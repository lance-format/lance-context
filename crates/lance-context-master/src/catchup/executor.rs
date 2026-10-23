use super::Result;
use crate::{config::MasterConfig, maintenance_execution, state::MasterState};
use futures::future::BoxFuture;
use lance_context_api::TaskKind;
use lance_context_core::rollout_append::SealedWalProbe;
use lance_context_core::{
    GenericStore, GenericStoreOptions, MergeMemoryBudget, RolloutStore, RolloutStoreOptions,
};
use lance_context_merge::MaintenanceKind;
use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

/// A successful empty pass is evidence to prioritize other shards, not proof
/// that this shard will stay empty. Recheck all shards every fourth pass and
/// whenever the previously productive set runs dry. State lasts one slice;
/// another claim always starts by checking every configured writer again.
struct ShardActivity(Vec<AtomicBool>);

impl ShardActivity {
    fn new(count: usize) -> Self {
        Self((0..count).map(|_| AtomicBool::new(true)).collect())
    }

    fn select(&self, pass: usize) -> Vec<usize> {
        let active: Vec<_> = self
            .0
            .iter()
            .enumerate()
            .filter_map(|(index, active)| active.load(Ordering::Relaxed).then_some(index))
            .collect();
        if pass.is_multiple_of(4) || active.is_empty() {
            (0..self.0.len()).collect()
        } else {
            active
        }
    }

    fn record(&self, index: usize, reclaimed: usize) {
        self.0[index].store(reclaimed > 0, Ordering::Relaxed);
    }

    fn caught_up(&self, selected: &[usize], reclaimed: usize) -> bool {
        selected.len() == self.0.len() && reclaimed == 0
    }
}

/// A busy admission never acquired a task or entered recovery/payload work.
/// Transport errors, including an unknown claim outcome, remain errors.
#[derive(Debug, PartialEq, Eq)]
pub enum ExecuteOutcome {
    Completed,
    AdmissionBusy,
}

/// One dedicated process, one table, ordered commits, bounded passes. No server,
/// scanner, ingestion writer, or fleet task consumer starts in this mode.
pub async fn execute(mut config: MasterConfig, target: &str) -> Result<ExecuteOutcome> {
    config.catchup.enabled = false;
    config.catchup.validate()?;
    if !config.merge_rollout.owned(target) || config.merge_rollout.draining(target) {
        return Err("catch-up executor requires a non-draining owned target".into());
    }
    let job = config
        .catchup
        .job_name
        .clone()
        .ok_or("missing catch-up Job identity")?;
    let startup = tokio::time::Instant::now();
    let state = MasterState::new(config).await.map_err(|e| e.to_string())?;
    tracing::info!(%target, startup_seconds = startup.elapsed().as_secs_f64(), "catch-up state ready");
    // Enqueue through the normal durable queue, then claim only this table.
    state
        .task_store
        .enqueue(TaskKind::MergeWal, target, Vec::new())
        .await
        .map_err(|e| e.to_string())?;
    let admission_started = tokio::time::Instant::now();
    let Some(claim) = claim_before_deadline(
        || async {
            state
                .task_store
                .claim_merge_target(target, &job)
                .await
                .map_err(|e| e.to_string())
        },
        admission_started + Duration::from_secs(30),
    )
    .await?
    else {
        super::outcome::report_deferred(&state.config.catchup, target)?;
        return Ok(ExecuteOutcome::AdmissionBusy);
    };
    tracing::info!(%target, admission_seconds = admission_started.elapsed().as_secs_f64(),
        "catch-up task claimed");
    let result = async {
        // The claim CAS has excluded any live reconciler. A previous process
        // may still have a storage PUT in flight: fence before opening writers.
        let coordinator = state.task_store.merge_coordinator();
        if let Some(old) = coordinator.get(target).await? {
            crate::merge_execution::ensure_recovery_due(&coordinator, &old).await?;
            crate::merge_execution::recover_execution(
                &state,
                &coordinator,
                &state.task_store.merge_claim(&claim),
                old,
            )
            .await?;
        }
        maintenance_execution::run_as(
            &state,
            &claim,
            MaintenanceKind::Catchup,
            merge_passes(&state, target),
        )
        .await
    }
    .await;
    let error = result.as_ref().err().cloned();
    state
        .task_store
        .finish(claim, result)
        .await
        .map_err(|e| e.to_string())?;
    if let Some(error) = error {
        return Err(error);
    }
    Ok(ExecuteOutcome::Completed)
}

async fn claim_before_deadline<T, F, Fut>(
    mut attempt: F,
    deadline: tokio::time::Instant,
) -> Result<Option<T>>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Option<T>>>,
{
    loop {
        // Bound retries between attempts, not by dropping an in-flight
        // mutating claim RPC. Individual etcd RPCs have their own timeout;
        // an accepted claim must be delivered even at the retry deadline.
        if tokio::time::Instant::now() >= deadline {
            return Ok(None);
        }
        if let Some(claim) = attempt().await? {
            return Ok(Some(claim));
        }
        tokio::time::sleep_until(
            deadline.min(tokio::time::Instant::now() + Duration::from_secs(1)),
        )
        .await;
    }
}

async fn merge_passes(state: &Arc<MasterState>, target: &str) -> Result<String> {
    if state.config.append.enabled(target) {
        return crate::rollout_append::run(state, target).await;
    }
    let session = RolloutStore::build_session(96 * 1024 * 1024, 32 * 1024 * 1024);
    let budget = MergeMemoryBudget::new(state.config.catchup.merge_memory_bytes);
    merge_local(state, target, session, budget).await
}

/// Same bounded storage implementation for resident masters and dedicated legacy executors.
/// Resources are supplied by the caller so resident tasks share one process budget.
pub(crate) async fn merge_local(
    state: &Arc<MasterState>,
    target: &str,
    session: Arc<lance::session::Session>,
    budget: Arc<MergeMemoryBudget>,
) -> Result<String> {
    let config = &state.config.catchup;
    let mut total = 0usize;
    // Legacy supervisors consume these progress messages and task details.
    let dedicated = config.target.is_some();
    let detail = |total| {
        let prefix = if dedicated {
            "dedicated catch-up merged"
        } else {
            "local WAL merge reclaimed"
        };
        format!("{prefix} {total} generations")
    };
    // Canonical claim reconciliation and the maintenance fence already surround
    // this future. Resolve the WAL root once and check fresh manifests
    // before opening full stores, including the empty shards of a busy table.
    // The normal activity/full-resweep policy below still applies.
    let uri = match target.strip_prefix("generic:") {
        Some(name) => state.generic_uri(name),
        None => state.rollout_uri(target),
    };
    let probe = SealedWalProbe::open(&uri, Some(session.clone()))
        .await
        .map_err(|e| e.to_string())?;
    let started = tokio::time::Instant::now();
    // This is only an admission time slice. A progressing operation already
    // admitted may run beyond it; cancellation is based on lack of progress.
    let admit_for = Duration::from_secs(config.slice_secs);
    let activity = ShardActivity::new(config.shards.len());
    // Bound each slice even under continuous ingestion. Resident masters publish
    // continuation demand before releasing their claim; no full scan is required.
    for pass in 0..16 {
        let selected = activity.select(pass);
        metrics::counter!("master_catchup_empty_shards_skipped_total")
            .increment((config.shards.len() - selected.len()) as u64);
        let reclaimed = run_pipeline(selected.iter().copied(), config.pipeline_enabled, |index| {
            let shard = &config.shards[index];
            let activity = &activity;
            let probe = &probe;
            let session = session.clone();
            let budget = budget.clone();
            async move {
                // Admission is checked after queue capacity becomes available.
                // A prepared batch already admitted may finish beyond the slice.
                if started.elapsed() >= admit_for
                    || crate::local_merge::drain_requested(state, target)
                {
                    return Ok(Box::pin(async { Ok(0usize) }) as BoxFuture<'_, Result<usize>>);
                }
                let read_started = tokio::time::Instant::now();
                // Keep the existing ordered activity/progress messages for
                // empty shards as well; only their full store opens disappear.
                let commit = prepare_nonempty_shard(state, target, shard, probe, session, budget)
                    .await?
                    .unwrap_or_else(|| Box::pin(async { Ok(0) }));
                let prepare_secs = read_started.elapsed().as_secs_f64();
                Ok(Box::pin(async move {
                    let commit_started = tokio::time::Instant::now();
                    let reclaimed = commit.await?;
                    // Only successful, actually admitted work informs the next
                    // pass. An expired slice or an error is not an empty scan.
                    activity.record(index, reclaimed);
                    let commit_secs = commit_started.elapsed().as_secs_f64();
                    metrics::counter!("master_catchup_generations_reclaimed_total")
                        .increment(reclaimed as u64);
                    if dedicated {
                        tracing::info!(%target, %shard, reclaimed, prepare_secs, commit_secs,
                            "dedicated catch-up committed");
                    } else {
                        tracing::info!(%target, %shard, reclaimed, prepare_secs, commit_secs,
                            "local WAL merge committed");
                    }
                    Ok(reclaimed)
                }) as BoxFuture<'_, Result<usize>>)
            }
        })
        .await?;
        total += reclaimed;
        // A skipped batch is not evidence of an empty WAL. Join all admitted
        // commits above, then publish continuation below before releasing the
        // task/fence, including when drain preceded the very first batch.
        if crate::local_merge::drain_requested(state, target) {
            break;
        }
        if activity.caught_up(&selected, reclaimed) && (total == 0 || started.elapsed() < admit_for)
        {
            // Empty metadata reads can consume the admission slice too. They
            // must not create a self-sustaining chain of zero-work tasks;
            // bounded resident discovery will reassert any remaining demand.
            return Ok(detail(total));
        }
        if started.elapsed() >= admit_for
            || crate::local_merge::yield_to_maintenance(state, target).await
        {
            break;
        }
    }
    if crate::local_merge::selected(&state.config, target) && config.target.is_none() {
        state
            .task_store
            .merge_coordinator()
            .request_merge(target)
            .await?;
        metrics::counter!("master_resident_merge_continuations_total").increment(1);
    }
    Ok(detail(total))
}

async fn prepare_nonempty_shard(
    state: &Arc<MasterState>,
    target: &str,
    shard: &str,
    probe: &SealedWalProbe,
    session: Arc<lance::session::Session>,
    budget: Arc<MergeMemoryBudget>,
) -> Result<Option<BoxFuture<'static, Result<usize>>>> {
    let started = tokio::time::Instant::now();
    // Every check refreshes the manifest; an earlier empty result does not hide
    // a later flush. Errors retain the normal failure/recovery path.
    let pending = probe
        .has_sealed_wal(shard)
        .await
        .map_err(|e| e.to_string())?;
    metrics::histogram!("master_catchup_shard_probe_seconds",
        "result" => if pending { "pending" } else { "empty" })
    .record(started.elapsed().as_secs_f64());
    if !pending {
        metrics::counter!("master_catchup_empty_shard_probes_total").increment(1);
        return Ok(None);
    }
    prepare_shard(state, target, shard, session, budget)
        .await
        .map(Some)
}

/// Read sealed WAL without acquiring a writer epoch. The returned future owns
/// the prepared data and its memory reservation; it performs no writes until
/// the single consumer polls it under the existing maintenance write scope.
async fn prepare_shard(
    state: &Arc<MasterState>,
    target: &str,
    shard: &str,
    session: Arc<lance::session::Session>,
    budget: Arc<MergeMemoryBudget>,
) -> Result<BoxFuture<'static, Result<usize>>> {
    if let Some(name) = target.strip_prefix("generic:") {
        let options = GenericStoreOptions {
            shard_id: Some(shard.to_owned()),
            merge_after_generations: Some(0),
            merge_max_generations: Some(state.config.catchup.merge_max_generations),
            merge_max_bytes: Some(state.config.catchup.merge_max_bytes),
            key_index_type: state.config.key_index_type,
            merge_budget: Some(budget.clone()),
            session: Some(session.clone()),
            pending_generations_max: Some(0),
            ..Default::default()
        };
        let mut store = GenericStore::open_existing(&state.generic_uri(name), options)
            .await
            .map_err(|e| e.to_string())?;
        // No resident writer was opened, so cleanup only sees sealed WAL.
        match store
            .prepare_cleanup_merge()
            .await
            .map_err(|e| e.to_string())?
        {
            Some((manifest_store, manifest, prepared)) => Ok(Box::pin(async move {
                store
                    .commit_prepared_merge(&manifest_store, &manifest, prepared)
                    .await
                    .map_err(|e| e.to_string())
            })),
            None => Ok(Box::pin(async { Ok(0usize) })),
        }
    } else {
        let options = RolloutStoreOptions {
            shard_id: Some(shard.to_owned()),
            merge_after_generations: Some(0),
            merge_max_generations: Some(state.config.catchup.merge_max_generations),
            merge_max_bytes: Some(state.config.catchup.merge_max_bytes),
            key_index_type: state.config.key_index_type,
            merge_budget: Some(budget.clone()),
            session: Some(session.clone()),
            pending_generations_max: Some(0),
            ..Default::default()
        };
        let mut store =
            RolloutStore::open_existing_with_options(&state.rollout_uri(target), options)
                .await
                .map_err(|e| e.to_string())?;
        match store
            .prepare_merge_if_ready(1)
            .await
            .map_err(|e| e.to_string())?
        {
            Some((manifest_store, manifest, mut prepared)) => {
                if state
                    .config
                    .catchup
                    .bulk_key_delete_targets
                    .iter()
                    .any(|t| t == target)
                {
                    prepared.use_bulk_key_delete();
                }
                Ok(Box::pin(async move {
                    store
                        .commit_prepared_merge(&manifest_store, &manifest, prepared)
                        .await
                        .map_err(|e| e.to_string())
                }))
            }
            None => Ok(Box::pin(async { Ok(0usize) })),
        }
    }
}

// StorageBase releases the failed generation/reservation before returning this
// error. A speculative reader can encounter it while the committer holds bytes.
const MERGE_BUDGET_BUSY: &str = "merge memory budget busy while reading first generation; retry";

/// At most one commit plus one preparing/queued batch. Reserve queue capacity
/// BEFORE preparing: otherwise a producer can retain a third batch. Do not use
/// `buffered(n)` here: a later read holding the entire memory budget can block
/// an earlier read forever while ordered delivery prevents its commit.
///
/// Both halves are polled in the caller's task, preserving the task-local
/// maintenance fence and progress scope. Cancellation drops both halves and
/// their reservations; already admitted manifest writes are drained by the scope.
async fn run_pipeline<I, P, F, C>(items: I, enabled: bool, mut prepare: P) -> Result<usize>
where
    I: IntoIterator,
    I::Item: Clone,
    P: FnMut(I::Item) -> F,
    F: Future<Output = Result<C>>,
    C: Future<Output = Result<usize>>,
{
    if !enabled {
        let mut total = 0;
        for item in items {
            total += prepare(item).await?.await?;
        }
        return Ok(total);
    }
    let (send, mut receive) = tokio::sync::mpsc::channel::<Result<C>>(1);
    let (completed, mut completion) = tokio::sync::watch::channel(0usize);
    let producer = async move {
        for (sent, item) in items.into_iter().enumerate() {
            let permit = send
                .reserve()
                .await
                .map_err(|_| "catch-up committer stopped")?;
            let mut prepared = prepare(item.clone()).await;
            if prepared
                .as_ref()
                .is_err_and(|error| error.contains(MERGE_BUDGET_BUSY))
            {
                // Finish the preceding commit before retrying this shard alone.
                // Do not prepare later shards while it waits: they could take
                // the freed memory and recreate the contention. Retry only once;
                // storage/other failures retain the normal durable backoff.
                let finished = completion
                    .wait_for(|count| *count >= sent)
                    .await
                    .map_err(|_| "catch-up committer stopped before memory retry")?;
                drop(finished);
                metrics::counter!("master_catchup_prefetch_memory_retries_total").increment(1);
                prepared = prepare(item).await;
            }
            let failed = prepared.is_err();
            permit.send(prepared);
            // Deliver read errors in order: let the preceding commit finish,
            // then stop. Never keep preparing later shards after a read failure.
            if failed {
                break;
            }
        }
        Ok::<_, String>(())
    };
    let consumer = async move {
        let mut total = 0;
        let mut finished = 0;
        while let Some(prepared) = receive.recv().await {
            total += prepared?.await?;
            finished += 1;
            completed.send_replace(finished);
        }
        Ok::<_, String>(total)
    };
    let (_, total) = tokio::try_join!(producer, consumer)?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::{Notify, Semaphore};

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS; local metadata benchmark"]
    async fn empty_preflight_storage_benchmark() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let mut config = MasterConfig::parse_from(["test"]);
        config.data_dir = dir.path().to_string_lossy().into();
        config.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        config.etcd.etcd_prefix = format!(
            "/empty-preflight-bench/{}",
            lance_context_core::generate_id()
        );
        config.catchup.shards = (0..20).map(|i| format!("writer-{i}")).collect();
        config.worker_endpoints.clear();
        config.stats_scan_interval_secs = 0;
        let state = MasterState::new(config).await.unwrap();

        async fn write_sample(
            state: &MasterState,
            target: &str,
            shard: &str,
            id: &str,
        ) -> RolloutStore {
            let writer = RolloutStore::open_with_options(
                &state.rollout_uri(target),
                RolloutStoreOptions {
                    shard_id: Some(shard.into()),
                    merge_after_generations: Some(0),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let dto = serde_json::from_value(serde_json::json!({
                "id": id, "rollout_id": "r", "content": "benchmark"
            }))
            .unwrap();
            writer
                .add(&[lance_context_core::rollout_record_from_add_request(&dto)])
                .await
                .unwrap();
            writer.flush().await.unwrap();
            writer
        }

        // Equivalent tables with real, drained manifests on all twenty shards.
        // Keep setup outside measurements and alternate which path runs first.
        for target in ["baseline", "filtered"] {
            for shard in &state.config.catchup.shards {
                let mut writer = write_sample(&state, target, shard, shard).await;
                assert_eq!(writer.cleanup_own_shard().await.unwrap(), 1);
                writer.close().await.unwrap();
            }
        }
        let session = RolloutStore::build_session(96 * 1024 * 1024, 32 * 1024 * 1024);
        let budget = MergeMemoryBudget::new(state.config.catchup.merge_memory_bytes);
        for case in ["empty", "first_shard", "last_shard", "all_shards"] {
            for sample in 0..6 {
                for filtered in if sample % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let target = if filtered { "filtered" } else { "baseline" };
                    let mut writers = Vec::new();
                    if case != "empty" {
                        for (index, shard) in state.config.catchup.shards.iter().enumerate() {
                            if case == "all_shards"
                                || case == "last_shard" && index == 19
                                || case == "first_shard" && index == 0
                            {
                                writers.push(
                                    write_sample(
                                        &state,
                                        target,
                                        shard,
                                        &format!("{case}-{sample}-{index}"),
                                    )
                                    .await,
                                );
                            }
                        }
                    }
                    let started = tokio::time::Instant::now();
                    let probe = if filtered {
                        Some(
                            SealedWalProbe::open(&state.rollout_uri(target), Some(session.clone()))
                                .await
                                .unwrap(),
                        )
                    } else {
                        None
                    };
                    let reclaimed = run_pipeline(
                        state.config.catchup.shards.iter(),
                        state.config.catchup.pipeline_enabled,
                        |shard| {
                            let probe = probe.as_ref();
                            let state = &state;
                            let session = session.clone();
                            let budget = budget.clone();
                            async move {
                                if let Some(probe) = probe {
                                    Ok(prepare_nonempty_shard(
                                        state, target, shard, probe, session, budget,
                                    )
                                    .await?
                                    .unwrap_or_else(|| Box::pin(async { Ok(0) })))
                                } else {
                                    prepare_shard(state, target, shard, session, budget).await
                                }
                            }
                        },
                    )
                    .await
                    .unwrap();
                    eprintln!("storage_benchmark case={case} sample={sample} filtered={filtered} micros={} reclaimed={reclaimed}", started.elapsed().as_micros());
                    assert_eq!(reclaimed, writers.len());
                    for writer in &mut writers {
                        writer.close().await.unwrap();
                    }
                    assert_eq!(budget.reserved(), 0);
                }
            }
        }
        for target in ["baseline", "filtered"] {
            assert_eq!(
                lance::Dataset::open(&state.rollout_uri(target))
                    .await
                    .unwrap()
                    .count_rows(None)
                    .await
                    .unwrap(),
                20 + 6 + 6 + 6 * 20
            );
        }
    }

    #[tokio::test]
    async fn sparse_shards_require_a_fresh_full_sweep_before_completion() {
        for pipelined in [false, true] {
            for late_write in [false, true] {
                let activity = ShardActivity::new(20);
                let pending: Vec<_> = (0..20)
                    .map(|index| AtomicUsize::new(if index == 0 { 24 } else { 0 }))
                    .collect();
                let empty_reads = AtomicUsize::new(0);
                let mut reclaimed = 0;
                let mut completed = false;
                let mut saw_partial_empty = false;
                for pass in 0..16 {
                    let selected = activity.select(pass);
                    let count = run_pipeline(selected.iter().copied(), pipelined, |index| {
                        let activity = &activity;
                        let pending = &pending;
                        let empty_reads = &empty_reads;
                        async move {
                            // Model a byte-bounded read followed by its joined
                            // commit. Each successful pass reclaims at most six.
                            let mut remaining = pending[index].load(Ordering::SeqCst);
                            let count = loop {
                                let count = remaining.min(6);
                                match pending[index].compare_exchange(
                                    remaining,
                                    remaining - count,
                                    Ordering::SeqCst,
                                    Ordering::SeqCst,
                                ) {
                                    Ok(_) => break count,
                                    Err(current) => remaining = current,
                                }
                            };
                            if count == 0 {
                                empty_reads.fetch_add(1, Ordering::SeqCst);
                            }
                            Ok(async move {
                                tokio::task::yield_now().await;
                                activity.record(index, count);
                                Ok(count)
                            })
                        }
                    })
                    .await
                    .unwrap();
                    reclaimed += count;
                    if pass == 1 && late_write {
                        // This writer was empty at discovery, then received new
                        // data while the hot writer was still catching up.
                        pending[19].store(2, Ordering::SeqCst);
                    }
                    if count == 0 && selected.len() < 20 {
                        saw_partial_empty = true;
                        assert!(!activity.caught_up(&selected, count));
                    }
                    if activity.caught_up(&selected, count) {
                        assert!(pending.iter().all(|x| x.load(Ordering::SeqCst) == 0));
                        completed = true;
                        break;
                    }
                }
                assert!(completed);
                assert_eq!(reclaimed, if late_write { 26 } else { 24 });
                if late_write {
                    assert!(
                        saw_partial_empty,
                        "completion must wait for the full resweep"
                    );
                } else {
                    // Initial and final discovery remain; the three middle
                    // rounds avoid 19 known-empty shards each (96 -> 39).
                    assert_eq!(empty_reads.load(Ordering::SeqCst), 39);
                }
            }
        }
    }

    #[tokio::test]
    async fn continuously_hot_shard_cannot_starve_a_newly_active_writer() {
        for pipelined in [false, true] {
            let activity = ShardActivity::new(20);
            let cold_pending = AtomicUsize::new(0);
            let mut cold_reclaimed_at = Vec::new();
            for pass in 0..12 {
                let selected = activity.select(pass);
                let count = run_pipeline(selected.iter().copied(), pipelined, |index| {
                    let activity = &activity;
                    let cold_pending = &cold_pending;
                    async move {
                        let count = match index {
                            0 => 6, // Continuous ingestion never lets this go idle.
                            19 => cold_pending.swap(0, Ordering::SeqCst),
                            _ => 0,
                        };
                        Ok(async move {
                            activity.record(index, count);
                            Ok(count)
                        })
                    }
                })
                .await
                .unwrap();
                assert!(!activity.caught_up(&selected, count));
                if count > 6 {
                    cold_reclaimed_at.push(pass);
                }
                if pass == 1 || pass == 6 {
                    cold_pending.store(2, Ordering::SeqCst);
                }
            }
            assert_eq!(cold_reclaimed_at, [4, 8]);
            assert_eq!(cold_pending.load(Ordering::SeqCst), 0);
            // A new claim cannot inherit an old empty observation.
            assert_eq!(
                ShardActivity::new(20).select(0),
                (0..20).collect::<Vec<_>>()
            );
        }
    }

    #[tokio::test]
    async fn admission_delivers_claim_accepted_across_retry_deadline() {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
        let result = claim_before_deadline(
            || async {
                tokio::time::sleep_until(deadline + Duration::from_millis(10)).await;
                Ok(Some("accepted claim"))
            },
            deadline,
        )
        .await
        .unwrap();
        assert_eq!(result, Some("accepted claim"));
    }

    #[tokio::test]
    async fn admission_stops_retrying_unclaimed_work_at_deadline() {
        let calls = AtomicUsize::new(0);
        let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
        let result = claim_before_deadline(
            || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(None::<()>)
            },
            deadline,
        )
        .await;
        assert_eq!(result.unwrap(), None);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn admission_unknown_claim_outcome_is_not_busy_even_after_deadline() {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
        let result = claim_before_deadline(
            || async {
                tokio::time::sleep_until(deadline + Duration::from_millis(10)).await;
                Err::<Option<()>, _>("claim response lost after accepted transaction".into())
            },
            deadline,
        )
        .await;
        assert_eq!(
            result.unwrap_err(),
            "claim response lost after accepted transaction"
        );
    }

    #[tokio::test]
    async fn pipeline_overlaps_reads_with_one_ordered_commit_and_bounds_lookahead() {
        let prepared = AtomicUsize::new(0);
        let committed = std::sync::Mutex::new(Vec::new());
        let next_ready = Notify::new();
        let release = Semaphore::new(0);
        let work = run_pipeline(0..5, true, |id| {
            let prepared = &prepared;
            let committed = &committed;
            let next_ready = &next_ready;
            let release = &release;
            async move {
                prepared.fetch_add(1, Ordering::SeqCst);
                if id == 1 {
                    next_ready.notify_one();
                }
                Ok(async move {
                    if id == 0 {
                        release.acquire().await.unwrap().forget();
                    }
                    committed.lock().unwrap().push(id);
                    Ok(1)
                })
            }
        });
        let check = async {
            next_ready.notified().await;
            assert_eq!(prepared.load(Ordering::SeqCst), 2);
            assert!(committed.lock().unwrap().is_empty());
            release.add_permits(1);
        };
        let (result, ()) =
            tokio::time::timeout(Duration::from_secs(5), async { tokio::join!(work, check) })
                .await
                .unwrap();
        assert_eq!(result.unwrap(), 5);
        assert_eq!(*committed.lock().unwrap(), [0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn pipeline_progresses_when_only_one_batch_fits_and_releases_on_failure() {
        for enabled in [false, true] {
            for fail_at in [None, Some(2)] {
                let budget = MergeMemoryBudget::new(1024 * 1024);
                let committed = AtomicUsize::new(0);
                let result = tokio::time::timeout(
                    Duration::from_secs(5),
                    run_pipeline(0..5, enabled, |id| {
                        let budget = &budget;
                        let committed = &committed;
                        async move {
                            let reservation = budget.reserve(budget.limit()).await;
                            Ok(async move {
                                let _reservation = reservation;
                                tokio::task::yield_now().await;
                                if fail_at == Some(id) {
                                    return Err("commit failed".into());
                                }
                                committed.fetch_add(1, Ordering::SeqCst);
                                Ok(1)
                            })
                        }
                    }),
                )
                .await
                .expect("a queued reservation must not block the committer");
                assert_eq!(
                    result,
                    fail_at.map_or(Ok(5), |_| Err("commit failed".into()))
                );
                assert_eq!(committed.load(Ordering::SeqCst), fail_at.unwrap_or(5));
                assert_eq!(budget.reserved(), 0);
            }
        }
    }

    #[tokio::test]
    async fn oversized_prefetch_retries_after_commit_releases_memory() {
        let budget = MergeMemoryBudget::new(2 * 1024 * 1024);
        let attempts = AtomicUsize::new(0);
        let contention = Notify::new();
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            run_pipeline(0..2, true, |id| {
                let budget = &budget;
                let attempts = &attempts;
                let contention = &contention;
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    let mut reservation = budget.reserve(1024 * 1024).await;
                    if id == 1 && !reservation.try_grow_to(2 * 1024 * 1024) {
                        contention.notify_one();
                        return Err(MERGE_BUDGET_BUSY.into());
                    }
                    Ok(async move {
                        let _reservation = reservation;
                        if id == 0 {
                            // Force the first attempt at shard 1 to contend with
                            // this commit, then permit the commit to release bytes.
                            contention.notified().await;
                        }
                        Ok(1)
                    })
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(result, Ok(2));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert_eq!(budget.reserved(), 0);
    }

    #[tokio::test]
    async fn read_failure_finishes_previous_commit_and_stops_admission() {
        let prepared = AtomicUsize::new(0);
        let committed = AtomicUsize::new(0);
        let failed_read = Notify::new();
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            run_pipeline(0..5, true, |id| {
                let prepared = &prepared;
                let committed = &committed;
                let failed_read = &failed_read;
                async move {
                    prepared.fetch_add(1, Ordering::SeqCst);
                    if id == 1 {
                        failed_read.notify_one();
                        return Err("read failed".into());
                    }
                    Ok(async move {
                        failed_read.notified().await;
                        committed.fetch_add(1, Ordering::SeqCst);
                        Ok(1)
                    })
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(result, Err("read failed".into()));
        assert_eq!(prepared.load(Ordering::SeqCst), 2);
        assert_eq!(committed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancellation_drops_active_and_prefetched_reservations() {
        let budget = MergeMemoryBudget::new(2 * 1024 * 1024);
        let next_ready = Notify::new();
        let mut work = Box::pin(run_pipeline(0..5, true, |id| {
            let budget = &budget;
            let next_ready = &next_ready;
            async move {
                let reservation = budget.reserve(1024 * 1024).await;
                if id == 1 {
                    next_ready.notify_one();
                }
                Ok(async move {
                    let _reservation = reservation;
                    std::future::pending::<Result<usize>>().await
                })
            }
        }));
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::select! {
                result = &mut work => panic!("unexpected completion: {result:?}"),
                () = next_ready.notified() => {}
            }
        })
        .await
        .unwrap();
        assert_eq!(budget.reserved(), budget.limit());
        drop(work);
        assert_eq!(budget.reserved(), 0);
    }
}
