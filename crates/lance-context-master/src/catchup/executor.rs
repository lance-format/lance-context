use super::Result;
use crate::{config::MasterConfig, maintenance_execution, state::MasterState};
use futures::future::BoxFuture;
use lance_context_api::TaskKind;
use lance_context_core::{
    GenericStore, GenericStoreOptions, MergeMemoryBudget, RolloutStore, RolloutStoreOptions,
};
use lance_context_merge::MaintenanceKind;
use std::{future::Future, sync::Arc, time::Duration};

/// One dedicated process, one table, ordered commits, bounded passes. No server,
/// scanner, ingestion writer, or fleet task consumer starts in this mode.
pub async fn execute(mut config: MasterConfig, target: &str) -> Result<()> {
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
    let state = MasterState::new(config).await.map_err(|e| e.to_string())?;
    // Enqueue through the normal durable queue, then claim only this table.
    state
        .task_store
        .enqueue(TaskKind::MergeWal, target, Vec::new())
        .await
        .map_err(|e| e.to_string())?;
    let claim = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Some(claim) = state
                .task_store
                .claim_merge_target(target, &job)
                .await
                .map_err(|e| e.to_string())?
            {
                return Ok::<_, String>(claim);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    })
    .await
    .map_err(|_| "catch-up admission remained busy for 30 seconds")??;
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
    Ok(())
}

async fn merge_passes(state: &Arc<MasterState>, target: &str) -> Result<String> {
    let config = &state.config.catchup;
    let session = RolloutStore::build_session(96 * 1024 * 1024, 32 * 1024 * 1024);
    let budget = MergeMemoryBudget::new(config.merge_memory_bytes);
    let mut total = 0usize;
    let started = tokio::time::Instant::now();
    // This is only an admission time slice. A progressing operation already
    // admitted may run beyond it; cancellation is based on lack of progress.
    let admit_for = Duration::from_secs(config.slice_secs);
    // Bound each slice even under continuous ingestion; later fresh stats may
    // request another Job. Visit every shard before repeating any hot shard.
    for _ in 0..16 {
        let before = total;
        total += run_pipeline(&config.shards, config.pipeline_enabled, |shard| {
            let session = session.clone();
            let budget = budget.clone();
            async move {
                // Admission is checked after queue capacity becomes available.
                // A prepared batch already admitted may finish beyond the slice.
                if started.elapsed() >= admit_for {
                    return Ok(Box::pin(async { Ok(0usize) }) as BoxFuture<'_, Result<usize>>);
                }
                let read_started = tokio::time::Instant::now();
                let commit = prepare_shard(state, target, shard, session, budget).await?;
                let prepare_secs = read_started.elapsed().as_secs_f64();
                Ok(Box::pin(async move {
                    let commit_started = tokio::time::Instant::now();
                    let reclaimed = commit.await?;
                    let commit_secs = commit_started.elapsed().as_secs_f64();
                    metrics::counter!("master_catchup_generations_reclaimed_total")
                        .increment(reclaimed as u64);
                    tracing::info!(%target, %shard, reclaimed, prepare_secs, commit_secs,
                        "dedicated catch-up committed");
                    Ok(reclaimed)
                }) as BoxFuture<'_, Result<usize>>)
            }
        })
        .await?;
        if total == before {
            break;
        }
    }
    Ok(format!("dedicated catch-up merged {total} generations"))
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
            Some((manifest_store, manifest, prepared)) => Ok(Box::pin(async move {
                store
                    .commit_prepared_merge(&manifest_store, &manifest, prepared)
                    .await
                    .map_err(|e| e.to_string())
            })),
            None => Ok(Box::pin(async { Ok(0usize) })),
        }
    }
}

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
    let producer = async move {
        for item in items {
            let permit = send
                .reserve()
                .await
                .map_err(|_| "catch-up committer stopped")?;
            let prepared = prepare(item).await;
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
        while let Some(prepared) = receive.recv().await {
            total += prepared?.await?;
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
