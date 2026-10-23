//! Bounded local table mutations sharing WAL merge's durable storage fence.
use crate::{state::MasterState, task_store::TaskClaim};
use futures::FutureExt;
use lance_context_api::TaskKind;
use lance_context_core::merge_write_scope::{CommitAuthorizer, MergeWriteScope};
use lance_context_merge::{Coordinator, Execution, MaintenanceKind, Phase};
use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

pub(crate) fn kind(task: TaskKind) -> Option<MaintenanceKind> {
    match task {
        TaskKind::Compact => Some(MaintenanceKind::Compact),
        TaskKind::IndexId => Some(MaintenanceKind::IndexId),
        TaskKind::Repair => Some(MaintenanceKind::Repair),
        TaskKind::MergeWal => None,
    }
}

pub(crate) fn retry_kind(endpoint: &str) -> Option<TaskKind> {
    match MaintenanceKind::from_endpoint(endpoint)? {
        MaintenanceKind::Compact => Some(TaskKind::Compact),
        MaintenanceKind::IndexId => Some(TaskKind::IndexId),
        MaintenanceKind::Repair => Some(TaskKind::Repair),
        MaintenanceKind::Catchup => Some(TaskKind::MergeWal),
    }
}

pub(crate) async fn enqueue_recovery(
    state: &Arc<MasterState>,
    coordinator: &Coordinator,
    execution: &Execution,
) -> Result<(), String> {
    let Some(task_kind) = retry_kind(&execution.endpoint) else {
        return Ok(());
    };
    // A recorded fenced execution needs reconciliation even if the
    // enrollment/maintenance flag was disabled after its admission.
    if !(state.config.merge_rollout.owned(&execution.target)
        || state.config.merge_rollout.draining(&execution.target)
        || execution.protocol == 2 && execution.maintenance.is_some())
    {
        return Ok(());
    }
    if coordinator
        .failure(&execution.target, &execution.endpoint)
        .await?
        .is_some_and(|f| f.next_retry_ms > lance_context_merge::failure::now_ms())
    {
        return Ok(());
    }
    crate::scheduler::enqueue(state, task_kind, &execution.target)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Claim admission excludes every other reconciler while this runs. A lost
/// process lease is permission to fence its admitted versions, not to delete
/// its persistent target lock or assume that its last PUT failed.
pub(crate) async fn reconcile_previous(
    state: &Arc<MasterState>,
    claim: &TaskClaim,
) -> Result<(), String> {
    let coordinator = state.task_store.merge_coordinator();
    let Some(old) = coordinator.get(&claim.task.target).await? else {
        return Ok(());
    };
    if old.maintenance.is_none() {
        return Ok(());
    }
    let proof = state.task_store.merge_claim(claim);
    crate::merge_execution::ensure_recovery_due(&coordinator, &old).await?;
    match old.phase {
        Phase::Reserved => {
            if !coordinator.cancel_reserved(&old).await? {
                return Err("maintenance reservation changed".into());
            }
            let terminal = coordinator
                .get(&old.target)
                .await?
                .ok_or("maintenance reservation disappeared")?;
            if !coordinator.release(&proof, &terminal).await? {
                return Err("claim lost releasing maintenance".into());
            }
        }
        Phase::Finished | Phase::Recovered => {
            if !coordinator.release(&proof, &old).await? {
                return Err("claim lost releasing maintenance".into());
            }
        }
        _ => crate::merge_execution::recover_execution(state, &coordinator, &proof, old).await?,
    }
    Ok(())
}

struct Guard {
    coordinator: Coordinator,
    execution: Execution,
    uri: String,
}
impl std::fmt::Debug for Guard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaintenanceGuard")
            .field("id", &self.execution.id)
            .finish()
    }
}
impl CommitAuthorizer for Guard {
    fn authorize<'a>(
        &'a self,
        resource: &'a str,
        version: u64,
    ) -> Pin<Box<dyn Future<Output = lance::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.coordinator
                .authorize_commit(&self.execution, &self.uri, resource, version)
                .await
                .map_err(lance::Error::io)
        })
    }
}

/// Recovery may just have published the first failure's two-second backoff.
/// Honor that deadline here instead of failing a new dedicated pass and making
/// its supervisor add another exponential delay. Budget at most two seconds
/// for cooldown sleeps, then re-read the deadline before admitting payload work.
async fn wait_for_short_backoff(
    coordinator: &Coordinator,
    target: &str,
    endpoint: &str,
) -> Result<(), String> {
    let wait_until = tokio::time::Instant::now() + Duration::from_secs(2);
    while let Some(failure) = coordinator.failure(target, endpoint).await? {
        let remaining = failure
            .next_retry_ms
            .saturating_sub(lance_context_merge::failure::now_ms());
        if remaining == 0 {
            break;
        }
        let delay = Duration::from_millis(remaining);
        if delay > wait_until.saturating_duration_since(tokio::time::Instant::now()) {
            return Err(format!(
                "maintenance retry at {} after {:?} failure",
                failure.next_retry_ms, failure.class
            ));
        }
        tracing::info!(
            target,
            endpoint,
            retry_at_ms = failure.next_retry_ms,
            wait_ms = remaining,
            "waiting for short maintenance retry deadline"
        );
        tokio::time::sleep(delay).await;
    }
    Ok(())
}

async fn watch(
    coordinator: &Coordinator,
    execution: &Execution,
    scope: &MergeWriteScope,
    executor_id: &str,
    task_id: &str,
) -> String {
    let mut publisher = coordinator.work_progress_publisher(execution);
    watch_with_publisher(execution, scope, executor_id, task_id, &mut publisher).await
}

trait ProgressPublisher: Send {
    fn publish(
        &mut self,
        sequence: u64,
        report: serde_json::Value,
    ) -> impl Future<Output = Result<bool, String>> + Send;
}

impl ProgressPublisher for lance_context_merge::progress::WorkProgressPublisher {
    async fn publish(&mut self, sequence: u64, report: serde_json::Value) -> Result<bool, String> {
        self.publish(sequence, report).await
    }
}

async fn watch_with_publisher(
    execution: &Execution,
    scope: &MergeWriteScope,
    executor_id: &str,
    task_id: &str,
    publisher: &mut impl ProgressPublisher,
) -> String {
    let mut sequence = scope.completed_steps();
    if sequence > 0 {
        crate::merge_capacity::observed_work();
    }
    let mut changed = tokio::time::Instant::now();
    let mut last_progress_at_ms = (sequence > 0).then(lance_context_merge::failure::now_ms);
    let mut publication_failed = false;
    // Work observation is local and stays responsive. Publishing every local
    // sample made each active task generate a replicated write every second,
    // even when the progress sequence was unchanged. Coalesce reports, leaving
    // several publication opportunities inside the actual-work idle budget.
    let publish_every = Duration::from_secs((execution.idle_timeout_secs / 4).clamp(1, 5));
    let mut next_publication = tokio::time::Instant::now();
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let current = scope.completed_steps();
        // Publish resumed work promptly after a quiet period. Otherwise a
        // buffered update near the idle boundary could unnecessarily widen
        // the remote observer's stale-progress window.
        let resumed = current != sequence && changed.elapsed() >= publish_every;
        if current > sequence {
            crate::merge_capacity::observed_work();
        }
        if current != sequence {
            changed = tokio::time::Instant::now();
            sequence = current;
            last_progress_at_ms = Some(lance_context_merge::failure::now_ms());
        }
        if changed.elapsed() >= Duration::from_secs(execution.idle_timeout_secs) {
            tracing::warn!(target = %execution.target, execution = %execution.id, task_id, executor_id,
                sequence, work = ?scope.work(), idle_seconds = changed.elapsed().as_secs(),
                "maintenance stopped making progress; cancelling preparation before drain/fence recovery");
            return "maintenance no-progress deadline exceeded".into();
        }
        if !publication_failed && !resumed && tokio::time::Instant::now() < next_publication {
            metrics::counter!("master_maintenance_progress_coalesced_samples_total").increment(1);
            continue;
        }
        // An occasional unchanged report checks ownership and freshness; it
        // never resets the local or remote actual-work deadline. Every storage
        // commit still checks its own fence, including between these reports.
        let report = serde_json::json!({
            "executor_id": executor_id,
            "task_id": task_id,
            "sequence": sequence,
            "observed_at_ms": lance_context_merge::failure::now_ms(),
            "last_progress_at_ms": last_progress_at_ms,
            "idle_for_ms": changed.elapsed().as_millis() as u64,
            "idle_timeout_secs": execution.idle_timeout_secs,
            "work": scope.work(),
        });
        metrics::counter!("master_maintenance_progress_publications_total").increment(1);
        match publisher.publish(sequence, report).await {
            Ok(true) => {
                next_publication = tokio::time::Instant::now() + publish_every;
                if publication_failed {
                    metrics::counter!("master_maintenance_progress_publication_recovered_total")
                        .increment(1);
                }
                publication_failed = false;
            }
            Ok(false) => return "maintenance ownership revoked".into(),
            Err(error) => {
                metrics::counter!("master_maintenance_progress_publication_errors_total")
                    .increment(1);
                if publication_failed {
                    return format!("maintenance progress publication failed twice: {error}");
                }
                publication_failed = true;
                // One bounded retry on the next normal tick. This does not
                // reset the actual-work deadline or authorize a storage write;
                // every manifest still obtains its own fenced commit permit.
                tracing::warn!(target = %execution.target, execution = %execution.id, task_id,
                    %error, "maintenance progress publication failed; retrying once without resetting progress");
            }
        }
    }
}

pub(crate) async fn run<F>(
    state: &Arc<MasterState>,
    claim: &TaskClaim,
    work: F,
) -> Result<String, String>
where
    F: Future<Output = Result<String, String>>,
{
    if !fenced(&state.config, &claim.task.target) {
        return observe_legacy(work).await;
    }
    let maintenance = kind(claim.task.kind).ok_or("invalid local maintenance kind")?;
    run_as(state, claim, maintenance, work).await
}

/// Report actual local work for legacy maintenance too, but retain its original
/// cancellation and commit behavior. A quiet observation never stops this work.
async fn observe_legacy<F>(work: F) -> F::Output
where
    F: Future,
{
    crate::merge_capacity::phase("legacy_maintenance");
    let scope = MergeWriteScope::for_legacy_file_observation();
    let work = scope.run(work);
    tokio::pin!(work);
    let mut observed = 0;
    loop {
        let outcome = tokio::select! {
            result = &mut work => Some(result),
            _ = tokio::time::sleep(Duration::from_secs(1)) => None,
        };
        let current = scope.completed_steps();
        if current > observed {
            observed = current;
            crate::merge_capacity::observed_work();
        }
        if let Some(result) = outcome {
            return result;
        }
    }
}

fn fenced(config: &crate::config::MasterConfig, target: &str) -> bool {
    config.merge_rollout.owned(target)
        || (config.maintenance.fence_legacy_maintenance && !target.starts_with("datagen:"))
}

pub(crate) async fn run_as<F>(
    state: &Arc<MasterState>,
    claim: &TaskClaim,
    maintenance: MaintenanceKind,
    work: F,
) -> Result<String, String>
where
    F: Future<Output = Result<String, String>>,
{
    crate::merge_capacity::phase("fenced_maintenance");
    let uri = match claim.task.target.strip_prefix("generic:") {
        Some(name) => state.generic_uri(name),
        None => state.rollout_uri(&claim.task.target),
    };
    if !lance_context_core::merge_write_scope::supports_version_fencing(&uri) {
        return Err("invalid storage backend for maintenance fencing".into());
    }
    let coordinator = state.task_store.merge_coordinator();
    let proof = state.task_store.merge_claim(claim);
    wait_for_short_backoff(&coordinator, &claim.task.target, maintenance.endpoint()).await?;
    let config = &state.config.maintenance;
    let mut reserved = Execution::new(
        &claim.task.target,
        maintenance.endpoint(),
        if maintenance == MaintenanceKind::Catchup {
            state
                .config
                .catchup
                .job_name
                .as_deref()
                .unwrap_or(&claim.task.id)
        } else {
            &claim.task.id
        },
        config.maintenance_timeout_secs,
    );
    reserved.maintenance = Some(maintenance);
    reserved.idle_timeout_secs = config.maintenance_idle_timeout_secs;
    if !coordinator.reserve(&proof, &reserved).await? {
        return Err("maintenance admission lost".into());
    }
    let running = coordinator
        .start(&reserved)
        .await?
        .ok_or("maintenance start lost")?;
    // These handles are ephemeral to this operation. Observe successful file
    // IO as well as logical checkpoints so a long legacy compact/index scan
    // does not look idle merely because Lance has not completed a whole phase.
    let scope = MergeWriteScope::with_preparation_authorizer(Arc::new(Guard {
        coordinator: coordinator.clone(),
        execution: running.clone(),
        uri: uri.clone(),
    }));
    let executor_id = state.admission.status().executor_id;
    let outcome = std::panic::AssertUnwindSafe(async {
        let watched = async {
            tokio::select! {
                result = scope.run(work) => result,
                error = watch(&coordinator, &running, &scope, &executor_id, &claim.task.id) => Err(error),
            }
        };
        watched.await
    })
    .catch_unwind()
    .await
    .unwrap_or_else(|_| Err("maintenance executor panicked".into()));
    tracing::info!(target = %running.target, execution = %running.id,
        task_id = %claim.task.id, %executor_id, work = ?scope.work(),
        success = outcome.is_ok(), "maintenance work outcome before commit drain");
    // The work future has been dropped, so no preparation may issue another
    // commit. The shielded manifest leaves still need acknowledgement or fencing.
    let drained = tokio::time::timeout(
        Duration::from_secs(config.maintenance_drain_timeout_secs),
        scope.drain(),
    )
    .await
    .is_ok();
    if !drained {
        retain_until_fenced(coordinator.clone(), running.clone(), scope.clone());
    }
    if !drained || scope.has_uncertain_commit() {
        let error = outcome
            .as_ref()
            .err()
            .cloned()
            .unwrap_or_else(|| "maintenance manifest result unknown".into());
        let _ = coordinator.report_uncertain(&running, error.clone()).await;
        let current = coordinator
            .get(&running.target)
            .await?
            .ok_or("maintenance ownership disappeared")?;
        match crate::merge_execution::recover_execution(state, &coordinator, &proof, current).await
        {
            Ok(()) => {
                scope.abort_fenced_leaves().await;
            }
            Err(recovery) => {
                return Err(recovery);
            }
        }
        return Err(format!(
            "maintenance storage fenced after ambiguous commit: {error}"
        ));
    }
    let terminal = running.finished(outcome.as_ref().map(|_| 0).map_err(Clone::clone));
    if !coordinator
        .finish(&running, outcome.as_ref().map(|_| 0).map_err(Clone::clone))
        .await?
    {
        return Err("maintenance ownership changed before completion".into());
    }
    if !coordinator.release(&proof, &terminal).await? {
        return Err("maintenance completion claim lost".into());
    }
    // A successful WAL catch-up reclaimed generations. Publish the merged
    // watermark per shard *after* release, detached and bounded, so the
    // release path never waits on a storage read, never contends with
    // writer flushes on the demand keys, and never exceeds the etcd
    // transaction size for a table with many shards. Each shard is its own
    // monotonic per-dimension join, so a lost or late publish cannot
    // misstate demand; the next scan reconciles it. Other maintenance kinds
    // do not move WAL watermarks.
    if maintenance == MaintenanceKind::Catchup && outcome.is_ok() {
        crate::demand_publish::spawn_merged_demand(
            coordinator.clone(),
            running.target.clone(),
            uri.clone(),
        );
    }
    outcome
}

// If storage is unavailable, preserve ownership and return the scheduler slot.
// Only these metadata commit leaves survive cancellation; payload work was
// dropped. A later successful barrier authorizes aborting the old leaves.
fn retain_until_fenced(
    coordinator: Coordinator,
    execution: Execution,
    scope: Arc<MergeWriteScope>,
) {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = scope.drain() => return,
                _ = tokio::time::sleep(Duration::from_secs(5)) => {}
            }
            match coordinator.get(&execution.target).await {
                Ok(None) => {
                    scope.abort_fenced_leaves().await;
                    return;
                }
                Ok(Some(current))
                    if current.id != execution.id || current.phase == Phase::Recovered =>
                {
                    scope.abort_fenced_leaves().await;
                    return;
                }
                _ => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::MasterConfig, task_store::TaskKinds};
    use clap::Parser;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ScriptedProgress {
        replies: std::collections::VecDeque<Result<bool, String>>,
        calls: usize,
    }

    impl ProgressPublisher for ScriptedProgress {
        async fn publish(
            &mut self,
            _sequence: u64,
            _report: serde_json::Value,
        ) -> Result<bool, String> {
            self.calls += 1;
            self.replies.pop_front().unwrap_or(Ok(true))
        }
    }

    #[tokio::test]
    async fn progress_publication_retry_is_bounded_and_revocation_is_terminal() {
        let mut execution = Execution::new("table", "master:catchup", "test", 60);
        execution.idle_timeout_secs = 10;
        let scope = MergeWriteScope::new();
        for (replies, expected, calls) in [
            (
                vec![Err("temporary etcd timeout".into()), Ok(true), Ok(false)],
                "ownership revoked",
                3,
            ),
            (
                vec![Err("first timeout".into()), Err("second timeout".into())],
                "publication failed twice",
                2,
            ),
            (vec![Ok(false)], "ownership revoked", 1),
        ] {
            let mut publisher = ScriptedProgress {
                replies: replies.into(),
                calls: 0,
            };
            let error = tokio::time::timeout(
                Duration::from_secs(6),
                watch_with_publisher(&execution, &scope, "executor", "task", &mut publisher),
            )
            .await
            .unwrap();
            assert!(error.contains(expected), "{error}");
            assert_eq!(publisher.calls, calls);
        }
    }

    #[tokio::test]
    async fn progress_reports_coalesce_real_work_without_hiding_new_progress() {
        struct Reports(tokio::sync::mpsc::UnboundedSender<serde_json::Value>);
        impl ProgressPublisher for Reports {
            async fn publish(
                &mut self,
                _sequence: u64,
                report: serde_json::Value,
            ) -> Result<bool, String> {
                self.0.send(report).unwrap();
                Ok(true)
            }
        }
        let mut execution = Execution::new("table", "master:catchup", "test", 60);
        execution.idle_timeout_secs = 60;
        let scope = MergeWriteScope::new();
        let (tx, mut reports) = tokio::sync::mpsc::unbounded_channel();
        let mut publisher = Reports(tx);
        let watch = watch_with_publisher(&execution, &scope, "executor", "task", &mut publisher);
        let producer = async {
            loop {
                scope
                    .run(async {
                        lance_context_core::merge_write_scope::checkpoint();
                    })
                    .await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        };
        let verify = async {
            let first = reports.recv().await.unwrap();
            let first_sequence = first["sequence"].as_u64().unwrap();
            assert!(first_sequence > 0);
            let started = tokio::time::Instant::now();
            // Work continues, but repeated samples need not hit etcd.
            assert!(tokio::time::timeout(Duration::from_secs(3), reports.recv())
                .await
                .is_err());
            assert!(scope.completed_steps() > first_sequence);
            let next = tokio::time::timeout(Duration::from_secs(4), reports.recv())
                .await
                .unwrap()
                .unwrap();
            assert!(started.elapsed() >= Duration::from_secs(4));
            assert!(next["sequence"].as_u64().unwrap() > first_sequence);
            assert!(next["last_progress_at_ms"].as_u64() > first["last_progress_at_ms"].as_u64());
        };
        tokio::time::timeout(Duration::from_secs(10), async {
            tokio::select! {
                error = watch => panic!("progressing work stopped: {error}"),
                _ = producer => unreachable!(),
                _ = verify => {},
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn progress_resuming_after_quiet_does_not_wait_for_coalescing() {
        struct ResumeAfterQuiet<'a> {
            calls: usize,
            scope: &'a Arc<MergeWriteScope>,
        }
        impl ProgressPublisher for ResumeAfterQuiet<'_> {
            async fn publish(
                &mut self,
                _sequence: u64,
                _report: serde_json::Value,
            ) -> Result<bool, String> {
                self.calls += 1;
                if self.calls == 2 {
                    // Resume just after a quiet freshness publication. The
                    // next local observation must not wait another five seconds.
                    self.scope
                        .run(async {
                            lance_context_core::merge_write_scope::checkpoint();
                        })
                        .await;
                }
                Ok(self.calls < 3)
            }
        }
        let mut execution = Execution::new("table", "master:catchup", "test", 60);
        execution.idle_timeout_secs = 60;
        let scope = MergeWriteScope::new();
        let mut publisher = ResumeAfterQuiet {
            calls: 0,
            scope: &scope,
        };
        let error = tokio::time::timeout(
            Duration::from_secs(9),
            watch_with_publisher(&execution, &scope, "executor", "task", &mut publisher),
        )
        .await
        .unwrap();
        assert!(error.contains("ownership revoked"));
        assert_eq!(publisher.calls, 3);
    }

    #[tokio::test]
    async fn quiet_reports_are_coalesced_without_extending_the_idle_deadline() {
        let mut execution = Execution::new("table", "master:catchup", "test", 60);
        execution.idle_timeout_secs = 8;
        let scope = MergeWriteScope::new();
        let mut publisher = ScriptedProgress {
            replies: Default::default(),
            calls: 0,
        };
        let error = tokio::time::timeout(
            Duration::from_secs(10),
            watch_with_publisher(&execution, &scope, "executor", "task", &mut publisher),
        )
        .await
        .unwrap();
        assert!(error.contains("no-progress"));
        assert!(
            (2..=4).contains(&publisher.calls),
            "quiet work must publish bounded freshness updates without writing every second: {}",
            publisher.calls
        );
    }

    #[tokio::test]
    async fn progress_publication_retry_does_not_reset_no_progress_deadline() {
        let mut execution = Execution::new("table", "master:catchup", "test", 60);
        execution.idle_timeout_secs = 2;
        let scope = MergeWriteScope::new();
        let mut publisher = ScriptedProgress {
            replies: vec![Err("temporary etcd timeout".into())].into(),
            calls: 0,
        };
        let error = tokio::time::timeout(
            Duration::from_secs(4),
            watch_with_publisher(&execution, &scope, "executor", "task", &mut publisher),
        )
        .await
        .unwrap();
        assert!(error.contains("no-progress"), "{error}");
        assert_eq!(publisher.calls, 1, "retry must not keep idle work alive");
    }

    #[tokio::test]
    async fn legacy_file_progress_is_visible_without_masking_an_idle_operation() {
        use crate::{admission::Admission, merge_capacity::Capacity};
        use lance_context_core::{
            ColumnSpec, ColumnType, GenericStore, GenericStoreOptions, ListSource, SchemaSpec,
        };
        use tokio::sync::{Notify, Semaphore};

        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let schema = SchemaSpec::new(vec![(
            "id".into(),
            ColumnSpec::required(ColumnType::String { large: false }),
        )]);
        let mut store = GenericStore::open(uri, schema, GenericStoreOptions::default())
            .await
            .unwrap();
        store
            .add(&[serde_json::json!({"id": "preserved"})
                .as_object()
                .unwrap()
                .clone()])
            .await
            .unwrap();
        store.flush().await.unwrap();
        store.cleanup_wal().await.unwrap();
        store.close().await.unwrap();

        let admission = Arc::new(Admission::default());
        let pool = Arc::new(Semaphore::new(2));
        let mut capacities = Vec::new();
        for name in ["reading", "idle"] {
            let operation = Arc::new(admission.try_admit_named("compact").unwrap());
            operation.task(name, name);
            capacities.push(Capacity::new(
                pool.clone().acquire_owned().await.unwrap(),
                pool.clone(),
                Arc::new(Semaphore::new(1)),
                Duration::from_secs(1),
                metrics::gauge!("test_maintenance_progress_running"),
                operation,
            ));
        }
        let read_done = Notify::new();
        let last_work_completed_at = std::sync::atomic::AtomicI64::new(0);
        let release = Semaphore::new(0);
        let busy = crate::merge_capacity::scope(
            capacities[0].clone(),
            observe_legacy(async {
                for step in 0..5 {
                    let mut store =
                        GenericStore::open_existing(uri, GenericStoreOptions::default())
                            .await
                            .unwrap();
                    // The observation scope must preserve real legacy commits,
                    // not just allow reads or record synthetic checkpoints.
                    store
                        .add(&[serde_json::json!({"id": format!("new-{step}")})
                            .as_object()
                            .unwrap()
                            .clone()])
                        .await
                        .unwrap();
                    store.flush().await.unwrap();
                    store.cleanup_wal().await.unwrap();
                    let rows = store.list_from(ListSource::Fragments).await.unwrap();
                    assert_eq!(rows.len(), step + 2);
                    assert!(rows.iter().any(|row| row["id"] == "preserved"));
                    store.close().await.unwrap();
                    last_work_completed_at
                        .store(chrono::Utc::now().timestamp_millis(), Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(350)).await;
                }
                read_done.notify_one();
                release.acquire().await.unwrap().forget();
            }),
        );
        let idle = crate::merge_capacity::scope(
            capacities[1].clone(),
            observe_legacy(async { release.acquire().await.unwrap().forget() }),
        );
        let inspect = async {
            read_done.notified().await;
            let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
            let before = loop {
                let status = admission.status();
                let reading = status
                    .operations
                    .values()
                    .find(|op| op.task_id.as_deref() == Some("reading"))
                    .unwrap();
                let idle = status
                    .operations
                    .values()
                    .find(|op| op.task_id.as_deref() == Some("idle"))
                    .unwrap();
                assert!(
                    idle.progress.is_none(),
                    "another operation's IO cannot count here"
                );
                assert_eq!(idle.phase, "legacy_maintenance");
                if let Some(progress) = &reading.progress {
                    if progress.completed >= 2
                        && progress.last_completed_at_ms
                            >= last_work_completed_at.load(Ordering::SeqCst)
                    {
                        assert_eq!(progress.unit, "work_progress_samples");
                        break serde_json::to_value(progress).unwrap();
                    }
                }
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "real file reads were not reported"
                );
                tokio::time::sleep(Duration::from_millis(50)).await;
            };
            tokio::time::sleep(Duration::from_millis(1200)).await;
            let status = admission.status();
            let reading = status
                .operations
                .values()
                .find(|op| op.task_id.as_deref() == Some("reading"))
                .unwrap();
            assert_eq!(
                serde_json::to_value(reading.progress.as_ref().unwrap()).unwrap(),
                before,
                "polling without completed IO must not refresh progress"
            );
            assert_eq!(
                status.active_operations, 2,
                "quiet legacy work must not be cancelled"
            );
            release.add_permits(2);
        };
        tokio::time::timeout(Duration::from_secs(15), async {
            tokio::join!(busy, idle, inspect);
        })
        .await
        .unwrap();
    }

    #[test]
    fn legacy_maintenance_fencing_does_not_enroll_worker_merges() {
        let default = MasterConfig::parse_from(["master"]);
        assert!(!fenced(&default, "new-hot-table"));
        let enabled = MasterConfig::parse_from(["master", "--fence-legacy-maintenance"]);
        assert!(fenced(&enabled, "new-hot-table"));
        assert!(fenced(&enabled, "generic:new-hot-table"));
        assert!(!fenced(&enabled, "datagen:legacy"));
        assert!(!enabled.merge_rollout.owned("new-hot-table"));
        assert!(!enabled.merge_rollout.owned("generic:new-hot-table"));
    }

    async fn fixture() -> (tempfile::TempDir, Arc<MasterState>) {
        configured_fixture(true, false).await
    }

    async fn configured_fixture(
        owned: bool,
        fence_legacy: bool,
    ) -> (tempfile::TempDir, Arc<MasterState>) {
        fixture_with_idle(owned, fence_legacy, 1).await
    }

    async fn fixture_with_idle(
        owned: bool,
        fence_legacy: bool,
        idle: u64,
    ) -> (tempfile::TempDir, Arc<MasterState>) {
        let dir = tempfile::tempdir().unwrap();
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let prefix = format!("/maintenance-test/{}", lance_context_core::generate_id());
        let mut config = MasterConfig::parse_from([
            "master",
            "--data-dir",
            dir.path().to_str().unwrap(),
            "--etcd-endpoints",
            &endpoints,
            "--etcd-prefix",
            &prefix,
        ]);
        if owned {
            config.merge_rollout.owned_targets = vec!["table".into()];
        }
        config.maintenance.fence_legacy_maintenance = fence_legacy;
        config.stats_scan_interval_secs = 0;
        config.compaction_interval_secs = 0;
        config.merge_wal_interval_secs = 0;
        config.maintenance.maintenance_timeout_secs = 60;
        config.maintenance.maintenance_idle_timeout_secs = idle;
        config.maintenance.maintenance_drain_timeout_secs = 1;
        (dir, MasterState::new(config).await.unwrap())
    }

    async fn claim(state: &Arc<MasterState>, kind: TaskKind) -> TaskClaim {
        crate::scheduler::enqueue(state, kind, "table")
            .await
            .unwrap();
        state.task_store.claim_next().await.unwrap().unwrap()
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS; ~8 s"]
    async fn legacy_maintenance_file_io_progress_survives_multiple_idle_windows() {
        use lance_context_core::{
            ColumnSpec, ColumnType, GenericStore, GenericStoreOptions, ListSource, SchemaSpec,
        };
        let (_dir, state) = fixture_with_idle(false, true, 3).await;
        let uri = state.rollout_uri("table");
        let schema = SchemaSpec::new(vec![
            (
                "id".into(),
                ColumnSpec::required(ColumnType::String { large: false }),
            ),
            (
                "content".into(),
                ColumnSpec::required(ColumnType::String { large: false }),
            ),
        ]);
        let mut store = GenericStore::open(
            &uri,
            schema,
            GenericStoreOptions {
                merge_after_generations: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let rows: Vec<_> = (0..512)
            .map(|n| {
                serde_json::json!({"id": n.to_string(), "content": "x".repeat(4096)})
                    .as_object()
                    .unwrap()
                    .clone()
            })
            .collect();
        store.add(&rows).await.unwrap();
        store.flush().await.unwrap();
        store.cleanup_wal().await.unwrap();
        store.close().await.unwrap();
        let claim = claim(&state, TaskKind::IndexId).await;
        let coordinator = state.task_store.merge_coordinator();
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            run(&state, &claim, async {
                // No logical checkpoint calls: only completed reads through the
                // scoped stores may keep this operation alive. Reopen to avoid
                // turning later scans into memory-cache hits with no IO.
                let mut observed_io = false;
                for step in 0..14 {
                    let store = GenericStore::open_existing(&uri, GenericStoreOptions::default())
                        .await
                        .map_err(|e| e.to_string())?;
                    let rows = store
                        .list_from(ListSource::Fragments)
                        .await
                        .map_err(|e| e.to_string())?;
                    assert_eq!(rows.len(), 512);
                    if step > 3 {
                        let execution = coordinator.get("table").await?.unwrap();
                        if let Some(report) = coordinator.work_progress(&execution).await? {
                            observed_io |= report["sequence"].as_u64().unwrap() > 0;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                assert!(observed_io, "the watchdog must publish completed file IO");
                Ok("file IO kept the operation alive".into())
            }),
        )
        .await
        .unwrap();
        assert!(result.is_ok(), "{result:?}");
        assert!(coordinator.get("table").await.unwrap().is_none());
        state.task_store.finish(claim, result).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn coalesced_progress_reaches_etcd_and_preserves_revocation_cas() {
        let (_dir, state) = fixture().await;
        let claim = claim(&state, TaskKind::MergeWal).await;
        let coordinator = state.task_store.merge_coordinator();
        let mut execution = Execution::new("table", "master:catchup", "test", 60);
        execution.idle_timeout_secs = 60;
        assert!(coordinator
            .reserve(&state.task_store.merge_claim(&claim), &execution)
            .await
            .unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        let scope = MergeWriteScope::new();
        let watchdog = watch(
            &coordinator,
            &running,
            &scope,
            "executor-test",
            &claim.task.id,
        );
        let verify = async {
            let first = loop {
                if let Some(report) = coordinator.work_progress(&running).await.unwrap() {
                    break report;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            };
            let first_sequence = first["sequence"].as_u64().unwrap();
            for _ in 0..20 {
                scope
                    .run(async {
                        lance_context_core::merge_write_scope::checkpoint();
                    })
                    .await;
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
            assert_eq!(
                coordinator.work_progress(&running).await.unwrap().unwrap(),
                first,
                "local progress samples must not rewrite the report every second"
            );
            let latest = loop {
                let report = coordinator.work_progress(&running).await.unwrap().unwrap();
                if report["sequence"].as_u64() == Some(scope.completed_steps()) {
                    break report;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            };
            assert!(latest["sequence"].as_u64().unwrap() > first_sequence);
            assert!(
                !coordinator
                    .revoke_stalled(&running, Some(first_sequence))
                    .await
                    .unwrap(),
                "an old observation cannot revoke newly published progress"
            );
            assert!(coordinator.finish(&running, Ok(0)).await.unwrap());
        };
        let (error, ()) = tokio::time::timeout(Duration::from_secs(20), async {
            tokio::join!(watchdog, verify)
        })
        .await
        .unwrap();
        assert!(error.contains("ownership revoked"));
        let terminal = coordinator.get("table").await.unwrap().unwrap();
        assert!(coordinator
            .release(&state.task_store.merge_claim(&claim), &terminal)
            .await
            .unwrap());
        state
            .task_store
            .finish(claim, Ok("coalescing fixture completed".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn work_report_heartbeats_do_not_extend_actual_progress_deadline() {
        let (_dir, state) = fixture().await;
        let claim = claim(&state, TaskKind::MergeWal).await;
        let coordinator = state.task_store.merge_coordinator();
        let mut execution = Execution::new("table", "master:catchup", "test", 60);
        execution.idle_timeout_secs = 5;
        assert!(coordinator
            .reserve(&state.task_store.merge_claim(&claim), &execution)
            .await
            .unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        let scope = MergeWriteScope::new();
        let watchdog = watch(
            &coordinator,
            &running,
            &scope,
            "executor-test",
            &claim.task.id,
        );
        let producer = async {
            // Make real checkpoint progress beyond the entire idle allowance.
            for _ in 0..12 {
                scope
                    .run(async {
                        lance_context_core::merge_write_scope::checkpoint();
                    })
                    .await;
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            let moving = coordinator.work_progress(&running).await.unwrap().unwrap();
            assert_eq!(moving["executor_id"], "executor-test");
            assert_eq!(moving["task_id"], claim.task.id);
            // Wait for actual publications instead of assuming an etcd RPC
            // plus a one-second timer always completes in 1.1 wall seconds.
            let first = loop {
                let report = coordinator.work_progress(&running).await.unwrap().unwrap();
                if report["sequence"].as_u64() == Some(scope.completed_steps()) {
                    break report;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            };
            let second = loop {
                let report = coordinator.work_progress(&running).await.unwrap().unwrap();
                if report["observed_at_ms"].as_u64() > first["observed_at_ms"].as_u64() {
                    break report;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            };
            assert_eq!(first["sequence"], second["sequence"]);
            assert_eq!(first["last_progress_at_ms"], second["last_progress_at_ms"]);
            assert!(
                second["observed_at_ms"].as_u64().unwrap()
                    > first["observed_at_ms"].as_u64().unwrap()
            );
        };
        let (error, ()) = tokio::time::timeout(Duration::from_secs(30), async {
            tokio::join!(watchdog, producer)
        })
        .await
        .unwrap();
        assert!(error.contains("no-progress"));
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn stalled_local_kinds_cancel_and_release_without_stopping_other_tables() {
        assert_stalled_local_kinds(true, false).await;
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn legacy_stalled_local_kinds_cancel_and_release_without_stopping_other_tables() {
        assert_stalled_local_kinds(false, true).await;
    }

    async fn assert_stalled_local_kinds(owned: bool, fence_legacy: bool) {
        struct Active(Arc<AtomicBool>);
        impl Drop for Active {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let (_dir, state) = configured_fixture(owned, fence_legacy).await;
        for kind in [TaskKind::IndexId, TaskKind::Compact, TaskKind::Repair] {
            let owned = claim(&state, kind).await;
            let active = Arc::new(AtomicBool::new(true));
            let guard = Active(active.clone());
            let result = run(&state, &owned, async move {
                let _guard = guard;
                std::future::pending::<Result<String, String>>().await
            });
            let other = async {
                let task = crate::scheduler::enqueue(&state, TaskKind::Compact, "healthy")
                    .await
                    .unwrap();
                let healthy = state.task_store.claim_next().await.unwrap().unwrap();
                assert_eq!(healthy.task.id, task.id);
                state
                    .task_store
                    .finish(healthy, Ok("healthy continued".into()))
                    .await
                    .unwrap();
            };
            let (result, ()) = tokio::time::timeout(Duration::from_secs(15), async {
                tokio::join!(result, other)
            })
            .await
            .expect("a no-progress operation must not strand the master");
            assert!(result.as_ref().unwrap_err().contains("no-progress"));
            assert!(!active.load(Ordering::SeqCst));
            assert!(state
                .task_store
                .merge_coordinator()
                .get("table")
                .await
                .unwrap()
                .is_none());
            state.task_store.finish(owned, result).await.unwrap();
            let failure = state
                .task_store
                .merge_coordinator()
                .failure("table", super::kind(kind).unwrap().endpoint())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(failure.consecutive_attempts, 1);
            assert!(failure.next_retry_ms > failure.last_failure_ms);
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn short_retry_is_honored_without_failing_another_pass() {
        let (_dir, state) = fixture().await;
        let first = claim(&state, TaskKind::IndexId).await;
        let result = run(&state, &first, async {
            Err("temporary storage timeout".into())
        })
        .await;
        state.task_store.finish(first, result).await.unwrap();
        let coordinator = state.task_store.merge_coordinator();
        let before = coordinator
            .failure("table", "master:index_id")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(before.consecutive_attempts, 1);
        assert_eq!(before.next_retry_ms - before.last_failure_ms, 2000);
        let next = claim(&state, TaskKind::IndexId).await;
        let result = run(&state, &next, async {
            assert!(lance_context_merge::failure::now_ms() >= before.next_retry_ms);
            Ok("continued after retry deadline".into())
        })
        .await;
        assert!(result.is_ok(), "{result:?}");
        state.task_store.finish(next, result).await.unwrap();
        assert!(coordinator
            .failure("table", "master:index_id")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn recovered_catchup_waits_for_newly_materialized_failure() {
        let (_dir, state) = fixture().await;
        let uri = state.rollout_uri("table");
        let store = lance_context_core::RolloutStore::open(&uri).await.unwrap();
        let old_claim = claim(&state, TaskKind::MergeWal).await;
        let coordinator = state.task_store.merge_coordinator();
        let endpoint = MaintenanceKind::Catchup.endpoint();
        let mut old = Execution::new("table", endpoint, "old-publisher", 60);
        old.maintenance = Some(MaintenanceKind::Catchup);
        assert!(coordinator
            .reserve(&state.task_store.merge_claim(&old_claim), &old)
            .await
            .unwrap());
        let old = coordinator.start(&old).await.unwrap().unwrap();
        coordinator
            .authorize_commit(&old, &uri, "base", store.version() + 1)
            .await
            .unwrap();
        coordinator
            .report_uncertain(&old, "maintenance progress publication timeout".into())
            .await
            .unwrap();
        assert!(coordinator
            .failure("table", endpoint)
            .await
            .unwrap()
            .is_none());
        state
            .task_store
            .abandon_claim_for_test(old_claim)
            .await
            .unwrap();
        let next = claim(&state, TaskKind::MergeWal).await;
        reconcile_previous(&state, &next).await.unwrap();
        let failure = coordinator
            .failure("table", endpoint)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(failure.consecutive_attempts, 1);
        assert_eq!(failure.next_retry_ms - failure.last_failure_ms, 2000);
        let result = run_as(&state, &next, MaintenanceKind::Catchup, async {
            assert!(lance_context_merge::failure::now_ms() >= failure.next_retry_ms);
            Ok("recovered publisher continued in the same pass".into())
        })
        .await;
        assert!(result.is_ok(), "{result:?}");
        state.task_store.finish(next, result).await.unwrap();
        assert!(coordinator
            .failure("table", endpoint)
            .await
            .unwrap()
            .is_none());
        assert!(coordinator
            .authorize_commit(&old, &uri, "base", store.version() + 2)
            .await
            .is_err());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn short_retry_wait_rechecks_an_extended_failure_deadline() {
        let (_dir, state) = fixture().await;
        let first = claim(&state, TaskKind::IndexId).await;
        let result = run(&state, &first, async {
            Err("temporary storage timeout".into())
        })
        .await;
        state.task_store.finish(first, result).await.unwrap();
        let coordinator = state.task_store.merge_coordinator();
        let next = claim(&state, TaskKind::IndexId).await;
        let proof = state.task_store.merge_claim(&next);
        let update = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            coordinator
                .record_failure(&proof, "table", "master:index_id", "invalid index schema")
                .await
                .unwrap()
        };
        let run = run(&state, &next, async {
            panic!("extended backoff must prevent payload work")
        });
        let (result, extended) = tokio::join!(run, update);
        assert!(result.as_ref().unwrap_err().contains("retry at"));
        state.task_store.finish(next, result).await.unwrap();
        let after = coordinator
            .failure("table", "master:index_id")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.consecutive_attempts, extended.consecutive_attempts);
        assert_eq!(after.next_retry_ms, extended.next_retry_ms);
        assert!(coordinator.get("table").await.unwrap().is_none());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn a_new_task_cannot_bypass_local_failure_backoff() {
        let (_dir, state) = fixture().await;
        let first = claim(&state, TaskKind::IndexId).await;
        let outcome = run(&state, &first, async { Err("invalid index schema".into()) }).await;
        state.task_store.finish(first, outcome).await.unwrap();
        let coordinator = state.task_store.merge_coordinator();
        let before = coordinator
            .failure("table", "master:index_id")
            .await
            .unwrap()
            .unwrap();
        assert!(before.needs_attention);
        let fresh = claim(&state, TaskKind::IndexId).await;
        let result = run(&state, &fresh, async {
            panic!("backoff must prevent new payload work")
        })
        .await;
        assert!(result.as_ref().unwrap_err().contains("retry at"));
        state.task_store.finish(fresh, result).await.unwrap();
        let after = coordinator
            .failure("table", "master:index_id")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(before.consecutive_attempts, after.consecutive_attempts);
        assert_eq!(before.next_retry_ms, after.next_retry_ms);
        assert!(coordinator.get("table").await.unwrap().is_none());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn lost_master_index_is_fenced_before_merge_takes_over() {
        assert_lost_master_fenced(true, false).await;
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn legacy_lost_master_index_is_fenced_before_merge_takes_over() {
        assert_lost_master_fenced(false, true).await;
    }

    async fn assert_lost_master_fenced(owned: bool, fence_legacy: bool) {
        let (_dir, state) = configured_fixture(owned, fence_legacy).await;
        let uri = state.rollout_uri("table");
        let store = lance_context_core::RolloutStore::open(&uri).await.unwrap();
        let high = store.version() + 1;
        let old_claim = claim(&state, TaskKind::IndexId).await;
        let coordinator = state.task_store.merge_coordinator();
        let proof = state.task_store.merge_claim(&old_claim);
        let mut old = Execution::new("table", "master:index_id", "old-master", 60);
        old.maintenance = Some(MaintenanceKind::IndexId);
        assert!(coordinator.reserve(&proof, &old).await.unwrap());
        let old = coordinator.start(&old).await.unwrap().unwrap();
        coordinator
            .authorize_commit(&old, &uri, "base", high)
            .await
            .unwrap();
        crate::scheduler::enqueue(&state, TaskKind::MergeWal, "table")
            .await
            .unwrap();
        assert!(
            state
                .task_store
                .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                .await
                .unwrap()
                .is_none(),
            "live reconciler must remain exclusive"
        );
        state
            .task_store
            .abandon_claim_for_test(old_claim)
            .await
            .unwrap();
        let next = state
            .task_store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        reconcile_previous(&state, &next).await.unwrap();
        assert!(coordinator.get("table").await.unwrap().is_none());
        let fresh =
            lance_context_core::RolloutStore::open_existing_with_options(&uri, Default::default())
                .await
                .unwrap();
        assert!(
            fresh.version() > high,
            "all old admitted immutable versions must be occupied"
        );
        assert!(coordinator
            .authorize_commit(&old, &uri, "base", high + 2)
            .await
            .is_err());
        state
            .task_store
            .finish(next, Ok("safe to continue".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn failed_storage_barrier_keeps_ownership_and_retries_do_not_reset_budget() {
        assert_failed_barrier(true).await;
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn legacy_fenced_records_recover_after_flag_rollback_and_retain_failed_barriers() {
        // A previous process admitted this record with fencing enabled; the
        // restarted master has neither owned targets nor the legacy flag.
        assert_failed_barrier(false).await;
    }

    async fn assert_failed_barrier(owned: bool) {
        let (_dir, state) = configured_fixture(owned, false).await;
        let old_claim = claim(&state, TaskKind::Repair).await;
        let coordinator = state.task_store.merge_coordinator();
        let mut old = Execution::new("table", "master:repair", "old-master", 60);
        old.maintenance = Some(MaintenanceKind::Repair);
        assert!(coordinator
            .reserve(&state.task_store.merge_claim(&old_claim), &old)
            .await
            .unwrap());
        let old = coordinator.start(&old).await.unwrap().unwrap();
        // No dataset exists: the metadata recovery barrier must fail closed.
        coordinator
            .authorize_commit(&old, &state.rollout_uri("table"), "base", 1)
            .await
            .unwrap();
        state
            .task_store
            .finish(old_claim, Err("lost ambiguous execution".into()))
            .await
            .unwrap();
        let (rows, _) = coordinator.execution_page(None).await.unwrap();
        assert_eq!(rows.len(), 1);
        // There is deliberately no failure ledger yet: execution inventory
        // alone must schedule recovery even with all stats sweeps disabled.
        enqueue_recovery(&state, &coordinator, &rows[0])
            .await
            .unwrap();
        let retry = state.task_store.claim_next().await.unwrap().unwrap();
        let error = reconcile_previous(&state, &retry).await.unwrap_err();
        assert_eq!(
            coordinator.get("table").await.unwrap().unwrap().phase,
            Phase::Recovering
        );
        assert!(coordinator
            .authorize_commit(&old, &state.rollout_uri("table"), "base", 2)
            .await
            .is_err());
        state.task_store.finish(retry, Err(error)).await.unwrap();
        let failure = coordinator
            .failure("table", "master:repair")
            .await
            .unwrap()
            .unwrap();
        enqueue_recovery(&state, &coordinator, &rows[0])
            .await
            .unwrap();
        assert!(
            state
                .task_store
                .get_active_id(TaskKind::Repair, "table")
                .await
                .unwrap()
                .is_none(),
            "backoff must not recreate a fresh retry budget"
        );
        assert_eq!(
            coordinator
                .failure("table", "master:repair")
                .await
                .unwrap()
                .unwrap()
                .consecutive_attempts,
            failure.consecutive_attempts
        );
    }
}
