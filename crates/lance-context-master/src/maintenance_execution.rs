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
    if !state.config.merge_rollout.owned(&execution.target)
        && !state.config.merge_rollout.draining(&execution.target)
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

async fn watch(
    coordinator: &Coordinator,
    execution: &Execution,
    scope: &MergeWriteScope,
) -> String {
    let mut sequence = scope.completed_steps();
    let mut changed = tokio::time::Instant::now();
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let current = scope.completed_steps();
        if current != sequence {
            changed = tokio::time::Instant::now();
            sequence = current;
        }
        if changed.elapsed() >= Duration::from_secs(execution.idle_timeout_secs) {
            return "maintenance no-progress deadline exceeded".into();
        }
        // Publishing an unchanged sequence checks ownership but does not count
        // as progress. A healthy lease never resets the idle deadline.
        match coordinator.publish_progress(execution, sequence).await {
            Ok(true) => {}
            Ok(false) => return "maintenance ownership revoked".into(),
            Err(error) => return format!("maintenance progress publication failed: {error}"),
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
    if !state.config.merge_rollout.owned(&claim.task.target) {
        return work.await;
    }
    let maintenance = kind(claim.task.kind).ok_or("invalid local maintenance kind")?;
    run_as(state, claim, maintenance, work).await
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
    let uri = match claim.task.target.strip_prefix("generic:") {
        Some(name) => state.generic_uri(name),
        None => state.rollout_uri(&claim.task.target),
    };
    if !lance_context_core::merge_write_scope::supports_version_fencing(&uri) {
        return Err("invalid storage backend for maintenance fencing".into());
    }
    let coordinator = state.task_store.merge_coordinator();
    let proof = state.task_store.merge_claim(claim);
    if let Some(failure) = coordinator
        .failure(&claim.task.target, maintenance.endpoint())
        .await?
    {
        if failure.next_retry_ms > lance_context_merge::failure::now_ms() {
            return Err(format!(
                "maintenance retry at {} after {:?} failure",
                failure.next_retry_ms, failure.class
            ));
        }
    }
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
    let scope = MergeWriteScope::with_pinned_authorizer(Arc::new(Guard {
        coordinator: coordinator.clone(),
        execution: running.clone(),
        uri,
    }));
    let outcome = std::panic::AssertUnwindSafe(async {
        let watched = async {
            tokio::select! {
                result = scope.run(work) => result,
                error = watch(&coordinator, &running, &scope) => Err(error),
            }
        };
        watched.await
    })
    .catch_unwind()
    .await
    .unwrap_or_else(|_| Err("maintenance executor panicked".into()));
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

    async fn fixture() -> (tempfile::TempDir, Arc<MasterState>) {
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
        config.merge_rollout.owned_targets = vec!["table".into()];
        config.stats_scan_interval_secs = 0;
        config.compaction_interval_secs = 0;
        config.merge_wal_interval_secs = 0;
        config.maintenance.maintenance_timeout_secs = 60;
        config.maintenance.maintenance_idle_timeout_secs = 1;
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
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn stalled_local_kinds_cancel_and_release_without_stopping_other_tables() {
        struct Active(Arc<AtomicBool>);
        impl Drop for Active {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let (_dir, state) = fixture().await;
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
            let (result, ()) = tokio::join!(result, other);
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
        let (_dir, state) = fixture().await;
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
        let (_dir, state) = fixture().await;
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
