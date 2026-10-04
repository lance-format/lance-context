//! HTTP handlers only admit/cancel work. The owned executor, not the connection,
//! holds the durable execution fence until the scoped storage future is gone.
use crate::{
    error::AppError,
    routes::{generic, rollouts},
    state::AppState,
};
use axum::{extract::State, http::StatusCode, Json};
use futures::FutureExt;
use lance_context_merge::{execute_scoped, execute_until_cancelled, Coordinator, Execution, Phase};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{watch, Mutex, OnceCell};

struct OwnedCommitGuard {
    coordinator: Coordinator,
    execution: Execution,
    dataset_uri: String,
}
impl std::fmt::Debug for OwnedCommitGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedCommitGuard")
            .field("execution", &self.execution.id)
            .finish()
    }
}
impl lance_context_core::merge_write_scope::CommitAuthorizer for OwnedCommitGuard {
    fn authorize<'a>(
        &'a self,
        resource: &'a str,
        version: u64,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<(), lance_context_core::LanceError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            self.coordinator
                .authorize_commit(&self.execution, &self.dataset_uri, resource, version)
                .await
                .map_err(lance_context_core::LanceError::io)
        })
    }
}

pub struct Executions {
    coordinator: OnceCell<Coordinator>,
    etcd: Option<lance_context_merge::EtcdConfig>,
    pub(crate) rollout: lance_context_merge::rollout::MergeRollout,
    queue_timeout_secs: u64,
    idle_timeout_secs: u64,
    instance: String,
    timeout_secs: u64,
    running: Mutex<HashMap<String, watch::Sender<bool>>>,
}

impl Executions {
    #[cfg(test)]
    pub fn new(coordinator: Option<Coordinator>, timeout_secs: u64) -> Self {
        Self {
            coordinator: OnceCell::new_with(coordinator),
            etcd: None,
            rollout: Default::default(),
            queue_timeout_secs: 600,
            idle_timeout_secs: 600,
            instance: uuid::Uuid::new_v4().to_string(),
            timeout_secs,
            running: Mutex::new(HashMap::new()),
        }
    }
    pub fn configured(
        etcd: lance_context_merge::EtcdConfig,
        rollout: lance_context_merge::rollout::MergeRollout,
        timeout_secs: u64,
        queue_timeout_secs: u64,
        idle_timeout_secs: u64,
    ) -> Self {
        Self {
            coordinator: OnceCell::new(),
            etcd: Some(etcd),
            rollout,
            timeout_secs,
            queue_timeout_secs,
            idle_timeout_secs,
            instance: uuid::Uuid::new_v4().to_string(),
            running: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn owned(&self, target: &str) -> bool {
        self.rollout.owned(target)
    }

    pub(crate) fn legacy_allowed(&self, target: &str) -> Result<(), AppError> {
        if self.owned(target) || self.rollout.draining(target) {
            return Err(AppError::Overloaded(
                "table is draining or requires owned merge protocol".into(),
            ));
        }
        Ok(())
    }

    pub(crate) async fn request_merge(&self, target: &str) -> Result<(), String> {
        self.coordinator()
            .await
            .map_err(|e| format!("{e:?}"))?
            .request_merge(target)
            .await
    }

    /// Invoked after HTTP admission has stopped. Detached executors are not
    /// drained by axum's connection shutdown.
    pub(crate) async fn shutdown(&self) {
        for cancel in self.running.lock().await.values() {
            let _ = cancel.send(true);
        }
        while !self.running.lock().await.is_empty() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn coordinator(&self) -> Result<Coordinator, AppError> {
        self.coordinator
            .get_or_try_init(|| async {
                let config = self.etcd.as_ref().ok_or_else(|| {
                    AppError::Overloaded("owned merge execution requires ETCD_ENDPOINTS".into())
                })?;
                config.connect().await.map_err(AppError::Internal)
            })
            .await
            .cloned()
    }
}

pub async fn capabilities(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    Ok(Json(
        serde_json::json!({"protocol": 2, "instance": state.merge_executions.instance,
        "timeout_secs": state.merge_executions.timeout_secs,
        "queue_timeout_secs": state.merge_executions.queue_timeout_secs,
        "idle_timeout_secs": state.merge_executions.idle_timeout_secs,
        "progress_protocol": 1,
        "rollout_append_protocol": 1,
        "shard_name": state.instance_id,
        "owned_targets": state.merge_executions.rollout.owned_targets,
        "drain_targets": state.merge_executions.rollout.drain_targets}),
    ))
}

pub async fn start(
    State(state): State<Arc<AppState>>,
    Json(execution): Json<Execution>,
) -> Result<StatusCode, AppError> {
    let coordinator = state.merge_executions.coordinator().await?;
    if !state.merge_executions.owned(&execution.target)
        || execution.maintenance.is_some()
        || execution.protocol != 2
        || execution.instance != state.merge_executions.instance
        || execution.phase != Phase::Reserved
        || execution.timeout_secs == 0
        || execution.timeout_secs > state.merge_executions.timeout_secs
        || execution.queue_timeout_secs == 0
        || execution.queue_timeout_secs > state.merge_executions.queue_timeout_secs
        || execution.idle_timeout_secs == 0
        || execution.idle_timeout_secs > state.merge_executions.idle_timeout_secs
    {
        return Err(AppError::InvalidRequest(
            "merge executor incarnation or deadline mismatch".into(),
        ));
    }
    let name = execution
        .target
        .strip_prefix("generic:")
        .unwrap_or(&execution.target);
    lance_context_core::validate_store_name(name).map_err(AppError::InvalidRequest)?;
    let mut running = state.merge_executions.running.lock().await;
    if running.contains_key(&execution.id) {
        return Ok(StatusCode::ACCEPTED);
    }
    let (cancel, cancelled) = watch::channel(false);
    running.insert(execution.id.clone(), cancel);
    // Spawn before any fallible admission I/O. Losing the HTTP connection must
    // never leave a Running fence without an executor (or cancel a live write).
    let owned = state.clone();
    tokio::spawn(async move {
        let result = run(owned.clone(), coordinator, execution.clone(), cancelled).await;
        if let Err(error) = result {
            tracing::error!(id = %execution.id, target = %execution.target, %error, "merge execution ownership unresolved");
        }
        owned
            .merge_executions
            .running
            .lock()
            .await
            .remove(&execution.id);
    });
    Ok(StatusCode::ACCEPTED)
}

async fn run(
    state: Arc<AppState>,
    coordinator: Coordinator,
    execution: Execution,
    cancelled: watch::Receiver<bool>,
) -> Result<(), String> {
    // CAS errors are ambiguous: never retry storage work. Reconciliation sees
    // the durable fence, so an uncertain admission cannot release ownership.
    let running = loop {
        match coordinator.start(&execution).await {
            Ok(Some(running)) => break running,
            Ok(None) => match coordinator.get(&execution.target).await {
                Ok(Some(current))
                    if current.id == execution.id && current.phase == Phase::Running =>
                {
                    break current
                }
                Ok(_) => return Ok(()),
                Err(_) => {}
            },
            Err(error) => {
                tracing::warn!(id = %execution.id, %error, "reconciling uncertain execution admission")
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    };
    let target = execution.target.clone();
    let mut slot = None;
    let uri = match target.strip_prefix("generic:") {
        Some(name) => state.generic_uri(name),
        None => state.rollout_uri(&target),
    };
    let work = async {
        if !lance_context_core::merge_write_scope::supports_version_fencing(&uri) {
            return Err("invalid storage backend for owned merge fencing".into());
        }
        let result = if let Some(name) = target.strip_prefix("generic:") {
            generic::merge_generic_wal_owned(
                State(state.clone()),
                axum::extract::Path(name.to_string()),
            )
            .await
            .map(|Json(reply)| reply["reclaimed"].as_u64().unwrap_or(0) as usize)
        } else {
            rollouts::merge_wal_owned(State(state.clone()), axum::extract::Path(target.clone()))
                .await
                .map(|Json(reply)| reply.reclaimed)
        };
        match result {
            Ok(n) => Ok(n),
            Err(AppError::NotFound(ref message))
                if message == &format!("Rollout store '{}' does not exist", target)
                    || message
                        == &format!(
                            "Generic store '{}' does not exist",
                            target.strip_prefix("generic:").unwrap_or(&target)
                        ) =>
            {
                Ok(0)
            }
            Err(e) => Err(format!("{e:?}")),
        }
    };
    let write_scope = lance_context_core::merge_write_scope::MergeWriteScope::with_authorizer(
        Arc::new(OwnedCommitGuard {
            coordinator: coordinator.clone(),
            execution: running.clone(),
            dataset_uri: uri.clone(),
        }),
    );
    // Erase the nested merge future so adding the independent watchdog does
    // not exceed the compiler's async layout recursion limit.
    let execute: futures::future::BoxFuture<'_, Result<usize, String>> = Box::pin(async {
        // No payload work starts before admission to the process-wide slot.
        slot = execute_scoped(
            async { Ok(state.acquire_merge_slot().await) },
            Duration::from_secs(execution.queue_timeout_secs),
            cancelled.clone(),
        )
        .await
        .map_err(|e| format!("merge queue wait: {e}"))?;
        if !coordinator.publish_progress(&running, 0).await? {
            return Err("merge ownership revoked before execution".into());
        }
        execute_until_cancelled(
            async {
                tokio::select! {
                    result = write_scope.run(work) => result,
                    error = watch_progress(&coordinator, &running, &write_scope) => Err(error),
                }
            },
            cancelled,
        )
        .await
    });
    let outcome = std::panic::AssertUnwindSafe(async {
        tokio::select! {
            result = execute => result,
            error = watch_ownership(&coordinator, &running) => Err(error),
        }
    })
    .catch_unwind()
    .await
    .unwrap_or_else(|_| Err("merge executor panicked".into()));
    // Top-level cancellation cannot acknowledge a write still in flight at
    // object storage. Join those leaf commits before publishing Finished.
    loop {
        tokio::select! {
            biased;
            _ = write_scope.drain() => break,
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                match coordinator.get(&execution.target).await {
                    Ok(None) => {
                        // Only a terminal outcome or a completed storage
                        // barrier permits removal of this execution record.
                        write_scope.abort_fenced_leaves().await;
                        return Ok(());
                    }
                    Ok(Some(current)) if current.id != execution.id || current.phase == Phase::Recovered => {
                        write_scope.abort_fenced_leaves().await;
                        return Ok(());
                    }
                    _ => {},
                }
            }
        }
    }
    // Do not admit another memory-heavy merge into this slot while a
    // cancelled execution is still joining storage commits.
    drop(slot);
    // A completed Rust future can still have an ambiguous remote result.
    // Preserve that distinction durably instead of authorizing another writer.
    let uncertain = write_scope.has_uncertain_commit();
    let mut expected = running.finished(outcome.clone());
    if uncertain {
        expected.phase = Phase::Uncertain;
        expected.reclaimed = 0;
        expected.error = Some(format!(
            "merge ownership unresolved: manifest commit result unknown; {:?}",
            outcome
        ));
    }
    loop {
        let acknowledged = if uncertain {
            coordinator
                .report_uncertain(&running, expected.error.clone().unwrap())
                .await
        } else {
            coordinator.finish(&running, outcome.clone()).await
        };
        match acknowledged {
            Ok(true) => break,
            Ok(false) => match coordinator.get(&execution.target).await {
                Ok(Some(current)) if current == expected => break,
                Ok(None) => return Ok(()),
                Ok(Some(current))
                    if current.id != execution.id
                        || matches!(current.phase, Phase::Recovering | Phase::Recovered) =>
                {
                    return Ok(())
                }
                Ok(_) => return Err("merge execution fence changed unexpectedly".into()),
                Err(error) => {
                    tracing::warn!(id = %execution.id, %error, "reconciling merge outcome acknowledgement")
                }
            },
            Err(error) => {
                tracing::warn!(id = %execution.id, %error, "retrying merge outcome acknowledgement")
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    Ok(())
}

// Check independently of progress and slot admission. A revoked executor may
// be stuck before its first checkpoint, and the HTTP cancel may never arrive.
// This only cancels preparation; the caller still drains or fences admitted
// storage writes before releasing the merge slot and publishing a terminal state.
async fn watch_ownership(coordinator: &Coordinator, execution: &Execution) -> String {
    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;
        match tokio::time::timeout(Duration::from_secs(2), coordinator.get(&execution.target)).await
        {
            Ok(Ok(Some(current))) if current == *execution => {}
            Ok(Ok(_)) => return "merge ownership revoked during execution".into(),
            // An unavailable coordinator is not evidence of a completed fence.
            // Queue and real-progress idle deadlines remain independent.
            _ => {}
        }
    }
}

async fn watch_progress(
    coordinator: &Coordinator,
    execution: &Execution,
    scope: &lance_context_core::merge_write_scope::MergeWriteScope,
) -> String {
    let mut sequence = 0;
    let mut changed = tokio::time::Instant::now();
    let mut published = 0;
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let current = scope.completed_steps();
        if current != sequence {
            sequence = current;
            changed = tokio::time::Instant::now();
        }
        if changed.elapsed() >= Duration::from_secs(execution.idle_timeout_secs) {
            return "merge no-progress deadline exceeded".into();
        }
        if current != published {
            match coordinator.publish_progress(execution, current).await {
                Ok(true) => published = current,
                Ok(false) => return "merge ownership revoked during execution".into(),
                Err(error) => {
                    tracing::warn!(id = %execution.id, %error, "merge progress publication failed")
                }
            }
        }
    }
}

pub async fn cancel(
    State(state): State<Arc<AppState>>,
    Json(execution): Json<Execution>,
) -> Result<StatusCode, AppError> {
    let coordinator = state.merge_executions.coordinator().await?;
    if coordinator
        .cancel_reserved(&execution)
        .await
        .map_err(AppError::Internal)?
    {
        return Ok(StatusCode::OK);
    }
    if let Some(cancel) = state
        .merge_executions
        .running
        .lock()
        .await
        .get(&execution.id)
    {
        let _ = cancel.send(true);
        return Ok(StatusCode::ACCEPTED);
    }
    // Absence from the local map is NOT proof that the old process stopped.
    // The caller must reconcile the durable result, not interpret HTTP 404 as
    // permission for a competing write.
    Err(AppError::NotFound(
        "execution is not owned by this worker incarnation".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_merge::{ClaimProof, EtcdConfig};
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn unavailable_etcd_does_not_block_startup_or_disable_legacy_self_merge() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let config = crate::config::ServerConfig::parse_from([
            "test",
            "--data-dir",
            dir.path().to_str().unwrap(),
            "--etcd-endpoints",
            "http://127.0.0.1:1",
            "--merge-owned-targets",
            "hot",
            "--rollout-merge-after-generations",
            "3",
            "--rollout-cleanup-interval-secs",
            "30",
        ]);
        let state = Arc::new(AppState::new(config).await.unwrap());
        assert!(state.merge_executions.coordinator.get().is_none());
        assert_eq!(state.rollout_merge_after_generations, 3);
        assert_eq!(state.rollout_cleanup_interval_secs, 30);
        assert!(state.merge_executions.legacy_allowed("other").is_ok());
        assert!(state.merge_executions.legacy_allowed("hot").is_err());
        let Json(reply) = capabilities(State(state.clone())).await.unwrap();
        assert_eq!(reply["owned_targets"][0], "hot");
        assert!(state.merge_executions.coordinator.get().is_none());
    }

    async fn deadline_fixture() -> (Arc<AppState>, Coordinator, ClaimProof, tempfile::TempDir) {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let prefix = format!("/server-deadlines/{}", uuid::Uuid::new_v4());
        let proof = ClaimProof {
            key: format!("{prefix}/claim"),
            token: "owner".into(),
            lease_id: 0,
        };
        client
            .put(proof.key.as_str(), proof.token.as_str(), None)
            .await
            .unwrap();
        client
            .put(
                lance_context_merge::target_lock_key(&prefix, "hot"),
                proof.token.as_str(),
                None,
            )
            .await
            .unwrap();
        let coordinator = Coordinator::new(client, prefix);
        let dir = tempfile::tempdir().unwrap();
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        state.merge_slots = Some(Arc::new(tokio::sync::Semaphore::new(1)));
        state.merge_executions = Executions::new(Some(coordinator.clone()), 3600);
        state
            .merge_executions
            .rollout
            .owned_targets
            .push("hot".into());
        let state = Arc::new(state);
        let _ = rollouts::create_rollout_store(
            State(state.clone()),
            Json(lance_context_api::CreateRolloutStoreRequest {
                name: "hot".into(),
                storage_options: None,
            }),
        )
        .await
        .unwrap();
        (state, coordinator, proof, dir)
    }

    async fn terminal(coordinator: &Coordinator) -> Execution {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let current = coordinator.get("hot").await.unwrap().unwrap();
                if current.phase == Phase::Finished {
                    return current;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn queue_wait_longer_than_execution_budget_still_runs_after_slot_release() {
        let (state, coordinator, proof, _dir) = deadline_fixture().await;
        let held = state.acquire_merge_slot().await;
        let mut execution = Execution::new("hot", "worker", &state.merge_executions.instance, 1);
        execution.queue_timeout_secs = 10;
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        start(State(state.clone()), Json(execution.clone()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(1300)).await;
        assert_eq!(
            coordinator.get("hot").await.unwrap().unwrap().phase,
            Phase::Running
        );
        assert!(coordinator.progress(&execution).await.unwrap().is_none());
        drop(held);
        let finished = terminal(&coordinator).await;
        assert!(finished.error.is_none(), "{finished:?}");
        assert!(coordinator.release(&proof, &finished).await.unwrap());
        state.merge_executions.shutdown().await;
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn no_progress_cancels_a_blocked_merge_before_the_total_deadline() {
        let (state, coordinator, proof, _dir) = deadline_fixture().await;
        let store = state.get_or_open_rollout_store("hot").await.unwrap();
        let held = store.write().await;
        let mut execution = Execution::new("hot", "worker", &state.merge_executions.instance, 3600);
        execution.idle_timeout_secs = 1;
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        start(State(state.clone()), Json(execution)).await.unwrap();
        let finished = terminal(&coordinator).await;
        assert!(finished.error.as_ref().unwrap().contains("no-progress"));
        assert!(coordinator.release(&proof, &finished).await.unwrap());
        assert_eq!(state.merge_slots.as_ref().unwrap().available_permits(), 1);
        drop(held);
        state.merge_executions.shutdown().await;
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn running_work_is_not_cancelled_by_legacy_total_timeout() {
        let (state, coordinator, proof, _dir) = deadline_fixture().await;
        let store = state.get_or_open_rollout_store("hot").await.unwrap();
        let held = store.write().await;
        let mut execution = Execution::new("hot", "worker", &state.merge_executions.instance, 1);
        execution.idle_timeout_secs = 10;
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        start(State(state.clone()), Json(execution.clone()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(2200)).await;
        assert_eq!(
            coordinator.get("hot").await.unwrap().unwrap().phase,
            Phase::Running
        );
        drop(held);
        let finished = terminal(&coordinator).await;
        assert!(finished.error.is_none(), "{finished:?}");
        assert!(coordinator.release(&proof, &finished).await.unwrap());
        state.merge_executions.shutdown().await;
    }

    // No cancel HTTP request is sent in either case. Revocation must interrupt
    // both slot wait and a merge that cannot reach even its first checkpoint.
    async fn revoked_executor_exits(queue_blocked: bool) {
        let (state, coordinator, proof, _dir) = deadline_fixture().await;
        let store = state.get_or_open_rollout_store("hot").await.unwrap();
        let held_store = store.write().await;
        let held_slot = if queue_blocked {
            state.acquire_merge_slot().await
        } else {
            None
        };
        let execution = Execution::new("hot", "worker", &state.merge_executions.instance, 3600);
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        start(State(state.clone()), Json(execution.clone()))
            .await
            .unwrap();
        let running = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let current = coordinator.get("hot").await.unwrap().unwrap();
                if current.phase == Phase::Running
                    && (queue_blocked || coordinator.progress(&current).await.unwrap().is_some())
                {
                    break current;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        let frozen = coordinator.freeze(&proof, &running).await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(6), async {
            while !state.merge_executions.running.lock().await.is_empty() {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("revoked executor must stop without waiting for idle/queue timeout");
        assert_eq!(
            state.merge_slots.as_ref().unwrap().available_permits(),
            usize::from(!queue_blocked)
        );
        // Local task exit alone does not unlock the table or acknowledge the
        // storage barrier. An attempted replacement is still rejected.
        assert_eq!(coordinator.get("hot").await.unwrap(), Some(frozen.clone()));
        let replacement = Execution::new("hot", "worker", &state.merge_executions.instance, 3600);
        assert!(!coordinator.reserve(&proof, &replacement).await.unwrap());
        let watermarks = coordinator.watermarks(&frozen).await.unwrap();
        assert!(watermarks.versions.is_empty());
        // There were no admitted writes, so the storage barrier is empty.
        assert!(coordinator.finish_recovery(&proof, &frozen).await.unwrap());
        let recovered = coordinator.get("hot").await.unwrap().unwrap();
        assert!(coordinator.release(&proof, &recovered).await.unwrap());
        drop(held_slot);
        drop(held_store);
        assert!(coordinator.reserve(&proof, &replacement).await.unwrap());
        start(State(state.clone()), Json(replacement))
            .await
            .unwrap();
        let finished = terminal(&coordinator).await;
        assert!(finished.error.is_none(), "{finished:?}");
        assert!(coordinator.release(&proof, &finished).await.unwrap());
        state.merge_executions.shutdown().await;
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn revoked_executor_stops_while_waiting_for_a_slot() {
        revoked_executor_exits(true).await;
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn revoked_executor_stops_without_any_progress() {
        revoked_executor_exits(false).await;
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn disconnected_http_call_is_cancelled_before_ownership_handoff() {
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let prefix = format!("/server-merge-test/{}", uuid::Uuid::new_v4());
        let coordinator = EtcdConfig {
            etcd_endpoints: endpoints.split(',').map(str::to_string).collect(),
            etcd_prefix: prefix.clone(),
            etcd_username: None,
            etcd_password: None,
            etcd_ca_cert: None,
            etcd_client_cert: None,
            etcd_client_key: None,
        }
        .connect()
        .await
        .unwrap();
        // Test-only claim creation through the isolated etcd connection.
        let mut client =
            etcd_client::Client::connect(endpoints.split(',').collect::<Vec<_>>(), None)
                .await
                .unwrap();
        let proof = ClaimProof {
            key: format!("{prefix}/claim"),
            token: "owner".into(),
            lease_id: 0,
        };
        client
            .put(proof.key.clone(), proof.token.clone(), None)
            .await
            .unwrap();
        client
            .put(
                lance_context_merge::target_lock_key(&prefix, "blocked"),
                proof.token.clone(),
                None,
            )
            .await
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        state.merge_executions = Executions::new(Some(coordinator.clone()), 600);
        state
            .merge_executions
            .rollout
            .owned_targets
            .push("blocked".into());
        let state = Arc::new(state);
        let name = "blocked";
        // Create via the real API so registry and resident handle agree.
        let _ = rollouts::create_rollout_store(
            State(state.clone()),
            Json(lance_context_api::CreateRolloutStoreRequest {
                name: name.into(),
                storage_options: None,
            }),
        )
        .await
        .unwrap();
        let store = state.get_or_open_rollout_store(name).await.unwrap();
        let held = store.write().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = crate::routes::router().with_state(state.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let execution = Execution::new(
            name,
            &format!("http://{address}"),
            &state.merge_executions.instance,
            600,
        );
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        let body = serde_json::to_string(&execution).unwrap();
        let mut connection = tokio::net::TcpStream::connect(address).await.unwrap();
        connection.write_all(format!("POST /api/v1/internal/merge-executor/start HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if coordinator.get(name).await.unwrap().unwrap().phase == Phase::Running {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        drop(connection);
        assert_eq!(
            coordinator.get(name).await.unwrap().unwrap().phase,
            Phase::Running
        );
        assert!(!coordinator
            .reserve(&proof, &Execution::new(name, "other", "other", 1))
            .await
            .unwrap());
        cancel(State(state.clone()), Json(execution.clone()))
            .await
            .unwrap();
        let finished = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let current = coordinator.get(name).await.unwrap().unwrap();
                if current.phase == Phase::Finished {
                    break current;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(finished.error.as_ref().unwrap().contains("cancelled"));
        assert!(coordinator.release(&proof, &finished).await.unwrap());
        drop(held);
        let next = Execution::new(name, &execution.endpoint, &execution.instance, 30);
        assert!(coordinator.reserve(&proof, &next).await.unwrap());
        start(State(state.clone()), Json(next)).await.unwrap();
        let done = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let current = coordinator.get(name).await.unwrap().unwrap();
                if current.phase == Phase::Finished {
                    break current;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(done.error.is_none(), "retry must run: {done:?}");
        assert!(coordinator.release(&proof, &done).await.unwrap());
        server.abort();
        client.delete(proof.key, None).await.unwrap();
    }
}

/// This endpoint only writes unreachable immutable files. Dropping its HTTP
/// body cancels preparation; there is no detached manifest publisher to fence.
pub async fn stage_append(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(plan): Json<lance_context_core::rollout_append::AppendPlan>,
) -> Result<axum::response::Response, AppError> {
    use axum::response::IntoResponse;
    use lance_context_core::{
        merge_write_scope::MergeWriteScope,
        rollout_append::{stage, StageEvent},
    };
    lance_context_core::validate_store_name(&name).map_err(AppError::InvalidRequest)?;
    plan.validate().map_err(AppError::from_lance)?;
    if !state.merge_executions.owned(&name) || state.merge_executions.rollout.draining(&name) {
        return Err(AppError::Overloaded(
            "staged append requires an owned, non-draining rollout".into(),
        ));
    }
    let budget = state.merge_budget.clone().ok_or_else(|| {
        AppError::Overloaded("staged append requires a shared merge memory budget".into())
    })?;
    if state.rollout_merge_max_bytes == 0 || plan.max_bytes > state.rollout_merge_max_bytes {
        return Err(AppError::InvalidRequest(
            "staged append exceeds worker byte limit".into(),
        ));
    }
    let scope = Arc::new(MergeWriteScope::default());
    let worker_scope = scope.clone();
    let idle = Duration::from_secs(state.merge_executions.idle_timeout_secs);
    let work = Box::pin(async move {
        let _slot = state.acquire_merge_slot().await;
        worker_scope
            .run(stage(
                &state.rollout_uri(&name),
                plan,
                budget,
                state.rollout_session.clone(),
            ))
            .await
    });
    let stream = futures::stream::unfold(
        Some((work, scope, 0, tokio::time::Instant::now())),
        move |next| async move {
            let (mut work, scope, mut sequence, mut changed) = next?;
            let event = tokio::select! {
                result = &mut work => match result {
                    Ok(part) => StageEvent::Complete(part),
                    Err(error) => StageEvent::Failed(error.to_string()),
                },
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    let current = scope.completed_steps();
                    if current > sequence { sequence = current; changed = tokio::time::Instant::now(); }
                    if changed.elapsed() >= idle { StageEvent::Failed("staged append made no progress".into()) }
                    else { StageEvent::Progress(sequence) }
                }
            };
            let terminal = !matches!(event, StageEvent::Progress(_));
            let mut bytes = serde_json::to_vec(&event).expect("serializable stage event");
            bytes.push(b'\n');
            let next = if terminal {
                None
            } else {
                Some((work, scope, sequence, changed))
            };
            Some((Ok::<_, std::convert::Infallible>(bytes), next))
        },
    );
    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
        axum::body::Body::from_stream(stream),
    )
        .into_response())
}

#[cfg(test)]
mod append_tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use lance_context_core::{
        rollout_append::{AppendCoordinator, StageEvent},
        MergeMemoryBudget, RolloutStore, RolloutStoreOptions,
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn staging_http_writes_only_files_and_uses_shared_budget() {
        let dir = tempfile::tempdir().unwrap();
        let mut state =
            AppState::new_for_test_with_instance(dir.path().to_path_buf(), Some("a".into())).await;
        state
            .merge_executions
            .rollout
            .owned_targets
            .push("hot".into());
        state.merge_budget = Some(MergeMemoryBudget::new(4 * 1024 * 1024));
        let uri = state.rollout_uri("hot");
        let store = RolloutStore::open_with_options(
            &uri,
            RolloutStoreOptions {
                shard_id: Some("a".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let dto = serde_json::from_value(
            serde_json::json!({"id":"a", "rollout_id":"r", "content":"hello"}),
        )
        .unwrap();
        store
            .add(&[lance_context_core::rollout_record_from_add_request(&dto)])
            .await
            .unwrap();
        store.flush().await.unwrap();
        let mut coordinator = AppendCoordinator::open(&uri, None).await.unwrap();
        let plan = coordinator
            .plan(&["a".into()], 1, 1024 * 1024)
            .await
            .unwrap()
            .0
            .remove(0);
        let version = coordinator.version();
        let state = Arc::new(state);
        let request = Request::builder()
            .method("POST")
            .uri("/api/v1/internal/rollout-append/hot")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&plan).unwrap()))
            .unwrap();
        let response = crate::routes::router()
            .with_state(state.clone())
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        let event: StageEvent = serde_json::from_slice(
            body.split(|b| *b == b'\n')
                .rfind(|line| !line.is_empty())
                .unwrap(),
        )
        .unwrap();
        let StageEvent::Complete(part) = event else {
            panic!("staging failed: {event:?}")
        };
        assert_eq!(part.rows, 1);
        assert_eq!(state.merge_budget.as_ref().unwrap().reserved(), 0);
        assert_eq!(
            lance::Dataset::open(&uri).await.unwrap().version().version,
            version
        );
        assert_eq!(
            lance::Dataset::open(&uri)
                .await
                .unwrap()
                .count_rows(None)
                .await
                .unwrap(),
            0
        );
        assert_eq!(coordinator.commit(vec![part]).await.unwrap(), 1);
    }
}
