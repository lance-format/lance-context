//! HTTP handlers only admit/cancel work. The owned executor, not the connection,
//! holds the durable execution fence until the scoped storage future is gone.
mod background;
#[cfg(test)]
mod progress_tests;
pub(crate) use background::{publish_background_request, spawn_background};

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
    publisher: lance_context_merge::fencing::CommitPermitPublisher,
    execution: Execution,
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
            self.publisher
                .authorize(resource, version)
                .await
                .map_err(lance_context_core::LanceError::io)
        })
    }
}

#[derive(Default)]
pub(crate) struct WorkerAdmission {
    pub writer_id: Option<String>,
    pub enable_fencing: bool,
}

pub struct Executions {
    pub(crate) legacy: crate::legacy_merge::LegacyMerges,
    coordinator: OnceCell<Coordinator>,
    etcd: Option<lance_context_merge::EtcdConfig>,
    pub(crate) rollout: lance_context_merge::rollout::MergeRollout,
    endpoint: Option<String>,
    worker_admission: WorkerAdmission,
    fencing_mode: OnceCell<bool>,
    mode_pending: std::sync::atomic::AtomicBool,
    background_targets: std::sync::Mutex<std::collections::HashSet<String>>,
    request_targets: std::sync::Mutex<std::collections::HashSet<String>>,
    request_slots: Arc<tokio::sync::Semaphore>,
    queue_timeout_secs: u64,
    idle_timeout_secs: u64,
    instance: String,
    timeout_secs: u64,
    running: Mutex<HashMap<String, watch::Sender<bool>>>,
    // Updated while holding `running`, so no executor can slip past shutdown.
    closing: std::sync::atomic::AtomicBool,
}

impl Executions {
    #[cfg(test)]
    pub fn new(coordinator: Option<Coordinator>, timeout_secs: u64) -> Self {
        Self {
            legacy: Default::default(),
            coordinator: OnceCell::new_with(coordinator),
            etcd: None,
            rollout: Default::default(),
            endpoint: None,
            worker_admission: WorkerAdmission::default(),
            fencing_mode: OnceCell::new(),
            mode_pending: std::sync::atomic::AtomicBool::new(false),
            background_targets: Default::default(),
            request_targets: Default::default(),
            request_slots: Arc::new(tokio::sync::Semaphore::new(4)),
            queue_timeout_secs: 600,
            idle_timeout_secs: 600,
            instance: uuid::Uuid::new_v4().to_string(),
            timeout_secs,
            running: Mutex::new(HashMap::new()),
            closing: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub(crate) fn configured(
        etcd: lance_context_merge::EtcdConfig,
        rollout: lance_context_merge::rollout::MergeRollout,
        endpoint: Option<String>,
        timeout_secs: u64,
        queue_timeout_secs: u64,
        idle_timeout_secs: u64,
        worker_admission: WorkerAdmission,
    ) -> Self {
        Self {
            legacy: Default::default(),
            coordinator: OnceCell::new(),
            etcd: Some(etcd),
            rollout,
            endpoint,
            worker_admission,
            fencing_mode: OnceCell::new(),
            mode_pending: std::sync::atomic::AtomicBool::new(false),
            background_targets: Default::default(),
            request_targets: Default::default(),
            request_slots: Arc::new(tokio::sync::Semaphore::new(4)),
            timeout_secs,
            queue_timeout_secs,
            idle_timeout_secs,
            instance: uuid::Uuid::new_v4().to_string(),
            running: Mutex::new(HashMap::new()),
            closing: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Unknown mode defers maintenance, never ingestion. An activated protocol
    /// is persisted so a configuration rollback cannot reopen untracked writes.
    pub(crate) fn fencing_mode_known(&self) -> Option<bool> {
        if !self.rollout.fence_legacy
            && !self.worker_admission.enable_fencing
            && self.coordinator.get().is_none()
            && self
                .etcd
                .as_ref()
                .is_none_or(|c| c.etcd_endpoints.is_empty())
        {
            return Some(false);
        }
        self.fencing_mode.get().copied()
    }

    pub(crate) async fn fenced_mode(&self) -> Result<bool, AppError> {
        if let Some(mode) = self.fencing_mode_known() {
            return Ok(mode);
        }
        self.fencing_mode.get_or_try_init(|| async {
            let coordinator = self.coordinator().await?;
            let cluster_mode = coordinator.legacy_fencing_mode(self.rollout.fence_legacy)
                .await.map_err(AppError::Internal)?;
            let mode = match self.worker_admission.writer_id.as_deref() {
                Some(writer) => coordinator.worker_fencing_mode(writer, self.worker_admission.enable_fencing)
                    .await.map_err(AppError::Internal)?,
                None if self.worker_admission.enable_fencing => return Err(AppError::InvalidRequest("worker fencing requires a stable writer identity".into())),
                None => cluster_mode,
            };
            if mode && self.endpoint.is_none() {
                return Err(AppError::InvalidRequest("persisted merge protocol requires MERGE_EXECUTOR_ENDPOINT; refusing legacy rollback".into()));
            }
            Ok(mode)
        }).await.copied()
    }

    pub(crate) fn resolve_mode_in_background(state: Arc<AppState>) {
        use std::sync::atomic::Ordering;
        if state.merge_executions.fencing_mode_known().is_some()
            || state
                .merge_executions
                .mode_pending
                .swap(true, Ordering::SeqCst)
        {
            return;
        }
        tokio::spawn(async move {
            if let Err(error) = state.merge_executions.fenced_mode().await {
                tracing::warn!(
                    ?error,
                    "merge protocol unresolved; ingestion continues, untracked merges deferred"
                );
            }
            state
                .merge_executions
                .mode_pending
                .store(false, Ordering::SeqCst);
        });
    }

    pub(crate) fn instance(&self) -> &str {
        &self.instance
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

    pub(crate) async fn publish_demand_event(
        &self,
        target: &str,
        event: &lance_context_merge::demand::DemandEvent,
    ) -> Result<bool, String> {
        self.coordinator()
            .await
            .map_err(|e| format!("{e:?}"))?
            .publish_demand_event(target, event)
            .await
    }

    pub(crate) async fn request_merge(&self, target: &str) -> Result<(), String> {
        self.coordinator()
            .await
            .map_err(|e| format!("{e:?}"))?
            .request_merge(target)
            .await
    }

    /// Close executor admission and request cancellation BEFORE joining HTTP
    /// handlers: a direct merge handler can itself be waiting on an executor.
    /// This does not acknowledge storage completion or discard its durable fence.
    pub(crate) async fn begin_shutdown(&self) {
        let running = self.running.lock().await;
        self.closing
            .store(true, std::sync::atomic::Ordering::Release);
        for cancel in running.values() {
            let _ = cancel.send(true);
        }
    }

    /// Detached executors retain their storage barriers while shutting down.
    pub(crate) async fn shutdown(&self) {
        self.begin_shutdown().await;
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
    Executions::resolve_mode_in_background(state.clone());
    Ok(Json(
        serde_json::json!({"protocol": 2, "instance": state.merge_executions.instance,
        "timeout_secs": state.merge_executions.timeout_secs,
        "queue_timeout_secs": state.merge_executions.queue_timeout_secs,
        "idle_timeout_secs": state.merge_executions.idle_timeout_secs,
        "progress_protocol": 1,
        "legacy_progress_protocol": 1,
        "merge_admission_ready": state.merge_executions.fencing_mode_known().is_some(),
        "fenced_legacy_merges": state.merge_executions.fencing_mode_known() == Some(true),
        "rollout_append_protocol": 1,
        "shard_name": state.instance_id,
        "owned_targets": state.merge_executions.rollout.owned_targets,
        "drain_targets": state.merge_executions.rollout.drain_targets}),
    ))
}

/// A replacement must resolve both the persisted admission mode and the exact
/// old execution from its own coordinator before legacy retirement can proceed.
#[derive(serde::Deserialize)]
pub struct RetirementQuery {
    target: String,
}

pub async fn retirement_status(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(query): axum::extract::Query<RetirementQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    if query.target.is_empty() || query.target.len() > 512 {
        return Err(AppError::InvalidRequest(
            "one explicit retirement target is required".into(),
        ));
    }
    if !state.merge_executions.fenced_mode().await? {
        return Err(AppError::Overloaded(
            "replacement must fence all legacy merge admission".into(),
        ));
    }
    let execution = state
        .merge_executions
        .coordinator()
        .await?
        .get(&query.target)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(serde_json::json!({
        "instance": state.merge_executions.instance,
        "base_uri": state.base_uri,
        "shard_name": state.instance_id,
        "fenced_legacy_merges": true,
        "execution": execution,
    })))
}

pub async fn start(
    State(state): State<Arc<AppState>>,
    Json(execution): Json<Execution>,
) -> Result<StatusCode, AppError> {
    let coordinator = state.merge_executions.coordinator().await?;
    if (!state.merge_executions.owned(&execution.target)
        && !state.merge_executions.fenced_mode().await?)
        || state.merge_executions.rollout.draining(&execution.target)
        || execution.background
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
    if state
        .merge_executions
        .closing
        .load(std::sync::atomic::Ordering::Acquire)
    {
        return Err(AppError::Overloaded(
            "merge executor is shutting down".into(),
        ));
    }
    if running.contains_key(&execution.id) {
        return Ok(StatusCode::ACCEPTED);
    }
    let (cancel, cancelled) = watch::channel(false);
    running.insert(execution.id.clone(), cancel);
    // Spawn before any fallible admission I/O. Losing the HTTP connection must
    // never leave a Running fence without an executor (or cancel a live write).
    let owned = state.clone();
    tokio::spawn(async move {
        let result = run(
            owned.clone(),
            coordinator,
            execution.clone(),
            cancelled,
            None,
        )
        .await;
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
    mut slot: Option<tokio::sync::OwnedSemaphorePermit>,
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
            publisher: coordinator.commit_permit_publisher(&running, &uri),
            execution: running.clone(),
        }),
    );
    // Erase the nested merge future so adding the independent watchdog does
    // not exceed the compiler's async layout recursion limit.
    let execute: futures::future::BoxFuture<'_, Result<usize, String>> = Box::pin(async {
        // No payload work starts before admission to the process-wide slot.
        if slot.is_none() {
            slot = execute_scoped(
                async { Ok(state.acquire_merge_slot().await) },
                Duration::from_secs(execution.queue_timeout_secs),
                cancelled.clone(),
            )
            .await
            .map_err(|e| format!("merge queue wait: {e}"))?;
        }
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
    tokio::select! {
        biased;
        _ = write_scope.drain() => {},
        _ = wait_for_execution_state(&coordinator, &execution.target, |current| {
            // Recovering alone cannot authorize aborting admitted writes.
            current.is_none_or(|current| {
                current.id != execution.id || current.phase == Phase::Recovered
            })
        }) => {
            // Only a terminal outcome or a completed storage barrier permits
            // removal/replacement of this execution record.
            write_scope.abort_fenced_leaves().await;
            return Ok(());
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
    wait_for_execution_state(coordinator, &execution.target, |current| {
        current != Some(execution)
    })
    .await;
    "merge ownership revoked during execution".into()
}

// No periodic read while the owner watch is healthy, and no subscription to
// progress (which would wake on our own publications). Every event/disconnect
// is reconciled with a new authoritative read; the revision covers setup races.
async fn wait_for_execution_state(
    coordinator: &Coordinator,
    target: &str,
    predicate: impl Fn(Option<&Execution>) -> bool,
) {
    loop {
        let observation = async {
            let (current, revision) = tokio::time::timeout(
                Duration::from_secs(2),
                coordinator.get_with_revision(target),
            )
            .await
            .map_err(|e| e.to_string())??;
            if predicate(current.as_ref()) {
                return Ok::<_, String>(true);
            }
            let mut watch = tokio::time::timeout(
                Duration::from_secs(2),
                coordinator.watch_execution_owner(target, revision),
            )
            .await
            .map_err(|e| e.to_string())??;
            watch.changed().await?;
            Ok(false)
        };
        match observation.await {
            Ok(true) => return,
            Ok(false) => {}
            // An unavailable coordinator is not evidence of a completed fence.
            // Queue and real-progress idle deadlines remain independent. Rate
            // limit retries only after errors, never delay an observed change.
            Err(_) => tokio::time::sleep(Duration::from_secs(2)).await,
        }
    }
}

async fn watch_progress(
    coordinator: &Coordinator,
    execution: &Execution,
    scope: &lance_context_core::merge_write_scope::MergeWriteScope,
) -> String {
    watch_progress_with(execution, scope, |sequence| {
        coordinator.publish_progress(execution, sequence)
    })
    .await
}

async fn watch_progress_with<F>(
    execution: &Execution,
    scope: &lance_context_core::merge_write_scope::MergeWriteScope,
    mut publish: impl FnMut(u64) -> F,
) -> String
where
    F: std::future::Future<Output = Result<bool, String>>,
{
    let mut sequence = 0;
    let mut changed = tokio::time::Instant::now();
    let mut published = 0;
    // Sample actual work every second, but coalesce continuous progress writes.
    // Keep several publication opportunities inside even a short idle budget.
    let publish_every = Duration::from_secs((execution.idle_timeout_secs / 4).clamp(1, 5));
    let mut next_publication = tokio::time::Instant::now();
    let mut publication_failed = false;
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let current = scope.completed_steps();
        let resumed = current != sequence && changed.elapsed() >= publish_every;
        if current != sequence {
            sequence = current;
            changed = tokio::time::Instant::now();
        }
        if changed.elapsed() >= Duration::from_secs(execution.idle_timeout_secs) {
            return "merge no-progress deadline exceeded".into();
        }
        if current != published {
            if !publication_failed && !resumed && tokio::time::Instant::now() < next_publication {
                metrics::counter!("worker_merge_progress_coalesced_samples_total").increment(1);
                continue;
            }
            metrics::counter!("worker_merge_progress_publication_attempts_total").increment(1);
            match publish(current).await {
                Ok(true) => {
                    published = current;
                    next_publication = tokio::time::Instant::now() + publish_every;
                    publication_failed = false;
                    metrics::counter!("worker_merge_progress_publications_total", "result" => "success")
                        .increment(1);
                }
                Ok(false) => {
                    metrics::counter!("worker_merge_progress_publications_total", "result" => "revoked")
                        .increment(1);
                    return "merge ownership revoked during execution".into();
                }
                Err(error) => {
                    // Retry at the existing local tick, using the latest work.
                    // Neither a retry nor its acknowledgement extends idle time.
                    publication_failed = true;
                    metrics::counter!("worker_merge_progress_publications_total", "result" => "error")
                        .increment(1);
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

    async fn isolated_etcd_rpc_counts() -> (u64, u64, u64) {
        use tokio::io::AsyncReadExt;
        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").as_deref(), Ok("1"));
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let address = endpoint.strip_prefix("http://127.0.0.1:").unwrap();
        let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{address}"))
            .await
            .unwrap();
        stream
            .write_all(b"GET /metrics HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = String::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_string(&mut response))
            .await
            .unwrap()
            .unwrap();
        assert!(response.starts_with("HTTP/1.0 200") || response.starts_with("HTTP/1.1 200"));
        let count = |method: &str| -> u64 {
            response
                .lines()
                .filter(|line| {
                    line.starts_with("grpc_server_started_total{")
                        && line.contains(&format!("grpc_method=\"{method}\""))
                })
                .map(|line| {
                    line.split_whitespace()
                        .last()
                        .unwrap()
                        .parse::<f64>()
                        .unwrap() as u64
                })
                .sum()
        };
        (count("Range"), count("Watch"), count("Txn"))
    }

    #[tokio::test]
    #[ignore = "requires exclusive isolated local ETCD_TEST_ENDPOINTS"]
    async fn ownership_observer_does_not_poll_or_wake_on_own_progress() {
        let (_state, original, proof, _dir) = deadline_fixture().await;
        let execution = Execution::new("hot", "worker", "boot", 600);
        assert!(original.reserve(&proof, &execution).await.unwrap());
        let running = original.start(&execution).await.unwrap().unwrap();
        // Use the real connection timeouts/keepalive, not just a bare test client.
        let coordinator = EtcdConfig {
            etcd_endpoints: vec![std::env::var("ETCD_TEST_ENDPOINTS").unwrap()],
            etcd_prefix: proof.key.strip_suffix("/claim").unwrap().into(),
            etcd_username: None,
            etcd_password: None,
            etcd_ca_cert: None,
            etcd_client_cert: None,
            etcd_client_key: None,
        }
        .connect()
        .await
        .unwrap();
        let initial = isolated_etcd_rpc_counts().await;
        let observer_coordinator = coordinator.clone();
        let observed = running.clone();
        let mut observer =
            tokio::spawn(async move { watch_ownership(&observer_coordinator, &observed).await });
        tokio::time::timeout(Duration::from_secs(3), async {
            while isolated_etcd_rpc_counts().await.1 == initial.1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let baseline = isolated_etcd_rpc_counts().await;
        // Beyond the configured 10-second RPC timeout, with continuous progress.
        for sequence in 1..=120 {
            assert!(coordinator
                .publish_progress(&running, sequence)
                .await
                .unwrap());
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert!(!observer.is_finished());
        }
        let after = isolated_etcd_rpc_counts().await;
        assert_eq!(
            (after.0, after.1),
            (baseline.0, baseline.1),
            "healthy owner stream must not reread or resubscribe on progress"
        );
        assert_eq!(baseline.0 - initial.0, 1, "one initial owner read");
        assert_eq!(baseline.1 - initial.1, 1, "one owner watch");
        assert!(coordinator
            .freeze(&proof, &running)
            .await
            .unwrap()
            .is_some());
        let result = tokio::time::timeout(Duration::from_secs(2), &mut observer)
            .await
            .unwrap()
            .unwrap();
        assert!(result.contains("revoked"));
    }

    #[tokio::test]
    #[ignore = "requires exclusive isolated local ETCD_TEST_ENDPOINTS"]
    async fn worker_progress_coalesces_real_etcd_writes_and_preserves_owner_watch() {
        use lance_context_core::merge_write_scope::{checkpoint, MergeWriteScope};

        let (_state, coordinator, proof, _dir) = deadline_fixture().await;
        let mut execution = Execution::new("hot", "worker", "boot", 600);
        execution.idle_timeout_secs = 120;
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        let scope = MergeWriteScope::new();
        let initial = isolated_etcd_rpc_counts().await;
        let observer = watch_ownership(&coordinator, &running);
        tokio::pin!(observer);
        let verify = async {
            tokio::time::timeout(Duration::from_secs(3), async {
                while isolated_etcd_rpc_counts().await.1 == initial.1 {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
            let baseline = isolated_etcd_rpc_counts().await;
            let progress = watch_progress(&coordinator, &running, &scope);
            tokio::pin!(progress);
            let work = async {
                // Real local work continues independently of metadata writes.
                for _ in 0..120 {
                    scope.run(async { checkpoint() }).await;
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                let after = isolated_etcd_rpc_counts().await;
                assert_eq!((after.0, after.1), (baseline.0, baseline.1));
                let writes = after.2 - baseline.2;
                assert!((2..=4).contains(&writes), "progress Txns: {writes}");
                let published = coordinator.progress(&running).await.unwrap().unwrap();
                assert!(published.sequence >= 50);
                assert!(published.sequence <= scope.completed_steps());
                assert!(
                    !coordinator.revoke_stalled(&running, Some(0)).await.unwrap(),
                    "an obsolete progress observation cannot revoke current work"
                );
                println!(
                    "worker progress: {} local steps, {writes} Txns, published sequence {}",
                    scope.completed_steps(),
                    published.sequence
                );
            };
            tokio::select! {
                error = &mut progress => panic!("progress ended during continuous work: {error}"),
                () = work => {},
            }
            let frozen = coordinator.freeze(&proof, &running).await.unwrap().unwrap();
            assert!(!coordinator
                .publish_progress(&running, scope.completed_steps() + 1)
                .await
                .unwrap());
            frozen
        };
        let (error, frozen) = tokio::time::timeout(Duration::from_secs(20), async {
            tokio::join!(&mut observer, verify)
        })
        .await
        .unwrap();
        assert!(error.contains("revoked"));
        assert!(coordinator.finish_recovery(&proof, &frozen).await.unwrap());
        let recovered = coordinator.get("hot").await.unwrap().unwrap();
        assert!(coordinator.release(&proof, &recovered).await.unwrap());
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn storage_drain_observer_waits_for_completed_fence() {
        let (_state, coordinator, proof, _dir) = deadline_fixture().await;
        let execution = Execution::new("hot", "worker", "boot", 600);
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        let wait = wait_for_execution_state(&coordinator, "hot", |current| {
            current
                .is_none_or(|current| current.id != running.id || current.phase == Phase::Recovered)
        });
        tokio::pin!(wait);
        assert!(tokio::time::timeout(Duration::from_millis(100), &mut wait)
            .await
            .is_err());
        let frozen = coordinator.freeze(&proof, &running).await.unwrap().unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(200), &mut wait)
            .await
            .is_err());
        assert!(coordinator.finish_recovery(&proof, &frozen).await.unwrap());
        tokio::time::timeout(Duration::from_secs(2), &mut wait)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn unavailable_etcd_does_not_block_startup_or_flush_configuration() {
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
        assert_eq!(reply["legacy_progress_protocol"], 1);
        let Json(progress) = crate::legacy_merge::status(
            State(state.clone()),
            axum::extract::Query(Default::default()),
        )
        .await;
        assert_eq!(progress["instance"], reply["instance"]);
        assert_eq!(progress["snapshot"]["takeover_authorized"], false);
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

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn legacy_retirement_status_resolves_mode_and_returns_exact_coordinator_record() {
        let (mut state, coordinator, proof, _dir) = deadline_fixture().await;
        let local = Arc::get_mut(&mut state).unwrap();
        local.merge_executions.worker_admission = WorkerAdmission {
            writer_id: Some("worker-9".into()),
            enable_fencing: true,
        };
        local.merge_executions.endpoint = Some("http://replacement:3000".into());
        local.instance_id = Some("worker-9".into());
        let execution = coordinator
            .admit_legacy(&proof, "hot", "http://old:3000")
            .await
            .unwrap();
        let Json(reply) = retirement_status(
            State(state.clone()),
            axum::extract::Query(RetirementQuery {
                target: "hot".into(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(reply["fenced_legacy_merges"], true);
        assert_eq!(
            reply["execution"],
            serde_json::to_value(&execution).unwrap()
        );
        assert_eq!(reply["base_uri"], state.base_uri);
        assert_eq!(reply["shard_name"], "worker-9");
        assert_eq!(coordinator.get("hot").await.unwrap(), Some(execution));
        assert!(coordinator
            .worker_fencing_mode("worker-9", false)
            .await
            .unwrap());
        assert!(!coordinator
            .worker_fencing_mode("worker-8", false)
            .await
            .unwrap());
        assert!(!coordinator.legacy_fencing_mode(false).await.unwrap());
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn legacy_retirement_status_rejects_unfenced_replacement() {
        let (state, coordinator, proof, _dir) = deadline_fixture().await;
        let execution = coordinator
            .admit_legacy(&proof, "hot", "http://old:3000")
            .await
            .unwrap();
        assert!(retirement_status(
            State(state),
            axum::extract::Query(RetirementQuery {
                target: "hot".into()
            })
        )
        .await
        .is_err());
        assert_eq!(coordinator.get("hot").await.unwrap(), Some(execution));
        assert!(!coordinator.legacy_fencing_mode(false).await.unwrap());
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
