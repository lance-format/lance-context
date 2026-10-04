//! Serial worker merges with durable execution ownership and targeted retries.
use crate::{state::MasterState, task_store::TaskClaim};
use lance_context_merge::{ClaimProof, Coordinator, Execution, Phase};
use std::{sync::Arc, time::Duration};

const RPC_TIMEOUT: Duration = Duration::from_secs(10);
const POLL_DELAY: Duration = Duration::from_millis(250);
const RETRY_DELAY: Duration = Duration::from_secs(2);
const ATTEMPTS: usize = 3;
const OWNER_PROBE_INTERVAL: Duration = Duration::from_secs(10);
const OWNER_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const OWNER_PROBE_FAILURES: u8 = 3;

#[derive(serde::Deserialize)]
struct ExecutorIdentity {
    instance: String,
}

#[derive(serde::Deserialize)]
struct Capabilities {
    protocol: u32,
    instance: String,
    timeout_secs: u64,
    queue_timeout_secs: u64,
    idle_timeout_secs: u64,
    progress_protocol: u32,
    owned_targets: Vec<String>,
}

pub(crate) async fn run_merge_wal(
    state: &Arc<MasterState>,
    claim: &TaskClaim,
) -> Result<String, String> {
    if state.config.worker_endpoints.is_empty() {
        return Err("no worker endpoints configured".into());
    }
    let coordinator = state.task_store.merge_coordinator();
    let proof = state.task_store.merge_claim(claim);
    let target = &claim.task.target;
    if state.config.merge_rollout.draining(target) {
        // A drain stops new work, but must still resolve an already-owned
        // execution. Otherwise rollback would strand its persistent lock.
        if let Some(old) = coordinator.get(target).await? {
            ensure_recovery_due(&coordinator, &old).await?;
            if reconcile(&state.http, &coordinator, &proof, old, false)
                .await
                .is_err()
            {
                if let Some(current) = coordinator.get(target).await? {
                    recover_execution(state, &coordinator, &proof, current).await?;
                }
            }
        }
        return Err("merge target draining for protocol transition".into());
    }
    if !state.config.merge_rollout.owned(target) {
        if coordinator.get(target).await?.is_some() {
            return Err("owned execution still present; refusing legacy downgrade".into());
        }
        return run_legacy(state, target).await;
    }
    if state.config.append.enabled(target) {
        if let Some(old) = coordinator.get(target).await? {
            ensure_recovery_due(&coordinator, &old).await?;
            recover_execution(state, &coordinator, &proof, old).await?;
        }
        return crate::maintenance_execution::run_as(
            state,
            claim,
            lance_context_merge::MaintenanceKind::Catchup,
            crate::rollout_append::run(state, target),
        )
        .await;
    }
    // Reconcile/fence before any other table mutation. One retry of the fan-out
    // after a completed barrier lets healthy shards progress in this task.
    for recovery_round in 0..2 {
        if let Some(old) = coordinator.get(&claim.task.target).await? {
            ensure_recovery_due(&coordinator, &old).await?;
            if recovery_round > 0 || matches!(old.phase, Phase::Recovering | Phase::Uncertain) {
                recover_execution(state, &coordinator, &proof, old).await?;
            } else if let Err(error) = reconcile(&state.http, &coordinator, &proof, old, true).await
            {
                if let Some(current) = coordinator.get(&claim.task.target).await? {
                    recover_execution(state, &coordinator, &proof, current).await?;
                } else {
                    tracing::info!(target = %claim.task.target, %error, "previous merge ended; resuming shards");
                }
            }
        }
        let outcome = run_workers(
            &state.http,
            &coordinator,
            &proof,
            &claim.task.target,
            &state.config.worker_endpoints,
        )
        .await;
        if outcome.is_ok()
            || recovery_round == 1
            || coordinator.get(&claim.task.target).await?.is_none()
        {
            return outcome;
        }
    }
    unreachable!("bounded merge recovery loop returns on last iteration")
}

// Compatibility path is chosen only by explicit configuration, never after a
// failed owned RPC. It retains legacy limitations until that table is drained.
async fn run_legacy(state: &Arc<MasterState>, target: &str) -> Result<String, String> {
    let mut reclaimed = 0u64;
    for endpoint in &state.config.worker_endpoints {
        let endpoint = endpoint.trim_end_matches('/');
        let url = match target.strip_prefix("generic:") {
            Some(name) => format!("{endpoint}/api/v1/generic/{name}/merge-wal"),
            None => format!("{endpoint}/api/v1/internal/merge-wal/{target}"),
        };
        let started = std::time::Instant::now();
        let result = async {
            let response = state
                .http
                .post(url)
                .send()
                .await
                .map_err(|e| e.to_string())?;
            if response.status() == reqwest::StatusCode::NOT_FOUND {
                return Ok(0);
            }
            let reply: serde_json::Value = response
                .error_for_status()
                .map_err(|e| e.to_string())?
                .json()
                .await
                .map_err(|e| e.to_string())?;
            reply["reclaimed"]
                .as_u64()
                .ok_or_else(|| "invalid legacy merge response".to_string())
        }
        .await;
        metrics::histogram!("master_merge_wal_worker_duration_seconds")
            .record(started.elapsed().as_secs_f64());
        metrics::counter!("master_merge_wal_workers_total", "result" => if result.is_ok() { "ok" } else { "failed" }).increment(1);
        reclaimed += result?;
    }
    metrics::counter!("master_merge_wal_generations_reclaimed_total").increment(reclaimed);
    Ok(format!(
        "merged {reclaimed} generations across {}/{} workers (legacy)",
        state.config.worker_endpoints.len(),
        state.config.worker_endpoints.len()
    ))
}

struct Deadlines {
    queue: tokio::time::Instant,
    idle_timeout: Duration,
    running: Option<tokio::time::Instant>,
    last_progress: tokio::time::Instant,
    sequence: Option<u64>,
}

impl Deadlines {
    fn new(execution: &Execution, now: tokio::time::Instant) -> Self {
        Self {
            queue: now + Duration::from_secs(execution.queue_timeout_secs) + RPC_TIMEOUT,
            idle_timeout: Duration::from_secs(execution.idle_timeout_secs) + RPC_TIMEOUT,
            running: None,
            last_progress: now,
            sequence: None,
        }
    }
    fn observe(&mut self, sequence: u64, now: tokio::time::Instant) {
        self.running.get_or_insert(now);
        if self.sequence.is_none_or(|previous| sequence > previous) {
            self.sequence = Some(sequence);
            self.last_progress = now;
        }
    }
    fn expired(&self, now: tokio::time::Instant) -> bool {
        match self.running {
            None => now >= self.queue,
            Some(_) => now.duration_since(self.last_progress) >= self.idle_timeout,
        }
    }
}

pub(crate) async fn ensure_recovery_due(
    coordinator: &Coordinator,
    old: &Execution,
) -> Result<(), String> {
    if old.phase == Phase::Recovering {
        if let Some(failure) = coordinator.failure(&old.target, &old.endpoint).await? {
            if failure.last_error.contains("recovery barrier failed")
                && failure.next_retry_ms > lance_context_merge::failure::now_ms()
            {
                return Err(format!(
                    "{}; recovery probe at {}",
                    failure.last_error, failure.next_retry_ms
                ));
            }
        }
    }
    Ok(())
}

pub(crate) async fn recover_execution(
    state: &Arc<MasterState>,
    coordinator: &Coordinator,
    proof: &ClaimProof,
    old: Execution,
) -> Result<(), String> {
    let result = async {
        // Completion can race a failed identity probe or the end of cancel
        // grace. A terminal executor needs release, not a new storage barrier.
        if matches!(old.phase, Phase::Finished | Phase::Recovered) {
            return coordinator.release(proof, &old).await?.then_some(())
                .ok_or_else(|| "claim lost before releasing terminal merge execution".to_string());
        }
        let frozen = coordinator.freeze(proof, &old).await?
            .ok_or_else(|| "merge execution changed while freezing commit admission".to_string())?;
        let watermarks = coordinator.watermarks(&frozen).await?;
        let uri = match frozen.target.strip_prefix("generic:") {
            Some(name) => state.generic_uri(name), None => state.rollout_uri(&frozen.target),
        };
        if watermarks.dataset_uri.as_deref().is_some_and(|worker_uri| worker_uri != uri) {
            return Err("dataset URI mismatch between worker and recovery master".into());
        }
        tokio::time::timeout(Duration::from_secs(120),
            lance_context_core::merge_write_scope::fence_manifest_versions(&uri, None, &watermarks.versions, &frozen.id)
        ).await.map_err(|_| "storage recovery barrier deadline exceeded".to_string())?
            .map_err(|e| e.to_string())?;
        if !coordinator.finish_recovery(proof, &frozen).await? {
            return Err("claim lost before publishing storage recovery barrier".into());
        }
        let recovered = coordinator.get(&frozen.target).await?.ok_or_else(|| "recovered execution disappeared".to_string())?;
        if recovered.id != frozen.id || !coordinator.release(proof, &recovered).await? {
            return Err("claim lost before releasing recovered execution".into());
        }
        metrics::counter!("master_merge_storage_recoveries_total", "result" => "ok").increment(1);
        tracing::warn!(target = %frozen.target, endpoint = %frozen.endpoint, id = %frozen.id, "old merge storage commits fenced; resuming healthy shards");
        Ok(())
    }.await;
    if let Err(error) = &result {
        let message = format!("merge ownership unresolved: recovery barrier failed: {error}");
        coordinator
            .record_failure(proof, &old.target, &old.endpoint, &message)
            .await?;
        metrics::counter!("master_merge_storage_recoveries_total", "result" => "failed")
            .increment(1);
        return Err(message);
    }
    result
}

async fn run_workers(
    http: &reqwest::Client,
    coordinator: &Coordinator,
    proof: &ClaimProof,
    target: &str,
    endpoints: &[String],
) -> Result<String, String> {
    let mut pending = Vec::new();
    let mut deferred = Vec::new();
    for endpoint in endpoints {
        match coordinator.failure(target, endpoint).await? {
            Some(failure) if failure.next_retry_ms > lance_context_merge::failure::now_ms() => {
                deferred.push(format!(
                    "{endpoint}: retry at {}; {:?}: {}",
                    failure.next_retry_ms, failure.class, failure.last_error
                ))
            }
            _ => pending.push(endpoint.clone()),
        }
    }

    let mut reclaimed = 0;
    let mut errors = Vec::new();
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(RETRY_DELAY).await;
        }
        let mut retry = Vec::new();
        errors.clear();
        for endpoint in pending {
            let failures_before = coordinator
                .failure(target, &endpoint)
                .await?
                .map_or(0, |failure| failure.consecutive_attempts);
            let started = std::time::Instant::now();
            let outcome = one(http, coordinator, proof, target, &endpoint).await;
            metrics::histogram!("master_merge_wal_worker_duration_seconds")
                .record(started.elapsed().as_secs_f64());
            metrics::counter!("master_merge_wal_workers_total", "result" => if outcome.is_ok() { "ok" } else { "failed" }).increment(1);
            match outcome {
                Ok(n) => {
                    reclaimed += n;
                }
                Err(error) => {
                    // An execution still in storage is not a failed endpoint we
                    // may skip. Ownership must first be reconciled to terminal.
                    if coordinator.get(target).await?.is_some() {
                        return Err(error);
                    }
                    // Executed failures are recorded atomically with release.
                    // Capability/admission failures have no execution result.
                    let failure = match coordinator.failure(target, &endpoint).await? {
                        Some(failure) if failure.consecutive_attempts > failures_before => failure,
                        _ => {
                            coordinator
                                .record_failure(proof, target, &endpoint, &error)
                                .await?
                        }
                    };
                    let message = format!(
                        "{endpoint}: {error} (attempt {}; retry at {}; attention={})",
                        failure.consecutive_attempts,
                        failure.next_retry_ms,
                        failure.needs_attention
                    );
                    if failure.consecutive_attempts < ATTEMPTS as u32 && !failure.needs_attention {
                        errors.push(message);
                        retry.push(endpoint);
                    } else {
                        deferred.push(message);
                    }
                }
            }
        }
        pending = retry;
        if pending.is_empty() {
            break;
        }
    }
    metrics::counter!("master_merge_wal_generations_reclaimed_total").increment(reclaimed as u64);
    errors.extend(deferred);
    if !errors.is_empty() {
        return Err(format!("merged {reclaimed} generations; {} worker(s) failed or cooling down (up to {ATTEMPTS} targeted attempts): {}", errors.len(), errors.join("; ")));
    }
    Ok(format!(
        "merged {reclaimed} generations across {}/{} workers",
        endpoints.len(),
        endpoints.len()
    ))
}

async fn one(
    http: &reqwest::Client,
    coordinator: &Coordinator,
    proof: &ClaimProof,
    target: &str,
    endpoint: &str,
) -> Result<usize, String> {
    let base = format!(
        "{}/api/v1/internal/merge-executor",
        endpoint.trim_end_matches('/')
    );
    let capabilities: Capabilities = http
        .get(&base)
        .timeout(RPC_TIMEOUT)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    if capabilities.protocol != 2
        || capabilities.progress_protocol != 1
        || !capabilities.owned_targets.iter().any(|t| t == target)
        || capabilities.timeout_secs == 0
        || capabilities.queue_timeout_secs == 0
        || capabilities.idle_timeout_secs == 0
    {
        return Err("worker lacks bounded owned-merge protocol".into());
    }
    let mut execution = Execution::new(
        target,
        endpoint,
        &capabilities.instance,
        capabilities.timeout_secs,
    );
    execution.queue_timeout_secs = capabilities.queue_timeout_secs;
    execution.idle_timeout_secs = capabilities.idle_timeout_secs;
    // Even a lost reserve response is ambiguous. Recover the fence before
    // issuing any other storage operation; never infer absence from an error.
    let admission = coordinator.reserve(proof, &execution).await;
    match admission {
        Ok(true) => {}
        _ => {
            if let Some(current) = coordinator.get(target).await? {
                if current.id == execution.id {
                    return reconcile(http, coordinator, proof, current, true).await;
                }
            }
            return Err("merge admission failed or task claim lost".into());
        }
    }
    let started = http
        .post(format!("{base}/start"))
        .timeout(RPC_TIMEOUT)
        .json(&execution)
        .send()
        .await;
    let cancel = !matches!(started, Ok(ref response) if response.status().is_success());
    // A transport error can mean the worker is already writing. It is NOT an
    // outcome, and cannot let this task advance to another endpoint.
    reconcile(http, coordinator, proof, execution, cancel).await
}

async fn reconcile(
    http: &reqwest::Client,
    coordinator: &Coordinator,
    proof: &ClaimProof,
    initial: Execution,
    cancel_immediately: bool,
) -> Result<usize, String> {
    reconcile_with_grace(
        http,
        coordinator,
        proof,
        initial,
        cancel_immediately,
        Duration::from_secs(60),
    )
    .await
}

async fn reconcile_with_grace(
    http: &reqwest::Client,
    coordinator: &Coordinator,
    proof: &ClaimProof,
    initial: Execution,
    cancel_immediately: bool,
    grace: Duration,
) -> Result<usize, String> {
    let mut deadlines = Deadlines::new(&initial, tokio::time::Instant::now());
    let mut cancel_started = cancel_immediately.then(tokio::time::Instant::now);
    let mut next_cancel = tokio::time::Instant::now();
    let mut next_probe = tokio::time::Instant::now() + OWNER_PROBE_INTERVAL;
    let mut failed_probes = 0;
    loop {
        // Losing etcd responses must not turn this into an unbounded scheduler
        // wait. Ownership remains durable when the reconciliation slot exits.
        if cancel_started.is_some_and(|at| at.elapsed() >= grace) {
            let error = "merge ownership unresolved: executor did not acknowledge termination; fence retained pending storage version recovery";
            coordinator
                .record_failure(proof, &initial.target, &initial.endpoint, error)
                .await?;
            return Err(error.into());
        }
        let current = match coordinator.get(&initial.target).await {
            Ok(Some(current)) if current.id == initial.id => current,
            Ok(_) => return Err("merge execution ownership changed during reconciliation".into()),
            Err(error) => {
                // Bound an unavailable coordinator independently. On successful
                // reads, sample progress before evaluating the idle deadline.
                if cancel_started.is_none() && deadlines.expired(tokio::time::Instant::now()) {
                    cancel_started = Some(tokio::time::Instant::now());
                }
                tracing::warn!(target = %initial.target, %error, "cannot confirm merge completion; retaining execution fence");
                tokio::time::sleep(RETRY_DELAY).await;
                continue;
            }
        };
        if matches!(current.phase, Phase::Uncertain | Phase::Recovering) {
            let error = current
                .error
                .unwrap_or_else(|| "merge ownership unresolved: ambiguous storage commit".into());
            coordinator
                .record_failure(proof, &current.target, &current.endpoint, &error)
                .await?;
            return Err(error);
        }
        if matches!(current.phase, Phase::Finished | Phase::Recovered) {
            if !coordinator.release(proof, &current).await? {
                return Err("task claim lost while releasing terminal merge execution".into());
            }
            return current.error.map_or(Ok(current.reclaimed), Err);
        }
        if cancel_started.is_none() {
            if let Ok(Some(progress)) = coordinator.progress(&current).await {
                if deadlines
                    .sequence
                    .is_none_or(|previous| progress.sequence > previous)
                {
                    // Completed work outweighs an unavailable HTTP endpoint.
                    failed_probes = 0;
                }
                deadlines.observe(progress.sequence, tokio::time::Instant::now());
            }
            if deadlines.expired(tokio::time::Instant::now()) {
                cancel_started = Some(tokio::time::Instant::now());
            }
        }
        if tokio::time::Instant::now() >= next_probe {
            let probe = async {
                http.get(format!(
                    "{}/api/v1/internal/merge-executor",
                    current.endpoint.trim_end_matches('/')
                ))
                .timeout(OWNER_PROBE_TIMEOUT)
                .send()
                .await?
                .error_for_status()?
                .json::<ExecutorIdentity>()
                .await
            }
            .await;
            next_probe = tokio::time::Instant::now() + OWNER_PROBE_INTERVAL;
            let reason = match probe {
                Ok(identity) if identity.instance == current.instance => {
                    failed_probes = 0;
                    None
                }
                Ok(_) => Some("merge executor incarnation changed"),
                Err(_) => {
                    failed_probes += 1;
                    (failed_probes >= OWNER_PROBE_FAILURES)
                        .then_some("merge executor unreachable without progress")
                }
            };
            if let Some(reason) = reason {
                if current.phase == Phase::Reserved {
                    // No execution has started. CAS cancellation either wins
                    // admission or the next observation sees the running owner.
                    let _ = coordinator.cancel_reserved(&current).await;
                    continue;
                }
                // This is a reason to revoke admission and run the storage
                // barrier, never permission to release ownership directly.
                let error = format!("merge ownership unresolved: {reason}; recovery required");
                coordinator
                    .record_failure(proof, &current.target, &current.endpoint, &error)
                    .await?;
                tracing::warn!(target = %current.target, endpoint = %current.endpoint,
                    id = %current.id, %reason, "recovering lost merge executor before execution deadline");
                return Err(error);
            }
        }
        if cancel_started.is_some() && tokio::time::Instant::now() >= next_cancel {
            next_cancel = tokio::time::Instant::now() + RETRY_DELAY;
            // CAS reserved work to terminal, or ask its owner to cancel the
            // actual storage future. HTTP success alone is not acknowledgement.
            if current.phase == Phase::Reserved {
                let _ = coordinator.cancel_reserved(&current).await;
            }
            let _ = http
                .post(format!(
                    "{}/api/v1/internal/merge-executor/cancel",
                    current.endpoint.trim_end_matches('/')
                ))
                .timeout(RPC_TIMEOUT)
                .json(&current)
                .send()
                .await;
        }
        tokio::time::sleep(POLL_DELAY).await;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn queue_execution_and_real_progress_have_independent_deadlines() {
        use super::*;
        let mut execution = Execution::new("hot", "worker", "boot", 1800);
        execution.queue_timeout_secs = 600;
        execution.idle_timeout_secs = 120;
        let now = tokio::time::Instant::now();
        let mut deadlines = Deadlines::new(&execution, now);
        assert!(!deadlines.expired(now + Duration::from_secs(599)));
        // A long queue wait consumes none of the running budget.
        let started = now + Duration::from_secs(599);
        deadlines.observe(0, started);
        for step in 1..=10 {
            let current = started + Duration::from_secs(step * 100);
            deadlines.observe(step, current);
            assert!(!deadlines.expired(current));
        }
        // Receiving the same progress repeatedly is only a heartbeat.
        let stalled = started + Duration::from_secs(1131);
        deadlines.observe(10, stalled);
        assert!(deadlines.expired(stalled));
        // Real progress continues beyond the legacy total runtime ceiling.
        deadlines.observe(11, started + Duration::from_secs(1811));
        assert!(!deadlines.expired(started + Duration::from_secs(1811)));
    }

    #[test]
    fn a_worker_that_never_acquires_a_slot_has_a_bounded_queue_wait() {
        use super::*;
        let mut execution = Execution::new("hot", "worker", "boot", 3600);
        execution.queue_timeout_secs = 10;
        let now = tokio::time::Instant::now();
        let deadlines = Deadlines::new(&execution, now);
        assert!(deadlines.expired(now + Duration::from_secs(21)));
    }
    use super::*;
    use axum::{
        extract::State,
        routing::{get, post},
        Json, Router,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };
    use tokio::sync::watch;

    #[derive(Clone)]
    struct Worker {
        coordinator: Coordinator,
        calls: Arc<AtomicUsize>,
        fail: bool,
        stall_first: bool,
        name: &'static str,
        events: Arc<Mutex<Vec<String>>>,
    }

    async fn worker(worker: Worker) -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route(
                "/api/v1/internal/merge-executor",
                get(|| async {
                    Json(serde_json::json!({"protocol":2,"instance":"test","timeout_secs":1,"queue_timeout_secs":1,"idle_timeout_secs":1,"progress_protocol":1,"owned_targets":["table"]}))
                }),
            )
            .route(
                "/api/v1/internal/merge-executor/start",
                post(
                    |State(w): State<Worker>, Json(e): Json<Execution>| async move {
                        let running = w.coordinator.start(&e).await.unwrap().unwrap();
                        tokio::spawn(async move {
                            let call = w.calls.fetch_add(1, Ordering::SeqCst);
                            let (_cancel, rx) = watch::channel(false);
                            let result = lance_context_merge::execute_scoped(
                                async {
                                    if w.stall_first && call == 0 {
                                        return std::future::pending().await;
                                    }
                                    if w.fail {
                                        Err("injected shard failure".into())
                                    } else {
                                        Ok(7)
                                    }
                                },
                                Duration::from_secs(1),
                                rx,
                            )
                            .await;
                            w.events.lock().unwrap().push(format!(
                                "{}:{}",
                                w.name,
                                if result.is_ok() { "ok" } else { "failed" }
                            ));
                            assert!(w.coordinator.finish(&running, result).await.unwrap());
                        });
                        axum::http::StatusCode::ACCEPTED
                    },
                ),
            )
            .with_state(worker);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        (
            address,
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            }),
        )
    }

    async fn fixture() -> (Coordinator, ClaimProof) {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").expect("isolated local etcd required");
        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let prefix = format!("/merge-fanout-tests/{}", Execution::new("", "", "", 1).id);
        let lease = client.lease_grant(120, None).await.unwrap().id();
        let proof = ClaimProof {
            key: format!("{prefix}/claim"),
            token: "owner".into(),
            lease_id: lease,
        };
        client
            .put(
                lance_context_merge::target_lock_key(&prefix, "table"),
                proof.token.clone(),
                Some(etcd_client::PutOptions::new().with_lease(lease)),
            )
            .await
            .unwrap();
        client
            .put(
                proof.key.clone(),
                proof.token.clone(),
                Some(etcd_client::PutOptions::new().with_lease(lease)),
            )
            .await
            .unwrap();
        (Coordinator::new(client, prefix), proof)
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn orphaned_merge_is_storage_fenced_and_healthy_shard_progresses() {
        use clap::Parser;
        use lance_context_api::TaskKind;
        let dir = tempfile::tempdir().unwrap();
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let prefix = format!("/merge-orphan-tests/{}", Execution::new("", "", "", 1).id);
        let cfg = crate::config::MasterConfig::parse_from([
            "test",
            "--data-dir",
            dir.path().to_str().unwrap(),
            "--etcd-endpoints",
            &endpoints,
            "--etcd-prefix",
            &prefix,
            "--merge-owned-targets",
            "table",
        ]);
        let mut state = MasterState::new(cfg).await.unwrap();
        let uri = state.rollout_uri("table");
        lance_context_core::RolloutStore::open(&uri)
            .await
            .unwrap()
            .close()
            .await
            .unwrap();
        let initial_version = lance::Dataset::open(&uri).await.unwrap().version().version;
        let coordinator = state.task_store.merge_coordinator();
        let good = Worker {
            coordinator: coordinator.clone(),
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
            stall_first: false,
            name: "healthy",
            events: Arc::new(Mutex::new(Vec::new())),
        };
        let (healthy, server) = worker(good.clone()).await;
        let lost = "http://127.0.0.1:1".to_string();
        Arc::get_mut(&mut state).unwrap().config.worker_endpoints = vec![lost.clone(), healthy];
        state
            .task_store
            .enqueue(TaskKind::MergeWal, "table", Vec::new())
            .await
            .unwrap();
        let claim = state
            .task_store
            .claim_next_of_kinds(crate::task_store::TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let proof = state.task_store.merge_claim(&claim);
        let execution = Execution::new("table", &lost, "dead-worker", 1);
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        coordinator
            .authorize_commit(&running, &uri, "base", initial_version + 1)
            .await
            .unwrap();
        assert!(coordinator
            .report_uncertain(&running, "lost storage response".into())
            .await
            .unwrap());
        let outcome = tokio::time::timeout(Duration::from_secs(30), run_merge_wal(&state, &claim))
            .await
            .unwrap();
        assert!(outcome.unwrap_err().contains("merged 7 generations"));
        assert_eq!(good.calls.load(Ordering::SeqCst), 1);
        assert!(coordinator.get("table").await.unwrap().is_none());
        assert!(lance::Dataset::open(&uri).await.unwrap().version().version > initial_version + 1);
        assert!(coordinator
            .authorize_commit(&running, &uri, "base", initial_version + 4)
            .await
            .is_err());

        // A probe may decide to recover just as the executor publishes its
        // terminal result. Release that result without trying to freeze it.
        let completed = Execution::new("table", &lost, "late-completion", 1);
        assert!(coordinator.reserve(&proof, &completed).await.unwrap());
        let completed = coordinator.start(&completed).await.unwrap().unwrap();
        assert!(coordinator.finish(&completed, Ok(0)).await.unwrap());
        let completed = coordinator.get("table").await.unwrap().unwrap();
        recover_execution(&state, &coordinator, &proof, completed)
            .await
            .unwrap();
        assert!(coordinator.get("table").await.unwrap().is_none());

        // A mismatched worker namespace must remain fenced, never recover by
        // committing a barrier to an unrelated same-named table.
        let wrong = Execution::new("table", &lost, "misconfigured-worker", 1);
        assert!(coordinator.reserve(&proof, &wrong).await.unwrap());
        let wrong = coordinator.start(&wrong).await.unwrap().unwrap();
        coordinator
            .authorize_commit(&wrong, "file:///different-table", "base", 7)
            .await
            .unwrap();
        let error = recover_execution(&state, &coordinator, &proof, wrong)
            .await
            .unwrap_err();
        assert!(error.contains("dataset URI mismatch"));
        assert_eq!(
            coordinator.get("table").await.unwrap().unwrap().phase,
            Phase::Recovering
        );
        state.task_store.finish(claim, Err(error)).await.unwrap();
        server.abort();
        let mut client =
            etcd_client::Client::connect(endpoints.split(',').collect::<Vec<_>>(), None)
                .await
                .unwrap();
        client
            .delete(
                prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    async fn verify_lost_executor(restarted: bool) {
        let (coordinator, proof) = fixture().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let app = if restarted {
            Router::new().route(
                "/api/v1/internal/merge-executor",
                get(|| async { Json(serde_json::json!({"instance": "new-process"})) }),
            )
        } else {
            Router::new()
        };
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let execution = Execution::new("table", &endpoint, "old-process", 3600);
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        assert!(coordinator.publish_progress(&running, 0).await.unwrap());
        let error = tokio::time::timeout(
            Duration::from_secs(if restarted { 15 } else { 40 }),
            reconcile(
                &reqwest::Client::new(),
                &coordinator,
                &proof,
                running.clone(),
                false,
            ),
        )
        .await
        .expect("lost executor must be detected before the 600-second idle timeout")
        .unwrap_err();
        assert!(
            error.contains(if restarted {
                "incarnation changed"
            } else {
                "unreachable"
            }),
            "{error}"
        );
        assert_eq!(coordinator.get("table").await.unwrap(), Some(running));
        assert!(!coordinator
            .reserve(&proof, &Execution::new("table", "other", "new", 3600))
            .await
            .unwrap());
        server.abort();
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn restarted_executor_is_detected_before_idle_deadline() {
        verify_lost_executor(true).await;
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn unreachable_executor_is_detected_before_idle_deadline() {
        verify_lost_executor(false).await;
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn real_progress_prevents_takeover_when_http_probes_fail() {
        let (coordinator, proof) = fixture().await;
        let execution = Execution::new("table", "http://127.0.0.1:1", "worker", 3600);
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        assert!(coordinator.publish_progress(&running, 0).await.unwrap());
        let publisher = coordinator.clone();
        let working = running.clone();
        let work = tokio::spawn(async move {
            for step in 1..=35 {
                tokio::time::sleep(Duration::from_secs(1)).await;
                assert!(publisher.publish_progress(&working, step).await.unwrap());
            }
            assert!(publisher.finish(&working, Ok(7)).await.unwrap());
        });
        assert_eq!(
            tokio::time::timeout(
                Duration::from_secs(45),
                reconcile(
                    &reqwest::Client::new(),
                    &coordinator,
                    &proof,
                    running,
                    false
                )
            )
            .await
            .unwrap()
            .unwrap(),
            7
        );
        work.await.unwrap();
        assert!(coordinator.get("table").await.unwrap().is_none());
        assert!(coordinator
            .failure("table", &execution.endpoint)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn missing_executor_returns_attention_without_unlocking_surviving_write() {
        let (coordinator, proof) = fixture().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server =
            tokio::spawn(async move { axum::serve(listener, Router::new()).await.unwrap() });
        let execution = Execution::new("table", &endpoint, "lost-process", 1);
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        let error = tokio::time::timeout(
            Duration::from_secs(3),
            reconcile_with_grace(
                &reqwest::Client::new(),
                &coordinator,
                &proof,
                running.clone(),
                true,
                Duration::from_millis(50),
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert!(error.contains("ownership unresolved"));
        assert_eq!(
            coordinator.get("table").await.unwrap(),
            Some(running.clone())
        );
        assert!(!coordinator
            .reserve(&proof, &Execution::new("table", "replacement", "new", 1))
            .await
            .unwrap());
        assert!(
            coordinator
                .failure("table", &endpoint)
                .await
                .unwrap()
                .unwrap()
                .needs_attention
        );
        // A late positive acknowledgement still permits normal recovery.
        assert!(coordinator
            .finish(&running, Err("cancelled".into()))
            .await
            .unwrap());
        let terminal = coordinator.get("table").await.unwrap().unwrap();
        assert!(coordinator.release(&proof, &terminal).await.unwrap());
        server.abort();
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn missing_owned_capability_never_falls_back_to_legacy_post() {
        let (coordinator, proof) = fixture().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let app = Router::new().route(
            "/api/v1/internal/merge-wal/table",
            post(move || {
                let counted = counted.clone();
                async move {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({"reclaimed": 9}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        assert!(one(
            &reqwest::Client::new(),
            &coordinator,
            &proof,
            "table",
            &endpoint
        )
        .await
        .is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(coordinator.get("table").await.unwrap().is_none());
        server.abort();
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn stalled_shard_is_terminated_healthy_shards_advance_then_only_failure_retries() {
        let (coordinator, proof) = fixture().await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let bad = Worker {
            coordinator: coordinator.clone(),
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
            stall_first: true,
            name: "stalled",
            events: events.clone(),
        };
        let good = Worker {
            stall_first: false,
            name: "healthy",
            calls: Arc::new(AtomicUsize::new(0)),
            ..bad.clone()
        };
        let (first, first_server) = worker(bad.clone()).await;
        let (second, second_server) = worker(good.clone()).await;
        let result = tokio::time::timeout(
            Duration::from_secs(20),
            run_workers(
                &reqwest::Client::new(),
                &coordinator,
                &proof,
                "table",
                &[first, second],
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(result.contains("merged 14"));
        assert_eq!(bad.calls.load(Ordering::SeqCst), 2);
        assert_eq!(good.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            *events.lock().unwrap(),
            ["stalled:failed", "healthy:ok", "stalled:ok"]
        );
        assert!(coordinator.get("table").await.unwrap().is_none());
        first_server.abort();
        second_server.abort();
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn partial_success_does_not_hide_exhausted_shard_failure() {
        let (coordinator, proof) = fixture().await;
        let bad = Worker {
            coordinator: coordinator.clone(),
            calls: Arc::new(AtomicUsize::new(0)),
            fail: true,
            stall_first: false,
            name: "bad",
            events: Arc::new(Mutex::new(Vec::new())),
        };
        let good = Worker {
            fail: false,
            name: "good",
            calls: Arc::new(AtomicUsize::new(0)),
            ..bad.clone()
        };
        let (first, first_server) = worker(bad.clone()).await;
        let (second, second_server) = worker(good.clone()).await;
        let result = tokio::time::timeout(
            Duration::from_secs(25),
            run_workers(
                &reqwest::Client::new(),
                &coordinator,
                &proof,
                "table",
                &[first.clone(), second.clone()],
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert!(result.contains("merged 7 generations"));
        assert!(result.contains("1 worker(s) failed or cooling down"));
        assert_eq!(bad.calls.load(Ordering::SeqCst), 3);
        assert_eq!(good.calls.load(Ordering::SeqCst), 1);
        // A fresh task must retain the failed shard's backoff, while still
        // admitting the healthy shard for newly arrived generations.
        let next = run_workers(
            &reqwest::Client::new(),
            &coordinator,
            &proof,
            "table",
            &[first.clone(), second],
        )
        .await
        .unwrap_err();
        assert!(next.contains("retry at"));
        assert_eq!(bad.calls.load(Ordering::SeqCst), 3);
        assert_eq!(good.calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            coordinator
                .failure("table", &first)
                .await
                .unwrap()
                .unwrap()
                .consecutive_attempts,
            3
        );
        assert!(coordinator.get("table").await.unwrap().is_none());
        first_server.abort();
        second_server.abort();
    }
}
