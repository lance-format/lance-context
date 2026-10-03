//! Execution fences outlive scheduler leases and HTTP connections.
//!
//! A claim authorizes admission, not cancellation of an existing storage write.
//! Only the executor can publish its terminal outcome. Recovery first cancels
//! and reconciles the old execution; it never clears a running fence on timeout.

pub mod failure;
pub mod fencing;
pub mod progress;
pub mod rollout;

use etcd_client::{Client, Compare, CompareOp, Txn, TxnOp};
use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug)]
pub struct ClaimProof {
    pub key: String,
    pub token: String,
    pub lease_id: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Reserved,
    Running,
    Uncertain,
    Recovering,
    Recovered,
    Finished,
}

/// Local master mutations use the same storage fence as WAL merging. Older
/// readers cannot CAS these records because they do not preserve this field.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MaintenanceKind {
    Compact,
    IndexId,
    Repair,
}

impl MaintenanceKind {
    pub fn endpoint(self) -> &'static str {
        match self {
            Self::Compact => "master:compact",
            Self::IndexId => "master:index_id",
            Self::Repair => "master:repair",
        }
    }
    pub fn from_endpoint(endpoint: &str) -> Option<Self> {
        match endpoint {
            "master:compact" => Some(Self::Compact),
            "master:index_id" => Some(Self::IndexId),
            "master:repair" => Some(Self::Repair),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Execution {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maintenance: Option<MaintenanceKind>,
    pub id: String,
    pub target: String,
    pub endpoint: String,
    pub instance: String,
    pub timeout_secs: u64,
    #[serde(default = "default_queue_timeout")]
    pub queue_timeout_secs: u64,
    #[serde(default = "default_queue_timeout")]
    pub idle_timeout_secs: u64,
    pub phase: Phase,
    pub reclaimed: usize,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "protocol_absent")]
    pub protocol: u32,
}

fn protocol_absent(protocol: &u32) -> bool {
    *protocol == 0
}

fn default_queue_timeout() -> u64 {
    600
}

impl Execution {
    pub fn new(target: &str, endpoint: &str, instance: &str, timeout_secs: u64) -> Self {
        Self {
            maintenance: None,
            id: uuid::Uuid::new_v4().to_string(),
            target: target.into(),
            endpoint: endpoint.into(),
            instance: instance.into(),
            timeout_secs,
            queue_timeout_secs: default_queue_timeout(),
            idle_timeout_secs: default_queue_timeout(),
            phase: Phase::Reserved,
            reclaimed: 0,
            error: None,
            protocol: 2,
        }
    }

    pub fn finished(&self, outcome: Result<usize>) -> Self {
        let mut next = self.clone();
        next.phase = Phase::Finished;
        match outcome {
            Ok(n) => next.reclaimed = n,
            Err(e) => next.error = Some(e),
        }
        next
    }
}

/// Deliberately no lease on these keys. An expired task claim must not let a
/// second endpoint commit while the first endpoint's HTTP handler is still alive.
#[derive(Clone)]
pub struct Coordinator {
    client: Client,
    prefix: String,
}

pub fn execution_key(prefix: &str, target: &str) -> String {
    let encoded: String = target
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!(
        "{}/merge-executions/{encoded}",
        prefix.trim_end_matches('/')
    )
}

pub fn target_lock_key(prefix: &str, target: &str) -> String {
    execution_key(prefix, target).replace("/merge-executions/", "/target-locks/")
}

pub fn execution_owner(execution: &Execution) -> String {
    format!("merge-execution:{}", execution.id)
}

fn encode(execution: &Execution) -> Vec<u8> {
    serde_json::to_vec(execution).expect("execution is JSON serializable")
}

impl Coordinator {
    pub fn new(client: Client, prefix: impl Into<String>) -> Self {
        Self {
            client,
            prefix: prefix.into(),
        }
    }

    pub async fn get(&self, target: &str) -> Result<Option<Execution>> {
        let response = self
            .client
            .clone()
            .get(execution_key(&self.prefix, target), None)
            .await
            .map_err(|e| e.to_string())?;
        response
            .kvs()
            .first()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()
    }

    /// Bounded recovery inventory; execution records themselves are durable
    /// demand even if a process died before writing its failure ledger.
    pub async fn execution_page(
        &self,
        after: Option<&str>,
    ) -> Result<(Vec<Execution>, Option<String>)> {
        let prefix = format!("{}/merge-executions/", self.prefix.trim_end_matches('/'));
        let (start, options) = match after {
            None => (prefix.clone(), etcd_client::GetOptions::new().with_prefix()),
            Some(key) if key.starts_with(&prefix) => {
                let mut end = prefix.as_bytes().to_vec();
                *end.last_mut().unwrap() += 1;
                (
                    format!("{key}\0"),
                    etcd_client::GetOptions::new().with_range(end),
                )
            }
            Some(_) => return Err("invalid execution cursor".into()),
        };
        let response = self
            .client
            .clone()
            .get(start, Some(options.with_limit(256)))
            .await
            .map_err(|e| e.to_string())?;
        let rows = response
            .kvs()
            .iter()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .collect::<Result<Vec<Execution>>>()?;
        let next = response
            .more()
            .then(|| {
                response
                    .kvs()
                    .last()
                    .map(|kv| String::from_utf8_lossy(kv.key()).into_owned())
            })
            .flatten();
        Ok((rows, next))
    }

    /// Atomic claim check and admission close the delayed-request/lease-loss race.
    pub async fn reserve(&self, claim: &ClaimProof, execution: &Execution) -> Result<bool> {
        if execution.phase != Phase::Reserved
            || execution.timeout_secs == 0
            || execution.protocol != 2
        {
            return Err("invalid merge execution reservation".into());
        }
        let key = execution_key(&self.prefix, &execution.target);
        self.transact(
            vec![
                Compare::value(claim.key.as_str(), CompareOp::Equal, claim.token.as_bytes()),
                Compare::version(key.as_str(), CompareOp::Equal, 0),
                Compare::value(
                    target_lock_key(&self.prefix, &execution.target),
                    CompareOp::Equal,
                    claim.token.as_bytes(),
                ),
            ],
            vec![
                TxnOp::put(key, encode(execution), None),
                TxnOp::put(
                    target_lock_key(&self.prefix, &execution.target),
                    execution_owner(execution),
                    None,
                ),
            ],
        )
        .await
    }

    pub async fn start(&self, execution: &Execution) -> Result<Option<Execution>> {
        if execution.phase != Phase::Reserved {
            return Err("execution is not reserved".into());
        }
        let mut running = execution.clone();
        running.phase = Phase::Running;
        Ok(self.replace(execution, &running).await?.then_some(running))
    }

    pub async fn finish(&self, running: &Execution, outcome: Result<usize>) -> Result<bool> {
        if running.phase != Phase::Running {
            return Err("execution is not running".into());
        }
        self.replace(running, &running.finished(outcome)).await
    }

    /// The Rust write future ended with an ambiguous storage result. This is
    /// durable diagnostic evidence, not permission to hand off storage writes.
    pub async fn report_uncertain(&self, running: &Execution, error: String) -> Result<bool> {
        if running.phase != Phase::Running {
            return Err("execution is not running".into());
        }
        let mut next = running.finished(Err(error));
        next.phase = Phase::Uncertain;
        self.replace(running, &next).await
    }

    /// Cancelling an unstarted request is a CAS. A late POST cannot start it.
    pub async fn cancel_reserved(&self, execution: &Execution) -> Result<bool> {
        if execution.phase != Phase::Reserved {
            return Ok(false);
        }
        self.replace(
            execution,
            &execution.finished(Err("cancelled before admission".into())),
        )
        .await
    }

    /// Never delete on elapsed time, an HTTP error, or claim expiry. Both the
    /// terminal result and the current task claim must still match atomically.
    pub async fn release(&self, claim: &ClaimProof, execution: &Execution) -> Result<bool> {
        if !matches!(execution.phase, Phase::Finished | Phase::Recovered) {
            return Err("cannot release a live execution".into());
        }
        let key = execution_key(&self.prefix, &execution.target);
        let (mut compares, mut operations) = self.completion_changes(execution).await?;
        compares.extend([
            Compare::value(claim.key.as_str(), CompareOp::Equal, claim.token.as_bytes()),
            Compare::value(key.as_str(), CompareOp::Equal, encode(execution)),
            Compare::value(
                target_lock_key(&self.prefix, &execution.target),
                CompareOp::Equal,
                execution_owner(execution),
            ),
        ]);
        operations.extend([
            self.remove_permits(execution),
            self.remove_progress(execution),
            TxnOp::delete(key, None),
            TxnOp::put(
                target_lock_key(&self.prefix, &execution.target),
                claim.token.clone(),
                Some(etcd_client::PutOptions::new().with_lease(claim.lease_id)),
            ),
        ]);
        // A byte-bounded batch can succeed while leaving historical WAL behind,
        // even when no new writes arrive to trigger demand. Persist continuation
        // with release so a master crash cannot lose it. The scheduler coalesces
        // it behind the current task, then queues another fair, bounded pass.
        // Zero progress, failures and local maintenance must not hot-loop.
        if execution.maintenance.is_none()
            && execution.phase == Phase::Finished
            && execution.error.is_none()
            && execution.reclaimed > 0
        {
            operations.push(TxnOp::put(
                self.request_key(&execution.target),
                execution.target.clone(),
                None,
            ));
        }
        self.transact(compares, operations).await
    }

    async fn replace(&self, old: &Execution, new: &Execution) -> Result<bool> {
        let key = execution_key(&self.prefix, &old.target);
        self.transact(
            vec![Compare::value(key.as_str(), CompareOp::Equal, encode(old))],
            vec![TxnOp::put(key, encode(new), None)],
        )
        .await
    }

    async fn transact(&self, compares: Vec<Compare>, operations: Vec<TxnOp>) -> Result<bool> {
        self.client
            .clone()
            .txn(Txn::new().when(compares).and_then(operations))
            .await
            .map(|r| r.succeeded())
            .map_err(|e| e.to_string())
    }
}

/// The executor owns this future independently of the request handler. On
/// cancellation, drop its scoped storage future *before* publishing Finished.
/// Callers must not pass a detached JoinHandle: dropping one does not stop it.
pub async fn execute_scoped<F, T>(
    work: F,
    timeout: std::time::Duration,
    mut cancel: tokio::sync::watch::Receiver<bool>,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    // Keep the work in this inner scope: select! only drops branch borrows when
    // the future was pinned outside it, which would publish completion too soon.
    tokio::select! {
        biased;
        _ = async {
            loop {
                if *cancel.borrow_and_update() { return; }
                if cancel.changed().await.is_err() { std::future::pending::<()>().await; }
            }
        } => Err("merge execution cancelled".into()),
        result = tokio::time::timeout(timeout, work) =>
            result.unwrap_or_else(|_| Err("merge execution deadline exceeded".into())),
    }
}

#[derive(Clone, Debug, clap::Args)]
pub struct EtcdConfig {
    /// Comma-separated etcd v3 endpoints. Scheduler state (task queue,
    /// lease-based claims, per-experiment write locks) lives in etcd so several
    /// stateless master replicas can share one queue. Required.
    #[arg(long, env = "ETCD_ENDPOINTS", value_delimiter = ',')]
    pub etcd_endpoints: Vec<String>,

    /// Namespace for all lance-context master keys in etcd.
    #[arg(long, env = "ETCD_PREFIX", default_value = "/lance-context/master")]
    pub etcd_prefix: String,

    /// Optional etcd username. `ETCD_PASSWORD` must also be set.
    #[arg(long, env = "ETCD_USERNAME")]
    pub etcd_username: Option<String>,

    /// Optional etcd password. `ETCD_USERNAME` must also be set.
    #[arg(long, env = "ETCD_PASSWORD")]
    pub etcd_password: Option<String>,

    /// Optional PEM CA certificate path for etcd TLS.
    #[arg(long, env = "ETCD_CA_CERT")]
    pub etcd_ca_cert: Option<String>,

    /// Optional PEM client certificate path for etcd mutual TLS.
    #[arg(long, env = "ETCD_CLIENT_CERT")]
    pub etcd_client_cert: Option<String>,

    /// Optional PEM client private-key path for etcd mutual TLS.
    #[arg(long, env = "ETCD_CLIENT_KEY")]
    pub etcd_client_key: Option<String>,
}

impl EtcdConfig {
    pub async fn connect(&self) -> Result<Coordinator> {
        use etcd_client::{Certificate, ConnectOptions, Identity, TlsOptions};
        use std::time::Duration;
        let config = self;
        let mut options = ConnectOptions::new()
            .with_connect_timeout(Duration::from_secs(5))
            .with_timeout(Duration::from_secs(10))
            .with_keep_alive(Duration::from_secs(10), Duration::from_secs(3))
            .with_require_leader(true);
        match (&config.etcd_username, &config.etcd_password) {
            (Some(username), Some(password)) => {
                options = options.with_user(username, password);
            }
            (None, None) => {}
            _ => {
                return Err(String::from(
                    "ETCD_USERNAME and ETCD_PASSWORD must be configured together",
                ))
            }
        }
        if let Some(path) = &config.etcd_ca_cert {
            let pem = std::fs::read(path)
                .map_err(|err| format!("failed to read ETCD_CA_CERT '{path}': {err}"))?;
            let mut tls = TlsOptions::new().ca_certificate(Certificate::from_pem(pem));
            match (&config.etcd_client_cert, &config.etcd_client_key) {
                (Some(cert), Some(key)) => {
                    let cert_pem = std::fs::read(cert).map_err(|err| {
                        format!("failed to read ETCD_CLIENT_CERT '{cert}': {err}")
                    })?;
                    let key_pem = std::fs::read(key)
                        .map_err(|err| format!("failed to read ETCD_CLIENT_KEY '{key}': {err}"))?;
                    tls = tls.identity(Identity::from_pem(cert_pem, key_pem));
                }
                (None, None) => {}
                _ => {
                    return Err(String::from(
                        "ETCD_CLIENT_CERT and ETCD_CLIENT_KEY must be configured together",
                    ))
                }
            }
            options = options.with_tls(tls);
        } else if config.etcd_client_cert.is_some() || config.etcd_client_key.is_some() {
            return Err(String::from(
                "ETCD_CA_CERT is required when configuring an etcd client certificate",
            ));
        }
        let client = Client::connect(config.etcd_endpoints.clone(), Some(options))
            .await
            .map_err(|e| e.to_string())?;

        Ok(Coordinator::new(client, self.etcd_prefix.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        time::Duration,
    };
    use tokio::sync::{oneshot, watch};

    struct InFlight(Arc<AtomicBool>);
    impl Drop for InFlight {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_drops_the_writer_before_terminal_acknowledgement() {
        let writing = Arc::new(AtomicBool::new(false));
        let flag = writing.clone();
        let (_cancel, rx) = watch::channel(false);
        let result = execute_scoped(
            async move {
                flag.store(true, Ordering::SeqCst);
                let _guard = InFlight(flag);
                std::future::pending::<Result<usize>>().await
            },
            Duration::from_secs(600),
            rx,
        )
        .await;
        assert!(result.unwrap_err().contains("deadline"));
        assert!(!writing.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn disconnected_caller_does_not_detach_uncontrolled_work() {
        let writing = Arc::new(AtomicBool::new(false));
        let flag = writing.clone();
        let (cancel, rx) = watch::channel(false);
        let (entered, began) = oneshot::channel();
        // The HTTP handler owns only admission, never this executor's lifetime.
        let executor = tokio::spawn(execute_scoped(
            async move {
                flag.store(true, Ordering::SeqCst);
                let _guard = InFlight(flag);
                entered.send(()).unwrap();
                std::future::pending::<Result<usize>>().await
            },
            Duration::from_secs(600),
            rx,
        ));
        began.await.unwrap();
        assert!(writing.load(Ordering::SeqCst));
        cancel.send(true).unwrap();
        assert!(executor.await.unwrap().unwrap_err().contains("cancelled"));
        assert!(!writing.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn pre_cancelled_request_never_polls_storage() {
        let (cancel, rx) = watch::channel(false);
        cancel.send(true).unwrap();
        let result: Result<usize> = execute_scoped(
            async { panic!("must not execute storage") },
            Duration::from_secs(1),
            rx,
        )
        .await;
        assert!(result.is_err());
    }

    #[test]
    fn maintenance_records_preserve_legacy_worker_encoding_and_require_new_readers() {
        let legacy = Execution::new("table", "http://worker", "instance", 600);
        let encoded = encode(&legacy);
        assert!(!String::from_utf8_lossy(&encoded).contains("maintenance"));
        assert_eq!(
            serde_json::from_slice::<Execution>(&encoded).unwrap(),
            legacy
        );
        for kind in [
            MaintenanceKind::Compact,
            MaintenanceKind::IndexId,
            MaintenanceKind::Repair,
        ] {
            let mut local = legacy.clone();
            local.maintenance = Some(kind);
            local.endpoint = kind.endpoint().into();
            let encoded = encode(&local);
            assert_eq!(
                serde_json::from_slice::<Execution>(&encoded).unwrap(),
                local
            );
            assert_eq!(MaintenanceKind::from_endpoint(&local.endpoint), Some(kind));
            // An old reader omits the new field on re-encoding: its value CAS
            // cannot start, finish, freeze or release this local execution.
            let mut old_reader = local.clone();
            old_reader.maintenance = None;
            assert_ne!(encode(&old_reader), encoded);
        }
        assert_eq!(MaintenanceKind::from_endpoint("http://worker"), None);
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn execution_inventory_pages_without_crossing_namespace() {
        let (coordinator, mut client, _, _) = fixture().await;
        for n in 0..257 {
            let execution = Execution::new(&format!("table-{n:03}"), "master:repair", "lost", 60);
            client
                .put(
                    execution_key(&coordinator.prefix, &execution.target),
                    encode(&execution),
                    None,
                )
                .await
                .unwrap();
        }
        let (first, cursor) = coordinator.execution_page(None).await.unwrap();
        assert_eq!(first.len(), 256);
        let (last, done) = coordinator.execution_page(cursor.as_deref()).await.unwrap();
        assert_eq!(last.len(), 1);
        assert!(done.is_none());
        assert!(!first.iter().any(|e| e.id == last[0].id));
        assert!(coordinator
            .execution_page(Some("/another-prefix/key"))
            .await
            .is_err());
    }

    // Isolated prefix on a LOCAL test etcd. These tests never use production.
    async fn fixture() -> (Coordinator, Client, ClaimProof, i64) {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS")
            .expect("ETCD_TEST_ENDPOINTS is required for ignored etcd tests");
        let mut client = Client::connect([endpoint], None).await.unwrap();
        let prefix = format!("/merge-execution-tests/{}", uuid::Uuid::new_v4());
        let lease = client.lease_grant(30, None).await.unwrap().id();
        let proof = ClaimProof {
            key: format!("{prefix}/claims/test"),
            token: "original".into(),
            lease_id: lease,
        };
        client
            .put(
                target_lock_key(&prefix, "table"),
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
        (
            Coordinator::new(client.clone(), prefix),
            client,
            proof,
            lease,
        )
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn successful_release_durably_requests_another_pass_without_resetting_other_failures() {
        let (coordinator, client, proof, _) = fixture().await;
        let failure = coordinator
            .record_failure(&proof, "table", "broken-worker", "schema mismatch")
            .await
            .unwrap();
        coordinator.request_merge("table").await.unwrap();
        let old_request = coordinator.request_page(None).await.unwrap().0.remove(0);
        let execution = Execution::new("table", "worker", "boot", 600);
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        assert!(coordinator.finish(&running, Ok(3)).await.unwrap());
        let finished = coordinator.get("table").await.unwrap().unwrap();
        let stale_claim = ClaimProof {
            token: "stale".into(),
            ..proof.clone()
        };
        assert!(!coordinator.release(&stale_claim, &finished).await.unwrap());
        assert_eq!(
            coordinator.request_page(None).await.unwrap().0[0].revision,
            old_request.revision
        );
        assert!(coordinator.release(&proof, &finished).await.unwrap());
        // Simulate restarting the master after releasing the execution, before
        // it has finished its task. The continuation survives that crash window.
        let reconnected = Coordinator::new(client, coordinator.prefix.clone());
        assert!(reconnected.get("table").await.unwrap().is_none());
        assert!(!reconnected.acknowledge_request(&old_request).await.unwrap());
        let requests = reconnected.request_page(None).await.unwrap().0;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].target, "table");
        let after = reconnected
            .failure("table", "broken-worker")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.consecutive_attempts, failure.consecutive_attempts);
        assert_eq!(after.next_retry_ms, failure.next_retry_ms);
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn empty_failed_and_maintenance_releases_do_not_request_more_merges() {
        let (coordinator, _, proof, _) = fixture().await;
        for (maintenance, outcome) in [
            (None, Ok(0)),
            (None, Err("storage timeout".into())),
            (Some(MaintenanceKind::Compact), Ok(4)),
            (Some(MaintenanceKind::IndexId), Ok(4)),
        ] {
            let mut execution = Execution::new("table", "worker", "boot", 600);
            execution.maintenance = maintenance;
            assert!(coordinator.reserve(&proof, &execution).await.unwrap());
            let running = coordinator.start(&execution).await.unwrap().unwrap();
            assert!(coordinator.finish(&running, outcome).await.unwrap());
            let finished = coordinator.get("table").await.unwrap().unwrap();
            assert!(coordinator.release(&proof, &finished).await.unwrap());
            assert!(coordinator.request_page(None).await.unwrap().0.is_empty());
        }
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn worker_demand_coalesces_without_resetting_failure_backoff() {
        let (coordinator, _, proof, _) = fixture().await;
        let mut last = None;
        for _ in 0..4 {
            last = Some(
                coordinator
                    .record_failure(&proof, "table", "worker", "storage timeout")
                    .await
                    .unwrap(),
            );
        }
        for _ in 0..10 {
            coordinator.request_merge("table").await.unwrap();
        }
        let (rows, next) = coordinator.request_page(None).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].target, "table");
        assert!(next.is_none());
        let after = coordinator
            .failure("table", "worker")
            .await
            .unwrap()
            .unwrap();
        let before = last.unwrap();
        assert_eq!(after.consecutive_attempts, 4);
        assert_eq!(after.next_retry_ms, before.next_retry_ms);
        // A later flush must survive the old demand acknowledgement.
        coordinator.request_merge("table").await.unwrap();
        assert!(!coordinator.acknowledge_request(&rows[0]).await.unwrap());
        let newer = coordinator.request_page(None).await.unwrap().0;
        assert!(coordinator.acknowledge_request(&newer[0]).await.unwrap());
        assert!(coordinator.request_page(None).await.unwrap().0.is_empty());
        coordinator.request_merge("table").await.unwrap();
        assert_eq!(
            coordinator.request_page(None).await.unwrap().0[0].target,
            "table"
        );
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn progress_cannot_resurrect_a_frozen_or_released_execution() {
        let (coordinator, _, proof, _) = fixture().await;
        let execution = Execution::new("table", "worker", "boot", 3600);
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        assert!(coordinator.publish_progress(&running, 4).await.unwrap());
        assert_eq!(
            coordinator
                .progress(&running)
                .await
                .unwrap()
                .unwrap()
                .sequence,
            4
        );
        let frozen = coordinator.freeze(&proof, &running).await.unwrap().unwrap();
        assert!(!coordinator.publish_progress(&running, 5).await.unwrap());
        // Empty watermarks: this test did not authorize any storage writes.
        assert!(coordinator.finish_recovery(&proof, &frozen).await.unwrap());
        let recovered = coordinator.get("table").await.unwrap().unwrap();
        assert!(coordinator.release(&proof, &recovered).await.unwrap());
        assert!(coordinator.progress(&running).await.unwrap().is_none());
        assert!(!coordinator.publish_progress(&running, 6).await.unwrap());
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn recovery_closes_commit_admission_and_retains_every_allowed_version() {
        let (coordinator, mut client, proof, lease) = fixture().await;
        let operation = Execution::new("table", "worker", "boot", 1);
        assert!(coordinator.reserve(&proof, &operation).await.unwrap());
        let running = coordinator.start(&operation).await.unwrap().unwrap();
        coordinator
            .authorize_commit(&running, "file:///table", "base", 7)
            .await
            .unwrap();
        assert!(coordinator
            .authorize_commit(&running, "file:///wrong-table", "base", 8)
            .await
            .is_err());
        client.lease_revoke(lease).await.unwrap();
        // The old executor remains protected independently of its scheduler.
        coordinator
            .authorize_commit(&running, "file:///table", "base", 8)
            .await
            .unwrap();
        assert!(coordinator
            .freeze(&proof, &running)
            .await
            .unwrap()
            .is_none());
        let next = ClaimProof {
            token: "recovery-owner".into(),
            lease_id: 0,
            ..proof.clone()
        };
        client
            .put(next.key.clone(), next.token.clone(), None)
            .await
            .unwrap();
        let shard = format!("shard:{}", uuid::Uuid::new_v4());
        coordinator
            .authorize_commit(&running, "file:///table", &shard, 91)
            .await
            .unwrap();
        let (admitted, frozen) = tokio::join!(
            coordinator.authorize_commit(&running, "file:///table", "base", 9),
            coordinator.freeze(&next, &running),
        );
        let frozen = frozen.unwrap().unwrap();
        let plan = coordinator.watermarks(&frozen).await.unwrap();
        assert_eq!(plan.dataset_uri.as_deref(), Some("file:///table"));
        assert_eq!(plan.versions["base"], if admitted.is_ok() { 9 } else { 8 });
        assert_eq!(plan.versions[&shard], 91);
        assert!(coordinator
            .authorize_commit(&running, "file:///table", "base", 10)
            .await
            .is_err());
        assert!(coordinator
            .authorize_commit(&running, "file:///table", &shard, 92)
            .await
            .is_err());
        assert!(coordinator.release(&next, &frozen).await.is_err());
        assert!(!coordinator.finish_recovery(&proof, &frozen).await.unwrap());
        // Storage fencing itself is tested against real manifests in core.
        assert!(coordinator.finish_recovery(&next, &frozen).await.unwrap());
        let recovered = coordinator.get("table").await.unwrap().unwrap();
        assert_eq!(recovered.phase, Phase::Recovered);
        assert!(coordinator.release(&next, &recovered).await.unwrap());
        assert!(coordinator
            .authorize_commit(&running, "file:///table", "base", 11)
            .await
            .is_err());
        // A successful barrier must not disguise a permanent data failure as
        // another ownership failure with a short metadata-probe backoff.
        let operation = Execution::new("table", "worker", "boot", 1);
        assert!(coordinator.reserve(&next, &operation).await.unwrap());
        let running = coordinator.start(&operation).await.unwrap().unwrap();
        assert!(coordinator
            .report_uncertain(
                &running,
                "merge ownership unresolved: manifest commit result unknown; schema mismatch"
                    .into(),
            )
            .await
            .unwrap());
        let uncertain = coordinator.get("table").await.unwrap().unwrap();
        let frozen = coordinator
            .freeze(&next, &uncertain)
            .await
            .unwrap()
            .unwrap();
        assert!(coordinator.finish_recovery(&next, &frozen).await.unwrap());
        let recovered = coordinator.get("table").await.unwrap().unwrap();
        assert!(coordinator.release(&next, &recovered).await.unwrap());
        let failure = coordinator
            .failure("table", "worker")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(failure.class, failure::FailureClass::DataOrConfiguration);
        assert!(failure.needs_attention);
        assert!(failure.last_error.contains("schema mismatch"));
        assert_eq!(failure.next_retry_ms - failure.last_failure_ms, 3_600_000);
        client
            .delete(
                coordinator.prefix.clone(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn ambiguous_storage_response_cannot_release_the_execution() {
        let (coordinator, mut client, proof, lease) = fixture().await;
        let operation = Execution::new("table", "worker", "boot", 1);
        assert!(coordinator.reserve(&proof, &operation).await.unwrap());
        let running = coordinator.start(&operation).await.unwrap().unwrap();
        assert!(coordinator
            .report_uncertain(&running, "storage response lost".into())
            .await
            .unwrap());
        let uncertain = coordinator.get("table").await.unwrap().unwrap();
        assert_eq!(uncertain.phase, Phase::Uncertain);
        assert!(coordinator.release(&proof, &uncertain).await.is_err());
        assert!(!coordinator.finish(&running, Ok(1)).await.unwrap());
        client.lease_revoke(lease).await.unwrap();
        assert_eq!(coordinator.get("table").await.unwrap(), Some(uncertain));
        client
            .delete(
                coordinator.prefix.clone(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn failure_budget_survives_reconnect_and_lost_claim_cannot_clear_it() {
        let (coordinator, mut client, proof, lease) = fixture().await;
        for count in 1..=3 {
            let failure = coordinator
                .record_failure(&proof, "table", "worker", "temporary failure")
                .await
                .unwrap();
            assert_eq!(failure.consecutive_attempts, count);
        }
        let reconnected = Coordinator::new(client.clone(), coordinator.prefix.clone());
        let failure = reconnected
            .failure("table", "worker")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(failure.consecutive_attempts, 3);
        assert!(failure.next_retry_ms > failure.last_failure_ms + 20_000);
        reconnected
            .record_failure(&proof, "table2", "worker", "schema mismatch")
            .await
            .unwrap();
        let (page, next) = reconnected.failure_page(None, 1).await.unwrap();
        assert_eq!(page.len(), 1);
        let (second, end) = reconnected.failure_page(next.as_deref(), 1).await.unwrap();
        assert_eq!(second.len(), 1);
        assert_ne!(page[0].target, second[0].target);
        assert!(end.is_none());
        client.lease_revoke(lease).await.unwrap();
        assert!(reconnected
            .clear_failure(&proof, "table", "worker")
            .await
            .is_err());
        assert!(reconnected
            .record_failure(&proof, "table", "worker", "stale writer")
            .await
            .is_err());
        assert_eq!(
            reconnected
                .failure("table", "worker")
                .await
                .unwrap()
                .unwrap()
                .consecutive_attempts,
            3
        );
        client
            .put(proof.key.clone(), proof.token.clone(), None)
            .await
            .unwrap();
        reconnected
            .clear_failure(&proof, "table", "worker")
            .await
            .unwrap();
        assert!(reconnected
            .failure("table", "worker")
            .await
            .unwrap()
            .is_none());
        client
            .delete(
                coordinator.prefix.clone(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn revoked_claim_cannot_release_surviving_server_work() {
        let (coordinator, mut client, proof, lease) = fixture().await;
        let operation = Execution::new("table", "worker", "boot", 600);
        assert!(coordinator.reserve(&proof, &operation).await.unwrap());
        let running = coordinator.start(&operation).await.unwrap().unwrap();
        client.lease_revoke(lease).await.unwrap();
        assert_eq!(
            coordinator.get("table").await.unwrap(),
            Some(running.clone())
        );
        let next = ClaimProof {
            key: proof.key.clone(),
            token: "replacement".into(),
            lease_id: 0,
        };
        client
            .put(next.key.clone(), next.token.clone(), None)
            .await
            .unwrap();
        assert!(!coordinator
            .reserve(&next, &Execution::new("table", "worker2", "boot2", 600))
            .await
            .unwrap());
        assert!(coordinator.release(&next, &running).await.is_err());
        assert!(coordinator
            .finish(&running, Err("cancelled".into()))
            .await
            .unwrap());
        let done = coordinator.get("table").await.unwrap().unwrap();
        assert!(!coordinator.release(&proof, &done).await.unwrap());
        assert!(coordinator
            .failure("table", "worker")
            .await
            .unwrap()
            .is_none());
        assert!(coordinator.release(&next, &done).await.unwrap());
        assert_eq!(
            coordinator
                .failure("table", "worker")
                .await
                .unwrap()
                .unwrap()
                .consecutive_attempts,
            1,
            "releasing a failed execution must atomically persist the retry budget"
        );
        assert!(coordinator
            .reserve(&next, &Execution::new("table", "worker2", "boot2", 600))
            .await
            .unwrap());
        client
            .delete(
                coordinator.prefix.clone(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn delayed_http_start_cannot_resurrect_cancelled_operation() {
        let (coordinator, mut client, proof, lease) = fixture().await;
        let old = Execution::new("table", "worker", "boot", 600);
        assert!(coordinator.reserve(&proof, &old).await.unwrap());
        assert!(coordinator.cancel_reserved(&old).await.unwrap());
        let done = coordinator.get("table").await.unwrap().unwrap();
        assert!(coordinator.release(&proof, &done).await.unwrap());
        let new = Execution::new("table", "worker2", "boot2", 600);
        assert!(coordinator.reserve(&proof, &new).await.unwrap());
        assert!(coordinator.start(&old).await.unwrap().is_none());
        assert_eq!(coordinator.get("table").await.unwrap(), Some(new));
        client.lease_revoke(lease).await.unwrap();
        client
            .delete(
                coordinator.prefix.clone(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }
}
