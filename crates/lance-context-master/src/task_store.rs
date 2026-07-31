//! Durable scheduler task storage backed by etcd.
//!
//! Scheduler state lives entirely in etcd: compare-and-swap enqueue,
//! lease-backed task claims, and distributed per-experiment write locks so
//! several stateless masters can safely drain one shared queue. Because claims
//! ride renewed leases, a master that disappears has its in-flight task requeued
//! after the lease expires — execution is at-least-once, so task
//! implementations must be idempotent.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use etcd_client::{
    Certificate, Client, Compare, CompareOp, ConnectOptions, GetOptions, Identity, PutOptions,
    TlsOptions, Txn, TxnOp,
};
use lance_context_api::{TaskKind, TaskRecord, TaskState};
use lance_context_core::generate_id;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::MasterConfig;

const TASK_POLL_BATCH: usize = 256;

#[derive(Clone)]
pub struct TaskStore {
    inner: Arc<EtcdTaskStore>,
    history_limit: usize,
    history_ttl_secs: u64,
}

struct EtcdTaskStore {
    client: Client,
    prefix: String,
    lease_ttl: i64,
}

/// Ownership of one running task. Dropping the claim stops lease renewal; etcd
/// then removes its claim and target-lock keys, allowing recovery.
pub struct TaskClaim {
    pub task: TaskRecord,
    backend: ClaimBackend,
}

struct ClaimBackend {
    token: String,
    lease_id: i64,
    claim_key: String,
    target_key: Option<String>,
    keepalive: LeaseKeepalive,
}

/// Short-lived distributed guard used to serialize stats-table writers across
/// master replicas.
pub struct CoordinationGuard {
    token: String,
    lease_id: i64,
    key: String,
    keepalive: LeaseKeepalive,
}

struct LeaseKeepalive {
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl Drop for LeaseKeepalive {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        self.task.abort();
    }
}

impl TaskStore {
    pub async fn open(config: &MasterConfig) -> lance::Result<Self> {
        let store = Self {
            inner: Arc::new(EtcdTaskStore::connect(config).await?),
            history_limit: config.task_history_limit.max(1),
            history_ttl_secs: config.task_history_ttl_secs,
        };
        store.recover_orphaned().await?;
        store.prune_terminal_history().await?;
        Ok(store)
    }

    /// Atomically enqueue a task. Standalone Compact, IndexId, and depless
    /// MergeWal tasks use an etcd dedupe key while queued or running. Tasks with
    /// dependencies are always distinct because they belong to a specific
    /// ordered chain.
    pub async fn enqueue(
        &self,
        kind: TaskKind,
        target: &str,
        depends_on: Vec<String>,
    ) -> lance::Result<TaskRecord> {
        self.inner.enqueue(kind, target, depends_on).await
    }

    pub async fn list(&self) -> lance::Result<Vec<TaskRecord>> {
        self.inner.list().await
    }

    pub async fn get(&self, id: &str) -> lance::Result<Option<TaskRecord>> {
        self.inner.get(id).await
    }

    pub async fn queue_depth(&self) -> lance::Result<usize> {
        self.inner.queue_depth().await
    }

    /// Claim the oldest runnable task. The queued->running update, claim lease,
    /// and per-experiment write lock are one etcd transaction.
    pub async fn claim_next(&self) -> lance::Result<Option<TaskClaim>> {
        self.inner.recover_orphaned().await?;
        let (claim, dependency_failed) = self.inner.claim_next().await?;
        if dependency_failed {
            if let Err(error) = self.prune_terminal_history().await {
                tracing::warn!(error = %error, "failed to prune task history");
            }
        }
        Ok(claim)
    }

    /// Finish a claimed task and release its claim/target lock.
    pub async fn finish(
        &self,
        claim: TaskClaim,
        outcome: Result<String, String>,
    ) -> lance::Result<()> {
        self.inner.finish(claim, outcome).await?;
        self.prune_terminal_history().await.map(|_| ())
    }

    /// Try to acquire a named coordination lock without waiting. This is used
    /// around the shared Lance stats table, whose mutations must have one writer
    /// across all master replicas.
    pub async fn try_coordination_lock(
        &self,
        name: &str,
    ) -> lance::Result<Option<CoordinationGuard>> {
        self.inner.try_coordination_lock(name).await
    }

    pub async fn coordination_lock(&self, name: &str) -> lance::Result<CoordinationGuard> {
        loop {
            if let Some(guard) = self.try_coordination_lock(name).await? {
                return Ok(guard);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub async fn release_coordination_lock(&self, guard: CoordinationGuard) -> lance::Result<()> {
        let CoordinationGuard {
            token,
            lease_id,
            key,
            keepalive,
        } = guard;
        drop(keepalive);
        self.inner.delete_owned_key(&key, &token).await?;
        self.inner.revoke_lease(lease_id).await
    }

    async fn recover_orphaned(&self) -> lance::Result<usize> {
        self.inner.recover_orphaned().await
    }

    async fn prune_terminal_history(&self) -> lance::Result<usize> {
        let tasks = self.list().await?;
        let ttl_cutoff = if self.history_ttl_secs > 0 {
            Some(now_ms() - (self.history_ttl_secs as i64) * 1_000)
        } else {
            None
        };
        let ids = prunable_terminal_ids(tasks, self.history_limit, ttl_cutoff);
        if ids.is_empty() {
            return Ok(0);
        }
        self.inner.delete_many(&ids).await?;
        Ok(ids.len())
    }
}

/// Select terminal (Done/Failed) task ids to prune under two independent
/// policies, whichever removes a task first:
/// - **count**: keep only the newest `history_limit` terminal tasks;
/// - **age (TTL)**: when `ttl_cutoff` is `Some`, drop terminal tasks whose
///   `finished_at` (fallback `enqueued_at`) is at or before the cutoff.
///
/// Queued/Running tasks, and any terminal task a live (Queued/Running) task
/// still lists in `depends_on`, are never selected regardless of age or count.
/// Pure and etcd-free so the policy is unit-testable.
fn prunable_terminal_ids(
    tasks: Vec<TaskRecord>,
    history_limit: usize,
    ttl_cutoff: Option<i64>,
) -> Vec<String> {
    let protected = tasks
        .iter()
        .filter(|task| matches!(task.state, TaskState::Queued | TaskState::Running))
        .flat_map(|task| task.depends_on.iter().cloned())
        .collect::<HashSet<_>>();
    let mut terminal = tasks
        .into_iter()
        .filter(|task| matches!(task.state, TaskState::Done | TaskState::Failed))
        .filter(|task| !protected.contains(&task.id))
        .collect::<Vec<_>>();
    // Newest first, so rank >= history_limit are the ones over the count cap.
    terminal.sort_by_key(|task| std::cmp::Reverse(task.finished_at.unwrap_or(task.enqueued_at)));
    terminal
        .into_iter()
        .enumerate()
        .filter(|(rank, task)| {
            let over_count = *rank >= history_limit;
            let expired = ttl_cutoff
                .is_some_and(|cutoff| task.finished_at.unwrap_or(task.enqueued_at) <= cutoff);
            over_count || expired
        })
        .map(|(_, task)| task.id)
        .collect()
}

impl EtcdTaskStore {
    async fn connect(config: &MasterConfig) -> lance::Result<Self> {
        if config.etcd_endpoints.is_empty() {
            return Err(lance::Error::io(
                "ETCD_ENDPOINTS is required to run the master",
            ));
        }
        if config.etcd_lease_ttl_secs < 5 {
            return Err(lance::Error::io("ETCD_LEASE_TTL_SECS must be at least 5"));
        }
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
                return Err(lance::Error::io(
                    "ETCD_USERNAME and ETCD_PASSWORD must be configured together",
                ))
            }
        }
        if let Some(path) = &config.etcd_ca_cert {
            let pem = std::fs::read(path).map_err(|err| {
                lance::Error::io(format!("failed to read ETCD_CA_CERT '{path}': {err}"))
            })?;
            let mut tls = TlsOptions::new().ca_certificate(Certificate::from_pem(pem));
            match (&config.etcd_client_cert, &config.etcd_client_key) {
                (Some(cert), Some(key)) => {
                    let cert_pem = std::fs::read(cert).map_err(|err| {
                        lance::Error::io(format!("failed to read ETCD_CLIENT_CERT '{cert}': {err}"))
                    })?;
                    let key_pem = std::fs::read(key).map_err(|err| {
                        lance::Error::io(format!("failed to read ETCD_CLIENT_KEY '{key}': {err}"))
                    })?;
                    tls = tls.identity(Identity::from_pem(cert_pem, key_pem));
                }
                (None, None) => {}
                _ => {
                    return Err(lance::Error::io(
                        "ETCD_CLIENT_CERT and ETCD_CLIENT_KEY must be configured together",
                    ))
                }
            }
            options = options.with_tls(tls);
        } else if config.etcd_client_cert.is_some() || config.etcd_client_key.is_some() {
            return Err(lance::Error::io(
                "ETCD_CA_CERT is required when configuring an etcd client certificate",
            ));
        }
        let client = Client::connect(config.etcd_endpoints.clone(), Some(options))
            .await
            .map_err(etcd_error("connect to etcd"))?;
        Ok(Self {
            client,
            prefix: config.etcd_prefix.trim_end_matches('/').to_string(),
            lease_ttl: config.etcd_lease_ttl_secs,
        })
    }

    async fn enqueue(
        &self,
        kind: TaskKind,
        target: &str,
        depends_on: Vec<String>,
    ) -> lance::Result<TaskRecord> {
        let task = new_task(kind, target, depends_on);
        let task_key = self.task_key(&task.id);
        let queue_key = self.queue_key(&task.id);
        let value = encode_task(&task)?;
        let mut client = self.client.clone();
        if let Some(dedupe_key) = self.dedupe_key(kind, target, &task.depends_on) {
            for _ in 0..4 {
                let txn = Txn::new()
                    .when([Compare::version(dedupe_key.as_str(), CompareOp::Equal, 0)])
                    .and_then([
                        TxnOp::put(task_key.as_str(), value.clone(), None),
                        TxnOp::put(queue_key.as_str(), value.clone(), None),
                        TxnOp::put(dedupe_key.as_str(), task.id.as_bytes(), None),
                    ]);
                if client
                    .txn(txn)
                    .await
                    .map_err(etcd_error("enqueue task"))?
                    .succeeded()
                {
                    return Ok(task);
                }
                if let Some(existing_id) = self.get_text(&dedupe_key).await? {
                    if let Some(existing) = self.get(&existing_id).await? {
                        if matches!(existing.state, TaskState::Queued | TaskState::Running) {
                            return Ok(existing);
                        }
                    }
                    self.delete_owned_key(&dedupe_key, &existing_id).await?;
                }
            }
            return Err(lance::Error::io(
                "failed to resolve concurrent etcd task enqueue",
            ));
        }
        client
            .txn(Txn::new().and_then([
                TxnOp::put(task_key, value.clone(), None),
                TxnOp::put(queue_key, value, None),
            ]))
            .await
            .map_err(etcd_error("enqueue task"))?;
        Ok(task)
    }

    async fn get(&self, id: &str) -> lance::Result<Option<TaskRecord>> {
        let mut client = self.client.clone();
        let response = client
            .get(self.task_key(id), None)
            .await
            .map_err(etcd_error("read task"))?;
        response
            .kvs()
            .first()
            .map(|kv| decode_task(kv.value(), id))
            .transpose()
    }

    async fn list(&self) -> lance::Result<Vec<TaskRecord>> {
        let mut client = self.client.clone();
        let response = client
            .get(self.tasks_prefix(), Some(GetOptions::new().with_prefix()))
            .await
            .map_err(etcd_error("list tasks"))?;
        response
            .kvs()
            .iter()
            .map(|kv| decode_task(kv.value(), &String::from_utf8_lossy(kv.key())))
            .collect()
    }

    async fn queue_depth(&self) -> lance::Result<usize> {
        let mut client = self.client.clone();
        let response = client
            .get(
                self.queue_prefix(),
                Some(GetOptions::new().with_prefix().with_count_only()),
            )
            .await
            .map_err(etcd_error("count queued tasks"))?;
        usize::try_from(response.count())
            .map_err(|_| lance::Error::io("etcd returned an invalid queue count"))
    }

    async fn claim_next(&self) -> lance::Result<(Option<TaskClaim>, bool)> {
        let prefix = self.queue_prefix();
        let range_end = prefix_range_end(prefix.as_bytes());
        let mut start_key = prefix.into_bytes();
        let mut dependency_failed = false;

        loop {
            let mut client = self.client.clone();
            let response = client
                .get(
                    start_key,
                    Some(
                        GetOptions::new()
                            .with_range(range_end.clone())
                            .with_limit(TASK_POLL_BATCH as i64),
                    ),
                )
                .await
                .map_err(etcd_error("list queued tasks"))?;
            let more = response.more();
            let next_key = response.kvs().last().map(|kv| {
                let mut key = kv.key().to_vec();
                key.push(0);
                key
            });
            let queued = response
                .kvs()
                .iter()
                .map(|kv| decode_task(kv.value(), &String::from_utf8_lossy(kv.key())))
                .collect::<lance::Result<Vec<_>>>()?;

            for mut task in queued {
                match self.dependency_status(&task).await? {
                    DependencyStatus::Ready => {}
                    DependencyStatus::Waiting => continue,
                    DependencyStatus::Failed(dependency) => {
                        if self.fail_dependency(&task, &dependency).await? {
                            dependency_failed = true;
                        }
                        continue;
                    }
                }

                let token = generate_id();
                let lease_id = self.grant_lease().await?;
                let claim_key = self.claim_key(&task.id);
                let target_key =
                    requires_target_lock(task.kind).then(|| self.target_lock_key(&task.target));
                let queue_key = self.queue_key(&task.id);
                let running_key = self.running_key(&task.id);
                let queued_value = encode_task(&task)?;
                task.state = TaskState::Running;
                task.started_at = Some(now_ms());
                let running_value = encode_task(&task)?;
                let mut compares = vec![
                    Compare::value(self.task_key(&task.id), CompareOp::Equal, queued_value),
                    Compare::version(claim_key.as_str(), CompareOp::Equal, 0),
                    Compare::version(queue_key.as_str(), CompareOp::Greater, 0),
                ];
                if let Some(key) = &target_key {
                    compares.push(Compare::version(key.as_str(), CompareOp::Equal, 0));
                }
                let lease_options = Some(PutOptions::new().with_lease(lease_id));
                let mut operations = vec![
                    TxnOp::put(self.task_key(&task.id), running_value.clone(), None),
                    TxnOp::delete(queue_key, None),
                    TxnOp::put(running_key, running_value, None),
                    TxnOp::put(claim_key.as_str(), token.as_bytes(), lease_options.clone()),
                ];
                if let Some(key) = &target_key {
                    operations.push(TxnOp::put(
                        key.as_str(),
                        token.as_bytes(),
                        lease_options.clone(),
                    ));
                }
                let mut client = self.client.clone();
                let claimed = client
                    .txn(Txn::new().when(compares).and_then(operations))
                    .await
                    .map_err(etcd_error("claim task"))?
                    .succeeded();
                if claimed {
                    let keepalive = self.start_keepalive(lease_id).await?;
                    return Ok((
                        Some(TaskClaim {
                            task,
                            backend: ClaimBackend {
                                token,
                                lease_id,
                                claim_key,
                                target_key,
                                keepalive,
                            },
                        }),
                        dependency_failed,
                    ));
                }
                self.revoke_lease(lease_id).await?;
            }

            if !more {
                return Ok((None, dependency_failed));
            }
            let Some(next_key) = next_key else {
                return Ok((None, dependency_failed));
            };
            start_key = next_key;
        }
    }

    async fn finish(&self, claim: TaskClaim, outcome: Result<String, String>) -> lance::Result<()> {
        let ClaimBackend {
            token,
            lease_id,
            claim_key,
            target_key,
            keepalive,
        } = claim.backend;
        let mut task = claim.task;
        apply_outcome(&mut task, outcome);
        let mut operations = vec![
            TxnOp::put(self.task_key(&task.id), encode_task(&task)?, None),
            TxnOp::delete(claim_key.as_str(), None),
            TxnOp::delete(self.running_key(&task.id), None),
        ];
        if let Some(key) = &target_key {
            operations.push(TxnOp::delete(key.as_str(), None));
        }
        if let Some(key) = self.dedupe_key(task.kind, &task.target, &task.depends_on) {
            operations.push(TxnOp::delete(key, None));
        }
        let mut client = self.client.clone();
        let completed = client
            .txn(
                Txn::new()
                    .when([Compare::value(
                        claim_key.as_str(),
                        CompareOp::Equal,
                        token.as_bytes(),
                    )])
                    .and_then(operations),
            )
            .await
            .map_err(etcd_error("complete task"))?
            .succeeded();
        drop(keepalive);
        self.revoke_lease(lease_id).await?;
        if !completed {
            return Err(lance::Error::io(format!(
                "task '{}' lost its etcd claim before completion",
                task.id
            )));
        }
        Ok(())
    }

    async fn recover_orphaned(&self) -> lance::Result<usize> {
        let mut client = self.client.clone();
        let response = client
            .get(self.running_prefix(), Some(GetOptions::new().with_prefix()))
            .await
            .map_err(etcd_error("list running tasks"))?;
        let running = response
            .kvs()
            .iter()
            .map(|kv| decode_task(kv.value(), &String::from_utf8_lossy(kv.key())))
            .collect::<lance::Result<Vec<_>>>()?;
        let mut recovered = 0;
        for mut task in running {
            let running_value = encode_task(&task)?;
            let claim_key = self.claim_key(&task.id);
            let running_key = self.running_key(&task.id);
            requeue(&mut task);
            let txn = Txn::new()
                .when([
                    Compare::value(self.task_key(&task.id), CompareOp::Equal, running_value),
                    Compare::version(claim_key.as_str(), CompareOp::Equal, 0),
                    Compare::version(running_key.as_str(), CompareOp::Greater, 0),
                ])
                .and_then([
                    TxnOp::put(self.task_key(&task.id), encode_task(&task)?, None),
                    TxnOp::put(self.queue_key(&task.id), encode_task(&task)?, None),
                    TxnOp::delete(running_key, None),
                ]);
            let mut client = self.client.clone();
            if client
                .txn(txn)
                .await
                .map_err(etcd_error("recover orphaned task"))?
                .succeeded()
            {
                recovered += 1;
            }
        }
        Ok(recovered)
    }

    async fn try_coordination_lock(&self, name: &str) -> lance::Result<Option<CoordinationGuard>> {
        let lease_id = self.grant_lease().await?;
        let token = generate_id();
        let key = format!("{}/coordination/{}", self.prefix, encode_segment(name));
        let txn = Txn::new()
            .when([Compare::version(key.as_str(), CompareOp::Equal, 0)])
            .and_then([TxnOp::put(
                key.as_str(),
                token.as_bytes(),
                Some(PutOptions::new().with_lease(lease_id)),
            )]);
        let mut client = self.client.clone();
        if !client
            .txn(txn)
            .await
            .map_err(etcd_error("acquire coordination lock"))?
            .succeeded()
        {
            self.revoke_lease(lease_id).await?;
            return Ok(None);
        }
        let keepalive = self.start_keepalive(lease_id).await?;
        Ok(Some(CoordinationGuard {
            token,
            lease_id,
            key,
            keepalive,
        }))
    }

    async fn grant_lease(&self) -> lance::Result<i64> {
        let mut client = self.client.clone();
        client
            .lease_grant(self.lease_ttl, None)
            .await
            .map(|response| response.id())
            .map_err(etcd_error("grant lease"))
    }

    async fn start_keepalive(&self, lease_id: i64) -> lance::Result<LeaseKeepalive> {
        let mut client = self.client.clone();
        let (mut keeper, mut stream) = client
            .lease_keep_alive(lease_id)
            .await
            .map_err(etcd_error("start lease keepalive"))?;
        let interval = Duration::from_secs((self.lease_ttl / 3).max(1) as u64);
        let (stop_tx, mut stop_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    _ = tokio::time::sleep(interval) => {
                        if keeper.keep_alive().await.is_err() {
                            break;
                        }
                        match tokio::time::timeout(interval, stream.message()).await {
                            Ok(Ok(Some(response))) if response.ttl() > 0 => {}
                            _ => break,
                        }
                    }
                }
            }
        });
        Ok(LeaseKeepalive {
            stop: Some(stop_tx),
            task,
        })
    }

    async fn revoke_lease(&self, lease_id: i64) -> lance::Result<()> {
        let mut client = self.client.clone();
        client
            .lease_revoke(lease_id)
            .await
            .map(|_| ())
            .map_err(etcd_error("revoke lease"))
    }

    async fn delete_owned_key(&self, key: &str, owner: &str) -> lance::Result<()> {
        let txn = Txn::new()
            .when([Compare::value(key, CompareOp::Equal, owner.as_bytes())])
            .and_then([TxnOp::delete(key, None)]);
        let mut client = self.client.clone();
        client
            .txn(txn)
            .await
            .map(|_| ())
            .map_err(etcd_error("release owned key"))
    }

    async fn get_text(&self, key: &str) -> lance::Result<Option<String>> {
        let mut client = self.client.clone();
        let response = client
            .get(key, None)
            .await
            .map_err(etcd_error("read etcd key"))?;
        response
            .kvs()
            .first()
            .map(|kv| {
                std::str::from_utf8(kv.value())
                    .map(str::to_string)
                    .map_err(|err| lance::Error::io(format!("invalid UTF-8 in etcd key: {err}")))
            })
            .transpose()
    }

    async fn dependency_status(&self, task: &TaskRecord) -> lance::Result<DependencyStatus> {
        let mut waiting = false;
        for dependency in &task.depends_on {
            match self.get(dependency).await? {
                Some(record) if record.state == TaskState::Done => {}
                Some(record) if record.state == TaskState::Failed => {
                    return Ok(DependencyStatus::Failed(dependency.clone()));
                }
                Some(_) => waiting = true,
                None => return Ok(DependencyStatus::Failed(dependency.clone())),
            }
        }
        Ok(if waiting {
            DependencyStatus::Waiting
        } else {
            DependencyStatus::Ready
        })
    }

    async fn fail_dependency(&self, task: &TaskRecord, dependency: &str) -> lance::Result<bool> {
        let queued_value = encode_task(task)?;
        let mut failed = task.clone();
        fail_dependency(&mut failed, dependency);
        let txn = Txn::new()
            .when([
                Compare::value(self.task_key(&task.id), CompareOp::Equal, queued_value),
                Compare::version(self.queue_key(&task.id), CompareOp::Greater, 0),
            ])
            .and_then([
                TxnOp::put(self.task_key(&task.id), encode_task(&failed)?, None),
                TxnOp::delete(self.queue_key(&task.id), None),
            ]);
        let mut client = self.client.clone();
        client
            .txn(txn)
            .await
            .map(|response| response.succeeded())
            .map_err(etcd_error("fail task with unsuccessful dependency"))
    }

    async fn delete_many(&self, ids: &[String]) -> lance::Result<()> {
        for chunk in ids.chunks(100) {
            let operations = chunk
                .iter()
                .map(|id| TxnOp::delete(self.task_key(id), None))
                .collect::<Vec<_>>();
            let mut client = self.client.clone();
            client
                .txn(Txn::new().and_then(operations))
                .await
                .map_err(etcd_error("prune task history"))?;
        }
        Ok(())
    }

    fn tasks_prefix(&self) -> String {
        format!("{}/tasks/", self.prefix)
    }

    fn queue_prefix(&self) -> String {
        format!("{}/queue/", self.prefix)
    }

    fn running_prefix(&self) -> String {
        format!("{}/running/", self.prefix)
    }

    fn task_key(&self, id: &str) -> String {
        format!("{}{id}", self.tasks_prefix())
    }

    fn queue_key(&self, id: &str) -> String {
        format!("{}{id}", self.queue_prefix())
    }

    fn running_key(&self, id: &str) -> String {
        format!("{}{id}", self.running_prefix())
    }

    fn claim_key(&self, id: &str) -> String {
        format!("{}/claims/{id}", self.prefix)
    }

    fn target_lock_key(&self, target: &str) -> String {
        format!("{}/target-locks/{}", self.prefix, encode_segment(target))
    }

    fn dedupe_key(&self, kind: TaskKind, target: &str, depends_on: &[String]) -> Option<String> {
        should_dedupe(kind, depends_on).then(|| {
            format!(
                "{}/dedupe/{}/{}",
                self.prefix,
                kind_label(kind),
                encode_segment(target)
            )
        })
    }
}

fn new_task(kind: TaskKind, target: &str, depends_on: Vec<String>) -> TaskRecord {
    TaskRecord {
        id: generate_id(),
        kind,
        target: target.to_string(),
        state: TaskState::Queued,
        error: None,
        detail: None,
        enqueued_at: now_ms(),
        started_at: None,
        finished_at: None,
        depends_on,
    }
}

fn apply_outcome(task: &mut TaskRecord, outcome: Result<String, String>) {
    task.finished_at = Some(now_ms());
    match outcome {
        Ok(detail) => {
            task.state = TaskState::Done;
            task.detail = Some(detail);
            task.error = None;
        }
        Err(error) => {
            task.state = TaskState::Failed;
            task.error = Some(error);
            task.detail = None;
        }
    }
}

fn requeue(task: &mut TaskRecord) {
    task.state = TaskState::Queued;
    task.started_at = None;
    task.finished_at = None;
    task.error = None;
    task.detail = None;
}

enum DependencyStatus {
    Ready,
    Waiting,
    Failed(String),
}

fn fail_dependency(task: &mut TaskRecord, dependency: &str) {
    apply_outcome(
        task,
        Err(format!("dependency {dependency} did not complete")),
    );
}

fn should_dedupe(kind: TaskKind, depends_on: &[String]) -> bool {
    // MergeWal is deduped for depless enqueues (periodic auto-sweep and the
    // manual "Merge WAL" button) so a slow fan-out cannot pile up duplicate
    // broadcasts for the same experiment. A MergeWal that is part of an ordered
    // Optimize chain carries `depends_on` and is intentionally not deduped.
    depends_on.is_empty()
        && matches!(
            kind,
            TaskKind::Compact | TaskKind::IndexId | TaskKind::MergeWal
        )
}

fn requires_target_lock(kind: TaskKind) -> bool {
    matches!(kind, TaskKind::Compact | TaskKind::IndexId)
}

fn kind_label(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Compact => "compact",
        TaskKind::MergeWal => "merge-wal",
        TaskKind::IndexId => "index-id",
    }
}

fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

fn encode_segment(value: &str) -> String {
    value
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn prefix_range_end(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    for index in (0..end.len()).rev() {
        if end[index] != u8::MAX {
            end[index] += 1;
            end.truncate(index + 1);
            return end;
        }
    }
    vec![0]
}

fn encode_task(task: &TaskRecord) -> lance::Result<Vec<u8>> {
    serde_json::to_vec(task)
        .map_err(|err| lance::Error::io(format!("failed to encode task '{}': {err}", task.id)))
}

fn decode_task(value: &[u8], key: &str) -> lance::Result<TaskRecord> {
    serde_json::from_slice(value)
        .map_err(|err| lance::Error::io(format!("failed to decode task '{key}': {err}")))
}

fn etcd_error(action: &'static str) -> impl FnOnce(etcd_client::Error) -> lance::Error {
    move |err| lance::Error::io(format!("failed to {action}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn config(dir: &TempDir) -> MasterConfig {
        MasterConfig {
            data_dir: dir.path().to_string_lossy().to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            stats_scan_interval_secs: 0,
            scan_concurrency: 4,
            rollout_cache_bytes: 2 * 1024 * 1024 * 1024,
            stats_maintenance_every_n_scans: 0,
            stats_history_ttl_secs: 3_600,
            stats_cold_retire_secs: 0,
            compaction_interval_secs: 0,
            min_fragments: 16,
            target_rows_per_fragment: 1_048_576,
            merge_wal_interval_secs: 0,
            merge_wal_min_generations: 8,
            worker_endpoints: vec![],
            task_concurrency: 4,
            etcd_endpoints: vec![],
            etcd_prefix: "/test".to_string(),
            etcd_username: None,
            etcd_password: None,
            etcd_ca_cert: None,
            etcd_client_cert: None,
            etcd_client_key: None,
            etcd_lease_ttl_secs: 30,
            task_history_limit: 1_000,
            task_history_ttl_secs: 86_400,
            ui_dir: None,
        }
    }

    #[tokio::test]
    async fn connect_requires_etcd_endpoints() {
        let dir = TempDir::new().unwrap();
        let error = match TaskStore::open(&config(&dir)).await {
            Ok(_) => panic!("etcd backend must require endpoints"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("ETCD_ENDPOINTS is required"));
    }

    fn terminal_task(id: &str, state: TaskState, finished_at: i64) -> TaskRecord {
        TaskRecord {
            id: id.to_string(),
            kind: TaskKind::Compact,
            target: "exp".to_string(),
            state,
            error: None,
            detail: None,
            enqueued_at: finished_at,
            started_at: Some(finished_at),
            finished_at: Some(finished_at),
            depends_on: Vec::new(),
        }
    }

    #[test]
    fn prune_keeps_newest_over_count_cap() {
        // 5 terminal tasks, cap 2, TTL disabled → oldest 3 pruned.
        let tasks = (0..5)
            .map(|i| terminal_task(&format!("t{i}"), TaskState::Done, 1_000 + i * 10))
            .collect::<Vec<_>>();
        let mut pruned = prunable_terminal_ids(tasks, 2, None);
        pruned.sort();
        assert_eq!(pruned, vec!["t0", "t1", "t2"]);
    }

    #[test]
    fn prune_expires_by_ttl_cutoff() {
        // High count cap so only the TTL policy fires.
        let tasks = vec![
            terminal_task("old-1", TaskState::Done, 100),
            terminal_task("old-2", TaskState::Failed, 200),
            terminal_task("fresh", TaskState::Done, 5_000),
        ];
        let mut pruned = prunable_terminal_ids(tasks, 1_000, Some(1_000));
        pruned.sort();
        assert_eq!(pruned, vec!["old-1", "old-2"]);
    }

    #[test]
    fn prune_never_touches_active_or_depended_on() {
        // A queued task depends on a terminal one that is otherwise TTL-expired;
        // that dependency must be protected. Queued/Running are never terminal
        // candidates in the first place.
        let mut dep = terminal_task("dep", TaskState::Done, 100);
        dep.id = "dep".to_string();
        let mut queued = terminal_task("live", TaskState::Queued, 100);
        queued.depends_on = vec!["dep".to_string()];
        let expired = terminal_task("expired", TaskState::Done, 100);
        let tasks = vec![dep, queued, expired];

        let pruned = prunable_terminal_ids(tasks, 0, Some(1_000));
        // `dep` protected by the live task; `live` is Queued (not terminal);
        // only the unreferenced expired terminal task is pruned.
        assert_eq!(pruned, vec!["expired"]);
    }

    #[test]
    fn prune_ttl_disabled_uses_count_only() {
        let tasks = vec![
            terminal_task("a", TaskState::Done, 1),
            terminal_task("b", TaskState::Done, 2),
        ];
        // TTL None + generous cap → nothing pruned even though timestamps are old.
        assert!(prunable_terminal_ids(tasks, 10, None).is_empty());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn etcd_coordinates_dedupe_claims_and_target_locks() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS")
            .expect("ETCD_TEST_ENDPOINTS must point to a test etcd");
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd_endpoints = endpoint.split(',').map(str::to_string).collect();
        cfg.etcd_prefix = format!("/lance-context/test/{}", generate_id());
        cfg.etcd_lease_ttl_secs = 5;

        let first = TaskStore::open(&cfg).await.unwrap();
        let second = TaskStore::open(&cfg).await.unwrap();
        let (a, b) = tokio::join!(
            first.enqueue(TaskKind::Compact, "experiment", Vec::new()),
            second.enqueue(TaskKind::Compact, "experiment", Vec::new())
        );
        let a = a.unwrap();
        let b = b.unwrap();
        assert_eq!(a.id, b.id, "concurrent enqueue must dedupe");

        let (claim_a, claim_b) = tokio::join!(first.claim_next(), second.claim_next());
        let mut claims = [claim_a.unwrap(), claim_b.unwrap()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_eq!(claims.len(), 1, "only one master may claim a task");
        let compact_claim = claims.pop().unwrap();

        first
            .enqueue(TaskKind::IndexId, "experiment", Vec::new())
            .await
            .unwrap();
        assert!(
            second.claim_next().await.unwrap().is_none(),
            "Compact and IndexId must share the experiment write lock"
        );

        first
            .finish(compact_claim, Ok("compacted".to_string()))
            .await
            .unwrap();
        let index_claim = second.claim_next().await.unwrap().unwrap();
        assert_eq!(index_claim.task.kind, TaskKind::IndexId);
        second
            .finish(index_claim, Ok("indexed".to_string()))
            .await
            .unwrap();

        let prerequisite = first
            .enqueue(TaskKind::MergeWal, "dependency", Vec::new())
            .await
            .unwrap();
        let dependent = second
            .enqueue(
                TaskKind::Compact,
                "dependency",
                vec![prerequisite.id.clone()],
            )
            .await
            .unwrap();
        let prerequisite_claim = first.claim_next().await.unwrap().unwrap();
        assert_eq!(prerequisite_claim.task.id, prerequisite.id);
        assert!(
            second.claim_next().await.unwrap().is_none(),
            "another master must not claim a task with a running dependency"
        );
        first
            .finish(prerequisite_claim, Ok("merged".to_string()))
            .await
            .unwrap();
        let dependent_claim = second.claim_next().await.unwrap().unwrap();
        assert_eq!(dependent_claim.task.id, dependent.id);
        second
            .finish(dependent_claim, Ok("compacted".to_string()))
            .await
            .unwrap();

        let orphan = first
            .enqueue(TaskKind::MergeWal, "orphan", Vec::new())
            .await
            .unwrap();
        let abandoned = first.claim_next().await.unwrap().unwrap();
        assert_eq!(abandoned.task.id, orphan.id);
        drop(abandoned);
        tokio::time::sleep(Duration::from_secs(6)).await;
        let recovered = second.claim_next().await.unwrap().unwrap();
        assert_eq!(recovered.task.id, orphan.id);
        second
            .finish(recovered, Ok("recovered".to_string()))
            .await
            .unwrap();

        let mut client = Client::connect(cfg.etcd_endpoints, None).await.unwrap();
        client
            .delete(
                cfg.etcd_prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }
}
