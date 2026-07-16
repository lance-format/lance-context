//! Durable scheduler task storage.
//!
//! RocksDB is the single-master backend. etcd adds compare-and-swap enqueue,
//! lease-backed task claims, and distributed target locks so several stateless
//! masters can safely drain one shared queue.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use etcd_client::{
    Certificate, Client, Compare, CompareOp, ConnectOptions, GetOptions, Identity, PutOptions,
    TlsOptions, Txn, TxnOp,
};
use lance_context_api::{TaskKind, TaskRecord, TaskState};
use lance_context_core::generate_id;
use rocksdb::{Direction, IteratorMode, Options, WriteBatch, WriteOptions, DB};
use tokio::sync::{oneshot, Mutex};
use tokio::task::JoinHandle;

use crate::config::{MasterConfig, TaskStoreBackend};

const TASK_KEY_PREFIX: &[u8] = b"task/";
const TASK_POLL_BATCH: usize = 256;

#[derive(Clone)]
pub struct TaskStore {
    inner: Arc<TaskStoreInner>,
    history_limit: usize,
}

enum TaskStoreInner {
    Rocks(Box<RocksTaskStore>),
    Etcd(Box<EtcdTaskStore>),
}

struct RocksTaskStore {
    db: DB,
    operation_lock: Mutex<()>,
    claimed_tasks: Mutex<HashSet<String>>,
    claimed_targets: Mutex<HashSet<String>>,
}

struct EtcdTaskStore {
    client: Client,
    prefix: String,
    lease_ttl: i64,
}

/// Ownership of one running task. Dropping an etcd claim stops lease renewal;
/// etcd then removes its claim and target-lock keys, allowing recovery.
pub struct TaskClaim {
    pub task: TaskRecord,
    backend: ClaimBackend,
}

enum ClaimBackend {
    Rocks {
        target_locked: bool,
    },
    Etcd {
        token: String,
        lease_id: i64,
        claim_key: String,
        target_key: Option<String>,
        keepalive: LeaseKeepalive,
    },
}

/// Short-lived distributed guard used to serialize stats-table writers.
pub struct CoordinationGuard {
    backend: GuardBackend,
}

enum GuardBackend {
    Rocks,
    Etcd {
        token: String,
        lease_id: i64,
        key: String,
        keepalive: LeaseKeepalive,
    },
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
        let inner = match config.task_store_backend {
            TaskStoreBackend::Rocksdb => {
                TaskStoreInner::Rocks(Box::new(RocksTaskStore::open(&config.task_db_path)?))
            }
            TaskStoreBackend::Etcd => {
                TaskStoreInner::Etcd(Box::new(EtcdTaskStore::connect(config).await?))
            }
        };
        let store = Self {
            inner: Arc::new(inner),
            history_limit: config.task_history_limit.max(1),
        };
        store.recover_orphaned().await?;
        store.prune_terminal_history().await?;
        Ok(store)
    }

    #[must_use]
    pub fn is_distributed(&self) -> bool {
        matches!(self.inner.as_ref(), TaskStoreInner::Etcd(_))
    }

    /// Atomically enqueue a task. Standalone Compact and IndexId tasks use an
    /// etcd/local dedupe key while queued or running. Tasks with dependencies
    /// are always distinct because they belong to a specific ordered chain.
    pub async fn enqueue(
        &self,
        kind: TaskKind,
        target: &str,
        depends_on: Vec<String>,
    ) -> lance::Result<TaskRecord> {
        match self.inner.as_ref() {
            TaskStoreInner::Rocks(store) => store.enqueue(kind, target, depends_on).await,
            TaskStoreInner::Etcd(store) => store.enqueue(kind, target, depends_on).await,
        }
    }

    pub async fn list(&self) -> lance::Result<Vec<TaskRecord>> {
        match self.inner.as_ref() {
            TaskStoreInner::Rocks(store) => store.list(),
            TaskStoreInner::Etcd(store) => store.list().await,
        }
    }

    pub async fn get(&self, id: &str) -> lance::Result<Option<TaskRecord>> {
        match self.inner.as_ref() {
            TaskStoreInner::Rocks(store) => store.get(id),
            TaskStoreInner::Etcd(store) => store.get(id).await,
        }
    }

    pub async fn queue_depth(&self) -> lance::Result<usize> {
        match self.inner.as_ref() {
            TaskStoreInner::Rocks(store) => Ok(store
                .list()?
                .into_iter()
                .filter(|task| task.state == TaskState::Queued)
                .count()),
            TaskStoreInner::Etcd(store) => store.queue_depth().await,
        }
    }

    /// Claim the oldest runnable task. In etcd mode the queued->running update,
    /// claim lease, and per-experiment write lock are one transaction.
    pub async fn claim_next(&self) -> lance::Result<Option<TaskClaim>> {
        let (claim, dependency_failed) = match self.inner.as_ref() {
            TaskStoreInner::Rocks(store) => store.claim_next().await,
            TaskStoreInner::Etcd(store) => {
                store.recover_orphaned().await?;
                store.claim_next().await
            }
        }?;
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
        match self.inner.as_ref() {
            TaskStoreInner::Rocks(store) => store.finish(claim, outcome).await?,
            TaskStoreInner::Etcd(store) => store.finish(claim, outcome).await?,
        }
        self.prune_terminal_history().await.map(|_| ())
    }

    /// Try to acquire a named coordination lock without waiting. This is used
    /// around the shared Lance stats table, whose mutations must have one writer
    /// across all master replicas.
    pub async fn try_coordination_lock(
        &self,
        name: &str,
    ) -> lance::Result<Option<CoordinationGuard>> {
        match self.inner.as_ref() {
            TaskStoreInner::Rocks(_) => Ok(Some(CoordinationGuard {
                backend: GuardBackend::Rocks,
            })),
            TaskStoreInner::Etcd(store) => store.try_coordination_lock(name).await,
        }
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
        match (self.inner.as_ref(), guard.backend) {
            (TaskStoreInner::Rocks(_), GuardBackend::Rocks) => Ok(()),
            (
                TaskStoreInner::Etcd(store),
                GuardBackend::Etcd {
                    token,
                    lease_id,
                    key,
                    keepalive,
                },
            ) => {
                drop(keepalive);
                store.delete_owned_key(&key, &token).await?;
                store.revoke_lease(lease_id).await
            }
            _ => Err(lance::Error::io(
                "coordination guard belongs to a different task backend",
            )),
        }
    }

    async fn recover_orphaned(&self) -> lance::Result<usize> {
        match self.inner.as_ref() {
            TaskStoreInner::Rocks(store) => store.recover_running().await,
            TaskStoreInner::Etcd(store) => store.recover_orphaned().await,
        }
    }

    async fn prune_terminal_history(&self) -> lance::Result<usize> {
        let tasks = self.list().await?;
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
        terminal
            .sort_by_key(|task| std::cmp::Reverse(task.finished_at.unwrap_or(task.enqueued_at)));
        let ids = terminal
            .into_iter()
            .skip(self.history_limit)
            .map(|task| task.id)
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Ok(0);
        }
        match self.inner.as_ref() {
            TaskStoreInner::Rocks(store) => store.delete_many(ids.iter().map(String::as_str))?,
            TaskStoreInner::Etcd(store) => store.delete_many(&ids).await?,
        }
        Ok(ids.len())
    }
}

impl RocksTaskStore {
    fn open(path: impl AsRef<Path>) -> lance::Result<Self> {
        let path = path.as_ref();
        let text = path.to_string_lossy();
        if text.contains("://") {
            return Err(lance::Error::io(format!(
                "task DB path must be local, got '{text}'"
            )));
        }
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|err| {
                lance::Error::io(format!(
                    "failed to create task DB parent '{}': {err}",
                    parent.display()
                ))
            })?;
        }
        let mut options = Options::default();
        options.create_if_missing(true);
        options.set_max_open_files(64);
        options.set_keep_log_file_num(4);
        let db = DB::open(&options, path).map_err(|err| {
            lance::Error::io(format!(
                "failed to open task DB '{}': {err}",
                path.display()
            ))
        })?;
        Ok(Self {
            db,
            operation_lock: Mutex::new(()),
            claimed_tasks: Mutex::new(HashSet::new()),
            claimed_targets: Mutex::new(HashSet::new()),
        })
    }

    async fn enqueue(
        &self,
        kind: TaskKind,
        target: &str,
        depends_on: Vec<String>,
    ) -> lance::Result<TaskRecord> {
        let _operation = self.operation_lock.lock().await;
        if should_dedupe(kind, &depends_on) {
            if let Some(existing) = self.list()?.into_iter().find(|task| {
                task.kind == kind
                    && task.target == target
                    && task.depends_on.is_empty()
                    && matches!(task.state, TaskState::Queued | TaskState::Running)
            }) {
                return Ok(existing);
            }
        }
        let task = new_task(kind, target, depends_on);
        self.put(&task)?;
        Ok(task)
    }

    fn put(&self, task: &TaskRecord) -> lance::Result<()> {
        let value = encode_task(task)?;
        self.db
            .put_opt(task_key(&task.id), value, &sync_write_options())
            .map_err(|err| lance::Error::io(format!("failed to persist task '{}': {err}", task.id)))
    }

    fn get(&self, id: &str) -> lance::Result<Option<TaskRecord>> {
        self.db
            .get(task_key(id))
            .map_err(|err| lance::Error::io(format!("failed to read task '{id}': {err}")))?
            .map(|value| decode_task(&value, id))
            .transpose()
    }

    fn list(&self) -> lance::Result<Vec<TaskRecord>> {
        let mut tasks = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(TASK_KEY_PREFIX, Direction::Forward))
        {
            let (key, value) =
                item.map_err(|err| lance::Error::io(format!("failed to iterate task DB: {err}")))?;
            if !key.starts_with(TASK_KEY_PREFIX) {
                break;
            }
            tasks.push(decode_task(&value, &String::from_utf8_lossy(&key))?);
        }
        Ok(tasks)
    }

    async fn claim_next(&self) -> lance::Result<(Option<TaskClaim>, bool)> {
        let _operation = self.operation_lock.lock().await;
        let claimed_tasks = self.claimed_tasks.lock().await;
        let mut claimed_targets = self.claimed_targets.lock().await;
        let tasks = self.list()?;
        let mut queued = tasks
            .iter()
            .filter(|task| task.state == TaskState::Queued && !claimed_tasks.contains(&task.id))
            .cloned()
            .collect::<Vec<_>>();
        queued.sort_by_key(|task| task.enqueued_at);
        let mut dependency_failed = false;
        let mut runnable = None;
        for mut task in queued {
            match dependency_status(&task, |id| {
                tasks
                    .iter()
                    .find(|candidate| candidate.id == id)
                    .map(|task| task.state)
            }) {
                DependencyStatus::Ready => {}
                DependencyStatus::Waiting => continue,
                DependencyStatus::Failed(dependency) => {
                    fail_dependency(&mut task, &dependency);
                    self.put(&task)?;
                    dependency_failed = true;
                    continue;
                }
            }
            if requires_target_lock(task.kind) && claimed_targets.contains(&task.target) {
                continue;
            }
            runnable = Some(task);
            break;
        }
        let Some(mut task) = runnable else {
            return Ok((None, dependency_failed));
        };
        let target_locked = requires_target_lock(task.kind);
        task.state = TaskState::Running;
        task.started_at = Some(now_ms());
        self.put(&task)?;
        if target_locked {
            claimed_targets.insert(task.target.clone());
        }
        drop(claimed_targets);
        drop(claimed_tasks);
        self.claimed_tasks.lock().await.insert(task.id.clone());
        Ok((
            Some(TaskClaim {
                task,
                backend: ClaimBackend::Rocks { target_locked },
            }),
            dependency_failed,
        ))
    }

    async fn finish(&self, claim: TaskClaim, outcome: Result<String, String>) -> lance::Result<()> {
        let ClaimBackend::Rocks { target_locked } = claim.backend else {
            return Err(lance::Error::io("task claim belongs to etcd"));
        };
        let _operation = self.operation_lock.lock().await;
        let mut task = claim.task;
        apply_outcome(&mut task, outcome);
        let result = self.put(&task);
        self.claimed_tasks.lock().await.remove(&task.id);
        if target_locked {
            self.claimed_targets.lock().await.remove(&task.target);
        }
        result
    }

    async fn recover_running(&self) -> lance::Result<usize> {
        let _operation = self.operation_lock.lock().await;
        let mut recovered = 0;
        for mut task in self.list()? {
            if task.state == TaskState::Running {
                requeue(&mut task);
                self.put(&task)?;
                recovered += 1;
            }
        }
        Ok(recovered)
    }

    fn delete_many<'a>(&self, ids: impl IntoIterator<Item = &'a str>) -> lance::Result<()> {
        let mut batch = WriteBatch::default();
        let mut count = 0;
        for id in ids {
            batch.delete(task_key(id));
            count += 1;
        }
        if count == 0 {
            return Ok(());
        }
        self.db
            .write_opt(batch, &sync_write_options())
            .map_err(|err| lance::Error::io(format!("failed to prune task history: {err}")))
    }
}

impl EtcdTaskStore {
    async fn connect(config: &MasterConfig) -> lance::Result<Self> {
        if config.etcd_endpoints.is_empty() {
            return Err(lance::Error::io(
                "ETCD_ENDPOINTS is required when TASK_STORE_BACKEND=etcd",
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
                            backend: ClaimBackend::Etcd {
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
        let ClaimBackend::Etcd {
            token,
            lease_id,
            claim_key,
            target_key,
            keepalive,
        } = claim.backend
        else {
            return Err(lance::Error::io("task claim belongs to RocksDB"));
        };
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
            backend: GuardBackend::Etcd {
                token,
                lease_id,
                key,
                keepalive,
            },
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

fn dependency_status(
    task: &TaskRecord,
    mut lookup: impl FnMut(&str) -> Option<TaskState>,
) -> DependencyStatus {
    let mut waiting = false;
    for dependency in &task.depends_on {
        match lookup(dependency) {
            Some(TaskState::Done) => {}
            Some(TaskState::Failed) | None => {
                return DependencyStatus::Failed(dependency.clone());
            }
            Some(TaskState::Queued | TaskState::Running) => waiting = true,
        }
    }
    if waiting {
        DependencyStatus::Waiting
    } else {
        DependencyStatus::Ready
    }
}

fn fail_dependency(task: &mut TaskRecord, dependency: &str) {
    apply_outcome(
        task,
        Err(format!("dependency {dependency} did not complete")),
    );
}

fn should_dedupe(kind: TaskKind, depends_on: &[String]) -> bool {
    depends_on.is_empty() && matches!(kind, TaskKind::Compact | TaskKind::IndexId)
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

fn task_key(id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(TASK_KEY_PREFIX.len() + id.len());
    key.extend_from_slice(TASK_KEY_PREFIX);
    key.extend_from_slice(id.as_bytes());
    key
}

fn sync_write_options() -> WriteOptions {
    let mut options = WriteOptions::default();
    options.set_sync(true);
    options
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
            compaction_interval_secs: 0,
            min_fragments: 16,
            target_rows_per_fragment: 1_048_576,
            worker_endpoints: vec![],
            task_concurrency: 4,
            task_store_backend: TaskStoreBackend::Rocksdb,
            task_db_path: dir.path().join("tasks").to_string_lossy().to_string(),
            etcd_endpoints: vec![],
            etcd_prefix: "/test".to_string(),
            etcd_username: None,
            etcd_password: None,
            etcd_ca_cert: None,
            etcd_client_cert: None,
            etcd_client_key: None,
            etcd_lease_ttl_secs: 30,
            task_history_limit: 1_000,
            ui_dir: None,
        }
    }

    #[tokio::test]
    async fn rocksdb_survives_reopen_and_dedupes() {
        let dir = TempDir::new().unwrap();
        let cfg = config(&dir);
        let first = TaskStore::open(&cfg).await.unwrap();
        let a = first
            .enqueue(TaskKind::Compact, "experiment", Vec::new())
            .await
            .unwrap();
        let b = first
            .enqueue(TaskKind::Compact, "experiment", Vec::new())
            .await
            .unwrap();
        assert_eq!(a.id, b.id);
        drop(first);

        let reopened = TaskStore::open(&cfg).await.unwrap();
        assert_eq!(reopened.get(&a.id).await.unwrap().unwrap(), a);
    }

    #[tokio::test]
    async fn rocksdb_claims_and_finishes() {
        let dir = TempDir::new().unwrap();
        let store = TaskStore::open(&config(&dir)).await.unwrap();
        let task = store
            .enqueue(TaskKind::IndexId, "experiment", Vec::new())
            .await
            .unwrap();
        let claim = store.claim_next().await.unwrap().unwrap();
        assert_eq!(claim.task.id, task.id);
        assert_eq!(claim.task.state, TaskState::Running);
        store
            .finish(claim, Ok("indexed".to_string()))
            .await
            .unwrap();
        let finished = store.get(&task.id).await.unwrap().unwrap();
        assert_eq!(finished.state, TaskState::Done);
        assert_eq!(finished.detail.as_deref(), Some("indexed"));
    }

    #[tokio::test]
    async fn rocksdb_waits_for_dependencies_and_propagates_failure() {
        let dir = TempDir::new().unwrap();
        let store = TaskStore::open(&config(&dir)).await.unwrap();

        let prerequisite = store
            .enqueue(TaskKind::MergeWal, "experiment", Vec::new())
            .await
            .unwrap();
        let dependent = store
            .enqueue(
                TaskKind::Compact,
                "experiment",
                vec![prerequisite.id.clone()],
            )
            .await
            .unwrap();
        let claim = store.claim_next().await.unwrap().unwrap();
        assert_eq!(claim.task.id, prerequisite.id);
        assert!(
            store.claim_next().await.unwrap().is_none(),
            "dependent task must wait while its prerequisite is running"
        );
        store.finish(claim, Ok("merged".to_string())).await.unwrap();
        let dependent_claim = store.claim_next().await.unwrap().unwrap();
        assert_eq!(dependent_claim.task.id, dependent.id);
        store
            .finish(dependent_claim, Ok("compacted".to_string()))
            .await
            .unwrap();

        let failed_prerequisite = store
            .enqueue(TaskKind::MergeWal, "failed-chain", Vec::new())
            .await
            .unwrap();
        let skipped = store
            .enqueue(
                TaskKind::IndexId,
                "failed-chain",
                vec![failed_prerequisite.id.clone()],
            )
            .await
            .unwrap();
        let claim = store.claim_next().await.unwrap().unwrap();
        assert_eq!(claim.task.id, failed_prerequisite.id);
        store
            .finish(claim, Err("merge failed".to_string()))
            .await
            .unwrap();
        assert!(store.claim_next().await.unwrap().is_none());
        let skipped = store.get(&skipped.id).await.unwrap().unwrap();
        assert_eq!(skipped.state, TaskState::Failed);
        assert!(skipped
            .error
            .as_deref()
            .is_some_and(|error| error.contains("dependency")));
    }

    #[tokio::test]
    async fn dependent_tasks_do_not_participate_in_standalone_dedupe() {
        let dir = TempDir::new().unwrap();
        let store = TaskStore::open(&config(&dir)).await.unwrap();
        let prerequisite = store
            .enqueue(TaskKind::MergeWal, "experiment", Vec::new())
            .await
            .unwrap();
        let dependent = store
            .enqueue(TaskKind::Compact, "experiment", vec![prerequisite.id])
            .await
            .unwrap();
        let standalone = store
            .enqueue(TaskKind::Compact, "experiment", Vec::new())
            .await
            .unwrap();
        assert_ne!(dependent.id, standalone.id);
    }

    #[tokio::test]
    async fn rejects_object_store_path_for_rocksdb() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.task_db_path = "s3://bucket/tasks".to_string();
        let error = match TaskStore::open(&cfg).await {
            Ok(_) => panic!("object-store URI should be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("must be local"));
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn etcd_coordinates_dedupe_claims_and_target_locks() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS")
            .expect("ETCD_TEST_ENDPOINTS must point to a test etcd");
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.task_store_backend = TaskStoreBackend::Etcd;
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
