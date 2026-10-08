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
use etcd_client::{Client, Compare, CompareOp, GetOptions, PutOptions, Txn, TxnOp, TxnOpResponse};
use lance_context_api::{RepairRecord, TaskCooldown, TaskKind, TaskRecord, TaskState};
use lance_context_core::generate_id;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::MasterConfig;

const TASK_POLL_BATCH: usize = 256;

/// A set of [`TaskKind`]s a claim is allowed to return.
///
/// Small enough to copy; used to let each scheduler execution pool claim only
/// the kinds it can actually run, so a saturated pool's backlog never hides a
/// runnable task belonging to an idle pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskKinds {
    compact: bool,
    merge_wal: bool,
    index_id: bool,
}

impl TaskKinds {
    /// Every kind -- the unfiltered claim used outside the scheduler.
    pub const ANY: Self = Self {
        compact: true,
        merge_wal: true,
        index_id: true,
    };
    /// Only `MergeWal`, the kind with its own dedicated budget.
    pub const MERGE_WAL: Self = Self {
        compact: false,
        merge_wal: true,
        index_id: false,
    };
    /// Everything the general pool runs: `Compact` and `IndexId`.
    pub const GENERAL: Self = Self {
        compact: true,
        merge_wal: false,
        index_id: true,
    };

    pub const COMPACT: Self = Self {
        compact: true,
        merge_wal: false,
        index_id: false,
    };

    pub fn without_compact(mut self) -> Self {
        self.compact = false;
        self
    }

    #[must_use]
    pub fn contains(self, kind: TaskKind) -> bool {
        match kind {
            TaskKind::Compact => self.compact,
            TaskKind::MergeWal => self.merge_wal,
            // Repair is a base-table write like IndexId and shares its pool.
            TaskKind::IndexId | TaskKind::Repair => self.index_id,
        }
    }
}

#[derive(Clone)]
pub struct TaskStore {
    inner: Arc<EtcdTaskStore>,
    history_limit: usize,
    history_ttl_secs: u64,
    cooldown: CooldownPolicy,
    global_maintenance: bool,
}

/// When to stop re-enqueueing a target that keeps failing, and for how long.
///
/// A store whose base-table manifest names a fragment that no longer exists
/// fails every merge and every compaction at the same point, forever. Without
/// memory of that, each sweep re-enqueued it, each attempt fanned out to
/// every worker before failing, and five such stores consumed roughly half
/// the fleet's task slots while healthy stores queued behind them.
#[derive(Debug, Clone, Copy)]
pub struct CooldownPolicy {
    /// Consecutive failures before a target is cooled down; `0` disables.
    pub after_failures: u32,
    /// Cooldown after the threshold is first reached; doubles per further failure.
    pub base: Duration,
    /// Longest cooldown.
    pub max: Duration,
}

impl CooldownPolicy {
    fn duration_for(&self, failures: u32) -> Duration {
        let over = failures.saturating_sub(self.after_failures);
        let secs = self
            .base
            .as_secs()
            .saturating_mul(1u64 << over.min(20))
            .min(self.max.as_secs());
        Duration::from_secs(secs)
    }
}

struct EtcdTaskStore {
    client: Client,
    prefix: String,
    lease_ttl: i64,
    prepare_targets: Vec<String>,
    index_prepare_targets: Vec<String>,
    maintenance_catchup_targets: Vec<String>,
    rollout: lance_context_merge::rollout::MergeRollout,
}

/// Ownership of one running task. Dropping the claim stops lease renewal; etcd
/// then removes its claim and target-lock keys, allowing recovery.
pub struct TaskClaim {
    pub task: TaskRecord,
    backend: ClaimBackend,
}

impl TaskClaim {
    pub(crate) fn preparing_maintenance(&self) -> bool {
        self.backend.preparation_key.is_some() && self.backend.target_key.is_none()
    }
}

struct ClaimBackend {
    token: String,
    lease_id: i64,
    claim_key: String,
    target_key: Option<String>,
    preparation_key: Option<String>,
    preparation_owner: Option<Vec<u8>>,
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
    pub(crate) async fn preparation_owned(&self, claim: &TaskClaim) -> lance::Result<bool> {
        let Some(key) = &claim.backend.preparation_key else {
            return Ok(false);
        };
        let values = self
            .inner
            .read_values(&[claim.backend.claim_key.clone(), key.clone()])
            .await?;
        Ok(values
            .iter()
            .all(|v| v.as_deref() == Some(claim.backend.token.as_bytes())))
    }

    /// Ask subsequent merge admissions to yield after files are ready. This
    /// marker shares the preparation lease: crash/finish removes the request.
    /// It grants no write ownership and never cancels an admitted merge.
    pub(crate) async fn request_maintenance_commit(
        &self,
        claim: &TaskClaim,
    ) -> lance::Result<bool> {
        let Some(preparation) = &claim.backend.preparation_key else {
            return Ok(false);
        };
        let b = &claim.backend;
        self.inner
            .client
            .clone()
            .txn(
                Txn::new()
                    .when([
                        Compare::value(b.claim_key.as_str(), CompareOp::Equal, b.token.as_bytes()),
                        Compare::value(preparation.as_str(), CompareOp::Equal, b.token.as_bytes()),
                    ])
                    .and_then([TxnOp::put(
                        self.inner.compaction_commit_key(&claim.task.target),
                        b.token.as_bytes(),
                        Some(PutOptions::new().with_lease(b.lease_id)),
                    )]),
            )
            .await
            .map(|r| r.succeeded())
            .map_err(etcd_error("request maintenance commit turn"))
    }

    /// Acquire the existing table writer protocol only after immutable files
    /// are ready. An ambiguous response can adopt only this claim's own token.
    pub(crate) async fn promote_maintenance(&self, claim: &mut TaskClaim) -> lance::Result<bool> {
        assert!(claim.preparing_maintenance());
        let b = &claim.backend;
        let target = &claim.task.target;
        let execution_key = lance_context_merge::execution_key(&self.inner.prefix, target);
        let target_key = self.inner.target_lock_key(target);
        let merge_key = execution_key.replace("/merge-executions/", "/merge-claims/");
        let active_key = crate::catchup::store::active_key(&self.inner.prefix, target);
        let values = self
            .inner
            .read_values(&[
                execution_key.clone(),
                target_key.clone(),
                merge_key.clone(),
                active_key.clone(),
            ])
            .await?;
        let old: Option<lance_context_merge::Execution> = values[0]
            .as_deref()
            .map(serde_json::from_slice)
            .transpose()
            .map_err(|e| lance::Error::io(e.to_string()))?;
        if values[2].is_some()
            || values[3] != b.preparation_owner
            || old.as_ref().is_some_and(|e| e.maintenance.is_none())
        {
            return Ok(false);
        }
        let expected_owner = old.as_ref().map(lance_context_merge::execution_owner);
        if values[1].as_deref() != expected_owner.as_deref().map(str::as_bytes) {
            return Ok(false);
        }
        let mut compares = vec![
            Compare::value(b.claim_key.as_str(), CompareOp::Equal, b.token.as_bytes()),
            Compare::value(
                b.preparation_key.as_ref().unwrap().as_str(),
                CompareOp::Equal,
                b.token.as_bytes(),
            ),
            Compare::version(merge_key.as_str(), CompareOp::Equal, 0),
            match &b.preparation_owner {
                Some(owner) => Compare::value(active_key, CompareOp::Equal, owner.clone()),
                None => Compare::version(active_key, CompareOp::Equal, 0),
            },
        ];
        for (key, value) in [
            (execution_key, &values[0]),
            (target_key.clone(), &values[1]),
        ] {
            compares.push(match value {
                Some(v) => Compare::value(key, CompareOp::Equal, v.clone()),
                None => Compare::version(key, CompareOp::Equal, 0),
            });
        }
        let opts = Some(PutOptions::new().with_lease(b.lease_id));
        let mut writes = vec![TxnOp::put(
            merge_key.clone(),
            b.token.as_bytes(),
            opts.clone(),
        )];
        if old.is_none() {
            writes.push(TxnOp::put(target_key.clone(), b.token.as_bytes(), opts));
        }
        let response = self
            .inner
            .client
            .clone()
            .txn(Txn::new().when(compares).and_then(writes))
            .await;
        let accepted = match response {
            Ok(r) => r.succeeded(),
            Err(error) => {
                let values = self
                    .inner
                    .read_values(&[b.claim_key.clone(), merge_key, target_key.clone()])
                    .await?;
                if values[0].as_deref() == Some(b.token.as_bytes())
                    && values[1].as_deref() == Some(b.token.as_bytes())
                    && values[2].as_deref()
                        == Some(expected_owner.as_deref().unwrap_or(&b.token).as_bytes())
                {
                    true
                } else {
                    return Err(etcd_error("promote maintenance")(error));
                }
            }
        };
        if accepted {
            claim.backend.target_key = Some(target_key);
        }
        Ok(accepted)
    }

    pub(crate) fn merge_coordinator(&self) -> lance_context_merge::Coordinator {
        lance_context_merge::Coordinator::new(self.inner.client.clone(), self.inner.prefix.clone())
    }

    pub(crate) fn merge_claim(&self, claim: &TaskClaim) -> lance_context_merge::ClaimProof {
        lance_context_merge::ClaimProof {
            key: self.inner.claim_key(&claim.task.id),
            token: claim.backend.token.clone(),
            lease_id: claim.backend.lease_id,
        }
    }

    /// A successful no-op is tied to the exact observed version and options.
    /// Expiry also bounds suppression when stats are stale or a URI is reused.
    pub(crate) async fn record_compact_noop(
        &self,
        target: &str,
        uri: &str,
        version: u64,
        options: &str,
    ) -> lance::Result<()> {
        let key = lance_context_merge::execution_key(&self.inner.prefix, target)
            .replace("/merge-executions/", "/compact-noops/");
        let lease = self
            .inner
            .client
            .clone()
            .lease_grant(900, None)
            .await
            .map_err(etcd_error("compact no-op lease"))?
            .id();
        let value = serde_json::json!({"uri": uri, "version": version, "options": options});
        self.inner
            .client
            .clone()
            .put(
                key,
                value.to_string(),
                Some(PutOptions::new().with_lease(lease)),
            )
            .await
            .map_err(etcd_error("compact no-op record"))?;
        Ok(())
    }

    pub(crate) async fn compact_is_unchanged(
        &self,
        target: &str,
        uri: &str,
        version: i64,
        options: &str,
    ) -> lance::Result<bool> {
        if version < 0 {
            return Ok(false);
        }
        let key = lance_context_merge::execution_key(&self.inner.prefix, target)
            .replace("/merge-executions/", "/compact-noops/");
        let response = self
            .inner
            .client
            .clone()
            .get(key, None)
            .await
            .map_err(etcd_error("compact no-op read"))?;
        let Some(kv) = response.kvs().first() else {
            return Ok(false);
        };
        let value: serde_json::Value = serde_json::from_slice(kv.value())
            .map_err(|e| lance::Error::io(format!("invalid compact no-op record: {e}")))?;
        Ok(
            value["uri"] == uri
                && value["version"] == version as u64
                && value["options"] == options,
        )
    }

    /// The underlying etcd client, for other etcd-backed components (the
    /// store registries) that should share one connection.
    pub fn etcd_client(&self) -> &Client {
        &self.inner.client
    }

    pub async fn open(config: &MasterConfig) -> lance::Result<Self> {
        let store = Self {
            inner: Arc::new(EtcdTaskStore::connect(config).await?),
            history_limit: config.task_history_limit.max(1),
            history_ttl_secs: config.task_history_ttl_secs,
            cooldown: CooldownPolicy {
                after_failures: config.task_cooldown_after_failures,
                base: Duration::from_secs(config.task_cooldown_base_secs),
                max: Duration::from_secs(config.task_cooldown_max_secs),
            },
            global_maintenance: config.catchup.target.is_none(),
        };
        // Short-lived exact-target executors must not sweep the fleet's running
        // tasks or terminal history on every invocation. Ordinary masters own
        // that maintenance; dedicated admission reconciles its own task below.
        if store.global_maintenance {
            store.recover_orphaned().await?;
            store.prune_terminal_history().await?;
        }
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
        self.claim_next_of_kinds(TaskKinds::ANY).await
    }

    /// Claim the oldest runnable task whose kind is in `kinds`, skipping over
    /// queued tasks of other kinds instead of stopping at them.
    ///
    /// The scheduler needs this because it runs several execution pools with
    /// separate budgets. Claiming is destructive -- it deletes the queue key,
    /// grants a lease, and takes the per-experiment target lock -- so claiming a
    /// task the caller has no free slot for does not merely waste a poll: it
    /// parks a claimed task on a semaphore while it holds its target lock. With
    /// one kind numerically dominant, an unfiltered claim returns that kind
    /// nearly every time, so a scarce kind can wait behind it indefinitely even
    /// though its own pool is idle. Filtering at claim time keeps the queue's
    /// FIFO order within each kind while letting an idle pool reach past a
    /// saturated one.
    pub async fn claim_next_of_kinds(&self, kinds: TaskKinds) -> lance::Result<Option<TaskClaim>> {
        self.inner.recover_orphaned().await?;
        let (claim, dependency_failed) = self.inner.claim_next(kinds, None, None).await?;
        if dependency_failed {
            if let Err(error) = self.prune_terminal_history().await {
                tracing::warn!(error = %error, "failed to prune task history");
            }
        }
        Ok(claim)
    }

    /// Reserved resident capacity filters before any dependency/ownership RPC.
    /// Keep canonical claims and all external-owner/fencing checks unchanged.
    pub(crate) async fn claim_resident_merge(
        &self,
        targets: &[String],
    ) -> lance::Result<Option<TaskClaim>> {
        self.inner.recover_orphaned().await?;
        let (claim, dependency_failed) = self
            .inner
            .claim_next(TaskKinds::MERGE_WAL, None, Some(targets))
            .await?;
        if dependency_failed {
            if let Err(error) = self.prune_terminal_history().await {
                tracing::warn!(error = %error, "failed to prune task history");
            }
        }
        Ok(claim)
    }

    /// Claim exactly this target, never consuming another table's queue entry.
    pub(crate) async fn claim_merge_target(
        &self,
        target: &str,
        job: &str,
    ) -> lance::Result<Option<TaskClaim>> {
        let Some(id) = self.get_active_id(TaskKind::MergeWal, target).await? else {
            return Ok(None);
        };
        if let Some(task) = self.inner.get(&id).await? {
            if task.state == TaskState::Running {
                self.inner.recover_tasks(&[task]).await?;
            }
        }
        Ok(self
            .inner
            .claim_next(TaskKinds::MERGE_WAL, Some((&id, job)), None)
            .await?
            .0)
    }

    /// Finish a claimed task and release its claim/target lock.
    pub async fn finish(
        &self,
        claim: TaskClaim,
        outcome: Result<String, String>,
    ) -> lance::Result<()> {
        let kind = claim.task.kind;
        let target = claim.task.target.clone();
        let failed = outcome.as_ref().err().cloned();
        self.inner.finish(claim, outcome).await?;
        // Cooldown bookkeeping is best-effort and never fails the completion:
        // the task's terminal state is already committed above.
        if kind != TaskKind::MergeWal && self.cooldown.after_failures > 0 {
            let result = match failed {
                Some(error) => {
                    self.inner
                        .record_failure(kind, &target, &error, self.cooldown)
                        .await
                }
                None => self.inner.clear_cooldown(kind, &target).await,
            };
            if let Err(error) = result {
                tracing::warn!(kind = ?kind, target = %target, %error, "cooldown bookkeeping failed");
            }
        }
        if self.global_maintenance {
            if let Err(error) = self.prune_terminal_history().await {
                tracing::warn!(kind = ?kind, target = %target, %error, "post-completion history pruning failed");
            }
        }
        Ok(())
    }

    /// Whether the sweeps should skip this target for now because it has
    /// failed repeatedly. Manual enqueues are not gated by this.
    /// Id of the queued or running task that `enqueue(kind, target)` would
    /// dedupe into, if any. Lets a sweep tell a new task from a no-op.
    pub async fn get_active_id(
        &self,
        kind: TaskKind,
        target: &str,
    ) -> lance::Result<Option<String>> {
        self.inner.get_active_id(kind, target).await
    }

    pub async fn is_cooling_down(&self, kind: TaskKind, target: &str) -> lance::Result<bool> {
        if kind == TaskKind::MergeWal || self.cooldown.after_failures == 0 {
            return Ok(false);
        }
        // Below the threshold the record only carries the failure count; it is
        // a cooldown only once `until_ms` is set.
        // The record outlives the cooldown window so the failure count keeps
        // climbing across windows; only `until_ms` says whether we are inside
        // one right now.
        Ok(self
            .inner
            .get_cooldown(kind, target)
            .await?
            .is_some_and(|c| c.until_ms.is_some_and(|until| until > now_ms())))
    }

    /// Every target currently in cooldown, for operators.
    pub async fn list_cooldowns(&self) -> lance::Result<Vec<TaskCooldown>> {
        self.inner.list_cooldowns().await
    }

    /// Persist what a repair dropped, keyed by target and time, so it can be
    /// answered for later. Kept for `history_ttl_secs` like task history.
    pub async fn record_repair(&self, record: &RepairRecord) -> lance::Result<()> {
        self.inner
            .record_repair(record, self.history_ttl_secs)
            .await
    }

    /// Every recorded repair, most recent first.
    pub async fn list_repairs(&self) -> lance::Result<Vec<RepairRecord>> {
        self.inner.list_repairs().await
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

    /// Simulate a master crash while `claim` was in flight.
    ///
    /// Dropping a `TaskClaim` only stops *renewal*; etcd keeps the claim and
    /// target-lock keys until the lease TTL elapses, which is exactly why a
    /// task remains `Running` for up to `etcd_lease_ttl_secs` after its owner
    /// dies. Revoking the lease collapses that TTL wait to zero, producing the
    /// same key state a real crash reaches once the lease expires, so recovery
    /// tests need not sleep out the TTL.
    #[cfg(test)]
    pub(crate) async fn abandon_claim_for_test(&self, claim: TaskClaim) -> lance::Result<()> {
        let lease_id = claim.backend.lease_id;
        drop(claim);
        self.inner.revoke_lease(lease_id).await
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
        if !config.etcd.is_configured() {
            return Err(lance::Error::io(
                "ETCD_ENDPOINTS is required to run the master",
            ));
        }
        if config.etcd_lease_ttl_secs < 5 {
            return Err(lance::Error::io("ETCD_LEASE_TTL_SECS must be at least 5"));
        }
        let client = config.etcd.connect().await?;
        Ok(Self {
            client,
            prefix: config.etcd.prefix().to_string(),
            lease_ttl: config.etcd_lease_ttl_secs,
            prepare_targets: config.maintenance.compaction_prepare_targets.clone(),
            index_prepare_targets: config.maintenance.index_prepare_targets.clone(),
            maintenance_catchup_targets: config.maintenance.maintenance_catchup_targets.clone(),
            rollout: config.merge_rollout.clone(),
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

    async fn claim_next(
        &self,
        kinds: TaskKinds,
        only_id: Option<(&str, &str)>,
        resident_targets: Option<&[String]>,
    ) -> lance::Result<(Option<TaskClaim>, bool)> {
        let prefix = only_id.map_or_else(|| self.queue_prefix(), |(id, _)| self.queue_key(id));
        let range_end = if only_id.is_some() {
            let mut end = prefix.as_bytes().to_vec();
            end.push(0);
            end
        } else {
            prefix_range_end(prefix.as_bytes())
        };
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
                // Skip kinds this caller cannot run *before* the dependency
                // probe: an unrunnable kind should cost nothing, and this is
                // what lets a scarce kind be found behind a dominant one.
                if !kinds.contains(task.kind) {
                    continue;
                }
                if resident_targets.is_some_and(|targets| {
                    task.kind != TaskKind::MergeWal
                        || task.target.starts_with("generic:")
                        || !self.rollout.owned(&task.target)
                        || self.rollout.draining(&task.target)
                        || !targets.iter().any(|t| t == "*" || t == &task.target)
                }) {
                    continue;
                }
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

                let preparation_targets: &[String] = match task.kind {
                    TaskKind::Compact => &self.prepare_targets,
                    TaskKind::IndexId => &self.index_prepare_targets,
                    _ => &[],
                };
                let preparing = matches!(task.kind, TaskKind::Compact | TaskKind::IndexId)
                    && !task.target.starts_with("generic:")
                    && !self.rollout.draining(&task.target)
                    && preparation_targets
                        .iter()
                        .any(|t| t == "*" || t == &task.target);
                let write_claim = requires_target_lock(task.kind) && !preparing;
                // Keep the existing wire keys so older masters also yield to
                // prepared index commits. Compact/IndexId share one preparation
                // slot per table; append merges need neither preparation slot.
                let preparation_key = preparing.then(|| {
                    format!(
                        "{}/compact-preparations/{}",
                        self.prefix,
                        encode_segment(&task.target)
                    )
                });
                let execution_key = lance_context_merge::execution_key(&self.prefix, &task.target);
                let snapshot = self
                    .read_values(&[
                        execution_key.clone(),
                        self.target_lock_key(&task.target),
                        crate::catchup::store::active_key(&self.prefix, &task.target),
                        execution_key.replace("/merge-executions/", "/merge-claims/"),
                        self.claim_key(&task.id),
                        preparation_key.clone().unwrap_or_else(|| {
                            format!(
                                "{}/compact-preparations/{}",
                                self.prefix,
                                encode_segment(&task.target)
                            )
                        }),
                        self.compaction_commit_key(&task.target),
                    ])
                    .await?;
                let merge_execution: Option<lance_context_merge::Execution> = snapshot[0]
                    .as_deref()
                    .map(serde_json::from_slice)
                    .transpose()
                    .map_err(|e| lance::Error::io(format!("invalid merge execution: {e}")))?;
                // Preparation itself stays concurrent. Once its output is
                // ready, yield the next merge turn so the short commit can win.
                // An unresolved execution still needs merge recovery; never
                // let a scheduling hint block that storage safety path.
                if task.kind == TaskKind::MergeWal
                    && merge_execution.is_none()
                    && snapshot[6].is_some()
                    && snapshot[6] == snapshot[5]
                {
                    metrics::counter!("master_merge_yield_to_compaction_total").increment(1);
                    continue;
                }
                // Avoid lease grant/revoke for a clearly blocked candidate.
                // The final transaction still checks every ownership predicate.
                let expected_owner = merge_execution
                    .as_ref()
                    .map(lance_context_merge::execution_owner);
                let shared_owner = snapshot[2].is_some()
                    && matches!(task.kind, TaskKind::Compact | TaskKind::IndexId)
                    && self.rollout.owned(&task.target)
                    && !self.rollout.draining(&task.target)
                    && !task.target.starts_with("generic:")
                    && self.maintenance_catchup_targets.contains(&task.target);
                let expected_job = only_id.map(|(_, job)| job.as_bytes()).or(if shared_owner {
                    snapshot[2].as_deref()
                } else {
                    None
                });
                if snapshot[4].is_some()
                    || (preparing
                        && (snapshot[5].is_some() || (snapshot[2].is_some() && !shared_owner)))
                    || (write_claim
                        && (snapshot[2].as_deref() != expected_job
                            || snapshot[3].is_some()
                            || snapshot[1].as_deref()
                                != expected_owner.as_deref().map(str::as_bytes)))
                {
                    metrics::counter!("master_task_admission_blocked_total").increment(1);
                    continue;
                }
                // Only MergeWal reconciles a worker execution. A local fenced
                // mutation may be recovered by any table writer after the
                // exclusive reconciler lease has expired.
                if !preparing
                    && task.kind != TaskKind::MergeWal
                    && merge_execution
                        .as_ref()
                        .is_some_and(|e| e.maintenance.is_none())
                {
                    continue;
                }
                let token = generate_id();
                let lease_id = self.grant_lease().await?;
                // Establish renewal before publishing ownership, so a slow
                // keepalive handshake cannot strand an already accepted claim.
                let keepalive = match self.start_keepalive(lease_id).await {
                    Ok(keepalive) => keepalive,
                    Err(error) => {
                        let _ = self.revoke_lease(lease_id).await;
                        return Err(error);
                    }
                };
                let claim_key = self.claim_key(&task.id);
                let target_key = write_claim.then(|| self.target_lock_key(&task.target));
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
                if let Some(key) = &preparation_key {
                    compares.push(Compare::version(key.as_str(), CompareOp::Equal, 0));
                    let active_key = crate::catchup::store::active_key(&self.prefix, &task.target);
                    compares.push(match &snapshot[2] {
                        Some(owner) => Compare::value(active_key, CompareOp::Equal, owner.clone()),
                        None => Compare::version(active_key, CompareOp::Equal, 0),
                    });
                }
                if task.kind == TaskKind::MergeWal && merge_execution.is_none() {
                    // A commit request arriving between the read and claim
                    // must win over this stale merge admission attempt.
                    compares.push(match &snapshot[6] {
                        Some(value) => Compare::value(
                            self.compaction_commit_key(&task.target),
                            CompareOp::Equal,
                            value.clone(),
                        ),
                        None => Compare::version(
                            self.compaction_commit_key(&task.target),
                            CompareOp::Equal,
                            0,
                        ),
                    });
                }
                let catchup_key = crate::catchup::store::active_key(&self.prefix, &task.target);
                if write_claim {
                    compares.push(match expected_job {
                        Some(job) => Compare::value(catchup_key, CompareOp::Equal, job),
                        None => Compare::version(catchup_key, CompareOp::Equal, 0),
                    });
                }
                if let Some(key) = &target_key {
                    if let Some(execution) = &merge_execution {
                        compares.push(Compare::value(
                            key.as_str(),
                            CompareOp::Equal,
                            lance_context_merge::execution_owner(execution),
                        ));
                    } else {
                        compares.push(Compare::version(key.as_str(), CompareOp::Equal, 0));
                    }
                }
                // Remote execution ownership persists, but its reconciler is
                // still exclusive and leased. Dependency-chain merge tasks
                // must not cancel another live scheduler's execution.
                let merge_claim_key =
                    lance_context_merge::execution_key(&self.prefix, &task.target)
                        .replace("/merge-executions/", "/merge-claims/");
                if write_claim {
                    compares.push(Compare::version(
                        merge_claim_key.as_str(),
                        CompareOp::Equal,
                        0,
                    ));
                }
                let lease_options = Some(PutOptions::new().with_lease(lease_id));
                let mut operations = vec![
                    TxnOp::put(self.task_key(&task.id), running_value.clone(), None),
                    TxnOp::delete(queue_key, None),
                    TxnOp::put(running_key, running_value, None),
                    TxnOp::put(claim_key.as_str(), token.as_bytes(), lease_options.clone()),
                ];
                if let Some(key) = &preparation_key {
                    operations.push(TxnOp::put(
                        key.as_str(),
                        token.as_bytes(),
                        lease_options.clone(),
                    ));
                }
                if write_claim {
                    operations.push(TxnOp::put(
                        merge_claim_key,
                        token.as_bytes(),
                        lease_options.clone(),
                    ));
                }
                if merge_execution.is_none() {
                    if let Some(key) = &target_key {
                        operations.push(TxnOp::put(
                            key.as_str(),
                            token.as_bytes(),
                            lease_options.clone(),
                        ));
                    }
                }
                let mut client = self.client.clone();
                let response = client
                    .txn(Txn::new().when(compares).and_then(operations))
                    .await;
                let claimed = match self
                    .resolve_claim_response(response, &claim_key, &token)
                    .await
                {
                    Ok(claimed) => claimed,
                    Err(error) => {
                        drop(keepalive);
                        let _ = self.revoke_lease(lease_id).await;
                        return Err(error);
                    }
                };
                if claimed {
                    return Ok((
                        Some(TaskClaim {
                            task,
                            backend: ClaimBackend {
                                token,
                                lease_id,
                                claim_key,
                                target_key,
                                preparation_key,
                                preparation_owner: snapshot[2].clone(),
                                keepalive,
                            },
                        }),
                        dependency_failed,
                    ));
                }
                drop(keepalive);
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
            preparation_key,
            keepalive,
            ..
        } = claim.backend;
        let mut task = claim.task;
        let owns_table = target_key.is_some();
        let unresolved = if owns_table {
            lance_context_merge::Coordinator::new(self.client.clone(), self.prefix.clone())
                .get(&task.target)
                .await
                .map_err(lance::Error::io)?
        } else {
            None
        };
        apply_outcome(&mut task, outcome);
        let mut operations = vec![
            TxnOp::put(self.task_key(&task.id), encode_task(&task)?, None),
            TxnOp::delete(claim_key.as_str(), None),
            TxnOp::delete(self.running_key(&task.id), None),
        ];
        if owns_table {
            operations.push(TxnOp::delete(
                lance_context_merge::execution_key(&self.prefix, &task.target)
                    .replace("/merge-executions/", "/merge-claims/"),
                None,
            ));
        }
        if let Some(key) = preparation_key {
            operations.push(TxnOp::delete(key, None));
        }
        if unresolved.is_none() {
            if let Some(key) = &target_key {
                operations.push(TxnOp::delete(key.as_str(), None));
            }
        }
        if let Some(key) = self.dedupe_key(task.kind, &task.target, &task.depends_on) {
            operations.push(TxnOp::delete(key, None));
        }
        let mut compares = vec![Compare::value(
            claim_key.as_str(),
            CompareOp::Equal,
            token.as_bytes(),
        )];
        if owns_table {
            compares.push(match &unresolved {
                Some(execution) => Compare::value(
                    lance_context_merge::execution_key(&self.prefix, &task.target),
                    CompareOp::Equal,
                    serde_json::to_vec(execution).map_err(|e| lance::Error::io(e.to_string()))?,
                ),
                None => Compare::version(
                    lance_context_merge::execution_key(&self.prefix, &task.target),
                    CompareOp::Equal,
                    0,
                ),
            });
        }
        let mut client = self.client.clone();
        let completed = client
            .txn(Txn::new().when(compares).and_then(operations))
            .await
            .map(|response| response.succeeded())
            .map_err(etcd_error("complete task"));
        drop(keepalive);
        // Cleanup cannot change whether the terminal-state CAS committed. In
        // particular, lease expiry must not hide a rejected CAS or turn a
        // committed merge into a failure/backoff in a dedicated executor.
        let cleanup = self.revoke_lease(lease_id).await;
        completion_result(&task, lease_id, completed, cleanup)
    }

    async fn resolve_claim_response(
        &self,
        response: Result<etcd_client::TxnResponse, etcd_client::Error>,
        claim_key: &str,
        token: &str,
    ) -> lance::Result<bool> {
        match response {
            Ok(response) => Ok(response.succeeded()),
            Err(error) => {
                // A transport deadline can arrive after the atomic claim was
                // accepted. Adopt only our unique token, never another owner.
                if self.get_text(claim_key).await?.as_deref() == Some(token) {
                    metrics::counter!("master_task_claim_response_recovered_total").increment(1);
                    Ok(true)
                } else {
                    Err(etcd_error("claim task")(error))
                }
            }
        }
    }

    async fn recover_orphaned(&self) -> lance::Result<usize> {
        // Bound each read and batch live-claim checks: a healthy running task
        // needs no conditional recovery transaction. Keep the original CAS
        // authority for missing claims, since preflight reads may become stale.
        let mut start = self.running_prefix().into_bytes();
        let end = prefix_range_end(&start);
        let mut recovered = 0;
        loop {
            let page = self
                .client
                .clone()
                .get(
                    start,
                    Some(GetOptions::new().with_range(end.clone()).with_limit(64)),
                )
                .await
                .map_err(etcd_error("list running tasks"))?;
            let tasks = page
                .kvs()
                .iter()
                .map(|kv| decode_task(kv.value(), &String::from_utf8_lossy(kv.key())))
                .collect::<lance::Result<Vec<_>>>()?;
            recovered += self.recover_tasks(&tasks).await?;
            if !page.more() {
                return Ok(recovered);
            }
            start = page
                .kvs()
                .last()
                .ok_or_else(|| lance::Error::io("empty recovery page"))?
                .key()
                .to_vec();
            start.push(0);
        }
    }

    /// Exact keys in one read-only transaction; never a fleet prefix scan.
    async fn read_values(&self, keys: &[String]) -> lance::Result<Vec<Option<Vec<u8>>>> {
        let response = self
            .client
            .clone()
            .txn(
                Txn::new().and_then(
                    keys.iter()
                        .map(|key| TxnOp::get(key.as_str(), None))
                        .collect::<Vec<_>>(),
                ),
            )
            .await
            .map_err(etcd_error("read admission snapshot"))?;
        let responses = response.op_responses();
        if responses.len() != keys.len() {
            return Err(lance::Error::io("incomplete admission snapshot"));
        }
        responses
            .into_iter()
            .map(|op| match op {
                TxnOpResponse::Get(value) => Ok(value.kvs().first().map(|kv| kv.value().to_vec())),
                _ => Err(lance::Error::io("invalid admission snapshot response")),
            })
            .collect()
    }

    async fn recover_tasks(&self, tasks: &[TaskRecord]) -> lance::Result<usize> {
        if tasks.is_empty() {
            return Ok(0);
        }
        let claims = self
            .read_values(
                &tasks
                    .iter()
                    .map(|t| self.claim_key(&t.id))
                    .collect::<Vec<_>>(),
            )
            .await?;
        let mut recovered = 0;
        for (task, claim) in tasks.iter().zip(claims) {
            if claim.is_some() {
                continue;
            }
            if self.recover_task(task).await? {
                recovered += 1;
            }
        }
        metrics::counter!("master_task_recovery_inspected_total").increment(tasks.len() as u64);
        metrics::counter!("master_task_recovery_requeued_total").increment(recovered as u64);
        Ok(recovered)
    }

    async fn recover_task(&self, task: &TaskRecord) -> lance::Result<bool> {
        let mut queued = task.clone();
        requeue(&mut queued);
        let value = encode_task(&queued)?;
        let txn = Txn::new()
            .when([
                Compare::value(
                    self.task_key(&task.id),
                    CompareOp::Equal,
                    encode_task(task)?,
                ),
                Compare::version(self.claim_key(&task.id), CompareOp::Equal, 0),
                Compare::value(
                    self.running_key(&task.id),
                    CompareOp::Equal,
                    encode_task(task)?,
                ),
            ])
            .and_then([
                TxnOp::put(self.task_key(&task.id), value.clone(), None),
                TxnOp::put(self.queue_key(&task.id), value, None),
                TxnOp::delete(self.running_key(&task.id), None),
            ]);
        Ok(self
            .client
            .clone()
            .txn(txn)
            .await
            .map_err(etcd_error("recover orphaned task"))?
            .succeeded())
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
        match client.lease_revoke(lease_id).await {
            Ok(_) => Ok(()),
            // This is the exact etcd lease RPC error, not a substring search
            // across an arbitrary storage error or request identifier.
            Err(etcd_client::Error::GRpcStatus(status))
                if status.message() == "etcdserver: requested lease not found" =>
            {
                Ok(())
            }
            Err(error) => Err(etcd_error("revoke lease")(error)),
        }
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

    fn compaction_commit_key(&self, target: &str) -> String {
        format!(
            "{}/compact-commit-ready/{}",
            self.prefix,
            encode_segment(target)
        )
    }

    fn target_lock_key(&self, target: &str) -> String {
        format!("{}/target-locks/{}", self.prefix, encode_segment(target))
    }

    fn cooldown_prefix(&self) -> String {
        format!("{}/cooldown/", self.prefix)
    }

    fn cooldown_key(&self, kind: TaskKind, target: &str) -> String {
        format!(
            "{}{}/{}",
            self.cooldown_prefix(),
            kind_label(kind),
            encode_segment(target)
        )
    }

    /// The queued or running task the dedupe key for `(kind, target)` points
    /// at, if the key exists and that task is still active.
    async fn get_active_id(&self, kind: TaskKind, target: &str) -> lance::Result<Option<String>> {
        let Some(dedupe_key) = self.dedupe_key(kind, target, &[]) else {
            return Ok(None);
        };
        let Some(existing_id) = self.get_text(&dedupe_key).await? else {
            return Ok(None);
        };
        Ok(self
            .get(&existing_id)
            .await?
            .filter(|t| matches!(t.state, TaskState::Queued | TaskState::Running))
            .map(|t| t.id))
    }

    async fn get_cooldown(
        &self,
        kind: TaskKind,
        target: &str,
    ) -> lance::Result<Option<TaskCooldown>> {
        Ok(self
            .get_cooldown_versioned(kind, target)
            .await?
            .map(|(record, _)| record))
    }

    /// The cooldown record plus the key's etcd `mod_revision` (0 when absent),
    /// so a writer can update it with a compare-and-swap.
    async fn get_cooldown_versioned(
        &self,
        kind: TaskKind,
        target: &str,
    ) -> lance::Result<Option<(TaskCooldown, i64)>> {
        let mut client = self.client.clone();
        let response = client
            .get(self.cooldown_key(kind, target), None)
            .await
            .map_err(etcd_error("get cooldown"))?;
        response
            .kvs()
            .first()
            .map(|kv| {
                serde_json::from_slice::<TaskCooldown>(kv.value())
                    .map(|record| (record, kv.mod_revision()))
                    .map_err(|e| lance::Error::io(format!("decode cooldown: {e}")))
            })
            .transpose()
    }

    /// Bump the consecutive-failure count and, past the threshold, set the
    /// cooldown window. The record is leased for `policy.max` regardless of
    /// the window, so the count survives the window lapsing: a target that
    /// fails again right after its cooldown ends is on failure N+1 and gets
    /// a doubled window, not a fresh threshold. Without that, every master's
    /// sweep re-probed the target the instant the window closed (a burst of
    /// one task per master), it failed the threshold again, and the window
    /// never grew. Only a success clears the record.
    async fn record_failure(
        &self,
        kind: TaskKind,
        target: &str,
        error: &str,
        policy: CooldownPolicy,
    ) -> lance::Result<()> {
        // Several masters finish a failing task for the same target within
        // seconds of each other (they all swept it at the same instant), so
        // the count is bumped with a compare-and-swap on the key's revision.
        // A plain get/put would let two of them read the same count and both
        // write count+1, losing a failure and delaying the cooldown.
        let key = self.cooldown_key(kind, target);
        for attempt in 0..64u32 {
            if attempt > 0 {
                // Contention is bounded by the number of masters (a handful),
                // so a short jittered backoff is enough to let the others land.
                let jitter = (now_ms() as u64 ^ u64::from(attempt)) % 20;
                tokio::time::sleep(Duration::from_millis(5 + jitter)).await;
            }
            let (prev_failures, revision) = self
                .get_cooldown_versioned(kind, target)
                .await?
                .map_or((0, 0), |(c, rev)| (c.failures, rev));
            let failures = prev_failures.saturating_add(1);
            let now = now_ms();
            let cooling = failures >= policy.after_failures;
            let duration = policy.duration_for(failures);
            let record = TaskCooldown {
                kind,
                target: target.to_string(),
                failures,
                until_ms: if cooling {
                    Some(now + duration.as_millis() as i64)
                } else {
                    None
                },
                last_error: error.chars().take(512).collect(),
            };
            // Lease for the longest window so the count outlives any single
            // cooldown; a slow trickle of unrelated failures still ages out.
            let mut client = self.client.clone();
            let lease = client
                .lease_grant(policy.max.as_secs().max(1) as i64, None)
                .await
                .map_err(etcd_error("grant cooldown lease"))?
                .id();
            let value = serde_json::to_vec(&record)
                .map_err(|e| lance::Error::io(format!("encode cooldown: {e}")))?;
            let committed = client
                .txn(
                    Txn::new()
                        .when([Compare::mod_revision(
                            key.as_str(),
                            CompareOp::Equal,
                            revision,
                        )])
                        .and_then([TxnOp::put(
                            key.as_str(),
                            value,
                            Some(PutOptions::new().with_lease(lease)),
                        )]),
                )
                .await
                .map_err(etcd_error("put cooldown"))?
                .succeeded();
            if !committed {
                // Someone else bumped it first; re-read and try again.
                let _ = client.lease_revoke(lease).await;
                continue;
            }
            if cooling {
                metrics::counter!("master_task_cooldowns_total", "kind" => kind_label(kind))
                    .increment(1);
                tracing::warn!(
                    kind = ?kind,
                    target = %target,
                    failures,
                    cooldown_secs = duration.as_secs(),
                    last_error = %record.last_error,
                    "target failed repeatedly; sweeps will skip it until the cooldown lapses"
                );
            }
            return Ok(());
        }
        Err(lance::Error::io(format!(
            "cooldown for {kind:?} '{target}' contended past 64 attempts"
        )))
    }

    async fn clear_cooldown(&self, kind: TaskKind, target: &str) -> lance::Result<()> {
        let mut client = self.client.clone();
        client
            .delete(self.cooldown_key(kind, target), None)
            .await
            .map_err(etcd_error("clear cooldown"))?;
        Ok(())
    }

    async fn list_cooldowns(&self) -> lance::Result<Vec<TaskCooldown>> {
        let mut client = self.client.clone();
        let response = client
            .get(
                self.cooldown_prefix(),
                Some(GetOptions::new().with_prefix()),
            )
            .await
            .map_err(etcd_error("list cooldowns"))?;
        response
            .kvs()
            .iter()
            .map(|kv| {
                serde_json::from_slice::<TaskCooldown>(kv.value())
                    .map_err(|e| lance::Error::io(format!("decode cooldown: {e}")))
            })
            .filter(|r| !matches!(r, Ok(c) if c.until_ms.is_none_or(|until| until <= now_ms())))
            .collect()
    }

    fn repairs_prefix(&self) -> String {
        format!("{}/repairs/", self.prefix)
    }

    async fn record_repair(&self, record: &RepairRecord, ttl_secs: u64) -> lance::Result<()> {
        let mut client = self.client.clone();
        let lease = client
            .lease_grant(ttl_secs.max(60) as i64, None)
            .await
            .map_err(etcd_error("grant repair lease"))?
            .id();
        let key = format!(
            "{}{}/{}",
            self.repairs_prefix(),
            encode_segment(&record.target),
            record.repaired_at_ms
        );
        let value = serde_json::to_vec(record)
            .map_err(|e| lance::Error::io(format!("encode repair: {e}")))?;
        client
            .put(key, value, Some(PutOptions::new().with_lease(lease)))
            .await
            .map_err(etcd_error("put repair"))?;
        Ok(())
    }

    async fn list_repairs(&self) -> lance::Result<Vec<RepairRecord>> {
        let mut client = self.client.clone();
        let response = client
            .get(self.repairs_prefix(), Some(GetOptions::new().with_prefix()))
            .await
            .map_err(etcd_error("list repairs"))?;
        let mut out = response
            .kvs()
            .iter()
            .map(|kv| {
                serde_json::from_slice::<RepairRecord>(kv.value())
                    .map_err(|e| lance::Error::io(format!("decode repair: {e}")))
            })
            .collect::<lance::Result<Vec<_>>>()?;
        out.sort_by_key(|r| std::cmp::Reverse(r.repaired_at_ms));
        Ok(out)
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
            TaskKind::Compact | TaskKind::IndexId | TaskKind::MergeWal | TaskKind::Repair
        )
}

fn requires_target_lock(kind: TaskKind) -> bool {
    matches!(
        kind,
        TaskKind::Compact | TaskKind::IndexId | TaskKind::Repair | TaskKind::MergeWal
    )
}

fn kind_label(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Compact => "compact",
        TaskKind::MergeWal => "merge-wal",
        TaskKind::IndexId => "index-id",
        TaskKind::Repair => "repair",
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

fn completion_result(
    task: &TaskRecord,
    lease_id: i64,
    completed: lance::Result<bool>,
    cleanup: lance::Result<()>,
) -> lance::Result<()> {
    let outcome = match &completed {
        Ok(true) => "committed",
        Ok(false) => "claim_lost",
        Err(_) => "unknown",
    };
    metrics::counter!("master_task_completion_total", "outcome" => outcome).increment(1);
    tracing::info!(task = %task.id, target = %task.target, lease_id, outcome, "task completion transaction result");
    if let Err(error) = cleanup {
        metrics::counter!("master_task_completion_cleanup_failed_total").increment(1);
        tracing::warn!(task = %task.id, target = %task.target, lease_id, outcome, %error,
            "task lease cleanup failed; preserving completion transaction result");
    }
    match completed {
        Ok(true) => Ok(()),
        Ok(false) => Err(lance::Error::io(format!(
            "task '{}' lost its etcd claim before completion",
            task.id
        ))),
        // A lost response is not evidence of either a committed or rejected
        // transaction. Leave recovery to the durable task/ownership protocol.
        Err(error) => Err(error),
    }
}

fn etcd_error(action: &'static str) -> impl FnOnce(etcd_client::Error) -> lance::Error {
    move |err| lance::Error::io(format!("failed to {action}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn completion_cleanup_never_changes_the_transaction_outcome() {
        let task = new_task(TaskKind::MergeWal, "hot", Vec::new());
        let cleanup_error = || Err(lance::Error::io("injected lease revoke timeout"));
        assert!(completion_result(&task, 1, Ok(true), cleanup_error()).is_ok());
        let lost = completion_result(&task, 1, Ok(false), cleanup_error()).unwrap_err();
        assert!(lost.to_string().contains("lost its etcd claim"));
        let unknown = completion_result(
            &task,
            1,
            Err(lance::Error::io("injected completion response loss")),
            cleanup_error(),
        )
        .unwrap_err();
        assert!(unknown.to_string().contains("completion response loss"));
        assert!(!unknown.to_string().contains("lease revoke"));
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn expired_merge_claim_reports_ownership_loss_and_preserves_replacement() {
        let (_dir, _cfg, store) = preparation_test_store().await;
        let task = store
            .enqueue(TaskKind::MergeWal, "hot", Vec::new())
            .await
            .unwrap();
        let old = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let lease = old.backend.lease_id;
        store.inner.revoke_lease(lease).await.unwrap();
        // An already absent lease is a successful idempotent cleanup.
        store.inner.revoke_lease(lease).await.unwrap();
        store.recover_orphaned().await.unwrap();
        let replacement = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(replacement.task.id, task.id);
        let error = store
            .finish(old, Ok("stale successful execution".into()))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("lost its etcd claim"), "{error}");
        assert_eq!(
            store
                .inner
                .get_text(&replacement.backend.claim_key)
                .await
                .unwrap()
                .as_deref(),
            Some(replacement.backend.token.as_str())
        );
        assert_eq!(
            store
                .inner
                .get_text(&store.inner.target_lock_key("hot"))
                .await
                .unwrap()
                .as_deref(),
            Some(replacement.backend.token.as_str())
        );
        store
            .finish(replacement, Ok("replacement completed".into()))
            .await
            .unwrap();
        assert_eq!(
            store
                .inner
                .get(&task.id)
                .await
                .unwrap()
                .unwrap()
                .detail
                .as_deref(),
            Some("replacement completed")
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn benchmark_target_admission_with_eighty_healthy_tasks() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/admission-benchmark/{}", generate_id());
        cfg.catchup.target = Some("hot".into());
        let store = TaskStore::open(&cfg).await.unwrap();
        let lease = store.inner.grant_lease().await.unwrap();
        let keepalive = store.inner.start_keepalive(lease).await.unwrap();
        let mut client = store.inner.client.clone();
        for i in 0..80 {
            let mut task = new_task(TaskKind::MergeWal, &format!("healthy-{i}"), Vec::new());
            task.state = TaskState::Running;
            let value = encode_task(&task).unwrap();
            client
                .txn(Txn::new().and_then([
                    TxnOp::put(store.inner.task_key(&task.id), value.clone(), None),
                    TxnOp::put(store.inner.running_key(&task.id), value, None),
                    TxnOp::put(
                        store.inner.claim_key(&task.id),
                        "live",
                        Some(PutOptions::new().with_lease(lease)),
                    ),
                ]))
                .await
                .unwrap();
        }
        // Reproduce the old sweep's sequential conditional transactions. All
        // predicates fail because these fixture tasks have healthy claims.
        let baseline = std::time::Instant::now();
        let running = client
            .get(
                store.inner.running_prefix(),
                Some(GetOptions::new().with_prefix()),
            )
            .await
            .unwrap();
        for kv in running.kvs() {
            let task = decode_task(kv.value(), "fixture").unwrap();
            assert!(!store.inner.recover_task(&task).await.unwrap());
        }
        let old_sweep_seconds = baseline.elapsed().as_secs_f64();
        let batch = std::time::Instant::now();
        assert_eq!(store.inner.recover_orphaned().await.unwrap(), 0);
        let batched_sweep_seconds = batch.elapsed().as_secs_f64();
        client
            .put(
                crate::catchup::store::active_key(&store.inner.prefix, "hot"),
                "job",
                None,
            )
            .await
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", Vec::new())
            .await
            .unwrap();
        let start = std::time::Instant::now();
        let claim = store
            .claim_merge_target("hot", "job")
            .await
            .unwrap()
            .unwrap();
        let target_claim_seconds = start.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::json!({"healthy_running":80, "old_sweep_seconds":old_sweep_seconds,
            "batched_sweep_seconds":batched_sweep_seconds, "target_claim_seconds":target_claim_seconds})
        );
        store.finish(claim, Ok("done".into())).await.unwrap();
        drop(keepalive);
        store.inner.revoke_lease(lease).await.unwrap();
        client
            .delete(
                cfg.etcd.etcd_prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn dedicated_admission_ignores_unrelated_history_and_recovers_own_orphan() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/admission-test/{}", generate_id());
        cfg.catchup.target = Some("hot".into());
        let backend = EtcdTaskStore::connect(&cfg).await.unwrap();
        let mut client = backend.client.clone();
        // These intentionally undecodable records expose any accidental fleet
        // scan during dedicated open, claim, recovery, or finish.
        let unrelated_running = backend.running_key("unrelated");
        let unrelated_history = backend.task_key("unrelated");
        for key in [&unrelated_running, &unrelated_history] {
            client
                .put(key.as_str(), "not a task record", None)
                .await
                .unwrap();
        }
        client
            .put(
                crate::catchup::store::active_key(&backend.prefix, "hot"),
                "job",
                None,
            )
            .await
            .unwrap();
        let store = TaskStore::open(&cfg).await.unwrap();
        let task = store
            .enqueue(TaskKind::MergeWal, "hot", Vec::new())
            .await
            .unwrap();
        assert!(store
            .claim_merge_target("hot", "wrong-job")
            .await
            .unwrap()
            .is_none());
        let claim = store
            .claim_merge_target("hot", "job")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, task.id);
        assert!(
            store
                .claim_merge_target("hot", "job")
                .await
                .unwrap()
                .is_none(),
            "a live claim must remain protected"
        );
        store.abandon_claim_for_test(claim).await.unwrap();
        let recovered = store
            .claim_merge_target("hot", "job")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(recovered.task.id, task.id);
        store.finish(recovered, Ok("merged".into())).await.unwrap();
        assert_eq!(
            store.get(&task.id).await.unwrap().unwrap().state,
            TaskState::Done
        );
        for key in [&unrelated_running, &unrelated_history] {
            assert_eq!(
                backend.get_text(key).await.unwrap().as_deref(),
                Some("not a task record")
            );
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn lost_claim_response_adopts_only_its_own_token() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/admission-test/{}", generate_id());
        let store = TaskStore::open(&cfg).await.unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", Vec::new())
            .await
            .unwrap();
        let claim = store.claim_next().await.unwrap().unwrap();
        let lost = || {
            Err(etcd_client::Error::Internal(
                "simulated lost response".into(),
            ))
        };
        assert!(store
            .inner
            .resolve_claim_response(lost(), &claim.backend.claim_key, &claim.backend.token)
            .await
            .unwrap());
        assert!(store
            .inner
            .resolve_claim_response(lost(), &claim.backend.claim_key, "another-owner")
            .await
            .is_err());
        store.abandon_claim_for_test(claim).await.unwrap();
        assert!(store
            .inner
            .resolve_claim_response(lost(), &store.inner.claim_key("absent"), "another-owner")
            .await
            .is_err());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn batched_recovery_preserves_live_claims_and_crosses_pages() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/admission-test/{}", generate_id());
        let store = TaskStore::open(&cfg).await.unwrap();
        store
            .enqueue(TaskKind::MergeWal, "live", Vec::new())
            .await
            .unwrap();
        let live = store.claim_next().await.unwrap().unwrap();
        let mut client = store.inner.client.clone();
        for i in 0..70 {
            let mut task = new_task(TaskKind::MergeWal, &format!("orphan-{i}"), Vec::new());
            task.state = TaskState::Running;
            let value = encode_task(&task).unwrap();
            client
                .txn(Txn::new().and_then([
                    TxnOp::put(store.inner.task_key(&task.id), value.clone(), None),
                    TxnOp::put(store.inner.running_key(&task.id), value, None),
                ]))
                .await
                .unwrap();
        }
        assert_eq!(store.inner.recover_orphaned().await.unwrap(), 70);
        assert_eq!(
            store.get(&live.task.id).await.unwrap().unwrap().state,
            TaskState::Running
        );
        assert_eq!(store.inner.recover_orphaned().await.unwrap(), 0);
        assert_eq!(store.queue_depth().await.unwrap(), 70);
        store.finish(live, Ok("done".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn compact_noop_lease_expiry_allows_bounded_recheck() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/test/{}", generate_id());
        let store = TaskStore::open(&cfg).await.unwrap();
        store
            .record_compact_noop("table", "/table", 10, "options")
            .await
            .unwrap();
        assert!(store
            .compact_is_unchanged("table", "/table", 10, "options")
            .await
            .unwrap());
        assert!(!store
            .compact_is_unchanged("table", "/elsewhere", 10, "options")
            .await
            .unwrap());
        assert!(!store
            .compact_is_unchanged("table", "/table", -1, "options")
            .await
            .unwrap());
        let key = lance_context_merge::execution_key(&cfg.etcd.etcd_prefix, "table")
            .replace("/merge-executions/", "/compact-noops/");
        let mut client = store.inner.client.clone();
        let kv = client.get(key, None).await.unwrap();
        let lease = kv.kvs()[0].lease();
        let ttl = client.lease_time_to_live(lease, None).await.unwrap().ttl();
        assert!((1..=900).contains(&ttl));
        client.lease_revoke(lease).await.unwrap();
        assert!(!store
            .compact_is_unchanged("table", "/table", 10, "options")
            .await
            .unwrap());
    }

    fn config(dir: &TempDir) -> MasterConfig {
        MasterConfig {
            append: Default::default(),
            catchup: Default::default(),
            wal_tail: Default::default(),
            maintenance: Default::default(),
            merge_rollout: Default::default(),
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
            compaction_concurrency: 1,
            compaction_threads: 1,
            compaction_batch_size: 8,
            compaction_max_source_fragments: 32,
            index_after_compaction: false,
            key_index_type: Default::default(),
            index_before_merge: false,
            compaction_max_bytes_per_file: 1024 * 1024 * 1024,
            merge_wal_interval_secs: 0,
            merge_wal_min_generations: 8,
            worker_endpoints: vec![],
            task_concurrency: 4,
            merge_wal_concurrency: 4,
            etcd: lance_context_core::etcd::EtcdConfig {
                etcd_endpoints: vec![],
                etcd_prefix: "/test".to_string(),
                ..Default::default()
            },
            registry: lance_context_core::etcd::RegistryConfig::default(),
            etcd_lease_ttl_secs: 30,
            task_history_limit: 1_000,
            task_history_ttl_secs: 86_400,
            task_cooldown_after_failures: 3,
            task_cooldown_base_secs: 600,
            task_cooldown_max_secs: 21_600,
            ui_dir: None,
        }
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn merge_serializes_other_writers_and_survives_claim_loss() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_string)
            .collect();
        cfg.etcd.etcd_prefix = format!("/merge-lock-test/{}", generate_id());
        let store = TaskStore::open(&cfg).await.unwrap();
        store
            .enqueue(TaskKind::MergeWal, "shared", Vec::new())
            .await
            .unwrap();
        let merge = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store
            .enqueue(TaskKind::Compact, "shared", Vec::new())
            .await
            .unwrap();
        store
            .enqueue(TaskKind::IndexId, "shared", Vec::new())
            .await
            .unwrap();
        assert!(
            store
                .claim_next_of_kinds(TaskKinds::GENERAL)
                .await
                .unwrap()
                .is_none(),
            "merge must lock against index and compact"
        );
        store
            .enqueue(TaskKind::Compact, "unrelated", Vec::new())
            .await
            .unwrap();
        let other = store
            .claim_next_of_kinds(TaskKinds::GENERAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(other.task.target, "unrelated");
        let dependency = other.task.id.clone();
        store.finish(other, Ok("done".into())).await.unwrap();
        // A dependency-chain task has a distinct id and bypasses dedupe.
        store
            .enqueue(TaskKind::MergeWal, "shared", vec![dependency])
            .await
            .unwrap();
        let coordinator = store.merge_coordinator();
        let execution = lance_context_merge::Execution::new("shared", "worker", "boot", 600);
        assert!(coordinator
            .reserve(&store.merge_claim(&merge), &execution)
            .await
            .unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        assert!(
            store
                .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                .await
                .unwrap()
                .is_none(),
            "only one scheduler may own reconciliation, even with a persistent execution lock"
        );
        store.abandon_claim_for_test(merge).await.unwrap();
        store.inner.recover_orphaned().await.unwrap();
        assert!(
            store
                .claim_next_of_kinds(TaskKinds::GENERAL)
                .await
                .unwrap()
                .is_none(),
            "expired claim is not permission to write past live worker execution"
        );
        let recovered = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store
            .finish(recovered, Err("merge ownership unresolved".into()))
            .await
            .unwrap();
        assert_eq!(
            coordinator.get("shared").await.unwrap(),
            Some(running.clone())
        );
        assert!(store
            .claim_next_of_kinds(TaskKinds::GENERAL)
            .await
            .unwrap()
            .is_none());
        let recovered = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert!(coordinator
            .finish(&running, Err("cancelled".into()))
            .await
            .unwrap());
        let done = coordinator.get("shared").await.unwrap().unwrap();
        assert!(coordinator
            .release(&store.merge_claim(&recovered), &done)
            .await
            .unwrap());
        store
            .finish(recovered, Ok("recovered".into()))
            .await
            .unwrap();
        let compact = store
            .claim_next_of_kinds(TaskKinds::GENERAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(compact.task.kind, TaskKind::Compact);
        store
            .enqueue(TaskKind::MergeWal, "shared", Vec::new())
            .await
            .unwrap();
        assert!(
            store
                .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                .await
                .unwrap()
                .is_none(),
            "compact must also block a new merge"
        );
        store.finish(compact, Ok("done".into())).await.unwrap();
        store
            .inner
            .client
            .clone()
            .delete(
                cfg.etcd.etcd_prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    async fn preparation_test_store() -> (TempDir, MasterConfig, TaskStore) {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_string)
            .collect();
        cfg.etcd.etcd_prefix = format!("/lance-context/test/{}", generate_id());
        cfg.maintenance.compaction_prepare_targets = vec!["*".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        (dir, cfg, store)
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn resident_claim_filter_preserves_legacy_draining_and_external_owners() {
        let (_dir, mut cfg, _) = preparation_test_store().await;
        cfg.merge_rollout.owned_targets = ["hot", "draining", "excluded", "external", "generic:gs"]
            .map(str::to_owned)
            .to_vec();
        cfg.merge_rollout.drain_targets = vec!["draining".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        let mut queued = Vec::new();
        for target in [
            "legacy",
            "draining",
            "excluded",
            "external",
            "generic:gs",
            "hot",
        ] {
            queued.push(
                store
                    .enqueue(TaskKind::MergeWal, target, vec![])
                    .await
                    .unwrap(),
            );
        }
        let key = crate::catchup::store::active_key(&cfg.etcd.etcd_prefix, "external");
        store
            .inner
            .client
            .clone()
            .put(key.clone(), "external-owner", None)
            .await
            .unwrap();
        let targets = ["hot", "draining", "external", "generic:gs"].map(str::to_owned);
        let claim = store.claim_resident_merge(&targets).await.unwrap().unwrap();
        assert_eq!(claim.task.target, "hot");
        // A second replica cannot take the running canonical task.
        let other = TaskStore::open(&cfg).await.unwrap();
        assert!(other
            .claim_resident_merge(&targets)
            .await
            .unwrap()
            .is_none());
        for task in &queued[..5] {
            assert_eq!(
                store.get(&task.id).await.unwrap().unwrap().state,
                TaskState::Queued
            );
        }
        assert_eq!(
            store.inner.get_text(&key).await.unwrap().as_deref(),
            Some("external-owner")
        );
        store.finish(claim, Ok("done".into())).await.unwrap();
        let wildcard = store
            .claim_resident_merge(&["*".into()])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(wildcard.task.target, "excluded");
        store.finish(wildcard, Ok("done".into())).await.unwrap();
        assert!(store
            .claim_resident_merge(&["*".into()])
            .await
            .unwrap()
            .is_none());
        store
            .inner
            .client
            .clone()
            .delete(
                cfg.etcd.etcd_prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn index_preparation_allows_merge_and_shares_compact_commit_turn() {
        let (_dir, mut cfg, _) = preparation_test_store().await;
        cfg.maintenance.index_prepare_targets = vec!["hot".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let merge = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store
            .enqueue(TaskKind::IndexId, "hot", vec![])
            .await
            .unwrap();
        let mut prep = store
            .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
            .await
            .unwrap()
            .unwrap();
        assert!(prep.preparing_maintenance());
        // Readers/file builders can start even when merge already owns the table.
        assert!(!store.promote_maintenance(&mut prep).await.unwrap());
        store
            .enqueue(TaskKind::Compact, "hot", vec![])
            .await
            .unwrap();
        assert!(store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .is_none());
        assert!(store.request_maintenance_commit(&prep).await.unwrap());
        store
            .finish(merge, Ok("merge advanced during index build".into()))
            .await
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        // This uses the old compact wire keys: old masters also yield the turn.
        assert!(store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .is_none());
        assert!(store.promote_maintenance(&mut prep).await.unwrap());
        assert!(store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .is_none());
        store
            .finish(prep, Ok("index published".into()))
            .await
            .unwrap();
        let merge = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store.finish(merge, Ok("continued".into())).await.unwrap();
        let compact = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        store.finish(compact, Ok("continued".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn expired_index_preparation_cannot_publish_or_release_replacement() {
        let (_dir, mut cfg, _) = preparation_test_store().await;
        cfg.maintenance.index_prepare_targets = vec!["*".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        let task = store
            .enqueue(TaskKind::IndexId, "hot", vec![])
            .await
            .unwrap();
        let kinds = TaskKinds::GENERAL.without_compact();
        let mut old = store.claim_next_of_kinds(kinds).await.unwrap().unwrap();
        assert!(store.request_maintenance_commit(&old).await.unwrap());
        store
            .inner
            .revoke_lease(old.backend.lease_id)
            .await
            .unwrap();
        store.recover_orphaned().await.unwrap();
        let mut replacement = store.claim_next_of_kinds(kinds).await.unwrap().unwrap();
        assert_eq!(replacement.task.id, task.id);
        assert!(!store.request_maintenance_commit(&old).await.unwrap());
        assert!(!store.promote_maintenance(&mut old).await.unwrap());
        assert!(store.finish(old, Ok("stale".into())).await.is_err());
        assert!(store.preparation_owned(&replacement).await.unwrap());
        assert!(store.promote_maintenance(&mut replacement).await.unwrap());
        store.finish(replacement, Ok("valid".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn compaction_preparation_allows_merge_but_commit_is_exclusive() {
        let (_dir, _cfg, store) = preparation_test_store().await;
        store
            .enqueue(TaskKind::Compact, "hot", Vec::new())
            .await
            .unwrap();
        let mut prep = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        assert!(prep.preparing_maintenance());
        assert!(store
            .inner
            .get_text(&store.inner.target_lock_key("hot"))
            .await
            .unwrap()
            .is_none());
        store
            .enqueue(TaskKind::MergeWal, "hot", Vec::new())
            .await
            .unwrap();
        let merge = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert!(!store.promote_maintenance(&mut prep).await.unwrap());
        store
            .finish(merge, Ok("append while rewriting".into()))
            .await
            .unwrap();
        assert!(store.promote_maintenance(&mut prep).await.unwrap());
        assert!(!prep.preparing_maintenance());
        store
            .enqueue(TaskKind::MergeWal, "hot", Vec::new())
            .await
            .unwrap();
        assert!(store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .is_none());
        store
            .finish(prep, Ok("compact published".into()))
            .await
            .unwrap();
        let next = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store.finish(next, Ok("continued".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn ready_compaction_gets_the_next_native_merge_turn_without_blocking_preparation() {
        let (_dir, mut cfg, _) = preparation_test_store().await;
        cfg.maintenance.maintenance_catchup_targets = vec!["hot".into()];
        cfg.merge_rollout.owned_targets = vec!["hot".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        let active = crate::catchup::store::active_key(&cfg.etcd.etcd_prefix, "hot");
        store
            .inner
            .client
            .clone()
            .put(active, "native", None)
            .await
            .unwrap();
        store
            .enqueue(TaskKind::Compact, "hot", vec![])
            .await
            .unwrap();
        let mut prep = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let native = store
            .claim_merge_target("hot", "native")
            .await
            .unwrap()
            .unwrap();
        assert!(
            prep.preparing_maintenance(),
            "preparation must not block native merge admission"
        );
        assert!(store.request_maintenance_commit(&prep).await.unwrap());
        assert!(
            !store.promote_maintenance(&mut prep).await.unwrap(),
            "running merge must finish cooperatively"
        );
        store
            .finish(native, Ok("joined pass".into()))
            .await
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        assert!(
            store
                .claim_merge_target("hot", "native")
                .await
                .unwrap()
                .is_none(),
            "new pass must yield to ready compact"
        );
        assert!(store.promote_maintenance(&mut prep).await.unwrap());
        store
            .finish(prep, Ok("compact committed".into()))
            .await
            .unwrap();
        assert!(store
            .inner
            .get_text(&store.inner.compaction_commit_key("hot"))
            .await
            .unwrap()
            .is_none());
        let next = store
            .claim_merge_target("hot", "native")
            .await
            .unwrap()
            .unwrap();
        store
            .finish(next, Ok("merge resumes".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn ready_compaction_lease_loss_unblocks_merges_and_cannot_publish_a_stale_request() {
        let (_dir, _cfg, store) = preparation_test_store().await;
        store
            .enqueue(TaskKind::Compact, "hot", vec![])
            .await
            .unwrap();
        let prep = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        assert!(store.request_maintenance_commit(&prep).await.unwrap());
        store
            .inner
            .revoke_lease(prep.backend.lease_id)
            .await
            .unwrap();
        assert!(!store.request_maintenance_commit(&prep).await.unwrap());
        assert!(store
            .inner
            .get_text(&store.inner.compaction_commit_key("hot"))
            .await
            .unwrap()
            .is_none());
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let merge = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store
            .finish(merge, Ok("merge remains available".into()))
            .await
            .unwrap();
        drop(prep);
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn ready_compaction_never_blocks_recovery_of_an_unresolved_merge_execution() {
        let (_dir, _cfg, store) = preparation_test_store().await;
        store
            .enqueue(TaskKind::Compact, "hot", vec![])
            .await
            .unwrap();
        let prep = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let merge = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let coordinator = store.merge_coordinator();
        let execution = lance_context_merge::Execution::new("hot", "worker", "boot", 600);
        assert!(coordinator
            .reserve(&store.merge_claim(&merge), &execution)
            .await
            .unwrap());
        assert!(store.request_maintenance_commit(&prep).await.unwrap());
        store.abandon_claim_for_test(merge).await.unwrap();
        let recovery = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(coordinator.get("hot").await.unwrap(), Some(execution));
        store
            .finish(
                recovery,
                Err("fixture retains unresolved storage ownership".into()),
            )
            .await
            .unwrap();
        store
            .finish(prep, Err("fixture preparation ends".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn opted_in_native_owner_allows_preparation_and_exclusive_maintenance() {
        let (_dir, mut cfg, _) = preparation_test_store().await;
        cfg.maintenance.maintenance_catchup_targets = vec!["hot".into()];
        cfg.merge_rollout.owned_targets = vec!["hot".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        let active = crate::catchup::store::active_key(&cfg.etcd.etcd_prefix, "hot");
        store
            .inner
            .client
            .clone()
            .put(active.clone(), "native", None)
            .await
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", Vec::new())
            .await
            .unwrap();
        assert!(store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .is_none());
        let native = store
            .claim_merge_target("hot", "native")
            .await
            .unwrap()
            .unwrap();
        store
            .enqueue(TaskKind::Compact, "hot", Vec::new())
            .await
            .unwrap();
        let mut prep = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        assert!(prep.preparing_maintenance());
        assert!(!store.promote_maintenance(&mut prep).await.unwrap());
        store
            .finish(native, Ok("native pass joined".into()))
            .await
            .unwrap();
        assert!(store.promote_maintenance(&mut prep).await.unwrap());
        store
            .enqueue(TaskKind::MergeWal, "hot", Vec::new())
            .await
            .unwrap();
        assert!(store
            .claim_merge_target("hot", "native")
            .await
            .unwrap()
            .is_none());
        store
            .finish(prep, Ok("compact published".into()))
            .await
            .unwrap();
        store
            .enqueue(TaskKind::IndexId, "hot", Vec::new())
            .await
            .unwrap();
        let index = store
            .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(index.task.kind, TaskKind::IndexId);
        assert!(store
            .claim_merge_target("hot", "native")
            .await
            .unwrap()
            .is_none());
        store
            .finish(index, Ok("index published".into()))
            .await
            .unwrap();
        let native = store
            .claim_merge_target("hot", "native")
            .await
            .unwrap()
            .unwrap();
        store
            .finish(native, Ok("native drainage continues".into()))
            .await
            .unwrap();
        assert_eq!(
            store.inner.get_text(&active).await.unwrap().as_deref(),
            Some("native")
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn native_maintenance_opt_in_cannot_bypass_legacy_generic_or_drain_guards() {
        for (target, owned, draining, opt_in) in [
            ("hot", false, false, "hot"),
            ("hot", true, true, "hot"),
            ("generic:hot", true, false, "generic:hot"),
            ("hot", true, false, "*"),
        ] {
            let (_dir, mut cfg, _) = preparation_test_store().await;
            cfg.maintenance.maintenance_catchup_targets = vec![opt_in.into()];
            if owned {
                cfg.merge_rollout.owned_targets = vec![target.into()];
            }
            if draining {
                cfg.merge_rollout.drain_targets = vec![target.into()];
            }
            let store = TaskStore::open(&cfg).await.unwrap();
            let active = crate::catchup::store::active_key(&cfg.etcd.etcd_prefix, target);
            store
                .inner
                .client
                .clone()
                .put(active.clone(), "native", None)
                .await
                .unwrap();
            store
                .enqueue(TaskKind::Compact, target, Vec::new())
                .await
                .unwrap();
            store
                .enqueue(TaskKind::IndexId, target, Vec::new())
                .await
                .unwrap();
            assert!(store
                .claim_next_of_kinds(TaskKinds::GENERAL)
                .await
                .unwrap()
                .is_none());
            assert_eq!(
                store.inner.get_text(&active).await.unwrap().as_deref(),
                Some("native")
            );
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn native_owner_change_invalidates_prepared_admission() {
        let (_dir, mut cfg, _) = preparation_test_store().await;
        cfg.maintenance.maintenance_catchup_targets = vec!["hot".into()];
        cfg.merge_rollout.owned_targets = vec!["hot".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        let active = crate::catchup::store::active_key(&cfg.etcd.etcd_prefix, "hot");
        store
            .inner
            .client
            .clone()
            .put(active.clone(), "native", None)
            .await
            .unwrap();
        store
            .enqueue(TaskKind::Compact, "hot", Vec::new())
            .await
            .unwrap();
        let mut prep = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        store
            .inner
            .client
            .clone()
            .put(active.clone(), "replacement", None)
            .await
            .unwrap();
        assert!(!store.promote_maintenance(&mut prep).await.unwrap());
        store
            .finish(prep, Err("native identity changed".into()))
            .await
            .unwrap();
        assert_eq!(
            store.inner.get_text(&active).await.unwrap().as_deref(),
            Some("replacement")
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn preparation_and_promotion_respect_dedicated_owner() {
        let (_dir, cfg, store) = preparation_test_store().await;
        store
            .enqueue(TaskKind::Compact, "hot", Vec::new())
            .await
            .unwrap();
        let active = crate::catchup::store::active_key(&cfg.etcd.etcd_prefix, "hot");
        store
            .inner
            .client
            .clone()
            .put(active.clone(), "dedicated", None)
            .await
            .unwrap();
        assert!(store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .is_none());
        store
            .inner
            .client
            .clone()
            .delete(active.clone(), None)
            .await
            .unwrap();
        let mut prep = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        store
            .inner
            .client
            .clone()
            .put(active.clone(), "new-dedicated", None)
            .await
            .unwrap();
        assert!(!store.promote_maintenance(&mut prep).await.unwrap());
        store.finish(prep, Err("superseded".into())).await.unwrap();
        assert_eq!(
            store.inner.get_text(&active).await.unwrap().as_deref(),
            Some("new-dedicated")
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn failed_preparation_cannot_release_an_active_merge() {
        let (_dir, _cfg, store) = preparation_test_store().await;
        store
            .enqueue(TaskKind::Compact, "hot", Vec::new())
            .await
            .unwrap();
        let prep = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", Vec::new())
            .await
            .unwrap();
        let merge = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let coordinator = store.merge_coordinator();
        let execution = lance_context_merge::Execution::new("hot", "worker", &merge.task.id, 600);
        assert!(coordinator
            .reserve(&store.merge_claim(&merge), &execution)
            .await
            .unwrap());
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        store
            .finish(prep, Err("preparation failed".into()))
            .await
            .unwrap();
        assert_eq!(coordinator.get("hot").await.unwrap(), Some(running.clone()));
        assert_eq!(
            store
                .inner
                .get_text(&store.inner.target_lock_key("hot"))
                .await
                .unwrap(),
            Some(lance_context_merge::execution_owner(&running))
        );
        assert!(coordinator.finish(&running, Ok(1)).await.unwrap());
        let terminal = coordinator.get("hot").await.unwrap().unwrap();
        assert!(coordinator
            .release(&store.merge_claim(&merge), &terminal)
            .await
            .unwrap());
        store.finish(merge, Ok("done".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn expired_preparation_cannot_promote_or_finish_replacement() {
        let (_dir, _cfg, store) = preparation_test_store().await;
        let task = store
            .enqueue(TaskKind::Compact, "hot", Vec::new())
            .await
            .unwrap();
        let mut old = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        store
            .inner
            .revoke_lease(old.backend.lease_id)
            .await
            .unwrap();
        store.recover_orphaned().await.unwrap();
        let mut replacement = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(replacement.task.id, task.id);
        assert!(!store.preparation_owned(&old).await.unwrap());
        assert!(!store.promote_maintenance(&mut old).await.unwrap());
        assert!(store.finish(old, Ok("stale".into())).await.is_err());
        assert!(store.preparation_owned(&replacement).await.unwrap());
        assert!(store.promote_maintenance(&mut replacement).await.unwrap());
        store.finish(replacement, Ok("valid".into())).await.unwrap();
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

    /// Several masters finish a failing task for the same target within
    /// seconds of one another. Their failure bookkeeping must not lose counts
    /// to a read-modify-write race, or a broken store takes longer to cool
    /// down than the policy says.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn concurrent_failure_records_are_not_lost() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS")
            .expect("ETCD_TEST_ENDPOINTS must point to a test etcd");
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = endpoint.split(',').map(str::to_string).collect();
        cfg.etcd.etcd_prefix = format!("/lance-context/test/{}", generate_id());
        cfg.task_cooldown_after_failures = 100; // stay below threshold; count only

        let store = TaskStore::open(&cfg).await.unwrap();
        let policy = store.cooldown;
        let n = 12;
        let calls = (0..n).map(|_| {
            let inner = store.inner.clone();
            async move {
                inner
                    .record_failure(TaskKind::MergeWal, "racy", "boom", policy)
                    .await
                    .unwrap();
            }
        });
        futures::future::join_all(calls).await;

        let record = store
            .inner
            .get_cooldown(TaskKind::MergeWal, "racy")
            .await
            .unwrap()
            .expect("record exists");
        assert_eq!(
            record.failures, n,
            "every concurrent failure must be counted; a lost update means a broken store cools down late"
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn etcd_coordinates_dedupe_claims_and_target_locks() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS")
            .expect("ETCD_TEST_ENDPOINTS must point to a test etcd");
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = endpoint.split(',').map(str::to_string).collect();
        cfg.etcd.etcd_prefix = format!("/lance-context/test/{}", generate_id());
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

        let mut client = Client::connect(cfg.etcd.etcd_endpoints, None)
            .await
            .unwrap();
        client
            .delete(
                cfg.etcd.etcd_prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    /// A kind-filtered claim reaches past a large backlog of another kind.
    ///
    /// This is the starvation fix in miniature: many `MergeWal` tasks are
    /// enqueued *before* a single `Compact`, so the `Compact` is last in the
    /// queue's FIFO order. An unfiltered claim returns MergeWal every time --
    /// asserted below, so the test still describes the old behavior -- while a
    /// `GENERAL`-filtered claim must skip all of them and find the Compact.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn filtered_claim_reaches_compact_behind_merge_wal_backlog() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS")
            .expect("ETCD_TEST_ENDPOINTS must point to a test etcd");
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = endpoint.split(',').map(str::to_string).collect();
        cfg.etcd.etcd_prefix = format!("/lance-context/test/{}", generate_id());
        cfg.etcd_lease_ttl_secs = 5;

        let store = TaskStore::open(&cfg).await.unwrap();

        // 30 MergeWal tasks (distinct targets, so dedupe keeps them all), then
        // the Compact last -- worst case for FIFO order.
        for i in 0..30 {
            store
                .enqueue(TaskKind::MergeWal, &format!("exp-{i}"), Vec::new())
                .await
                .unwrap();
        }
        let compact = store
            .enqueue(TaskKind::Compact, "starved", Vec::new())
            .await
            .unwrap();

        // Unfiltered: FIFO hands back a MergeWal, never the trailing Compact.
        let any = store.claim_next().await.unwrap().unwrap();
        assert_eq!(
            any.task.kind,
            TaskKind::MergeWal,
            "the backlog head must still be MergeWal -- otherwise this test is \
             not exercising the starvation scenario"
        );
        store.finish(any, Ok("done".to_string())).await.unwrap();

        // Filtered to what the general pool runs: must skip the whole backlog.
        let claimed = store
            .claim_next_of_kinds(TaskKinds::GENERAL)
            .await
            .unwrap()
            .expect("a Compact behind a MergeWal backlog must still be claimable");
        assert_eq!(claimed.task.kind, TaskKind::Compact);
        assert_eq!(claimed.task.id, compact.id);
        store.finish(claimed, Ok("done".to_string())).await.unwrap();

        // And with no Compact left, the general filter reports empty rather
        // than falling back to the still-large MergeWal backlog.
        assert!(
            store
                .claim_next_of_kinds(TaskKinds::GENERAL)
                .await
                .unwrap()
                .is_none(),
            "GENERAL must not claim MergeWal"
        );
        // The merge poller still sees its own backlog.
        let merge = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .expect("MergeWal backlog must remain claimable by its own pool");
        assert_eq!(merge.task.kind, TaskKind::MergeWal);
        store.finish(merge, Ok("done".to_string())).await.unwrap();

        let mut client = Client::connect(cfg.etcd.etcd_endpoints, None)
            .await
            .unwrap();
        client
            .delete(
                cfg.etcd.etcd_prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }
}
