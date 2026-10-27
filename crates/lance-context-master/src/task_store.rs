//! Durable scheduler task storage backed by etcd.
//!
//! Scheduler state lives entirely in etcd: compare-and-swap enqueue,
//! lease-backed task claims, and distributed per-experiment write locks so
//! several stateless masters can safely drain one shared queue. Because claims
//! ride renewed leases, a master that disappears has its in-flight task requeued
//! after the lease expires — execution is at-least-once, so task
//! implementations must be idempotent.

use lance_context_metrics::etcd::{ObserveEtcd, Operation};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(test)]
use std::sync::OnceLock;
use std::time::Duration;

use chrono::Utc;
use etcd_client::{Client, Compare, CompareOp, GetOptions, PutOptions, Txn, TxnOp, TxnOpResponse};
use lance_context_api::{RepairRecord, TaskCooldown, TaskKind, TaskRecord, TaskState};
use lance_context_core::generate_id;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::admission_telemetry::{Context as AdmissionContext, Origin, Outcome};
use crate::config::MasterConfig;
use crate::eligibility;

#[path = "task_admission.rs"]
mod diagnostic;
pub(crate) use diagnostic::AdmissionExplanation;

#[path = "maintenance_watch.rs"]
mod maintenance_watch;
pub(crate) use maintenance_watch::MaintenanceChange;

#[path = "preparation_watch.rs"]
mod preparation_watch;

const TASK_POLL_BATCH: usize = 256;
const SCHEDULER_CLAIM_MAX_CANDIDATES: usize = 64;
const SCHEDULER_CLAIM_MAX_PAGES: usize = 4;
// Bound advisory lookahead independently of queue pages and eligible-candidate
// budgets. Claims are still serial CAS operations; this batches reads only.
// Eight ordinary tasks fill the existing 64-key envelope. Dependency-heavy
// batches still stop at the key limit below.
const CLAIM_READ_BATCH_TASKS: usize = 8;
const CLAIM_READ_BATCH_KEYS: usize = 64;

/// Owned by one scheduler pool, never shared between kind/target filters.
/// A fixed etcd revision closes each sweep even while new tasks keep arriving.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct SchedulerClaimCursor {
    queue: ScanCursor,
    recovery: ScanCursor,
}

#[cfg(test)]
impl SchedulerClaimCursor {
    pub(crate) fn has_more_pages(&self) -> bool {
        self.queue.next.is_some() || self.recovery.next.is_some()
    }
}

#[derive(Default)]
struct ScanCursor {
    next: Option<Vec<u8>>,
    revision: Option<i64>,
}

impl ScanCursor {
    fn advance(&mut self, key: &[u8], last_in_round: bool) {
        if last_in_round {
            *self = Self::default();
        } else {
            let mut next = key.to_vec();
            next.push(0);
            self.next = Some(next);
        }
    }
}

struct SchedulerClaimScan<'a> {
    cursor: &'a mut ScanCursor,
    operation: &'a crate::admission::Operation,
    candidates: usize,
    pages: usize,
}

impl SchedulerClaimScan<'_> {
    fn advance(&mut self, key: &[u8], last_in_round: bool) {
        self.cursor.advance(key, last_in_round);
        self.operation.claim_completed(0, 1, 0);
    }
}

/// A set of [`TaskKind`]s a claim is allowed to return.
///
/// Small enough to copy; used to let each scheduler execution pool claim only
/// the kinds it can actually run, so a saturated pool's backlog never hides a
/// runnable task belonging to an idle pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    pending_history_prune: Arc<AtomicBool>,
    #[cfg(test)]
    scheduler_notifications: Arc<OnceLock<tokio::sync::watch::Receiver<u64>>>,
    publication_fence: Option<PublicationFence>,
}

/// A controller's lease-backed token must still match at publication, not
/// merely when its local reconciliation began.
#[derive(Clone)]
struct PublicationFence {
    key: String,
    token: String,
}

fn with_publication_fence(
    mut compares: Vec<Compare>,
    fence: Option<&PublicationFence>,
) -> Vec<Compare> {
    if let Some(fence) = fence {
        compares.push(Compare::value(
            fence.key.as_str(),
            CompareOp::Equal,
            fence.token.as_str(),
        ));
    }
    compares
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
    repair_validate_only_targets: Vec<String>,
    wal_claim_targets: Vec<String>,
    rollout: lance_context_merge::rollout::MergeRollout,
    auto_local_merge: bool,
    #[cfg(test)]
    claim_owner_swap: std::sync::Mutex<Option<(String, String)>>,
    #[cfg(test)]
    lose_claim_response: AtomicBool,
}

/// Ownership of one running task. Dropping the claim stops lease renewal; etcd
/// then removes its claim and target-lock keys, allowing recovery.
pub struct TaskClaim {
    pub task: TaskRecord,
    backend: ClaimBackend,
}

impl TaskClaim {
    pub(crate) fn recovering_orphaned_catchup(&self) -> bool {
        self.backend.orphaned_catchup
    }

    pub(crate) fn preparing_maintenance(&self) -> bool {
        self.backend.preparation_key.is_some() && self.backend.target_key.is_none()
    }
}

/// A controller offer is permission to attempt one canonical claim, not table
/// ownership. Its ticket is backed by a permit held by the receiving pool.
#[derive(Clone, Debug)]
pub(crate) struct DispatchAssignment {
    pub task_id: String,
    pub offer_key: String,
    pub offer_value: Vec<u8>,
    pub ticket_key: String,
    pub ticket_value: Vec<u8>,
    pub reservation_key: String,
}

#[derive(Clone, Copy)]
enum ClaimSource<'a> {
    Queue,
    Catchup { task_id: &'a str, job: &'a str },
    Assigned(&'a DispatchAssignment),
}

impl ClaimSource<'_> {
    fn task_id(&self) -> Option<&str> {
        match self {
            Self::Queue => None,
            Self::Catchup { task_id, .. } => Some(task_id),
            Self::Assigned(offer) => Some(&offer.task_id),
        }
    }
    fn catchup_job(&self) -> Option<&str> {
        match self {
            Self::Catchup { job, .. } => Some(job),
            _ => None,
        }
    }
}

struct ClaimBackend {
    token: String,
    lease_id: i64,
    claim_key: String,
    target_key: Option<String>,
    preparation_key: Option<String>,
    preparation_owner: Option<Vec<u8>>,
    orphaned_catchup: bool,
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

impl CoordinationGuard {
    pub(crate) fn key(&self) -> &str {
        &self.key
    }

    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    pub(crate) fn is_live(&self) -> bool {
        !self.keepalive.task.is_finished()
    }
}

struct LeaseKeepalive {
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

/// Private to one claim call. Failed claim transactions have published no keys
/// on this lease; the first successful transaction transfers it to TaskClaim.
struct UnclaimedLease {
    id: i64,
    keepalive: LeaseKeepalive,
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
    /// Existing leased commit-turn request; presence asks an admitted writer
    /// to yield at a completed pass, never grants ownership to another writer.
    pub(crate) async fn maintenance_commit_requested(&self, target: &str) -> lance::Result<bool> {
        let values = self
            .inner
            .read_values(&[self.inner.compaction_commit_key(target)])
            .await?;
        Ok(values[0].is_some())
    }

    #[cfg(test)]
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
            .observed(Operation::TaskOther)
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
            .observed(Operation::TaskClaim)
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
            .observed(Operation::LeaseGrant)
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
            .observed(Operation::TaskOther)
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
            .observed(Operation::TaskRead)
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

    /// Share admission-change notifications across this master's scheduler pools.
    #[cfg(test)]
    pub(crate) fn scheduler_notifications(&self) -> tokio::sync::watch::Receiver<u64> {
        self.scheduler_notifications
            .get_or_init(|| {
                crate::scheduler_wakeup::subscribe(
                    self.inner.client.clone(),
                    self.inner.prefix.clone(),
                )
            })
            .clone()
    }

    /// The underlying etcd client, for other etcd-backed components (the
    /// store registries) that should share one connection.
    pub fn etcd_client(&self) -> &Client {
        &self.inner.client
    }

    pub async fn open(config: &MasterConfig) -> lance::Result<Self> {
        Self::open_mode(config, true).await
    }

    /// Native scheduler replicas leave global recovery/history to the leader.
    pub(crate) async fn open_executor(config: &MasterConfig) -> lance::Result<Self> {
        Self::open_mode(config, false).await
    }

    async fn open_mode(config: &MasterConfig, legacy_maintenance: bool) -> lance::Result<Self> {
        let store = Self {
            inner: Arc::new(EtcdTaskStore::connect(config).await?),
            history_limit: config.task_history_limit.max(1),
            history_ttl_secs: config.task_history_ttl_secs,
            cooldown: CooldownPolicy {
                after_failures: config.task_cooldown_after_failures,
                base: Duration::from_secs(config.task_cooldown_base_secs),
                max: Duration::from_secs(config.task_cooldown_max_secs),
            },
            global_maintenance: legacy_maintenance && config.catchup.target.is_none(),
            pending_history_prune: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            scheduler_notifications: Arc::new(OnceLock::new()),
            publication_fence: None,
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

    pub(crate) async fn enqueue_with_context(
        &self,
        kind: TaskKind,
        target: &str,
        depends_on: Vec<String>,
        context: AdmissionContext,
    ) -> lance::Result<TaskRecord> {
        self.inner
            .enqueue_guarded(kind, target, depends_on, None, None, context)
            .await?
            .ok_or_else(|| lance::Error::io("unguarded enqueue unexpectedly superseded"))
    }

    /// Automatic discovery must not publish after losing its elected owner.
    /// Manual enqueue retains its existing semantics and does not use this API.
    #[cfg(test)]
    pub(crate) async fn enqueue_discovered(
        &self,
        kind: TaskKind,
        target: &str,
    ) -> lance::Result<TaskRecord> {
        self.enqueue_discovered_with_context(
            kind,
            target,
            AdmissionContext::new(Origin::PeriodicStats),
        )
        .await
    }

    pub(crate) async fn enqueue_discovered_with_context(
        &self,
        kind: TaskKind,
        target: &str,
        context: AdmissionContext,
    ) -> lance::Result<TaskRecord> {
        if self.publication_fence.is_none() {
            return Err(lance::Error::io(
                "discovery enqueue requires a leader fence",
            ));
        }
        self.inner
            .enqueue_guarded(
                kind,
                target,
                Vec::new(),
                None,
                self.publication_fence.as_ref(),
                context,
            )
            .await?
            .ok_or_else(|| lance::Error::io("discovery enqueue lost leadership"))
    }

    /// Commit the observed seal frontier only when creating a new durable task.
    /// Dedupe never consumes a frontier: a running pass may have already visited
    /// a writer, and a failed CAS must leave all new seals available on replay.
    pub(crate) async fn enqueue_aggregate_wal(
        &self,
        target: &str,
        frontier_revision: i64,
        frontier_value: &[u8],
    ) -> lance::Result<crate::demand_controller::aggregate::Admission> {
        use crate::demand_controller::aggregate::{frontier_key, Admission};
        let fence = self
            .publication_fence
            .as_ref()
            .ok_or_else(|| lance::Error::io("aggregate admission requires leader fence"))?;
        if frontier_revision < 0 || frontier_value.len() > 256 * 1024 {
            return Err(lance::Error::io("invalid aggregate frontier observation"));
        }
        let context = AdmissionContext::revision(Origin::AggregateFrontier, frontier_revision);
        let inner = &self.inner;
        let checkpoint = frontier_key(&inner.prefix, target);
        let dedupe = inner.dedupe_key(TaskKind::MergeWal, target, &[]).unwrap();
        let task = new_task(TaskKind::MergeWal, target, vec![]);
        let value = encode_task(&task)?;
        let mut frontier: crate::demand_controller::aggregate::Frontier =
            serde_json::from_slice(frontier_value).map_err(|e| lance::Error::io(e.to_string()))?;
        frontier.task = Some(task.id.clone());
        let frontier_value =
            serde_json::to_vec(&frontier).map_err(|e| lance::Error::io(e.to_string()))?;
        if frontier_value.len() > 256 * 1024 {
            return Err(lance::Error::io(
                "oversized aggregate frontier with task identity",
            ));
        }
        let mut client = inner.client.clone();
        let response = client
            .txn(
                Txn::new()
                    .when(with_publication_fence(
                        vec![
                            Compare::mod_revision(
                                checkpoint.as_str(),
                                CompareOp::Equal,
                                frontier_revision,
                            ),
                            Compare::version(dedupe.as_str(), CompareOp::Equal, 0),
                        ],
                        Some(fence),
                    ))
                    .and_then([
                        TxnOp::put(inner.task_key(&task.id), value.clone(), None),
                        TxnOp::put(inner.queue_key(&task.id), value, None),
                        TxnOp::put(dedupe.as_str(), task.id.as_str(), None),
                        TxnOp::put(checkpoint.as_str(), frontier_value.clone(), None),
                    ])
                    .or_else([
                        TxnOp::get(checkpoint.as_str(), None),
                        TxnOp::get(dedupe.as_str(), None),
                    ]),
            )
            .observed(Operation::TaskEnqueue)
            .await
            .map_err(etcd_error("admit aggregate WAL demand"))?;
        if response.succeeded() {
            context.record(TaskKind::MergeWal, target, Some(&task.id), Outcome::Created);
            return Ok(Admission::Admitted {
                revision: response
                    .header()
                    .ok_or_else(|| lance::Error::io("missing WAL admission revision"))?
                    .revision(),
                frontier: frontier_value,
                task: task.id,
            });
        }
        let replies = response.op_responses();
        let [TxnOpResponse::Get(current), TxnOpResponse::Get(existing)] = replies.as_slice() else {
            return Err(lance::Error::io("invalid WAL admission conflict response"));
        };
        if current.kvs().first().map_or(0, |kv| kv.mod_revision()) != frontier_revision {
            context.record(TaskKind::MergeWal, target, None, Outcome::Superseded);
            return Ok(Admission::Changed);
        }
        let Some(existing) = existing.kvs().first() else {
            context.record(TaskKind::MergeWal, target, None, Outcome::Superseded);
            return Ok(Admission::Changed);
        };
        let id =
            std::str::from_utf8(existing.value()).map_err(|e| lance::Error::io(e.to_string()))?;
        let stored = client
            .get(inner.task_key(id), None)
            .observed(Operation::TaskRead)
            .await
            .map_err(etcd_error("resolve aggregate WAL dedupe"))?;
        let mut compares = vec![Compare::value(dedupe.as_str(), CompareOp::Equal, id)];
        if let Some(kv) = stored.kvs().first() {
            let record = decode_task(kv.value(), id)?;
            if record.target != target || record.kind != TaskKind::MergeWal {
                return Err(lance::Error::io("aggregate WAL dedupe target mismatch"));
            }
            if matches!(record.state, TaskState::Queued | TaskState::Running) {
                context.record(TaskKind::MergeWal, target, Some(id), Outcome::Deduplicated);
                return Ok(Admission::Blocked {
                    id: id.to_string(),
                    revision: existing.mod_revision(),
                });
            }
            compares.push(Compare::mod_revision(
                inner.task_key(id),
                CompareOp::Equal,
                kv.mod_revision(),
            ));
        } else {
            compares.push(Compare::version(inner.task_key(id), CompareOp::Equal, 0));
        }
        client
            .txn(
                Txn::new()
                    .when(with_publication_fence(compares, Some(fence)))
                    .and_then([TxnOp::delete(dedupe.as_str(), None)]),
            )
            .observed(Operation::TaskEnqueue)
            .await
            .map_err(etcd_error("clean terminal aggregate WAL dedupe"))?;
        context.record(TaskKind::MergeWal, target, None, Outcome::Superseded);
        Ok(Admission::Changed)
    }

    /// Resolve an ambiguous aggregate publication without waiting for its old
    /// RPC. Rewriting the PREVIOUS frontier at the expected revision invalidates
    /// that RPC's CAS without consuming any unadmitted seal. If it already won,
    /// the comparison fails and returns its actual frontier/dedupe instead.
    pub(crate) async fn resolve_aggregate_admission(
        &self,
        target: &str,
        revision: i64,
        previous: &[u8],
    ) -> lance::Result<crate::demand_controller::aggregate::AdmissionSnapshot> {
        use crate::demand_controller::aggregate::{frontier_key, AdmissionSnapshot};
        let fence = self
            .publication_fence
            .as_ref()
            .ok_or_else(|| lance::Error::io("aggregate resolution requires leader fence"))?;
        if revision < 0 || previous.len() > 256 * 1024 {
            return Err(lance::Error::io("invalid aggregate resolution observation"));
        }
        let checkpoint = frontier_key(&self.inner.prefix, target);
        let dedupe = self
            .inner
            .dedupe_key(TaskKind::MergeWal, target, &[])
            .unwrap();
        let response = self
            .inner
            .client
            .clone()
            .txn(
                Txn::new()
                    .when(with_publication_fence(
                        vec![Compare::mod_revision(
                            checkpoint.as_str(),
                            CompareOp::Equal,
                            revision,
                        )],
                        Some(fence),
                    ))
                    .and_then([
                        TxnOp::put(checkpoint.as_str(), previous, None),
                        TxnOp::get(checkpoint.as_str(), None),
                        TxnOp::get(dedupe.as_str(), None),
                    ])
                    .or_else([
                        TxnOp::get(checkpoint.as_str(), None),
                        TxnOp::get(dedupe.as_str(), None),
                    ]),
            )
            .observed(Operation::TaskEnqueue)
            .await
            .map_err(etcd_error("resolve aggregate admission outcome"))?;
        let observed = response
            .header()
            .ok_or_else(|| lance::Error::io("missing resolution revision"))?
            .revision();
        let mut replies = response.op_responses();
        if response.succeeded() {
            replies.remove(0);
        }
        let [TxnOpResponse::Get(frontier), TxnOpResponse::Get(dedupe)] = replies.as_slice() else {
            return Err(lance::Error::io("invalid aggregate resolution response"));
        };
        let row = |response: &etcd_client::GetResponse| {
            response.kvs().first().map_or((observed, None), |kv| {
                (kv.mod_revision(), Some(kv.value().to_vec()))
            })
        };
        Ok(AdmissionSnapshot {
            frontier: row(frontier),
            dedupe: row(dedupe),
        })
    }

    /// Recovery inventory is a hint, not new demand. Bind a new queue entry to
    /// the observed execution version without adding an ownership read/RPC.
    pub(crate) async fn enqueue_execution_recovery(
        &self,
        kind: TaskKind,
        observed: &lance_context_merge::ExecutionObservation,
    ) -> lance::Result<Option<TaskRecord>> {
        if observed.revision <= 0 {
            return Err(lance::Error::io("recovery requires an execution revision"));
        }
        let target = &observed.execution.target;
        let key = lance_context_merge::execution_key(&self.inner.prefix, target);
        self.inner
            .enqueue_guarded(
                kind,
                target,
                Vec::new(),
                Some((key, observed.revision)),
                self.publication_fence.as_ref(),
                AdmissionContext::revision(Origin::ExecutionRecovery, observed.revision),
            )
            .await
    }

    pub(crate) async fn enqueue_failure_recovery(
        &self,
        kind: TaskKind,
        observed: &lance_context_merge::failure::FailureObservation,
    ) -> lance::Result<Option<TaskRecord>> {
        if observed.revision <= 0 {
            return Err(lance::Error::io("recovery requires a failure revision"));
        }
        let failure = &observed.failure;
        let key = lance_context_merge::failure::failure_key(
            &self.inner.prefix,
            &failure.target,
            &failure.endpoint,
        );
        self.inner
            .enqueue_guarded(
                kind,
                &failure.target,
                Vec::new(),
                Some((key, observed.revision)),
                self.publication_fence.as_ref(),
                AdmissionContext::revision(Origin::FailureRecovery, observed.revision),
            )
            .await
    }

    /// Advisory duplicate suppression for a recovery inventory page. Missing
    /// or terminal tasks still go through the ordinary revision-guarded enqueue.
    /// These reads never create a claim, acknowledge demand, or retire an owner.
    #[cfg(test)]
    pub(crate) async fn recovery_enqueue_needed(
        &self,
        targets: &[(TaskKind, String)],
    ) -> lance::Result<Vec<bool>> {
        let mut needed = vec![true; targets.len()];
        for (batch_index, batch) in targets.chunks(CLAIM_READ_BATCH_TASKS).enumerate() {
            // A singleton gains nothing from speculative reads.
            if batch.len() == 1 {
                continue;
            }
            let keys = batch
                .iter()
                .map(|(kind, target)| {
                    self.inner
                        .dedupe_key(*kind, target, &[])
                        .ok_or_else(|| lance::Error::io("recovery kind does not deduplicate"))
                })
                .collect::<lance::Result<Vec<_>>>()?;
            let ids = self.inner.read_values(&keys).await?;
            let mut active = Vec::new();
            let mut task_keys = Vec::new();
            for (index, value) in ids.into_iter().enumerate() {
                if let Some(value) = value {
                    let Ok(id) = String::from_utf8(value) else {
                        // Leave malformed entries to the ordinary per-target
                        // path; one bad record must not suppress its neighbors.
                        continue;
                    };
                    task_keys.push(self.inner.task_key(&id));
                    active.push((index, id));
                }
            }
            if task_keys.is_empty() {
                continue;
            }
            let tasks = self.inner.read_values(&task_keys).await?;
            for ((index, id), value) in active.into_iter().zip(tasks) {
                if let Some(value) = value {
                    let Ok(task) = decode_task(&value, &id) else {
                        continue;
                    };
                    let (kind, target) = &batch[index];
                    if task.id != id || task.kind != *kind || task.target != *target {
                        continue;
                    }
                    if matches!(task.state, TaskState::Queued | TaskState::Running) {
                        needed[batch_index * CLAIM_READ_BATCH_TASKS + index] = false;
                    }
                }
            }
        }
        Ok(needed)
    }

    /// Admit and acknowledge a writer hint atomically. An old scan page must
    /// not create another merge after a peer has served the same request.
    pub(crate) async fn enqueue_merge_request(
        &self,
        request: &lance_context_merge::rollout::MergeRequest,
    ) -> lance::Result<Option<TaskRecord>> {
        self.inner
            .enqueue_merge_request(request, self.publication_fence.as_ref())
            .await
    }

    /// An informer already has a queued task snapshot. Acknowledge the hint
    /// with one fenced transaction instead of rereading dedupe and task keys.
    /// A changed task/dedupe/request falls back to ordinary admission; a running
    /// pass must never consume a hint for a shard it may already have visited.
    pub(crate) async fn acknowledge_cached_merge_request(
        &self,
        request: &lance_context_merge::rollout::MergeRequest,
        queued_value: &[u8],
    ) -> lance::Result<bool> {
        let queued: TaskRecord = serde_json::from_slice(queued_value)
            .map_err(|e| lance::Error::io(format!("decode queued demand hint: {e}")))?;
        if request.revision <= 0
            || queued.state != TaskState::Queued
            || queued.kind != TaskKind::MergeWal
            || queued.target != request.target
            || !queued.depends_on.is_empty()
        {
            return Ok(false);
        }
        let request_key = lance_context_merge::execution_key(&self.inner.prefix, &request.target)
            .replace("/merge-executions/", "/merge-requests/");
        let dedupe = self
            .inner
            .dedupe_key(TaskKind::MergeWal, &request.target, &[])
            .unwrap();
        let compares = with_publication_fence(
            vec![
                Compare::mod_revision(request_key.as_str(), CompareOp::Equal, request.revision),
                Compare::value(dedupe, CompareOp::Equal, queued.id.as_str()),
                Compare::value(
                    self.inner.task_key(&queued.id),
                    CompareOp::Equal,
                    queued_value,
                ),
                Compare::value(
                    self.inner.queue_key(&queued.id),
                    CompareOp::Equal,
                    queued_value,
                ),
            ],
            self.publication_fence.as_ref(),
        );
        self.inner
            .client
            .clone()
            .txn(
                Txn::new()
                    .when(compares)
                    .and_then([TxnOp::delete(request_key, None)]),
            )
            .observed(Operation::TaskOther)
            .await
            .map(|response| {
                if response.succeeded() {
                    AdmissionContext::revision(Origin::WriterRequest, request.revision).record(
                        TaskKind::MergeWal,
                        &request.target,
                        Some(&queued.id),
                        Outcome::Deduplicated,
                    );
                }
                // A failed cache acknowledgement falls back to normal admission.
                // It is not a separate terminal outcome.
                response.succeeded()
            })
            .map_err(etcd_error("acknowledge cached merge demand"))
    }

    pub async fn list(&self) -> lance::Result<Vec<TaskRecord>> {
        self.inner.list().await
    }

    pub async fn get(&self, id: &str) -> lance::Result<Option<TaskRecord>> {
        self.inner.get(id).await
    }

    pub(crate) async fn explain_admission(
        &self,
        id: &str,
    ) -> lance::Result<Option<AdmissionExplanation>> {
        self.inner.explain_admission(id).await
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
        let (claim, dependency_failed) = self
            .inner
            .claim_next(kinds, ClaimSource::Queue, None, None)
            .await?;
        if dependency_failed {
            if let Err(error) = self.prune_terminal_history().await {
                tracing::warn!(error = %error, "failed to prune task history");
            }
        }
        Ok(claim)
    }

    /// Reserved resident capacity filters before any dependency/ownership RPC.
    /// Keep canonical claims and all external-owner/fencing checks unchanged.
    #[cfg(test)]
    pub(crate) async fn claim_resident_merge(
        &self,
        targets: &[String],
    ) -> lance::Result<Option<TaskClaim>> {
        self.inner.recover_orphaned().await?;
        let (claim, dependency_failed) = self
            .inner
            .claim_next(
                TaskKinds::MERGE_WAL,
                ClaimSource::Queue,
                Some(targets),
                None,
            )
            .await?;
        if dependency_failed {
            if let Err(error) = self.prune_terminal_history().await {
                tracing::warn!(error = %error, "failed to prune task history");
            }
        }
        Ok(claim)
    }

    /// Yield between bounded queue/recovery slices without cancelling any
    /// ownership RPC. The caller keeps this cursor for its own execution pool.
    #[cfg(test)]
    pub(crate) async fn claim_scheduler_slice(
        &self,
        kinds: TaskKinds,
        resident_targets: Option<&[String]>,
        cursor: &mut SchedulerClaimCursor,
        operation: &crate::admission::Operation,
    ) -> lance::Result<Option<TaskClaim>> {
        self.inner
            .recover_scheduler_slice(&mut cursor.recovery, operation)
            .await?;
        let mut scan = SchedulerClaimScan {
            cursor: &mut cursor.queue,
            operation,
            candidates: 0,
            pages: 0,
        };
        let (claim, dependency_failed) = self
            .inner
            .claim_next(kinds, ClaimSource::Queue, resident_targets, Some(&mut scan))
            .await?;
        if let Some(claim) = &claim {
            // Publish accepted ownership before any terminal-history work.
            operation.task(&claim.task.id, &claim.task.target);
        }
        if dependency_failed && self.global_maintenance {
            // Do not keep an accepted claim / local slot idle behind a fleet
            // history scan. An independently admitted master loop performs it.
            self.pending_history_prune.store(true, Ordering::Release);
        }
        operation.claim_phase(if claim.is_some() {
            "claimed"
        } else {
            "scan_complete"
        });
        Ok(claim)
    }

    #[cfg(test)]
    pub(crate) fn lose_next_claim_response_for_test(&self, owner_swap: Option<(String, String)>) {
        *self.inner.claim_owner_swap.lock().unwrap() = owner_swap;
        self.inner
            .lose_claim_response
            .store(true, Ordering::Release);
    }

    /// Consume one controller offer through the same claim/fencing transaction
    /// as ordinary scheduling. The receiver has already reserved local capacity.
    pub(crate) async fn claim_assignment(
        &self,
        offer: &DispatchAssignment,
        kinds: TaskKinds,
        resident_targets: Option<&[String]>,
        operation: &crate::admission::Operation,
    ) -> lance::Result<Option<TaskClaim>> {
        operation.claim_phase("assigned_task");
        let (claim, dependency_failed) = self
            .inner
            .claim_next(kinds, ClaimSource::Assigned(offer), resident_targets, None)
            .await?;
        if let Some(claim) = &claim {
            operation.task(&claim.task.id, &claim.task.target);
        }
        if dependency_failed && self.global_maintenance {
            self.pending_history_prune.store(true, Ordering::Release);
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
                self.inner.recover_tasks(&[task], None).await?;
            }
        }
        Ok(self
            .inner
            .claim_next(
                TaskKinds::MERGE_WAL,
                ClaimSource::Catchup { task_id: &id, job },
                None,
                None,
            )
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
    /// Targets with a `kind` task queued and targets with one running,
    /// separately, read in pages. For the shadow planner's comparison
    /// against reality: "queued" is discovery, "running" is execution, and
    /// the two must not be conflated.
    pub async fn list_queued_and_running_targets(
        &self,
        kind: TaskKind,
    ) -> lance::Result<(Vec<String>, Vec<String>)> {
        let mut queued = Vec::new();
        let mut running = Vec::new();
        for (prefix, out) in [
            (self.inner.queue_prefix(), &mut queued),
            (self.inner.running_prefix(), &mut running),
        ] {
            let (kvs, _) = crate::planner::read_prefix_paged(&self.inner.client, &prefix)
                .await
                .map_err(lance::Error::io)?;
            for kv in kvs {
                if let Ok(task) = decode_task(&kv.value, &String::from_utf8_lossy(&kv.key)) {
                    if task.kind == kind {
                        out.push(task.target);
                    }
                }
            }
            out.sort();
            out.dedup();
        }
        Ok((queued, running))
    }

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

    pub(crate) fn coordination_key(&self, name: &str) -> String {
        format!(
            "{}/coordination/{}",
            self.inner.prefix,
            encode_segment(name)
        )
    }

    /// Register immutable discovery policy once per process lease. Followers
    /// watch the leader key; they do not grant a new lease on every election
    /// attempt or periodically rewrite the same capability record.
    pub(crate) async fn discovery_session(
        &self,
        policy: Vec<u8>,
    ) -> lance::Result<CoordinationGuard> {
        let lease_id = self.inner.grant_lease().await?;
        let keepalive = self.inner.start_keepalive(lease_id).await?;
        let token = generate_id();
        self.inner
            .client
            .clone()
            .put(
                format!("{}/demand-controller/members/{token}", self.inner.prefix),
                policy,
                Some(PutOptions::new().with_lease(lease_id)),
            )
            .observed(Operation::DispatchWrite)
            .await
            .map_err(etcd_error("register discovery policy"))?;
        Ok(CoordinationGuard {
            token,
            lease_id,
            key: format!("{}/demand-controller/leader-v2", self.inner.prefix),
            keepalive,
        })
    }

    /// A ticket exists only while its execution pool holds a real slot. Its
    /// lease also bounds an offer that was published but never claimed.
    pub(crate) async fn create_dispatch_ticket(
        &self,
        policy: Vec<u8>,
    ) -> lance::Result<CoordinationGuard> {
        let token = String::from_utf8(policy).map_err(|e| lance::Error::io(e.to_string()))?;
        let lease_id = self.inner.grant_lease().await?;
        let keepalive = self.inner.start_keepalive(lease_id).await?;
        let key = format!(
            "{}/demand-controller/tickets/{}",
            self.inner.prefix,
            generate_id()
        );
        self.inner
            .client
            .clone()
            .put(
                key.as_str(),
                token.as_str(),
                Some(PutOptions::new().with_lease(lease_id)),
            )
            .observed(Operation::DispatchWrite)
            .await
            .map_err(etcd_error("register dispatch capacity"))?;
        Ok(CoordinationGuard {
            token,
            lease_id,
            key,
            keepalive,
        })
    }

    /// Re-advertise a reserved local slot after its previous offer was consumed.
    /// Reuses the pool's lease; running tasks have their own independent leases.
    pub(crate) async fn rearm_dispatch_ticket(
        &self,
        guard: &mut CoordinationGuard,
        policy: Vec<u8>,
    ) -> lance::Result<i64> {
        if !guard.is_live() {
            return Err(lance::Error::io("dispatch session lease ended"));
        }
        let token = String::from_utf8(policy).map_err(|e| lance::Error::io(e.to_string()))?;
        let response = self
            .inner
            .client
            .clone()
            .txn(
                Txn::new()
                    .when([
                        Compare::version(guard.key.as_str(), CompareOp::Equal, 0),
                        Compare::version(
                            guard.key.replace("/tickets/", "/offers/"),
                            CompareOp::Equal,
                            0,
                        ),
                    ])
                    .and_then([TxnOp::put(
                        guard.key.as_str(),
                        token.as_str(),
                        Some(PutOptions::new().with_lease(guard.lease_id)),
                    )]),
            )
            .observed(Operation::DispatchWrite)
            .await
            .map_err(etcd_error("rearm dispatch capacity"))?;
        if !response.succeeded() {
            return Err(lance::Error::io(
                "previous dispatch capacity still advertised",
            ));
        }
        guard.token = token;
        response
            .header()
            .map(|header| header.revision())
            .ok_or_else(|| lance::Error::io("missing dispatch capacity revision"))
    }

    pub(crate) async fn publish_assignment(
        &self,
        offer: &DispatchAssignment,
        ticket_lease: i64,
        queue_revision: i64,
    ) -> lance::Result<Option<i64>> {
        if self.publication_fence.is_none() || ticket_lease <= 0 || queue_revision <= 0 {
            return Err(lance::Error::io(
                "assignment requires a leader fence and live observations",
            ));
        }
        let compares = with_publication_fence(
            vec![
                Compare::value(
                    offer.ticket_key.as_str(),
                    CompareOp::Equal,
                    offer.ticket_value.clone(),
                ),
                Compare::lease(offer.ticket_key.as_str(), CompareOp::Equal, ticket_lease),
                Compare::version(offer.offer_key.as_str(), CompareOp::Equal, 0),
                Compare::version(offer.reservation_key.as_str(), CompareOp::Equal, 0),
                Compare::mod_revision(
                    self.inner.queue_key(&offer.task_id),
                    CompareOp::Equal,
                    queue_revision,
                ),
            ],
            self.publication_fence.as_ref(),
        );
        let options = Some(PutOptions::new().with_lease(ticket_lease));
        let response = self
            .inner
            .client
            .clone()
            .txn(Txn::new().when(compares).and_then([
                TxnOp::put(
                    offer.offer_key.as_str(),
                    offer.offer_value.clone(),
                    options.clone(),
                ),
                TxnOp::put(
                    offer.reservation_key.as_str(),
                    offer.offer_value.clone(),
                    options,
                ),
            ]))
            .observed(Operation::DispatchWrite)
            .await
            .map_err(etcd_error("offer task to reserved executor capacity"))?;
        Ok(response
            .succeeded()
            .then(|| response.header().map_or(0, |h| h.revision())))
    }

    pub(crate) async fn dispatch_dependency_observations(
        &self,
        ids: &[String],
        revision: i64,
    ) -> lance::Result<(Vec<Option<TaskState>>, i64)> {
        if ids.len() > 64 || ids.is_empty() {
            return Err(lance::Error::io("invalid dependency observation batch"));
        }
        let response = self
            .inner
            .client
            .clone()
            .txn(
                Txn::new().and_then(
                    ids.iter()
                        .map(|id| {
                            TxnOp::get(
                                self.inner.task_key(id),
                                (revision > 0).then(|| GetOptions::new().with_revision(revision)),
                            )
                        })
                        .collect::<Vec<_>>(),
                ),
            )
            .observed(Operation::TaskReadTxn)
            .await
            .map_err(etcd_error("observe dispatch dependencies"))?;
        let snapshot = if revision > 0 {
            revision
        } else {
            response
                .header()
                .ok_or_else(|| lance::Error::io("missing dependency snapshot revision"))?
                .revision()
        };
        let responses = response.op_responses();
        if responses.len() != ids.len() {
            return Err(lance::Error::io("incomplete dependency observations"));
        }
        let states = responses
            .into_iter()
            .zip(ids)
            .map(|(op, id)| match op {
                TxnOpResponse::Get(value) => match value.kvs().first() {
                    Some(kv) => {
                        let task = decode_task(kv.value(), id)?;
                        if task.id != *id {
                            return Err(lance::Error::io("dependency identity mismatch"));
                        }
                        Ok(Some(task.state))
                    }
                    None => Ok(None),
                },
                _ => Err(lance::Error::io("invalid dependency observation response")),
            })
            .collect::<lance::Result<Vec<_>>>()?;
        Ok((states, snapshot))
    }

    pub(crate) async fn recover_observed_task(&self, task: &TaskRecord) -> lance::Result<bool> {
        if self.publication_fence.is_none() {
            return Err(lance::Error::io(
                "controller recovery requires leader fence",
            ));
        }
        self.inner
            .recover_task(task, self.publication_fence.as_ref())
            .await
    }

    pub(crate) async fn elect_discovery_leader(
        &self,
        session: &CoordinationGuard,
    ) -> lance::Result<bool> {
        let response = self
            .inner
            .client
            .clone()
            .txn(
                Txn::new()
                    .when([Compare::version(session.key.as_str(), CompareOp::Equal, 0)])
                    .and_then([TxnOp::put(
                        session.key.as_str(),
                        session.token.as_str(),
                        Some(PutOptions::new().with_lease(session.lease_id)),
                    )]),
            )
            .observed(Operation::DispatchWrite)
            .await
            .map_err(etcd_error("elect discovery leader"))?;
        Ok(response.succeeded())
    }

    pub(crate) fn discovery_publisher(&self, session: &CoordinationGuard) -> Self {
        let mut publisher = self.clone();
        publisher.publication_fence = Some(PublicationFence {
            key: session.key.clone(),
            token: session.token.clone(),
        });
        publisher
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

    /// A compact metrics summary, outside scheduler watch ranges. One write
    /// per completed scan, conditional on the original stats writer's token.
    pub(crate) async fn publish_backlog_snapshot(
        &self,
        guard: &CoordinationGuard,
        snapshot: &crate::backlog_metrics::Snapshot,
    ) -> lance::Result<()> {
        let expected = format!(
            "{}/coordination/{}",
            self.inner.prefix,
            encode_segment("stats-writer")
        );
        if guard.key != expected || !snapshot.validate() {
            return Err(lance::Error::io(
                "invalid backlog snapshot or stats-writer guard",
            ));
        }
        let value = serde_json::to_vec(snapshot).map_err(|e| lance::Error::io(e.to_string()))?;
        if value.len() > crate::backlog_metrics::MAX_SNAPSHOT_BYTES {
            return Err(lance::Error::io(
                "backlog snapshot exceeds fixed size limit",
            ));
        }
        let result = self
            .inner
            .client
            .clone()
            .txn(
                Txn::new()
                    .when([Compare::value(
                        guard.key.clone(),
                        CompareOp::Equal,
                        guard.token.clone(),
                    )])
                    .and_then([TxnOp::put(
                        format!("{}/observability/wal-backlog", self.inner.prefix),
                        value,
                        None,
                    )]),
            )
            .observed(Operation::TaskOther)
            .await
            .map_err(etcd_error("publish backlog metrics"))?;
        if !result.succeeded() {
            return Err(lance::Error::io(
                "stats writer ownership changed before metric publication",
            ));
        }
        Ok(())
    }

    /// Read exactly one bounded summary key. No queue or table enumeration.
    pub(crate) async fn read_backlog_snapshot(
        &self,
    ) -> lance::Result<Option<(crate::backlog_metrics::Snapshot, i64)>> {
        let result = self
            .inner
            .client
            .clone()
            .get(
                format!("{}/observability/wal-backlog", self.inner.prefix),
                None,
            )
            .observed(Operation::TaskRead)
            .await
            .map_err(etcd_error("read backlog metrics"))?;
        let Some(kv) = result.kvs().first() else {
            return Ok(None);
        };
        if kv.value().len() > crate::backlog_metrics::MAX_SNAPSHOT_BYTES {
            return Err(lance::Error::io("backlog metrics exceed fixed size limit"));
        }
        let snapshot = serde_json::from_slice(kv.value())
            .map_err(|e| lance::Error::io(format!("invalid backlog metrics: {e}")))?;
        Ok(Some((snapshot, kv.mod_revision())))
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

    #[cfg(test)]
    pub(crate) fn history_prune_pending(&self) -> bool {
        self.pending_history_prune.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) async fn prune_pending_history(&self) -> lance::Result<usize> {
        if !self.pending_history_prune.swap(false, Ordering::AcqRel) {
            return Ok(0);
        }
        match self.prune_terminal_history().await {
            Ok(n) => Ok(n),
            Err(error) => {
                self.pending_history_prune.store(true, Ordering::Release);
                Err(error)
            }
        }
    }

    pub(crate) async fn prune_controller_history(&self) -> lance::Result<usize> {
        if self.publication_fence.is_none() {
            return Err(lance::Error::io(
                "history maintenance requires leader fence",
            ));
        }
        let mut tasks = Vec::new();
        let mut start = self.inner.tasks_prefix().into_bytes();
        let end = prefix_range_end(&start);
        let mut revision = 0;
        let mut bytes = 0;
        loop {
            let options = GetOptions::new().with_range(end.clone()).with_limit(128);
            let options = if revision > 0 {
                options.with_revision(revision)
            } else {
                options
            };
            let response = self
                .inner
                .client
                .clone()
                .get(start, Some(options))
                .observed(Operation::TaskScan)
                .await
                .map_err(etcd_error("page controller task history"))?;
            if revision == 0 {
                revision = response
                    .header()
                    .ok_or_else(|| lance::Error::io("missing task history revision"))?
                    .revision();
            }
            for kv in response.kvs() {
                bytes += kv.key().len() + kv.value().len();
                if tasks.len() >= 100_000 || bytes > 64 * 1024 * 1024 {
                    return Err(lance::Error::io(
                        "task history maintenance capacity exceeded",
                    ));
                }
                tasks.push(decode_task(kv.value(), &String::from_utf8_lossy(kv.key()))?);
            }
            if !response.more() {
                break;
            }
            start = response
                .kvs()
                .last()
                .ok_or_else(|| lance::Error::io("empty task history continuation"))?
                .key()
                .to_vec();
            start.push(0);
        }
        let cutoff =
            (self.history_ttl_secs > 0).then(|| now_ms() - self.history_ttl_secs as i64 * 1_000);
        let ids = prunable_terminal_ids(tasks, self.history_limit, cutoff);
        self.inner
            .delete_many(&ids, self.publication_fence.as_ref())
            .await?;
        Ok(ids.len())
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
        self.inner.delete_many(&ids, None).await?;
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
            #[cfg(test)]
            claim_owner_swap: Default::default(),
            #[cfg(test)]
            lose_claim_response: AtomicBool::new(false),
            prefix: config.etcd.prefix().to_string(),
            lease_ttl: config.etcd_lease_ttl_secs,
            prepare_targets: config.maintenance.compaction_prepare_targets.clone(),
            index_prepare_targets: config.maintenance.index_prepare_targets.clone(),
            maintenance_catchup_targets: config.maintenance.maintenance_catchup_targets.clone(),
            repair_validate_only_targets: config.maintenance.repair_validate_only_targets.clone(),
            wal_claim_targets: config.maintenance.wal_claim_targets.clone(),
            rollout: config.merge_rollout.clone(),
            auto_local_merge: config.local_merge.master_local_merge_auto,
        })
    }

    async fn enqueue(
        &self,
        kind: TaskKind,
        target: &str,
        depends_on: Vec<String>,
    ) -> lance::Result<TaskRecord> {
        self.enqueue_guarded(
            kind,
            target,
            depends_on,
            None,
            None,
            AdmissionContext::new(Origin::Explicit),
        )
        .await?
        .ok_or_else(|| lance::Error::io("unguarded enqueue unexpectedly superseded"))
    }

    async fn delete_owned_key_fenced(
        &self,
        key: &str,
        value: &str,
        fence: Option<&PublicationFence>,
    ) -> lance::Result<()> {
        self.client
            .clone()
            .txn(
                Txn::new()
                    .when(with_publication_fence(
                        vec![Compare::value(key, CompareOp::Equal, value)],
                        fence,
                    ))
                    .and_then([TxnOp::delete(key, None)]),
            )
            .observed(Operation::TaskOther)
            .await
            .map_err(etcd_error("clean controller dedupe"))?;
        Ok(())
    }

    async fn enqueue_guarded(
        &self,
        kind: TaskKind,
        target: &str,
        depends_on: Vec<String>,
        execution: Option<(String, i64)>,
        fence: Option<&PublicationFence>,
        context: AdmissionContext,
    ) -> lance::Result<Option<TaskRecord>> {
        let task = new_task(kind, target, depends_on);
        let task_key = self.task_key(&task.id);
        let queue_key = self.queue_key(&task.id);
        let value = encode_task(&task)?;
        let mut client = self.client.clone();
        if let Some(dedupe_key) = self.dedupe_key(kind, target, &task.depends_on) {
            for _ in 0..4 {
                // Recovery scanners mostly rediscover an already-active task.
                // Resolve those with reads instead of first proposing a write
                // transaction that is guaranteed to lose the dedupe compare.
                let existing_id = if let Some((key, revision)) = &execution {
                    // Replace the usual dedupe GET with one read-only Txn;
                    // stale inventory does not propose a guaranteed-losing write.
                    let snapshot = client
                        .txn(
                            Txn::new()
                                .when(with_publication_fence(
                                    vec![Compare::mod_revision(
                                        key.as_str(),
                                        CompareOp::Equal,
                                        *revision,
                                    )],
                                    fence,
                                ))
                                .and_then([TxnOp::get(dedupe_key.as_str(), None)]),
                        )
                        .observed(Operation::TaskEnqueue)
                        .await
                        .map_err(etcd_error("read recovery enqueue observation"))?;
                    if !snapshot.succeeded() {
                        context.record(kind, target, None, Outcome::Superseded);
                        return Ok(None);
                    }
                    let responses = snapshot.op_responses();
                    let [TxnOpResponse::Get(dedupe)] = responses.as_slice() else {
                        return Err(lance::Error::io("missing recovery dedupe snapshot"));
                    };
                    dedupe
                        .kvs()
                        .first()
                        .map(|kv| String::from_utf8(kv.value().to_vec()))
                        .transpose()
                        .map_err(|e| lance::Error::io(e.to_string()))?
                } else {
                    self.get_text(&dedupe_key).await?
                };
                if let Some(existing_id) = existing_id {
                    if let Some(existing) = self.get(&existing_id).await? {
                        if matches!(existing.state, TaskState::Queued | TaskState::Running) {
                            context.record(kind, target, Some(&existing.id), Outcome::Deduplicated);
                            return Ok(Some(existing));
                        }
                    }
                    self.delete_owned_key_fenced(&dedupe_key, &existing_id, fence)
                        .await?;
                }
                let mut compares = vec![Compare::version(dedupe_key.as_str(), CompareOp::Equal, 0)];
                let mut otherwise = Vec::new();
                if let Some((key, revision)) = &execution {
                    compares.push(Compare::mod_revision(
                        key.as_str(),
                        CompareOp::Equal,
                        *revision,
                    ));
                    // Resolve a stale observation from the failed CAS itself;
                    // do not add a second query or retry an obsolete page.
                    otherwise.push(TxnOp::get(key.as_str(), None));
                }
                let txn = Txn::new()
                    .when(with_publication_fence(compares, fence))
                    .and_then([
                        TxnOp::put(task_key.as_str(), value.clone(), None),
                        TxnOp::put(queue_key.as_str(), value.clone(), None),
                        TxnOp::put(dedupe_key.as_str(), task.id.as_bytes(), None),
                    ])
                    .or_else(otherwise);
                let response = client
                    .txn(txn)
                    .observed(Operation::TaskEnqueue)
                    .await
                    .map_err(etcd_error("enqueue task"))?;
                if response.succeeded() {
                    context.record(kind, target, Some(&task.id), Outcome::Created);
                    return Ok(Some(task));
                }
                if let Some((_, revision)) = &execution {
                    let responses = response.op_responses();
                    let [TxnOpResponse::Get(current)] = responses.as_slice() else {
                        return Err(lance::Error::io(
                            "missing recovery enqueue conflict snapshot",
                        ));
                    };
                    if current.kvs().first().map(|kv| kv.mod_revision()) != Some(*revision) {
                        context.record(kind, target, None, Outcome::Superseded);
                        return Ok(None);
                    }
                }
            }
            return Err(lance::Error::io(
                "failed to resolve concurrent etcd task enqueue",
            ));
        }
        // Recovery tasks are always depless and all four kinds deduplicate.
        // Ordinary explicit dependency chains retain their existing behavior.
        assert!(execution.is_none());
        client
            .txn(Txn::new().and_then([
                TxnOp::put(task_key, value.clone(), None),
                TxnOp::put(queue_key, value, None),
            ]))
            .observed(Operation::TaskEnqueue)
            .await
            .map_err(etcd_error("enqueue task"))?;
        context.record(kind, target, Some(&task.id), Outcome::Created);
        Ok(Some(task))
    }

    async fn enqueue_merge_request(
        &self,
        request: &lance_context_merge::rollout::MergeRequest,
        fence: Option<&PublicationFence>,
    ) -> lance::Result<Option<TaskRecord>> {
        if request.revision <= 0 {
            return Err(lance::Error::io(
                "merge demand requires an existing request revision",
            ));
        }
        let target = &request.target;
        let context = AdmissionContext::revision(Origin::WriterRequest, request.revision);
        let request_key = lance_context_merge::execution_key(&self.prefix, target)
            .replace("/merge-executions/", "/merge-requests/");
        let dedupe_key = self.dedupe_key(TaskKind::MergeWal, target, &[]).unwrap();
        let task = new_task(TaskKind::MergeWal, target, vec![]);
        let value = encode_task(&task)?;
        let request_matches =
            || Compare::mod_revision(request_key.as_str(), CompareOp::Equal, request.revision);
        let mut client = self.client.clone();
        for _ in 0..4 {
            // This transaction is read-only on both branches. Stale hints and
            // running-task dedupe therefore need no replicated write proposal.
            let snapshot = client
                .txn(
                    Txn::new()
                        .when(with_publication_fence(vec![request_matches()], fence))
                        .and_then([TxnOp::get(dedupe_key.as_str(), None)]),
                )
                .observed(Operation::TaskEnqueue)
                .await
                .map_err(etcd_error("read merge demand admission"))?;
            if !snapshot.succeeded() {
                context.record(TaskKind::MergeWal, target, None, Outcome::Superseded);
                return Ok(None);
            }
            let existing_id = match snapshot.op_responses().into_iter().next() {
                Some(TxnOpResponse::Get(response)) => response
                    .kvs()
                    .first()
                    .map(|kv| String::from_utf8(kv.value().to_vec()))
                    .transpose()
                    .map_err(|e| lance::Error::io(e.to_string()))?,
                _ => return Err(lance::Error::io("missing merge demand dedupe snapshot")),
            };
            if let Some(existing_id) = existing_id {
                let record = client
                    .get(self.task_key(&existing_id), None)
                    .observed(Operation::TaskRead)
                    .await
                    .map_err(etcd_error("read merge demand task"))?;
                if let Some(kv) = record.kvs().first() {
                    let existing = decode_task(kv.value(), &existing_id)?;
                    if existing.state == TaskState::Running {
                        // This pass may already have visited the requesting
                        // shard. Keep the demand for its durable continuation.
                        context.record(
                            TaskKind::MergeWal,
                            target,
                            Some(&existing.id),
                            Outcome::Deduplicated,
                        );
                        return Ok(Some(existing));
                    }
                    if existing.state == TaskState::Queued {
                        let acknowledged = client
                            .txn(
                                Txn::new()
                                    .when(with_publication_fence(
                                        vec![
                                            request_matches(),
                                            Compare::value(
                                                dedupe_key.as_str(),
                                                CompareOp::Equal,
                                                existing_id.as_str(),
                                            ),
                                            Compare::mod_revision(
                                                self.task_key(&existing_id),
                                                CompareOp::Equal,
                                                kv.mod_revision(),
                                            ),
                                        ],
                                        fence,
                                    ))
                                    .and_then([TxnOp::delete(request_key.as_str(), None)]),
                            )
                            .observed(Operation::TaskEnqueue)
                            .await
                            .map_err(etcd_error("acknowledge queued merge demand"))?;
                        if acknowledged.succeeded() {
                            context.record(
                                TaskKind::MergeWal,
                                target,
                                Some(&existing.id),
                                Outcome::Deduplicated,
                            );
                            return Ok(Some(existing));
                        }
                        continue;
                    }
                }
                self.delete_owned_key_fenced(&dedupe_key, &existing_id, fence)
                    .await?;
                continue;
            }
            let admitted = client
                .txn(
                    Txn::new()
                        .when(with_publication_fence(
                            vec![
                                request_matches(),
                                Compare::version(dedupe_key.as_str(), CompareOp::Equal, 0),
                            ],
                            fence,
                        ))
                        .and_then([
                            TxnOp::put(self.task_key(&task.id), value.clone(), None),
                            TxnOp::put(self.queue_key(&task.id), value.clone(), None),
                            TxnOp::put(dedupe_key.as_str(), task.id.as_str(), None),
                            TxnOp::delete(request_key.as_str(), None),
                        ]),
                )
                .observed(Operation::TaskEnqueue)
                .await
                .map_err(etcd_error("admit merge demand"))?;
            if admitted.succeeded() {
                context.record(TaskKind::MergeWal, target, Some(&task.id), Outcome::Created);
                return Ok(Some(task));
            }
        }
        Err(lance::Error::io(
            "merge demand admission changed repeatedly; durable request retained",
        ))
    }

    async fn get(&self, id: &str) -> lance::Result<Option<TaskRecord>> {
        let mut client = self.client.clone();
        let response = client
            .get(self.task_key(id), None)
            .observed(Operation::TaskRead)
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
            .observed(Operation::TaskScan)
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
            .observed(Operation::TaskScan)
            .await
            .map_err(etcd_error("count queued tasks"))?;
        usize::try_from(response.count())
            .map_err(|_| lance::Error::io("etcd returned an invalid queue count"))
    }

    fn eligibility_policy(&self) -> eligibility::Policy<'_> {
        eligibility::Policy {
            auto_local_merge: self.auto_local_merge,
            rollout: &self.rollout,
            compaction_prepare_targets: &self.prepare_targets,
            index_prepare_targets: &self.index_prepare_targets,
            maintenance_catchup_targets: &self.maintenance_catchup_targets,
            repair_validate_only_targets: &self.repair_validate_only_targets,
        }
    }

    fn locally_eligible(
        &self,
        task: &TaskRecord,
        kinds: TaskKinds,
        resident_targets: Option<&[String]>,
    ) -> bool {
        !(task.kind == TaskKind::MergeWal
            && !self.wal_claim_targets.is_empty()
            && !self.wal_claim_targets.contains(&task.target))
            && eligibility::pre_dependency_filter(
                task.kind,
                &task.target,
                |kind| kinds.contains(kind),
                resident_targets,
                &self.eligibility_policy(),
            )
            .is_ok()
    }

    fn admission_keys(&self, task: &TaskRecord) -> [String; 8] {
        let execution = lance_context_merge::execution_key(&self.prefix, &task.target);
        [
            execution.clone(),
            self.target_lock_key(&task.target),
            crate::catchup::store::active_key(&self.prefix, &task.target),
            execution.replace("/merge-executions/", "/merge-claims/"),
            self.claim_key(&task.id),
            format!(
                "{}/compact-preparations/{}",
                self.prefix,
                encode_segment(&task.target)
            ),
            self.compaction_commit_key(&task.target),
            self.queue_key(&task.id),
        ]
    }

    async fn read_claim_candidates(
        &self,
        candidates: &[&TaskRecord],
        operation: Option<&crate::admission::Operation>,
    ) -> lance::Result<HashMap<String, ClaimCandidateRead>> {
        let mut dependency_keys = Vec::new();
        let mut selected = Vec::new();
        let mut total_keys = 0;
        for task in candidates.iter().take(CLAIM_READ_BATCH_TASKS) {
            let needed = 8usize.saturating_add(task.depends_on.len());
            if needed > CLAIM_READ_BATCH_KEYS.saturating_sub(total_keys) {
                break;
            }
            total_keys += needed;
            selected.push(*task);
            dependency_keys.extend(task.depends_on.iter().map(|id| self.task_key(id)));
        }
        if selected.is_empty() {
            // Very long dependency chains retain the existing bounded-per-key
            // path instead of creating an oversized transaction.
            return Ok(HashMap::new());
        }
        let dependencies = if dependency_keys.is_empty() {
            Vec::new()
        } else {
            if let Some(operation) = operation {
                operation.claim_phase("dependencies_batch");
            }
            let values = self.read_values(&dependency_keys).await?;
            if let Some(operation) = operation {
                operation.claim_completed(1, 0, 0);
            }
            values
        };
        let mut dependencies = dependencies.into_iter();
        let mut out = HashMap::new();
        let mut admission_keys = Vec::new();
        let mut ready = Vec::new();
        for task in selected {
            let dependencies: Vec<Option<Vec<u8>>> =
                dependencies.by_ref().take(task.depends_on.len()).collect();
            // Avoid multiplying key reads for dependency-blocked queues. A
            // malformed later dependency is reported only when that candidate
            // is reached, so lookahead cannot poison an earlier runnable task.
            if matches!(
                dependency_status_from_values(task, &dependencies),
                Ok(DependencyStatus::Ready)
            ) {
                admission_keys.extend(self.admission_keys(task));
                ready.push(task.id.clone());
            }
            out.insert(
                task.id.clone(),
                ClaimCandidateRead {
                    admission: Vec::new(),
                    dependencies,
                },
            );
        }
        if !admission_keys.is_empty() {
            if let Some(operation) = operation {
                operation.claim_phase("admission_batch");
            }
            let values = self.read_values(&admission_keys).await?;
            if let Some(operation) = operation {
                operation.claim_completed(1, 0, 0);
            }
            let mut values = values.into_iter();
            for id in ready {
                out.get_mut(&id)
                    .expect("batched candidate was inserted")
                    .admission = values.by_ref().take(8).collect();
            }
        }
        Ok(out)
    }

    async fn claim_next(
        &self,
        kinds: TaskKinds,
        source: ClaimSource<'_>,
        resident_targets: Option<&[String]>,
        scan: Option<&mut SchedulerClaimScan<'_>>,
    ) -> lance::Result<(Option<TaskClaim>, bool)> {
        let mut lease = None;
        let result = self
            .claim_next_with_lease(kinds, source, resident_targets, scan, &mut lease)
            .await;
        if let Some(lease) = lease {
            drop(lease.keepalive);
            let cleanup = self.revoke_lease(lease.id).await;
            // Preserve the original failure, but do not hide a failed cleanup
            // after an otherwise successful empty scan.
            if result.is_ok() {
                cleanup?;
            }
        }
        result
    }

    async fn claim_next_with_lease(
        &self,
        kinds: TaskKinds,
        source: ClaimSource<'_>,
        resident_targets: Option<&[String]>,
        mut scan: Option<&mut SchedulerClaimScan<'_>>,
        lease: &mut Option<UnclaimedLease>,
    ) -> lance::Result<(Option<TaskClaim>, bool)> {
        let prefix = source
            .task_id()
            .map_or_else(|| self.queue_prefix(), |id| self.queue_key(id));
        let range_end = if source.task_id().is_some() {
            let mut end = prefix.as_bytes().to_vec();
            end.push(0);
            end
        } else {
            prefix_range_end(prefix.as_bytes())
        };
        let mut start_key = prefix.into_bytes();
        let mut dependency_failed = false;

        loop {
            if let Some(scan) = scan.as_mut() {
                if scan.pages == SCHEDULER_CLAIM_MAX_PAGES {
                    metrics::counter!("master_task_claim_scan_yields_total", "budget" => "pages")
                        .increment(1);
                    return Ok((None, dependency_failed));
                }
                scan.operation.claim_phase("queue_page");
            }
            let response = if let Some(scan) = scan.as_mut() {
                let response = self
                    .scan_page(
                        &self.queue_prefix(),
                        scan.cursor,
                        TASK_POLL_BATCH,
                        scan.operation,
                    )
                    .await?;
                scan.pages += 1;
                response
            } else {
                self.client
                    .clone()
                    .get(
                        start_key,
                        Some(
                            GetOptions::new()
                                .with_range(range_end.clone())
                                .with_limit(TASK_POLL_BATCH as i64),
                        ),
                    )
                    .observed(Operation::TaskScan)
                    .await
                    .map_err(etcd_error("list queued tasks"))?
            };
            let more = response.more();
            let next_key = response.kvs().last().map(|kv| {
                let mut key = kv.key().to_vec();
                key.push(0);
                key
            });
            let mut queued = response
                .kvs()
                .iter()
                .map(|kv| decode_task(kv.value(), &String::from_utf8_lossy(kv.key())))
                .collect::<lance::Result<VecDeque<_>>>()?;
            let mut candidate_reads = HashMap::new();

            while let Some(mut task) = queued.pop_front() {
                let candidate_key = self.queue_key(&task.id);
                let mut candidate_next = candidate_key.as_bytes().to_vec();
                candidate_next.push(0);
                let last_in_round = !more && next_key.as_ref() == Some(&candidate_next);
                let advance = |scan: &mut Option<&mut SchedulerClaimScan<'_>>| {
                    if let Some(scan) = scan.as_mut() {
                        scan.advance(candidate_key.as_bytes(), last_in_round);
                    }
                };
                // Local routing policy only. Apply before dependency probes or
                // ownership RPCs, including shared and reserved resident pools.
                // Unselected tasks stay available to other masters. The normal
                // claim transaction and execution protocol remain unchanged.
                if task.kind == TaskKind::MergeWal
                    && !self.wal_claim_targets.is_empty()
                    && !self.wal_claim_targets.contains(&task.target)
                {
                    advance(&mut scan);
                    continue;
                }
                // Cheap kind/resident filtering happens before the dependency
                // probe: an unrunnable kind should cost nothing, and this is
                // what lets a scarce kind be found behind a dominant one.
                let policy = self.eligibility_policy();
                if let Err(skip) = eligibility::pre_dependency_filter(
                    task.kind,
                    &task.target,
                    |kind| kinds.contains(kind),
                    resident_targets,
                    &policy,
                ) {
                    tracing::trace!(task = %task.id, target = %task.target, ?skip, "task skipped");
                    advance(&mut scan);
                    continue;
                }
                if let Some(scan) = scan.as_mut() {
                    if scan.candidates == SCHEDULER_CLAIM_MAX_CANDIDATES {
                        metrics::counter!("master_task_claim_scan_yields_total", "budget" => "candidates").increment(1);
                        return Ok((None, dependency_failed));
                    }
                    scan.candidates += 1;
                    scan.operation.claim_phase("dependencies");
                }
                let operation = scan.as_ref().map(|scan| scan.operation);
                if !candidate_reads.contains_key(&task.id) {
                    let remaining = scan.as_ref().map_or(CLAIM_READ_BATCH_TASKS, |scan| {
                        SCHEDULER_CLAIM_MAX_CANDIDATES - scan.candidates + 1
                    });
                    let candidates: Vec<_> = std::iter::once(&task)
                        .chain(queued.iter())
                        .filter(|task| self.locally_eligible(task, kinds, resident_targets))
                        .take(remaining.min(CLAIM_READ_BATCH_TASKS))
                        .collect();
                    candidate_reads = self.read_claim_candidates(&candidates, operation).await?;
                }
                let read = candidate_reads.remove(&task.id);
                let dependency = match &read {
                    Some(read) => dependency_status_from_values(&task, &read.dependencies)?,
                    None => self.dependency_status(&task, operation).await?,
                };
                match dependency {
                    DependencyStatus::Ready => {}
                    DependencyStatus::Waiting => {
                        advance(&mut scan);
                        continue;
                    }
                    DependencyStatus::Failed(dependency) => {
                        candidate_reads.clear();
                        if self.fail_dependency(&task, &dependency).await? {
                            dependency_failed = true;
                        }
                        advance(&mut scan);
                        continue;
                    }
                }

                // Keep the existing wire keys so older masters also yield to
                // prepared index commits. Compact/IndexId share one preparation
                // slot per table; append merges need neither preparation slot.
                let preparation_key = format!(
                    "{}/compact-preparations/{}",
                    self.prefix,
                    encode_segment(&task.target)
                );
                if let Some(operation) = operation {
                    operation.claim_phase("admission_snapshot");
                }
                let snapshot = match read {
                    Some(read) => read.admission,
                    None => {
                        let snapshot = self.read_values(&self.admission_keys(&task)).await?;
                        if let Some(operation) = operation {
                            operation.claim_completed(1, 0, 0);
                        }
                        snapshot
                    }
                };
                // A task from the paged snapshot may already have left the
                // live queue. Check that in the same fresh admission transaction
                // before granting any lease; never replay a historical owner.
                if snapshot[7].is_none() {
                    advance(&mut scan);
                    continue;
                }
                let merge_execution: Option<lance_context_merge::Execution> = snapshot[0]
                    .as_deref()
                    .map(serde_json::from_slice)
                    .transpose()
                    .map_err(|e| lance::Error::io(format!("invalid merge execution: {e}")))?;
                let shape = eligibility::claim_shape(
                    task.kind,
                    &task.target,
                    &policy,
                    snapshot[2].is_some(),
                );
                let preparing = shape.preparing;
                let write_claim = shape.write_claim;
                let preparation_key = preparing.then_some(preparation_key);
                // Avoid lease grant/revoke for a clearly blocked candidate.
                // The final transaction still checks every ownership predicate.
                let expected_owner = merge_execution
                    .as_ref()
                    .map(lance_context_merge::execution_owner);
                let orphaned_catchup = source.catchup_job().is_none()
                    && snapshot[3].is_none()
                    && snapshot[4].is_none()
                    && snapshot[1].as_deref() == expected_owner.as_deref().map(str::as_bytes)
                    && eligibility::orphaned_catchup_candidate(
                        task.kind,
                        &task.target,
                        &policy,
                        merge_execution.as_ref(),
                        snapshot[2].as_deref(),
                    );
                let catchup_record_key =
                    lance_context_merge::execution_key(&self.prefix, &task.target)
                        .replace("/merge-executions/", "/catchup-records/");
                if orphaned_catchup {
                    // Rare recovery candidate only: regular admission keeps its
                    // existing eight-key batch and incurs no additional reads.
                    let record = self
                        .read_values(std::slice::from_ref(&catchup_record_key))
                        .await?;
                    if let Some(operation) = operation {
                        operation.claim_completed(1, 0, 0);
                    }
                    if record[0].is_some() {
                        advance(&mut scan);
                        continue;
                    }
                }
                let expected_job = source.catchup_job().map(str::as_bytes).or(
                    if shape.shared_owner || orphaned_catchup {
                        snapshot[2].as_deref()
                    } else {
                        None
                    },
                );
                let view = eligibility::Snapshot {
                    merge_execution: snapshot[0].as_deref(),
                    target_lock: snapshot[1].as_deref(),
                    catchup_active: snapshot[2].as_deref(),
                    merge_claim: snapshot[3].as_deref(),
                    task_claim: snapshot[4].as_deref(),
                    compact_preparation: snapshot[5].as_deref(),
                    compact_commit_ready: snapshot[6].as_deref(),
                };
                match eligibility::admission_filter(
                    task.kind,
                    shape,
                    &view,
                    expected_owner.as_deref(),
                    merge_execution
                        .as_ref()
                        .is_some_and(|e| e.maintenance.is_none()),
                    expected_job,
                ) {
                    Ok(()) => {}
                    Err(eligibility::Skip::YieldToPreparedCommit) => {
                        metrics::counter!("master_merge_yield_to_compaction_total").increment(1);
                        advance(&mut scan);
                        continue;
                    }
                    Err(eligibility::Skip::OwnershipBlocked) => {
                        metrics::counter!("master_task_admission_blocked_total").increment(1);
                        advance(&mut scan);
                        continue;
                    }
                    Err(skip) => {
                        tracing::trace!(task = %task.id, target = %task.target, ?skip, "task skipped");
                        advance(&mut scan);
                        continue;
                    }
                }
                let token = generate_id();
                // Do not reuse advisory reads after any ownership mutation.
                // Every candidate still passes the original claim/fence CAS.
                candidate_reads.clear();
                if lease.is_none() {
                    if let Some(operation) = operation {
                        operation.claim_phase("lease_grant");
                    }
                    metrics::counter!("master_scheduler_claim_lease_attempts_total").increment(1);
                    let id = self.grant_lease().await?;
                    if let Some(operation) = operation {
                        operation.claim_completed(1, 0, 0);
                        operation.claim_phase("lease_keepalive");
                    }
                    // Establish renewal before publishing ownership, so a slow
                    // handshake cannot strand an already accepted claim.
                    let keepalive = match self.start_keepalive(id).await {
                        Ok(keepalive) => keepalive,
                        Err(error) => {
                            let _ = self.revoke_lease(id).await;
                            return Err(error);
                        }
                    };
                    *lease = Some(UnclaimedLease { id, keepalive });
                } else {
                    metrics::counter!("master_scheduler_claim_lease_reuses_total").increment(1);
                }
                let unclaimed = lease.as_ref().expect("claim lease was initialized");
                if unclaimed.keepalive.task.is_finished() {
                    return Err(lance::Error::io("unclaimed task lease renewal stopped"));
                }
                let lease_id = unclaimed.id;
                if let Some(operation) = operation {
                    operation.claim_phase("claim_transaction");
                }
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
                if orphaned_catchup {
                    compares.extend([
                        Compare::value(
                            lance_context_merge::execution_key(&self.prefix, &task.target),
                            CompareOp::Equal,
                            snapshot[0].clone().expect("recovery execution"),
                        ),
                        Compare::version(catchup_record_key, CompareOp::Equal, 0),
                    ]);
                }
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
                if let ClaimSource::Assigned(offer) = source {
                    compares.extend([
                        Compare::value(
                            offer.offer_key.as_str(),
                            CompareOp::Equal,
                            offer.offer_value.clone(),
                        ),
                        Compare::value(
                            offer.ticket_key.as_str(),
                            CompareOp::Equal,
                            offer.ticket_value.clone(),
                        ),
                        Compare::value(
                            offer.reservation_key.as_str(),
                            CompareOp::Equal,
                            offer.offer_value.clone(),
                        ),
                    ]);
                    operations.extend([
                        TxnOp::delete(offer.offer_key.as_str(), None),
                        TxnOp::delete(offer.ticket_key.as_str(), None),
                        TxnOp::delete(offer.reservation_key.as_str(), None),
                    ]);
                }
                #[cfg(test)]
                {
                    // Deterministic race after snapshot/compare construction.
                    let swap = self.claim_owner_swap.lock().unwrap().take();
                    if let Some((key, replacement)) = swap {
                        self.client
                            .clone()
                            .put(key, replacement, None)
                            .await
                            .map_err(etcd_error("test owner replacement"))?;
                    }
                }
                let mut client = self.client.clone();
                let response = client
                    .txn(Txn::new().when(compares).and_then(operations))
                    .observed(Operation::TaskClaim)
                    .await;
                #[cfg(test)]
                let response = if self.lose_claim_response.swap(false, Ordering::AcqRel) {
                    // The real transaction has completed, but the caller sees
                    // a lost response and must reconcile its original token.
                    Err(etcd_client::Error::Internal(
                        "injected lost claim response".into(),
                    ))
                } else {
                    response
                };
                if let Some(operation) = operation {
                    operation.claim_phase("claim_reconciliation");
                }
                let claimed = self
                    .resolve_claim_response(response, &claim_key, &token)
                    .await?;
                if let Some(operation) = operation {
                    operation.claim_completed(1, 0, 0);
                }
                metrics::counter!("master_scheduler_claim_results_total", "outcome" => if claimed { "accepted" } else { "contended" }).increment(1);
                if claimed {
                    let keepalive = lease.take().expect("accepted claim owns lease").keepalive;
                    advance(&mut scan);
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
                                orphaned_catchup,
                                keepalive,
                            },
                        }),
                        dependency_failed,
                    ));
                }
                // A definitively rejected transaction published no ownership.
                // Keep this private lease for the next candidate in this call;
                // never share it between calls, pools, or accepted tasks.
                advance(&mut scan);
            }

            if !more {
                if let Some(scan) = scan.as_mut() {
                    *scan.cursor = ScanCursor::default();
                }
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
            .observed(Operation::TaskFinish)
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

    /// Bounded server-side pages without sorting/materializing the whole prefix.
    /// Only queue membership uses this snapshot: dependency, lease and ownership
    /// reads plus every claim/recovery CAS always use the live revision.
    async fn scan_page(
        &self,
        prefix: &str,
        cursor: &mut ScanCursor,
        limit: usize,
        operation: &crate::admission::Operation,
    ) -> lance::Result<etcd_client::GetResponse> {
        let start = cursor
            .next
            .clone()
            .unwrap_or_else(|| prefix.as_bytes().to_vec());
        let mut options = GetOptions::new()
            .with_range(prefix_range_end(prefix.as_bytes()))
            .with_limit(limit as i64);
        if let Some(revision) = cursor.revision {
            options = options.with_revision(revision);
        }
        let response = self
            .client
            .clone()
            .get(start, Some(options))
            .observed(Operation::TaskScan)
            .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                if matches!(&error, etcd_client::Error::GRpcStatus(status) if status.message().contains("required revision has been compacted"))
                {
                    // No ownership RPC has started for this page. Restart a
                    // fresh bounded sweep on the next poll, never a hot retry.
                    *cursor = ScanCursor::default();
                    metrics::counter!("master_task_claim_scan_resets_total", "reason" => "compacted").increment(1);
                }
                return Err(etcd_error("read scheduler scan page")(error));
            }
        };
        operation.claim_completed(1, 0, 0);
        if cursor.revision.is_none() {
            cursor.revision = Some(
                response
                    .header()
                    .ok_or_else(|| lance::Error::io("missing scan revision"))?
                    .revision(),
            );
        }
        Ok(response)
    }

    #[cfg(test)]
    async fn recover_scheduler_slice(
        &self,
        cursor: &mut ScanCursor,
        operation: &crate::admission::Operation,
    ) -> lance::Result<()> {
        operation.claim_phase("recovery_page");
        let page = self
            .scan_page(&self.running_prefix(), cursor, 64, operation)
            .await?;
        let tasks = page
            .kvs()
            .iter()
            .map(|kv| decode_task(kv.value(), &String::from_utf8_lossy(kv.key())))
            .collect::<lance::Result<Vec<_>>>()?;
        operation.claim_phase("recovery_ownership");
        self.recover_tasks(&tasks, Some(operation)).await?;
        if let Some(last) = page.kvs().last() {
            cursor.advance(last.key(), !page.more());
        } else {
            *cursor = ScanCursor::default();
        }
        Ok(())
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
                .observed(Operation::TaskScan)
                .await
                .map_err(etcd_error("list running tasks"))?;
            let tasks = page
                .kvs()
                .iter()
                .map(|kv| decode_task(kv.value(), &String::from_utf8_lossy(kv.key())))
                .collect::<lance::Result<Vec<_>>>()?;
            recovered += self.recover_tasks(&tasks, None).await?;
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
            .observed(Operation::TaskReadTxn)
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

    async fn recover_tasks(
        &self,
        tasks: &[TaskRecord],
        operation: Option<&crate::admission::Operation>,
    ) -> lance::Result<usize> {
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
        if let Some(operation) = operation {
            operation.claim_completed(1, 0, 0);
        }
        let mut recovered = 0;
        for (task, claim) in tasks.iter().zip(claims) {
            if claim.is_some() {
                if let Some(operation) = operation {
                    operation.claim_completed(0, 0, 1);
                }
                continue;
            }
            if self.recover_task(task, None).await? {
                recovered += 1;
            }
            if let Some(operation) = operation {
                operation.claim_completed(1, 0, 1);
            }
        }
        metrics::counter!("master_task_recovery_inspected_total").increment(tasks.len() as u64);
        metrics::counter!("master_task_recovery_requeued_total").increment(recovered as u64);
        Ok(recovered)
    }

    async fn recover_task(
        &self,
        task: &TaskRecord,
        fence: Option<&PublicationFence>,
    ) -> lance::Result<bool> {
        let mut queued = task.clone();
        requeue(&mut queued);
        let value = encode_task(&queued)?;
        let txn = Txn::new()
            .when(with_publication_fence(
                vec![
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
                ],
                fence,
            ))
            .and_then([
                TxnOp::put(self.task_key(&task.id), value.clone(), None),
                TxnOp::put(self.queue_key(&task.id), value, None),
                TxnOp::delete(self.running_key(&task.id), None),
            ]);
        Ok(self
            .client
            .clone()
            .txn(txn)
            .observed(Operation::TaskOther)
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
            .observed(Operation::TaskOther)
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
            .observed(Operation::LeaseGrant)
            .await
            .map(|response| response.id())
            .map_err(etcd_error("grant lease"))
    }

    async fn start_keepalive(&self, lease_id: i64) -> lance::Result<LeaseKeepalive> {
        let mut client = self.client.clone();
        let interval = Duration::from_secs((self.lease_ttl / 3).max(1) as u64);
        let ttl = Duration::from_secs(self.lease_ttl as u64);
        let (mut keeper, mut stream) = tokio::time::timeout(
            interval,
            client
                .lease_keep_alive(lease_id)
                .observed(Operation::LeaseStreamOpen),
        )
        .await
        .map_err(|_| lance::Error::io("start lease keepalive timed out"))?
        .map_err(etcd_error("start lease keepalive"))?;
        // etcd-client consumes a positive-TTL response during the handshake.
        let (stop_tx, mut stop_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => return,
                    _ = tokio::time::sleep(interval) => {}
                }
                let response = tokio::time::timeout(interval, async {
                    keeper.keep_alive().await?;
                    stream.message().await
                })
                .await;
                match response {
                    Ok(Ok(Some(response))) => {
                        if response.id() != lease_id || response.ttl() <= 0 {
                            tracing::warn!(lease_id, "task lease is no longer renewable");
                            return;
                        }
                        continue;
                    }
                    failure => tracing::warn!(
                        lease_id,
                        ?failure,
                        "task lease stream interrupted; reconnecting the same lease"
                    ),
                }

                // A missing response is not proof that the server failed to
                // renew. Even beyond our last confirmation, etcd may still
                // hold this lease. Bound recovery from the stream failure,
                // letting etcd decide whether this same lease is renewable.
                // This never grants a lease or recreates ownership keys, and
                // renewal does not replace a task's publication ownership checks.
                let recovery_until = tokio::time::Instant::now() + ttl;
                loop {
                    let reconnect_started = tokio::time::Instant::now();
                    if reconnect_started >= recovery_until {
                        tracing::warn!(
                            lease_id,
                            "task lease stream recovery exhausted its retry window"
                        );
                        return;
                    }
                    let result = tokio::time::timeout_at(
                        recovery_until.min(reconnect_started + interval),
                        client
                            .lease_keep_alive(lease_id)
                            .observed(Operation::LeaseStreamOpen),
                    )
                    .await;
                    if let Ok(Ok((replacement, replacement_stream))) = result {
                        if replacement.id() != lease_id {
                            tracing::warn!(lease_id, "task lease reconnect returned another lease");
                            return;
                        }
                        keeper = replacement;
                        stream = replacement_stream;
                        metrics::counter!("master_task_lease_stream_recoveries_total").increment(1);
                        tracing::info!(
                            lease_id,
                            "task lease stream recovered without changing ownership"
                        );
                        break;
                    }
                    // Bound reconnect traffic during an etcd outage. The lease
                    // identity staggers retries without changing their deadline.
                    let retry_delay =
                        interval / 2 + Duration::from_millis(lease_id.unsigned_abs() % 500);
                    tokio::select! {
                        _ = &mut stop_rx => return,
                        _ = tokio::time::sleep_until(recovery_until.min(tokio::time::Instant::now() + retry_delay)) => {}
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
        match client
            .lease_revoke(lease_id)
            .observed(Operation::LeaseRevoke)
            .await
        {
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
            .observed(Operation::TaskOther)
            .await
            .map(|_| ())
            .map_err(etcd_error("release owned key"))
    }

    async fn get_text(&self, key: &str) -> lance::Result<Option<String>> {
        let mut client = self.client.clone();
        let response = client
            .get(key, None)
            .observed(Operation::TaskRead)
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

    async fn dependency_status(
        &self,
        task: &TaskRecord,
        operation: Option<&crate::admission::Operation>,
    ) -> lance::Result<DependencyStatus> {
        let mut waiting = false;
        for dependencies in task.depends_on.chunks(64) {
            let keys = dependencies
                .iter()
                .map(|id| self.task_key(id))
                .collect::<Vec<_>>();
            let values = self.read_values(&keys).await?;
            if let Some(operation) = operation {
                operation.claim_completed(1, 0, 0);
            }
            for (dependency, value) in dependencies.iter().zip(values) {
                let record = value
                    .as_deref()
                    .map(|value| decode_task(value, dependency))
                    .transpose()?;
                match record {
                    Some(record) if record.state == TaskState::Done => {}
                    Some(record) if record.state == TaskState::Failed => {
                        return Ok(DependencyStatus::Failed(dependency.clone()))
                    }
                    Some(_) => waiting = true,
                    None => return Ok(DependencyStatus::Failed(dependency.clone())),
                }
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
            .observed(Operation::TaskOther)
            .await
            .map(|response| response.succeeded())
            .map_err(etcd_error("fail task with unsuccessful dependency"))
    }

    async fn delete_many(
        &self,
        ids: &[String],
        fence: Option<&PublicationFence>,
    ) -> lance::Result<()> {
        for chunk in ids.chunks(100) {
            let operations = chunk
                .iter()
                .map(|id| TxnOp::delete(self.task_key(id), None))
                .collect::<Vec<_>>();
            let mut client = self.client.clone();
            let response = client
                .txn(
                    Txn::new()
                        .when(with_publication_fence(Vec::new(), fence))
                        .and_then(operations),
                )
                .observed(Operation::TaskOther)
                .await
                .map_err(etcd_error("prune task history"))?;
            if !response.succeeded() {
                return Err(lance::Error::io("task history controller lost leadership"));
            }
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
            .observed(Operation::TaskRead)
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
                .observed(Operation::LeaseGrant)
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
                .observed(Operation::TaskOther)
                .await
                .map_err(etcd_error("put cooldown"))?
                .succeeded();
            if !committed {
                // Someone else bumped it first; re-read and try again.
                let _ = client
                    .lease_revoke(lease)
                    .observed(Operation::LeaseRevoke)
                    .await;
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
            .observed(Operation::TaskOther)
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
            .observed(Operation::TaskScan)
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
            .observed(Operation::LeaseGrant)
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
            .observed(Operation::TaskOther)
            .await
            .map_err(etcd_error("put repair"))?;
        Ok(())
    }

    async fn list_repairs(&self) -> lance::Result<Vec<RepairRecord>> {
        let mut client = self.client.clone();
        let response = client
            .get(self.repairs_prefix(), Some(GetOptions::new().with_prefix()))
            .observed(Operation::TaskScan)
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

struct ClaimCandidateRead {
    admission: Vec<Option<Vec<u8>>>,
    dependencies: Vec<Option<Vec<u8>>>,
}

fn dependency_status_from_values(
    task: &TaskRecord,
    values: &[Option<Vec<u8>>],
) -> lance::Result<DependencyStatus> {
    if task.depends_on.len() != values.len() {
        return Err(lance::Error::io("incomplete dependency snapshot"));
    }
    let mut waiting = false;
    for (id, value) in task.depends_on.iter().zip(values) {
        let record = value.as_deref().map(|v| decode_task(v, id)).transpose()?;
        match record {
            Some(record) if record.state == TaskState::Done => {}
            Some(record) if record.state == TaskState::Failed => {
                return Ok(DependencyStatus::Failed(id.clone()));
            }
            Some(_) => waiting = true,
            None => return Ok(DependencyStatus::Failed(id.clone())),
        }
    }
    Ok(if waiting {
        DependencyStatus::Waiting
    } else {
        DependencyStatus::Ready
    })
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
pub(crate) mod tests {
    use super::*;
    use tempfile::TempDir;

    mod admission_outcomes {
        include!("admission_tests.rs");
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn failure_recovery_observation_preserves_new_budget_and_writer_demand() {
        use lance_context_merge::failure::{failure_key, FailureClass, ShardFailure};
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let c = store.merge_coordinator();
        let mut client = store.inner.client.clone();
        for kind in [
            TaskKind::MergeWal,
            TaskKind::Compact,
            TaskKind::IndexId,
            TaskKind::Repair,
        ] {
            let target = format!("failure-{}", kind_label(kind));
            let key = failure_key(&store.inner.prefix, &target, "worker");
            let mut failure = ShardFailure {
                target: target.clone(),
                endpoint: "worker".into(),
                consecutive_attempts: 1,
                class: FailureClass::Retryable,
                last_error: "transport failure".into(),
                last_failure_ms: 1,
                next_retry_ms: 0,
                needs_attention: false,
            };
            client
                .put(key.clone(), serde_json::to_vec(&failure).unwrap(), None)
                .await
                .unwrap();
            let old = c
                .failure_observation_page(None, 256)
                .await
                .unwrap()
                .0
                .into_iter()
                .find(|row| row.failure.target == target)
                .unwrap();
            failure.consecutive_attempts = 15;
            failure.next_retry_ms = lance_context_merge::failure::now_ms() + 60_000;
            client
                .put(key.clone(), serde_json::to_vec(&failure).unwrap(), None)
                .await
                .unwrap();
            c.request_merge(&target).await.unwrap();
            let request = c.get_merge_request(&target).await.unwrap().unwrap();
            assert!(store
                .enqueue_failure_recovery(kind, &old)
                .await
                .unwrap()
                .is_none());
            assert!(store.get_active_id(kind, &target).await.unwrap().is_none());
            let seen = c.failure(&target, "worker").await.unwrap().unwrap();
            assert_eq!(seen.consecutive_attempts, 15);
            assert_eq!(seen.next_retry_ms, failure.next_retry_ms);
            assert_eq!(
                c.get_merge_request(&target)
                    .await
                    .unwrap()
                    .unwrap()
                    .revision,
                request.revision
            );
            // A current due observation still gets one deduplicated recovery.
            failure.next_retry_ms = 0;
            client
                .put(key.clone(), serde_json::to_vec(&failure).unwrap(), None)
                .await
                .unwrap();
            let fresh = c
                .failure_observation_page(None, 256)
                .await
                .unwrap()
                .0
                .into_iter()
                .find(|row| row.failure.target == target)
                .unwrap();
            let (a, b) = tokio::join!(
                store.enqueue_failure_recovery(kind, &fresh),
                store.enqueue_failure_recovery(kind, &fresh)
            );
            let a = a.unwrap().unwrap();
            assert_eq!(a.id, b.unwrap().unwrap().id);
            client.delete(key, None).await.unwrap();
            assert!(store
                .enqueue_failure_recovery(kind, &fresh)
                .await
                .unwrap()
                .is_none());
            assert_eq!(
                store.get_active_id(kind, &target).await.unwrap(),
                Some(a.id)
            );
            assert_eq!(
                c.get_merge_request(&target)
                    .await
                    .unwrap()
                    .unwrap()
                    .revision,
                request.revision
            );
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn recovery_prefilter_retains_missing_terminal_and_malformed_work() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let mut client = store.inner.client.clone();
        let kinds = [
            TaskKind::MergeWal,
            TaskKind::Compact,
            TaskKind::IndexId,
            TaskKind::Repair,
        ];
        let mut targets = Vec::new();
        let mut tasks = Vec::new();
        for i in 0..11 {
            let kind = kinds[i % kinds.len()];
            let target = format!("prefilter-{i}");
            tasks.push(store.enqueue(kind, &target, vec![]).await.unwrap());
            targets.push((kind, target));
        }
        for (index, state) in [
            (1, TaskState::Running),
            (2, TaskState::Done),
            (3, TaskState::Failed),
        ] {
            tasks[index].state = state;
            client
                .put(
                    store.inner.task_key(&tasks[index].id),
                    encode_task(&tasks[index]).unwrap(),
                    None,
                )
                .await
                .unwrap();
        }
        for index in [4, 9] {
            client
                .delete(store.inner.task_key(&tasks[index].id), None)
                .await
                .unwrap();
        }
        let dedupe = |index: usize| {
            store
                .inner
                .dedupe_key(targets[index].0, &targets[index].1, &[])
                .unwrap()
        };
        client.put(dedupe(5), vec![255], None).await.unwrap();
        client
            .put(store.inner.task_key(&tasks[6].id), "not json", None)
            .await
            .unwrap();
        let mut wrong = tasks[7].clone();
        wrong.target = "another-target".into();
        client
            .put(
                store.inner.task_key(&wrong.id),
                encode_task(&wrong).unwrap(),
                None,
            )
            .await
            .unwrap();
        store
            .merge_coordinator()
            .request_merge(&targets[10].1)
            .await
            .unwrap();
        assert_eq!(
            store.recovery_enqueue_needed(&targets).await.unwrap(),
            vec![false, false, true, true, true, true, true, true, false, true, false]
        );
        assert!(store
            .merge_coordinator()
            .get_merge_request(&targets[10].1)
            .await
            .unwrap()
            .is_some());
        // No speculative reads for a singleton; the existing enqueue handles it.
        assert_eq!(
            store.recovery_enqueue_needed(&targets[..1]).await.unwrap(),
            vec![true]
        );
        assert!(store.recovery_enqueue_needed(&[]).await.unwrap().is_empty());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn recovery_prefilter_revisits_completed_tasks_without_losing_execution() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let coordinator = store.merge_coordinator();
        let mut client = store.inner.client.clone();
        let mut targets = Vec::new();
        let mut tasks = Vec::new();
        for target in ["finishing", "still-active"] {
            let mut execution = lance_context_merge::Execution::new(target, "worker", "old", 60);
            execution.background = true;
            assert!(coordinator.reserve_background(&execution).await.unwrap());
            tasks.push(
                store
                    .enqueue(TaskKind::MergeWal, target, vec![])
                    .await
                    .unwrap(),
            );
            targets.push((TaskKind::MergeWal, target.to_string()));
        }
        let observed = coordinator
            .execution_observation_page(None)
            .await
            .unwrap()
            .0;
        assert_eq!(
            store.recovery_enqueue_needed(&targets).await.unwrap(),
            vec![false, false]
        );
        // Completion just after an advisory read is seen on the next sweep.
        tasks[0].state = TaskState::Done;
        client
            .put(
                store.inner.task_key(&tasks[0].id),
                encode_task(&tasks[0]).unwrap(),
                None,
            )
            .await
            .unwrap();
        coordinator.request_merge("finishing").await.unwrap();
        assert_eq!(
            store.recovery_enqueue_needed(&targets).await.unwrap(),
            vec![true, false]
        );
        let old = observed
            .iter()
            .find(|row| row.execution.target == "finishing")
            .unwrap();
        let next = store
            .enqueue_execution_recovery(TaskKind::MergeWal, old)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(next.id, tasks[0].id);
        assert_eq!(
            coordinator.get("finishing").await.unwrap().unwrap().id,
            old.execution.id
        );
        assert!(coordinator
            .get_merge_request("finishing")
            .await
            .unwrap()
            .is_some());
        assert_eq!(
            store
                .get_active_id(TaskKind::MergeWal, "still-active")
                .await
                .unwrap(),
            Some(tasks[1].id.clone())
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn recovery_enqueue_versions_preserve_dedupe_and_new_writer_demand() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let coordinator = store.merge_coordinator();
        let mut client = store.inner.client.clone();
        for kind in [
            TaskKind::MergeWal,
            TaskKind::Compact,
            TaskKind::IndexId,
            TaskKind::Repair,
        ] {
            let target = format!("recovery-{}", kind_label(kind));
            let execution = lance_context_merge::Execution::new(&target, "worker", "old", 60);
            let key = lance_context_merge::execution_key(&store.inner.prefix, &target);
            // Legacy/default-omitted and unknown fields must not force a
            // reserialization comparison that silently disables recovery.
            let mut value = serde_json::to_value(execution).unwrap();
            value.as_object_mut().unwrap().remove("protocol");
            value["future_metadata"] = serde_json::json!("preserve");
            client
                .put(key.clone(), value.to_string(), None)
                .await
                .unwrap();
            let observed = coordinator
                .execution_observation_page(None)
                .await
                .unwrap()
                .0
                .into_iter()
                .find(|o| o.execution.target == target)
                .unwrap();
            // Identical bytes at a different revision are still a new record.
            client
                .put(key.clone(), value.to_string(), None)
                .await
                .unwrap();
            coordinator.request_merge(&target).await.unwrap();
            let demand = coordinator
                .get_merge_request(&target)
                .await
                .unwrap()
                .unwrap();
            assert!(store
                .enqueue_execution_recovery(kind, &observed)
                .await
                .unwrap()
                .is_none());
            assert!(store.get_active_id(kind, &target).await.unwrap().is_none());
            assert_eq!(
                coordinator
                    .get_merge_request(&target)
                    .await
                    .unwrap()
                    .unwrap()
                    .revision,
                demand.revision
            );
            let fresh = coordinator
                .execution_observation_page(None)
                .await
                .unwrap()
                .0
                .into_iter()
                .find(|o| o.execution.target == target)
                .unwrap();
            let (first, second) = tokio::join!(
                store.enqueue_execution_recovery(kind, &fresh),
                store.enqueue_execution_recovery(kind, &fresh),
            );
            let first = first.unwrap().unwrap();
            assert_eq!(first.id, second.unwrap().unwrap().id);
            assert_eq!(first.kind, kind);
            client.delete(key, None).await.unwrap();
            assert!(store
                .enqueue_execution_recovery(kind, &fresh)
                .await
                .unwrap()
                .is_none());
            assert_eq!(
                store.get_active_id(kind, &target).await.unwrap(),
                Some(first.id)
            );
            assert_eq!(
                coordinator
                    .get_merge_request(&target)
                    .await
                    .unwrap()
                    .unwrap()
                    .revision,
                demand.revision
            );
        }
        assert_eq!(store.list().await.unwrap().len(), 4);
    }

    #[tokio::test]
    #[ignore = "requires exclusive loopback ETCD_TEST_ENDPOINTS and ETCD_TEST_EXCLUSIVE=1"]
    async fn recovery_enqueue_rechecks_execution_after_dedupe_snapshot() {
        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").as_deref(), Ok("1"));
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        assert!(!endpoint.contains(',') && endpoint.starts_with("http://127.0.0.1:"));
        async fn transactions(endpoint: &str) -> u64 {
            reqwest::get(format!("{endpoint}/metrics"))
                .await
                .unwrap()
                .text()
                .await
                .unwrap()
                .lines()
                .filter(|line| {
                    line.starts_with("grpc_server_handled_total{")
                        && line.contains("grpc_method=\"Txn\"")
                        && line.contains("grpc_code=\"OK\"")
                })
                .map(|line| {
                    line.split_whitespace()
                        .last()
                        .unwrap()
                        .parse::<f64>()
                        .unwrap() as u64
                })
                .sum()
        }
        let (_dir, proxy, store, mut observer) = lease_proxy_store().await;
        let coordinator = store.merge_coordinator();
        for replace in [false, true] {
            let target = format!("race-{replace}");
            let key = lance_context_merge::execution_key(&store.inner.prefix, &target);
            let execution = lance_context_merge::Execution::new(&target, "worker", "old", 60);
            observer
                .put(key.clone(), serde_json::to_vec(&execution).unwrap(), None)
                .await
                .unwrap();
            let observed = coordinator
                .execution_observation_page(None)
                .await
                .unwrap()
                .0
                .into_iter()
                .find(|o| o.execution.target == target)
                .unwrap();
            proxy.responses.send(false).unwrap();
            let before = transactions(&endpoint).await;
            let candidate = store.clone();
            let stale = observed.clone();
            let enqueue = tokio::spawn(async move {
                candidate
                    .enqueue_execution_recovery(TaskKind::MergeWal, &stale)
                    .await
            });
            // The read-only dedupe transaction has completed in real etcd,
            // but its successful response has not reached the scheduler yet.
            tokio::time::timeout(Duration::from_secs(5), async {
                while transactions(&endpoint).await == before {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
            assert!(!enqueue.is_finished());
            if replace {
                let replacement = lance_context_merge::Execution::new(&target, "worker", "new", 60);
                observer
                    .put(key.clone(), serde_json::to_vec(&replacement).unwrap(), None)
                    .await
                    .unwrap();
            } else {
                observer.delete(key.clone(), None).await.unwrap();
            }
            proxy.responses.send(true).unwrap();
            assert!(tokio::time::timeout(Duration::from_secs(5), enqueue)
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .is_none());
            assert!(store
                .get_active_id(TaskKind::MergeWal, &target)
                .await
                .unwrap()
                .is_none());
            // Lease cleanup elsewhere can advance the cluster revision even
            // while every obsolete enqueue remains read-only.
            let unrelated_key = format!("/unrelated-recovery-test/{}", generate_id());
            let unrelated_lease = observer.lease_grant(60, None).await.unwrap().id();
            observer
                .put(
                    unrelated_key,
                    "unrelated",
                    Some(PutOptions::new().with_lease(unrelated_lease)),
                )
                .await
                .unwrap();
            // Watch only this fixture's keys, including transient writes that
            // a before/after key snapshot would miss.
            let revision = observer
                .get(key.clone(), None)
                .await
                .unwrap()
                .header()
                .unwrap()
                .revision();
            let mut events = observer
                .watch(
                    format!("{}/", store.inner.prefix),
                    Some(
                        etcd_client::WatchOptions::new()
                            .with_prefix()
                            .with_start_revision(revision + 1),
                    ),
                )
                .await
                .unwrap();
            for _ in 0..20 {
                assert!(store
                    .enqueue_execution_recovery(TaskKind::MergeWal, &observed)
                    .await
                    .unwrap()
                    .is_none());
            }
            observer.lease_revoke(unrelated_lease).await.unwrap();
            // A sentinel after the attempts closes the observation window.
            // Ordered delivery must contain only this event, even though the
            // unrelated lease deletion advanced the global revision.
            let sentinel = format!("{}/no-write-boundary/{replace}", store.inner.prefix);
            let boundary = observer
                .put(sentinel.clone(), "done", None)
                .await
                .unwrap()
                .header()
                .unwrap()
                .revision();
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    let response = events.message().await.unwrap().unwrap();
                    assert!(!response.canceled());
                    if response.events().is_empty() {
                        continue;
                    }
                    assert_eq!(response.events().len(), 1);
                    let event = &response.events()[0];
                    assert_eq!(event.event_type(), etcd_client::EventType::Put);
                    let kv = event.kv().unwrap();
                    assert_eq!(kv.key(), sentinel.as_bytes());
                    assert_eq!(kv.mod_revision(), boundary);
                    break;
                }
            })
            .await
            .unwrap();
            drop(events);
        }
        assert!(store.list().await.unwrap().is_empty());
    }

    pub(crate) struct LeaseTestProxy {
        pub(crate) endpoint: String,
        pub(crate) online: tokio::sync::watch::Sender<bool>,
        responses: tokio::sync::watch::Sender<bool>,
        task: JoinHandle<()>,
    }

    impl LeaseTestProxy {
        pub(crate) async fn start(upstream: &str) -> Self {
            let upstream = upstream.strip_prefix("http://").unwrap().to_owned();
            assert!(upstream.starts_with("127.0.0.1:") || upstream.starts_with("localhost:"));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let (online, mut status) = tokio::sync::watch::channel(true);
            let (responses, response_status) = tokio::sync::watch::channel(true);
            let task = tokio::spawn(async move {
                let mut connections = tokio::task::JoinSet::new();
                loop {
                    tokio::select! {
                        accepted = listener.accept() => {
                            let (mut downstream, _) = accepted.unwrap();
                            if !*status.borrow_and_update() {
                                continue;
                            }
                            let mut status = status.clone();
                            let mut response_status = response_status.clone();
                            let upstream = upstream.clone();
                            connections.spawn(async move {
                                tokio::select! {
                                    _ = status.changed() => {},
                                    _ = async {
                                        if let Ok(mut upstream) = tokio::net::TcpStream::connect(upstream).await {
                                            let (mut client_read, mut client_write) = downstream.split();
                                            let (mut server_read, mut server_write) = upstream.split();
                                            let replies = async {
                                                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                                                let mut buffer = [0; 16 * 1024];
                                                loop {
                                                    let count = server_read.read(&mut buffer).await?;
                                                    if count == 0 {
                                                        return Ok::<(), std::io::Error>(());
                                                    }
                                                    if response_status.wait_for(|enabled| *enabled).await.is_err() {
                                                        return Ok(());
                                                    }
                                                    client_write.write_all(&buffer[..count]).await?;
                                                }
                                            };
                                            tokio::select! {
                                                _ = tokio::io::copy(&mut client_read, &mut server_write) => {},
                                                _ = replies => {},
                                            }
                                        }
                                    } => {},
                                }
                            });
                        }
                        _ = connections.join_next(), if !connections.is_empty() => {}
                    }
                }
            });
            Self {
                endpoint,
                online,
                responses,
                task,
            }
        }
    }

    impl Drop for LeaseTestProxy {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn lease_proxy_store() -> (TempDir, LeaseTestProxy, TaskStore, Client) {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_test_writer()
            .try_init();
        let upstream = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let upstream = upstream.split(',').next().unwrap();
        let proxy = LeaseTestProxy::start(upstream).await;
        let observer = Client::connect([upstream], None).await.unwrap();
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = vec![proxy.endpoint.clone()];
        cfg.etcd.etcd_prefix = format!("/lease-stream-test/{}", generate_id());
        cfg.etcd_lease_ttl_secs = 6;
        let store = TaskStore::open(&cfg).await.unwrap();
        (dir, proxy, store, observer)
    }

    async fn bounded_scheduler_store(targets: Vec<String>) -> (TempDir, TaskStore) {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/bounded-claim-test/{}", generate_id());
        cfg.maintenance.wal_claim_targets = targets;
        let store = TaskStore::open(&cfg).await.unwrap();
        (dir, store)
    }

    #[tokio::test]
    #[ignore = "requires loopback ETCD_TEST_ENDPOINTS"]
    async fn preparation_owner_stream_recovers_transport_without_trusting_old_claim() {
        let (_dir, mut cfg, _) = preparation_test_store().await;
        let upstream = cfg.etcd.etcd_endpoints[0].clone();
        let proxy = LeaseTestProxy::start(&upstream).await;
        let mut direct = Client::connect([upstream], None).await.unwrap();
        cfg.etcd.etcd_endpoints = vec![proxy.endpoint.clone()];
        cfg.etcd_lease_ttl_secs = 30;
        let store = TaskStore::open(&cfg).await.unwrap();
        store
            .enqueue(TaskKind::Compact, "hot", vec![])
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        let mut observer = store.preparation_observer(&claim).unwrap();
        assert!(observer.observe().await.unwrap());
        proxy.online.send(false).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(10), observer.observe())
                .await
                .unwrap()
                .is_err()
        );
        proxy.online.send(true).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(10), observer.observe())
                .await
                .unwrap()
                .unwrap()
        );
        direct
            .delete(claim.backend.preparation_key.clone().unwrap(), None)
            .await
            .unwrap();
        assert!(
            !tokio::time::timeout(Duration::from_secs(2), observer.observe())
                .await
                .unwrap()
                .unwrap()
        );
        store
            .finish(claim, Err("test ownership revoked".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires loopback ETCD_TEST_ENDPOINTS"]
    async fn preparation_watch_recovers_transport_but_rejects_replaced_lease() {
        use lance_context_core::merge_write_scope::{checkpoint, MergeWriteScope};
        let (_dir, mut cfg, _) = preparation_test_store().await;
        let upstream = cfg.etcd.etcd_endpoints[0].clone();
        let proxy = LeaseTestProxy::start(&upstream).await;
        let observer = Client::connect([upstream], None).await.unwrap();
        cfg.etcd.etcd_endpoints = vec![proxy.endpoint.clone()];
        let store = TaskStore::open(&cfg).await.unwrap();
        store
            .enqueue(TaskKind::Compact, "hot", vec![])
            .await
            .unwrap();
        let mut old = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        assert!(old.preparing_maintenance());
        let scope = MergeWriteScope::new();
        let mut calls = 0;
        let watch =
            crate::scheduler::watch_preparation_ownership(&scope, Duration::from_secs(3), || {
                calls += 1;
                let attempt = calls;
                let proxy = &proxy;
                let store = &store;
                let claim = &old;
                let mut observer = observer.clone();
                async move {
                    if attempt == 1 {
                        proxy.online.send(false).unwrap();
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                    let result = store.preparation_owned(claim).await;
                    if attempt == 1 {
                        assert!(
                            result.is_err(),
                            "TCP fault did not reach the ownership read"
                        );
                        proxy.online.send(true).unwrap();
                    }
                    if attempt == 2 {
                        assert!(matches!(result, Ok(true)), "{result:?}");
                        let current = observer
                            .get(claim.backend.claim_key.clone(), None)
                            .await
                            .unwrap();
                        assert_eq!(current.kvs()[0].lease(), claim.backend.lease_id);
                        assert_eq!(current.kvs()[0].value(), claim.backend.token.as_bytes());
                        // Successful read uncertainty recovery cannot hide a
                        // subsequent real lease loss, even with ongoing work.
                        observer.lease_revoke(claim.backend.lease_id).await.unwrap();
                    }
                    result
                }
            });
        let work = async {
            loop {
                scope.run(async { checkpoint() }).await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        };
        let result = tokio::time::timeout(Duration::from_secs(25), async {
            tokio::select! {
                result = watch => result,
                () = work => unreachable!(),
            }
        })
        .await
        .unwrap();
        assert!(result.contains("claim lost"), "{result}");
        assert_eq!(calls, 3);
        store.recover_orphaned().await.unwrap();
        let mut replacement = store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(replacement.task.id, old.task.id);
        assert!(!store.promote_maintenance(&mut old).await.unwrap());
        assert!(store.finish(old, Ok("stale".into())).await.is_err());
        assert!(store.preparation_owned(&replacement).await.unwrap());
        assert!(store.promote_maintenance(&mut replacement).await.unwrap());
        store.finish(replacement, Ok("valid".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn backlog_snapshot_handoff_fences_old_scanner() {
        let (_dir, first) = bounded_scheduler_store(vec![]).await;
        let mut cfg = config(&_dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = first.inner.prefix.clone();
        let second = TaskStore::open(&cfg).await.unwrap();
        assert!(first.read_backlog_snapshot().await.unwrap().is_none());
        let old = first.coordination_lock("stats-writer").await.unwrap();
        let mut high = crate::backlog_metrics::Snapshot::new(&[], 0, now_ms(), 0, 0);
        high.stores = 1;
        high.generations = 3913;
        high.largest_store = 3913;
        first.publish_backlog_snapshot(&old, &high).await.unwrap();
        let (seen, before) = second.read_backlog_snapshot().await.unwrap().unwrap();
        assert_eq!(seen.generations, 3913);
        // Real etcd lease loss, while the old process and its cached high value
        // survive. Only the successor may replace the shared observation.
        first.inner.revoke_lease(old.lease_id).await.unwrap();
        let new = second.coordination_lock("stats-writer").await.unwrap();
        let mut low = high.clone();
        low.generations = 1964;
        low.largest_store = 1964;
        second.publish_backlog_snapshot(&new, &low).await.unwrap();
        assert!(first.publish_backlog_snapshot(&old, &high).await.is_err());
        let (seen, after) = first.read_backlog_snapshot().await.unwrap().unwrap();
        assert_eq!(seen.generations, 1964);
        assert!(after > before);
        // A failed observation is published as incomplete, not erased/zeroed.
        low.failed_observations = 1;
        second.publish_backlog_snapshot(&new, &low).await.unwrap();
        assert_eq!(
            first
                .read_backlog_snapshot()
                .await
                .unwrap()
                .unwrap()
                .0
                .failed_observations,
            1
        );
        second.release_coordination_lock(new).await.unwrap();
        drop(old);
        second
            .inner
            .client
            .clone()
            .delete(
                format!("{}/observability/wal-backlog", second.inner.prefix),
                None,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn scheduler_notifications_reconnect_after_transport_loss() {
        let (_dir, proxy, store, mut observer) = lease_proxy_store().await;
        let mut rx = store.scheduler_notifications();
        tokio::time::timeout(Duration::from_secs(5), async {
            while *rx.borrow_and_update() < 10 {
                rx.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        proxy.online.send(false).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        // The write occurs during the disconnected window. Re-subscription
        // must wake a fresh scan even though this event itself was missed.
        observer
            .put(
                format!("{}/queue/disconnected", store.inner.prefix),
                "fixture",
                None,
            )
            .await
            .unwrap();
        proxy.online.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(10), rx.changed())
            .await
            .unwrap()
            .unwrap();
        rx.borrow_and_update();
        assert_eq!(
            observer
                .get(format!("{}/queue/disconnected", store.inner.prefix), None)
                .await
                .unwrap()
                .count(),
            1
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn idle_scheduler_avoids_repeated_claims_and_wakes_for_a_new_task() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let runner_store = store.clone();
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let (quiet, quiet_rx) = tokio::sync::oneshot::channel();
        let runner = tokio::spawn(async move {
            let mut rx = runner_store.scheduler_notifications();
            // Let subscription acknowledgements settle before measuring quiet
            // queue behavior; these acknowledgements intentionally trigger scans.
            tokio::time::sleep(Duration::from_secs(1)).await;
            rx.borrow_and_update();
            let admission = Arc::new(crate::admission::Admission::default());
            let mut cursor = SchedulerClaimCursor::default();
            let pool = Arc::new(tokio::sync::Semaphore::new(1));
            let operation = admission.try_admit_named("fixture").unwrap();
            counter.fetch_add(1, Ordering::SeqCst);
            assert!(runner_store
                .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &operation)
                .await
                .unwrap()
                .is_none());
            quiet.send(()).unwrap();
            loop {
                crate::scheduler_wakeup::wait(&mut rx, &pool, cursor.has_more_pages()).await;
                rx.borrow_and_update();
                counter.fetch_add(1, Ordering::SeqCst);
                if let Some(claim) = runner_store
                    .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &operation)
                    .await
                    .unwrap()
                {
                    return claim;
                }
            }
        });
        quiet_rx.await.unwrap();
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "unchanged queue should not be scanned every 500ms"
        );
        let queued = store
            .enqueue(TaskKind::MergeWal, "new-demand", vec![])
            .await
            .unwrap();
        let claim = tokio::time::timeout(Duration::from_secs(2), runner)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, queued.id);
        store
            .finish(claim, Ok("fixture completed".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn shared_scheduler_notifications_never_replace_authoritative_claims() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let mut first = store.scheduler_notifications();
        let second = store.clone().scheduler_notifications();
        assert!(
            first.same_channel(&second),
            "one stream shared by all pools/clones"
        );
        // Initial subscription also wakes a scan, covering changes made while
        // the watch was being installed.
        tokio::time::timeout(Duration::from_secs(5), first.changed())
            .await
            .unwrap()
            .unwrap();
        let task = store
            .enqueue(TaskKind::MergeWal, "notification", vec![])
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), first.changed())
            .await
            .unwrap()
            .unwrap();
        let claimed = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claimed.task.id, task.id);
        // A stale notification cannot grant the same durable task twice.
        assert!(store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .is_none());
        store
            .finish(claimed, Ok("fixture completed".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn merge_demand_admission_does_not_replay_a_served_fresh_page() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let coordinator = store.merge_coordinator();
        coordinator.request_merge("hot").await.unwrap();
        let request = coordinator.request_page(None).await.unwrap().0.remove(0);
        let first = store
            .enqueue_merge_request(&request)
            .await
            .unwrap()
            .unwrap();
        assert!(coordinator
            .get_merge_request("hot")
            .await
            .unwrap()
            .is_none());
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, first.id);
        store
            .finish(claim, Ok("peer served this request".into()))
            .await
            .unwrap();
        // Another master fetched the same page before the peer served it. It
        // may still be processing that page, without any failure or pause.
        assert!(store
            .enqueue_merge_request(&request)
            .await
            .unwrap()
            .is_none());
        assert_eq!(store.queue_depth().await.unwrap(), 0);
        assert!(store
            .get_active_id(TaskKind::MergeWal, "hot")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn merge_demand_admission_preserves_new_flushes_and_running_continuation() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let coordinator = store.merge_coordinator();
        let queued = store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        coordinator.request_merge("hot").await.unwrap();
        let stale = coordinator.get_merge_request("hot").await.unwrap().unwrap();
        coordinator.request_merge("hot").await.unwrap();
        let fresh = coordinator.get_merge_request("hot").await.unwrap().unwrap();
        assert!(fresh.revision > stale.revision);
        assert!(store.enqueue_merge_request(&stale).await.unwrap().is_none());
        assert_eq!(
            coordinator
                .get_merge_request("hot")
                .await
                .unwrap()
                .unwrap()
                .revision,
            fresh.revision
        );
        assert_eq!(
            store
                .enqueue_merge_request(&fresh)
                .await
                .unwrap()
                .unwrap()
                .id,
            queued.id
        );
        assert!(coordinator
            .get_merge_request("hot")
            .await
            .unwrap()
            .is_none());
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        coordinator.request_merge("hot").await.unwrap();
        let during_run = coordinator.get_merge_request("hot").await.unwrap().unwrap();
        assert_eq!(
            store
                .enqueue_merge_request(&during_run)
                .await
                .unwrap()
                .unwrap()
                .id,
            queued.id
        );
        assert!(coordinator
            .get_merge_request("hot")
            .await
            .unwrap()
            .is_some());
        store
            .finish(claim, Ok("pass completed".into()))
            .await
            .unwrap();
        let next = store
            .enqueue_merge_request(&during_run)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(next.id, queued.id);
        assert!(coordinator
            .get_merge_request("hot")
            .await
            .unwrap()
            .is_none());
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store
            .finish(claim, Ok("continuation completed".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn merge_demand_admission_concurrent_scanners_create_one_task() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let coordinator = store.merge_coordinator();
        coordinator.request_merge("hot").await.unwrap();
        let request = coordinator.get_merge_request("hot").await.unwrap().unwrap();
        let mut scanners = tokio::task::JoinSet::new();
        for _ in 0..20 {
            let store = store.clone();
            let request = request.clone();
            scanners.spawn(async move { store.enqueue_merge_request(&request).await.unwrap() });
        }
        let mut admitted = Vec::new();
        while let Some(result) = scanners.join_next().await {
            if let Some(task) = result.unwrap() {
                admitted.push(task);
            }
        }
        assert_eq!(admitted.len(), 1);
        assert_eq!(store.queue_depth().await.unwrap(), 1);
        assert!(coordinator
            .get_merge_request("hot")
            .await
            .unwrap()
            .is_none());
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, admitted[0].id);
        store
            .finish(claim, Ok("single pass completed".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn bounded_scheduler_claim_resumes_and_wraps_before_new_arrivals() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let mut client = store.inner.client.clone();
        let mut blocked = Vec::new();
        for i in 0..130 {
            let task = store
                .enqueue(TaskKind::MergeWal, &format!("blocked-{i}"), vec![])
                .await
                .unwrap();
            client
                .put(
                    store.inner.target_lock_key(&task.target),
                    "external-owner",
                    None,
                )
                .await
                .unwrap();
            blocked.push(task);
        }
        let tail = store
            .enqueue(TaskKind::MergeWal, "tail", vec![])
            .await
            .unwrap();
        let admission = Arc::new(crate::admission::Admission::default());
        let mut cursor = SchedulerClaimCursor::default();
        let first = admission.try_admit_named("merge").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &first)
            .await
            .unwrap()
            .is_none());
        let snapshot = admission.status();
        let p = snapshot
            .operations
            .values()
            .next()
            .unwrap()
            .claim_progress
            .as_ref()
            .unwrap();
        assert_eq!(p.queue_candidates, 64);
        // A blocked candidate must not require its own network round trip.
        // Includes the recovery page and queue page, not just preflight reads.
        assert!((2..=12).contains(&p.completed_etcd_responses), "{p:?}");
        assert!(snapshot
            .operations
            .values()
            .next()
            .unwrap()
            .progress
            .is_none());
        drop(first);
        // A formerly blocked old task becomes runnable while new tasks arrive.
        // The retained revision guarantees this round ends, so the old task
        // is reconsidered before the new tail can extend the scan forever.
        client
            .delete(store.inner.target_lock_key(&blocked[0].target), None)
            .await
            .unwrap();
        let newer = store
            .enqueue(TaskKind::MergeWal, "newer", vec![])
            .await
            .unwrap();
        let second = admission.try_admit_named("merge").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &second)
            .await
            .unwrap()
            .is_none());
        drop(second);
        let third = admission.try_admit_named("merge").unwrap();
        let claim = store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &third)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, tail.id);
        assert_eq!(
            admission
                .status()
                .operations
                .values()
                .next()
                .unwrap()
                .task_id
                .as_deref(),
            Some(tail.id.as_str())
        );
        store
            .finish(claim, Ok("test completion".into()))
            .await
            .unwrap();
        drop(third);
        let fourth = admission.try_admit_named("merge").unwrap();
        let claim = store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &fourth)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, blocked[0].id);
        assert_eq!(
            store.get(&newer.id).await.unwrap().unwrap().state,
            TaskState::Queued
        );
        assert_eq!(
            client
                .get(store.inner.target_lock_key(&blocked[1].target), None)
                .await
                .unwrap()
                .kvs()[0]
                .value(),
            b"external-owner"
        );
        store
            .finish(claim, Ok("test completion".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn batched_admission_reads_waiting_dependencies_then_wakes_after_completion() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let dependency = store
            .enqueue(TaskKind::IndexId, "dependency", vec![])
            .await
            .unwrap();
        for i in 0..64 {
            store
                .enqueue(
                    TaskKind::MergeWal,
                    &format!("dependent-{i}"),
                    vec![dependency.id.clone()],
                )
                .await
                .unwrap();
        }
        let admission = Arc::new(crate::admission::Admission::default());
        let mut cursor = SchedulerClaimCursor::default();
        let op = admission.try_admit_named("merge").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &op)
            .await
            .unwrap()
            .is_none());
        let status = admission.status();
        let progress = status
            .operations
            .values()
            .next()
            .unwrap()
            .claim_progress
            .as_ref()
            .unwrap();
        assert!(
            (2..=20).contains(&progress.completed_etcd_responses),
            "{progress:?}"
        );
        drop(op);
        let dependency_claim = store
            .claim_next_of_kinds(TaskKinds::GENERAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(dependency_claim.task.id, dependency.id);
        store
            .finish(dependency_claim, Ok("dependency completed".into()))
            .await
            .unwrap();
        // No advisory read survives the previous slice or hides the new state.
        let op = admission.try_admit_named("merge").unwrap();
        let mut cursor = SchedulerClaimCursor::default();
        let claim = store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &op)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.depends_on, [dependency.id]);
        store.finish(claim, Ok("done".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn batched_admission_reads_preserve_exclusive_claims_under_competition() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let dependency = store
            .enqueue(TaskKind::IndexId, "dependency", vec![])
            .await
            .unwrap();
        let dependency_claim = store
            .claim_next_of_kinds(TaskKinds::GENERAL)
            .await
            .unwrap()
            .unwrap();
        store
            .finish(dependency_claim, Ok("done".into()))
            .await
            .unwrap();
        // Dependency-chain tasks intentionally bypass ordinary target dedupe.
        for _ in 0..4 {
            store
                .enqueue(TaskKind::MergeWal, "contended", vec![dependency.id.clone()])
                .await
                .unwrap();
        }
        let results = futures::future::join_all(
            (0..4).map(|_| store.claim_next_of_kinds(TaskKinds::MERGE_WAL)),
        )
        .await;
        let mut claims: Vec<_> = results.into_iter().filter_map(Result::unwrap).collect();
        assert_eq!(
            claims.len(),
            1,
            "advisory batching must not grant concurrent table writers"
        );
        let first = claims.pop().unwrap();
        let first_id = first.task.id.clone();
        assert_eq!(store.queue_depth().await.unwrap(), 3);
        store.finish(first, Ok("done".into())).await.unwrap();
        let next = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(next.task.id, first_id);
        store.finish(next, Ok("done".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn batched_admission_reads_do_not_poison_earlier_work_with_later_corruption() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let first = store
            .enqueue(TaskKind::MergeWal, "first", vec![])
            .await
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "later", vec!["malformed".into()])
            .await
            .unwrap();
        store
            .inner
            .client
            .clone()
            .put(store.inner.task_key("malformed"), "not-json", None)
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, first.id);
        store.finish(claim, Ok("done".into())).await.unwrap();
        assert!(store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .is_err());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn batched_admission_reads_fall_back_for_long_dependency_chains() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let dependency = store
            .enqueue(TaskKind::IndexId, "dependency", vec![])
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::GENERAL)
            .await
            .unwrap()
            .unwrap();
        store.finish(claim, Ok("done".into())).await.unwrap();
        let task = store
            .enqueue(TaskKind::MergeWal, "long-chain", vec![dependency.id; 64])
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, task.id);
        store.finish(claim, Ok("done".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn bounded_scheduler_filtered_pages_preserve_other_pool_work() {
        let (_dir, store) = bounded_scheduler_store(vec!["chosen".into()]).await;
        let mut client = store.inner.client.clone();
        // Real canonical task/queue keys; batch fixture creation to avoid
        // making setup latency dominate the bounded-scan assertion.
        let mut tasks = Vec::new();
        for i in 0..1025 {
            let mut task = new_task(TaskKind::MergeWal, &format!("excluded-{i}"), vec![]);
            task.id = format!("00000000-0000-7000-8000-{i:012x}");
            tasks.push(task);
        }
        for chunk in tasks.chunks(32) {
            let ops = chunk
                .iter()
                .flat_map(|task| {
                    [
                        TxnOp::put(
                            store.inner.task_key(&task.id),
                            encode_task(task).unwrap(),
                            None,
                        ),
                        TxnOp::put(
                            store.inner.queue_key(&task.id),
                            encode_task(task).unwrap(),
                            None,
                        ),
                    ]
                })
                .collect::<Vec<_>>();
            client.txn(Txn::new().and_then(ops)).await.unwrap();
        }
        let chosen = store
            .enqueue(TaskKind::MergeWal, "chosen", vec![])
            .await
            .unwrap();
        let compact = store
            .enqueue(TaskKind::Compact, "maintenance", vec![])
            .await
            .unwrap();
        let admission = Arc::new(crate::admission::Admission::default());
        let mut merge_cursor = SchedulerClaimCursor::default();
        let first = admission.try_admit_named("merge").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut merge_cursor, &first)
            .await
            .unwrap()
            .is_none());
        let snapshot = admission.status();
        let p = snapshot
            .operations
            .values()
            .next()
            .unwrap()
            .claim_progress
            .as_ref()
            .unwrap();
        assert_eq!(p.queue_candidates, 1024);
        // 1 empty recovery page + 4 queue pages: excluded merge targets
        // trigger no dependency/eligibility RPCs or whole-prefix sort.
        assert_eq!(p.completed_etcd_responses, 5);
        drop(first);
        let merge = admission.try_admit_named("merge").unwrap();
        let claim = store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut merge_cursor, &merge)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, chosen.id);
        let mut general_cursor = SchedulerClaimCursor::default();
        let general = admission.try_admit_named("general").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::GENERAL, None, &mut general_cursor, &general)
            .await
            .unwrap()
            .is_none());
        let maintenance = store
            .claim_scheduler_slice(TaskKinds::GENERAL, None, &mut general_cursor, &general)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(maintenance.task.id, compact.id);
        assert_eq!(
            store.get(&tasks[0].id).await.unwrap().unwrap().state,
            TaskState::Queued
        );
        store
            .finish(maintenance, Ok("test completion".into()))
            .await
            .unwrap();
        store
            .finish(claim, Ok("test completion".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn claim_scan_reuses_private_lease_after_rejected_cas() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let mut client = store.inner.client.clone();
        let mut changed = Vec::new();
        for i in 0..10 {
            let mut task = store
                .enqueue(TaskKind::MergeWal, &format!("changed-{i}"), vec![])
                .await
                .unwrap();
            // Deterministic lost CAS: canonical state changed after the queue
            // snapshot, while the advisory ownership keys remain unoccupied.
            task.detail = Some("concurrent update".into());
            client
                .put(
                    store.inner.task_key(&task.id),
                    encode_task(&task).unwrap(),
                    None,
                )
                .await
                .unwrap();
            changed.push(task);
        }
        let tail = store
            .enqueue(TaskKind::MergeWal, "runnable", vec![])
            .await
            .unwrap();
        let admission = Arc::new(crate::admission::Admission::default());
        let operation = admission.try_admit_named("merge").unwrap();
        let mut cursor = SchedulerClaimCursor::default();
        let claim = store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &operation)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, tail.id);
        let status = admission.status();
        let progress = status
            .operations
            .values()
            .next()
            .unwrap()
            .claim_progress
            .as_ref()
            .unwrap();
        assert_eq!(progress.queue_candidates, 11);
        assert_eq!(
            progress.completed_etcd_responses, 25,
            "two pages + eleven snapshots + eleven CAS responses + one lease grant; {progress:?}"
        );
        let id = claim.backend.lease_id;
        assert!(client.lease_time_to_live(id, None).await.unwrap().ttl() > 0);
        for task in changed {
            assert_eq!(store.get(&task.id).await.unwrap().unwrap(), task);
            assert!(client
                .get(store.inner.target_lock_key(&task.target), None)
                .await
                .unwrap()
                .kvs()
                .is_empty());
        }
        // Finishing the accepted task revokes the transferred lease normally.
        store.finish(claim, Ok("done".into())).await.unwrap();
        assert!(client.lease_time_to_live(id, None).await.unwrap().ttl() <= 0);
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn claim_scan_never_shares_a_lease_between_accepted_tasks() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        for target in ["first", "second"] {
            store
                .enqueue(TaskKind::MergeWal, target, vec![])
                .await
                .unwrap();
        }
        let first = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let second = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(first.backend.lease_id, second.backend.lease_id);
        let second_id = second.backend.lease_id;
        store.finish(first, Ok("done".into())).await.unwrap();
        assert!(
            store
                .inner
                .client
                .clone()
                .lease_time_to_live(second_id, None)
                .await
                .unwrap()
                .ttl()
                > 0
        );
        assert_eq!(
            store
                .inner
                .get_text(&second.backend.claim_key)
                .await
                .unwrap()
                .as_deref(),
            Some(second.backend.token.as_str())
        );
        store.finish(second, Ok("done".into())).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn claim_scan_cleans_up_after_exhaustion_and_late_dependency_error() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let mut client = store.inner.client.clone();
        let mut changed = store
            .enqueue(TaskKind::MergeWal, "changed", vec![])
            .await
            .unwrap();
        changed.detail = Some("concurrent update".into());
        client
            .put(
                store.inner.task_key(&changed.id),
                encode_task(&changed).unwrap(),
                None,
            )
            .await
            .unwrap();
        assert!(store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .is_none());
        let later = store
            .enqueue(TaskKind::MergeWal, "later", vec!["malformed".into()])
            .await
            .unwrap();
        client
            .put(store.inner.task_key("malformed"), "invalid-json", None)
            .await
            .unwrap();
        assert!(store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .is_err());
        for task in [&changed, &later] {
            assert_eq!(store.get(&task.id).await.unwrap().unwrap(), *task);
            assert!(client
                .get(store.inner.claim_key(&task.id), None)
                .await
                .unwrap()
                .kvs()
                .is_empty());
            assert!(client
                .get(store.inner.target_lock_key(&task.target), None)
                .await
                .unwrap()
                .kvs()
                .is_empty());
        }
        // The isolated-etcd validation also verifies no lease remains after
        // these empty/error paths, not just absence of task ownership keys.
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn bounded_scheduler_snapshot_uses_fresh_queue_and_ownership() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let mut client = store.inner.client.clone();
        for i in 0..65 {
            let t = store
                .enqueue(TaskKind::MergeWal, &format!("blocked-{i}"), vec![])
                .await
                .unwrap();
            client
                .put(store.inner.target_lock_key(&t.target), "live-owner", None)
                .await
                .unwrap();
        }
        let tail = store
            .enqueue(TaskKind::MergeWal, "completed-by-peer", vec![])
            .await
            .unwrap();
        let admission = Arc::new(crate::admission::Admission::default());
        let mut cursor = SchedulerClaimCursor::default();
        let first = admission.try_admit_named("merge").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &first)
            .await
            .unwrap()
            .is_none());
        drop(first);
        // Another executor finishes a task still visible in our old queue snapshot.
        let peer = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(peer.task.id, tail.id);
        store
            .finish(peer, Ok("peer completed".into()))
            .await
            .unwrap();
        let second = admission.try_admit_named("merge").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &second)
            .await
            .unwrap()
            .is_none());
        let status = admission.status();
        let progress = status
            .operations
            .values()
            .next()
            .unwrap()
            .claim_progress
            .as_ref()
            .unwrap();
        assert_eq!(progress.queue_candidates, 2);
        assert_eq!(
            progress.completed_etcd_responses, 3,
            "two pages + one batched snapshot, without a lease for the historical task"
        );
        assert_eq!(
            store.get(&tail.id).await.unwrap().unwrap().state,
            TaskState::Done
        );
        assert_eq!(
            client
                .get(store.inner.target_lock_key("blocked-64"), None)
                .await
                .unwrap()
                .kvs()[0]
                .value(),
            b"live-owner"
        );
    }

    #[tokio::test]
    #[ignore = "requires dedicated loopback etcd and ETCD_TEST_ALLOW_COMPACTION=1"]
    async fn bounded_scheduler_compacted_snapshot_restarts_without_stealing_locks() {
        // Compaction is cluster-wide, not isolated by our test key prefix.
        // Require explicit opt-in against a disposable local instance.
        assert_eq!(
            std::env::var("ETCD_TEST_ALLOW_COMPACTION").as_deref(),
            Ok("1"),
            "this test requires a dedicated disposable etcd instance"
        );
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        assert!(
            !endpoint.contains(',')
                && (endpoint.starts_with("http://127.0.0.1:")
                    || endpoint.starts_with("http://localhost:")),
            "cluster-wide compaction is restricted to a disposable loopback etcd"
        );
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        let mut client = store.inner.client.clone();
        for i in 0..65 {
            let t = store
                .enqueue(TaskKind::MergeWal, &format!("blocked-{i}"), vec![])
                .await
                .unwrap();
            client
                .put(store.inner.target_lock_key(&t.target), "live-owner", None)
                .await
                .unwrap();
        }
        let tail = store
            .enqueue(TaskKind::MergeWal, "tail", vec![])
            .await
            .unwrap();
        let admission = Arc::new(crate::admission::Admission::default());
        let mut cursor = SchedulerClaimCursor::default();
        let first = admission.try_admit_named("merge").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &first)
            .await
            .unwrap()
            .is_none());
        let revision = cursor.queue.revision.unwrap();
        drop(first);
        let advanced = client
            .put(
                format!("{}/advance-revision", store.inner.prefix),
                "1",
                None,
            )
            .await
            .unwrap()
            .header()
            .unwrap()
            .revision();
        assert!(advanced > revision);
        client.compact(advanced, None).await.unwrap();
        let compacted = admission.try_admit_named("merge").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &compacted)
            .await
            .is_err());
        assert!(cursor.queue.revision.is_none() && cursor.queue.next.is_none());
        assert_eq!(
            store.get(&tail.id).await.unwrap().unwrap().state,
            TaskState::Queued
        );
        drop(compacted);
        let mut claimed = None;
        for _ in 0..3 {
            let operation = admission.try_admit_named("merge").unwrap();
            claimed = store
                .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &operation)
                .await
                .unwrap();
            if claimed.is_some() {
                break;
            }
        }
        let claim = claimed.expect("a fresh sweep must reach the tail");
        assert_eq!(claim.task.id, tail.id);
        assert_eq!(
            client
                .get(store.inner.target_lock_key("blocked-0"), None)
                .await
                .unwrap()
                .kvs()[0]
                .value(),
            b"live-owner"
        );
        store
            .finish(claim, Ok("test completion".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn bounded_scheduler_dependency_cleanup_does_not_hold_accepted_claim() {
        let (_dir, mut store) = bounded_scheduler_store(vec![]).await;
        store.history_limit = 1;
        let failed = store
            .enqueue(TaskKind::MergeWal, "bad-1", vec!["missing".into()])
            .await
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "bad-2", vec!["missing".into()])
            .await
            .unwrap();
        let wanted = store
            .enqueue(TaskKind::MergeWal, "runnable", vec![])
            .await
            .unwrap();
        let admission = Arc::new(crate::admission::Admission::default());
        let operation = admission.try_admit_named("merge").unwrap();
        let mut cursor = SchedulerClaimCursor::default();
        let claim = store
            .claim_scheduler_slice(TaskKinds::MERGE_WAL, None, &mut cursor, &operation)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(claim.task.id, wanted.id);
        assert!(store.history_prune_pending());
        assert!(
            store.get(&failed.id).await.unwrap().is_some(),
            "claim path must not run the history scan"
        );
        assert_eq!(
            admission
                .status()
                .operations
                .values()
                .next()
                .unwrap()
                .task_id
                .as_deref(),
            Some(wanted.id.as_str())
        );
        assert_eq!(store.prune_pending_history().await.unwrap(), 1);
        assert!(!store.history_prune_pending());
        assert_eq!(
            store.get(&wanted.id).await.unwrap().unwrap().state,
            TaskState::Running
        );
        store
            .finish(claim, Ok("test completion".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn bounded_scheduler_recovery_visits_all_pages_without_stealing_live_claims() {
        let (_dir, store) = bounded_scheduler_store(vec![]).await;
        store
            .enqueue(TaskKind::MergeWal, "live", vec![])
            .await
            .unwrap();
        let live = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let mut client = store.inner.client.clone();
        for i in 0..70 {
            let mut task = new_task(TaskKind::MergeWal, &format!("orphan-{i}"), vec![]);
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
        let admission = Arc::new(crate::admission::Admission::default());
        let mut cursor = SchedulerClaimCursor::default();
        let first = admission.try_admit_named("general").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::GENERAL, None, &mut cursor, &first)
            .await
            .unwrap()
            .is_none());
        let depth = store.queue_depth().await.unwrap();
        assert!(depth > 0 && depth <= 64 && depth < 70);
        assert_eq!(
            admission
                .status()
                .operations
                .values()
                .next()
                .unwrap()
                .claim_progress
                .as_ref()
                .unwrap()
                .recovery_candidates,
            64
        );
        drop(first);
        let second = admission.try_admit_named("general").unwrap();
        assert!(store
            .claim_scheduler_slice(TaskKinds::GENERAL, None, &mut cursor, &second)
            .await
            .unwrap()
            .is_none());
        assert_eq!(store.queue_depth().await.unwrap(), 70);
        assert_eq!(
            store.get(&live.task.id).await.unwrap().unwrap().state,
            TaskState::Running
        );
        assert!(!client
            .get(live.backend.claim_key.clone(), None)
            .await
            .unwrap()
            .kvs()
            .is_empty());
        store
            .finish(live, Ok("test completion".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires loopback ETCD_TEST_ENDPOINTS"]
    async fn task_lease_stream_reconnect_preserves_the_same_claim() {
        let (_dir, proxy, store, mut observer) = lease_proxy_store().await;
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let lease = claim.backend.lease_id;
        let key = claim.backend.claim_key.clone();
        let token = claim.backend.token.clone();
        proxy.online.send(false).unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        proxy.online.send(true).unwrap();
        // Prove that the proxy really restored transport before checking lease
        // recovery. New connections must not inherit an unseen old cut event.
        tokio::time::timeout(Duration::from_secs(2), async {
            let mut probe = Client::connect([proxy.endpoint.as_str()], None)
                .await
                .unwrap();
            let values = probe.get(key.clone(), None).await.unwrap();
            assert_eq!(values.kvs()[0].value(), token.as_bytes());
        })
        .await
        .expect("proxy did not restore etcd connectivity");
        // Survive more than the original lease lifetime after a real TCP cut.
        tokio::time::sleep(Duration::from_secs(10)).await;
        let values = observer.get(key.clone(), None).await.unwrap();
        assert_eq!(
            values.kvs().len(),
            1,
            "a transient stream failure expired the claim"
        );
        assert_eq!(values.kvs()[0].value(), token.as_bytes());
        assert_eq!(
            values.kvs()[0].lease(),
            lease,
            "recovery must not replace the lease"
        );
        assert!(!claim.backend.keepalive.task.is_finished());
        // Dropping a claim must still stop renewal, including after reconnect.
        drop(claim);
        tokio::time::timeout(Duration::from_secs(9), async {
            while !observer
                .get(key.clone(), None)
                .await
                .unwrap()
                .kvs()
                .is_empty()
            {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .unwrap();
        observer
            .delete(
                store.inner.prefix.clone(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires loopback ETCD_TEST_ENDPOINTS"]
    async fn task_lease_stream_recovers_when_renewal_replies_are_delayed() {
        let (_dir, proxy, store, mut observer) = lease_proxy_store().await;
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let key = claim.backend.claim_key.clone();
        let token = claim.backend.token.clone();
        let lease = claim.backend.lease_id;
        // Let the first ordinary renewal complete. Then forward requests to
        // etcd while withholding responses: an unacknowledged renewal can
        // extend the real lease beyond the client's last confirmation.
        tokio::time::sleep(Duration::from_secs(3)).await;
        proxy.responses.send(false).unwrap();
        tokio::time::sleep(Duration::from_secs(6)).await;
        // Restore responses before any observer RPC: a slow linearizable
        // read must not silently extend the injected six-second outage.
        let renewal_still_running = !claim.backend.keepalive.task.is_finished();
        proxy.responses.send(true).unwrap();
        let values = observer.get(key.clone(), None).await.unwrap();
        assert_eq!(
            values.kvs().len(),
            1,
            "fixture must retain a live server lease"
        );
        assert_eq!(values.kvs()[0].value(), token.as_bytes());
        assert_eq!(values.kvs()[0].lease(), lease);
        assert!(
            observer
                .lease_time_to_live(lease, None)
                .await
                .unwrap()
                .ttl()
                > 0
        );
        assert!(renewal_still_running,
            "unacknowledged renewals must be retried on the same lease within a bounded recovery window");
        tokio::time::sleep(Duration::from_secs(8)).await;
        let values = observer.get(key, None).await.unwrap();
        assert_eq!(
            values.kvs().len(),
            1,
            "restored responses did not resume renewal"
        );
        assert_eq!(values.kvs()[0].value(), token.as_bytes());
        assert_eq!(values.kvs()[0].lease(), lease);
        assert!(!claim.backend.keepalive.task.is_finished());
        drop(claim);
        observer.lease_revoke(lease).await.unwrap();
        observer
            .delete(
                store.inner.prefix.clone(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires loopback ETCD_TEST_ENDPOINTS"]
    async fn task_lease_stream_recovery_stops_after_a_persistent_outage() {
        let (_dir, proxy, store, mut observer) = lease_proxy_store().await;
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let key = claim.backend.claim_key.clone();
        proxy.online.send(false).unwrap();
        tokio::time::timeout(Duration::from_secs(10), async {
            while !claim.backend.keepalive.task.is_finished() {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .expect("renewal task exceeded its bounded recovery window");
        // Client termination and server-side lease deletion are separate:
        // etcd may delay its expiry transaction or this linearizable read.
        // A slow observer must not mask whether the client stopped in time.
        tokio::time::timeout(Duration::from_secs(10), async {
            while !observer
                .get(key.clone(), None)
                .await
                .unwrap()
                .kvs()
                .is_empty()
            {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .unwrap();
        proxy.online.send(true).unwrap();
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(observer.get(key, None).await.unwrap().kvs().is_empty());
        drop(claim);
        observer
            .delete(
                store.inner.prefix.clone(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires loopback ETCD_TEST_ENDPOINTS"]
    async fn task_lease_stream_recovery_never_resurrects_revoked_ownership() {
        let (_dir, proxy, store, mut observer) = lease_proxy_store().await;
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let lease = claim.backend.lease_id;
        let key = claim.backend.claim_key.clone();
        proxy.online.send(false).unwrap();
        observer.lease_revoke(lease).await.unwrap();
        observer
            .put(key.clone(), "replacement-owner", None)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        proxy.online.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(9), async {
            while !claim.backend.keepalive.task.is_finished() {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .unwrap();
        let values = observer.get(key, None).await.unwrap();
        assert_eq!(values.kvs()[0].value(), b"replacement-owner");
        assert_eq!(values.kvs()[0].lease(), 0);
        assert!(
            observer
                .lease_time_to_live(lease, None)
                .await
                .unwrap()
                .ttl()
                <= 0
        );
        drop(claim);
        observer
            .delete(
                store.inner.prefix.clone(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }

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
            assert!(!store.inner.recover_task(&task, None).await.unwrap());
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
            legacy_retirement_request: None,
            local_merge: Default::default(),
            append: Default::default(),
            catchup: Default::default(),
            wal_tail: Default::default(),
            planner: Default::default(),
            maintenance: Default::default(),
            merge_rollout: Default::default(),
            data_dir: dir.path().to_string_lossy().to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            stats_scan_interval_secs: 0,
            backlog_metrics_max_age_secs: 900,
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
            legacy_merge_wait_concurrency: 32,
            legacy_merge_foreground_secs: 30,
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
    async fn wal_claim_targets_preserve_queue_peers_and_writer_exclusion() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/wal-claim-targets/{}", generate_id());
        let peer = TaskStore::open(&cfg).await.unwrap();
        cfg.maintenance.wal_claim_targets = vec!["hot".into()];
        let selected = TaskStore::open(&cfg).await.unwrap();
        let broken = peer
            .enqueue(
                TaskKind::MergeWal,
                "excluded-dependency",
                vec!["missing".into()],
            )
            .await
            .unwrap();
        // Reach the selected target on a later queue page, without consuming
        // unrelated tasks or evaluating their dependencies.
        for i in 0..TASK_POLL_BATCH {
            peer.enqueue(TaskKind::MergeWal, &format!("other-{i}"), vec![])
                .await
                .unwrap();
        }
        let hot = peer
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let claim = selected.claim_next().await.unwrap().unwrap();
        assert_eq!(claim.task.id, hot.id);
        assert_eq!(
            selected.get(&broken.id).await.unwrap().unwrap().state,
            TaskState::Queued
        );
        assert_eq!(peer.queue_depth().await.unwrap(), TASK_POLL_BATCH + 1);
        // The filter cannot bypass an existing owner, even on the same process.
        peer.enqueue(TaskKind::Compact, "hot", vec![])
            .await
            .unwrap();
        assert!(selected.claim_next().await.unwrap().is_none());
        assert!(selected
            .claim_resident_merge(&["other-0".into()])
            .await
            .unwrap()
            .is_none());
        // Maintenance on unselected tables and claims by another master remain
        // independent of this master's local routing restriction.
        let index = peer
            .enqueue(TaskKind::IndexId, "outside", vec![])
            .await
            .unwrap();
        let maintenance = selected.claim_next().await.unwrap().unwrap();
        assert_eq!(maintenance.task.id, index.id);
        selected
            .finish(maintenance, Ok("index done".into()))
            .await
            .unwrap();
        let other = peer
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert!(other.task.target.starts_with("other-"));
        peer.finish(other, Ok("peer done".into())).await.unwrap();
        selected.finish(claim, Ok("hot done".into())).await.unwrap();
        let compact = selected.claim_next().await.unwrap().unwrap();
        assert_eq!(compact.task.kind, TaskKind::Compact);
        assert_eq!(compact.task.target, "hot");
        selected
            .finish(compact, Ok("compact done".into()))
            .await
            .unwrap();
        peer.inner
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

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn generic_recovery_releases_reserved_catchup_after_claim_loss() {
        use lance_context_merge::{Execution, MaintenanceKind, Phase};
        let (_dir, cfg, store) = preparation_test_store().await;
        let state = crate::state::MasterState::new(cfg).await.unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let old_claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let coordinator = store.merge_coordinator();
        let mut reserved = Execution::new("hot", "master:catchup", "old-instance", 600);
        reserved.maintenance = Some(MaintenanceKind::Catchup);
        assert!(coordinator
            .reserve(&store.merge_claim(&old_claim), &reserved)
            .await
            .unwrap());
        store.abandon_claim_for_test(old_claim).await.unwrap();
        let recovery = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let proof = store.merge_claim(&recovery);
        crate::merge_execution::recover_execution(&state, &coordinator, &proof, reserved.clone())
            .await
            .unwrap();
        assert!(coordinator.get("hot").await.unwrap().is_none());
        assert!(coordinator.start(&reserved).await.unwrap().is_none());
        let mut late = reserved.clone();
        late.phase = Phase::Running;
        assert!(coordinator
            .authorize_commit(&late, &state.rollout_uri("hot"), "base", 1)
            .await
            .is_err());
        let successor = Execution::new("hot", "master:catchup", "new-instance", 600);
        assert!(coordinator.reserve(&proof, &successor).await.unwrap());
        let running = coordinator.start(&successor).await.unwrap().unwrap();
        assert!(coordinator.finish(&running, Ok(0)).await.unwrap());
        let terminal = coordinator.get("hot").await.unwrap().unwrap();
        assert!(coordinator.release(&proof, &terminal).await.unwrap());
        store
            .finish(recovery, Ok("replacement completed".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn orphaned_catchup_reservation_reaches_normal_recovery() {
        use lance_context_merge::{Execution, MaintenanceKind, Phase};
        let (_dir, mut cfg, _) = preparation_test_store().await;
        cfg.merge_rollout.owned_targets = vec!["hot".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        let state = crate::state::MasterState::new(cfg.clone()).await.unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let old_claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let coordinator = store.merge_coordinator();
        let mut reserved = Execution::new("hot", "master:catchup", "old-instance", 600);
        reserved.maintenance = Some(MaintenanceKind::Catchup);
        assert!(coordinator
            .reserve(&store.merge_claim(&old_claim), &reserved)
            .await
            .unwrap());
        let active = crate::catchup::store::active_key(&cfg.etcd.etcd_prefix, "hot");
        store
            .inner
            .client
            .clone()
            .put(active.clone(), reserved.instance.clone(), None)
            .await
            .unwrap();
        // The old task has lost its lease, but the catchup reservation and
        // execution remain. No matching Job identity is supplied by this poller.
        store.abandon_claim_for_test(old_claim).await.unwrap();
        let recovery = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let proof = store.merge_claim(&recovery);
        crate::maintenance_execution::reconcile_previous(&state, &recovery)
            .await
            .unwrap();
        assert!(store
            .inner
            .client
            .clone()
            .get(active, None)
            .await
            .unwrap()
            .kvs()
            .is_empty());
        assert!(coordinator.get("hot").await.unwrap().is_none());
        assert!(coordinator.start(&reserved).await.unwrap().is_none());
        let mut late = reserved.clone();
        late.phase = Phase::Running;
        assert!(coordinator
            .authorize_commit(&late, &state.rollout_uri("hot"), "base", 1)
            .await
            .is_err());
        let successor = Execution::new("hot", "master:catchup", "new-instance", 600);
        assert!(coordinator.reserve(&proof, &successor).await.unwrap());
        let running = coordinator.start(&successor).await.unwrap().unwrap();
        assert!(coordinator.finish(&running, Ok(0)).await.unwrap());
        let terminal = coordinator.get("hot").await.unwrap().unwrap();
        assert!(coordinator.release(&proof, &terminal).await.unwrap());
        store
            .finish(recovery, Ok("replacement completed".into()))
            .await
            .unwrap();
    }

    async fn orphan_catchup_fixture() -> (
        TempDir,
        Arc<crate::state::MasterState>,
        TaskStore,
        lance_context_merge::Execution,
    ) {
        use lance_context_merge::{Execution, MaintenanceKind};
        let (dir, mut cfg, _) = preparation_test_store().await;
        cfg.merge_rollout.owned_targets = vec!["hot".into()];
        let state = crate::state::MasterState::new(cfg.clone()).await.unwrap();
        let store = TaskStore::open(&cfg).await.unwrap();
        store
            .enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let mut execution = Execution::new("hot", "master:catchup", "old-job", 600);
        execution.maintenance = Some(MaintenanceKind::Catchup);
        let coordinator = store.merge_coordinator();
        assert!(coordinator
            .reserve(&store.merge_claim(&claim), &execution)
            .await
            .unwrap());
        store
            .inner
            .client
            .clone()
            .put(
                crate::catchup::store::active_key(&cfg.etcd.etcd_prefix, "hot"),
                execution.instance.clone(),
                None,
            )
            .await
            .unwrap();
        store.abandon_claim_for_test(claim).await.unwrap();
        (dir, state, store, execution)
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn orphaned_catchup_reservation_preserves_failure_history_and_cancelled_crash() {
        let (_dir, state, store, reserved) = orphan_catchup_fixture().await;
        let coordinator = store.merge_coordinator();
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let proof = store.merge_claim(&claim);
        let error = "merge ownership unresolved: recovery barrier failed: execution lacks recoverable storage fencing protocol";
        for _ in 0..15 {
            coordinator
                .record_failure(&proof, "hot", "master:catchup", error)
                .await
                .unwrap();
        }
        let before = coordinator
            .failure("hot", "master:catchup")
            .await
            .unwrap()
            .unwrap();
        // The first recovery crashes after cancelling, before releasing either
        // persistent key. A normal replacement must resume the same terminal CAS.
        let cancelled = coordinator
            .cancel_reserved_for_recovery(&proof, &reserved)
            .await
            .unwrap()
            .unwrap();
        store.abandon_claim_for_test(claim).await.unwrap();
        let replacement = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let explanation = store
            .inner
            .explain_admission(&replacement.task.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(explanation.reason, "not_queued");
        crate::maintenance_execution::reconcile_previous(&state, &replacement)
            .await
            .unwrap();
        assert!(coordinator.get("hot").await.unwrap().is_none());
        let after = coordinator
            .failure("hot", "master:catchup")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.consecutive_attempts, before.consecutive_attempts);
        assert_eq!(after.last_error, before.last_error);
        assert_eq!(after.last_failure_ms, before.last_failure_ms);
        assert!(after.next_retry_ms <= lance_context_merge::failure::now_ms());
        assert!(!coordinator
            .release_orphaned_catchup(&proof, &cancelled)
            .await
            .unwrap());
        let result = crate::maintenance_execution::run_as(
            &state,
            &replacement,
            lance_context_merge::MaintenanceKind::Catchup,
            async { Err("invalid schema from actual payload work".into()) },
        )
        .await;
        assert_eq!(
            result.unwrap_err(),
            "invalid schema from actual payload work"
        );
        let failure = coordinator
            .failure("hot", "master:catchup")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(failure.consecutive_attempts, 16);
        assert!(failure.next_retry_ms > lance_context_merge::failure::now_ms());
        store
            .finish(replacement, Err("fixture payload failed".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn orphaned_catchup_reservation_claim_rejects_changed_ownership() {
        for changed in [
            "start",
            "active",
            "record",
            "target",
            "merge_claim",
            "task_claim",
        ] {
            let (_dir, state, store, reserved) = orphan_catchup_fixture().await;
            store.inner.recover_orphaned().await.unwrap();
            let id = store
                .get_active_id(TaskKind::MergeWal, "hot")
                .await
                .unwrap()
                .unwrap();
            let before = store.inner.explain_admission(&id).await.unwrap().unwrap();
            assert!(before.orphaned_catchup_recovery);
            assert!(before.skip.is_none());
            let execution_key =
                lance_context_merge::execution_key(&state.config.etcd.etcd_prefix, "hot");
            let (key, value) = match changed {
                "start" => {
                    let mut running = reserved.clone();
                    running.phase = lance_context_merge::Phase::Running;
                    (
                        execution_key.clone(),
                        serde_json::to_string(&running).unwrap(),
                    )
                }
                "active" => (
                    crate::catchup::store::active_key(&state.config.etcd.etcd_prefix, "hot"),
                    "new-job".into(),
                ),
                "record" => (
                    execution_key.replace("/merge-executions/", "/catchup-records/"),
                    "new-inventory".into(),
                ),
                "target" => (store.inner.target_lock_key("hot"), "new-owner".into()),
                "merge_claim" => (
                    execution_key.replace("/merge-executions/", "/merge-claims/"),
                    "live".into(),
                ),
                "task_claim" => (store.inner.claim_key(&id), "live".into()),
                _ => unreachable!(),
            };
            // Change after the advisory snapshot, immediately before the claim
            // transaction, including Reserved -> Running with the same owner ID.
            *store.inner.claim_owner_swap.lock().unwrap() = Some((key.clone(), value.clone()));
            assert!(
                store
                    .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                    .await
                    .unwrap()
                    .is_none(),
                "{changed}"
            );
            assert_eq!(
                store.inner.get_text(&key).await.unwrap().as_deref(),
                Some(value.as_str())
            );
            assert_eq!(
                store.get(&id).await.unwrap().unwrap().state,
                TaskState::Queued
            );
            assert!(store
                .inner
                .explain_admission(&id)
                .await
                .unwrap()
                .unwrap()
                .skip
                .is_some());
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn orphaned_catchup_reservation_release_is_atomic_and_preserves_replacements() {
        for changed in ["active", "record", "execution", "target", "claim"] {
            let (_dir, state, store, reserved) = orphan_catchup_fixture().await;
            let coordinator = store.merge_coordinator();
            let claim = store
                .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                .await
                .unwrap()
                .unwrap();
            let proof = store.merge_claim(&claim);
            let terminal = coordinator
                .cancel_reserved_for_recovery(&proof, &reserved)
                .await
                .unwrap()
                .unwrap();
            let execution_key =
                lance_context_merge::execution_key(&state.config.etcd.etcd_prefix, "hot");
            let active = crate::catchup::store::active_key(&state.config.etcd.etcd_prefix, "hot");
            let (key, value) = match changed {
                "active" => (active.clone(), "new-job".into()),
                "record" => (
                    execution_key.replace("/merge-executions/", "/catchup-records/"),
                    "new-inventory".into(),
                ),
                "execution" => {
                    let mut replacement = terminal.clone();
                    replacement.id = "successor".into();
                    (
                        execution_key.clone(),
                        serde_json::to_string(&replacement).unwrap(),
                    )
                }
                "target" => (store.inner.target_lock_key("hot"), "new-owner".into()),
                "claim" => (proof.key.clone(), "new-claim".into()),
                _ => unreachable!(),
            };
            store
                .inner
                .client
                .clone()
                .put(key.clone(), value.clone(), None)
                .await
                .unwrap();
            assert!(
                !coordinator
                    .release_orphaned_catchup(&proof, &terminal)
                    .await
                    .unwrap(),
                "{changed}"
            );
            assert_eq!(
                store.inner.get_text(&key).await.unwrap().as_deref(),
                Some(value.as_str())
            );
            assert!(coordinator.get("hot").await.unwrap().is_some());
            assert!(store.inner.get_text(&active).await.unwrap().is_some());
            store.abandon_claim_for_test(claim).await.unwrap();
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn orphaned_catchup_reservation_racing_start_keeps_marker_and_writer() {
        let (_dir, state, store, reserved) = orphan_catchup_fixture().await;
        let coordinator = store.merge_coordinator();
        let claim = store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let running = coordinator.start(&reserved).await.unwrap().unwrap();
        assert!(
            crate::maintenance_execution::reconcile_previous(&state, &claim)
                .await
                .is_err()
        );
        assert_eq!(coordinator.get("hot").await.unwrap(), Some(running));
        let active = crate::catchup::store::active_key(&state.config.etcd.etcd_prefix, "hot");
        assert_eq!(
            store.inner.get_text(&active).await.unwrap().as_deref(),
            Some(reserved.instance.as_str())
        );
        assert!(store
            .claim_merge_target("hot", &reserved.instance)
            .await
            .unwrap()
            .is_none());
        store
            .finish(claim, Err("racing executor remains protected".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn reserved_recovery_admits_retry_without_resetting_failure_history() {
        use lance_context_merge::{failure::FailureClass, Execution, MaintenanceKind};
        for canonical_preflight in [false, true] {
            let (_dir, mut cfg, store) = preparation_test_store().await;
            cfg.merge_rollout.owned_targets = vec!["hot".into()];
            let state = crate::state::MasterState::new(cfg).await.unwrap();
            store
                .enqueue(TaskKind::MergeWal, "hot", vec![])
                .await
                .unwrap();
            let claim = store
                .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                .await
                .unwrap()
                .unwrap();
            let proof = store.merge_claim(&claim);
            let coordinator = store.merge_coordinator();
            let mut reserved = Execution::new("hot", "master:catchup", "old", 600);
            reserved.maintenance = Some(MaintenanceKind::Catchup);
            assert!(coordinator.reserve(&proof, &reserved).await.unwrap());
            let error = "merge ownership unresolved: recovery barrier failed: execution lacks recoverable storage fencing protocol";
            for _ in 0..15 {
                coordinator
                    .record_failure(&proof, "hot", "master:catchup", error)
                    .await
                    .unwrap();
            }
            let before = coordinator
                .failure("hot", "master:catchup")
                .await
                .unwrap()
                .unwrap();
            assert!(before.next_retry_ms > lance_context_merge::failure::now_ms());
            if canonical_preflight {
                crate::maintenance_execution::reconcile_previous(&state, &claim)
                    .await
                    .unwrap();
            } else {
                crate::merge_execution::recover_execution(&state, &coordinator, &proof, reserved)
                    .await
                    .unwrap();
            }
            let after = coordinator
                .failure("hot", "master:catchup")
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                after.consecutive_attempts, before.consecutive_attempts,
                "recovering an unstarted reservation is not another failed merge"
            );
            assert_eq!(after.last_error, before.last_error);
            assert_eq!(after.last_failure_ms, before.last_failure_ms);
            assert!(after.next_retry_ms <= lance_context_merge::failure::now_ms());
            // The resolved protocol problem permits one real attempt. If that
            // attempt fails for a different reason, its original budget survives.
            let outcome = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                crate::maintenance_execution::run_as(
                    &state,
                    &claim,
                    MaintenanceKind::Catchup,
                    async { Err("invalid schema from actual payload work".into()) },
                ),
            )
            .await
            .unwrap();
            assert_eq!(
                outcome.unwrap_err(),
                "invalid schema from actual payload work"
            );
            let failed = coordinator
                .failure("hot", "master:catchup")
                .await
                .unwrap()
                .unwrap();
            assert_eq!(failed.consecutive_attempts, 16);
            assert_eq!(failed.class, FailureClass::DataOrConfiguration);
            assert!(failed.next_retry_ms > lance_context_merge::failure::now_ms());
            store
                .finish(claim, Err("fixture payload failed".into()))
                .await
                .unwrap();
        }
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
        let targets = ["hot", "draining", "external"].map(str::to_owned);
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
        let generic = store
            .claim_resident_merge(&["generic:gs".into()])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(generic.task.target, "generic:gs");
        assert!(other
            .claim_resident_merge(&["generic:gs".into()])
            .await
            .unwrap()
            .is_none());
        store.finish(generic, Ok("done".into())).await.unwrap();
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
    async fn repair_validation_owner_swap_rejects_stale_claim_and_preserves_marker() {
        let (_dir, mut cfg, _) = preparation_test_store().await;
        cfg.merge_rollout.owned_targets = vec!["hot".into()];
        cfg.maintenance.repair_validate_only_targets = vec!["hot".into()];
        cfg.maintenance.maintenance_catchup_targets = vec!["hot".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        let active = crate::catchup::store::active_key(&cfg.etcd.etcd_prefix, "hot");
        store
            .inner
            .client
            .clone()
            .put(active.clone(), "old", None)
            .await
            .unwrap();
        let task = store
            .enqueue(TaskKind::Repair, "hot", vec![])
            .await
            .unwrap();
        *store.inner.claim_owner_swap.lock().unwrap() = Some((active.clone(), "new".into()));
        assert!(
            store.claim_next().await.unwrap().is_none(),
            "owner changed after read must reject final CAS"
        );
        assert!(store
            .inner
            .get_text(&store.inner.target_lock_key("hot"))
            .await
            .unwrap()
            .is_none());
        let claim = store.claim_next().await.unwrap().unwrap();
        assert_eq!(claim.task.id, task.id);
        assert_eq!(
            claim.backend.preparation_owner.as_deref(),
            Some(b"new".as_slice())
        );
        assert!(!claim.preparing_maintenance());
        let peer = TaskStore::open(&cfg).await.unwrap();
        peer.enqueue(TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        assert!(peer
            .claim_merge_target("hot", "new")
            .await
            .unwrap()
            .is_none());
        peer.enqueue(TaskKind::IndexId, "hot", vec![])
            .await
            .unwrap();
        assert!(peer.claim_next().await.unwrap().is_none());
        store.finish(claim, Ok("validated".into())).await.unwrap();
        assert_eq!(
            store.inner.get_text(&active).await.unwrap().as_deref(),
            Some("new")
        );
        let merge = peer
            .claim_merge_target("hot", "new")
            .await
            .unwrap()
            .unwrap();
        store
            .enqueue(TaskKind::Repair, "hot", vec![])
            .await
            .unwrap();
        assert!(
            store.claim_next().await.unwrap().is_none(),
            "repair cannot overlap active merge"
        );
        peer.finish(merge, Ok("pass finished".into()))
            .await
            .unwrap();
        let index = peer.claim_next().await.unwrap().unwrap();
        assert_eq!(index.task.kind, TaskKind::IndexId);
        assert!(
            store.claim_next().await.unwrap().is_none(),
            "repair cannot overlap maintenance"
        );
        peer.finish(index, Ok("index finished".into()))
            .await
            .unwrap();
        let next = store.claim_next().await.unwrap().unwrap();
        assert_eq!(next.task.kind, TaskKind::Repair);
        store.finish(next, Ok("validated".into())).await.unwrap();
        assert_eq!(
            store.inner.get_text(&active).await.unwrap().as_deref(),
            Some("new")
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn repair_validation_exact_opt_in_cannot_bypass_scope_guards() {
        for (target, owned, draining, opt_in) in [
            ("hot", true, false, ""),
            ("hot", true, false, "*"),
            ("hot", false, false, "hot"),
            ("hot", true, true, "hot"),
            ("generic:hot", true, false, "generic:hot"),
        ] {
            let (_dir, mut cfg, _) = preparation_test_store().await;
            cfg.maintenance.maintenance_catchup_targets = vec![target.into()];
            cfg.maintenance.repair_validate_only_targets = vec![opt_in.into()];
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
                .enqueue(TaskKind::Repair, target, vec![])
                .await
                .unwrap();
            assert!(
                store.claim_next().await.unwrap().is_none(),
                "{target}/{opt_in}/{owned}/{draining}"
            );
            assert_eq!(
                store.inner.get_text(&active).await.unwrap().as_deref(),
                Some("native")
            );
        }
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
