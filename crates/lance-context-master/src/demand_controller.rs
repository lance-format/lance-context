//! One leased controller discovers durable merge demand for all executors.
//! Followers watch only the leader key. The leader lists at one revision, then
//! maintains bounded state from resumable watches; executing tasks is unchanged.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use etcd_client::{Client, EventType, GetOptions, WatchOptions};
use lance_context_api::{TaskKind, TaskRecord, TaskState};
use lance_context_merge::{
    failure::{FailureObservation, ShardFailure},
    rollout::MergeRequest,
    Execution, ExecutionObservation, Phase,
};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::{
    config::MasterConfig,
    state::MasterState,
    task_store::{CoordinationGuard, TaskStore},
};

pub(crate) mod aggregate;
pub(crate) mod dispatch;

const RANGES: &[&str] = &[
    "merge-requests",
    "merge-executions",
    "merge-failures",
    "demand-controller/members",
    "queue",
    "running",
    "demand-controller/tickets",
    "demand-controller/offers",
    "demand-controller/reservations",
    "claims",
    "target-locks",
    "merge-claims",
    "catchup-active",
    "catchup-records",
    "compact-preparations",
    "compact-commit-ready",
    // Watch dependency changes without listing retained task history.
    "tasks",
];
const MAX_KEYS: usize = 100_000;
const MAX_BYTES: usize = 64 * 1024 * 1024;
const REVISIT: Duration = Duration::from_secs(15);
const RECONNECT: Duration = Duration::from_secs(5);

/// Admission policy, not execution placement. Preserve the union of the old
/// replica-local discovery rules, even when a narrowly scoped master leads.
/// The lease keeps this record live without periodically rewriting its value.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Policy {
    v: u32,
    automatic: bool,
    #[serde(default)]
    aggregate_min_generations: Option<i64>,
    /// Age-based admission is opt-in through the existing WAL tail setting.
    /// Older members advertise no tail policy during a mixed-version rollout.
    #[serde(default)]
    aggregate_tail_interval_secs: u64,
    #[serde(default)]
    aggregate_merge_slots: usize,
    endpoints: Vec<String>,
    fence_legacy: bool,
    owned: Vec<String>,
    draining: Vec<String>,
    local: Vec<String>,
    local_auto: bool,
}

impl Policy {
    fn from_config(config: &MasterConfig) -> Self {
        Self {
            v: 1,
            automatic: config.merge_wal_interval_secs > 0,
            aggregate_min_generations: Some(config.merge_wal_min_generations),
            aggregate_tail_interval_secs: if config.wal_tail.wal_tail_stats_max_age_secs > 0 {
                config.wal_tail.wal_tail_interval_secs
            } else {
                0
            },
            aggregate_merge_slots: if !config.maintenance.claim_wal_merges {
                0
            } else if config.merge_wal_concurrency > 0 {
                config.merge_wal_concurrency
            } else {
                config.task_concurrency.max(1)
            },
            endpoints: config.worker_endpoints.clone(),
            fence_legacy: config.merge_rollout.fence_legacy,
            owned: config.merge_rollout.owned_targets.clone(),
            draining: config.merge_rollout.drain_targets.clone(),
            local: config
                .merge_rollout
                .owned_targets
                .iter()
                .filter(|t| crate::local_merge::selected(config, t))
                .cloned()
                .collect(),
            // Validated automatic local routing requires wildcard local targets.
            local_auto: config.local_merge.master_local_merge_auto
                && config.merge_rollout.fence_legacy,
        }
    }

    fn admits_request(&self, target: &str, execution: Option<&Execution>) -> bool {
        let dynamic = self.local_auto
            && !target.starts_with("datagen:")
            && lance_context_core::validate_store_name(
                target.strip_prefix("generic:").unwrap_or(target),
            )
            .is_ok();
        let ordinary = self.automatic
            && !self.draining.iter().any(|t| t == target)
            && (!self.endpoints.is_empty() || dynamic || self.local.iter().any(|t| t == target));
        ordinary
            || self.fence_legacy
            || self.owned.iter().any(|t| t == target)
            || execution.is_some_and(|e| e.maintenance.is_none())
    }

    fn admits_failure(
        &self,
        failure: &ShardFailure,
        execution: Option<&Execution>,
    ) -> Option<TaskKind> {
        let kind = crate::maintenance_execution::retry_kind(&failure.endpoint).or_else(|| {
            self.endpoints
                .contains(&failure.endpoint)
                .then_some(TaskKind::MergeWal)
        })?;
        let draining = self.draining.contains(&failure.target);
        ((!draining && (self.fence_legacy || self.owned.contains(&failure.target)))
            || (draining && execution.is_some()))
        .then_some(kind)
    }
}

#[derive(Clone)]
struct Entry {
    value: Vec<u8>,
    revision: i64,
    lease_id: i64,
    due: Instant,
    active: Option<(TaskKind, String)>,
    policy: Option<Result<Policy, String>>,
    offered_target: Option<String>,
}

struct Cache {
    entries: BTreeMap<Vec<u8>, Entry>,
    // One timer per key, including keys being reconciled. Repeated updates
    // replace the value without growing the queue or moving a cold key back.
    ready: BTreeSet<(Instant, Vec<u8>)>,
    bytes: usize,
    revisions: Vec<i64>,
    active: BTreeMap<(TaskKind, String), BTreeSet<Vec<u8>>>,
    placement: dispatch::Placement,
    offered_targets: BTreeMap<String, usize>,
}

impl Cache {
    fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            ready: BTreeSet::new(),
            bytes: 0,
            revisions: vec![0; RANGES.len()],
            active: BTreeMap::new(),
            placement: dispatch::Placement::default(),
            offered_targets: BTreeMap::new(),
        }
    }

    fn apply(&mut self, key: &[u8], value: Option<&[u8]>, revision: i64) -> Result<(), String> {
        let old = self.entries.get(key);
        if old.is_some_and(|entry| entry.revision >= revision) {
            return Ok(());
        }
        let old_bytes = old.map_or(0, |e| key.len() + e.value.len());
        let bytes = self.bytes - old_bytes + value.map_or(0, |v| key.len() + v.len());
        if value.is_some()
            && ((old.is_none() && self.entries.len() >= MAX_KEYS) || bytes > MAX_BYTES)
        {
            return Err("demand informer capacity exceeded; durable records retained".into());
        }
        let location = std::str::from_utf8(key)
            .ok()
            .and_then(|key| key.rsplit_once('/'));
        let offered_target =
            if location.is_some_and(|(parent, _)| parent.ends_with("/demand-controller/offers")) {
                value.map(dispatch::offer_target).transpose()?
            } else {
                None
            };
        let due = old.map_or_else(Instant::now, |e| e.due);
        self.placement
            .observe_ticket(key, old.map(|entry| entry.value.as_slice()), value);
        self.placement.forget_queue_entry(key);
        if let Some(old) = self.entries.remove(key) {
            self.ready.remove(&(old.due, key.to_vec()));
            if let Some(target) = old.offered_target {
                if let Some(count) = self.offered_targets.get_mut(&target) {
                    *count -= 1;
                    if *count == 0 {
                        self.offered_targets.remove(&target);
                    }
                }
            }
            if let Some(active) = old.active {
                if let Some(keys) = self.active.get_mut(&active) {
                    keys.remove(key);
                    if keys.is_empty() {
                        self.active.remove(&active);
                    }
                }
            }
        }
        self.bytes = bytes;
        if let Some(value) = value {
            if let Some(target) = &offered_target {
                *self.offered_targets.entry(target.clone()).or_default() += 1;
            }
            let task_state = location.and_then(|(parent, _)| {
                if parent.ends_with("/queue") {
                    Some(TaskState::Queued)
                } else if parent.ends_with("/running") {
                    Some(TaskState::Running)
                } else {
                    None
                }
            });
            let policy = location
                .filter(|(parent, _)| parent.ends_with("/demand-controller/members"))
                .map(|_| serde_json::from_slice::<Policy>(value).map_err(|e| e.to_string()));
            let active = task_state.and_then(|expected| {
                let task: TaskRecord = serde_json::from_slice(value).ok()?;
                // A queued dependency chain must not absorb independent work.
                // Once running, its live claim must suppress recovery of the
                // same kind/target just like a task without dependencies.
                (task.state == expected
                    && (task.depends_on.is_empty() || expected == TaskState::Running)
                    && location.is_some_and(|(_, id)| id == task.id))
                .then_some((task.kind, task.target))
            });
            if let Some(active) = &active {
                self.active
                    .entry(active.clone())
                    .or_default()
                    .insert(key.to_vec());
            }
            let auxiliary = location.is_some_and(|(parent, _)| {
                RANGES
                    .iter()
                    .skip(3)
                    .any(|range| parent.ends_with(&format!("/{range}")))
            });
            self.entries.insert(
                key.to_vec(),
                Entry {
                    value: value.to_vec(),
                    revision,
                    lease_id: 0,
                    due,
                    active,
                    policy,
                    offered_target,
                },
            );
            if !auxiliary {
                self.ready.insert((due, key.to_vec()));
            }
        }
        Ok(())
    }

    fn next_due(&self) -> Instant {
        self.ready
            .first()
            .map_or_else(|| Instant::now() + REVISIT, |(due, _)| *due)
    }

    fn take_due(&mut self) -> Option<(Vec<u8>, Entry)> {
        if self.ready.first()?.0 > Instant::now() {
            return None;
        }
        let (_, key) = self.ready.pop_first()?;
        let entry = self.entries.get_mut(&key)?;
        let observed = entry.clone();
        entry.due = Instant::now() + REVISIT;
        self.ready.insert((entry.due, key.clone()));
        Some((key, observed))
    }

    fn value<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<Option<T>, String> {
        self.entries
            .get(key.as_bytes())
            .map(|e| serde_json::from_slice(&e.value).map_err(|error| error.to_string()))
            .transpose()
    }

    fn has_active(&self, kind: TaskKind, target: &str) -> bool {
        self.active.contains_key(&(kind, target.to_string()))
    }

    fn queued(&self, prefix: &str, kind: TaskKind, target: &str) -> Option<&Entry> {
        self.active
            .get(&(kind, target.to_string()))?
            .iter()
            .find(|key| key.starts_with(prefix_key(prefix, "queue").as_bytes()))
            .and_then(|key| self.entries.get(key))
    }

    fn policies(&self, prefix: &str) -> Result<Vec<&Policy>, String> {
        let prefix = format!("{prefix}/demand-controller/members/");
        self.entries
            .range(prefix.as_bytes().to_vec()..)
            .take_while(|(key, _)| key.starts_with(prefix.as_bytes()))
            .map(|(_, e)| {
                let policy = e
                    .policy
                    .as_ref()
                    .ok_or("missing decoded policy")?
                    .as_ref()
                    .map_err(Clone::clone)?;
                if policy.v != 1 {
                    return Err("unsupported discovery policy version".into());
                }
                Ok(policy)
            })
            .collect()
    }
}

fn prefix_key(prefix: &str, range: &str) -> String {
    format!("{}/{range}/", prefix.trim_end_matches('/'))
}

/// Every page across every range uses the first page's revision. No dispatch
/// starts with a partial snapshot. Memory is bounded even during bootstrap.
async fn list(client: &Client, prefix: &str) -> Result<Cache, String> {
    let mut client = client.clone();
    let mut cache = Cache::new();
    let mut revision = 0;
    for range in RANGES {
        if *range == "tasks" {
            continue;
        }
        let start = prefix_key(prefix, range);
        let mut end = start.as_bytes().to_vec();
        *end.last_mut().unwrap() += 1;
        let mut key = start.into_bytes();
        let mut limit = 128;
        loop {
            let mut options = GetOptions::new().with_range(end.clone()).with_limit(limit);
            if revision > 0 {
                options = options.with_revision(revision);
            }
            metrics::counter!("master_demand_controller_list_rpcs_total").increment(1);
            let response = match client.get(key.clone(), Some(options)).await {
                Ok(response) => response,
                Err(error) if limit > 1 && error.to_string().contains("message length") => {
                    limit = (limit / 2).max(1);
                    continue;
                }
                Err(error) => return Err(error.to_string()),
            };
            if revision == 0 {
                revision = response.header().map_or(0, |h| h.revision());
                if revision <= 0 {
                    return Err("missing informer snapshot revision".into());
                }
            }
            for kv in response.kvs() {
                cache.apply(kv.key(), Some(kv.value()), kv.mod_revision())?;
                if let Some(entry) = cache.entries.get_mut(kv.key()) {
                    entry.lease_id = kv.lease();
                }
            }
            if !response.more() {
                break;
            }
            key = response
                .kvs()
                .last()
                .ok_or("empty informer continuation page")?
                .key()
                .to_vec();
            key.push(0);
        }
    }
    cache.revisions.fill(revision);
    metrics::counter!("master_demand_controller_snapshots_total").increment(1);
    Ok(cache)
}

async fn watch(
    client: &Client,
    prefix: &str,
    cache: &Cache,
) -> Result<etcd_client::WatchStream, String> {
    let options = |i: usize| {
        WatchOptions::new()
            .with_prefix()
            .with_start_revision(cache.revisions[i] + 1)
            .with_watch_id(i as i64 + 1)
            .with_progress_notify()
    };
    let mut stream = client
        .clone()
        .watch(prefix_key(prefix, RANGES[0]), Some(options(0)))
        .await
        .map_err(|e| e.to_string())?;
    for (i, range) in RANGES.iter().enumerate().skip(1) {
        stream
            .watch(prefix_key(prefix, range), Some(options(i)))
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(stream)
}

fn apply_response(cache: &mut Cache, response: &etcd_client::WatchResponse) -> Result<(), String> {
    if response.canceled() {
        if response.compact_revision() > 0 {
            return Err("compacted".into());
        }
        return Err(format!(
            "demand watch canceled: {}",
            response.cancel_reason()
        ));
    }
    // A create acknowledgement's header can be ahead of replayed events.
    // Advancing to that header would lose updates on an immediate reconnect.
    if response.created() {
        return Ok(());
    }
    let index =
        usize::try_from(response.watch_id() - 1).map_err(|_| "invalid informer watch id")?;
    if index >= RANGES.len() {
        return Err("unknown informer watch id".into());
    }
    for event in response.events() {
        let kv = event.kv().ok_or("informer event missing key")?;
        let value = (event.event_type() != EventType::Delete).then_some(kv.value());
        if RANGES[index] == "tasks" {
            cache
                .placement
                .observe_dependency(kv.key(), value, kv.mod_revision());
        } else {
            cache.apply(kv.key(), value, kv.mod_revision())?;
            if let Some(entry) = cache.entries.get_mut(kv.key()) {
                if entry.revision == kv.mod_revision() {
                    entry.lease_id = kv.lease();
                }
            }
        }
    }
    // Each prefix advances separately; events on another watch are no proof
    // that this prefix has delivered that global revision.
    let revision = if response.events().is_empty() {
        response.header().map_or(0, |h| h.revision())
    } else {
        response
            .events()
            .iter()
            .filter_map(|e| e.kv().map(|kv| kv.mod_revision()))
            .max()
            .unwrap_or(0)
    };
    cache.revisions[index] = cache.revisions[index].max(revision);
    Ok(())
}

async fn reconcile(
    publisher: &TaskStore,
    prefix: &str,
    cache: &Cache,
    key: &[u8],
    entry: &Entry,
) -> Result<(), String> {
    let policies = cache.policies(prefix)?;
    let execution_for = |target: &str| {
        cache.value::<Execution>(&lance_context_merge::execution_key(prefix, target))
    };
    if key.starts_with(prefix_key(prefix, "merge-requests").as_bytes()) {
        let target = std::str::from_utf8(&entry.value).map_err(|e| e.to_string())?;
        let expected = lance_context_merge::execution_key(prefix, target)
            .replace("/merge-executions/", "/merge-requests/");
        if key != expected.as_bytes() {
            return Err("merge request key/target mismatch".into());
        }
        let execution = execution_for(target)?;
        if policies
            .iter()
            .any(|p| p.admits_request(target, execution.as_ref()))
        {
            let request = MergeRequest {
                target: target.to_string(),
                revision: entry.revision,
            };
            if let Some(queued) = cache.queued(prefix, TaskKind::MergeWal, target) {
                if publisher
                    .acknowledge_cached_merge_request(&request, &queued.value)
                    .await
                    .map_err(|e| e.to_string())?
                {
                    metrics::counter!("master_worker_demand_admission_total", "outcome" => "queued").increment(1);
                    return Ok(());
                }
            } else if cache.has_active(TaskKind::MergeWal, target) {
                // Keep the request durable until this running pass completes.
                return Ok(());
            }
            publisher
                .enqueue_merge_request(&request)
                .await
                .map_err(|e| e.to_string())?;
        }
    } else if key.starts_with(prefix_key(prefix, "merge-executions").as_bytes()) {
        let execution: Execution =
            serde_json::from_slice(&entry.value).map_err(|e| e.to_string())?;
        if key != lance_context_merge::execution_key(prefix, &execution.target).as_bytes() {
            return Err("execution key/target mismatch".into());
        }
        let kind = if execution.maintenance.is_some() {
            let Some(kind) = crate::maintenance_execution::retry_kind(&execution.endpoint) else {
                return Ok(());
            };
            if execution.protocol != 2
                && !policies.iter().any(|p| {
                    p.owned.contains(&execution.target) || p.draining.contains(&execution.target)
                })
            {
                return Ok(());
            }
            kind
        } else {
            TaskKind::MergeWal
        };
        if cache.has_active(kind, &execution.target) {
            metrics::counter!("master_execution_recovery_already_active_total").increment(1);
            return Ok(());
        }
        let failure = cache
            .value::<ShardFailure>(&lance_context_merge::failure::failure_key(
                prefix,
                &execution.target,
                &execution.endpoint,
            ))?
            .map(ShardFailure::corrected);
        let delayed = failure.is_some_and(|f| {
            f.next_retry_ms > lance_context_merge::failure::now_ms()
                && (execution.maintenance.is_some()
                    || (execution.phase == Phase::Recovering
                        && f.last_error.contains("recovery barrier failed")))
        });
        if !delayed {
            publisher
                .enqueue_execution_recovery(
                    kind,
                    &ExecutionObservation {
                        execution,
                        revision: entry.revision,
                    },
                )
                .await
                .map_err(|e| e.to_string())?;
        }
    } else if key.starts_with(prefix_key(prefix, "merge-failures").as_bytes()) {
        let failure = serde_json::from_slice::<ShardFailure>(&entry.value)
            .map_err(|e| e.to_string())?
            .corrected();
        if key
            != lance_context_merge::failure::failure_key(prefix, &failure.target, &failure.endpoint)
                .as_bytes()
        {
            return Err("failure key/target mismatch".into());
        }
        if failure.next_retry_ms > lance_context_merge::failure::now_ms() {
            return Ok(());
        }
        let execution = execution_for(&failure.target)?;
        if let Some(kind) = policies
            .iter()
            .find_map(|p| p.admits_failure(&failure, execution.as_ref()))
        {
            if cache.has_active(kind, &failure.target) {
                metrics::counter!("master_failure_recovery_already_active_total").increment(1);
                return Ok(());
            }
            publisher
                .enqueue_failure_recovery(
                    kind,
                    &FailureObservation {
                        failure,
                        revision: entry.revision,
                    },
                )
                .await
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Followers use one quiet watch. No prefix scan, per-table read, election
/// transaction, or new lease while a healthy leader holds the key.
pub(crate) async fn wait_for_vacancy(
    client: &Client,
    key: &str,
    owner: Option<&str>,
) -> Result<(), String> {
    let response = client
        .clone()
        .get(key, None)
        .await
        .map_err(|e| e.to_string())?;
    if response.kvs().is_empty()
        || owner.is_some_and(|token| response.kvs()[0].value() != token.as_bytes())
    {
        return Ok(());
    }
    let revision = response
        .header()
        .ok_or("missing election revision")?
        .revision();
    let mut stream = client
        .clone()
        .watch(
            key,
            Some(WatchOptions::new().with_start_revision(revision + 1)),
        )
        .await
        .map_err(|e| e.to_string())?;
    while let Some(response) = stream.message().await.map_err(|e| e.to_string())? {
        if response.canceled() {
            return Err("leader watch canceled".into());
        }
        if response.events().iter().any(|e| {
            e.event_type() == EventType::Delete
                || owner
                    .is_some_and(|token| e.kv().is_some_and(|kv| kv.value() != token.as_bytes()))
        }) {
            return Ok(());
        }
    }
    Err("leader watch ended".into())
}

async fn lead(state: &Arc<MasterState>, session: &CoordinationGuard) -> Result<(), String> {
    let store = &state.task_store;
    let publisher = store.discovery_publisher(session);
    let prefix = state.config.etcd.prefix().trim_end_matches('/');
    let client = store.etcd_client();
    let mut cache = list(client, prefix).await?;
    let mut place_more = true;
    let mut place_dirty = false;
    let mut recovery_due = Instant::now();
    loop {
        let mut stream = match watch(client, prefix, &cache).await {
            Ok(stream) => stream,
            Err(error) => {
                tracing::warn!(%error, "demand watch connection failed; retaining cursor");
                tokio::time::sleep(RECONNECT).await;
                continue;
            }
        };
        let failure = loop {
            let next = cache.next_due();
            tokio::select! {
                response = stream.message() => {
                    match response {
                        Ok(Some(response)) => {
                            if !response.events().is_empty() { place_dirty = true; place_more = true; }
                            if let Err(error) = apply_response(&mut cache, &response) { break error; }
                        },
                        Ok(None) => break "demand watch ended".to_string(),
                        Err(error) => break error.to_string(),
                    }
                }
                _ = tokio::task::yield_now(), if place_more => {
                    let Some(operation) = state.admission.try_admit_named("task_dispatch") else { return Ok(()); };
                    operation.phase("offer_reserved_capacity");
                    match dispatch::place(&publisher, prefix, &mut cache).await {
                        Ok(more) => {
                            place_more = more || place_dirty;
                            if !more { place_dirty = false; }
                        },
                        Err(error) => break error,
                    }
                }
                _ = tokio::time::sleep_until(recovery_due) => {
                    recovery_due = Instant::now() + Duration::from_secs(1);
                    let Some(operation) = state.admission.try_admit_named("task_recovery") else { return Ok(()); };
                    operation.phase("recover_observed_orphan");
                    // No RPC unless the cache contains a running task with no
                    // claim. A fenced transaction rechecks actual ownership.
                    if let Err(error) = dispatch::recover(&publisher, prefix, &cache).await { break error; }
                }
                _ = tokio::time::sleep_until(next) => {
                    if let Some((key, entry)) = cache.take_due() {
                        // This operation only publishes metadata. Storage writes
                        // still require the executor's existing claim and fence.
                        let Some(operation) = state.admission.try_admit_named("worker_merge_demand") else { return Ok(()); };
                        operation.phase("controller_reconcile");
                        let result = tokio::time::timeout(Duration::from_secs(10), reconcile(&publisher, prefix, &cache, &key, &entry)).await;
                        match result {
                            Ok(Ok(())) => operation.observed_work(),
                            other => {
                                metrics::counter!("master_demand_controller_reconcile_errors_total").increment(1);
                                tracing::warn!(error = ?other, "demand reconciliation deferred; durable record retained");
                            }
                        }
                    }
                }
            }
            let queued = cache
                .entries
                .range(prefix_key(prefix, "queue").into_bytes()..)
                .take_while(|(key, _)| key.starts_with(prefix_key(prefix, "queue").as_bytes()))
                .count();
            metrics::gauge!("master_task_queue_depth").set(queued as f64);
            metrics::gauge!("master_demand_controller_cache_keys").set(cache.entries.len() as f64);
            metrics::gauge!("master_demand_controller_cache_bytes").set(cache.bytes as f64);
        };
        metrics::counter!("master_demand_controller_watch_reconnects_total").increment(1);
        tracing::warn!(%failure, "demand watch reconnecting");
        tokio::time::sleep(RECONNECT).await;
        if failure == "compacted" || failure.contains("message length") {
            cache = list(client, prefix).await?;
            place_more = true;
        }
        // An oversized historical response also needs a bounded fresh List;
        // reconnecting the same cursor would otherwise fail forever.
        // For transport errors retain each cursor and replay only deltas.
    }
}

async fn maintain_history(
    state: &Arc<MasterState>,
    session: &CoordinationGuard,
) -> Result<(), String> {
    let publisher = state.task_store.discovery_publisher(session);
    let mut ticker = tokio::time::interval(Duration::from_secs(300));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        let Some(operation) = state.admission.try_admit_named("task_history") else {
            return Ok(());
        };
        operation.phase("controller_history_maintenance");
        if let Err(error) = publisher.prune_controller_history().await {
            tracing::warn!(%error, "controller history maintenance deferred");
        }
    }
}

async fn become_leader(
    store: &TaskStore,
    session: &CoordinationGuard,
    key: &str,
) -> Result<(), String> {
    loop {
        wait_for_vacancy(store.etcd_client(), key, None).await?;
        if store
            .elect_discovery_leader(session)
            .await
            .map_err(|e| e.to_string())?
        {
            return Ok(());
        }
    }
}

pub(crate) fn spawn(state: &Arc<MasterState>) -> tokio::task::JoinHandle<()> {
    let state = state.clone();
    tokio::spawn(async move {
        let policy = serde_json::to_vec(&Policy::from_config(&state.config))
            .expect("serialize discovery policy");
        let store = &state.task_store;
        metrics::gauge!("master_demand_controller_leader").set(0.0);
        metrics::gauge!("master_task_queue_depth").set(0.0);
        loop {
            if !state.admission.is_accepting() {
                return;
            }
            let session = match store.discovery_session(policy.clone()).await {
                Ok(session) => session,
                Err(error) => {
                    tracing::warn!(%error, "discovery registration failed");
                    tokio::time::sleep(RECONNECT).await;
                    continue;
                }
            };
            let work = async {
                let key = format!(
                    "{}/demand-controller/leader-v2",
                    state.config.etcd.prefix().trim_end_matches('/')
                );
                become_leader(store, &session, &key).await?;
                metrics::gauge!("master_demand_controller_leader").set(1.0);
                // The exact leader-key watcher stops discovery promptly
                // if our lease expires. Publication also has a CAS fence.
                tokio::select! {
                    result = lead(&state, &session) => result,
                    _ = aggregate::run(&state, &session) => Ok(()),
                    result = maintain_history(&state, &session) => result,
                    result = wait_for_vacancy(store.etcd_client(), &key, Some(session.token())) => result,
                }
            };
            let health = async {
                while session.is_live() && state.admission.is_accepting() {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            };
            tokio::select! {
                result = work => if let Err(error) = result { tracing::warn!(%error, "demand controller session ended"); },
                _ = health => {},
            }
            metrics::gauge!("master_demand_controller_leader").set(0.0);
            metrics::gauge!("master_task_queue_depth").set(0.0);
            metrics::gauge!("master_demand_controller_cache_keys").set(0.0);
            metrics::gauge!("master_demand_controller_cache_bytes").set(0.0);
            if let Err(error) = store.release_coordination_lock(session).await {
                tracing::warn!(%error, "discovery lease release failed");
            }
            tokio::time::sleep(RECONNECT).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use lance_context_merge::failure::FailureClass;

    fn config() -> MasterConfig {
        MasterConfig::parse_from(["test"])
    }

    pub(super) async fn fixture() -> (tempfile::TempDir, MasterConfig, TaskStore) {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config();
        cfg.data_dir = dir.path().to_string_lossy().into_owned();
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_string)
            .collect();
        cfg.etcd.etcd_prefix = format!(
            "/demand-controller-test/{}",
            lance_context_core::generate_id()
        );
        cfg.etcd_lease_ttl_secs = 5;
        cfg.merge_rollout.fence_legacy = true;
        cfg.worker_endpoints = vec!["worker-a".into()];
        let store = TaskStore::open(&cfg).await.unwrap();
        (dir, cfg, store)
    }

    fn failure(target: &str) -> ShardFailure {
        ShardFailure {
            target: target.into(),
            endpoint: "worker-a".into(),
            consecutive_attempts: 7,
            class: FailureClass::Retryable,
            last_error: "transport failure".into(),
            last_failure_ms: 1,
            next_retry_ms: 0,
            needs_attention: false,
        }
    }

    #[test]
    fn coalesces_hot_keys_without_starving_cold_keys_or_growing_timers() {
        let mut cache = Cache::new();
        cache.apply(b"hot", Some(b"1"), 1).unwrap();
        cache.apply(b"cold", Some(b"1"), 2).unwrap();
        let due = cache.entries[b"hot".as_slice()].due;
        for n in 3..10_000 {
            cache.apply(b"hot", Some(b"latest"), n).unwrap();
        }
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.ready.len(), 2);
        assert_eq!(cache.entries[b"hot".as_slice()].due, due);
        assert_eq!(cache.take_due().unwrap().0, b"hot");
        assert_eq!(cache.take_due().unwrap().0, b"cold");
        assert!(cache.take_due().is_none());
        cache.apply(b"hot", None, 10_000).unwrap();
        assert_eq!(cache.ready.len(), 1);
        assert_eq!(cache.bytes, 5);
    }

    #[test]
    fn capacity_failure_preserves_previous_value_and_does_not_drop_other_work() {
        let mut cache = Cache::new();
        cache.apply(b"a", Some(b"old"), 1).unwrap();
        cache.apply(b"b", Some(b"safe"), 2).unwrap();
        let oversized = vec![0; MAX_BYTES];
        assert!(cache.apply(b"a", Some(&oversized), 3).is_err());
        assert_eq!(cache.entries[b"a".as_slice()].value, b"old");
        assert_eq!(cache.entries[b"b".as_slice()].value, b"safe");
        assert_eq!(cache.ready.len(), 2);
        assert_eq!(cache.bytes, 9);
    }

    #[test]
    fn discovery_uses_policy_union_without_expanding_scoped_admission() {
        let mut narrow = config();
        narrow.worker_endpoints.clear();
        narrow.merge_wal_interval_secs = 0;
        narrow.merge_rollout.owned_targets = vec!["narrow".into()];
        let narrow = Policy::from_config(&narrow);
        assert!(narrow.admits_request("narrow", None));
        assert!(!narrow.admits_request("broad", None));
        let mut broad = config();
        broad.worker_endpoints = vec!["worker-a".into()];
        broad.merge_wal_interval_secs = 30;
        let mut broad = Policy::from_config(&broad);
        assert!(broad.admits_request("broad", None));
        broad.draining = vec!["broad".into()];
        assert!(!broad.admits_request("broad", None));
        broad.fence_legacy = true;
        assert!(broad.admits_failure(&failure("narrow"), None).is_some());
        assert!(narrow.admits_failure(&failure("narrow"), None).is_none());
        assert!(broad.admits_failure(&failure("broad"), None).is_none());
        let execution = Execution::new("broad", "worker-a", "old", 60);
        assert_eq!(
            broad.admits_failure(&failure("broad"), Some(&execution)),
            Some(TaskKind::MergeWal)
        );
        assert!([narrow, broad]
            .iter()
            .any(|p| p.admits_request("narrow", None)));
    }

    pub(super) async fn next_events(
        stream: &mut etcd_client::WatchStream,
        cache: &mut Cache,
        predicate: impl Fn(&Cache) -> bool,
    ) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !predicate(cache) {
                let response = stream.message().await.unwrap().unwrap();
                apply_response(cache, &response).unwrap();
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn informer_replays_list_watch_gap_and_reconnect_without_relisting() {
        let (_dir, cfg, store) = fixture().await;
        let prefix = cfg.etcd.prefix();
        let coordinator = store.merge_coordinator();
        coordinator.request_merge("early").await.unwrap();
        let mut cache = list(store.etcd_client(), prefix).await.unwrap();
        let early = lance_context_merge::execution_key(prefix, "early")
            .replace("/merge-executions/", "/merge-requests/");
        // Changes after List, before watch setup, must be replayed.
        coordinator.request_merge("late").await.unwrap();
        store
            .etcd_client()
            .clone()
            .delete(early.clone(), None)
            .await
            .unwrap();
        let late = lance_context_merge::execution_key(prefix, "late")
            .replace("/merge-executions/", "/merge-requests/");
        let mut stream = watch(store.etcd_client(), prefix, &cache).await.unwrap();
        next_events(&mut stream, &mut cache, |c| {
            c.entries.contains_key(late.as_bytes()) && !c.entries.contains_key(early.as_bytes())
        })
        .await;
        let observed = cache.entries[late.as_bytes()].revision;
        drop(stream);
        coordinator.request_merge("late").await.unwrap();
        let mut stream = watch(store.etcd_client(), prefix, &cache).await.unwrap();
        next_events(&mut stream, &mut cache, |c| {
            c.entries[late.as_bytes()].revision > observed
        })
        .await;
        assert!(!cache.entries.contains_key(early.as_bytes()));
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn stale_leader_cannot_acknowledge_demand_or_publish_recovery() {
        let (_dir, cfg, store) = fixture().await;
        let policy = serde_json::to_vec(&Policy::from_config(&cfg)).unwrap();
        let old = store.discovery_session(policy.clone()).await.unwrap();
        assert!(store.elect_discovery_leader(&old).await.unwrap());
        let stale = store.discovery_publisher(&old);
        let coordinator = store.merge_coordinator();
        coordinator.request_merge("fresh").await.unwrap();
        coordinator.request_merge("queued").await.unwrap();
        store
            .enqueue(TaskKind::MergeWal, "queued", vec![])
            .await
            .unwrap();
        let execution = Execution::new("orphan", "worker-a", "old", 60);
        let execution_key = lance_context_merge::execution_key(cfg.etcd.prefix(), "orphan");
        store
            .etcd_client()
            .clone()
            .put(execution_key, serde_json::to_vec(&execution).unwrap(), None)
            .await
            .unwrap();
        let f = failure("retry");
        let failure_key =
            lance_context_merge::failure::failure_key(cfg.etcd.prefix(), &f.target, &f.endpoint);
        store
            .etcd_client()
            .clone()
            .put(failure_key, serde_json::to_vec(&f).unwrap(), None)
            .await
            .unwrap();
        let cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        let old_token = old.token().to_string();
        store.release_coordination_lock(old).await.unwrap();
        let new = store.discovery_session(policy).await.unwrap();
        assert!(store.elect_discovery_leader(&new).await.unwrap());
        // A new owner may already exist before the old leader sets up its
        // ownership watch. That must stop the old leader immediately.
        tokio::time::timeout(
            Duration::from_secs(1),
            wait_for_vacancy(
                store.etcd_client(),
                &format!("{}/demand-controller/leader-v2", cfg.etcd.prefix()),
                Some(&old_token),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        for (key, entry) in &cache.entries {
            reconcile(&stale, cfg.etcd.prefix(), &cache, key, entry)
                .await
                .unwrap();
        }
        assert_eq!(store.list().await.unwrap().len(), 1);
        assert!(coordinator
            .get_merge_request("fresh")
            .await
            .unwrap()
            .is_some());
        assert!(coordinator
            .get_merge_request("queued")
            .await
            .unwrap()
            .is_some());
        let publisher = store.discovery_publisher(&new);
        for (key, entry) in &cache.entries {
            reconcile(&publisher, cfg.etcd.prefix(), &cache, key, entry)
                .await
                .unwrap();
        }
        assert_eq!(store.list().await.unwrap().len(), 4);
        assert!(coordinator
            .get_merge_request("fresh")
            .await
            .unwrap()
            .is_none());
        assert!(coordinator
            .get_merge_request("queued")
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            coordinator
                .failure("retry", "worker-a")
                .await
                .unwrap()
                .unwrap()
                .consecutive_attempts,
            7
        );
        store.release_coordination_lock(new).await.unwrap();
    }
    pub(super) async fn rpc_count(endpoint: &str, methods: &[&str]) -> u64 {
        reqwest::get(format!("{endpoint}/metrics"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
            .lines()
            .filter(|line| {
                line.starts_with("grpc_server_started_total{")
                    && methods
                        .iter()
                        .any(|method| line.contains(&format!("grpc_method=\"{method}\"")))
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

    #[tokio::test]
    #[ignore = "requires exclusive loopback ETCD_TEST_ENDPOINTS"]
    async fn one_ten_fifty_followers_do_not_multiply_discovery_reads_and_take_over() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").as_deref(), Ok("1"));
        assert!(endpoint.starts_with("http://127.0.0.1:") && !endpoint.contains(','));
        for replicas in [1, 10, 50] {
            let (_dir, cfg, store) = fixture().await;
            let policy = serde_json::to_vec(&Policy::from_config(&cfg)).unwrap();
            let leader = store.discovery_session(policy.clone()).await.unwrap();
            let key = format!("{}/demand-controller/leader-v2", cfg.etcd.prefix());
            become_leader(&store, &leader, &key).await.unwrap();
            let coordinator = store.merge_coordinator();
            for i in 0..256 {
                coordinator
                    .request_merge(&format!("table-{i:03}"))
                    .await
                    .unwrap();
            }
            let before_watches = rpc_count(&endpoint, &["Watch"]).await;
            let mut followers = tokio::task::JoinSet::new();
            for _ in 1..replicas {
                let session = store.discovery_session(policy.clone()).await.unwrap();
                let follower_store = store.clone();
                let follower_key = key.clone();
                followers.spawn(async move {
                    become_leader(&follower_store, &session, &follower_key)
                        .await
                        .unwrap();
                    session
                });
            }
            tokio::time::timeout(Duration::from_secs(5), async {
                while rpc_count(&endpoint, &["Watch"]).await < before_watches + replicas - 1 {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .unwrap();
            let baseline = rpc_count(&endpoint, &["Range", "Txn"]).await;
            let cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
            tokio::time::sleep(if replicas == 50 {
                Duration::from_secs(16)
            } else {
                Duration::from_millis(250)
            })
            .await;
            let reads = rpc_count(&endpoint, &["Range", "Txn"]).await - baseline;
            // Two request pages plus execution/failure/member/queue/running pages. All
            // membership records fit on one page even with fifty masters.
            assert_eq!(
                reads,
                RANGES.len() as u64,
                "discovery reads grew with replicas={replicas}"
            );
            assert_eq!(cache.entries.len(), 256 + replicas as usize);
            eprintln!("replicas={replicas} demand_keys=256 discovery_range_txn={reads} follower_discovery_range_txn=0");
            let before = Instant::now();
            store.release_coordination_lock(leader).await.unwrap();
            if replicas > 1 {
                let winner = tokio::time::timeout(Duration::from_secs(5), followers.join_next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                // Exactly one winner; the other followers return to watching.
                assert!(
                    tokio::time::timeout(Duration::from_millis(100), followers.join_next())
                        .await
                        .is_err()
                );
                let replacement = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
                assert_eq!(
                    replacement
                        .entries
                        .keys()
                        .filter(|k| k.starts_with(
                            prefix_key(cfg.etcd.prefix(), "merge-requests").as_bytes()
                        ))
                        .count(),
                    256
                );
                eprintln!(
                    "replicas={replicas} takeover_ms={} all_256_requests_rebuilt=true",
                    before.elapsed().as_millis()
                );
                followers.abort_all();
                while followers.join_next().await.is_some() {}
                store.release_coordination_lock(winner).await.unwrap();
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS and ETCD_TEST_ALLOW_COMPACTION=1"]
    async fn compaction_forces_consistent_relist_and_retains_new_demand() {
        assert_eq!(
            std::env::var("ETCD_TEST_ALLOW_COMPACTION").as_deref(),
            Ok("1")
        );
        let (_dir, cfg, store) = fixture().await;
        let coordinator = store.merge_coordinator();
        coordinator.request_merge("before").await.unwrap();
        let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        let before = lance_context_merge::execution_key(cfg.etcd.prefix(), "before")
            .replace("/merge-executions/", "/merge-requests/");
        store
            .etcd_client()
            .clone()
            .delete(before.clone(), None)
            .await
            .unwrap();
        coordinator.request_merge("after").await.unwrap();
        let revision = store
            .etcd_client()
            .clone()
            .get("unrelated", None)
            .await
            .unwrap()
            .header()
            .unwrap()
            .revision();
        store
            .etcd_client()
            .clone()
            .compact(revision, None)
            .await
            .unwrap();
        let mut stream = watch(store.etcd_client(), cfg.etcd.prefix(), &cache)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let response = stream.message().await.unwrap().unwrap();
                if apply_response(&mut cache, &response).is_err_and(|e| e == "compacted") {
                    break;
                }
            }
        })
        .await
        .unwrap();
        let cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        assert!(!cache.entries.contains_key(before.as_bytes()));
        let after = lance_context_merge::execution_key(cfg.etcd.prefix(), "after")
            .replace("/merge-executions/", "/merge-requests/");
        assert!(cache.entries.contains_key(after.as_bytes()));
    }
    async fn until(mut predicate: impl AsyncFnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(12), async {
            while !predicate().await {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn running_controller_preserves_union_and_recovers_from_leader_abort() {
        let (_dir, mut cfg, store) = fixture().await;
        cfg.merge_rollout.fence_legacy = false;
        cfg.worker_endpoints.clear();
        cfg.merge_wal_interval_secs = 0;
        cfg.merge_rollout.owned_targets = vec!["narrow".into()];
        cfg.stats_scan_interval_secs = 0;
        cfg.compaction_interval_secs = 0;
        let first = crate::state::MasterState::new(cfg.clone()).await.unwrap();
        let mut broad = cfg.clone();
        broad.worker_endpoints = vec!["http://worker-a:8080".into()];
        broad.merge_wal_interval_secs = 30;
        let second = crate::state::MasterState::new(broad).await.unwrap();
        let key = format!("{}/demand-controller/leader-v2", cfg.etcd.prefix());
        let controller_a = spawn(&first);
        until(async || {
            !store
                .etcd_client()
                .clone()
                .get(key.clone(), None)
                .await
                .unwrap()
                .kvs()
                .is_empty()
        })
        .await;
        let original = store
            .etcd_client()
            .clone()
            .get(key.clone(), None)
            .await
            .unwrap()
            .kvs()[0]
            .value()
            .to_vec();
        let controller_b = spawn(&second);
        let member_prefix = prefix_key(cfg.etcd.prefix(), "demand-controller/members");
        until(async || {
            store
                .etcd_client()
                .clone()
                .get(member_prefix.clone(), Some(GetOptions::new().with_prefix()))
                .await
                .unwrap()
                .kvs()
                .len()
                == 2
        })
        .await;
        store
            .merge_coordinator()
            .request_merge("broad-before")
            .await
            .unwrap();
        until(async || {
            store
                .get_active_id(TaskKind::MergeWal, "broad-before")
                .await
                .unwrap()
                .is_some()
        })
        .await;
        // Abrupt task loss does not revoke the lease; this exercises actual
        // lease expiry, not just the explicit-release fast path above.
        controller_a.abort();
        assert!(controller_a.await.unwrap_err().is_cancelled());
        until(async || {
            store
                .etcd_client()
                .clone()
                .get(key.clone(), None)
                .await
                .unwrap()
                .kvs()
                .first()
                .is_some_and(|kv| kv.value() != original)
        })
        .await;
        store
            .merge_coordinator()
            .request_merge("broad-after")
            .await
            .unwrap();
        until(async || {
            store
                .get_active_id(TaskKind::MergeWal, "broad-after")
                .await
                .unwrap()
                .is_some()
        })
        .await;
        controller_b.abort();
        assert!(controller_b.await.unwrap_err().is_cancelled());
        assert_eq!(store.list().await.unwrap().len(), 2);
    }
    #[tokio::test]
    #[ignore = "requires exclusive loopback ETCD_TEST_ENDPOINTS"]
    async fn active_inventory_reconciles_without_rpc_and_watch_releases_suppression() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").as_deref(), Ok("1"));
        assert!(endpoint.starts_with("http://127.0.0.1:") && !endpoint.contains(','));
        let (_dir, cfg, store) = fixture().await;
        let session = store
            .discovery_session(serde_json::to_vec(&Policy::from_config(&cfg)).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let coordinator = store.merge_coordinator();
        for i in 0..64 {
            let target = format!("active-{i:03}");
            let mut execution = Execution::new(&target, "worker-a", "old", 60);
            execution.background = true;
            assert!(coordinator.reserve_background(&execution).await.unwrap());
            store
                .enqueue(TaskKind::MergeWal, &target, vec![])
                .await
                .unwrap();
        }
        let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        let before = rpc_count(&endpoint, &["Range", "Txn"]).await;
        for (key, entry) in &cache.entries {
            reconcile(&publisher, cfg.etcd.prefix(), &cache, key, entry)
                .await
                .unwrap();
        }
        let reads = rpc_count(&endpoint, &["Range", "Txn"]).await - before;
        // reserve_background emits a fresh writer hint as well as inventory.
        // Each hint is acknowledged by one transaction, with no advisory reads.
        assert_eq!(
            reads, 64,
            "writer hints should only need their acknowledgement transaction"
        );
        let mut stream = watch(store.etcd_client(), cfg.etcd.prefix(), &cache)
            .await
            .unwrap();
        next_events(&mut stream, &mut cache, |c| {
            !c.entries.keys().any(|key| {
                key.starts_with(prefix_key(cfg.etcd.prefix(), "merge-requests").as_bytes())
            })
        })
        .await;
        let before = rpc_count(&endpoint, &["Range", "Txn"]).await;
        for (key, entry) in &cache.entries {
            reconcile(&publisher, cfg.etcd.prefix(), &cache, key, entry)
                .await
                .unwrap();
        }
        let reads = rpc_count(&endpoint, &["Range", "Txn"]).await - before;
        assert_eq!(reads, 0, "active inventory must not repeat advisory reads");
        eprintln!("active_inventory_rows=64 writer_hint_acknowledgements=64 repeated_reconciliation_range_txn={reads}");
        let claim = store
            .claim_next_of_kinds(crate::task_store::TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let target = claim.task.target.clone();
        store
            .finish(claim, Ok("test metadata completion".into()))
            .await
            .unwrap();
        next_events(&mut stream, &mut cache, |c| {
            !c.has_active(TaskKind::MergeWal, &target)
        })
        .await;
        let key = lance_context_merge::execution_key(cfg.etcd.prefix(), &target);
        reconcile(
            &publisher,
            cfg.etcd.prefix(),
            &cache,
            key.as_bytes(),
            &cache.entries[key.as_bytes()],
        )
        .await
        .unwrap();
        assert!(store
            .get_active_id(TaskKind::MergeWal, &target)
            .await
            .unwrap()
            .is_some());
        store.release_coordination_lock(session).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn dependency_chain_does_not_hide_independent_recovery() {
        let (_dir, cfg, store) = fixture().await;
        let dependency = store
            .enqueue(TaskKind::IndexId, "dependency", vec![])
            .await
            .unwrap();
        store
            .enqueue(TaskKind::MergeWal, "waiting", vec![dependency.id])
            .await
            .unwrap();
        let cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        assert!(!cache.has_active(TaskKind::MergeWal, "waiting"));
        assert!(cache.has_active(TaskKind::IndexId, "dependency"));
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn running_dependent_index_suppresses_recovery_until_claim_finishes() {
        let (_dir, cfg, store) = fixture().await;
        let compact = store
            .enqueue(TaskKind::Compact, "table", vec![])
            .await
            .unwrap();
        let index = store
            .enqueue(TaskKind::IndexId, "table", vec![compact.id.clone()])
            .await
            .unwrap();
        let cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        assert!(
            !cache.has_active(TaskKind::IndexId, "table"),
            "queued dependency chains must not hide independent work"
        );
        let claim = store.claim_next().await.unwrap().unwrap();
        assert_eq!(claim.task.id, compact.id);
        store
            .finish(claim, Ok("compaction finished".into()))
            .await
            .unwrap();
        let claim = store.claim_next().await.unwrap().unwrap();
        assert_eq!(claim.task.id, index.id);
        let cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        let active = cache.has_active(TaskKind::IndexId, "table");
        // Always release the real lease, including on the regression's red run.
        store
            .finish(claim, Ok("index finished".into()))
            .await
            .unwrap();
        assert!(
            active,
            "an already running dependent index must suppress duplicate recovery"
        );
        let cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        assert!(!cache.has_active(TaskKind::IndexId, "table"));
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn cached_acknowledgement_preserves_newer_flush_and_running_followup() {
        let (_dir, cfg, store) = fixture().await;
        let session = store
            .discovery_session(serde_json::to_vec(&Policy::from_config(&cfg)).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let coordinator = store.merge_coordinator();
        store
            .enqueue(TaskKind::MergeWal, "table", vec![])
            .await
            .unwrap();
        coordinator.request_merge("table").await.unwrap();
        let cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        let queued = &cache
            .queued(cfg.etcd.prefix(), TaskKind::MergeWal, "table")
            .unwrap()
            .value;
        let old = coordinator
            .get_merge_request("table")
            .await
            .unwrap()
            .unwrap();
        coordinator.request_merge("table").await.unwrap();
        assert!(!publisher
            .acknowledge_cached_merge_request(&old, queued)
            .await
            .unwrap());
        let latest = coordinator
            .get_merge_request("table")
            .await
            .unwrap()
            .unwrap();
        assert!(latest.revision > old.revision);
        let claim = store
            .claim_next_of_kinds(crate::task_store::TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        assert!(!publisher
            .acknowledge_cached_merge_request(&latest, queued)
            .await
            .unwrap());
        publisher.enqueue_merge_request(&latest).await.unwrap();
        assert!(coordinator
            .get_merge_request("table")
            .await
            .unwrap()
            .is_some());
        store
            .finish(claim, Ok("completed earlier pass".into()))
            .await
            .unwrap();
        let successor = publisher
            .enqueue_merge_request(&latest)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(successor.state, TaskState::Queued);
        assert!(coordinator
            .get_merge_request("table")
            .await
            .unwrap()
            .is_none());
        store.release_coordination_lock(session).await.unwrap();
    }
}
