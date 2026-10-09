//! Leader-elected demand planner, shadow mode (P1.5).
//!
//! Spec: `docs/design/scheduler-p1-data-model.md` §1.2, §3. One master holds
//! `P/leader` under a lease and folds every `P/demand-events/<hex>/<shard>`
//! record into a per-table `P/demand/<hex>` cache. The cache is a pure
//! function of the stored events (see `demand::TableDemand::fold`), so it can
//! be rebuilt from scratch by any leader and is never a source of truth.
//!
//! This module **places nothing and removes nothing**. Every existing loop
//! keeps running for every table. Its only outputs are the `demand/` cache,
//! the `/scheduler/demand` view and metrics. Scores, classes and shadow
//! placements come in later PRs once this cache is trusted.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use etcd_client::{Compare, CompareOp, GetOptions, PutOptions, Txn, TxnOp};
use lance_context_core::generate_id;
use lance_context_merge::demand::{DemandEvent, TableDemand};
use serde::{Deserialize, Serialize};

use crate::state::MasterState;

#[derive(Clone, Debug, clap::Args)]
pub struct PlannerConfig {
    /// Run the demand planner (leader election + demand cache). Shadow only:
    /// nothing is scheduled from it yet.
    #[arg(long, env = "PLANNER_ENABLED", default_value_t = false)]
    pub planner_enabled: bool,
    /// Full reconcile interval; between ticks the leader reacts to changes.
    #[arg(long, env = "PLANNER_RECONCILE_SECS", default_value_t = 30)]
    pub planner_reconcile_secs: u64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            planner_enabled: false,
            planner_reconcile_secs: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LeaderRecord {
    pub v: u32,
    pub token: String,
    pub instance: String,
    pub since_ms: i64,
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn hex(target: &str) -> String {
    target.bytes().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn unhex(encoded: &str) -> Option<String> {
    if !encoded.len().is_multiple_of(2) {
        return None;
    }
    let bytes: Option<Vec<u8>> = (0..encoded.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&encoded[i..i + 2], 16).ok())
        .collect();
    String::from_utf8(bytes?).ok()
}

/// A key/value/revision triple from a paged prefix read.
pub(crate) struct Kv {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub mod_revision: i64,
}

/// Byte budget for one prefix page. The etcd client caps a response at
/// 4 MiB; records are ~200–400 B but a shard record with a full
/// `sealed_times` map is ~1.5 KiB, so a key count is not a bound. Pages
/// are sized adaptively: start at `PAGE_KEYS`, and when a page comes back
/// over `PAGE_BYTES` halve the key limit for the next one (etcd does not
/// let a client ask for "N bytes"). A page that is still too big at the
/// minimum limit is a hard error, not a silent retry loop.
const PAGE_KEYS: i64 = 512;
const PAGE_MIN_KEYS: i64 = 8;
const PAGE_BYTES: usize = 1 << 20;

/// Byte budget for one write transaction. etcd's `--max-request-bytes`
/// defaults to 1.5 MiB; stay well under so a request is never rejected for
/// size. Op count is capped separately at etcd's `--max-txn-ops` (128).
const TXN_BYTES: usize = 768 * 1024;
const TXN_OPS: usize = 100;

/// Split `ops` into transactions that respect both the op-count and the
/// request-size limits. Each `TxnOp` is measured by its serialised value
/// size plus the key; an op larger than the budget on its own still gets a
/// transaction of its own.
pub(crate) fn chunk_ops(ops: Vec<(Vec<u8>, usize)>) -> Vec<Vec<usize>> {
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut bytes = 0usize;
    for (i, (_, size)) in ops.iter().enumerate() {
        if !current.is_empty() && (current.len() >= TXN_OPS || bytes + size > TXN_BYTES) {
            chunks.push(std::mem::take(&mut current));
            bytes = 0;
        }
        current.push(i);
        bytes += size;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Read every key under `prefix` in pages, all at one revision so the
/// result is a consistent snapshot. Returns the kvs and that revision.
pub(crate) async fn read_prefix_paged(
    client: &etcd_client::Client,
    prefix: &str,
) -> Result<(Vec<Kv>, i64), String> {
    let mut client = client.clone();
    let range_end = {
        let mut end = prefix.as_bytes().to_vec();
        for i in (0..end.len()).rev() {
            if end[i] != u8::MAX {
                end[i] += 1;
                end.truncate(i + 1);
                break;
            }
        }
        end
    };
    let mut out = Vec::new();
    let mut start = prefix.as_bytes().to_vec();
    let mut revision: i64 = 0;
    let mut limit = PAGE_KEYS;
    loop {
        let mut opts = GetOptions::new()
            .with_range(range_end.clone())
            .with_limit(limit);
        if revision > 0 {
            opts = opts.with_revision(revision);
        }
        let response = match client.get(start.clone(), Some(opts)).await {
            Ok(r) => r,
            // A response over the client's message cap surfaces as a gRPC
            // error; shrink the page and retry the same start key.
            Err(e) if limit > PAGE_MIN_KEYS && e.to_string().contains("message length") => {
                limit = (limit / 2).max(PAGE_MIN_KEYS);
                metrics::counter!("planner_page_shrink_total").increment(1);
                continue;
            }
            Err(e) => return Err(e.to_string()),
        };
        if revision == 0 {
            revision = response.header().map_or(0, |h| h.revision());
        }
        let page_bytes: usize = response
            .kvs()
            .iter()
            .map(|kv| kv.key().len() + kv.value().len())
            .sum();
        if page_bytes > PAGE_BYTES && limit > PAGE_MIN_KEYS {
            // Proactively shrink before we get close to the cap.
            limit = (limit / 2).max(PAGE_MIN_KEYS);
        } else if page_bytes < PAGE_BYTES / 4 && limit < PAGE_KEYS {
            limit = (limit * 2).min(PAGE_KEYS);
        }
        let more = response.more();
        let last = response.kvs().last().map(|kv| kv.key().to_vec());
        for kv in response.kvs() {
            out.push(Kv {
                key: kv.key().to_vec(),
                value: kv.value().to_vec(),
                mod_revision: kv.mod_revision(),
            });
        }
        let Some(last) = last else { break };
        if !more {
            break;
        }
        start = last;
        start.push(0);
    }
    Ok((out, revision))
}

/// Write `ops` under the leader token in size- and count-bounded txns.
async fn write_chunked(
    client: &mut etcd_client::Client,
    keys: &Keys,
    leader_token: &str,
    ops: Vec<SizedOp>,
    what: &str,
) -> Result<(), String> {
    let sizes: Vec<(Vec<u8>, usize)> = ops.iter().map(|o| (Vec::new(), o.size)).collect();
    let mut ops: Vec<Option<TxnOp>> = ops.into_iter().map(|o| Some(o.op)).collect();
    for chunk in chunk_ops(sizes) {
        let batch: Vec<TxnOp> = chunk.into_iter().filter_map(|i| ops[i].take()).collect();
        let response = client
            .txn(
                Txn::new()
                    .when(vec![Compare::value(
                        keys.leader_token().as_str(),
                        CompareOp::Equal,
                        leader_token,
                    )])
                    .and_then(batch),
            )
            .await
            .map_err(|e| e.to_string())?;
        if !response.succeeded() {
            return Err(format!("leadership lost during {what}"));
        }
    }
    Ok(())
}

/// A txn op with the request bytes it will cost.
pub(crate) struct SizedOp {
    op: TxnOp,
    size: usize,
}

impl SizedOp {
    fn put(key: String, value: Vec<u8>) -> Self {
        let size = key.len() + value.len() + 32;
        Self {
            op: TxnOp::put(key, value, None),
            size,
        }
    }
    fn delete(key: Vec<u8>) -> Self {
        let size = key.len() + 32;
        Self {
            op: TxnOp::delete(key, None),
            size,
        }
    }
}

pub(crate) struct Keys {
    prefix: String,
}

impl Keys {
    pub(crate) fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.trim_end_matches('/').to_string(),
        }
    }
    pub(crate) fn leader(&self) -> String {
        format!("{}/leader", self.prefix)
    }
    /// Bare token beside the record so planner writes can compare on it.
    pub(crate) fn leader_token(&self) -> String {
        format!("{}/leader-token", self.prefix)
    }
    pub(crate) fn events_prefix(&self) -> String {
        format!("{}/demand-events/", self.prefix)
    }
    pub(crate) fn demand_prefix(&self) -> String {
        format!("{}/demand/", self.prefix)
    }
    pub(crate) fn demand(&self, target: &str) -> String {
        format!("{}/demand/{}", self.prefix, hex(target))
    }
    /// `demand-events/<hex>/<shard>` → (target, shard).
    fn parse_event_key(&self, key: &str) -> Option<(String, String)> {
        let rest = key.strip_prefix(&self.events_prefix())?;
        let (encoded, shard) = rest.split_once('/')?;
        Some((unhex(encoded)?, shard.to_string()))
    }
}

/// Fold every stored event into per-table records. Pure; the planner's one
/// unit of work. Events whose key or body cannot be read are counted and
/// skipped, never fatal.
pub(crate) fn fold_events<'a>(
    keys: &Keys,
    kvs: impl Iterator<Item = (&'a [u8], &'a [u8], i64)>,
    now: i64,
) -> (BTreeMap<String, TableDemand>, usize) {
    let mut tables: BTreeMap<String, TableDemand> = BTreeMap::new();
    let mut skipped = 0usize;
    for (key, value, revision) in kvs {
        let Some((target, _shard)) =
            keys.parse_event_key(std::str::from_utf8(key).unwrap_or_default())
        else {
            skipped += 1;
            continue;
        };
        let Ok(event) = serde_json::from_slice::<DemandEvent>(value) else {
            skipped += 1;
            continue;
        };
        let table = tables.entry(target).or_default();
        if let Err(lance_context_merge::demand::Ignored::UnknownVersion(_)) =
            table.fold(&event, revision, now)
        {
            skipped += 1;
        }
    }
    (tables, skipped)
}

pub(crate) fn spawn(state: &Arc<MasterState>) {
    if !state.config.planner.planner_enabled {
        return;
    }
    let weak = Arc::downgrade(state);
    tokio::spawn(async move {
        loop {
            let Some(state) = weak.upgrade() else { return };
            match lead(&state).await {
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(%error, "planner leadership loop ended");
                    metrics::counter!("planner_leader_errors_total").increment(1);
                }
            }
            drop(state);
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

/// Campaign for leadership; while leader, run reconcile cycles until the
/// lease is lost or the process drains. Returns `Ok(true)` if this call held
/// leadership at all, `Ok(false)` if the campaign was lost.
async fn lead(state: &Arc<MasterState>) -> Result<bool, String> {
    let keys = Keys::new(&state.config.etcd.etcd_prefix);
    let mut client = state.task_store.etcd_client().clone();
    let ttl = state.config.etcd_lease_ttl_secs.max(5);
    let lease = client
        .lease_grant(ttl, None)
        .await
        .map_err(|e| e.to_string())?
        .id();
    let token = generate_id();
    let record = LeaderRecord {
        v: 1,
        token: token.clone(),
        instance: state.admission.status().executor_id,
        since_ms: now_ms(),
    };
    let won = client
        .txn(
            Txn::new()
                .when(vec![Compare::version(
                    keys.leader().as_str(),
                    CompareOp::Equal,
                    0,
                )])
                .and_then(vec![
                    TxnOp::put(
                        keys.leader(),
                        serde_json::to_vec(&record).unwrap(),
                        Some(PutOptions::new().with_lease(lease)),
                    ),
                    TxnOp::put(
                        keys.leader_token(),
                        token.clone(),
                        Some(PutOptions::new().with_lease(lease)),
                    ),
                ]),
        )
        .await
        .map_err(|e| e.to_string())?
        .succeeded();
    if !won {
        let _ = client.lease_revoke(lease).await;
        metrics::gauge!("planner_is_leader").set(0.0);
        return Ok(false);
    }
    metrics::gauge!("planner_is_leader").set(1.0);
    tracing::info!(token = %token, "planner leadership acquired");
    let (mut keeper, mut stream) = client
        .lease_keep_alive(lease)
        .await
        .map_err(|e| e.to_string())?;
    let keepalive_every = Duration::from_secs((ttl as u64 / 3).max(1));
    let reconcile_every = Duration::from_secs(state.config.planner.planner_reconcile_secs.max(5));
    let mut reconcile = tokio::time::interval(reconcile_every);
    let mut keepalive = tokio::time::interval(keepalive_every);
    let mut inflight: Option<tokio::task::JoinHandle<()>> = None;
    let result = loop {
        tokio::select! {
            _ = keepalive.tick() => {
                if keeper.keep_alive().await.is_err() {
                    break Err("lease keepalive failed".to_string());
                }
                match tokio::time::timeout(Duration::from_secs(5), stream.message()).await {
                    Ok(Ok(Some(response))) if response.ttl() > 0 => {}
                    _ => break Err("lease expired or keepalive stream closed".to_string()),
                }
            }
            _ = reconcile.tick() => {
                if !state.admission.status().accepting {
                    break Ok(true);
                }
                // Reconcile runs detached so a slow pass (13k tables, many
                // pages) can never starve this loop's lease keepalive. At
                // most one pass in flight; a tick that finds one running is
                // skipped and counted.
                if let Some(h) = &inflight {
                    if !h.is_finished() {
                        metrics::counter!("planner_reconcile_skipped_total").increment(1);
                        continue;
                    }
                }
                let st = state.clone();
                let k = Keys::new(&state.config.etcd.etcd_prefix);
                let t = token.clone();
                inflight = Some(tokio::spawn(async move {
                    if let Err(error) = reconcile_once(&st, &k, &t).await {
                        tracing::warn!(%error, "planner reconcile failed");
                        metrics::counter!("planner_reconcile_errors_total").increment(1);
                    }
                }));
            }
        }
    };
    metrics::gauge!("planner_is_leader").set(0.0);
    if let Some(h) = inflight.take() {
        h.abort();
    }
    drop(keeper);
    let _ = client.lease_revoke(lease).await;
    tracing::info!(token = %token, ?result, "planner leadership released");
    result
}

/// One reconcile: read every event, fold, write every changed table record
/// under a leader-token compare, delete records for tables with no events.
pub(crate) async fn reconcile_once(
    state: &Arc<MasterState>,
    keys: &Keys,
    leader_token: &str,
) -> Result<usize, String> {
    let started = std::time::Instant::now();
    let mut client = state.task_store.etcd_client().clone();
    let (events, header_revision) =
        read_prefix_paged(state.task_store.etcd_client(), &keys.events_prefix()).await?;
    let now = now_ms();
    let (tables, skipped) = fold_events(
        keys,
        events
            .iter()
            .map(|kv| (kv.key.as_slice(), kv.value.as_slice(), kv.mod_revision)),
        now,
    );
    metrics::gauge!("planner_events").set(events.len() as f64);
    if skipped > 0 {
        metrics::counter!("planner_events_skipped_total").increment(skipped as u64);
    }
    let (existing, _) =
        read_prefix_paged(state.task_store.etcd_client(), &keys.demand_prefix()).await?;
    let existing_count = existing.len();
    let mut existing_by_key: BTreeMap<Vec<u8>, TableDemand> = BTreeMap::new();
    for kv in &existing {
        if let Ok(record) = serde_json::from_slice::<TableDemand>(&kv.value) {
            existing_by_key.insert(kv.key.clone(), record);
        }
    }
    let mut ops: Vec<SizedOp> = Vec::new();
    let mut written = 0usize;
    let mut total_pending = 0u64;
    let folded: BTreeMap<String, TableDemand> = tables.clone();
    for (target, mut table) in tables {
        total_pending += table.pending_generations();
        let key = keys.demand(&target);
        // Only the fold's inputs decide equality; the clock does not.
        let unchanged = existing_by_key
            .remove(key.as_bytes())
            .is_some_and(|mut old| {
                old.updated_ms = 0;
                let mut cmp = table.clone();
                cmp.updated_ms = 0;
                old == cmp
            });
        if unchanged {
            continue;
        }
        table.updated_ms = now;
        ops.push(SizedOp::put(key, serde_json::to_vec(&table).unwrap()));
        written += 1;
    }
    // Tables that no longer have any events: drop the cache row.
    for stale_key in existing_by_key.into_keys() {
        ops.push(SizedOp::delete(stale_key));
        written += 1;
    }
    // Always confirm leadership, even with nothing to write, so a deposed
    // leader learns it is deposed on the very next tick rather than when it
    // next has a change.
    let still_leader = client
        .get(keys.leader_token(), None)
        .await
        .map_err(|e| e.to_string())?
        .kvs()
        .first()
        .is_some_and(|kv| kv.value() == leader_token.as_bytes());
    if !still_leader {
        return Err("leadership lost during reconcile".into());
    }
    // Chunk by both op count (etcd --max-txn-ops) and request bytes
    // (--max-request-bytes); each txn guarded by the leader token so a
    // deposed leader cannot write a stale cache.
    write_chunked(&mut client, keys, leader_token, ops, "reconcile").await?;
    // Executor view: heartbeats plus reservations, then the shadow pass.
    let executors_keys = crate::executors::Keys::new(&state.config.etcd.etcd_prefix);
    match crate::executors::load_headroom(&client, &executors_keys).await {
        Ok(headroom) => {
            metrics::gauge!("planner_executors").set(headroom.len() as f64);
            let reserved: u64 = headroom.values().map(|h| h.bytes_reserved).sum();
            metrics::gauge!("planner_reserved_bytes_total").set(reserved as f64);
            if let Err(error) = shadow_pass(
                state,
                keys,
                &executors_keys,
                leader_token,
                &folded,
                headroom,
                now,
            )
            .await
            {
                tracing::warn!(%error, "planner shadow pass failed");
                metrics::counter!("planner_shadow_errors_total").increment(1);
            }
        }
        Err(error) => {
            tracing::warn!(%error, "planner could not load executor headroom");
        }
    }
    metrics::gauge!("planner_tables").set(existing_count as f64);
    metrics::gauge!("planner_pending_generations_total").set(total_pending as f64);
    metrics::histogram!("planner_reconcile_seconds").record(started.elapsed().as_secs_f64());
    metrics::gauge!("planner_last_revision").set(header_revision as f64);
    tracing::debug!(
        written,
        skipped,
        revision = header_revision,
        "planner reconcile"
    );
    Ok(written)
}

/// Shadow placement: score every table, order per design §4.2, place
/// against headroom, and record what *would* be dispatched as `Shadow`
/// assignments that no executor reads. Then compare with reality: for each
/// table the planner would place, is a MergeWal task already active or
/// queued? For each table it would not, is one running anyway? Each
/// mismatch is a disagreement; the counter has to approach zero before P2.
///
/// Shadow assignments are replaced wholesale every pass (they are a view,
/// not a reservation anyone binds), guarded by the leader token.
async fn shadow_pass(
    state: &Arc<MasterState>,
    keys: &Keys,
    executors_keys: &crate::executors::Keys,
    leader_token: &str,
    tables: &BTreeMap<String, TableDemand>,
    mut headroom: BTreeMap<String, crate::executors::Headroom>,
    now: i64,
) -> Result<(), String> {
    use crate::executors::{Assignment, AssignmentState};
    use crate::scoring::{planner_order, score_merge, Class, MergePolicy};
    use lance_context_api::TaskKind;

    let policy = MergePolicy {
        min_generations: state.config.merge_wal_min_generations.max(1) as u64,
        ..MergePolicy::default()
    };
    let mut scored: Vec<_> = tables
        .iter()
        .filter_map(|(target, demand)| score_merge(target, demand, &policy, now))
        .collect();
    scored.sort_by(planner_order);
    for s in &scored {
        metrics::gauge!("planner_merge_score", "target" => s.target.clone()).set(s.score);
    }
    let by_class = |c: Class| scored.iter().filter(|s| s.class == c).count();
    metrics::gauge!("planner_units", "class" => "critical").set(by_class(Class::Critical) as f64);
    metrics::gauge!("planner_units", "class" => "normal").set(by_class(Class::Normal) as f64);
    metrics::gauge!("planner_units", "class" => "tail").set(by_class(Class::Tail) as f64);

    // Only Shadow records from previous passes are replaced; nothing else
    // under assignments/ is touched.
    let mut client = state.task_store.etcd_client().clone();
    let (existing, _) = read_prefix_paged(
        state.task_store.etcd_client(),
        &executors_keys.assignments_prefix(),
    )
    .await?;
    let mut ops: Vec<SizedOp> = existing
        .iter()
        .filter(|kv| {
            serde_json::from_slice::<Assignment>(&kv.value)
                .is_ok_and(|a| a.state == AssignmentState::Shadow)
        })
        .map(|kv| SizedOp::delete(kv.key.clone()))
        .collect();
    // Headroom already counts previous Shadow records; release them in the
    // arithmetic too, since they are about to be replaced.
    for h in headroom.values_mut() {
        h.slots_reserved.clear();
        h.bytes_reserved = 0;
        h.assignments = 0;
    }
    for kv in &existing {
        if let Ok(a) = serde_json::from_slice::<Assignment>(&kv.value) {
            if a.state != AssignmentState::Shadow && a.holds_capacity() {
                if let Some(h) = headroom.get_mut(&a.executor) {
                    for (k, n) in &a.reserved_slots {
                        *h.slots_reserved.entry(*k).or_insert(0) += n;
                    }
                    h.bytes_reserved += a.reserved_bytes;
                    h.assignments += 1;
                }
            }
        }
    }

    let mut would_place: Vec<(String, String)> = Vec::new();
    let mut unplaceable = 0usize;
    for s in &scored {
        if s.class == Class::Tail {
            continue; // the real sweeps do not touch tails either
        }
        // Estimated cost. ROLLOUT_APPEND_MAX_BYTES is the *target* for one
        // pass, not a ceiling: the append path reads a whole generation
        // before checking it, so the largest single generation sets the
        // real floor, and buffers cost ~2x decoded. The planner does not
        // know per-generation sizes - a scan reports bytes-through-sealed,
        // from which only an *average* per generation follows, and an
        // average is not a maximum. Until per-generation sizes are
        // reported the only honest reservation for a table whose largest
        // generation is unknown is the executor's whole local budget. The
        // average is recorded as a diagnostic so the gap between "what we
        // reserve" and "what we would reserve with real sizes" is visible.
        let per_pass_target = state.config.append.rollout_append_max_bytes as u64;
        let local_budget = state.config.append.rollout_append_local_memory_bytes as u64;
        let average_generation = if s.pending_bytes > 0 && s.pending_generations > 0 {
            s.pending_bytes / s.pending_generations
        } else {
            0
        };
        metrics::histogram!("planner_shadow_average_generation_bytes")
            .record(average_generation as f64);
        let expected_bytes = local_budget.max(per_pass_target * 2).max(1);
        metrics::counter!(
            "planner_shadow_reservation_basis_total",
            "basis" => if s.pending_bytes == 0 { "no_size_data" } else { "average_only" }
        )
        .increment(1);
        let pick = headroom
            .values_mut()
            .filter(|h| h.fits(TaskKind::MergeWal, expected_bytes))
            .max_by_key(|h| h.bytes_free());
        let Some(h) = pick else {
            unplaceable += 1;
            continue;
        };
        *h.slots_reserved.entry(TaskKind::MergeWal).or_insert(0) += 1;
        h.bytes_reserved += expected_bytes;
        h.assignments += 1;
        let unit_id = generate_id();
        let a = Assignment {
            v: crate::executors::SCHEMA_VERSION,
            unit_id: unit_id.clone(),
            kind: TaskKind::MergeWal,
            target: s.target.clone(),
            needs_write_turn: true,
            executor: h.executor.clone(),
            planner_token: leader_token.to_string(),
            reserved_slots: [(TaskKind::MergeWal, 1)].into_iter().collect(),
            reserved_bytes: expected_bytes,
            state: AssignmentState::Shadow,
            created_ms: now,
            bind_deadline_ms: now + 30_000,
        };
        ops.push(SizedOp::put(
            executors_keys.assignment(&s.target, &unit_id),
            serde_json::to_vec(&a).unwrap(),
        ));
        would_place.push((s.target.clone(), h.executor.clone()));
    }
    metrics::gauge!("planner_shadow_placements").set(would_place.len() as f64);
    metrics::gauge!("planner_shadow_unplaceable").set(unplaceable as f64);

    write_chunked(&mut client, keys, leader_token, ops, "shadow pass").await?;

    // Comparison with the live system. These are **observations, not a
    // takeover gate**: the shadow planner has only demand and capacity,
    // while the real queue also reflects ownership, eligibility, cooldowns
    // and executors this planner cannot see. Reality is read once, in
    // pages, as two separate sets - queued (discovery happened) and running
    // (execution happened) - and the comparison is **per due table**, not
    // per placement, so a table the planner wanted but could not place is
    // still compared on discovery. If reality cannot be read the pass
    // records nothing rather than zeros.
    //
    //  discovery_miss   due (non-tail) demand, but reality has this table
    //                   neither queued nor running: the real discovery
    //                   loops have not noticed it.
    //  queued_only      due demand that reality has queued but not running:
    //                   discovered, waiting on admission or capacity.
    //  running          due demand that reality is executing.
    //  placement_gap    due demand the planner could not place for lack of
    //                   headroom (independent of reality): the planner's
    //                   capacity model vs. the real pools.
    //  phantom_task     reality has this table queued or running with no
    //                   scored demand at all: a lost demand event, or work
    //                   the planner would not have scheduled.
    //
    // The summed `scheduler_shadow_disagreements_total` is kept only for
    // continuity; read the components.
    let (queued, running) = match state
        .task_store
        .list_queued_and_running_targets(TaskKind::MergeWal)
        .await
    {
        Ok(sets) => sets,
        Err(error) => {
            tracing::warn!(%error, "shadow comparison skipped: could not read task queue");
            metrics::counter!("planner_shadow_comparison_skipped_total").increment(1);
            return Ok(());
        }
    };
    let queued: std::collections::BTreeSet<&str> = queued.iter().map(String::as_str).collect();
    let running: std::collections::BTreeSet<&str> = running.iter().map(String::as_str).collect();
    let placed: std::collections::BTreeSet<&str> =
        would_place.iter().map(|(t, _)| t.as_str()).collect();
    let mut discovery_miss = 0u64;
    let mut queued_only = 0u64;
    let mut running_n = 0u64;
    let mut placement_gap_n = 0u64;
    let mut scored_targets: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for s in &scored {
        scored_targets.insert(s.target.as_str());
        if s.class == Class::Tail {
            continue;
        }
        let t = s.target.as_str();
        if running.contains(t) {
            running_n += 1;
        } else if queued.contains(t) {
            queued_only += 1;
        } else {
            discovery_miss += 1;
            tracing::debug!(target = %t, class = ?s.class, score = s.score,
                "shadow: due demand, reality has no task queued or running");
        }
        if !placed.contains(t) {
            placement_gap_n += 1;
        }
    }
    let mut phantom_task = 0u64;
    for t in queued.union(&running) {
        if !scored_targets.contains(t) {
            phantom_task += 1;
            tracing::debug!(target = %t,
                "shadow: reality has a MergeWal task with no scored demand");
        }
    }
    for (label, n) in [
        ("discovery_miss", discovery_miss),
        ("queued_only", queued_only),
        ("running", running_n),
        ("placement_gap", placement_gap_n),
        ("phantom_task", phantom_task),
    ] {
        metrics::gauge!("planner_shadow_comparison", "outcome" => label).set(n as f64);
    }
    let disagreements = discovery_miss + phantom_task;
    metrics::counter!("scheduler_shadow_disagreements_total", "kind" => "merge_wal")
        .increment(disagreements);
    metrics::gauge!("planner_shadow_disagreements_last_pass").set(disagreements as f64);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_merge::demand::{EventSource, SCHEMA_VERSION};

    fn event(shard: &str, sealed: u64, merged: Option<u64>) -> Vec<u8> {
        serde_json::to_vec(&DemandEvent {
            v: SCHEMA_VERSION,
            shard: shard.into(),
            sealed_through: sealed,
            sealed_bytes_through: 0,
            merged_through: merged,
            flushed_at_ms: 1,
            writer_epoch: 1,
            source: EventSource::Writer,
            merged_epoch: None,
            sealed_times: Default::default(),
        })
        .unwrap()
    }

    #[test]
    fn fold_events_groups_by_table_and_skips_garbage() {
        let keys = Keys::new("/p");
        let a1 = format!("/p/demand-events/{}/s1", hex("alpha"));
        let a2 = format!("/p/demand-events/{}/s2", hex("alpha"));
        let b1 = format!("/p/demand-events/{}/s1", hex("generic:beta"));
        let bad_key = "/p/demand-events/zz/s1".to_string();
        let bad_body = format!("/p/demand-events/{}/s9", hex("alpha"));
        let future = serde_json::json!({"v": 999, "shard": "s", "sealed_through": 1,
            "flushed_at_ms": 0, "writer_epoch": 1, "source": "writer"})
        .to_string();
        let future_key = format!("/p/demand-events/{}/s1", hex("gamma"));
        let kvs: Vec<(Vec<u8>, Vec<u8>, i64)> = vec![
            (a1.into_bytes(), event("s1", 10, Some(4)), 1),
            (a2.into_bytes(), event("s2", 3, None), 2),
            (b1.into_bytes(), event("s1", 7, Some(7)), 3),
            (bad_key.into_bytes(), event("s1", 1, None), 4),
            (bad_body.into_bytes(), b"not json".to_vec(), 5),
            (future_key.into_bytes(), future.into_bytes(), 6),
        ];
        let (tables, skipped) = fold_events(
            &keys,
            kvs.iter().map(|(k, v, r)| (k.as_slice(), v.as_slice(), *r)),
            0,
        );
        assert_eq!(skipped, 3, "bad key, bad body, future version");
        assert_eq!(tables.len(), 3, "alpha, generic:beta, gamma (empty)");
        assert_eq!(tables["alpha"].pending_generations(), 6 + 3);
        assert_eq!(tables["alpha"].shards.len(), 2);
        assert_eq!(tables["generic:beta"].pending_generations(), 0);
        assert_eq!(tables["alpha"].observed_revision, 2);
        assert!(tables["gamma"].shards.is_empty());
    }

    #[test]
    fn hex_round_trips_and_rejects_odd_input() {
        for s in ["alpha", "generic:beta", "", "ünïcödé"] {
            assert_eq!(unhex(&hex(s)).as_deref(), Some(s));
        }
        assert_eq!(unhex("abc"), None);
        assert_eq!(unhex("zz"), None);
    }

    /// Spec test 4 / review rounds 5 and 7: leader failover. A first leader
    /// builds the cache and shadow placements, then loses its lease. A
    /// second master, with nothing but etcd, must (a) win, (b) rebuild an
    /// identical cache from the same events, (c) rebuild identical headroom
    /// from heartbeats and durable assignments, and (d) reach the same
    /// shadow placement decisions. The deposed leader must be refused on
    /// its next write. Nothing lives only in a leader's memory.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn leader_failover_rebuilds_identical_state_and_fences_the_old_leader() {
        use crate::executors::{Assignment, AssignmentState};
        use crate::state::MasterState;
        use clap::Parser;
        use lance_context_api::TaskKind;
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = crate::config::MasterConfig::parse_from([
            "test",
            "--data-dir",
            dir.path().to_str().unwrap(),
        ]);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/planner-failover/{}", generate_id());
        cfg.planner.planner_enabled = true;
        cfg.planner.planner_reconcile_secs = 1;
        cfg.etcd_lease_ttl_secs = 5;
        cfg.merge_wal_concurrency = 2;
        cfg.stats_scan_interval_secs = 0;
        cfg.compaction_interval_secs = 0;
        cfg.merge_wal_interval_secs = 0;
        let a = MasterState::new(cfg.clone()).await.unwrap();
        let b = MasterState::new(cfg.clone()).await.unwrap();
        let keys = Keys::new(&cfg.etcd.etcd_prefix);
        let ekeys = crate::executors::Keys::new(&cfg.etcd.etcd_prefix);
        let client = a.task_store.etcd_client().clone();

        // Demand: three tables, mixed shapes.
        let coordinator = a.task_store.merge_coordinator();
        for (target, shard, sealed, merged) in [
            ("hot", "s1", 400u64, Some(40u64)),
            ("hot", "s2", 30, None),
            ("warm", "s1", 20, Some(4)),
            ("idle", "s1", 7, Some(7)),
        ] {
            coordinator
                .publish_demand_event(
                    target,
                    &DemandEvent {
                        v: SCHEMA_VERSION,
                        shard: shard.into(),
                        sealed_through: sealed,
                        sealed_bytes_through: sealed * 1_000_000,
                        merged_through: merged,
                        flushed_at_ms: 1_700_000_000_000,
                        writer_epoch: 1,
                        source: EventSource::Writer,
                        merged_epoch: None,
                        sealed_times: Default::default(),
                    },
                )
                .await
                .unwrap();
        }
        // Both masters heartbeat as executors throughout.
        let hb_a = tokio::spawn({
            let a = a.clone();
            async move { crate::executors::heartbeat_loop_for_test(&a).await }
        });
        let hb_b = tokio::spawn({
            let b = b.clone();
            async move { crate::executors::heartbeat_loop_for_test(&b).await }
        });
        // A durable non-shadow assignment, as a bound unit would leave.
        let bound = Assignment {
            v: crate::executors::SCHEMA_VERSION,
            unit_id: "bound-1".into(),
            kind: TaskKind::MergeWal,
            target: "elsewhere".into(),
            needs_write_turn: true,
            executor: a.admission.status().executor_id.clone(),
            planner_token: "old".into(),
            reserved_slots: [(TaskKind::MergeWal, 1)].into_iter().collect(),
            reserved_bytes: 1 << 20,
            state: AssignmentState::Bound,
            created_ms: 0,
            bind_deadline_ms: 0,
        };
        client
            .clone()
            .put(
                ekeys.assignment("elsewhere", &bound.unit_id),
                serde_json::to_vec(&bound).unwrap(),
                None,
            )
            .await
            .unwrap();

        // Leader 1 (master a) campaigns alone and completes a pass.
        let lead_a = tokio::spawn({
            let a = a.clone();
            async move { lead(&a).await }
        });
        let snapshot = |client: etcd_client::Client, keys: Keys, ekeys: crate::executors::Keys| async move {
            let (cache, _) = read_prefix_paged(&client, &keys.demand_prefix())
                .await
                .unwrap();
            let mut demand: BTreeMap<String, TableDemand> = BTreeMap::new();
            for kv in cache {
                let mut t: TableDemand = serde_json::from_slice(&kv.value).unwrap();
                t.updated_ms = 0;
                demand.insert(String::from_utf8_lossy(&kv.key).to_string(), t);
            }
            let (asg, _) = read_prefix_paged(&client, &ekeys.assignments_prefix())
                .await
                .unwrap();
            let mut shadows: Vec<(String, String, u64)> = asg
                .iter()
                .filter_map(|kv| serde_json::from_slice::<Assignment>(&kv.value).ok())
                .filter(|a| a.state == AssignmentState::Shadow)
                .map(|a| (a.target, a.executor, a.reserved_bytes))
                .collect();
            shadows.sort();
            let headroom = crate::executors::load_headroom(&client, &ekeys)
                .await
                .unwrap();
            (demand, shadows, headroom)
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let first = loop {
            let snap = snapshot(
                client.clone(),
                Keys::new(&cfg.etcd.etcd_prefix),
                crate::executors::Keys::new(&cfg.etcd.etcd_prefix),
            )
            .await;
            if snap.0.len() == 3 && !snap.1.is_empty() && snap.2.len() == 2 {
                break snap;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "leader 1 never completed a pass: {:?}",
                (snap.0.len(), snap.1.len(), snap.2.len())
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        };
        let token_1: String = String::from_utf8(
            client
                .clone()
                .get(keys.leader_token(), None)
                .await
                .unwrap()
                .kvs()[0]
                .value()
                .to_vec(),
        )
        .unwrap();
        assert_eq!(
            first.1.len(),
            2,
            "hot and warm are due; idle is not: {:?}",
            first.1
        );

        // Leader 1 dies without releasing: abort the task so no revoke runs,
        // then let the lease expire naturally.
        lead_a.abort();
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            if client
                .clone()
                .get(keys.leader(), None)
                .await
                .unwrap()
                .kvs()
                .is_empty()
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "leader 1 lease never expired"
            );
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        // The deposed leader's token can no longer write.
        let err = reconcile_once(&a, &keys, &token_1).await.unwrap_err();
        assert!(err.contains("leadership lost"), "{err}");

        // Leader 2 (master b) campaigns with only etcd to go on.
        let lead_b = tokio::spawn({
            let b = b.clone();
            async move { lead(&b).await }
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            let lk = client.clone().get(keys.leader(), None).await.unwrap();
            if let Some(kv) = lk.kvs().first() {
                let rec: LeaderRecord = serde_json::from_slice(kv.value()).unwrap();
                if rec.token != token_1 {
                    break;
                }
            }
            assert!(std::time::Instant::now() < deadline, "leader 2 never won");
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        // Wait for leader 2's first pass to replace the shadow set, then compare.
        tokio::time::sleep(Duration::from_millis(2500)).await;
        let second = snapshot(
            client.clone(),
            Keys::new(&cfg.etcd.etcd_prefix),
            crate::executors::Keys::new(&cfg.etcd.etcd_prefix),
        )
        .await;
        assert_eq!(second.0, first.0, "demand cache differs after failover");
        assert_eq!(second.1, first.1, "shadow placements differ after failover");
        // Headroom: same executors, same durable (non-shadow) reservations.
        for (id, h1) in &first.2 {
            let h2 = &second.2[id];
            assert_eq!(h2.slots_total, h1.slots_total, "{id}");
            assert_eq!(h2.bytes_total, h1.bytes_total, "{id}");
        }
        let durable = |h: &BTreeMap<String, crate::executors::Headroom>| -> u64 {
            // The Bound assignment's bytes survive on master a's row in both.
            h[&a.admission.status().executor_id].bytes_reserved
        };
        assert!(durable(&first.2) >= 1 << 20);
        assert!(
            durable(&second.2) >= 1 << 20,
            "durable reservation lost on failover"
        );
        lead_b.abort();
        hb_a.abort();
        hb_b.abort();
        let _ = client
            .clone()
            .delete(
                cfg.etcd.etcd_prefix.as_str(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await;
    }

    /// Round-5 finding 1 at production shape: 13,405 tables, two shards
    /// each (about 12 MB of events, three times the etcd client's 4 MiB
    /// message cap). A single-shot prefix read fails; the paged read must
    /// return every record at one revision, the fold must count them all,
    /// and a reconcile must complete while the leader keeps renewing its
    /// lease (the slow-reconcile-starves-keepalive finding).
    /// Seeds ~27k keys and reads them back several times; on a shared
    /// single-node etcd this starves concurrently running tests with
    /// `request timed out`. Run alone: `cargo test -p lance-context-master
    /// --lib reconcile_at_production_scale -- --ignored --test-threads=1`.
    /// Gated on `PLANNER_SCALE_TEST=1` so the ordinary `--ignored` sweep in
    /// CI does not include it.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS and PLANNER_SCALE_TEST=1; run alone"]
    async fn reconcile_at_production_scale_pages_reads_and_keeps_the_lease() {
        if std::env::var("PLANNER_SCALE_TEST").is_err() {
            eprintln!("PLANNER_SCALE_TEST unset; skipping");
            return;
        }
        use crate::state::MasterState;
        use clap::Parser;
        const TABLES: usize = 13_405;
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = crate::config::MasterConfig::parse_from([
            "test",
            "--data-dir",
            dir.path().to_str().unwrap(),
        ]);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/planner-scale/{}", generate_id());
        let _cleanup = PrefixCleanup(
            etcd_client::Client::connect(cfg.etcd.etcd_endpoints.clone(), None)
                .await
                .unwrap(),
            cfg.etcd.etcd_prefix.clone(),
        );
        cfg.planner.planner_enabled = true;
        cfg.planner.planner_reconcile_secs = 5;
        cfg.etcd_lease_ttl_secs = 5;
        cfg.stats_scan_interval_secs = 0;
        cfg.compaction_interval_secs = 0;
        cfg.merge_wal_interval_secs = 0;
        let state = MasterState::new(cfg.clone()).await.unwrap();
        let keys = Keys::new(&cfg.etcd.etcd_prefix);
        let mut client = state.task_store.etcd_client().clone();

        // Seed events directly, in batched txns, with realistic bodies.
        let mut ops = Vec::with_capacity(128);
        for i in 0..TABLES {
            let target = format!("exp-{i:05}");
            let pending = (i % 50) as u64; // 354-ish tables with real backlog is
                                           // fewer than this; worst case is fine
            for shard in [
                "aaaaaaaa-0000-0000-0000-000000000001",
                "aaaaaaaa-0000-0000-0000-000000000002",
            ] {
                let event = DemandEvent {
                    v: SCHEMA_VERSION,
                    shard: shard.into(),
                    sealed_through: 1000 + pending,
                    sealed_bytes_through: 64 << 20,
                    merged_through: Some(1000),
                    flushed_at_ms: 1_700_000_000_000,
                    writer_epoch: 3,
                    source: EventSource::Scan,
                    merged_epoch: None,
                    sealed_times: (1001..=1000 + pending)
                        .map(|g| (g, 1_700_000_000_000 + g as i64))
                        .collect(),
                };
                ops.push(TxnOp::put(
                    format!("{}{}/{shard}", keys.events_prefix(), hex(&target)),
                    serde_json::to_vec(&event).unwrap(),
                    None,
                ));
                if ops.len() == 100 {
                    client
                        .txn(Txn::new().and_then(std::mem::take(&mut ops)))
                        .await
                        .unwrap();
                }
            }
        }
        if !ops.is_empty() {
            client.txn(Txn::new().and_then(ops)).await.unwrap();
        }

        // A single unpaged read of this prefix exceeds the client limit.
        let single = client
            .get(keys.events_prefix(), Some(GetOptions::new().with_prefix()))
            .await;
        assert!(
            single.is_err(),
            "expected the unpaged read to exceed the message cap; got {} kvs",
            single.map(|r| r.kvs().len()).unwrap_or(0)
        );

        // The paged read returns everything at one revision.
        let (kvs, revision) = read_prefix_paged(&client, &keys.events_prefix())
            .await
            .unwrap();
        assert_eq!(kvs.len(), TABLES * 2);
        assert!(revision > 0);
        let (tables, skipped) = fold_events(
            &keys,
            kvs.iter()
                .map(|kv| (kv.key.as_slice(), kv.value.as_slice(), kv.mod_revision)),
            0,
        );
        assert_eq!(skipped, 0);
        assert_eq!(tables.len(), TABLES);
        let expected_pending: u64 = (0..TABLES as u64).map(|i| (i % 50) * 2).sum();
        assert_eq!(
            tables
                .values()
                .map(|t| t.pending_generations())
                .sum::<u64>(),
            expected_pending
        );

        // Lead for real: the reconcile runs detached, so the keepalive must
        // keep firing and the leader key must survive a full pass.
        let leader = tokio::spawn({
            let state = state.clone();
            async move { lead(&state).await }
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        loop {
            let (cache, _) = read_prefix_paged(&client, &keys.demand_prefix())
                .await
                .unwrap();
            if cache.len() == TABLES {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "cache incomplete: {}",
                cache.len()
            );
            assert!(
                !leader.is_finished(),
                "leader loop ended during reconcile: {:?}",
                leader.await
            );
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        // Lease is alive well past several TTLs.
        tokio::time::sleep(Duration::from_secs(12)).await;
        let lk = client.get(keys.leader(), None).await.unwrap();
        assert_eq!(
            lk.kvs().len(),
            1,
            "leader lost its lease during/after the pass"
        );
        assert!(!leader.is_finished());
        leader.abort();
        let _ = client
            .delete(
                cfg.etcd.etcd_prefix.as_str(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await;
    }

    /// Deletes a test prefix on drop so a failed assertion cannot leave
    /// tens of thousands of keys behind to slow every later test.
    struct PrefixCleanup(etcd_client::Client, String);
    impl Drop for PrefixCleanup {
        fn drop(&mut self) {
            let mut client = self.0.clone();
            let prefix = self.1.clone();
            tokio::spawn(async move {
                let _ = client
                    .delete(
                        prefix.as_str(),
                        Some(etcd_client::DeleteOptions::new().with_prefix()),
                    )
                    .await;
            });
        }
    }

    #[test]
    fn chunk_ops_respects_both_count_and_bytes() {
        // 300 small ops -> 3 chunks of 100.
        let small: Vec<(Vec<u8>, usize)> = (0..300).map(|_| (Vec::new(), 100)).collect();
        let chunks = chunk_ops(small);
        assert_eq!(
            chunks.iter().map(Vec::len).collect::<Vec<_>>(),
            [100, 100, 100]
        );
        // 20 ops of 100 KiB -> bytes bind at 7 per chunk (7 * 100 KiB < 768 KiB).
        let big: Vec<(Vec<u8>, usize)> = (0..20).map(|_| (Vec::new(), 100 * 1024)).collect();
        let chunks = chunk_ops(big);
        assert!(chunks.iter().all(|c| c.len() <= 7), "{chunks:?}");
        assert_eq!(chunks.iter().map(Vec::len).sum::<usize>(), 20);
        // One op larger than the budget still goes out, alone.
        let huge = vec![(Vec::new(), 2 * TXN_BYTES), (Vec::new(), 10)];
        let chunks = chunk_ops(huge);
        assert_eq!(chunks, vec![vec![0], vec![1]]);
    }

    /// Round-6 findings 1 and 2 at the reviewer's shapes: 512 tables x 8
    /// shards with full `sealed_times` maps (~5.6 MB of events: a 512-key
    /// page exceeds the 4 MiB cap) and a cache write for 20-shard tables
    /// (100 records ~2.7 MB: exceeds etcd's 1.5 MiB request cap). Both must
    /// complete; the paged read must have shrunk its page at least once.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS and PLANNER_SCALE_TEST=1; run alone"]
    async fn large_records_page_by_bytes_and_write_within_request_limits() {
        if std::env::var("PLANNER_SCALE_TEST").is_err() {
            eprintln!("PLANNER_SCALE_TEST unset; skipping");
            return;
        }
        use crate::state::MasterState;
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = crate::config::MasterConfig::parse_from([
            "test",
            "--data-dir",
            dir.path().to_str().unwrap(),
        ]);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/planner-bytes/{}", generate_id());
        let _cleanup = PrefixCleanup(
            etcd_client::Client::connect(cfg.etcd.etcd_endpoints.clone(), None)
                .await
                .unwrap(),
            cfg.etcd.etcd_prefix.clone(),
        );
        cfg.planner.planner_enabled = true;
        cfg.stats_scan_interval_secs = 0;
        cfg.compaction_interval_secs = 0;
        cfg.merge_wal_interval_secs = 0;
        let state = MasterState::new(cfg.clone()).await.unwrap();
        let keys = Keys::new(&cfg.etcd.etcd_prefix);
        let mut client = state.task_store.etcd_client().clone();

        // Shape A: 512 tables x 8 shards, each with a 64-entry times map.
        let full_times: BTreeMap<u64, i64> = (1..=64u64)
            .map(|g| (1_000_000_000_000 + g, 1_700_000_000_000 + g as i64 * 1000))
            .collect();
        let mut ops = Vec::new();
        for i in 0..512 {
            let target = format!("wide-{i:04}");
            for sh in 0..8 {
                let event = DemandEvent {
                    v: SCHEMA_VERSION,
                    shard: format!("bbbbbbbb-0000-0000-0000-{sh:012}"),
                    sealed_through: 64,
                    sealed_bytes_through: 64 << 20,
                    merged_through: None,
                    flushed_at_ms: 1_700_000_000_000,
                    writer_epoch: 1,
                    source: EventSource::Writer,
                    merged_epoch: None,
                    sealed_times: full_times.clone(),
                };
                ops.push(TxnOp::put(
                    format!("{}{}/{}", keys.events_prefix(), hex(&target), event.shard),
                    serde_json::to_vec(&event).unwrap(),
                    None,
                ));
                if ops.len() == 50 {
                    client
                        .txn(Txn::new().and_then(std::mem::take(&mut ops)))
                        .await
                        .unwrap();
                }
            }
        }
        if !ops.is_empty() {
            client.txn(Txn::new().and_then(ops)).await.unwrap();
        }
        // The whole prefix exceeds the cap (so this fixture is not vacuous);
        // the paged reader must still return every record.
        let unpaged = client
            .get(keys.events_prefix(), Some(GetOptions::new().with_prefix()))
            .await;
        assert!(
            unpaged.is_err(),
            "expected the unpaged read to exceed the message cap"
        );
        let (kvs, _) = read_prefix_paged(&client, &keys.events_prefix())
            .await
            .unwrap();
        assert_eq!(kvs.len(), 512 * 8);
        let total: usize = kvs.iter().map(|kv| kv.key.len() + kv.value.len()).sum();
        assert!(
            total > 4 << 20,
            "fixture must exceed 4 MiB in total: {total}"
        );
        let (tables, skipped) = fold_events(
            &keys,
            kvs.iter()
                .map(|kv| (kv.key.as_slice(), kv.value.as_slice(), kv.mod_revision)),
            0,
        );
        assert_eq!((tables.len(), skipped), (512, 0));

        // Shape B: the cache write. 20-shard tables produce ~27 KB records;
        // 100 of them in one txn is ~2.7 MB. Drive a real reconcile under a
        // real leader token and assert every record landed.
        let wide20: BTreeMap<u64, i64> = (1..=64u64).map(|g| (g, 1_700_000_000_000)).collect();
        let mut ops = Vec::new();
        for i in 0..100 {
            let target = format!("deep-{i:03}");
            for sh in 0..20 {
                let event = DemandEvent {
                    v: SCHEMA_VERSION,
                    shard: format!("cccccccc-0000-0000-0000-{sh:012}"),
                    sealed_through: 64,
                    sealed_bytes_through: 0,
                    merged_through: None,
                    flushed_at_ms: 1_700_000_000_000,
                    writer_epoch: 1,
                    source: EventSource::Writer,
                    merged_epoch: None,
                    sealed_times: wide20.clone(),
                };
                ops.push(TxnOp::put(
                    format!("{}{}/{}", keys.events_prefix(), hex(&target), event.shard),
                    serde_json::to_vec(&event).unwrap(),
                    None,
                ));
                if ops.len() == 40 {
                    client
                        .txn(Txn::new().and_then(std::mem::take(&mut ops)))
                        .await
                        .unwrap();
                }
            }
        }
        if !ops.is_empty() {
            client.txn(Txn::new().and_then(ops)).await.unwrap();
        }
        let token = generate_id();
        client
            .put(keys.leader_token(), token.clone(), None)
            .await
            .unwrap();
        let written = reconcile_once(&state, &keys, &token).await.unwrap();
        assert_eq!(
            written,
            512 + 100,
            "every table record written in one reconcile"
        );
        let (cache, _) = read_prefix_paged(&client, &keys.demand_prefix())
            .await
            .unwrap();
        assert_eq!(cache.len(), 612);
        let deep: TableDemand = serde_json::from_slice(
            &cache
                .iter()
                .find(|kv| kv.key.ends_with(hex("deep-000").as_bytes()))
                .unwrap()
                .value,
        )
        .unwrap();
        assert_eq!(deep.shards.len(), 20);
        let _ = client
            .delete(
                cfg.etcd.etcd_prefix.as_str(),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await;
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn single_leader_builds_a_cache_equal_to_a_direct_fold_and_deposed_leader_cannot_write() {
        use crate::state::MasterState;
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = crate::config::MasterConfig::parse_from([
            "test",
            "--data-dir",
            dir.path().to_str().unwrap(),
        ]);
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        cfg.etcd.etcd_prefix = format!("/planner-test/{}", generate_id());
        cfg.planner.planner_enabled = true;
        cfg.planner.planner_reconcile_secs = 1;
        cfg.stats_scan_interval_secs = 0;
        cfg.compaction_interval_secs = 0;
        cfg.merge_wal_interval_secs = 0;
        let first = MasterState::new(cfg.clone()).await.unwrap();
        let second = MasterState::new(cfg.clone()).await.unwrap();
        let keys = Keys::new(&cfg.etcd.etcd_prefix);

        // Two tables, one with two shards, published through the real path.
        let coordinator = first.task_store.merge_coordinator();
        for (target, shard, sealed, merged) in [
            ("hot", "a", 40u64, Some(10u64)),
            ("hot", "b", 12, None),
            ("cold", "a", 3, Some(3)),
        ] {
            coordinator
                .publish_demand_event(
                    target,
                    &DemandEvent {
                        v: SCHEMA_VERSION,
                        shard: shard.into(),
                        sealed_through: sealed,
                        sealed_bytes_through: sealed * 100,
                        merged_through: merged,
                        flushed_at_ms: 1000,
                        writer_epoch: 1,
                        source: EventSource::Writer,
                        merged_epoch: None,
                        sealed_times: Default::default(),
                    },
                )
                .await
                .unwrap();
        }

        // Both campaign; exactly one wins. The loser returns Ok(false)
        // promptly; the winner keeps leading until aborted.
        let a = tokio::spawn({
            let first = first.clone();
            async move { lead(&first).await }
        });
        let b = tokio::spawn({
            let second = second.clone();
            async move { lead(&second).await }
        });
        let mut client = first.task_store.etcd_client().clone();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let cache = loop {
            let got = client
                .get(keys.demand_prefix(), Some(GetOptions::new().with_prefix()))
                .await
                .unwrap();
            if got.kvs().len() == 2 {
                break got;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "planner never wrote the cache"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        let leader_kv = client.get(keys.leader(), None).await.unwrap();
        assert_eq!(leader_kv.kvs().len(), 1, "exactly one leader");
        let leader: LeaderRecord = serde_json::from_slice(leader_kv.kvs()[0].value()).unwrap();

        // Cache equals a direct fold of the events.
        let events = client
            .get(keys.events_prefix(), Some(GetOptions::new().with_prefix()))
            .await
            .unwrap();
        let (direct, skipped) = fold_events(
            &keys,
            events
                .kvs()
                .iter()
                .map(|kv| (kv.key(), kv.value(), kv.mod_revision())),
            0,
        );
        assert_eq!(skipped, 0);
        for kv in cache.kvs() {
            let mut stored: TableDemand = serde_json::from_slice(kv.value()).unwrap();
            let target = unhex(
                String::from_utf8_lossy(kv.key())
                    .strip_prefix(&keys.demand_prefix())
                    .unwrap(),
            )
            .unwrap();
            stored.updated_ms = 0;
            let mut expect = direct[&target].clone();
            expect.updated_ms = 0;
            assert_eq!(stored, expect, "{target}");
        }
        let hot: TableDemand = serde_json::from_slice(
            client.get(keys.demand("hot"), None).await.unwrap().kvs()[0].value(),
        )
        .unwrap();
        assert_eq!(hot.pending_generations(), 30 + 12);

        // Via the HTTP view.
        let axum::Json(report) = crate::routes::scheduler_demand(
            axum::extract::State(first.clone()),
            axum::extract::Query(crate::routes::DemandQuery {
                target: Some("hot".into()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            report.leader.as_ref().map(|l| &l.token),
            Some(&leader.token)
        );
        assert_eq!(report.tables.len(), 1);
        assert_eq!(report.tables[0].pending_generations, 42);
        assert_eq!(report.tables[0].shards, 2);
        assert!(
            report.tables[0].record.is_some(),
            "single-target view carries the record"
        );
        // List mode: paged, bounded, pending-sorted, summary only.
        let axum::Json(list) = crate::routes::scheduler_demand(
            axum::extract::State(first.clone()),
            axum::extract::Query(crate::routes::DemandQuery {
                target: None,
                limit: Some(1),
                include_idle: false,
            }),
        )
        .await
        .unwrap();
        assert_eq!(list.total_tables, 2, "hot and cold are both cached");
        assert_eq!(list.tables_with_pending, 1);
        assert_eq!(list.tables.len(), 1, "limit applied");
        assert_eq!(list.tables[0].target, "hot");
        assert!(list.tables[0].record.is_none(), "list view is summary only");

        // Shadow placement: with this master's heartbeat live, the hot table
        // (42 pending >= 8) gets exactly one Shadow assignment against it;
        // cold (0 pending) gets none. No real MergeWal task exists, so the
        // pass records one disagreement for hot.
        let hb_runner = tokio::spawn({
            let first = first.clone();
            async move { crate::executors::heartbeat_loop_for_test(&first).await }
        });
        let ekeys = crate::executors::Keys::new(&cfg.etcd.etcd_prefix);
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        let shadows: Vec<crate::executors::Assignment> = loop {
            let got = client
                .get(
                    ekeys.assignments_prefix(),
                    Some(GetOptions::new().with_prefix()),
                )
                .await
                .unwrap();
            let v: Vec<crate::executors::Assignment> = got
                .kvs()
                .iter()
                .filter_map(|kv| serde_json::from_slice(kv.value()).ok())
                .collect();
            if !v.is_empty() {
                break v;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no shadow assignment written"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert_eq!(shadows.len(), 1, "{shadows:?}");
        assert_eq!(shadows[0].target, "hot");
        assert_eq!(shadows[0].state, crate::executors::AssignmentState::Shadow);
        assert_eq!(shadows[0].kind, lance_context_api::TaskKind::MergeWal);
        assert_eq!(shadows[0].executor, first.admission.status().executor_id);
        assert!(shadows[0].reserved_bytes > 0);
        // Headroom reflects the shadow reservation.
        let h = crate::executors::load_headroom(&client, &ekeys)
            .await
            .unwrap();
        assert_eq!(h[&shadows[0].executor].assignments, 1);
        // Enqueue a real MergeWal task for hot: on the next pass the planner
        // and reality agree, and the shadow row is replaced, not duplicated.
        first
            .task_store
            .enqueue(lance_context_api::TaskKind::MergeWal, "hot", vec![])
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let got = client
            .get(
                ekeys.assignments_prefix(),
                Some(GetOptions::new().with_prefix()),
            )
            .await
            .unwrap();
        assert_eq!(
            got.kvs().len(),
            1,
            "shadow rows are replaced, never accumulated"
        );
        hb_runner.abort();

        // A deposed leader's reconcile is refused: forge a different token.
        let err = reconcile_once(&first, &keys, "not-the-leader")
            .await
            .unwrap_err();
        assert!(err.contains("leadership lost"), "{err}");

        // Dropping the cold table's events removes its cache row on the next
        // reconcile; hot stays.
        client
            .delete(
                format!("{}{}/", keys.events_prefix(), hex("cold")),
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let got = client
                .get(keys.demand_prefix(), Some(GetOptions::new().with_prefix()))
                .await
                .unwrap();
            if got.kvs().len() == 1 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "stale cache row not dropped"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        // Exactly one campaigner lost (finished with Ok(false)); the other is
        // still leading. If both led, both would still be running.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let finished = [a.is_finished(), b.is_finished()];
        assert_eq!(
            finished.iter().filter(|f| **f).count(),
            1,
            "exactly one campaigner must have lost and returned (finished={finished:?})"
        );
        let (loser, winner) = if finished[0] { (a, b) } else { (b, a) };
        assert_eq!(loser.await.unwrap(), Ok(false));
        winner.abort();
    }
}
