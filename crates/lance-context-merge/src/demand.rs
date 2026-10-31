//! Per-table maintenance demand folded from per-shard watermarks.
//!
//! Schema: `docs/design/scheduler-p1-data-model.md` §1. The record stored
//! under `P/demand-events/<hex>/<shard>` is the per-dimension **join** of
//! everything every publisher (writer, executor, scan) has reported for that
//! shard, and `TableDemand::fold` applies the same join in memory. Joins are
//! idempotent and commutative, so any order or duplication of the same
//! events - and a rebuild from the stored records - yields the same answer.
//!
//! Generation numbers are monotonic per shard across writer epochs: Lance
//! keeps `current_generation` and `flushed_generations` when a new writer
//! claims a shard (`ShardManifest { writer_epoch: next, ..base }`). The epoch
//! is therefore a writer fence and a tiebreak, never a numbering restart.

use lance_context_metrics::etcd::{ObserveEtcd, Operation};
use std::collections::BTreeMap;

use etcd_client::{Compare, CompareOp, GetResponse, TxnOp, TxnOpResponse};
use serde::{Deserialize, Serialize};

use crate::{Coordinator, Result};

fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or_default()
}

pub const SCHEMA_VERSION: u32 = 1;

// Bound both the response allocation and transaction op count. Only reads are
// batched: writes retain independent revision comparisons and retry budgets.
const DEMAND_READ_BATCH_SIZE: usize = 32;

/// How many `(generation, flushed_at)` pairs a shard record keeps for age
/// tracking. Only the lowest unmerged generations matter for "oldest
/// pending", so the set is pruned to the smallest `SEALED_TIMES_CAP`
/// generations above `merged`. Union-then-truncate-to-smallest is order
/// independent.
pub const SEALED_TIMES_CAP: usize = 64;

/// Who produced a watermark event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSource {
    /// The shard's own writer, on flush.
    Writer,
    /// An executor that merged generations, after its release.
    Executor,
    /// The periodic stats scan observing the shard from storage.
    Scan,
}

/// One shard's watermark record. Published values are absolute; the stored
/// record is the join of all of them (see `join_events`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DemandEvent {
    #[serde(default = "schema_version")]
    pub v: u32,
    pub shard: String,
    /// Highest generation sealed (flushed) on this shard.
    pub sealed_through: u64,
    #[serde(default)]
    pub sealed_bytes_through: u64,
    /// Highest generation merged into the base table, if known.
    #[serde(default)]
    pub merged_through: Option<u64>,
    /// Epoch under which `merged_through` was reported, when it differs from
    /// `writer_epoch`. Provenance only; absent means "same as writer_epoch".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_epoch: Option<u64>,
    /// When `sealed_through` was flushed.
    pub flushed_at_ms: i64,
    /// The shard writer's epoch: provenance and tiebreak only.
    pub writer_epoch: u64,
    pub source: EventSource,
    /// Flush times of the lowest unmerged generations (≤ `SEALED_TIMES_CAP`),
    /// carried in the stored record so the age clock survives a rebuild.
    /// Publishers that only know their current flush leave this empty; the
    /// join fills and prunes it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sealed_times: BTreeMap<u64, i64>,
}

fn schema_version() -> u32 {
    SCHEMA_VERSION
}

/// A watermark with the epoch that produced it. Orders by **value** first,
/// epoch second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Mark {
    pub epoch: u64,
    pub value: u64,
}

impl PartialOrd for Mark {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Mark {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.value, self.epoch).cmp(&(other.value, other.epoch))
    }
}

/// One shard's folded state. `sealed`, `merged` and `sealed_bytes` are pure
/// maxima, so pending counts and bytes are exactly order independent.
/// `sealed_times` is a union (min time per generation) pruned to the lowest
/// `SEALED_TIMES_CAP` generations above `merged`; `oldest_pending_ms` is
/// therefore exact while a shard has at most that many pending generations
/// and the best-known lower bound beyond it. `sealed` and `merged` are
/// independent: nothing about one lowers or replaces the other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ShardDemand {
    pub sealed: Mark,
    #[serde(default)]
    pub merged: Option<Mark>,
    #[serde(default)]
    pub sealed_bytes: u64,
    #[serde(default)]
    pub sealed_times: BTreeMap<u64, i64>,
}

/// How much to trust an oldest-pending time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgeBound {
    Exact,
    /// The true flush time is at or before this. As a bound on *waiting*
    /// it is a floor (waited at least this long), which is sound for
    /// promotion; as a bound on *score aging* it understates.
    Upper,
}

impl ShardDemand {
    pub fn pending_generations(&self) -> u64 {
        self.sealed
            .value
            .saturating_sub(self.merged.map_or(0, |m| m.value))
    }

    /// Flush time of the oldest generation still unmerged, with how much
    /// to trust it.
    ///
    /// The record keeps flush times for at most `SEALED_TIMES_CAP` of the
    /// lowest pending generations. While the oldest pending generation's
    /// own time is kept the answer is **exact**. Once merging has consumed
    /// every kept entry (a shard that had more than the cap pending), only
    /// the newest sealed time is known; that is an **upper** bound on the
    /// oldest pending flush time, so it must never drive age-based
    /// promotion. Never `None` while something is pending.
    pub fn oldest_pending(&self) -> Option<(i64, AgeBound)> {
        if self.pending_generations() == 0 {
            return None;
        }
        let oldest = self.merged.map_or(0, |m| m.value) + 1;
        if let Some(t) = self.sealed_times.get(&oldest) {
            return Some((*t, AgeBound::Exact));
        }
        // Entries at or below merged are stale and ignored. The lowest kept
        // entry above `oldest` is an upper bound on its flush time; the
        // newest sealed generation is always kept (see `prune_times`), so
        // there is always one to give.
        self.sealed_times
            .range(oldest..)
            .next()
            .map(|(_, t)| (*t, AgeBound::Upper))
    }

    /// Convenience: the time only, whatever its bound.
    pub fn oldest_pending_ms(&self) -> Option<i64> {
        self.oldest_pending().map(|(t, _)| t)
    }

    /// Oldest-pending time safe to use for **waiting-time** decisions.
    ///
    /// An `Upper` bound on the flush time is a **lower** bound on how long
    /// the oldest generation has waited (it flushed no later than this), so
    /// "waited at least `max_turn_wait`" is a sound conclusion from it.
    /// Returns the time for both bounds; the distinction matters only when
    /// a caller needs the exact age, not a floor on it.
    pub fn oldest_pending_wait_floor_ms(&self) -> Option<i64> {
        self.oldest_pending().map(|(t, _)| t)
    }

    /// Exact oldest-pending time, or `None` when only a bound is known.
    pub fn oldest_pending_exact_ms(&self) -> Option<i64> {
        match self.oldest_pending() {
            Some((t, AgeBound::Exact)) => Some(t),
            _ => None,
        }
    }

    /// Bound the map by cap **only**. Dropping entries at or below `merged`
    /// here would make the kept set depend on whether the merge or the
    /// flushes arrived first (merge-first keeps 65..128, flush-first keeps
    /// 1..63 and 128), which breaks order independence. Entries at or below
    /// merged are ignored at read time instead.
    fn prune(&mut self) {
        prune_times(&mut self.sealed_times);
    }
}

/// The per-table record under `P/demand/<hex>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableDemand {
    #[serde(default = "schema_version")]
    pub v: u32,
    #[serde(default)]
    pub shards: BTreeMap<String, ShardDemand>,
    #[serde(default)]
    pub fragment_count: u64,
    #[serde(default)]
    pub index_stale: bool,
    #[serde(default)]
    pub uncovered_fragments: u64,
    #[serde(default)]
    pub missing_fragments: bool,
    /// etcd revision of the newest input folded in. Freshness bookkeeping
    /// only; never an ordering input.
    #[serde(default)]
    pub observed_revision: i64,
    #[serde(default)]
    pub updated_ms: i64,
}

impl Default for TableDemand {
    fn default() -> Self {
        Self {
            v: SCHEMA_VERSION,
            shards: BTreeMap::new(),
            fragment_count: 0,
            index_stale: false,
            uncovered_fragments: 0,
            missing_fragments: false,
            observed_revision: 0,
            updated_ms: 0,
        }
    }
}

/// Why an event did not change the table record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ignored {
    /// Nothing advanced.
    NotNewer,
    /// The event's schema major version is not understood.
    UnknownVersion(u32),
}

/// Bound a times map: the lowest `SEALED_TIMES_CAP - 1` generations plus
/// the single highest. Both parts are functions of the set, so the result
/// is order independent. Keeping the highest means an upper bound on every
/// pending age survives even when merging consumes all the low entries.
fn prune_times(times: &mut BTreeMap<u64, i64>) {
    if times.len() <= SEALED_TIMES_CAP {
        return;
    }
    let newest = times.pop_last();
    while times.len() > SEALED_TIMES_CAP - 1 {
        times.pop_last();
    }
    if let Some((g, t)) = newest {
        times.insert(g, t);
    }
}

/// Insert `time` for `generation`, keeping the earliest. Returns whether the
/// map changed.
fn note_time(times: &mut BTreeMap<u64, i64>, generation: u64, time: i64) -> bool {
    match times.entry(generation) {
        std::collections::btree_map::Entry::Occupied(mut slot) => {
            if time < *slot.get() {
                slot.insert(time);
                true
            } else {
                false
            }
        }
        std::collections::btree_map::Entry::Vacant(slot) => {
            slot.insert(time);
            true
        }
    }
}

impl TableDemand {
    /// Reject records from a future schema. `v == 0` is a pre-versioning
    /// record and is read as version 1.
    pub fn check_version(v: u32) -> std::result::Result<(), Ignored> {
        match v {
            0 | SCHEMA_VERSION => Ok(()),
            other => Err(Ignored::UnknownVersion(other)),
        }
    }

    pub fn pending_generations(&self) -> u64 {
        self.shards
            .values()
            .map(ShardDemand::pending_generations)
            .sum()
    }

    pub fn pending_bytes(&self) -> u64 {
        // Bytes are tracked only as "through sealed"; approximate pending
        // bytes by the sealed-bytes watermark of shards with pending work.
        self.shards
            .values()
            .filter(|s| s.pending_generations() > 0)
            .map(|s| s.sealed_bytes)
            .sum()
    }

    /// Flush time of the oldest generation still unmerged on any shard,
    /// whatever its bound. For display.
    pub fn oldest_pending_ms(&self) -> Option<i64> {
        self.shards
            .values()
            .filter_map(ShardDemand::oldest_pending_ms)
            .min()
    }

    /// Oldest **exact** pending time across shards.
    pub fn oldest_pending_exact_ms(&self) -> Option<i64> {
        self.shards
            .values()
            .filter_map(ShardDemand::oldest_pending_exact_ms)
            .min()
    }

    /// Earliest time such that every shard's oldest pending generation
    /// flushed **no later** than it. A sound floor on the table's wait: the
    /// scheduler may promote on `now - this >= max_turn_wait` for any bound.
    pub fn oldest_pending_wait_floor_ms(&self) -> Option<i64> {
        self.shards
            .values()
            .filter_map(ShardDemand::oldest_pending_wait_floor_ms)
            .min()
    }

    /// Fold one event. Returns `Err(reason)` when the record is unchanged.
    pub fn fold(
        &mut self,
        event: &DemandEvent,
        revision: i64,
        now_ms: i64,
    ) -> std::result::Result<(), Ignored> {
        Self::check_version(event.v)?;
        let shard = self.shards.entry(event.shard.clone()).or_default();
        let mut changed = false;

        let sealed = Mark {
            epoch: event.writer_epoch,
            value: event.sealed_through,
        };
        if sealed > shard.sealed {
            shard.sealed = sealed;
            changed = true;
        }
        if event.sealed_bytes_through > shard.sealed_bytes {
            shard.sealed_bytes = event.sealed_bytes_through;
            changed = true;
        }
        if let Some(m) = event.merged_through {
            let merged = Mark {
                epoch: event.merged_epoch.unwrap_or(event.writer_epoch),
                value: m,
            };
            if shard.merged.is_none_or(|current| merged > current) {
                shard.merged = Some(merged);
                changed = true;
            }
        }
        // Age bookkeeping: this event's own flush time plus any times it
        // carries from a stored record, min per generation, only for
        // generations still above merged.
        // An executor's flushed_at_ms is its own observation time, not when
        // the generation was flushed; only writer and scan events seed the
        // age clock from their own timestamp.
        if event.source != EventSource::Executor
            && event.sealed_through > 0
            && note_time(
                &mut shard.sealed_times,
                event.sealed_through,
                event.flushed_at_ms,
            )
        {
            changed = true;
        }
        for (g, t) in &event.sealed_times {
            if note_time(&mut shard.sealed_times, *g, *t) {
                changed = true;
            }
        }

        if !changed {
            return Err(Ignored::NotNewer);
        }
        shard.prune();
        self.observed_revision = self.observed_revision.max(revision);
        self.updated_ms = now_ms;
        Ok(())
    }
}

/// Per-dimension join of a stored record with a new event. Returns `None`
/// when the stored record already dominates on every dimension.
///
/// `sealed` is joined by `(value, epoch)` then bytes; `merged` by
/// `(value, epoch)`; each independently, so a writer event without
/// `merged_through` keeps the executor's mark and an executor event with an
/// older sealed still advances merged. The result keeps the stored record's
/// writer fields unless the new event's sealed mark wins. `sealed_times` is
/// the union of both records' times and both records' own flush times, min
/// per generation, pruned above merged and to the cap, so the age clock is
/// preserved in storage.
pub fn join_events(stored: Option<&DemandEvent>, event: &DemandEvent) -> Option<DemandEvent> {
    let Some(stored) = stored else {
        let mut fresh = event.clone();
        fresh.v = SCHEMA_VERSION;
        if fresh.merged_epoch == Some(fresh.writer_epoch) {
            fresh.merged_epoch = None;
        }
        if fresh.source != EventSource::Executor && fresh.sealed_through > 0 {
            note_time(
                &mut fresh.sealed_times,
                fresh.sealed_through,
                fresh.flushed_at_ms,
            );
        }
        prune_times(&mut fresh.sealed_times);
        return Some(fresh);
    };
    let s_sealed = Mark {
        epoch: stored.writer_epoch,
        value: stored.sealed_through,
    };
    let e_sealed = Mark {
        epoch: event.writer_epoch,
        value: event.sealed_through,
    };
    let s_merged = stored.merged_through.map(|v| Mark {
        epoch: stored.merged_epoch.unwrap_or(stored.writer_epoch),
        value: v,
    });
    let e_merged = event.merged_through.map(|v| Mark {
        epoch: event.merged_epoch.unwrap_or(event.writer_epoch),
        value: v,
    });
    let sealed_wins = e_sealed > s_sealed
        || (e_sealed == s_sealed && event.sealed_bytes_through > stored.sealed_bytes_through);
    let merged_wins = match (s_merged, e_merged) {
        (_, None) => false,
        (None, Some(_)) => true,
        (Some(s), Some(e)) => e > s,
    };
    let best_merged = match (s_merged, e_merged) {
        (Some(s), Some(e)) => Some(s.max(e)),
        (s, e) => s.or(e),
    };

    // Times: union of everything known, min per generation.
    let mut times = stored.sealed_times.clone();
    for (g, t) in &event.sealed_times {
        note_time(&mut times, *g, *t);
    }
    for (src, g, t) in [
        (stored.source, stored.sealed_through, stored.flushed_at_ms),
        (event.source, event.sealed_through, event.flushed_at_ms),
    ] {
        if src != EventSource::Executor && g > 0 {
            note_time(&mut times, g, t);
        }
    }
    prune_times(&mut times);
    let times_changed = times != stored.sealed_times;

    if !sealed_wins && !merged_wins && !times_changed {
        return None;
    }
    let mut next = if sealed_wins {
        event.clone()
    } else {
        stored.clone()
    };
    next.sealed_bytes_through = stored.sealed_bytes_through.max(event.sealed_bytes_through);
    next.merged_through = best_merged.map(|m| m.value);
    next.merged_epoch = best_merged
        .map(|m| m.epoch)
        .filter(|e| *e != next.writer_epoch);
    // `source` and `flushed_at_ms` describe who set `sealed_through` and
    // when it was flushed; they stay with the sealed mark so a rebuild
    // re-derives exactly the time entries the fold did.
    next.sealed_times = times;
    next.v = SCHEMA_VERSION;
    Some(next)
}

/// Decode a stored record. A record from a newer schema is never overwritten:
/// refusing is the only safe answer to a value this binary cannot interpret.
/// An unreadable record (corrupt JSON) is treated as absent and replaced.
fn stored_event(kv: Option<&etcd_client::KeyValue>) -> Result<Option<DemandEvent>> {
    let Some(kv) = kv else { return Ok(None) };
    let Ok(stored) = serde_json::from_slice::<DemandEvent>(kv.value()) else {
        return Ok(None);
    };
    TableDemand::check_version(stored.v)
        .map_err(|e| format!("stored demand record has an unsupported version: {e:?}"))?;
    Ok(Some(stored))
}

/// A shard's merged watermark as observed by an executor after its commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardMerged {
    pub shard: String,
    pub merged_through: u64,
}

impl Coordinator {
    fn demand_event_key(&self, target: &str, shard: &str) -> String {
        let encoded: String = target.bytes().map(|b| format!("{b:02x}")).collect();
        format!(
            "{}/demand-events/{encoded}/{shard}",
            self.prefix.trim_end_matches('/')
        )
    }

    /// Read-join-CAS one event into the stored record, retrying a bounded
    /// number of times when a concurrent publisher wins the race, so no
    /// publisher's progress is lost to another's. Never deletes.
    async fn join_into_store(&self, key: &str, event: &DemandEvent) -> Result<bool> {
        let mut client = self.client.clone();
        let current = client
            .get(key, None)
            .observed(Operation::DemandRead)
            .await
            .map_err(|e| e.to_string())?;
        self.join_from_read(key, event, current).await
    }

    async fn join_from_read(
        &self,
        key: &str,
        event: &DemandEvent,
        mut current: GetResponse,
    ) -> Result<bool> {
        let mut client = self.client.clone();
        for attempt in 0..5 {
            if attempt > 0 {
                // A stale batch read is only a starting point. On conflict,
                // re-read this key and re-join, preserving concurrent progress.
                current = client
                    .get(key, None)
                    .observed(Operation::DemandRead)
                    .await
                    .map_err(|e| e.to_string())?;
            }
            let stored = stored_event(current.kvs().first())?;
            let Some(next) = join_events(stored.as_ref(), event) else {
                return Ok(false);
            };
            let value = serde_json::to_vec(&next).map_err(|e| e.to_string())?;
            let compare = match current.kvs().first() {
                Some(kv) => Compare::mod_revision(key, CompareOp::Equal, kv.mod_revision()),
                None => Compare::version(key, CompareOp::Equal, 0),
            };
            let response = client
                .txn(
                    etcd_client::Txn::new()
                        .when(vec![compare])
                        .and_then(vec![TxnOp::put(key, value, None)]),
                )
                .observed(Operation::DemandPublish)
                .await
                .map_err(|e| e.to_string())?;
            if response.succeeded() {
                return Ok(true);
            }
        }
        Err(format!(
            "demand record {key} kept losing to concurrent writes"
        ))
    }

    /// Publish a shard watermark from a writer or scan. Safe on every flush;
    /// `Ok(false)` when nothing advanced.
    pub async fn publish_demand_event(&self, target: &str, event: &DemandEvent) -> Result<bool> {
        TableDemand::check_version(event.v).map_err(|e| format!("{e:?}"))?;
        let key = self.demand_event_key(target, &event.shard);
        self.join_into_store(&key, event).await
    }

    /// Publish scan observations with bounded, read-only batches. Each changed
    /// shard still gets its own CAS and conflict retries; unchanged shards need
    /// no write. Duplicates are processed in input order and stale observations
    /// re-join on conflict. Like single-event publication, an error may follow
    /// successful writes; callers can safely replay the observations.
    pub async fn publish_demand_events(
        &self,
        target: &str,
        events: &[DemandEvent],
    ) -> Result<usize> {
        for event in events {
            TableDemand::check_version(event.v).map_err(|e| format!("{e:?}"))?;
        }
        let mut written = 0;
        let mut client = self.client.clone();
        for chunk in events.chunks(DEMAND_READ_BATCH_SIZE) {
            if chunk.len() == 1 {
                written += usize::from(self.publish_demand_event(target, &chunk[0]).await?);
                continue;
            }
            let keys: Vec<_> = chunk
                .iter()
                .map(|event| self.demand_event_key(target, &event.shard))
                .collect();
            let response = client
                .txn(
                    etcd_client::Txn::new().and_then(
                        keys.iter()
                            .map(|key| TxnOp::get(key.as_str(), None))
                            .collect::<Vec<_>>(),
                    ),
                )
                .observed(Operation::DemandRead)
                .await
                .map_err(|e| e.to_string())?;
            let reads = response.op_responses();
            if reads.len() != chunk.len() {
                return Err("demand batch read returned an unexpected response count".into());
            }
            for ((key, event), read) in keys.iter().zip(chunk).zip(reads) {
                let TxnOpResponse::Get(current) = read else {
                    return Err("demand batch read returned a non-Get response".into());
                };
                written += usize::from(self.join_from_read(key, event, current).await?);
            }
        }
        Ok(written)
    }

    /// Publish one shard's merged watermark from an executor. Only `merged`
    /// can advance; the writer's sealed/epoch/bytes/times are kept. A shard
    /// with no stored record gets a fresh executor-sourced record with
    /// `sealed_through = merged_through` at epoch 0.
    pub async fn publish_merged_watermark(
        &self,
        target: &str,
        shard: &ShardMerged,
    ) -> Result<bool> {
        let key = self.demand_event_key(target, &shard.shard);
        let event = DemandEvent {
            v: SCHEMA_VERSION,
            shard: shard.shard.clone(),
            sealed_through: shard.merged_through,
            sealed_bytes_through: 0,
            merged_through: Some(shard.merged_through),
            flushed_at_ms: chrono_now_ms(),
            writer_epoch: 0,
            source: EventSource::Executor,
            merged_epoch: None,
            sealed_times: BTreeMap::new(),
        };
        self.join_into_store(&key, &event).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(
        shard: &str,
        sealed: u64,
        merged: Option<u64>,
        epoch: u64,
        source: EventSource,
    ) -> DemandEvent {
        DemandEvent {
            v: 1,
            shard: shard.into(),
            sealed_through: sealed,
            sealed_bytes_through: sealed * 1000,
            merged_through: merged,
            flushed_at_ms: sealed as i64 * 10,
            writer_epoch: epoch,
            source,
            merged_epoch: None,
            sealed_times: BTreeMap::new(),
        }
    }

    fn fold_all(events: &[DemandEvent]) -> TableDemand {
        let mut table = TableDemand::default();
        for (i, event) in events.iter().enumerate() {
            let _ = table.fold(event, i as i64, 0);
        }
        table.observed_revision = 0;
        table.updated_ms = 0;
        table
    }

    fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }
        let mut out = Vec::new();
        for i in 0..items.len() {
            let mut rest = items.to_vec();
            let head = rest.remove(i);
            for mut tail in permutations(&rest) {
                tail.insert(0, head.clone());
                out.push(tail);
            }
        }
        out
    }

    fn assert_order_independent(events: &[DemandEvent]) -> TableDemand {
        let canonical = fold_all(events);
        for perm in permutations(events) {
            assert_eq!(fold_all(&perm), canonical, "order {perm:?}");
            let doubled: Vec<_> = perm.iter().flat_map(|e| [e.clone(), e.clone()]).collect();
            assert_eq!(fold_all(&doubled), canonical, "doubled {perm:?}");
        }
        let mut again = canonical.clone();
        for event in events {
            let _ = again.fold(event, 99, 99);
        }
        again.observed_revision = 0;
        again.updated_ms = 0;
        assert_eq!(again, canonical, "replay onto final state");
        canonical
    }

    /// Storage-layer join over a stream, as etcd would end up holding it.
    fn join_all(events: &[DemandEvent]) -> Option<DemandEvent> {
        let mut stored: Option<DemandEvent> = None;
        for e in events {
            if let Some(next) = join_events(stored.as_ref(), e) {
                stored = Some(next);
            }
        }
        stored
    }

    /// Rebuild a table from stored records only (one per shard), as a new
    /// leader would.
    fn rebuild(events: &[DemandEvent]) -> TableDemand {
        let mut by_shard: BTreeMap<String, Option<DemandEvent>> = BTreeMap::new();
        for e in events {
            let slot = by_shard.entry(e.shard.clone()).or_default();
            if let Some(next) = join_events(slot.as_ref(), e) {
                *slot = Some(next);
            }
        }
        let mut t = TableDemand::default();
        for stored in by_shard.into_values().flatten() {
            let _ = t.fold(&stored, 0, 0);
        }
        t.observed_revision = 0;
        t.updated_ms = 0;
        t
    }

    #[test]
    fn fold_is_order_independent_and_idempotent() {
        let events = [
            ev("a", 10, None, 1, EventSource::Writer),
            ev("a", 12, Some(8), 1, EventSource::Executor),
            ev("a", 11, Some(11), 1, EventSource::Executor),
            ev("b", 3, None, 2, EventSource::Writer),
            ev("b", 5, None, 2, EventSource::Scan),
        ];
        let table = assert_order_independent(&events);
        assert_eq!(table.shards["a"].sealed.value, 12);
        assert_eq!(table.shards["a"].merged.unwrap().value, 11);
        assert_eq!(table.shards["b"].sealed.value, 5);
        assert_eq!(table.pending_generations(), 1 + 5);
        assert_eq!(rebuild(&events), table, "rebuild from storage agrees");
    }

    /// Round-3 counterexamples.
    #[test]
    fn round3_counterexamples_fold_to_one_answer() {
        let table = assert_order_independent(&[
            ev("s", 100, Some(40), 1, EventSource::Executor),
            ev("s", 60, None, 2, EventSource::Writer),
        ]);
        assert_eq!(table.pending_generations(), 60);
        let table = assert_order_independent(&[
            ev("s", 30, Some(10), 1, EventSource::Writer),
            ev("s", 20, Some(20), 1, EventSource::Scan),
        ]);
        assert_eq!(table.pending_generations(), 10);
    }

    /// Round-4 counterexample: merged to 40, writer changes epoch and flushes
    /// to 60. Generation numbering survives the epoch, so pending is 20.
    #[test]
    fn merged_from_an_older_epoch_still_counts_against_a_newer_flush() {
        let events = [
            ev("s", 40, Some(40), 1, EventSource::Executor),
            ev("s", 60, None, 2, EventSource::Writer),
        ];
        let table = assert_order_independent(&events);
        assert_eq!(
            table.shards["s"].sealed,
            Mark {
                epoch: 2,
                value: 60
            }
        );
        assert_eq!(
            table.shards["s"].merged,
            Some(Mark {
                epoch: 1,
                value: 40
            })
        );
        assert_eq!(table.pending_generations(), 20);
        let stored = join_all(&events).unwrap();
        assert_eq!(
            (
                stored.writer_epoch,
                stored.sealed_through,
                stored.merged_through
            ),
            (2, 60, Some(40))
        );
        assert_eq!(rebuild(&events).pending_generations(), 20);
    }

    #[test]
    fn writer_event_never_forgets_merged() {
        let events = [
            ev("s", 10, Some(8), 1, EventSource::Executor),
            ev("s", 12, None, 1, EventSource::Writer),
        ];
        let table = assert_order_independent(&events);
        assert_eq!(table.pending_generations(), 4);
        assert_eq!(join_all(&events).unwrap().merged_through, Some(8));
        assert_eq!(rebuild(&events).pending_generations(), 4);
    }

    #[test]
    fn a_lower_value_from_any_source_changes_nothing() {
        let mut table = TableDemand::default();
        table
            .fold(&ev("a", 20, Some(5), 3, EventSource::Writer), 1, 0)
            .unwrap();
        // Lower sealed and lower merged: a time entry for generation 15 is
        // learned (useful for age) but no watermark moves.
        table
            .fold(&ev("a", 15, Some(2), 3, EventSource::Scan), 2, 0)
            .unwrap();
        assert_eq!(table.shards["a"].sealed.value, 20);
        assert_eq!(table.shards["a"].merged.unwrap().value, 5);
        assert_eq!(
            table.fold(&ev("a", 15, Some(2), 3, EventSource::Scan), 3, 0),
            Err(Ignored::NotNewer)
        );
        // A "newer epoch" with a lower generation cannot occur in Lance; if
        // reported it loses the max. Its flush time is recorded once (the
        // map is bounded by cap only; see ShardDemand::prune) and is inert:
        // generation 2 is below merged and never read.
        table
            .fold(&ev("a", 2, None, 4, EventSource::Scan), 5, 0)
            .unwrap();
        assert_eq!(
            table.fold(&ev("a", 2, None, 4, EventSource::Scan), 6, 0),
            Err(Ignored::NotNewer)
        );
        assert_eq!(
            table.shards["a"].oldest_pending_ms(),
            Some(150),
            "gen 15's time, not gen 2's"
        );
        assert_eq!(
            table.shards["a"].sealed,
            Mark {
                epoch: 3,
                value: 20
            }
        );
    }

    #[test]
    fn marks_order_by_generation_with_epoch_as_tiebreak() {
        let mut table = TableDemand::default();
        table
            .fold(&ev("a", 50, Some(40), 2, EventSource::Executor), 1, 0)
            .unwrap();
        table
            .fold(&ev("a", 60, Some(55), 1, EventSource::Writer), 2, 0)
            .unwrap();
        assert_eq!(
            table.shards["a"].sealed,
            Mark {
                epoch: 1,
                value: 60
            }
        );
        assert_eq!(
            table.shards["a"].merged,
            Some(Mark {
                epoch: 1,
                value: 55
            })
        );
        table
            .fold(&ev("a", 60, None, 3, EventSource::Writer), 4, 0)
            .unwrap();
        assert_eq!(
            table.shards["a"].sealed,
            Mark {
                epoch: 3,
                value: 60
            }
        );
        assert_eq!(table.pending_generations(), 5);
    }

    /// Round-3 finding 4 and round-4 finding 4: the age clock is the oldest
    /// unmerged generation's flush time, does not reset on newer flushes,
    /// and survives a rebuild from the stored record.
    #[test]
    fn oldest_pending_is_stable_and_rebuildable() {
        let flush = |g: u64, t: i64| DemandEvent {
            flushed_at_ms: t,
            ..ev("s", g, None, 1, EventSource::Writer)
        };
        let events = [flush(1, 1000), flush(2, 2000), flush(3, 3000)];
        let table = assert_order_independent(&events);
        assert_eq!(table.oldest_pending_ms(), Some(1000));
        let stored = join_all(&events).unwrap();
        assert_eq!(stored.sealed_times.values().next(), Some(&1000));
        assert_eq!(
            rebuild(&events).oldest_pending_ms(),
            Some(1000),
            "rebuild must agree"
        );
        // Merging 1 moves the clock to generation 2 in both places.
        let merged = ev("s", 3, Some(1), 1, EventSource::Executor);
        let mut table = table;
        table.fold(&merged, 4, 0).unwrap();
        assert_eq!(table.oldest_pending_ms(), Some(2000));
        let stored = join_events(Some(&stored), &merged).unwrap();
        let mut from_stored = TableDemand::default();
        from_stored.fold(&stored, 0, 0).unwrap();
        assert_eq!(from_stored.oldest_pending_ms(), Some(2000));
        let all = [flush(1, 1000), flush(2, 2000), flush(3, 3000), merged];
        assert_eq!(rebuild(&all).oldest_pending_ms(), Some(2000));
        assert_order_independent(&all);
    }

    /// Round-5 finding 6: 128 pending, merge the first 64. The oldest
    /// pending is generation 65, whose time the cap never kept. The answer
    /// must not be unknown and must not drive promotion as if exact: it is
    /// reported as an Upper bound (the newest kept or sealed time), excluded
    /// from `oldest_pending_exact_ms`, and identical after a rebuild.
    #[test]
    fn age_after_merging_past_the_cap_is_a_flagged_bound() {
        let flush = |g: u64| DemandEvent {
            flushed_at_ms: g as i64 * 1000,
            ..ev("s", g, None, 1, EventSource::Writer)
        };
        let mut events: Vec<_> = (1..=128).map(flush).collect();
        let table = fold_all(&events);
        assert_eq!(
            table.shards["s"].oldest_pending(),
            Some((1000, AgeBound::Exact))
        );
        assert_eq!(table.oldest_pending_exact_ms(), Some(1000));
        events.push(ev("s", 128, Some(64), 1, EventSource::Executor));
        let table = fold_all(&events);
        let (t, bound) = table.shards["s"]
            .oldest_pending()
            .expect("64 pending: not unknown");
        assert_eq!(bound, AgeBound::Upper, "gen 65's time was never kept");
        assert!(
            t >= 65_000,
            "an upper bound is at or after the true time: {t}"
        );
        assert_eq!(
            table.oldest_pending_exact_ms(),
            None,
            "a bound never promotes"
        );
        assert_eq!(table.oldest_pending_ms(), Some(t), "but is shown");
        assert_eq!(
            rebuild(&events).shards["s"].oldest_pending(),
            Some((t, AgeBound::Upper))
        );
        let mut reversed = events.clone();
        reversed.reverse();
        assert_eq!(
            fold_all(&reversed).shards["s"].oldest_pending(),
            Some((t, AgeBound::Upper))
        );
        // Below the cap the answer is exact again: 10 pending, merge 3 -> gen 4.
        let small: Vec<_> = (1..=10)
            .map(flush)
            .chain([ev("s", 10, Some(3), 1, EventSource::Executor)])
            .collect();
        assert_eq!(
            fold_all(&small).shards["s"].oldest_pending(),
            Some((4000, AgeBound::Exact))
        );
    }

    /// Round-6 finding 3: a long-lived table whose first 70 generations
    /// each merged immediately, then two generations sit for a day. The
    /// first 63 timestamps occupy the bounded map forever, so the tail's
    /// own times are known only as an upper bound - which is still a sound
    /// *floor* on how long it has waited, and must drive promotion.
    #[test]
    fn long_lived_tail_still_reports_a_usable_wait() {
        let mut events = Vec::new();
        for g in 1..=70u64 {
            events.push(DemandEvent {
                flushed_at_ms: g as i64 * 1000,
                ..ev("s", g, None, 1, EventSource::Writer)
            });
            events.push(ev("s", g, Some(g), 1, EventSource::Executor));
        }
        let day = 86_400_000i64;
        events.push(DemandEvent {
            flushed_at_ms: 100_000,
            ..ev("s", 71, None, 1, EventSource::Writer)
        });
        events.push(DemandEvent {
            flushed_at_ms: 100_000 + 30_000,
            ..ev("s", 72, None, 1, EventSource::Writer)
        });
        let table = fold_all(&events);
        assert_eq!(table.pending_generations(), 2);
        let (t, bound) = table.shards["s"].oldest_pending().unwrap();
        // Gen 71's own time (100_000) may or may not be kept; the newest
        // sealed (gen 72, 130_000) always is. Either way the floor is no
        // later than 130_000 and no earlier than 100_000.
        assert!((100_000..=130_000).contains(&t), "{t} {bound:?}");
        let now = 130_000 + day;
        let wait = now - table.oldest_pending_wait_floor_ms().unwrap();
        assert!(
            wait >= day,
            "a day-old tail must show at least a day of wait: {wait}"
        );
        assert_eq!(rebuild(&events).oldest_pending_wait_floor_ms(), Some(t));
    }

    #[test]
    fn beyond_cap_counts_stay_exact_and_age_is_a_bound() {
        let n = (SEALED_TIMES_CAP as u64) + 20;
        let forward: Vec<_> = (1..=n)
            .map(|g| DemandEvent {
                flushed_at_ms: g as i64 * 10,
                ..ev("s", g, None, 1, EventSource::Writer)
            })
            .collect();
        let mut reverse = forward.clone();
        reverse.reverse();
        let f = fold_all(&forward);
        let r = fold_all(&reverse);
        assert_eq!(f.pending_generations(), r.pending_generations());
        assert_eq!(f.pending_generations(), n);
        assert_eq!(f.pending_bytes(), r.pending_bytes());
        assert_eq!(f.oldest_pending_ms(), Some(10));
        assert_eq!(r.oldest_pending_ms(), Some(10));
        assert!(f.shards["s"].sealed_times.len() <= SEALED_TIMES_CAP);
        assert_eq!(
            join_all(&forward).unwrap().sealed_times,
            join_all(&reverse).unwrap().sealed_times
        );
        assert_eq!(rebuild(&forward).oldest_pending_ms(), Some(10));
    }

    /// Randomized streams over two shards, three epochs, all sources, with
    /// and without merged. Small streams: every permutation. Large streams:
    /// forward, reverse, shuffle. The in-memory fold must be order
    /// independent and a rebuild from stored records must agree with it,
    /// including the age clock while below the cap.
    #[test]
    fn randomized_streams_fold_order_independently_and_rebuild_identically() {
        let mut seed: u64 = 0x5eed_1234_abcd_0001;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let gen_event = |r: &mut dyn FnMut() -> u64| {
            let shard = if r().is_multiple_of(2) { "a" } else { "b" };
            let epoch = 1 + r() % 3;
            let sealed = r() % 40;
            let merged = if r().is_multiple_of(3) {
                None
            } else {
                Some(r() % 40)
            };
            let source = match r() % 3 {
                0 => EventSource::Writer,
                1 => EventSource::Executor,
                _ => EventSource::Scan,
            };
            DemandEvent {
                v: 1,
                shard: shard.into(),
                sealed_through: sealed,
                sealed_bytes_through: sealed * 7,
                merged_through: merged,
                flushed_at_ms: (r() % 1000) as i64,
                writer_epoch: epoch,
                source,
                merged_epoch: None,
                sealed_times: BTreeMap::new(),
            }
        };
        for _ in 0..200 {
            let events: Vec<_> = (0..5).map(|_| gen_event(&mut next)).collect();
            let canonical = fold_all(&events);
            for perm in permutations(&events) {
                assert_eq!(fold_all(&perm), canonical, "fold order {perm:?}");
                assert_eq!(rebuild(&perm), canonical, "rebuild {perm:?}");
            }
        }
        for _ in 0..50 {
            let events: Vec<_> = (0..200).map(|_| gen_event(&mut next)).collect();
            let canonical = fold_all(&events);
            let mut reversed = events.clone();
            reversed.reverse();
            let mut shuffled = events.clone();
            for i in (1..shuffled.len()).rev() {
                shuffled.swap(i, (next() as usize) % (i + 1));
            }
            for other in [reversed, shuffled] {
                let folded = fold_all(&other);
                assert_eq!(
                    folded.pending_generations(),
                    canonical.pending_generations()
                );
                assert_eq!(folded.pending_bytes(), canonical.pending_bytes());
                for (k, s) in &canonical.shards {
                    assert_eq!(folded.shards[k].sealed, s.sealed);
                    assert_eq!(folded.shards[k].merged, s.merged);
                }
                assert_eq!(
                    rebuild(&other).pending_generations(),
                    canonical.pending_generations()
                );
            }
        }
    }

    #[test]
    fn aggregates_and_revision_tracking() {
        let mut table = TableDemand::default();
        table
            .fold(&ev("a", 10, Some(4), 1, EventSource::Writer), 7, 100)
            .unwrap();
        table
            .fold(&ev("b", 3, Some(3), 1, EventSource::Writer), 9, 200)
            .unwrap();
        table
            .fold(&ev("c", 8, None, 1, EventSource::Writer), 8, 300)
            .unwrap();
        assert_eq!(table.pending_generations(), 14, "a: 6, b: 0, c: 8");
        assert_eq!(table.pending_bytes(), 10_000 + 8_000);
        assert_eq!(table.observed_revision, 9);
        assert_eq!(table.updated_ms, 300);
        assert_eq!(
            table.fold(&ev("b", 3, None, 1, EventSource::Writer), 50, 400),
            Err(Ignored::NotNewer)
        );
        assert_eq!(table.observed_revision, 9, "ignored events do not advance");
    }

    #[test]
    fn schema_version_is_set_and_enforced() {
        assert_eq!(TableDemand::default().v, SCHEMA_VERSION);
        let mut table = TableDemand::default();
        let future = DemandEvent {
            v: 999,
            ..ev("s", 1, None, 1, EventSource::Writer)
        };
        assert_eq!(table.fold(&future, 1, 0), Err(Ignored::UnknownVersion(999)));
        assert!(table.shards.is_empty());
        let legacy = DemandEvent {
            v: 0,
            ..ev("s", 1, None, 1, EventSource::Writer)
        };
        table.fold(&legacy, 1, 0).unwrap();
        let wire = serde_json::to_value(&table).unwrap();
        assert_eq!(wire["v"], SCHEMA_VERSION);
        assert_eq!(join_events(None, &legacy).unwrap().v, SCHEMA_VERSION);
    }

    #[test]
    fn wire_format_is_stable() {
        let event = ev("s", 1, Some(1), 1, EventSource::Writer);
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["v"], 1);
        assert_eq!(json["source"], "writer");
        assert!(json.get("sealed_times").is_none(), "empty times omitted");
        assert!(
            json.get("merged_epoch").is_none(),
            "same-epoch merged omitted"
        );
        let back: DemandEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
        let minimal: DemandEvent = serde_json::from_str(
            r#"{"shard":"s","sealed_through":1,"flushed_at_ms":0,"writer_epoch":1,"source":"scan"}"#,
        )
        .unwrap();
        assert_eq!(minimal.v, 1);
        assert_eq!(minimal.merged_through, None);
        assert_eq!(minimal.merged_epoch, None);
        assert!(minimal.sealed_times.is_empty());
    }

    async fn batch_test_coordinator() -> Coordinator {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        Coordinator::new(client, format!("/demand-batch/{}", uuid::Uuid::new_v4()))
    }

    fn operation_count(
        handle: &metrics_exporter_prometheus::PrometheusHandle,
        operation: &str,
    ) -> u64 {
        let prefix = format!("etcd_operation_attempts_total{{operation=\"{operation}\"}} ");
        handle
            .render()
            .lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .map(|value| value.parse().unwrap())
            .unwrap_or(0)
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn batch_scan_bounds_reads_and_unchanged_scans_do_not_write() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let coordinator = batch_test_coordinator().await;
        let events: Vec<_> = (0..97)
            .map(|i| ev(&format!("s{i}"), 10, Some(4), 1, EventSource::Scan))
            .collect();
        assert_eq!(
            coordinator
                .publish_demand_events("t", &events)
                .await
                .unwrap(),
            97
        );
        assert_eq!(operation_count(&handle, "demand_read"), 4);
        assert_eq!(operation_count(&handle, "demand_publish"), 97);
        let prefix = format!("{}/demand-events/", coordinator.prefix);
        let before = coordinator
            .client
            .clone()
            .get(
                prefix.as_str(),
                Some(etcd_client::GetOptions::new().with_prefix()),
            )
            .await
            .unwrap();
        assert_eq!(before.kvs().len(), 97);
        assert_eq!(
            coordinator
                .publish_demand_events("t", &events)
                .await
                .unwrap(),
            0
        );
        assert_eq!(operation_count(&handle, "demand_read"), 8);
        assert_eq!(operation_count(&handle, "demand_publish"), 97);
        let after = coordinator
            .client
            .clone()
            .get(
                prefix.as_str(),
                Some(etcd_client::GetOptions::new().with_prefix()),
            )
            .await
            .unwrap();
        for (old, new) in before.kvs().iter().zip(after.kvs()) {
            assert_eq!(old.key(), new.key());
            assert_eq!(old.mod_revision(), new.mod_revision());
        }
        assert_eq!(
            coordinator.publish_demand_events("t", &[]).await.unwrap(),
            0
        );
        assert_eq!(operation_count(&handle, "demand_read"), 8);
        assert_eq!(
            coordinator
                .publish_demand_events("t", &events[..1])
                .await
                .unwrap(),
            0
        );
        assert_eq!(operation_count(&handle, "demand_read"), 9);
        println!(
            "97 unchanged shard observations: 4 read calls, 0 writes; single event: 1 read call"
        );
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn batch_stale_read_rejoins_writer_and_executor_progress() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let coordinator = batch_test_coordinator().await;
        coordinator
            .publish_demand_event("t", &ev("s", 10, None, 1, EventSource::Writer))
            .await
            .unwrap();
        let key = coordinator.demand_event_key("t", "s");
        let stale = coordinator
            .client
            .clone()
            .get(key.as_str(), None)
            .await
            .unwrap();
        // Both publishers win after the scan snapshot, before its CAS.
        coordinator
            .publish_demand_event("t", &ev("s", 30, None, 2, EventSource::Writer))
            .await
            .unwrap();
        coordinator
            .publish_merged_watermark(
                "t",
                &ShardMerged {
                    shard: "s".into(),
                    merged_through: 15,
                },
            )
            .await
            .unwrap();
        let reads_before = operation_count(&handle, "demand_read");
        assert!(coordinator
            .join_from_read(&key, &ev("s", 20, Some(18), 1, EventSource::Scan), stale)
            .await
            .unwrap());
        assert_eq!(
            operation_count(&handle, "demand_read") - reads_before,
            1,
            "only the conflicted key is reread"
        );
        let current = coordinator
            .client
            .clone()
            .get(key.as_str(), None)
            .await
            .unwrap();
        let stored = stored_event(current.kvs().first()).unwrap().unwrap();
        assert_eq!(stored.sealed_through, 30);
        assert_eq!(stored.writer_epoch, 2);
        assert_eq!(stored.sealed_bytes_through, 30_000);
        assert_eq!(stored.merged_through, Some(18));
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn batch_duplicate_shards_and_create_race_preserve_both_advancements() {
        let coordinator = batch_test_coordinator().await;
        let events = [
            ev("s", 20, None, 1, EventSource::Writer),
            ev("s", 20, Some(10), 1, EventSource::Scan),
            ev("other", 8, None, 1, EventSource::Scan),
        ];
        assert_eq!(
            coordinator
                .publish_demand_events("t", &events)
                .await
                .unwrap(),
            3
        );
        let key = coordinator.demand_event_key("t", "s");
        let current = coordinator
            .client
            .clone()
            .get(key.as_str(), None)
            .await
            .unwrap();
        let stored = stored_event(current.kvs().first()).unwrap().unwrap();
        assert_eq!(
            (stored.sealed_through, stored.merged_through),
            (20, Some(10))
        );
        assert_eq!(
            coordinator
                .publish_demand_events("t", &events)
                .await
                .unwrap(),
            0
        );
        assert!(
            coordinator
                .client
                .clone()
                .get(coordinator.demand_event_key("t", "other"), None)
                .await
                .unwrap()
                .kvs()
                .len()
                == 1
        );
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn batch_rejects_future_schema_in_snapshot_and_after_conflict() {
        let coordinator = batch_test_coordinator().await;
        let key = coordinator.demand_event_key("t", "s");
        let stale_absent = coordinator
            .client
            .clone()
            .get(key.as_str(), None)
            .await
            .unwrap();
        let mut future = ev("s", 5, None, 1, EventSource::Writer);
        future.v = SCHEMA_VERSION + 1;
        coordinator
            .client
            .clone()
            .put(key.as_str(), serde_json::to_vec(&future).unwrap(), None)
            .await
            .unwrap();
        let before = coordinator
            .client
            .clone()
            .get(key.as_str(), None)
            .await
            .unwrap();
        let event = ev("s", 50, None, 1, EventSource::Scan);
        let error = coordinator
            .join_from_read(&key, &event, stale_absent)
            .await
            .unwrap_err();
        assert!(error.contains("unsupported version"), "{error}");
        let error = coordinator
            .publish_demand_events("t", &[event, ev("other", 1, None, 1, EventSource::Scan)])
            .await
            .unwrap_err();
        assert!(error.contains("unsupported version"), "{error}");
        let after = coordinator
            .client
            .clone()
            .get(key.as_str(), None)
            .await
            .unwrap();
        assert_eq!(
            before.kvs()[0].mod_revision(),
            after.kvs()[0].mod_revision()
        );
        assert_eq!(before.kvs()[0].value(), after.kvs()[0].value());
        // Validate every input before publishing even the first chunk.
        let error = coordinator
            .publish_demand_events(
                "other-target",
                &[ev("ok", 1, None, 1, EventSource::Scan), future],
            )
            .await
            .unwrap_err();
        assert!(error.contains("Version"), "{error}");
        assert!(coordinator
            .client
            .clone()
            .get(coordinator.demand_event_key("other-target", "ok"), None)
            .await
            .unwrap()
            .kvs()
            .is_empty());
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn publish_is_a_join_and_never_lowers() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let coordinator =
            Coordinator::new(client, format!("/demand-publish/{}", uuid::Uuid::new_v4()));
        let publish = |e: DemandEvent| {
            let c = coordinator.clone();
            async move { c.publish_demand_event("t", &e).await.unwrap() }
        };
        assert!(publish(ev("s", 10, None, 1, EventSource::Writer)).await);
        assert!(
            !publish(ev("s", 10, None, 1, EventSource::Writer)).await,
            "duplicate"
        );
        assert!(
            publish(ev("s", 9, None, 1, EventSource::Scan)).await,
            "lower sealed: learns a flush time only"
        );
        assert!(
            !publish(ev("s", 9, None, 1, EventSource::Scan)).await,
            "then nothing"
        );
        assert!(
            publish(ev("s", 10, Some(4), 1, EventSource::Executor)).await,
            "merged advances"
        );
        assert!(
            !publish(ev("s", 10, Some(3), 1, EventSource::Executor)).await,
            "merged regress"
        );
        assert!(
            publish(ev("s", 12, None, 1, EventSource::Writer)).await,
            "sealed advances"
        );
        assert!(
            publish(ev("s", 12, None, 2, EventSource::Writer)).await,
            "same gen, higher epoch"
        );
        assert!(
            publish(ev("other", 1, None, 1, EventSource::Writer)).await,
            "separate shard"
        );
    }

    /// Round-4 finding 5: a stored record from a newer schema is refused,
    /// never overwritten.
    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn stored_record_from_a_newer_schema_is_never_overwritten() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let coordinator = Coordinator::new(
            client.clone(),
            format!("/demand-version/{}", uuid::Uuid::new_v4()),
        );
        let key = coordinator.demand_event_key("t", "s");
        let future = serde_json::json!({
            "v": 999, "shard": "s", "sealed_through": 5, "flushed_at_ms": 0,
            "writer_epoch": 1, "source": "writer", "some_future_field": true
        });
        client
            .put(key.as_str(), future.to_string(), None)
            .await
            .unwrap();
        let before = client.get(key.as_str(), None).await.unwrap().kvs()[0].mod_revision();
        let err = coordinator
            .publish_demand_event("t", &ev("s", 50, None, 1, EventSource::Writer))
            .await
            .unwrap_err();
        assert!(err.contains("unsupported version"), "{err}");
        let err = coordinator
            .publish_merged_watermark(
                "t",
                &ShardMerged {
                    shard: "s".into(),
                    merged_through: 3,
                },
            )
            .await
            .unwrap_err();
        assert!(err.contains("unsupported version"), "{err}");
        let after = client.get(key.as_str(), None).await.unwrap();
        assert_eq!(after.kvs()[0].mod_revision(), before, "record untouched");
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn merged_publish_advances_only_merged_and_survives_writer_races() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let coordinator = Coordinator::new(
            client.clone(),
            format!("/demand-merged/{}", uuid::Uuid::new_v4()),
        );
        assert!(coordinator
            .publish_demand_event("t", &ev("s", 20, None, 3, EventSource::Writer))
            .await
            .unwrap());
        let shard = |m: u64| ShardMerged {
            shard: "s".into(),
            merged_through: m,
        };
        assert!(coordinator
            .publish_merged_watermark("t", &shard(15))
            .await
            .unwrap());
        let key = coordinator.demand_event_key("t", "s");
        let read = |client: etcd_client::Client, key: String| async move {
            serde_json::from_slice::<DemandEvent>(
                client.clone().get(key, None).await.unwrap().kvs()[0].value(),
            )
            .unwrap()
        };
        let after = read(client.clone(), key.clone()).await;
        assert_eq!(after.merged_through, Some(15));
        assert_eq!(after.sealed_through, 20, "writer's sealed kept");
        assert_eq!(after.writer_epoch, 3, "writer's epoch kept");
        assert_eq!(after.sealed_bytes_through, 20_000, "writer's bytes kept");
        assert!(
            !coordinator
                .publish_merged_watermark("t", &shard(12))
                .await
                .unwrap(),
            "lower merged is a no-op"
        );
        let c1 = coordinator.clone();
        let c2 = coordinator.clone();
        let (w, m) = tokio::join!(
            async move {
                c1.publish_demand_event("t", &ev("s", 25, None, 3, EventSource::Writer))
                    .await
                    .unwrap()
            },
            async move { c2.publish_merged_watermark("t", &shard(18)).await.unwrap() }
        );
        assert!(w && m, "both writes must succeed (writer={w}, merged={m})");
        let stored = read(client.clone(), key.clone()).await;
        assert_eq!(
            (stored.sealed_through, stored.merged_through),
            (25, Some(18))
        );
        assert!(coordinator
            .publish_merged_watermark(
                "t",
                &ShardMerged {
                    shard: "new".into(),
                    merged_through: 7
                }
            )
            .await
            .unwrap());
        let fresh = read(client.clone(), coordinator.demand_event_key("t", "new")).await;
        assert_eq!(
            (
                fresh.sealed_through,
                fresh.merged_through,
                fresh.writer_epoch
            ),
            (7, Some(7), 0)
        );
    }
}
