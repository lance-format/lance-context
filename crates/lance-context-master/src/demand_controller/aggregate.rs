//! Table-wide WAL admission from existing writer watermarks. Only the elected
//! demand controller runs this loop. Its streaming snapshot is independent of
//! task dispatch, and retained values omit payloads and generation timestamps.
//!
//! An admitted frontier means "durably queued", not "merged". It is committed
//! atomically with a NEW queue entry. Neither dedupe against a running task nor
//! stale merged watermarks may consume new seals or generate endless empty work.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use etcd_client::{Client, EventType, GetOptions, WatchOptions};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use super::{Policy, RECONNECT};
use crate::state::MasterState;
use crate::task_store::{CoordinationGuard, TaskStore};

const RANGES: &[&str] = &[
    "demand-events",
    "demand-controller/admitted-wal",
    "demand-controller/members",
    "dedupe/merge-wal",
];
const CAPACITY_ERROR: &str = "aggregate WAL informer capacity exceeded; durable inputs retained";
const MAX_TABLES: usize = 32_768;
const MAX_SHARDS: usize = 524_288;
const MAX_BYTES: usize = 128 * 1024 * 1024;
const MAX_VALUE_BYTES: usize = 256 * 1024;
const MAX_SHARDS_PER_TABLE: usize = 1024;

#[derive(Debug, Deserialize)]
struct Event {
    #[serde(default = "version")]
    v: u32,
    shard: String,
    sealed_through: u64,
    #[serde(default)]
    merged_through: Option<u64>,
}

fn version() -> u32 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Frontier {
    #[serde(default = "version")]
    pub v: u32,
    pub shards: BTreeMap<String, u64>,
    /// Identity of the task entrusted with these seals, retained across restart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
}

impl Frontier {
    fn validate(&self) -> Result<(), String> {
        if self.v != 1
            || self.shards.len() > MAX_SHARDS_PER_TABLE
            || self.shards.keys().any(|id| id.len() > 4096)
            || self
                .task
                .as_ref()
                .is_some_and(|id| id.is_empty() || id.len() > 256)
        {
            return Err("unsupported or oversized WAL admission frontier".into());
        }
        Ok(())
    }
}

#[derive(Default)]
struct Shard {
    sealed: u64,
    merged: u64,
    admitted: u64,
    revision: i64,
    invalid: bool,
}

#[derive(Default)]
struct Table {
    shards: BTreeMap<String, Shard>,
    frontier_revision: i64,
    frontier_seen: i64,
    frontier_invalid: bool,
    dedupe_revision: i64,
    dedupe_invalid: bool,
    retry_after: Option<Instant>,
    dedupe_id: Option<String>,
    // Resolve a dedupe record at most once until its identity changes. New
    // writer events do not repeatedly GET an already-known running task.
    checked_dedupe: Option<String>,
    frontier_task: Option<String>,
}

impl Table {
    fn pending(&self) -> u64 {
        self.shards.values().fold(0u64, |sum, s| {
            sum.saturating_add(s.sealed.saturating_sub(s.merged.max(s.admitted)))
        })
    }

    fn frontier(&self) -> Frontier {
        Frontier {
            v: 1,
            task: None,
            shards: self
                .shards
                .iter()
                .map(|(id, s)| (id.clone(), s.admitted.max(s.sealed)))
                .collect(),
        }
    }

    fn previous_frontier(&self) -> Frontier {
        Frontier {
            v: 1,
            task: self.frontier_task.clone().or_else(|| {
                (self.frontier_revision > 0 && self.frontier_revision == self.dedupe_revision)
                    .then(|| self.dedupe_id.clone())
                    .flatten()
            }),
            shards: self
                .shards
                .iter()
                .map(|(id, s)| (id.clone(), s.admitted))
                .collect(),
        }
    }

    fn owns_outstanding(&self) -> bool {
        !self.frontier_invalid
            && !self.dedupe_invalid
            && self.dedupe_id.is_some()
            && (self.frontier_task.as_ref() == self.dedupe_id.as_ref()
                || (self.frontier_task.is_none()
                    && self.frontier_revision > 0
                    && self.frontier_revision == self.dedupe_revision))
    }

    fn blocked(&self) -> bool {
        self.frontier_invalid
            || self.dedupe_invalid
            || self.shards.values().any(|s| s.invalid)
            || (self.dedupe_id.is_some() && self.dedupe_id == self.checked_dedupe)
    }
}

struct Cache {
    tables: BTreeMap<String, Table>,
    members: BTreeMap<Vec<u8>, Policy>,
    member_bytes: BTreeMap<Vec<u8>, usize>,
    ready: BTreeSet<(Instant, String)>,
    due: BTreeMap<String, Instant>,
    revisions: [i64; 4],
    shards: usize,
    bytes: usize,
    outstanding: BTreeSet<String>,
    waiting: BTreeMap<String, i64>,
    pending: Option<Attempt>,
}

struct Attempt {
    target: String,
    revision: i64,
    previous: Vec<u8>,
    retry_at: Instant,
}

impl Cache {
    fn new() -> Self {
        Self {
            tables: BTreeMap::new(),
            members: BTreeMap::new(),
            member_bytes: BTreeMap::new(),
            ready: BTreeSet::new(),
            due: BTreeMap::new(),
            revisions: [0; 4],
            shards: 0,
            bytes: 0,
            outstanding: BTreeSet::new(),
            waiting: BTreeMap::new(),
            pending: None,
        }
    }

    fn capacity(&self) -> usize {
        // One queued turn per advertised execution slot, in addition to running
        // work. Old members do not invent capacity. Scale with live masters.
        self.members
            .values()
            .fold(0usize, |sum, p| {
                sum.saturating_add(p.aggregate_merge_slots.saturating_mul(2))
            })
            .min(4096)
    }

    fn refresh_outstanding(&mut self, target: &str) {
        let reserved = self.pending.as_ref().is_some_and(|p| p.target == target);
        if reserved || self.tables.get(target).is_some_and(Table::owns_outstanding) {
            self.outstanding.insert(target.to_string());
        } else {
            self.outstanding.remove(target);
        }
    }

    fn wait_capacity(&mut self, target: &str) {
        if let Some(due) = self.due.remove(target) {
            self.ready.remove(&(due, target.to_string()));
        }
        let revision = self.tables[target]
            .shards
            .values()
            .map(|s| s.revision)
            .max()
            .unwrap_or(0);
        self.waiting.insert(target.to_string(), revision);
    }

    fn wake_capacity(&mut self) {
        if self.pending.is_some() || self.outstanding.len() >= self.capacity() {
            return;
        }
        // Current writes get the next available credit instead of waiting for
        // every historical table. Existing periodic discovery still serves old
        // storage backlog independently of these advisory hints.
        let Some(target) = self
            .waiting
            .iter()
            .max_by_key(|(_, revision)| *revision)
            .map(|(target, _)| target.clone())
        else {
            return;
        };
        self.waiting.remove(&target);
        self.mark(&target, Duration::ZERO);
    }

    fn mark(&mut self, target: &str, delay: Duration) {
        if self.waiting.contains_key(target) {
            self.wait_capacity(target);
            return;
        }
        let next = (Instant::now() + delay).max(
            self.tables
                .get(target)
                .and_then(|table| table.retry_after)
                .unwrap_or_else(Instant::now),
        );
        if let Some(old) = self.due.get(target) {
            if *old <= next {
                return;
            }
            self.ready.remove(&(*old, target.to_string()));
        }
        self.due.insert(target.to_string(), next);
        self.ready.insert((next, target.to_string()));
    }

    fn defer_error(&mut self, target: &str) {
        let retry_after = Instant::now() + super::REVISIT;
        if let Some(pending) = self.pending.as_mut() {
            pending.retry_at = retry_after;
        }
        if let Some(table) = self.tables.get_mut(target) {
            table.retry_after = Some(retry_after);
        }
        if let Some(old) = self.due.remove(target) {
            self.ready.remove(&(old, target.to_string()));
        }
        self.mark(target, super::REVISIT);
    }

    fn table(&mut self, target: &str, shard_ids: &[&str]) -> Result<&mut Table, String> {
        let old = self.tables.get(target);
        let additional: BTreeSet<_> = shard_ids
            .iter()
            .filter(|id| old.is_none_or(|table| !table.shards.contains_key(**id)))
            .copied()
            .collect();
        let extra = if old.is_none() {
            1280 + target.len() * 6
        } else {
            0
        } + additional.iter().map(|id| 192 + id.len()).sum::<usize>();
        if (old.is_none() && self.tables.len() >= MAX_TABLES)
            || old.map_or(0, |t| t.shards.len()) + additional.len() > MAX_SHARDS_PER_TABLE
            || self.shards + additional.len() > MAX_SHARDS
            || self.bytes + extra > MAX_BYTES
        {
            return Err(CAPACITY_ERROR.into());
        }
        self.bytes += extra;
        self.shards += additional.len();
        let table = self.tables.entry(target.to_string()).or_default();
        for id in additional {
            table.shards.entry(id.to_string()).or_default();
        }
        Ok(table)
    }

    fn apply(
        &mut self,
        range: usize,
        prefix: &str,
        key: &[u8],
        value: Option<&[u8]>,
        revision: i64,
    ) -> Result<(), String> {
        let Some(location) = std::str::from_utf8(key).ok() else {
            invalid_input("key");
            return Ok(());
        };
        let Some(rest) = location.strip_prefix(&super::prefix_key(prefix, RANGES[range])) else {
            invalid_input("key");
            return Ok(());
        };
        if range == 2 {
            let policy = value.and_then(|value| {
                if value.len() > MAX_VALUE_BYTES {
                    return None;
                }
                serde_json::from_slice::<Policy>(value)
                    .ok()
                    .filter(|p| p.v == 1)
            });
            if let Some(policy) = policy {
                if !self.members.contains_key(key) && self.members.len() >= 1024 {
                    return Err(CAPACITY_ERROR.into());
                }
                let charge = key.len() * 2 + value.unwrap().len() * 4 + 512;
                let bytes = self.bytes - self.member_bytes.get(key).copied().unwrap_or(0) + charge;
                if bytes > MAX_BYTES {
                    return Err(CAPACITY_ERROR.into());
                }
                self.bytes = bytes;
                self.member_bytes.insert(key.to_vec(), charge);
                self.members.insert(key.to_vec(), policy);
            } else {
                // A corrupt member cannot preserve an old broad policy or
                // poison every table's snapshot. A later valid Put repairs it.
                if value.is_some() {
                    invalid_input("member");
                }
                self.members.remove(key);
                self.bytes -= self.member_bytes.remove(key).unwrap_or(0);
            }
            let targets: Vec<_> = self.tables.keys().cloned().collect();
            for target in targets {
                self.mark(&target, Duration::ZERO);
            }
            self.wake_capacity();
            return Ok(());
        }
        let (encoded, shard) = if range == 0 {
            let Some((target, shard)) = rest.split_once('/') else {
                invalid_input("key");
                return Ok(());
            };
            (target, Some(shard))
        } else {
            (rest, None)
        };
        // Validate ASCII before the shared hex decoder's byte slicing.
        if !encoded.is_ascii() || encoded.len() > 8192 {
            invalid_input("key");
            return Ok(());
        }
        let Some(target) = crate::planner::unhex(encoded) else {
            invalid_input("key");
            return Ok(());
        };
        match range {
            0 => {
                let id = shard.unwrap();
                if id.len() > 4096 {
                    invalid_input("key");
                    return Ok(());
                }
                let table = self.table(&target, &[id])?;
                let state = table.shards.get_mut(id).unwrap();
                if revision <= state.revision {
                    return Ok(());
                }
                let event = value
                    .map(|value| {
                        if value.len() > MAX_VALUE_BYTES {
                            return Err("oversized WAL event".into());
                        }
                        let event: Event =
                            serde_json::from_slice(value).map_err(|e| e.to_string())?;
                        lance_context_merge::demand::TableDemand::check_version(event.v)
                            .map_err(|e| format!("unsupported WAL event: {e:?}"))?;
                        if event.shard != id {
                            return Err("WAL event shard/key mismatch".into());
                        }
                        Ok::<_, String>(event)
                    })
                    .transpose();
                let event = match event {
                    Ok(event) => event,
                    Err(error) => {
                        state.revision = revision;
                        state.invalid = true;
                        metrics::counter!("master_aggregate_wal_invalid_inputs_total", "kind" => "event").increment(1);
                        tracing::warn!(%error, %target, shard = id, "invalid WAL hint quarantined; other tables continue");
                        return Ok(());
                    }
                };
                state.invalid = false;
                state.revision = revision;
                if let Some(event) = event {
                    // Counts use the same monotonic per-dimension join as
                    // TableDemand::fold; writer epochs never reset numbering.
                    state.sealed = state.sealed.max(event.sealed_through);
                    state.merged = state.merged.max(event.merged_through.unwrap_or(0));
                } else {
                    state.sealed = 0;
                    state.merged = 0;
                }
            }
            1 => {
                let frontier = value
                    .map(|value| {
                        if value.len() > MAX_VALUE_BYTES {
                            return Err("oversized WAL frontier".into());
                        }
                        let frontier: Frontier =
                            serde_json::from_slice(value).map_err(|e| e.to_string())?;
                        frontier.validate()?;
                        Ok::<_, String>(frontier)
                    })
                    .transpose();
                let frontier = match frontier {
                    Ok(frontier) => frontier,
                    Err(error) => {
                        let table = self.table(&target, &[])?;
                        if revision > table.frontier_seen {
                            table.frontier_seen = revision;
                            table.frontier_invalid = true;
                            metrics::counter!("master_aggregate_wal_invalid_inputs_total", "kind" => "frontier").increment(1);
                            tracing::warn!(%error, %target, "invalid WAL frontier quarantined; other tables continue");
                        }
                        self.refresh_outstanding(&target);
                        self.wake_capacity();
                        return Ok(());
                    }
                };
                let ids: Vec<_> = frontier
                    .as_ref()
                    .map(|f| f.shards.keys().map(String::as_str).collect())
                    .unwrap_or_default();
                let table = self.table(&target, &ids)?;
                if revision <= table.frontier_seen {
                    return Ok(());
                }
                table.frontier_invalid = false;
                table.frontier_seen = revision;
                table.frontier_revision = revision;
                table.frontier_task = frontier.as_ref().and_then(|f| f.task.clone());
                // Deletion is an explicit reset. Ordinary recovery never
                // deletes a frontier; task/history pruning does not touch it.
                for shard in table.shards.values_mut() {
                    shard.admitted = 0;
                }
                if let Some(frontier) = frontier {
                    for (id, admitted) in frontier.shards {
                        table.shards.get_mut(&id).unwrap().admitted = admitted;
                    }
                }
                if value.is_none() {
                    table.frontier_revision = 0;
                }
            }
            3 => {
                let table = self.table(&target, &[])?;
                if revision <= table.dedupe_revision {
                    return Ok(());
                }
                table.dedupe_revision = revision;
                let id = value.and_then(|v| {
                    if v.is_empty() || v.len() > 256 {
                        return None;
                    }
                    std::str::from_utf8(v).ok().map(str::to_string)
                });
                table.dedupe_invalid = value.is_some() && id.is_none();
                if table.dedupe_invalid {
                    table.checked_dedupe = None;
                    invalid_input("dedupe");
                    self.refresh_outstanding(&target);
                    self.wake_capacity();
                    return Ok(());
                }
                if table.dedupe_id != id {
                    table.checked_dedupe = None;
                }
                table.dedupe_id = id;
            }
            _ => unreachable!(),
        }
        self.refresh_outstanding(&target);
        self.mark(&target, Duration::ZERO);
        self.wake_capacity();
        Ok(())
    }

    fn take_ready(&mut self) -> Option<String> {
        if let Some(pending) = &self.pending {
            return (pending.retry_at <= Instant::now()).then(|| pending.target.clone());
        }
        if self.ready.first()?.0 > Instant::now() {
            return None;
        }
        let (_, target) = self.ready.pop_first()?;
        self.due.remove(&target);
        Some(target)
    }

    fn next_due(&self) -> Instant {
        if let Some(pending) = &self.pending {
            return pending.retry_at;
        }
        self.ready.first().map_or_else(
            || Instant::now() + Duration::from_secs(3600),
            |(due, _)| *due,
        )
    }
}

fn invalid_input(kind: &'static str) {
    metrics::counter!("master_aggregate_wal_invalid_inputs_total", "kind" => kind).increment(1);
}

// Only failed bootstrap attempts are paced. A healthy controller uses Watch
// and never polls the fleet. Hard capacity failures must not repeatedly read
// the entire prefix every few seconds while the other master loops keep working.
fn bootstrap_delay(error: &str, failures: u32) -> Duration {
    if error == CAPACITY_ERROR {
        return Duration::from_secs(300);
    }
    Duration::from_secs(
        5u64.saturating_mul(1u64 << failures.saturating_sub(1).min(6))
            .min(300),
    )
}

pub(crate) fn frontier_key(prefix: &str, target: &str) -> String {
    let hex: String = target.bytes().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}/demand-controller/admitted-wal/{hex}",
        prefix.trim_end_matches('/')
    )
}

pub(crate) enum Admission {
    Admitted {
        revision: i64,
        frontier: Vec<u8>,
        task: String,
    },
    Blocked {
        id: String,
        revision: i64,
    },
    Changed,
}

pub(crate) struct AdmissionSnapshot {
    pub frontier: (i64, Option<Vec<u8>>),
    pub dedupe: (i64, Option<Vec<u8>>),
}

fn dedupe_key(prefix: &str, target: &str) -> String {
    let hex: String = target.bytes().map(|b| format!("{b:02x}")).collect();
    format!("{}/dedupe/merge-wal/{hex}", prefix.trim_end_matches('/'))
}

async fn resolve_pending(
    publisher: &TaskStore,
    prefix: &str,
    cache: &mut Cache,
) -> Result<(), String> {
    let pending = cache.pending.as_ref().ok_or("missing pending admission")?;
    let target = pending.target.clone();
    let observed = publisher
        .resolve_aggregate_admission(&target, pending.revision, &pending.previous)
        .await
        .map_err(|e| e.to_string())?;
    cache.apply(
        1,
        prefix,
        frontier_key(prefix, &target).as_bytes(),
        observed.frontier.1.as_deref(),
        observed.frontier.0,
    )?;
    cache.apply(
        3,
        prefix,
        dedupe_key(prefix, &target).as_bytes(),
        observed.dedupe.1.as_deref(),
        observed.dedupe.0,
    )?;
    cache.pending = None;
    cache.refresh_outstanding(&target);
    cache.mark(&target, Duration::ZERO);
    cache.wake_capacity();
    Ok(())
}

async fn snapshot(client: &Client, prefix: &str) -> Result<Cache, String> {
    let mut cache = Cache::new();
    let mut client = client.clone();
    let mut revision = 0;
    for (index, range) in RANGES.iter().enumerate() {
        let start = super::prefix_key(prefix, range);
        let mut end = start.as_bytes().to_vec();
        *end.last_mut().unwrap() += 1;
        let mut key = start.into_bytes();
        let mut limit = 128;
        loop {
            let mut options = GetOptions::new().with_range(end.clone()).with_limit(limit);
            if revision > 0 {
                options = options.with_revision(revision);
            }
            metrics::counter!("master_aggregate_wal_list_rpcs_total").increment(1);
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
                    return Err("missing aggregate snapshot revision".into());
                }
            }
            for kv in response.kvs() {
                cache.apply(index, prefix, kv.key(), Some(kv.value()), kv.mod_revision())?;
            }
            if !response.more() {
                break;
            }
            key = response
                .kvs()
                .last()
                .ok_or("empty aggregate continuation page")?
                .key()
                .to_vec();
            key.push(0);
            tokio::task::yield_now().await;
        }
    }
    cache.revisions.fill(revision);
    metrics::counter!("master_aggregate_wal_snapshots_total").increment(1);
    Ok(cache)
}

async fn watch(
    client: &Client,
    prefix: &str,
    cache: &Cache,
) -> Result<etcd_client::WatchStream, String> {
    let options = |index: usize| {
        WatchOptions::new()
            .with_prefix()
            .with_start_revision(cache.revisions[index] + 1)
            .with_watch_id(index as i64 + 1)
            .with_progress_notify()
    };
    let mut stream = client
        .clone()
        .watch(super::prefix_key(prefix, RANGES[0]), Some(options(0)))
        .await
        .map_err(|e| e.to_string())?;
    for (index, range) in RANGES.iter().enumerate().skip(1) {
        stream
            .watch(super::prefix_key(prefix, range), Some(options(index)))
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(stream)
}

fn apply_response(
    cache: &mut Cache,
    prefix: &str,
    response: &etcd_client::WatchResponse,
) -> Result<(), String> {
    if response.canceled() {
        if response.compact_revision() > 0 {
            return Err("compacted".into());
        }
        return Err(format!(
            "aggregate watch canceled: {}",
            response.cancel_reason()
        ));
    }
    if response.created() {
        return Ok(());
    }
    let index =
        usize::try_from(response.watch_id() - 1).map_err(|_| "invalid aggregate watch id")?;
    if index >= RANGES.len() {
        return Err("unknown aggregate watch id".into());
    }
    for event in response.events() {
        let kv = event.kv().ok_or("aggregate event missing key")?;
        cache.apply(
            index,
            prefix,
            kv.key(),
            (event.event_type() != EventType::Delete).then_some(kv.value()),
            kv.mod_revision(),
        )?;
    }
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
    cache: &mut Cache,
    target: &str,
) -> Result<(), String> {
    let result = reconcile_table(publisher, prefix, cache, target).await;
    // A waiter may no longer qualify when it receives its turn. Return that
    // unused credit even on no-op/error paths so other tables cannot strand.
    cache.wake_capacity();
    result
}

async fn reconcile_table(
    publisher: &TaskStore,
    prefix: &str,
    cache: &mut Cache,
    target: &str,
) -> Result<(), String> {
    if cache.pending.is_some() {
        return resolve_pending(publisher, prefix, cache).await;
    }
    let Some(threshold) = cache
        .members
        .values()
        .filter(|p| {
            p.automatic && !p.draining.iter().any(|t| t == target) && p.admits_request(target, None)
        })
        .filter_map(|p| p.aggregate_min_generations)
        .map(|n| n.max(1) as u64)
        .min()
    else {
        return Ok(());
    };
    let table = cache.tables.get(target).ok_or("missing aggregate table")?;
    if table.blocked() || table.pending() < threshold {
        return Ok(());
    }
    if !cache.outstanding.contains(target) && cache.outstanding.len() >= cache.capacity() {
        cache.wait_capacity(target);
        return Ok(());
    }
    let frontier = table.frontier();
    let value = serde_json::to_vec(&frontier).map_err(|e| e.to_string())?;
    if value.len() > MAX_VALUE_BYTES {
        return Err("oversized admission frontier".into());
    }
    let revision = table.frontier_revision;
    let previous = serde_json::to_vec(&table.previous_frontier()).map_err(|e| e.to_string())?;
    if previous.len() > MAX_VALUE_BYTES {
        return Err("oversized previous admission frontier".into());
    }
    // Reserve before the RPC. A lost response must not admit another table
    // against this credit until the original CAS has been fenced/resolved.
    cache.pending = Some(Attempt {
        target: target.to_string(),
        revision,
        previous,
        retry_at: Instant::now(),
    });
    cache.refresh_outstanding(target);
    let result = publisher
        .enqueue_aggregate_wal(target, revision, &value)
        .await
        .map_err(|e| e.to_string())?;
    match result {
        Admission::Admitted {
            revision,
            frontier,
            task,
        } => {
            cache.apply(
                1,
                prefix,
                frontier_key(prefix, target).as_bytes(),
                Some(&frontier),
                revision,
            )?;
            cache.apply(
                3,
                prefix,
                dedupe_key(prefix, target).as_bytes(),
                Some(task.as_bytes()),
                revision,
            )?;
            cache.tables.get_mut(target).unwrap().checked_dedupe = Some(task);
            cache.pending = None;
            cache.refresh_outstanding(target);
            metrics::counter!("master_aggregate_wal_admission_total", "outcome" => "queued")
                .increment(1);
        }
        Admission::Blocked { id, revision } => {
            let table = cache.tables.get_mut(target).unwrap();
            if revision >= table.dedupe_revision {
                table.dedupe_revision = revision;
                table.dedupe_id = Some(id.clone());
                table.checked_dedupe = Some(id);
            }
            cache.pending = None;
            cache.refresh_outstanding(target);
            metrics::counter!("master_aggregate_wal_admission_total", "outcome" => "active")
                .increment(1);
        }
        Admission::Changed => {
            // A concurrent enqueue or frontier update won. The watch normally
            // wakes us first; this bounded retry also handles stale dedupe cleanup.
            resolve_pending(publisher, prefix, cache).await?;
        }
    }
    cache.wake_capacity();
    Ok(())
}

/// Shares the dispatch leader's lease/fence, but never delays its snapshot,
/// offers or task recovery while bootstrapping 13k tables of writer hints.
struct MetricsGuard;
impl Drop for MetricsGuard {
    fn drop(&mut self) {
        metrics::gauge!("master_aggregate_wal_cache_tables").set(0.0);
        metrics::gauge!("master_aggregate_wal_cache_shards").set(0.0);
        metrics::gauge!("master_aggregate_wal_cache_bytes").set(0.0);
        metrics::gauge!("master_aggregate_wal_outstanding").set(0.0);
        metrics::gauge!("master_aggregate_wal_waiting").set(0.0);
        metrics::gauge!("master_aggregate_wal_capacity").set(0.0);
    }
}

pub(super) async fn run(state: &Arc<MasterState>, session: &CoordinationGuard) {
    let _metrics_guard = MetricsGuard;
    let prefix = state.config.etcd.prefix().trim_end_matches('/');
    let publisher = state.task_store.discovery_publisher(session);
    let client = state.task_store.etcd_client();
    let mut bootstrap_failures = 0u32;
    loop {
        let mut cache = match snapshot(client, prefix).await {
            Ok(cache) => cache,
            Err(error) => {
                metrics::counter!("master_aggregate_wal_errors_total", "phase" => "snapshot")
                    .increment(1);
                tracing::warn!(%error, "aggregate WAL bootstrap deferred; dispatch remains active");
                bootstrap_failures = bootstrap_failures.saturating_add(1);
                tokio::time::sleep(bootstrap_delay(&error, bootstrap_failures)).await;
                continue;
            }
        };
        bootstrap_failures = 0;
        loop {
            let mut stream = match watch(client, prefix, &cache).await {
                Ok(stream) => stream,
                Err(error) => {
                    tracing::warn!(%error, "aggregate WAL watch reconnecting");
                    tokio::time::sleep(RECONNECT).await;
                    continue;
                }
            };
            let failure = loop {
                tokio::select! {
                    response = stream.message() => {
                        match response {
                            Ok(Some(response)) => if let Err(error) = apply_response(&mut cache, prefix, &response) { break error; },
                            Ok(None) => break "aggregate watch ended".to_string(),
                            Err(error) => break error.to_string(),
                        }
                    }
                    _ = tokio::time::sleep_until(cache.next_due()) => {
                        if let Some(target) = cache.take_ready() {
                            let Some(operation) = state.admission.try_admit_named("aggregate_wal_demand") else { return; };
                            operation.phase("admit_table_watermarks");
                            match tokio::time::timeout(Duration::from_secs(10), reconcile(&publisher, prefix, &mut cache, &target)).await {
                                Ok(Ok(())) => operation.observed_work(),
                                error => {
                                    metrics::counter!("master_aggregate_wal_errors_total", "phase" => "admit").increment(1);
                                    tracing::warn!(?error, %target, "aggregate WAL demand retained for retry");
                                    cache.defer_error(&target);
                                }
                            }
                        }
                    }
                }
                metrics::gauge!("master_aggregate_wal_cache_tables").set(cache.tables.len() as f64);
                metrics::gauge!("master_aggregate_wal_cache_shards").set(cache.shards as f64);
                metrics::gauge!("master_aggregate_wal_cache_bytes").set(cache.bytes as f64);
                metrics::gauge!("master_aggregate_wal_outstanding")
                    .set(cache.outstanding.len() as f64);
                metrics::gauge!("master_aggregate_wal_waiting").set(cache.waiting.len() as f64);
                metrics::gauge!("master_aggregate_wal_capacity").set(cache.capacity() as f64);
            };
            metrics::counter!("master_aggregate_wal_errors_total", "phase" => "watch").increment(1);
            tracing::warn!(%failure, "aggregate WAL watch reconnecting; task dispatch continues");
            tokio::time::sleep(RECONNECT).await;
            if failure == "compacted" || failure.contains("message length") {
                // A fresh snapshot alone cannot fence an RPC still in flight
                // under this same leader. Resolve it before discarding its
                // reservation, otherwise replay may exceed the admission cap.
                while let Some(attempt) = &cache.pending {
                    let target = attempt.target.clone();
                    tokio::time::sleep_until(attempt.retry_at).await;
                    let Some(operation) =
                        state.admission.try_admit_named("aggregate_wal_resolution")
                    else {
                        return;
                    };
                    operation.phase("resolve_before_snapshot");
                    match tokio::time::timeout(
                        Duration::from_secs(10),
                        resolve_pending(&publisher, prefix, &mut cache),
                    )
                    .await
                    {
                        Ok(Ok(())) => operation.observed_work(),
                        error => {
                            metrics::counter!("master_aggregate_wal_errors_total", "phase" => "resolve_snapshot").increment(1);
                            tracing::warn!(?error, %target, "retaining uncertain admission before aggregate resnapshot");
                            cache.defer_error(&target);
                        }
                    }
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MasterConfig;
    use clap::Parser;
    use lance_context_api::{TaskKind, TaskState};
    use lance_context_merge::demand::{DemandEvent, EventSource, TableDemand};

    fn event(shard: &str, sealed: u64, merged: Option<u64>) -> Vec<u8> {
        serde_json::to_vec(&DemandEvent {
            v: 1,
            shard: shard.into(),
            sealed_through: sealed,
            sealed_bytes_through: sealed * 100,
            merged_through: merged,
            merged_epoch: None,
            flushed_at_ms: 1,
            writer_epoch: 1,
            source: EventSource::Writer,
            sealed_times: BTreeMap::new(),
        })
        .unwrap()
    }

    fn publish(
        cache: &mut Cache,
        target: &str,
        shard: &str,
        sealed: u64,
        merged: Option<u64>,
        revision: i64,
    ) {
        let encoded: String = target.bytes().map(|b| format!("{b:02x}")).collect();
        cache
            .apply(
                0,
                "/p",
                format!("/p/demand-events/{encoded}/{shard}").as_bytes(),
                Some(&event(shard, sealed, merged)),
                revision,
            )
            .unwrap();
    }

    #[test]
    fn aggregate_counts_cross_shard_threshold_and_join_matches_existing_semantics() {
        let mut cache = Cache::new();
        publish(&mut cache, "t", "a", 7, None, 1);
        publish(&mut cache, "t", "b", 7, None, 2);
        assert_eq!(cache.tables["t"].pending(), 14);
        let mut reference = TableDemand::default();
        for (shard, sealed, merged, revision) in [
            ("a", 7, None, 1),
            ("b", 7, None, 2),
            ("a", 4, Some(5), 3),
            ("b", 10, Some(2), 4),
            ("a", 9, Some(3), 5),
            ("a", 5, Some(8), 6),
        ] {
            let value = event(shard, sealed, merged);
            let e: DemandEvent = serde_json::from_slice(&value).unwrap();
            let _ = reference.fold(&e, revision, 1);
            publish(&mut cache, "t", shard, sealed, merged, revision);
        }
        let expected: u64 = reference
            .shards
            .values()
            .map(|s| s.pending_generations())
            .sum();
        assert_eq!(cache.tables["t"].pending(), expected);
        assert_eq!(expected, 9);
    }

    #[test]
    fn stale_watermarks_do_not_regenerate_work_but_new_seals_survive() {
        let mut cache = Cache::new();
        publish(&mut cache, "t", "a", 100, Some(1), 1);
        publish(&mut cache, "t", "b", 100, Some(1), 2);
        let old = cache.tables["t"].frontier();
        let value = serde_json::to_vec(&old).unwrap();
        cache
            .apply(1, "/p", frontier_key("/p", "t").as_bytes(), Some(&value), 3)
            .unwrap();
        assert_eq!(cache.tables["t"].pending(), 0);
        for rev in 4..100 {
            publish(&mut cache, "t", "a", 100, Some(1), rev);
        }
        assert_eq!(cache.tables["t"].pending(), 0);
        publish(&mut cache, "t", "a", 107, Some(1), 101);
        publish(&mut cache, "t", "b", 107, Some(1), 102);
        // Replaying the earlier admission cannot consume writes that arrived
        // after its snapshot, even when merged progress was never published.
        cache
            .apply(1, "/p", frontier_key("/p", "t").as_bytes(), Some(&value), 3)
            .unwrap();
        assert_eq!(cache.tables["t"].pending(), 14);
        let mut restarted = Cache::new();
        restarted
            .apply(1, "/p", frontier_key("/p", "t").as_bytes(), Some(&value), 3)
            .unwrap();
        publish(&mut restarted, "t", "b", 107, Some(1), 102);
        publish(&mut restarted, "t", "a", 107, Some(1), 101);
        assert_eq!(restarted.tables["t"].pending(), 14);
    }

    #[test]
    fn cache_handles_thirteen_thousand_tables_without_retaining_payload_maps() {
        let mut cache = Cache::new();
        for table in 0..13_000 {
            let target = format!("traffic_workload_experiment_{table:05}_with_a_realistic_name");
            for shard in 0..20 {
                publish(
                    &mut cache,
                    &target,
                    &format!("worker-{shard:02}"),
                    7,
                    None,
                    1,
                );
            }
        }
        assert_eq!(cache.tables.len(), 13_000);
        assert_eq!(cache.shards, 260_000);
        assert_eq!(cache.ready.len(), 13_000);
        assert!(cache.bytes < MAX_BYTES);
        assert!(cache.tables.values().all(|t| t.pending() == 140));
        let before = cache.bytes;
        let target = "traffic_workload_experiment_00000_with_a_realistic_name";
        for revision in 2..1000 {
            publish(&mut cache, target, "worker-00", 7, None, revision);
        }
        assert_eq!(cache.bytes, before);
        assert_eq!(cache.ready.len(), 13_000);
    }

    #[test]
    fn memory_limit_preserves_prior_state_and_queued_wakeup() {
        let mut cache = Cache::new();
        publish(&mut cache, "t", "a", 7, None, 1);
        cache.bytes = MAX_BYTES;
        let result = cache.apply(
            0,
            "/p",
            b"/p/demand-events/74/new",
            Some(&event("new", 7, None)),
            2,
        );
        assert!(result.is_err());
        assert_eq!(cache.tables["t"].pending(), 7);
        assert_eq!(cache.shards, 1);
        assert_eq!(cache.ready.len(), 1);
    }

    #[test]
    fn deletion_replay_does_not_restore_a_removed_frontier() {
        let mut cache = Cache::new();
        publish(&mut cache, "t", "a", 10, None, 1);
        let value = serde_json::to_vec(&cache.tables["t"].frontier()).unwrap();
        let key = frontier_key("/p", "t");
        cache
            .apply(1, "/p", key.as_bytes(), Some(&value), 2)
            .unwrap();
        cache.apply(1, "/p", key.as_bytes(), None, 3).unwrap();
        cache
            .apply(1, "/p", key.as_bytes(), Some(&value), 2)
            .unwrap();
        assert_eq!(cache.tables["t"].frontier_revision, 0);
        assert_eq!(cache.tables["t"].pending(), 10);
    }

    #[test]
    fn older_members_do_not_invent_an_unknown_threshold() {
        let config = MasterConfig::parse_from(["test"]);
        let mut value = serde_json::to_value(Policy::from_config(&config)).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("aggregate_min_generations");
        let old: Policy = serde_json::from_value(value).unwrap();
        assert!(old.aggregate_min_generations.is_none());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn atomic_admission_preserves_concurrent_seals_and_fences_old_owner() {
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let session = store
            .discovery_session(serde_json::to_vec(&Policy::from_config(&cfg)).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let prefix = cfg.etcd.prefix();
        let mut cache = Cache::new();
        cache
            .members
            .insert(b"member".to_vec(), Policy::from_config(&cfg));
        // Use fixture prefix for durable operations; the projection helper's
        // prefix is only a key parser for this local in-memory state.
        publish(&mut cache, "t", "a", 7, None, 1);
        publish(&mut cache, "t", "b", 7, None, 2);
        reconcile(&publisher, prefix, &mut cache, "t")
            .await
            .unwrap();
        assert_eq!(cache.tables["t"].pending(), 0);
        let tasks = store.list().await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].kind, TaskKind::MergeWal);
        assert_eq!(tasks[0].state, TaskState::Queued);
        let first_revision = cache.tables["t"].frontier_revision;
        publish(&mut cache, "t", "a", 14, None, first_revision + 1);
        publish(&mut cache, "t", "b", 14, None, first_revision + 2);
        reconcile(&publisher, prefix, &mut cache, "t")
            .await
            .unwrap();
        assert!(cache.tables["t"].blocked());
        assert_eq!(cache.tables["t"].pending(), 14);
        assert_eq!(cache.tables["t"].frontier_revision, first_revision);
        assert_eq!(store.list().await.unwrap().len(), 1);
        store.release_coordination_lock(session).await.unwrap();
        let value = serde_json::to_vec(&cache.tables["t"].frontier()).unwrap();
        assert!(matches!(
            publisher
                .enqueue_aggregate_wal("other", 0, &value)
                .await
                .unwrap(),
            Admission::Changed
        ));
        let result = store
            .etcd_client()
            .clone()
            .get(frontier_key(prefix, "other"), None)
            .await
            .unwrap();
        assert!(result.kvs().is_empty());
        assert_eq!(store.list().await.unwrap().len(), 1);
    }
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn running_merge_preserves_new_seals_and_watch_schedules_next_pass() {
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let policy = serde_json::to_vec(&Policy::from_config(&cfg)).unwrap();
        let session = store.discovery_session(policy).await.unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let prefix = cfg.etcd.prefix();
        let mut client = store.etcd_client().clone();
        for shard in ["a", "b"] {
            client
                .put(
                    format!("{prefix}/demand-events/74/{shard}"),
                    event(shard, 7, None),
                    None,
                )
                .await
                .unwrap();
        }
        let mut cache = snapshot(&client, prefix).await.unwrap();
        assert_eq!(cache.tables["t"].pending(), 14);
        reconcile(&publisher, prefix, &mut cache, "t")
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(crate::task_store::TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        for shard in ["a", "b"] {
            client
                .put(
                    format!("{prefix}/demand-events/74/{shard}"),
                    event(shard, 14, None),
                    None,
                )
                .await
                .unwrap();
        }
        // Restart/re-list while the first pass is still running, without any
        // executor merged watermark. The persisted admission must survive.
        let mut cache = snapshot(&client, prefix).await.unwrap();
        assert_eq!(cache.tables["t"].pending(), 14);
        reconcile(&publisher, prefix, &mut cache, "t")
            .await
            .unwrap();
        assert!(cache.tables["t"].blocked());
        for _ in 0..100 {
            reconcile(&publisher, prefix, &mut cache, "t")
                .await
                .unwrap();
        }
        assert_eq!(store.list().await.unwrap().len(), 1);
        let mut stream = watch(&client, prefix, &cache).await.unwrap();
        store
            .finish(claim, Ok("first bounded pass".into()))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while cache.tables["t"].blocked() {
                let response = stream.message().await.unwrap().unwrap();
                apply_response(&mut cache, prefix, &response).unwrap();
            }
        })
        .await
        .unwrap();
        assert_eq!(cache.tables["t"].pending(), 14);
        reconcile(&publisher, prefix, &mut cache, "t")
            .await
            .unwrap();
        assert_eq!(store.list().await.unwrap().len(), 2);
        assert_eq!(cache.tables["t"].pending(), 0);
        let restarted = snapshot(&client, prefix).await.unwrap();
        assert_eq!(restarted.tables["t"].pending(), 0);
        let next = store
            .claim_next_of_kinds(crate::task_store::TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store
            .finish(next, Ok("second bounded pass".into()))
            .await
            .unwrap();
        store.release_coordination_lock(session).await.unwrap();
    }

    #[test]
    fn invalid_watermark_isolated_to_one_table_and_recovers_on_valid_update() {
        let mut cache = Cache::new();
        publish(&mut cache, "bad", "a", 10, None, 1);
        publish(&mut cache, "good", "a", 10, None, 1);
        cache
            .apply(0, "/p", b"/p/demand-events/626164/a", Some(b"broken"), 2)
            .unwrap();
        assert!(cache.tables["bad"].blocked());
        assert!(!cache.tables["good"].blocked());
        assert_eq!(cache.tables["good"].pending(), 10);
        publish(&mut cache, "bad", "a", 11, None, 3);
        assert!(!cache.tables["bad"].blocked());
        assert_eq!(cache.tables["bad"].pending(), 11);
    }
    #[test]
    fn corrupt_members_and_oversized_hints_do_not_poison_other_tables() {
        let mut cache = Cache::new();
        let cfg = MasterConfig::parse_from(["test"]);
        let member = b"/p/demand-controller/members/a";
        let good = serde_json::to_vec(&Policy::from_config(&cfg)).unwrap();
        cache.apply(2, "/p", member, Some(&good), 1).unwrap();
        assert_eq!(cache.members.len(), 1);
        cache.apply(2, "/p", member, Some(b"broken"), 2).unwrap();
        assert!(cache.members.is_empty());
        cache.apply(2, "/p", member, Some(&good), 3).unwrap();
        publish(&mut cache, "bad", "a", 10, None, 1);
        publish(&mut cache, "good", "a", 10, None, 1);
        let oversized = vec![b' '; MAX_VALUE_BYTES + 1];
        cache
            .apply(0, "/p", b"/p/demand-events/626164/a", Some(&oversized), 2)
            .unwrap();
        assert!(cache.tables["bad"].blocked());
        assert!(!cache.tables["good"].blocked());
        for key in [
            b"/p/demand-events/invalid/shard".as_slice(),
            b"/p/demand-events/\xff/shard",
            b"/p/demand-events/74",
        ] {
            cache.apply(0, "/p", key, Some(b"{}"), 3).unwrap();
        }
        cache
            .apply(3, "/p", b"/p/dedupe/merge-wal/676f6f64", Some(b"\xff"), 4)
            .unwrap();
        assert!(cache.tables["good"].blocked());
        cache
            .apply(3, "/p", b"/p/dedupe/merge-wal/676f6f64", None, 5)
            .unwrap();
        assert!(!cache.tables["good"].blocked());
        assert_eq!(cache.tables["good"].pending(), 10);
        let key = b"/p/dedupe/merge-wal/676f6f64";
        cache.apply(3, "/p", key, Some(b"old-id"), 6).unwrap();
        cache.tables.get_mut("good").unwrap().checked_dedupe = Some("old-id".into());
        assert!(cache.tables["good"].blocked());
        cache.apply(3, "/p", key, Some(b"\xff"), 7).unwrap();
        cache.apply(3, "/p", key, Some(b"old-id"), 8).unwrap();
        // Repairing the same id still needs a fresh task-state check: its
        // completion could not remove the corrupted dedupe value in between.
        assert!(!cache.tables["good"].blocked());
    }

    #[tokio::test]
    async fn fresh_writer_events_do_not_bypass_a_failed_admission_retry() {
        let mut cache = Cache::new();
        publish(&mut cache, "hot", "a", 10, None, 1);
        assert_eq!(cache.take_ready().as_deref(), Some("hot"));
        cache.defer_error("hot");
        for revision in 2..100 {
            publish(&mut cache, "hot", "a", revision, None, revision as i64);
            assert!(cache.take_ready().is_none());
        }
        publish(&mut cache, "healthy", "a", 10, None, 1);
        assert_eq!(cache.take_ready().as_deref(), Some("healthy"));
        tokio::time::sleep_until(cache.next_due() + Duration::from_millis(1)).await;
        assert_eq!(cache.take_ready().as_deref(), Some("hot"));
        assert!(cache.take_ready().is_none());
        assert_eq!(bootstrap_delay(CAPACITY_ERROR, 1), Duration::from_secs(300));
        assert_eq!(bootstrap_delay("transport", 1), Duration::from_secs(5));
        assert_eq!(bootstrap_delay("transport", 100), Duration::from_secs(300));
    }

    #[tokio::test]
    #[ignore = "requires exclusive loopback ETCD_TEST_ENDPOINTS"]
    async fn active_dedupe_and_unchanged_frontier_issue_no_repeated_rpc() {
        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").as_deref(), Ok("1"));
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        assert!(endpoint.starts_with("http://127.0.0.1:") && !endpoint.contains(','));
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let session = store
            .discovery_session(serde_json::to_vec(&Policy::from_config(&cfg)).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let mut cache = Cache::new();
        cache
            .members
            .insert(b"a".to_vec(), Policy::from_config(&cfg));
        publish(&mut cache, "t", "a", 14, None, 1);
        reconcile(&publisher, cfg.etcd.prefix(), &mut cache, "t")
            .await
            .unwrap();
        let before = super::super::tests::rpc_count(&endpoint, &["Range", "Txn", "Put"]).await;
        for revision in 2..102 {
            publish(&mut cache, "t", "a", 14, None, revision);
            reconcile(&publisher, cfg.etcd.prefix(), &mut cache, "t")
                .await
                .unwrap();
        }
        assert_eq!(
            super::super::tests::rpc_count(&endpoint, &["Range", "Txn", "Put"]).await,
            before
        );
        publish(&mut cache, "t", "a", 28, None, 103);
        reconcile(&publisher, cfg.etcd.prefix(), &mut cache, "t")
            .await
            .unwrap();
        assert!(cache.tables["t"].blocked());
        let before = super::super::tests::rpc_count(&endpoint, &["Range", "Txn", "Put"]).await;
        for revision in 104..204 {
            publish(&mut cache, "t", "a", revision, None, revision as i64);
            reconcile(&publisher, cfg.etcd.prefix(), &mut cache, "t")
                .await
                .unwrap();
        }
        assert_eq!(
            super::super::tests::rpc_count(&endpoint, &["Range", "Txn", "Put"]).await,
            before
        );
        assert_eq!(store.list().await.unwrap().len(), 1);
        store.release_coordination_lock(session).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn paginated_snapshot_stays_at_one_revision_during_writes() {
        use etcd_client::{Txn, TxnOp};
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let prefix = cfg.etcd.prefix();
        let mut client = store.etcd_client().clone();
        for page in 0..6 {
            let ops: Vec<_> = (page * 50..(page + 1) * 50)
                .map(|n| {
                    let shard = format!("s{n:03}");
                    TxnOp::put(
                        format!("{prefix}/demand-events/74/{shard}"),
                        event(&shard, 7, None),
                        None,
                    )
                })
                .collect();
            client.txn(Txn::new().and_then(ops)).await.unwrap();
        }
        let mut writer = client.clone();
        let key = format!("{prefix}/demand-events/74/s299");
        let running = tokio::spawn(async move {
            for generation in 8..50 {
                writer
                    .put(key.clone(), event("s299", generation, None), None)
                    .await
                    .unwrap();
                tokio::task::yield_now().await;
            }
        });
        let cache = snapshot(&client, prefix).await.unwrap();
        running.await.unwrap();
        assert_eq!(cache.shards, 300);
        assert!(cache.revisions.iter().all(|r| *r == cache.revisions[0]));
        let actual = client
            .get(
                format!("{prefix}/demand-events/"),
                Some(
                    GetOptions::new()
                        .with_prefix()
                        .with_revision(cache.revisions[0]),
                ),
            )
            .await
            .unwrap();
        assert_eq!(actual.kvs().len(), 300);
        for kv in actual.kvs() {
            let event: Event = serde_json::from_slice(kv.value()).unwrap();
            assert_eq!(
                cache.tables["t"].shards[&event.shard].sealed,
                event.sealed_through
            );
        }
    }

    #[tokio::test]
    #[ignore = "requires isolated etcd and ETCD_TEST_ALLOW_COMPACTION=1"]
    async fn compacted_watch_requires_new_snapshot_then_replays_new_seals() {
        assert_eq!(
            std::env::var("ETCD_TEST_ALLOW_COMPACTION").as_deref(),
            Ok("1")
        );
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let prefix = cfg.etcd.prefix();
        let mut client = store.etcd_client().clone();
        let key = format!("{prefix}/demand-events/74/a");
        client
            .put(key.clone(), event("a", 7, None), None)
            .await
            .unwrap();
        let mut old = snapshot(&client, prefix).await.unwrap();
        client
            .put(key.clone(), event("a", 14, None), None)
            .await
            .unwrap();
        let newer = client
            .put(key.clone(), event("a", 21, None), None)
            .await
            .unwrap();
        client
            .compact(newer.header().unwrap().revision(), None)
            .await
            .unwrap();
        let mut stream = watch(&client, prefix, &old).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let response = stream.message().await.unwrap().unwrap();
                if apply_response(&mut old, prefix, &response).is_err_and(|e| e == "compacted") {
                    break;
                }
            }
        })
        .await
        .unwrap();
        drop(stream);
        let mut cache = snapshot(&client, prefix).await.unwrap();
        assert_eq!(cache.tables["t"].pending(), 21);
        let mut stream = watch(&client, prefix, &cache).await.unwrap();
        client.put(key, event("a", 28, None), None).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while cache.tables["t"].pending() != 28 {
                let response = stream.message().await.unwrap().unwrap();
                apply_response(&mut cache, prefix, &response).unwrap();
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn restart_after_unacknowledged_publication_does_not_duplicate_task() {
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let policy = serde_json::to_vec(&Policy::from_config(&cfg)).unwrap();
        let session = store.discovery_session(policy.clone()).await.unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let prefix = cfg.etcd.prefix();
        let mut client = store.etcd_client().clone();
        for shard in ["a", "b"] {
            client
                .put(
                    format!("{prefix}/demand-events/74/{shard}"),
                    event(shard, 7, None),
                    None,
                )
                .await
                .unwrap();
        }
        let before = snapshot(&client, prefix).await.unwrap();
        let value = serde_json::to_vec(&before.tables["t"].frontier()).unwrap();
        // Deliberately discard the transaction response and the old cache,
        // as when the controller dies after commit but before acknowledging it.
        let _ = publisher
            .enqueue_aggregate_wal("t", 0, &value)
            .await
            .unwrap();
        store.release_coordination_lock(session).await.unwrap();
        let next = store.discovery_session(policy).await.unwrap();
        assert!(store.elect_discovery_leader(&next).await.unwrap());
        let next_publisher = store.discovery_publisher(&next);
        let mut rebuilt = snapshot(&client, prefix).await.unwrap();
        assert_eq!(rebuilt.tables["t"].pending(), 0);
        reconcile(&next_publisher, prefix, &mut rebuilt, "t")
            .await
            .unwrap();
        assert_eq!(store.list().await.unwrap().len(), 1);
        assert!(matches!(
            publisher
                .enqueue_aggregate_wal("other", 0, &value)
                .await
                .unwrap(),
            Admission::Changed
        ));
        store.release_coordination_lock(next).await.unwrap();
    }
    #[test]
    fn admission_capacity_and_credit_survive_watch_order_and_shrink() {
        let cfg = MasterConfig::parse_from(["test"]);
        let mut policy = Policy::from_config(&cfg);
        policy.aggregate_merge_slots = 2;
        let mut old = serde_json::to_value(&policy).unwrap();
        old.as_object_mut().unwrap().remove("aggregate_merge_slots");
        let old: Policy = serde_json::from_value(old).unwrap();
        assert_eq!(old.aggregate_merge_slots, 0);
        for dedupe_first in [false, true] {
            let mut cache = Cache::new();
            cache.members.insert(b"m".to_vec(), policy.clone());
            assert_eq!(cache.capacity(), 4);
            publish(&mut cache, "a", "s", 14, None, 1);
            let mut frontier = cache.tables["a"].frontier();
            frontier.task = Some("task-a".into());
            let bytes = serde_json::to_vec(&frontier).unwrap();
            for range in if dedupe_first { [3, 1] } else { [1, 3] } {
                let key = if range == 1 {
                    frontier_key("/p", "a")
                } else {
                    dedupe_key("/p", "a")
                };
                let value = if range == 1 {
                    bytes.as_slice()
                } else {
                    b"task-a"
                };
                cache
                    .apply(range, "/p", key.as_bytes(), Some(value), 2)
                    .unwrap();
            }
            assert_eq!(cache.outstanding.len(), 1);
            cache.members.clear();
            assert_eq!(cache.capacity(), 0);
            assert!(cache.outstanding.contains("a"));
            // Resolution may rewrite a frontier without changing its task.
            cache
                .apply(1, "/p", frontier_key("/p", "a").as_bytes(), Some(&bytes), 3)
                .unwrap();
            assert!(cache.outstanding.contains("a"));
            cache
                .apply(3, "/p", dedupe_key("/p", "a").as_bytes(), None, 4)
                .unwrap();
            assert!(cache.outstanding.is_empty());
        }
    }

    #[tokio::test]
    #[ignore = "requires exclusive loopback ETCD_TEST_ENDPOINTS"]
    async fn bootstrap_budget_retains_hints_and_watch_releases_credit_to_fresh_demand() {
        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").as_deref(), Ok("1"));
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        assert!(endpoint.starts_with("http://127.0.0.1:") && !endpoint.contains(','));
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let mut policy = Policy::from_config(&cfg);
        policy.aggregate_merge_slots = 2;
        let session = store
            .discovery_session(serde_json::to_vec(&policy).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let prefix = cfg.etcd.prefix();
        let mut client = store.etcd_client().clone();
        for n in 0..100 {
            let name = format!("table-{n:03}");
            let hex: String = name.bytes().map(|b| format!("{b:02x}")).collect();
            client
                .put(
                    format!("{prefix}/demand-events/{hex}/a"),
                    event("a", 14, None),
                    None,
                )
                .await
                .unwrap();
        }
        let mut cache = snapshot(&client, prefix).await.unwrap();
        while let Some(target) = cache.take_ready() {
            reconcile(&publisher, prefix, &mut cache, &target)
                .await
                .unwrap();
        }
        assert_eq!(store.list().await.unwrap().len(), 4);
        assert_eq!(cache.outstanding.len(), 4);
        assert_eq!(cache.waiting.len(), 96);
        assert_eq!(
            cache.tables.values().map(Table::pending).sum::<u64>(),
            96 * 14
        );
        let before = super::super::tests::rpc_count(&endpoint, &["Range", "Txn", "Put"]).await;
        for _ in 0..100 {
            assert!(cache.take_ready().is_none());
        }
        assert_eq!(
            super::super::tests::rpc_count(&endpoint, &["Range", "Txn", "Put"]).await,
            before
        );
        // Rebuild consumes the persisted outstanding credits, even with no
        // executor heartbeat/merged watermark and no subsequent writer event.
        let mut cache = snapshot(&client, prefix).await.unwrap();
        while let Some(target) = cache.take_ready() {
            reconcile(&publisher, prefix, &mut cache, &target)
                .await
                .unwrap();
        }
        assert_eq!(cache.outstanding.len(), 4);
        assert_eq!(store.list().await.unwrap().len(), 4);
        let mut stream = watch(&client, prefix, &cache).await.unwrap();
        // New demand is below each writer's own threshold, above table total.
        for shard in ["a", "b"] {
            client
                .put(
                    format!("{prefix}/demand-events/686f74/{shard}"),
                    event(shard, 7, None),
                    None,
                )
                .await
                .unwrap();
        }
        tokio::time::timeout(Duration::from_secs(5), async {
            while cache.tables.get("hot").is_none_or(|t| t.pending() != 14) {
                apply_response(
                    &mut cache,
                    prefix,
                    &stream.message().await.unwrap().unwrap(),
                )
                .unwrap();
            }
        })
        .await
        .unwrap();
        while let Some(target) = cache.take_ready() {
            reconcile(&publisher, prefix, &mut cache, &target)
                .await
                .unwrap();
        }
        assert!(cache.waiting.contains_key("hot"));
        let claim = store
            .claim_next_of_kinds(crate::task_store::TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store.finish(claim, Ok("done".into())).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while cache.outstanding.len() == 4 {
                apply_response(
                    &mut cache,
                    prefix,
                    &stream.message().await.unwrap().unwrap(),
                )
                .unwrap();
            }
        })
        .await
        .unwrap();
        while let Some(target) = cache.take_ready() {
            reconcile(&publisher, prefix, &mut cache, &target)
                .await
                .unwrap();
        }
        let tasks = store.list().await.unwrap();
        assert_eq!(tasks.len(), 5);
        assert!(tasks
            .iter()
            .any(|t| t.target == "hot" && t.state == TaskState::Queued));
        assert_eq!(cache.outstanding.len(), 4);
        assert_eq!(cache.waiting.len(), 96);
        store.release_coordination_lock(session).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn ambiguous_publication_is_fenced_before_releasing_admission_credit() {
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let session = store
            .discovery_session(serde_json::to_vec(&Policy::from_config(&cfg)).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let prefix = cfg.etcd.prefix();
        for committed in [false, true] {
            let target = if committed { "committed" } else { "delayed" };
            let mut cache = Cache::new();
            publish(&mut cache, target, "a", 14, None, 1);
            let value = serde_json::to_vec(&cache.tables[target].frontier()).unwrap();
            let previous = serde_json::to_vec(&cache.tables[target].previous_frontier()).unwrap();
            cache.pending = Some(Attempt {
                target: target.into(),
                revision: 0,
                previous,
                retry_at: Instant::now(),
            });
            cache.refresh_outstanding(target);
            assert_eq!(cache.outstanding.len(), 1);
            if committed {
                let _lost = publisher
                    .enqueue_aggregate_wal(target, 0, &value)
                    .await
                    .unwrap();
            }
            resolve_pending(&publisher, prefix, &mut cache)
                .await
                .unwrap();
            assert!(cache.pending.is_none());
            assert_eq!(cache.outstanding.len(), usize::from(committed));
            assert_eq!(
                cache.tables[target].pending(),
                if committed { 0 } else { 14 }
            );
            assert!(cache.tables[target].frontier_revision > 0);
            // Even if the originally timed-out RPC arrives now, its previous
            // frontier CAS cannot publish a task against the released credit.
            assert!(matches!(
                publisher
                    .enqueue_aggregate_wal(target, 0, &value)
                    .await
                    .unwrap(),
                Admission::Changed
            ));
        }
        assert_eq!(store.list().await.unwrap().len(), 1);
        store.release_coordination_lock(session).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn unused_waiter_turn_does_not_strand_remaining_demand() {
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let mut cache = Cache::new();
        cache
            .members
            .insert(b"m".to_vec(), Policy::from_config(&cfg));
        for target in ["a", "b"] {
            publish(&mut cache, target, "s", 1, None, 1);
            cache.wait_capacity(target);
        }
        cache.wake_capacity();
        let target = cache.take_ready().unwrap();
        // Below threshold requires no publisher RPC or elected session.
        reconcile(&store, cfg.etcd.prefix(), &mut cache, &target)
            .await
            .unwrap();
        assert!(cache.take_ready().is_some());
    }
    #[test]
    fn advertised_budget_matches_wal_claiming_pool() {
        let mut cfg = MasterConfig::parse_from(["test"]);
        cfg.merge_wal_concurrency = 0;
        cfg.task_concurrency = 0;
        assert_eq!(Policy::from_config(&cfg).aggregate_merge_slots, 1);
        cfg.task_concurrency = 9;
        assert_eq!(Policy::from_config(&cfg).aggregate_merge_slots, 9);
        cfg.merge_wal_concurrency = 3;
        assert_eq!(Policy::from_config(&cfg).aggregate_merge_slots, 3);
        cfg.maintenance.claim_wal_merges = false;
        assert_eq!(Policy::from_config(&cfg).aggregate_merge_slots, 0);
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn member_capacity_changes_wake_waiters_without_cancelling_active_tasks() {
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let mut policy = Policy::from_config(&cfg);
        policy.aggregate_merge_slots = 1;
        let session = store
            .discovery_session(serde_json::to_vec(&policy).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let prefix = cfg.etcd.prefix();
        let mut client = store.etcd_client().clone();
        for target in ["a", "b", "c", "d", "e"] {
            let hex: String = target.bytes().map(|b| format!("{b:02x}")).collect();
            client
                .put(
                    format!("{prefix}/demand-events/{hex}/s"),
                    event("s", 14, None),
                    None,
                )
                .await
                .unwrap();
        }
        let mut cache = snapshot(&client, prefix).await.unwrap();
        while let Some(target) = cache.take_ready() {
            reconcile(&publisher, prefix, &mut cache, &target)
                .await
                .unwrap();
        }
        assert_eq!(cache.outstanding.len(), 2);
        assert_eq!(cache.waiting.len(), 3);
        let mut stream = watch(&client, prefix, &cache).await.unwrap();
        let second = store
            .discovery_session(serde_json::to_vec(&policy).unwrap())
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while cache.capacity() < 4 {
                apply_response(
                    &mut cache,
                    prefix,
                    &stream.message().await.unwrap().unwrap(),
                )
                .unwrap();
            }
        })
        .await
        .unwrap();
        while let Some(target) = cache.take_ready() {
            reconcile(&publisher, prefix, &mut cache, &target)
                .await
                .unwrap();
        }
        assert_eq!(cache.outstanding.len(), 4);
        assert_eq!(cache.waiting.len(), 1);
        store.release_coordination_lock(second).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while cache.capacity() != 2 {
                apply_response(
                    &mut cache,
                    prefix,
                    &stream.message().await.unwrap().unwrap(),
                )
                .unwrap();
            }
        })
        .await
        .unwrap();
        while let Some(target) = cache.take_ready() {
            reconcile(&publisher, prefix, &mut cache, &target)
                .await
                .unwrap();
        }
        assert_eq!(cache.outstanding.len(), 4);
        let tasks = store.list().await.unwrap();
        assert_eq!(tasks.len(), 4);
        assert!(tasks.iter().all(|t| t.state == TaskState::Queued));
        assert_eq!(cache.waiting.len(), 1);
        store.release_coordination_lock(session).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated etcd and ETCD_TEST_ALLOW_COMPACTION=1"]
    async fn uncertain_publication_after_completion_or_compaction_keeps_credit_and_seals_correct() {
        assert_eq!(
            std::env::var("ETCD_TEST_ALLOW_COMPACTION").as_deref(),
            Ok("1")
        );
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let policy = Policy::from_config(&cfg);
        let session = store
            .discovery_session(serde_json::to_vec(&policy).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&session).await.unwrap());
        let publisher = store.discovery_publisher(&session);
        let prefix = cfg.etcd.prefix();
        let mut client = store.etcd_client().clone();
        let target = "t";
        let key = format!("{prefix}/demand-events/74/a");
        client
            .put(key.clone(), event("a", 14, None), None)
            .await
            .unwrap();
        let mut cache = snapshot(&client, prefix).await.unwrap();
        let value = serde_json::to_vec(&cache.tables[target].frontier()).unwrap();
        cache.pending = Some(Attempt {
            target: target.into(),
            revision: 0,
            previous: serde_json::to_vec(&cache.tables[target].previous_frontier()).unwrap(),
            retry_at: Instant::now(),
        });
        cache.refresh_outstanding(target);
        let _lost = publisher
            .enqueue_aggregate_wal(target, 0, &value)
            .await
            .unwrap();
        let claim = store
            .claim_next_of_kinds(crate::task_store::TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        store
            .finish(claim, Ok("completed before reply".into()))
            .await
            .unwrap();
        // Compaction prevents restoring the old watch. Same-owner resolution
        // must reconstruct completed admission, not leak the reserved credit.
        let response = client
            .put(key.clone(), event("a", 28, None), None)
            .await
            .unwrap();
        client
            .compact(response.header().unwrap().revision(), None)
            .await
            .unwrap();
        let mut stream = watch(&client, prefix, &cache).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if apply_response(
                    &mut cache,
                    prefix,
                    &stream.message().await.unwrap().unwrap(),
                )
                .is_err_and(|e| e == "compacted")
                {
                    break;
                }
            }
        })
        .await
        .unwrap();
        resolve_pending(&publisher, prefix, &mut cache)
            .await
            .unwrap();
        assert!(cache.pending.is_none());
        assert!(cache.outstanding.is_empty());
        let mut cache = snapshot(&client, prefix).await.unwrap();
        assert_eq!(cache.tables[target].pending(), 14);
        assert!(cache.outstanding.is_empty());
        reconcile(&publisher, prefix, &mut cache, target)
            .await
            .unwrap();
        assert_eq!(store.list().await.unwrap().len(), 2);
        assert_eq!(cache.outstanding.len(), 1);
        // A lost request which has NOT committed is invalidated before a
        // same-leader snapshot too, including a delayed arrival afterwards.
        let other = "u";
        client
            .put(
                format!("{prefix}/demand-events/75/a"),
                event("a", 14, None),
                None,
            )
            .await
            .unwrap();
        let mut cache = snapshot(&client, prefix).await.unwrap();
        let value = serde_json::to_vec(&cache.tables[other].frontier()).unwrap();
        cache.pending = Some(Attempt {
            target: other.into(),
            revision: 0,
            previous: serde_json::to_vec(&cache.tables[other].previous_frontier()).unwrap(),
            retry_at: Instant::now(),
        });
        cache.refresh_outstanding(other);
        resolve_pending(&publisher, prefix, &mut cache)
            .await
            .unwrap();
        let mut rebuilt = snapshot(&client, prefix).await.unwrap();
        assert_eq!(rebuilt.tables[other].pending(), 14);
        assert!(!rebuilt.outstanding.contains(other));
        assert!(matches!(
            publisher
                .enqueue_aggregate_wal(other, 0, &value)
                .await
                .unwrap(),
            Admission::Changed
        ));
        reconcile(&publisher, prefix, &mut rebuilt, other)
            .await
            .unwrap();
        assert_eq!(store.list().await.unwrap().len(), 3);
        store.release_coordination_lock(session).await.unwrap();
    }
}
