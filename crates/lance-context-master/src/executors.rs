//! Executor heartbeats and durable assignment reservations (P1.6).
//!
//! Spec: `docs/design/scheduler-p1-data-model.md` §2. Every master publishes
//! a leased `P/executors/<id>` record describing what it can run and how
//! much. The planner (leader) computes headroom per executor from the
//! `P/assignments/<hex>/<unit-id>` records it has written, **never** from
//! the heartbeat's sampled `bytes_free`: assignments are durable, survive a
//! leader change, and count from the moment they are written, not from
//! when they bind.
//!
//! P1 scope: heartbeats are real; assignments exist only as `shadow`
//! records that no executor reads. The headroom arithmetic and failover
//! rebuild are what this module tests.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use etcd_client::PutOptions;
use lance_context_api::TaskKind;
use serde::{Deserialize, Serialize};

use crate::state::MasterState;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorKind {
    ResidentMaster,
    Worker,
    CatchupJob,
    LegacyRpc,
}

/// Leased heartbeat under `P/executors/<id>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Heartbeat {
    pub v: u32,
    pub kind: ExecutorKind,
    pub instance: String,
    pub version: String,
    pub kinds: Vec<TaskKind>,
    /// Slots per task kind this executor will run concurrently.
    pub slots_total: BTreeMap<TaskKind, u32>,
    /// Memory budget for staging/merge buffers. 0 = unknown/unbounded.
    pub bytes_total: u64,
    /// Diagnostic sample only; never used for placement.
    pub bytes_free_sample: u64,
    pub draining: bool,
    pub heartbeat_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentState {
    /// Written by the planner in shadow mode; no executor reads it.
    Shadow,
    Assigned,
    Bound,
    Released,
}

/// Durable reservation under `P/assignments/<hex>/<unit-id>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assignment {
    pub v: u32,
    pub unit_id: String,
    pub kind: TaskKind,
    pub target: String,
    pub needs_write_turn: bool,
    pub executor: String,
    pub planner_token: String,
    pub reserved_slots: BTreeMap<TaskKind, u32>,
    pub reserved_bytes: u64,
    pub state: AssignmentState,
    pub created_ms: i64,
    pub bind_deadline_ms: i64,
}

impl Assignment {
    /// Assignments that hold capacity. `Shadow` is included so the shadow
    /// planner's own arithmetic is exercised; `Released` is not.
    pub fn holds_capacity(&self) -> bool {
        !matches!(self.state, AssignmentState::Released)
    }
}

/// What the planner knows about one executor after folding heartbeats and
/// assignments. Pure data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Headroom {
    pub executor: String,
    pub kind: ExecutorKind,
    pub draining: bool,
    pub slots_total: BTreeMap<TaskKind, u32>,
    pub slots_reserved: BTreeMap<TaskKind, u32>,
    pub bytes_total: u64,
    pub bytes_reserved: u64,
    /// Live assignments counted into the reservation.
    pub assignments: usize,
}

impl Headroom {
    pub fn slots_free(&self, kind: TaskKind) -> u32 {
        self.slots_total
            .get(&kind)
            .copied()
            .unwrap_or(0)
            .saturating_sub(self.slots_reserved.get(&kind).copied().unwrap_or(0))
    }
    pub fn bytes_free(&self) -> u64 {
        self.bytes_total.saturating_sub(self.bytes_reserved)
    }
    /// Can this executor take a unit of `kind` needing `bytes`? Draining
    /// executors take nothing. An executor with `bytes_total == 0` has no
    /// byte budget to check.
    pub fn fits(&self, kind: TaskKind, bytes: u64) -> bool {
        !self.draining
            && self.slots_free(kind) > 0
            && (self.bytes_total == 0 || self.bytes_free() >= bytes)
    }
}

/// Fold heartbeats and assignments into per-executor headroom. Pure: this
/// is what a new leader does from etcd before its first cycle. Assignments
/// pointing at an executor with no heartbeat still count (its reservation
/// must not evaporate because the heartbeat lapsed first); they appear with
/// zero totals and are reported so the planner can release them.
pub fn headroom<'a>(
    heartbeats: impl IntoIterator<Item = (&'a str, &'a Heartbeat)>,
    assignments: impl IntoIterator<Item = &'a Assignment>,
) -> BTreeMap<String, Headroom> {
    let mut out: BTreeMap<String, Headroom> = BTreeMap::new();
    for (id, hb) in heartbeats {
        out.insert(
            id.to_string(),
            Headroom {
                executor: id.to_string(),
                kind: hb.kind,
                draining: hb.draining,
                slots_total: hb.slots_total.clone(),
                slots_reserved: BTreeMap::new(),
                bytes_total: hb.bytes_total,
                bytes_reserved: 0,
                assignments: 0,
            },
        );
    }
    for a in assignments {
        if !a.holds_capacity() {
            continue;
        }
        let entry = out.entry(a.executor.clone()).or_insert_with(|| Headroom {
            executor: a.executor.clone(),
            kind: ExecutorKind::ResidentMaster,
            draining: true,
            slots_total: BTreeMap::new(),
            slots_reserved: BTreeMap::new(),
            bytes_total: 0,
            bytes_reserved: 0,
            assignments: 0,
        });
        for (kind, n) in &a.reserved_slots {
            *entry.slots_reserved.entry(*kind).or_insert(0) += n;
        }
        entry.bytes_reserved = entry.bytes_reserved.saturating_add(a.reserved_bytes);
        entry.assignments += 1;
    }
    out
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
    pub(crate) fn executors_prefix(&self) -> String {
        format!("{}/executors/", self.prefix)
    }
    pub(crate) fn executor(&self, id: &str) -> String {
        format!("{}/executors/{id}", self.prefix)
    }
    pub(crate) fn assignments_prefix(&self) -> String {
        format!("{}/assignments/", self.prefix)
    }
    /// Used by the shadow placer (next PR) and by tests now.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn assignment(&self, target: &str, unit_id: &str) -> String {
        let encoded: String = target.bytes().map(|b| format!("{b:02x}")).collect();
        format!("{}/assignments/{encoded}/{unit_id}", self.prefix)
    }
}

/// What this master advertises. Slot counts mirror the dispatch pools so
/// the heartbeat describes real capacity, not a wish.
pub(crate) fn local_heartbeat(state: &MasterState) -> Heartbeat {
    let cfg = &state.config;
    let mut slots_total = BTreeMap::new();
    let mut kinds = vec![TaskKind::IndexId, TaskKind::Repair];
    slots_total.insert(TaskKind::IndexId, cfg.task_concurrency.max(1) as u32);
    slots_total.insert(TaskKind::Repair, cfg.task_concurrency.max(1) as u32);
    if cfg.compaction_concurrency > 0 {
        kinds.push(TaskKind::Compact);
        slots_total.insert(TaskKind::Compact, cfg.compaction_concurrency as u32);
    }
    let merge_slots = if cfg.merge_wal_concurrency > 0 {
        cfg.merge_wal_concurrency
    } else {
        cfg.task_concurrency.max(1)
    } + if crate::local_merge::enabled(cfg) {
        cfg.append.rollout_append_local_task_concurrency
    } else {
        0
    };
    if merge_slots > 0 {
        kinds.push(TaskKind::MergeWal);
        slots_total.insert(TaskKind::MergeWal, merge_slots as u32);
    }
    let bytes_total = if crate::local_merge::enabled(cfg) {
        cfg.append.rollout_append_local_memory_bytes as u64
    } else {
        0
    };
    let status = state.admission.status();
    Heartbeat {
        v: SCHEMA_VERSION,
        kind: ExecutorKind::ResidentMaster,
        instance: status.executor_id.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        kinds,
        slots_total,
        bytes_total,
        bytes_free_sample: state.local_append.as_ref().map_or(bytes_total, |l| {
            (l.budget.limit() as u64).saturating_sub(l.budget.reserved() as u64)
        }),
        draining: !status.accepting,
        heartbeat_ms: chrono::Utc::now().timestamp_millis(),
    }
}

/// Publish this master's heartbeat under a lease, renewing every `TTL/3`.
/// Independent of leadership: every master is an executor.
pub(crate) fn spawn(state: &Arc<MasterState>) {
    if !state.config.planner.planner_enabled {
        return;
    }
    let weak = Arc::downgrade(state);
    tokio::spawn(async move {
        loop {
            let Some(state) = weak.upgrade() else { return };
            if let Err(error) = heartbeat_loop(&state).await {
                tracing::warn!(%error, "executor heartbeat loop ended");
                metrics::counter!("executor_heartbeat_errors_total").increment(1);
            }
            drop(state);
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

#[cfg(test)]
pub(crate) async fn heartbeat_loop_for_test(state: &Arc<MasterState>) -> Result<(), String> {
    heartbeat_loop(state).await
}

async fn heartbeat_loop(state: &Arc<MasterState>) -> Result<(), String> {
    let keys = Keys::new(&state.config.etcd.etcd_prefix);
    let mut client = state.task_store.etcd_client().clone();
    let ttl = state.config.etcd_lease_ttl_secs.max(5);
    let lease = client
        .lease_grant(ttl, None)
        .await
        .map_err(|e| e.to_string())?
        .id();
    let (mut keeper, mut stream) = client
        .lease_keep_alive(lease)
        .await
        .map_err(|e| e.to_string())?;
    let id = state.admission.status().executor_id;
    let key = keys.executor(&id);
    let mut tick = tokio::time::interval(Duration::from_secs((ttl as u64 / 3).max(1)));
    let result = loop {
        tick.tick().await;
        let hb = local_heartbeat(state);
        if let Err(e) = client
            .put(
                key.as_str(),
                serde_json::to_vec(&hb).unwrap(),
                Some(PutOptions::new().with_lease(lease)),
            )
            .await
        {
            break Err(e.to_string());
        }
        if keeper.keep_alive().await.is_err() {
            break Err("lease keepalive failed".into());
        }
        match tokio::time::timeout(Duration::from_secs(5), stream.message()).await {
            Ok(Ok(Some(r))) if r.ttl() > 0 => {}
            _ => break Err("lease expired or keepalive stream closed".into()),
        }
        metrics::gauge!("executor_heartbeat_bytes_total").set(hb.bytes_total as f64);
        if hb.draining {
            // Stop advertising; let the lease expire so the planner sees us go.
            break Ok(());
        }
    };
    drop(keeper);
    let _ = client.lease_revoke(lease).await;
    result
}

/// Read every heartbeat and live assignment and fold them. What a leader
/// does before its first cycle and on every reconcile.
pub(crate) async fn load_headroom(
    client: &etcd_client::Client,
    keys: &Keys,
) -> Result<BTreeMap<String, Headroom>, String> {
    let (hbs, _) = crate::planner::read_prefix_paged(client, &keys.executors_prefix()).await?;
    let prefix = keys.executors_prefix();
    let heartbeats: Vec<(String, Heartbeat)> = hbs
        .iter()
        .filter_map(|kv| {
            let id = std::str::from_utf8(&kv.key)
                .ok()?
                .strip_prefix(prefix.as_str())?
                .to_string();
            let hb: Heartbeat = serde_json::from_slice(&kv.value).ok()?;
            (hb.v == SCHEMA_VERSION).then_some((id, hb))
        })
        .collect();
    let (asg, _) = crate::planner::read_prefix_paged(client, &keys.assignments_prefix()).await?;
    let assignments: Vec<Assignment> = asg
        .iter()
        .filter_map(|kv| serde_json::from_slice::<Assignment>(&kv.value).ok())
        .filter(|a| a.v == SCHEMA_VERSION)
        .collect();
    Ok(headroom(
        heartbeats.iter().map(|(id, hb)| (id.as_str(), hb)),
        assignments.iter(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hb(kind: ExecutorKind, merge: u32, bytes: u64, draining: bool) -> Heartbeat {
        let mut slots_total = BTreeMap::new();
        slots_total.insert(TaskKind::MergeWal, merge);
        slots_total.insert(TaskKind::Compact, 1);
        Heartbeat {
            v: SCHEMA_VERSION,
            kind,
            instance: "i".into(),
            version: "t".into(),
            kinds: vec![TaskKind::MergeWal, TaskKind::Compact],
            slots_total,
            bytes_total: bytes,
            bytes_free_sample: bytes, // deliberately lies: says everything is free
            draining,
            heartbeat_ms: 0,
        }
    }

    fn asg(executor: &str, kind: TaskKind, bytes: u64, state: AssignmentState) -> Assignment {
        let mut reserved_slots = BTreeMap::new();
        reserved_slots.insert(kind, 1);
        Assignment {
            v: SCHEMA_VERSION,
            unit_id: lance_context_core::generate_id(),
            kind,
            target: "t".into(),
            needs_write_turn: true,
            executor: executor.into(),
            planner_token: "p".into(),
            reserved_slots,
            reserved_bytes: bytes,
            state,
            created_ms: 0,
            bind_deadline_ms: 0,
        }
    }

    /// Spec test 3: headroom comes from assignments, not samples. An executor
    /// whose sample says 2 GiB free but with 2 GiB reserved gets no placement.
    #[test]
    fn headroom_ignores_the_sample_and_counts_unbound_assignments() {
        let gib = 1u64 << 30;
        let heartbeats = [(
            "e1".to_string(),
            hb(ExecutorKind::ResidentMaster, 2, 2 * gib, false),
        )];
        let assignments = [
            asg("e1", TaskKind::MergeWal, gib, AssignmentState::Assigned), // not yet bound
            asg("e1", TaskKind::MergeWal, gib, AssignmentState::Shadow),
            asg("e1", TaskKind::Compact, 0, AssignmentState::Released), // does not count
        ];
        let h = headroom(
            heartbeats.iter().map(|(id, hb)| (id.as_str(), hb)),
            assignments.iter(),
        );
        let e1 = &h["e1"];
        assert_eq!(e1.bytes_reserved, 2 * gib);
        assert_eq!(e1.bytes_free(), 0);
        assert_eq!(e1.slots_free(TaskKind::MergeWal), 0);
        assert_eq!(
            e1.slots_free(TaskKind::Compact),
            1,
            "released does not count"
        );
        assert_eq!(e1.assignments, 2);
        assert!(
            !e1.fits(TaskKind::MergeWal, 1),
            "sample said free; reservation says no"
        );
        assert!(e1.fits(TaskKind::Compact, 0));
        assert!(
            !e1.fits(TaskKind::Compact, 1),
            "no bytes left even for a slot that is free"
        );
    }

    #[test]
    fn draining_and_unknown_budget_rules() {
        let heartbeats = [
            ("d".to_string(), hb(ExecutorKind::Worker, 4, 0, true)),
            ("u".to_string(), hb(ExecutorKind::Worker, 4, 0, false)),
        ];
        let h = headroom(
            heartbeats.iter().map(|(id, hb)| (id.as_str(), hb)),
            [].iter(),
        );
        assert!(
            !h["d"].fits(TaskKind::MergeWal, 0),
            "draining takes nothing"
        );
        assert!(
            h["u"].fits(TaskKind::MergeWal, u64::MAX),
            "no byte budget: slots decide"
        );
    }

    /// Spec test 4: a new leader rebuilds reservations from assignments even
    /// when the executor's heartbeat is gone, so capacity cannot be double
    /// booked across a failover; such orphans are visible for release.
    #[test]
    fn assignments_survive_a_missing_heartbeat() {
        let assignments = [asg("gone", TaskKind::MergeWal, 7, AssignmentState::Bound)];
        let h = headroom(std::iter::empty(), assignments.iter());
        let gone = &h["gone"];
        assert_eq!(gone.bytes_reserved, 7);
        assert_eq!(gone.bytes_total, 0);
        assert!(
            gone.draining,
            "an executor with no heartbeat is treated as draining"
        );
        assert!(!gone.fits(TaskKind::MergeWal, 0));
        assert_eq!(gone.assignments, 1);
    }

    #[test]
    fn headroom_is_a_pure_fold_of_its_inputs() {
        let heartbeats = [
            (
                "a".to_string(),
                hb(ExecutorKind::ResidentMaster, 2, 100, false),
            ),
            ("b".to_string(), hb(ExecutorKind::CatchupJob, 1, 50, false)),
        ];
        let assignments = [
            asg("a", TaskKind::MergeWal, 30, AssignmentState::Bound),
            asg("b", TaskKind::MergeWal, 50, AssignmentState::Assigned),
            asg("a", TaskKind::Compact, 10, AssignmentState::Shadow),
        ];
        let once = headroom(
            heartbeats.iter().map(|(id, hb)| (id.as_str(), hb)),
            assignments.iter(),
        );
        let mut reversed = assignments.to_vec();
        reversed.reverse();
        let again = headroom(
            heartbeats.iter().map(|(id, hb)| (id.as_str(), hb)),
            reversed.iter(),
        );
        assert_eq!(once, again);
        assert_eq!(once["a"].bytes_free(), 60);
        assert_eq!(once["b"].bytes_free(), 0);
        assert_eq!(once["a"].slots_free(TaskKind::MergeWal), 1);
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn heartbeat_is_published_under_a_lease_and_load_headroom_reads_it() {
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
        cfg.etcd.etcd_prefix = format!("/executors-test/{}", lance_context_core::generate_id());
        cfg.planner.planner_enabled = true;
        cfg.merge_wal_concurrency = 3;
        cfg.task_concurrency = 2;
        cfg.stats_scan_interval_secs = 0;
        cfg.compaction_interval_secs = 0;
        cfg.merge_wal_interval_secs = 0;
        let state = MasterState::new(cfg.clone()).await.unwrap();
        let keys = Keys::new(&cfg.etcd.etcd_prefix);
        let runner = tokio::spawn({
            let state = state.clone();
            async move { heartbeat_loop(&state).await }
        });
        let client = state.task_store.etcd_client().clone();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let id = state.admission.status().executor_id;
        let hb: Heartbeat = loop {
            let got = client.clone().get(keys.executor(&id), None).await.unwrap();
            if let Some(kv) = got.kvs().first() {
                assert!(kv.lease() != 0, "heartbeat must be leased");
                break serde_json::from_slice(kv.value()).unwrap();
            }
            assert!(std::time::Instant::now() < deadline, "no heartbeat");
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert_eq!(hb.kind, ExecutorKind::ResidentMaster);
        assert_eq!(hb.slots_total[&TaskKind::MergeWal], 3);
        assert_eq!(hb.slots_total[&TaskKind::IndexId], 2);
        assert!(!hb.draining);
        // Write a shadow assignment against it and read headroom back.
        let a = asg(&id, TaskKind::MergeWal, 0, AssignmentState::Shadow);
        client
            .clone()
            .put(
                keys.assignment("t", &a.unit_id),
                serde_json::to_vec(&a).unwrap(),
                None,
            )
            .await
            .unwrap();
        let h = load_headroom(&client, &keys).await.unwrap();
        assert_eq!(h[&id].slots_free(TaskKind::MergeWal), 2);
        assert_eq!(h[&id].assignments, 1);
        // Draining stops the heartbeat and the loop exits; the lease is
        // revoked so the key disappears promptly.
        state.admission.begin_drain();
        let outcome = tokio::time::timeout(Duration::from_secs(15), runner)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(outcome, Ok(()));
        let got = client.clone().get(keys.executor(&id), None).await.unwrap();
        assert!(got.kvs().is_empty(), "heartbeat gone after drain");
        // But its assignment still reserves capacity for a future leader.
        let h = load_headroom(&client, &keys).await.unwrap();
        assert_eq!(h[&id].assignments, 1);
        assert!(h[&id].draining);
    }
}

#[cfg(test)]
mod shadow_isolation {
    /// Spec test 5: shadow assignments are written only by the planner and
    /// read only by the planner, `load_headroom` and the operator route. No
    /// scheduler, merge, catch-up or executor code path may consult them
    /// until P2 deliberately introduces binding. This is a source-level
    /// check so a stray read fails CI, not production.
    #[test]
    fn no_executor_code_path_reads_assignments() {
        let allowed = ["planner.rs", "executors.rs", "routes.rs"];
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders = Vec::new();
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !name.ends_with(".rs") || allowed.contains(&name) {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            if text.contains("assignments_prefix")
                || text.contains(".assignment(")
                || text.contains("/assignments/")
            {
                offenders.push(name.to_string());
            }
        }
        // The catchup/ subdirectory too.
        for sub in ["catchup"] {
            if let Ok(rd) = std::fs::read_dir(dir.join(sub)) {
                for entry in rd.flatten() {
                    let text = std::fs::read_to_string(entry.path()).unwrap_or_default();
                    if text.contains("assignments") {
                        offenders.push(format!("{sub}/{}", entry.file_name().to_string_lossy()));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "executor paths read assignments: {offenders:?}"
        );
        // routes.rs may only read through load_headroom.
        let routes = std::fs::read_to_string(dir.join("routes.rs")).unwrap();
        assert!(
            !routes.contains("assignments_prefix"),
            "routes must go through load_headroom"
        );
    }
}
