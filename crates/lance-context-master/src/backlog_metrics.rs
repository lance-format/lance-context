//! A small, shared scan result, not a second fleet scan. Scrapes do no I/O.
use std::fmt::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::stats_store::StatRow;
use crate::task_store::TaskStore;

const REFRESH: Duration = Duration::from_secs(60);
const READ_TIMEOUT: Duration = Duration::from_secs(5);
pub(crate) const MAX_SNAPSHOT_BYTES: usize = 4096;

/// Fixed-cardinality summary. Failed observations are explicit, including
/// failures without an older stats row to carry forward.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Snapshot {
    pub schema: u32,
    pub scan_started_ms: i64,
    pub committed_ms: i64,
    pub oldest_observation_ms: i64,
    pub stores: u64,
    pub failed_observations: u64,
    pub generations: u64,
    pub largest_store: u64,
    pub stores_over_1000: u64,
    pub largest_shard: u64,
    pub stores_with_shard_over_4096: u64,
}

impl Snapshot {
    pub(crate) fn new(
        rows: &[StatRow],
        failed: usize,
        scan_started_ms: i64,
        largest_shard: u64,
        stores_with_shard_over_4096: u64,
    ) -> Self {
        Self {
            schema: 1,
            scan_started_ms,
            committed_ms: Utc::now().timestamp_millis(),
            oldest_observation_ms: rows
                .iter()
                .map(|r| r.scanned_at)
                .min()
                .unwrap_or(scan_started_ms),
            stores: rows.len() as u64,
            failed_observations: failed as u64,
            generations: rows
                .iter()
                .map(|r| r.pending_wal_generations.max(0) as u64)
                .sum(),
            largest_store: rows
                .iter()
                .map(|r| r.pending_wal_generations.max(0) as u64)
                .max()
                .unwrap_or(0),
            stores_over_1000: rows
                .iter()
                .filter(|r| r.pending_wal_generations > 1000)
                .count() as u64,
            largest_shard,
            stores_with_shard_over_4096,
        }
    }

    pub(crate) fn validate(&self) -> bool {
        self.schema == 1
            && self.scan_started_ms > 0
            && self.oldest_observation_ms > 0
            && self.scan_started_ms <= self.committed_ms
            && self.oldest_observation_ms <= self.committed_ms
            && self.largest_store <= self.generations
            && self.largest_shard <= self.largest_store
            && self.stores_over_1000 <= self.stores
            && self.stores_with_shard_over_4096 <= self.stores
    }
}

#[derive(Default)]
struct Cached {
    snapshot: Option<Snapshot>,
    revision: i64,
    confirmed: Option<Instant>,
}

/// Only the compact metrics snapshot is cached here, never per-table payloads.
pub struct BacklogMetrics {
    cached: Mutex<Cached>,
    max_age: Duration,
}

impl BacklogMetrics {
    pub fn new(max_age: Duration) -> Arc<Self> {
        Arc::new(Self {
            cached: Mutex::new(Cached::default()),
            max_age,
        })
    }

    fn update(&self, result: Result<Option<(Snapshot, i64)>, ()>, now: Instant) {
        let mut cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        match result {
            Ok(Some((snapshot, revision)))
                if snapshot.validate() && revision >= cached.revision =>
            {
                cached.snapshot = Some(snapshot);
                cached.revision = revision;
                cached.confirmed = Some(now);
            }
            Ok(None) => {
                *cached = Cached::default();
            }
            _ => {
                cached.confirmed = None;
            }
        }
    }

    /// Dynamic exposition avoids retaining a dead publisher's last gauge.
    /// Failure, cold start and expiration never emit a false backlog of zero.
    pub fn render(&self) -> String {
        self.render_at(Instant::now(), Utc::now().timestamp_millis())
    }

    fn render_at(&self, now: Instant, wall_ms: i64) -> String {
        let mut out = String::new();
        let Ok(cached) = self.cached.try_lock() else {
            gauge(
                &mut out,
                "snapshot_available",
                0.0,
                "A complete fresh shared snapshot is available (1), otherwise unknown (0).",
            );
            return out;
        };
        let readable = cached
            .confirmed
            .is_some_and(|at| now.saturating_duration_since(at) <= REFRESH + READ_TIMEOUT);
        gauge(
            &mut out,
            "snapshot_readable",
            f64::from(readable),
            "The shared summary was successfully read within the refresh deadline.",
        );
        let Some(snapshot) = &cached.snapshot else {
            gauge(
                &mut out,
                "snapshot_available",
                0.0,
                "A complete fresh shared snapshot is available (1), otherwise unknown (0).",
            );
            return out;
        };
        let age_ms =
            wall_ms.saturating_sub(snapshot.oldest_observation_ms.min(snapshot.scan_started_ms));
        // A materially future-dated snapshot is unknown, not artificially fresh.
        let fresh = age_ms >= 0
            && age_ms as u128 <= self.max_age.as_millis()
            && snapshot.committed_ms <= wall_ms;
        let complete = snapshot.failed_observations == 0;
        let available = readable && fresh && complete;
        gauge(
            &mut out,
            "snapshot_age_seconds",
            age_ms.max(0) as f64 / 1000.0,
            "Age of the oldest observation or start of the shared scan; never the scrape time.",
        );
        gauge(
            &mut out,
            "snapshot_complete",
            f64::from(complete),
            "All registry entries were observed successfully in this shared scan.",
        );
        gauge(
            &mut out,
            "snapshot_fresh",
            f64::from(fresh),
            "The shared scan observations are within the configured age limit.",
        );
        gauge(
            &mut out,
            "snapshot_available",
            f64::from(available),
            "A complete fresh shared snapshot is available (1), otherwise unknown (0).",
        );
        gauge(
            &mut out,
            "snapshot_revision",
            cached.revision as f64,
            "Shared summary etcd revision; replicas converge without summing snapshots.",
        );
        gauge(
            &mut out,
            "snapshot_failed_stores",
            snapshot.failed_observations as f64,
            "Registry entries not successfully observed; these are unknown, not zero backlog.",
        );
        if available {
            gauge(&mut out, "generations", snapshot.generations as f64, "Pending WAL generations summed across stores in the shared scan. Gauge, not a rate.");
            gauge(
                &mut out,
                "largest_store_generations",
                snapshot.largest_store as f64,
                "Largest sum across all shards of one store; not the per-shard read cap.",
            );
            gauge(
                &mut out,
                "stores_over_1000",
                snapshot.stores_over_1000 as f64,
                "Stores with more than 1000 pending generations, including those above 4096.",
            );
            gauge(
                &mut out,
                "largest_shard_generations",
                snapshot.largest_shard as f64,
                "Largest pending count on a single WAL shard in the shared scan.",
            );
            gauge(&mut out, "stores_with_shard_over_4096", snapshot.stores_with_shard_over_4096 as f64, "Stores with a shard above the reference threshold 4096; risk, not observed HTTP refusals. Actual configured caps may differ.");
            gauge(
                &mut out,
                "observed_stores",
                snapshot.stores as f64,
                "Number of stores represented by the complete shared snapshot.",
            );
        }
        out
    }

    /// One fixed-key read per minute; never list tables or query on a scrape.
    /// Failed reads invalidate this replica until a later successful read.
    pub fn spawn(self: &Arc<Self>, store: TaskStore) -> tokio::task::JoinHandle<()> {
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                let Some(metrics) = weak.upgrade() else {
                    return;
                };
                let result =
                    tokio::time::timeout(READ_TIMEOUT, store.read_backlog_snapshot()).await;
                metrics.update(
                    match result {
                        Ok(Ok(snapshot)) => Ok(snapshot),
                        _ => Err(()),
                    },
                    Instant::now(),
                );
                drop(metrics);
                tokio::time::sleep(REFRESH).await;
            }
        })
    }
}

fn gauge(out: &mut String, suffix: &str, value: f64, help: &str) {
    let name = format!("master_wal_backlog_{suffix}");
    let _ = writeln!(
        out,
        "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(generations: u64) -> Snapshot {
        Snapshot {
            schema: 1,
            scan_started_ms: 100_000,
            committed_ms: 110_000,
            oldest_observation_ms: 101_000,
            stores: 2,
            failed_observations: 0,
            generations,
            largest_store: generations,
            stores_over_1000: 1,
            largest_shard: generations / 2,
            stores_with_shard_over_4096: 0,
        }
    }

    #[test]
    fn publisher_handoff_replaces_old_high_value_on_all_replicas() {
        let now = Instant::now();
        let a = BacklogMetrics::new(Duration::from_secs(900));
        let b = BacklogMetrics::new(Duration::from_secs(900));
        a.update(Ok(Some((snapshot(3913), 10))), now);
        for replica in [&a, &b] {
            replica.update(Ok(Some((snapshot(1964), 11))), now);
        }
        assert_eq!(a.render_at(now, 120_000), b.render_at(now, 120_000));
        assert!(a
            .render_at(now, 120_000)
            .contains("master_wal_backlog_generations 1964\n"));
        // An old replica cannot revive the previous number with a stale read.
        a.update(Ok(Some((snapshot(3913), 10))), now);
        assert!(!a
            .render_at(now, 120_000)
            .contains("master_wal_backlog_generations "));
    }

    #[test]
    fn unavailable_partial_and_expired_snapshots_do_not_emit_zero_backlog() {
        let now = Instant::now();
        let metrics = BacklogMetrics::new(Duration::from_secs(300));
        assert!(!metrics
            .render_at(now, 120_000)
            .contains("master_wal_backlog_generations "));
        metrics.update(Ok(Some((snapshot(3913), 10))), now);
        assert!(metrics
            .render_at(now, 120_000)
            .contains("master_wal_backlog_snapshot_age_seconds 20\n"));
        assert!(!metrics
            .render_at(now + Duration::from_secs(66), 186_000)
            .contains("master_wal_backlog_generations "));
        // Reading the old summary again does not reset its observation age.
        metrics.update(Ok(Some((snapshot(3913), 10))), now);
        assert!(!metrics
            .render_at(now, 401_000)
            .contains("master_wal_backlog_generations "));
        let mut partial = snapshot(3913);
        partial.failed_observations = 1;
        metrics.update(Ok(Some((partial, 11))), now);
        assert!(metrics
            .render_at(now, 120_000)
            .contains("master_wal_backlog_snapshot_complete 0\n"));
        assert!(!metrics
            .render_at(now, 120_000)
            .contains("master_wal_backlog_generations "));
        metrics.update(Err(()), now);
        assert!(metrics
            .render_at(now, 120_000)
            .contains("master_wal_backlog_snapshot_readable 0\n"));
    }

    #[test]
    fn empty_committed_scan_is_zero_but_future_or_invalid_snapshot_is_unknown() {
        let now = Instant::now();
        let metrics = BacklogMetrics::new(Duration::from_secs(300));
        let mut empty = snapshot(0);
        empty.stores_over_1000 = 0;
        empty.stores = 0;
        metrics.update(Ok(Some((empty, 1))), now);
        assert!(metrics
            .render_at(now, 120_000)
            .contains("master_wal_backlog_generations 0\n"));
        assert!(!metrics
            .render_at(now, 90_000)
            .contains("master_wal_backlog_generations "));
        let mut invalid = snapshot(100);
        invalid.schema = 2;
        metrics.update(Ok(Some((invalid, 2))), now);
        assert!(!metrics
            .render_at(now, 120_000)
            .contains("master_wal_backlog_generations "));
    }
}
