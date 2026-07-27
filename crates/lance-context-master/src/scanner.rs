//! Background stats scanner.
//!
//! Every `stats_scan_interval_secs` the scanner enumerates all experiments from
//! the registry, opens each one read-only with bounded concurrency, records its
//! [`RolloutObservation`] into the stats table, and reconciles away rows for
//! experiments that have since been deleted from the registry. Failures on a
//! single experiment are logged and skipped; the next round retries.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use futures::stream::{self, StreamExt};
use lance_context_core::{RolloutStore, RolloutStoreOptions};
use tokio::task::JoinHandle;

use crate::state::MasterState;
use crate::stats_store::StatRow;

/// Per-experiment open+observe timeout so one wedged dataset cannot stall a
/// scan round.
const OBSERVE_TIMEOUT: Duration = Duration::from_secs(30);

/// Bound on one steady-state `_stats` maintenance pass so a slow object store
/// cannot wedge the scanner loop.
///
/// Deliberately not applied to the first pass after startup: see
/// [`maintain_stats`].
const MAINTENANCE_TIMEOUT: Duration = Duration::from_secs(300);

/// Compact `_stats` and prune its old manifest versions.
///
/// Scans now write the table as one `Overwrite` snapshot per round rather than
/// two commits per experiment, so in steady state the version chain grows by
/// one per round and this pass is cheap. It remains necessary to reclaim those
/// versions, and to recover deployments that ran the old per-row path.
///
/// # The first pass runs without a timeout
///
/// A deployment upgraded from the per-row path can arrive with a chain
/// hundreds of thousands of versions long (246k+ observed). Compacting and
/// pruning that cannot finish inside `MAINTENANCE_TIMEOUT`, so every pass timed
/// out, rolled back, and left the table exactly as bloated as before — the
/// bound guaranteed the table could never recover. The first pass after startup
/// therefore runs unbounded, and subsequent passes take the bound: by then the
/// backlog is gone and any pass exceeding it is a genuine fault.
///
/// Callers must hold the `stats-writer` coordination lock so only one replica
/// ever rewrites the dataset.
pub async fn maintain_stats(state: &Arc<MasterState>) -> lance::Result<()> {
    let ttl = Duration::from_secs(state.config.stats_history_ttl_secs);
    let start = std::time::Instant::now();
    let mut stats = state.stats.lock().await;

    // `swap` so exactly one pass per process is unbounded, even if several
    // scanner ticks race here.
    let first_pass = state.stats_maintenance_done.swap(true, Ordering::SeqCst);
    let outcome = if first_pass {
        tokio::time::timeout(MAINTENANCE_TIMEOUT, stats.maintain(ttl))
            .await
            .unwrap_or_else(|_| Err(lance::Error::io("stats maintenance timed out")))
    } else {
        tracing::info!(
            version = stats.version(),
            "running first stats maintenance pass without a timeout; \
             a table carried over from the per-row write path can take a while to reclaim"
        );
        stats.maintain(ttl).await
    };

    match outcome {
        Ok((compaction, removal)) => {
            state.stats_maintenance_failures.store(0, Ordering::Relaxed);
            state
                .stats_last_reclaimed_version
                .store(stats.version(), Ordering::Relaxed);
            metrics::histogram!("master_stats_maintenance_duration_seconds")
                .record(start.elapsed().as_secs_f64());
            metrics::counter!("master_stats_versions_removed_total")
                .increment(removal.old_versions);
            metrics::gauge!("master_stats_version").set(stats.version() as f64);
            metrics::gauge!("master_stats_maintenance_consecutive_failures").set(0.0);
            metrics::gauge!("master_stats_unreclaimed_versions").set(0.0);
            tracing::info!(
                fragments_removed = compaction.fragments_removed,
                fragments_added = compaction.fragments_added,
                old_versions_removed = removal.old_versions,
                bytes_removed = removal.bytes_removed,
                version = stats.version(),
                "stats maintenance complete"
            );
            Ok(())
        }
        Err(e) => {
            // Failure was previously silent: `master_stats_versions_removed_total`
            // only moves on success, so a maintenance pass that kept failing
            // (object-store outage, permissions, a genuinely stuck compaction)
            // showed up as a counter that stopped incrementing -- indistinguishable
            // from "nothing to reclaim". Meanwhile versions accumulate again and
            // the table walks back toward the state this whole path exists to
            // prevent.
            //
            // Export the two things worth alerting on. Note the raw version
            // *number* is not one of them: it climbs by design and says nothing
            // about disk. What matters is how many versions have gone unreclaimed
            // since the last successful pass.
            let failures = state
                .stats_maintenance_failures
                .fetch_add(1, Ordering::Relaxed)
                + 1;
            let last_reclaimed = state.stats_last_reclaimed_version.load(Ordering::Relaxed);
            let unreclaimed = stats.version().saturating_sub(last_reclaimed);

            metrics::counter!("master_stats_maintenance_failures_total").increment(1);
            metrics::gauge!("master_stats_maintenance_consecutive_failures").set(failures as f64);
            metrics::gauge!("master_stats_unreclaimed_versions").set(unreclaimed as f64);

            tracing::warn!(
                error = %e,
                consecutive_failures = failures,
                unreclaimed_versions = unreclaimed,
                version = stats.version(),
                "stats maintenance failed; old manifests are not being reclaimed"
            );
            Err(e)
        }
    }
}

/// Run a single scan pass: refresh every experiment's stats row and drop rows
/// for experiments no longer in the registry. Returns the number of
/// experiments successfully observed.
pub async fn scan_once(state: &Arc<MasterState>) -> lance::Result<usize> {
    let guard = state.task_store.coordination_lock("stats-writer").await?;
    let result = scan_once_inner(state).await;
    let release = state.task_store.release_coordination_lock(guard).await;
    match (result, release) {
        (Ok(count), Ok(())) => Ok(count),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

async fn try_scan_once(state: &Arc<MasterState>, maintain: bool) -> lance::Result<Option<usize>> {
    let Some(guard) = state
        .task_store
        .try_coordination_lock("stats-writer")
        .await?
    else {
        return Ok(None);
    };
    let result = scan_once_inner(state).await;
    // Maintenance runs under the same writer lock as the scan that just
    // created the versions, and its failure never fails the scan round.
    if maintain {
        if let Err(e) = maintain_stats(state).await {
            tracing::warn!(error = %e, "stats maintenance failed");
        }
    }
    let release = state.task_store.release_coordination_lock(guard).await;
    match (result, release) {
        (Ok(count), Ok(())) => Ok(Some(count)),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

async fn scan_once_inner(state: &Arc<MasterState>) -> lance::Result<usize> {
    let scan_start = std::time::Instant::now();
    let entries = state.registry.write().await.list().await?;
    let live: HashSet<String> = entries.iter().map(|e| e.name.clone()).collect();
    let concurrency = state.config.scan_concurrency.max(1);

    // Observe experiments concurrently (bounded), preserving prior compaction
    // counters by reading the existing stats row first. `None` means this round
    // could not observe that experiment; its previous row is carried over below
    // rather than dropped.
    let observed: Vec<(String, Option<StatRow>)> = stream::iter(entries)
        .map(|entry| {
            let state = state.clone();
            async move {
                match observe_one(&state, &entry.name, &entry.uri).await {
                    Ok(row) => (entry.name, Some(row)),
                    Err(e) => {
                        tracing::warn!(store = %entry.name, error = %e, "scan: observe failed");
                        (entry.name, None)
                    }
                }
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    let count = observed.iter().filter(|(_, row)| row.is_some()).count();

    let mut stats = state.stats.lock().await;

    // Build the round's snapshot. This replaces the whole table in one commit
    // instead of two commits per experiment, and subsumes the old
    // reconcile-remove pass: an experiment absent from the registry is simply
    // absent from the snapshot.
    //
    // An experiment this round failed to observe keeps its previous row, so a
    // transient storage error does not blank it from the UI or, worse, hide it
    // from the compaction and WAL-merge sweeps that read this table.
    let previous = stats.list(None, usize::MAX, 0).await?;
    let mut carry_over: HashMap<String, StatRow> = previous
        .into_iter()
        .map(|row| (row.name.clone(), row))
        .collect();

    let mut snapshot: Vec<StatRow> = Vec::with_capacity(observed.len());
    for (name, row) in observed {
        match row {
            Some(row) => snapshot.push(row),
            None => {
                if let Some(stale) = carry_over.remove(&name) {
                    snapshot.push(stale);
                }
            }
        }
    }
    snapshot.sort_by(|a, b| a.name.cmp(&b.name));

    let mut total_rows: i64 = 0;
    let mut total_fragments: i64 = 0;
    let mut live_count: usize = 0;
    for row in &snapshot {
        if live.contains(&row.name) {
            live_count += 1;
            total_rows += row.row_count;
            total_fragments += row.fragment_count;
        }
    }

    if let Err(e) = stats.write_snapshot(&snapshot).await {
        tracing::warn!(error = %e, "stats snapshot write failed");
    }

    metrics::histogram!("master_scan_duration_seconds").record(scan_start.elapsed().as_secs_f64());
    // Named without a `_total` suffix: these are gauges (current value), and
    // `_total` is the counter convention. Datadog's OpenMetrics check infers
    // type from the name, so `_total` on a gauge was being ingested as a
    // monotonic count -- graphing "experiments created per second" for a metric
    // whose meaning is "how many exist right now". `rate()` was equally
    // meaningless in Prometheus. The `_total` names are still emitted alongside,
    // deprecated, so existing dashboards keep working.
    metrics::gauge!("master_experiments").set(live_count as f64);
    metrics::gauge!("master_rollout_rows").set(total_rows as f64);
    metrics::gauge!("master_rollout_fragments").set(total_fragments as f64);
    // Deprecated aliases -- remove once dashboards have migrated.
    metrics::gauge!("master_experiments_total").set(live_count as f64);
    metrics::gauge!("master_rollout_rows_total").set(total_rows as f64);
    metrics::gauge!("master_rollout_fragments_total").set(total_fragments as f64);
    Ok(count)
}

/// Open one experiment read-only, observe it, and build its stats row. Preserves
/// `last_compaction`/`total_compactions` from any existing stats row. Returns
/// an error on open/observe failure or timeout.
async fn observe_one(state: &Arc<MasterState>, name: &str, uri: &str) -> lance::Result<StatRow> {
    let opts = RolloutStoreOptions::default();
    let open = RolloutStore::open_existing_with_options(uri, opts);
    let store = match tokio::time::timeout(OBSERVE_TIMEOUT, open).await {
        Ok(Ok(store)) => store,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(lance::Error::io(format!(
                "open timed out for store '{name}'"
            )))
        }
    };
    let obs = match tokio::time::timeout(OBSERVE_TIMEOUT, store.observe()).await {
        Ok(Ok(obs)) => obs,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(lance::Error::io(format!(
                "observe timed out for store '{name}'"
            )));
        }
    };

    // Carry compaction counters forward across scans.
    let (last_compaction, total_compactions) = {
        let mut stats = state.stats.lock().await;
        match stats.get(name).await {
            Ok(Some(prev)) => (prev.last_compaction, prev.total_compactions),
            _ => (StatRow::NO_COMPACTION, 0),
        }
    };

    Ok(StatRow {
        name: name.to_string(),
        uri: uri.to_string(),
        row_count: obs.row_count,
        fragment_count: obs.fragment_count,
        last_updated: obs.last_updated,
        pending_wal_generations: obs.pending_wal_generations,
        last_compaction,
        total_compactions,
        scanned_at: Utc::now().timestamp_millis(),
    })
}

/// Refresh one experiment immediately and persist its new stats row.
pub async fn refresh_one(state: &Arc<MasterState>, name: &str, uri: &str) -> lance::Result<()> {
    let guard = state.task_store.coordination_lock("stats-writer").await?;
    let result = match observe_one(state, name, uri).await {
        Ok(row) => state.stats.lock().await.upsert(&row).await,
        Err(error) => Err(error),
    };
    let release = state.task_store.release_coordination_lock(guard).await;
    result.and(release)
}

/// Spawn the periodic scanner. Returns `None` when the interval is `0`.
pub fn spawn_scanner(state: &Arc<MasterState>) -> Option<JoinHandle<()>> {
    let interval_secs = state.config.stats_scan_interval_secs;
    if interval_secs == 0 {
        return None;
    }
    let state = state.clone();
    Some(tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
        let every_n = state.config.stats_maintenance_every_n_scans;
        let mut round: u64 = 0;
        loop {
            ticker.tick().await;
            round += 1;
            // Round 1 included: an existing deployment may start with a very
            // long version chain, and waiting N rounds to reclaim it would
            // leave cold start slow for another N intervals.
            let maintain = every_n > 0 && (round == 1 || round.is_multiple_of(every_n));
            match try_scan_once(&state, maintain).await {
                Ok(Some(n)) => tracing::info!(experiments = n, "stats scan complete"),
                Ok(None) => {
                    tracing::debug!("stats scan skipped; another master owns the writer lock")
                }
                Err(e) => tracing::warn!(error = %e, "stats scan round failed"),
            }
        }
    }))
}

#[cfg(test)]
mod maintenance_alerting_tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Mirrors the failure bookkeeping in `maintain_stats`, which cannot be
    /// exercised directly without an etcd-backed `MasterState`.
    ///
    /// The property under test: a failing maintenance pass must leave a signal.
    /// Previously it left none -- `master_stats_versions_removed_total` only
    /// moves on success, so a pass failing for days looked exactly like a pass
    /// with nothing to reclaim, while manifests piled back up.
    struct Bookkeeping {
        failures: AtomicU64,
        last_reclaimed_version: AtomicU64,
    }

    impl Bookkeeping {
        fn new() -> Self {
            Self {
                failures: AtomicU64::new(0),
                last_reclaimed_version: AtomicU64::new(0),
            }
        }

        fn on_success(&self, version: u64) -> (u64, u64) {
            self.failures.store(0, Ordering::Relaxed);
            self.last_reclaimed_version
                .store(version, Ordering::Relaxed);
            (0, 0)
        }

        fn on_failure(&self, version: u64) -> (u64, u64) {
            let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
            let unreclaimed =
                version.saturating_sub(self.last_reclaimed_version.load(Ordering::Relaxed));
            (failures, unreclaimed)
        }
    }

    #[test]
    fn failures_accumulate_and_reset_on_success() {
        let b = Bookkeeping::new();

        // A first success establishes the reclaim watermark.
        assert_eq!(b.on_success(10), (0, 0));

        // Each subsequent failure raises the consecutive count, and the
        // unreclaimed gap tracks versions written since that watermark.
        assert_eq!(b.on_failure(11), (1, 1));
        assert_eq!(b.on_failure(12), (2, 2));
        assert_eq!(b.on_failure(20), (3, 10));

        // One success clears both signals.
        assert_eq!(b.on_success(20), (0, 0));
        assert_eq!(b.on_failure(21), (1, 1));
    }

    /// The watermark starts at 0, so failures before any successful pass still
    /// report a non-zero gap rather than silently reporting "nothing pending".
    #[test]
    fn failure_before_first_success_still_reports_a_gap() {
        let b = Bookkeeping::new();
        let (failures, unreclaimed) = b.on_failure(246_000);
        assert_eq!(failures, 1);
        assert_eq!(
            unreclaimed, 246_000,
            "a bloated table that has never been reclaimed must report its full backlog"
        );
    }
}
