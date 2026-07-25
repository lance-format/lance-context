//! Background stats scanner.
//!
//! Every `stats_scan_interval_secs` the scanner enumerates all experiments from
//! the registry, opens each one read-only with bounded concurrency, records its
//! [`RolloutObservation`] into the stats table, and reconciles away rows for
//! experiments that have since been deleted from the registry. Failures on a
//! single experiment are logged and skipped; the next round retries.

use std::collections::HashSet;
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

/// Bound on one `_stats` maintenance pass (compaction + version cleanup) so a
/// slow object store cannot wedge the scanner loop.
const MAINTENANCE_TIMEOUT: Duration = Duration::from_secs(300);

/// Compact `_stats` and prune its old manifest versions.
///
/// `_stats` is written delete-then-append (one upsert per experiment per
/// round), so its version chain and fragment count grow every scan and Lance
/// never reclaims them on its own. Callers must hold the `stats-writer`
/// coordination lock so only one replica ever rewrites the dataset.
pub async fn maintain_stats(state: &Arc<MasterState>) -> lance::Result<()> {
    let ttl = Duration::from_secs(state.config.stats_history_ttl_secs);
    let start = std::time::Instant::now();
    let mut stats = state.stats.lock().await;
    match tokio::time::timeout(MAINTENANCE_TIMEOUT, stats.maintain(ttl)).await {
        Ok(Ok((compaction, removal))) => {
            metrics::histogram!("master_stats_maintenance_duration_seconds")
                .record(start.elapsed().as_secs_f64());
            metrics::counter!("master_stats_versions_removed_total")
                .increment(removal.old_versions);
            metrics::gauge!("master_stats_version").set(stats.version() as f64);
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
        Ok(Err(e)) => Err(e),
        Err(_) => Err(lance::Error::io("stats maintenance timed out")),
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
    // counters by reading the existing stats row first.
    let observed: Vec<StatRow> = stream::iter(entries)
        .map(|entry| {
            let state = state.clone();
            async move {
                match observe_one(&state, &entry.name, &entry.uri).await {
                    Ok(row) => Some(row),
                    Err(e) => {
                        tracing::warn!(store = %entry.name, error = %e, "scan: observe failed");
                        None
                    }
                }
            }
        })
        .buffer_unordered(concurrency)
        .filter_map(|row| async move { row })
        .collect()
        .await;

    let count = observed.len();

    // Upsert observed rows and reconcile deletions under the stats lock.
    let mut stats = state.stats.lock().await;
    for row in observed {
        if let Err(e) = stats.upsert(&row).await {
            tracing::warn!(store = %row.name, error = %e, "stats upsert failed");
        }
    }
    // Drop stats rows for experiments removed from the registry.
    let existing = stats.list(None, usize::MAX, 0).await?;
    let mut total_rows: i64 = 0;
    let mut total_fragments: i64 = 0;
    let mut live_count: usize = 0;
    for row in existing {
        if !live.contains(&row.name) {
            if let Err(e) = stats.remove(&row.name).await {
                tracing::warn!(store = %row.name, error = %e, "stats reconcile-remove failed");
            }
        } else {
            live_count += 1;
            total_rows += row.row_count;
            total_fragments += row.fragment_count;
        }
    }

    metrics::histogram!("master_scan_duration_seconds").record(scan_start.elapsed().as_secs_f64());
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
