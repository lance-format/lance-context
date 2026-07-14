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

/// Run a single scan pass: refresh every experiment's stats row and drop rows
/// for experiments no longer in the registry. Returns the number of
/// experiments successfully observed.
pub async fn scan_once(state: &Arc<MasterState>) -> lance::Result<usize> {
    let scan_start = std::time::Instant::now();
    let entries = state.registry.read().await.list().await?;
    let live: HashSet<String> = entries.iter().map(|e| e.name.clone()).collect();
    let concurrency = state.config.scan_concurrency.max(1);

    // Observe experiments concurrently (bounded), preserving prior compaction
    // counters by reading the existing stats row first.
    let observed: Vec<StatRow> = stream::iter(entries)
        .map(|entry| {
            let state = state.clone();
            async move { observe_one(&state, &entry.name, &entry.uri).await }
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
/// `None` (logged) on any error or timeout.
async fn observe_one(state: &Arc<MasterState>, name: &str, uri: &str) -> Option<StatRow> {
    let opts = RolloutStoreOptions::default();
    let open = RolloutStore::open_existing_with_options(uri, opts);
    let store = match tokio::time::timeout(OBSERVE_TIMEOUT, open).await {
        Ok(Ok(store)) => store,
        Ok(Err(e)) => {
            tracing::warn!(store = %name, error = %e, "scan: open failed");
            return None;
        }
        Err(_) => {
            tracing::warn!(store = %name, "scan: open timed out");
            return None;
        }
    };
    let obs = match tokio::time::timeout(OBSERVE_TIMEOUT, store.observe()).await {
        Ok(Ok(obs)) => obs,
        Ok(Err(e)) => {
            tracing::warn!(store = %name, error = %e, "scan: observe failed");
            return None;
        }
        Err(_) => {
            tracing::warn!(store = %name, "scan: observe timed out");
            return None;
        }
    };

    // Carry compaction counters forward across scans.
    let (last_compaction, total_compactions) = {
        let stats = state.stats.lock().await;
        match stats.get(name).await {
            Ok(Some(prev)) => (prev.last_compaction, prev.total_compactions),
            _ => (StatRow::NO_COMPACTION, 0),
        }
    };

    Some(StatRow {
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

/// Spawn the periodic scanner. Returns `None` when the interval is `0`.
pub fn spawn_scanner(state: &Arc<MasterState>) -> Option<JoinHandle<()>> {
    let interval_secs = state.config.stats_scan_interval_secs;
    if interval_secs == 0 {
        return None;
    }
    let state = state.clone();
    Some(tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
        loop {
            ticker.tick().await;
            match scan_once(&state).await {
                Ok(n) => tracing::info!(experiments = n, "stats scan complete"),
                Err(e) => tracing::warn!(error = %e, "stats scan round failed"),
            }
        }
    }))
}
