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
use lance_context_api::ExperimentSummary;

/// Per-experiment open+observe timeout so one wedged dataset cannot stall a
/// scan round.
const OBSERVE_TIMEOUT: Duration = Duration::from_secs(30);

/// Bound on one steady-state `_stats` maintenance pass so a slow object store
/// cannot wedge the scanner loop.
///
/// Deliberately not applied to the first pass after startup: see
/// [`maintain_stats`].
const MAINTENANCE_TIMEOUT: Duration = Duration::from_secs(300);

/// Bound on the merge and compaction steps of retiring one experiment. Larger
/// than `OBSERVE_TIMEOUT` because both genuinely rewrite data; a retirement that
/// exceeds it is abandoned and retried next round, which is harmless.
const RETIRE_TIMEOUT: Duration = Duration::from_secs(600);

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

    // Read the previous snapshot once, up front. Each experiment's row supplies
    // both its last observed version (to decide whether a full observation is
    // needed) and its compaction counters (which are carried forward). This
    // used to be a `stats.get()` per experiment, taking the stats lock once per
    // experiment per round.
    let previous: HashMap<String, StatRow> = {
        let mut stats = state.stats.lock().await;
        stats
            .list(None, usize::MAX, 0)
            .await?
            .into_iter()
            .map(|row| (row.name.clone(), row))
            .collect()
    };
    let previous = Arc::new(previous);
    let rollout_options = state.rollout_store_options();

    // Observe experiments concurrently (bounded). `None` means this round could
    // not observe that experiment; its previous row is carried over below
    // rather than dropped.
    let observed: Vec<(String, Option<(StatRow, bool)>)> = stream::iter(entries)
        .map(|entry| {
            let previous = previous.clone();
            let rollout_options = rollout_options.clone();
            async move {
                let prev = previous.get(&entry.name);
                match observe_one(&entry.name, &entry.uri, prev, rollout_options).await {
                    Ok(result) => (entry.name, Some(result)),
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
    let failed = observed.len() - count;
    let skipped = observed
        .iter()
        .filter(|(_, row)| matches!(row, Some((_, true))))
        .count();

    let mut stats = state.stats.lock().await;

    // Build the round's snapshot. This replaces the whole table in one commit
    // instead of two commits per experiment, and subsumes the old
    // reconcile-remove pass: an experiment absent from the registry is simply
    // absent from the snapshot.
    //
    // An experiment this round failed to observe keeps its previous row, so a
    // transient storage error does not blank it from the UI or, worse, hide it
    // from the compaction and WAL-merge sweeps that read this table.
    let mut carry_over = (*previous).clone();

    let mut snapshot: Vec<StatRow> = Vec::with_capacity(observed.len());
    for (name, row) in observed {
        match row {
            Some((row, _skipped)) => snapshot.push(row),
            None => {
                if let Some(stale) = carry_over.remove(&name) {
                    snapshot.push(stale);
                }
            }
        }
    }
    snapshot.sort_by(|a, b| a.name.cmp(&b.name));

    // Retire experiments with no writes for the configured window. Each is
    // merged, compacted and verified quiescent first; only then is its row
    // dropped, because a row absent from this table is invisible to both
    // auto-sweeps forever. See `retire_cold_experiments`.
    //
    // Done before the snapshot is written so a retirement takes effect in the
    // same commit rather than leaving a round where the row is stale.
    let retire_after = Duration::from_secs(state.config.stats_cold_retire_secs);
    let retired = retire_cold_experiments(
        &snapshot,
        retire_after,
        Utc::now().timestamp_millis(),
        state.rollout_store_options(),
    )
    .await;
    if !retired.is_empty() {
        snapshot.retain(|row| !retired.contains(&row.name));
    }
    metrics::gauge!("master_stats_hot_experiments").set(snapshot.len() as f64);

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

    // An empty snapshot is ambiguous at the store layer, so disambiguate here
    // where the inputs are in scope. `carry_over` above already re-adds the
    // previous row for every experiment that is still registered but failed to
    // observe, so an empty snapshot with `failed > 0` means the failures were
    // total -- refuse that wipe (write_snapshot no-ops). Empty with no failures
    // means the registry drained to zero, or everything left was retired: a
    // legitimate wipe that must be written through, otherwise rows for
    // deregistered experiments persist forever and keep feeding the compaction
    // and WAL-merge sweeps.
    let write = if snapshot.is_empty() && failed == 0 {
        stats.replace_snapshot(&snapshot).await
    } else {
        stats.write_snapshot(&snapshot).await
    };
    if let Err(e) = write {
        tracing::warn!(error = %e, "stats snapshot write failed");
    }

    metrics::histogram!("master_scan_duration_seconds").record(scan_start.elapsed().as_secs_f64());
    // How much of the round was avoided by the version check. A ratio near 1 is
    // the healthy state at scale: most experiments are cold and cost only an
    // open. A ratio trending to 0 means the incremental path is not engaging --
    // usually a stats table that keeps losing its rows.
    metrics::gauge!("master_scan_experiments_skipped").set(skipped as f64);
    metrics::gauge!("master_scan_experiments_observed").set((count - skipped) as f64);
    tracing::debug!(
        total = count,
        skipped,
        observed = count - skipped,
        "scan round complete"
    );
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

/// Open one experiment read-only and refresh its stats row, skipping the
/// expensive observation when nothing can have changed.
///
/// `previous` is the row from the last scan, if any.
///
/// # Why the version check pays for itself
///
/// Opening the dataset reads its manifest, which already carries the current
/// version — so comparing it against the last observed version is free. If they
/// match, no write has landed since, and every derived metric (`row_count`,
/// `fragment_count`, `pending_wal_generations`) is necessarily unchanged. The
/// previous row is then reused verbatim.
///
/// What that skips is the whole cost of `observe`: listing every MemWAL shard
/// and reading its manifest, opening each flushed generation to count pending
/// rows, and a `count_rows` over the base table. At tens of thousands of
/// experiments, almost all of them cold, that is the difference between a scan
/// that finishes inside its interval and one that never does.
async fn observe_one(
    name: &str,
    uri: &str,
    previous: Option<&StatRow>,
    options: RolloutStoreOptions,
) -> lance::Result<(StatRow, bool)> {
    let open = RolloutStore::open_existing_with_options(uri, options);
    let store = match tokio::time::timeout(OBSERVE_TIMEOUT, open).await {
        Ok(Ok(store)) => store,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(lance::Error::io(format!(
                "open timed out for store '{name}'"
            )))
        }
    };

    // Free: the manifest is already in hand from the open above.
    let current_version = store.version() as i64;
    if let Some(prev) = previous {
        if prev.version != StatRow::UNKNOWN_VERSION && prev.version == current_version {
            let mut row = prev.clone();
            row.uri = uri.to_string();
            row.scanned_at = Utc::now().timestamp_millis();
            return Ok((row, true));
        }
    }

    let obs = match tokio::time::timeout(OBSERVE_TIMEOUT, store.observe()).await {
        Ok(Ok(obs)) => obs,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(lance::Error::io(format!(
                "observe timed out for store '{name}'"
            )));
        }
    };

    // Carry compaction counters forward across scans. Taken from the caller's
    // snapshot rather than a per-experiment `stats.get()`, which used to take
    // the stats lock once per experiment per round.
    let (last_compaction, total_compactions) = previous
        .map_or((StatRow::NO_COMPACTION, 0), |prev| {
            (prev.last_compaction, prev.total_compactions)
        });

    Ok((
        StatRow {
            name: name.to_string(),
            uri: uri.to_string(),
            row_count: obs.row_count,
            fragment_count: obs.fragment_count,
            last_updated: obs.last_updated,
            pending_wal_generations: obs.pending_wal_generations,
            last_compaction,
            total_compactions,
            scanned_at: Utc::now().timestamp_millis(),
            version: obs.version as i64,
        },
        false,
    ))
}

/// Decide which of `rows` are cold enough to retire, and leave each one in a
/// state that needs no further maintenance before dropping it.
///
/// Returns the names that were successfully retired.
///
/// # Ordering is a correctness property, not an optimisation
///
/// Both auto-sweeps read the stats table and nothing else, so an experiment
/// absent from it is invisible to them forever. Retiring one that still has
/// un-merged MemWAL generations would strand those generations permanently:
/// their rows stay readable (the read path unions them) but they are never
/// folded into the base table, read amplification never goes down, and the
/// `_mem_wal/{shard}/` directories are never reclaimed. That is a storage leak
/// with no process left to notice it.
///
/// So retirement is: merge the WAL, compact, **verify the WAL actually drained**,
/// and only then drop the row. `compact_files` deliberately does not touch WAL
/// generations, so compaction alone would not have been enough.
///
/// Any step failing leaves the experiment in the table to be retried next
/// round. Refusing to retire is always safe; retiring early is not.
async fn retire_cold_experiments(
    rows: &[StatRow],
    retire_after: Duration,
    now_ms: i64,
    options: RolloutStoreOptions,
) -> HashSet<String> {
    if retire_after.is_zero() {
        return HashSet::new();
    }
    let cutoff_ms = now_ms.saturating_sub(retire_after.as_millis() as i64);

    let mut retired = HashSet::new();
    for row in rows {
        if row.last_updated > cutoff_ms {
            continue;
        }
        match prepare_for_retirement(&row.name, &row.uri, options.clone()).await {
            Ok(true) => {
                retired.insert(row.name.clone());
                metrics::counter!("master_stats_experiments_retired_total").increment(1);
                tracing::info!(
                    store = %row.name,
                    idle_ms = now_ms.saturating_sub(row.last_updated),
                    "retiring cold experiment from the stats table"
                );
            }
            Ok(false) => {
                // Still had pending generations after the merge, so something
                // else is writing or the merge did not fully drain. Keep it.
                tracing::debug!(
                    store = %row.name,
                    "cold experiment not retired: WAL still pending after merge"
                );
            }
            Err(e) => {
                metrics::counter!("master_stats_retire_failures_total").increment(1);
                tracing::warn!(
                    store = %row.name,
                    error = %e,
                    "failed to prepare cold experiment for retirement; keeping it"
                );
            }
        }
    }
    retired
}

/// Merge, compact, and verify one experiment is quiescent.
///
/// `Ok(true)` means it is safe to drop from the stats table.
async fn prepare_for_retirement(
    name: &str,
    uri: &str,
    options: RolloutStoreOptions,
) -> lance::Result<bool> {
    let mut store = match tokio::time::timeout(
        OBSERVE_TIMEOUT,
        RolloutStore::open_existing_with_options(uri, options),
    )
    .await
    {
        Ok(Ok(store)) => store,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(lance::Error::io(format!(
                "open timed out retiring store '{name}'"
            )))
        }
    };

    // 1. Drain the WAL. Must precede compaction: `compact_files` rewrites base
    //    fragments and leaves MemWAL generations untouched.
    tokio::time::timeout(RETIRE_TIMEOUT, store.cleanup_own_shard())
        .await
        .map_err(|_| lance::Error::io(format!("WAL merge timed out retiring '{name}'")))??;

    // 2. Compact, so the retired table is not left as many small fragments that
    //    nothing will ever come back to tidy.
    tokio::time::timeout(RETIRE_TIMEOUT, store.compact(None))
        .await
        .map_err(|_| lance::Error::io(format!("compaction timed out retiring '{name}'")))??;

    // 3. Verify. Only a genuinely drained shard may leave the sweeps' view.
    let obs = tokio::time::timeout(OBSERVE_TIMEOUT, store.observe())
        .await
        .map_err(|_| lance::Error::io(format!("observe timed out retiring '{name}'")))??;

    Ok(obs.pending_wal_generations == 0)
}

/// Observe one experiment on demand, without touching the stats table.
///
/// Used for experiments the stats table no longer holds — retired ones surfaced
/// by search or a detail request. Deliberately does not write a row back:
/// reading about a cold experiment must not make it hot again, or browsing the
/// UI would undo retirement and the table would creep back toward holding
/// everything.
pub async fn observe_cold(
    state: &MasterState,
    name: &str,
    uri: &str,
) -> lance::Result<ExperimentSummary> {
    observe_cold_with_options(name, uri, state.rollout_store_options()).await
}

async fn observe_cold_with_options(
    name: &str,
    uri: &str,
    options: RolloutStoreOptions,
) -> lance::Result<ExperimentSummary> {
    let (row, _) = observe_one(name, uri, None, options).await?;
    Ok(row.into_summary())
}

/// Refresh one experiment immediately and persist its new stats row.
///
/// Passes `None` as the previous row so this always does a full observation:
/// callers use it precisely when they believe something just changed (after a
/// compaction, or on an explicit refresh request), and the version check would
/// otherwise short-circuit exactly the observation they asked for.
pub async fn refresh_one(state: &Arc<MasterState>, name: &str, uri: &str) -> lance::Result<()> {
    let guard = state.task_store.coordination_lock("stats-writer").await?;
    let result = match observe_one(name, uri, None, state.rollout_store_options()).await {
        Ok((row, _)) => state.stats.lock().await.upsert(&row).await,
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

#[cfg(test)]
mod incremental_scan_tests {
    use super::*;
    use lance_context_core::{RolloutRecord, RolloutStore, RolloutStoreOptions, ROLE_ASSISTANT};
    use tempfile::TempDir;

    fn rec(id: &str) -> RolloutRecord {
        RolloutRecord {
            id: id.to_string(),
            rollout_id: "r".to_string(),
            problem_id: "p".to_string(),
            dataset: None,
            sequence_order: 0,
            role: ROLE_ASSISTANT.to_string(),
            created_at: Utc::now(),
            content: Some("x".to_string()),
            content_type: "text/plain".to_string(),
            model_input_string: None,
            model_output_string: None,
            rationale: None,
            problem_text: None,
            user_metadata: None,
            input_tokens: None,
            output_tokens: None,
            num_input_tokens: None,
            num_output_tokens: None,
            output_logprobs: None,
            input_logprobs: None,
            ref_logprobs: None,
            loss_mask: None,
            advantage: None,
            reward: None,
            raw_reward: None,
            grader_id: None,
            score: None,
            include_in_training: None,
            exclude_reason: None,
            policy_version: None,
            relationships: vec![],
            binary_payload: None,
            payload_size: None,
            payload_checksum: None,
            artifact_type: None,
            metadata: None,
        }
    }

    /// An experiment whose base version has not moved must not be re-observed.
    ///
    /// This is the property the whole incremental path rests on: at tens of
    /// thousands of experiments, almost all cold, a scan that fully observes
    /// every one cannot finish inside its interval. Opening the dataset is
    /// unavoidable (it is how the version is read), but everything after it --
    /// listing MemWAL shards, opening each flushed generation, counting rows --
    /// must be skipped when nothing has changed.
    #[tokio::test]
    async fn unchanged_experiment_is_not_reobserved() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("e.lance").to_string_lossy().to_string();
        {
            let store = RolloutStore::open_with_options(&uri, RolloutStoreOptions::default())
                .await
                .unwrap();
            store.add(&[rec("a")]).await.unwrap();
            store.flush().await.unwrap();
        }

        // First pass: nothing known, so a full observation happens.
        let (first, skipped) = observe_one("e", &uri, None, RolloutStoreOptions::default())
            .await
            .unwrap();
        assert!(!skipped, "the first observation cannot be skipped");
        assert_ne!(first.version, StatRow::UNKNOWN_VERSION);

        // Second pass with the row from the first: the version is unchanged, so
        // the expensive observation is skipped and the row is reused.
        let (second, skipped) =
            observe_one("e", &uri, Some(&first), RolloutStoreOptions::default())
                .await
                .unwrap();
        assert!(skipped, "an unchanged experiment must skip re-observation");
        assert_eq!(second.row_count, first.row_count);
        assert_eq!(second.fragment_count, first.fragment_count);
        assert_eq!(second.version, first.version);
        assert!(
            second.scanned_at >= first.scanned_at,
            "a skipped row still refreshes scanned_at so staleness stays visible"
        );
    }

    /// A write moves the base version, so the next scan must observe fully and
    /// pick up the new counts.
    #[tokio::test]
    async fn changed_experiment_is_reobserved() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("e.lance").to_string_lossy().to_string();
        let mut store = RolloutStore::open_with_options(&uri, RolloutStoreOptions::default())
            .await
            .unwrap();
        store.add(&[rec("a")]).await.unwrap();
        store.flush().await.unwrap();

        let (first, _) = observe_one("e", &uri, None, RolloutStoreOptions::default())
            .await
            .unwrap();

        // Merge the WAL into the base table: this advances the base version.
        store.cleanup_own_shard().await.unwrap();

        let (second, skipped) =
            observe_one("e", &uri, Some(&first), RolloutStoreOptions::default())
                .await
                .unwrap();
        assert!(
            !skipped,
            "a changed base version must force a full observation"
        );
        assert_ne!(
            second.version, first.version,
            "the new row must record the new version"
        );
    }

    /// A row from before the `version` column existed carries the sentinel and
    /// must force one full observation rather than being treated as unchanged.
    #[tokio::test]
    async fn unknown_version_forces_observation() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("e.lance").to_string_lossy().to_string();
        {
            let store = RolloutStore::open_with_options(&uri, RolloutStoreOptions::default())
                .await
                .unwrap();
            store.add(&[rec("a")]).await.unwrap();
            store.flush().await.unwrap();
        }

        let (mut legacy, _) = observe_one("e", &uri, None, RolloutStoreOptions::default())
            .await
            .unwrap();
        legacy.version = StatRow::UNKNOWN_VERSION;

        let (refreshed, skipped) =
            observe_one("e", &uri, Some(&legacy), RolloutStoreOptions::default())
                .await
                .unwrap();
        assert!(
            !skipped,
            "an unknown version must not be mistaken for an unchanged one"
        );
        assert_ne!(refreshed.version, StatRow::UNKNOWN_VERSION);
    }

    /// Compaction counters survive a skipped round.
    #[tokio::test]
    async fn skipped_round_preserves_compaction_counters() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("e.lance").to_string_lossy().to_string();
        {
            let store = RolloutStore::open_with_options(&uri, RolloutStoreOptions::default())
                .await
                .unwrap();
            store.add(&[rec("a")]).await.unwrap();
            store.flush().await.unwrap();
        }

        let (mut prev, _) = observe_one("e", &uri, None, RolloutStoreOptions::default())
            .await
            .unwrap();
        prev.last_compaction = 1_700_000_000_000;
        prev.total_compactions = 7;

        let (next, skipped) = observe_one("e", &uri, Some(&prev), RolloutStoreOptions::default())
            .await
            .unwrap();
        assert!(skipped);
        assert_eq!(next.last_compaction, 1_700_000_000_000);
        assert_eq!(next.total_compactions, 7);
    }
}

#[cfg(test)]
mod retirement_tests {
    use super::*;
    use lance_context_core::{RolloutRecord, RolloutStore, RolloutStoreOptions, ROLE_ASSISTANT};
    use tempfile::TempDir;

    fn rec(id: &str) -> RolloutRecord {
        RolloutRecord {
            id: id.to_string(),
            rollout_id: "r".to_string(),
            problem_id: "p".to_string(),
            dataset: None,
            sequence_order: 0,
            role: ROLE_ASSISTANT.to_string(),
            created_at: Utc::now(),
            content: Some("x".to_string()),
            content_type: "text/plain".to_string(),
            model_input_string: None,
            model_output_string: None,
            rationale: None,
            problem_text: None,
            user_metadata: None,
            input_tokens: None,
            output_tokens: None,
            num_input_tokens: None,
            num_output_tokens: None,
            output_logprobs: None,
            input_logprobs: None,
            ref_logprobs: None,
            loss_mask: None,
            advantage: None,
            reward: None,
            raw_reward: None,
            grader_id: None,
            score: None,
            include_in_training: None,
            exclude_reason: None,
            policy_version: None,
            relationships: vec![],
            binary_payload: None,
            payload_size: None,
            payload_checksum: None,
            artifact_type: None,
            metadata: None,
        }
    }

    fn row(name: &str, uri: &str, last_updated: i64) -> StatRow {
        StatRow {
            name: name.to_string(),
            uri: uri.to_string(),
            row_count: 1,
            fragment_count: 1,
            last_updated,
            pending_wal_generations: 0,
            last_compaction: StatRow::NO_COMPACTION,
            total_compactions: 0,
            scanned_at: last_updated,
            version: 1,
        }
    }

    /// Retirement must drain the WAL before dropping the row.
    ///
    /// The sweeps read the stats table and nothing else, so a retired
    /// experiment is invisible to them forever. Dropping one with pending
    /// generations would strand them: never merged, read amplification never
    /// recovers, `_mem_wal/` never reclaimed, and no process left to notice.
    #[tokio::test]
    async fn retirement_drains_the_wal_before_dropping_the_row() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("e.lance").to_string_lossy().to_string();
        {
            let store = RolloutStore::open_with_options(&uri, RolloutStoreOptions::default())
                .await
                .unwrap();
            for i in 0..3 {
                store.add(&[rec(&format!("r{i}"))]).await.unwrap();
                store.flush().await.unwrap();
            }
            // Pending generations exist at this point.
            let obs = store.observe().await.unwrap();
            assert!(
                obs.pending_wal_generations > 0,
                "test setup must leave un-merged generations"
            );
        }

        let now = Utc::now().timestamp_millis();
        let old = now - Duration::from_secs(30 * 86_400).as_millis() as i64;
        let retired = retire_cold_experiments(
            &[row("e", &uri, old)],
            Duration::from_secs(7 * 86_400),
            now,
            RolloutStoreOptions::default(),
        )
        .await;

        assert!(retired.contains("e"), "a cold experiment should retire");

        // The decisive assertion: nothing was left behind for a sweep that will
        // never look again.
        let store = RolloutStore::open_existing_with_options(&uri, RolloutStoreOptions::default())
            .await
            .unwrap();
        let obs = store.observe().await.unwrap();
        assert_eq!(
            obs.pending_wal_generations, 0,
            "retirement must leave zero pending generations, or they are stranded forever"
        );
        assert_eq!(obs.row_count, 3, "no rows may be lost by retirement");
    }

    /// An experiment written recently must not be retired.
    #[tokio::test]
    async fn recently_written_experiment_is_not_retired() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("e.lance").to_string_lossy().to_string();
        {
            let store = RolloutStore::open_with_options(&uri, RolloutStoreOptions::default())
                .await
                .unwrap();
            store.add(&[rec("a")]).await.unwrap();
            store.flush().await.unwrap();
        }

        let now = Utc::now().timestamp_millis();
        let retired = retire_cold_experiments(
            &[row("e", &uri, now)],
            Duration::from_secs(7 * 86_400),
            now,
            RolloutStoreOptions::default(),
        )
        .await;
        assert!(
            retired.is_empty(),
            "a hot experiment must stay in the table"
        );
    }

    /// Retirement disabled means nothing is ever retired, however old.
    #[tokio::test]
    async fn zero_window_disables_retirement() {
        let now = Utc::now().timestamp_millis();
        let retired = retire_cold_experiments(
            &[row("e", "/nonexistent", 0)],
            Duration::from_secs(0),
            now,
            RolloutStoreOptions::default(),
        )
        .await;
        assert!(retired.is_empty());
    }

    /// An experiment that cannot be prepared stays in the table rather than
    /// being dropped. Refusing to retire is always safe; retiring early is not.
    #[tokio::test]
    async fn unpreparable_experiment_is_kept() {
        let now = Utc::now().timestamp_millis();
        let old = now - Duration::from_secs(30 * 86_400).as_millis() as i64;
        let retired = retire_cold_experiments(
            &[row("gone", "/no/such/dataset.lance", old)],
            Duration::from_secs(7 * 86_400),
            now,
            RolloutStoreOptions::default(),
        )
        .await;
        assert!(
            retired.is_empty(),
            "an experiment that failed to prepare must not be retired"
        );
    }
    /// A cold observation must not write a stats row.
    ///
    /// Reading about a retired experiment (search, or opening its detail page)
    /// must not make it hot again -- otherwise browsing the UI would silently
    /// undo retirement and the table would creep back toward holding every
    /// experiment that ever existed, which is the state retirement exists to
    /// prevent.
    #[tokio::test]
    async fn observe_cold_reports_without_rehydrating() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("e.lance").to_string_lossy().to_string();
        {
            let store = RolloutStore::open_with_options(&uri, RolloutStoreOptions::default())
                .await
                .unwrap();
            store.add(&[rec("a")]).await.unwrap();
            store.flush().await.unwrap();
        }

        let summary = observe_cold_with_options("e", &uri, RolloutStoreOptions::default())
            .await
            .unwrap();
        assert_eq!(summary.name, "e");
        assert_eq!(
            summary.row_count, 1,
            "a cold read still reports real counts"
        );

        let second = observe_cold_with_options("e", &uri, RolloutStoreOptions::default())
            .await
            .unwrap();
        assert_eq!(second.row_count, summary.row_count);
    }

    /// A missing dataset surfaces as an error rather than a panic, so a search
    /// hit on a registry entry whose data is gone degrades to one skipped row.
    #[tokio::test]
    async fn observe_cold_errors_on_missing_dataset() {
        assert!(observe_cold_with_options(
            "gone",
            "/no/such/dataset.lance",
            RolloutStoreOptions::default(),
        )
        .await
        .is_err());
    }
}
