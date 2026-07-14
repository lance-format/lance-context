//! Centralized compaction scheduler.
//!
//! Compaction rewrites fragments (`Rewrite` in Lance's conflict matrix). Two
//! concurrent `Rewrite`s on the same dataset conflict, so compaction must have a
//! **single serial driver**. This module is that driver: one background task
//! drains a single queue, so automatic sweeps and manual API triggers share the
//! same serial execution path and can never overlap. `Rewrite` vs the
//! data-plane's `Append` is non-conflicting, so this runs safely alongside live
//! ingest.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use lance_context_api::CompactJobStatus;
use lance_context_core::{CompactionConfig, RolloutStore, RolloutStoreOptions};
use tokio::task::JoinHandle;

use crate::state::MasterState;
use crate::stats_store::StatRow;

/// Build the [`CompactionConfig`] the scheduler applies, from master config.
pub fn compaction_config(state: &MasterState) -> CompactionConfig {
    CompactionConfig {
        enabled: true,
        min_fragments: state.config.min_fragments,
        target_rows_per_fragment: state.config.target_rows_per_fragment,
        ..Default::default()
    }
}

/// Enqueue an experiment for compaction and mark it `Queued`. De-dupes: if a
/// job for `name` is already queued or running, this is a no-op returning the
/// existing status.
pub async fn enqueue(state: &Arc<MasterState>, name: &str) -> CompactJobStatus {
    {
        let jobs = state.jobs.lock().await;
        if matches!(
            jobs.get(name),
            Some(CompactJobStatus::Queued) | Some(CompactJobStatus::Running)
        ) {
            return jobs.get(name).cloned().unwrap();
        }
    }
    state
        .jobs
        .lock()
        .await
        .insert(name.to_string(), CompactJobStatus::Queued);
    // Unbounded send only fails if the receiver was dropped (scheduler gone).
    let _ = state.compact_tx.send(name.to_string());
    metrics::counter!("master_compaction_enqueued_total").increment(1);
    metrics::gauge!("master_compaction_queue_depth").increment(1.0);
    CompactJobStatus::Queued
}

/// Compact one experiment now (serial, on the scheduler task). Updates the job
/// status map and the stats table's compaction counters on success.
async fn run_compaction(state: &Arc<MasterState>, name: &str) {
    // This job is leaving the queue and entering execution.
    metrics::gauge!("master_compaction_queue_depth").decrement(1.0);
    state
        .jobs
        .lock()
        .await
        .insert(name.to_string(), CompactJobStatus::Running);

    let uri = state.rollout_uri(name);
    let config = compaction_config(state);
    let opts = RolloutStoreOptions::default();

    let compact_start = std::time::Instant::now();
    let status = match RolloutStore::open_existing_with_options(&uri, opts).await {
        Ok(mut store) => match store.compact(Some(config)).await {
            Ok(metrics) => {
                update_stats_after_compaction(state, name, &store).await;
                CompactJobStatus::Done {
                    fragments_removed: metrics.fragments_removed,
                    fragments_added: metrics.fragments_added,
                }
            }
            Err(e) => CompactJobStatus::Failed {
                error: e.to_string(),
            },
        },
        Err(e) => CompactJobStatus::Failed {
            error: e.to_string(),
        },
    };

    metrics::histogram!("master_compaction_duration_seconds")
        .record(compact_start.elapsed().as_secs_f64());
    let result = if matches!(status, CompactJobStatus::Done { .. }) {
        "success"
    } else {
        "failed"
    };
    metrics::counter!("master_compactions_total", "result" => result).increment(1);

    if let CompactJobStatus::Failed { error } = &status {
        tracing::warn!(store = %name, error = %error, "compaction failed");
    }
    state.jobs.lock().await.insert(name.to_string(), status);
}

/// Refresh the stats row for `name` after a successful compaction: re-observe
/// fragment/row counts and bump `last_compaction`/`total_compactions`.
async fn update_stats_after_compaction(state: &Arc<MasterState>, name: &str, store: &RolloutStore) {
    let obs = match store.observe().await {
        Ok(obs) => obs,
        Err(e) => {
            tracing::warn!(store = %name, error = %e, "post-compaction observe failed");
            return;
        }
    };
    let mut stats = state.stats.lock().await;
    let prev_total = match stats.get(name).await {
        Ok(Some(row)) => row.total_compactions,
        _ => 0,
    };
    let row = StatRow {
        name: name.to_string(),
        uri: state.rollout_uri(name),
        row_count: obs.row_count,
        fragment_count: obs.fragment_count,
        last_updated: obs.last_updated,
        pending_wal_generations: obs.pending_wal_generations,
        last_compaction: Utc::now().timestamp_millis(),
        total_compactions: prev_total + 1,
        scanned_at: Utc::now().timestamp_millis(),
    };
    if let Err(e) = stats.upsert(&row).await {
        tracing::warn!(store = %name, error = %e, "post-compaction stats upsert failed");
    }
}

/// Enqueue every experiment whose fragment count is at or above the configured
/// threshold, honoring quiet hours. Reads candidates from the stats table.
pub async fn sweep_candidates(state: &Arc<MasterState>) -> lance::Result<usize> {
    let config = compaction_config(state);
    // Quiet-hours gate applies to the whole sweep.
    if in_quiet_hours(&config) {
        return Ok(0);
    }
    let rows = state.stats.lock().await.list(None, usize::MAX, 0).await?;
    let mut queued = 0;
    for row in rows {
        if row.fragment_count as usize >= config.min_fragments {
            enqueue(state, &row.name).await;
            queued += 1;
        }
    }
    Ok(queued)
}

fn in_quiet_hours(config: &CompactionConfig) -> bool {
    if config.quiet_hours.is_empty() {
        return false;
    }
    use chrono::Timelike;
    let hour = Utc::now().hour() as u8;
    config
        .quiet_hours
        .iter()
        .any(|(start, end)| hour >= *start && hour < *end)
}

/// Spawn the single serial compaction worker plus (optionally) the periodic
/// auto-sweep. The worker owns the queue receiver and processes one experiment
/// at a time. Returns the worker handle. Panics if called twice (receiver
/// already taken).
pub fn spawn_scheduler(state: &Arc<MasterState>) -> JoinHandle<()> {
    let mut rx = state
        .compact_rx
        .try_lock()
        .expect("scheduler: state lock")
        .take()
        .expect("spawn_scheduler called more than once");

    // Optional periodic auto-sweep feeds the same queue.
    let interval_secs = state.config.compaction_interval_secs;
    if interval_secs > 0 {
        let sweep_state = state.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
            ticker.tick().await; // skip immediate tick
            loop {
                ticker.tick().await;
                match sweep_candidates(&sweep_state).await {
                    Ok(n) if n > 0 => tracing::info!(queued = n, "auto-sweep queued experiments"),
                    Ok(_) => {}
                    Err(e) => tracing::warn!(error = %e, "auto-sweep failed"),
                }
            }
        });
    }

    let worker_state = state.clone();
    tokio::spawn(async move {
        while let Some(name) = rx.recv().await {
            run_compaction(&worker_state, &name).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MasterConfig;
    use tempfile::TempDir;

    fn config(dir: &TempDir) -> MasterConfig {
        MasterConfig {
            data_dir: dir.path().to_string_lossy().to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            stats_scan_interval_secs: 0,
            scan_concurrency: 4,
            compaction_interval_secs: 0,
            // Low threshold so a handful of appends crosses it.
            min_fragments: 2,
            target_rows_per_fragment: 1_048_576,
            ui_dir: None,
        }
    }

    /// Manual enqueue -> serial worker compacts -> job reaches Done and the
    /// stats table records a compaction.
    #[tokio::test]
    async fn manual_compaction_runs_and_updates_stats() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        let worker = spawn_scheduler(&state);

        // Build a store with several fragments via repeated base-table appends.
        let name = "exp";
        let uri = state.rollout_uri(name);
        {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            for i in 0..4 {
                let rec = crate::scheduler::tests::rollout_record(&format!("r{i}"));
                store.add(&[rec]).await.unwrap();
                store.cleanup_own_shard().await.unwrap();
            }
        }
        state
            .registry
            .write()
            .await
            .upsert(name, &uri)
            .await
            .unwrap();
        // Seed a stats row so post-compaction upsert has a prior counter.
        crate::scanner::scan_once(&state).await.unwrap();

        enqueue(&state, name).await;

        // Wait for the worker to reach a terminal state.
        let mut done = None;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if let Some(s) = state.jobs.lock().await.get(name) {
                if matches!(
                    s,
                    CompactJobStatus::Done { .. } | CompactJobStatus::Failed { .. }
                ) {
                    done = Some(s.clone());
                    break;
                }
            }
        }
        let status = done.expect("job reached terminal state");
        assert!(
            matches!(status, CompactJobStatus::Done { .. }),
            "expected Done, got {status:?}"
        );

        let row = state.stats.lock().await.get(name).await.unwrap().unwrap();
        assert_eq!(row.total_compactions, 1);
        assert!(row.last_compaction >= 0);

        worker.abort();
    }

    #[tokio::test]
    async fn enqueue_dedupes_queued_jobs() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        // Do NOT spawn the worker, so jobs stay Queued.
        let s1 = enqueue(&state, "x").await;
        let s2 = enqueue(&state, "x").await;
        assert!(matches!(s1, CompactJobStatus::Queued));
        assert!(matches!(s2, CompactJobStatus::Queued));
        // Only one message should be in the channel; drain and count.
        let mut rx = state.compact_rx.lock().await.take().unwrap();
        let mut n = 0;
        while rx.try_recv().is_ok() {
            n += 1;
        }
        assert_eq!(n, 1);
    }

    /// Minimal rollout record builder for tests (the core struct has no
    /// `Default`).
    pub fn rollout_record(id: &str) -> lance_context_core::RolloutRecord {
        use chrono::TimeZone;
        lance_context_core::RolloutRecord {
            id: id.to_string(),
            rollout_id: "rollout-1".to_string(),
            problem_id: "problem-1".to_string(),
            dataset: None,
            sequence_order: 0,
            role: lance_context_core::ROLE_ASSISTANT.to_string(),
            created_at: Utc.timestamp_micros(1_700_000_000_000_000).unwrap(),
            content: Some("x".to_string()),
            content_type: "text/plain".to_string(),
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
}
