//! Unified task scheduler.
//!
//! Each master polls one durable queue with **bounded concurrency**. The queue
//! lives in etcd, which uses atomic lease-backed claims so multiple stateless
//! masters can drain the same queue. It executes three kinds of task
//! ([`TaskKind`]):
//!
//! - **Compact** — rewrites an experiment's base-table fragments. Two `Rewrite`s
//!   on the *same* dataset conflict in Lance's conflict matrix, so compaction of
//!   one experiment is serialized against itself (and against `IndexId`) via a
//!   per-name task-store lock. Distinct experiments compact concurrently.
//!   `Rewrite` vs the data-plane's `Append` is non-conflicting, so this runs
//!   safely alongside live ingest.
//! - **MergeWal** — folds flushed MemWAL generations back into the base table.
//!   The master cannot do this itself without fencing the live shard writer, so
//!   it fans out to every configured worker endpoint and each worker merges its
//!   own shard (`POST /api/v1/internal/merge-wal/{name}`).
//! - **IndexId** — builds a ZoneMap scalar index on the base table's `id` column
//!   (runs locally on the master). It commits a `CreateIndex`, which can conflict
//!   with a concurrent `Compact` `Rewrite` on the same dataset, so it shares the
//!   per-name task-store lock with `Compact`.
//!
//! Each task runs in its own `tokio::spawn`, so one task's failure never affects
//! another. A global [`Semaphore`] bounds how many run at once.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use lance_context_api::{TaskKind, TaskRecord};
use lance_context_core::{CompactionConfig, RolloutStore, RolloutStoreOptions};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

use crate::state::MasterState;
use crate::stats_store::StatRow;
use crate::task_store::TaskClaim;

/// Maximum tasks a single auto-sweep may enqueue.
///
/// Previously unbounded: a sweep read the whole stats table and enqueued one
/// task per row over the threshold, so at tens of thousands of experiments a
/// single tick could flood the queue. Capping keeps each tick's work bounded;
/// anything still over the threshold is picked up by the next tick.
const MAX_SWEEP_ENQUEUE: usize = 256;

/// Build the [`CompactionConfig`] the scheduler applies, from master config.
pub fn compaction_config(state: &MasterState) -> CompactionConfig {
    CompactionConfig {
        enabled: true,
        min_fragments: state.config.min_fragments,
        target_rows_per_fragment: state.config.target_rows_per_fragment,
        ..Default::default()
    }
}

fn kind_label(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Compact => "compact",
        TaskKind::MergeWal => "merge_wal",
        TaskKind::IndexId => "index_id",
    }
}

/// Enqueue a task and return its record. For [`TaskKind::Compact`],
/// [`TaskKind::IndexId`], and depless [`TaskKind::MergeWal`] this de-dupes
/// against an existing non-terminal task for the same target: if one is already
/// `Queued` or `Running`, its record is returned unchanged and nothing new is
/// enqueued. A `MergeWal` that is part of a dependency chain (non-empty
/// `depends_on`) is not de-duped.
pub async fn enqueue(
    state: &Arc<MasterState>,
    kind: TaskKind,
    target: &str,
) -> lance::Result<TaskRecord> {
    enqueue_with_deps(state, kind, target, Vec::new()).await
}

/// Like [`enqueue`] but the task waits for `depends_on` (task ids) to reach
/// `Done` before it runs. De-dup is skipped when dependencies are present: a
/// dependent task is part of an ordered chain and must not collapse into an
/// unrelated in-flight task for the same target.
pub async fn enqueue_with_deps(
    state: &Arc<MasterState>,
    kind: TaskKind,
    target: &str,
    depends_on: Vec<String>,
) -> lance::Result<TaskRecord> {
    let record = state.task_store.enqueue(kind, target, depends_on).await?;
    metrics::counter!("master_task_enqueued_total", "kind" => kind_label(kind)).increment(1);
    Ok(record)
}

/// Time spent getting a task to the point where its work can start.
#[derive(Debug, Clone, Copy, Default)]
struct TaskClaimTiming {
    /// The etcd claim transaction (queued→running, lease, target lock).
    claim: std::time::Duration,
    /// Waiting for a concurrency permit once the task was already claimed.
    permit_wait: std::time::Duration,
}

/// Execute one claimed task and atomically publish its terminal state.
///
/// `timing` carries how long the dispatch loop spent claiming this task and
/// waiting for a concurrency permit, so every phase of the task's life lands on
/// one metric rather than only the work window.
async fn run_task(state: &Arc<MasterState>, claim: TaskClaim, timing: TaskClaimTiming) {
    let task = claim.task.clone();
    let kind = kind_label(task.kind);

    metrics::histogram!("master_task_phase_duration_seconds", "kind" => kind, "phase" => "claim")
        .record(timing.claim.as_secs_f64());
    metrics::histogram!(
        "master_task_phase_duration_seconds",
        "kind" => kind,
        "phase" => "permit_wait",
    )
    .record(timing.permit_wait.as_secs_f64());

    let started = std::time::Instant::now();
    let outcome = match task.kind {
        TaskKind::Compact => run_compaction(state, &task.target).await,
        TaskKind::MergeWal => run_merge_wal(state, &task.target).await,
        TaskKind::IndexId => run_index_id(state, &task.target).await,
    };
    let work_elapsed = started.elapsed();
    let result = if outcome.is_ok() { "success" } else { "failed" };

    metrics::histogram!("master_task_phase_duration_seconds", "kind" => kind, "phase" => "work")
        .record(work_elapsed.as_secs_f64());
    // Same scope as the `work` phase above, kept for back-compat with existing
    // dashboards. Success/failure is carried by `master_tasks_total{result}`, a
    // counter — putting `result` on the histogram would double its series count
    // (every bucket, twice) to describe the latency of a rare event.
    metrics::histogram!("master_task_duration_seconds", "kind" => kind)
        .record(work_elapsed.as_secs_f64());
    metrics::counter!("master_tasks_total", "kind" => kind, "result" => result).increment(1);

    if let Err(error) = &outcome {
        tracing::warn!(task = %task.id, target = %task.target, error, "task failed");
    }
    let commit_start = std::time::Instant::now();
    let finished = state.task_store.finish(claim, outcome).await;
    metrics::histogram!("master_task_phase_duration_seconds", "kind" => kind, "phase" => "commit")
        .record(commit_start.elapsed().as_secs_f64());
    if let Err(error) = finished {
        tracing::error!(task = %task.id, error = %error, "failed to persist task completion");
    }
}

/// Compact one experiment. The task-store claim owns the per-experiment write
/// lock for the full execution.
async fn run_compaction(state: &Arc<MasterState>, name: &str) -> Result<String, String> {
    compact_inner(state, name).await
}

/// Build a ZoneMap scalar index on one experiment's `id` column. Shares the
/// per-name base-table write gate with [`run_compaction`] so an `IndexId` and a
/// `Compact` for the same experiment never commit concurrently (`CreateIndex`
/// vs `Rewrite` can conflict). Distinct experiments index concurrently.
async fn run_index_id(state: &Arc<MasterState>, name: &str) -> Result<String, String> {
    index_id_inner(state, name).await
}

async fn index_id_inner(state: &Arc<MasterState>, name: &str) -> Result<String, String> {
    let uri = state.rollout_uri(name);
    let opts = RolloutStoreOptions::default();
    let mut store = RolloutStore::open_existing_with_options(&uri, opts)
        .await
        .map_err(|e| e.to_string())?;
    store
        .create_id_zonemap_index()
        .await
        .map_err(|e| e.to_string())?;
    Ok("built zonemap index on id".to_string())
}

async fn compact_inner(state: &Arc<MasterState>, name: &str) -> Result<String, String> {
    let uri = state.rollout_uri(name);
    let config = compaction_config(state);
    let opts = RolloutStoreOptions::default();

    let mut store = RolloutStore::open_existing_with_options(&uri, opts)
        .await
        .map_err(|e| e.to_string())?;
    let metrics = store
        .compact(Some(config))
        .await
        .map_err(|e| e.to_string())?;
    update_stats_after_compaction(state, name, &store).await;
    Ok(format!(
        "removed {} / added {} fragments",
        metrics.fragments_removed, metrics.fragments_added
    ))
}

/// Shape of the worker's merge-wal response (`{ "reclaimed": n }`).
#[derive(serde::Deserialize)]
struct MergeWalReply {
    reclaimed: usize,
}

/// Fan a WAL-merge out to every configured worker endpoint. Each worker merges
/// its own shard; a worker that owns no data for `name` reports 0 (or 404, which
/// we tolerate). Succeeds if at least one endpoint responded; fails only when
/// there are no endpoints or every one errored.
async fn run_merge_wal(state: &Arc<MasterState>, name: &str) -> Result<String, String> {
    let endpoints = &state.config.worker_endpoints;
    if endpoints.is_empty() {
        return Err("no worker endpoints configured (--worker-endpoints)".to_string());
    }

    let calls = endpoints.iter().map(|ep| {
        let http = state.http.clone();
        let url = format!(
            "{}/api/v1/internal/merge-wal/{}",
            ep.trim_end_matches('/'),
            name
        );
        async move {
            // Per-worker timing: `join_all` means the slowest worker sets the
            // whole task's latency, so without this one straggler is
            // indistinguishable from every worker being slow. Unlabelled --
            // outcome is carried by the counter below, which costs one series
            // per value instead of one per bucket per value.
            let started = std::time::Instant::now();
            let outcome = merge_wal_one(&http, &url).await;
            let result = match &outcome {
                Ok(WorkerMerge::Reclaimed(_)) => "ok",
                Ok(WorkerMerge::NotFound) => "not_found",
                Err(WorkerMergeError::Http(_)) => "http_error",
                Err(WorkerMergeError::Transport(_)) => "transport_error",
            };
            metrics::histogram!("master_merge_wal_worker_duration_seconds")
                .record(started.elapsed().as_secs_f64());
            // Counted per worker per attempt: a 404 is tolerated as "owns no
            // shard" and N-1 failures still report task success, so this counter
            // is the only place partial failure is visible at all.
            metrics::counter!("master_merge_wal_workers_total", "result" => result).increment(1);
            outcome
        }
    });

    let results = futures::future::join_all(calls).await;
    let total_workers = results.len();
    let mut reclaimed = 0usize;
    let mut ok_workers = 0usize;
    let mut last_err = None;
    for r in results {
        match r {
            Ok(WorkerMerge::Reclaimed(n)) => {
                reclaimed += n;
                ok_workers += 1;
            }
            Ok(WorkerMerge::NotFound) => {
                ok_workers += 1;
            }
            Err(e) => last_err = Some(e.to_string()),
        }
    }

    metrics::counter!("master_merge_wal_generations_reclaimed_total").increment(reclaimed as u64);

    if ok_workers == 0 {
        return Err(last_err.unwrap_or_else(|| "all workers failed".to_string()));
    }
    Ok(format!(
        "merged {reclaimed} generations across {ok_workers}/{total_workers} workers"
    ))
}

/// One worker's response to a WAL-merge fan-out.
enum WorkerMerge {
    Reclaimed(usize),
    /// The worker owns no shard for this experiment; tolerated as success.
    NotFound,
}

enum WorkerMergeError {
    /// A non-success HTTP status.
    Http(String),
    /// Connection/timeout/decode failure.
    Transport(String),
}

impl std::fmt::Display for WorkerMergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(m) | Self::Transport(m) => f.write_str(m),
        }
    }
}

/// Issue the merge call to one worker, classifying the failure mode so the
/// caller can label its metrics.
async fn merge_wal_one(http: &reqwest::Client, url: &str) -> Result<WorkerMerge, WorkerMergeError> {
    let resp = http
        .post(url)
        .send()
        .await
        .map_err(|e| WorkerMergeError::Transport(e.to_string()))?;
    let status = resp.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(WorkerMerge::NotFound);
    }
    if !status.is_success() {
        return Err(WorkerMergeError::Http(format!("{url}: HTTP {status}")));
    }
    let body: MergeWalReply = resp
        .json()
        .await
        .map_err(|e| WorkerMergeError::Transport(e.to_string()))?;
    Ok(WorkerMerge::Reclaimed(body.reclaimed))
}

/// Refresh the stats row for `name` after a successful compaction: re-observe
/// fragment/row counts and bump `last_compaction`/`total_compactions`.
async fn update_stats_after_compaction(state: &Arc<MasterState>, name: &str, store: &RolloutStore) {
    let guard = match state.task_store.coordination_lock("stats-writer").await {
        Ok(guard) => guard,
        Err(e) => {
            tracing::warn!(store = %name, error = %e, "stats writer lock failed");
            return;
        }
    };
    let obs = match store.observe().await {
        Ok(obs) => obs,
        Err(e) => {
            tracing::warn!(store = %name, error = %e, "post-compaction observe failed");
            let _ = state.task_store.release_coordination_lock(guard).await;
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
        version: obs.version as i64,
    };
    if let Err(e) = stats.upsert(&row).await {
        tracing::warn!(store = %name, error = %e, "post-compaction stats upsert failed");
    }
    drop(stats);
    if let Err(e) = state.task_store.release_coordination_lock(guard).await {
        tracing::warn!(store = %name, error = %e, "stats writer unlock failed");
    }
}

/// Enqueue a `Compact` task for every experiment whose fragment count is at or
/// above the configured threshold, honoring quiet hours. Reads candidates from
/// the stats table.
pub async fn sweep_candidates(state: &Arc<MasterState>) -> lance::Result<usize> {
    let Some(guard) = state
        .task_store
        .try_coordination_lock("compaction-sweep")
        .await?
    else {
        return Ok(0);
    };
    let result = sweep_candidates_inner(state).await;
    let release = state.task_store.release_coordination_lock(guard).await;
    match (result, release) {
        (Ok(count), Ok(())) => Ok(count),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

async fn sweep_candidates_inner(state: &Arc<MasterState>) -> lance::Result<usize> {
    let config = compaction_config(state);
    // Quiet-hours gate applies to the whole sweep.
    if in_quiet_hours(&config) {
        return Ok(0);
    }
    // Threshold pushed into the scan and capped, rather than reading the whole
    // stats table and filtering in this loop. At tens of thousands of
    // experiments the full read dominated every sweep, and the enqueue count
    // was unbounded.
    let rows = state
        .stats
        .lock()
        .await
        .list_above_fragment_count(config.min_fragments, MAX_SWEEP_ENQUEUE)
        .await?;
    let mut queued = 0;
    for row in rows {
        enqueue(state, TaskKind::Compact, &row.name).await?;
        queued += 1;
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

/// Enqueue a `MergeWal` task for every experiment whose pending MemWAL
/// generation count is at or above the configured threshold, reading candidates
/// from the stats table. Coordinated across master replicas by a dedicated
/// task-store lock so only one replica sweeps at a time. Depless `MergeWal`
/// enqueues de-dupe, so a still-running fan-out is not re-queued.
pub async fn sweep_merge_wal_candidates(state: &Arc<MasterState>) -> lance::Result<usize> {
    let Some(guard) = state
        .task_store
        .try_coordination_lock("merge-wal-sweep")
        .await?
    else {
        return Ok(0);
    };
    let result = sweep_merge_wal_inner(state).await;
    let release = state.task_store.release_coordination_lock(guard).await;
    match (result, release) {
        (Ok(count), Ok(())) => Ok(count),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

async fn sweep_merge_wal_inner(state: &Arc<MasterState>) -> lance::Result<usize> {
    let threshold = state.config.merge_wal_min_generations;
    // See `sweep_candidates_inner`: predicate pushed down, enqueue capped.
    let rows = state
        .stats
        .lock()
        .await
        .list_above_pending_wal(threshold, MAX_SWEEP_ENQUEUE)
        .await?;
    let mut queued = 0;
    for row in rows {
        enqueue(state, TaskKind::MergeWal, &row.name).await?;
        queued += 1;
    }
    Ok(queued)
}

/// Spawn the scheduler poller plus the optional periodic auto-sweep.
pub fn spawn_scheduler(state: &Arc<MasterState>) -> JoinHandle<()> {
    // Optional periodic compaction auto-sweep feeds the same queue.
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

    // Optional periodic WAL-merge auto-sweep, independent of compaction.
    let merge_interval_secs = state.config.merge_wal_interval_secs;
    if merge_interval_secs > 0 {
        let sweep_state = state.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(merge_interval_secs));
            ticker.tick().await; // skip immediate tick
            loop {
                ticker.tick().await;
                match sweep_merge_wal_candidates(&sweep_state).await {
                    Ok(n) if n > 0 => {
                        tracing::info!(queued = n, "auto merge-wal sweep queued experiments")
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!(error = %e, "auto merge-wal sweep failed"),
                }
            }
        });
    }

    let concurrency = state.config.task_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(concurrency));
    let dispatch_state = state.clone();
    tokio::spawn(async move {
        loop {
            if let Ok(queued) = dispatch_state.task_store.queue_depth().await {
                metrics::gauge!("master_task_queue_depth").set(queued as f64);
            }
            while sem.available_permits() > 0 {
                let claim_start = std::time::Instant::now();
                match dispatch_state.task_store.claim_next().await {
                    Ok(Some(claim)) => {
                        let claim_elapsed = claim_start.elapsed();
                        // The task is already claimed at this point (queue key
                        // deleted, lease granted, target lock held), so time
                        // spent here is a claimed-but-idle task holding its
                        // per-experiment lock — worth seeing separately.
                        let permit_start = std::time::Instant::now();
                        let permit = sem
                            .clone()
                            .acquire_owned()
                            .await
                            .expect("semaphore never closed");
                        let timing = TaskClaimTiming {
                            claim: claim_elapsed,
                            permit_wait: permit_start.elapsed(),
                        };
                        let st = dispatch_state.clone();
                        tokio::spawn(async move {
                            run_task(&st, claim, timing).await;
                            drop(permit);
                        });
                    }
                    Ok(None) => break,
                    Err(error) => {
                        tracing::warn!(error = %error, "scheduler queue poll failed");
                        break;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MasterConfig;
    use lance_context_api::TaskState;
    use lance_context_core::generate_id;
    use tempfile::TempDir;

    fn config(dir: &TempDir) -> MasterConfig {
        MasterConfig {
            data_dir: dir.path().to_string_lossy().to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            stats_scan_interval_secs: 0,
            scan_concurrency: 4,
            stats_maintenance_every_n_scans: 0,
            stats_history_ttl_secs: 3_600,
            stats_cold_retire_secs: 0,
            compaction_interval_secs: 0,
            // Low threshold so a handful of appends crosses it.
            min_fragments: 2,
            target_rows_per_fragment: 1_048_576,
            merge_wal_interval_secs: 0,
            merge_wal_min_generations: 2,
            worker_endpoints: vec![],
            task_concurrency: 4,
            etcd_endpoints: std::env::var("ETCD_TEST_ENDPOINTS")
                .map(|value| value.split(',').map(str::to_string).collect())
                .unwrap_or_default(),
            etcd_prefix: format!("/lance-context/test/{}", generate_id()),
            etcd_username: None,
            etcd_password: None,
            etcd_ca_cert: None,
            etcd_client_cert: None,
            etcd_client_key: None,
            etcd_lease_ttl_secs: 5,
            task_history_limit: 1_000,
            task_history_ttl_secs: 86_400,
            ui_dir: None,
        }
    }

    /// Wait until the task reaches a terminal state, returning its final record.
    async fn await_terminal(state: &Arc<MasterState>, id: &str) -> TaskRecord {
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if let Some(t) = state.task_store.get(id).await.unwrap() {
                if matches!(t.state, TaskState::Done | TaskState::Failed) {
                    return t;
                }
            }
        }
        panic!("task {id} did not reach a terminal state");
    }

    /// Manual enqueue -> dispatcher compacts -> task reaches Done and the stats
    /// table records a compaction.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
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
                let rec = rollout_record(&format!("r{i}"));
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

        let rec = enqueue(&state, TaskKind::Compact, name).await.unwrap();
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Done, "got {status:?}");
        assert_eq!(
            state
                .task_store
                .list()
                .await
                .unwrap()
                .into_iter()
                .find(|task| task.id == rec.id)
                .unwrap()
                .state,
            TaskState::Done
        );

        let row = state.stats.lock().await.get(name).await.unwrap().unwrap();
        assert_eq!(row.total_compactions, 1);
        assert!(row.last_compaction >= 0);

        worker.abort();
    }

    /// Manual enqueue of an `IndexId` task -> dispatcher builds the ZoneMap
    /// index -> task reaches Done with the expected detail summary.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn index_id_task_builds_index_and_reaches_done() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        let worker = spawn_scheduler(&state);

        let name = "exp";
        let uri = state.rollout_uri(name);
        {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            for i in 0..3 {
                let rec = rollout_record(&format!("r{i}"));
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

        let rec = enqueue(&state, TaskKind::IndexId, name).await.unwrap();
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Done, "got {status:?}");
        assert_eq!(status.detail.as_deref(), Some("built zonemap index on id"));

        worker.abort();
    }

    /// Enqueuing the same experiment twice while queued de-dupes to one task.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn enqueue_dedupes_queued_compactions() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        // Do NOT spawn the dispatcher, so the first task stays Queued.
        let a = enqueue(&state, TaskKind::Compact, "x").await.unwrap();
        let b = enqueue(&state, TaskKind::Compact, "x").await.unwrap();
        assert_eq!(a.id, b.id, "second enqueue returns the same task");
        assert_eq!(state.task_store.list().await.unwrap().len(), 1);
    }

    /// A MergeWal task with no configured endpoints fails fast with a clear msg.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn merge_wal_without_endpoints_fails() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        let worker = spawn_scheduler(&state);
        let rec = enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Failed);
        assert!(status.error.unwrap().contains("worker endpoints"));
        worker.abort();
    }

    /// MergeWal fans out to every configured worker endpoint and sums the
    /// reclaimed counts. Uses a tiny in-process stub server per "worker".
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn merge_wal_broadcasts_and_sums_reclaimed() {
        use axum::{routing::post, Json, Router};

        // A stub worker that always reports `reclaimed` for any merge call.
        async fn spawn_stub(reclaimed: usize) -> String {
            let app = Router::new().route(
                "/api/v1/internal/merge-wal/{name}",
                post(move || async move { Json(serde_json::json!({ "reclaimed": reclaimed })) }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            format!("http://{addr}")
        }

        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.worker_endpoints = vec![spawn_stub(3).await, spawn_stub(2).await];
        let state = MasterState::new(cfg).await.unwrap();
        let worker = spawn_scheduler(&state);

        let rec = enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Done, "got {status:?}");
        let detail = status.detail.unwrap();
        assert!(detail.contains("merged 5 generations"), "detail: {detail}");
        assert!(detail.contains("2/2 workers"), "detail: {detail}");
        worker.abort();
    }

    /// A dependent task runs only after its dependency reaches `Done`: an
    /// `index_id` depending on a `compact` must start after compaction finishes,
    /// so the two never contend for the shared per-experiment base-table gate.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn dependent_task_runs_after_dependency_done() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        let worker = spawn_scheduler(&state);

        let name = "exp";
        let uri = state.rollout_uri(name);
        {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            for i in 0..4 {
                let rec = rollout_record(&format!("r{i}"));
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
        crate::scanner::scan_once(&state).await.unwrap();

        let compact = enqueue(&state, TaskKind::Compact, name).await.unwrap();
        let index = enqueue_with_deps(&state, TaskKind::IndexId, name, vec![compact.id.clone()])
            .await
            .unwrap();

        let compact_final = await_terminal(&state, &compact.id).await;
        assert_eq!(
            compact_final.state,
            TaskState::Done,
            "got {compact_final:?}"
        );
        let index_final = await_terminal(&state, &index.id).await;
        assert_eq!(index_final.state, TaskState::Done, "got {index_final:?}");
        // The dependent could not have started before the dependency finished.
        assert!(
            index_final.started_at.unwrap() >= compact_final.finished_at.unwrap(),
            "index started {:?} before compact finished {:?}",
            index_final.started_at,
            compact_final.finished_at
        );

        worker.abort();
    }

    /// A dependent whose dependency `Failed` is skipped (marked `Failed`) rather
    /// than run.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn dependent_skipped_when_dependency_fails() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        let worker = spawn_scheduler(&state);

        // MergeWal with no worker endpoints fails; its dependent must be skipped.
        let merge = enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        let dependent = enqueue_with_deps(&state, TaskKind::Compact, "exp", vec![merge.id.clone()])
            .await
            .unwrap();

        let merge_final = await_terminal(&state, &merge.id).await;
        assert_eq!(merge_final.state, TaskState::Failed);
        let dep_final = await_terminal(&state, &dependent.id).await;
        assert_eq!(dep_final.state, TaskState::Failed, "got {dep_final:?}");
        assert!(dep_final.error.unwrap().contains("dependency"));

        worker.abort();
    }

    /// The WAL-merge sweep enqueues a `MergeWal` only for experiments whose
    /// pending generation count is at or above the threshold, and de-dupes so a
    /// second sweep does not pile up a duplicate for the same target.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn sweep_merge_wal_enqueues_over_threshold_and_dedupes() {
        use crate::stats_store::StatRow;

        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.merge_wal_min_generations = 3;
        let state = MasterState::new(cfg).await.unwrap();

        let seed = |name: &str, pending: i64| StatRow {
            version: StatRow::UNKNOWN_VERSION,
            name: name.to_string(),
            uri: state.rollout_uri(name),
            row_count: 0,
            fragment_count: 0,
            last_updated: 0,
            pending_wal_generations: pending,
            last_compaction: StatRow::NO_COMPACTION,
            total_compactions: 0,
            scanned_at: 0,
        };
        {
            let mut stats = state.stats.lock().await;
            stats.upsert(&seed("hot", 5)).await.unwrap(); // >= threshold
            stats.upsert(&seed("cold", 1)).await.unwrap(); // < threshold
        }

        let queued = sweep_merge_wal_candidates(&state).await.unwrap();
        assert_eq!(queued, 1, "only the over-threshold experiment is swept");

        let tasks = state.task_store.list().await.unwrap();
        let merge_tasks: Vec<_> = tasks
            .iter()
            .filter(|t| t.kind == TaskKind::MergeWal)
            .collect();
        assert_eq!(merge_tasks.len(), 1);
        assert_eq!(merge_tasks[0].target, "hot");

        // Second sweep must de-dupe against the still-queued MergeWal.
        sweep_merge_wal_candidates(&state).await.unwrap();
        let merge_after = state
            .task_store
            .list()
            .await
            .unwrap()
            .into_iter()
            .filter(|t| t.kind == TaskKind::MergeWal)
            .count();
        assert_eq!(merge_after, 1, "duplicate MergeWal is de-duped");
    }

    /// Minimal rollout record builder for tests (the core struct has no    /// `Default`).
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
}
