//! Unified task scheduler.
//!
//! The master runs a single scheduler that drains one queue with **bounded
//! concurrency**. It executes three kinds of task ([`TaskKind`]):
//!
//! - **Compact** — rewrites an experiment's base-table fragments. Two `Rewrite`s
//!   on the *same* dataset conflict in Lance's conflict matrix, so compaction of
//!   one experiment is serialized against itself (and against `IndexId`) via a
//!   per-name in-flight gate ([`MasterState::inflight_dataset_writes`]). Distinct
//!   experiments compact concurrently. `Rewrite` vs the data-plane's `Append` is
//!   non-conflicting, so this runs safely alongside live ingest.
//! - **MergeWal** — folds flushed MemWAL generations back into the base table.
//!   The master cannot do this itself without fencing the live shard writer, so
//!   it fans out to every configured worker endpoint and each worker merges its
//!   own shard (`POST /api/v1/internal/merge-wal/{name}`).
//! - **IndexId** — builds a ZoneMap scalar index on the base table's `id` column
//!   (runs locally on the master). It commits a `CreateIndex`, which can conflict
//!   with a concurrent `Compact` `Rewrite` on the same dataset, so it shares the
//!   per-name in-flight gate with `Compact`.
//!
//! Each task runs in its own `tokio::spawn`, so one task's failure never affects
//! another. A global [`Semaphore`] bounds how many run at once.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use lance_context_api::{TaskKind, TaskRecord, TaskState};
use lance_context_core::{generate_id, CompactionConfig, RolloutStore, RolloutStoreOptions};
use tokio::sync::Semaphore;
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

/// Current Unix-millisecond timestamp.
fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

fn kind_label(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Compact => "compact",
        TaskKind::MergeWal => "merge_wal",
        TaskKind::IndexId => "index_id",
    }
}

/// Enqueue a task and return its record. For [`TaskKind::Compact`] and
/// [`TaskKind::IndexId`] this de-dupes against an existing non-terminal task for
/// the same target: if one is already `Queued` or `Running`, its record is
/// returned unchanged and nothing new is enqueued. `MergeWal` tasks are not
/// de-duped (each fan-out is independent).
pub async fn enqueue(state: &Arc<MasterState>, kind: TaskKind, target: &str) -> TaskRecord {
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
) -> TaskRecord {
    if depends_on.is_empty() && matches!(kind, TaskKind::Compact | TaskKind::IndexId) {
        let tasks = state.tasks.lock().await;
        if let Some(existing) = tasks.values().find(|t| {
            t.kind == kind
                && t.target == target
                && matches!(t.state, TaskState::Queued | TaskState::Running)
        }) {
            return existing.clone();
        }
    }

    let record = TaskRecord {
        id: generate_id(),
        kind,
        target: target.to_string(),
        state: TaskState::Queued,
        error: None,
        detail: None,
        enqueued_at: now_ms(),
        started_at: None,
        finished_at: None,
        depends_on,
    };
    state
        .tasks
        .lock()
        .await
        .insert(record.id.clone(), record.clone());
    // Unbounded send only fails if the receiver was dropped (scheduler gone).
    let _ = state.task_tx.send(record.id.clone());
    metrics::counter!("master_task_enqueued_total", "kind" => kind_label(kind)).increment(1);
    metrics::gauge!("master_task_queue_depth").increment(1.0);
    record
}

/// Mutate a task record in place under the lock, if it still exists.
async fn update_task(state: &Arc<MasterState>, id: &str, f: impl FnOnce(&mut TaskRecord)) {
    if let Some(rec) = state.tasks.lock().await.get_mut(id) {
        f(rec);
    }
}

/// Readiness of a task with respect to its declared dependencies.
enum DepStatus {
    /// All dependencies reached `Done` (or the task has none): ok to run.
    Ready,
    /// At least one dependency is still `Queued`/`Running`: defer and re-check.
    Waiting,
    /// At least one dependency `Failed` (or vanished): the dependent cannot run.
    /// Carries the id of the offending dependency for the error message.
    DepFailed(String),
}

/// Inspect a task's `depends_on` against the current task map. A missing
/// dependency is treated as failed (its record was never created or was
/// dropped), so a dependent never waits forever on a ghost id.
async fn dep_status(state: &Arc<MasterState>, id: &str) -> DepStatus {
    let tasks = state.tasks.lock().await;
    let Some(task) = tasks.get(id) else {
        return DepStatus::Ready; // vanished; run_task will no-op
    };
    for dep in &task.depends_on {
        match tasks.get(dep) {
            Some(d) => match d.state {
                TaskState::Done => {}
                TaskState::Failed => return DepStatus::DepFailed(dep.clone()),
                TaskState::Queued | TaskState::Running => return DepStatus::Waiting,
            },
            None => return DepStatus::DepFailed(dep.clone()),
        }
    }
    DepStatus::Ready
}

/// Execute one task by id: dispatch on its kind, recording lifecycle timestamps
/// and the terminal state.
async fn run_task(state: &Arc<MasterState>, id: String) {
    metrics::gauge!("master_task_queue_depth").decrement(1.0);

    let Some(task) = state.tasks.lock().await.get(&id).cloned() else {
        return; // task vanished (should not happen)
    };
    update_task(state, &id, |t| {
        t.state = TaskState::Running;
        t.started_at = Some(now_ms());
    })
    .await;

    let started = std::time::Instant::now();
    let outcome = match task.kind {
        TaskKind::Compact => run_compaction(state, &task.target).await,
        TaskKind::MergeWal => run_merge_wal(state, &task.target).await,
        TaskKind::IndexId => run_index_id(state, &task.target).await,
    };

    metrics::histogram!("master_task_duration_seconds", "kind" => kind_label(task.kind))
        .record(started.elapsed().as_secs_f64());
    let result = if outcome.is_ok() { "success" } else { "failed" };
    metrics::counter!("master_tasks_total", "kind" => kind_label(task.kind), "result" => result)
        .increment(1);

    update_task(state, &id, |t| {
        t.finished_at = Some(now_ms());
        match outcome {
            Ok(detail) => {
                t.state = TaskState::Done;
                t.detail = Some(detail);
            }
            Err(error) => {
                tracing::warn!(task = %id, target = %t.target, error = %error, "task failed");
                t.state = TaskState::Failed;
                t.error = Some(error);
            }
        }
    })
    .await;
}

/// Compact one experiment. Serialized per experiment via the shared base-table
/// write gate so two `Rewrite`s (or a `Rewrite` racing a `CreateIndex`) on the
/// same dataset never conflict. Returns a human-readable outcome summary on
/// success.
async fn run_compaction(state: &Arc<MasterState>, name: &str) -> Result<String, String> {
    // Per-name serial gate: refuse if another base-table write for this
    // experiment is already running. (De-dup at enqueue makes this rare, but
    // the gate is the real guarantee against conflicting commits.)
    {
        let mut inflight = state.inflight_dataset_writes.lock().await;
        if !inflight.insert(name.to_string()) {
            return Err(format!(
                "a base-table write for '{name}' is already in progress"
            ));
        }
    }
    // Ensure the gate is released no matter how the inner call exits.
    let result = compact_inner(state, name).await;
    state.inflight_dataset_writes.lock().await.remove(name);
    result
}

/// Build a ZoneMap scalar index on one experiment's `id` column. Shares the
/// per-name base-table write gate with [`run_compaction`] so an `IndexId` and a
/// `Compact` for the same experiment never commit concurrently (`CreateIndex`
/// vs `Rewrite` can conflict). Distinct experiments index concurrently.
async fn run_index_id(state: &Arc<MasterState>, name: &str) -> Result<String, String> {
    {
        let mut inflight = state.inflight_dataset_writes.lock().await;
        if !inflight.insert(name.to_string()) {
            return Err(format!(
                "a base-table write for '{name}' is already in progress"
            ));
        }
    }
    let result = index_id_inner(state, name).await;
    state.inflight_dataset_writes.lock().await.remove(name);
    result
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
            let resp = http.post(&url).send().await.map_err(|e| e.to_string())?;
            let status = resp.status();
            if status == reqwest::StatusCode::NOT_FOUND {
                // This worker owns no shard for `name`; treat as 0 reclaimed.
                return Ok::<usize, String>(0);
            }
            if !status.is_success() {
                return Err(format!("{url}: HTTP {status}"));
            }
            let body: MergeWalReply = resp.json().await.map_err(|e| e.to_string())?;
            Ok(body.reclaimed)
        }
    });

    let results = futures::future::join_all(calls).await;
    let total_workers = results.len();
    let mut reclaimed = 0usize;
    let mut ok_workers = 0usize;
    let mut last_err = None;
    for r in results {
        match r {
            Ok(n) => {
                reclaimed += n;
                ok_workers += 1;
            }
            Err(e) => last_err = Some(e),
        }
    }

    if ok_workers == 0 {
        return Err(last_err.unwrap_or_else(|| "all workers failed".to_string()));
    }
    Ok(format!(
        "merged {reclaimed} generations across {ok_workers}/{total_workers} workers"
    ))
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

/// Enqueue a `Compact` task for every experiment whose fragment count is at or
/// above the configured threshold, honoring quiet hours. Reads candidates from
/// the stats table.
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
            enqueue(state, TaskKind::Compact, &row.name).await;
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

/// Spawn the scheduler dispatcher plus (optionally) the periodic auto-sweep. The
/// dispatcher owns the queue receiver and runs up to `task_concurrency` tasks at
/// once, each in its own detached task. Returns the dispatcher handle. Panics if
/// called twice (receiver already taken).
pub fn spawn_scheduler(state: &Arc<MasterState>) -> JoinHandle<()> {
    let mut rx = state
        .task_rx
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

    let concurrency = state.config.task_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(concurrency));
    let dispatch_state = state.clone();
    tokio::spawn(async move {
        while let Some(id) = rx.recv().await {
            // Resolve dependencies before taking a slot, so a task waiting on a
            // chain never occupies a permit that its own dependency needs to
            // finish (which would deadlock at concurrency == 1).
            match dep_status(&dispatch_state, &id).await {
                DepStatus::Ready => {}
                DepStatus::DepFailed(dep) => {
                    // Skip: mark the dependent Failed and move on.
                    update_task(&dispatch_state, &id, |t| {
                        t.state = TaskState::Failed;
                        t.finished_at = Some(now_ms());
                        t.error = Some(format!("dependency {dep} did not complete"));
                    })
                    .await;
                    metrics::gauge!("master_task_queue_depth").decrement(1.0);
                    continue;
                }
                DepStatus::Waiting => {
                    // Re-check later without busy-looping. Re-send the id after a
                    // short delay; the queue is unbounded so this always succeeds.
                    let tx = dispatch_state.task_tx.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(200)).await;
                        let _ = tx.send(id);
                    });
                    continue;
                }
            }
            // Acquire a slot before spawning so at most `concurrency` run at once.
            let permit = sem
                .clone()
                .acquire_owned()
                .await
                .expect("semaphore never closed");
            let st = dispatch_state.clone();
            tokio::spawn(async move {
                run_task(&st, id).await;
                drop(permit);
            });
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
            worker_endpoints: vec![],
            task_concurrency: 4,
            ui_dir: None,
        }
    }

    /// Wait until the task reaches a terminal state, returning its final record.
    async fn await_terminal(state: &Arc<MasterState>, id: &str) -> TaskRecord {
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if let Some(t) = state.tasks.lock().await.get(id) {
                if matches!(t.state, TaskState::Done | TaskState::Failed) {
                    return t.clone();
                }
            }
        }
        panic!("task {id} did not reach a terminal state");
    }

    /// Manual enqueue -> dispatcher compacts -> task reaches Done and the stats
    /// table records a compaction.
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

        let rec = enqueue(&state, TaskKind::Compact, name).await;
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Done, "got {status:?}");

        let row = state.stats.lock().await.get(name).await.unwrap().unwrap();
        assert_eq!(row.total_compactions, 1);
        assert!(row.last_compaction >= 0);

        worker.abort();
    }

    /// Manual enqueue of an `IndexId` task -> dispatcher builds the ZoneMap
    /// index -> task reaches Done with the expected detail summary.
    #[tokio::test]
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

        let rec = enqueue(&state, TaskKind::IndexId, name).await;
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Done, "got {status:?}");
        assert_eq!(status.detail.as_deref(), Some("built zonemap index on id"));

        worker.abort();
    }

    /// Enqueuing the same experiment twice while queued de-dupes to one task.
    #[tokio::test]
    async fn enqueue_dedupes_queued_compactions() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        // Do NOT spawn the dispatcher, so the first task stays Queued.
        let a = enqueue(&state, TaskKind::Compact, "x").await;
        let b = enqueue(&state, TaskKind::Compact, "x").await;
        assert_eq!(a.id, b.id, "second enqueue returns the same task");
        // Only one task in the map, one message on the queue.
        assert_eq!(state.tasks.lock().await.len(), 1);
        let mut rx = state.task_rx.lock().await.take().unwrap();
        let mut n = 0;
        while rx.try_recv().is_ok() {
            n += 1;
        }
        assert_eq!(n, 1);
    }

    /// A MergeWal task with no configured endpoints fails fast with a clear msg.
    #[tokio::test]
    async fn merge_wal_without_endpoints_fails() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        let worker = spawn_scheduler(&state);
        let rec = enqueue(&state, TaskKind::MergeWal, "exp").await;
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Failed);
        assert!(status.error.unwrap().contains("worker endpoints"));
        worker.abort();
    }

    /// MergeWal fans out to every configured worker endpoint and sums the
    /// reclaimed counts. Uses a tiny in-process stub server per "worker".
    #[tokio::test]
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

        let rec = enqueue(&state, TaskKind::MergeWal, "exp").await;
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

        let compact = enqueue(&state, TaskKind::Compact, name).await;
        let index =
            enqueue_with_deps(&state, TaskKind::IndexId, name, vec![compact.id.clone()]).await;

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
    async fn dependent_skipped_when_dependency_fails() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        let worker = spawn_scheduler(&state);

        // MergeWal with no worker endpoints fails; its dependent must be skipped.
        let merge = enqueue(&state, TaskKind::MergeWal, "exp").await;
        let dependent =
            enqueue_with_deps(&state, TaskKind::Compact, "exp", vec![merge.id.clone()]).await;

        let merge_final = await_terminal(&state, &merge.id).await;
        assert_eq!(merge_final.state, TaskState::Failed);
        let dep_final = await_terminal(&state, &dependent.id).await;
        assert_eq!(dep_final.state, TaskState::Failed, "got {dep_final:?}");
        assert!(dep_final.error.unwrap().contains("dependency"));

        worker.abort();
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
