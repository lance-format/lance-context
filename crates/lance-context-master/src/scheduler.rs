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
//! - **IndexId** — builds the configured scalar index on `id` locally. Opted-in
//!   targets prepare immutable files without table write ownership, then acquire
//!   the shared writer fence to validate and publish the index metadata.
//!
//! Each task runs in its own `tokio::spawn`, so one task's failure never affects
//! another. A global [`Semaphore`] bounds how many run at once.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use lance_context_api::{TaskKind, TaskRecord};
use lance_context_core::{CompactionConfig, RolloutStore};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

use crate::state::MasterState;
use crate::stats_store::StatRow;
use crate::task_store::{TaskClaim, TaskKinds};

/// Maximum tasks a single auto-sweep may enqueue.
///
/// Previously unbounded: a sweep read the whole stats table and enqueued one
/// task per row over the threshold, so at tens of thousands of experiments a
/// single tick could flood the queue. Capping keeps each tick's work bounded;
/// anything still over the threshold is picked up by the next tick.
const MAX_SWEEP_ENQUEUE: usize = 256;

/// How many candidate rows a merge sweep reads per new task it may enqueue.
/// Candidates already in flight are deduped and do not count against the cap.
const SWEEP_CANDIDATE_MULTIPLIER: usize = 8;

/// How long a finished compaction waits for the `stats-writer` lock to refresh
/// its stats row before giving up and leaving it to the next scan round.
const STATS_REFRESH_LOCK_WAIT: Duration = Duration::from_secs(10);

/// Task-target prefix marking a generic store. Store names match
/// `[A-Za-z0-9_][A-Za-z0-9._-]*`, so a `:` can never appear in a bare name
/// and the prefix is unambiguous. Carrying the kind in the target string --
/// rather than adding a field to `TaskRecord` -- keeps the etcd task schema,
/// dedupe keys and per-target locks exactly as they are: a generic store's
/// MergeWal is locked and de-duped under `generic:<name>`, so it can never
/// collide with a rollout experiment of the same name.
const GENERIC_TARGET_PREFIX: &str = "generic:";

/// Which store kind a task target names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoreKind {
    Rollout,
    Generic,
}

/// Split a task target into its store kind and bare name.
fn parse_target(target: &str) -> (StoreKind, &str) {
    match target.strip_prefix(GENERIC_TARGET_PREFIX) {
        Some(name) => (StoreKind::Generic, name),
        None => (StoreKind::Rollout, target),
    }
}

/// Build the task target for a generic store.
pub(crate) fn generic_target(name: &str) -> String {
    format!("{GENERIC_TARGET_PREFIX}{name}")
}

fn kind_label(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Compact => "compact",
        TaskKind::MergeWal => "merge_wal",
        TaskKind::IndexId => "index_id",
        TaskKind::Repair => "repair",
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
async fn run_task(state: &Arc<MasterState>, mut claim: TaskClaim, timing: TaskClaimTiming) {
    let task = claim.task.clone();
    let kind = kind_label(task.kind);

    metrics::histogram!("master_task_phase_duration_seconds", "kind" => kind, "phase" => "claim")
        .record(timing.claim.as_secs_f64());
    // Time from enqueue to claim. Sustained growth with idle pools means
    // admission is blocking (ownership, fairness), growth with full pools
    // means capacity; either way it is the single backlog signal.
    if let Some(started_at) = task.started_at {
        let waited = (started_at - task.enqueued_at).max(0) as f64 / 1000.0;
        metrics::histogram!("master_task_schedule_to_bind_seconds", "kind" => kind).record(waited);
    }
    metrics::histogram!(
        "master_task_phase_duration_seconds",
        "kind" => kind,
        "phase" => "permit_wait",
    )
    .record(timing.permit_wait.as_secs_f64());

    let started = std::time::Instant::now();
    let mut refresh_compaction_stats = false;
    let outcome = if claim.preparing_maintenance() && task.kind == TaskKind::IndexId {
        run_prepared_index(state, &mut claim).await
    } else if claim.preparing_maintenance() {
        run_prepared_compaction(state, &mut claim)
            .await
            .map(|(detail, changed)| {
                refresh_compaction_stats = changed;
                detail
            })
    } else {
        match crate::maintenance_execution::reconcile_previous(state, &claim).await {
            Err(error) => Err(error),
            Ok(())
                if state.config.merge_rollout.draining(&task.target)
                    && task.kind != TaskKind::MergeWal =>
            {
                Err("target draining legacy writers for merge protocol transition".into())
            }
            Ok(()) => match task.kind {
                TaskKind::MergeWal => crate::merge_execution::run_merge_wal(state, &claim).await,
                _ => {
                    crate::maintenance_execution::run(state, &claim, async {
                        match task.kind {
                            TaskKind::Compact => run_compaction(state, &task).await,
                            TaskKind::IndexId => run_index_id(state, &task.target).await,
                            TaskKind::Repair => run_repair(state, &claim).await,
                            TaskKind::MergeWal => unreachable!(),
                        }
                    })
                    .await
                }
            },
        }
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
        if !matches!(task.kind, TaskKind::Repair | TaskKind::MergeWal)
            && is_missing_fragment_error(error)
        {
            // The manifest names a file storage does not have. No retry and
            // no cooldown changes that; a repair does, so enqueue one now
            // and re-run this task behind it.
            schedule_repair(state, &task).await;
        }
    }
    let commit_start = std::time::Instant::now();
    let finished = state.task_store.finish(claim, outcome).await;
    metrics::histogram!("master_task_phase_duration_seconds", "kind" => kind, "phase" => "commit")
        .record(commit_start.elapsed().as_secs_f64());
    if let Err(error) = finished {
        tracing::error!(task = %task.id, error = %error, "failed to persist task completion");
    } else if refresh_compaction_stats {
        // finish releases table ownership durably. Stats bookkeeping must not
        // hold up the next writer, but stays inside the caller's task/compact
        // permits so slow refreshes cannot create unbounded background work.
        let started = std::time::Instant::now();
        let (_, name) = parse_target(&task.target);
        match RolloutStore::open_existing_with_options(
            &state.rollout_uri(name),
            state.rollout_store_options(),
        )
        .await
        {
            Ok(store) => update_stats_after_compaction(state, name, &store).await,
            Err(error) => {
                tracing::warn!(store = %name, error = %error, "post-compaction stats open failed");
            }
        }
        metrics::histogram!("master_compaction_phase_duration_seconds", "phase" => "stats_refresh")
            .record(started.elapsed().as_secs_f64());
    }
}

/// Whether a task error is the deterministic "manifest names a data file
/// that is not there" failure. Every scan of the base table then fails the
/// same way, from the master's compaction and from every worker's merge
/// alike, until the fragment is dropped.
fn is_missing_fragment_error(error: &str) -> bool {
    error.contains("Not found")
        && (error.contains("/data/") || error.contains("/_deletions/"))
        && (error.contains(".lance") || error.contains(".arrow") || error.contains(".bin"))
}

/// Enqueue a `Repair` for the failed task's target, then the original task
/// again depending on it. Both are best-effort: the failure is already
/// recorded, and the next sweep enqueues the original kind anyway.
async fn schedule_repair(state: &Arc<MasterState>, failed: &TaskRecord) {
    let repair = match enqueue(state, TaskKind::Repair, &failed.target).await {
        Ok(repair) => repair,
        Err(error) => {
            tracing::warn!(target = %failed.target, %error, "failed to enqueue repair");
            return;
        }
    };
    tracing::warn!(
        target = %failed.target,
        after = ?failed.kind,
        repair = %repair.id,
        "base table names missing files; repair enqueued"
    );
    if let Err(error) = enqueue_with_deps(state, failed.kind, &failed.target, vec![repair.id]).await
    {
        tracing::warn!(target = %failed.target, %error, "failed to re-enqueue after repair");
    }
}

/// Drop base-table fragments whose files are missing (see
/// `StorageBase::repair_missing_fragments`) and record what was dropped.
async fn run_repair(state: &Arc<MasterState>, claim: &TaskClaim) -> Result<String, String> {
    let task = &claim.task;
    let (kind, name) = parse_target(&task.target);
    if kind != StoreKind::Rollout {
        return Err(format!("repair is not implemented for {kind:?} stores"));
    }
    let uri = state.rollout_uri(name);
    let mut store = RolloutStore::open_existing_with_options(&uri, state.rollout_store_options())
        .await
        .map_err(|e| e.to_string())?;
    let report = store
        .repair_missing_fragments()
        .await
        .map_err(|e| e.to_string())?;
    let Some(committed_version) = report.committed_version else {
        return Ok("nothing missing; no repair needed".to_string());
    };
    // A committed repair changed the data that caused these failures. Permit
    // the dependent retry immediately; a new task or a no-op repair must never
    // clear a budget. Other error classes (schema, timeout, ownership) survive.
    let coordinator = state.task_store.merge_coordinator();
    for endpoint in ["master:compact", "master:index_id"] {
        if coordinator
            .failure(&task.target, endpoint)
            .await?
            .is_some_and(|failure| is_missing_fragment_error(&failure.last_error))
        {
            coordinator
                .clear_failure(&state.task_store.merge_claim(claim), &task.target, endpoint)
                .await?;
        }
    }
    let rows: usize = report
        .dropped
        .iter()
        .map(|d| d.physical_rows.unwrap_or(0))
        .sum();
    // The task whose failure scheduled this repair is the one re-enqueued
    // behind it (see `schedule_repair`); a manual repair has none.
    let triggered_by = state
        .task_store
        .list()
        .await
        .ok()
        .and_then(|tasks| {
            tasks
                .into_iter()
                .find(|t| t.depends_on.contains(&task.id))
                .map(|t| t.kind)
        })
        .unwrap_or(TaskKind::Repair);
    let record = lance_context_api::RepairRecord {
        target: task.target.clone(),
        repaired_at_ms: chrono::Utc::now().timestamp_millis(),
        read_version: report.read_version,
        committed_version,
        triggered_by,
        dropped: report
            .dropped
            .iter()
            .map(|d| lance_context_api::RepairedFragment {
                id: d.id,
                physical_rows: d.physical_rows,
                missing_files: d.missing_files.clone(),
            })
            .collect(),
    };
    if let Err(error) = state.task_store.record_repair(&record).await {
        tracing::warn!(target = %task.target, %error, "failed to record repair");
    }
    metrics::counter!("master_repairs_total").increment(1);
    metrics::counter!("master_repair_fragments_dropped_total")
        .increment(report.dropped.len() as u64);
    Ok(format!(
        "dropped {} fragments ({} rows) whose files were missing; version {} -> {}",
        report.dropped.len(),
        rows,
        report.read_version,
        committed_version
    ))
}

/// Preparation must never publish a manifest, even if a future Lance version
/// adds a hidden metadata write. Handles opened here capture this rejection.
#[derive(Debug)]
struct PreparationOnly;
impl lance_context_core::merge_write_scope::CommitAuthorizer for PreparationOnly {
    fn authorize<'a>(
        &'a self,
        _: &'a str,
        _: u64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = lance::Result<()>> + Send + 'a>> {
        Box::pin(async { Err(lance::Error::io("maintenance preparation cannot commit")) })
    }
}

async fn watch_maintenance_preparation(
    state: &Arc<MasterState>,
    claim: &TaskClaim,
    scope: &lance_context_core::merge_write_scope::MergeWriteScope,
) -> String {
    let mut sequence = scope.completed_steps();
    let mut changed = std::time::Instant::now();
    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;
        match state.task_store.preparation_owned(claim).await {
            Ok(true) => {}
            Ok(false) => return "maintenance preparation claim lost".into(),
            Err(error) => return error.to_string(),
        }
        let current = scope.completed_steps();
        if current != sequence {
            changed = std::time::Instant::now();
            sequence = current;
        }
        if changed.elapsed()
            >= Duration::from_secs(state.config.maintenance.maintenance_idle_timeout_secs)
        {
            return "maintenance preparation made no progress".into();
        }
    }
}

/// How long a prepared compaction/index waited for its write turn, and
/// whether it got one. `expired` means the preparation is discarded and
/// rebuilt: the fairness failure #324 addressed. Both are per kind.
fn record_commit_turn(kind: TaskKind, waited: Duration, result: &'static str) {
    let kind = kind_label(kind);
    metrics::histogram!("master_commit_turn_wait_seconds", "kind" => kind, "result" => result)
        .record(waited.as_secs_f64());
    metrics::counter!("master_commit_turn_requests_total", "kind" => kind, "result" => result)
        .increment(1);
}

async fn wait_for_maintenance_commit(
    state: &Arc<MasterState>,
    claim: &mut TaskClaim,
) -> Result<(), String> {
    let waiting = std::time::Instant::now();
    let wait_budget = match claim.task.kind {
        TaskKind::IndexId => state.config.maintenance.index_commit_wait_secs,
        _ => state.config.maintenance.compaction_commit_wait_secs,
    };
    if !state
        .task_store
        .request_maintenance_commit(claim)
        .await
        .map_err(|e| e.to_string())?
    {
        return Err("maintenance preparation claim lost before requesting commit".into());
    }
    loop {
        if !state
            .task_store
            .preparation_owned(claim)
            .await
            .map_err(|e| e.to_string())?
        {
            return Err("maintenance preparation claim lost before commit".into());
        }
        if state
            .task_store
            .promote_maintenance(claim)
            .await
            .map_err(|e| e.to_string())?
        {
            record_commit_turn(claim.task.kind, waiting.elapsed(), "served");
            return Ok(());
        }
        // Check only after promotion has definitively declined. Never cancel
        // an in-flight acquisition and mistake an accepted claim for a timeout.
        if waiting.elapsed() >= Duration::from_secs(wait_budget) {
            record_commit_turn(claim.task.kind, waiting.elapsed(), "expired");
            return Err("maintenance commit ownership wait budget exhausted; reprepare".into());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn run_prepared_compaction(
    state: &Arc<MasterState>,
    claim: &mut TaskClaim,
) -> Result<(String, bool), String> {
    let target = claim.task.target.clone();
    let (_, name) = parse_target(&target);
    let uri = state.rollout_uri(name);
    let scope = lance_context_core::merge_write_scope::MergeWriteScope::with_preparation_authorizer(
        Arc::new(PreparationOnly),
    );
    let began = std::time::Instant::now();
    tracing::info!(task = %claim.task.id, target = %claim.task.target, "compaction file preparation started without table ownership");
    let prepared = {
        let work = scope.run(async {
            let store =
                RolloutStore::open_existing_with_options(&uri, state.rollout_store_options())
                    .await
                    .map_err(|e| e.to_string())?;
            store
                .prepare_compaction(state.compaction_config())
                .await
                .map_err(|e| e.to_string())
        });
        tokio::select! {
            result = work => result?,
            error = watch_maintenance_preparation(state, claim, &scope) => return Err(error),
        }
    };
    metrics::histogram!("master_compaction_phase_duration_seconds", "phase" => "prepare")
        .record(began.elapsed().as_secs_f64());
    if prepared.is_empty() {
        // Suppress only the observed snapshot; any concurrent append changes
        // the version and makes this hint inapplicable.
        let options = compact_options_key(&state.compaction_config());
        let _ = tokio::time::timeout(
            Duration::from_secs(3),
            state
                .task_store
                .record_compact_noop(name, &uri, prepared.read_version(), &options),
        )
        .await;
        return Ok(("compaction snapshot needs no rewrite".into(), false));
    }
    let waiting = std::time::Instant::now();
    tracing::info!(task = %claim.task.id, target = %claim.task.target, seconds = began.elapsed().as_secs_f64(), "compaction files ready; waiting for commit ownership");
    wait_for_maintenance_commit(state, claim).await?;
    metrics::histogram!("master_compaction_phase_duration_seconds", "phase" => "commit_wait")
        .record(waiting.elapsed().as_secs_f64());
    // No preparation handle is reused: only this scope may authorize metadata
    // commits, including Lance's fragment-ID reservation and index remap.
    crate::maintenance_execution::reconcile_previous(state, claim).await?;
    let started = std::time::Instant::now();
    let outcome = crate::maintenance_execution::run(state, claim, async {
        let mut store =
            RolloutStore::open_existing_with_options(&uri, state.rollout_store_options())
                .await
                .map_err(|e| e.to_string())?;
        let metrics = store
            .commit_prepared_compaction(prepared)
            .await
            .map_err(|e| e.to_string())?;
        finish_compaction(state, &claim.task, metrics).await
    })
    .await;
    metrics::histogram!("master_compaction_phase_duration_seconds", "phase" => "commit")
        .record(started.elapsed().as_secs_f64());
    outcome.map(|detail| (detail, true))
}

/// Compact one experiment. The task-store claim owns the per-experiment write
/// lock for the full execution.
///
/// A compaction that rewrote fragments invalidates the `id` ZoneMap index
/// (per-fragment min/max), so it enqueues an [`TaskKind::IndexId`] that
/// depends on this task. The dependency keeps the two from contending for the
/// per-target lock and lets the index wait its turn behind the queue. Skipped
/// when nothing was rewritten and when `index_after_compaction` is off.
async fn run_compaction(state: &Arc<MasterState>, task: &TaskRecord) -> Result<String, String> {
    let (kind, name) = parse_target(&task.target);
    if kind != StoreKind::Rollout {
        // Base-table compaction of generic stores is not scheduled by the
        // master yet; only WAL merges are. Refuse rather than open the store
        // through the rollout code path with the wrong URI and schema.
        return Err(format!("compaction is not scheduled for {kind:?} stores"));
    }
    let metrics = compact_inner(state, name).await?;
    finish_compaction(state, task, metrics).await
}

async fn finish_compaction(
    state: &Arc<MasterState>,
    task: &TaskRecord,
    metrics: lance::dataset::optimize::CompactionMetrics,
) -> Result<String, String> {
    if state.config.index_after_compaction && metrics.fragments_added > 0 {
        // Best-effort: the compaction itself is done; the next compaction of
        // this target re-enqueues the index anyway.
        if let Err(error) = enqueue_with_deps(
            state,
            TaskKind::IndexId,
            &task.target,
            vec![task.id.clone()],
        )
        .await
        {
            tracing::warn!(target = %task.target, %error, "failed to enqueue post-compaction id index");
        }
    }
    Ok(format!(
        "removed {} / added {} fragments",
        metrics.fragments_removed, metrics.fragments_added
    ))
}

/// Build the configured scalar index on one experiment's `id` column. Shares the
/// per-name base-table write gate with [`run_compaction`] so an `IndexId` and a
/// `Compact` for the same experiment never commit concurrently (`CreateIndex`
/// vs `Rewrite` can conflict). Distinct experiments index concurrently.
async fn run_index_id(state: &Arc<MasterState>, target: &str) -> Result<String, String> {
    let (kind, name) = parse_target(target);
    if kind != StoreKind::Rollout {
        return Err(format!("id indexing is not scheduled for {kind:?} stores"));
    }
    index_id_inner(state, name).await
}

async fn index_id_inner(state: &Arc<MasterState>, name: &str) -> Result<String, String> {
    let uri = state.rollout_uri(name);
    let mut store = RolloutStore::open_existing_with_options(&uri, state.rollout_store_options())
        .await
        .map_err(|e| e.to_string())?;
    store
        .create_id_key_index()
        .await
        .map_err(|e| e.to_string())?;
    Ok(format!(
        "built {} index on id",
        state.config.key_index_type.as_str()
    ))
}

async fn run_prepared_index(
    state: &Arc<MasterState>,
    claim: &mut TaskClaim,
) -> Result<String, String> {
    let uri = state.rollout_uri(&claim.task.target);
    let scope = lance_context_core::merge_write_scope::MergeWriteScope::with_preparation_authorizer(
        Arc::new(PreparationOnly),
    );
    let began = std::time::Instant::now();
    tracing::info!(task = %claim.task.id, target = %claim.task.target,
        "index file preparation started without table ownership");
    let prepared = {
        let work = scope.run(async {
            let store =
                RolloutStore::open_existing_with_options(&uri, state.rollout_store_options())
                    .await
                    .map_err(|e| e.to_string())?;
            store
                .prepare_id_key_index()
                .await
                .map_err(|e| e.to_string())
        });
        tokio::select! {
            result = work => result?,
            error = watch_maintenance_preparation(state, claim, &scope) => return Err(error),
        }
    };
    metrics::histogram!("master_index_phase_duration_seconds", "phase" => "prepare")
        .record(began.elapsed().as_secs_f64());
    tracing::info!(task = %claim.task.id, target = %claim.task.target,
        seconds = began.elapsed().as_secs_f64(), "index files ready; waiting for commit ownership");
    let waiting = std::time::Instant::now();
    wait_for_maintenance_commit(state, claim).await?;
    metrics::histogram!("master_index_phase_duration_seconds", "phase" => "commit_wait")
        .record(waiting.elapsed().as_secs_f64());
    crate::maintenance_execution::reconcile_previous(state, claim).await?;
    let started = std::time::Instant::now();
    let result = crate::maintenance_execution::run(state, claim, async {
        let mut store =
            RolloutStore::open_existing_with_options(&uri, state.rollout_store_options())
                .await
                .map_err(|e| e.to_string())?;
        store
            .commit_prepared_id_key_index(prepared)
            .await
            .map_err(|e| e.to_string())?;
        Ok(format!(
            "built {} index on id (prepared outside write lock)",
            state.config.key_index_type.as_str()
        ))
    })
    .await;
    metrics::histogram!("master_index_phase_duration_seconds", "phase" => "commit")
        .record(started.elapsed().as_secs_f64());
    result
}

async fn compact_inner(
    state: &Arc<MasterState>,
    name: &str,
) -> Result<lance::dataset::optimize::CompactionMetrics, String> {
    let uri = state.rollout_uri(name);
    let config = state.compaction_config();

    let mut store = RolloutStore::open_existing_with_options(&uri, state.rollout_store_options())
        .await
        .map_err(|e| e.to_string())?;
    let options = compact_options_key(&config);
    let before = store.version();
    let metrics = store
        .compact(Some(config))
        .await
        .map_err(|e| e.to_string())?;
    if metrics.fragments_removed == 0 && metrics.fragments_added == 0 && before == store.version() {
        // Optimization only: an etcd failure must not turn a completed compact
        // into a failed task or retain its write lock indefinitely.
        if !matches!(
            tokio::time::timeout(
                Duration::from_secs(3),
                state
                    .task_store
                    .record_compact_noop(name, &uri, before, &options)
            )
            .await,
            Ok(Ok(()))
        ) {
            tracing::warn!(
                store = name,
                "could not persist compact no-op; future sweep may retry"
            );
        }
    }
    update_stats_after_compaction(state, name, &store).await;
    Ok(metrics)
}

// Include every effective rewrite option. Scheduling intervals/quiet hours do
// not affect the plan. JSON keeps the encoding stable across master restarts.
fn compact_options_key(config: &CompactionConfig) -> String {
    serde_json::json!({
        "target_rows": config.target_rows_per_fragment,
        "row_group": config.max_rows_per_group,
        "deletions": config.materialize_deletions,
        "deletion_threshold": config.materialize_deletions_threshold,
        "threads": config.num_threads,
        "max_bytes": config.max_bytes_per_file,
        "batch_size": config.batch_size,
        "max_sources": config.max_source_fragments,
        "binary_copy": config.try_binary_copy,
    })
    .to_string()
}

/// Refresh the stats row for `name` after a successful compaction: re-observe
/// fragment/row counts and bump `last_compaction`/`total_compactions`.
///
/// Best-effort. The compaction itself is already committed, and the next scan
/// round refreshes the row anyway, so this never waits long for the
/// `stats-writer` lock: a holder mid-maintenance would otherwise pin this
/// task's concurrency slot on every master for as long as it runs.
async fn update_stats_after_compaction(state: &Arc<MasterState>, name: &str, store: &RolloutStore) {
    let guard = match tokio::time::timeout(
        STATS_REFRESH_LOCK_WAIT,
        state.task_store.coordination_lock("stats-writer"),
    )
    .await
    {
        Ok(Ok(guard)) => guard,
        Ok(Err(e)) => {
            tracing::warn!(store = %name, error = %e, "stats writer lock failed");
            return;
        }
        Err(_) => {
            tracing::info!(
                store = %name,
                "stats writer busy; leaving the post-compaction refresh to the next scan"
            );
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
    let config = state.compaction_config();
    // Quiet-hours gate applies to the whole sweep.
    if in_quiet_hours(&config) {
        return Ok(0);
    }
    // The stats implementation already materializes all above-threshold metadata
    // to rank candidates. Keep the tail so suppressed/active leaders cannot
    // starve useful work below the enqueue cap; never load table payloads here.
    let rows = state
        .stats
        .lock()
        .await
        .list_above_fragment_count(config.min_fragments, usize::MAX)
        .await?;
    let options = compact_options_key(&config);
    let mut queued = 0;
    for row in rows {
        if queued >= MAX_SWEEP_ENQUEUE {
            break;
        }
        // Generic rows share the stats table (their name carries the
        // `generic:` prefix) so the WAL-merge sweep sees them; compaction of
        // generic stores is not master-scheduled, so they are skipped here.
        if parse_target(&row.name).0 != StoreKind::Rollout {
            continue;
        }
        if state
            .task_store
            .is_cooling_down(TaskKind::Compact, &row.name)
            .await?
        {
            continue;
        }
        if state
            .task_store
            .compact_is_unchanged(
                &row.name,
                &state.rollout_uri(&row.name),
                row.version,
                &options,
            )
            .await?
        {
            metrics::counter!("master_compact_noop_suppressed_total").increment(1);
            continue;
        }
        let before = state
            .task_store
            .get_active_id(TaskKind::Compact, &row.name)
            .await?;
        let task = enqueue(state, TaskKind::Compact, &row.name).await?;
        if before.as_deref() != Some(task.id.as_str()) {
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
    // The cap bounds *new* tasks per tick, not candidates considered. The
    // first MAX_SWEEP_ENQUEUE rows by pending count are mostly stores that
    // already have a task queued or running (enqueue dedupes them), so a
    // sweep that stopped there spent its whole budget on no-ops while every
    // store ranked below the cap starved: with 286 stores over the read cap,
    // a 500-generation store sat unreadable with no task for hours. Read
    // more rows than the cap and keep going until the cap is spent on tasks
    // that did not exist before.
    let rows = state
        .stats
        .lock()
        .await
        .list_above_pending_wal(threshold, MAX_SWEEP_ENQUEUE * SWEEP_CANDIDATE_MULTIPLIER)
        .await?;
    let mut queued = 0;
    let mut deduped = 0;
    for row in rows {
        if queued >= MAX_SWEEP_ENQUEUE {
            break;
        }
        if state
            .task_store
            .is_cooling_down(TaskKind::MergeWal, &row.name)
            .await?
        {
            continue;
        }
        let before = state
            .task_store
            .get_active_id(TaskKind::MergeWal, &row.name)
            .await?;
        let task = enqueue(state, TaskKind::MergeWal, &row.name).await?;
        if before.as_deref() == Some(task.id.as_str()) {
            deduped += 1;
        } else {
            queued += 1;
        }
    }
    metrics::gauge!("master_merge_sweep_deduped").set(deduped as f64);
    Ok(queued)
}

// A draining target still needs metadata recovery while ownership is unresolved.
// Once released, its old failure ledger must not enqueue fresh merge tasks.
async fn should_probe_failure(
    state: &Arc<MasterState>,
    failure: &lance_context_merge::failure::ShardFailure,
) -> bool {
    if state.config.merge_rollout.owned(&failure.target) {
        return true;
    }
    if state.config.merge_rollout.draining(&failure.target) {
        match state
            .task_store
            .merge_coordinator()
            .get(&failure.target)
            .await
        {
            Ok(execution) => return execution.is_some(),
            Err(error) => {
                tracing::warn!(target = %failure.target, %error, "cannot inspect draining merge ownership")
            }
        }
    }
    false
}

pub(crate) async fn enqueue_merge_request(
    state: &Arc<MasterState>,
    coordinator: &lance_context_merge::Coordinator,
    request: &lance_context_merge::rollout::MergeRequest,
) -> Result<(), String> {
    if !state.config.merge_rollout.owned(&request.target) {
        return Ok(());
    }
    let task = enqueue(state, TaskKind::MergeWal, &request.target)
        .await
        .map_err(|e| e.to_string())?;
    if task.state == lance_context_api::TaskState::Queued {
        coordinator.acknowledge_request(request).await?;
    }
    Ok(())
}

/// Spawn the scheduler pollers plus the optional periodic auto-sweep.
///
/// Returns the handle of the *general* poller only. When a separate WAL-merge
/// budget is configured there is also a second poller (and the auto-sweep task)
/// whose handles are dropped: like the sweep, they are meant to live as long as
/// the process. Aborting the returned handle therefore stops general dispatch,
/// not every scheduler task -- adequate for tests, which drop the whole
/// `MasterState` immediately after.
pub fn spawn_scheduler(state: &Arc<MasterState>) -> JoinHandle<()> {
    crate::wal_tail::spawn(state);
    crate::resident_recovery::spawn(state);
    crate::planner::spawn(state);
    crate::executors::spawn(state);
    // Retry metadata independently of coarse stats sweeps. Every replica may
    // enqueue; the existing task dedupe/claim transaction elects one executor.
    let retry_state = state.clone();
    tokio::spawn(async move {
        let coordinator = retry_state.task_store.merge_coordinator();
        let mut cursor = None;
        let mut request_cursor = None;
        let mut execution_cursor = None;
        let mut ticker = tokio::time::interval(Duration::from_secs(15));
        loop {
            ticker.tick().await;
            match coordinator.request_page(request_cursor.as_deref()).await {
                Ok((targets, next)) => {
                    request_cursor = next;
                    for request in targets {
                        if let Err(error) =
                            enqueue_merge_request(&retry_state, &coordinator, &request).await
                        {
                            tracing::warn!(target = %request.target, %error, "worker merge demand enqueue failed");
                        }
                    }
                }
                Err(error) => tracing::warn!(%error, "worker merge demand scan failed"),
            }
            match coordinator
                .execution_page(execution_cursor.as_deref())
                .await
            {
                Ok((executions, next)) => {
                    execution_cursor = next;
                    for execution in executions {
                        if execution.maintenance.is_none() {
                            continue;
                        }
                        if let Err(error) = crate::maintenance_execution::enqueue_recovery(
                            &retry_state,
                            &coordinator,
                            &execution,
                        )
                        .await
                        {
                            tracing::warn!(target = %execution.target, %error, "local execution recovery enqueue failed");
                        }
                    }
                }
                Err(error) => tracing::warn!(%error, "local execution recovery scan failed"),
            }
            match coordinator.failure_page(cursor.as_deref(), 256).await {
                Ok((rows, next)) => {
                    cursor = next;
                    let mut targets = std::collections::HashSet::new();
                    for failure in rows {
                        let task_kind = crate::maintenance_execution::retry_kind(&failure.endpoint)
                            .or_else(|| {
                                retry_state
                                    .config
                                    .worker_endpoints
                                    .contains(&failure.endpoint)
                                    .then_some(TaskKind::MergeWal)
                            });
                        let Some(task_kind) = task_kind else {
                            continue;
                        };
                        if failure.next_retry_ms <= lance_context_merge::failure::now_ms()
                            && targets.insert((failure.target.clone(), kind_label(task_kind)))
                            && should_probe_failure(&retry_state, &failure).await
                        {
                            if let Err(error) =
                                enqueue(&retry_state, task_kind, &failure.target).await
                            {
                                tracing::warn!(target = %failure.target, %error, "maintenance recovery enqueue failed");
                            }
                        }
                    }
                }
                Err(error) => tracing::warn!(%error, "merge recovery metadata scan failed"),
            }
        }
    });

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
    // WAL merge draws from its own budget so a backlog of slow HTTP fan-outs
    // cannot occupy every general slot and starve compaction. `0` opts back
    // into the shared pool.
    let merge_sem = (state.config.merge_wal_concurrency > 0)
        .then(|| Arc::new(Semaphore::new(state.config.merge_wal_concurrency)));

    // One poller per pool, each claiming only the kinds its pool runs.
    //
    // A single poller drawing from one unfiltered `claim_next` could not keep
    // the pools independent, because claiming is destructive: it deletes the
    // queue key, grants a lease, and takes the per-experiment target lock. The
    // claimed task's kind then decided which pool it drew from, so a claim made
    // on behalf of an idle pool could return a task belonging to a saturated
    // one and park it on `acquire_owned()` -- holding its target lock while
    // idle. With `MergeWal` numerically dominant (the 600s sweep enqueues one
    // per over-threshold experiment), nearly every claim returned a MergeWal,
    // and the loop's own `while` guard stayed true only because the *general*
    // pool had permits. The result was a dispatcher that spun claiming MergeWal
    // tasks it could not run while a lone `Compact` sat queued behind them --
    // starvation that adding replicas or raising `task_concurrency` could not
    // fix, since every added slot was filled the same way.
    //
    // Splitting the pollers makes each one claim only what it can run, so a
    // free compaction slot reaches past any number of queued MergeWal tasks.
    // FIFO order within a kind is unchanged.
    let general = spawn_pool_poller(
        state.clone(),
        sem.clone(),
        if merge_sem.is_some() {
            TaskKinds::GENERAL.without_compact()
        } else {
            // No separate merge budget: the general pool runs everything, so it
            // must still be allowed to claim MergeWal.
            TaskKinds::ANY.without_compact()
        },
        true,
    );
    spawn_pool_poller(state.clone(), sem, TaskKinds::COMPACT, false);
    if let Some(merge) = merge_sem {
        spawn_pool_poller(state.clone(), merge, TaskKinds::MERGE_WAL, false);
    }
    if state.config.append.rollout_append_local
        && state.config.append.rollout_append_local_task_concurrency > 0
    {
        // Reserve before claiming. A saturated legacy RPC pool cannot consume
        // this capacity; buffers still use the process-wide local Arrow budget.
        spawn_filtered_pool_poller(
            state.clone(),
            Arc::new(Semaphore::new(
                state.config.append.rollout_append_local_task_concurrency,
            )),
            TaskKinds::MERGE_WAL,
            false,
            true,
        );
    }
    general
}

/// Spawn one dispatch loop bound to a single execution pool.
///
/// `kinds` restricts what this loop will claim; `report_depth` designates the
/// one loop that publishes the shared queue-depth gauge, so running several
/// pollers does not multiply that metric.
fn spawn_pool_poller(
    state: Arc<MasterState>,
    pool: Arc<Semaphore>,
    kinds: TaskKinds,
    report_depth: bool,
) -> tokio::task::JoinHandle<()> {
    spawn_filtered_pool_poller(state, pool, kinds, report_depth, false)
}

/// Decrements a gauge when dropped, so unwinding pays it back.
struct GaugeHold(metrics::Gauge);
impl Drop for GaugeHold {
    fn drop(&mut self) {
        self.0.decrement(1.0);
    }
}

fn spawn_filtered_pool_poller(
    state: Arc<MasterState>,
    pool: Arc<Semaphore>,
    kinds: TaskKinds,
    report_depth: bool,
    resident_only: bool,
) -> tokio::task::JoinHandle<()> {
    let pool_label: &'static str = if resident_only {
        "resident_merge"
    } else if kinds == TaskKinds::COMPACT {
        "compact"
    } else if kinds == TaskKinds::MERGE_WAL {
        "merge"
    } else {
        "general"
    };
    // Permits in use are split by what holds them: `claiming` while the etcd
    // claim is outstanding, `running` once a task owns the permit. Both are
    // updated at the permit boundaries themselves, so a claim that blocks on
    // etcd for minutes shows as a held permit rather than as idle capacity.
    let claiming =
        metrics::gauge!("master_task_pool_in_use", "pool" => pool_label, "state" => "claiming");
    let running =
        metrics::gauge!("master_task_pool_in_use", "pool" => pool_label, "state" => "running");
    claiming.set(0.0);
    running.set(0.0);
    tokio::spawn(async move {
        loop {
            if report_depth {
                if let Ok(queued) = state.task_store.queue_depth().await {
                    metrics::gauge!("master_task_queue_depth").set(queued as f64);
                }
            }
            while pool.available_permits() > 0 {
                // Reserve both budgets before taking durable table ownership.
                let compact_permit = if kinds == TaskKinds::COMPACT {
                    match state.compaction_permits.clone().try_acquire_owned() {
                        Ok(permit) => Some(permit),
                        Err(_) => break,
                    }
                } else {
                    None
                };
                let Ok(permit) = pool.clone().try_acquire_owned() else {
                    break;
                };
                let Some(operation) = state.admission.try_admit() else {
                    break;
                };
                // Held across the await so a poller aborted mid-claim (drain,
                // shutdown, test teardown) still pays the gauge back.
                claiming.increment(1.0);
                let claiming_hold = GaugeHold(claiming.clone());
                let claim_start = std::time::Instant::now();
                let claim = if resident_only {
                    state
                        .task_store
                        .claim_resident_merge(&state.config.append.rollout_append_targets)
                        .await
                } else {
                    state.task_store.claim_next_of_kinds(kinds).await
                };
                drop(claiming_hold);
                match claim {
                    Ok(Some(claim)) => {
                        let claim_elapsed = claim_start.elapsed();
                        // Local capacity was reserved before the claim. No
                        // claimed table waits on a compaction semaphore.
                        let timing = TaskClaimTiming {
                            claim: claim_elapsed,
                            permit_wait: Duration::ZERO,
                        };
                        let st = state.clone();
                        let running = running.clone();
                        running.increment(1.0);
                        tokio::spawn(async move {
                            // Decrement on every exit, including a panic in
                            // run_task: the permits are dropped by unwinding,
                            // so the gauge must be too, or capacity reads as
                            // consumed forever.
                            let _held = GaugeHold(running);
                            run_task(&st, claim, timing).await;
                            drop(permit);
                            drop(compact_permit);
                            drop(operation);
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
            append: Default::default(),
            catchup: Default::default(),
            wal_tail: Default::default(),
            planner: Default::default(),
            maintenance: Default::default(),
            merge_rollout: lance_context_merge::rollout::MergeRollout {
                owned_targets: ["exp", "generic:gs", "broken"]
                    .into_iter()
                    .map(str::to_string)
                    .chain((0..20).map(|i| format!("exp-{i}")))
                    .collect(),
                drain_targets: vec![],
            },
            data_dir: dir.path().to_string_lossy().to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            stats_scan_interval_secs: 0,
            scan_concurrency: 4,
            rollout_cache_bytes: 2 * 1024 * 1024 * 1024,
            stats_maintenance_every_n_scans: 0,
            stats_history_ttl_secs: 3_600,
            stats_cold_retire_secs: 0,
            compaction_interval_secs: 0,
            // Low threshold so a handful of appends crosses it.
            min_fragments: 2,
            target_rows_per_fragment: 1_048_576,
            compaction_concurrency: 1,
            compaction_threads: 1,
            compaction_batch_size: 8,
            compaction_max_source_fragments: 32,
            index_after_compaction: false,
            key_index_type: Default::default(),
            index_before_merge: false,
            compaction_max_bytes_per_file: 1024 * 1024 * 1024,
            merge_wal_interval_secs: 0,
            merge_wal_min_generations: 2,
            worker_endpoints: vec![],
            task_concurrency: 4,
            merge_wal_concurrency: 4,
            task_cooldown_after_failures: 3,
            task_cooldown_base_secs: 600,
            task_cooldown_max_secs: 21_600,
            etcd: lance_context_core::etcd::EtcdConfig {
                etcd_endpoints: std::env::var("ETCD_TEST_ENDPOINTS")
                    .map(|value| value.split(',').map(str::to_string).collect())
                    .unwrap_or_default(),
                etcd_prefix: format!("/lance-context/test/{}", generate_id()),
                ..Default::default()
            },
            registry: lance_context_core::etcd::RegistryConfig::default(),
            etcd_lease_ttl_secs: 5,
            task_history_limit: 1_000,
            task_history_ttl_secs: 86_400,
            ui_dir: None,
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn prepared_compact_yields_capacity_without_cancelling_busy_writer() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.maintenance.compaction_prepare_targets = vec!["exp".into()];
        cfg.maintenance.compaction_commit_wait_secs = 0;
        let state = MasterState::new(cfg).await.unwrap();
        enqueue(&state, TaskKind::Compact, "exp").await.unwrap();
        let mut prep = state
            .task_store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        let merge = state
            .task_store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let merge_id = merge.task.id.clone();
        let error = wait_for_maintenance_commit(&state, &mut prep)
            .await
            .unwrap_err();
        assert!(error.contains("wait budget exhausted"));
        state.task_store.finish(prep, Err(error)).await.unwrap();
        assert_eq!(
            state
                .task_store
                .get(&merge_id)
                .await
                .unwrap()
                .unwrap()
                .state,
            TaskState::Running
        );
        // The other writer still owns its exact claim and can finish normally.
        state
            .task_store
            .finish(merge, Ok("writer kept progressing".into()))
            .await
            .unwrap();
        assert_eq!(
            state
                .task_store
                .get(&merge_id)
                .await
                .unwrap()
                .unwrap()
                .state,
            TaskState::Done
        );
        enqueue(&state, TaskKind::Compact, "exp").await.unwrap();
        let mut next = state
            .task_store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        wait_for_maintenance_commit(&state, &mut next)
            .await
            .unwrap();
        state
            .task_store
            .finish(next, Ok("retry acquired free writer".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn compact_capacity_is_reserved_before_claim_without_blocking_other_kinds() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        RolloutStore::open(&state.rollout_uri("exp")).await.unwrap();
        let occupied = state
            .compaction_permits
            .clone()
            .acquire_owned()
            .await
            .unwrap();
        let compact = enqueue(&state, TaskKind::Compact, "exp").await.unwrap();
        enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        enqueue(&state, TaskKind::IndexId, "other").await.unwrap();
        let pool = Arc::new(Semaphore::new(2));
        let poller = spawn_pool_poller(state.clone(), pool.clone(), TaskKinds::COMPACT, false);
        tokio::time::sleep(Duration::from_millis(650)).await;
        assert_eq!(
            state
                .task_store
                .get(&compact.id)
                .await
                .unwrap()
                .unwrap()
                .state,
            TaskState::Queued
        );
        assert_eq!(pool.available_permits(), 2);
        let merge = state
            .task_store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let index = state
            .task_store
            .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(index.task.kind, TaskKind::IndexId);
        state
            .task_store
            .finish(merge, Ok("merge is not pinned by idle compact".into()))
            .await
            .unwrap();
        state
            .task_store
            .finish(index, Ok("index remains dispatchable".into()))
            .await
            .unwrap();
        drop(occupied);
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let task = state.task_store.get(&compact.id).await.unwrap().unwrap();
                if task.state == TaskState::Done {
                    break;
                }
                assert_ne!(task.state, TaskState::Failed, "{:?}", task.error);
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        poller.abort();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn prepared_compact_runs_through_owned_commit_and_releases_both_claims() {
        prepared_compact_commit_and_release(true).await;
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn prepared_compact_runs_through_legacy_commit_and_releases_both_claims() {
        prepared_compact_commit_and_release(false).await;
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn prepared_compact_releases_writer_before_waiting_for_stats() {
        for owned in [true, false] {
            let dir = TempDir::new().unwrap();
            let mut cfg = config(&dir);
            cfg.maintenance.compaction_prepare_targets = vec!["exp".into()];
            cfg.etcd_lease_ttl_secs = 30;
            if !owned {
                cfg.merge_rollout.owned_targets.clear();
            }
            let state = MasterState::new(cfg).await.unwrap();
            let mut store = RolloutStore::open(&state.rollout_uri("exp")).await.unwrap();
            for id in ["a", "b", "c", "d"] {
                store.add(&[rollout_record(id)]).await.unwrap();
                store.cleanup_own_shard().await.unwrap();
            }
            let stats_guard = state
                .task_store
                .coordination_lock("stats-writer")
                .await
                .unwrap();
            let compact = enqueue(&state, TaskKind::Compact, "exp").await.unwrap();
            let claim = state
                .task_store
                .claim_next_of_kinds(TaskKinds::COMPACT)
                .await
                .unwrap()
                .unwrap();
            let permit = state
                .compaction_permits
                .clone()
                .acquire_owned()
                .await
                .unwrap();
            let runner_state = state.clone();
            let runner = tokio::spawn(async move {
                let _permit = permit;
                run_task(&runner_state, claim, TaskClaimTiming::default()).await;
            });
            // This is shorter than STATS_REFRESH_LOCK_WAIT. Completion must
            // become durable while the competing stats writer still holds on.
            let finished = await_terminal(&state, &compact.id).await;
            assert_eq!(finished.state, TaskState::Done, "{:?}", finished.error);
            assert!(!runner.is_finished());
            assert_eq!(state.compaction_permits.available_permits(), 0);
            enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
            let merge = state
                .task_store
                .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                .await
                .unwrap()
                .expect("stats contention must not retain table ownership");
            state
                .task_store
                .finish(merge, Ok("next writer admitted during stats wait".into()))
                .await
                .unwrap();
            assert!(!runner.is_finished());
            state
                .task_store
                .release_coordination_lock(stats_guard)
                .await
                .unwrap();
            tokio::time::timeout(Duration::from_secs(5), runner)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(state.compaction_permits.available_permits(), 1);
            let stats = state.stats.lock().await.get("exp").await.unwrap().unwrap();
            assert_eq!(stats.total_compactions, 1);
            assert_eq!(stats.row_count, 4);
        }
    }

    async fn prepared_compact_commit_and_release(owned: bool) {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.maintenance.compaction_prepare_targets = vec!["exp".into()];
        if !owned {
            cfg.merge_rollout.owned_targets.clear();
        }
        let state = MasterState::new(cfg).await.unwrap();
        let uri = state.rollout_uri("exp");
        let mut store = RolloutStore::open(&uri).await.unwrap();
        for id in ["a", "b", "c", "d"] {
            store.add(&[rollout_record(id)]).await.unwrap();
            store.cleanup_own_shard().await.unwrap();
        }
        let task = enqueue(&state, TaskKind::Compact, "exp").await.unwrap();
        let claim = state
            .task_store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        assert!(claim.preparing_maintenance());
        run_task(
            &state,
            claim,
            TaskClaimTiming {
                claim: Duration::ZERO,
                permit_wait: Duration::ZERO,
            },
        )
        .await;
        let finished = state.task_store.get(&task.id).await.unwrap().unwrap();
        assert_eq!(finished.state, TaskState::Done, "{:?}", finished.error);
        assert!(state
            .task_store
            .merge_coordinator()
            .get("exp")
            .await
            .unwrap()
            .is_none());
        enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        let merge = state
            .task_store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        state
            .task_store
            .finish(merge, Ok("resumed".into()))
            .await
            .unwrap();
        let reader = RolloutStore::open_existing_with_options(&uri, Default::default())
            .await
            .unwrap();
        for id in ["a", "b", "c", "d"] {
            assert!(reader.get_by_id(id).await.unwrap().is_some());
        }
    }

    /// The stats scan publishes a scan-sourced demand event for every shard
    /// it observes, including shards whose writer never ran the server
    /// flush path, and a later scan cannot lower a watermark a writer or
    /// executor set at the same epoch.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn scan_publishes_every_shard_and_never_lowers_a_watermark() {
        use lance_context_core::RolloutStoreOptions;
        use lance_context_merge::demand::{DemandEvent, EventSource, TableDemand};
        let dir = TempDir::new().unwrap();
        let cfg = config(&dir);
        let state = MasterState::new(cfg).await.unwrap();
        let uri = state.rollout_uri("exp");
        for shard in ["a", "b"] {
            let writer = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    shard_id: Some(shard.into()),
                    merge_after_generations: Some(0),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            for i in 0..3 {
                writer
                    .add(&[rollout_record(&format!("{shard}-{i}"))])
                    .await
                    .unwrap();
                writer.flush().await.unwrap();
            }
        }
        state.registry.upsert("exp", &uri).await.unwrap();
        crate::scanner::scan_once(&state).await.unwrap();

        let prefix = state
            .config
            .etcd
            .etcd_prefix
            .trim_end_matches('/')
            .to_string();
        let encoded: String = "exp".bytes().map(|b| format!("{b:02x}")).collect();
        let events_prefix = format!("{prefix}/demand-events/{encoded}/");
        let client = state.task_store.etcd_client().clone();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let events = loop {
            let kvs = client
                .clone()
                .get(
                    events_prefix.as_str(),
                    Some(etcd_client::GetOptions::new().with_prefix()),
                )
                .await
                .unwrap();
            if kvs.kvs().len() >= 2 {
                break kvs;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "scan demand never landed"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(events.kvs().len(), 2, "one scan event per shard");
        let mut table = TableDemand::default();
        for kv in events.kvs() {
            let event: DemandEvent = serde_json::from_slice(kv.value()).unwrap();
            assert_eq!(event.source, EventSource::Scan);
            assert!(event.sealed_through >= 3, "{event:?}");
            table.fold(&event, kv.mod_revision(), 0).unwrap();
        }
        assert_eq!(table.pending_generations(), 6);

        // An executor reports merged=2 on shard "a" at the same epoch. A
        // second scan (which knows nothing about merged) must not lower it.
        let coordinator = state.task_store.merge_coordinator();
        let (key, mut a): (String, DemandEvent) = events
            .kvs()
            .iter()
            .map(|kv| {
                (
                    String::from_utf8_lossy(kv.key()).to_string(),
                    serde_json::from_slice::<DemandEvent>(kv.value()).unwrap(),
                )
            })
            .next()
            .unwrap();
        a.merged_through = Some(2);
        a.source = EventSource::Executor;
        assert!(coordinator.publish_demand_event("exp", &a).await.unwrap());
        let before = client.clone().get(key.as_str(), None).await.unwrap().kvs()[0].mod_revision();
        crate::scanner::scan_once(&state).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        let after = client.clone().get(key.as_str(), None).await.unwrap();
        let stored: DemandEvent = serde_json::from_slice(after.kvs()[0].value()).unwrap();
        assert_eq!(
            stored.merged_through,
            Some(2),
            "scan lowered merged: {stored:?}"
        );
        assert_eq!(stored.sealed_through, a.sealed_through);
        assert_eq!(
            after.kvs()[0].mod_revision(),
            before,
            "an unchanged scan must not rewrite the record"
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn compact_noop_survives_restart_and_changes_invalidate_it() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.min_fragments = 1;
        let state = MasterState::new(cfg.clone()).await.unwrap();
        let uri = state.rollout_uri("exp");
        let mut store = RolloutStore::open(&uri).await.unwrap();
        store.add(&[rollout_record("a")]).await.unwrap();
        store.cleanup_own_shard().await.unwrap();
        state.registry.upsert("exp", &uri).await.unwrap();
        crate::scanner::scan_once(&state).await.unwrap();
        let metrics = compact_inner(&state, "exp").await.unwrap();
        assert_eq!(metrics.fragments_removed, 0);
        assert_eq!(sweep_candidates(&state).await.unwrap(), 0);
        drop(state);
        let restarted = MasterState::new(cfg.clone()).await.unwrap();
        assert_eq!(sweep_candidates(&restarted).await.unwrap(), 0);
        // Explicit operator requests are still admitted.
        let manual = enqueue(&restarted, TaskKind::Compact, "exp").await.unwrap();
        assert_eq!(manual.state, TaskState::Queued);
        // A version change (including index/deletion commits) invalidates it.
        let key = compact_options_key(&restarted.compaction_config());
        assert!(!restarted
            .task_store
            .compact_is_unchanged("exp", &uri, store.version() as i64 + 1, &key)
            .await
            .unwrap());
        cfg.target_rows_per_fragment += 1;
        let changed = MasterState::new(cfg).await.unwrap();
        assert!(!changed
            .task_store
            .compact_is_unchanged(
                "exp",
                &uri,
                store.version() as i64,
                &compact_options_key(&changed.compaction_config())
            )
            .await
            .unwrap());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn compact_sweep_reaches_useful_tail_after_suppressed_leaders() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        let options = compact_options_key(&state.compaction_config());
        let mut rows = Vec::new();
        for i in 0..=MAX_SWEEP_ENQUEUE {
            let name = format!("noop-{i}");
            let uri = state.rollout_uri(&name);
            state
                .task_store
                .record_compact_noop(&name, &uri, 1, &options)
                .await
                .unwrap();
            rows.push(StatRow {
                name,
                uri,
                version: 1,
                fragment_count: 100,
                row_count: 0,
                last_updated: 0,
                pending_wal_generations: 0,
                last_compaction: 0,
                total_compactions: 1,
                scanned_at: 0,
            });
        }
        let mut useful = rows[0].clone();
        useful.name = "useful".into();
        useful.uri = state.rollout_uri("useful");
        useful.fragment_count = 20;
        rows.push(useful);
        state
            .stats
            .lock()
            .await
            .replace_snapshot(&rows)
            .await
            .unwrap();
        assert_eq!(sweep_candidates(&state).await.unwrap(), 1);
        assert!(state
            .task_store
            .get_active_id(TaskKind::Compact, "useful")
            .await
            .unwrap()
            .is_some());
        assert_eq!(
            sweep_candidates(&state).await.unwrap(),
            0,
            "deduped tasks do not spend the budget"
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn demand_during_running_merge_retains_a_followup_without_stats_sweep() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.merge_rollout.owned_targets = vec!["hot".into()];
        let state = MasterState::new(cfg).await.unwrap();
        let coordinator = state.task_store.merge_coordinator();
        let first = enqueue(&state, TaskKind::MergeWal, "hot").await.unwrap();
        let claim = state.task_store.claim_next().await.unwrap().unwrap();
        assert_eq!(claim.task.id, first.id);
        coordinator.request_merge("hot").await.unwrap();
        let request = coordinator.request_page(None).await.unwrap().0.remove(0);
        enqueue_merge_request(&state, &coordinator, &request)
            .await
            .unwrap();
        assert_eq!(coordinator.request_page(None).await.unwrap().0.len(), 1);
        state
            .task_store
            .finish(claim, Ok("old pass complete".into()))
            .await
            .unwrap();
        enqueue_merge_request(&state, &coordinator, &request)
            .await
            .unwrap();
        let next = state
            .task_store
            .get_active_id(TaskKind::MergeWal, "hot")
            .await
            .unwrap()
            .unwrap();
        assert_ne!(first.id, next);
        assert!(coordinator.request_page(None).await.unwrap().0.is_empty());
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
    async fn draining_resolves_existing_owned_merge_and_blocks_new_target_mutations() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.merge_rollout.owned_targets.retain(|t| t != "exp");
        cfg.merge_rollout.drain_targets.push("exp".into());
        cfg.worker_endpoints = vec!["http://127.0.0.1:1".into()];
        let state = MasterState::new(cfg).await.unwrap();
        enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        let claim = state
            .task_store
            .claim_next_of_kinds(crate::task_store::TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let coordinator = state.task_store.merge_coordinator();
        let proof = state.task_store.merge_claim(&claim);
        let failure = coordinator
            .record_failure(&proof, "exp", "http://127.0.0.1:1", "storage timeout")
            .await
            .unwrap();
        assert!(!should_probe_failure(&state, &failure).await);
        let execution =
            lance_context_merge::Execution::new("exp", "http://127.0.0.1:1", "boot", 30);
        assert!(coordinator.reserve(&proof, &execution).await.unwrap());
        assert!(should_probe_failure(&state, &failure).await);
        let running = coordinator.start(&execution).await.unwrap().unwrap();
        assert!(coordinator.finish(&running, Ok(0)).await.unwrap());
        let outcome = crate::merge_execution::run_merge_wal(&state, &claim).await;
        assert!(outcome.as_ref().unwrap_err().contains("draining"));
        assert!(coordinator.get("exp").await.unwrap().is_none());
        assert!(!should_probe_failure(&state, &failure).await);
        state.task_store.finish(claim, outcome).await.unwrap();

        let compact = enqueue(&state, TaskKind::Compact, "exp").await.unwrap();
        let claim = state
            .task_store
            .claim_next_of_kinds(crate::task_store::TaskKinds::GENERAL)
            .await
            .unwrap()
            .unwrap();
        run_task(
            &state,
            claim,
            TaskClaimTiming {
                claim: Duration::ZERO,
                permit_wait: Duration::ZERO,
            },
        )
        .await;
        let result = state.task_store.get(&compact.id).await.unwrap().unwrap();
        assert!(result.error.as_deref().unwrap().contains("draining"));
    }

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
        state.registry.upsert(name, &uri).await.unwrap();
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

    /// A compaction that rewrote fragments enqueues an `IndexId` that depends
    /// on it, and that index runs to Done after the compaction.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn compaction_that_rewrites_fragments_enqueues_id_index() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.index_after_compaction = true;
        let state = MasterState::new(cfg).await.unwrap();
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
        state.registry.upsert(name, &uri).await.unwrap();
        crate::scanner::scan_once(&state).await.unwrap();

        let compact = enqueue(&state, TaskKind::Compact, name).await.unwrap();
        assert_eq!(
            await_terminal(&state, &compact.id).await.state,
            TaskState::Done
        );

        let index = state
            .task_store
            .list()
            .await
            .unwrap()
            .into_iter()
            .find(|task| task.kind == TaskKind::IndexId && task.target == name)
            .expect("compaction enqueued an IndexId task");
        assert_eq!(index.depends_on, vec![compact.id.clone()]);
        let status = await_terminal(&state, &index.id).await;
        assert_eq!(status.state, TaskState::Done, "got {status:?}");
        assert_eq!(status.detail.as_deref(), Some("built btree index on id"));

        // Compact until nothing is left to rewrite. The first merge built the
        // id BTree over the first fragment, and Lance compacts indexed and
        // unindexed fragments in separate groups, so reaching one fragment
        // can take more than one pass. Only passes that rewrote fragments
        // enqueue an IndexId; the final no-op pass must not.
        let mut rewriting_compactions = 1;
        loop {
            let again = enqueue(&state, TaskKind::Compact, name).await.unwrap();
            let status = await_terminal(&state, &again.id).await;
            assert_eq!(status.state, TaskState::Done, "got {status:?}");
            if status.detail.as_deref() == Some("removed 0 / added 0 fragments") {
                break;
            }
            rewriting_compactions += 1;
            assert!(rewriting_compactions <= 4, "compaction never converged");
        }
        let indexes = state
            .task_store
            .list()
            .await
            .unwrap()
            .into_iter()
            .filter(|task| task.kind == TaskKind::IndexId)
            .count();
        assert_eq!(indexes, rewriting_compactions);

        worker.abort();
    }

    #[test]
    fn missing_fragment_error_is_recognised() {
        assert!(is_missing_fragment_error(
            "Wrapped error: Not found: rocketkeep/x.rollout.lance/data/0101abcd.lance, /rustc/..."
        ));
        assert!(is_missing_fragment_error(
            "LanceError(IO): Not found: rocketkeep/x.rollout.lance/_deletions/3-12-7.arrow"
        ));
        // A missing manifest or WAL generation is a different failure.
        assert!(!is_missing_fragment_error(
            "Not found: rocketkeep/x.rollout.lance/_versions/12.manifest"
        ));
        assert!(!is_missing_fragment_error(
            "Not found: rocketkeep/x.rollout.lance/_mem_wal/shard/gen_5/_versions"
        ));
        assert!(!is_missing_fragment_error("HTTP 500 Internal Server Error"));
    }

    /// A merge-wal on a store with no id BTree builds one before fanning out,
    /// so the workers' `merge_insert` probes instead of scanning; a second
    /// merge finds it present and builds nothing.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn owned_merge_delegates_index_safety_to_worker() {
        use axum::{routing::post, Json, Router};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.worker_endpoints = vec![format!("http://{addr}")];
        cfg.index_before_merge = true;
        let base = cfg.data_dir.clone();
        let app = Router::new().route(
            "/api/v1/internal/merge-wal/{name}",
            post(
                move |axum::extract::Path(name): axum::extract::Path<String>| {
                    let base = base.clone();
                    async move {
                        let uri =
                            lance_context_core::join_uri(&base, &format!("{name}.rollout.lance"));
                        let mut store = RolloutStore::open(&uri).await.unwrap();
                        if !store.has_id_btree_index().await.unwrap() {
                            store.create_id_btree_index().await.unwrap();
                        }
                        Json(serde_json::json!({"reclaimed":0}))
                    }
                },
            ),
        );
        let app = owned_stub(app, &cfg).await;
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = MasterState::new(cfg).await.unwrap();
        let worker = spawn_scheduler(&state);

        let name = "exp";
        let uri = state.rollout_uri(name);
        {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            store.add(&[rollout_record("r0")]).await.unwrap();
            store.cleanup_own_shard().await.unwrap();
            assert!(!store.has_id_btree_index().await.unwrap());
        }
        state.registry.upsert(name, &uri).await.unwrap();

        let rec = enqueue(&state, TaskKind::MergeWal, name).await.unwrap();
        assert_eq!(await_terminal(&state, &rec.id).await.state, TaskState::Done);
        let store = RolloutStore::open_existing_with_options(&uri, Default::default())
            .await
            .unwrap();
        assert!(
            store.has_id_btree_index().await.unwrap(),
            "merge built the index"
        );
        let version_after_first = store.version();

        // Present now: the next merge does not rebuild it.
        let rec = enqueue(&state, TaskKind::MergeWal, name).await.unwrap();
        assert_eq!(await_terminal(&state, &rec.id).await.state, TaskState::Done);
        let store = RolloutStore::open_existing_with_options(&uri, Default::default())
            .await
            .unwrap();
        assert_eq!(store.version(), version_after_first);

        worker.abort();
    }

    /// A base table whose manifest names a missing data file fails every
    /// compaction with `Not found`. That failure enqueues a `Repair` and the
    /// compaction again behind it; the repair drops the dead fragment and is
    /// recorded, and the re-run compaction succeeds.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn missing_fragment_failure_triggers_repair_and_rerun() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        // Repairing a missing fragment must not erase an unrelated failure.
        enqueue(&state, TaskKind::IndexId, "exp").await.unwrap();
        let index_claim = state.task_store.claim_next().await.unwrap().unwrap();
        let coordinator = state.task_store.merge_coordinator();
        let unrelated = coordinator
            .record_failure(
                &state.task_store.merge_claim(&index_claim),
                "exp",
                "master:index_id",
                "invalid index schema",
            )
            .await
            .unwrap();
        state
            .task_store
            .finish(index_claim, Err("invalid index schema".into()))
            .await
            .unwrap();
        let worker = spawn_scheduler(&state);

        let name = "exp";
        let uri = state.rollout_uri(name);
        let victim_path = {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            for i in 0..4 {
                store
                    .add(&[rollout_record(&format!("r{i}"))])
                    .await
                    .unwrap();
                store.cleanup_own_shard().await.unwrap();
            }
            store.base_data_files()[1].clone()
        };
        std::fs::remove_file(
            dir.path()
                .join(format!("{name}.rollout.lance/data/{victim_path}")),
        )
        .unwrap();
        state.registry.upsert(name, &uri).await.unwrap();
        crate::scanner::scan_once(&state).await.unwrap();

        let compact = enqueue(&state, TaskKind::Compact, name).await.unwrap();
        let failed = await_terminal(&state, &compact.id).await;
        assert_eq!(failed.state, TaskState::Failed, "got {failed:?}");
        assert!(
            failed.error.as_deref().unwrap_or("").contains("Not found"),
            "{failed:?}"
        );

        // The failure enqueued a repair and a dependent compaction.
        let tasks = state.task_store.list().await.unwrap();
        let repair = tasks
            .iter()
            .find(|t| t.kind == TaskKind::Repair && t.target == name)
            .expect("repair enqueued")
            .clone();
        let rerun = tasks
            .iter()
            .find(|t| t.kind == TaskKind::Compact && t.target == name && t.id != compact.id)
            .expect("compaction re-enqueued")
            .clone();
        assert_eq!(rerun.depends_on, vec![repair.id.clone()]);

        let repair = await_terminal(&state, &repair.id).await;
        assert_eq!(repair.state, TaskState::Done, "got {repair:?}");
        assert!(repair
            .detail
            .as_deref()
            .unwrap()
            .starts_with("dropped 1 fragments (1 rows)"));
        let rerun = await_terminal(&state, &rerun.id).await;
        assert_eq!(rerun.state, TaskState::Done, "got {rerun:?}");

        let retained = coordinator
            .failure(name, "master:index_id")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retained.next_retry_ms, unrelated.next_retry_ms);
        assert_eq!(
            retained.consecutive_attempts,
            unrelated.consecutive_attempts
        );
        assert!(coordinator
            .failure(name, "master:compact")
            .await
            .unwrap()
            .is_none());

        let repairs = state.task_store.list_repairs().await.unwrap();
        assert_eq!(repairs.len(), 1);
        assert_eq!(repairs[0].target, name);
        assert_eq!(repairs[0].triggered_by, TaskKind::Compact);
        assert_eq!(repairs[0].dropped.len(), 1);
        assert_eq!(repairs[0].dropped[0].missing_files, vec![victim_path]);

        // Three rows remain readable.
        let store = RolloutStore::open_existing_with_options(&uri, Default::default())
            .await
            .unwrap();
        assert_eq!(store.list(None, None).await.unwrap().len(), 3);

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
        state.registry.upsert(name, &uri).await.unwrap();

        let rec = enqueue(&state, TaskKind::IndexId, name).await.unwrap();
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Done, "got {status:?}");
        assert_eq!(status.detail.as_deref(), Some("built btree index on id"));

        worker.abort();
    }

    /// The preparation watchdog is progress-based, not wall-clock based.
    /// A build that completes a step every second must outlive an idle
    /// timeout shorter than its total duration; one that goes silent must be
    /// cancelled close to the idle timeout; and a real index build under the
    /// same short timeout must finish, proving data IO is observed as
    /// progress. Guards the #327 + #330 + #328 interaction: a cancelled
    /// slow-but-alive build would be classified retryable and rebuilt every
    /// few seconds forever.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn slow_but_alive_preparation_outlives_idle_timeout() {
        use lance_context_core::merge_write_scope::{checkpoint, MergeWriteScope};
        const IDLE: u64 = 3;
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.maintenance.index_prepare_targets = vec!["exp".into()];
        cfg.maintenance.maintenance_idle_timeout_secs = IDLE;
        cfg.merge_rollout.owned_targets = vec!["exp".into()];
        let state = MasterState::new(cfg).await.unwrap();
        let uri = state.rollout_uri("exp");
        let mut writer = RolloutStore::open(&uri).await.unwrap();
        writer.add(&[rollout_record("seed")]).await.unwrap();
        writer.cleanup_own_shard().await.unwrap();

        let claim_preparation = |state: Arc<MasterState>| async move {
            let task = enqueue(&state, TaskKind::IndexId, "exp").await.unwrap();
            let claim = state
                .task_store
                .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
                .await
                .unwrap()
                .unwrap();
            assert!(claim.preparing_maintenance());
            (task, claim)
        };

        // (a) Slow but alive: 8 s of work, one completed step per second.
        let (task, claim) = claim_preparation(state.clone()).await;
        let scope = MergeWriteScope::with_preparation_authorizer(Arc::new(PreparationOnly));
        let began = std::time::Instant::now();
        let work = scope.run(async {
            for _ in 0..8 {
                tokio::time::sleep(Duration::from_secs(1)).await;
                checkpoint();
            }
            "finished"
        });
        let outcome = tokio::select! {
            done = work => Ok(done),
            error = watch_maintenance_preparation(&state, &claim, &scope) => Err(error),
        };
        assert_eq!(outcome, Ok("finished"), "alive preparation was cancelled");
        assert!(began.elapsed() >= Duration::from_secs(8));
        state
            .task_store
            .finish(claim, Ok("a".into()))
            .await
            .unwrap();
        assert_eq!(
            state.task_store.get(&task.id).await.unwrap().unwrap().state,
            TaskState::Done
        );

        // (b) Silent: no steps at all. Cancelled near IDLE, not before.
        let (task, claim) = claim_preparation(state.clone()).await;
        let scope = MergeWriteScope::with_preparation_authorizer(Arc::new(PreparationOnly));
        let began = std::time::Instant::now();
        let work = scope.run(async {
            tokio::time::sleep(Duration::from_secs(30)).await;
            "finished"
        });
        let outcome = tokio::select! {
            done = work => Ok(done),
            error = watch_maintenance_preparation(&state, &claim, &scope) => Err(error),
        };
        assert_eq!(
            outcome,
            Err("maintenance preparation made no progress".to_string())
        );
        let elapsed = began.elapsed();
        assert!(
            elapsed >= Duration::from_secs(IDLE) && elapsed < Duration::from_secs(IDLE + 4),
            "stall detected after {elapsed:?}"
        );
        state
            .task_store
            .finish(claim, Err("stalled".into()))
            .await
            .unwrap();
        assert_eq!(
            state.task_store.get(&task.id).await.unwrap().unwrap().state,
            TaskState::Failed
        );

        // (c) A real index build under the same short idle timeout completes:
        // data/_indices IO is counted as progress by the preparation store.
        let (task, claim) = claim_preparation(state.clone()).await;
        run_task(&state, claim, TaskClaimTiming::default()).await;
        let status = state.task_store.get(&task.id).await.unwrap().unwrap();
        assert_eq!(status.state, TaskState::Done, "{status:?}");
        assert!(status
            .detail
            .as_deref()
            .is_some_and(|d| d.contains("prepared outside write lock")));
    }

    /// Review rounds 5 and 7: sustained concurrent progress isolation.
    /// Four real index preparations run at once for several idle timeouts:
    /// two throttled-but-alive at different rates, one frozen from the
    /// start, one frozen partway through. Each watchdog must judge only its
    /// own scope's completed IO. The alive pair must finish regardless of
    /// the frozen pair's silence; the frozen pair must be cancelled within
    /// the idle window regardless of the alive pair's progress; and the
    /// partway freeze must be detected from its own last progress, not the
    /// test start. Runs long enough (> 4 idle windows) that a shared or
    /// leaked progress counter would show up as a wrong verdict.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS; ~20 s"]
    async fn concurrent_preparations_are_judged_only_by_their_own_progress() {
        use lance_context_core::merge_write_scope::MergeWriteScope;
        use lance_context_core::preparation_io::test_support::{Throttle, ThrottleWrapper};
        use std::sync::atomic::Ordering;
        const IDLE: u64 = 3;
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.data_dir = format!("file-object-store://{}", dir.path().display());
        cfg.maintenance.index_prepare_targets = vec!["*".into()];
        cfg.maintenance.maintenance_idle_timeout_secs = IDLE;
        cfg.merge_rollout.owned_targets = ["slow-a", "slow-b", "frozen-start", "frozen-mid"]
            .map(String::from)
            .to_vec();
        cfg.task_concurrency = 4;
        let state = MasterState::new(cfg).await.unwrap();
        for target in ["slow-a", "slow-b", "frozen-start", "frozen-mid"] {
            let mut writer = RolloutStore::open(&state.rollout_uri(target))
                .await
                .unwrap();
            let records: Vec<_> = (0..64)
                .map(|i| rollout_record(&format!("{target}-{i}")))
                .collect();
            writer.add(&records).await.unwrap();
            writer.cleanup_own_shard().await.unwrap();
        }
        let mut claims = std::collections::HashMap::new();
        for _ in 0..4 {
            let claim = state
                .task_store
                .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
                .await
                .unwrap();
            let claim = match claim {
                Some(c) => c,
                None => {
                    // Enqueue lazily so claims come out one per table.
                    for target in ["slow-a", "slow-b", "frozen-start", "frozen-mid"] {
                        if !claims.contains_key(target) {
                            enqueue(&state, TaskKind::IndexId, target).await.unwrap();
                        }
                    }
                    state
                        .task_store
                        .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
                        .await
                        .unwrap()
                        .unwrap()
                }
            };
            assert!(claim.preparing_maintenance());
            claims.insert(claim.task.target.clone(), claim);
        }
        assert_eq!(claims.len(), 4);

        let run =
            |target: &'static str, claim: crate::task_store::TaskClaim, throttle: Throttle| {
                let state = state.clone();
                let scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
                    Arc::new(PreparationOnly),
                    Arc::new(ThrottleWrapper(throttle.clone())),
                );
                let uri = state.rollout_uri(target);
                async move {
                    let began = std::time::Instant::now();
                    let work = scope.run(async {
                        let store = RolloutStore::open_existing_with_options(
                            &uri,
                            state.rollout_store_options(),
                        )
                        .await
                        .map_err(|e| e.to_string())?;
                        store
                            .prepare_id_key_index()
                            .await
                            .map(|_| "prepared")
                            .map_err(|e| e.to_string())
                    });
                    let outcome = tokio::select! {
                        done = work => done,
                        error = watch_maintenance_preparation(&state, &claim, &scope) => Err(error),
                    };
                    (
                        outcome,
                        began.elapsed(),
                        throttle.ops.load(Ordering::Relaxed),
                        claim,
                    )
                }
            };
        let slow_a = Throttle::new(Duration::from_millis(1200)); // 2.5x per op below IDLE
        let slow_b = Throttle::new(Duration::from_millis(2500)); // just under IDLE per op
        let frozen_start = Throttle::new(Duration::ZERO);
        frozen_start.frozen.store(true, Ordering::Relaxed);
        // Slow enough that the build is still mid-flight when frozen: at
        // 1.2 s/op a 64-row build runs well past 2 x IDLE.
        let frozen_mid = Throttle::new(Duration::from_millis(1200));
        let freezer = {
            let f = frozen_mid.clone();
            tokio::spawn(async move {
                // Let it make real progress for ~1.5 idle windows, then stop.
                tokio::time::sleep(Duration::from_millis(IDLE * 1500)).await;
                f.frozen.store(true, Ordering::Relaxed);
                std::time::Instant::now()
            })
        };
        let started = std::time::Instant::now();
        let c_a = claims.remove("slow-a").unwrap();
        let c_b = claims.remove("slow-b").unwrap();
        let c_fs = claims.remove("frozen-start").unwrap();
        let c_fm = claims.remove("frozen-mid").unwrap();
        // A frozen build that is never cancelled would hang here; bound the
        // whole scenario so a regression fails fast instead.
        let (a, b, fs, fm) = tokio::time::timeout(Duration::from_secs(60), async {
            tokio::join!(
                run("slow-a", c_a, slow_a.clone()),
                run("slow-b", c_b, slow_b.clone()),
                run("frozen-start", c_fs, frozen_start.clone()),
                run("frozen-mid", c_fm, frozen_mid.clone()),
            )
        })
        .await
        .expect("a frozen preparation was never cancelled: watchdogs are not isolated");
        let froze_at = freezer.await.unwrap();
        let idle = Duration::from_secs(IDLE);

        // Alive pair: finished, each past the idle window, with real IO.
        for (name, (outcome, elapsed, ops, _)) in [("slow-a", &a), ("slow-b", &b)] {
            assert_eq!(
                *outcome,
                Ok("prepared"),
                "{name} was cancelled: {outcome:?}"
            );
            assert!(
                *elapsed > idle,
                "{name} finished in {elapsed:?}, inside one idle window; vacuous"
            );
            assert!(*ops >= 3, "{name} saw {ops} ops; shim not intercepting");
        }
        assert!(
            b.1 > a.1,
            "the slower build must take longer (a={:?}, b={:?}); else rates are not real",
            a.1,
            b.1
        );
        // Frozen from the start: cancelled in [IDLE, IDLE+4) from test start.
        assert_eq!(
            fs.0,
            Err("maintenance preparation made no progress".to_string())
        );
        assert!(
            fs.1 >= idle && fs.1 < idle + Duration::from_secs(4),
            "frozen-start at {:?}",
            fs.1
        );
        // Frozen midway: cancelled in [IDLE, IDLE+4) from *its own* freeze,
        // not from the start and not from a neighbour's progress.
        assert_eq!(
            fm.0,
            Err("maintenance preparation made no progress".to_string())
        );
        let since_freeze = (started + fm.1).saturating_duration_since(froze_at);
        assert!(
            since_freeze >= idle - Duration::from_millis(500)
                && since_freeze < idle + Duration::from_secs(4),
            "frozen-mid cancelled {since_freeze:?} after its freeze (total {:?})",
            fm.1
        );
        assert!(
            fm.2 >= 2,
            "frozen-mid should have made progress before freezing: {} ops",
            fm.2
        );
        // And the alive builds outlived both cancellations.
        assert!(
            a.1 > fs.1 && b.1 > fs.1,
            "alive builds ended before the frozen one was cancelled"
        );
        for (claim, ok) in [(a.3, true), (b.3, true), (fs.3, false), (fm.3, false)] {
            let result = if ok {
                Ok("prepared".to_string())
            } else {
                Err("stalled".to_string())
            };
            state.task_store.finish(claim, result).await.unwrap();
        }
    }

    /// Review finding 6 on #340: prove the watchdog on a *real* index build
    /// with real storage behaviour, not hand-placed checkpoints.
    ///
    /// (a) Throttled IO: every data/index operation sleeps longer than the
    ///     idle timeout would allow between manual checkpoints, yet because
    ///     each completed operation is progress, the build runs past the idle
    ///     timeout and finishes.
    /// (b) Frozen IO: the same build with storage parked part-way is cancelled
    ///     by the watchdog with the stall error, within [IDLE, IDLE + 4).
    /// (c) Isolation: a second, healthy preparation on another table running
    ///     concurrently with (b) is not cancelled and does not keep (b) alive.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn real_index_build_survives_slow_io_and_is_cancelled_when_frozen() {
        use lance_context_core::merge_write_scope::MergeWriteScope;
        use lance_context_core::preparation_io::test_support::{Throttle, ThrottleWrapper};
        const IDLE: u64 = 3;
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        // Plain local paths take Lance's direct file IO and bypass object-store
        // wrappers; progress there comes from the io_tracker. The
        // `file-object-store` scheme drives the same directory through the
        // ObjectStore API, so the shim and the progress wrapper see every
        // data/index operation exactly as they would on a cloud store, which
        // is the path this test is about.
        cfg.data_dir = format!("file-object-store://{}", dir.path().display());
        cfg.maintenance.index_prepare_targets = vec!["*".into()];
        cfg.maintenance.maintenance_idle_timeout_secs = IDLE;
        cfg.merge_rollout.owned_targets = vec!["slow".into(), "frozen".into(), "healthy".into()];
        let state = MasterState::new(cfg).await.unwrap();
        // Enough rows to force several data-file operations during the build.
        for target in ["slow", "frozen", "healthy"] {
            let mut writer = RolloutStore::open(&state.rollout_uri(target))
                .await
                .unwrap();
            let records: Vec<_> = (0..64)
                .map(|i| rollout_record(&format!("{target}-{i}")))
                .collect();
            writer.add(&records).await.unwrap();
            writer.cleanup_own_shard().await.unwrap();
        }
        let claim_for = |state: Arc<MasterState>, target: &'static str| async move {
            enqueue(&state, TaskKind::IndexId, target).await.unwrap();
            let claim = state
                .task_store
                .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(claim.task.target, target);
            assert!(claim.preparing_maintenance());
            claim
        };
        let prepare =
            |state: Arc<MasterState>, scope: Arc<MergeWriteScope>, target: &'static str| {
                let uri = state.rollout_uri(target);
                async move {
                    scope
                        .run(async {
                            let store = RolloutStore::open_existing_with_options(
                                &uri,
                                state.rollout_store_options(),
                            )
                            .await
                            .map_err(|e| e.to_string())?;
                            store
                                .prepare_id_key_index()
                                .await
                                .map(|_| "prepared")
                                .map_err(|e| e.to_string())
                        })
                        .await
                }
            };

        // (a) Throttled: 1.2 s per data-file op, idle timeout 3 s. The build
        // must take longer than IDLE in total and still complete.
        let claim = claim_for(state.clone(), "slow").await;
        let throttle = Throttle::new(Duration::from_millis(1200));
        let scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
            Arc::new(PreparationOnly),
            Arc::new(ThrottleWrapper(throttle.clone())),
        );
        let began = std::time::Instant::now();
        let outcome = tokio::select! {
            done = prepare(state.clone(), scope.clone(), "slow") => done,
            error = watch_maintenance_preparation(&state, &claim, &scope) => Err(error),
        };
        let elapsed = began.elapsed();
        let ops = throttle.ops.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            ops >= 3,
            "shim did not intercept data IO (ops={ops}); test is vacuous"
        );
        assert_eq!(
            outcome,
            Ok("prepared"),
            "slow-but-alive real build was cancelled"
        );
        assert!(
            elapsed > Duration::from_secs(IDLE),
            "build finished in {elapsed:?}; did not outlive the idle timeout, test is vacuous"
        );
        state
            .task_store
            .finish(claim, Ok("a".into()))
            .await
            .unwrap();

        // (b) + (c): freeze "frozen" after its first data op; run "healthy"
        // concurrently with no throttle. Only "frozen" is cancelled.
        let frozen_claim = claim_for(state.clone(), "frozen").await;
        let healthy_claim = claim_for(state.clone(), "healthy").await;
        let throttle = Throttle::new(Duration::ZERO);
        throttle
            .frozen
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let frozen_scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
            Arc::new(PreparationOnly),
            Arc::new(ThrottleWrapper(throttle.clone())),
        );
        let healthy_scope = MergeWriteScope::with_preparation_authorizer(Arc::new(PreparationOnly));
        let began = std::time::Instant::now();
        let frozen_run = async {
            tokio::select! {
                done = prepare(state.clone(), frozen_scope.clone(), "frozen") => done,
                error = watch_maintenance_preparation(&state, &frozen_claim, &frozen_scope) => Err(error),
            }
        };
        let healthy_run = async {
            // Give the healthy build a slow start so it overlaps the frozen
            // one's stall window rather than finishing instantly.
            tokio::time::sleep(Duration::from_millis(500)).await;
            tokio::select! {
                done = prepare(state.clone(), healthy_scope.clone(), "healthy") => done,
                error = watch_maintenance_preparation(&state, &healthy_claim, &healthy_scope) => Err(error),
            }
        };
        let (frozen_outcome, healthy_outcome) = tokio::join!(frozen_run, healthy_run);
        let elapsed = began.elapsed();
        assert_eq!(
            frozen_outcome,
            Err("maintenance preparation made no progress".to_string())
        );
        assert!(
            elapsed >= Duration::from_secs(IDLE) && elapsed < Duration::from_secs(IDLE + 4),
            "frozen build cancelled after {elapsed:?}"
        );
        assert_eq!(
            healthy_outcome,
            Ok("prepared"),
            "healthy build was collateral damage"
        );
        assert!(
            throttle.ops.load(std::sync::atomic::Ordering::Relaxed) >= 1,
            "frozen shim never saw IO; test is vacuous"
        );
        throttle
            .frozen
            .store(false, std::sync::atomic::Ordering::Relaxed);
        state
            .task_store
            .finish(frozen_claim, Err("stalled".into()))
            .await
            .unwrap();
        state
            .task_store
            .finish(healthy_claim, Ok("h".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn prepared_index_task_runs_with_fenced_publication() {
        for kind in [
            lance_context_core::KeyIndexType::Btree,
            lance_context_core::KeyIndexType::Zonemap,
        ] {
            let dir = TempDir::new().unwrap();
            let mut cfg = config(&dir);
            cfg.maintenance.index_prepare_targets = vec!["exp".into()];
            cfg.merge_rollout.owned_targets = vec!["exp".into()];
            cfg.key_index_type = kind;
            let state = MasterState::new(cfg).await.unwrap();
            let uri = state.rollout_uri("exp");
            let mut writer = RolloutStore::open(&uri).await.unwrap();
            writer.add(&[rollout_record("seed")]).await.unwrap();
            writer.cleanup_own_shard().await.unwrap();
            let task = enqueue(&state, TaskKind::IndexId, "exp").await.unwrap();
            let claim = state
                .task_store
                .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
                .await
                .unwrap()
                .unwrap();
            assert!(claim.preparing_maintenance());
            run_task(&state, claim, TaskClaimTiming::default()).await;
            let status = state.task_store.get(&task.id).await.unwrap().unwrap();
            assert_eq!(status.state, TaskState::Done, "{status:?}");
            assert_eq!(
                status.detail,
                Some(format!(
                    "built {} index on id (prepared outside write lock)",
                    kind.as_str()
                ))
            );
            assert!(state
                .task_store
                .merge_coordinator()
                .get("exp")
                .await
                .unwrap()
                .is_none());
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn resident_merge_waiting_for_memory_keeps_general_tasks_running() {
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.append.rollout_append_local = true;
        cfg.append.rollout_append_targets = vec!["exp".into()];
        cfg.catchup.shards = vec!["historical".into()];
        cfg.merge_wal_concurrency = 1;
        cfg.task_concurrency = 1;
        let state = MasterState::new(cfg).await.unwrap();
        let hot = RolloutStore::open_with_options(
            &state.rollout_uri("exp"),
            lance_context_core::RolloutStoreOptions {
                shard_id: Some("historical".into()),
                merge_after_generations: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        hot.add(&[rollout_record("hot-row")]).await.unwrap();
        hot.flush().await.unwrap();
        let mut cold = RolloutStore::open(&state.rollout_uri("other"))
            .await
            .unwrap();
        cold.add(&[rollout_record("cold-row")]).await.unwrap();
        cold.cleanup_own_shard().await.unwrap();
        let budget = state.local_append.as_ref().unwrap().budget.clone();
        let held = budget.reserve(budget.limit()).await;
        let merge = enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        let merge_poller = spawn_pool_poller(
            state.clone(),
            Arc::new(Semaphore::new(1)),
            TaskKinds::MERGE_WAL,
            false,
        );
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if state
                    .task_store
                    .get(&merge.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .state
                    == TaskState::Running
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        let index = enqueue(&state, TaskKind::IndexId, "other").await.unwrap();
        let general = spawn_pool_poller(
            state.clone(),
            Arc::new(Semaphore::new(1)),
            TaskKinds::GENERAL.without_compact(),
            false,
        );
        let indexed = await_terminal(&state, &index.id).await;
        assert_eq!(indexed.state, TaskState::Done, "{indexed:?}");
        assert_eq!(
            state
                .task_store
                .get(&merge.id)
                .await
                .unwrap()
                .unwrap()
                .state,
            TaskState::Running
        );
        drop(held);
        let merged = await_terminal(&state, &merge.id).await;
        assert_eq!(merged.state, TaskState::Done, "{merged:?}");
        assert_eq!(budget.reserved(), 0);
        assert_eq!(
            lance::Dataset::open(&state.rollout_uri("exp"))
                .await
                .unwrap()
                .count_rows(None)
                .await
                .unwrap(),
            1
        );
        merge_poller.abort();
        general.abort();
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

    /// Wrap test workers in the production ownership wire protocol, preserving
    /// their individual route assertions and injected delays/errors.
    async fn owned_stub(app: axum::Router, cfg: &MasterConfig) -> axum::Router {
        use axum::{
            routing::{get, post},
            Json,
        };
        use lance_context_merge::{Coordinator, Execution};
        use tower::ServiceExt;
        let client = etcd_client::Client::connect(cfg.etcd.etcd_endpoints.clone(), None)
            .await
            .unwrap();
        let coordinator = Coordinator::new(client, cfg.etcd.etcd_prefix.clone());
        let owned_targets = cfg.merge_rollout.owned_targets.clone();
        axum::Router::new()
            .route(
                "/api/v1/internal/merge-executor",
                get(move || {
                    let owned_targets = owned_targets.clone();
                    async move { Json(serde_json::json!({"protocol":2,"instance":"stub","timeout_secs":600,"queue_timeout_secs":600,"idle_timeout_secs":600,"progress_protocol":1,"owned_targets":owned_targets})) }
                }),
            )
            .route(
                "/api/v1/internal/merge-executor/start",
                post(move |Json(execution): Json<Execution>| {
                    let coordinator = coordinator.clone();
                    let app = app.clone();
                    async move {
                        let running = coordinator.start(&execution).await.unwrap().unwrap();
                        tokio::spawn(async move {
                            let path = match execution.target.strip_prefix("generic:") {
                                Some(name) => format!("/api/v1/generic/{name}/merge-wal"),
                                None => format!("/api/v1/internal/merge-wal/{}", execution.target),
                            };
                            let response = app
                                .oneshot(
                                    axum::http::Request::post(path)
                                        .body(axum::body::Body::empty())
                                        .unwrap(),
                                )
                                .await
                                .unwrap();
                            let status = response.status();
                            let body = axum::body::to_bytes(response.into_body(), 65536)
                                .await
                                .unwrap();
                            let outcome = if status.is_success() {
                                Ok(serde_json::from_slice::<serde_json::Value>(&body).unwrap()
                                    ["reclaimed"]
                                    .as_u64()
                                    .unwrap() as usize)
                            } else if status == axum::http::StatusCode::NOT_FOUND {
                                Ok(0)
                            } else {
                                Err(format!("HTTP {status}: {}", String::from_utf8_lossy(&body)))
                            };
                            assert!(coordinator.finish(&running, outcome).await.unwrap());
                        });
                        axum::http::StatusCode::ACCEPTED
                    }
                }),
            )
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn executor_drain_finishes_current_merge_and_leaves_queue_for_other_master() {
        use axum::{routing::post, Json, Router};
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(Semaphore::new(0));
        let app = Router::new().route(
            "/api/v1/internal/merge-wal/{name}",
            post({
                let started = started.clone();
                let release = release.clone();
                move || {
                    let started = started.clone();
                    let release = release.clone();
                    async move {
                        started.notify_one();
                        release.acquire().await.unwrap().forget();
                        Json(serde_json::json!({"reclaimed": 3}))
                    }
                }
            }),
        );
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.merge_wal_concurrency = 1;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        cfg.worker_endpoints = vec![format!("http://{}", listener.local_addr().unwrap())];
        let app = owned_stub(app, &cfg).await;
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = MasterState::new(cfg.clone()).await.unwrap();
        let worker = spawn_scheduler(&state);
        let current = enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        tokio::time::timeout(Duration::from_secs(10), started.notified())
            .await
            .unwrap();
        use tower::ServiceExt;
        let api = crate::routes::api_router().with_state(state.clone());
        let request = |id: &str| {
            axum::http::Request::builder()
                .method("POST")
                .uri("/executor/drain")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({"executor_id":id}).to_string(),
                ))
                .unwrap()
        };
        let wrong = api
            .clone()
            .oneshot(request("previous-process"))
            .await
            .unwrap();
        assert_eq!(wrong.status(), axum::http::StatusCode::CONFLICT);
        assert!(state.admission.status().accepting);
        let id = state.admission.status().executor_id;
        let response = api.oneshot(request(&id)).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let draining = state.admission.status();
        assert!(!draining.accepting);
        assert!(draining.active_operations > 0);
        let next = enqueue(&state, TaskKind::MergeWal, "exp-1").await.unwrap();
        release.add_permits(2);
        tokio::time::timeout(Duration::from_secs(10), state.admission.wait_drained())
            .await
            .unwrap();
        assert_eq!(
            state
                .task_store
                .get(&current.id)
                .await
                .unwrap()
                .unwrap()
                .state,
            TaskState::Done
        );
        assert_eq!(
            state.task_store.get(&next.id).await.unwrap().unwrap().state,
            TaskState::Queued
        );
        assert!(state.admission.try_admit().is_none());
        assert!(crate::scanner::scan_once(&state).await.is_err());

        let other = MasterState::new(cfg).await.unwrap();
        let other_worker = spawn_scheduler(&other);
        assert_eq!(
            await_terminal(&other, &next.id).await.state,
            TaskState::Done
        );
        assert!(state.admission.status().drained());
        worker.abort();
        other_worker.abort();
        server.abort();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn reserved_resident_merge_finishes_while_legacy_pool_is_saturated() {
        use axum::{routing::post, Json, Router};
        use lance_context_core::{RolloutStore, RolloutStoreOptions};
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(Semaphore::new(0));
        let app = Router::new().route(
            "/api/v1/internal/merge-wal/{name}",
            post({
                let started = started.clone();
                let release = release.clone();
                move || {
                    let started = started.clone();
                    let release = release.clone();
                    async move {
                        started.notify_one();
                        release.acquire().await.unwrap().forget();
                        Json(serde_json::json!({"reclaimed": 1}))
                    }
                }
            }),
        );
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.merge_wal_concurrency = 1;
        cfg.append.rollout_append_local = true;
        cfg.append.rollout_append_targets = vec!["exp".into()];
        cfg.catchup.shards = vec!["writer".into()];
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        cfg.worker_endpoints = vec![format!("http://{}", listener.local_addr().unwrap())];
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = MasterState::new(cfg).await.unwrap();
        let writer = RolloutStore::open_with_options(
            &state.rollout_uri("exp"),
            RolloutStoreOptions {
                shard_id: Some("writer".into()),
                merge_after_generations: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let dto = serde_json::from_value(serde_json::json!({
            "id":"one", "rollout_id":"r", "content":"test"
        }))
        .unwrap();
        writer
            .add(&[lance_context_core::rollout_record_from_add_request(&dto)])
            .await
            .unwrap();
        writer.flush().await.unwrap();
        let legacy = enqueue(&state, TaskKind::MergeWal, "legacy-blocked")
            .await
            .unwrap();
        let scheduler = spawn_scheduler(&state);
        tokio::time::timeout(Duration::from_secs(10), started.notified())
            .await
            .unwrap();
        let queued = enqueue(&state, TaskKind::MergeWal, "legacy-queued")
            .await
            .unwrap();
        let resident = enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        let result = await_terminal(&state, &resident.id).await;
        assert_eq!(result.state, TaskState::Done, "{:?}", result.error);
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
        let reader = lance::Dataset::open(&state.rollout_uri("exp"))
            .await
            .unwrap();
        assert_eq!(reader.count_rows(None).await.unwrap(), 1);
        assert_eq!(
            state
                .task_store
                .get(&legacy.id)
                .await
                .unwrap()
                .unwrap()
                .state,
            TaskState::Running
        );
        assert_eq!(
            state
                .task_store
                .get(&queued.id)
                .await
                .unwrap()
                .unwrap()
                .state,
            TaskState::Queued
        );
        // The reserved poller obeys the same joined admission drain.
        state.admission.begin_drain();
        release.add_permits(1);
        tokio::time::timeout(Duration::from_secs(10), state.admission.wait_drained())
            .await
            .unwrap();
        assert_eq!(
            state
                .task_store
                .get(&queued.id)
                .await
                .unwrap()
                .unwrap()
                .state,
            TaskState::Queued
        );
        scheduler.abort();
        server.abort();
    }

    /// MergeWal fans out to every configured worker endpoint and sums the
    /// reclaimed counts. Uses a tiny in-process stub server per "worker".
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn merge_wal_broadcasts_and_sums_reclaimed() {
        use axum::{routing::post, Json, Router};

        // A stub worker that always reports `reclaimed` for any merge call.
        async fn spawn_stub(reclaimed: usize, cfg: &MasterConfig) -> String {
            let app = Router::new().route(
                "/api/v1/internal/merge-wal/{name}",
                post(move || async move { Json(serde_json::json!({ "reclaimed": reclaimed })) }),
            );
            let app = owned_stub(app, cfg).await;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            format!("http://{addr}")
        }

        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.worker_endpoints = vec![spawn_stub(3, &cfg).await, spawn_stub(2, &cfg).await];
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

    /// A generic-store MergeWal task fans out to the generic route on each
    /// worker, not the rollout one. Targets carry the kind as a prefix so the
    /// etcd task schema is unchanged.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn merge_wal_routes_generic_targets_to_the_generic_endpoint() {
        use axum::{extract::Path, routing::post, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let rollout_hits = Arc::new(AtomicUsize::new(0));
        let generic_hits = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route("/api/v1/internal/merge-wal/{name}", {
                let hits = rollout_hits.clone();
                post(move |Path(_name): Path<String>| async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({ "reclaimed": 1 }))
                })
            })
            .route("/api/v1/generic/{name}/merge-wal", {
                let hits = generic_hits.clone();
                post(move |Path(name): Path<String>| async move {
                    assert_eq!(
                        name, "gs",
                        "the bare name reaches the worker, not the prefix"
                    );
                    hits.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({ "reclaimed": 4 }))
                })
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.worker_endpoints = vec![format!("http://{addr}")];
        let app = owned_stub(app, &cfg).await;
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = MasterState::new(cfg).await.unwrap();
        let worker = spawn_scheduler(&state);

        let rec = enqueue(&state, TaskKind::MergeWal, &generic_target("gs"))
            .await
            .unwrap();
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Done, "got {status:?}");
        assert!(status.detail.unwrap().contains("merged 4 generations"));
        assert_eq!(generic_hits.load(Ordering::SeqCst), 1);
        assert_eq!(rollout_hits.load(Ordering::SeqCst), 0);
        worker.abort();
    }

    /// A worker that reclaims nothing plus a worker that errors is a failed
    /// task, not "merged 0 across 1/2 workers": that shape is exactly a store
    /// whose base table is broken (the shard owner 500s, the others have no
    /// data), and the failure cooldown can only act on it if it is reported
    /// as a failure.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn merge_wal_with_an_erroring_worker_and_no_progress_fails() {
        use axum::{http::StatusCode, routing::post, Json, Router};

        async fn spawn(ok: bool, cfg: &MasterConfig) -> String {
            let app = Router::new().route(
                "/api/v1/internal/merge-wal/{name}",
                post(move || async move {
                    if ok {
                        (StatusCode::OK, Json(serde_json::json!({ "reclaimed": 0 })))
                    } else {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({ "error": "Not found: data/frag.lance" })),
                        )
                    }
                }),
            );
            let app = owned_stub(app, cfg).await;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("http://{addr}")
        }

        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.worker_endpoints = vec![spawn(true, &cfg).await, spawn(false, &cfg).await];
        let state = MasterState::new(cfg).await.unwrap();
        let worker = spawn_scheduler(&state);

        let rec = enqueue(&state, TaskKind::MergeWal, "broken").await.unwrap();
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Failed, "got {status:?}");
        assert!(status.error.unwrap().contains("merged 0 generations"));
        assert!(
            state
                .task_store
                .list()
                .await
                .unwrap()
                .iter()
                .all(|task| task.kind != TaskKind::Repair),
            "a merge failure must not schedule destructive fragment removal"
        );
        assert!(
            !state
                .task_store
                .is_cooling_down(TaskKind::MergeWal, "broken")
                .await
                .unwrap(),
            "one broken shard must not suppress healthy shard scheduling"
        );
        let (failures, _) = state
            .task_store
            .merge_coordinator()
            .failure_page(None, 256)
            .await
            .unwrap();
        assert_eq!(failures.len(), 1);
        assert!(failures[0].needs_attention);
        worker.abort();
    }

    /// Workers are called one at a time. Every worker's merge commits to the
    /// same base table and Lance's commit-conflict retry gives up after 30s, so
    /// concurrent fan-out is N writers racing one commit point -- the failure
    /// the master exists to prevent.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn merge_wal_calls_workers_serially() {
        use axum::{routing::post, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Each stub records the max number of in-flight calls it observed
        // across the fleet through a shared counter.
        let inflight = Arc::new(AtomicUsize::new(0));
        let max_inflight = Arc::new(AtomicUsize::new(0));
        async fn spawn_stub(
            inflight: Arc<AtomicUsize>,
            max_inflight: Arc<AtomicUsize>,
            cfg: &MasterConfig,
        ) -> String {
            let app = Router::new().route(
                "/api/v1/internal/merge-wal/{name}",
                post(move || {
                    let inflight = inflight.clone();
                    let max_inflight = max_inflight.clone();
                    async move {
                        let now = inflight.fetch_add(1, Ordering::SeqCst) + 1;
                        max_inflight.fetch_max(now, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(150)).await;
                        inflight.fetch_sub(1, Ordering::SeqCst);
                        Json(serde_json::json!({ "reclaimed": 1 }))
                    }
                }),
            );
            let app = owned_stub(app, cfg).await;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("http://{addr}")
        }

        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.worker_endpoints = vec![
            spawn_stub(inflight.clone(), max_inflight.clone(), &cfg).await,
            spawn_stub(inflight.clone(), max_inflight.clone(), &cfg).await,
            spawn_stub(inflight.clone(), max_inflight.clone(), &cfg).await,
        ];
        let state = MasterState::new(cfg).await.unwrap();
        let worker = spawn_scheduler(&state);

        let rec = enqueue(&state, TaskKind::MergeWal, "exp").await.unwrap();
        let status = await_terminal(&state, &rec.id).await;
        assert_eq!(status.state, TaskState::Done, "got {status:?}");
        assert!(status.detail.unwrap().contains("3/3 workers"));
        assert_eq!(
            max_inflight.load(Ordering::SeqCst),
            1,
            "fan-out must never have more than one worker merging at a time"
        );
        worker.abort();
    }

    /// The compaction sweep must not enqueue generic rows: the master does not
    /// compact generic stores, and `run_compaction` would otherwise open the
    /// wrong URI through the rollout code path.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn compaction_sweep_skips_generic_rows() {
        use crate::stats_store::StatRow;

        let dir = TempDir::new().unwrap();
        let state = MasterState::new(config(&dir)).await.unwrap();
        let seed = |name: String, uri: String| StatRow {
            version: StatRow::UNKNOWN_VERSION,
            name,
            uri,
            row_count: 0,
            fragment_count: 50,
            last_updated: 0,
            pending_wal_generations: 0,
            last_compaction: StatRow::NO_COMPACTION,
            total_compactions: 0,
            scanned_at: 0,
        };
        {
            let mut stats = state.stats.lock().await;
            stats
                .upsert(&seed("r".to_string(), state.rollout_uri("r")))
                .await
                .unwrap();
            stats
                .upsert(&seed(generic_target("g"), state.generic_uri("g")))
                .await
                .unwrap();
        }
        let queued = sweep_candidates(&state).await.unwrap();
        assert_eq!(queued, 1, "only the rollout row is compacted");
        let tasks = state.task_store.list().await.unwrap();
        assert!(tasks.iter().all(|t| t.target == "r"), "{tasks:?}");
    }

    /// A `Compact` runs even while every `MergeWal` slot is occupied by slow
    /// fan-outs and the queue is dominated by MergeWal tasks.
    ///
    /// This is the end-to-end form of the starvation bug: the merge pool is
    /// saturated by stub workers that never return, and 20 MergeWal tasks are
    /// enqueued *ahead* of the Compact. Before the split-poller fix the single
    /// dispatch loop kept claiming MergeWal tasks (parking them on the merge
    /// semaphore while they held their locks) and the trailing Compact was
    /// never reached, so this test would hang to its timeout.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn compact_runs_while_merge_wal_pool_is_saturated() {
        use axum::{routing::post, Router};

        // A worker that accepts the merge call and then never responds, so the
        // MergeWal task occupies its slot for the duration of the test.
        let hang = Router::new().route(
            "/api/v1/internal/merge-wal/{name}",
            post(|| async {
                std::future::pending::<()>().await;
                String::new()
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.worker_endpoints = vec![format!("http://{addr}")];
        cfg.merge_wal_concurrency = 2;
        cfg.task_concurrency = 2;
        let hang = owned_stub(hang, &cfg).await;
        tokio::spawn(async move { axum::serve(listener, hang).await.unwrap() });
        let state = MasterState::new(cfg).await.unwrap();

        // Build a compactable store before the dispatcher starts.
        let name = "starved";
        let uri = state.rollout_uri(name);
        {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            for i in 0..4 {
                let rec = rollout_record(&format!("r{i}"));
                store.add(&[rec]).await.unwrap();
                store.cleanup_own_shard().await.unwrap();
            }
        }
        state.registry.upsert(name, &uri).await.unwrap();
        crate::scanner::scan_once(&state).await.unwrap();

        // Saturate and over-fill the merge queue *before* the Compact.
        for i in 0..20 {
            enqueue(&state, TaskKind::MergeWal, &format!("exp-{i}"))
                .await
                .unwrap();
        }
        let compact = enqueue(&state, TaskKind::Compact, name).await.unwrap();

        let worker = spawn_scheduler(&state);

        let status = await_terminal(&state, &compact.id).await;
        assert_eq!(
            status.state,
            TaskState::Done,
            "Compact must drain while MergeWal saturates its own pool: {status:?}"
        );

        // Sanity: the merge backlog really is still stuck, i.e. the Compact did
        // not simply win because the MergeWal tasks all completed.
        let stuck = state
            .task_store
            .list()
            .await
            .unwrap()
            .into_iter()
            .filter(|t| {
                t.kind == TaskKind::MergeWal
                    && matches!(t.state, TaskState::Queued | TaskState::Running)
            })
            .count();
        assert!(
            stuck > 0,
            "test must leave MergeWal work outstanding, else it proves nothing"
        );

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
        state.registry.upsert(name, &uri).await.unwrap();
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

    /// Non-merge maintenance retains target-wide cooldown. Merge failures now
    /// use the independent per-shard circuit tested in merge_execution.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn repeated_compaction_failures_cool_the_target_down_and_sweeps_skip_it() {
        use crate::stats_store::StatRow;

        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.min_fragments = 1;
        cfg.task_cooldown_after_failures = 2;
        cfg.task_cooldown_base_secs = 3600;
        // Missing base dataset: every compaction fails immediately.
        cfg.worker_endpoints = vec![];
        let state = MasterState::new(cfg).await.unwrap();
        let worker = spawn_scheduler(&state);

        {
            let mut stats = state.stats.lock().await;
            stats
                .upsert(&StatRow {
                    version: StatRow::UNKNOWN_VERSION,
                    name: "broken".to_string(),
                    uri: state.rollout_uri("broken"),
                    row_count: 0,
                    fragment_count: 50,
                    last_updated: 0,
                    pending_wal_generations: 50,
                    last_compaction: StatRow::NO_COMPACTION,
                    total_compactions: 0,
                    scanned_at: 0,
                })
                .await
                .unwrap();
        }

        // Two sweeps, two failures: the target crosses the threshold.
        for round in 0..2 {
            assert!(
                !state
                    .task_store
                    .is_cooling_down(TaskKind::Compact, "broken")
                    .await
                    .unwrap(),
                "round {round}: must not cool down below the failure threshold"
            );
            assert_eq!(sweep_candidates(&state).await.unwrap(), 1);
            let id = state
                .task_store
                .list()
                .await
                .unwrap()
                .into_iter()
                .filter(|t| t.kind == TaskKind::Compact && t.target == "broken")
                .max_by_key(|t| t.enqueued_at)
                .unwrap()
                .id;
            assert_eq!(await_terminal(&state, &id).await.state, TaskState::Failed);
        }
        // `finish` commits the task's terminal state first and then records the
        // failure; `await_terminal` returns on the former, so give the latter a
        // moment.
        let mut cooling = false;
        for _ in 0..40 {
            if state
                .task_store
                .is_cooling_down(TaskKind::Compact, "broken")
                .await
                .unwrap()
            {
                cooling = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(cooling, "target must be cooling down after 2 failures");
        let cooldowns = state.task_store.list_cooldowns().await.unwrap();
        assert_eq!(cooldowns.len(), 1);
        assert_eq!(cooldowns[0].target, "broken");
        assert_eq!(cooldowns[0].failures, 2);
        assert!(cooldowns[0].until_ms.is_some());

        // The sweep now skips it.
        assert_eq!(
            sweep_candidates(&state).await.unwrap(),
            0,
            "a cooled-down target must not be re-enqueued by the sweep"
        );

        // A manual enqueue is not gated.
        let manual = enqueue(&state, TaskKind::Compact, "broken").await.unwrap();
        assert_eq!(manual.target, "broken");
        assert_eq!(
            await_terminal(&state, &manual.id).await.state,
            TaskState::Failed
        );

        // The count keeps climbing across windows: that third failure (the
        // manual one) is recorded as failure 3, not a fresh 1, so the window
        // doubles. This is what stops every master re-probing the target the
        // instant a window closes.
        let mut cd = None;
        for _ in 0..40 {
            let cs = state.task_store.list_cooldowns().await.unwrap();
            if let Some(c) = cs.iter().find(|c| c.target == "broken" && c.failures >= 3) {
                cd = Some(c.clone());
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let cd = cd.expect("cooldown record must carry the cumulative count");
        assert_eq!(cd.failures, 3);
        let window_ms = cd.until_ms.unwrap() - chrono::Utc::now().timestamp_millis();
        assert!(
            window_ms > 3600 * 1000 * 3 / 2,
            "third failure must get a doubled (2h) window, got {}s",
            window_ms / 1000
        );

        worker.abort();
    }

    /// WAL sweeps enqueue over-threshold targets and dedupe a second sweep.
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
                                                           // A generic store over threshold is swept too, as a generic task.
            let mut g = seed(&generic_target("ghot"), 7);
            g.uri = state.generic_uri("ghot");
            stats.upsert(&g).await.unwrap();
        }

        let queued = sweep_merge_wal_candidates(&state).await.unwrap();
        assert_eq!(
            queued, 2,
            "both over-threshold stores are swept, rollout and generic"
        );

        let tasks = state.task_store.list().await.unwrap();
        let merge_tasks: Vec<_> = tasks
            .iter()
            .filter(|t| t.kind == TaskKind::MergeWal)
            .collect();
        assert_eq!(merge_tasks.len(), 2);
        let mut targets: Vec<&str> = merge_tasks.iter().map(|t| t.target.as_str()).collect();
        targets.sort();
        assert_eq!(targets, vec!["generic:ghot", "hot"]);

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
        assert_eq!(merge_after, 2, "duplicate MergeWal is de-duped");
        // And a second sweep reports 0 new tasks, not 2 no-ops.
        assert_eq!(sweep_merge_wal_candidates(&state).await.unwrap(), 0);
    }

    /// The per-tick cap bounds new tasks, not candidates: stores that already
    /// have a task in flight must not eat the budget and starve the rest.
    #[tokio::test]
    #[ignore = "needs etcd (ETCD_TEST_ENDPOINTS)"]
    async fn sweep_budget_is_spent_on_new_tasks_not_deduped_ones() {
        use crate::stats_store::StatRow;
        let dir = TempDir::new().unwrap();
        let mut cfg = config(&dir);
        cfg.merge_wal_min_generations = 1;
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
            // MAX_SWEEP_ENQUEUE stores with the biggest backlog...
            for i in 0..MAX_SWEEP_ENQUEUE {
                stats
                    .upsert(&seed(&format!("big-{i:03}"), 10_000))
                    .await
                    .unwrap();
            }
            // ...and one smaller store ranked below all of them.
            stats.upsert(&seed("small", 500)).await.unwrap();
        }
        // First sweep fills the cap with the big stores.
        assert_eq!(
            sweep_merge_wal_candidates(&state).await.unwrap(),
            MAX_SWEEP_ENQUEUE
        );
        // Second sweep: every big store dedupes; the budget must reach "small".
        assert_eq!(sweep_merge_wal_candidates(&state).await.unwrap(), 1);
        assert!(state
            .task_store
            .list()
            .await
            .unwrap()
            .iter()
            .any(|t| t.kind == TaskKind::MergeWal && t.target == "small"));
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
