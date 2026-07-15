//! Admin JSON API routes (PR A: read-only observability endpoints).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use lance_context_api::{
    CompactJobStatus, EnqueueTaskRequest, ExperimentDetail, ExperimentListResponse,
    ExperimentSummary, TaskKind, TaskListResponse, TaskRecord, TaskState,
};

use crate::error::MasterError;
use crate::scanner;
use crate::scheduler;
use crate::state::MasterState;

/// Query params for the experiment listing.
#[derive(Debug, Deserialize)]
pub struct ListParams {
    #[serde(default)]
    pub search: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    50
}

/// Query params for the detail endpoint.
#[derive(Debug, Deserialize)]
pub struct DetailParams {
    /// When true, open the dataset and recompute metrics on-demand rather than
    /// returning the last scanned snapshot.
    #[serde(default)]
    pub fresh: bool,
}

/// `GET /api/v1/experiments`
pub async fn list_experiments(
    State(state): State<Arc<MasterState>>,
    Query(params): Query<ListParams>,
) -> Result<Json<ExperimentListResponse>, MasterError> {
    let search = params.search.as_deref().filter(|s| !s.is_empty());
    let stats = state.stats.lock().await;
    let total = stats.count(search).await.map_err(MasterError::from_lance)?;
    let rows = stats
        .list(search, params.limit, params.offset)
        .await
        .map_err(MasterError::from_lance)?;
    let experiments: Vec<ExperimentSummary> = rows.into_iter().map(|r| r.into_summary()).collect();
    Ok(Json(ExperimentListResponse { experiments, total }))
}

/// `GET /api/v1/experiments/{name}`
pub async fn get_experiment(
    State(state): State<Arc<MasterState>>,
    Path(name): Path<String>,
    Query(params): Query<DetailParams>,
) -> Result<Json<ExperimentDetail>, MasterError> {
    if params.fresh {
        // Recompute this one experiment now, updating the stats table as a
        // side effect, then read it back.
        let entries = state
            .registry
            .write()
            .await
            .list()
            .await
            .map_err(MasterError::from_lance)?;
        if !entries.iter().any(|e| e.name == name) {
            return Err(MasterError::NotFound(format!(
                "experiment '{}' does not exist",
                name
            )));
        }
        scanner::scan_once(&state)
            .await
            .map_err(MasterError::from_lance)?;
    }
    let stats = state.stats.lock().await;
    match stats.get(&name).await.map_err(MasterError::from_lance)? {
        Some(row) => Ok(Json(ExperimentDetail {
            summary: row.into_summary(),
        })),
        None => Err(MasterError::NotFound(format!(
            "experiment '{}' not found in stats",
            name
        ))),
    }
}

/// `POST /api/v1/rescan` — trigger one immediate full scan.
pub async fn rescan(
    State(state): State<Arc<MasterState>>,
) -> Result<Json<serde_json::Value>, MasterError> {
    let n = scanner::scan_once(&state)
        .await
        .map_err(MasterError::from_lance)?;
    Ok(Json(serde_json::json!({ "scanned": n })))
}

/// Map a task's terminal `detail` string back into the fragment counts the
/// legacy `CompactJobStatus::Done` shape carries. Best-effort: unparsable
/// details yield zeros (the UI only needs the state to stop polling).
fn parse_fragment_detail(detail: Option<&str>) -> (usize, usize) {
    // Detail format: "removed {r} / added {a} fragments".
    let parse = |s: &str, key: &str| -> usize {
        s.split_whitespace()
            .skip_while(|w| *w != key)
            .nth(1)
            .and_then(|w| w.parse().ok())
            .unwrap_or(0)
    };
    match detail {
        Some(d) => (parse(d, "removed"), parse(d, "added")),
        None => (0, 0),
    }
}

/// Project a [`TaskRecord`] into the legacy [`CompactJobStatus`] shape so the
/// existing compaction UI keeps working while the queue view rolls out.
fn task_to_compact_status(task: &TaskRecord) -> CompactJobStatus {
    match task.state {
        TaskState::Queued => CompactJobStatus::Queued,
        TaskState::Running => CompactJobStatus::Running,
        TaskState::Done => {
            let (fragments_removed, fragments_added) = parse_fragment_detail(task.detail.as_deref());
            CompactJobStatus::Done {
                fragments_removed,
                fragments_added,
            }
        }
        TaskState::Failed => CompactJobStatus::Failed {
            error: task.error.clone().unwrap_or_default(),
        },
    }
}

/// The most recent `Compact` task for `name`, if any (by `enqueued_at`).
async fn latest_compact_task(state: &Arc<MasterState>, name: &str) -> Option<TaskRecord> {
    state
        .tasks
        .lock()
        .await
        .values()
        .filter(|t| t.kind == TaskKind::Compact && t.target == name)
        .max_by_key(|t| t.enqueued_at)
        .cloned()
}

/// `POST /api/v1/experiments/{name}/compact` — enqueue a manual compaction.
/// Returns 202 Accepted with the (possibly de-duped) job status. Retained for
/// backward compatibility; internally this is just a `Compact` task.
pub async fn compact_experiment(
    State(state): State<Arc<MasterState>>,
    Path(name): Path<String>,
) -> Result<(StatusCode, Json<CompactJobStatus>), MasterError> {
    // Only enqueue known experiments.
    let exists = state
        .registry
        .write()
        .await
        .contains(&name)
        .await
        .map_err(MasterError::from_lance)?;
    if !exists {
        return Err(MasterError::NotFound(format!(
            "experiment '{}' does not exist",
            name
        )));
    }
    let task = scheduler::enqueue(&state, TaskKind::Compact, &name).await;
    Ok((StatusCode::ACCEPTED, Json(task_to_compact_status(&task))))
}

/// `GET /api/v1/experiments/{name}/compact/status` — latest compaction job
/// state for an experiment (`None` when never requested).
pub async fn compact_status(
    State(state): State<Arc<MasterState>>,
    Path(name): Path<String>,
) -> Json<CompactJobStatus> {
    match latest_compact_task(&state, &name).await {
        Some(task) => Json(task_to_compact_status(&task)),
        None => Json(CompactJobStatus::None),
    }
}

/// `POST /api/v1/tasks` — enqueue a task of any kind. Returns 202 Accepted with
/// the (possibly de-duped) task record.
pub async fn enqueue_task(
    State(state): State<Arc<MasterState>>,
    Json(req): Json<EnqueueTaskRequest>,
) -> Result<(StatusCode, Json<TaskRecord>), MasterError> {
    let exists = state
        .registry
        .write()
        .await
        .contains(&req.target)
        .await
        .map_err(MasterError::from_lance)?;
    if !exists {
        return Err(MasterError::NotFound(format!(
            "experiment '{}' does not exist",
            req.target
        )));
    }
    let task = scheduler::enqueue(&state, req.kind, &req.target).await;
    Ok((StatusCode::ACCEPTED, Json(task)))
}

/// `GET /api/v1/tasks` — all tasks (queue + recent history), newest first.
pub async fn list_tasks(State(state): State<Arc<MasterState>>) -> Json<TaskListResponse> {
    let mut tasks: Vec<TaskRecord> = state.tasks.lock().await.values().cloned().collect();
    tasks.sort_by_key(|t| std::cmp::Reverse(t.enqueued_at));
    Json(TaskListResponse { tasks })
}

/// `GET /api/v1/tasks/{id}` — a single task by id.
pub async fn get_task(
    State(state): State<Arc<MasterState>>,
    Path(id): Path<String>,
) -> Result<Json<TaskRecord>, MasterError> {
    match state.tasks.lock().await.get(&id).cloned() {
        Some(task) => Ok(Json(task)),
        None => Err(MasterError::NotFound(format!("task '{}' not found", id))),
    }
}

/// Build the admin API router (mounted under `/api/v1`).
pub fn api_router() -> Router<Arc<MasterState>> {
    Router::new()
        .route("/experiments", get(list_experiments))
        .route("/experiments/{name}", get(get_experiment))
        .route("/experiments/{name}/compact", post(compact_experiment))
        .route("/experiments/{name}/compact/status", get(compact_status))
        .route("/tasks", post(enqueue_task).get(list_tasks))
        .route("/tasks/{id}", get(get_task))
        .route("/rescan", post(rescan))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MasterConfig;
    use lance_context_core::{RolloutRegistry, RolloutStore};
    use tempfile::TempDir;

    fn test_config(dir: &TempDir) -> MasterConfig {
        MasterConfig {
            data_dir: dir.path().to_string_lossy().to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            stats_scan_interval_secs: 0,
            scan_concurrency: 4,
            compaction_interval_secs: 0,
            min_fragments: 16,
            target_rows_per_fragment: 1_048_576,
            worker_endpoints: vec![],
            task_concurrency: 4,
            ui_dir: None,
        }
    }

    /// Create N experiments via core, register them, scan, and assert the stats
    /// table + list endpoint reflect them.
    #[tokio::test]
    async fn scan_populates_and_list_returns_experiments() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();

        for i in 0..3 {
            let name = format!("exp-{i}");
            let uri = state.rollout_uri(&name);
            // Creating the store materializes an (empty) base table on disk.
            RolloutStore::open(&uri).await.unwrap();
            state
                .registry
                .write()
                .await
                .upsert(&name, &uri)
                .await
                .unwrap();
        }

        let scanned = scanner::scan_once(&state).await.unwrap();
        assert_eq!(scanned, 3);

        let Json(resp) = list_experiments(
            State(state.clone()),
            Query(ListParams {
                search: None,
                limit: 50,
                offset: 0,
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp.total, 3);
        assert_eq!(resp.experiments.len(), 3);
        assert_eq!(resp.experiments[0].name, "exp-0");

        // Search narrows.
        let Json(one) = list_experiments(
            State(state.clone()),
            Query(ListParams {
                search: Some("exp-1".to_string()),
                limit: 50,
                offset: 0,
            }),
        )
        .await
        .unwrap();
        assert_eq!(one.total, 1);
        assert_eq!(one.experiments[0].name, "exp-1");
    }

    #[tokio::test]
    async fn scan_sees_registry_commits_from_another_handle() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();
        let name = "external";
        let uri = state.rollout_uri(name);
        RolloutStore::open(&uri).await.unwrap();

        let registry_uri = dir.path().join("_registry.rollout.lance");
        let mut worker_registry =
            RolloutRegistry::open_or_create(registry_uri.to_str().unwrap(), None)
                .await
                .unwrap();
        worker_registry.upsert(name, &uri).await.unwrap();

        let scanned = scanner::scan_once(&state).await.unwrap();
        assert_eq!(scanned, 1);
        assert!(state.stats.lock().await.get(name).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn scan_reconciles_removed_experiments() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();

        let name = "gone";
        let uri = state.rollout_uri(name);
        RolloutStore::open(&uri).await.unwrap();
        state
            .registry
            .write()
            .await
            .upsert(name, &uri)
            .await
            .unwrap();
        scanner::scan_once(&state).await.unwrap();
        assert!(state.stats.lock().await.get(name).await.unwrap().is_some());

        // Remove from registry -> next scan drops the stats row.
        state.registry.write().await.remove(name).await.unwrap();
        scanner::scan_once(&state).await.unwrap();
        assert!(state.stats.lock().await.get(name).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn get_experiment_not_found() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();
        let res = get_experiment(
            State(state),
            Path("missing".to_string()),
            Query(DetailParams { fresh: false }),
        )
        .await;
        assert!(matches!(res, Err(MasterError::NotFound(_))));
    }
}
