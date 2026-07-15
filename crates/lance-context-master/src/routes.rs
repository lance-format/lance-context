//! Admin JSON API routes (PR A: read-only observability endpoints).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use lance_context_api::{
    CompactJobStatus, ExperimentDetail, ExperimentListResponse, ExperimentSummary,
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
        let entry = state
            .registry
            .write()
            .await
            .get(&name)
            .await
            .map_err(MasterError::from_lance)?
            .ok_or_else(|| {
                MasterError::NotFound(format!("experiment '{}' does not exist", name))
            })?;
        scanner::refresh_one(&state, &entry.name, &entry.uri)
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

/// `POST /api/v1/experiments/{name}/compact` — enqueue a manual compaction.
/// Returns 202 Accepted with the (possibly de-duped) job status.
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
    let status = scheduler::enqueue(&state, &name).await;
    Ok((StatusCode::ACCEPTED, Json(status)))
}

/// `GET /api/v1/experiments/{name}/compact/status` — latest compaction job
/// state for an experiment (`None` when never requested).
pub async fn compact_status(
    State(state): State<Arc<MasterState>>,
    Path(name): Path<String>,
) -> Json<CompactJobStatus> {
    let status = state
        .jobs
        .lock()
        .await
        .get(&name)
        .cloned()
        .unwrap_or(CompactJobStatus::None);
    Json(status)
}

/// Build the admin API router (mounted under `/api/v1`).
pub fn api_router() -> Router<Arc<MasterState>> {
    Router::new()
        .route("/experiments", get(list_experiments))
        .route("/experiments/{name}", get(get_experiment))
        .route("/experiments/{name}/compact", post(compact_experiment))
        .route("/experiments/{name}/compact/status", get(compact_status))
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

    #[tokio::test]
    async fn fresh_detail_refreshes_only_requested_experiment() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();

        for name in ["target", "other"] {
            let uri = state.rollout_uri(name);
            RolloutStore::open(&uri).await.unwrap();
            state
                .registry
                .write()
                .await
                .upsert(name, &uri)
                .await
                .unwrap();
        }

        let Json(detail) = get_experiment(
            State(state.clone()),
            Path("target".to_string()),
            Query(DetailParams { fresh: true }),
        )
        .await
        .unwrap();
        assert_eq!(detail.summary.name, "target");
        assert!(state
            .stats
            .lock()
            .await
            .get("target")
            .await
            .unwrap()
            .is_some());
        assert!(state
            .stats
            .lock()
            .await
            .get("other")
            .await
            .unwrap()
            .is_none());
    }
}
