//! Admin JSON API routes (PR A: read-only observability endpoints).

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use lance_context_api::{
    CompactJobStatus, ExperimentDetail, ExperimentListResponse, ExperimentRecordsResponse,
    ExperimentSummary,
};
use lance_context_core::{
    rollout_record_to_dto, RolloutFilters, RolloutStore, RolloutStoreOptions,
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

fn default_records_limit() -> usize {
    25
}

/// Query params for the detail endpoint.
#[derive(Debug, Deserialize)]
pub struct DetailParams {
    /// When true, open the dataset and recompute metrics on-demand rather than
    /// returning the last scanned snapshot.
    #[serde(default)]
    pub fresh: bool,
}

/// Query params for server-side rollout record filtering and pagination.
#[derive(Debug, Deserialize)]
pub struct RecordListParams {
    #[serde(default = "default_records_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub rollout_id: Option<String>,
    #[serde(default)]
    pub problem_id: Option<String>,
    #[serde(default)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content_type: Option<String>,
    #[serde(default)]
    pub policy_version: Option<String>,
    #[serde(default)]
    pub artifact_type: Option<String>,
    #[serde(default)]
    pub include_in_training: Option<bool>,
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

/// `GET /api/v1/experiments/{name}/records`
pub async fn list_experiment_records(
    State(state): State<Arc<MasterState>>,
    Path(name): Path<String>,
    Query(params): Query<RecordListParams>,
) -> Result<Json<ExperimentRecordsResponse>, MasterError> {
    let store = open_registered_store(&state, &name).await?;
    let limit = params.limit.clamp(1, 100);
    let filters = RolloutFilters {
        id: non_empty(params.id),
        rollout_id: non_empty(params.rollout_id),
        problem_id: non_empty(params.problem_id),
        dataset: non_empty(params.dataset),
        role: non_empty(params.role),
        content_type: non_empty(params.content_type),
        policy_version: non_empty(params.policy_version),
        artifact_type: non_empty(params.artifact_type),
        include_in_training: params.include_in_training,
    };
    let page = store
        .list_filtered(&filters, limit, params.offset)
        .await
        .map_err(MasterError::from_lance)?;

    Ok(Json(ExperimentRecordsResponse {
        records: page
            .records
            .into_iter()
            .map(rollout_record_to_dto)
            .collect(),
        has_more: page.has_more,
        limit,
        offset: params.offset,
    }))
}

/// `GET /api/v1/experiments/{name}/records/{id}/blob`
pub async fn download_experiment_blob(
    State(state): State<Arc<MasterState>>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Response, MasterError> {
    let store = open_registered_store(&state, &name).await?;
    let record = store
        .get_by_id(&id)
        .await
        .map_err(MasterError::from_lance)?
        .ok_or_else(|| MasterError::NotFound(format!("record '{}' does not exist", id)))?;
    let bytes = store
        .get_blob(&id)
        .await
        .map_err(MasterError::from_lance)?
        .ok_or_else(|| MasterError::NotFound(format!("record '{}' has no blob", id)))?;

    let content_type = HeaderValue::from_str(&record.content_type)
        .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream"));
    let filename = blob_filename(&record);
    let disposition = HeaderValue::from_str(&format!("attachment; filename=\"{filename}\""))
        .map_err(|err| MasterError::Internal(err.to_string()))?;

    Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_DISPOSITION, disposition)
        .body(Body::from(bytes))
        .map_err(|err| MasterError::Internal(err.to_string()))
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
        .route("/experiments/{name}/records", get(list_experiment_records))
        .route(
            "/experiments/{name}/records/{id}/blob",
            get(download_experiment_blob),
        )
        .route("/experiments/{name}/compact", post(compact_experiment))
        .route("/experiments/{name}/compact/status", get(compact_status))
        .route("/rescan", post(rescan))
}

async fn open_registered_store(
    state: &MasterState,
    name: &str,
) -> Result<RolloutStore, MasterError> {
    let entry = state
        .registry
        .write()
        .await
        .get(name)
        .await
        .map_err(MasterError::from_lance)?
        .ok_or_else(|| MasterError::NotFound(format!("experiment '{}' does not exist", name)))?;
    RolloutStore::open_existing_with_options(&entry.uri, RolloutStoreOptions::default())
        .await
        .map_err(MasterError::from_lance)
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.is_empty())
}

fn blob_filename(record: &lance_context_core::RolloutRecord) -> String {
    let candidate = record
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("filename"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(&record.id);
    let sanitized: String = candidate
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .take(180)
        .collect();
    if sanitized.is_empty() {
        "rollout-blob".to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MasterConfig;
    use chrono::Utc;
    use lance_context_core::{RolloutRecord, RolloutRegistry, RolloutStore};
    use serde_json::json;
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

    fn test_record(id: &str, with_blob: bool) -> RolloutRecord {
        let bytes = with_blob.then(|| b"master-blob".to_vec());
        RolloutRecord {
            id: id.to_string(),
            rollout_id: "rollout-1".to_string(),
            problem_id: "problem-1".to_string(),
            dataset: Some("gsm8k".to_string()),
            sequence_order: 1,
            role: if with_blob { "artifact" } else { "assistant" }.to_string(),
            created_at: Utc::now(),
            content: Some("answer".to_string()),
            content_type: if with_blob { "image/png" } else { "text/plain" }.to_string(),
            input_tokens: None,
            output_tokens: Some(vec![1, 2]),
            num_input_tokens: None,
            num_output_tokens: Some(2),
            output_logprobs: None,
            input_logprobs: None,
            ref_logprobs: None,
            loss_mask: None,
            advantage: None,
            reward: Some(1.0),
            raw_reward: None,
            grader_id: None,
            score: Some(0.9),
            include_in_training: Some(true),
            exclude_reason: None,
            policy_version: Some("policy-a".to_string()),
            relationships: Vec::new(),
            payload_size: bytes.as_ref().map(|bytes| bytes.len() as i64),
            binary_payload: bytes,
            payload_checksum: None,
            artifact_type: with_blob.then(|| "screenshot".to_string()),
            metadata: with_blob.then(|| json!({"filename": "grade result.png"})),
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

    #[tokio::test]
    async fn records_endpoint_filters_pages_and_rejects_unknown_experiment() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();
        let uri = state.rollout_uri("records");
        let mut store = RolloutStore::open(&uri).await.unwrap();
        store
            .add(&[
                test_record("assistant-1", false),
                test_record("artifact-1", true),
            ])
            .await
            .unwrap();
        state
            .registry
            .write()
            .await
            .upsert("records", &uri)
            .await
            .unwrap();

        let Json(page) = list_experiment_records(
            State(state.clone()),
            Path("records".to_string()),
            Query(RecordListParams {
                limit: 1,
                offset: 0,
                id: None,
                rollout_id: Some("rollout-1".to_string()),
                problem_id: None,
                dataset: Some("gsm8k".to_string()),
                role: Some("artifact".to_string()),
                content_type: None,
                policy_version: Some("policy-a".to_string()),
                artifact_type: Some("screenshot".to_string()),
                include_in_training: Some(true),
            }),
        )
        .await
        .unwrap();
        assert!(!page.has_more);
        assert_eq!(page.limit, 1);
        assert_eq!(page.records[0].id, "artifact-1");
        assert!(page.records[0].binary_payload.is_none());

        let missing = list_experiment_records(
            State(state),
            Path("missing".to_string()),
            Query(RecordListParams {
                limit: 25,
                offset: 0,
                id: None,
                rollout_id: None,
                problem_id: None,
                dataset: None,
                role: None,
                content_type: None,
                policy_version: None,
                artifact_type: None,
                include_in_training: None,
            }),
        )
        .await;
        assert!(matches!(missing, Err(MasterError::NotFound(_))));
    }

    #[tokio::test]
    async fn blob_endpoint_sets_download_headers_and_returns_bytes() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();
        let uri = state.rollout_uri("blobs");
        let mut store = RolloutStore::open(&uri).await.unwrap();
        store
            .add(&[
                test_record("artifact-1", true),
                test_record("assistant-1", false),
            ])
            .await
            .unwrap();
        state
            .registry
            .write()
            .await
            .upsert("blobs", &uri)
            .await
            .unwrap();

        let response = download_experiment_blob(
            State(state.clone()),
            Path(("blobs".to_string(), "artifact-1".to_string())),
        )
        .await
        .unwrap();
        assert_eq!(response.headers()[header::CONTENT_TYPE], "image/png");
        assert_eq!(
            response.headers()[header::CONTENT_DISPOSITION],
            "attachment; filename=\"grade_result.png\""
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&bytes[..], b"master-blob");

        let no_blob = download_experiment_blob(
            State(state),
            Path(("blobs".to_string(), "assistant-1".to_string())),
        )
        .await;
        assert!(matches!(no_blob, Err(MasterError::NotFound(_))));
    }
}
