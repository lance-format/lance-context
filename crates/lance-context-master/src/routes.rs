//! Admin JSON API routes (PR A: read-only observability endpoints).

use std::collections::HashSet;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use lance_context_api::{
    CompactJobStatus, EnqueueTaskRequest, ExperimentDetail, ExperimentListResponse,
    ExperimentRecordsResponse, ExperimentSummary, SqlQueryRequest, SqlQueryResponse, TaskKind,
    TaskListResponse, TaskRecord, TaskState,
};
use lance_context_core::{rollout_record_to_dto, ListSource, RolloutFilters, RolloutStore};
use tokio::sync::RwLock;

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

/// Query params for the paginated task list.
#[derive(Debug, Deserialize)]
pub struct TaskListParams {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
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
    /// Data source to scan: `fragments` (base table only, the default), `wal`
    /// (pending MemWAL generations only), or `all` (the union). Absent/empty
    /// defaults to `fragments`.
    #[serde(default)]
    pub source: Option<String>,
}

/// Parse the `source` query param into a [`ListSource`]. Absent or empty →
/// `Fragments` (the fast default); an explicitly unknown value is rejected.
fn parse_list_source(raw: Option<&str>) -> Result<ListSource, MasterError> {
    match raw.map(str::trim) {
        None | Some("") | Some("fragments") => Ok(ListSource::Fragments),
        Some("wal") => Ok(ListSource::Wal),
        Some("all") => Ok(ListSource::All),
        Some(other) => Err(MasterError::InvalidRequest(format!(
            "invalid source '{other}': expected one of fragments, wal, all"
        ))),
    }
}

fn list_source_label(source: ListSource) -> &'static str {
    match source {
        ListSource::Fragments => "fragments",
        ListSource::Wal => "wal",
        ListSource::All => "all",
    }
}

/// `GET /api/v1/experiments`
///
/// Without `search`, returns the experiments the stats table holds — the ones
/// written recently enough not to have been retired. That is the useful default
/// view at scale: a flat list of every experiment ever created is neither
/// renderable nor interesting once there are tens of thousands of them.
///
/// With `search`, falls back to the registry for names the stats table no
/// longer holds, and observes those on demand. Retirement therefore removes an
/// experiment from the *default* list but never makes it unfindable.
pub async fn list_experiments(
    State(state): State<Arc<MasterState>>,
    Query(params): Query<ListParams>,
) -> Result<Json<ExperimentListResponse>, MasterError> {
    let search = params.search.as_deref().filter(|s| !s.is_empty());

    let (mut experiments, mut total) = {
        let mut stats = state.stats.lock().await;
        let total = stats.count(search).await.map_err(MasterError::from_lance)?;
        let rows = stats
            .list(search, params.limit, params.offset)
            .await
            .map_err(MasterError::from_lance)?;
        let experiments: Vec<ExperimentSummary> =
            rows.into_iter().map(|r| r.into_summary()).collect();
        (experiments, total)
    };

    if let Some(query) = search {
        // Retired experiments have no stats row, so a search that only consulted
        // the stats table would silently omit them. The registry is the
        // authoritative list of what exists.
        let known: HashSet<String> = experiments.iter().map(|e| e.name.clone()).collect();
        let matches: Vec<_> = state
            .registry
            .write()
            .await
            .list()
            .await
            .map_err(MasterError::from_lance)?
            .into_iter()
            .filter(|entry| entry.name.contains(query) && !known.contains(&entry.name))
            .collect();

        total += matches.len() as i64;

        // Observe the retired matches this page needs. Bounded by the page
        // size: a search matching thousands of cold experiments must not open
        // thousands of datasets to render one page.
        let want = params.limit.saturating_sub(experiments.len());
        for entry in matches.into_iter().take(want) {
            match scanner::observe_cold(&state, &entry.name, &entry.uri).await {
                Ok(summary) => experiments.push(summary),
                Err(e) => {
                    tracing::warn!(
                        store = %entry.name,
                        error = %e,
                        "search: failed to observe retired experiment"
                    );
                }
            }
        }
        experiments.sort_by(|a, b| a.name.cmp(&b.name));
    }

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
    let cached = {
        let mut stats = state.stats.lock().await;
        stats.get(&name).await.map_err(MasterError::from_lance)?
    };
    if let Some(row) = cached {
        return Ok(Json(ExperimentDetail {
            summary: row.into_summary(),
        }));
    }

    // No stats row: either retired, or never scanned. Resolve through the
    // registry and observe on demand rather than reporting it as missing.
    let entry = state
        .registry
        .write()
        .await
        .get(&name)
        .await
        .map_err(MasterError::from_lance)?
        .ok_or_else(|| MasterError::NotFound(format!("experiment '{}' does not exist", name)))?;
    let summary = scanner::observe_cold(&state, &entry.name, &entry.uri)
        .await
        .map_err(MasterError::from_lance)?;
    Ok(Json(ExperimentDetail { summary }))
}

/// `GET /api/v1/experiments/{name}/records`
pub async fn list_experiment_records(
    State(state): State<Arc<MasterState>>,
    Path(name): Path<String>,
    Query(params): Query<RecordListParams>,
) -> Result<Json<ExperimentRecordsResponse>, MasterError> {
    let store = open_registered_store(&state, &name).await?;
    let mut store = store.write().await;
    store
        .refresh_latest()
        .await
        .map_err(MasterError::from_lance)?;
    let limit = params.limit.clamp(1, 100);
    let source = parse_list_source(params.source.as_deref())?;
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
        .list_filtered_source(&filters, limit, params.offset, source)
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
        source: list_source_label(source).to_string(),
    }))
}

/// `POST /api/v1/experiments/{name}/query` — run a read-only `SELECT` against
/// one experiment's rollout records (exposed as a table named `records`,
/// merged base ∪ pending WAL view). Non-`SELECT` or malformed SQL returns 400.
pub async fn query_experiment_sql(
    State(state): State<Arc<MasterState>>,
    Path(name): Path<String>,
    Json(body): Json<SqlQueryRequest>,
) -> Result<Json<SqlQueryResponse>, MasterError> {
    let store = open_registered_store(&state, &name).await?;
    let mut store = store.write().await;
    store
        .refresh_latest()
        .await
        .map_err(MasterError::from_lance)?;
    let result = store
        .query_sql(&body.sql)
        .await
        .map_err(MasterError::from_lance_user)?;
    let row_count = result.rows.len();
    Ok(Json(SqlQueryResponse {
        columns: result.columns,
        rows: result.rows,
        row_count,
        truncated: result.truncated,
    }))
}

/// `GET /api/v1/experiments/{name}/records/{id}/blob`
pub async fn download_experiment_blob(
    State(state): State<Arc<MasterState>>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Response, MasterError> {
    let store = open_registered_store(&state, &name).await?;
    let mut store = store.write().await;
    store
        .refresh_latest()
        .await
        .map_err(MasterError::from_lance)?;
    // Single base-first scan returns the row metadata and its payload together,
    // instead of a separate get_by_id + get_blob (two point scans over the same
    // shard) as before.
    let (record, payload) = store
        .get_record_with_blob(&id)
        .await
        .map_err(MasterError::from_lance)?
        .ok_or_else(|| MasterError::NotFound(format!("record '{}' does not exist", id)))?;
    let bytes =
        payload.ok_or_else(|| MasterError::NotFound(format!("record '{}' has no blob", id)))?;

    let content_type = HeaderValue::from_str(&record.content_type)
        .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream"));
    let filename = blob_filename(&record);
    let disposition = HeaderValue::from_str(&format!("attachment; filename=\"{filename}\""))
        .map_err(|err| MasterError::Internal(err.to_string()))?;

    Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_DISPOSITION, disposition)
        .header(header::CONTENT_LENGTH, bytes.len())
        .body(blob_stream_body(bytes))
        .map_err(|err| MasterError::Internal(err.to_string()))
}

/// Frame size for chunked blob download responses (see [`blob_stream_body`]).
const BLOB_STREAM_CHUNK_BYTES: usize = 256 * 1024;

/// Build a chunked [`Body`] from an owned blob so the HTTP send path never holds
/// a second full-blob copy and a slow client applies backpressure at frame
/// granularity. `Bytes` frames are refcounted slices of one allocation, so no
/// per-frame copy of the payload is made.
fn blob_stream_body(bytes: Vec<u8>) -> Body {
    let buf = bytes::Bytes::from(bytes);
    let chunks = (0..buf.len())
        .step_by(BLOB_STREAM_CHUNK_BYTES.max(1))
        .map(move |start| {
            let end = (start + BLOB_STREAM_CHUNK_BYTES).min(buf.len());
            Ok::<_, std::convert::Infallible>(buf.slice(start..end))
        });
    Body::from_stream(futures::stream::iter(chunks))
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
            let (fragments_removed, fragments_added) =
                parse_fragment_detail(task.detail.as_deref());
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
async fn latest_compact_task(
    state: &Arc<MasterState>,
    name: &str,
) -> lance::Result<Option<TaskRecord>> {
    Ok(state
        .task_store
        .list()
        .await?
        .into_iter()
        .filter(|t| t.kind == TaskKind::Compact && t.target == name)
        .max_by_key(|t| t.enqueued_at))
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
    let task = scheduler::enqueue(&state, TaskKind::Compact, &name)
        .await
        .map_err(MasterError::from_lance)?;
    Ok((StatusCode::ACCEPTED, Json(task_to_compact_status(&task))))
}

/// `GET /api/v1/experiments/{name}/compact/status` — latest compaction job
/// state for an experiment (`None` when never requested).
pub async fn compact_status(
    State(state): State<Arc<MasterState>>,
    Path(name): Path<String>,
) -> Result<Json<CompactJobStatus>, MasterError> {
    Ok(
        match latest_compact_task(&state, &name)
            .await
            .map_err(MasterError::from_lance)?
        {
            Some(task) => Json(task_to_compact_status(&task)),
            None => Json(CompactJobStatus::None),
        },
    )
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
    let task = scheduler::enqueue_with_deps(&state, req.kind, &req.target, req.depends_on)
        .await
        .map_err(MasterError::from_lance)?;
    Ok((StatusCode::ACCEPTED, Json(task)))
}

/// `GET /api/v1/tasks` — paginated tasks (queue + recent history), newest first.
pub async fn list_tasks(
    State(state): State<Arc<MasterState>>,
    Query(params): Query<TaskListParams>,
) -> Result<Json<TaskListResponse>, MasterError> {
    let mut tasks = state
        .task_store
        .list()
        .await
        .map_err(MasterError::from_lance)?;
    tasks.sort_by_key(|t| std::cmp::Reverse(t.enqueued_at));
    let total = tasks.len();
    let limit = params.limit.clamp(1, 200);
    let page = tasks.into_iter().skip(params.offset).take(limit).collect();
    Ok(Json(TaskListResponse {
        tasks: page,
        total,
        limit,
        offset: params.offset,
    }))
}

/// `GET /api/v1/tasks/{id}` — a single task by id.
pub async fn get_task(
    State(state): State<Arc<MasterState>>,
    Path(id): Path<String>,
) -> Result<Json<TaskRecord>, MasterError> {
    match state
        .task_store
        .get(&id)
        .await
        .map_err(MasterError::from_lance)?
    {
        Some(task) => Ok(Json(task)),
        None => Err(MasterError::NotFound(format!("task '{}' not found", id))),
    }
}

/// Build the admin API router (mounted under `/api/v1`).
pub fn api_router() -> Router<Arc<MasterState>> {
    Router::new()
        .route("/experiments", get(list_experiments))
        .route("/experiments/{name}", get(get_experiment))
        .route("/experiments/{name}/records", get(list_experiment_records))
        .route("/experiments/{name}/query", post(query_experiment_sql))
        .route(
            "/experiments/{name}/records/{id}/blob",
            get(download_experiment_blob),
        )
        .route("/experiments/{name}/compact", post(compact_experiment))
        .route("/experiments/{name}/compact/status", get(compact_status))
        .route("/tasks", post(enqueue_task).get(list_tasks))
        .route("/tasks/{id}", get(get_task))
        .route("/rescan", post(rescan))
}

async fn open_registered_store(
    state: &MasterState,
    name: &str,
) -> Result<Arc<RwLock<RolloutStore>>, MasterError> {
    let entry = state
        .registry
        .write()
        .await
        .get(name)
        .await
        .map_err(MasterError::from_lance)?
        .ok_or_else(|| MasterError::NotFound(format!("experiment '{}' does not exist", name)))?;
    state
        .get_or_open_record_store(name, &entry.uri)
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
    use lance_context_core::{generate_id, RolloutRecord, RolloutRegistry, RolloutStore};
    use serde_json::json;
    use tempfile::TempDir;

    fn test_config(dir: &TempDir) -> MasterConfig {
        MasterConfig {
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
            min_fragments: 16,
            target_rows_per_fragment: 1_048_576,
            compaction_concurrency: 1,
            compaction_threads: 1,
            compaction_batch_size: 8,
            compaction_max_source_fragments: 32,
            compaction_max_bytes_per_file: 1024 * 1024 * 1024,
            merge_wal_interval_secs: 0,
            merge_wal_min_generations: 8,
            worker_endpoints: vec![],
            task_concurrency: 4,
            etcd_endpoints: test_etcd_endpoints(),
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

    fn test_etcd_endpoints() -> Vec<String> {
        std::env::var("ETCD_TEST_ENDPOINTS")
            .map(|value| value.split(',').map(str::to_string).collect())
            .unwrap_or_default()
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
            model_input_string: None,
            model_output_string: None,
            rationale: None,
            problem_text: None,
            user_metadata: None,
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
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
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
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
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
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
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
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
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
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
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
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn records_endpoint_filters_pages_and_rejects_unknown_experiment() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();
        let uri = state.rollout_uri("records");
        let store = RolloutStore::open(&uri).await.unwrap();
        store
            .add(&[
                test_record("assistant-1", false),
                test_record("artifact-1", true),
            ])
            .await
            .unwrap();
        // `add` only appends to this handle's in-memory memtable; the rows
        // become visible to the separate handle the endpoint opens only once
        // the memtable is sealed into a committed WAL generation. Flush rather
        // than relaxing the assertions below.
        store.flush().await.unwrap();
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
                source: Some("all".to_string()),
            }),
        )
        .await
        .unwrap();
        assert!(!page.has_more);
        assert_eq!(page.limit, 1);
        assert_eq!(page.records[0].id, "artifact-1");
        assert!(page.records[0].binary_payload.is_none());
        assert_eq!(page.source, "all");

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
                source: None,
            }),
        )
        .await;
        assert!(matches!(missing, Err(MasterError::NotFound(_))));
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn query_endpoint_runs_select_and_rejects_mutations() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();
        let uri = state.rollout_uri("records");
        let store = RolloutStore::open(&uri).await.unwrap();
        store
            .add(&[
                test_record("assistant-1", false),
                test_record("artifact-1", true),
            ])
            .await
            .unwrap();
        // `add` only appends to this handle's in-memory memtable; the rows
        // become visible to the separate handle the endpoint opens only once
        // the memtable is sealed into a committed WAL generation. Flush rather
        // than relaxing the assertions below.
        store.flush().await.unwrap();
        state
            .registry
            .write()
            .await
            .upsert("records", &uri)
            .await
            .unwrap();

        // A valid SELECT returns rows over the merged view.
        let Json(result) = query_experiment_sql(
            State(state.clone()),
            Path("records".to_string()),
            Json(SqlQueryRequest {
                sql: "SELECT count(*) AS n FROM records".to_string(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(result.columns, vec!["n".to_string()]);
        assert_eq!(result.row_count, 1);
        assert!(!result.truncated);

        // A mutation is rejected as a 400-class InvalidRequest.
        let rejected = query_experiment_sql(
            State(state.clone()),
            Path("records".to_string()),
            Json(SqlQueryRequest {
                sql: "DROP TABLE records".to_string(),
            }),
        )
        .await;
        assert!(matches!(rejected, Err(MasterError::InvalidRequest(_))));

        // Unknown experiment is a 404.
        let missing = query_experiment_sql(
            State(state),
            Path("missing".to_string()),
            Json(SqlQueryRequest {
                sql: "SELECT 1".to_string(),
            }),
        )
        .await;
        assert!(matches!(missing, Err(MasterError::NotFound(_))));
    }

    #[test]
    fn parse_list_source_maps_and_defaults() {
        assert_eq!(parse_list_source(None).unwrap(), ListSource::Fragments);
        assert_eq!(parse_list_source(Some("")).unwrap(), ListSource::Fragments);
        assert_eq!(
            parse_list_source(Some("fragments")).unwrap(),
            ListSource::Fragments
        );
        assert_eq!(parse_list_source(Some("wal")).unwrap(), ListSource::Wal);
        assert_eq!(parse_list_source(Some("all")).unwrap(), ListSource::All);
        assert!(matches!(
            parse_list_source(Some("bogus")),
            Err(MasterError::InvalidRequest(_))
        ));
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn blob_endpoint_sets_download_headers_and_returns_bytes() {
        let dir = TempDir::new().unwrap();
        let state = MasterState::new(test_config(&dir)).await.unwrap();
        let uri = state.rollout_uri("blobs");
        let store = RolloutStore::open(&uri).await.unwrap();
        store
            .add(&[
                test_record("artifact-1", true),
                test_record("assistant-1", false),
            ])
            .await
            .unwrap();
        // See the note in `records_endpoint_...`: rows are only visible to the
        // handle the endpoint opens after the memtable is sealed.
        store.flush().await.unwrap();
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
