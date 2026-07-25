use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{FromRequest, Multipart, Path, Query, Request, State};
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::Json;
use chrono::Utc;
use lance_context_api::{
    AddRolloutRequest, AddRolloutsRequest, AddRolloutsResponse, CheckoutRequest, CompactRequest,
    CompactResponse, CompactStatsResponse, CreateRolloutStoreRequest, GetRolloutResponse,
    ListRolloutStoresResponse, ListRolloutsResponse, RelationshipDto, RolloutRecordDto,
    RolloutStoreInfo, VersionResponse,
};
use lance_context_core::{
    CompactionConfig, Relationship, RolloutFilters, RolloutRecord, RolloutStore,
    RolloutStoreOptions,
};
use tokio::sync::RwLock;

use crate::error::AppError;
use crate::state::AppState;

/// Upper bound on a single rollout append request body (JSON or multipart).
/// Artifact blobs routinely exceed axum's 2 MiB default, so the batch endpoint
/// raises the ceiling well above it while still bounding memory.
pub const MAX_ROLLOUT_UPLOAD_BYTES: usize = 1024 * 1024 * 1024;

/// Frame size for chunked blob download responses. The in-memory `Vec<u8>` is
/// sliced into frames of this size so the HTTP send path never holds an extra
/// full-blob copy and a slow reader exerts backpressure at frame granularity
/// rather than after the whole payload is queued. 256 KiB keeps per-frame
/// overhead negligible while capping the amount buffered ahead of the socket.
const BLOB_STREAM_CHUNK_BYTES: usize = 256 * 1024;

/// Build a chunked [`Body`] from an owned blob. `Bytes` frames share one
/// backing allocation (cheap refcounted slices — no per-frame copy), so this
/// does not duplicate the payload; it only changes how it is fed to the socket.
///
/// `reservation` (the in-flight blob-budget guard, if any) is moved into the
/// stream and released only when the last frame has been produced, so the
/// budget accounts for a slow download for its full lifetime.
fn blob_stream_body(
    bytes: Vec<u8>,
    mut reservation: Option<crate::state::BlobReservation>,
) -> Body {
    let buf = bytes::Bytes::from(bytes);
    let len = buf.len();
    let mut offset = 0usize;
    let stream = futures::stream::poll_fn(move |_| {
        if offset >= len {
            // Drop the reservation exactly when the stream ends.
            let _ = reservation.take();
            return std::task::Poll::Ready(None);
        }
        let end = (offset + BLOB_STREAM_CHUNK_BYTES).min(len);
        let frame = buf.slice(offset..end);
        offset = end;
        std::task::Poll::Ready(Some(Ok::<_, std::convert::Infallible>(frame)))
    });
    Body::from_stream(stream)
}

/// Parse the `Content-Length` header into a byte count, defaulting to `0` when
/// absent or unparsable (a chunked upload without a declared length reserves
/// nothing up-front; the body-size limit still caps it).
fn content_length(headers: &header::HeaderMap) -> usize {
    headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Reserve `bytes` from the instance's in-flight blob budget (if configured),
/// returning the RAII guard to hold for the request's duration. Returns
/// `503 Overloaded` when the budget cannot currently admit the request. When no
/// budget is configured (`None`) this is a no-op that always admits.
fn acquire_blob_budget(
    state: &AppState,
    bytes: usize,
) -> Result<Option<crate::state::BlobReservation>, AppError> {
    match &state.blob_budget {
        None => Ok(None),
        Some(budget) => budget.try_acquire(bytes).map(Some).ok_or_else(|| {
            metrics::counter!("rollout_blob_budget_rejections_total").increment(1);
            AppError::Overloaded(
                "server is at its in-flight blob memory limit; retry shortly".to_string(),
            )
        }),
    }
}

/// Response for the internal WAL-merge trigger: how many flushed generations
/// this worker's shard folded into the base table.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct MergeWalResponse {
    pub reclaimed: usize,
}

pub async fn create_rollout_store(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateRolloutStoreRequest>,
) -> Result<(StatusCode, Json<RolloutStoreInfo>), AppError> {
    AppState::validate_name(&req.name)?;
    // Existence is tracked durably in the registry, not by cache membership.
    if state
        .rollout_registry
        .write()
        .await
        .contains(&req.name)
        .await
        .map_err(AppError::from_lance)?
    {
        return Err(AppError::AlreadyExists(format!(
            "Rollout store '{}' already exists",
            req.name
        )));
    }

    let uri = state.rollout_uri(&req.name);
    let options = RolloutStoreOptions {
        storage_options: req.storage_options,
        shard_id: state.instance_id.clone(),
        merge_after_generations: (state.rollout_merge_after_generations > 0)
            .then_some(state.rollout_merge_after_generations),
    };

    let store = RolloutStore::open_with_options(&uri, options)
        .await
        .map_err(AppError::from_lance)?;
    let version = store.version();

    let store = Arc::new(RwLock::new(store));
    // Record in the durable registry and the LRU. WAL cleanup is handled by the
    // single process-wide sweeper (see `AppState::spawn_global_sweeper`), so no
    // per-store timer is started here.
    state.register_rollout(&req.name, &uri, store).await?;

    Ok((
        StatusCode::CREATED,
        Json(RolloutStoreInfo {
            name: req.name,
            uri,
            version: Some(version),
        }),
    ))
}

pub async fn list_rollout_stores(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ListRolloutStoresResponse>, AppError> {
    // Served from the durable registry so it enumerates *all* stores, not just
    // those currently resident in the LRU. `version` is omitted here to avoid
    // opening every dataset.
    let entries = state
        .rollout_registry
        .write()
        .await
        .list()
        .await
        .map_err(AppError::from_lance)?;
    let out = entries
        .into_iter()
        .map(|entry| RolloutStoreInfo {
            name: entry.name,
            uri: entry.uri,
            version: None,
        })
        .collect();

    Ok(Json(ListRolloutStoresResponse { stores: out }))
}

pub async fn get_rollout_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<RolloutStoreInfo>, AppError> {
    let store_lock = state.get_or_open_rollout_store(&name).await?;
    let store = store_lock.read().await;

    Ok(Json(RolloutStoreInfo {
        name: name.clone(),
        uri: state.rollout_uri(&name),
        version: Some(store.version()),
    }))
}

pub async fn delete_rollout_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    // Removes from the durable registry and evicts any resident handle; 404 when
    // the store is not registered.
    if !state.unregister_rollout(&name).await? {
        return Err(AppError::NotFound(format!(
            "Rollout store '{}' does not exist",
            name
        )));
    }

    let uri = state.rollout_uri(&name);
    if let Err(e) = tokio::fs::remove_dir_all(&uri).await {
        tracing::warn!("Failed to remove rollout data at {}: {}", uri, e);
    }

    Ok(StatusCode::NO_CONTENT)
}

/// Append rollout rows. The single POST endpoint accepts either encoding,
/// chosen by `Content-Type`:
///
/// - `application/json`: an [`AddRolloutsRequest`]; any `binary_payload` rides
///   inline as base64.
/// - `multipart/form-data`: a first part named `metadata` holding the records
///   array (with each `binary_payload` null), followed by one raw binary part
///   per record that carries a blob, named for that record's zero-based index.
///
/// Both encodings funnel into a single [`RolloutStore::add`]. A rollout append
/// is atomic and feeds training, so the multipart path is validated strictly:
/// any part whose name is not a valid record index, any index with no matching
/// record, a duplicate index, a byte length disagreeing with the record's
/// `payload_size`, or a record that declares `payload_size` yet receives no
/// bytes rejects the whole request with `400` before anything is written.
pub async fn add_rollouts(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    req: Request,
) -> Result<(StatusCode, Json<AddRolloutsResponse>), AppError> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();

    // Admit against the in-flight blob budget before buffering the body, using
    // the declared Content-Length as the reservation size. Held for the whole
    // handler so concurrent uploads cannot collectively exceed the budget and
    // OOM the worker; dropped when the request completes.
    let _budget = acquire_blob_budget(&state, content_length(req.headers()))?;

    let records = if content_type_is(&content_type, "multipart/form-data") {
        let multipart = Multipart::from_request(req, &state)
            .await
            .map_err(|e| AppError::InvalidRequest(format!("invalid multipart request: {e}")))?;
        parse_multipart_rollouts(multipart).await?
    } else if content_type_is(&content_type, "application/json") {
        let bytes = axum::body::to_bytes(req.into_body(), MAX_ROLLOUT_UPLOAD_BYTES)
            .await
            .map_err(|e| AppError::InvalidRequest(format!("failed to read request body: {e}")))?;
        let manifest: AddRolloutsRequest = serde_json::from_slice(&bytes)
            .map_err(|e| AppError::InvalidRequest(format!("invalid JSON body: {e}")))?;
        manifest.records
    } else {
        return Err(AppError::InvalidRequest(format!(
            "unsupported Content-Type '{content_type}': expected application/json or multipart/form-data"
        )));
    };

    if records.is_empty() {
        return Err(AppError::InvalidRequest(
            "records array must not be empty".to_string(),
        ));
    }

    let store_lock = state.get_or_open_rollout_store(&name).await?;

    let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
    let core_records: Vec<RolloutRecord> = records
        .iter()
        .map(rollout_record_from_add_request)
        .collect();
    let count = core_records.len();

    // A read lock: `add` is `&self` and MemWAL appends are internally
    // concurrent, so multiple ingest requests to the same store run in parallel.
    // Mutating ops (merge, compact, checkout, close) still take the write lock.
    let store = store_lock.read().await;
    let version = store
        .add(&core_records)
        .await
        .map_err(AppError::from_lance)?;

    metrics::counter!("rollout_appends_total").increment(1);
    metrics::counter!("rollout_records_appended_total").increment(count as u64);

    Ok((
        StatusCode::CREATED,
        Json(AddRolloutsResponse {
            version,
            ids,
            count,
        }),
    ))
}

/// Parse a `multipart/form-data` rollout append into records with their blob
/// bytes attached. Enforces the strict-reject contract described on
/// [`add_rollouts`].
async fn parse_multipart_rollouts(
    mut multipart: Multipart,
) -> Result<Vec<AddRolloutRequest>, AppError> {
    // The manifest must arrive first so blob parts can be matched by index as
    // they stream in.
    let first = multipart
        .next_field()
        .await
        .map_err(|e| AppError::InvalidRequest(format!("malformed multipart body: {e}")))?
        .ok_or_else(|| AppError::InvalidRequest("multipart body is empty".to_string()))?;
    if first.name() != Some("metadata") {
        return Err(AppError::InvalidRequest(format!(
            "first multipart part must be named 'metadata', found '{}'",
            first.name().unwrap_or("<unnamed>")
        )));
    }
    let metadata_bytes = first
        .bytes()
        .await
        .map_err(|e| AppError::InvalidRequest(format!("failed to read metadata part: {e}")))?;
    let manifest: AddRolloutsRequest = serde_json::from_slice(&metadata_bytes)
        .map_err(|e| AppError::InvalidRequest(format!("invalid metadata JSON: {e}")))?;
    let mut records = manifest.records;

    // Collect the raw binary parts, keyed by the record index they target.
    let mut blobs: HashMap<usize, Vec<u8>> = HashMap::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::InvalidRequest(format!("malformed multipart body: {e}")))?
    {
        let part_name = field
            .name()
            .ok_or_else(|| {
                AppError::InvalidRequest("multipart part is missing a name".to_string())
            })?
            .to_string();
        let idx: usize = part_name.parse().map_err(|_| {
            AppError::InvalidRequest(format!(
                "unexpected multipart part '{part_name}': expected a record index"
            ))
        })?;
        if idx >= records.len() {
            return Err(AppError::InvalidRequest(format!(
                "multipart part '{idx}' has no matching record ({} records supplied)",
                records.len()
            )));
        }
        let bytes = field.bytes().await.map_err(|e| {
            AppError::InvalidRequest(format!("failed to read binary part '{idx}': {e}"))
        })?;
        if blobs.insert(idx, bytes.to_vec()).is_some() {
            return Err(AppError::InvalidRequest(format!(
                "duplicate multipart part for record index {idx}"
            )));
        }
    }

    // Attach each blob to its record, verifying the declared size.
    for (idx, bytes) in blobs {
        let record = &mut records[idx];
        if let Some(expected) = record.payload_size {
            if expected != bytes.len() as i64 {
                return Err(AppError::InvalidRequest(format!(
                    "record index {idx}: payload_size {expected} does not match received {} bytes",
                    bytes.len()
                )));
            }
        }
        record.binary_payload = Some(bytes);
    }

    // Symmetric strict-reject: a record that declares a payload but received no
    // bytes (no inline payload and no matching part) would be written with an
    // empty artifact, silently dropping training data. Reject it.
    for (idx, record) in records.iter().enumerate() {
        if record.payload_size.is_some() && record.binary_payload.is_none() {
            return Err(AppError::InvalidRequest(format!(
                "record index {idx} declares payload_size but no binary part was supplied"
            )));
        }
    }

    Ok(records)
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct RolloutListParams {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub filters: Option<String>,
}

pub async fn list_rollouts(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<RolloutListParams>,
) -> Result<Json<ListRolloutsResponse>, AppError> {
    let filters = params
        .filters
        .as_deref()
        .map(|raw| {
            let value = serde_json::from_str(raw).map_err(|err| {
                AppError::InvalidRequest(format!("invalid rollout filters: {err}"))
            })?;
            RolloutFilters::from_json_value(value).map_err(AppError::InvalidRequest)
        })
        .transpose()?;

    let store_lock = state.get_or_open_rollout_store(&name).await?;

    let store = store_lock.read().await;
    let records = store
        .list_with_filters(params.limit, params.offset, filters.as_ref())
        .await
        .map_err(AppError::from_lance)?;

    Ok(Json(ListRolloutsResponse {
        records: records.into_iter().map(rollout_record_to_dto).collect(),
    }))
}

pub async fn get_rollout(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Json<GetRolloutResponse>, AppError> {
    let store_lock = state.get_or_open_rollout_store(&name).await?;

    let store = store_lock.read().await;
    let record = store.get_by_id(&id).await.map_err(AppError::from_lance)?;

    Ok(Json(GetRolloutResponse {
        record: record.map(rollout_record_to_dto),
    }))
}

/// Materialize a rollout row's `binary_payload` bytes. The artifact
/// bytes are opaque, so they stream as `application/octet-stream`. `404` when
/// the row is absent or carries no payload.
///
/// The payload is sent as a **chunked** body ([`blob_stream_body`]) rather than
/// a single giant frame: the response yields fixed-size frames so the HTTP
/// layer never re-buffers the whole blob and a slow client applies backpressure
/// instead of forcing the server to hold an extra full copy in the send queue.
pub async fn fetch_rollout_blob(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Response, AppError> {
    let store_lock = state.get_or_open_rollout_store(&name).await?;

    let store = store_lock.read().await;
    let bytes = store
        .get_blob(&id)
        .await
        .map_err(AppError::from_lance)?
        .ok_or_else(|| AppError::NotFound(format!("Rollout '{}' has no payload", id)))?;

    // Reserve now that the payload size is known, and hold the reservation for
    // the whole streamed send (moved into the body): a slow client keeps the
    // blob resident until the last frame flushes, so the budget must account for
    // it until then. Reject with 503 if the budget is currently exhausted.
    let reservation = acquire_blob_budget(&state, bytes.len())?;
    drop(store);

    let len = bytes.len();
    Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, len)
        .body(blob_stream_body(bytes, reservation))
        .map_err(|err| AppError::Internal(err.to_string()))
}

pub async fn checkout_rollout(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<CheckoutRequest>,
) -> Result<Json<VersionResponse>, AppError> {
    let store_lock = state.get_or_open_rollout_store(&name).await?;

    let mut store = store_lock.write().await;
    store
        .checkout(req.version)
        .await
        .map_err(AppError::from_lance)?;

    Ok(Json(VersionResponse {
        version: store.version(),
    }))
}

/// Compact the rollout store's base table (fold small fragments produced by WAL
/// merges into larger ones).
///
/// Intended to be driven by an external scheduler (cron / k8s CronJob) from a
/// single caller — not every worker — since two concurrent base-table rewrites
/// conflict and waste work. Safe to run while workers append or WAL-merge.
pub async fn compact_rollout(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<CompactRequest>,
) -> Result<Json<CompactResponse>, AppError> {
    let store_lock = state.get_or_open_rollout_store(&name).await?;

    let config = if req.target_rows_per_fragment.is_some() || req.materialize_deletions.is_some() {
        let mut c = CompactionConfig::default();
        if let Some(v) = req.target_rows_per_fragment {
            c.target_rows_per_fragment = v;
        }
        if let Some(v) = req.materialize_deletions {
            c.materialize_deletions = v;
        }
        Some(c)
    } else {
        None
    };

    let mut store = store_lock.write().await;
    let compact_start = std::time::Instant::now();
    let compact_result = store.compact(config).await;
    ::metrics::histogram!("rollout_compaction_duration_seconds")
        .record(compact_start.elapsed().as_secs_f64());
    let metrics = match compact_result {
        Ok(m) => {
            ::metrics::counter!("rollout_compactions_total", "result" => "success").increment(1);
            m
        }
        Err(e) => {
            ::metrics::counter!("rollout_compactions_total", "result" => "failed").increment(1);
            return Err(AppError::from_lance(e));
        }
    };

    Ok(Json(CompactResponse {
        fragments_removed: metrics.fragments_removed,
        fragments_added: metrics.fragments_added,
        files_removed: metrics.files_removed,
        files_added: metrics.files_added,
    }))
}

/// Base-table compaction statistics for a rollout store.
pub async fn compact_rollout_stats(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<CompactStatsResponse>, AppError> {
    let store_lock = state.get_or_open_rollout_store(&name).await?;

    let store = store_lock.read().await;
    let stats = store.compaction_stats();

    Ok(Json(CompactStatsResponse {
        total_fragments: stats.total_fragments,
        is_compacting: stats.is_compacting,
        last_compaction: stats.last_compaction,
        last_error: stats.last_error,
        total_compactions: stats.total_compactions,
    }))
}

/// Trigger a WAL merge on this worker's own shard for `name`.
///
/// This is the worker half of the master-driven "MergeWal" task: the master
/// cannot merge a shard it does not own without fencing the live writer, so it
/// fans this call out to every worker and each worker folds *its own* flushed
/// MemWAL generations into the base table via [`RolloutStore::cleanup_own_shard`]
/// (which merges whatever is pending, no count threshold). A worker whose shard
/// has nothing pending simply reports `reclaimed: 0`.
pub async fn merge_wal(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<MergeWalResponse>, AppError> {
    let store_lock = state.get_or_open_rollout_store(&name).await?;
    let mut store = store_lock.write().await;
    let reclaimed = store
        .cleanup_own_shard()
        .await
        .map_err(AppError::from_lance)?;
    if reclaimed > 0 {
        ::metrics::counter!("rollout_wal_cleanup_total", "result" => "merged").increment(1);
        ::metrics::counter!("rollout_wal_generations_reclaimed_total").increment(reclaimed as u64);
    }
    Ok(Json(MergeWalResponse { reclaimed }))
}

fn content_type_is(content_type: &str, expected: &str) -> bool {
    content_type
        .split(';')
        .next()
        .map(str::trim)
        .is_some_and(|mime| mime.eq_ignore_ascii_case(expected))
}

fn dto_to_relationship(r: RelationshipDto) -> Relationship {
    Relationship {
        target_id: r.target_id,
        relation: r.relation,
        weight: r.weight,
    }
}

fn relationship_to_dto(r: Relationship) -> RelationshipDto {
    RelationshipDto {
        target_id: r.target_id,
        relation: r.relation,
        weight: r.weight,
    }
}

fn rollout_record_from_add_request(r: &AddRolloutRequest) -> RolloutRecord {
    RolloutRecord {
        id: r.id.clone(),
        rollout_id: r.rollout_id.clone(),
        problem_id: r.problem_id.clone().unwrap_or_else(|| r.rollout_id.clone()),
        dataset: r.dataset.clone(),
        sequence_order: r.sequence_order,
        role: r.role.clone(),
        created_at: r.created_at.unwrap_or_else(Utc::now),
        content: r.content.clone(),
        content_type: r.content_type.clone(),
        model_input_string: r.model_input_string.clone(),
        model_output_string: r.model_output_string.clone(),
        rationale: r.rationale.clone(),
        problem_text: r.problem_text.clone(),
        user_metadata: r.user_metadata.clone(),
        input_tokens: r.input_tokens.clone(),
        output_tokens: r.output_tokens.clone(),
        num_input_tokens: r.num_input_tokens,
        num_output_tokens: r.num_output_tokens,
        output_logprobs: r.output_logprobs.clone(),
        input_logprobs: r.input_logprobs.clone(),
        ref_logprobs: r.ref_logprobs.clone(),
        loss_mask: r.loss_mask.clone(),
        advantage: r.advantage,
        reward: r.reward,
        raw_reward: r.raw_reward,
        grader_id: r.grader_id.clone(),
        score: r.score,
        include_in_training: r.include_in_training,
        exclude_reason: r.exclude_reason.clone(),
        policy_version: r.policy_version.clone(),
        relationships: r
            .relationships
            .iter()
            .cloned()
            .map(dto_to_relationship)
            .collect(),
        binary_payload: r.binary_payload.clone(),
        payload_size: r.payload_size,
        payload_checksum: r.payload_checksum.clone(),
        artifact_type: r.artifact_type.clone(),
        metadata: r.metadata.clone(),
    }
}

fn rollout_record_to_dto(r: RolloutRecord) -> RolloutRecordDto {
    RolloutRecordDto {
        id: r.id,
        rollout_id: r.rollout_id,
        problem_id: r.problem_id,
        dataset: r.dataset,
        sequence_order: r.sequence_order,
        role: r.role,
        created_at: r.created_at,
        content: r.content,
        content_type: r.content_type,
        model_input_string: r.model_input_string,
        model_output_string: r.model_output_string,
        rationale: r.rationale,
        problem_text: r.problem_text,
        user_metadata: r.user_metadata,
        input_tokens: r.input_tokens,
        output_tokens: r.output_tokens,
        num_input_tokens: r.num_input_tokens,
        num_output_tokens: r.num_output_tokens,
        output_logprobs: r.output_logprobs,
        input_logprobs: r.input_logprobs,
        ref_logprobs: r.ref_logprobs,
        loss_mask: r.loss_mask,
        advantage: r.advantage,
        reward: r.reward,
        raw_reward: r.raw_reward,
        grader_id: r.grader_id,
        score: r.score,
        include_in_training: r.include_in_training,
        exclude_reason: r.exclude_reason,
        policy_version: r.policy_version,
        relationships: r
            .relationships
            .into_iter()
            .map(relationship_to_dto)
            .collect(),
        binary_payload: r.binary_payload,
        payload_size: r.payload_size,
        payload_checksum: r.payload_checksum,
        artifact_type: r.artifact_type,
        metadata: r.metadata,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lance_context_api::CreateRolloutStoreRequest;
    use tempfile::TempDir;

    use super::*;
    use crate::state::AppState;

    #[tokio::test]
    async fn blob_stream_body_reassembles_across_chunk_boundaries() {
        // A payload spanning several BLOB_STREAM_CHUNK_BYTES frames must
        // reassemble byte-for-byte, and a reservation moved into the body is
        // held until the stream is fully drained.
        let payload: Vec<u8> = (0..(BLOB_STREAM_CHUNK_BYTES * 2 + 123))
            .map(|i| (i % 251) as u8)
            .collect();
        let budget = crate::state::BlobBudget::new(payload.len());
        let reservation = budget.try_acquire(payload.len());
        assert!(reservation.is_some());
        // Budget is now fully occupied.
        assert!(budget.try_acquire(1).is_none());

        let body = blob_stream_body(payload.clone(), reservation);
        let collected = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        assert_eq!(&collected[..], &payload[..]);

        // Once the body is consumed the reservation dropped, freeing the budget.
        assert!(budget.try_acquire(payload.len()).is_some());
    }

    #[test]
    fn content_length_parses_or_defaults_zero() {
        let mut headers = header::HeaderMap::new();
        assert_eq!(content_length(&headers), 0);
        headers.insert(header::CONTENT_LENGTH, header::HeaderValue::from(4096));
        assert_eq!(content_length(&headers), 4096);
    }

    async fn rollout_state() -> (Arc<AppState>, TempDir) {
        let dir = TempDir::new().unwrap();
        let state = Arc::new(AppState::new_for_test(dir.path().to_path_buf()).await);
        let (_status, _info) = create_rollout_store(
            State(state.clone()),
            Json(CreateRolloutStoreRequest {
                name: "rl".to_string(),
                storage_options: None,
            }),
        )
        .await
        .expect("create rollout store");
        (state, dir)
    }

    #[tokio::test]
    async fn create_rejects_names_that_are_not_portable_path_segments() {
        let dir = TempDir::new().unwrap();
        let state = Arc::new(AppState::new_for_test(dir.path().to_path_buf()).await);

        for name in [
            "../escape",
            "nested/store",
            r"nested\store",
            "_registry",
            &"a".repeat(lance_context_core::MAX_STORE_NAME_LEN + 1),
        ] {
            let err = create_rollout_store(
                State(state.clone()),
                Json(CreateRolloutStoreRequest {
                    name: name.to_string(),
                    storage_options: None,
                }),
            )
            .await
            .unwrap_err();
            assert!(matches!(err, AppError::InvalidRequest(_)), "{name}");
        }
        assert!(state.rollout_stores.lock().await.is_empty());
        assert!(state
            .rollout_registry
            .write()
            .await
            .list()
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn open_rejects_invalid_rollout_names_before_registry_access() {
        let dir = TempDir::new().unwrap();
        let state = AppState::new_for_test(dir.path().to_path_buf()).await;

        let err = match state.get_or_open_rollout_store("../escape").await {
            Ok(_) => panic!("invalid name unexpectedly opened"),
            Err(err) => err,
        };
        assert!(matches!(err, AppError::InvalidRequest(_)));
    }

    fn record_with_size(id: &str, payload_size: Option<i64>) -> AddRolloutRequest {
        AddRolloutRequest {
            id: id.to_string(),
            rollout_id: format!("traj-{id}"),
            payload_size,
            ..Default::default()
        }
    }

    fn filtered_record(id: &str, policy_version: &str, training: bool) -> AddRolloutRequest {
        AddRolloutRequest {
            id: id.to_string(),
            rollout_id: format!("traj-{id}"),
            problem_id: Some(format!("problem-{id}")),
            role: "assistant".to_string(),
            policy_version: Some(policy_version.to_string()),
            include_in_training: Some(training),
            ..Default::default()
        }
    }

    fn json_request(body: Vec<u8>) -> Request {
        Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap()
    }

    /// Flush a rollout store's MemWAL so just-added rows become visible to reads.
    /// `add` is durable-but-async now (visible only after a flush / the periodic
    /// flush sweeper), so tests that add then read must flush explicitly.
    async fn flush_store(state: &Arc<AppState>, name: &str) {
        let store = state.get_or_open_rollout_store(name).await.unwrap();
        store.read().await.flush().await.unwrap();
    }

    /// Assemble a `multipart/form-data` body from ordered (name, bytes) parts.
    fn multipart_request(boundary: &str, parts: &[(&str, Vec<u8>)]) -> Request {
        let mut body = Vec::new();
        for (name, data) in parts {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        Request::builder()
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap()
    }

    async fn count_rollouts(state: &Arc<AppState>) -> usize {
        let Json(list) = list_rollouts(
            State(state.clone()),
            Path("rl".to_string()),
            Query(RolloutListParams::default()),
        )
        .await
        .unwrap();
        list.records.len()
    }

    #[tokio::test]
    async fn add_rollouts_json_projects_blob_out_but_fetch_materializes() {
        let (state, _dir) = rollout_state().await;
        let payload = b"artifact-bytes".to_vec();
        let mut record = record_with_size("r0", Some(payload.len() as i64));
        record.binary_payload = Some(payload.clone());
        let body = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![record],
        })
        .unwrap();

        let (status, Json(resp)) = add_rollouts(
            State(state.clone()),
            Path("rl".to_string()),
            json_request(body),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(resp.count, 1);
        assert_eq!(resp.ids, vec!["r0".to_string()]);
        flush_store(&state, "rl").await;

        // A plain get projects the binary column out, reading it back as None...
        let Json(got) = get_rollout(
            State(state.clone()),
            Path(("rl".to_string(), "r0".to_string())),
        )
        .await
        .unwrap();
        let dto = got.record.expect("row present");
        assert_eq!(dto.payload_size, Some(payload.len() as i64));
        assert!(dto.binary_payload.is_none());

        // ...but the blob endpoint materializes the bytes.
        let resp = fetch_rollout_blob(
            State(state.clone()),
            Path(("rl".to_string(), "r0".to_string())),
        )
        .await
        .unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), payload.as_slice());
    }

    #[tokio::test]
    async fn add_rollouts_multipart_attaches_blob_by_index() {
        let (state, _dir) = rollout_state().await;
        let payload = b"raw-multipart-bytes".to_vec();
        let metadata = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![record_with_size("r0", Some(payload.len() as i64))],
        })
        .unwrap();
        let req = multipart_request("BOUND", &[("metadata", metadata), ("0", payload.clone())]);

        let (status, Json(resp)) = add_rollouts(State(state.clone()), Path("rl".to_string()), req)
            .await
            .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(resp.count, 1);
        flush_store(&state, "rl").await;

        let resp = fetch_rollout_blob(
            State(state.clone()),
            Path(("rl".to_string(), "r0".to_string())),
        )
        .await
        .unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), payload.as_slice());
    }

    #[tokio::test]
    async fn add_rollouts_multipart_rejects_metadata_not_first() {
        let (state, _dir) = rollout_state().await;
        let req = multipart_request("BOUND", &[("0", b"bytes".to_vec())]);
        let err = add_rollouts(State(state.clone()), Path("rl".to_string()), req)
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::InvalidRequest(_)));
        assert_eq!(count_rollouts(&state).await, 0);
    }

    #[tokio::test]
    async fn add_rollouts_multipart_rejects_orphan_index() {
        let (state, _dir) = rollout_state().await;
        let metadata = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![record_with_size("r0", Some(4))],
        })
        .unwrap();
        // Part "1" has no matching record (only index 0 exists).
        let req = multipart_request("BOUND", &[("metadata", metadata), ("1", b"data".to_vec())]);
        let err = add_rollouts(State(state.clone()), Path("rl".to_string()), req)
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::InvalidRequest(_)));
        assert_eq!(count_rollouts(&state).await, 0);
    }

    #[tokio::test]
    async fn add_rollouts_multipart_rejects_duplicate_index() {
        let (state, _dir) = rollout_state().await;
        let metadata = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![record_with_size("r0", Some(4))],
        })
        .unwrap();
        let req = multipart_request(
            "BOUND",
            &[
                ("metadata", metadata),
                ("0", b"data".to_vec()),
                ("0", b"dupe".to_vec()),
            ],
        );
        let err = add_rollouts(State(state.clone()), Path("rl".to_string()), req)
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::InvalidRequest(_)));
        assert_eq!(count_rollouts(&state).await, 0);
    }

    #[tokio::test]
    async fn add_rollouts_multipart_rejects_size_mismatch() {
        let (state, _dir) = rollout_state().await;
        // Declared payload_size (999) disagrees with the 4 bytes received.
        let metadata = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![record_with_size("r0", Some(999))],
        })
        .unwrap();
        let req = multipart_request("BOUND", &[("metadata", metadata), ("0", b"data".to_vec())]);
        let err = add_rollouts(State(state.clone()), Path("rl".to_string()), req)
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::InvalidRequest(_)));
        assert_eq!(count_rollouts(&state).await, 0);
    }

    #[tokio::test]
    async fn add_rollouts_multipart_rejects_declared_payload_without_part() {
        let (state, _dir) = rollout_state().await;
        // Record declares payload_size but no binary part is attached.
        let metadata = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![record_with_size("r0", Some(4))],
        })
        .unwrap();
        let req = multipart_request("BOUND", &[("metadata", metadata)]);
        let err = add_rollouts(State(state.clone()), Path("rl".to_string()), req)
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::InvalidRequest(_)));
        assert_eq!(count_rollouts(&state).await, 0);
    }

    #[tokio::test]
    async fn add_rollouts_rejects_empty_records() {
        let (state, _dir) = rollout_state().await;
        let body = serde_json::to_vec(&AddRolloutsRequest { records: vec![] }).unwrap();
        let err = add_rollouts(
            State(state.clone()),
            Path("rl".to_string()),
            json_request(body),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::InvalidRequest(_)));
    }

    /// The internal merge-wal endpoint folds this worker's pending flushed
    /// generations into the base table and reports how many it reclaimed. A
    /// second call with nothing pending is a no-op reporting `0`.
    #[tokio::test]
    async fn merge_wal_reclaims_pending_then_is_noop() {
        let (state, _dir) = rollout_state().await;
        // One append flushes one generation into this instance's shard.
        let body = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![record_with_size("r0", None)],
        })
        .unwrap();
        let _ = add_rollouts(
            State(state.clone()),
            Path("rl".to_string()),
            json_request(body),
        )
        .await
        .expect("append succeeds");
        // Seal the append into a flushed generation so the merge has something
        // to reclaim (appends no longer flush inline).
        flush_store(&state, "rl").await;

        let Json(first) = merge_wal(State(state.clone()), Path("rl".to_string()))
            .await
            .expect("merge succeeds");
        assert_eq!(first.reclaimed, 1, "one pending generation merged");

        let Json(second) = merge_wal(State(state.clone()), Path("rl".to_string()))
            .await
            .expect("second merge succeeds");
        assert_eq!(second.reclaimed, 0, "nothing left to merge");

        // The row survives the merge, readable exactly once.
        let Json(got) = get_rollout(
            State(state.clone()),
            Path(("rl".to_string(), "r0".to_string())),
        )
        .await
        .unwrap();
        assert!(got.record.is_some());
    }

    /// A second server instance (distinct in-memory cache, shared data dir)
    /// that never `create`d the store must still serve reads/writes for it by
    /// lazily loading from storage — the multi-replica 404 regression.
    #[tokio::test]
    async fn second_instance_lazily_opens_store_created_elsewhere() {
        // Pod A: create the store and write a row.
        let (state_a, dir) = rollout_state().await;
        let body = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![record_with_size("r0", None)],
        })
        .unwrap();
        let _ = add_rollouts(
            State(state_a.clone()),
            Path("rl".to_string()),
            json_request(body),
        )
        .await
        .expect("write on instance A");
        // Flush A's shard so its row is visible to any reader (reads union all
        // shards' flushed generations).
        flush_store(&state_a, "rl").await;

        // Pod B: a fresh AppState over the SAME data dir, with an empty cache and
        // a different instance id (its own shard). It never saw the `create`.
        let state_b = Arc::new(
            AppState::new_for_test_with_instance(
                dir.path().to_path_buf(),
                Some("rollout-1".to_string()),
            )
            .await,
        );

        // Read routed to B must lazily open the store, not 404.
        let Json(list) = list_rollouts(
            State(state_b.clone()),
            Path("rl".to_string()),
            Query(RolloutListParams::default()),
        )
        .await
        .expect("instance B lazily opens the store instead of 404");
        assert_eq!(list.records.len(), 1);
        assert_eq!(list.records[0].id, "r0");

        // And a write on B lands in the same dataset (visible back on A).
        let body = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![record_with_size("r1", None)],
        })
        .unwrap();
        let _ = add_rollouts(
            State(state_b.clone()),
            Path("rl".to_string()),
            json_request(body),
        )
        .await
        .expect("write on instance B");
        // Flush B's shard so its row is visible to A's reader too.
        flush_store(&state_b, "rl").await;
        assert_eq!(count_rollouts(&state_a).await, 2);
    }

    /// A read for a store that was never created anywhere must still 404 — lazy
    /// open must not silently materialize an empty dataset for a bad name.
    #[tokio::test]
    async fn read_of_nonexistent_store_still_404s() {
        let (state, _dir) = rollout_state().await;
        let err = list_rollouts(
            State(state.clone()),
            Path("no-such-store".to_string()),
            Query(RolloutListParams::default()),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::NotFound(_)));
    }

    /// With a capacity-1 LRU, creating a second store must evict the first — but
    /// the evicted store is still reachable, transparently reopened from storage
    /// via the durable registry, not falsely 404'd.
    #[tokio::test]
    async fn evicted_store_is_transparently_reopened() {
        let dir = TempDir::new().unwrap();
        // Force a capacity-1 cache so the second create evicts the first.
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        state.rollout_stores =
            tokio::sync::Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(1).unwrap()));
        let state = Arc::new(state);

        for name in ["exp-a", "exp-b"] {
            let _ = create_rollout_store(
                State(state.clone()),
                Json(CreateRolloutStoreRequest {
                    name: name.to_string(),
                    storage_options: None,
                }),
            )
            .await
            .expect("create");
        }

        // exp-a was evicted (cache holds only exp-b now), yet a lookup succeeds.
        assert_eq!(state.rollout_stores.lock().await.len(), 1);
        let info = get_rollout_store(State(state.clone()), Path("exp-a".to_string()))
            .await
            .expect("evicted store reopens instead of 404");
        assert_eq!(info.name, "exp-a");
    }

    /// `list` is served from the durable registry, so it enumerates every store
    /// that exists — including ones evicted from (or never resident in) the LRU.
    #[tokio::test]
    async fn list_enumerates_all_registered_stores() {
        let dir = TempDir::new().unwrap();
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        state.rollout_stores =
            tokio::sync::Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(1).unwrap()));
        let state = Arc::new(state);

        for name in ["e0", "e1", "e2"] {
            let _ = create_rollout_store(
                State(state.clone()),
                Json(CreateRolloutStoreRequest {
                    name: name.to_string(),
                    storage_options: None,
                }),
            )
            .await
            .expect("create");
        }
        // Only one store is resident, but list must return all three.
        assert_eq!(state.rollout_stores.lock().await.len(), 1);
        let Json(listed) = list_rollout_stores(State(state.clone()))
            .await
            .expect("list");
        let mut names: Vec<String> = listed.stores.into_iter().map(|s| s.name).collect();
        names.sort();
        assert_eq!(names, vec!["e0", "e1", "e2"]);
    }

    /// Delete removes the store from the durable registry, so a subsequent read
    /// 404s and it no longer appears in `list`.
    #[tokio::test]
    async fn delete_unregisters_store() {
        let (state, _dir) = rollout_state().await;
        delete_rollout_store(State(state.clone()), Path("rl".to_string()))
            .await
            .expect("delete");
        let err = get_rollout_store(State(state.clone()), Path("rl".to_string()))
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::NotFound(_)));
        let Json(listed) = list_rollout_stores(State(state.clone())).await.unwrap();
        assert!(listed.stores.is_empty());
    }

    #[tokio::test]
    async fn list_rollouts_applies_json_filters() {
        let (state, _dir) = rollout_state().await;
        let body = serde_json::to_vec(&AddRolloutsRequest {
            records: vec![
                filtered_record("a", "ckpt-1", true),
                filtered_record("b", "ckpt-2", false),
            ],
        })
        .unwrap();
        let _ = add_rollouts(
            State(state.clone()),
            Path("rl".to_string()),
            json_request(body),
        )
        .await
        .unwrap();
        flush_store(&state, "rl").await;

        let Json(response) = list_rollouts(
            State(state),
            Path("rl".to_string()),
            Query(RolloutListParams {
                filters: Some(
                    serde_json::json!({
                        "policy_version": "ckpt-2",
                        "include_in_training": false
                    })
                    .to_string(),
                ),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(response.records.len(), 1);
        assert_eq!(response.records[0].id, "b");
    }

    #[tokio::test]
    async fn list_rollouts_rejects_invalid_filters() {
        let (state, _dir) = rollout_state().await;
        let invalid_json = list_rollouts(
            State(state.clone()),
            Path("rl".to_string()),
            Query(RolloutListParams {
                filters: Some("{not-json}".to_string()),
                ..Default::default()
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(invalid_json, AppError::InvalidRequest(_)));

        let unsupported = list_rollouts(
            State(state),
            Path("rl".to_string()),
            Query(RolloutListParams {
                filters: Some(serde_json::json!({"reward": 1.0}).to_string()),
                ..Default::default()
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(unsupported, AppError::InvalidRequest(_)));
    }
}
