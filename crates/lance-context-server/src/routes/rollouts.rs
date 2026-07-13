use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{FromRequest, Multipart, Path, Query, Request, State};
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::Json;
use chrono::Utc;
use lance_context_api::{
    AddRolloutRequest, AddRolloutsRequest, AddRolloutsResponse, CheckoutRequest,
    CreateRolloutStoreRequest, GetRolloutResponse, ListRolloutStoresResponse, ListRolloutsResponse,
    RelationshipDto, RolloutRecordDto, RolloutStoreInfo, VersionResponse,
};
use lance_context_core::{Relationship, RolloutRecord, RolloutStore, RolloutStoreOptions};
use tokio::sync::RwLock;

use crate::error::AppError;
use crate::state::AppState;

/// Upper bound on a single rollout append request body (JSON or multipart).
/// Artifact blobs routinely exceed axum's 2 MiB default, so the batch endpoint
/// raises the ceiling well above it while still bounding memory.
pub const MAX_ROLLOUT_UPLOAD_BYTES: usize = 1024 * 1024 * 1024;

pub async fn create_rollout_store(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateRolloutStoreRequest>,
) -> Result<(StatusCode, Json<RolloutStoreInfo>), AppError> {
    let stores = state.rollout_stores.read().await;
    if stores.contains_key(&req.name) {
        return Err(AppError::AlreadyExists(format!(
            "Rollout store '{}' already exists",
            req.name
        )));
    }
    drop(stores);

    let uri = state.rollout_uri(&req.name);
    let options = RolloutStoreOptions {
        storage_options: req.storage_options,
        shard_id: state.instance_id.clone(),
        merge_after_generations: (state.rollout_merge_after_generations > 0)
            .then_some(state.rollout_merge_after_generations),
        cleanup_interval_secs: (state.rollout_cleanup_interval_secs > 0)
            .then_some(state.rollout_cleanup_interval_secs),
        cleanup_min_generations: Some(state.rollout_cleanup_min_generations),
    };

    let store = RolloutStore::open_with_options(&uri, options)
        .await
        .map_err(AppError::from_lance)?;
    let version = store.version();

    let store = Arc::new(RwLock::new(store));
    // Start the periodic per-shard WAL-cleanup timer (no-op when the interval is
    // disabled). The handle is detached: it is aborted when the store is dropped
    // on delete, and otherwise runs for the server's lifetime.
    let _cleanup = RolloutStore::spawn_periodic_cleanup(store.clone());

    let mut stores = state.rollout_stores.write().await;
    stores.insert(req.name.clone(), store);

    Ok((
        StatusCode::CREATED,
        Json(RolloutStoreInfo {
            name: req.name,
            uri,
            version,
        }),
    ))
}

pub async fn list_rollout_stores(
    State(state): State<Arc<AppState>>,
) -> Json<ListRolloutStoresResponse> {
    let stores = state.rollout_stores.read().await;
    let mut out = Vec::with_capacity(stores.len());

    for (name, store_lock) in stores.iter() {
        let store = store_lock.read().await;
        out.push(RolloutStoreInfo {
            name: name.clone(),
            uri: state.rollout_uri(name),
            version: store.version(),
        });
    }

    Json(ListRolloutStoresResponse { stores: out })
}

pub async fn get_rollout_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<RolloutStoreInfo>, AppError> {
    let stores = state.rollout_stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Rollout store '{}' does not exist", name)))?;
    let store = store_lock.read().await;

    Ok(Json(RolloutStoreInfo {
        name: name.clone(),
        uri: state.rollout_uri(&name),
        version: store.version(),
    }))
}

pub async fn delete_rollout_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    let mut stores = state.rollout_stores.write().await;
    if stores.remove(&name).is_none() {
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

    let stores = state.rollout_stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Rollout store '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
    let core_records: Vec<RolloutRecord> = records
        .iter()
        .map(rollout_record_from_add_request)
        .collect();
    let count = core_records.len();

    let mut store = store_lock.write().await;
    let version = store
        .add(&core_records)
        .await
        .map_err(AppError::from_lance)?;

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
}

pub async fn list_rollouts(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<RolloutListParams>,
) -> Result<Json<ListRolloutsResponse>, AppError> {
    let stores = state.rollout_stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Rollout store '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let records = store
        .list(params.limit, params.offset)
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
    let stores = state.rollout_stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Rollout store '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let record = store.get_by_id(&id).await.map_err(AppError::from_lance)?;

    Ok(Json(GetRolloutResponse {
        record: record.map(rollout_record_to_dto),
    }))
}

/// Materialize a rollout row's `binary_payload` bytes. The artifact
/// bytes are opaque, so they stream as `application/octet-stream`. `404` when
/// the row is absent or carries no payload.
pub async fn fetch_rollout_blob(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Response, AppError> {
    let stores = state.rollout_stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Rollout store '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let bytes = store
        .get_blob(&id)
        .await
        .map_err(AppError::from_lance)?
        .ok_or_else(|| AppError::NotFound(format!("Rollout '{}' has no payload", id)))?;

    Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .body(Body::from(bytes))
        .map_err(|err| AppError::Internal(err.to_string()))
}

pub async fn checkout_rollout(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<CheckoutRequest>,
) -> Result<Json<VersionResponse>, AppError> {
    let stores = state.rollout_stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Rollout store '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let mut store = store_lock.write().await;
    store
        .checkout(req.version)
        .await
        .map_err(AppError::from_lance)?;

    Ok(Json(VersionResponse {
        version: store.version(),
    }))
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
    use std::collections::HashMap;
    use std::sync::Arc;

    use lance_context_api::CreateRolloutStoreRequest;
    use tempfile::TempDir;
    use tokio::sync::RwLock;

    use super::*;
    use crate::state::AppState;

    async fn rollout_state() -> (Arc<AppState>, TempDir) {
        let dir = TempDir::new().unwrap();
        let state = Arc::new(AppState {
            stores: RwLock::new(HashMap::new()),
            rollout_stores: RwLock::new(HashMap::new()),
            base_path: dir.path().to_path_buf(),
            instance_id: None,
            rollout_merge_after_generations: 0,
        });
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

    fn record_with_size(id: &str, payload_size: Option<i64>) -> AddRolloutRequest {
        AddRolloutRequest {
            id: id.to_string(),
            rollout_id: format!("traj-{id}"),
            payload_size,
            ..Default::default()
        }
    }

    fn json_request(body: Vec<u8>) -> Request {
        Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap()
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
}
