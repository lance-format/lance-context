use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::Response;
use axum::Json;
use chrono::Utc;
use lance_context_api::{
    AddRecordsRequest, AddRecordsResponse, DeleteRecordResponse, GetRecordResponse,
    ListRecordsResponse, RecordDto, RecordPatchDto, RelationshipDto, StateMetadataDto,
    UpdateRecordRequest, UpdateRecordResponse, UpsertRecordRequest, UpsertRecordResponse,
    UpsertRecordsRequest, UpsertRecordsResponse, UpsertResultDto,
};
use lance_context_core::{
    ContextRecord, LifecycleQueryOptions, RecordFilters, RecordPatch, Relationship, StateMetadata,
    LIFECYCLE_ACTIVE,
};
use uuid::Uuid;

use crate::error::AppError;
use crate::state::AppState;

pub async fn add_records(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddRecordsRequest>,
) -> Result<(axum::http::StatusCode, Json<AddRecordsResponse>), AppError> {
    if req.records.is_empty() {
        return Err(AppError::InvalidRequest(
            "records array must not be empty".to_string(),
        ));
    }

    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let run_id = Uuid::new_v4().to_string();
    let mut ids = Vec::with_capacity(req.records.len());
    let mut core_records = Vec::with_capacity(req.records.len());

    for r in &req.records {
        let id = Uuid::new_v4().to_string();
        ids.push(id.clone());
        core_records.push(record_from_add_request(r, id, run_id.clone()));
    }

    let count = core_records.len();
    let mut store = store_lock.write().await;
    let version = store
        .add(&core_records)
        .await
        .map_err(AppError::from_lance)?;

    Ok((
        axum::http::StatusCode::CREATED,
        Json(AddRecordsResponse {
            version,
            ids,
            count,
        }),
    ))
}

pub async fn upsert_record(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpsertRecordRequest>,
) -> Result<(axum::http::StatusCode, Json<UpsertRecordResponse>), AppError> {
    if req.key != "external_id" {
        return Err(AppError::InvalidRequest(format!(
            "upsert key '{}' is not supported; use 'external_id'",
            req.key
        )));
    }
    if req.record.external_id.as_deref().is_none_or(str::is_empty) {
        return Err(AppError::InvalidRequest(
            "upsert requires record.external_id".to_string(),
        ));
    }

    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let record = record_from_add_request(
        &req.record,
        Uuid::new_v4().to_string(),
        Uuid::new_v4().to_string(),
    );
    let mut store = store_lock.write().await;
    let result = store
        .upsert_by_external_id(record)
        .await
        .map_err(AppError::from_lance)?;
    let status = if result.inserted {
        axum::http::StatusCode::CREATED
    } else {
        axum::http::StatusCode::OK
    };

    Ok((
        status,
        Json(UpsertRecordResponse {
            version: result.version,
            inserted: result.inserted,
            replaced_id: result.replaced_id,
            record: record_to_dto(result.record),
        }),
    ))
}

pub async fn upsert_records(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpsertRecordsRequest>,
) -> Result<(axum::http::StatusCode, Json<UpsertRecordsResponse>), AppError> {
    if req.key != "external_id" {
        return Err(AppError::InvalidRequest(format!(
            "upsert key '{}' is not supported; use 'external_id'",
            req.key
        )));
    }
    if req.records.is_empty() {
        return Err(AppError::InvalidRequest(
            "records array must not be empty".to_string(),
        ));
    }
    for (index, record) in req.records.iter().enumerate() {
        if record.external_id.as_deref().is_none_or(str::is_empty) {
            return Err(AppError::InvalidRequest(format!(
                "upsert requires record.external_id (records[{index}])"
            )));
        }
    }

    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let core_records: Vec<ContextRecord> = req
        .records
        .iter()
        .map(|r| record_from_add_request(r, Uuid::new_v4().to_string(), Uuid::new_v4().to_string()))
        .collect();

    let mut store = store_lock.write().await;
    let results = store
        .upsert_many_by_external_id(core_records)
        .await
        .map_err(AppError::from_lance)?;
    let version = results
        .last()
        .map(|r| r.version)
        .unwrap_or_else(|| store.version());

    Ok((
        axum::http::StatusCode::OK,
        Json(UpsertRecordsResponse {
            version,
            results: results
                .into_iter()
                .map(|r| UpsertResultDto {
                    inserted: r.inserted,
                    replaced_id: r.replaced_id,
                    record: record_to_dto(r.record),
                })
                .collect(),
        }),
    ))
}

pub async fn update_record(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateRecordRequest>,
) -> Result<Json<UpdateRecordResponse>, AppError> {
    if req.patch.is_empty() {
        return Err(AppError::InvalidRequest(
            "update requires at least one patch field".to_string(),
        ));
    }

    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let patch = patch_from_dto(&req.patch);
    let mut store = store_lock.write().await;
    let result = match (&req.id, &req.external_id) {
        (Some(id), None) => store.update_by_id(id, patch).await,
        (None, Some(external_id)) => store.update_by_external_id(external_id, patch).await,
        (None, None) => {
            return Err(AppError::InvalidRequest(
                "update requires either id or external_id".to_string(),
            ));
        }
        (Some(_), Some(_)) => {
            return Err(AppError::InvalidRequest(
                "update accepts only one of id or external_id".to_string(),
            ));
        }
    }
    .map_err(AppError::from_lance)?;

    Ok(Json(match result {
        Some(result) => UpdateRecordResponse {
            version: result.version,
            updated: true,
            replaced_id: Some(result.replaced_id),
            record: Some(record_to_dto(result.record)),
        },
        None => UpdateRecordResponse {
            version: store.version(),
            updated: false,
            replaced_id: None,
            record: None,
        },
    }))
}

pub async fn get_record(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Json<GetRecordResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let record = store.get(&id).await.map_err(AppError::from_lance)?;

    Ok(Json(GetRecordResponse {
        record: record.map(record_to_dto),
    }))
}

/// Resolve a record's external payload reference to its raw bytes.
///
/// Returns the bytes with the record's `content_type` (defaulting to
/// `application/octet-stream`). `404` if no such record; `400` if the record
/// carries no external payload reference.
pub async fn fetch_payload(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Response, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let record = store
        .get_by_id(&id)
        .await
        .map_err(AppError::from_lance)?
        .ok_or_else(|| AppError::NotFound(format!("Record '{}' does not exist", id)))?;
    if record.payload_uri.is_none() {
        return Err(AppError::InvalidRequest(format!(
            "record '{}' has no external payload reference to fetch",
            id
        )));
    }
    let bytes = store
        .fetch_payload(&id)
        .await
        .map_err(AppError::from_lance)?
        .ok_or_else(|| AppError::NotFound(format!("Record '{}' does not exist", id)))?;

    let content_type = if record.content_type.is_empty() {
        "application/octet-stream".to_string()
    } else {
        record.content_type
    };
    Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(bytes))
        .map_err(|err| AppError::Internal(err.to_string()))
}

#[derive(serde::Deserialize)]
pub struct ExternalIdParams {
    pub external_id: String,
}

pub async fn get_record_by_external_id(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<ExternalIdParams>,
) -> Result<Json<GetRecordResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let record = store
        .get_by_external_id(&params.external_id)
        .await
        .map_err(AppError::from_lance)?;

    Ok(Json(GetRecordResponse {
        record: record.map(record_to_dto),
    }))
}

pub async fn delete_record(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Json<DeleteRecordResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let mut store = store_lock.write().await;
    let deleted = store
        .delete_by_id(&id)
        .await
        .map_err(AppError::from_lance)?;
    let version = store.version();

    Ok(Json(DeleteRecordResponse { deleted, version }))
}

pub async fn delete_record_by_external_id(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<ExternalIdParams>,
) -> Result<Json<DeleteRecordResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let mut store = store_lock.write().await;
    let deleted = store
        .delete_by_external_id(&params.external_id)
        .await
        .map_err(AppError::from_lance)?;
    let version = store.version();

    Ok(Json(DeleteRecordResponse { deleted, version }))
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct ListParams {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    /// JSON object encoding `RecordFilters`, URL-encoded into the query string.
    pub filters: Option<String>,
    #[serde(default)]
    pub include_expired: bool,
    #[serde(default)]
    pub include_retired: bool,
}

pub async fn list_records(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<ListParams>,
) -> Result<Json<ListRecordsResponse>, AppError> {
    let filters = params
        .filters
        .as_deref()
        .map(|raw| {
            serde_json::from_str(raw)
                .map_err(|err| AppError::InvalidRequest(format!("invalid filters JSON: {err}")))
                .and_then(|value| {
                    RecordFilters::from_json_value(value).map_err(AppError::InvalidRequest)
                })
        })
        .transpose()?;

    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let records = store
        .list_filtered_with_options(
            params.limit,
            params.offset,
            filters.as_ref(),
            LifecycleQueryOptions::new(params.include_expired, params.include_retired),
        )
        .await
        .map_err(AppError::from_lance)?;

    let dtos: Vec<RecordDto> = records.into_iter().map(record_to_dto).collect();

    Ok(Json(ListRecordsResponse { records: dtos }))
}

#[derive(serde::Deserialize)]
pub struct RelatedParams {
    pub target_id: String,
    pub relation: Option<String>,
    pub limit: Option<usize>,
    #[serde(default)]
    pub include_expired: bool,
    #[serde(default)]
    pub include_retired: bool,
}

pub async fn related_records(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<RelatedParams>,
) -> Result<Json<ListRecordsResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let records = store
        .list_related_with_options(
            &params.target_id,
            params.relation.as_deref(),
            params.limit,
            LifecycleQueryOptions::new(params.include_expired, params.include_retired),
        )
        .await
        .map_err(AppError::from_lance)?;

    let dtos: Vec<RecordDto> = records.into_iter().map(record_to_dto).collect();

    Ok(Json(ListRecordsResponse { records: dtos }))
}

pub fn record_to_dto(r: ContextRecord) -> RecordDto {
    RecordDto {
        id: r.id,
        external_id: r.external_id,
        run_id: r.run_id,
        bot_id: r.bot_id,
        session_id: r.session_id,
        tenant: r.tenant,
        source: r.source,
        created_at: r.created_at,
        role: r.role,
        content_type: r.content_type,
        text_payload: r.text_payload,
        binary_payload: r.binary_payload,
        payload_uri: r.payload_uri,
        payload_size: r.payload_size,
        payload_checksum: r.payload_checksum,
        embedding: r.embedding,
        state_metadata: r.state_metadata.map(|sm| StateMetadataDto {
            step: sm.step,
            active_plan_id: sm.active_plan_id,
            tokens_used: sm.tokens_used,
            custom: sm.custom,
        }),
        metadata: r.metadata,
        relationships: r
            .relationships
            .into_iter()
            .map(relationship_to_dto)
            .collect(),
        expires_at: r.expires_at,
        retention_policy: r.retention_policy,
        lifecycle_status: r.lifecycle_status,
        retired_at: r.retired_at,
        retired_reason: r.retired_reason,
        supersedes_id: r.supersedes_id,
        superseded_by_id: r.superseded_by_id,
    }
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

fn patch_from_dto(patch: &RecordPatchDto) -> RecordPatch {
    RecordPatch {
        bot_id: patch.bot_id.clone(),
        session_id: patch.session_id.clone(),
        tenant: patch.tenant.clone(),
        source: patch.source.clone(),
        state_metadata: patch.state_metadata.as_ref().map(|sm| StateMetadata {
            step: sm.step,
            active_plan_id: sm.active_plan_id.clone(),
            tokens_used: sm.tokens_used,
            custom: sm.custom.clone(),
        }),
        metadata: patch.metadata.clone(),
        relationships: patch.relationships.as_ref().map(|relationships| {
            relationships
                .iter()
                .cloned()
                .map(dto_to_relationship)
                .collect()
        }),
        expires_at: patch.expires_at,
        retention_policy: patch.retention_policy.clone(),
        lifecycle_status: patch.lifecycle_status.clone(),
        retired_at: patch.retired_at,
        retired_reason: patch.retired_reason.clone(),
        embedding: patch.embedding.clone(),
        payload_uri: patch.payload_uri.clone(),
        payload_size: patch.payload_size,
        payload_checksum: patch.payload_checksum.clone(),
    }
}

fn record_from_add_request(
    r: &lance_context_api::AddRecordRequest,
    id: String,
    run_id: String,
) -> ContextRecord {
    ContextRecord {
        id,
        external_id: r.external_id.clone(),
        run_id,
        bot_id: r.bot_id.clone(),
        session_id: r.session_id.clone(),
        tenant: r.tenant.clone(),
        source: r.source.clone(),
        created_at: Utc::now(),
        role: r.role.clone(),
        state_metadata: r.state_metadata.as_ref().map(|sm| StateMetadata {
            step: sm.step,
            active_plan_id: sm.active_plan_id.clone(),
            tokens_used: sm.tokens_used,
            custom: sm.custom.clone(),
        }),
        metadata: r.metadata.clone(),
        relationships: r
            .relationships
            .iter()
            .cloned()
            .map(dto_to_relationship)
            .collect(),
        expires_at: r.expires_at,
        retention_policy: r.retention_policy.clone(),
        lifecycle_status: LIFECYCLE_ACTIVE.to_string(),
        retired_at: None,
        retired_reason: None,
        supersedes_id: r.supersedes_id.clone(),
        superseded_by_id: None,
        content_type: r.content_type.clone(),
        text_payload: r.text_payload.clone(),
        binary_payload: r.binary_payload.clone(),
        payload_uri: r.payload_uri.clone(),
        payload_size: r.payload_size,
        payload_checksum: r.payload_checksum.clone(),
        embedding: r.embedding.clone(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::extract::{Path, Query, State};
    use axum::Json;
    use chrono::{Duration, Utc};
    use lance_context_api::{
        AddRecordRequest, AddRecordsRequest, RecordPatchDto, UpdateRecordRequest,
        UpsertRecordRequest, UpsertRecordsRequest,
    };
    use lance_context_core::ContextStore;
    use tempfile::TempDir;
    use tokio::sync::RwLock;

    use super::*;
    use crate::state::AppState;

    async fn test_state(context_name: &str) -> (Arc<AppState>, TempDir) {
        let dir = TempDir::new().unwrap();
        let uri = dir
            .path()
            .join(format!("{context_name}.lance"))
            .to_string_lossy()
            .to_string();
        let store = ContextStore::open(&uri).await.unwrap();
        let mut stores = HashMap::new();
        stores.insert(context_name.to_string(), Arc::new(RwLock::new(store)));
        let state = Arc::new(AppState {
            stores: RwLock::new(stores),
            base_path: dir.path().to_path_buf(),
        });
        (state, dir)
    }

    fn text_record(text: &str) -> AddRecordRequest {
        AddRecordRequest {
            role: "user".to_string(),
            content_type: "text/plain".to_string(),
            text_payload: Some(text.to_string()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn fetch_payload_returns_bytes_404_and_400() {
        let context_name = "ctx";
        let (state, dir) = test_state(context_name).await;
        let object_uri = dir.path().join("media.bin").to_string_lossy().to_string();
        let payload = b"external media bytes".to_vec();

        // Offload the object through the store's object-store path.
        {
            let stores = state.stores.read().await;
            let store = stores.get(context_name).unwrap().read().await;
            store.put_payload(&object_uri, &payload).await.unwrap();
        }

        // Add a record that references the object instead of inlining bytes.
        let record = AddRecordRequest {
            role: "user".to_string(),
            content_type: "image/png".to_string(),
            payload_uri: Some(object_uri.clone()),
            payload_size: Some(payload.len() as i64),
            ..Default::default()
        };
        let (_, Json(add_response)) = add_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(AddRecordsRequest {
                records: vec![record],
            }),
        )
        .await
        .unwrap();
        let id = add_response.ids[0].clone();

        // The payload endpoint streams the resolved bytes with the content type.
        let resp = fetch_payload(
            State(state.clone()),
            Path((context_name.to_string(), id.clone())),
        )
        .await
        .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/png"
        );
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), payload.as_slice());

        // Unknown id -> 404.
        let missing = fetch_payload(
            State(state.clone()),
            Path((context_name.to_string(), "does-not-exist".to_string())),
        )
        .await
        .unwrap_err();
        assert!(matches!(missing, AppError::NotFound(_)));

        // Record without an external reference -> 400.
        let (_, Json(inline)) = add_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(AddRecordsRequest {
                records: vec![text_record("inline only")],
            }),
        )
        .await
        .unwrap();
        let inline_id = inline.ids[0].clone();
        let no_ref = fetch_payload(State(state), Path((context_name.to_string(), inline_id)))
            .await
            .unwrap_err();
        assert!(matches!(no_ref, AppError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn get_and_delete_by_external_id() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;
        let external_id = "s3://bucket/path/doc.md#chunk?index=1";
        let mut record = text_record("stable source chunk");
        record.external_id = Some(external_id.to_string());

        let (_, Json(add_response)) = add_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(AddRecordsRequest {
                records: vec![record],
            }),
        )
        .await
        .unwrap();

        let Json(get_response) = get_record_by_external_id(
            State(state.clone()),
            Path(context_name.to_string()),
            Query(ExternalIdParams {
                external_id: external_id.to_string(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            get_response.record.unwrap().text_payload.as_deref(),
            Some("stable source chunk")
        );

        let Json(delete_response) = delete_record_by_external_id(
            State(state.clone()),
            Path(context_name.to_string()),
            Query(ExternalIdParams {
                external_id: external_id.to_string(),
            }),
        )
        .await
        .unwrap();
        assert!(delete_response.deleted);
        assert!(delete_response.version >= add_response.version);

        let Json(missing_response) = get_record_by_external_id(
            State(state),
            Path(context_name.to_string()),
            Query(ExternalIdParams {
                external_id: external_id.to_string(),
            }),
        )
        .await
        .unwrap();
        assert!(missing_response.record.is_none());
    }

    #[tokio::test]
    async fn delete_by_internal_id_returns_false_when_already_absent() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;

        let (_, Json(add_response)) = add_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(AddRecordsRequest {
                records: vec![text_record("temporary note")],
            }),
        )
        .await
        .unwrap();
        let id = add_response.ids[0].clone();

        let Json(delete_response) = delete_record(
            State(state.clone()),
            Path((context_name.to_string(), id.clone())),
        )
        .await
        .unwrap();
        assert!(delete_response.deleted);

        let Json(second_response) =
            delete_record(State(state), Path((context_name.to_string(), id)))
                .await
                .unwrap();
        assert!(!second_response.deleted);
    }

    #[tokio::test]
    async fn upsert_by_external_id_inserts_then_replaces_visible_record() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;
        let external_id = "doc-123#chunk-1";

        let mut first = text_record("old value");
        first.external_id = Some(external_id.to_string());
        let (insert_status, Json(inserted)) = upsert_record(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(UpsertRecordRequest {
                record: first,
                key: "external_id".to_string(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(insert_status, axum::http::StatusCode::CREATED);
        assert!(inserted.inserted);
        assert!(inserted.replaced_id.is_none());

        let mut replacement = text_record("new value");
        replacement.external_id = Some(external_id.to_string());
        let (replace_status, Json(replaced)) = upsert_record(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(UpsertRecordRequest {
                record: replacement,
                key: "external_id".to_string(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(replace_status, axum::http::StatusCode::OK);
        assert!(!replaced.inserted);
        assert_eq!(
            replaced.replaced_id.as_deref(),
            Some(inserted.record.id.as_str())
        );
        assert_eq!(
            replaced.record.supersedes_id.as_deref(),
            Some(inserted.record.id.as_str())
        );

        let Json(response) = list_records(
            State(state),
            Path(context_name.to_string()),
            Query(ListParams {
                limit: None,
                offset: None,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(response.records.len(), 1);
        assert_eq!(
            response.records[0].text_payload.as_deref(),
            Some("new value")
        );
    }

    #[tokio::test]
    async fn upsert_records_batch_inserts_and_replaces() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;

        let mut a = text_record("a-old");
        a.external_id = Some("ext-a".to_string());
        let mut b = text_record("b-value");
        b.external_id = Some("ext-b".to_string());

        // First batch: two inserts.
        let (status, Json(first)) = upsert_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(UpsertRecordsRequest {
                records: vec![a, b],
                key: "external_id".to_string(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(first.results.len(), 2);
        assert!(first.results.iter().all(|r| r.inserted));
        let a_id = first.results[0].record.id.clone();

        // Second batch: replace ext-a, insert ext-c.
        let mut a_new = text_record("a-new");
        a_new.external_id = Some("ext-a".to_string());
        let mut c = text_record("c-value");
        c.external_id = Some("ext-c".to_string());
        let (_, Json(second)) = upsert_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(UpsertRecordsRequest {
                records: vec![a_new, c],
                key: "external_id".to_string(),
            }),
        )
        .await
        .unwrap();
        assert!(!second.results[0].inserted);
        assert_eq!(
            second.results[0].replaced_id.as_deref(),
            Some(a_id.as_str())
        );
        assert!(second.results[1].inserted);

        // ext-a now resolves to the replacement; three external_ids visible.
        let Json(after) = get_record_by_external_id(
            State(state.clone()),
            Path(context_name.to_string()),
            Query(ExternalIdParams {
                external_id: "ext-a".to_string(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(after.record.unwrap().text_payload.as_deref(), Some("a-new"));

        let Json(listed) = list_records(
            State(state),
            Path(context_name.to_string()),
            Query(ListParams {
                limit: None,
                offset: None,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(listed.records.len(), 3);
    }

    #[tokio::test]
    async fn upsert_records_batch_rejects_empty() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;
        let err = upsert_records(
            State(state),
            Path(context_name.to_string()),
            Json(UpsertRecordsRequest {
                records: vec![],
                key: "external_id".to_string(),
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn upsert_records_batch_requires_external_id() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;
        let err = upsert_records(
            State(state),
            Path(context_name.to_string()),
            Json(UpsertRecordsRequest {
                records: vec![text_record("no external id")],
                key: "external_id".to_string(),
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn update_by_external_id_patches_visible_record() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;
        let external_id = "doc-123#chunk-1";

        let mut record = text_record("stable value");
        record.external_id = Some(external_id.to_string());
        let (_, Json(add_response)) = add_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(AddRecordsRequest {
                records: vec![record],
            }),
        )
        .await
        .unwrap();
        let old_id = add_response.ids[0].clone();

        let Json(updated) = update_record(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(UpdateRecordRequest {
                id: None,
                external_id: Some(external_id.to_string()),
                patch: RecordPatchDto {
                    metadata: Some(serde_json::json!({"revision": 2})),
                    relationships: Some(vec![RelationshipDto {
                        target_id: "doc-123".to_string(),
                        relation: "derived_from".to_string(),
                        weight: None,
                    }]),
                    ..Default::default()
                },
            }),
        )
        .await
        .unwrap();

        assert!(updated.updated);
        assert_eq!(updated.replaced_id.as_deref(), Some(old_id.as_str()));
        let record = updated.record.unwrap();
        assert_ne!(record.id, old_id);
        assert_eq!(record.external_id.as_deref(), Some(external_id));
        assert_eq!(record.text_payload.as_deref(), Some("stable value"));
        assert_eq!(record.metadata, Some(serde_json::json!({"revision": 2})));
        assert_eq!(record.relationships.len(), 1);
        assert_eq!(record.supersedes_id.as_deref(), Some(old_id.as_str()));

        let Json(response) = list_records(
            State(state),
            Path(context_name.to_string()),
            Query(ListParams {
                limit: None,
                offset: None,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(response.records.len(), 1);
        assert_eq!(response.records[0].id, record.id);
    }

    #[tokio::test]
    async fn related_records_filters_by_target_and_relation() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;
        let mut related = text_record("record that cites the runbook");
        related.relationships = vec![RelationshipDto {
            target_id: "doc://runbook#chunk-1".to_string(),
            relation: "cites".to_string(),
            weight: Some(0.75),
        }];

        let _ = add_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(AddRecordsRequest {
                records: vec![related, text_record("unrelated record")],
            }),
        )
        .await
        .unwrap();

        let Json(response) = related_records(
            State(state),
            Path(context_name.to_string()),
            Query(RelatedParams {
                target_id: "doc://runbook#chunk-1".to_string(),
                relation: Some("cites".to_string()),
                limit: Some(10),
                include_expired: false,
                include_retired: false,
            }),
        )
        .await
        .unwrap();

        assert_eq!(response.records.len(), 1);
        assert_eq!(
            response.records[0].text_payload.as_deref(),
            Some("record that cites the runbook")
        );
        assert_eq!(response.records[0].relationships.len(), 1);
    }

    async fn list_with(
        state: &Arc<AppState>,
        context_name: &str,
        params: ListParams,
    ) -> Vec<RecordDto> {
        let Json(response) = list_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Query(params),
        )
        .await
        .unwrap();
        response.records
    }

    #[tokio::test]
    async fn list_filters_by_metadata_and_builtin_fields() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;

        let mut alpha = text_record("alpha");
        alpha.metadata = Some(serde_json::json!({"tenant": "acme"}));
        let mut bravo = text_record("bravo");
        bravo.role = "assistant".to_string();
        bravo.metadata = Some(serde_json::json!({"tenant": "globex"}));
        let mut charlie = text_record("charlie");
        charlie.metadata = Some(serde_json::json!({"tenant": "acme"}));
        let _ = add_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(AddRecordsRequest {
                records: vec![alpha, bravo, charlie],
            }),
        )
        .await
        .unwrap();

        // Metadata filter restricts to tenant=acme (alpha + charlie).
        let records = list_with(
            &state,
            context_name,
            ListParams {
                filters: Some(r#"{"tenant": "acme"}"#.to_string()),
                ..Default::default()
            },
        )
        .await;
        let texts: Vec<&str> = records
            .iter()
            .filter_map(|r| r.text_payload.as_deref())
            .collect();
        assert_eq!(records.len(), 2);
        assert!(texts.contains(&"alpha"));
        assert!(texts.contains(&"charlie"));

        // Built-in field filter restricts to role=assistant (bravo).
        let records = list_with(
            &state,
            context_name,
            ListParams {
                filters: Some(r#"{"role": "assistant"}"#.to_string()),
                ..Default::default()
            },
        )
        .await;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].text_payload.as_deref(), Some("bravo"));
    }

    #[tokio::test]
    async fn list_respects_expired_visibility() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;

        let fresh = text_record("fresh");
        let mut stale = text_record("stale");
        stale.expires_at = Some(Utc::now() - Duration::hours(1));
        let _ = add_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(AddRecordsRequest {
                records: vec![fresh, stale],
            }),
        )
        .await
        .unwrap();

        // Default listing hides the expired record.
        let records = list_with(&state, context_name, ListParams::default()).await;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].text_payload.as_deref(), Some("fresh"));

        // include_expired surfaces it.
        let records = list_with(
            &state,
            context_name,
            ListParams {
                include_expired: true,
                ..Default::default()
            },
        )
        .await;
        assert_eq!(records.len(), 2);
    }

    #[tokio::test]
    async fn list_respects_retired_visibility() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;

        let mut original = text_record("v1");
        original.external_id = Some("doc-1".to_string());
        let (_, Json(add_response)) = add_records(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(AddRecordsRequest {
                records: vec![original],
            }),
        )
        .await
        .unwrap();
        let old_id = add_response.ids[0].clone();

        let Json(updated) = update_record(
            State(state.clone()),
            Path(context_name.to_string()),
            Json(UpdateRecordRequest {
                id: None,
                external_id: Some("doc-1".to_string()),
                patch: RecordPatchDto {
                    metadata: Some(serde_json::json!({"revision": 2})),
                    ..Default::default()
                },
            }),
        )
        .await
        .unwrap();
        assert!(updated.updated);

        // Default listing returns only the visible successor.
        let records = list_with(&state, context_name, ListParams::default()).await;
        assert_eq!(records.len(), 1);
        assert_ne!(records[0].id, old_id);

        // include_retired surfaces the superseded original too.
        let records = list_with(
            &state,
            context_name,
            ListParams {
                include_retired: true,
                ..Default::default()
            },
        )
        .await;
        assert_eq!(records.len(), 2);
        assert!(records.iter().any(|r| r.id == old_id));
    }

    #[tokio::test]
    async fn list_rejects_invalid_filters_json() {
        let context_name = "ctx";
        let (state, _dir) = test_state(context_name).await;

        let result = list_records(
            State(state),
            Path(context_name.to_string()),
            Query(ListParams {
                filters: Some("not json".to_string()),
                ..Default::default()
            }),
        )
        .await;
        assert!(matches!(result, Err(AppError::InvalidRequest(_))));
    }
}
