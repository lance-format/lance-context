use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::Json;
use chrono::Utc;
use lance_context_api::{
    AddRecordsRequest, AddRecordsResponse, GetRecordResponse, ListRecordsResponse, RecordDto,
    StateMetadataDto,
};
use lance_context_core::{ContextRecord, StateMetadata, LIFECYCLE_ACTIVE};
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
        core_records.push(ContextRecord {
            id,
            external_id: r.external_id.clone(),
            run_id: run_id.clone(),
            bot_id: r.bot_id.clone(),
            session_id: r.session_id.clone(),
            created_at: Utc::now(),
            role: r.role.clone(),
            state_metadata: r.state_metadata.as_ref().map(|sm| StateMetadata {
                step: sm.step,
                active_plan_id: sm.active_plan_id.clone(),
                tokens_used: sm.tokens_used,
                custom: sm.custom.clone(),
            }),
            metadata: r.metadata.clone(),
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
            embedding: r.embedding.clone(),
        });
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

#[derive(serde::Deserialize)]
pub struct ListParams {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

pub async fn list_records(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<ListParams>,
) -> Result<Json<ListRecordsResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let records = store
        .list(params.limit, params.offset)
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
        created_at: r.created_at,
        role: r.role,
        content_type: r.content_type,
        text_payload: r.text_payload,
        binary_payload: r.binary_payload,
        embedding: r.embedding,
        state_metadata: r.state_metadata.map(|sm| StateMetadataDto {
            step: sm.step,
            active_plan_id: sm.active_plan_id,
            tokens_used: sm.tokens_used,
            custom: sm.custom,
        }),
        metadata: r.metadata,
        expires_at: r.expires_at,
        retention_policy: r.retention_policy,
        lifecycle_status: r.lifecycle_status,
        retired_at: r.retired_at,
        retired_reason: r.retired_reason,
        supersedes_id: r.supersedes_id,
        superseded_by_id: r.superseded_by_id,
    }
}
