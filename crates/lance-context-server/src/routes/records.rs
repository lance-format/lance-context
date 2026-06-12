use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::Json;
use chrono::Utc;
use lance_context_api::{
    AddRecordsRequest, AddRecordsResponse, DeleteRecordResponse, GetRecordResponse,
    ListRecordsResponse, RecordDto, RelationshipDto, StateMetadataDto,
};
use lance_context_core::{
    ContextRecord, LifecycleQueryOptions, Relationship, StateMetadata, LIFECYCLE_ACTIVE,
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::extract::{Path, Query, State};
    use axum::Json;
    use lance_context_api::{AddRecordRequest, AddRecordsRequest};
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
}
