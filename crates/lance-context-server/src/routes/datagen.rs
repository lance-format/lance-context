use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use lance_context_api::{
    AddDatagenEventsRequest, AddDatagenEventsResponse, CreateDatagenStoreRequest,
    DatagenRootItemStatusesResponse, DatagenStoreApi, DatagenStoreInfo,
    GetFoldedDatagenItemResponse, ListDatagenEventsResponse, ListDatagenFailuresResponse,
    ListDatagenStoresResponse,
};
use lance_context_core::{DatagenStore, DatagenStoreOptions};
use tokio::sync::RwLock;

use crate::error::AppError;
use crate::state::AppState;

/// Upper bound on a single datagen append request body. FIELD blobs are
/// offloaded to a content-addressed artifact store before the event reaches the
/// log, so events themselves are small; the ceiling still bounds a pathological
/// batch.
pub const MAX_DATAGEN_UPLOAD_BYTES: usize = 256 * 1024 * 1024;

pub async fn create_datagen_store(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateDatagenStoreRequest>,
) -> Result<(StatusCode, Json<DatagenStoreInfo>), AppError> {
    AppState::validate_name(&req.name)?;
    if state
        .datagen_registry
        .write()
        .await
        .contains(&req.name)
        .await
        .map_err(AppError::from_lance)?
    {
        return Err(AppError::AlreadyExists(format!(
            "Datagen store '{}' already exists",
            req.name
        )));
    }

    let uri = state.datagen_uri(&req.name);
    let options = DatagenStoreOptions {
        storage_options: req.storage_options,
        shard_id: state.instance_id.clone(),
        merge_after_generations: None,
        cleanup_interval_secs: None,
    };
    let store = DatagenStore::open_with_options(&uri, options)
        .await
        .map_err(AppError::from_lance)?;
    let version = store.version();

    let store = Arc::new(RwLock::new(store));
    state.register_datagen(&req.name, &uri, store).await?;

    Ok((
        StatusCode::CREATED,
        Json(DatagenStoreInfo {
            name: req.name,
            uri,
            version: Some(version),
        }),
    ))
}

pub async fn list_datagen_stores(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ListDatagenStoresResponse>, AppError> {
    let entries = state
        .datagen_registry
        .write()
        .await
        .list()
        .await
        .map_err(AppError::from_lance)?;
    let stores = entries
        .into_iter()
        .map(|entry| DatagenStoreInfo {
            name: entry.name,
            uri: entry.uri,
            version: None,
        })
        .collect();
    Ok(Json(ListDatagenStoresResponse { stores }))
}

pub async fn get_datagen_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<DatagenStoreInfo>, AppError> {
    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let store = store_lock.read().await;
    Ok(Json(DatagenStoreInfo {
        name: name.clone(),
        uri: state.datagen_uri(&name),
        version: Some(store.version()),
    }))
}

pub async fn delete_datagen_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    if !state.unregister_datagen(&name).await? {
        return Err(AppError::NotFound(format!(
            "Datagen store '{}' does not exist",
            name
        )));
    }
    let uri = state.datagen_uri(&name);
    if let Err(e) = tokio::fs::remove_dir_all(&uri).await {
        tracing::warn!("Failed to remove datagen data at {}: {}", uri, e);
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Which append semantics the `/events` endpoint applies.
#[derive(Debug, Default, serde::Deserialize)]
pub struct AppendParams {
    /// When `true`, the batch is committed as one atomic checkpoint (FIELD_*
    /// events plus exactly one STEP_COMPLETED) via `append_checkpoint`. Default
    /// appends the events as one raw MemWAL generation.
    #[serde(default)]
    pub checkpoint: bool,
}

pub async fn add_datagen_events(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<AppendParams>,
    Json(req): Json<AddDatagenEventsRequest>,
) -> Result<(StatusCode, Json<AddDatagenEventsResponse>), AppError> {
    if req.events.is_empty() {
        return Err(AppError::InvalidRequest(
            "events array must not be empty".to_string(),
        ));
    }

    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let mut store = store_lock.write().await;
    let resp = if params.checkpoint {
        DatagenStoreApi::append_checkpoint(&mut *store, &req.events).await
    } else {
        DatagenStoreApi::append(&mut *store, &req.events).await
    }
    .map_err(AppError::from_context)?;

    Ok((StatusCode::CREATED, Json(resp)))
}

/// Blob projection for the fold endpoint.
#[derive(Debug, Default, serde::Deserialize)]
pub struct FoldParams {
    /// When `true`, blob fields are materialized inline instead of left lazy for `get_blob`.
    #[serde(default)]
    pub load_blobs: bool,
}

async fn fold_datagen_item_refreshing_on_miss(
    store_lock: &RwLock<DatagenStore>,
    item_id: &str,
    load_blobs: bool,
) -> Result<Option<lance_context_api::FoldedDatagenItemDto>, AppError> {
    {
        let store = store_lock.read().await;
        let item = DatagenStoreApi::fold_item_with_blobs(&*store, item_id, load_blobs)
            .await
            .map_err(AppError::from_context)?;
        if item.is_some() || store.is_version_pinned() {
            return Ok(item);
        }
    }

    let mut store = store_lock.write().await;
    if !store.is_version_pinned() {
        store.refresh_latest().await.map_err(AppError::from_lance)?;
    }
    DatagenStoreApi::fold_item_with_blobs(&*store, item_id, load_blobs)
        .await
        .map_err(AppError::from_context)
}

async fn datagen_failures_refreshing_on_empty(
    store_lock: &RwLock<DatagenStore>,
    item_id: &str,
) -> Result<Vec<lance_context_api::DatagenFailureDto>, AppError> {
    {
        let store = store_lock.read().await;
        let failures = DatagenStoreApi::item_failures(&*store, item_id)
            .await
            .map_err(AppError::from_context)?;
        if !failures.is_empty() || store.is_version_pinned() {
            return Ok(failures);
        }
    }

    let mut store = store_lock.write().await;
    if !store.is_version_pinned() {
        store.refresh_latest().await.map_err(AppError::from_lance)?;
    }
    DatagenStoreApi::item_failures(&*store, item_id)
        .await
        .map_err(AppError::from_context)
}

async fn datagen_events_for_root_refreshing_on_empty(
    store_lock: &RwLock<DatagenStore>,
    root_item_id: &str,
) -> Result<Vec<lance_context_api::DatagenEventDto>, AppError> {
    {
        let store = store_lock.read().await;
        let events = DatagenStoreApi::events_for_root(&*store, root_item_id)
            .await
            .map_err(AppError::from_context)?;
        if !events.is_empty() || store.is_version_pinned() {
            return Ok(events);
        }
    }

    let mut store = store_lock.write().await;
    if !store.is_version_pinned() {
        store.refresh_latest().await.map_err(AppError::from_lance)?;
    }
    DatagenStoreApi::events_for_root(&*store, root_item_id)
        .await
        .map_err(AppError::from_context)
}

async fn datagen_root_statuses_refreshing_on_missing(
    store_lock: &RwLock<DatagenStore>,
    ids: &[String],
) -> Result<DatagenRootItemStatusesResponse, AppError> {
    {
        let store = store_lock.read().await;
        let statuses = DatagenStoreApi::root_item_statuses(&*store, ids)
            .await
            .map_err(AppError::from_context)?;
        if statuses.statuses.len() == ids.len() || store.is_version_pinned() {
            return Ok(statuses);
        }
    }

    let mut store = store_lock.write().await;
    if !store.is_version_pinned() {
        store.refresh_latest().await.map_err(AppError::from_lance)?;
    }
    DatagenStoreApi::root_item_statuses(&*store, ids)
        .await
        .map_err(AppError::from_context)
}

async fn get_datagen_blob_refreshing_on_miss(
    store_lock: &RwLock<DatagenStore>,
    event_id: &str,
) -> Result<Option<Vec<u8>>, AppError> {
    {
        let store = store_lock.read().await;
        let bytes = DatagenStoreApi::get_blob(&*store, event_id)
            .await
            .map_err(AppError::from_context)?;
        if bytes.is_some() || store.is_version_pinned() {
            return Ok(bytes);
        }
    }

    let mut store = store_lock.write().await;
    if !store.is_version_pinned() {
        store.refresh_latest().await.map_err(AppError::from_lance)?;
    }
    DatagenStoreApi::get_blob(&*store, event_id)
        .await
        .map_err(AppError::from_context)
}

pub async fn fold_datagen_item(
    State(state): State<Arc<AppState>>,
    Path((name, item_id)): Path<(String, String)>,
    Query(params): Query<FoldParams>,
) -> Result<Json<GetFoldedDatagenItemResponse>, AppError> {
    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let item =
        fold_datagen_item_refreshing_on_miss(&store_lock, &item_id, params.load_blobs).await?;
    Ok(Json(GetFoldedDatagenItemResponse { item }))
}

/// Aggregate the whole log into a run overview (per-status root counts, failure
/// counts by error type, completed-step counts).
pub async fn datagen_overview(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<lance_context_api::DatagenRunOverviewDto>, AppError> {
    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let store = store_lock.read().await;
    let overview = DatagenStoreApi::overview(&*store)
        .await
        .map_err(AppError::from_context)?;
    Ok(Json(overview))
}

pub async fn datagen_item_failures(
    State(state): State<Arc<AppState>>,
    Path((name, item_id)): Path<(String, String)>,
) -> Result<Json<ListDatagenFailuresResponse>, AppError> {
    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let failures = datagen_failures_refreshing_on_empty(&store_lock, &item_id).await?;
    Ok(Json(ListDatagenFailuresResponse { failures }))
}

/// Dump every raw event whose root item is `root_item_id`. The server does no
/// fold/tree assembly; the client builds the item tree from these events.
pub async fn datagen_events_for_root(
    State(state): State<Arc<AppState>>,
    Path((name, root_item_id)): Path<(String, String)>,
) -> Result<Json<lance_context_api::ListDatagenEventsResponse>, AppError> {
    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let events = datagen_events_for_root_refreshing_on_empty(&store_lock, &root_item_id).await?;
    Ok(Json(ListDatagenEventsResponse { events }))
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct RootStatusParams {
    /// Comma-separated list of root item ids to classify.
    pub ids: Option<String>,
}

pub async fn datagen_root_item_statuses(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<RootStatusParams>,
) -> Result<Json<lance_context_api::DatagenRootItemStatusesResponse>, AppError> {
    let ids: Vec<String> = params
        .ids
        .as_deref()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let resp = datagen_root_statuses_refreshing_on_missing(&store_lock, &ids).await?;
    Ok(Json(resp))
}

/// Materialize one FIELD_* event's offloaded blob bytes by event id. The bytes
/// are opaque, so they return as `application/octet-stream`; `404` when the
/// event or its payload is absent.
pub async fn fetch_datagen_blob(
    State(state): State<Arc<AppState>>,
    Path((name, event_id)): Path<(String, String)>,
) -> Result<axum::response::Response, AppError> {
    use axum::http::header;
    use axum::response::IntoResponse;

    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let bytes = get_datagen_blob_refreshing_on_miss(&store_lock, &event_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Datagen event '{}' has no blob", event_id)))?;

    Ok(([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response())
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use lance_context_api::CreateDatagenStoreRequest;
    use lance_context_core::{
        datagen_event_id, DatagenBlobValue, DatagenEvent, DatagenEventType, DatagenItemStatus,
        DatagenStepKind, DatagenValue, DATAGEN_SCHEMA_VERSION,
    };
    use tempfile::TempDir;

    use super::*;

    async fn test_state() -> (Arc<AppState>, TempDir) {
        let dir = TempDir::new().unwrap();
        let state = Arc::new(AppState::new_for_test(dir.path().to_path_buf()).await);
        (state, dir)
    }

    fn event(
        item_id: &str,
        seq: i64,
        checkpoint_id: &str,
        ordinal: u32,
        event_type: DatagenEventType,
    ) -> DatagenEvent {
        DatagenEvent {
            event_id: datagen_event_id(item_id, checkpoint_id, ordinal),
            item_id: item_id.to_string(),
            root_item_id: item_id.to_string(),
            parent_item_id: None,
            item_seq: seq,
            checkpoint_id: checkpoint_id.to_string(),
            event_type,
            step_name: None,
            step_kind: None,
            step_index: None,
            enclosing_step: None,
            selector_step: None,
            attempt: 0,
            run_id: "external-run".to_string(),
            writer_epoch: "external-writer".to_string(),
            field_name: None,
            field_type: None,
            codec_version: None,
            value: None,
            query_tags: None,
            status: Some(DatagenItemStatus::Running),
            error_type: None,
            error_dump: None,
            traceback: None,
            event_ts: Utc::now(),
            schema_version: DATAGEN_SCHEMA_VERSION,
        }
    }

    #[tokio::test]
    async fn point_reads_refresh_a_base_advanced_by_an_external_writer() {
        let (state, _dir) = test_state().await;
        for name in ["fold-store", "blob-store"] {
            let _ = create_datagen_store(
                State(state.clone()),
                Json(CreateDatagenStoreRequest {
                    name: name.to_string(),
                    storage_options: None,
                }),
            )
            .await
            .unwrap();
        }

        let cached_fold = state.get_or_open_datagen_store("fold-store").await.unwrap();
        let mut fold_writer = DatagenStore::open_existing_with_options(
            &state.datagen_uri("fold-store"),
            DatagenStoreOptions {
                shard_id: Some("external-fold-writer".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let created = event(
            "merged-item",
            0,
            "created",
            0,
            DatagenEventType::ItemCreated,
        );
        fold_writer.append(&[created]).await.unwrap();
        assert_eq!(fold_writer.cleanup_own_shard().await.unwrap(), 1);
        assert_eq!(fold_writer.pending_wal_generations().await.unwrap(), 0);
        assert!(fold_writer.version() > cached_fold.read().await.version());

        let Json(found) = fold_datagen_item(
            State(state.clone()),
            Path(("fold-store".to_string(), "merged-item".to_string())),
            Query(FoldParams { load_blobs: false }),
        )
        .await
        .unwrap();
        assert_eq!(found.item.unwrap().item_id, "merged-item");
        assert_eq!(cached_fold.read().await.version(), fold_writer.version());

        let cached_blob = state.get_or_open_datagen_store("blob-store").await.unwrap();
        let mut blob_writer = DatagenStore::open_existing_with_options(
            &state.datagen_uri("blob-store"),
            DatagenStoreOptions {
                shard_id: Some("external-blob-writer".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let payload = b"externally merged blob".to_vec();
        let mut blob_event = event("blob-item", 0, "blob", 0, DatagenEventType::FieldSet);
        blob_event.step_name = Some("capture".to_string());
        blob_event.step_kind = Some(DatagenStepKind::Leaf);
        blob_event.step_index = Some(0);
        blob_event.field_name = Some("artifact".to_string());
        blob_event.field_type = Some("blob".to_string());
        blob_event.codec_version = Some(1);
        blob_event.value = Some(DatagenValue::Blob(DatagenBlobValue {
            bytes: Some(payload.clone()),
            size: payload.len() as i64,
            checksum: None,
        }));
        blob_event.status = None;
        let event_id = blob_event.event_id.clone();
        blob_writer.append(&[blob_event]).await.unwrap();
        assert_eq!(blob_writer.cleanup_own_shard().await.unwrap(), 1);
        assert_eq!(blob_writer.pending_wal_generations().await.unwrap(), 0);
        assert!(blob_writer.version() > cached_blob.read().await.version());

        let response = fetch_datagen_blob(State(state), Path(("blob-store".to_string(), event_id)))
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), payload.as_slice());
        assert_eq!(cached_blob.read().await.version(), blob_writer.version());
    }
}
