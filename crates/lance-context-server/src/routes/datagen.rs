use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use lance_context_api::{
    AddDatagenEventsRequest, AddDatagenEventsResponse, CreateDatagenStoreRequest, DatagenStoreApi,
    DatagenStoreInfo, GetFoldedDatagenItemResponse, ListDatagenFailuresResponse,
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

pub async fn fold_datagen_item(
    State(state): State<Arc<AppState>>,
    Path((name, item_id)): Path<(String, String)>,
    Query(params): Query<FoldParams>,
) -> Result<Json<GetFoldedDatagenItemResponse>, AppError> {
    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let store = store_lock.read().await;
    let item = DatagenStoreApi::fold_item_with_blobs(&*store, &item_id, params.load_blobs)
        .await
        .map_err(AppError::from_context)?;
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
    let store = store_lock.read().await;
    let failures = DatagenStoreApi::item_failures(&*store, &item_id)
        .await
        .map_err(AppError::from_context)?;
    Ok(Json(ListDatagenFailuresResponse { failures }))
}

/// Dump every raw event whose root item is `root_item_id`. The server does no
/// fold/tree assembly; the client builds the item tree from these events.
pub async fn datagen_events_for_root(
    State(state): State<Arc<AppState>>,
    Path((name, root_item_id)): Path<(String, String)>,
) -> Result<Json<lance_context_api::ListDatagenEventsResponse>, AppError> {
    let store_lock = state.get_or_open_datagen_store(&name).await?;
    let store = store_lock.read().await;
    let events = DatagenStoreApi::events_for_root(&*store, &root_item_id)
        .await
        .map_err(AppError::from_context)?;
    Ok(Json(lance_context_api::ListDatagenEventsResponse {
        events,
    }))
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
    let store = store_lock.read().await;
    let resp = DatagenStoreApi::root_item_statuses(&*store, &ids)
        .await
        .map_err(AppError::from_context)?;
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
    let store = store_lock.read().await;
    let bytes = DatagenStoreApi::get_blob(&*store, &event_id)
        .await
        .map_err(AppError::from_context)?
        .ok_or_else(|| AppError::NotFound(format!("Datagen event '{}' has no blob", event_id)))?;

    Ok(([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response())
}
