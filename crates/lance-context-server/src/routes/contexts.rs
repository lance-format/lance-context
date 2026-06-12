use std::collections::HashSet;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use lance_context_api::{ContextInfo, CreateContextRequest, ListContextsResponse};
use lance_context_core::{ContextStore, ContextStoreOptions, DistanceMetric, IdIndexType};
use tokio::sync::RwLock;

use crate::error::AppError;
use crate::state::AppState;

pub async fn create_context(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateContextRequest>,
) -> Result<(axum::http::StatusCode, Json<ContextInfo>), AppError> {
    let stores = state.stores.read().await;
    if stores.contains_key(&req.name) {
        return Err(AppError::AlreadyExists(format!(
            "Context '{}' already exists",
            req.name
        )));
    }
    drop(stores);

    let id_index_type = match req.id_index_type.as_deref() {
        Some("btree") => IdIndexType::BTree,
        Some("zonemap") => IdIndexType::ZoneMap,
        Some("none") | None => IdIndexType::None,
        Some(other) => {
            return Err(AppError::InvalidRequest(format!(
                "Invalid id_index_type: '{}'. Must be 'none', 'zonemap', or 'btree'",
                other
            )));
        }
    };

    let blob_columns: HashSet<String> = req.blob_columns.unwrap_or_default().into_iter().collect();

    let distance_metric = match req.distance_metric.as_deref() {
        Some(value) => Some(
            DistanceMetric::parse(value).map_err(|e| AppError::InvalidRequest(e.to_string()))?,
        ),
        None => None,
    };

    let uri = state.context_uri(&req.name);
    let options = ContextStoreOptions {
        storage_options: req.storage_options,
        embedding_dim: req.embedding_dim,
        blob_columns,
        id_index_type,
        distance_metric,
        ..Default::default()
    };

    let store = ContextStore::open_with_options(&uri, options)
        .await
        .map_err(AppError::from_lance)?;

    let version = store.version();

    let mut stores = state.stores.write().await;
    stores.insert(req.name.clone(), Arc::new(RwLock::new(store)));

    Ok((
        axum::http::StatusCode::CREATED,
        Json(ContextInfo {
            name: req.name,
            uri,
            version,
        }),
    ))
}

pub async fn list_contexts(State(state): State<Arc<AppState>>) -> Json<ListContextsResponse> {
    let stores = state.stores.read().await;
    let mut contexts = Vec::with_capacity(stores.len());

    for (name, store_lock) in stores.iter() {
        let store = store_lock.read().await;
        contexts.push(ContextInfo {
            name: name.clone(),
            uri: state.context_uri(name),
            version: store.version(),
        });
    }

    Json(ListContextsResponse { contexts })
}

pub async fn get_context(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<ContextInfo>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?;
    let store = store_lock.read().await;

    Ok(Json(ContextInfo {
        name: name.clone(),
        uri: state.context_uri(&name),
        version: store.version(),
    }))
}

pub async fn delete_context(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<axum::http::StatusCode, AppError> {
    let mut stores = state.stores.write().await;
    if stores.remove(&name).is_none() {
        return Err(AppError::NotFound(format!(
            "Context '{}' does not exist",
            name
        )));
    }

    let uri = state.context_uri(&name);
    if let Err(e) = tokio::fs::remove_dir_all(&uri).await {
        tracing::warn!("Failed to remove context data at {}: {}", uri, e);
    }

    Ok(axum::http::StatusCode::NO_CONTENT)
}
