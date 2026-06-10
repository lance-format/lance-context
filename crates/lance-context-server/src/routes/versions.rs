use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use lance_context_api::{CheckoutRequest, VersionResponse};

use crate::error::AppError;
use crate::state::AppState;

pub async fn get_version(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<VersionResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    Ok(Json(VersionResponse {
        version: store.version(),
    }))
}

pub async fn checkout(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<CheckoutRequest>,
) -> Result<Json<VersionResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
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
