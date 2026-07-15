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
    AppState::validate_name(&req.name)?;
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
    let store_lock = state.get_or_open_context_store(&name).await?;
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
    AppState::validate_name(&name)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_request(name: &str) -> CreateContextRequest {
        CreateContextRequest {
            name: name.to_string(),
            storage_options: None,
            id_index_type: None,
            blob_columns: None,
            embedding_dim: None,
            distance_metric: None,
        }
    }

    #[tokio::test]
    async fn create_rejects_names_that_are_not_portable_path_segments() {
        let dir = TempDir::new().unwrap();
        let state = Arc::new(AppState::new_for_test(dir.path().to_path_buf()).await);

        for name in ["../escape", "nested/store", r"nested\store", "_registry"] {
            let err = create_context(State(state.clone()), Json(create_request(name)))
                .await
                .unwrap_err();
            assert!(matches!(err, AppError::InvalidRequest(_)), "{name}");
        }
        assert!(state.stores.read().await.is_empty());
    }

    #[tokio::test]
    async fn open_rejects_invalid_context_names_before_storage_access() {
        let dir = TempDir::new().unwrap();
        let state = AppState::new_for_test(dir.path().to_path_buf()).await;

        let err = match state.get_or_open_context_store("../escape").await {
            Ok(_) => panic!("invalid name unexpectedly opened"),
            Err(err) => err,
        };
        assert!(matches!(err, AppError::InvalidRequest(_)));
    }
}
