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
        ..state.context_store_options()
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
    use crate::routes::records::add_records;
    use lance_context_api::{AddRecordRequest, AddRecordsRequest};
    use std::time::Duration;
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

    #[tokio::test]
    async fn server_instances_write_contexts_to_distinct_wal_shards() {
        let dir = TempDir::new().unwrap();
        let context_name = "shared";
        let state_a = Arc::new(
            AppState::new_for_test_with_instance(
                dir.path().to_path_buf(),
                Some("context-0".to_string()),
            )
            .await,
        );

        let (_, Json(_)) =
            create_context(State(state_a.clone()), Json(create_request(context_name)))
                .await
                .unwrap();
        add_text_record(state_a, context_name, "from context-0").await;

        // A second server lazily opens the same dataset. Its instance id must
        // select a different shard instead of fencing the first server's writer.
        let state_b = Arc::new(
            AppState::new_for_test_with_instance(
                dir.path().to_path_buf(),
                Some("context-1".to_string()),
            )
            .await,
        );
        add_text_record(state_b.clone(), context_name, "from context-1").await;

        let store = state_b
            .get_or_open_context_store(context_name)
            .await
            .unwrap();
        assert_eq!(store.read().await.list(None, None).await.unwrap().len(), 2);

        let mem_wal = dir
            .path()
            .join(format!("{context_name}.lance"))
            .join("_mem_wal");
        let shard_count = std::fs::read_dir(mem_wal)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .count();
        assert_eq!(shard_count, 2, "each server instance must own one shard");
    }

    async fn add_text_record(state: Arc<AppState>, context_name: &str, text: &str) {
        let request = AddRecordsRequest {
            records: vec![AddRecordRequest {
                role: "user".to_string(),
                content_type: "text/plain".to_string(),
                text_payload: Some(text.to_string()),
                ..Default::default()
            }],
        };
        let (_, Json(_)) = tokio::time::timeout(
            Duration::from_secs(30),
            add_records(State(state), Path(context_name.to_string()), Json(request)),
        )
        .await
        .expect("context write must not hang on shard contention")
        .unwrap();
    }
}
