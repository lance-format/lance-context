//! HTTP surface for stores over user-declared schemas.
//!
//! Unlike the fixed-schema routes, rows are opaque JSON objects in both
//! directions: the schema is declared at store creation, so there is no static
//! DTO to convert through. Validation happens in the core codec against the
//! store's own schema, which is where the column list actually lives.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use lance_context_api::{
    AddRowsRequest, AddRowsResponse, CreateGenericStoreRequest, GenericStoreInfo,
    ListGenericStoresResponse, ListRowsResponse,
};
use lance_context_core::{GenericStore, GenericStoreOptions};
use serde::Deserialize;
use tokio::sync::RwLock;

use crate::error::AppError;
use crate::state::AppState;

/// Upper bound on one generic append request body.
///
/// Generic stores hold blob columns inline — that is the point of them — so
/// this matches the rollout ceiling rather than the smaller datagen one.
pub const MAX_GENERIC_UPLOAD_BYTES: usize = 1024 * 1024 * 1024;

/// `POST /api/v1/generic`
pub async fn create_generic_store(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateGenericStoreRequest>,
) -> Result<(StatusCode, Json<GenericStoreInfo>), AppError> {
    AppState::validate_name(&req.name)?;
    // Reject an invalid schema before creating anything, so a bad declaration
    // cannot leave an empty dataset behind.
    req.schema.validate().map_err(AppError::InvalidRequest)?;

    if state
        .generic_registry
        .write()
        .await
        .contains(&req.name)
        .await
        .map_err(AppError::from_lance)?
    {
        return Err(AppError::AlreadyExists(format!(
            "Generic store '{}' already exists",
            req.name
        )));
    }

    let uri = state.generic_uri(&req.name);
    let options = GenericStoreOptions {
        storage_options: req.storage_options,
        shard_id: state.instance_id.clone(),
        merge_after_generations: None,
        merge_max_generations: Some(state.rollout_merge_max_generations),
        session: None,
        seal_on_add: req.seal_on_add,
    };
    let store = GenericStore::open(&uri, req.schema.clone(), options)
        .await
        .map_err(AppError::from_lance)?;
    let version = store.version();

    let store = Arc::new(RwLock::new(store));
    state.register_generic(&req.name, &uri, store).await?;

    Ok((
        StatusCode::CREATED,
        Json(GenericStoreInfo {
            name: req.name,
            uri,
            version: Some(version),
            schema: Some(req.schema),
        }),
    ))
}

/// `GET /api/v1/generic`
///
/// Served from the registry without opening each dataset, so `version` and
/// `schema` are omitted — both live in the dataset itself.
pub async fn list_generic_stores(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ListGenericStoresResponse>, AppError> {
    let entries = state
        .generic_registry
        .write()
        .await
        .list()
        .await
        .map_err(AppError::from_lance)?;

    Ok(Json(ListGenericStoresResponse {
        stores: entries
            .into_iter()
            .map(|entry| GenericStoreInfo {
                name: entry.name,
                uri: entry.uri,
                version: None,
                schema: None,
            })
            .collect(),
    }))
}

/// `GET /api/v1/generic/{name}`
pub async fn get_generic_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<GenericStoreInfo>, AppError> {
    let store = state.get_or_open_generic_store(&name).await?;
    let guard = store.read().await;
    Ok(Json(GenericStoreInfo {
        name: name.clone(),
        uri: state.generic_uri(&name),
        version: Some(guard.version()),
        schema: Some(guard.spec().clone()),
    }))
}

/// `DELETE /api/v1/generic/{name}`
pub async fn delete_generic_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    if state.unregister_generic(&name).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "Generic store '{name}' does not exist"
        )))
    }
}

/// `POST /api/v1/generic/{name}/rows`
///
/// Rows are validated against the store's schema: an undeclared column is an
/// error rather than being dropped, and an omitted nullable column is null.
pub async fn add_rows(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddRowsRequest>,
) -> Result<(StatusCode, Json<AddRowsResponse>), AppError> {
    let store = state.get_or_open_generic_store(&name).await?;
    // Read lock: `add` takes `&self`, so appends do not serialize here.
    let guard = store.read().await;
    let count = req.rows.len();
    let version = guard.add(&req.rows).await.map_err(AppError::from_lance)?;
    Ok((
        StatusCode::CREATED,
        Json(AddRowsResponse { version, count }),
    ))
}

/// Query parameters for listing rows.
#[derive(Debug, Deserialize)]
pub struct ListRowsQuery {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
    /// SQL predicate over the store's own columns.
    #[serde(default)]
    pub filter: Option<String>,
}

/// `GET /api/v1/generic/{name}/rows`
///
/// Blob columns are projected out, so a list never materializes a large
/// payload; fetch those per row via [`get_row`].
pub async fn list_rows(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(query): Query<ListRowsQuery>,
) -> Result<Json<ListRowsResponse>, AppError> {
    let store = state.get_or_open_generic_store(&name).await?;
    let guard = store.read().await;

    let rows = match &query.filter {
        Some(filter) => guard
            .list_filtered(filter, query.limit, query.offset)
            .await
            .map_err(AppError::from_lance)?,
        None => guard
            .list(query.limit, query.offset)
            .await
            .map_err(AppError::from_lance)?,
    };
    Ok(Json(ListRowsResponse { rows }))
}

/// Query parameters for a point read.
#[derive(Debug, Deserialize)]
pub struct GetRowQuery {
    /// Comma-separated columns to read. Absent reads everything except blob
    /// columns; name a blob column explicitly to fetch it.
    #[serde(default)]
    pub columns: Option<String>,
}

async fn get_generic_row_refreshing_on_miss(
    store_lock: &RwLock<GenericStore>,
    id: &str,
    columns: Option<&[String]>,
) -> Result<Option<serde_json::Map<String, serde_json::Value>>, AppError> {
    {
        let store = store_lock.read().await;
        let row = store.get(id, columns).await.map_err(AppError::from_lance)?;
        if row.is_some() || store.is_version_pinned() {
            return Ok(row);
        }
    }

    let store = store_lock.read().await;
    if !store.is_version_pinned() {
        store.refresh_latest().await.map_err(AppError::from_lance)?;
    }
    store.get(id, columns).await.map_err(AppError::from_lance)
}

/// `GET /api/v1/generic/{name}/rows/{id}`
pub async fn get_row(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
    Query(query): Query<GetRowQuery>,
) -> Result<Json<serde_json::Map<String, serde_json::Value>>, AppError> {
    let store = state.get_or_open_generic_store(&name).await?;

    let columns: Option<Vec<String>> = query.columns.as_ref().map(|raw| {
        raw.split(',')
            .map(str::trim)
            .filter(|column| !column.is_empty())
            .map(str::to_string)
            .collect()
    });

    let row = get_generic_row_refreshing_on_miss(&store, &id, columns.as_deref())
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Row '{id}' does not exist")))?;
    Ok(Json(row))
}

/// `POST /api/v1/generic/{name}/flush`
///
/// Seals the active memtable so previously added rows become readable. Needed
/// only when the store was created with `seal_on_add: false`.
pub async fn flush_generic_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    let store = state.get_or_open_generic_store(&name).await?;
    let guard = store.read().await;
    guard.flush().await.map_err(AppError::from_lance)?;
    Ok(StatusCode::NO_CONTENT)
}

/// `POST /api/v1/generic/{name}/merge-wal`
///
/// Folds this instance's flushed MemWAL generations into the base table.
/// Returns how many were reclaimed.
pub async fn merge_generic_wal(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let store = state.get_or_open_generic_store(&name).await?;
    let mut guard = store.write().await;
    let reclaimed = guard.cleanup_wal().await.map_err(AppError::from_lance)?;
    Ok(Json(serde_json::json!({ "reclaimed": reclaimed })))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lance_context_api::{ColumnSpec, ColumnType, SchemaSpec, ID_COLUMN};
    use tempfile::TempDir;

    use super::*;
    use crate::state::AppState;

    fn spec() -> SchemaSpec {
        SchemaSpec::new(vec![
            (
                ID_COLUMN.to_string(),
                ColumnSpec::required(ColumnType::String { large: false }),
            ),
            (
                "user".to_string(),
                ColumnSpec::new(ColumnType::String { large: false }),
            ),
            (
                "blob".to_string(),
                ColumnSpec::new(ColumnType::Binary { blob: true }),
            ),
        ])
    }

    async fn test_state() -> (Arc<AppState>, TempDir) {
        let dir = TempDir::new().unwrap();
        let state = Arc::new(AppState::new_for_test(dir.path().to_path_buf()).await);
        (state, dir)
    }

    fn row(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        value.as_object().unwrap().clone()
    }

    #[tokio::test]
    async fn point_read_refreshes_a_base_advanced_by_an_external_writer() {
        let (state, _dir) = test_state().await;
        let _ = create_generic_store(
            State(state.clone()),
            Json(CreateGenericStoreRequest {
                name: "s1".to_string(),
                schema: spec(),
                storage_options: None,
                seal_on_add: true,
            }),
        )
        .await
        .unwrap();
        let cached = state.get_or_open_generic_store("s1").await.unwrap();
        let mut writer = GenericStore::open_existing(
            &state.generic_uri("s1"),
            GenericStoreOptions {
                shard_id: Some("external-writer".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        writer
            .add(&[row(
                serde_json::json!({"id": "merged-row", "user": "external"}),
            )])
            .await
            .unwrap();
        assert_eq!(writer.cleanup_wal().await.unwrap(), 1);
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
        assert!(writer.version() > cached.read().await.version());

        let Json(found) = get_row(
            State(state),
            Path(("s1".to_string(), "merged-row".to_string())),
            Query(GetRowQuery { columns: None }),
        )
        .await
        .unwrap();
        assert_eq!(found["user"], serde_json::json!("external"));
        assert_eq!(cached.read().await.version(), writer.version());
    }

    #[tokio::test]
    async fn create_list_add_and_read_round_trip() {
        let (state, _dir) = test_state().await;

        let (status, Json(info)) = create_generic_store(
            State(state.clone()),
            Json(CreateGenericStoreRequest {
                name: "s1".to_string(),
                schema: spec(),
                storage_options: None,
                seal_on_add: true,
            }),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert!(info.schema.is_some(), "create echoes the schema back");

        let _ = add_rows(
            State(state.clone()),
            Path("s1".to_string()),
            Json(AddRowsRequest {
                rows: vec![
                    row(serde_json::json!({"id": "r1", "user": "u1", "blob": [1, 2, 3]})),
                    row(serde_json::json!({"id": "r2", "user": "u2"})),
                ],
            }),
        )
        .await
        .unwrap();

        let Json(listed) = list_rows(
            State(state.clone()),
            Path("s1".to_string()),
            Query(ListRowsQuery {
                limit: None,
                offset: None,
                filter: None,
            }),
        )
        .await
        .unwrap();
        assert_eq!(listed.rows.len(), 2);
        assert!(
            !listed.rows[0].contains_key("blob"),
            "listing must not materialize blob columns"
        );

        // A blob is fetchable per row by naming the column.
        let Json(fetched) = get_row(
            State(state.clone()),
            Path(("s1".to_string(), "r1".to_string())),
            Query(GetRowQuery {
                columns: Some("blob".to_string()),
            }),
        )
        .await
        .unwrap();
        assert_eq!(fetched["blob"], serde_json::json!("AQID"));
    }

    #[tokio::test]
    async fn invalid_schema_is_rejected_before_the_store_is_created() {
        let (state, _dir) = test_state().await;

        // No `id` column.
        let bad = SchemaSpec::new(vec![(
            "name".to_string(),
            ColumnSpec::new(ColumnType::String { large: false }),
        )]);
        let err = create_generic_store(
            State(state.clone()),
            Json(CreateGenericStoreRequest {
                name: "bad".to_string(),
                schema: bad,
                storage_options: None,
                seal_on_add: true,
            }),
        )
        .await
        .expect_err("an invalid schema must be rejected");
        assert!(matches!(err, AppError::InvalidRequest(_)), "{err:?}");

        // And nothing was registered.
        let Json(listed) = list_generic_stores(State(state)).await.unwrap();
        assert!(listed.stores.is_empty());
    }

    #[tokio::test]
    async fn duplicate_create_conflicts_and_missing_store_is_not_found() {
        let (state, _dir) = test_state().await;
        let request = || CreateGenericStoreRequest {
            name: "s1".to_string(),
            schema: spec(),
            storage_options: None,
            seal_on_add: true,
        };

        let _ = create_generic_store(State(state.clone()), Json(request()))
            .await
            .unwrap();
        let err = create_generic_store(State(state.clone()), Json(request()))
            .await
            .expect_err("a duplicate name must conflict");
        assert!(matches!(err, AppError::AlreadyExists(_)), "{err:?}");

        let err = get_generic_store(State(state.clone()), Path("absent".to_string()))
            .await
            .expect_err("an unknown store must 404");
        assert!(matches!(err, AppError::NotFound(_)), "{err:?}");
    }

    #[tokio::test]
    async fn undeclared_columns_are_rejected_rather_than_dropped() {
        // A field-name typo must fail the write, not become missing data.
        let (state, _dir) = test_state().await;
        let _ = create_generic_store(
            State(state.clone()),
            Json(CreateGenericStoreRequest {
                name: "s1".to_string(),
                schema: spec(),
                storage_options: None,
                seal_on_add: true,
            }),
        )
        .await
        .unwrap();

        let err = add_rows(
            State(state.clone()),
            Path("s1".to_string()),
            Json(AddRowsRequest {
                rows: vec![row(serde_json::json!({"id": "r1", "usr": "typo"}))],
            }),
        )
        .await
        .expect_err("an undeclared column must be rejected");
        assert!(
            format!("{err:?}").contains("not declared"),
            "expected an undeclared-column error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn delete_removes_the_store_from_the_registry() {
        let (state, _dir) = test_state().await;
        let _ = create_generic_store(
            State(state.clone()),
            Json(CreateGenericStoreRequest {
                name: "s1".to_string(),
                schema: spec(),
                storage_options: None,
                seal_on_add: true,
            }),
        )
        .await
        .unwrap();

        assert_eq!(
            delete_generic_store(State(state.clone()), Path("s1".to_string()))
                .await
                .unwrap(),
            StatusCode::NO_CONTENT
        );
        let Json(listed) = list_generic_stores(State(state.clone())).await.unwrap();
        assert!(listed.stores.is_empty());

        let err = delete_generic_store(State(state), Path("s1".to_string()))
            .await
            .expect_err("deleting twice must 404");
        assert!(matches!(err, AppError::NotFound(_)), "{err:?}");
    }

    /// The bug this guards: `seal_on_add` used to live only in the open
    /// options, so a store created with it lost read-your-write the first time
    /// it was evicted from the LRU and reopened — silently, with no error and
    /// no data loss, just "sometimes I can't read what I just wrote".
    #[tokio::test]
    async fn seal_mode_survives_eviction_and_reopen() {
        let (state, _dir) = test_state().await;
        let _ = create_generic_store(
            State(state.clone()),
            Json(CreateGenericStoreRequest {
                name: "s1".to_string(),
                schema: spec(),
                storage_options: None,
                seal_on_add: true,
            }),
        )
        .await
        .unwrap();

        // Evict, forcing the next request to reopen from object storage. This
        // is exactly what the LRU does under capacity pressure.
        state.generic_stores.lock().await.pop("s1");

        let _ = add_rows(
            State(state.clone()),
            Path("s1".to_string()),
            Json(AddRowsRequest {
                rows: vec![row(serde_json::json!({"id": "r1"}))],
            }),
        )
        .await
        .unwrap();

        let Json(listed) = list_rows(
            State(state),
            Path("s1".to_string()),
            Query(ListRowsQuery {
                limit: None,
                offset: None,
                filter: None,
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            listed.rows.len(),
            1,
            "a reopened store must keep the seal mode it was created with"
        );
    }

    /// The mirror case: a store created with the deferred default must not
    /// start sealing after a reopen either.
    #[tokio::test]
    async fn deferred_seal_mode_also_survives_eviction() {
        let (state, _dir) = test_state().await;
        let _ = create_generic_store(
            State(state.clone()),
            Json(CreateGenericStoreRequest {
                name: "s1".to_string(),
                schema: spec(),
                storage_options: None,
                seal_on_add: false,
            }),
        )
        .await
        .unwrap();

        state.generic_stores.lock().await.pop("s1");

        let _ = add_rows(
            State(state.clone()),
            Path("s1".to_string()),
            Json(AddRowsRequest {
                rows: vec![row(serde_json::json!({"id": "r1"}))],
            }),
        )
        .await
        .unwrap();

        let query = || ListRowsQuery {
            limit: None,
            offset: None,
            filter: None,
        };
        let Json(listed) = list_rows(State(state.clone()), Path("s1".to_string()), Query(query()))
            .await
            .unwrap();
        assert_eq!(listed.rows.len(), 0, "deferred seal must stay deferred");

        flush_generic_store(State(state.clone()), Path("s1".to_string()))
            .await
            .unwrap();
        let Json(after) = list_rows(State(state), Path("s1".to_string()), Query(query()))
            .await
            .unwrap();
        assert_eq!(after.rows.len(), 1);
    }

    #[tokio::test]
    async fn deferred_seal_needs_a_flush_and_wal_merges() {
        let (state, _dir) = test_state().await;
        let _ = create_generic_store(
            State(state.clone()),
            Json(CreateGenericStoreRequest {
                name: "s1".to_string(),
                schema: spec(),
                storage_options: None,
                // Default profile: durable on return, not yet visible.
                seal_on_add: false,
            }),
        )
        .await
        .unwrap();

        let _ = add_rows(
            State(state.clone()),
            Path("s1".to_string()),
            Json(AddRowsRequest {
                rows: vec![row(serde_json::json!({"id": "r1"}))],
            }),
        )
        .await
        .unwrap();

        let query = || ListRowsQuery {
            limit: None,
            offset: None,
            filter: None,
        };
        let Json(before) = list_rows(State(state.clone()), Path("s1".to_string()), Query(query()))
            .await
            .unwrap();
        assert_eq!(before.rows.len(), 0, "deferred seal must not publish yet");

        flush_generic_store(State(state.clone()), Path("s1".to_string()))
            .await
            .unwrap();
        let Json(after) = list_rows(State(state.clone()), Path("s1".to_string()), Query(query()))
            .await
            .unwrap();
        assert_eq!(after.rows.len(), 1);

        // And the WAL generation folds into the base table.
        let Json(merged) = merge_generic_wal(State(state.clone()), Path("s1".to_string()))
            .await
            .unwrap();
        assert!(merged["reclaimed"].as_u64().unwrap() > 0);
        let Json(kept) = list_rows(State(state), Path("s1".to_string()), Query(query()))
            .await
            .unwrap();
        assert_eq!(kept.rows.len(), 1, "rows survive the merge");
    }
}
