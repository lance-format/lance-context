use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use lance_context_api::{
    RetrieveRequest, RetrieveResponse, RetrieveResultDto, SearchRequest, SearchResponse,
    SearchResultDto,
};
use lance_context_core::{LifecycleQueryOptions, RecordFilters};

use crate::error::AppError;
use crate::routes::records::record_to_dto;
use crate::state::AppState;

pub async fn search(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    let filters = req
        .filters
        .clone()
        .map(RecordFilters::from_json_value)
        .transpose()
        .map_err(AppError::InvalidRequest)?;

    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let results = store
        .search_filtered_with_options(
            &req.query,
            Some(req.limit),
            filters.as_ref(),
            LifecycleQueryOptions::new(req.include_expired, req.include_retired),
        )
        .await
        .map_err(AppError::from_lance)?;

    let dtos: Vec<SearchResultDto> = results
        .into_iter()
        .map(|mut sr| {
            if !req.include_relationships {
                sr.record.relationships.clear();
            }
            SearchResultDto {
                record: record_to_dto(sr.record),
                distance: sr.distance,
            }
        })
        .collect();

    Ok(Json(SearchResponse { results: dtos }))
}

pub async fn retrieve(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<RetrieveRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    if req.fusion != "rrf" {
        return Err(AppError::InvalidRequest(
            "retrieve fusion currently supports only 'rrf'".to_string(),
        ));
    }

    let filters = req
        .filters
        .clone()
        .map(RecordFilters::from_json_value)
        .transpose()
        .map_err(AppError::InvalidRequest)?;

    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let results = store
        .retrieve_filtered_with_options(
            req.text.as_deref(),
            req.vector.as_deref(),
            Some(req.limit),
            filters.as_ref(),
            LifecycleQueryOptions::new(req.include_expired, req.include_retired),
        )
        .await
        .map_err(AppError::from_lance)?;

    let dtos: Vec<RetrieveResultDto> = results
        .into_iter()
        .map(|mut result| {
            if !req.include_relationships {
                result.record.relationships.clear();
            }
            RetrieveResultDto {
                record: record_to_dto(result.record),
                score: result.score,
                vector_distance: result.vector_distance,
                text_score: result.text_score,
                matched_channels: result.matched_channels,
            }
        })
        .collect();

    Ok(Json(RetrieveResponse { results: dtos }))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::extract::{Path, State};
    use axum::Json;
    use chrono::{Duration, Utc};
    use lance_context_api::{
        AddRecordRequest, AddRecordsRequest, RecordPatchDto, RelationshipDto, UpdateRecordRequest,
    };
    use lance_context_core::{ContextStore, ContextStoreOptions};
    use tempfile::TempDir;
    use tokio::sync::RwLock;

    use super::*;
    use crate::routes::records::{add_records, update_record};
    use crate::state::AppState;

    const CTX: &str = "ctx";

    async fn test_state() -> (Arc<AppState>, TempDir) {
        let dir = TempDir::new().unwrap();
        let uri = dir
            .path()
            .join(format!("{CTX}.lance"))
            .to_string_lossy()
            .to_string();
        let store = ContextStore::open_with_options(
            &uri,
            ContextStoreOptions {
                embedding_dim: Some(3),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let mut stores = HashMap::new();
        stores.insert(CTX.to_string(), Arc::new(RwLock::new(store)));
        let state = Arc::new(AppState {
            stores: RwLock::new(stores),
            base_path: dir.path().to_path_buf(),
        });
        (state, dir)
    }

    fn embedded_record(text: &str, embedding: [f32; 3]) -> AddRecordRequest {
        AddRecordRequest {
            role: "user".to_string(),
            content_type: "text/plain".to_string(),
            text_payload: Some(text.to_string()),
            embedding: Some(embedding.to_vec()),
            ..Default::default()
        }
    }

    async fn add(state: &Arc<AppState>, records: Vec<AddRecordRequest>) -> Vec<String> {
        let (_, Json(resp)) = add_records(
            State(state.clone()),
            Path(CTX.to_string()),
            Json(AddRecordsRequest { records }),
        )
        .await
        .unwrap();
        resp.ids
    }

    async fn run_search(state: &Arc<AppState>, req: SearchRequest) -> Vec<SearchResultDto> {
        let Json(resp) = search(State(state.clone()), Path(CTX.to_string()), Json(req))
            .await
            .unwrap();
        resp.results
    }

    fn search_for(query: [f32; 3]) -> SearchRequest {
        SearchRequest {
            query: query.to_vec(),
            limit: 10,
            filters: None,
            include_expired: false,
            include_retired: false,
            include_relationships: false,
        }
    }

    #[tokio::test]
    async fn search_filters_by_metadata_and_builtin_fields() {
        let (state, _dir) = test_state().await;

        let mut alpha = embedded_record("alpha", [1.0, 0.0, 0.0]);
        alpha.metadata = Some(serde_json::json!({"tenant": "acme"}));
        let mut bravo = embedded_record("bravo", [0.0, 1.0, 0.0]);
        bravo.role = "assistant".to_string();
        bravo.metadata = Some(serde_json::json!({"tenant": "globex"}));
        let mut charlie = embedded_record("charlie", [0.0, 0.0, 1.0]);
        charlie.metadata = Some(serde_json::json!({"tenant": "acme"}));
        add(&state, vec![alpha, bravo, charlie]).await;

        // Metadata filter restricts to tenant=acme (alpha + charlie).
        let mut req = search_for([1.0, 0.0, 0.0]);
        req.filters = Some(serde_json::json!({"tenant": "acme"}));
        let results = run_search(&state, req).await;
        let texts: Vec<&str> = results
            .iter()
            .filter_map(|r| r.record.text_payload.as_deref())
            .collect();
        assert_eq!(results.len(), 2);
        assert!(texts.contains(&"alpha"));
        assert!(texts.contains(&"charlie"));

        // Built-in field filter restricts to role=assistant (bravo).
        let mut req = search_for([0.0, 1.0, 0.0]);
        req.filters = Some(serde_json::json!({"role": "assistant"}));
        let results = run_search(&state, req).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.text_payload.as_deref(), Some("bravo"));
    }

    #[tokio::test]
    async fn search_respects_expired_visibility() {
        let (state, _dir) = test_state().await;

        let fresh = embedded_record("fresh", [1.0, 0.0, 0.0]);
        let mut stale = embedded_record("stale", [1.0, 0.0, 0.0]);
        stale.expires_at = Some(Utc::now() - Duration::hours(1));
        add(&state, vec![fresh, stale]).await;

        // Default search hides the expired record.
        let results = run_search(&state, search_for([1.0, 0.0, 0.0])).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.text_payload.as_deref(), Some("fresh"));

        // include_expired surfaces it.
        let mut req = search_for([1.0, 0.0, 0.0]);
        req.include_expired = true;
        let results = run_search(&state, req).await;
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn search_respects_retired_visibility() {
        let (state, _dir) = test_state().await;

        let mut original = embedded_record("v1", [1.0, 0.0, 0.0]);
        original.external_id = Some("doc-1".to_string());
        let ids = add(&state, vec![original]).await;
        let old_id = ids[0].clone();

        // Updating supersedes the original; the successor keeps the embedding.
        let Json(updated) = update_record(
            State(state.clone()),
            Path(CTX.to_string()),
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

        // Default search returns only the visible successor.
        let results = run_search(&state, search_for([1.0, 0.0, 0.0])).await;
        assert_eq!(results.len(), 1);
        assert_ne!(results[0].record.id, old_id);

        // include_retired surfaces the superseded original too.
        let mut req = search_for([1.0, 0.0, 0.0]);
        req.include_retired = true;
        let results = run_search(&state, req).await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.record.id == old_id));
    }

    #[tokio::test]
    async fn search_include_relationships_toggles_relationship_payload() {
        let (state, _dir) = test_state().await;

        let mut record = embedded_record("cites runbook", [1.0, 0.0, 0.0]);
        record.relationships = vec![RelationshipDto {
            target_id: "doc://runbook".to_string(),
            relation: "cites".to_string(),
            weight: Some(0.5),
        }];
        add(&state, vec![record]).await;

        // Default omits relationships.
        let results = run_search(&state, search_for([1.0, 0.0, 0.0])).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].record.relationships.is_empty());

        // include_relationships returns them.
        let mut req = search_for([1.0, 0.0, 0.0]);
        req.include_relationships = true;
        let results = run_search(&state, req).await;
        assert_eq!(results[0].record.relationships.len(), 1);
    }
}
