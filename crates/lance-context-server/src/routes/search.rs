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
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let results = store
        .search(&req.query, Some(req.limit))
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
