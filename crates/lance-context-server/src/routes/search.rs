use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use lance_context_api::{SearchRequest, SearchResponse, SearchResultDto};

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
        .map(|sr| SearchResultDto {
            record: record_to_dto(sr.record),
            distance: sr.distance,
        })
        .collect();

    Ok(Json(SearchResponse { results: dtos }))
}
