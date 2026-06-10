use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use lance_context_api::{CompactRequest, CompactResponse, CompactStatsResponse};
use lance_context_core::CompactionConfig;

use crate::error::AppError;
use crate::state::AppState;

pub async fn compact(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<CompactRequest>,
) -> Result<Json<CompactResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let config = if req.target_rows_per_fragment.is_some() || req.materialize_deletions.is_some() {
        let mut c = CompactionConfig::default();
        if let Some(v) = req.target_rows_per_fragment {
            c.target_rows_per_fragment = v;
        }
        if let Some(v) = req.materialize_deletions {
            c.materialize_deletions = v;
        }
        Some(c)
    } else {
        None
    };

    let mut store = store_lock.write().await;
    let metrics = store.compact(config).await.map_err(AppError::from_lance)?;

    Ok(Json(CompactResponse {
        fragments_removed: metrics.fragments_removed,
        fragments_added: metrics.fragments_added,
        files_removed: metrics.files_removed,
        files_added: metrics.files_added,
    }))
}

pub async fn compact_stats(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<CompactStatsResponse>, AppError> {
    let stores = state.stores.read().await;
    let store_lock = stores
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("Context '{}' does not exist", name)))?
        .clone();
    drop(stores);

    let store = store_lock.read().await;
    let stats = store
        .compaction_stats()
        .await
        .map_err(AppError::from_lance)?;

    Ok(Json(CompactStatsResponse {
        total_fragments: stats.total_fragments,
        is_compacting: stats.is_compacting,
        last_compaction: stats.last_compaction,
        last_error: stats.last_error,
        total_compactions: stats.total_compactions,
    }))
}
