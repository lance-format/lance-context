pub mod compact;
pub mod contexts;
pub mod datagen;
pub mod generic;
pub mod health;
pub mod records;
pub mod rollouts;
pub mod search;
pub mod versions;

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::routing::{delete, get, patch, post, put};
use axum::Router;

use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/v1/health", get(health::health_check))
        .route("/api/v1/contexts", post(contexts::create_context))
        .route("/api/v1/contexts", get(contexts::list_contexts))
        .route("/api/v1/contexts/{name}", get(contexts::get_context))
        .route("/api/v1/contexts/{name}", delete(contexts::delete_context))
        .route(
            "/api/v1/contexts/{name}/records",
            post(records::add_records),
        )
        .route(
            "/api/v1/contexts/{name}/records",
            put(records::upsert_record),
        )
        .route(
            "/api/v1/contexts/{name}/records/batch",
            put(records::upsert_records),
        )
        .route(
            "/api/v1/contexts/{name}/records",
            patch(records::update_record),
        )
        .route(
            "/api/v1/contexts/{name}/records",
            get(records::list_records),
        )
        .route(
            "/api/v1/contexts/{name}/records",
            delete(records::delete_record_by_external_id),
        )
        .route(
            "/api/v1/contexts/{name}/records/by-external-id",
            get(records::get_record_by_external_id),
        )
        .route(
            "/api/v1/contexts/{name}/records/related",
            get(records::related_records),
        )
        .route(
            "/api/v1/contexts/{name}/records/{id}",
            get(records::get_record),
        )
        .route(
            "/api/v1/contexts/{name}/records/{id}",
            delete(records::delete_record),
        )
        .route(
            "/api/v1/contexts/{name}/records/{id}/payload",
            get(records::fetch_payload),
        )
        .route("/api/v1/contexts/{name}/search", post(search::search))
        .route("/api/v1/contexts/{name}/retrieve", post(search::retrieve))
        .route(
            "/api/v1/contexts/{name}/version",
            get(versions::get_version),
        )
        .route("/api/v1/contexts/{name}/checkout", post(versions::checkout))
        .route("/api/v1/contexts/{name}/compact", post(compact::compact))
        .route(
            "/api/v1/contexts/{name}/compact/stats",
            get(compact::compact_stats),
        )
        .route("/api/v1/rollouts", post(rollouts::create_rollout_store))
        .route("/api/v1/rollouts", get(rollouts::list_rollout_stores))
        .route("/api/v1/rollouts/{name}", get(rollouts::get_rollout_store))
        .route(
            "/api/v1/rollouts/{name}",
            delete(rollouts::delete_rollout_store),
        )
        .route(
            "/api/v1/rollouts/{name}/records",
            post(rollouts::add_rollouts)
                .layer(DefaultBodyLimit::max(rollouts::MAX_ROLLOUT_UPLOAD_BYTES)),
        )
        .route(
            "/api/v1/rollouts/{name}/records",
            get(rollouts::list_rollouts),
        )
        .route(
            "/api/v1/rollouts/{name}/records/{id}",
            get(rollouts::get_rollout),
        )
        .route(
            "/api/v1/rollouts/{name}/records/{id}/blob",
            get(rollouts::fetch_rollout_blob),
        )
        .route(
            "/api/v1/rollouts/{name}/checkout",
            post(rollouts::checkout_rollout),
        )
        .route(
            "/api/v1/rollouts/{name}/compact",
            post(rollouts::compact_rollout),
        )
        .route(
            "/api/v1/rollouts/{name}/compact/stats",
            get(rollouts::compact_rollout_stats),
        )
        .route(
            "/api/v1/internal/merge-wal/{name}",
            post(rollouts::merge_wal),
        )
        .route("/api/v1/datagen", post(datagen::create_datagen_store))
        .route("/api/v1/datagen", get(datagen::list_datagen_stores))
        .route("/api/v1/datagen/{name}", get(datagen::get_datagen_store))
        .route(
            "/api/v1/datagen/{name}",
            delete(datagen::delete_datagen_store),
        )
        .route(
            "/api/v1/datagen/{name}/events",
            post(datagen::add_datagen_events)
                .layer(DefaultBodyLimit::max(datagen::MAX_DATAGEN_UPLOAD_BYTES)),
        )
        .route(
            "/api/v1/datagen/{name}/items/{item_id}",
            get(datagen::fold_datagen_item),
        )
        .route(
            "/api/v1/datagen/{name}/items/{item_id}/failures",
            get(datagen::datagen_item_failures),
        )
        .route(
            "/api/v1/datagen/{name}/overview",
            get(datagen::datagen_overview),
        )
        .route(
            "/api/v1/datagen/{name}/root-status",
            get(datagen::datagen_root_item_statuses),
        )
        .route(
            "/api/v1/datagen/{name}/roots/{root_item_id}/events",
            get(datagen::datagen_events_for_root),
        )
        .route(
            "/api/v1/datagen/{name}/blobs/{event_id}",
            get(datagen::fetch_datagen_blob),
        )
        .route("/api/v1/generic", post(generic::create_generic_store))
        .route("/api/v1/generic", get(generic::list_generic_stores))
        .route("/api/v1/generic/{name}", get(generic::get_generic_store))
        .route(
            "/api/v1/generic/{name}",
            delete(generic::delete_generic_store),
        )
        .route(
            "/api/v1/generic/{name}/rows",
            post(generic::add_rows).layer(DefaultBodyLimit::max(generic::MAX_GENERIC_UPLOAD_BYTES)),
        )
        .route("/api/v1/generic/{name}/rows", get(generic::list_rows))
        .route("/api/v1/generic/{name}/rows/{id}", get(generic::get_row))
        .route(
            "/api/v1/generic/{name}/flush",
            post(generic::flush_generic_store),
        )
        .route(
            "/api/v1/generic/{name}/merge-wal",
            post(generic::merge_generic_wal),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_builds_with_record_parity_routes() {
        let _ = router();
    }
}
