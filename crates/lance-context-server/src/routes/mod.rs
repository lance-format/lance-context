pub mod compact;
pub mod contexts;
pub mod health;
pub mod records;
pub mod search;
pub mod versions;

use std::sync::Arc;

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_builds_with_record_parity_routes() {
        let _ = router();
    }
}
