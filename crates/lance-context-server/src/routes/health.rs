use axum::Json;

#[derive(serde::Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}
