//! Admin HTTP error type.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// Errors surfaced by the master admin API.
#[derive(Debug)]
pub enum MasterError {
    NotFound(String),
    Internal(String),
}

impl MasterError {
    /// Map a Lance error to an internal server error.
    pub fn from_lance(err: lance::Error) -> Self {
        MasterError::Internal(err.to_string())
    }
}

impl IntoResponse for MasterError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            MasterError::NotFound(m) => (StatusCode::NOT_FOUND, m),
            MasterError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        };
        (status, Json(json!({ "error": msg }))).into_response()
    }
}
