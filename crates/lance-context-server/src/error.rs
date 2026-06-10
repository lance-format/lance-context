use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use lance_context_api::{ErrorBody, ErrorResponse};

#[derive(Debug)]
pub enum AppError {
    NotFound(String),
    AlreadyExists(String),
    InvalidRequest(String),
    Internal(String),
    CompactionInProgress,
}

impl AppError {
    pub fn from_lance(err: impl std::fmt::Display) -> Self {
        let msg = err.to_string();
        if msg.contains("already in progress") {
            AppError::CompactionInProgress
        } else if msg.contains("not found") || msg.contains("DatasetNotFound") {
            AppError::NotFound(msg)
        } else if msg.contains("Invalid") {
            AppError::InvalidRequest(msg)
        } else {
            AppError::Internal(msg)
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, "NOT_FOUND", msg),
            AppError::AlreadyExists(msg) => (StatusCode::CONFLICT, "ALREADY_EXISTS", msg),
            AppError::InvalidRequest(msg) => (StatusCode::BAD_REQUEST, "INVALID_REQUEST", msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL", msg),
            AppError::CompactionInProgress => (
                StatusCode::CONFLICT,
                "COMPACTION_IN_PROGRESS",
                "Compaction already in progress".to_string(),
            ),
        };

        let body = ErrorResponse {
            error: ErrorBody {
                code: code.to_string(),
                message,
            },
        };

        (status, Json(body)).into_response()
    }
}
