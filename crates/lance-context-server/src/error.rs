use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use lance_context_api::{ErrorBody, ErrorResponse};
use lance_context_core::LanceError;

#[derive(Debug)]
pub enum AppError {
    NotFound(String),
    AlreadyExists(String),
    InvalidRequest(String),
    Internal(String),
    CompactionInProgress,
    /// The in-flight blob-byte budget is exhausted; the client should retry
    /// later. Maps to `503 Service Unavailable`.
    Overloaded(String),
}

impl AppError {
    /// Map a Lance error onto the API's error taxonomy.
    ///
    /// Prefers Lance's **typed** variants over string matching: `DatasetNotFound`
    /// / `NotFound` → 404, `DatasetAlreadyExists` → 409, `InvalidInput` /
    /// `SchemaMismatch` → 400. The one unavoidable string match is the
    /// "compaction already in progress" signal, which the core raises as an
    /// `ArrowError::InvalidArgumentError` that Lance folds into its generic
    /// `Arrow` variant (no dedicated variant, see `store.rs`); it is checked
    /// first so it still maps to 409 `COMPACTION_IN_PROGRESS`.
    pub fn from_lance(err: LanceError) -> Self {
        // Checked before the typed match: the compaction-in-progress signal is an
        // Arrow-wrapped error with no dedicated variant, so only its text
        // distinguishes it.
        if err.to_string().contains("already in progress") {
            return AppError::CompactionInProgress;
        }
        match err {
            LanceError::DatasetNotFound { .. } | LanceError::NotFound { .. } => {
                AppError::NotFound(err.to_string())
            }
            LanceError::DatasetAlreadyExists { .. } => AppError::AlreadyExists(err.to_string()),
            LanceError::InvalidInput { .. } | LanceError::SchemaMismatch { .. } => {
                AppError::InvalidRequest(err.to_string())
            }
            other => AppError::Internal(other.to_string()),
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
            AppError::Overloaded(msg) => (StatusCode::SERVICE_UNAVAILABLE, "OVERLOADED", msg),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_typed_variants_to_taxonomy() {
        assert!(matches!(
            AppError::from_lance(LanceError::dataset_not_found("db/x", "missing".into())),
            AppError::NotFound(_)
        ));
        assert!(matches!(
            AppError::from_lance(LanceError::dataset_already_exists("db/x")),
            AppError::AlreadyExists(_)
        ));
        assert!(matches!(
            AppError::from_lance(LanceError::invalid_input("bad field")),
            AppError::InvalidRequest(_)
        ));
        assert!(matches!(
            AppError::from_lance(LanceError::io("disk gone".to_string())),
            AppError::Internal(_)
        ));
    }

    #[test]
    fn compaction_in_progress_is_detected_from_arrow_variant() {
        // Reproduce how the core raises it: an ArrowError folded into Lance's
        // generic `Arrow` variant (NOT `InvalidInput`), so only the text
        // identifies it. `LanceError::arrow` is exactly what `From<ArrowError>`
        // produces for a non-external ArrowError.
        let err = LanceError::arrow("Compaction already in progress");
        assert!(matches!(
            AppError::from_lance(err),
            AppError::CompactionInProgress
        ));
    }
}
