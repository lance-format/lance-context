use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use lance_context_api::{ContextError, ErrorBody, ErrorResponse};
use lance_context_core::LanceError;
use lance_context_metrics::{ReadFailureDiagnostic, ReadFailureKind, ReadFailureStage};

/// Adds a closed diagnostic to a failed read's response extensions, preserving
/// the original wire response. Raw error messages never enter recovery signals.
#[derive(Debug)]
pub struct ReadFailureError {
    error: AppError,
    stage: ReadFailureStage,
    kind: ReadFailureKind,
}

impl ReadFailureError {
    pub fn new(error: AppError, stage: ReadFailureStage) -> Self {
        let kind = if matches!(error, AppError::Overloaded(_)) {
            ReadFailureKind::Overloaded
        } else {
            ReadFailureKind::Internal
        };
        Self { error, stage, kind }
    }

    pub fn from_lance(error: LanceError, stage: ReadFailureStage) -> Self {
        let kind = match error.find_root() {
            LanceError::IO { .. } => ReadFailureKind::LanceIo,
            LanceError::Timeout { .. } => ReadFailureKind::LanceTimeout,
            LanceError::Arrow { .. } => ReadFailureKind::LanceArrow,
            _ => ReadFailureKind::LanceOther,
        };
        let mut result = Self::new(AppError::from_lance(error), stage);
        if !matches!(result.error, AppError::Overloaded(_)) {
            result.kind = kind;
        }
        result
    }
}

impl IntoResponse for ReadFailureError {
    fn into_response(self) -> Response {
        let mut response = self.error.into_response();
        if response.status().is_server_error() {
            response
                .extensions_mut()
                .insert(ReadFailureDiagnostic::new(self.stage, self.kind));
        }
        response
    }
}

#[derive(Debug)]
pub enum AppError {
    NotFound(String),
    AlreadyExists(String),
    InvalidRequest(String),
    Internal(String),
    CompactionInProgress,
    /// An opted-in read stopped after its outer HTTP future was dropped.
    ClientDisconnected,
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
        if lance_context_core::is_pending_generations_exceeded(&err) {
            return AppError::Overloaded(err.to_string());
        }
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

    /// Map an API-layer [`ContextError`] onto the server's error taxonomy.
    /// Datagen routes call the `DatagenStoreApi` trait (which owns the DTO↔core
    /// conversion) and so surface `ContextError` rather than raw `LanceError`.
    pub fn from_context(err: ContextError) -> Self {
        match err {
            ContextError::NotFound(msg) => AppError::NotFound(msg),
            ContextError::AlreadyExists(msg) => AppError::AlreadyExists(msg),
            ContextError::InvalidRequest(msg) => AppError::InvalidRequest(msg),
            ContextError::Internal(msg) => AppError::Internal(msg),
            ContextError::CompactionInProgress => AppError::CompactionInProgress,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            AppError::ClientDisconnected => (
                StatusCode::from_u16(499).expect("499 is a valid HTTP status"),
                "CLIENT_DISCONNECTED",
                "client disconnected".to_string(),
            ),
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

    #[tokio::test]
    async fn read_diagnostics_preserve_wire_errors_and_exclude_raw_messages() {
        use axum::body::to_bytes;
        for make in [
            AppError::Internal as fn(String) -> AppError,
            AppError::Overloaded,
            AppError::NotFound,
            AppError::InvalidRequest,
        ] {
            let message = "private-row-id https://example.invalid/path?sig=secret";
            let original = make(message.into()).into_response();
            let annotated =
                ReadFailureError::new(make(message.into()), ReadFailureStage::GenericStoreOpen)
                    .into_response();
            assert_eq!(annotated.status(), original.status());
            assert_eq!(annotated.headers(), original.headers());
            let diagnostic = annotated.extensions().get::<ReadFailureDiagnostic>();
            assert_eq!(diagnostic.is_some(), original.status().is_server_error());
            let safe = format!("{diagnostic:?}");
            assert!(!safe.contains("private-row-id"));
            assert!(!safe.contains("sig=secret"));
            assert_eq!(
                to_bytes(annotated.into_body(), 4096).await.unwrap(),
                to_bytes(original.into_body(), 4096).await.unwrap()
            );
        }
    }

    #[test]
    fn read_failure_family_uses_typed_errors_without_inspecting_text() {
        let error = ReadFailureError::from_lance(
            LanceError::io("arbitrary sensitive text"),
            ReadFailureStage::GenericRefresh,
        );
        assert_eq!(error.kind, ReadFailureKind::LanceIo);
        let error = ReadFailureError::from_lance(
            LanceError::arrow("arbitrary sensitive text"),
            ReadFailureStage::GenericRetryLookup,
        );
        assert_eq!(error.kind, ReadFailureKind::LanceArrow);
        let error = ReadFailureError::from_lance(
            LanceError::io(format!(
                "{}: details",
                lance_context_core::PENDING_GENERATIONS_EXCEEDED
            )),
            ReadFailureStage::GenericLookup,
        );
        assert_eq!(error.kind, ReadFailureKind::Overloaded);
    }

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
    fn pending_generation_cap_is_overloaded() {
        let err = LanceError::io(format!(
            "{}: shard rollout-0 has 5000 flushed generations pending merge (cap 4096)",
            lance_context_core::PENDING_GENERATIONS_EXCEEDED
        ));
        assert!(matches!(AppError::from_lance(err), AppError::Overloaded(_)));
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
