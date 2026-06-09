use chrono::{DateTime, Utc};

use crate::serde::CONTENT_TYPE_TOMBSTONE;

/// Structured metadata captured alongside each context entry.
#[derive(Debug, Clone, Default)]
pub struct StateMetadata {
    pub step: Option<i32>,
    pub active_plan_id: Option<String>,
    pub tokens_used: Option<i32>,
    pub custom: Option<String>,
}

/// User-facing representation of a context entry written to storage.
#[derive(Debug, Clone)]
pub struct ContextRecord {
    pub id: String,
    pub external_id: Option<String>,
    pub run_id: String,
    pub bot_id: Option<String>,
    pub session_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub role: String,
    pub state_metadata: Option<StateMetadata>,
    pub content_type: String,
    pub text_payload: Option<String>,
    pub binary_payload: Option<Vec<u8>>,
    pub embedding: Option<Vec<f32>>,
}

impl ContextRecord {
    #[must_use]
    pub fn is_tombstone(&self) -> bool {
        self.content_type == CONTENT_TYPE_TOMBSTONE
    }
}

/// Result returned from a vector similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub record: ContextRecord,
    pub distance: f32,
}
