use chrono::{DateTime, Utc};

pub const LIFECYCLE_ACTIVE: &str = "active";
pub const LIFECYCLE_CONTRADICTED: &str = "contradicted";

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
    pub expires_at: Option<DateTime<Utc>>,
    pub retention_policy: Option<String>,
    pub lifecycle_status: String,
    pub retired_at: Option<DateTime<Utc>>,
    pub retired_reason: Option<String>,
    pub supersedes_id: Option<String>,
    pub superseded_by_id: Option<String>,
    pub content_type: String,
    pub text_payload: Option<String>,
    pub binary_payload: Option<Vec<u8>>,
    pub embedding: Option<Vec<f32>>,
}

impl ContextRecord {
    #[must_use]
    pub fn is_expired_at(&self, now: DateTime<Utc>) -> bool {
        self.expires_at.is_some_and(|expires_at| expires_at <= now)
    }

    #[must_use]
    pub fn is_hidden_by_lifecycle(&self) -> bool {
        if self.lifecycle_status == LIFECYCLE_ACTIVE
            || self.lifecycle_status == LIFECYCLE_CONTRADICTED
        {
            return self.retired_at.is_some() || self.superseded_by_id.is_some();
        }

        true
    }

    #[must_use]
    pub fn has_non_default_lifecycle(&self) -> bool {
        self.expires_at.is_some()
            || self.retention_policy.is_some()
            || self.lifecycle_status != LIFECYCLE_ACTIVE
            || self.retired_at.is_some()
            || self.retired_reason.is_some()
            || self.supersedes_id.is_some()
            || self.superseded_by_id.is_some()
    }
}

/// Query-time controls for lifecycle-aware retrieval.
#[derive(Debug, Clone)]
pub struct LifecycleQueryOptions {
    pub include_expired: bool,
    pub include_retired: bool,
    pub reference_time: DateTime<Utc>,
}

impl Default for LifecycleQueryOptions {
    fn default() -> Self {
        Self {
            include_expired: false,
            include_retired: false,
            reference_time: Utc::now(),
        }
    }
}

impl LifecycleQueryOptions {
    #[must_use]
    pub fn new(include_expired: bool, include_retired: bool) -> Self {
        Self {
            include_expired,
            include_retired,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn is_visible(&self, record: &ContextRecord) -> bool {
        (self.include_expired || !record.is_expired_at(self.reference_time))
            && (self.include_retired || !record.is_hidden_by_lifecycle())
    }
}

/// Result returned from a vector similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub record: ContextRecord,
    pub distance: f32,
}
