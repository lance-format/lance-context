use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;

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
    pub metadata: Option<Value>,
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

/// Metadata matching operation for filtered retrieval.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataFilter {
    Equals(Value),
    Contains(Value),
}

/// Filters applied to records before list pagination or search ranking.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecordFilters {
    pub bot_id: Option<String>,
    pub session_id: Option<String>,
    pub role: Option<String>,
    pub content_type: Option<String>,
    pub created_at_start: Option<DateTime<Utc>>,
    pub created_at_end: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, MetadataFilter>,
}

impl RecordFilters {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bot_id.is_none()
            && self.session_id.is_none()
            && self.role.is_none()
            && self.content_type.is_none()
            && self.created_at_start.is_none()
            && self.created_at_end.is_none()
            && self.metadata.is_empty()
    }

    #[must_use]
    pub fn matches(&self, record: &ContextRecord) -> bool {
        if self
            .bot_id
            .as_deref()
            .is_some_and(|value| record.bot_id.as_deref() != Some(value))
        {
            return false;
        }
        if self
            .session_id
            .as_deref()
            .is_some_and(|value| record.session_id.as_deref() != Some(value))
        {
            return false;
        }
        if self
            .role
            .as_deref()
            .is_some_and(|value| record.role != value)
        {
            return false;
        }
        if self
            .content_type
            .as_deref()
            .is_some_and(|value| record.content_type != value)
        {
            return false;
        }
        if self
            .created_at_start
            .is_some_and(|start| record.created_at < start)
        {
            return false;
        }
        if self
            .created_at_end
            .is_some_and(|end| record.created_at > end)
        {
            return false;
        }

        self.metadata.iter().all(|(key, filter)| {
            let Some(Value::Object(metadata)) = &record.metadata else {
                return false;
            };
            let Some(value) = metadata.get(key) else {
                return false;
            };
            match filter {
                MetadataFilter::Equals(expected) => value == expected,
                MetadataFilter::Contains(expected) => metadata_contains(value, expected),
            }
        })
    }
}

fn metadata_contains(value: &Value, expected: &Value) -> bool {
    match (value, expected) {
        (Value::Array(items), expected) => items.iter().any(|item| item == expected),
        (Value::String(value), Value::String(expected)) => value.contains(expected),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    fn record() -> ContextRecord {
        ContextRecord {
            id: "rec-1".to_string(),
            external_id: None,
            run_id: "run-1".to_string(),
            bot_id: Some("support-bot".to_string()),
            session_id: Some("incident-1".to_string()),
            created_at: Utc.with_ymd_and_hms(2026, 6, 9, 3, 0, 0).unwrap(),
            role: "assistant".to_string(),
            state_metadata: None,
            metadata: Some(json!({
                "scope": "team",
                "tags": ["runbook", "ownership"],
                "confidence": 0.92
            })),
            content_type: "text/plain".to_string(),
            text_payload: Some("hello".to_string()),
            binary_payload: None,
            embedding: None,
        }
    }

    #[test]
    fn filters_match_builtin_fields_timestamps_and_metadata() {
        let mut filters = RecordFilters {
            bot_id: Some("support-bot".to_string()),
            session_id: Some("incident-1".to_string()),
            role: Some("assistant".to_string()),
            content_type: Some("text/plain".to_string()),
            created_at_start: Some(Utc.with_ymd_and_hms(2026, 6, 9, 2, 0, 0).unwrap()),
            created_at_end: Some(Utc.with_ymd_and_hms(2026, 6, 9, 4, 0, 0).unwrap()),
            metadata: HashMap::new(),
        };
        filters
            .metadata
            .insert("scope".to_string(), MetadataFilter::Equals(json!("team")));
        filters.metadata.insert(
            "tags".to_string(),
            MetadataFilter::Contains(json!("runbook")),
        );

        assert!(filters.matches(&record()));

        filters.session_id = Some("other".to_string());
        assert!(!filters.matches(&record()));
    }
}
