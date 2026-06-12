use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::serde::CONTENT_TYPE_TOMBSTONE;

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

/// Directed relationship from this record to another graph node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Relationship {
    pub target_id: String,
    pub relation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
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
    pub relationships: Vec<Relationship>,
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
    pub fn is_tombstone(&self) -> bool {
        self.content_type == CONTENT_TYPE_TOMBSTONE
    }

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
        !record.is_tombstone()
            && (self.include_expired || !record.is_expired_at(self.reference_time))
            && (self.include_retired || !record.is_hidden_by_lifecycle())
    }
}

/// Result returned from a vector similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub record: ContextRecord,
    pub distance: f32,
}

/// Result returned from hybrid retrieval over context records.
#[derive(Debug, Clone)]
pub struct RetrieveResult {
    pub record: ContextRecord,
    pub score: f32,
    pub vector_distance: Option<f32>,
    pub text_score: Option<f32>,
    pub matched_channels: Vec<String>,
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
    pub fn from_json_value(value: Value) -> Result<Self, String> {
        let Value::Object(object) = value else {
            return Err("filters must be a JSON object".to_string());
        };

        let mut filters = RecordFilters::default();
        for (key, value) in object {
            match key.as_str() {
                "bot_id" => filters.bot_id = filter_string(key.as_str(), value)?,
                "session_id" => filters.session_id = filter_string(key.as_str(), value)?,
                "role" => filters.role = filter_string(key.as_str(), value)?,
                "content_type" => filters.content_type = filter_string(key.as_str(), value)?,
                "created_at" => apply_created_at_filter(&mut filters, value)?,
                "created_at_start" | "created_after" | "created_at_gte" => {
                    filters.created_at_start = Some(parse_filter_datetime(&key, &value)?);
                }
                "created_at_end" | "created_before" | "created_at_lte" => {
                    filters.created_at_end = Some(parse_filter_datetime(&key, &value)?);
                }
                _ => {
                    let filter = match value {
                        Value::Object(mut object)
                            if object.len() == 1 && object.contains_key("contains") =>
                        {
                            MetadataFilter::Contains(object.remove("contains").unwrap())
                        }
                        value => MetadataFilter::Equals(value),
                    };
                    filters.metadata.insert(key, filter);
                }
            }
        }

        Ok(filters)
    }

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

fn filter_string(name: &str, value: Value) -> Result<Option<String>, String> {
    match value {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value)),
        _ => Err(format!("filter '{name}' must be a string or null")),
    }
}

fn apply_created_at_filter(filters: &mut RecordFilters, value: Value) -> Result<(), String> {
    let Value::Object(object) = value else {
        return Err("filter 'created_at' must be an object with gte/lte bounds".to_string());
    };

    for (key, value) in object {
        match key.as_str() {
            "gte" | "start" | "after" => {
                filters.created_at_start = Some(parse_filter_datetime(&key, &value)?);
            }
            "lte" | "end" | "before" => {
                filters.created_at_end = Some(parse_filter_datetime(&key, &value)?);
            }
            other => {
                return Err(format!("unsupported created_at filter operator '{other}'"));
            }
        }
    }

    Ok(())
}

fn parse_filter_datetime(name: &str, value: &Value) -> Result<DateTime<Utc>, String> {
    let Some(value) = value.as_str() else {
        return Err(format!(
            "filter '{name}' must be an ISO-8601 timestamp string"
        ));
    };
    DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|err| err.to_string())
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
            relationships: Vec::new(),
            expires_at: None,
            retention_policy: None,
            lifecycle_status: LIFECYCLE_ACTIVE.to_string(),
            retired_at: None,
            retired_reason: None,
            supersedes_id: None,
            superseded_by_id: None,
            content_type: "text/plain".to_string(),
            text_payload: Some("hello".to_string()),
            binary_payload: None,
            embedding: None,
        }
    }

    #[test]
    fn filters_match_builtin_fields_timestamps_and_metadata() {
        let mut filters = RecordFilters::from_json_value(json!({
            "bot_id": "support-bot",
            "session_id": "incident-1",
            "role": "assistant",
            "content_type": "text/plain",
            "created_at": {
                "gte": "2026-06-09T02:00:00Z",
                "lte": "2026-06-09T04:00:00Z"
            },
            "scope": "team",
            "tags": {"contains": "runbook"}
        }))
        .unwrap();

        assert!(filters.matches(&record()));

        filters.session_id = Some("other".to_string());
        assert!(!filters.matches(&record()));
    }
}
