//! A typed conversation-turn schema over the shared generic storage API.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    AddRowsResponse, ColumnSpec, ColumnType, ContextError, ContextResult, GenericStoreApi,
    SchemaSpec,
};

/// One observed turn, not a complete model call or a copy of its input history.
///
/// `turn_id` is a zero-based position in the observed session, independent of
/// arrival order. It is not the storage key: different content at the same
/// `(session_id, turn_id)` must use different `id`s. Replays must reuse their ID.
/// A caller can derive `id` from session, turn, role, content type and exact
/// content (including null versus empty); the store does not generate it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceRecord {
    pub id: String,
    pub session_id: String,
    pub turn_id: i64,
    /// Open vocabulary: user, assistant, system, tool, or a source-specific role.
    pub role: String,
    /// Exact turn content. Null and an empty string remain distinct.
    pub content: Option<String>,
    #[serde(default = "text_plain")]
    pub content_type: String,
    /// Source dataset/system, not an individual occurrence of this shared turn.
    #[serde(default)]
    pub source: Option<String>,
    /// Optional source-specific annotations, physically encoded as JSON text.
    /// This does not implicitly accumulate provenance when an ID is replayed.
    #[serde(default)]
    pub metadata: Option<Value>,
}

fn text_plain() -> String {
    "text/plain".into()
}

impl TraceRecord {
    pub fn validate(&self) -> ContextResult<()> {
        for (name, value) in [
            ("id", &self.id),
            ("session_id", &self.session_id),
            ("role", &self.role),
            ("content_type", &self.content_type),
        ] {
            if value.trim().is_empty() {
                return Err(ContextError::InvalidRequest(format!(
                    "trace {name} must not be empty"
                )));
            }
        }
        if self.turn_id < 0 {
            return Err(ContextError::InvalidRequest(
                "trace turn_id must be nonnegative".into(),
            ));
        }
        Ok(())
    }

    /// Encode for the existing generic batch/HTTP API, validating before I/O.
    pub fn to_row(&self) -> ContextResult<Map<String, Value>> {
        self.validate()?;
        let Value::Object(mut row) = serde_json::to_value(self).map_err(codec_error)? else {
            unreachable!("TraceRecord serializes as an object")
        };
        row.insert(
            "metadata".into(),
            match &self.metadata {
                Some(value) => Value::String(serde_json::to_string(value).map_err(codec_error)?),
                None => Value::Null,
            },
        );
        Ok(row)
    }

    /// Decode a complete generic row. Projected rows are not complete records.
    pub fn from_row(mut row: Map<String, Value>) -> ContextResult<Self> {
        // Decode separately to preserve Some(JSON null) versus absent metadata.
        let metadata = match row.remove("metadata") {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) => Some(serde_json::from_str(&text).map_err(codec_error)?),
            Some(_) => {
                return Err(ContextError::InvalidRequest(
                    "trace metadata column must contain JSON text".into(),
                ))
            }
        };
        let mut record: Self = serde_json::from_value(Value::Object(row)).map_err(codec_error)?;
        record.metadata = metadata;
        record.validate()?;
        Ok(record)
    }
}

fn codec_error(error: serde_json::Error) -> ContextError {
    ContextError::InvalidRequest(format!("invalid trace record: {error}"))
}

/// Built-in trace schema, persisted by GenericStore like every other SchemaSpec.
/// `id` is the indexed storage key; `(session_id, turn_id)` is deliberately not
/// unique so branches and conflicting observations can coexist.
pub fn trace_schema() -> SchemaSpec {
    let string = || ColumnType::String { large: false };
    SchemaSpec::new(vec![
        ("id".into(), ColumnSpec::required(string())),
        ("session_id".into(), ColumnSpec::required(string())),
        ("turn_id".into(), ColumnSpec::required(ColumnType::Int64)),
        ("role".into(), ColumnSpec::required(string())),
        (
            "content".into(),
            ColumnSpec::new(ColumnType::String { large: true }),
        ),
        ("content_type".into(), ColumnSpec::required(string())),
        ("source".into(), ColumnSpec::new(string())),
        (
            "metadata".into(),
            ColumnSpec::new(ColumnType::String { large: true }),
        ),
    ])
}

/// Typed trace access over an embedded or remote GenericStore, with the same
/// WAL visibility, indexed batch lookup, and newest-ID resolution.
///
/// This is one dataset, not an additional store or index. `existing_ids` is a
/// read, not an atomic insert-if-absent operation: the caller must coordinate
/// competing writers and flush before relying on a previous append's visibility.
pub struct TraceStore<S> {
    store: S,
}

impl<S: GenericStoreApi> TraceStore<S> {
    /// Reject a mismatched schema before writing to an existing store.
    pub fn new(store: S) -> ContextResult<Self> {
        if store.spec() != &trace_schema() {
            return Err(ContextError::InvalidRequest(
                "store does not have the trace schema".into(),
            ));
        }
        Ok(Self { store })
    }

    /// Access projections, filtering, and the backend's maintenance interfaces.
    pub fn as_generic(&self) -> &S {
        &self.store
    }

    /// Recover the owned backend for shutdown or exclusive index/WAL maintenance.
    pub fn into_generic(self) -> S {
        self.store
    }

    /// Validate the entire batch before appending once. Repeated IDs have the
    /// generic store's newest-write-wins semantics, not immutable conflict checks.
    /// Distinct content variants must have distinct IDs. Visibility on return
    /// requires seal-on-add, or a subsequent `flush`.
    pub async fn add(&self, records: &[TraceRecord]) -> ContextResult<AddRowsResponse> {
        let rows = records
            .iter()
            .map(TraceRecord::to_row)
            .collect::<ContextResult<Vec<_>>>()?;
        self.store.add(&rows).await
    }

    /// Fetch up to MAX_BATCH_GET_IDS records, including pending flushed WAL.
    /// Missing IDs are omitted; duplicates appear once in first-requested order.
    pub async fn get_many(&self, ids: &[String]) -> ContextResult<Vec<TraceRecord>> {
        self.store
            .get_many(ids, None)
            .await?
            .into_iter()
            .map(TraceRecord::from_row)
            .collect()
    }

    /// Batch deduplication probe using an ID-only projection, without reading
    /// content or metadata. Shares get_many's bounds, order and WAL semantics.
    pub async fn existing_ids(&self, ids: &[String]) -> ContextResult<Vec<String>> {
        self.store
            .get_many(ids, Some(&["id".into()]))
            .await?
            .into_iter()
            .map(|mut row| match row.remove("id") {
                Some(Value::String(id)) => Ok(id),
                _ => Err(ContextError::Internal("trace lookup returned no ID".into())),
            })
            .collect()
    }

    /// Publish the active WAL generation for subsequent deduplication probes.
    pub async fn flush(&self) -> ContextResult<()> {
        self.store.flush().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> TraceRecord {
        serde_json::from_value(serde_json::json!({
            "id":"variant", "session_id":"s", "turn_id":0,
            "role":"user", "content":"你好"
        }))
        .unwrap()
    }

    #[test]
    fn trace_codec_preserves_content_and_json_nulls() {
        trace_schema().validate().unwrap();
        for content in [None, Some(String::new()), Some("原样\n雪".into())] {
            for metadata in [
                None,
                Some(Value::Null),
                Some(serde_json::json!({"x":[1,true]})),
            ] {
                let mut record = record();
                record.content = content.clone();
                record.metadata = metadata;
                assert_eq!(
                    TraceRecord::from_row(record.to_row().unwrap()).unwrap(),
                    record
                );
            }
        }
    }

    #[test]
    fn invalid_identity_and_unknown_columns_are_rejected() {
        let original = record();
        let mut invalid = original.clone();
        invalid.turn_id = -1;
        assert!(invalid.to_row().is_err());
        for field in ["id", "session_id", "role", "content_type"] {
            let mut row = original.to_row().unwrap();
            row.insert(field.into(), Value::String(" ".into()));
            assert!(TraceRecord::from_row(row).is_err());
        }
        let mut row = original.to_row().unwrap();
        row.insert("position".into(), Value::from(0));
        assert!(TraceRecord::from_row(row).is_err());
        let mut row = original.to_row().unwrap();
        row.insert("metadata".into(), Value::String("invalid JSON".into()));
        assert!(TraceRecord::from_row(row).is_err());
    }
}
