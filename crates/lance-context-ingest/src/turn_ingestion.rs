//! Body schema and business identity for reusable turn ingestion.
//!
//! This prepares identities without changing or deduplicating the body arrays.
//! First-create decisions belong to ordered WAL application, where concurrent
//! producers see one authoritative state. A prepared batch is not a durable ACK.
use anyhow::{ensure, Context, Result};
use arrow_array::{Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Schema, SchemaRef};
use lance_context_wal::{ingest_contract::arrow_schema_fingerprint, placement::Layout};
use serde::{Deserialize, Serialize};
#[cfg(feature = "materializer")]
use std::sync::Arc;
use std::{collections::HashSet, mem::size_of};

const SPEC_KEY: &str = "lance-context.turn-ingestion-spec.v1";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Persisted {
    schema: Schema,
    primary_key: Vec<String>,
    routing_column: String,
}

/// Immutable body and identity definition. The table incarnation supplies the
/// outer key scope. The routing column must be part of the primary key so equal
/// business keys cannot be sent to different shards.
///
/// A specification is bound at table provisioning. Changing it requires a new
/// binding; it is not a local restart option.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "Persisted", into = "Persisted")]
pub struct TurnIngestionSpec {
    body_schema: SchemaRef,
    primary_key: Vec<String>,
    key_indices: Vec<usize>,
    routing_index: usize,
}

impl TryFrom<Persisted> for TurnIngestionSpec {
    type Error = anyhow::Error;
    fn try_from(value: Persisted) -> Result<Self> {
        Self::new(
            std::sync::Arc::new(value.schema),
            value.primary_key,
            &value.routing_column,
        )
    }
}

impl From<TurnIngestionSpec> for Persisted {
    fn from(value: TurnIngestionSpec) -> Self {
        Self {
            schema: value.body_schema.as_ref().clone(),
            primary_key: value.primary_key.clone(),
            routing_column: value.routing_column().into(),
        }
    }
}

impl TurnIngestionSpec {
    pub fn new(
        body_schema: SchemaRef,
        primary_key: Vec<String>,
        routing_column: &str,
    ) -> Result<Self> {
        ensure!(
            !body_schema.metadata().contains_key(SPEC_KEY),
            "body schema contains reserved turn-ingestion metadata"
        );
        let mut names = HashSet::new();
        ensure!(
            body_schema.fields().iter().all(|f| names.insert(f.name())),
            "ambiguous body column names"
        );
        let mut keys = HashSet::new();
        ensure!(
            !primary_key.is_empty() && primary_key.iter().all(|key| keys.insert(key)),
            "primary key must contain distinct columns"
        );
        let key_indices = primary_key
            .iter()
            .map(|name| {
                let index = body_schema.index_of(name)?;
                let field = body_schema.field(index);
                ensure!(
                    field.data_type() == &DataType::Utf8 && !field.is_nullable(),
                    "turn key columns must be non-null UTF8"
                );
                Ok(index)
            })
            .collect::<Result<Vec<_>>>()?;
        let routing_index = body_schema.index_of(routing_column)?;
        ensure!(
            key_indices.contains(&routing_index),
            "routing column must be part of the primary key"
        );
        Ok(Self {
            body_schema,
            primary_key,
            key_indices,
            routing_index,
        })
    }

    pub fn body_schema(&self) -> &SchemaRef {
        &self.body_schema
    }

    pub fn primary_key(&self) -> &[String] {
        &self.primary_key
    }

    pub fn routing_column(&self) -> &str {
        self.body_schema.field(self.routing_index).name()
    }

    /// Persist the complete definition with a transport or narrow-table schema.
    /// Schema metadata maps must have a stable order across independent hosts.
    #[cfg(feature = "materializer")]
    pub(crate) fn bind_schema(&self, schema: &Schema) -> Result<SchemaRef> {
        fn canonical(value: serde_json::Value) -> serde_json::Value {
            use serde_json::Value;
            match value {
                Value::Object(fields) => Value::Object(
                    fields
                        .into_iter()
                        .map(|(key, value)| (key, canonical(value)))
                        .collect::<std::collections::BTreeMap<_, _>>()
                        .into_iter()
                        .collect(),
                ),
                Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
                value => value,
            }
        }
        let value = serde_json::to_string(&canonical(serde_json::to_value(Persisted {
            schema: self.body_schema.as_ref().clone(),
            primary_key: self.primary_key.clone(),
            routing_column: self.routing_column().into(),
        })?))?;
        ensure!(
            value.len() <= 32 << 10,
            "turn ingestion specification too large"
        );
        ensure!(
            schema
                .metadata()
                .get(SPEC_KEY)
                .is_none_or(|old| old == &value),
            "schema already binds a different turn ingestion specification"
        );
        let mut metadata = schema.metadata().clone();
        metadata.insert(SPEC_KEY.into(), value);
        Ok(Arc::new(Schema::new_with_metadata(
            schema.fields().clone(),
            metadata,
        )))
    }

    #[cfg(feature = "materializer")]
    pub(crate) fn from_schema(schema: &Schema) -> Result<Option<Self>> {
        let Some(value) = schema.metadata().get(SPEC_KEY) else {
            return Ok(None);
        };
        ensure!(
            value.len() <= 32 << 10,
            "turn ingestion specification too large"
        );
        let p: Persisted = serde_json::from_str(value)?;
        Ok(Some(Self::new(
            Arc::new(p.schema),
            p.primary_key,
            &p.routing_column,
        )?))
    }

    #[cfg(feature = "materializer")]
    pub(crate) fn key_values<'a>(&self, key: &'a str) -> Result<Vec<std::borrow::Cow<'a, str>>> {
        // JSON escaping may require owned components. This is a narrow point
        // key, never a body decode or history scan.
        let values: Vec<std::borrow::Cow<'a, str>> = serde_json::from_str(key)?;
        let borrowed = values
            .iter()
            .map(|value| value.as_ref())
            .collect::<Vec<_>>();
        ensure!(
            self.business_key(&borrowed, key.len())? == key,
            "noncanonical turn key"
        );
        Ok(values)
    }

    #[cfg(feature = "materializer")]
    pub(crate) fn routing_key(&self, key: &str) -> Result<String> {
        let values = self.key_values(key)?;
        let index = self
            .key_indices
            .iter()
            .position(|&i| i == self.routing_index)
            .unwrap();
        Ok(values[index].to_string())
    }

    #[cfg(feature = "materializer")]
    pub(crate) fn row_key(&self, body: &RecordBatch, row: usize, maximum: usize) -> Result<String> {
        ensure!(
            row < body.num_rows() && body.schema() == self.body_schema,
            "foreign turn row"
        );
        let values = self
            .key_indices
            .iter()
            .map(|&index| {
                let column = body
                    .column(index)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("invalid turn key column")?;
                ensure!(!column.is_null(row), "null turn key component");
                Ok(column.value(row))
            })
            .collect::<Result<Vec<_>>>()?;
        self.business_key(&values, maximum)
    }

    /// Includes the actual body schema and its metadata, ordered key columns,
    /// and routing rule. It is not a rollout/mixed wire-schema fingerprint.
    pub fn fingerprint(&self) -> Result<[u8; 32]> {
        let mut metadata = self.body_schema.metadata().clone();
        metadata.insert(
            SPEC_KEY.into(),
            serde_json::to_string(&(&self.primary_key, self.routing_column()))?,
        );
        arrow_schema_fingerprint(&Schema::new_with_metadata(
            self.body_schema.fields().clone(),
            metadata,
        ))
        .map_err(Into::into)
    }

    /// Canonical tuple encoding; delimiters, escaping and non-ASCII values do
    /// not alias. Original IDs and relationship columns stay in the body.
    pub fn business_key(&self, values: &[&str], max_key_bytes: usize) -> Result<String> {
        ensure!(
            values.len() == self.primary_key.len() && values.iter().all(|v| !v.is_empty()),
            "turn key arity differs or a key component is empty"
        );
        let input_bytes = values
            .iter()
            .try_fold(0usize, |n, value| n.checked_add(value.len()))
            .context("turn key length overflow")?;
        // Bound input before escaping can allocate. Encoded length is checked
        // independently because JSON escaping may expand the tuple.
        ensure!(input_bytes <= max_key_bytes, "turn key exceeds value limit");
        let key = serde_json::to_string(values)?;
        ensure!(
            key.len() <= max_key_bytes,
            "encoded turn key exceeds value limit"
        );
        Ok(key)
    }

    /// Preserve all rows, including duplicate keys and their original order.
    /// The caller reserves identity workspace before calling this method, using
    /// its existing shared admission budget. This is not a private memory pool
    /// or complete body validation for a durable ingestion endpoint.
    pub fn prepare(&self, body: RecordBatch, max_key_bytes: usize) -> Result<PreparedTurns> {
        ensure!(
            body.schema() == self.body_schema,
            "foreign turn body schema"
        );
        let columns = self
            .key_indices
            .iter()
            .map(|&index| {
                let values = body
                    .column(index)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("invalid turn key column")?;
                ensure!(values.null_count() == 0, "null turn key component");
                Ok(values)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut keys = Vec::with_capacity(body.num_rows());
        let mut values = Vec::with_capacity(columns.len());
        for row in 0..body.num_rows() {
            values.clear();
            values.extend(columns.iter().map(|column| column.value(row)));
            keys.push(self.business_key(&values, max_key_bytes)?);
        }
        Ok(PreparedTurns {
            body,
            keys,
            routing_index: self.routing_index,
        })
    }
}

/// Original Arrow arrays plus bounded, owned narrow identities. No body fields
/// are renamed, flattened, serialized or copied by this preparation.
pub struct PreparedTurns {
    body: RecordBatch,
    keys: Vec<String>,
    routing_index: usize,
}

impl PreparedTurns {
    pub fn body(&self) -> &RecordBatch {
        &self.body
    }

    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    /// Route with the same tenant/table namespace as the resolved binding.
    pub fn bucket(&self, row: usize, layout: &Layout, tenant: &str, table: &str) -> Result<u32> {
        ensure!(row < self.body.num_rows(), "turn row out of bounds");
        let values = self
            .body
            .column(self.routing_index)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("validated UTF8 routing column");
        layout.session_bucket(tenant, table, values.value(row))
    }

    /// Logical owned identity allocation, excluding Arrow buffers, allocator
    /// overhead and preparation scratch. Body ownership remains caller-charged.
    pub fn identity_bytes(&self) -> usize {
        self.keys.capacity() * size_of::<String>()
            + self.keys.iter().map(String::capacity).sum::<usize>()
    }

    pub fn into_parts(self) -> (RecordBatch, Vec<String>) {
        (self.body, self.keys)
    }
}

#[cfg(test)]
mod tests;
