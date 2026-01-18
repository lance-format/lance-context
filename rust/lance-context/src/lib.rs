//! Core types for the lance-context storage layer.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    FixedSizeListBuilder, Float32Builder, Int32Builder, LargeBinaryBuilder, LargeStringBuilder,
    StringBuilder, StringDictionaryBuilder, StructBuilder, TimestampMicrosecondBuilder,
};
use arrow_array::types::Int8Type;
use arrow_array::{ArrayRef, DictionaryArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema, TimeUnit};
use chrono::{DateTime, Utc};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::{Error as LanceError, Result as LanceResult};

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
    pub run_id: String,
    pub created_at: DateTime<Utc>,
    pub role: String,
    pub state_metadata: Option<StateMetadata>,
    pub content_type: String,
    pub text_payload: Option<String>,
    pub binary_payload: Option<Vec<u8>>,
    pub embedding: Option<Vec<f32>>,
}

/// Embedding length used for the semantic index column.
const DEFAULT_EMBEDDING_DIM: i32 = 1536;

/// Persistent Lance-backed context store.
#[derive(Clone)]
pub struct ContextStore {
    dataset: Dataset,
}

impl ContextStore {
    /// Open an existing context dataset or create a new one with the project schema.
    pub async fn open(uri: &str) -> LanceResult<Self> {
        match Dataset::open(uri).await {
            Ok(dataset) => Ok(Self { dataset }),
            Err(LanceError::DatasetNotFound { .. }) => {
                let schema = Arc::new(Self::schema());
                let empty_batch = RecordBatch::new_empty(schema.clone());
                let batches = RecordBatchIterator::new(
                    vec![Ok::<RecordBatch, ArrowError>(empty_batch)].into_iter(),
                    schema.clone(),
                );
                let params = WriteParams {
                    mode: WriteMode::Create,
                    ..Default::default()
                };
                let dataset = Dataset::write(batches, uri, Some(params)).await?;
                Ok(Self { dataset })
            }
            Err(err) => Err(err),
        }
    }

    /// Append context records to the store and return the new dataset version.
    pub async fn add(&mut self, entries: &[ContextRecord]) -> LanceResult<u64> {
        if entries.is_empty() {
            return Ok(self.dataset.manifest.version);
        }

        let batch = Self::records_to_batch(entries)?;
        let schema = batch.schema();
        let reader =
            RecordBatchIterator::new(vec![Ok::<RecordBatch, ArrowError>(batch)].into_iter(), schema);
        self.dataset.append(reader, None).await?;

        Ok(self.dataset.manifest.version)
    }

    /// Lance schema for the context store.
    pub fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("run_id", DataType::Utf8, false),
            Field::new(
                "created_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "role",
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Utf8),
                ),
                false,
            ),
            Field::new(
                "state_metadata",
                DataType::Struct(vec![
                    Field::new("step", DataType::Int32, true),
                    Field::new("active_plan_id", DataType::Utf8, true),
                    Field::new("tokens_used", DataType::Int32, true),
                    Field::new("custom", DataType::Utf8, true),
                ]
                .into()),
                true,
            ),
            Field::new("content_type", DataType::Utf8, false),
            Field::new("text_payload", DataType::LargeUtf8, true),
            Field::new("binary_payload", DataType::LargeBinary, true),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DEFAULT_EMBEDDING_DIM,
                ),
                true,
            ),
        ])
    }

    fn records_to_batch(entries: &[ContextRecord]) -> LanceResult<RecordBatch> {
        let mut id_builder = StringBuilder::new();
        let mut run_id_builder = StringBuilder::new();
        let mut created_at_builder = TimestampMicrosecondBuilder::with_capacity(entries.len());
        let mut role_builder = StringDictionaryBuilder::<Int8Type>::new();
        let mut content_type_builder = StringBuilder::new();
        let mut text_builder = LargeStringBuilder::new();
        let mut binary_builder = LargeBinaryBuilder::new();

        let state_fields: Vec<FieldRef> = vec![
            Arc::new(Field::new("step", DataType::Int32, true)),
            Arc::new(Field::new("active_plan_id", DataType::Utf8, true)),
            Arc::new(Field::new("tokens_used", DataType::Int32, true)),
            Arc::new(Field::new("custom", DataType::Utf8, true)),
        ];
        let mut state_builder = StructBuilder::new(
            state_fields,
            vec![
                Box::new(Int32Builder::new()),
                Box::new(StringBuilder::new()),
                Box::new(Int32Builder::new()),
                Box::new(StringBuilder::new()),
            ],
        );

        let mut embedding_builder =
            FixedSizeListBuilder::new(Float32Builder::new(), DEFAULT_EMBEDDING_DIM);

        for entry in entries {
            id_builder.append_value(&entry.id);
            run_id_builder.append_value(&entry.run_id);
            created_at_builder.append_value(entry.created_at.timestamp_micros());
            role_builder.append(&entry.role)?;
            content_type_builder.append_value(&entry.content_type);

            match &entry.text_payload {
                Some(value) => text_builder.append_value(value),
                None => text_builder.append_null(),
            }

            match &entry.binary_payload {
                Some(value) => binary_builder.append_value(value),
                None => binary_builder.append_null(),
            }

            if let Some(metadata) = &entry.state_metadata {
                state_builder
                    .field_builder::<Int32Builder>(0)
                    .unwrap()
                    .append_option(metadata.step);
                state_builder
                    .field_builder::<StringBuilder>(1)
                    .unwrap()
                    .append_option(metadata.active_plan_id.as_deref());
                state_builder
                    .field_builder::<Int32Builder>(2)
                    .unwrap()
                    .append_option(metadata.tokens_used);
                state_builder
                    .field_builder::<StringBuilder>(3)
                    .unwrap()
                    .append_option(metadata.custom.as_deref());
                state_builder.append(true);
            } else {
                state_builder
                    .field_builder::<Int32Builder>(0)
                    .unwrap()
                    .append_null();
                state_builder
                    .field_builder::<StringBuilder>(1)
                    .unwrap()
                    .append_null();
                state_builder
                    .field_builder::<Int32Builder>(2)
                    .unwrap()
                    .append_null();
                state_builder
                    .field_builder::<StringBuilder>(3)
                    .unwrap()
                    .append_null();
                state_builder.append(false);
            }

            if let Some(embedding) = &entry.embedding {
                if embedding.len() != DEFAULT_EMBEDDING_DIM as usize {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "embedding length {} does not match expected dimension {}",
                        embedding.len(),
                        DEFAULT_EMBEDDING_DIM
                    ))
                    .into());
                }
                {
                    let values_builder = embedding_builder.values();
                    for value in embedding {
                        values_builder.append_value(*value);
                    }
                }
                embedding_builder.append(true);
            } else {
                embedding_builder.append(false);
            }
        }

        let id_array: ArrayRef = Arc::new(id_builder.finish());
        let run_id_array: ArrayRef = Arc::new(run_id_builder.finish());
        let created_at_array: ArrayRef = Arc::new(created_at_builder.finish());
        let role_array: ArrayRef =
            Arc::new(DictionaryArray::<Int8Type>::from(role_builder.finish()));
        let content_type_array: ArrayRef = Arc::new(content_type_builder.finish());
        let text_array: ArrayRef = Arc::new(text_builder.finish());
        let binary_array: ArrayRef = Arc::new(binary_builder.finish());
        let state_array: ArrayRef = Arc::new(state_builder.finish());
        let embedding_array: ArrayRef = Arc::new(embedding_builder.finish());

        let schema = Arc::new(Self::schema());
        let batch = RecordBatch::try_new(
            schema,
            vec![
                id_array,
                run_id_array,
                created_at_array,
                role_array,
                state_array,
                content_type_array,
                text_array,
                binary_array,
                embedding_array,
            ],
        )?;

        Ok(batch)
    }
}

#[derive(Debug, Clone)]
pub struct Context {
    uri: String,
    branch: String,
    next_id: u64,
    entries: Vec<ContextEntry>,
    snapshots: HashMap<String, Snapshot>,
}

impl Context {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            branch: "main".to_string(),
            next_id: 1,
            entries: Vec::new(),
            snapshots: HashMap::new(),
        }
    }

    pub fn uri(&self) -> &str {
        &self.uri
    }

    pub fn branch(&self) -> &str {
        &self.branch
    }

    pub fn entries(&self) -> u64 {
        self.entries.len() as u64
    }

    pub fn add(&mut self, role: &str, content: &str, data_type: Option<&str>) -> u64 {
        let entry_id = self.next_id;
        self.next_id += 1;
        self.entries.push(ContextEntry {
            id: entry_id,
            role: role.to_string(),
            data_type: data_type.map(|value| value.to_string()),
            content: content.to_string(),
        });
        entry_id
    }

    pub fn snapshot(&mut self, label: Option<&str>) -> String {
        let id = match label {
            Some(label) if !label.is_empty() => label.to_string(),
            _ => format!("snapshot-{}", self.entries.len()),
        };
        let snapshot = Snapshot {
            id: id.clone(),
            label: label.map(|value| value.to_string()),
            entry_count: self.entries.len() as u64,
            branch: self.branch.clone(),
        };
        self.snapshots.insert(id.clone(), snapshot);
        id
    }

    pub fn fork(&self, branch_name: impl Into<String>) -> Self {
        Self {
            uri: self.uri.clone(),
            branch: branch_name.into(),
            next_id: self.next_id,
            entries: self.entries.clone(),
            snapshots: self.snapshots.clone(),
        }
    }

    pub fn checkout(&mut self, snapshot_id: &str) {
        if let Some(snapshot) = self.snapshots.get(snapshot_id) {
            self.entries.truncate(snapshot.entry_count as usize);
            self.next_id = self.entries.len() as u64 + 1;
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub id: u64,
    pub role: String,
    pub data_type: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub id: String,
    pub label: Option<String>,
    pub entry_count: u64,
    pub branch: String,
}
