use std::sync::Arc;

use arrow_array::builder::{
    FixedSizeListBuilder, Float32Builder, Int32Builder, LargeBinaryBuilder, LargeStringBuilder,
    StringBuilder, StringDictionaryBuilder, StructBuilder, TimestampMicrosecondBuilder,
};
use arrow_array::types::Int8Type;
use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema, TimeUnit};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::{Error as LanceError, Result as LanceResult};

use crate::record::ContextRecord;

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
        let reader = RecordBatchIterator::new(
            vec![Ok::<RecordBatch, ArrowError>(batch)].into_iter(),
            schema,
        );
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
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                false,
            ),
            Field::new(
                "state_metadata",
                DataType::Struct(
                    vec![
                        Field::new("step", DataType::Int32, true),
                        Field::new("active_plan_id", DataType::Utf8, true),
                        Field::new("tokens_used", DataType::Int32, true),
                        Field::new("custom", DataType::Utf8, true),
                    ]
                    .into(),
                ),
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
                // FixedSizeListBuilder requires padding values for null slots.
                let values_builder = embedding_builder.values();
                for _ in 0..DEFAULT_EMBEDDING_DIM {
                    values_builder.append_null();
                }
                embedding_builder.append(false);
            }
        }

        let id_array: ArrayRef = Arc::new(id_builder.finish());
        let run_id_array: ArrayRef = Arc::new(run_id_builder.finish());
        let created_at_array: ArrayRef = Arc::new(created_at_builder.finish());
        let role_array: ArrayRef = Arc::new(role_builder.finish());
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
