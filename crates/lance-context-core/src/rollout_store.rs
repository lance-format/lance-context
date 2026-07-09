//! Storage surface for the RL rollout schema.
//!
//! [`RolloutStore`] is a second, independent first-class store alongside
//! [`crate::store::ContextStore`]. It owns its own Lance dataset and Arrow
//! schema (see [`crate::rollout::RolloutRecord`] and spec §5) but reuses the
//! schema-agnostic infrastructure: native dataset versioning, blob v2 payload
//! offload, and the relationship graph.
//!
//! Unlike `ContextStore`, the v1 write path is a plain atomic
//! [`Dataset::append`] — no MemWAL region grouping (that machinery is specific
//! to conversation `bot_id`/`session_id` sharding). Rollout writes still get
//! native versioning (`checkout`), blob offload, and relationships, all at the
//! schema level.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float32Builder, Int32Builder, Int64Builder, Int8Builder, LargeBinaryBuilder,
    LargeStringBuilder, ListBuilder, StringBuilder, StringDictionaryBuilder,
    TimestampMicrosecondBuilder,
};
use arrow_array::types::Int8Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, Float32Array, Int32Array, Int64Array,
    Int8Array, LargeBinaryArray, LargeStringArray, ListArray, RecordBatch, RecordBatchIterator,
    StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use futures::TryStreamExt;
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::datatypes::BlobHandling;
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};

use crate::rollout::RolloutRecord;
use crate::store::{
    column_as, column_as_optional, relationship_field, relationship_list_item_field,
    relationship_struct_builder, relationships_from_list, timestamp_from_micros,
    RELATIONSHIPS_COLUMN,
};

/// Blob v2 columns permitted on a rollout dataset. `binary_payload` holds
/// artifact bytes offloaded to independent files (spec §6).
const VALID_ROLLOUT_BLOB_COLUMNS: &[&str] = &["binary_payload"];

/// Configuration for opening a [`RolloutStore`].
#[derive(Debug, Clone, Default)]
pub struct RolloutStoreOptions {
    /// Object-store credentials/config (e.g. S3), forwarded to Lance.
    pub storage_options: Option<HashMap<String, String>>,
    /// Columns physically offloaded via blob v2. Only `binary_payload` is valid.
    /// Empty means `binary_payload` is stored inline (no offload).
    pub blob_columns: HashSet<String>,
}

/// A Lance-backed store for RL rollout trajectories.
pub struct RolloutStore {
    dataset: Dataset,
    storage_options: Option<HashMap<String, String>>,
    blob_columns: HashSet<String>,
}

impl RolloutStore {
    /// Open an existing rollout dataset or create a new one, defaulting
    /// `binary_payload` to a blob v2 (offloaded) column.
    pub async fn open(uri: &str) -> LanceResult<Self> {
        let mut blob_columns = HashSet::new();
        blob_columns.insert("binary_payload".to_string());
        Self::open_with_options(
            uri,
            RolloutStoreOptions {
                storage_options: None,
                blob_columns,
            },
        )
        .await
    }

    /// Open a rollout dataset with explicit storage and blob configuration.
    pub async fn open_with_options(uri: &str, options: RolloutStoreOptions) -> LanceResult<Self> {
        for col in &options.blob_columns {
            if !VALID_ROLLOUT_BLOB_COLUMNS.contains(&col.as_str()) {
                return Err(LanceError::from(ArrowError::InvalidArgumentError(format!(
                    "invalid rollout blob column '{}': valid columns are {:?}",
                    col, VALID_ROLLOUT_BLOB_COLUMNS
                ))));
            }
        }

        let storage_options = options.storage_options.clone();
        let blob_columns = options.blob_columns.clone();
        let dataset = match Self::load_with_options(uri, storage_options.clone()).await {
            Ok(dataset) => dataset,
            Err(LanceError::DatasetNotFound { .. }) => {
                Self::create_with_options(uri, storage_options.clone(), &blob_columns).await?
            }
            Err(err) => return Err(err),
        };

        Ok(Self {
            dataset,
            storage_options,
            blob_columns,
        })
    }

    /// URI of the underlying Lance dataset.
    #[must_use]
    pub fn uri(&self) -> &str {
        self.dataset.uri()
    }

    /// Current dataset version.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.dataset.manifest.version
    }

    /// Checkout a specific dataset version — recovers the exact rollout set that
    /// trained a checkpoint (spec §3, reproducibility).
    pub async fn checkout(&mut self, version_id: u64) -> LanceResult<()> {
        self.dataset = self.dataset.checkout_version(version_id).await?;
        Ok(())
    }

    /// Append rollout rows in one atomic write; returns the new dataset version.
    pub async fn add(&mut self, records: &[RolloutRecord]) -> LanceResult<u64> {
        if records.is_empty() {
            return Ok(self.dataset.manifest.version);
        }

        let batch = self.records_to_batch(records)?;
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(
            vec![Ok::<RecordBatch, ArrowError>(batch)].into_iter(),
            schema,
        );

        let mut params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        if let Some(options) = &self.storage_options {
            params.store_params = Some(ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(options.clone()),
                )),
                ..Default::default()
            });
        }

        self.dataset.append(batches, Some(params)).await?;
        Ok(self.dataset.manifest.version)
    }

    /// List rollout rows in storage order.
    pub async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> LanceResult<Vec<RolloutRecord>> {
        let scanner = self.dataset.scan();
        let mut stream = scanner.try_into_stream().await?;
        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            results.extend(batch_to_rollout_records(&batch)?);
        }

        if let Some(offset) = offset {
            results = results.into_iter().skip(offset).collect();
        }
        if let Some(limit) = limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    /// Retrieve a single rollout row by its unique id.
    pub async fn get_by_id(&self, id: &str) -> LanceResult<Option<RolloutRecord>> {
        let escaped_id = id.replace('\'', "''");
        let mut scanner = self.dataset.scan();
        scanner.filter(&format!("id = '{}'", escaped_id))?;
        scanner.limit(Some(1), None)?;

        let mut stream = scanner.try_into_stream().await?;
        if let Some(batch) = stream.try_next().await? {
            return Ok(batch_to_rollout_records(&batch)?.into_iter().next());
        }
        Ok(None)
    }

    /// Fetch a single artifact row's `binary_payload` bytes on demand.
    ///
    /// When `binary_payload` is offloaded via blob v2 (the default, see
    /// [`RolloutStore::open`]), a normal scan returns the column as a
    /// description and [`RolloutRecord::binary_payload`] reads back as `None`.
    /// This method forces byte materialization with
    /// [`BlobHandling::AllBinary`]. Returns `None` if the row or its payload is
    /// absent.
    pub async fn get_blob(&self, id: &str) -> LanceResult<Option<Vec<u8>>> {
        let escaped_id = id.replace('\'', "''");
        let mut scanner = self.dataset.scan();
        scanner.project(&["id", "binary_payload"])?;
        scanner.filter(&format!("id = '{}'", escaped_id))?;
        if self.blob_columns.contains("binary_payload") {
            scanner.blob_handling(BlobHandling::AllBinary);
        }

        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            let id_array = column_as::<StringArray>(&batch, "id")?;
            let binary_array = column_as_optional::<LargeBinaryArray>(&batch, "binary_payload");
            for row in 0..batch.num_rows() {
                if id_array.value(row) == id {
                    return Ok(match binary_array {
                        Some(arr) if !arr.is_null(row) => Some(arr.value(row).to_vec()),
                        _ => None,
                    });
                }
            }
        }
        Ok(None)
    }

    async fn load_with_options(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> LanceResult<Dataset> {
        if let Some(options) = storage_options {
            DatasetBuilder::from_uri(uri)
                .with_storage_options(options)
                .load()
                .await
        } else {
            Dataset::open(uri).await
        }
    }

    async fn create_with_options(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
        blob_columns: &HashSet<String>,
    ) -> LanceResult<Dataset> {
        let schema = Arc::new(rollout_schema(blob_columns));
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let batches = RecordBatchIterator::new(
            vec![Ok::<RecordBatch, ArrowError>(empty_batch)].into_iter(),
            schema.clone(),
        );

        let mut params = WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        };
        if let Some(options) = storage_options {
            params.store_params = Some(ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(options),
                )),
                ..Default::default()
            });
        }

        Dataset::write(batches, uri, Some(params)).await
    }

    fn records_to_batch(&self, records: &[RolloutRecord]) -> LanceResult<RecordBatch> {
        let field_paths = self.dataset.schema().field_paths();
        let has = |name: &str| field_paths.iter().any(|path| path == name);
        let include_relationships = has(RELATIONSHIPS_COLUMN);
        let include_metadata = has("metadata");

        if !include_relationships && records.iter().any(|r| !r.relationships.is_empty()) {
            return Err(ArrowError::InvalidArgumentError(
                "relationships require a rollout dataset created with relationships support"
                    .to_string(),
            )
            .into());
        }
        if !include_metadata && records.iter().any(|r| r.metadata.is_some()) {
            return Err(ArrowError::InvalidArgumentError(
                "metadata requires a rollout dataset created with metadata support".to_string(),
            )
            .into());
        }

        let mut id_builder = StringBuilder::new();
        let mut rollout_id_builder = StringBuilder::new();
        let mut problem_id_builder = StringBuilder::new();
        let mut dataset_builder = StringBuilder::new();
        let mut sequence_order_builder = Int32Builder::new();
        let mut role_builder = StringDictionaryBuilder::<Int8Type>::new();
        let mut created_at_builder = TimestampMicrosecondBuilder::with_capacity(records.len());
        let mut content_builder = LargeStringBuilder::new();
        let mut content_type_builder = StringBuilder::new();
        let mut input_tokens_builder = ListBuilder::new(Int32Builder::new());
        let mut output_tokens_builder = ListBuilder::new(Int32Builder::new());
        let mut num_input_tokens_builder = Int32Builder::new();
        let mut num_output_tokens_builder = Int32Builder::new();
        let mut output_logprobs_builder = ListBuilder::new(Float32Builder::new());
        let mut input_logprobs_builder = ListBuilder::new(Float32Builder::new());
        let mut ref_logprobs_builder = ListBuilder::new(Float32Builder::new());
        let mut loss_mask_builder = ListBuilder::new(Int8Builder::new());
        let mut advantage_builder = Float32Builder::new();
        let mut reward_builder = Float32Builder::new();
        let mut raw_reward_builder = Float32Builder::new();
        let mut grader_id_builder = StringBuilder::new();
        let mut score_builder = Float32Builder::new();
        let mut include_in_training_builder = BooleanBuilder::new();
        let mut exclude_reason_builder = StringBuilder::new();
        let mut policy_version_builder = StringBuilder::new();
        let mut relationships_builder = ListBuilder::new(relationship_struct_builder())
            .with_field(relationship_list_item_field());
        let mut binary_payload_builder = LargeBinaryBuilder::new();
        let mut payload_size_builder = Int64Builder::new();
        let mut payload_checksum_builder = StringBuilder::new();
        let mut metadata_builder = LargeStringBuilder::new();

        for record in records {
            id_builder.append_value(&record.id);
            rollout_id_builder.append_value(&record.rollout_id);
            problem_id_builder.append_value(&record.problem_id);
            dataset_builder.append_option(record.dataset.as_deref());
            sequence_order_builder.append_value(record.sequence_order);
            role_builder.append(&record.role)?;
            created_at_builder.append_value(record.created_at.timestamp_micros());
            content_builder.append_option(record.content.as_deref());
            content_type_builder.append_value(&record.content_type);
            append_i32_list(&mut input_tokens_builder, record.input_tokens.as_deref());
            append_i32_list(&mut output_tokens_builder, record.output_tokens.as_deref());
            num_input_tokens_builder.append_option(record.num_input_tokens);
            num_output_tokens_builder.append_option(record.num_output_tokens);
            append_f32_list(
                &mut output_logprobs_builder,
                record.output_logprobs.as_deref(),
            );
            append_f32_list(
                &mut input_logprobs_builder,
                record.input_logprobs.as_deref(),
            );
            append_f32_list(&mut ref_logprobs_builder, record.ref_logprobs.as_deref());
            append_i8_list(&mut loss_mask_builder, record.loss_mask.as_deref());
            advantage_builder.append_option(record.advantage);
            reward_builder.append_option(record.reward);
            raw_reward_builder.append_option(record.raw_reward);
            grader_id_builder.append_option(record.grader_id.as_deref());
            score_builder.append_option(record.score);
            include_in_training_builder.append_option(record.include_in_training);
            exclude_reason_builder.append_option(record.exclude_reason.as_deref());
            policy_version_builder.append_option(record.policy_version.as_deref());

            for relationship in &record.relationships {
                let values_builder = relationships_builder.values();
                values_builder
                    .field_builder::<StringBuilder>(0)
                    .unwrap()
                    .append_value(&relationship.target_id);
                values_builder
                    .field_builder::<StringBuilder>(1)
                    .unwrap()
                    .append_value(&relationship.relation);
                values_builder
                    .field_builder::<Float32Builder>(2)
                    .unwrap()
                    .append_option(relationship.weight);
                values_builder.append(true);
            }
            relationships_builder.append(true);

            match &record.binary_payload {
                Some(bytes) => binary_payload_builder.append_value(bytes),
                None => binary_payload_builder.append_null(),
            }
            payload_size_builder.append_option(record.payload_size);
            payload_checksum_builder.append_option(record.payload_checksum.as_deref());
            match &record.metadata {
                Some(metadata) => metadata_builder.append_value(metadata.to_string()),
                None => metadata_builder.append_null(),
            }
        }

        let mut arrays_by_name: HashMap<String, ArrayRef> = HashMap::new();
        arrays_by_name.insert("id".to_string(), Arc::new(id_builder.finish()));
        arrays_by_name.insert(
            "rollout_id".to_string(),
            Arc::new(rollout_id_builder.finish()),
        );
        arrays_by_name.insert(
            "problem_id".to_string(),
            Arc::new(problem_id_builder.finish()),
        );
        arrays_by_name.insert("dataset".to_string(), Arc::new(dataset_builder.finish()));
        arrays_by_name.insert(
            "sequence_order".to_string(),
            Arc::new(sequence_order_builder.finish()),
        );
        arrays_by_name.insert("role".to_string(), Arc::new(role_builder.finish()));
        arrays_by_name.insert(
            "created_at".to_string(),
            Arc::new(created_at_builder.finish()),
        );
        arrays_by_name.insert("content".to_string(), Arc::new(content_builder.finish()));
        arrays_by_name.insert(
            "content_type".to_string(),
            Arc::new(content_type_builder.finish()),
        );
        arrays_by_name.insert(
            "input_tokens".to_string(),
            Arc::new(input_tokens_builder.finish()),
        );
        arrays_by_name.insert(
            "output_tokens".to_string(),
            Arc::new(output_tokens_builder.finish()),
        );
        arrays_by_name.insert(
            "num_input_tokens".to_string(),
            Arc::new(num_input_tokens_builder.finish()),
        );
        arrays_by_name.insert(
            "num_output_tokens".to_string(),
            Arc::new(num_output_tokens_builder.finish()),
        );
        arrays_by_name.insert(
            "output_logprobs".to_string(),
            Arc::new(output_logprobs_builder.finish()),
        );
        arrays_by_name.insert(
            "input_logprobs".to_string(),
            Arc::new(input_logprobs_builder.finish()),
        );
        arrays_by_name.insert(
            "ref_logprobs".to_string(),
            Arc::new(ref_logprobs_builder.finish()),
        );
        arrays_by_name.insert(
            "loss_mask".to_string(),
            Arc::new(loss_mask_builder.finish()),
        );
        arrays_by_name.insert(
            "advantage".to_string(),
            Arc::new(advantage_builder.finish()),
        );
        arrays_by_name.insert("reward".to_string(), Arc::new(reward_builder.finish()));
        arrays_by_name.insert(
            "raw_reward".to_string(),
            Arc::new(raw_reward_builder.finish()),
        );
        arrays_by_name.insert(
            "grader_id".to_string(),
            Arc::new(grader_id_builder.finish()),
        );
        arrays_by_name.insert("score".to_string(), Arc::new(score_builder.finish()));
        arrays_by_name.insert(
            "include_in_training".to_string(),
            Arc::new(include_in_training_builder.finish()),
        );
        arrays_by_name.insert(
            "exclude_reason".to_string(),
            Arc::new(exclude_reason_builder.finish()),
        );
        arrays_by_name.insert(
            "policy_version".to_string(),
            Arc::new(policy_version_builder.finish()),
        );
        if include_relationships {
            arrays_by_name.insert(
                RELATIONSHIPS_COLUMN.to_string(),
                Arc::new(relationships_builder.finish()),
            );
        }
        arrays_by_name.insert(
            "binary_payload".to_string(),
            Arc::new(binary_payload_builder.finish()),
        );
        arrays_by_name.insert(
            "payload_size".to_string(),
            Arc::new(payload_size_builder.finish()),
        );
        arrays_by_name.insert(
            "payload_checksum".to_string(),
            Arc::new(payload_checksum_builder.finish()),
        );
        if include_metadata {
            arrays_by_name.insert("metadata".to_string(), Arc::new(metadata_builder.finish()));
        }

        let schema: Arc<Schema> = Arc::new(self.dataset.schema().into());
        let arrays = schema
            .fields()
            .iter()
            .map(|field| {
                arrays_by_name.remove(field.name().as_str()).ok_or_else(|| {
                    LanceError::from(ArrowError::InvalidArgumentError(format!(
                        "unsupported rollout dataset column '{}'",
                        field.name()
                    )))
                })
            })
            .collect::<LanceResult<Vec<_>>>()?;

        Ok(RecordBatch::try_new(schema, arrays)?)
    }
}

/// Arrow schema for a rollout dataset (spec §5). `binary_payload` is marked as a
/// blob v2 column when present in `blob_columns`, so its bytes sink to an
/// independent file and column scans skip them.
#[must_use]
pub fn rollout_schema(blob_columns: &HashSet<String>) -> Schema {
    let mut id_metadata = HashMap::new();
    id_metadata.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );

    let binary_field = if blob_columns.contains("binary_payload") {
        let mut metadata = HashMap::new();
        metadata.insert("lance-encoding:blob".to_string(), "true".to_string());
        Field::new("binary_payload", DataType::LargeBinary, true).with_metadata(metadata)
    } else {
        Field::new("binary_payload", DataType::LargeBinary, true)
    };

    let fields = vec![
        // Identity & grouping.
        Field::new("id", DataType::Utf8, false).with_metadata(id_metadata),
        Field::new("rollout_id", DataType::Utf8, false),
        Field::new("problem_id", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, true),
        Field::new("sequence_order", DataType::Int32, false),
        Field::new(
            "role",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        // Message content.
        Field::new("content", DataType::LargeUtf8, true),
        Field::new("content_type", DataType::Utf8, false),
        // Tokens.
        list_field("input_tokens", DataType::Int32),
        list_field("output_tokens", DataType::Int32),
        Field::new("num_input_tokens", DataType::Int32, true),
        Field::new("num_output_tokens", DataType::Int32, true),
        // Training signals.
        list_field("output_logprobs", DataType::Float32),
        list_field("input_logprobs", DataType::Float32),
        list_field("ref_logprobs", DataType::Float32),
        list_field("loss_mask", DataType::Int8),
        Field::new("advantage", DataType::Float32, true),
        // Reward.
        Field::new("reward", DataType::Float32, true),
        Field::new("raw_reward", DataType::Float32, true),
        Field::new("grader_id", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
        // Training control & provenance.
        Field::new("include_in_training", DataType::Boolean, true),
        Field::new("exclude_reason", DataType::Utf8, true),
        Field::new("policy_version", DataType::Utf8, true),
        // Graph, artifacts, escape hatch.
        relationship_field(),
        binary_field,
        Field::new("payload_size", DataType::Int64, true),
        Field::new("payload_checksum", DataType::Utf8, true),
        Field::new("metadata", DataType::LargeUtf8, true),
    ];

    Schema::new(fields)
}

/// A nullable `List<item>` field carrying a nullable primitive child.
fn list_field(name: &str, item_type: DataType) -> Field {
    Field::new(
        name,
        DataType::List(Arc::new(Field::new("item", item_type, true))),
        true,
    )
}

fn append_i32_list(builder: &mut ListBuilder<Int32Builder>, values: Option<&[i32]>) {
    match values {
        Some(values) => {
            let child = builder.values();
            for value in values {
                child.append_value(*value);
            }
            builder.append(true);
        }
        None => builder.append(false),
    }
}

fn append_f32_list(builder: &mut ListBuilder<Float32Builder>, values: Option<&[f32]>) {
    match values {
        Some(values) => {
            let child = builder.values();
            for value in values {
                child.append_value(*value);
            }
            builder.append(true);
        }
        None => builder.append(false),
    }
}

fn append_i8_list(builder: &mut ListBuilder<Int8Builder>, values: Option<&[i8]>) {
    match values {
        Some(values) => {
            let child = builder.values();
            for value in values {
                child.append_value(*value);
            }
            builder.append(true);
        }
        None => builder.append(false),
    }
}

fn batch_to_rollout_records(batch: &RecordBatch) -> LanceResult<Vec<RolloutRecord>> {
    let id_array = column_as::<StringArray>(batch, "id")?;
    let rollout_id_array = column_as::<StringArray>(batch, "rollout_id")?;
    let problem_id_array = column_as::<StringArray>(batch, "problem_id")?;
    let dataset_array = column_as_optional::<StringArray>(batch, "dataset");
    let sequence_order_array = column_as::<Int32Array>(batch, "sequence_order")?;
    let role_array = column_as::<DictionaryArray<Int8Type>>(batch, "role")?;
    let created_at_array = column_as::<TimestampMicrosecondArray>(batch, "created_at")?;
    let content_array = column_as_optional::<LargeStringArray>(batch, "content");
    let content_type_array = column_as::<StringArray>(batch, "content_type")?;
    let input_tokens_array = column_as_optional::<ListArray>(batch, "input_tokens");
    let output_tokens_array = column_as_optional::<ListArray>(batch, "output_tokens");
    let num_input_tokens_array = column_as_optional::<Int32Array>(batch, "num_input_tokens");
    let num_output_tokens_array = column_as_optional::<Int32Array>(batch, "num_output_tokens");
    let output_logprobs_array = column_as_optional::<ListArray>(batch, "output_logprobs");
    let input_logprobs_array = column_as_optional::<ListArray>(batch, "input_logprobs");
    let ref_logprobs_array = column_as_optional::<ListArray>(batch, "ref_logprobs");
    let loss_mask_array = column_as_optional::<ListArray>(batch, "loss_mask");
    let advantage_array = column_as_optional::<Float32Array>(batch, "advantage");
    let reward_array = column_as_optional::<Float32Array>(batch, "reward");
    let raw_reward_array = column_as_optional::<Float32Array>(batch, "raw_reward");
    let grader_id_array = column_as_optional::<StringArray>(batch, "grader_id");
    let score_array = column_as_optional::<Float32Array>(batch, "score");
    let include_in_training_array =
        column_as_optional::<BooleanArray>(batch, "include_in_training");
    let exclude_reason_array = column_as_optional::<StringArray>(batch, "exclude_reason");
    let policy_version_array = column_as_optional::<StringArray>(batch, "policy_version");
    let relationships_array = column_as_optional::<ListArray>(batch, RELATIONSHIPS_COLUMN);
    let binary_payload_array = column_as_optional::<LargeBinaryArray>(batch, "binary_payload");
    let payload_size_array = column_as_optional::<Int64Array>(batch, "payload_size");
    let payload_checksum_array = column_as_optional::<StringArray>(batch, "payload_checksum");
    let metadata_array = column_as_optional::<LargeStringArray>(batch, "metadata");

    let mut results = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let created_at = timestamp_from_micros(created_at_array.value(row), "created_at")?;

        let role = {
            let values = role_array
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    LanceError::from(ArrowError::InvalidArgumentError(
                        "role dictionary values are not strings".to_string(),
                    ))
                })?;
            if role_array.is_null(row) {
                return Err(LanceError::from(ArrowError::InvalidArgumentError(
                    "role column contains null values".to_string(),
                )));
            }
            values
                .value(role_array.keys().value(row) as usize)
                .to_string()
        };

        let metadata = match metadata_array {
            Some(arr) if !arr.is_null(row) => {
                Some(serde_json::from_str(arr.value(row)).map_err(|err| {
                    LanceError::from(ArrowError::InvalidArgumentError(format!(
                        "invalid metadata JSON for rollout row {}: {}",
                        id_array.value(row),
                        err
                    )))
                })?)
            }
            _ => None,
        };
        let relationships = match relationships_array {
            Some(arr) if !arr.is_null(row) => relationships_from_list(arr, row)?,
            _ => Vec::new(),
        };

        results.push(RolloutRecord {
            id: id_array.value(row).to_string(),
            rollout_id: rollout_id_array.value(row).to_string(),
            problem_id: problem_id_array.value(row).to_string(),
            dataset: optional_string(dataset_array, row),
            sequence_order: sequence_order_array.value(row),
            role,
            created_at,
            content: optional_large_string(content_array, row),
            content_type: content_type_array.value(row).to_string(),
            input_tokens: optional_i32_list(input_tokens_array, row)?,
            output_tokens: optional_i32_list(output_tokens_array, row)?,
            num_input_tokens: optional_i32(num_input_tokens_array, row),
            num_output_tokens: optional_i32(num_output_tokens_array, row),
            output_logprobs: optional_f32_list(output_logprobs_array, row)?,
            input_logprobs: optional_f32_list(input_logprobs_array, row)?,
            ref_logprobs: optional_f32_list(ref_logprobs_array, row)?,
            loss_mask: optional_i8_list(loss_mask_array, row)?,
            advantage: optional_f32(advantage_array, row),
            reward: optional_f32(reward_array, row),
            raw_reward: optional_f32(raw_reward_array, row),
            grader_id: optional_string(grader_id_array, row),
            score: optional_f32(score_array, row),
            include_in_training: include_in_training_array.and_then(|arr| {
                if arr.is_null(row) {
                    None
                } else {
                    Some(arr.value(row))
                }
            }),
            exclude_reason: optional_string(exclude_reason_array, row),
            policy_version: optional_string(policy_version_array, row),
            relationships,
            binary_payload: match binary_payload_array {
                Some(arr) if !arr.is_null(row) => Some(arr.value(row).to_vec()),
                _ => None,
            },
            payload_size: payload_size_array.and_then(|arr| {
                if arr.is_null(row) {
                    None
                } else {
                    Some(arr.value(row))
                }
            }),
            payload_checksum: optional_string(payload_checksum_array, row),
            metadata,
        });
    }

    Ok(results)
}

fn optional_string(array: Option<&StringArray>, row: usize) -> Option<String> {
    array.and_then(|arr| {
        if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row).to_string())
        }
    })
}

fn optional_large_string(array: Option<&LargeStringArray>, row: usize) -> Option<String> {
    array.and_then(|arr| {
        if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row).to_string())
        }
    })
}

fn optional_i32(array: Option<&Int32Array>, row: usize) -> Option<i32> {
    array.and_then(|arr| {
        if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row))
        }
    })
}

fn optional_f32(array: Option<&Float32Array>, row: usize) -> Option<f32> {
    array.and_then(|arr| {
        if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row))
        }
    })
}

fn optional_i32_list(array: Option<&ListArray>, row: usize) -> LanceResult<Option<Vec<i32>>> {
    match array {
        Some(arr) if !arr.is_null(row) => {
            let values = arr.value(row);
            let typed = values
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    LanceError::from(ArrowError::InvalidArgumentError(
                        "token list column does not contain int32 values".to_string(),
                    ))
                })?;
            Ok(Some((0..typed.len()).map(|i| typed.value(i)).collect()))
        }
        _ => Ok(None),
    }
}

fn optional_f32_list(array: Option<&ListArray>, row: usize) -> LanceResult<Option<Vec<f32>>> {
    match array {
        Some(arr) if !arr.is_null(row) => {
            let values = arr.value(row);
            let typed = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    LanceError::from(ArrowError::InvalidArgumentError(
                        "logprob list column does not contain float32 values".to_string(),
                    ))
                })?;
            Ok(Some((0..typed.len()).map(|i| typed.value(i)).collect()))
        }
        _ => Ok(None),
    }
}

fn optional_i8_list(array: Option<&ListArray>, row: usize) -> LanceResult<Option<Vec<i8>>> {
    match array {
        Some(arr) if !arr.is_null(row) => {
            let values = arr.value(row);
            let typed = values.as_any().downcast_ref::<Int8Array>().ok_or_else(|| {
                LanceError::from(ArrowError::InvalidArgumentError(
                    "loss_mask list column does not contain int8 values".to_string(),
                ))
            })?;
            Ok(Some((0..typed.len()).map(|i| typed.value(i)).collect()))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Relationship;
    use crate::rollout::{ROLE_ARTIFACT, ROLE_ASSISTANT};
    use chrono::{TimeZone, Utc};
    use serde_json::json;
    use tempfile::TempDir;

    fn assistant_record(id: &str) -> RolloutRecord {
        RolloutRecord {
            id: id.to_string(),
            rollout_id: "rollout-1".to_string(),
            problem_id: "problem-1".to_string(),
            dataset: Some("gsm8k".to_string()),
            sequence_order: 0,
            role: ROLE_ASSISTANT.to_string(),
            created_at: Utc.timestamp_micros(1_700_000_000_000_000).unwrap(),
            content: Some("the answer is 42".to_string()),
            content_type: "text/plain".to_string(),
            input_tokens: Some(vec![10, 11, 12]),
            output_tokens: Some(vec![20, 21]),
            num_input_tokens: Some(3),
            num_output_tokens: Some(2),
            output_logprobs: Some(vec![-0.5, -1.25]),
            input_logprobs: None,
            ref_logprobs: Some(vec![-0.4, -1.1]),
            loss_mask: Some(vec![1, 0]),
            advantage: Some(0.75),
            reward: Some(1.0),
            raw_reward: Some(0.9),
            grader_id: Some("grader-a".to_string()),
            score: Some(0.95),
            include_in_training: Some(true),
            exclude_reason: None,
            policy_version: Some("ckpt-42".to_string()),
            relationships: vec![Relationship {
                target_id: "problem-1".to_string(),
                relation: "derived_from".to_string(),
                weight: Some(1.0),
            }],
            binary_payload: None,
            payload_size: None,
            payload_checksum: None,
            metadata: Some(json!({"harness": "verifiers"})),
        }
    }

    fn artifact_record(id: &str, bytes: &[u8]) -> RolloutRecord {
        RolloutRecord {
            id: id.to_string(),
            rollout_id: "rollout-1".to_string(),
            problem_id: "problem-1".to_string(),
            dataset: None,
            sequence_order: 1,
            role: ROLE_ARTIFACT.to_string(),
            created_at: Utc.timestamp_micros(1_700_000_000_500_000).unwrap(),
            content: None,
            content_type: "application/octet-stream".to_string(),
            input_tokens: None,
            output_tokens: None,
            num_input_tokens: None,
            num_output_tokens: None,
            output_logprobs: None,
            input_logprobs: None,
            ref_logprobs: None,
            loss_mask: None,
            advantage: None,
            reward: None,
            raw_reward: None,
            grader_id: None,
            score: None,
            include_in_training: None,
            exclude_reason: None,
            policy_version: None,
            relationships: Vec::new(),
            binary_payload: Some(bytes.to_vec()),
            payload_size: Some(bytes.len() as i64),
            payload_checksum: Some("sha256:cafef00d".to_string()),
            metadata: Some(json!({"filename": "trace.bin", "artifact_type": "trace"})),
        }
    }

    fn assert_records_eq(actual: &RolloutRecord, expected: &RolloutRecord) {
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.rollout_id, expected.rollout_id);
        assert_eq!(actual.problem_id, expected.problem_id);
        assert_eq!(actual.dataset, expected.dataset);
        assert_eq!(actual.sequence_order, expected.sequence_order);
        assert_eq!(actual.role, expected.role);
        assert_eq!(actual.created_at, expected.created_at);
        assert_eq!(actual.content, expected.content);
        assert_eq!(actual.content_type, expected.content_type);
        assert_eq!(actual.input_tokens, expected.input_tokens);
        assert_eq!(actual.output_tokens, expected.output_tokens);
        assert_eq!(actual.num_input_tokens, expected.num_input_tokens);
        assert_eq!(actual.num_output_tokens, expected.num_output_tokens);
        assert_eq!(actual.output_logprobs, expected.output_logprobs);
        assert_eq!(actual.input_logprobs, expected.input_logprobs);
        assert_eq!(actual.ref_logprobs, expected.ref_logprobs);
        assert_eq!(actual.loss_mask, expected.loss_mask);
        assert_eq!(actual.advantage, expected.advantage);
        assert_eq!(actual.reward, expected.reward);
        assert_eq!(actual.raw_reward, expected.raw_reward);
        assert_eq!(actual.grader_id, expected.grader_id);
        assert_eq!(actual.score, expected.score);
        assert_eq!(actual.include_in_training, expected.include_in_training);
        assert_eq!(actual.exclude_reason, expected.exclude_reason);
        assert_eq!(actual.policy_version, expected.policy_version);
        assert_eq!(actual.relationships.len(), expected.relationships.len());
        for (a, e) in actual.relationships.iter().zip(&expected.relationships) {
            assert_eq!(a.target_id, e.target_id);
            assert_eq!(a.relation, e.relation);
            assert_eq!(a.weight, e.weight);
        }
        // `binary_payload` is offloaded via blob v2, so `list`/`get_by_id`
        // scans read it back as `None` (byte materialization is verified
        // separately through `get_blob`). The inline sidecar columns still
        // round-trip.
        assert_eq!(actual.payload_size, expected.payload_size);
        assert_eq!(actual.payload_checksum, expected.payload_checksum);
        assert_eq!(actual.metadata, expected.metadata);
    }

    #[test]
    fn append_list_and_fetch_roundtrip() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let assistant = assistant_record("row-0");
        let artifact_bytes = b"\x00\x01\x02trace-bytes";
        let artifact = artifact_record("row-1", artifact_bytes);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            let version = store
                .add(&[assistant.clone(), artifact.clone()])
                .await
                .unwrap();
            assert!(version > 1, "append should advance the dataset version");

            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 2);
            assert_records_eq(&listed[0], &assistant);
            assert_records_eq(&listed[1], &artifact);
            // Offloaded blob column is not materialized by a plain scan.
            assert_eq!(listed[1].binary_payload, None);

            let fetched = store.get_by_id("row-1").await.unwrap().unwrap();
            assert_records_eq(&fetched, &artifact);
            assert!(fetched.is_artifact());

            // The bytes are recoverable on demand from the sidecar file.
            let blob = store.get_blob("row-1").await.unwrap();
            assert_eq!(blob.as_deref(), Some(&artifact_bytes[..]));
            // The assistant row carries no payload.
            assert_eq!(store.get_blob("row-0").await.unwrap(), None);

            assert!(store.get_by_id("missing").await.unwrap().is_none());
            assert_eq!(store.get_blob("missing").await.unwrap(), None);
        });
    }

    #[test]
    fn checkout_recovers_earlier_version() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();

        let artifact_bytes = b"\x00\x01\x02checkpoint-trace";
        let artifact = artifact_record("row-0", artifact_bytes);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            let first_version = store.add(&[artifact]).await.unwrap();
            store.add(&[assistant_record("row-1")]).await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 2);

            store.checkout(first_version).await.unwrap();
            let recovered = store.list(None, None).await.unwrap();
            assert_eq!(recovered.len(), 1);
            assert_eq!(recovered[0].id, "row-0");

            // Blob offload survives version checkout: the artifact bytes
            // written in the earlier version are still materializable after
            // rewinding the dataset to it.
            let blob = store.get_blob("row-0").await.unwrap();
            assert_eq!(blob.as_deref(), Some(&artifact_bytes[..]));
        });
    }
}
