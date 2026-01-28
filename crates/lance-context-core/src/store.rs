use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{
    FixedSizeListBuilder, Float32Builder, Int32Builder, LargeBinaryBuilder, LargeStringBuilder,
    StringBuilder, StringDictionaryBuilder, StructBuilder, TimestampMicrosecondBuilder,
};
use arrow_array::types::Int8Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float32Array, Int32Array,
    LargeBinaryArray, LargeStringArray, RecordBatch, RecordBatchIterator, StringArray, StructArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema, TimeUnit};
use chrono::{DateTime, Timelike, Utc};
use futures::TryStreamExt;
use lance::dataset::optimize::{compact_files, CompactionMetrics, CompactionOptions};
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::io::ObjectStoreParams;
use lance::{Error as LanceError, Result as LanceResult};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::record::{ContextRecord, SearchResult, StateMetadata};

/// Embedding length used for the semantic index column.
const DEFAULT_EMBEDDING_DIM: i32 = 1536;
const DEFAULT_SEARCH_LIMIT: usize = 10;

/// Configuration for background compaction.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Whether background compaction is enabled.
    pub enabled: bool,
    /// Minimum number of fragments to trigger compaction.
    pub min_fragments: usize,
    /// Target rows per fragment after compaction.
    pub target_rows_per_fragment: usize,
    /// Maximum rows per row group.
    pub max_rows_per_group: usize,
    /// Whether to materialize (remove) deleted rows during compaction.
    pub materialize_deletions: bool,
    /// Deletion threshold (0.0-1.0) to trigger materialization.
    pub materialize_deletions_threshold: f32,
    /// Number of threads for compaction (None = auto).
    pub num_threads: Option<usize>,
    /// Interval in seconds between compaction checks.
    pub check_interval_secs: u64,
    /// Quiet hours during which compaction is skipped [(start_hour, end_hour)].
    pub quiet_hours: Vec<(u8, u8)>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_fragments: 5,
            target_rows_per_fragment: 1_000_000,
            max_rows_per_group: 1024,
            materialize_deletions: true,
            materialize_deletions_threshold: 0.1,
            num_threads: None,
            check_interval_secs: 300,
            quiet_hours: vec![],
        }
    }
}

/// Statistics about compaction status and history.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Current number of fragments in the dataset.
    pub total_fragments: usize,
    /// Whether a compaction is currently in progress.
    pub is_compacting: bool,
    /// Timestamp of the last successful compaction.
    pub last_compaction: Option<DateTime<Utc>>,
    /// Error message from the last failed compaction.
    pub last_error: Option<String>,
    /// Total number of successful compactions performed.
    pub total_compactions: u64,
}

/// Internal state for tracking background compaction.
struct CompactionState {
    background_task: Option<JoinHandle<()>>,
    is_compacting: bool,
    last_compaction: Option<DateTime<Utc>>,
    last_error: Option<String>,
    total_compactions: u64,
}

/// Persistent Lance-backed context store.
#[derive(Clone)]
pub struct ContextStore {
    dataset: Dataset,
    compaction_state: Arc<Mutex<CompactionState>>,
    pub compaction_config: CompactionConfig,
}

/// Additional configuration when opening a [`ContextStore`].
#[derive(Debug, Clone, Default)]
pub struct ContextStoreOptions {
    pub storage_options: Option<HashMap<String, String>>,
    pub compaction: CompactionConfig,
}

impl ContextStoreOptions {
    #[must_use]
    pub fn storage_options(&self) -> Option<HashMap<String, String>> {
        self.storage_options.clone()
    }
}

impl ContextStore {
    /// Open an existing context dataset or create a new one with the project schema.
    pub async fn open(uri: &str) -> LanceResult<Self> {
        Self::open_with_options(uri, ContextStoreOptions::default()).await
    }

    /// Open a dataset with explicit object store configuration (e.g. S3 credentials).
    pub async fn open_with_options(uri: &str, options: ContextStoreOptions) -> LanceResult<Self> {
        let storage_options = options.storage_options();
        let dataset = match Self::load_with_options(uri, storage_options.clone()).await {
            Ok(dataset) => dataset,
            Err(LanceError::DatasetNotFound { .. }) => {
                Self::create_with_options(uri, storage_options).await?
            }
            Err(err) => return Err(err),
        };

        let mut store = Self {
            dataset,
            compaction_state: Arc::new(Mutex::new(CompactionState {
                background_task: None,
                is_compacting: false,
                last_compaction: None,
                last_error: None,
                total_compactions: 0,
            })),
            compaction_config: options.compaction,
        };

        // Start background compaction if enabled
        store.start_background_compaction().await?;

        Ok(store)
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

    /// Current dataset version.
    pub fn version(&self) -> u64 {
        self.dataset.manifest.version
    }

    /// Checkout a specific dataset version.
    pub async fn checkout(&mut self, version_id: u64) -> LanceResult<()> {
        let dataset = self.dataset.checkout_version(version_id).await?;
        self.dataset = dataset;
        Ok(())
    }

    /// List all records in the dataset.
    pub async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> LanceResult<Vec<ContextRecord>> {
        let mut scanner = self.dataset.scan();
        if let Some(limit) = limit {
            scanner.limit(Some(limit as i64), offset.map(|o| o as i64))?;
        } else if let Some(offset) = offset {
            scanner.limit(None, Some(offset as i64))?;
        }

        let mut stream = scanner.try_into_stream().await?;
        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            results.extend(batch_to_records(&batch)?);
        }
        Ok(results)
    }

    /// Perform a nearest-neighbor search over stored embeddings.
    pub async fn search(
        &self,
        query: &[f32],
        limit: Option<usize>,
    ) -> LanceResult<Vec<SearchResult>> {
        if query.len() != DEFAULT_EMBEDDING_DIM as usize {
            return Err(ArrowError::InvalidArgumentError(format!(
                "query length {} does not match embedding dimension {}",
                query.len(),
                DEFAULT_EMBEDDING_DIM
            ))
            .into());
        }

        let top_k = limit.unwrap_or(DEFAULT_SEARCH_LIMIT);
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let query_array = Float32Array::from(query.to_vec());

        let mut scanner = self.dataset.scan();
        scanner.nearest("embedding", &query_array, top_k)?;
        scanner.limit(Some(top_k as i64), Some(0))?;

        let mut stream = scanner.try_into_stream().await?;
        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            results.extend(batch_to_search_results(&batch)?);
        }
        Ok(results)
    }

    /// Manually trigger compaction to merge small fragments.
    pub async fn compact(
        &mut self,
        options: Option<CompactionConfig>,
    ) -> LanceResult<CompactionMetrics> {
        let config = options.unwrap_or_else(|| self.compaction_config.clone());

        info!(
            "Starting compaction: {} fragments",
            self.dataset.count_fragments()
        );
        let start = std::time::Instant::now();

        // Mark as compacting
        {
            let mut state = self.compaction_state.lock().await;
            if state.is_compacting {
                warn!("Compaction already in progress, skipping");
                return Err(LanceError::from(ArrowError::InvalidArgumentError(
                    "Compaction already in progress".to_string(),
                )));
            }
            state.is_compacting = true;
        }

        // Build Lance CompactionOptions
        let lance_options = CompactionOptions {
            target_rows_per_fragment: config.target_rows_per_fragment,
            max_rows_per_group: config.max_rows_per_group,
            materialize_deletions: config.materialize_deletions,
            materialize_deletions_threshold: config.materialize_deletions_threshold,
            num_threads: config.num_threads,
            ..Default::default()
        };

        // Run compaction
        let result = compact_files(&mut self.dataset, lance_options, None).await;

        // Update state
        let mut state = self.compaction_state.lock().await;
        state.is_compacting = false;

        match result {
            Ok(metrics) => {
                state.last_compaction = Some(Utc::now());
                state.total_compactions += 1;
                state.last_error = None;

                info!(
                    "Compaction completed in {:?}: removed {} fragments ({}files), added {} fragments ({} files)",
                    start.elapsed(),
                    metrics.fragments_removed,
                    metrics.files_removed,
                    metrics.fragments_added,
                    metrics.files_added
                );

                // Reload dataset to see new version
                self.dataset = Dataset::open(self.dataset.uri()).await?;

                Ok(metrics)
            }
            Err(e) => {
                error!("Compaction failed: {}", e);
                state.last_error = Some(e.to_string());
                Err(e)
            }
        }
    }

    /// Check if compaction should run based on configuration thresholds.
    pub async fn should_compact(&self) -> LanceResult<bool> {
        let fragment_count = self.dataset.count_fragments();

        if fragment_count < self.compaction_config.min_fragments {
            return Ok(false);
        }

        // Check quiet hours
        if !self.compaction_config.quiet_hours.is_empty() {
            let now = Utc::now();
            let current_hour = now.hour() as u8;

            for (start, end) in &self.compaction_config.quiet_hours {
                if current_hour >= *start && current_hour < *end {
                    info!("Skipping compaction during quiet hours ({}-{})", start, end);
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Get current compaction statistics.
    pub async fn compaction_stats(&self) -> LanceResult<CompactionStats> {
        let state = self.compaction_state.lock().await;

        Ok(CompactionStats {
            total_fragments: self.dataset.count_fragments(),
            is_compacting: state.is_compacting,
            last_compaction: state.last_compaction,
            last_error: state.last_error.clone(),
            total_compactions: state.total_compactions,
        })
    }

    /// Start background compaction task if enabled.
    async fn start_background_compaction(&mut self) -> LanceResult<()> {
        if !self.compaction_config.enabled {
            return Ok(());
        }

        let mut state = self.compaction_state.lock().await;
        if state.background_task.is_some() {
            warn!("Background compaction already running");
            return Ok(());
        }

        info!(
            "Starting background compaction (interval: {}s, min fragments: {})",
            self.compaction_config.check_interval_secs, self.compaction_config.min_fragments
        );

        let mut store_clone = self.clone();
        let interval_secs = self.compaction_config.check_interval_secs;

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

            loop {
                interval.tick().await;

                match store_clone.should_compact().await {
                    Ok(true) => {
                        info!("Background compaction triggered");
                        if let Err(e) = store_clone.compact(None).await {
                            error!("Background compaction failed: {}", e);
                        }
                    }
                    Ok(false) => {
                        // Not needed or in quiet hours
                    }
                    Err(e) => {
                        error!("Error checking compaction need: {}", e);
                    }
                }
            }
        });

        state.background_task = Some(task);
        Ok(())
    }

    /// Stop background compaction task.
    pub async fn stop_background_compaction(&mut self) -> LanceResult<()> {
        let mut state = self.compaction_state.lock().await;

        if let Some(task) = state.background_task.take() {
            info!("Stopping background compaction");
            task.abort();
        }

        Ok(())
    }

    /// Lance schema for the context store.
    pub fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("run_id", DataType::Utf8, false),
            Field::new("bot_id", DataType::Utf8, true),
            Field::new("session_id", DataType::Utf8, true),
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
    ) -> LanceResult<Dataset> {
        let schema = Arc::new(Self::schema());
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
            let store_params = ObjectStoreParams {
                storage_options: Some(options),
                ..Default::default()
            };
            params.store_params = Some(store_params);
        }

        Dataset::write(batches, uri, Some(params)).await
    }

    fn records_to_batch(entries: &[ContextRecord]) -> LanceResult<RecordBatch> {
        let mut id_builder = StringBuilder::new();
        let mut run_id_builder = StringBuilder::new();
        let mut bot_id_builder = StringBuilder::new();
        let mut session_id_builder = StringBuilder::new();
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
            bot_id_builder.append_option(entry.bot_id.as_deref());
            session_id_builder.append_option(entry.session_id.as_deref());
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
        let bot_id_array: ArrayRef = Arc::new(bot_id_builder.finish());
        let session_id_array: ArrayRef = Arc::new(session_id_builder.finish());
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
                bot_id_array,
                session_id_array,
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

impl Drop for ContextStore {
    fn drop(&mut self) {
        // Best-effort cleanup of background task
        if let Ok(mut state) = self.compaction_state.try_lock() {
            if let Some(task) = state.background_task.take() {
                task.abort();
            }
        }
    }
}

fn batch_to_search_results(batch: &RecordBatch) -> LanceResult<Vec<SearchResult>> {
    let records = batch_to_records(batch)?;

    let distance_column = batch.column_by_name("_distance").ok_or_else(|| {
        LanceError::from(ArrowError::InvalidArgumentError(
            "search results missing _distance column".to_string(),
        ))
    })?;
    let distance_array = distance_column
        .as_ref()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "_distance column has unexpected data type".to_string(),
            ))
        })?;

    Ok(records
        .into_iter()
        .enumerate()
        .map(|(i, record)| SearchResult {
            record,
            distance: distance_array.value(i),
        })
        .collect())
}

/// Convert a record batch to context records.
fn batch_to_records(batch: &RecordBatch) -> LanceResult<Vec<ContextRecord>> {
    let id_array = column_as::<StringArray>(batch, "id")?;
    let run_id_array = column_as::<StringArray>(batch, "run_id")?;
    let bot_id_array = column_as_optional::<StringArray>(batch, "bot_id");
    let session_id_array = column_as_optional::<StringArray>(batch, "session_id");
    let created_at_array = column_as::<TimestampMicrosecondArray>(batch, "created_at")?;
    let role_array = column_as::<DictionaryArray<Int8Type>>(batch, "role")?;
    let state_array = column_as::<StructArray>(batch, "state_metadata")?;
    let content_type_array = column_as::<StringArray>(batch, "content_type")?;
    let text_array = column_as::<LargeStringArray>(batch, "text_payload")?;
    let binary_array = column_as::<LargeBinaryArray>(batch, "binary_payload")?;
    let embedding_array = column_as::<FixedSizeListArray>(batch, "embedding")?;

    let step_array = state_array
        .column(0)
        .as_ref()
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "step column has unexpected data type".to_string(),
            ))
        })?;
    let active_plan_array = state_array
        .column(1)
        .as_ref()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "active_plan_id column has unexpected data type".to_string(),
            ))
        })?;
    let tokens_used_array = state_array
        .column(2)
        .as_ref()
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "tokens_used column has unexpected data type".to_string(),
            ))
        })?;
    let custom_array = state_array
        .column(3)
        .as_ref()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "custom column has unexpected data type".to_string(),
            ))
        })?;

    let mut results = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let created_at =
            DateTime::from_timestamp_micros(created_at_array.value(row)).ok_or_else(|| {
                LanceError::from(ArrowError::InvalidArgumentError(format!(
                    "invalid timestamp value {}",
                    created_at_array.value(row)
                )))
            })?;

        let state_metadata = if state_array.is_null(row) {
            None
        } else {
            Some(StateMetadata {
                step: if step_array.is_null(row) {
                    None
                } else {
                    Some(step_array.value(row))
                },
                active_plan_id: if active_plan_array.is_null(row) {
                    None
                } else {
                    Some(active_plan_array.value(row).to_string())
                },
                tokens_used: if tokens_used_array.is_null(row) {
                    None
                } else {
                    Some(tokens_used_array.value(row))
                },
                custom: if custom_array.is_null(row) {
                    None
                } else {
                    Some(custom_array.value(row).to_string())
                },
            })
        };

        let text_payload = if text_array.is_null(row) {
            None
        } else {
            Some(text_array.value(row).to_string())
        };

        let binary_payload = if binary_array.is_null(row) {
            None
        } else {
            Some(binary_array.value(row).to_vec())
        };

        let embedding = if embedding_array.is_null(row) {
            None
        } else {
            Some(embedding_from_list(embedding_array, row)?)
        };

        let role = if role_array.is_null(row) {
            return Err(LanceError::from(ArrowError::InvalidArgumentError(
                "role column contains null values".to_string(),
            )));
        } else {
            let role_values = role_array
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    LanceError::from(ArrowError::InvalidArgumentError(
                        "role dictionary values are not strings".to_string(),
                    ))
                })?;
            let key = role_array.keys().value(row) as usize;
            role_values.value(key).to_string()
        };

        let bot_id = bot_id_array.and_then(|arr| {
            if arr.is_null(row) {
                None
            } else {
                Some(arr.value(row).to_string())
            }
        });

        let session_id = session_id_array.and_then(|arr| {
            if arr.is_null(row) {
                None
            } else {
                Some(arr.value(row).to_string())
            }
        });

        results.push(ContextRecord {
            id: id_array.value(row).to_string(),
            run_id: run_id_array.value(row).to_string(),
            bot_id,
            session_id,
            created_at,
            role,
            state_metadata,
            content_type: content_type_array.value(row).to_string(),
            text_payload,
            binary_payload,
            embedding,
        });
    }

    Ok(results)
}

fn embedding_from_list(list: &FixedSizeListArray, row: usize) -> LanceResult<Vec<f32>> {
    let values = list.value(row);
    let float_array = values
        .as_ref()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "embedding column does not contain float32 values".to_string(),
            ))
        })?;
    let mut embedding = Vec::with_capacity(float_array.len());
    for idx in 0..float_array.len() {
        embedding.push(float_array.value(idx));
    }
    Ok(embedding)
}

fn column_as<'a, A>(batch: &'a RecordBatch, name: &str) -> LanceResult<&'a A>
where
    A: Array + 'static,
{
    let column = batch.column_by_name(name).ok_or_else(|| {
        LanceError::from(ArrowError::InvalidArgumentError(format!(
            "column '{name}' not found"
        )))
    })?;
    column.as_ref().as_any().downcast_ref::<A>().ok_or_else(|| {
        LanceError::from(ArrowError::InvalidArgumentError(format!(
            "column '{name}' has unexpected data type"
        )))
    })
}

fn column_as_optional<'a, A>(batch: &'a RecordBatch, name: &str) -> Option<&'a A>
where
    A: Array + 'static,
{
    batch
        .column_by_name(name)
        .and_then(|col| col.as_ref().as_any().downcast_ref::<A>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serde::CONTENT_TYPE_TEXT;
    use chrono::Utc;
    use tempfile::TempDir;

    fn make_embedding(pivot: f32) -> Vec<f32> {
        let mut values = vec![0.0; DEFAULT_EMBEDDING_DIM as usize];
        if !values.is_empty() {
            values[0] = pivot;
        }
        values
    }

    fn text_record(id: &str, embedding_pivot: f32) -> ContextRecord {
        ContextRecord {
            id: id.to_string(),
            run_id: format!("run-{id}"),
            bot_id: None,
            session_id: None,
            created_at: Utc::now(),
            role: "user".to_string(),
            state_metadata: Some(StateMetadata {
                step: Some(1),
                active_plan_id: Some("plan".to_string()),
                tokens_used: Some(10),
                custom: None,
            }),
            content_type: CONTENT_TYPE_TEXT.to_string(),
            text_payload: Some(format!("payload-{id}")),
            binary_payload: None,
            embedding: Some(make_embedding(embedding_pivot)),
        }
    }

    #[test]
    fn search_orders_by_distance() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let first = text_record("a", 0.0);
            let second = text_record("b", 1.0);
            store.add(&[first.clone(), second.clone()]).await.unwrap();

            let query = make_embedding(1.0);
            let results = store.search(&query, Some(2)).await.unwrap();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0].record.id, second.id);
            assert!(
                results[0].distance <= results[1].distance,
                "results not ordered by distance: {:?}",
                results
            );
        });
    }

    #[test]
    fn search_validates_query_length() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let err = store.search(&[0.0_f32], None).await.unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("embedding dimension"),
                "unexpected error message: {message}"
            );
        });
    }
}
