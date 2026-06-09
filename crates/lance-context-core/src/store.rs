use std::collections::{HashMap, HashSet};
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
use lance::dataset::mem_wal::{
    DatasetMemWalExt, LsmScanner, ShardManifestStore, ShardSnapshot, ShardWriterConfig,
};
use lance::dataset::optimize::{compact_files, CompactionMetrics, CompactionOptions};
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};
use lance_index::mem_wal::MEM_WAL_INDEX_NAME;
use lance_index::scalar::ScalarIndexParams;
use lance_index::IndexType;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::record::{ContextRecord, RecordFilters, SearchResult, StateMetadata};
use crate::serde::CONTENT_TYPE_TOMBSTONE;

/// Embedding length used for the semantic index column.
const DEFAULT_EMBEDDING_DIM: i32 = 1536;
const DEFAULT_SEARCH_LIMIT: usize = 10;
const DEFAULT_MANIFEST_SCAN_BATCH_SIZE: usize = 16;
const ID_INDEX_NAME: &str = "id_idx";

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

/// Type of scalar index on the `id` column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IdIndexType {
    /// No index on the id column.
    #[default]
    None,
    /// Zone-map index (min/max per fragment, lightweight).
    ZoneMap,
    /// B-tree index (point lookups, heavier).
    BTree,
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

/// Valid column names that may use blob encoding.
const VALID_BLOB_COLUMNS: &[&str] = &["text_payload", "binary_payload"];

/// Persistent Lance-backed context store.
#[derive(Clone)]
pub struct ContextStore {
    dataset: Dataset,
    compaction_state: Arc<Mutex<CompactionState>>,
    pub compaction_config: CompactionConfig,
    blob_columns: HashSet<String>,
    id_index_type: IdIndexType,
}

/// Additional configuration when opening a [`ContextStore`].
#[derive(Debug, Clone, Default)]
pub struct ContextStoreOptions {
    pub storage_options: Option<HashMap<String, String>>,
    pub compaction: CompactionConfig,
    /// Column names that should use Lance V1 blob encoding.
    /// Valid values: `"text_payload"`, `"binary_payload"`.
    pub blob_columns: HashSet<String>,
    /// Type of scalar index to create on the `id` column.
    pub id_index_type: IdIndexType,
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
        // Validate blob_columns
        for col in &options.blob_columns {
            if !VALID_BLOB_COLUMNS.contains(&col.as_str()) {
                return Err(LanceError::from(ArrowError::InvalidArgumentError(format!(
                    "invalid blob column '{}': valid columns are {:?}",
                    col, VALID_BLOB_COLUMNS
                ))));
            }
        }

        let storage_options = options.storage_options();
        let blob_columns = options.blob_columns.clone();
        let dataset = match Self::load_with_options(uri, storage_options.clone()).await {
            Ok(dataset) => dataset,
            Err(LanceError::DatasetNotFound { .. }) => {
                Self::create_with_options(uri, storage_options, &blob_columns).await?
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
            blob_columns,
            id_index_type: options.id_index_type,
        };

        // Ensure id index if configured
        store.ensure_id_index().await?;

        // Start background compaction if enabled
        store.start_background_compaction().await?;

        Ok(store)
    }

    /// Append context records to the store and return the new dataset version.
    pub async fn add(&mut self, entries: &[ContextRecord]) -> LanceResult<u64> {
        if entries.is_empty() {
            return Ok(self.dataset.manifest.version);
        }

        self.validate_unique_ids(entries).await?;
        self.write_entries(entries).await
    }

    async fn write_entries(&mut self, entries: &[ContextRecord]) -> LanceResult<u64> {
        if entries.is_empty() {
            return Ok(self.dataset.manifest.version);
        }

        // Group entries by (bot_id, session_id)
        let mut groups: HashMap<(Option<String>, Option<String>), Vec<ContextRecord>> =
            HashMap::new();
        for entry in entries {
            let key = (entry.bot_id.clone(), entry.session_id.clone());
            groups.entry(key).or_default().push(entry.clone());
        }

        // Ensure MemWAL is initialized (once for the dataset)
        {
            let indices = self.dataset.load_indices().await?;
            let has_mem_wal = indices.iter().any(|i| i.name == MEM_WAL_INDEX_NAME);

            if !has_mem_wal {
                // ZoneMap indices are not supported by MemWAL; exclude them
                let maintained_indexes: Vec<String> = indices
                    .iter()
                    .filter(|i| {
                        !(self.id_index_type == IdIndexType::ZoneMap && i.name == ID_INDEX_NAME)
                    })
                    .map(|i| i.name.clone())
                    .collect();
                self.dataset
                    .initialize_mem_wal()
                    .unsharded()
                    .maintained_indexes(maintained_indexes)
                    .execute()
                    .await?;
            }
        }

        for ((bot_id, session_id), group_entries) in groups {
            let region_id = Self::derive_region_id(&bot_id, &session_id);
            let batch = self.records_to_batch(&group_entries)?;
            let config = ShardWriterConfig {
                shard_id: region_id,
                ..Default::default()
            };

            let writer = self.dataset.mem_wal_writer(region_id, config).await?;
            writer.put(vec![batch]).await?;
            writer.close().await?;
        }

        Ok(self.dataset.manifest.version)
    }

    /// Logically forget a record by internal storage id.
    ///
    /// This writes a tombstone with the same primary key, preserving prior
    /// dataset versions while hiding the record from default reads.
    pub async fn delete_by_id(&mut self, id: &str) -> LanceResult<bool> {
        let Some(record) = self.get_by_id(id).await? else {
            return Ok(false);
        };
        self.write_tombstone_for(record).await?;
        Ok(true)
    }

    /// Logically forget a record by caller-supplied external id.
    pub async fn delete_by_external_id(&mut self, external_id: &str) -> LanceResult<bool> {
        let Some(record) = self.get_by_external_id(external_id).await? else {
            return Ok(false);
        };
        self.write_tombstone_for(record).await?;
        Ok(true)
    }

    async fn write_tombstone_for(&mut self, record: ContextRecord) -> LanceResult<u64> {
        let tombstone = ContextRecord {
            id: record.id,
            external_id: record.external_id,
            run_id: record.run_id,
            bot_id: record.bot_id,
            session_id: record.session_id,
            created_at: Utc::now(),
            role: record.role,
            state_metadata: None,
            metadata: None,
            content_type: CONTENT_TYPE_TOMBSTONE.to_string(),
            text_payload: None,
            binary_payload: None,
            embedding: None,
        };
        self.write_entries(std::slice::from_ref(&tombstone)).await
    }

    async fn validate_unique_ids(&self, entries: &[ContextRecord]) -> LanceResult<()> {
        let mut ids = HashSet::new();
        let mut external_ids = HashSet::new();
        for entry in entries {
            if entry.is_tombstone() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "content_type '{}' is reserved for internal tombstones",
                    CONTENT_TYPE_TOMBSTONE
                ))
                .into());
            }
            if !ids.insert(entry.id.as_str()) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "duplicate id '{}' in batch",
                    entry.id
                ))
                .into());
            }
            if let Some(external_id) = &entry.external_id {
                if !external_ids.insert(external_id.as_str()) {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "duplicate external_id '{}' in batch",
                        external_id
                    ))
                    .into());
                }
            }
        }

        for record in self.list(None, None).await? {
            if ids.contains(record.id.as_str()) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "id '{}' already exists",
                    record.id
                ))
                .into());
            }
            if let Some(external_id) = record.external_id {
                if external_ids.contains(external_id.as_str()) {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "external_id '{}' already exists",
                        external_id
                    ))
                    .into());
                }
            }
        }

        Ok(())
    }

    fn derive_region_id(bot_id: &Option<String>, session_id: &Option<String>) -> Uuid {
        let mut input = String::new();

        if let Some(bid) = bot_id {
            input.push_str(bid);
        }
        input.push('#');
        if let Some(sid) = session_id {
            input.push_str(sid);
        }

        // Use OID namespace as a base for our deterministic UUIDs
        Uuid::new_v5(&Uuid::NAMESPACE_OID, input.as_bytes())
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
        self.list_filtered(limit, offset, None).await
    }

    /// List records matching filters.
    pub async fn list_filtered(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<&RecordFilters>,
    ) -> LanceResult<Vec<ContextRecord>> {
        let scanner = self.lsm_scanner().await?;
        let mut stream = scanner.try_into_stream().await?;
        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            results.extend(
                batch_to_records(&batch)?
                    .into_iter()
                    .filter(|record| !record.is_tombstone()),
            );
        }

        if let Some(filters) = filters.filter(|filters| !filters.is_empty()) {
            results.retain(|record| filters.matches(record));
        }

        if let Some(offset) = offset {
            results = results.into_iter().skip(offset).collect();
        }
        if let Some(limit) = limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    /// Find a record by its internal storage id.
    pub async fn get_by_id(&self, id: &str) -> LanceResult<Option<ContextRecord>> {
        Ok(self
            .list(None, None)
            .await?
            .into_iter()
            .find(|record| record.id == id))
    }

    /// Find a record by its caller-supplied external id.
    pub async fn get_by_external_id(
        &self,
        external_id: &str,
    ) -> LanceResult<Option<ContextRecord>> {
        Ok(self
            .list(None, None)
            .await?
            .into_iter()
            .find(|record| record.external_id.as_deref() == Some(external_id)))
    }

    /// Perform a nearest-neighbor search over stored embeddings.
    pub async fn search(
        &self,
        query: &[f32],
        limit: Option<usize>,
    ) -> LanceResult<Vec<SearchResult>> {
        self.search_filtered(query, limit, None).await
    }

    /// Perform a nearest-neighbor search over stored embeddings matching filters.
    pub async fn search_filtered(
        &self,
        query: &[f32],
        limit: Option<usize>,
        filters: Option<&RecordFilters>,
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

        let mut results: Vec<SearchResult> = self
            .list_filtered(None, None, filters)
            .await?
            .into_iter()
            .filter_map(|record| {
                let distance = l2_distance(query, record.embedding.as_ref()?);
                Some(SearchResult { record, distance })
            })
            .collect();
        results.sort_by(|left, right| left.distance.total_cmp(&right.distance));
        results.truncate(top_k);
        Ok(results)
    }

    async fn lsm_scanner(&self) -> LanceResult<LsmScanner> {
        let object_store = self.dataset.object_store(None).await?;
        let branch_location = self.dataset.branch_location();
        let shard_ids = self.dataset.list_mem_wal_latest_shard_ids().await?;

        let mut shard_snapshots = Vec::with_capacity(shard_ids.len());
        for shard_id in shard_ids {
            let manifest_store = ShardManifestStore::new(
                object_store.clone(),
                &branch_location.path,
                shard_id,
                DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
            );
            let Some(manifest) = manifest_store.read_latest().await? else {
                continue;
            };

            let mut snapshot = ShardSnapshot::new(shard_id)
                .with_spec_id(manifest.shard_spec_id)
                .with_current_generation(manifest.current_generation);
            for flushed in manifest.flushed_generations {
                snapshot = snapshot.with_flushed_generation(flushed.generation, flushed.path);
            }
            shard_snapshots.push(snapshot);
        }

        Ok(LsmScanner::new(
            Arc::new(self.dataset.clone()),
            shard_snapshots,
            vec!["id".to_string()],
        ))
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
                drop(state); // Release lock before ensure_id_index

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

                // Ensure id index exists after compaction
                // (handles first-time creation on previously empty dataset)
                if let Err(e) = self.ensure_id_index().await {
                    warn!("Failed to ensure id index after compaction: {}", e);
                }

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

    /// Ensure the configured id index exists on the dataset.
    async fn ensure_id_index(&mut self) -> LanceResult<()> {
        if self.id_index_type == IdIndexType::None {
            return Ok(());
        }

        let indices = self.dataset.load_indices().await?;
        if indices.iter().any(|i| i.name == ID_INDEX_NAME) {
            return Ok(());
        }

        self.create_id_index().await
    }

    /// Create (or replace) the scalar index on the `id` column.
    pub async fn create_id_index(&mut self) -> LanceResult<()> {
        let index_type = match self.id_index_type {
            IdIndexType::ZoneMap => IndexType::ZoneMap,
            IdIndexType::BTree => IndexType::BTree,
            IdIndexType::None => return Ok(()),
        };

        info!("Creating {:?} index on id column", index_type);

        let params = ScalarIndexParams::default();

        self.dataset
            .create_index_builder(&["id"], index_type, &params)
            .name(ID_INDEX_NAME.to_string())
            .replace(true)
            .await?;

        // Reload dataset to pick up new index
        self.dataset = Dataset::open(self.dataset.uri()).await?;

        Ok(())
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
    ///
    /// When `blob_columns` contains a column name, that column is stored using
    /// Lance V1 blob encoding (out-of-line binary buffers). For `text_payload`,
    /// this also changes the Arrow type from `LargeUtf8` to `LargeBinary`.
    pub fn schema(blob_columns: &HashSet<String>) -> Schema {
        Self::schema_with_options(blob_columns, true, true)
    }

    fn schema_with_options(
        blob_columns: &HashSet<String>,
        include_external_id: bool,
        include_metadata: bool,
    ) -> Schema {
        let mut id_metadata = HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );

        let text_field = if blob_columns.contains("text_payload") {
            let mut metadata = HashMap::new();
            metadata.insert("lance-encoding:blob".to_string(), "true".to_string());
            Field::new("text_payload", DataType::LargeBinary, true).with_metadata(metadata)
        } else {
            Field::new("text_payload", DataType::LargeUtf8, true)
        };

        let binary_field = if blob_columns.contains("binary_payload") {
            let mut metadata = HashMap::new();
            metadata.insert("lance-encoding:blob".to_string(), "true".to_string());
            Field::new("binary_payload", DataType::LargeBinary, true).with_metadata(metadata)
        } else {
            Field::new("binary_payload", DataType::LargeBinary, true)
        };

        let mut fields = vec![Field::new("id", DataType::Utf8, false).with_metadata(id_metadata)];
        if include_external_id {
            fields.push(Field::new("external_id", DataType::Utf8, true));
        }
        fields.extend([
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
        ]);
        if include_metadata {
            fields.push(Field::new("metadata", DataType::LargeUtf8, true));
        }
        fields.extend([
            Field::new("content_type", DataType::Utf8, false),
            text_field,
            binary_field,
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DEFAULT_EMBEDDING_DIM,
                ),
                true,
            ),
        ]);

        Schema::new(fields)
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
        let schema = Arc::new(Self::schema(blob_columns));
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
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(options),
                )),
                ..Default::default()
            };
            params.store_params = Some(store_params);
        }

        Dataset::write(batches, uri, Some(params)).await
    }

    fn records_to_batch(&self, entries: &[ContextRecord]) -> LanceResult<RecordBatch> {
        let include_external_id = self
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == "external_id");
        if !include_external_id && entries.iter().any(|entry| entry.external_id.is_some()) {
            return Err(ArrowError::InvalidArgumentError(
                "external_id requires a context dataset created with external_id support"
                    .to_string(),
            )
            .into());
        }
        let include_metadata = self
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == "metadata");
        if !include_metadata && entries.iter().any(|entry| entry.metadata.is_some()) {
            return Err(ArrowError::InvalidArgumentError(
                "metadata requires a context dataset created with metadata support".to_string(),
            )
            .into());
        }

        let mut id_builder = StringBuilder::new();
        let mut external_id_builder = StringBuilder::new();
        let mut run_id_builder = StringBuilder::new();
        let mut bot_id_builder = StringBuilder::new();
        let mut session_id_builder = StringBuilder::new();
        let mut created_at_builder = TimestampMicrosecondBuilder::with_capacity(entries.len());
        let mut role_builder = StringDictionaryBuilder::<Int8Type>::new();
        let mut metadata_builder = LargeStringBuilder::new();
        let mut content_type_builder = StringBuilder::new();
        let mut binary_builder = LargeBinaryBuilder::new();

        let text_is_blob = self.blob_columns.contains("text_payload");
        let mut text_string_builder = if !text_is_blob {
            Some(LargeStringBuilder::new())
        } else {
            None
        };
        let mut text_binary_builder = if text_is_blob {
            Some(LargeBinaryBuilder::new())
        } else {
            None
        };

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
            external_id_builder.append_option(entry.external_id.as_deref());
            run_id_builder.append_value(&entry.run_id);
            bot_id_builder.append_option(entry.bot_id.as_deref());
            session_id_builder.append_option(entry.session_id.as_deref());
            created_at_builder.append_value(entry.created_at.timestamp_micros());
            role_builder.append(&entry.role)?;
            match &entry.metadata {
                Some(metadata) => metadata_builder.append_value(metadata.to_string()),
                None => metadata_builder.append_null(),
            }
            content_type_builder.append_value(&entry.content_type);

            if text_is_blob {
                match &entry.text_payload {
                    Some(value) => text_binary_builder
                        .as_mut()
                        .unwrap()
                        .append_value(value.as_bytes()),
                    None => text_binary_builder.as_mut().unwrap().append_null(),
                }
            } else {
                match &entry.text_payload {
                    Some(value) => text_string_builder.as_mut().unwrap().append_value(value),
                    None => text_string_builder.as_mut().unwrap().append_null(),
                }
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
        let external_id_array: ArrayRef = Arc::new(external_id_builder.finish());
        let run_id_array: ArrayRef = Arc::new(run_id_builder.finish());
        let bot_id_array: ArrayRef = Arc::new(bot_id_builder.finish());
        let session_id_array: ArrayRef = Arc::new(session_id_builder.finish());
        let created_at_array: ArrayRef = Arc::new(created_at_builder.finish());
        let role_array: ArrayRef = Arc::new(role_builder.finish());
        let metadata_array: ArrayRef = Arc::new(metadata_builder.finish());
        let content_type_array: ArrayRef = Arc::new(content_type_builder.finish());
        let text_array: ArrayRef = if text_is_blob {
            Arc::new(text_binary_builder.unwrap().finish())
        } else {
            Arc::new(text_string_builder.unwrap().finish())
        };
        let binary_array: ArrayRef = Arc::new(binary_builder.finish());
        let state_array: ArrayRef = Arc::new(state_builder.finish());
        let embedding_array: ArrayRef = Arc::new(embedding_builder.finish());

        let schema = Arc::new(Self::schema_with_options(
            &self.blob_columns,
            include_external_id,
            include_metadata,
        ));
        let mut arrays = vec![id_array];
        if include_external_id {
            arrays.push(external_id_array);
        }
        arrays.extend([
            run_id_array,
            bot_id_array,
            session_id_array,
            created_at_array,
            role_array,
            state_array,
        ]);
        if include_metadata {
            arrays.push(metadata_array);
        }
        arrays.extend([
            content_type_array,
            text_array,
            binary_array,
            embedding_array,
        ]);
        let batch = RecordBatch::try_new(schema, arrays)?;

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

/// Convert a record batch to context records.
fn batch_to_records(batch: &RecordBatch) -> LanceResult<Vec<ContextRecord>> {
    let id_array = column_as::<StringArray>(batch, "id")?;
    let external_id_array = column_as_optional::<StringArray>(batch, "external_id");
    let run_id_array = column_as::<StringArray>(batch, "run_id")?;
    let bot_id_array = column_as_optional::<StringArray>(batch, "bot_id");
    let session_id_array = column_as_optional::<StringArray>(batch, "session_id");
    let created_at_array = column_as::<TimestampMicrosecondArray>(batch, "created_at")?;
    let role_array = column_as::<DictionaryArray<Int8Type>>(batch, "role")?;
    let state_array = column_as::<StructArray>(batch, "state_metadata")?;
    let metadata_array = column_as_optional::<LargeStringArray>(batch, "metadata");
    let content_type_array = column_as::<StringArray>(batch, "content_type")?;
    let binary_array = column_as::<LargeBinaryArray>(batch, "binary_payload")?;
    let embedding_array = column_as::<FixedSizeListArray>(batch, "embedding")?;

    // Auto-detect whether text_payload is LargeBinary (blob) or LargeUtf8 (default)
    let text_is_binary = batch
        .schema()
        .field_with_name("text_payload")
        .is_ok_and(|f| f.data_type() == &DataType::LargeBinary);

    let text_string_array = if !text_is_binary {
        Some(column_as::<LargeStringArray>(batch, "text_payload")?)
    } else {
        None
    };
    let text_binary_array = if text_is_binary {
        Some(column_as::<LargeBinaryArray>(batch, "text_payload")?)
    } else {
        None
    };

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

        let text_payload = if text_is_binary {
            let arr = text_binary_array.unwrap();
            if arr.is_null(row) {
                None
            } else {
                Some(String::from_utf8_lossy(arr.value(row)).to_string())
            }
        } else {
            let arr = text_string_array.unwrap();
            if arr.is_null(row) {
                None
            } else {
                Some(arr.value(row).to_string())
            }
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

        let metadata = match metadata_array {
            Some(arr) if !arr.is_null(row) => {
                Some(serde_json::from_str(arr.value(row)).map_err(|err| {
                    LanceError::from(ArrowError::InvalidArgumentError(format!(
                        "invalid metadata JSON for record {}: {}",
                        id_array.value(row),
                        err
                    )))
                })?)
            }
            _ => None,
        };

        results.push(ContextRecord {
            id: id_array.value(row).to_string(),
            external_id: external_id_array.and_then(|arr| {
                if arr.is_null(row) {
                    None
                } else {
                    Some(arr.value(row).to_string())
                }
            }),
            run_id: run_id_array.value(row).to_string(),
            bot_id,
            session_id,
            created_at,
            role,
            state_metadata,
            metadata,
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

fn l2_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum::<f32>()
        .sqrt()
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
            external_id: None,
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
            metadata: None,
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

    #[test]
    fn external_id_roundtrips_and_supports_lookup() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let mut record = text_record("a", 0.0);
            record.external_id = Some("doc-123#chunk-1".to_string());
            store.add(std::slice::from_ref(&record)).await.unwrap();

            let by_external_id = store
                .get_by_external_id("doc-123#chunk-1")
                .await
                .unwrap()
                .unwrap();
            assert_eq!(by_external_id.id, record.id);
            assert_eq!(by_external_id.external_id, record.external_id);

            let by_id = store.get_by_id(&record.id).await.unwrap().unwrap();
            assert_eq!(by_id.external_id.as_deref(), Some("doc-123#chunk-1"));

            let missing = store.get_by_external_id("missing").await.unwrap();
            assert!(missing.is_none());
        });
    }

    #[test]
    fn add_rejects_duplicate_external_id() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let mut first = text_record("a", 0.0);
            first.external_id = Some("doc-123#chunk-1".to_string());
            store.add(std::slice::from_ref(&first)).await.unwrap();

            let mut duplicate = text_record("b", 0.0);
            duplicate.external_id = first.external_id.clone();
            let err = store.add(&[duplicate]).await.unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("external_id") && message.contains("already exists"),
                "unexpected error message: {message}"
            );
        });
    }

    #[test]
    fn add_rejects_reserved_tombstone_content_type() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let mut record = text_record("a", 0.0);
            record.content_type = CONTENT_TYPE_TOMBSTONE.to_string();

            let err = store.add(&[record]).await.unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("reserved") && message.contains("tombstone"),
                "unexpected error message: {message}"
            );
        });
    }

    #[test]
    fn delete_by_external_id_hides_record_from_default_reads() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let mut first = text_record("a", 0.0);
            first.external_id = Some("doc-123#chunk-1".to_string());
            let second = text_record("b", 2.0);
            store.add(&[first.clone(), second.clone()]).await.unwrap();

            assert!(store
                .delete_by_external_id("doc-123#chunk-1")
                .await
                .unwrap());

            assert!(store
                .get_by_external_id("doc-123#chunk-1")
                .await
                .unwrap()
                .is_none());
            assert!(store.get_by_id(&first.id).await.unwrap().is_none());

            let records = store.list(None, None).await.unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].id, second.id);

            let query = make_embedding(0.0);
            let hits = store.search(&query, Some(10)).await.unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].record.id, second.id);
        });
    }

    #[test]
    fn delete_by_id_hides_record_from_default_reads() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let mut first = text_record("a", 0.0);
            first.external_id = Some("doc-123#chunk-1".to_string());
            let second = text_record("b", 2.0);
            store.add(&[first.clone(), second.clone()]).await.unwrap();

            assert!(store.delete_by_id(&first.id).await.unwrap());

            assert!(store.get_by_id(&first.id).await.unwrap().is_none());
            assert!(store
                .get_by_external_id("doc-123#chunk-1")
                .await
                .unwrap()
                .is_none());

            let records = store.list(None, None).await.unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].id, second.id);

            let query = make_embedding(0.0);
            let hits = store.search(&query, Some(10)).await.unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].record.id, second.id);
        });
    }

    #[test]
    fn delete_missing_id_is_noop() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            assert!(!store.delete_by_id("missing").await.unwrap());
            assert!(!store.delete_by_external_id("missing").await.unwrap());
        });
    }

    #[test]
    fn external_id_can_be_reused_after_delete() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let mut first = text_record("a", 0.0);
            first.external_id = Some("doc-123#chunk-1".to_string());
            store.add(std::slice::from_ref(&first)).await.unwrap();
            assert!(store
                .delete_by_external_id("doc-123#chunk-1")
                .await
                .unwrap());

            let mut replacement = text_record("b", 1.0);
            replacement.external_id = first.external_id.clone();
            store.add(std::slice::from_ref(&replacement)).await.unwrap();

            let by_external_id = store
                .get_by_external_id("doc-123#chunk-1")
                .await
                .unwrap()
                .unwrap();
            assert_eq!(by_external_id.id, replacement.id);
            assert_eq!(store.list(None, None).await.unwrap().len(), 1);
        });
    }

    #[test]
    fn test_region_id_derivation_explicit() {
        let bot_id = Some("bot-123".to_string());
        let session_id = Some("session-456".to_string());

        let region_id_1 = ContextStore::derive_region_id(&bot_id, &session_id);
        let region_id_2 = ContextStore::derive_region_id(&bot_id, &session_id);

        assert_eq!(
            region_id_1, region_id_2,
            "Region ID should be deterministic for same inputs"
        );

        let other_session = Some("session-789".to_string());
        let region_id_3 = ContextStore::derive_region_id(&bot_id, &other_session);

        assert_ne!(
            region_id_1, region_id_3,
            "Region ID should differ for different inputs"
        );

        // Test None/None case (now deterministic based on empty strings)
        let region_id_none = ContextStore::derive_region_id(&None, &None);
        let region_id_none_2 = ContextStore::derive_region_id(&None, &None);
        assert_eq!(
            region_id_none, region_id_none_2,
            "Region ID for None/None should be deterministic"
        );
    }

    #[test]
    fn test_add_multiple_regions() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();

            // Create records for different regions
            let mut record1 = text_record("r1", 0.0);
            record1.bot_id = Some("bot-A".to_string());
            record1.session_id = Some("session-1".to_string());

            let mut record2 = text_record("r2", 0.0);
            record2.bot_id = Some("bot-B".to_string());
            record2.session_id = Some("session-2".to_string());

            // Add them in a single batch
            store
                .add(&[record1.clone(), record2.clone()])
                .await
                .unwrap();

            // Reload store to verify persistence
            let store = ContextStore::open(&uri).await.unwrap();

            // Verify we can list them back
            let results = store.list(None, None).await.unwrap();
            assert_eq!(results.len(), 2);

            let ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
            assert!(ids.contains(&"r1".to_string()));
            assert!(ids.contains(&"r2".to_string()));
        });
    }

    #[test]
    fn test_blob_binary_payload() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let options = ContextStoreOptions {
                blob_columns: HashSet::from(["binary_payload".to_string()]),
                ..Default::default()
            };
            let mut store = ContextStore::open_with_options(&uri, options)
                .await
                .unwrap();

            let mut record = text_record("blob-bin-1", 0.0);
            record.binary_payload = Some(vec![0xDE, 0xAD, 0xBE, 0xEF]);
            store.add(std::slice::from_ref(&record)).await.unwrap();

            // Verify schema has blob metadata on binary_payload
            let schema = ContextStore::schema(&store.blob_columns);
            let field = schema.field_with_name("binary_payload").unwrap();
            assert_eq!(
                field.metadata().get("lance-encoding:blob"),
                Some(&"true".to_string()),
            );
            // text_payload should remain LargeUtf8 without blob metadata
            let text_field = schema.field_with_name("text_payload").unwrap();
            assert_eq!(text_field.data_type(), &DataType::LargeUtf8);
            assert!(text_field.metadata().get("lance-encoding:blob").is_none());
        });
    }

    #[test]
    fn test_blob_text_payload() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let options = ContextStoreOptions {
                blob_columns: HashSet::from(["text_payload".to_string()]),
                ..Default::default()
            };
            let mut store = ContextStore::open_with_options(&uri, options)
                .await
                .unwrap();

            let record = text_record("blob-txt-1", 0.0);
            store.add(std::slice::from_ref(&record)).await.unwrap();

            // Roundtrip: records_to_batch -> batch_to_records
            let batch = store
                .records_to_batch(std::slice::from_ref(&record))
                .unwrap();
            let batch_schema = batch.schema();
            let text_field = batch_schema.field_with_name("text_payload").unwrap();
            assert_eq!(
                text_field.data_type(),
                &DataType::LargeBinary,
                "text_payload should be LargeBinary when blob-encoded"
            );

            let roundtripped = batch_to_records(&batch).unwrap();
            assert_eq!(roundtripped.len(), 1);
            assert_eq!(
                roundtripped[0].text_payload, record.text_payload,
                "text payload should survive blob roundtrip"
            );
        });
    }

    #[test]
    fn test_blob_both_columns() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let options = ContextStoreOptions {
                blob_columns: HashSet::from([
                    "text_payload".to_string(),
                    "binary_payload".to_string(),
                ]),
                ..Default::default()
            };
            let mut store = ContextStore::open_with_options(&uri, options)
                .await
                .unwrap();

            let mut record = text_record("blob-both-1", 0.0);
            record.binary_payload = Some(b"hello binary".to_vec());
            store.add(std::slice::from_ref(&record)).await.unwrap();

            // Both columns should have blob metadata
            let schema = ContextStore::schema(&store.blob_columns);
            let text_field = schema.field_with_name("text_payload").unwrap();
            let bin_field = schema.field_with_name("binary_payload").unwrap();
            assert_eq!(
                text_field.metadata().get("lance-encoding:blob"),
                Some(&"true".to_string()),
            );
            assert_eq!(
                bin_field.metadata().get("lance-encoding:blob"),
                Some(&"true".to_string()),
            );

            // Roundtrip via batch
            let batch = store
                .records_to_batch(std::slice::from_ref(&record))
                .unwrap();
            let roundtripped = batch_to_records(&batch).unwrap();
            assert_eq!(roundtripped.len(), 1);
            assert_eq!(roundtripped[0].text_payload, record.text_payload);
            assert_eq!(roundtripped[0].binary_payload, record.binary_payload);
        });
    }

    #[test]
    fn test_no_blob_default() {
        // Default options should produce no blob metadata
        let schema = ContextStore::schema(&HashSet::new());
        let text_field = schema.field_with_name("text_payload").unwrap();
        let bin_field = schema.field_with_name("binary_payload").unwrap();

        assert_eq!(text_field.data_type(), &DataType::LargeUtf8);
        assert!(text_field.metadata().get("lance-encoding:blob").is_none());
        assert_eq!(bin_field.data_type(), &DataType::LargeBinary);
        assert!(bin_field.metadata().get("lance-encoding:blob").is_none());
    }

    #[test]
    fn test_blob_schema_metadata() {
        let blob_columns =
            HashSet::from(["text_payload".to_string(), "binary_payload".to_string()]);
        let schema = ContextStore::schema(&blob_columns);

        let text_field = schema.field_with_name("text_payload").unwrap();
        assert_eq!(text_field.data_type(), &DataType::LargeBinary);
        assert_eq!(
            text_field.metadata().get("lance-encoding:blob"),
            Some(&"true".to_string()),
        );

        let bin_field = schema.field_with_name("binary_payload").unwrap();
        assert_eq!(bin_field.data_type(), &DataType::LargeBinary);
        assert_eq!(
            bin_field.metadata().get("lance-encoding:blob"),
            Some(&"true".to_string()),
        );

        // Non-blob fields should have no blob metadata
        let id_field = schema.field_with_name("id").unwrap();
        assert!(id_field.metadata().get("lance-encoding:blob").is_none());
    }

    #[test]
    fn test_blob_invalid_column_name() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let options = ContextStoreOptions {
                blob_columns: HashSet::from(["nonexistent_column".to_string()]),
                ..Default::default()
            };
            let result = ContextStore::open_with_options(&uri, options).await;
            assert!(result.is_err(), "should reject invalid blob column names");
            let err_msg = result.err().unwrap().to_string();
            assert!(
                err_msg.contains("invalid blob column"),
                "error should mention invalid blob column: {err_msg}"
            );
        });
    }

    #[test]
    fn test_batch_to_records_autodetects_text_type() {
        // Verify that batch_to_records works on both LargeUtf8 and LargeBinary
        // text_payload without needing configuration.
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            // Build a batch with text_payload as LargeUtf8 (default)
            let dir1 = TempDir::new().unwrap();
            let uri1 = dir1.path().to_string_lossy().to_string();
            let store_default = ContextStore::open(&uri1).await.unwrap();
            let record = text_record("auto-1", 0.0);
            let batch_utf8 = store_default
                .records_to_batch(std::slice::from_ref(&record))
                .unwrap();
            let results_utf8 = batch_to_records(&batch_utf8).unwrap();
            assert_eq!(results_utf8[0].text_payload, record.text_payload);

            // Build a batch with text_payload as LargeBinary (blob)
            let dir2 = TempDir::new().unwrap();
            let uri2 = dir2.path().to_string_lossy().to_string();
            let options = ContextStoreOptions {
                blob_columns: HashSet::from(["text_payload".to_string()]),
                ..Default::default()
            };
            let store_blob = ContextStore::open_with_options(&uri2, options)
                .await
                .unwrap();
            let batch_binary = store_blob
                .records_to_batch(std::slice::from_ref(&record))
                .unwrap();
            let results_binary = batch_to_records(&batch_binary).unwrap();
            assert_eq!(results_binary[0].text_payload, record.text_payload);
        });
    }

    #[test]
    fn test_id_index_btree() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let options = ContextStoreOptions {
                id_index_type: IdIndexType::BTree,
                ..Default::default()
            };
            let mut store = ContextStore::open_with_options(&uri, options)
                .await
                .unwrap();

            // Index should be created eagerly on open
            let indices = store.dataset.load_indices().await.unwrap();
            assert!(
                indices.iter().any(|i| i.name == ID_INDEX_NAME),
                "btree index should be created on open"
            );

            // Add data and verify it still works with the index
            for i in 0..5 {
                store
                    .add(&[text_record(&format!("btree-{i}"), i as f32)])
                    .await
                    .unwrap();
            }
            store.compact(None).await.unwrap();

            // Index should still exist after compaction
            let indices = store.dataset.load_indices().await.unwrap();
            assert!(
                indices.iter().any(|i| i.name == ID_INDEX_NAME),
                "btree index should persist after compaction"
            );
        });
    }

    #[test]
    fn test_id_index_zonemap() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let options = ContextStoreOptions {
                id_index_type: IdIndexType::ZoneMap,
                ..Default::default()
            };
            let mut store = ContextStore::open_with_options(&uri, options)
                .await
                .unwrap();

            // Index should be created eagerly on open
            let indices = store.dataset.load_indices().await.unwrap();
            assert!(
                indices.iter().any(|i| i.name == ID_INDEX_NAME),
                "zonemap index should be created on open"
            );

            for i in 0..5 {
                store
                    .add(&[text_record(&format!("zm-{i}"), i as f32)])
                    .await
                    .unwrap();
            }
            store.compact(None).await.unwrap();

            let indices = store.dataset.load_indices().await.unwrap();
            assert!(
                indices.iter().any(|i| i.name == ID_INDEX_NAME),
                "zonemap index should persist after compaction"
            );
        });
    }

    #[test]
    fn test_id_index_none_by_default() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();

            store.add(&[text_record("no-idx-1", 0.0)]).await.unwrap();
            store.compact(None).await.unwrap();

            let indices = store.dataset.load_indices().await.unwrap();
            assert!(
                !indices.iter().any(|i| i.name == ID_INDEX_NAME),
                "no id index should be created when IdIndexType::None"
            );
        });
    }

    #[test]
    fn test_id_index_idempotent() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let options = ContextStoreOptions {
                id_index_type: IdIndexType::BTree,
                ..Default::default()
            };
            let mut store = ContextStore::open_with_options(&uri, options)
                .await
                .unwrap();

            for i in 0..5 {
                store
                    .add(&[text_record(&format!("idem-{i}"), i as f32)])
                    .await
                    .unwrap();
            }

            // Create index twice -- second call should be a no-op
            store.create_id_index().await.unwrap();
            let v1 = store.version();
            store.ensure_id_index().await.unwrap();
            let v2 = store.version();
            assert_eq!(v1, v2, "ensure_id_index should not recreate existing index");
        });
    }
}
