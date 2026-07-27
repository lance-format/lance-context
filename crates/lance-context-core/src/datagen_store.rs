//! Lance-backed append-only checkpoint log for datagen pipelines.
//!
//! A [`DatagenStore`] owns exactly one Lance dataset. Item state, failures, and
//! trajectories are all derived by folding immutable events from that dataset;
//! there are no cross-dataset writes or correctness-critical projections.
//!
//! Concurrent writers append through per-instance MemWAL shards. Reads union
//! the base table with every flushed shard and de-duplicate by deterministic
//! `event_id`, making a retried checkpoint batch idempotent.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int32Builder, Int64Builder, LargeBinaryBuilder,
    LargeStringBuilder, StringBuilder, TimestampMicrosecondBuilder,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, RecordBatch, RecordBatchIterator, StringArray, TimestampMicrosecondArray,
    UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use futures::{stream, StreamExt, TryStreamExt};
use lance::dataset::mem_wal::{
    DatasetMemWalExt, LsmScanner, ShardManifestStore, ShardSnapshot, ShardWriter, ShardWriterConfig,
};
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};
use lance_index::mem_wal::{ShardManifest, MEM_WAL_INDEX_NAME};
use tokio::task::JoinHandle;
use tracing::{info, warn};
use uuid::Uuid;

use crate::datagen::{
    datagen_failures, datagen_trajectory, fold_datagen_events, DatagenBlobValue, DatagenEvent,
    DatagenEventType, DatagenFailure, DatagenItemLookup, DatagenItemStatus, DatagenRootItemStatuses,
    DatagenStepCursor, DatagenStepKind, DatagenValue,
};
use crate::rollout_store::derive_shard_id;
use crate::store::{column_as, column_as_optional, timestamp_from_micros};

const DEFAULT_MANIFEST_SCAN_BATCH_SIZE: usize = 16;
const DEFAULT_SHARD_SCAN_CONCURRENCY: usize = 16;

/// Configuration for opening a [`DatagenStore`].
#[derive(Debug, Clone, Default)]
pub struct DatagenStoreOptions {
    pub storage_options: Option<HashMap<String, String>>,
    /// Stable identity of this writer instance. Multi-writer deployments must
    /// assign a distinct value to each live instance.
    pub shard_id: Option<String>,
    /// Merge this writer's flushed generations into the base table after the
    /// threshold is reached. `None` or zero disables count-triggered merging.
    pub merge_after_generations: Option<usize>,
    /// Periodically merge this writer's pending generations. `None` or zero
    /// disables the timer.
    pub cleanup_interval_secs: Option<u64>,
}

/// A single append-only Lance dataset for datagen checkpoint events.
pub struct DatagenStore {
    dataset: Dataset,
    write_shard: Uuid,
    storage_options: Option<HashMap<String, String>>,
    merge_after_generations: usize,
    cleanup_interval_secs: u64,
    write_writer: Option<ShardWriter>,
}

impl DatagenStore {
    pub async fn open(uri: &str) -> LanceResult<Self> {
        Self::open_with_options(uri, DatagenStoreOptions::default()).await
    }

    pub async fn open_with_options(uri: &str, options: DatagenStoreOptions) -> LanceResult<Self> {
        Self::open_inner(uri, options, true).await
    }

    pub async fn open_existing_with_options(
        uri: &str,
        options: DatagenStoreOptions,
    ) -> LanceResult<Self> {
        Self::open_inner(uri, options, false).await
    }

    async fn open_inner(
        uri: &str,
        options: DatagenStoreOptions,
        create_if_missing: bool,
    ) -> LanceResult<Self> {
        let storage_options = options.storage_options.clone();
        let dataset = match Self::load_with_options(uri, storage_options.clone()).await {
            Ok(dataset) => dataset,
            Err(LanceError::DatasetNotFound { .. }) if create_if_missing => {
                Self::create_with_options(uri, storage_options.clone()).await?
            }
            Err(error) => return Err(error),
        };

        Ok(Self {
            dataset,
            write_shard: derive_shard_id(options.shard_id.as_deref()),
            storage_options,
            merge_after_generations: options.merge_after_generations.unwrap_or(0),
            cleanup_interval_secs: options.cleanup_interval_secs.unwrap_or(0),
            write_writer: None,
        })
    }

    #[must_use]
    pub fn uri(&self) -> &str {
        self.dataset.uri()
    }

    #[must_use]
    pub fn version(&self) -> u64 {
        self.dataset.manifest.version
    }

    /// Append one or more complete checkpoint batches.
    ///
    /// The supplied slice is persisted as one MemWAL generation. Callers should
    /// include FIELD_* events and the corresponding STEP_COMPLETED marker in
    /// the same call so a crash cannot expose a partially checkpointed step.
    pub async fn append(&mut self, events: &[DatagenEvent]) -> LanceResult<u64> {
        if events.is_empty() {
            return Ok(self.dataset.manifest.version);
        }
        validate_write_batch(events)?;
        let batch = events_to_batch(events)?;

        self.ensure_mem_wal().await?;
        self.write_with_resident_writer(&batch).await?;
        if self.merge_after_generations > 0 {
            self.merge_own_shard_if_ready(self.merge_after_generations)
                .await?;
        }
        Ok(self.dataset.manifest.version)
    }

    /// Append one completed step boundary atomically.
    ///
    /// The batch may contain any number of FIELD_SET/FIELD_APPEND events,
    /// including zero, but must contain exactly one STEP_COMPLETED marker and
    /// must belong to one item/checkpoint/writer attempt.
    pub async fn append_checkpoint(&mut self, events: &[DatagenEvent]) -> LanceResult<u64> {
        validate_checkpoint_batch(events)?;
        self.append(events).await
    }

    async fn write_with_resident_writer(&mut self, batch: &RecordBatch) -> LanceResult<()> {
        match self.put_seal_drain(batch).await {
            Ok(()) => Ok(()),
            Err(error) if is_fenced_error(&error) => {
                self.write_writer = None;
                self.put_seal_drain(batch).await
            }
            Err(error) => Err(error),
        }
    }

    async fn put_seal_drain(&mut self, batch: &RecordBatch) -> LanceResult<()> {
        self.ensure_write_writer().await?;
        let writer = self
            .write_writer
            .as_ref()
            .expect("ensure_write_writer set the writer");
        writer.put(vec![batch.clone()]).await?;
        writer.force_seal_active().await?;
        writer.wait_for_flush_drain().await?;
        Ok(())
    }

    async fn ensure_write_writer(&mut self) -> LanceResult<()> {
        if self.write_writer.is_some() {
            return Ok(());
        }
        let config = ShardWriterConfig {
            shard_id: self.write_shard,
            ..Default::default()
        };
        self.write_writer = Some(
            self.dataset
                .mem_wal_writer(self.write_shard, config)
                .await?,
        );
        Ok(())
    }

    /// Gracefully stop this store's resident MemWAL writer.
    pub async fn close(&mut self) -> LanceResult<()> {
        if let Some(writer) = self.write_writer.take() {
            writer.close().await?;
        }
        Ok(())
    }

    /// Read one item's event history without materializing blob bytes.
    pub async fn events_for_item(&self, item_id: &str) -> LanceResult<Vec<DatagenEvent>> {
        self.filtered_events(&format!("item_id = '{}'", escape_sql_literal(item_id)))
            .await
    }

    /// Read a root item and every projected descendant without blob bytes.
    pub async fn events_for_root(&self, root_item_id: &str) -> LanceResult<Vec<DatagenEvent>> {
        self.filtered_events(&format!(
            "root_item_id = '{}'",
            escape_sql_literal(root_item_id)
        ))
        .await
    }

    /// Read failure events directly from the source-of-truth log.
    pub async fn failures(&self, run_id: Option<&str>) -> LanceResult<Vec<DatagenEvent>> {
        let filter = match run_id {
            Some(run_id) => format!(
                "event_type = 'FAILED' AND run_id = '{}'",
                escape_sql_literal(run_id)
            ),
            None => "event_type = 'FAILED'".to_string(),
        };
        self.filtered_events(&filter).await
    }

    /// Reconstruct one item's latest state exclusively from its event log. Returns
    /// [`DatagenItemLookup::NeverStarted`] when the item has no ITEM_CREATED (the fresh-vs-resume fork).
    pub async fn fold_item(&self, item_id: &str) -> LanceResult<DatagenItemLookup> {
        let events = self.events_for_item(item_id).await?;
        match fold_datagen_events(&events).map_err(invalid_input)? {
            Some(item) => Ok(DatagenItemLookup::Found(item)),
            None => Ok(DatagenItemLookup::NeverStarted),
        }
    }

    /// Classify every root item that shares `root_item_id` with the given roots by folded lifecycle
    /// status. A root not present in the log is simply absent from the result (never started).
    pub async fn root_item_statuses(
        &self,
        root_item_ids: &[&str],
    ) -> LanceResult<DatagenRootItemStatuses> {
        let mut statuses = HashMap::new();
        for root_item_id in root_item_ids {
            let events = self.events_for_root(root_item_id).await?;
            let root_events: Vec<DatagenEvent> = events
                .into_iter()
                .filter(|event| &event.item_id == root_item_id)
                .collect();
            if let Some(item) = fold_datagen_events(&root_events).map_err(invalid_input)? {
                statuses.insert(root_item_id.to_string(), item.status);
            }
        }
        Ok(DatagenRootItemStatuses::from_map(statuses))
    }

    /// Read all failure records for an item directly from the failure lens.
    pub async fn item_failures(&self, item_id: &str) -> LanceResult<Vec<DatagenFailure>> {
        let events = self.events_for_item(item_id).await?;
        datagen_failures(&events).map_err(invalid_input)
    }

    /// Reconstruct the ordered step cursors an item passed through, without loading blob bytes.
    pub async fn trajectory(&self, item_id: &str) -> LanceResult<Vec<DatagenStepCursor>> {
        let events = self.events_for_item(item_id).await?;
        datagen_trajectory(&events).map_err(invalid_input)
    }

    /// Materialize one FIELD_* event's blob by id.
    ///
    /// Each candidate dataset is first filtered using only `event_id`; the
    /// matching `_rowid` is then passed to `take_rows` for an O(single blob)
    /// payload read.
    pub async fn get_blob(&self, event_id: &str) -> LanceResult<Option<Vec<u8>>> {
        let snapshots = self.wal_shard_snapshots().await?;
        let mut generations: Vec<(u64, String)> = snapshots
            .iter()
            .flat_map(|snapshot| {
                snapshot.flushed_generations.iter().map(|generation| {
                    (
                        generation.generation,
                        self.flushed_generation_uri(snapshot.shard_id, &generation.path),
                    )
                })
            })
            .collect();
        generations.sort_by_key(|(generation, _)| std::cmp::Reverse(*generation));

        for (_, uri) in generations {
            let dataset = self.open_flushed_dataset(&uri).await?;
            if let Some(payload) = Self::get_blob_from_dataset(&dataset, event_id).await? {
                return Ok(payload);
            }
        }

        Ok(Self::get_blob_from_dataset(&self.dataset, event_id)
            .await?
            .flatten())
    }

    /// Number of flushed generations waiting across all writer shards.
    pub async fn pending_wal_generations(&self) -> LanceResult<usize> {
        Ok(self
            .wal_shard_snapshots()
            .await?
            .iter()
            .map(|snapshot| snapshot.flushed_generations.len())
            .sum())
    }

    /// Merge every currently flushed generation owned by this writer into the
    /// base table.
    pub async fn cleanup_own_shard(&mut self) -> LanceResult<usize> {
        self.merge_own_shard_if_ready(1).await
    }

    /// Start periodic cleanup after the configured interval. The task keeps
    /// only a weak reference and exits when the store is dropped.
    pub fn spawn_periodic_cleanup(store: Arc<tokio::sync::RwLock<Self>>) -> Option<JoinHandle<()>> {
        let interval_secs = store.try_read().ok()?.cleanup_interval_secs;
        if interval_secs == 0 {
            return None;
        }

        let weak = Arc::downgrade(&store);
        Some(tokio::spawn(async move {
            let interval = std::time::Duration::from_secs(interval_secs);
            let pass_timeout = interval
                .saturating_mul(5)
                .max(std::time::Duration::from_secs(30));
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let Some(store) = weak.upgrade() else {
                    return;
                };
                let mut guard = store.write().await;
                match tokio::time::timeout(pass_timeout, guard.cleanup_own_shard()).await {
                    Ok(Ok(0)) => {}
                    Ok(Ok(reclaimed)) => info!(
                        shard = %guard.write_shard,
                        reclaimed,
                        "datagen WAL cleanup merged flushed generations"
                    ),
                    Ok(Err(error)) => warn!(
                        shard = %guard.write_shard,
                        error = %error,
                        "datagen WAL cleanup failed"
                    ),
                    Err(_) => warn!(
                        shard = %guard.write_shard,
                        timeout_secs = pass_timeout.as_secs(),
                        "datagen WAL cleanup timed out"
                    ),
                }
            }
        }))
    }

    async fn filtered_events(&self, filter: &str) -> LanceResult<Vec<DatagenEvent>> {
        let columns = self.non_blob_columns();
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let scanner = self.lsm_scanner().await?.project(&refs).filter(filter)?;
        let mut stream = scanner.try_into_stream().await?;
        let mut events = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            events.extend(batch_to_events(&batch)?);
        }
        events.sort_by(|left, right| {
            left.item_id
                .cmp(&right.item_id)
                .then_with(|| left.item_seq.cmp(&right.item_seq))
                .then_with(|| left.event_id.cmp(&right.event_id))
        });
        Ok(events)
    }

    async fn merge_own_shard_if_ready(&mut self, threshold: usize) -> LanceResult<usize> {
        let object_store = self.dataset.object_store(None).await?;
        let branch_location = self.dataset.branch_location();
        let manifest_store = ShardManifestStore::new(
            object_store,
            &branch_location.path,
            self.write_shard,
            DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
        );
        let Some(manifest) = manifest_store.read_latest().await? else {
            return Ok(0);
        };
        let pending = manifest.flushed_generations.len();
        if pending == 0 || pending < threshold.max(1) {
            return Ok(0);
        }
        self.merge_own_shard(&manifest_store, &manifest).await?;
        Ok(pending)
    }

    async fn merge_own_shard(
        &mut self,
        manifest_store: &ShardManifestStore,
        manifest: &ShardManifest,
    ) -> LanceResult<()> {
        if manifest.flushed_generations.is_empty() {
            return Ok(());
        }

        // claim_epoch below fences the resident writer. Drain it first and
        // reopen lazily on the next append.
        self.close().await?;

        let base_uri = self.dataset.uri().trim_end_matches('/').to_string();
        let mut merged_generations = HashSet::new();
        let mut merged_paths = Vec::new();
        let mut batches = Vec::new();
        for flushed in &manifest.flushed_generations {
            let generation_uri = format!(
                "{}/_mem_wal/{}/{}",
                base_uri, self.write_shard, flushed.path
            );
            let generation =
                Self::load_with_options(&generation_uri, self.storage_options.clone()).await?;
            let mut stream = generation.scan().try_into_stream().await?;
            while let Some(batch) = stream.try_next().await? {
                if batch.num_rows() > 0 {
                    batches.push(batch);
                }
            }
            merged_generations.insert(flushed.generation);
            merged_paths.push(flushed.path.clone());
        }

        if !batches.is_empty() {
            let schema = Arc::new(datagen_log_schema());
            let reader = RecordBatchIterator::new(
                batches.into_iter().map(Ok::<RecordBatch, ArrowError>),
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
            self.dataset.append(reader, Some(params)).await?;
        }

        let (epoch, _) = manifest_store.claim_epoch(manifest.shard_spec_id).await?;
        manifest_store
            .commit_update(epoch, |current| ShardManifest {
                version: current.version + 1,
                flushed_generations: current
                    .flushed_generations
                    .iter()
                    .filter(|generation| !merged_generations.contains(&generation.generation))
                    .cloned()
                    .collect(),
                ..current.clone()
            })
            .await?;

        let object_store = self.dataset.object_store(None).await?;
        let branch_path = self.dataset.branch_location().path.clone();
        for path in merged_paths {
            let generation_path = branch_path
                .clone()
                .join("_mem_wal")
                .join(self.write_shard.to_string().as_str())
                .join(path.as_str());
            if let Err(error) = object_store.remove_dir_all(generation_path).await {
                warn!(
                    shard = %self.write_shard,
                    generation_path = %path,
                    error = %error,
                    "failed to delete merged datagen WAL generation"
                );
            }
        }
        Ok(())
    }

    async fn get_blob_from_dataset(
        dataset: &Dataset,
        event_id: &str,
    ) -> LanceResult<Option<Option<Vec<u8>>>> {
        let mut scanner = dataset.scan();
        scanner
            .project(&["event_id"])?
            .filter(&format!("event_id = '{}'", escape_sql_literal(event_id)))?
            .with_row_id()
            .limit(Some(1), None)?;

        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            let event_id_array = column_as::<StringArray>(&batch, "event_id")?;
            let row_id_array = column_as::<UInt64Array>(&batch, "_rowid")?;
            for row in 0..batch.num_rows() {
                if event_id_array.value(row) != event_id {
                    continue;
                }
                let projection = dataset.schema().project(&["value_blob"])?;
                let payload_batch = dataset
                    .take_rows(&[row_id_array.value(row)], projection)
                    .await?;
                let payload = column_as_optional::<LargeBinaryArray>(&payload_batch, "value_blob");
                return Ok(Some(match payload {
                    Some(array) if !array.is_null(0) => Some(array.value(0).to_vec()),
                    _ => None,
                }));
            }
        }
        Ok(None)
    }

    fn non_blob_columns(&self) -> Vec<String> {
        self.dataset
            .schema()
            .fields
            .iter()
            .map(|field| field.name.clone())
            .filter(|name| name != "value_blob")
            .collect()
    }

    async fn lsm_scanner(&self) -> LanceResult<LsmScanner> {
        Ok(LsmScanner::new(
            Arc::new(self.dataset.clone()),
            self.wal_shard_snapshots().await?,
            vec!["event_id".to_string()],
        ))
    }

    async fn wal_shard_snapshots(&self) -> LanceResult<Vec<ShardSnapshot>> {
        let object_store = self.dataset.object_store(None).await?;
        let branch_path = self.dataset.branch_location().path.clone();
        let shard_ids = self.dataset.list_mem_wal_latest_shard_ids().await?;

        let snapshots: Vec<Option<ShardSnapshot>> = stream::iter(shard_ids)
            .map(|shard_id| {
                let object_store = object_store.clone();
                let branch_path = branch_path.clone();
                async move {
                    let manifest_store = ShardManifestStore::new(
                        object_store,
                        &branch_path,
                        shard_id,
                        DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
                    );
                    let Some(manifest) = manifest_store.read_latest().await? else {
                        return Ok(None);
                    };
                    let mut snapshot = ShardSnapshot::new(shard_id)
                        .with_spec_id(manifest.shard_spec_id)
                        .with_current_generation(manifest.current_generation);
                    for flushed in manifest.flushed_generations {
                        snapshot =
                            snapshot.with_flushed_generation(flushed.generation, flushed.path);
                    }
                    Ok::<_, LanceError>(Some(snapshot))
                }
            })
            .buffer_unordered(DEFAULT_SHARD_SCAN_CONCURRENCY)
            .try_collect()
            .await?;
        Ok(snapshots.into_iter().flatten().collect())
    }

    async fn ensure_mem_wal(&mut self) -> LanceResult<()> {
        if self.mem_wal_index_present().await? {
            return Ok(());
        }
        match self
            .dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
        {
            Ok(()) => Ok(()),
            Err(error) => {
                let uri = self.dataset.uri().to_string();
                self.dataset = Self::load_with_options(&uri, self.storage_options.clone()).await?;
                if self.mem_wal_index_present().await? {
                    Ok(())
                } else {
                    Err(error)
                }
            }
        }
    }

    async fn mem_wal_index_present(&self) -> LanceResult<bool> {
        let indices = self.dataset.load_indices().await?;
        Ok(indices.iter().any(|index| index.name == MEM_WAL_INDEX_NAME))
    }

    fn flushed_generation_uri(&self, shard_id: Uuid, path: &str) -> String {
        format!(
            "{}/_mem_wal/{shard_id}/{path}",
            self.dataset.uri().trim_end_matches('/')
        )
    }

    async fn open_flushed_dataset(&self, uri: &str) -> LanceResult<Dataset> {
        let mut builder = DatasetBuilder::from_uri(uri).with_session(self.dataset.session());
        if let Some(options) = self.storage_options.clone() {
            builder = builder.with_storage_options(options);
        }
        builder.load().await
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
        let schema = Arc::new(datagen_log_schema());
        let batches = RecordBatchIterator::new(
            vec![Ok::<RecordBatch, ArrowError>(RecordBatch::new_empty(
                schema.clone(),
            ))]
            .into_iter(),
            schema,
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
}

/// Arrow schema for the single append-only datagen checkpoint log.
#[must_use]
pub fn datagen_log_schema() -> Schema {
    let mut event_id_metadata = HashMap::new();
    event_id_metadata.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );

    Schema::new(vec![
        Field::new("event_id", DataType::Utf8, false).with_metadata(event_id_metadata),
        Field::new("item_id", DataType::Utf8, false),
        Field::new("root_item_id", DataType::Utf8, false),
        Field::new("parent_item_id", DataType::Utf8, true),
        Field::new("item_seq", DataType::Int64, false),
        Field::new("checkpoint_id", DataType::Utf8, false),
        Field::new("event_type", DataType::Utf8, false),
        Field::new("step_name", DataType::Utf8, true),
        Field::new("step_kind", DataType::Utf8, true),
        Field::new("step_index", DataType::Int64, true),
        Field::new("enclosing_step", DataType::Utf8, true),
        Field::new("selector_step", DataType::Utf8, true),
        Field::new("attempt", DataType::Int32, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("writer_epoch", DataType::Utf8, false),
        Field::new("field_name", DataType::Utf8, true),
        Field::new("field_type", DataType::Utf8, true),
        Field::new("codec_version", DataType::Int32, true),
        Field::new("value_kind", DataType::Utf8, true),
        Field::new("value_i64", DataType::Int64, true),
        Field::new("value_f64", DataType::Float64, true),
        Field::new("value_bool", DataType::Boolean, true),
        Field::new("value_str", DataType::LargeUtf8, true),
        Field::new("value_json", DataType::LargeUtf8, true),
        // Inline LargeBinary is required while MemWAL's LSM scanner does not
        // materialize blob-v2 columns.
        Field::new("value_blob", DataType::LargeBinary, true),
        Field::new("payload_size", DataType::Int64, true),
        Field::new("payload_checksum", DataType::Utf8, true),
        Field::new("query_tags_json", DataType::LargeUtf8, true),
        Field::new("status", DataType::Utf8, true),
        Field::new("error_type", DataType::Utf8, true),
        Field::new("error_dump", DataType::LargeUtf8, true),
        Field::new("traceback", DataType::LargeUtf8, true),
        Field::new(
            "event_ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("schema_version", DataType::Int32, false),
    ])
}

fn validate_write_batch(events: &[DatagenEvent]) -> LanceResult<()> {
    let mut by_id: HashMap<&str, &DatagenEvent> = HashMap::new();
    let mut item_sequences: HashMap<(&str, i64), &str> = HashMap::new();
    for event in events {
        event.validate().map_err(invalid_input)?;
        if let Some(previous) = by_id.insert(&event.event_id, event) {
            if previous != event {
                return Err(invalid_input(format!(
                    "event_id '{}' was reused with different content",
                    event.event_id
                )));
            }
        }
        if let Some(previous_event_id) =
            item_sequences.insert((&event.item_id, event.item_seq), &event.event_id)
        {
            if previous_event_id != event.event_id {
                return Err(invalid_input(format!(
                    "item '{}' has duplicate item_seq {} in one batch",
                    event.item_id, event.item_seq
                )));
            }
        }
        if let Some(DatagenValue::Blob(blob)) = &event.value {
            let bytes = blob.bytes.as_ref().ok_or_else(|| {
                invalid_input(format!(
                    "blob event '{}' has no bytes on the write path",
                    event.event_id
                ))
            })?;
            if blob.size != bytes.len() as i64 {
                return Err(invalid_input(format!(
                    "blob event '{}' declares {} bytes but contains {}",
                    event.event_id,
                    blob.size,
                    bytes.len()
                )));
            }
        }
    }
    Ok(())
}

fn validate_checkpoint_batch(events: &[DatagenEvent]) -> LanceResult<()> {
    let first = events
        .first()
        .ok_or_else(|| invalid_input("checkpoint batch must not be empty"))?;
    let mut completed = 0;
    for event in events {
        if event.item_id != first.item_id
            || event.checkpoint_id != first.checkpoint_id
            || event.run_id != first.run_id
            || event.writer_epoch != first.writer_epoch
            || event.attempt != first.attempt
        {
            return Err(invalid_input(
                "checkpoint batch must share item_id, checkpoint_id, run_id, writer_epoch, and attempt",
            ));
        }
        match event.event_type {
            DatagenEventType::FieldSet | DatagenEventType::FieldAppend => {}
            DatagenEventType::StepCompleted => completed += 1,
            _ => {
                return Err(invalid_input(
                    "checkpoint batch may contain only FIELD_SET, FIELD_APPEND, and STEP_COMPLETED events",
                ));
            }
        }
    }
    if completed != 1 {
        return Err(invalid_input(format!(
            "checkpoint batch requires exactly one STEP_COMPLETED event, found {completed}"
        )));
    }
    let completion = events
        .iter()
        .find(|event| event.event_type == DatagenEventType::StepCompleted)
        .unwrap();
    for event in events.iter().filter(|event| {
        matches!(
            event.event_type,
            DatagenEventType::FieldSet | DatagenEventType::FieldAppend
        )
    }) {
        if event.step_name != completion.step_name
            || event.step_index != completion.step_index
            || event.step_kind != completion.step_kind
            || event.enclosing_step != completion.enclosing_step
            || event.selector_step != completion.selector_step
        {
            return Err(invalid_input(
                "all field events must share the STEP_COMPLETED step identity",
            ));
        }
    }
    validate_write_batch(events)
}

fn events_to_batch(events: &[DatagenEvent]) -> LanceResult<RecordBatch> {
    let mut event_id = StringBuilder::new();
    let mut item_id = StringBuilder::new();
    let mut root_item_id = StringBuilder::new();
    let mut parent_item_id = StringBuilder::new();
    let mut item_seq = Int64Builder::new();
    let mut checkpoint_id = StringBuilder::new();
    let mut event_type = StringBuilder::new();
    let mut step_name = StringBuilder::new();
    let mut step_kind = StringBuilder::new();
    let mut step_index = Int64Builder::new();
    let mut enclosing_step = StringBuilder::new();
    let mut selector_step = StringBuilder::new();
    let mut attempt = Int32Builder::new();
    let mut run_id = StringBuilder::new();
    let mut writer_epoch = StringBuilder::new();
    let mut field_name = StringBuilder::new();
    let mut field_type = StringBuilder::new();
    let mut codec_version = Int32Builder::new();
    let mut value_kind = StringBuilder::new();
    let mut value_i64 = Int64Builder::new();
    let mut value_f64 = Float64Builder::new();
    let mut value_bool = BooleanBuilder::new();
    let mut value_str = LargeStringBuilder::new();
    let mut value_json = LargeStringBuilder::new();
    let mut value_blob = LargeBinaryBuilder::new();
    let mut payload_size = Int64Builder::new();
    let mut payload_checksum = StringBuilder::new();
    let mut query_tags_json = LargeStringBuilder::new();
    let mut status = StringBuilder::new();
    let mut error_type = StringBuilder::new();
    let mut error_dump = LargeStringBuilder::new();
    let mut traceback = LargeStringBuilder::new();
    let mut event_ts =
        TimestampMicrosecondBuilder::with_capacity(events.len()).with_timezone("UTC");
    let mut schema_version = Int32Builder::new();

    for event in events {
        event_id.append_value(&event.event_id);
        item_id.append_value(&event.item_id);
        root_item_id.append_value(&event.root_item_id);
        parent_item_id.append_option(event.parent_item_id.as_deref());
        item_seq.append_value(event.item_seq);
        checkpoint_id.append_value(&event.checkpoint_id);
        event_type.append_value(event.event_type.as_str());
        step_name.append_option(event.step_name.as_deref());
        step_kind.append_option(event.step_kind.map(DatagenStepKind::as_str));
        step_index.append_option(event.step_index);
        enclosing_step.append_option(event.enclosing_step.as_deref());
        selector_step.append_option(event.selector_step.as_deref());
        attempt.append_value(event.attempt);
        run_id.append_value(&event.run_id);
        writer_epoch.append_value(&event.writer_epoch);
        field_name.append_option(event.field_name.as_deref());
        field_type.append_option(event.field_type.as_deref());
        codec_version.append_option(event.codec_version);

        value_kind.append_option(event.value.as_ref().map(DatagenValue::kind));
        value_i64.append_option(match &event.value {
            Some(DatagenValue::Int(value)) => Some(*value),
            _ => None,
        });
        value_f64.append_option(match &event.value {
            Some(DatagenValue::Float(value)) => Some(*value),
            _ => None,
        });
        value_bool.append_option(match &event.value {
            Some(DatagenValue::Bool(value)) => Some(*value),
            _ => None,
        });
        value_str.append_option(match &event.value {
            Some(DatagenValue::Str(value)) => Some(value.as_str()),
            _ => None,
        });
        match &event.value {
            Some(DatagenValue::Json(value)) => value_json.append_value(value.to_string()),
            _ => value_json.append_null(),
        }
        match &event.value {
            Some(DatagenValue::Blob(blob)) => {
                value_blob.append_option(blob.bytes.as_deref());
                payload_size.append_value(blob.size);
                payload_checksum.append_option(blob.checksum.as_deref());
            }
            _ => {
                value_blob.append_null();
                payload_size.append_null();
                payload_checksum.append_null();
            }
        }
        match &event.query_tags {
            Some(tags) => query_tags_json.append_value(tags.to_string()),
            None => query_tags_json.append_null(),
        }
        status.append_option(event.status.map(DatagenItemStatus::as_str));
        error_type.append_option(event.error_type.as_deref());
        error_dump.append_option(event.error_dump.as_deref());
        traceback.append_option(event.traceback.as_deref());
        event_ts.append_value(event.event_ts.timestamp_micros());
        schema_version.append_value(event.schema_version);
    }

    let schema = Arc::new(datagen_log_schema());
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(event_id.finish()),
        Arc::new(item_id.finish()),
        Arc::new(root_item_id.finish()),
        Arc::new(parent_item_id.finish()),
        Arc::new(item_seq.finish()),
        Arc::new(checkpoint_id.finish()),
        Arc::new(event_type.finish()),
        Arc::new(step_name.finish()),
        Arc::new(step_kind.finish()),
        Arc::new(step_index.finish()),
        Arc::new(enclosing_step.finish()),
        Arc::new(selector_step.finish()),
        Arc::new(attempt.finish()),
        Arc::new(run_id.finish()),
        Arc::new(writer_epoch.finish()),
        Arc::new(field_name.finish()),
        Arc::new(field_type.finish()),
        Arc::new(codec_version.finish()),
        Arc::new(value_kind.finish()),
        Arc::new(value_i64.finish()),
        Arc::new(value_f64.finish()),
        Arc::new(value_bool.finish()),
        Arc::new(value_str.finish()),
        Arc::new(value_json.finish()),
        Arc::new(value_blob.finish()),
        Arc::new(payload_size.finish()),
        Arc::new(payload_checksum.finish()),
        Arc::new(query_tags_json.finish()),
        Arc::new(status.finish()),
        Arc::new(error_type.finish()),
        Arc::new(error_dump.finish()),
        Arc::new(traceback.finish()),
        Arc::new(event_ts.finish()),
        Arc::new(schema_version.finish()),
    ];
    Ok(RecordBatch::try_new(schema, arrays)?)
}

fn batch_to_events(batch: &RecordBatch) -> LanceResult<Vec<DatagenEvent>> {
    let event_id = column_as::<StringArray>(batch, "event_id")?;
    let item_id = column_as::<StringArray>(batch, "item_id")?;
    let root_item_id = column_as::<StringArray>(batch, "root_item_id")?;
    let parent_item_id = column_as_optional::<StringArray>(batch, "parent_item_id");
    let item_seq = column_as::<Int64Array>(batch, "item_seq")?;
    let checkpoint_id = column_as::<StringArray>(batch, "checkpoint_id")?;
    let event_type = column_as::<StringArray>(batch, "event_type")?;
    let step_name = column_as_optional::<StringArray>(batch, "step_name");
    let step_kind = column_as_optional::<StringArray>(batch, "step_kind");
    let step_index = column_as_optional::<Int64Array>(batch, "step_index");
    let enclosing_step = column_as_optional::<StringArray>(batch, "enclosing_step");
    let selector_step = column_as_optional::<StringArray>(batch, "selector_step");
    let attempt = column_as::<Int32Array>(batch, "attempt")?;
    let run_id = column_as::<StringArray>(batch, "run_id")?;
    let writer_epoch = column_as::<StringArray>(batch, "writer_epoch")?;
    let field_name = column_as_optional::<StringArray>(batch, "field_name");
    let field_type = column_as_optional::<StringArray>(batch, "field_type");
    let codec_version = column_as_optional::<Int32Array>(batch, "codec_version");
    let value_kind = column_as_optional::<StringArray>(batch, "value_kind");
    let value_i64 = column_as_optional::<Int64Array>(batch, "value_i64");
    let value_f64 = column_as_optional::<Float64Array>(batch, "value_f64");
    let value_bool = column_as_optional::<BooleanArray>(batch, "value_bool");
    let value_str = column_as_optional::<LargeStringArray>(batch, "value_str");
    let value_json = column_as_optional::<LargeStringArray>(batch, "value_json");
    let value_blob = column_as_optional::<LargeBinaryArray>(batch, "value_blob");
    let payload_size = column_as_optional::<Int64Array>(batch, "payload_size");
    let payload_checksum = column_as_optional::<StringArray>(batch, "payload_checksum");
    let query_tags_json = column_as_optional::<LargeStringArray>(batch, "query_tags_json");
    let status = column_as_optional::<StringArray>(batch, "status");
    let error_type = column_as_optional::<StringArray>(batch, "error_type");
    let error_dump = column_as_optional::<LargeStringArray>(batch, "error_dump");
    let traceback = column_as_optional::<LargeStringArray>(batch, "traceback");
    let event_ts = column_as::<TimestampMicrosecondArray>(batch, "event_ts")?;
    let schema_version = column_as::<Int32Array>(batch, "schema_version")?;

    let mut events = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let event_id_value = event_id.value(row).to_string();
        let value = match optional_string(value_kind, row).as_deref() {
            None => None,
            Some("int") => Some(DatagenValue::Int(required_i64(
                value_i64,
                row,
                "value_i64",
            )?)),
            Some("float") => Some(DatagenValue::Float(required_f64(
                value_f64,
                row,
                "value_f64",
            )?)),
            Some("bool") => Some(DatagenValue::Bool(required_bool(
                value_bool,
                row,
                "value_bool",
            )?)),
            Some("str") => Some(DatagenValue::Str(
                optional_large_string(value_str, row)
                    .ok_or_else(|| invalid_input("value_kind=str requires value_str"))?,
            )),
            Some("json") => {
                let json = optional_large_string(value_json, row)
                    .ok_or_else(|| invalid_input("value_kind=json requires value_json"))?;
                Some(DatagenValue::Json(serde_json::from_str(&json).map_err(
                    |error| {
                        invalid_input(format!(
                            "event '{}' contains invalid value_json: {}",
                            event_id_value, error
                        ))
                    },
                )?))
            }
            Some("blob") => Some(DatagenValue::Blob(DatagenBlobValue {
                bytes: optional_bytes(value_blob, row),
                size: required_i64(payload_size, row, "payload_size")?,
                checksum: optional_string(payload_checksum, row),
            })),
            Some(other) => {
                return Err(invalid_input(format!(
                    "event '{}' has unsupported value_kind '{}'",
                    event_id_value, other
                )));
            }
        };

        let query_tags = match optional_large_string(query_tags_json, row) {
            Some(json) => Some(serde_json::from_str(&json).map_err(|error| {
                invalid_input(format!(
                    "event '{}' contains invalid query_tags_json: {}",
                    event_id_value, error
                ))
            })?),
            None => None,
        };
        let event = DatagenEvent {
            event_id: event_id_value,
            item_id: item_id.value(row).to_string(),
            root_item_id: root_item_id.value(row).to_string(),
            parent_item_id: optional_string(parent_item_id, row),
            item_seq: item_seq.value(row),
            checkpoint_id: checkpoint_id.value(row).to_string(),
            event_type: DatagenEventType::parse(event_type.value(row)).map_err(invalid_input)?,
            step_name: optional_string(step_name, row),
            step_kind: match optional_string(step_kind, row) {
                Some(value) => Some(DatagenStepKind::parse(&value).map_err(invalid_input)?),
                None => None,
            },
            step_index: optional_i64(step_index, row),
            enclosing_step: optional_string(enclosing_step, row),
            selector_step: optional_string(selector_step, row),
            attempt: attempt.value(row),
            run_id: run_id.value(row).to_string(),
            writer_epoch: writer_epoch.value(row).to_string(),
            field_name: optional_string(field_name, row),
            field_type: optional_string(field_type, row),
            codec_version: optional_i32(codec_version, row),
            value,
            query_tags,
            status: match optional_string(status, row) {
                Some(value) => Some(DatagenItemStatus::parse(&value).map_err(invalid_input)?),
                None => None,
            },
            error_type: optional_string(error_type, row),
            error_dump: optional_large_string(error_dump, row),
            traceback: optional_large_string(traceback, row),
            event_ts: timestamp_from_micros(event_ts.value(row), "event_ts")?,
            schema_version: schema_version.value(row),
        };
        event.validate().map_err(invalid_input)?;
        events.push(event);
    }
    Ok(events)
}

fn optional_string(array: Option<&StringArray>, row: usize) -> Option<String> {
    array
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row).to_string())
}

fn optional_large_string(array: Option<&LargeStringArray>, row: usize) -> Option<String> {
    array
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row).to_string())
}

fn optional_bytes(array: Option<&LargeBinaryArray>, row: usize) -> Option<Vec<u8>> {
    array
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row).to_vec())
}

fn optional_i64(array: Option<&Int64Array>, row: usize) -> Option<i64> {
    array
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row))
}

fn optional_i32(array: Option<&Int32Array>, row: usize) -> Option<i32> {
    array
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row))
}

fn required_i64(array: Option<&Int64Array>, row: usize, name: &str) -> LanceResult<i64> {
    optional_i64(array, row).ok_or_else(|| invalid_input(format!("{name} must not be null")))
}

fn required_f64(array: Option<&Float64Array>, row: usize, name: &str) -> LanceResult<f64> {
    array
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row))
        .ok_or_else(|| invalid_input(format!("{name} must not be null")))
}

fn required_bool(array: Option<&BooleanArray>, row: usize, name: &str) -> LanceResult<bool> {
    array
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row))
        .ok_or_else(|| invalid_input(format!("{name} must not be null")))
}

fn escape_sql_literal(value: &str) -> String {
    value.replace('\'', "''")
}

fn invalid_input(message: impl Into<String>) -> LanceError {
    LanceError::from(ArrowError::InvalidArgumentError(message.into()))
}

impl Drop for DatagenStore {
    fn drop(&mut self) {
        if let Some(writer) = self.write_writer.take() {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    let _ = writer.close().await;
                });
            }
        }
    }
}

fn is_fenced_error(error: &LanceError) -> bool {
    let text = error.to_string();
    text.contains("fenced") || text.contains("Fenced")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::{
        datagen_event_id, DatagenFieldState, DatagenItemId, DatagenItemLookup, DatagenItemStatus,
        DatagenStepKind, DATAGEN_SCHEMA_VERSION,
    };
    use chrono::{TimeZone, Utc};
    use serde_json::json;
    use tempfile::TempDir;

    fn event(
        item_id: &str,
        seq: i64,
        checkpoint_id: &str,
        ordinal: u32,
        event_type: DatagenEventType,
    ) -> DatagenEvent {
        DatagenEvent {
            event_id: datagen_event_id(item_id, checkpoint_id, ordinal),
            item_id: item_id.to_string(),
            root_item_id: item_id.split('/').next().unwrap().to_string(),
            parent_item_id: None,
            item_seq: seq,
            checkpoint_id: checkpoint_id.to_string(),
            event_type,
            step_name: None,
            step_kind: None,
            step_index: None,
            enclosing_step: None,
            selector_step: None,
            attempt: 0,
            run_id: "run-1".to_string(),
            writer_epoch: "writer-1".to_string(),
            field_name: None,
            field_type: None,
            codec_version: None,
            value: None,
            query_tags: None,
            status: None,
            error_type: None,
            error_dump: None,
            traceback: None,
            event_ts: Utc.timestamp_micros(1_700_000_000_000_000 + seq).unwrap(),
            schema_version: DATAGEN_SCHEMA_VERSION,
        }
    }

    fn field_event(
        seq: i64,
        ordinal: u32,
        event_type: DatagenEventType,
        field_name: &str,
        field_type: &str,
        value: DatagenValue,
    ) -> DatagenEvent {
        let mut event = event("item-1", seq, "grade-0", ordinal, event_type);
        event.step_name = Some("grade".to_string());
        event.step_kind = Some(DatagenStepKind::Leaf);
        event.step_index = Some(2);
        event.field_name = Some(field_name.to_string());
        event.field_type = Some(field_type.to_string());
        event.codec_version = Some(1);
        event.value = Some(value);
        event
    }

    fn completed_step(seq: i64, ordinal: u32) -> DatagenEvent {
        let mut event = event(
            "item-1",
            seq,
            "grade-0",
            ordinal,
            DatagenEventType::StepCompleted,
        );
        event.step_name = Some("grade".to_string());
        event.step_kind = Some(DatagenStepKind::Leaf);
        event.step_index = Some(2);
        event
    }

    #[test]
    fn single_log_roundtrip_retry_fold_trajectory_and_blob() {
        let directory = TempDir::new().unwrap();
        let uri = directory.path().to_string_lossy().to_string();
        let blob_bytes = b"small-screenshot".to_vec();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = DatagenStore::open(&uri).await.unwrap();

            let mut created = event("item-1", 0, "created", 0, DatagenEventType::ItemCreated);
            created.query_tags = Some(json!({"domain": "math"}));
            store.append(&[created]).await.unwrap();

            let checkpoint = vec![
                field_event(
                    1,
                    0,
                    DatagenEventType::FieldSet,
                    "score",
                    "int",
                    DatagenValue::Int(i64::MAX),
                ),
                field_event(
                    2,
                    1,
                    DatagenEventType::FieldAppend,
                    "messages",
                    "json",
                    DatagenValue::Json(json!({"role": "assistant", "content": "42"})),
                ),
                field_event(
                    3,
                    2,
                    DatagenEventType::FieldSet,
                    "screenshot",
                    "image",
                    DatagenValue::Blob(DatagenBlobValue {
                        bytes: Some(blob_bytes.clone()),
                        size: blob_bytes.len() as i64,
                        checksum: Some("sha256:test".to_string()),
                    }),
                ),
                completed_step(4, 3),
            ];
            store.append_checkpoint(&checkpoint).await.unwrap();
            // Simulate a client retry after an ambiguous response.
            store.append_checkpoint(&checkpoint).await.unwrap();

            let mut terminal = event("item-1", 5, "terminal", 0, DatagenEventType::Terminal);
            terminal.status = Some(DatagenItemStatus::Completed);
            store.append(&[terminal]).await.unwrap();

            let events = store.events_for_item("item-1").await.unwrap();
            assert_eq!(events.len(), 6);
            let blob_event = events
                .iter()
                .find(|event| event.field_name.as_deref() == Some("screenshot"))
                .unwrap();
            let DatagenValue::Blob(blob) = blob_event.value.as_ref().unwrap() else {
                panic!("screenshot should be a blob");
            };
            assert!(blob.bytes.is_none());
            assert_eq!(blob.size, blob_bytes.len() as i64);
            assert_eq!(
                store.get_blob(&blob_event.event_id).await.unwrap(),
                Some(blob_bytes)
            );

            let folded = store.fold_item("item-1").await.unwrap();
            let folded = folded.folded().expect("item-1 was created");
            assert_eq!(folded.status, DatagenItemStatus::Completed);
            assert_eq!(
                folded.fields.get("score"),
                Some(&DatagenFieldState::Set(DatagenValue::Int(i64::MAX)))
            );
            assert_eq!(folded.trajectory.ordered.len(), 1);
            assert_eq!(folded.query_tags, Some(json!({"domain": "math"})));

            let trajectory = store.trajectory("item-1").await.unwrap();
            assert_eq!(trajectory.len(), 1);
            assert_eq!(trajectory[0].position.step.name, "grade");
        });
    }

    #[test]
    fn no_op_checkpoint_requires_and_persists_completion_marker() {
        let directory = TempDir::new().unwrap();
        let uri = directory.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = DatagenStore::open(&uri).await.unwrap();
            let created = event("item-1", 0, "created", 0, DatagenEventType::ItemCreated);
            store.append(&[created]).await.unwrap();
            let no_op = completed_step(1, 0);
            store.append_checkpoint(&[no_op]).await.unwrap();
            let folded = store.fold_item("item-1").await.unwrap();
            let folded = folded.folded().expect("item-1 was created");
            assert_eq!(folded.trajectory.ordered.len(), 1);

            let field_only = field_event(
                2,
                0,
                DatagenEventType::FieldSet,
                "score",
                "int",
                DatagenValue::Int(1),
            );
            let error = store
                .append_checkpoint(&[field_only])
                .await
                .unwrap_err()
                .to_string();
            assert!(error.contains("exactly one STEP_COMPLETED"));
        });
    }

    #[test]
    fn all_wal_shards_are_visible_and_failures_stay_in_the_log() {
        let directory = TempDir::new().unwrap();
        let uri = directory.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut writer_a = DatagenStore::open_with_options(
                &uri,
                DatagenStoreOptions {
                    storage_options: None,
                    shard_id: Some("writer-a".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let mut writer_b = DatagenStore::open_with_options(
                &uri,
                DatagenStoreOptions {
                    storage_options: None,
                    shard_id: Some("writer-b".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

            writer_a
                .append(&[event(
                    "root-a",
                    0,
                    "created-a",
                    0,
                    DatagenEventType::ItemCreated,
                )])
                .await
                .unwrap();

            let mut failed = event("root-b", 0, "failed-b", 0, DatagenEventType::Failed);
            failed.run_id = "run-failed".to_string();
            failed.writer_epoch = "writer-b".to_string();
            failed.step_name = Some("expand".to_string());
            failed.step_kind = Some(DatagenStepKind::Leaf);
            failed.step_index = Some(0);
            failed.error_type = Some("ValueError".to_string());
            failed.error_dump = Some("bad source item".to_string());
            writer_b.append(&[failed]).await.unwrap();

            assert_eq!(writer_a.events_for_item("root-b").await.unwrap().len(), 1);
            let failures = writer_a.failures(Some("run-failed")).await.unwrap();
            assert_eq!(failures.len(), 1);
            assert_eq!(failures[0].error_type.as_deref(), Some("ValueError"));
        });
    }

    #[test]
    fn cleanup_moves_events_to_base_without_changing_logical_results() {
        let directory = TempDir::new().unwrap();
        let uri = directory.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = DatagenStore::open_with_options(
                &uri,
                DatagenStoreOptions {
                    storage_options: None,
                    shard_id: Some("writer-a".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let created = event("item-1", 0, "created", 0, DatagenEventType::ItemCreated);
            let blob = field_event(
                1,
                0,
                DatagenEventType::FieldSet,
                "image",
                "image",
                DatagenValue::Blob(DatagenBlobValue {
                    bytes: Some(b"payload".to_vec()),
                    size: 7,
                    checksum: None,
                }),
            );
            store.append(&[created]).await.unwrap();
            store
                .append_checkpoint(&[blob.clone(), completed_step(2, 1)])
                .await
                .unwrap();
            assert_eq!(store.pending_wal_generations().await.unwrap(), 2);

            assert_eq!(store.cleanup_own_shard().await.unwrap(), 2);
            assert_eq!(store.pending_wal_generations().await.unwrap(), 0);
            assert_eq!(store.events_for_item("item-1").await.unwrap().len(), 3);
            assert_eq!(
                store.get_blob(&blob.event_id).await.unwrap(),
                Some(b"payload".to_vec())
            );
        });
    }

    #[test]
    fn fan_out_tree_reads_by_root_and_classifies_status() {
        let directory = TempDir::new().unwrap();
        let uri = directory.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = DatagenStore::open(&uri).await.unwrap();

            // Root item "7" fans out into one sub-item "7/solve_twice:0".
            let mut root_created = event("7", 0, "created-root", 0, DatagenEventType::ItemCreated);
            root_created.status = Some(DatagenItemStatus::Running);
            store.append(&[root_created]).await.unwrap();

            let mut child_created = event(
                "7/solve_twice:0",
                0,
                "created-child",
                0,
                DatagenEventType::ItemCreated,
            );
            child_created.parent_item_id = Some("7".to_string());
            child_created.status = Some(DatagenItemStatus::Running);
            store.append(&[child_created]).await.unwrap();

            let mut root_terminal = event("7", 1, "terminal", 0, DatagenEventType::Terminal);
            root_terminal.status = Some(DatagenItemStatus::Completed);
            store.append(&[root_terminal]).await.unwrap();

            // The whole tree is one root filter, no join.
            let tree = store.events_for_root("7").await.unwrap();
            assert_eq!(tree.len(), 3);

            // Bulk classification sees the root as terminated; the child is still running.
            let statuses = store.root_item_statuses(&["7"]).await.unwrap();
            let root_id = DatagenItemId::from_source_key("7");
            assert!(statuses.is_terminated(&root_id));

            let child = store.fold_item("7/solve_twice:0").await.unwrap();
            assert_eq!(
                child.folded().unwrap().status,
                DatagenItemStatus::Running
            );
            assert_eq!(
                child.folded().unwrap().parent_item_id,
                Some(DatagenItemId::from_source_key("7"))
            );

            // A never-started sibling folds to NeverStarted.
            assert_eq!(
                store.fold_item("7/solve_twice:1").await.unwrap(),
                DatagenItemLookup::NeverStarted
            );
        });
    }
}
