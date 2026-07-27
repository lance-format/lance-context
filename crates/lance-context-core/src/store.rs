use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{
    FixedSizeListBuilder, Float32Builder, Int32Builder, Int64Builder, LargeBinaryBuilder,
    LargeStringBuilder, ListBuilder, StringBuilder, StringDictionaryBuilder, StructBuilder,
    TimestampMicrosecondBuilder,
};
use arrow_array::types::Int8Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float32Array, Int32Array, Int64Array,
    LargeBinaryArray, LargeStringArray, ListArray, RecordBatch, StringArray, StructArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema, TimeUnit};
use chrono::{DateTime, Timelike, Utc};
use futures::TryStreamExt;
use lance::dataset::mem_wal::LsmScanner;
use lance::dataset::optimize::CompactionMetrics;
use lance::dataset::Dataset;
use lance::dataset::NewColumnTransform;
use lance::index::DatasetIndexExt;
use lance::io::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};
use lance_index::scalar::ScalarIndexParams;
use lance_index::IndexType;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::record::{
    ContextRecord, LifecycleQueryOptions, RecordFilters, RecordPatch, Relationship, RetrieveResult,
    SearchResult, StateMetadata, UpdateResult, UpsertResult, LIFECYCLE_ACTIVE,
};
use crate::serde::CONTENT_TYPE_TOMBSTONE;
use crate::store_base::{StorageBase, StorageBaseOptions};

/// Embedding length used for the semantic index column.
const DEFAULT_EMBEDDING_DIM: i32 = 1536;
const DEFAULT_SEARCH_LIMIT: usize = 10;
const RRF_K: f32 = 60.0;
const ID_INDEX_NAME: &str = "id_idx";
pub(crate) const RELATIONSHIPS_COLUMN: &str = "relationships";
/// Schema-metadata key under which the configured [`DistanceMetric`] is persisted
/// so it round-trips on `open` without being re-specified by the caller.
const DISTANCE_METRIC_METADATA_KEY: &str = "lance-context:distance_metric";

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

/// Distance metric used to rank candidates during vector search.
///
/// All variants are normalized so that a **smaller** value means a closer
/// match, keeping the search ranking ascending regardless of metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance. Default for backward compatibility.
    #[default]
    L2,
    /// Cosine distance (`1 - cosine_similarity`). Common for normalized
    /// embeddings from most modern models.
    Cosine,
    /// Negated dot product (maximum inner product search).
    Dot,
}

impl DistanceMetric {
    /// Parse a metric from its string identifier (`"l2"`, `"cosine"`, `"dot"`).
    /// Matching is case-insensitive.
    ///
    /// # Errors
    /// Returns an error if the identifier is not a recognized metric.
    pub fn parse(value: &str) -> LanceResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            "dot" | "dot_product" => Ok(Self::Dot),
            other => Err(LanceError::from(ArrowError::InvalidArgumentError(format!(
                "invalid distance metric '{other}': valid values are 'l2', 'cosine', 'dot'"
            )))),
        }
    }

    /// Compute the metric between a query and a candidate vector.
    ///
    /// The returned value is always "smaller is better".
    #[must_use]
    pub fn distance(self, query: &[f32], candidate: &[f32]) -> f32 {
        match self {
            Self::L2 => l2_distance(query, candidate),
            Self::Cosine => cosine_distance(query, candidate),
            Self::Dot => dot_distance(query, candidate),
        }
    }

    /// Stable string identifier for this metric, used when persisting it in
    /// dataset schema metadata. Round-trips through [`DistanceMetric::parse`].
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::Dot => "dot",
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

/// Valid column names that may use blob encoding.
const VALID_BLOB_COLUMNS: &[&str] = &["text_payload", "binary_payload"];

/// Open a store handle for the background compactor, with the recursion broken.
///
/// `ContextStore::open_inner` may spawn the background compaction task, and that
/// task needs to reopen the store — a mutual recursion between two `async fn`s.
/// Rust cannot infer `Send` for either of the resulting opaque future types, so
/// the task fails to compile even though the recursion is unreachable at runtime
/// (the reopen passes `spawn_background_compaction: false`).
///
/// Boxing to `dyn Future` here erases the type and cuts the cycle. It lives
/// outside the `impl` so the erasure is what the spawned task names, rather than
/// the opaque type it is trying to break out of.
fn open_for_compaction(
    uri: &str,
    options: ContextStoreOptions,
) -> Pin<Box<dyn Future<Output = LanceResult<ContextStore>> + Send + '_>> {
    Box::pin(ContextStore::open_inner(uri, options, false, false))
}

/// Persistent Lance-backed context store.
pub struct ContextStore {
    /// Shared storage layer: dataset handle, resident MemWAL writer, durable
    /// append, flush, WAL merge, compaction, indexing and the LSM scanner.
    ///
    /// Behind an `Arc` because `ContextStore` is `Clone` (the background
    /// compactor and the Python `fork` both rely on it) while `StorageBase`
    /// owns a resident `ShardWriter` and a `Drop` impl, so it must have exactly
    /// one owner. Sharing the base is also what makes clones agree about the
    /// resident writer instead of each opening their own.
    base: StorageBase,
    compaction_state: Arc<Mutex<CompactionState>>,
    pub compaction_config: CompactionConfig,
    blob_columns: HashSet<String>,
    id_index_type: IdIndexType,
    embedding_dim: i32,
    distance_metric: DistanceMetric,
}

/// Additional configuration when opening a [`ContextStore`].
///
/// [`Default`] is hand-written rather than derived because `seal_on_add`
/// must default to `true` (preserving read-your-write), which `derive(Default)`
/// would silently turn into `false`.
#[derive(Debug, Clone)]
pub struct ContextStoreOptions {
    pub storage_options: Option<HashMap<String, String>>,
    pub compaction: CompactionConfig,
    /// Width of the fixed-size embedding vector for newly-created datasets.
    /// Existing datasets always use the dimension persisted in their schema.
    pub embedding_dim: Option<i32>,
    /// Column names that should use Lance V1 blob encoding.
    /// Valid values: `"text_payload"`, `"binary_payload"`.
    pub blob_columns: HashSet<String>,
    /// Type of scalar index to create on the `id` column.
    pub id_index_type: IdIndexType,
    /// Distance metric used to rank vector-search results.
    ///
    /// For newly-created datasets this is persisted in the schema metadata and
    /// becomes the dataset's metric. For existing datasets the persisted metric
    /// is used; passing a different metric here is an error. `None` defaults to
    /// the persisted metric (or `L2` for datasets created before persistence).
    pub distance_metric: Option<DistanceMetric>,
    /// Stable identity of this writer instance, mapped to a MemWAL shard.
    ///
    /// Each instance owns exactly one shard, so no two instances ever contend
    /// and the epoch-fencing invariant holds by construction. `None` falls back
    /// to a single `"default"` shard, which is correct for single-instance use
    /// but must be set per-instance when running multiple writers.
    pub shard_id: Option<String>,
    /// Fold this instance's flushed MemWAL generations into the base table once
    /// it has accumulated this many. `None`/`0` disables the count trigger.
    ///
    /// Without a merge trigger, generations accumulate under `_mem_wal/` forever
    /// and every read unions all of them — the read amplification that
    /// previously had no bound at all on this store.
    pub merge_after_generations: Option<usize>,
    /// Whether [`ContextStore::add`] seals the memtable before returning, so the
    /// rows it wrote are immediately readable.
    ///
    /// Defaults to `true`, which is this store's historical behavior and is
    /// **required for correctness on the paths that read the table back**:
    /// id/external-id uniqueness validation, `upsert_by_external_id`, and
    /// tombstone-based deletes.
    ///
    /// Set `false` to get [`crate::RolloutStore`]'s throughput profile: `add`
    /// becomes a durable WAL append only, concurrent appends stop serializing
    /// behind a per-append seal, and one flushed generation per call is no
    /// longer emitted. In exchange there is **no read-your-write guarantee** —
    /// visibility then depends on [`ContextStore::flush`] being driven
    /// periodically. Only appropriate for trusted bulk appends that do not rely
    /// on the read-back paths above.
    pub seal_on_add: bool,
}

impl Default for ContextStoreOptions {
    fn default() -> Self {
        Self {
            storage_options: None,
            compaction: CompactionConfig::default(),
            embedding_dim: None,
            blob_columns: HashSet::new(),
            id_index_type: IdIndexType::default(),
            distance_metric: None,
            shard_id: None,
            merge_after_generations: None,
            // Read-your-write by default; see the field docs.
            seal_on_add: true,
        }
    }
}

impl ContextStoreOptions {
    #[must_use]
    pub fn storage_options(&self) -> Option<HashMap<String, String>> {
        self.storage_options.clone()
    }
}

fn relationship_struct_fields() -> Vec<Field> {
    vec![
        Field::new("target_id", DataType::Utf8, true),
        Field::new("relation", DataType::Utf8, true),
        Field::new("weight", DataType::Float32, true),
    ]
}

fn relationship_struct_data_type() -> DataType {
    DataType::Struct(relationship_struct_fields().into())
}

pub(crate) fn relationship_list_item_field() -> FieldRef {
    Arc::new(Field::new("item", relationship_struct_data_type(), true))
}

pub(crate) fn relationship_field() -> Field {
    Field::new(
        RELATIONSHIPS_COLUMN,
        DataType::List(relationship_list_item_field()),
        true,
    )
}

pub(crate) fn relationship_struct_builder() -> StructBuilder {
    let fields: Vec<FieldRef> = relationship_struct_fields()
        .into_iter()
        .map(|field| Arc::new(field) as FieldRef)
        .collect();
    StructBuilder::new(
        fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(StringBuilder::new()),
            Box::new(Float32Builder::new()),
        ],
    )
}

/// Per-`external_id` resolution computed in a single scan for batch upsert.
#[derive(Default)]
struct ExternalIdState {
    /// Ids of currently-visible records carrying this external_id (default
    /// lifecycle: not tombstoned/expired/retired/superseded).
    visible_ids: Vec<String>,
    /// Whether any non-tombstone row (visible or hidden) carries it. Mirrors
    /// the uniqueness check `add` performs on the single-upsert insert path.
    has_non_tombstone: bool,
}

/// Which large/optional payload columns to load on a read.
///
/// Excluding `binary` (and optionally `text` / `embedding`) lets metadata and
/// search queries avoid materializing large media bytes; the omitted fields
/// come back as `None`. Fetch a single record's bytes on demand with
/// [`ContextStore::get_blob`]. Defaults to loading everything (backward
/// compatible).
#[derive(Debug, Clone, Copy)]
pub struct ReadProjection {
    pub text: bool,
    pub binary: bool,
    pub embedding: bool,
}

impl Default for ReadProjection {
    fn default() -> Self {
        Self {
            text: true,
            binary: true,
            embedding: true,
        }
    }
}

impl ReadProjection {
    /// Load only scalar/metadata columns (no text, binary, or embedding).
    #[must_use]
    pub fn metadata_only() -> Self {
        Self {
            text: false,
            binary: false,
            embedding: false,
        }
    }

    /// Load everything except the (potentially large) `binary_payload`.
    #[must_use]
    pub fn without_binary() -> Self {
        Self {
            binary: false,
            ..Self::default()
        }
    }

    fn loads_all(self) -> bool {
        self.text && self.binary && self.embedding
    }
}

impl ContextStore {
    /// Open an existing context dataset or create a new one with the project schema.
    pub async fn open(uri: &str) -> LanceResult<Self> {
        Self::open_with_options(uri, ContextStoreOptions::default()).await
    }

    /// Open a dataset with explicit object store configuration (e.g. S3 credentials).
    /// Creates the dataset if it does not exist.
    pub async fn open_with_options(uri: &str, options: ContextStoreOptions) -> LanceResult<Self> {
        Self::open_inner(uri, options, true, true).await
    }

    /// Open an **existing** context dataset. Unlike [`Self::open_with_options`],
    /// this does **not** create the dataset when it is absent — it returns the
    /// underlying [`LanceError::DatasetNotFound`] instead.
    ///
    /// Used by the server's lazy cache-fill path: a cache miss must load a store
    /// that genuinely exists on object storage. Create-on-absence would silently
    /// materialize an empty context for a mistyped name, masking the 404. Store
    /// creation goes exclusively through [`Self::open_with_options`].
    pub async fn open_existing_with_options(
        uri: &str,
        options: ContextStoreOptions,
    ) -> LanceResult<Self> {
        Self::open_inner(uri, options, false, true).await
    }

    async fn open_inner(
        uri: &str,
        options: ContextStoreOptions,
        create_if_missing: bool,
        spawn_background_compaction: bool,
    ) -> LanceResult<Self> {
        // Validate blob_columns
        for col in &options.blob_columns {
            if !VALID_BLOB_COLUMNS.contains(&col.as_str()) {
                return Err(LanceError::from(ArrowError::InvalidArgumentError(format!(
                    "invalid blob column '{}': valid columns are {:?}",
                    col, VALID_BLOB_COLUMNS
                ))));
            }
        }

        let requested_embedding_dim = match options.embedding_dim {
            Some(dim) => {
                validate_embedding_dim(dim)?;
                dim
            }
            None => DEFAULT_EMBEDDING_DIM,
        };
        let storage_options = options.storage_options();
        let blob_columns = options.blob_columns.clone();
        let (dataset, created) = match Self::load_with_options(uri, storage_options.clone()).await {
            Ok(dataset) => (dataset, false),
            Err(LanceError::DatasetNotFound { .. }) if create_if_missing => {
                let dataset = Self::create_with_options(
                    uri,
                    storage_options.clone(),
                    &blob_columns,
                    requested_embedding_dim,
                    options.distance_metric.unwrap_or_default(),
                )
                .await?;
                (dataset, true)
            }
            Err(err) => return Err(err),
        };
        let arrow_schema: Schema = dataset.schema().into();
        let embedding_dim = embedding_dim_from_schema(&arrow_schema)?;
        if !created && options.embedding_dim.is_some() && embedding_dim != requested_embedding_dim {
            return Err(LanceError::from(ArrowError::InvalidArgumentError(format!(
                "existing context embedding dimension {} does not match requested dimension {}",
                embedding_dim, requested_embedding_dim
            ))));
        }
        let distance_metric = distance_metric_from_schema(&arrow_schema)?;
        if !created {
            if let Some(requested) = options.distance_metric {
                if requested != distance_metric {
                    return Err(LanceError::from(ArrowError::InvalidArgumentError(format!(
                        "existing context distance metric '{}' does not match requested metric '{}'",
                        distance_metric.as_str(),
                        requested.as_str()
                    ))));
                }
            }
        }

        let base = StorageBase::from_dataset(
            dataset,
            StorageBaseOptions {
                storage_options,
                shard_id: options.shard_id.clone(),
                merge_after_generations: options.merge_after_generations,
                session: None,
                schema: Arc::new(arrow_schema.clone()),
                key_column: "id".to_string(),
                // The context schema is parameterized per dataset (blob columns,
                // embedding width, optional feature columns), so there is no
                // single "latest" schema to evolve an older table to. Schema
                // migration stays explicit, via `migrate_relationships_column`.
                latest_schema: None,
                // Context writes read the table back — id/external-id uniqueness
                // validation, upsert, and tombstones all need read-your-write —
                // so seal by default. Callers appending trusted batches can opt
                // into rollout's deferred-seal throughput instead.
                seal_on_put: options.seal_on_add,
            },
        )
        .await?;

        let mut store = Self {
            base,
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
            embedding_dim,
            distance_metric,
        };

        // Ensure id index if configured
        store.ensure_id_index().await?;

        // Start background compaction if enabled.
        //
        // Skipped when this open *is* the background compactor reopening the
        // store: that path would otherwise re-enter here and spawn another
        // compactor, making the recursion type-level (the future can never be
        // proven `Send`) rather than merely infinite.
        if spawn_background_compaction {
            store.start_background_compaction().await?;
        }

        Ok(store)
    }

    /// Embedding vector width persisted in this context dataset schema.
    #[must_use]
    pub fn embedding_dim(&self) -> i32 {
        self.embedding_dim
    }

    /// URI of the underlying Lance dataset.
    #[must_use]
    pub fn uri(&self) -> &str {
        self.base.dataset.uri()
    }

    /// Distance metric this context ranks vector-search results with.
    #[must_use]
    pub fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    /// Append context records to the store and return the new dataset version.
    pub async fn add(&self, entries: &[ContextRecord]) -> LanceResult<u64> {
        if entries.is_empty() {
            return Ok(self.base.version());
        }

        self.validate_unique_ids(entries).await?;
        self.write_entries(entries).await
    }

    /// Encode and append `entries` through the shared storage layer.
    ///
    /// Takes `&self`: the resident writer lives behind the base's mutex, so
    /// appends no longer need exclusive access. Whether the rows are readable
    /// when this returns is governed by
    /// [`ContextStoreOptions::seal_on_add`] (default `true`).
    async fn write_entries(&self, entries: &[ContextRecord]) -> LanceResult<u64> {
        if entries.is_empty() {
            return Ok(self.base.version());
        }
        let batch = self.records_to_batch(entries)?;
        self.base.put(vec![batch]).await?;
        Ok(self.base.version())
    }

    /// Resolve a record's external payload reference to its bytes on demand.
    ///
    /// Records may carry a typed [`ContextRecord::payload_uri`] pointing at media
    /// stored outside the dataset (e.g. `gs://bucket/prefix/<id>`); `list`/`search`
    /// return that reference without materializing the bytes. This opt-in fetch
    /// resolves them using the context's configured `storage_options`, so it works
    /// for `gs://`, `s3://`, and local paths through the same object-store path the
    /// dataset itself uses.
    ///
    /// Returns `Ok(None)` if no record with `id` exists. Returns an error if the
    /// record exists but carries no external payload reference.
    // TODO(#115): offer a signed-URL variant (`fetch_payload_url`) where the
    // backend supports presigning, instead of always streaming the bytes back.
    pub async fn fetch_payload(&self, id: &str) -> LanceResult<Option<Vec<u8>>> {
        // Use the list-backed accessor so freshly written (MemWAL-buffered) rows
        // and lifecycle visibility are handled exactly like every other read.
        let Some(record) = self.get_by_id(id).await? else {
            return Ok(None);
        };
        let Some(uri) = record.payload_uri.as_deref() else {
            return Err(LanceError::from(ArrowError::InvalidArgumentError(format!(
                "record '{id}' has no external payload reference to fetch"
            ))));
        };
        let registry = Arc::new(ObjectStoreRegistry::default());
        let (store, path) =
            ObjectStore::from_uri_and_params(registry, uri, &self.payload_store_params()).await?;
        let bytes = store.read_one_all(&path).await?;
        Ok(Some(bytes.to_vec()))
    }

    /// Offload caller-provided bytes to an object at `uri` using the context's
    /// configured `storage_options`, returning the number of bytes written.
    ///
    /// Pairs with [`ContextStore::fetch_payload`]: write the media object here,
    /// then `add` a record whose `payload_uri` points at `uri`. Inline
    /// [`ContextRecord::binary_payload`] remains the small-payload path.
    pub async fn put_payload(&self, uri: &str, bytes: &[u8]) -> LanceResult<u64> {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let (store, path) =
            ObjectStore::from_uri_and_params(registry, uri, &self.payload_store_params()).await?;
        store.put(&path, bytes).await?;
        Ok(bytes.len() as u64)
    }

    /// Object-store parameters threading the context's `storage_options` so the
    /// same credentials/endpoint apply when resolving external payload URIs.
    fn payload_store_params(&self) -> ObjectStoreParams {
        let mut params = ObjectStoreParams::default();
        if let Some(options) = &self.base.storage_options {
            params.storage_options_accessor = Some(Arc::new(
                StorageOptionsAccessor::with_static_options(options.clone()),
            ));
        }
        params
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

    /// Insert a record or replace the currently-visible record with the same external id.
    ///
    /// Replacement is append-only: the new record keeps the same `external_id`
    /// and gets `supersedes_id` set to the old record id. Default reads hide
    /// the superseded record while `include_retired` reads can still inspect
    /// both versions. Caller-supplied supersession fields are ignored because
    /// this method manages replacement by `external_id`.
    pub async fn upsert_by_external_id(
        &mut self,
        mut record: ContextRecord,
    ) -> LanceResult<UpsertResult> {
        let Some(external_id) = record.external_id.clone() else {
            return Err(ArrowError::InvalidArgumentError(
                "upsert_by_external_id requires external_id".to_string(),
            )
            .into());
        };
        if external_id.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "upsert_by_external_id requires a non-empty external_id".to_string(),
            )
            .into());
        }
        if record.is_tombstone() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "content_type '{}' is reserved for internal tombstones",
                CONTENT_TYPE_TOMBSTONE
            ))
            .into());
        }
        record.supersedes_id = None;
        record.superseded_by_id = None;
        self.validate_new_record_id(&record).await?;

        let matches: Vec<ContextRecord> = self
            .list(None, None)
            .await?
            .into_iter()
            .filter(|existing| existing.external_id.as_deref() == Some(external_id.as_str()))
            .collect();

        match matches.as_slice() {
            [] => {
                let version = self.add(std::slice::from_ref(&record)).await?;
                Ok(UpsertResult {
                    record,
                    inserted: true,
                    replaced_id: None,
                    version,
                })
            }
            [existing] => {
                record.supersedes_id = Some(existing.id.clone());
                let version = self.write_entries(std::slice::from_ref(&record)).await?;
                Ok(UpsertResult {
                    record,
                    inserted: false,
                    replaced_id: Some(existing.id.clone()),
                    version,
                })
            }
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "external_id '{}' matches multiple visible records",
                external_id
            ))
            .into()),
        }
    }

    /// Insert-or-replace a batch of records keyed by `external_id`, in one
    /// logical operation.
    ///
    /// For each record: if a currently-visible record with the same
    /// `external_id` exists, it is replaced append-only (the successor gets
    /// `supersedes_id` set to the existing record id and the original is hidden
    /// from default reads); otherwise the record is inserted. All rows are
    /// written in a single pass, so records sharing a shard land in a single
    /// version bump.
    ///
    /// Semantics (parity with [`Self::upsert_by_external_id`] and `add_many`):
    /// - every record must carry a non-empty `external_id`;
    /// - duplicate `id`s or `external_id`s *within* the batch are rejected;
    /// - validation is all-or-nothing — if any record is invalid, nothing is
    ///   written;
    /// - caller-supplied supersession fields are ignored (replacement is
    ///   managed by `external_id`);
    /// - an insert whose `external_id` already exists on a non-tombstone but
    ///   hidden record is rejected, exactly as a single insert would be.
    ///
    /// Existing-key resolution and `id` uniqueness validation are each done in
    /// a single scan for the whole batch (not per record), composing with the
    /// indexed `id` validation so a batch does not full-scan per record.
    ///
    /// Returns one [`UpsertResult`] per input record, in input order, all
    /// carrying the final dataset version.
    pub async fn upsert_many_by_external_id(
        &mut self,
        mut records: Vec<ContextRecord>,
    ) -> LanceResult<Vec<UpsertResult>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Per-record validation + within-batch duplicate detection.
        let mut seen_ids: HashSet<&str> = HashSet::with_capacity(records.len());
        let mut seen_external_ids: HashSet<&str> = HashSet::with_capacity(records.len());
        for record in &records {
            let Some(external_id) = record.external_id.as_deref() else {
                return Err(ArrowError::InvalidArgumentError(
                    "upsert_many_by_external_id requires external_id on every record".to_string(),
                )
                .into());
            };
            if external_id.is_empty() {
                return Err(ArrowError::InvalidArgumentError(
                    "upsert_many_by_external_id requires a non-empty external_id".to_string(),
                )
                .into());
            }
            if record.is_tombstone() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "content_type '{}' is reserved for internal tombstones",
                    CONTENT_TYPE_TOMBSTONE
                ))
                .into());
            }
            if !seen_ids.insert(record.id.as_str()) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "duplicate id '{}' in batch",
                    record.id
                ))
                .into());
            }
            if !seen_external_ids.insert(external_id) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "duplicate external_id '{}' in batch",
                    external_id
                ))
                .into());
            }
        }

        // 2. Replacement is managed here; ignore caller-supplied supersession.
        for record in &mut records {
            record.supersedes_id = None;
            record.superseded_by_id = None;
        }

        // 3. id uniqueness against the store (indexed). external_id is NOT
        //    rejected here: an existing external_id means "replace".
        let id_list: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
        let (existing_ids, _) = self.find_existing_keys(&id_list, &[]).await?;
        if let Some(record) = records
            .iter()
            .find(|r| existing_ids.contains(r.id.as_str()))
        {
            return Err(ArrowError::InvalidArgumentError(format!(
                "id '{}' already exists",
                record.id
            ))
            .into());
        }

        // 4. Resolve every external_id to its visible record (if any) in one scan.
        let external_id_list: Vec<&str> = records
            .iter()
            .map(|r| r.external_id.as_deref().unwrap_or_default())
            .collect();
        let states = self.external_id_states(&external_id_list).await?;

        // 5. Wire supersession + per-record outcomes, mirroring the single path.
        let mut outcomes: Vec<(bool, Option<String>)> = Vec::with_capacity(records.len());
        for record in &mut records {
            let external_id = record.external_id.as_deref().unwrap_or_default();
            match states.get(external_id) {
                Some(state) if state.visible_ids.len() > 1 => {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "external_id '{}' matches multiple visible records",
                        external_id
                    ))
                    .into());
                }
                Some(state) if state.visible_ids.len() == 1 => {
                    let existing_id = state.visible_ids[0].clone();
                    record.supersedes_id = Some(existing_id.clone());
                    outcomes.push((false, Some(existing_id)));
                }
                Some(state) if state.has_non_tombstone => {
                    // No visible record, but a hidden non-tombstone row already
                    // holds this external_id — an insert would collide, exactly
                    // as the single-record insert path (via `add`) rejects it.
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "external_id '{}' already exists",
                        external_id
                    ))
                    .into());
                }
                _ => outcomes.push((true, None)),
            }
        }

        // 6. Single write for the whole batch.
        let version = self.write_entries(&records).await?;

        Ok(records
            .into_iter()
            .zip(outcomes)
            .map(|(record, (inserted, replaced_id))| UpsertResult {
                record,
                inserted,
                replaced_id,
                version,
            })
            .collect())
    }

    /// Resolve, for each candidate `external_id`, the set of currently-visible
    /// record ids and whether any non-tombstone row carries it — in a single
    /// projected, filtered scan rather than a full dataset list.
    ///
    /// A record that supersedes another keeps the same `external_id`, so the
    /// supersession relation is resolved correctly within the filtered set for
    /// every flow that creates supersession through the public API.
    async fn external_id_states(
        &self,
        external_ids: &[&str],
    ) -> LanceResult<HashMap<String, ExternalIdState>> {
        let mut states: HashMap<String, ExternalIdState> = HashMap::new();
        let candidates: HashSet<&str> = external_ids
            .iter()
            .copied()
            .filter(|value| !value.is_empty())
            .collect();
        if candidates.is_empty() {
            return Ok(states);
        }

        let filter_values: Vec<&str> = candidates.iter().copied().collect();
        let filter = format!("external_id IN ({})", sql_quoted_list(&filter_values));
        let scanner = self.lsm_scanner().await?.filter(&filter)?;
        let mut stream = scanner.try_into_stream().await?;
        let mut rows: Vec<ContextRecord> = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            rows.extend(batch_to_records(&batch)?);
        }

        let superseded_ids: HashSet<String> = rows
            .iter()
            .filter_map(|record| {
                let supersedes_id = record.supersedes_id.as_ref()?;
                if supersedes_id == &record.id {
                    None
                } else {
                    Some(supersedes_id.clone())
                }
            })
            .collect();

        let options = LifecycleQueryOptions::default();
        for record in rows {
            let Some(external_id) = record.external_id.as_deref() else {
                continue;
            };
            if !candidates.contains(external_id) {
                continue;
            }
            let entry = states.entry(external_id.to_string()).or_default();
            if !record.is_tombstone() {
                entry.has_non_tombstone = true;
            }
            if options.is_visible(&record) && !superseded_ids.contains(&record.id) {
                entry.visible_ids.push(record.id);
            }
        }

        Ok(states)
    }

    /// Partially update mutable fields on a visible record by internal id.
    ///
    /// The update is append-only: it writes a replacement record that
    /// supersedes the current visible record, preserving the original payload
    /// and embedding while changing only the requested patch fields.
    pub async fn update_by_id(
        &mut self,
        id: &str,
        patch: RecordPatch,
    ) -> LanceResult<Option<UpdateResult>> {
        if id.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "update_by_id requires a non-empty id".to_string(),
            )
            .into());
        }
        let Some(existing) = self.get_by_id(id).await? else {
            return Ok(None);
        };
        self.update_visible_record(existing, patch).await.map(Some)
    }

    /// Partially update mutable fields on a visible record by external id.
    ///
    /// Returns `Ok(None)` when no visible record currently has the external id.
    pub async fn update_by_external_id(
        &mut self,
        external_id: &str,
        patch: RecordPatch,
    ) -> LanceResult<Option<UpdateResult>> {
        if external_id.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "update_by_external_id requires a non-empty external_id".to_string(),
            )
            .into());
        }

        let matches: Vec<ContextRecord> = self
            .list(None, None)
            .await?
            .into_iter()
            .filter(|existing| existing.external_id.as_deref() == Some(external_id))
            .collect();

        match matches.as_slice() {
            [] => Ok(None),
            [existing] => self
                .update_visible_record(existing.clone(), patch)
                .await
                .map(Some),
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "external_id '{}' matches multiple visible records",
                external_id
            ))
            .into()),
        }
    }

    async fn update_visible_record(
        &mut self,
        existing: ContextRecord,
        patch: RecordPatch,
    ) -> LanceResult<UpdateResult> {
        if patch.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "update requires at least one patch field".to_string(),
            )
            .into());
        }

        let mut record = existing.clone();
        record.id = Uuid::new_v4().to_string();
        record.run_id = Uuid::new_v4().to_string();
        record.created_at = Utc::now();
        record.supersedes_id = Some(existing.id.clone());
        record.superseded_by_id = None;

        if let Some(bot_id) = patch.bot_id {
            record.bot_id = Some(bot_id);
        }
        if let Some(session_id) = patch.session_id {
            record.session_id = Some(session_id);
        }
        if let Some(tenant) = patch.tenant {
            record.tenant = Some(tenant);
        }
        if let Some(source) = patch.source {
            record.source = Some(source);
        }
        if let Some(state_metadata) = patch.state_metadata {
            record.state_metadata = Some(state_metadata);
        }
        if let Some(metadata) = patch.metadata {
            record.metadata = Some(metadata);
        }
        if let Some(relationships) = patch.relationships {
            record.relationships = relationships;
        }
        if let Some(expires_at) = patch.expires_at {
            record.expires_at = Some(expires_at);
        }
        if let Some(retention_policy) = patch.retention_policy {
            record.retention_policy = Some(retention_policy);
        }
        if let Some(lifecycle_status) = patch.lifecycle_status {
            record.lifecycle_status = lifecycle_status;
        }
        if let Some(retired_at) = patch.retired_at {
            record.retired_at = Some(retired_at);
        }
        if let Some(retired_reason) = patch.retired_reason {
            record.retired_reason = Some(retired_reason);
        }
        if let Some(embedding) = patch.embedding {
            record.embedding = Some(embedding);
        }
        if let Some(payload_uri) = patch.payload_uri {
            record.payload_uri = Some(payload_uri);
        }
        if let Some(payload_size) = patch.payload_size {
            record.payload_size = Some(payload_size);
        }
        if let Some(payload_checksum) = patch.payload_checksum {
            record.payload_checksum = Some(payload_checksum);
        }

        self.validate_new_record_id(&record).await?;
        let version = self.write_entries(std::slice::from_ref(&record)).await?;
        Ok(UpdateResult {
            record,
            replaced_id: existing.id,
            version,
        })
    }

    async fn write_tombstone_for(&mut self, record: ContextRecord) -> LanceResult<u64> {
        let tombstone = ContextRecord {
            id: record.id,
            external_id: record.external_id,
            run_id: record.run_id,
            bot_id: record.bot_id,
            session_id: record.session_id,
            tenant: record.tenant,
            source: record.source,
            created_at: Utc::now(),
            role: record.role,
            state_metadata: None,
            metadata: None,
            relationships: Vec::new(),
            expires_at: None,
            retention_policy: None,
            lifecycle_status: LIFECYCLE_ACTIVE.to_string(),
            retired_at: None,
            retired_reason: None,
            supersedes_id: None,
            superseded_by_id: None,
            content_type: CONTENT_TYPE_TOMBSTONE.to_string(),
            text_payload: None,
            binary_payload: None,
            payload_uri: None,
            payload_size: None,
            payload_checksum: None,
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

        let id_list: Vec<&str> = ids.iter().copied().collect();
        let external_id_list: Vec<&str> = external_ids.iter().copied().collect();
        let (existing_ids, existing_external_ids) =
            self.find_existing_keys(&id_list, &external_id_list).await?;

        // Report collisions in input order for deterministic, intuitive errors.
        for entry in entries {
            if existing_ids.contains(entry.id.as_str()) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "id '{}' already exists",
                    entry.id
                ))
                .into());
            }
            if let Some(external_id) = &entry.external_id {
                if existing_external_ids.contains(external_id.as_str()) {
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

    async fn validate_new_record_id(&self, entry: &ContextRecord) -> LanceResult<()> {
        let id = entry.id.as_str();
        let (existing_ids, _) = self.find_existing_keys(&[id], &[]).await?;
        if existing_ids.contains(id) {
            return Err(ArrowError::InvalidArgumentError(format!(
                "id '{}' already exists",
                entry.id
            ))
            .into());
        }
        Ok(())
    }

    /// Resolve which of the candidate `id` / `external_id` values already exist
    /// among non-tombstone records.
    ///
    /// Unlike a full `list` + deserialize, this issues projected, filtered
    /// scans that read only the `id` / `external_id` / `content_type` columns
    /// for rows matching the candidate keys. When an `id` scalar index is
    /// configured the `id` lookup is index-accelerated, so uniqueness
    /// validation cost is bounded by the candidate batch (and index
    /// selectivity) rather than the total number of stored records — history,
    /// retired, expired, and superseded rows included.
    ///
    /// Tombstones are skipped to preserve existing behavior: a key whose only
    /// surviving row is a tombstone is free for reuse, while a
    /// superseded/retired/expired (non-tombstone) row still reserves its key.
    async fn find_existing_keys(
        &self,
        ids: &[&str],
        external_ids: &[&str],
    ) -> LanceResult<(HashSet<String>, HashSet<String>)> {
        let mut existing_ids = HashSet::new();
        let mut existing_external_ids = HashSet::new();

        let candidate_ids: HashSet<&str> = ids.iter().copied().collect();
        let candidate_external_ids: HashSet<&str> = external_ids.iter().copied().collect();

        if !candidate_ids.is_empty() {
            let filter = format!("id IN ({})", sql_quoted_list(ids));
            let scanner = self
                .lsm_scanner()
                .await?
                .project(&["id", "content_type"])
                .filter(&filter)?;
            let mut stream = scanner.try_into_stream().await?;
            while let Some(batch) = stream.try_next().await? {
                let id_array = column_as::<StringArray>(&batch, "id")?;
                let content_type_array = column_as::<StringArray>(&batch, "content_type")?;
                for row in 0..batch.num_rows() {
                    if content_type_array.value(row) == CONTENT_TYPE_TOMBSTONE {
                        continue;
                    }
                    let id = id_array.value(row);
                    if candidate_ids.contains(id) {
                        existing_ids.insert(id.to_string());
                    }
                }
            }
        }

        if !candidate_external_ids.is_empty() && self.has_external_id_column() {
            let filter = format!("external_id IN ({})", sql_quoted_list(external_ids));
            let scanner = self
                .lsm_scanner()
                .await?
                .project(&["external_id", "content_type"])
                .filter(&filter)?;
            let mut stream = scanner.try_into_stream().await?;
            while let Some(batch) = stream.try_next().await? {
                let content_type_array = column_as::<StringArray>(&batch, "content_type")?;
                let Some(external_id_array) =
                    column_as_optional::<StringArray>(&batch, "external_id")
                else {
                    continue;
                };
                for row in 0..batch.num_rows() {
                    if content_type_array.value(row) == CONTENT_TYPE_TOMBSTONE {
                        continue;
                    }
                    if external_id_array.is_null(row) {
                        continue;
                    }
                    let external_id = external_id_array.value(row);
                    if candidate_external_ids.contains(external_id) {
                        existing_external_ids.insert(external_id.to_string());
                    }
                }
            }
        }

        Ok((existing_ids, existing_external_ids))
    }

    fn has_relationships_column(&self) -> bool {
        self.base
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == RELATIONSHIPS_COLUMN)
    }

    fn has_external_id_column(&self) -> bool {
        self.base
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == "external_id")
    }

    /// Current dataset version.
    pub fn version(&self) -> u64 {
        self.base.version()
    }

    /// Add the relationships column to an older dataset if it is missing.
    ///
    /// Existing rows are stored as null in the new column and read back as an
    /// empty relationship list.
    pub async fn migrate_relationships_column(&mut self) -> LanceResult<bool> {
        if self.has_relationships_column() {
            return Ok(false);
        }

        let schema = Arc::new(Schema::new(vec![relationship_field()]));
        self.base
            .dataset
            .add_columns(NewColumnTransform::AllNulls(schema), None, None)
            .await?;
        Ok(true)
    }

    /// Checkout a specific dataset version.
    pub async fn checkout(&mut self, version_id: u64) -> LanceResult<()> {
        let dataset = self.base.dataset.checkout_version(version_id).await?;
        self.base.dataset = dataset;
        Ok(())
    }

    /// Retrieve a single record by its unique ID.
    pub async fn get(&self, id: &str) -> LanceResult<Option<ContextRecord>> {
        let escaped_id = id.replace('\'', "''");
        let mut scanner = self.base.dataset.scan();
        scanner.filter(&format!("id = '{}'", escaped_id))?;
        scanner.limit(Some(1), None)?;

        let mut stream = scanner.try_into_stream().await?;
        if let Some(batch) = stream.try_next().await? {
            let records = batch_to_records(&batch)?;
            return Ok(records.into_iter().next());
        }
        Ok(None)
    }

    /// List all records in the dataset.
    pub async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> LanceResult<Vec<ContextRecord>> {
        self.list_filtered_with_options(limit, offset, None, LifecycleQueryOptions::default())
            .await
    }

    /// List records matching filters.
    pub async fn list_filtered(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<&RecordFilters>,
    ) -> LanceResult<Vec<ContextRecord>> {
        self.list_filtered_with_options(limit, offset, filters, LifecycleQueryOptions::default())
            .await
    }

    /// List records, applying lifecycle visibility and supersession before offset/limit.
    pub async fn list_with_options(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        options: LifecycleQueryOptions,
    ) -> LanceResult<Vec<ContextRecord>> {
        self.list_filtered_with_options(limit, offset, None, options)
            .await
    }

    /// List records matching filters, applying lifecycle visibility before offset/limit.
    pub async fn list_filtered_with_options(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<&RecordFilters>,
        options: LifecycleQueryOptions,
    ) -> LanceResult<Vec<ContextRecord>> {
        self.list_filtered_projected(limit, offset, filters, options, ReadProjection::default())
            .await
    }

    /// Like [`Self::list_filtered_with_options`] but with column projection, so
    /// large payload columns can be skipped (see [`ReadProjection`]). Omitted
    /// payloads come back as `None`; fetch bytes on demand via
    /// [`Self::get_blob`].
    pub async fn list_filtered_projected(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<&RecordFilters>,
        options: LifecycleQueryOptions,
        projection: ReadProjection,
    ) -> LanceResult<Vec<ContextRecord>> {
        let scanner = self.lsm_scanner_projected(projection).await?;
        let mut stream = scanner.try_into_stream().await?;
        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            results.extend(batch_to_records(&batch)?);
        }

        let superseded_ids: HashSet<String> = results
            .iter()
            .filter_map(|record| {
                let supersedes_id = record.supersedes_id.as_ref()?;
                if supersedes_id == &record.id {
                    None
                } else {
                    Some(supersedes_id.clone())
                }
            })
            .collect();
        results.retain(|record| {
            options.is_visible(record)
                && (options.include_retired || !superseded_ids.contains(&record.id))
        });
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

    /// List records that have a relationship targeting `target_id`.
    pub async fn list_related(
        &self,
        target_id: &str,
        relation: Option<&str>,
        limit: Option<usize>,
    ) -> LanceResult<Vec<ContextRecord>> {
        self.list_related_with_options(target_id, relation, limit, LifecycleQueryOptions::default())
            .await
    }

    /// List related records, applying lifecycle visibility before relationship matching.
    pub async fn list_related_with_options(
        &self,
        target_id: &str,
        relation: Option<&str>,
        limit: Option<usize>,
        options: LifecycleQueryOptions,
    ) -> LanceResult<Vec<ContextRecord>> {
        let mut results: Vec<ContextRecord> = self
            .list_with_options(None, None, options)
            .await?
            .into_iter()
            .filter(|record| {
                record.relationships.iter().any(|relationship| {
                    relationship.target_id == target_id
                        && relation.is_none_or(|value| relationship.relation == value)
                })
            })
            .collect();

        if let Some(limit) = limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    /// Perform a nearest-neighbor search over stored embeddings.
    pub async fn search(
        &self,
        query: &[f32],
        limit: Option<usize>,
    ) -> LanceResult<Vec<SearchResult>> {
        self.search_filtered_with_options(query, limit, None, LifecycleQueryOptions::default())
            .await
    }

    /// Perform a nearest-neighbor search over stored embeddings matching filters.
    pub async fn search_filtered(
        &self,
        query: &[f32],
        limit: Option<usize>,
        filters: Option<&RecordFilters>,
    ) -> LanceResult<Vec<SearchResult>> {
        self.search_filtered_with_options(query, limit, filters, LifecycleQueryOptions::default())
            .await
    }

    /// Perform nearest-neighbor search after applying lifecycle visibility.
    pub async fn search_with_options(
        &self,
        query: &[f32],
        limit: Option<usize>,
        options: LifecycleQueryOptions,
    ) -> LanceResult<Vec<SearchResult>> {
        self.search_filtered_with_options(query, limit, None, options)
            .await
    }

    /// Perform nearest-neighbor search after applying filters and lifecycle visibility.
    pub async fn search_filtered_with_options(
        &self,
        query: &[f32],
        limit: Option<usize>,
        filters: Option<&RecordFilters>,
        options: LifecycleQueryOptions,
    ) -> LanceResult<Vec<SearchResult>> {
        self.search_filtered_projected(query, limit, filters, options, ReadProjection::default())
            .await
    }

    /// Like [`Self::search_filtered_with_options`] but with column projection on
    /// the returned records (see [`ReadProjection`]). Embeddings are always read
    /// internally to score, then dropped from the results if `projection.embedding`
    /// is `false`.
    pub async fn search_filtered_projected(
        &self,
        query: &[f32],
        limit: Option<usize>,
        filters: Option<&RecordFilters>,
        options: LifecycleQueryOptions,
        projection: ReadProjection,
    ) -> LanceResult<Vec<SearchResult>> {
        validate_query_dimension(query, self.embedding_dim)?;

        let top_k = limit.unwrap_or(DEFAULT_SEARCH_LIMIT);
        if top_k == 0 {
            return Ok(Vec::new());
        }

        // Embedding is required to score; force it on for the scan but honor the
        // caller's text/binary choices.
        let scan_projection = ReadProjection {
            embedding: true,
            ..projection
        };
        let mut results: Vec<SearchResult> = self
            .list_filtered_projected(None, None, filters, options, scan_projection)
            .await?
            .into_iter()
            .filter_map(|mut record| {
                let distance = self
                    .distance_metric
                    .distance(query, record.embedding.as_ref()?);
                if !projection.embedding {
                    record.embedding = None;
                }
                Some(SearchResult { record, distance })
            })
            .collect();
        results.sort_by(|left, right| left.distance.total_cmp(&right.distance));
        results.truncate(top_k);
        Ok(results)
    }

    /// Retrieve records using optional text and vector channels, after filters and lifecycle visibility.
    pub async fn retrieve_filtered_with_options(
        &self,
        text: Option<&str>,
        vector: Option<&[f32]>,
        limit: Option<usize>,
        filters: Option<&RecordFilters>,
        options: LifecycleQueryOptions,
    ) -> LanceResult<Vec<RetrieveResult>> {
        let text_terms = text.map(unique_query_terms).unwrap_or_default();
        let has_text = !text_terms.is_empty();

        if !has_text && vector.is_none() {
            return Err(ArrowError::InvalidArgumentError(
                "retrieve requires text or vector".to_string(),
            )
            .into());
        }

        if let Some(query) = vector {
            validate_query_dimension(query, self.embedding_dim)?;
        }

        let top_k = limit.unwrap_or(DEFAULT_SEARCH_LIMIT);
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let records = self
            .list_filtered_with_options(None, None, filters, options)
            .await?;
        let mut candidates: HashMap<String, RetrieveResult> = HashMap::new();

        if let Some(query) = vector {
            let mut vector_hits: Vec<(usize, f32)> = records
                .iter()
                .enumerate()
                .filter_map(|(index, record)| {
                    let distance = self
                        .distance_metric
                        .distance(query, record.embedding.as_ref()?);
                    Some((index, distance))
                })
                .collect();
            vector_hits.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| records[left.0].id.cmp(&records[right.0].id))
            });

            for (rank, (index, distance)) in vector_hits.into_iter().enumerate() {
                add_retrieve_channel(
                    &mut candidates,
                    &records[index],
                    rank + 1,
                    "vector",
                    Some(distance),
                    None,
                );
            }
        }

        if has_text {
            let mut text_hits: Vec<(usize, f32)> = records
                .iter()
                .enumerate()
                .filter_map(|(index, record)| {
                    lexical_score(&text_terms, record.text_payload.as_deref())
                        .map(|score| (index, score))
                })
                .collect();
            text_hits.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| records[left.0].id.cmp(&records[right.0].id))
            });

            for (rank, (index, score)) in text_hits.into_iter().enumerate() {
                add_retrieve_channel(
                    &mut candidates,
                    &records[index],
                    rank + 1,
                    "text",
                    None,
                    Some(score),
                );
            }
        }

        let mut results: Vec<RetrieveResult> = candidates.into_values().collect();
        results.sort_by(compare_retrieve_results);
        results.truncate(top_k);
        Ok(results)
    }

    /// LSM scanner over the base table unioned with every shard's flushed
    /// MemWAL generations, deduplicating by `id`.
    async fn lsm_scanner(&self) -> LanceResult<LsmScanner> {
        self.base.lsm_scanner().await
    }

    /// Top-level column names to read for a projection (drops the excluded
    /// payload columns; everything else is always loaded so lifecycle
    /// filtering and metadata stay correct).
    fn projected_columns(&self, projection: ReadProjection) -> Vec<String> {
        self.base
            .dataset
            .schema()
            .fields
            .iter()
            .map(|field| field.name.clone())
            .filter(|name| {
                (projection.text || name != "text_payload")
                    && (projection.binary || name != "binary_payload")
                    && (projection.embedding || name != "embedding")
            })
            .collect()
    }

    /// An LSM scanner that only reads the columns required by `projection`.
    async fn lsm_scanner_projected(&self, projection: ReadProjection) -> LanceResult<LsmScanner> {
        let scanner = self.lsm_scanner().await?;
        if projection.loads_all() {
            return Ok(scanner);
        }
        let columns = self.projected_columns(projection);
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        Ok(scanner.project(&refs))
    }

    /// Fetch a single record's `binary_payload` on demand, without loading it
    /// during `list`/`search`. Returns `None` if the record or its binary
    /// payload is absent.
    pub async fn get_blob(&self, id: &str) -> LanceResult<Option<Vec<u8>>> {
        let filter = format!("id IN ({})", sql_quoted_list(&[id]));
        let scanner = self
            .lsm_scanner()
            .await?
            .project(&["id", "binary_payload"])
            .filter(&filter)?;
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

    /// Manually trigger compaction to merge small fragments.
    pub async fn compact(
        &mut self,
        options: Option<CompactionConfig>,
    ) -> LanceResult<CompactionMetrics> {
        let config = options.unwrap_or_else(|| self.compaction_config.clone());

        info!(
            "Starting compaction: {} fragments",
            self.base.dataset.count_fragments()
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

        // Run compaction through the shared layer, which sets
        // `defer_index_remap` (the base table carries a fieldless MemWAL index
        // that Lance's inline remap panics on) and reloads the handle with this
        // store's storage options and session.
        let result = self.base.compact(Some(config.clone())).await;

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

                // `StorageBase::compact` already reloaded the handle, with the
                // storage options and session attached -- the bare
                // `Dataset::open` that used to run here dropped both, which
                // broke reopening on any credentialed object store.

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

    /// Seal the active memtable so previously added rows become readable on
    /// every instance.
    ///
    /// A no-op when no writer is resident, and unnecessary under the default
    /// [`ContextStoreOptions::seal_on_add`] (`true`), where `add` already seals.
    /// Stores that opted into deferred sealing must drive this — periodically or
    /// explicitly — or their writes stay durable but invisible.
    pub async fn flush(&self) -> LanceResult<()> {
        self.base.flush().await
    }

    /// Gracefully close the resident MemWAL writer, draining its background
    /// tasks and sealing whatever it still buffers. Idempotent.
    pub async fn close(&mut self) -> LanceResult<()> {
        self.base.close().await
    }

    /// Fold this instance's flushed MemWAL generations into the base table when
    /// it has accumulated [`ContextStoreOptions::merge_after_generations`] of
    /// them. Returns the number reclaimed; a no-op when the trigger is unset.
    ///
    /// # This store previously never merged at all
    ///
    /// Flushed generations accumulated under `_mem_wal/` forever, and every read
    /// unioned all of them, so read cost grew without bound in the number of
    /// writes. Merging is what keeps that bounded — drive it from a sweeper, or
    /// use [`Self::cleanup_wal`] for the time-based trigger.
    pub async fn maybe_merge_wal(&mut self) -> LanceResult<usize> {
        self.base.maybe_merge_own_shard().await
    }

    /// Seal, then fold **every** pending flushed generation into the base table.
    /// The time half of the "time OR count" trigger, so deliberately not gated
    /// by the count threshold. Returns the number of generations reclaimed.
    pub async fn cleanup_wal(&mut self) -> LanceResult<usize> {
        self.base.cleanup_own_shard().await
    }

    /// Number of flushed MemWAL generations pending merge into the base table,
    /// across all shards. Read-only: never merges.
    pub async fn pending_wal_generations(&self) -> LanceResult<usize> {
        self.base.pending_wal_generations().await
    }

    /// Check if compaction should run based on configuration thresholds.
    pub async fn should_compact(&self) -> LanceResult<bool> {
        let fragment_count = self.base.dataset.count_fragments();

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
            total_fragments: self.base.dataset.count_fragments(),
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

        let indices = self.base.dataset.load_indices().await?;
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

        self.base
            .dataset
            .create_index_builder(&["id"], index_type, &params)
            .name(ID_INDEX_NAME.to_string())
            .replace(true)
            .await?;

        // Reload through the base so the new index is visible to subsequent
        // reads, keeping the storage options and session (a bare
        // `Dataset::open` here silently dropped them).
        self.base.reload().await
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

        // The task opens its *own* store handle rather than cloning this one.
        // `ContextStore` is no longer `Clone`: it owns a `StorageBase`, which
        // owns the resident MemWAL writer and a `Drop` that seals it, so it must
        // have exactly one owner. A second handle is the right model anyway --
        // compaction only rewrites base-table fragments and takes `&mut`, so
        // sharing a handle with the write path would mean contending for it.
        let uri = self.uri().to_string();
        let interval_secs = self.compaction_config.check_interval_secs;
        let options = ContextStoreOptions {
            storage_options: self.base.storage_options.clone(),
            // `enabled: false` is load-bearing, not tidiness: the reopened
            // handle would otherwise spawn its own background compactor, which
            // makes `start_background_compaction` infinitely recursive (and the
            // resulting future un-`Send`, so it does not even compile).
            compaction: CompactionConfig {
                enabled: false,
                ..self.compaction_config.clone()
            },
            embedding_dim: Some(self.embedding_dim),
            blob_columns: self.blob_columns.clone(),
            id_index_type: self.id_index_type,
            distance_metric: Some(self.distance_metric),
            shard_id: None,
            merge_after_generations: None,
            // A compactor never appends, so the seal mode is irrelevant to it;
            // deferring keeps it from ever emitting a generation.
            seal_on_add: false,
        };

        let task = tokio::spawn(async move {
            let compaction_options = options;
            let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

            loop {
                interval.tick().await;

                let mut store = match open_for_compaction(&uri, compaction_options.clone()).await {
                    Ok(store) => store,
                    Err(e) => {
                        error!("Background compaction could not open store: {}", e);
                        continue;
                    }
                };

                match store.should_compact().await {
                    Ok(true) => {
                        info!("Background compaction triggered");
                        if let Err(e) = store.compact(None).await {
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
        Self::schema_with_embedding_dim(blob_columns, DEFAULT_EMBEDDING_DIM)
    }

    /// Lance schema for a context store using a caller-selected embedding width.
    pub fn schema_with_embedding_dim(blob_columns: &HashSet<String>, embedding_dim: i32) -> Schema {
        Self::schema_with_options(
            blob_columns,
            true,
            true,
            true,
            true,
            true,
            embedding_dim,
            DistanceMetric::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn schema_with_options(
        blob_columns: &HashSet<String>,
        include_external_id: bool,
        include_metadata: bool,
        include_relationships: bool,
        include_lifecycle: bool,
        include_external_reference: bool,
        embedding_dim: i32,
        distance_metric: DistanceMetric,
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
            Field::new("tenant", DataType::Utf8, true),
            Field::new("source", DataType::Utf8, true),
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
        if include_relationships {
            fields.push(relationship_field());
        }
        if include_lifecycle {
            fields.extend([
                Field::new(
                    "expires_at",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                ),
                Field::new("retention_policy", DataType::Utf8, true),
                Field::new("lifecycle_status", DataType::Utf8, false),
                Field::new(
                    "retired_at",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                ),
                Field::new("retired_reason", DataType::Utf8, true),
                Field::new("supersedes_id", DataType::Utf8, true),
                Field::new("superseded_by_id", DataType::Utf8, true),
            ]);
        }
        fields.extend([
            Field::new("content_type", DataType::Utf8, false),
            text_field,
            binary_field,
        ]);
        if include_external_reference {
            fields.extend([
                Field::new("payload_uri", DataType::Utf8, true),
                Field::new("payload_size", DataType::Int64, true),
                Field::new("payload_checksum", DataType::Utf8, true),
            ]);
        }
        fields.push(Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim,
            ),
            true,
        ));

        let schema_metadata = HashMap::from([(
            DISTANCE_METRIC_METADATA_KEY.to_string(),
            distance_metric.as_str().to_string(),
        )]);

        Schema::new_with_metadata(fields, schema_metadata)
    }

    async fn load_with_options(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> LanceResult<Dataset> {
        StorageBase::load_with_options(uri, storage_options, None).await
    }

    async fn create_with_options(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
        blob_columns: &HashSet<String>,
        embedding_dim: i32,
        distance_metric: DistanceMetric,
    ) -> LanceResult<Dataset> {
        let schema = Arc::new(Self::schema_with_options(
            blob_columns,
            true,
            true,
            true,
            true,
            true,
            embedding_dim,
            distance_metric,
        ));
        StorageBase::create_with_options(uri, schema, storage_options, None).await
    }

    fn records_to_batch(&self, entries: &[ContextRecord]) -> LanceResult<RecordBatch> {
        let include_external_id = self
            .base
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == "external_id");
        let include_lifecycle = self
            .base
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == "expires_at");
        let include_metadata = self
            .base
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == "metadata");
        let include_tenant = self
            .base
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == "tenant");
        let include_source = self
            .base
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == "source");
        let include_external_reference = self
            .base
            .dataset
            .schema()
            .field_paths()
            .iter()
            .any(|path| path == "payload_uri");
        let include_relationships = self.has_relationships_column();
        if !include_external_id && entries.iter().any(|entry| entry.external_id.is_some()) {
            return Err(ArrowError::InvalidArgumentError(
                "external_id requires a context dataset created with external_id support"
                    .to_string(),
            )
            .into());
        }
        if !include_metadata && entries.iter().any(|entry| entry.metadata.is_some()) {
            return Err(ArrowError::InvalidArgumentError(
                "metadata requires a context dataset created with metadata support".to_string(),
            )
            .into());
        }
        if !include_tenant && entries.iter().any(|entry| entry.tenant.is_some()) {
            return Err(ArrowError::InvalidArgumentError(
                "tenant requires a context dataset created with partition-key column support"
                    .to_string(),
            )
            .into());
        }
        if !include_source && entries.iter().any(|entry| entry.source.is_some()) {
            return Err(ArrowError::InvalidArgumentError(
                "source requires a context dataset created with partition-key column support"
                    .to_string(),
            )
            .into());
        }
        if !include_relationships && entries.iter().any(|entry| !entry.relationships.is_empty()) {
            return Err(ArrowError::InvalidArgumentError(
                "relationships require a context dataset with relationships support; run migrate_relationships_column() on older datasets".to_string(),
            )
            .into());
        }
        if !include_external_reference
            && entries.iter().any(|entry| {
                entry.payload_uri.is_some()
                    || entry.payload_size.is_some()
                    || entry.payload_checksum.is_some()
            })
        {
            return Err(ArrowError::InvalidArgumentError(
                "external payload references require a context dataset created with external-reference support".to_string(),
            )
            .into());
        }
        if !include_lifecycle && entries.iter().any(ContextRecord::has_non_default_lifecycle) {
            return Err(ArrowError::InvalidArgumentError(
                "lifecycle fields require a context dataset created with lifecycle support"
                    .to_string(),
            )
            .into());
        }

        let mut id_builder = StringBuilder::new();
        let mut external_id_builder = StringBuilder::new();
        let mut run_id_builder = StringBuilder::new();
        let mut bot_id_builder = StringBuilder::new();
        let mut session_id_builder = StringBuilder::new();
        let mut tenant_builder = StringBuilder::new();
        let mut source_builder = StringBuilder::new();
        let mut created_at_builder = TimestampMicrosecondBuilder::with_capacity(entries.len());
        let mut role_builder = StringDictionaryBuilder::<Int8Type>::new();
        let mut metadata_builder = LargeStringBuilder::new();
        let mut relationships_builder = ListBuilder::new(relationship_struct_builder())
            .with_field(relationship_list_item_field());
        let mut expires_at_builder = TimestampMicrosecondBuilder::with_capacity(entries.len());
        let mut retention_policy_builder = StringBuilder::new();
        let mut lifecycle_status_builder = StringBuilder::new();
        let mut retired_at_builder = TimestampMicrosecondBuilder::with_capacity(entries.len());
        let mut retired_reason_builder = StringBuilder::new();
        let mut supersedes_id_builder = StringBuilder::new();
        let mut superseded_by_id_builder = StringBuilder::new();
        let mut content_type_builder = StringBuilder::new();
        let mut binary_builder = LargeBinaryBuilder::new();
        let mut payload_uri_builder = StringBuilder::new();
        let mut payload_size_builder = Int64Builder::new();
        let mut payload_checksum_builder = StringBuilder::new();

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
            FixedSizeListBuilder::new(Float32Builder::new(), self.embedding_dim);

        for entry in entries {
            id_builder.append_value(&entry.id);
            external_id_builder.append_option(entry.external_id.as_deref());
            run_id_builder.append_value(&entry.run_id);
            bot_id_builder.append_option(entry.bot_id.as_deref());
            session_id_builder.append_option(entry.session_id.as_deref());
            tenant_builder.append_option(entry.tenant.as_deref());
            source_builder.append_option(entry.source.as_deref());
            created_at_builder.append_value(entry.created_at.timestamp_micros());
            role_builder.append(&entry.role)?;
            match &entry.metadata {
                Some(metadata) => metadata_builder.append_value(metadata.to_string()),
                None => metadata_builder.append_null(),
            }
            for relationship in &entry.relationships {
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
            expires_at_builder
                .append_option(entry.expires_at.map(|value| value.timestamp_micros()));
            retention_policy_builder.append_option(entry.retention_policy.as_deref());
            lifecycle_status_builder.append_value(&entry.lifecycle_status);
            retired_at_builder
                .append_option(entry.retired_at.map(|value| value.timestamp_micros()));
            retired_reason_builder.append_option(entry.retired_reason.as_deref());
            supersedes_id_builder.append_option(entry.supersedes_id.as_deref());
            superseded_by_id_builder.append_option(entry.superseded_by_id.as_deref());
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

            payload_uri_builder.append_option(entry.payload_uri.as_deref());
            payload_size_builder.append_option(entry.payload_size);
            payload_checksum_builder.append_option(entry.payload_checksum.as_deref());

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
                if embedding.len() != self.embedding_dim as usize {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "embedding length {} does not match expected dimension {}",
                        embedding.len(),
                        self.embedding_dim
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
                for _ in 0..self.embedding_dim {
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
        let tenant_array: ArrayRef = Arc::new(tenant_builder.finish());
        let source_array: ArrayRef = Arc::new(source_builder.finish());
        let created_at_array: ArrayRef = Arc::new(created_at_builder.finish());
        let role_array: ArrayRef = Arc::new(role_builder.finish());
        let metadata_array: ArrayRef = Arc::new(metadata_builder.finish());
        let relationships_array: ArrayRef = Arc::new(relationships_builder.finish());
        let expires_at_array: ArrayRef = Arc::new(expires_at_builder.finish());
        let retention_policy_array: ArrayRef = Arc::new(retention_policy_builder.finish());
        let lifecycle_status_array: ArrayRef = Arc::new(lifecycle_status_builder.finish());
        let retired_at_array: ArrayRef = Arc::new(retired_at_builder.finish());
        let retired_reason_array: ArrayRef = Arc::new(retired_reason_builder.finish());
        let supersedes_id_array: ArrayRef = Arc::new(supersedes_id_builder.finish());
        let superseded_by_id_array: ArrayRef = Arc::new(superseded_by_id_builder.finish());
        let content_type_array: ArrayRef = Arc::new(content_type_builder.finish());
        let text_array: ArrayRef = if text_is_blob {
            Arc::new(text_binary_builder.unwrap().finish())
        } else {
            Arc::new(text_string_builder.unwrap().finish())
        };
        let binary_array: ArrayRef = Arc::new(binary_builder.finish());
        let payload_uri_array: ArrayRef = Arc::new(payload_uri_builder.finish());
        let payload_size_array: ArrayRef = Arc::new(payload_size_builder.finish());
        let payload_checksum_array: ArrayRef = Arc::new(payload_checksum_builder.finish());
        let state_array: ArrayRef = Arc::new(state_builder.finish());
        let embedding_array: ArrayRef = Arc::new(embedding_builder.finish());

        let mut arrays_by_name = HashMap::from([("id".to_string(), id_array)]);
        if include_external_id {
            arrays_by_name.insert("external_id".to_string(), external_id_array);
        }
        arrays_by_name.extend([
            ("run_id".to_string(), run_id_array),
            ("bot_id".to_string(), bot_id_array),
            ("session_id".to_string(), session_id_array),
            ("created_at".to_string(), created_at_array),
            ("role".to_string(), role_array),
            ("state_metadata".to_string(), state_array),
        ]);
        if include_tenant {
            arrays_by_name.insert("tenant".to_string(), tenant_array);
        }
        if include_source {
            arrays_by_name.insert("source".to_string(), source_array);
        }
        if include_metadata {
            arrays_by_name.insert("metadata".to_string(), metadata_array);
        }
        if include_relationships {
            arrays_by_name.insert(RELATIONSHIPS_COLUMN.to_string(), relationships_array);
        }
        if include_lifecycle {
            arrays_by_name.extend([
                ("expires_at".to_string(), expires_at_array),
                ("retention_policy".to_string(), retention_policy_array),
                ("lifecycle_status".to_string(), lifecycle_status_array),
                ("retired_at".to_string(), retired_at_array),
                ("retired_reason".to_string(), retired_reason_array),
                ("supersedes_id".to_string(), supersedes_id_array),
                ("superseded_by_id".to_string(), superseded_by_id_array),
            ]);
        }
        arrays_by_name.extend([
            ("content_type".to_string(), content_type_array),
            ("text_payload".to_string(), text_array),
            ("binary_payload".to_string(), binary_array),
            ("embedding".to_string(), embedding_array),
        ]);
        if include_external_reference {
            arrays_by_name.extend([
                ("payload_uri".to_string(), payload_uri_array),
                ("payload_size".to_string(), payload_size_array),
                ("payload_checksum".to_string(), payload_checksum_array),
            ]);
        }

        let schema: Arc<Schema> = Arc::new(self.base.dataset.schema().into());
        let arrays = schema
            .fields()
            .iter()
            .map(|field| {
                arrays_by_name.remove(field.name().as_str()).ok_or_else(|| {
                    LanceError::from(ArrowError::InvalidArgumentError(format!(
                        "unsupported dataset column '{}'",
                        field.name()
                    )))
                })
            })
            .collect::<LanceResult<Vec<_>>>()?;
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
    let tenant_array = column_as_optional::<StringArray>(batch, "tenant");
    let source_array = column_as_optional::<StringArray>(batch, "source");
    let created_at_array = column_as::<TimestampMicrosecondArray>(batch, "created_at")?;
    let role_array = column_as::<DictionaryArray<Int8Type>>(batch, "role")?;
    let state_array = column_as::<StructArray>(batch, "state_metadata")?;
    let metadata_array = column_as_optional::<LargeStringArray>(batch, "metadata");
    let relationships_array = column_as_optional::<ListArray>(batch, RELATIONSHIPS_COLUMN);
    let expires_at_array = column_as_optional::<TimestampMicrosecondArray>(batch, "expires_at");
    let retention_policy_array = column_as_optional::<StringArray>(batch, "retention_policy");
    let lifecycle_status_array = column_as_optional::<StringArray>(batch, "lifecycle_status");
    let retired_at_array = column_as_optional::<TimestampMicrosecondArray>(batch, "retired_at");
    let retired_reason_array = column_as_optional::<StringArray>(batch, "retired_reason");
    let supersedes_id_array = column_as_optional::<StringArray>(batch, "supersedes_id");
    let superseded_by_id_array = column_as_optional::<StringArray>(batch, "superseded_by_id");
    let content_type_array = column_as::<StringArray>(batch, "content_type")?;
    let binary_array = column_as_optional::<LargeBinaryArray>(batch, "binary_payload");
    let payload_uri_array = column_as_optional::<StringArray>(batch, "payload_uri");
    let payload_size_array = column_as_optional::<Int64Array>(batch, "payload_size");
    let payload_checksum_array = column_as_optional::<StringArray>(batch, "payload_checksum");
    let embedding_array = column_as_optional::<FixedSizeListArray>(batch, "embedding");

    // `text_payload` may be projected out, or stored as LargeBinary (blob) or LargeUtf8.
    let has_text = batch.schema().field_with_name("text_payload").is_ok();
    let text_is_binary = batch
        .schema()
        .field_with_name("text_payload")
        .is_ok_and(|f| f.data_type() == &DataType::LargeBinary);

    let text_string_array = if has_text && !text_is_binary {
        Some(column_as::<LargeStringArray>(batch, "text_payload")?)
    } else {
        None
    };
    let text_binary_array = if has_text && text_is_binary {
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
        let created_at = timestamp_from_micros(created_at_array.value(row), "created_at")?;

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

        let text_payload = if let Some(arr) = text_binary_array {
            if arr.is_null(row) {
                None
            } else {
                Some(String::from_utf8_lossy(arr.value(row)).to_string())
            }
        } else if let Some(arr) = text_string_array {
            if arr.is_null(row) {
                None
            } else {
                Some(arr.value(row).to_string())
            }
        } else {
            None
        };

        let binary_payload = match binary_array {
            Some(arr) if !arr.is_null(row) => Some(arr.value(row).to_vec()),
            _ => None,
        };

        let embedding = match embedding_array {
            Some(arr) if !arr.is_null(row) => Some(embedding_from_list(arr, row)?),
            _ => None,
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

        let tenant = tenant_array.and_then(|arr| {
            if arr.is_null(row) {
                None
            } else {
                Some(arr.value(row).to_string())
            }
        });

        let source = source_array.and_then(|arr| {
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
        let relationships = match relationships_array {
            Some(arr) if !arr.is_null(row) => relationships_from_list(arr, row)?,
            _ => Vec::new(),
        };
        let expires_at = optional_timestamp_from_array(expires_at_array, row, "expires_at")?;
        let retention_policy = optional_string_from_array(retention_policy_array, row);
        let lifecycle_status = optional_string_from_array(lifecycle_status_array, row)
            .unwrap_or_else(|| LIFECYCLE_ACTIVE.to_string());
        let retired_at = optional_timestamp_from_array(retired_at_array, row, "retired_at")?;
        let retired_reason = optional_string_from_array(retired_reason_array, row);
        let supersedes_id = optional_string_from_array(supersedes_id_array, row);
        let superseded_by_id = optional_string_from_array(superseded_by_id_array, row);
        let payload_uri = optional_string_from_array(payload_uri_array, row);
        let payload_size = payload_size_array.and_then(|arr| {
            if arr.is_null(row) {
                None
            } else {
                Some(arr.value(row))
            }
        });
        let payload_checksum = optional_string_from_array(payload_checksum_array, row);

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
            tenant,
            source,
            created_at,
            role,
            state_metadata,
            metadata,
            relationships,
            expires_at,
            retention_policy,
            lifecycle_status,
            retired_at,
            retired_reason,
            supersedes_id,
            superseded_by_id,
            content_type: content_type_array.value(row).to_string(),
            text_payload,
            binary_payload,
            payload_uri,
            payload_size,
            payload_checksum,
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

pub(crate) fn relationships_from_list(
    list: &ListArray,
    row: usize,
) -> LanceResult<Vec<Relationship>> {
    let values = list.value(row);
    let struct_array = values
        .as_ref()
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "relationships column does not contain struct values".to_string(),
            ))
        })?;

    let target_id_array = struct_array
        .column(0)
        .as_ref()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "relationships.target_id column has unexpected data type".to_string(),
            ))
        })?;
    let relation_array = struct_array
        .column(1)
        .as_ref()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "relationships.relation column has unexpected data type".to_string(),
            ))
        })?;
    let weight_array = struct_array
        .column(2)
        .as_ref()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            LanceError::from(ArrowError::InvalidArgumentError(
                "relationships.weight column has unexpected data type".to_string(),
            ))
        })?;

    let mut relationships = Vec::with_capacity(struct_array.len());
    for idx in 0..struct_array.len() {
        if struct_array.is_null(idx) {
            continue;
        }
        if target_id_array.is_null(idx) {
            return Err(LanceError::from(ArrowError::InvalidArgumentError(
                "relationships.target_id contains null values".to_string(),
            )));
        }
        if relation_array.is_null(idx) {
            return Err(LanceError::from(ArrowError::InvalidArgumentError(
                "relationships.relation contains null values".to_string(),
            )));
        }

        relationships.push(Relationship {
            target_id: target_id_array.value(idx).to_string(),
            relation: relation_array.value(idx).to_string(),
            weight: if weight_array.is_null(idx) {
                None
            } else {
                Some(weight_array.value(idx))
            },
        });
    }
    Ok(relationships)
}

pub(crate) fn timestamp_from_micros(value: i64, column: &str) -> LanceResult<DateTime<Utc>> {
    DateTime::from_timestamp_micros(value).ok_or_else(|| {
        LanceError::from(ArrowError::InvalidArgumentError(format!(
            "invalid timestamp value {value} in column '{column}'"
        )))
    })
}

fn optional_timestamp_from_array(
    array: Option<&TimestampMicrosecondArray>,
    row: usize,
    column: &str,
) -> LanceResult<Option<DateTime<Utc>>> {
    let Some(array) = array else {
        return Ok(None);
    };
    if array.is_null(row) {
        Ok(None)
    } else {
        timestamp_from_micros(array.value(row), column).map(Some)
    }
}

fn optional_string_from_array(array: Option<&StringArray>, row: usize) -> Option<String> {
    array.and_then(|arr| {
        if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row).to_string())
        }
    })
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

fn validate_embedding_dim(embedding_dim: i32) -> LanceResult<()> {
    if embedding_dim <= 0 {
        return Err(LanceError::from(ArrowError::InvalidArgumentError(format!(
            "embedding_dim must be positive, got {embedding_dim}"
        ))));
    }
    Ok(())
}

fn validate_query_dimension(query: &[f32], embedding_dim: i32) -> LanceResult<()> {
    if query.len() != embedding_dim as usize {
        return Err(ArrowError::InvalidArgumentError(format!(
            "query length {} does not match embedding dimension {}",
            query.len(),
            embedding_dim
        ))
        .into());
    }
    Ok(())
}

fn unique_query_terms(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    tokenize_for_retrieval(text)
        .into_iter()
        .filter(|term| seen.insert(term.clone()))
        .collect()
}

fn tokenize_for_retrieval(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        if character.is_alphanumeric() {
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        terms.push(current);
    }

    terms
}

fn lexical_score(query_terms: &[String], text: Option<&str>) -> Option<f32> {
    let text = text?;
    if query_terms.is_empty() {
        return None;
    }

    let payload_terms: HashSet<String> = tokenize_for_retrieval(text).into_iter().collect();
    if payload_terms.is_empty() {
        return None;
    }

    let matched_terms = query_terms
        .iter()
        .filter(|term| payload_terms.contains(*term))
        .count();
    if matched_terms == 0 {
        return None;
    }

    Some(matched_terms as f32 / query_terms.len() as f32)
}

fn add_retrieve_channel(
    candidates: &mut HashMap<String, RetrieveResult>,
    record: &ContextRecord,
    rank: usize,
    channel: &str,
    vector_distance: Option<f32>,
    text_score: Option<f32>,
) {
    let candidate = candidates
        .entry(record.id.clone())
        .or_insert_with(|| RetrieveResult {
            record: record.clone(),
            score: 0.0,
            vector_distance: None,
            text_score: None,
            matched_channels: Vec::new(),
        });
    candidate.score += 1.0 / (RRF_K + rank as f32);
    if let Some(distance) = vector_distance {
        candidate.vector_distance = Some(distance);
    }
    if let Some(score) = text_score {
        candidate.text_score = Some(score);
    }
    if !candidate
        .matched_channels
        .iter()
        .any(|existing| existing == channel)
    {
        candidate.matched_channels.push(channel.to_string());
    }
}

fn compare_retrieve_results(left: &RetrieveResult, right: &RetrieveResult) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| compare_optional_distance(left.vector_distance, right.vector_distance))
        .then_with(|| compare_optional_score(left.text_score, right.text_score))
        .then_with(|| left.record.id.cmp(&right.record.id))
}

fn compare_optional_distance(left: Option<f32>, right: Option<f32>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn compare_optional_score(left: Option<f32>, right: Option<f32>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.total_cmp(&left),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn embedding_dim_from_schema(schema: &Schema) -> LanceResult<i32> {
    let field = schema
        .field_with_name("embedding")
        .map_err(LanceError::from)?;
    let DataType::FixedSizeList(item_field, embedding_dim) = field.data_type() else {
        return Err(LanceError::from(ArrowError::InvalidArgumentError(
            "embedding column must be a FixedSizeList<Float32>".to_string(),
        )));
    };
    if item_field.data_type() != &DataType::Float32 {
        return Err(LanceError::from(ArrowError::InvalidArgumentError(
            "embedding column must contain Float32 values".to_string(),
        )));
    }
    validate_embedding_dim(*embedding_dim)?;
    Ok(*embedding_dim)
}

/// Read the persisted [`DistanceMetric`] from the dataset's schema metadata.
///
/// Datasets created before metric persistence (no key present) default to
/// [`DistanceMetric::L2`], preserving historical ranking behavior.
fn distance_metric_from_schema(schema: &Schema) -> LanceResult<DistanceMetric> {
    match schema.metadata.get(DISTANCE_METRIC_METADATA_KEY) {
        Some(value) => DistanceMetric::parse(value),
        None => Ok(DistanceMetric::default()),
    }
}

/// Dot product of two vectors.
fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>()
}

/// Cosine distance (`1 - cosine_similarity`), ranging from 0 (identical
/// direction) to 2 (opposite). If either vector has zero magnitude the
/// similarity is undefined, so we return the maximum distance (`1.0`) to keep
/// such records ranked last without producing `NaN`.
fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
    let dot = dot_product(left, right);
    let left_norm = dot_product(left, left).sqrt();
    let right_norm = dot_product(right, right).sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (left_norm * right_norm))
}

/// Negated dot product, so that a larger inner product (a closer match for
/// maximum-inner-product search) sorts first under ascending ordering.
fn dot_distance(left: &[f32], right: &[f32]) -> f32 {
    -dot_product(left, right)
}

pub(crate) fn column_as<'a, A>(batch: &'a RecordBatch, name: &str) -> LanceResult<&'a A>
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

pub(crate) fn column_as_optional<'a, A>(batch: &'a RecordBatch, name: &str) -> Option<&'a A>
where
    A: Array + 'static,
{
    batch
        .column_by_name(name)
        .and_then(|col| col.as_ref().as_any().downcast_ref::<A>())
}

/// Render a SQL `IN (...)` value list as comma-separated quoted string
/// literals, escaping any embedded single quotes. Callers ensure the input is
/// non-empty before building the surrounding `IN ()` clause.
fn sql_quoted_list(values: &[&str]) -> String {
    values
        .iter()
        .map(|value| format!("'{}'", value.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::serde::CONTENT_TYPE_TEXT;
    use arrow_array::RecordBatchIterator;
    use chrono::{Duration as ChronoDuration, Utc};
    use lance::dataset::{WriteMode, WriteParams};
    use tempfile::TempDir;

    fn make_embedding_with_dim(dim: usize, pivot: f32) -> Vec<f32> {
        let mut values = vec![0.0; dim];
        if !values.is_empty() {
            values[0] = pivot;
        }
        values
    }

    fn make_embedding(pivot: f32) -> Vec<f32> {
        make_embedding_with_dim(DEFAULT_EMBEDDING_DIM as usize, pivot)
    }

    fn text_record(id: &str, embedding_pivot: f32) -> ContextRecord {
        ContextRecord {
            id: id.to_string(),
            external_id: None,
            run_id: format!("run-{id}"),
            bot_id: None,
            session_id: None,
            tenant: None,
            source: None,
            created_at: Utc::now(),
            role: "user".to_string(),
            state_metadata: Some(StateMetadata {
                step: Some(1),
                active_plan_id: Some("plan".to_string()),
                tokens_used: Some(10),
                custom: None,
            }),
            metadata: None,
            relationships: Vec::new(),
            expires_at: None,
            retention_policy: None,
            lifecycle_status: LIFECYCLE_ACTIVE.to_string(),
            retired_at: None,
            retired_reason: None,
            supersedes_id: None,
            superseded_by_id: None,
            content_type: CONTENT_TYPE_TEXT.to_string(),
            text_payload: Some(format!("payload-{id}")),
            binary_payload: None,
            payload_uri: None,
            payload_size: None,
            payload_checksum: None,
            embedding: Some(make_embedding(embedding_pivot)),
        }
    }

    #[test]
    fn external_payload_reference_roundtrips_add_list_and_fetch() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let media_dir = TempDir::new().unwrap();
        let object_uri = media_dir
            .path()
            .join("media-001.bin")
            .to_string_lossy()
            .to_string();
        let payload = b"\x89PNG\r\n\x1a\n external media bytes".to_vec();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();

            // Offload bytes to the object store, then reference them by URI.
            let written = store.put_payload(&object_uri, &payload).await.unwrap();
            assert_eq!(written, payload.len() as u64);

            let mut record = text_record("media-001", 0.5);
            record.content_type = "image/png".to_string();
            record.text_payload = None;
            record.payload_uri = Some(object_uri.clone());
            record.payload_size = Some(payload.len() as i64);
            record.payload_checksum = Some("sha256:deadbeef".to_string());
            store.add(std::slice::from_ref(&record)).await.unwrap();

            // list returns the reference without materializing the bytes.
            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 1);
            let listed = &listed[0];
            assert_eq!(listed.payload_uri.as_deref(), Some(object_uri.as_str()));
            assert_eq!(listed.payload_size, Some(payload.len() as i64));
            assert_eq!(listed.payload_checksum.as_deref(), Some("sha256:deadbeef"));
            assert_eq!(listed.binary_payload, None);

            // opt-in fetch resolves the bytes via the context's storage path.
            let fetched = store.fetch_payload(&record.id).await.unwrap();
            assert_eq!(fetched, Some(payload.clone()));
        });
    }

    #[test]
    fn fetch_payload_handles_missing_record_and_missing_reference() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();

            // Unknown id resolves to None rather than erroring.
            assert_eq!(store.fetch_payload("does-not-exist").await.unwrap(), None);

            // A record without an external reference is an error to fetch.
            let record = text_record("inline-1", 0.1);
            store.add(std::slice::from_ref(&record)).await.unwrap();
            let err = store.fetch_payload(&record.id).await.unwrap_err();
            assert!(err.to_string().contains("no external payload reference"));
        });
    }

    #[test]
    fn search_orders_by_distance() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
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

    fn make_embedding2(x0: f32, x1: f32) -> Vec<f32> {
        let mut values = vec![0.0; DEFAULT_EMBEDDING_DIM as usize];
        values[0] = x0;
        values[1] = x1;
        values
    }

    fn text_record_with(id: &str, embedding: Vec<f32>) -> ContextRecord {
        let mut record = text_record(id, 0.0);
        record.embedding = Some(embedding);
        record
    }

    #[test]
    fn distance_metric_parse_and_math() {
        assert_eq!(DistanceMetric::parse("l2").unwrap(), DistanceMetric::L2);
        assert_eq!(DistanceMetric::parse("L2").unwrap(), DistanceMetric::L2);
        assert_eq!(
            DistanceMetric::parse("cosine").unwrap(),
            DistanceMetric::Cosine
        );
        assert_eq!(DistanceMetric::parse("DOT").unwrap(), DistanceMetric::Dot);
        assert!(DistanceMetric::parse("manhattan").is_err());
        assert_eq!(DistanceMetric::default(), DistanceMetric::L2);

        let a = [1.0_f32, 0.0];
        let b = [1.0_f32, 1.0];
        // L2: sqrt(0 + 1) = 1
        assert!((DistanceMetric::L2.distance(&a, &b) - 1.0).abs() < 1e-6);
        // Cosine distance: 1 - (1 / (1 * sqrt(2))) = 1 - 0.70710677
        assert!((DistanceMetric::Cosine.distance(&a, &b) - (1.0 - 0.707_106_77)).abs() < 1e-5);
        // Dot: -(1*1 + 0*1) = -1
        assert!((DistanceMetric::Dot.distance(&a, &b) + 1.0).abs() < 1e-6);
        // Zero-magnitude vectors yield max cosine distance, never NaN.
        let zero = [0.0_f32, 0.0];
        assert!((DistanceMetric::Cosine.distance(&a, &zero) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_metric_changes_ranking() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            // query direction matches "aligned" but "near" is closer in L2.
            let query = make_embedding2(1.0, 0.0);
            // aligned: same direction as query, larger magnitude -> far in L2,
            //          but cosine distance 0 and largest dot product.
            let aligned = make_embedding2(10.0, 0.0);
            // near: closest in L2, but off-axis -> larger cosine distance.
            let near = make_embedding2(1.0, 1.0);

            // Default (L2): `near` should rank first.
            let l2_dir = TempDir::new().unwrap();
            let l2_store = ContextStore::open(&l2_dir.path().to_string_lossy())
                .await
                .unwrap();
            l2_store
                .add(&[
                    text_record_with("aligned", aligned.clone()),
                    text_record_with("near", near.clone()),
                ])
                .await
                .unwrap();
            let l2_results = l2_store.search(&query, Some(2)).await.unwrap();
            assert_eq!(l2_results[0].record.id, "near");

            // Cosine: `aligned` should rank first despite the larger L2 distance.
            let cos_dir = TempDir::new().unwrap();
            let cos_opts = ContextStoreOptions {
                distance_metric: Some(DistanceMetric::Cosine),
                ..Default::default()
            };
            let cos_store =
                ContextStore::open_with_options(&cos_dir.path().to_string_lossy(), cos_opts)
                    .await
                    .unwrap();
            cos_store
                .add(&[
                    text_record_with("aligned", aligned.clone()),
                    text_record_with("near", near.clone()),
                ])
                .await
                .unwrap();
            let cos_results = cos_store.search(&query, Some(2)).await.unwrap();
            assert_eq!(cos_results[0].record.id, "aligned");

            // Dot: `aligned` has the largest inner product -> first.
            let dot_dir = TempDir::new().unwrap();
            let dot_opts = ContextStoreOptions {
                distance_metric: Some(DistanceMetric::Dot),
                ..Default::default()
            };
            let dot_store =
                ContextStore::open_with_options(&dot_dir.path().to_string_lossy(), dot_opts)
                    .await
                    .unwrap();
            dot_store
                .add(&[
                    text_record_with("aligned", aligned),
                    text_record_with("near", near),
                ])
                .await
                .unwrap();
            let dot_results = dot_store.search(&query, Some(2)).await.unwrap();
            assert_eq!(dot_results[0].record.id, "aligned");
        });
    }

    #[test]
    fn distance_metric_persists_across_reopen() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let dir = TempDir::new().unwrap();
            let uri = dir.path().to_string_lossy().to_string();
            let query = make_embedding2(1.0, 0.0);
            let aligned = make_embedding2(10.0, 0.0);
            let near = make_embedding2(1.0, 1.0);

            // Create with cosine and write records.
            {
                let opts = ContextStoreOptions {
                    distance_metric: Some(DistanceMetric::Cosine),
                    ..Default::default()
                };
                let store = ContextStore::open_with_options(&uri, opts).await.unwrap();
                store
                    .add(&[
                        text_record_with("aligned", aligned.clone()),
                        text_record_with("near", near.clone()),
                    ])
                    .await
                    .unwrap();
            }

            // Reopen WITHOUT passing the metric: it must be recovered from the
            // schema, so cosine ranking (`aligned` first) still applies.
            let store = ContextStore::open(&uri).await.unwrap();
            assert_eq!(store.distance_metric, DistanceMetric::Cosine);
            let results = store.search(&query, Some(2)).await.unwrap();
            assert_eq!(results[0].record.id, "aligned");
        });
    }

    #[test]
    fn distance_metric_mismatch_errors() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let dir = TempDir::new().unwrap();
            let uri = dir.path().to_string_lossy().to_string();
            ContextStore::open_with_options(
                &uri,
                ContextStoreOptions {
                    distance_metric: Some(DistanceMetric::Cosine),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

            let result = ContextStore::open_with_options(
                &uri,
                ContextStoreOptions {
                    distance_metric: Some(DistanceMetric::Dot),
                    ..Default::default()
                },
            )
            .await;
            let err = match result {
                Ok(_) => panic!("expected a distance-metric mismatch error"),
                Err(err) => err,
            };
            assert!(
                err.to_string().contains("distance metric"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn distance_metric_from_schema_defaults_l2_when_absent() {
        // Datasets created before metric persistence carry no metadata key.
        let schema = Schema::new(vec![Field::new("id", DataType::Utf8, false)]);
        assert_eq!(
            distance_metric_from_schema(&schema).unwrap(),
            DistanceMetric::L2
        );
    }

    #[test]
    fn retrieve_fuses_text_and_vector_channels() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let mut semantic_near = text_record("semantic-near", 0.0);
            semantic_near.text_payload = Some("general rollout risk guidance".to_string());
            let mut exact_policy = text_record("exact-policy", 1.0);
            exact_policy.text_payload = Some("POLICY-123 blocks service-a rollouts".to_string());

            store
                .add(&[semantic_near.clone(), exact_policy.clone()])
                .await
                .unwrap();

            let query = make_embedding(0.0);
            let results = store
                .retrieve_filtered_with_options(
                    Some("POLICY-123 service-a"),
                    Some(&query),
                    Some(2),
                    None,
                    LifecycleQueryOptions::default(),
                )
                .await
                .unwrap();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0].record.id, exact_policy.id);
            assert!(results[0].score > results[1].score);
            assert!(results[0].vector_distance.is_some());
            assert_eq!(results[0].text_score, Some(1.0));
            assert_eq!(results[0].matched_channels, ["vector", "text"]);
        });
    }

    #[test]
    fn custom_embedding_dimension_round_trips_add_search_and_reopen() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let options = ContextStoreOptions {
                embedding_dim: Some(3),
                ..Default::default()
            };
            let store = ContextStore::open_with_options(&uri, options)
                .await
                .unwrap();
            assert_eq!(store.embedding_dim(), 3);

            let mut first = text_record("custom-a", 0.0);
            first.embedding = Some(make_embedding_with_dim(3, 0.0));
            let mut second = text_record("custom-b", 0.0);
            second.embedding = Some(make_embedding_with_dim(3, 1.0));
            store.add(&[first.clone(), second.clone()]).await.unwrap();

            let query = make_embedding_with_dim(3, 1.0);
            let results = store.search(&query, Some(2)).await.unwrap();
            assert_eq!(results[0].record.id, second.id);

            let reopened = ContextStore::open(&uri).await.unwrap();
            assert_eq!(reopened.embedding_dim(), 3);
            let results = reopened.search(&query, Some(1)).await.unwrap();
            assert_eq!(results[0].record.id, second.id);

            let err = reopened
                .search(&make_embedding(1.0), None)
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains("embedding dimension 3"),
                "unexpected error message: {err}"
            );
        });
    }

    #[test]
    fn existing_default_dimension_dataset_opens_without_options() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            assert_eq!(store.embedding_dim(), DEFAULT_EMBEDDING_DIM);
            store.add(&[text_record("default-dim", 0.0)]).await.unwrap();
            drop(store);

            let reopened = ContextStore::open(&uri).await.unwrap();
            assert_eq!(reopened.embedding_dim(), DEFAULT_EMBEDDING_DIM);
            reopened
                .search(&make_embedding(0.0), Some(1))
                .await
                .unwrap();
        });
    }

    #[test]
    fn opening_existing_dataset_rejects_mismatched_requested_dimension() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let options = ContextStoreOptions {
                embedding_dim: Some(3),
                ..Default::default()
            };
            ContextStore::open_with_options(&uri, options)
                .await
                .unwrap();

            let mismatched = ContextStoreOptions {
                embedding_dim: Some(4),
                ..Default::default()
            };
            let err = match ContextStore::open_with_options(&uri, mismatched).await {
                Ok(_) => panic!("expected mismatched embedding dimension to fail"),
                Err(err) => err,
            };
            assert!(
                err.to_string()
                    .contains("does not match requested dimension 4"),
                "unexpected error message: {err}"
            );
        });
    }

    #[test]
    fn list_hides_expired_and_retired_records_by_default() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let active = text_record("active", 0.0);
            let mut expired = text_record("expired", 0.0);
            expired.expires_at = Some(Utc::now() - ChronoDuration::minutes(1));
            let mut superseded = text_record("superseded", 0.0);
            superseded.lifecycle_status = "superseded".to_string();
            superseded.retired_reason = Some("replaced by newer fact".to_string());
            superseded.superseded_by_id = Some("active".to_string());

            store
                .add(&[active.clone(), expired.clone(), superseded.clone()])
                .await
                .unwrap();

            let visible = store.list(None, None).await.unwrap();
            assert_eq!(visible.len(), 1);
            assert_eq!(visible[0].id, active.id);

            let all = store
                .list_with_options(None, None, LifecycleQueryOptions::new(true, true))
                .await
                .unwrap();
            assert_eq!(all.len(), 3);
            let expired_roundtrip = all.iter().find(|record| record.id == expired.id).unwrap();
            assert_eq!(
                expired_roundtrip
                    .expires_at
                    .map(|value| value.timestamp_micros()),
                expired.expires_at.map(|value| value.timestamp_micros())
            );
            let superseded_roundtrip = all
                .iter()
                .find(|record| record.id == superseded.id)
                .unwrap();
            assert_eq!(superseded_roundtrip.lifecycle_status, "superseded");
            assert_eq!(
                superseded_roundtrip.superseded_by_id.as_deref(),
                Some("active")
            );
        });
    }

    #[test]
    fn list_hides_records_superseded_by_newer_pointer() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let old = text_record("old", 0.0);
            let mut replacement = text_record("new", 1.0);
            replacement.supersedes_id = Some(old.id.clone());
            store
                .add(&[old.clone(), replacement.clone()])
                .await
                .unwrap();

            let visible = store.list(None, None).await.unwrap();
            assert_eq!(visible.len(), 1);
            assert_eq!(visible[0].id, replacement.id);

            let history = store
                .list_with_options(None, None, LifecycleQueryOptions::new(false, true))
                .await
                .unwrap();
            assert_eq!(history.len(), 2);
            assert!(history.iter().any(|record| record.id == old.id));
            assert!(history.iter().any(|record| record.id == replacement.id));
        });
    }

    #[test]
    fn search_filters_lifecycle_before_ranking() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let active = text_record("active", 1.0);
            let mut expired_better_match = text_record("expired", 0.0);
            expired_better_match.expires_at = Some(Utc::now() - ChronoDuration::minutes(1));
            store
                .add(&[active.clone(), expired_better_match.clone()])
                .await
                .unwrap();

            let query = make_embedding(0.0);
            let visible = store.search(&query, Some(1)).await.unwrap();
            assert_eq!(visible.len(), 1);
            assert_eq!(visible[0].record.id, active.id);

            let all = store
                .search_with_options(&query, Some(1), LifecycleQueryOptions::new(true, false))
                .await
                .unwrap();
            assert_eq!(all.len(), 1);
            assert_eq!(all[0].record.id, expired_better_match.id);
        });
    }

    #[test]
    fn external_id_roundtrips_and_supports_lookup() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
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
    fn upsert_by_external_id_inserts_then_replaces_visible_record() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();

            let mut first = text_record("first", 0.0);
            first.external_id = Some("doc-123#chunk-1".to_string());
            let inserted = store.upsert_by_external_id(first.clone()).await.unwrap();
            assert!(inserted.inserted);
            assert_eq!(inserted.replaced_id, None);
            assert_eq!(inserted.record.id, first.id);

            let mut replacement = text_record("replacement", 1.0);
            replacement.external_id = first.external_id.clone();
            let replaced = store
                .upsert_by_external_id(replacement.clone())
                .await
                .unwrap();
            assert!(!replaced.inserted);
            assert_eq!(replaced.replaced_id.as_deref(), Some(first.id.as_str()));
            assert_eq!(
                replaced.record.supersedes_id.as_deref(),
                Some(first.id.as_str())
            );

            let visible = store.list(None, None).await.unwrap();
            assert_eq!(visible.len(), 1);
            assert_eq!(visible[0].id, replacement.id);

            let by_external_id = store
                .get_by_external_id("doc-123#chunk-1")
                .await
                .unwrap()
                .unwrap();
            assert_eq!(by_external_id.id, replacement.id);

            let history = store
                .list_with_options(None, None, LifecycleQueryOptions::new(false, true))
                .await
                .unwrap();
            assert_eq!(history.len(), 2);
            assert!(history.iter().any(|record| record.id == first.id));
            assert!(history.iter().any(|record| record.id == replacement.id));
        });
    }

    fn upsert_record(id: &str, external_id: &str, pivot: f32) -> ContextRecord {
        let mut record = text_record(id, pivot);
        record.external_id = Some(external_id.to_string());
        record
    }

    #[test]
    fn upsert_many_inserts_new_records() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();

            let batch = vec![
                upsert_record("a", "ext-a", 0.0),
                upsert_record("b", "ext-b", 1.0),
            ];
            let results = store.upsert_many_by_external_id(batch).await.unwrap();

            assert_eq!(results.len(), 2);
            assert!(results.iter().all(|r| r.inserted));
            assert!(results.iter().all(|r| r.replaced_id.is_none()));
            assert_eq!(results[0].version, results[1].version);

            let visible = store.list(None, None).await.unwrap();
            assert_eq!(visible.len(), 2);
        });
    }

    #[test]
    fn upsert_many_replaces_existing_and_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();

            let first = vec![
                upsert_record("a1", "ext-a", 0.0),
                upsert_record("b1", "ext-b", 1.0),
            ];
            store.upsert_many_by_external_id(first).await.unwrap();

            // Re-apply with new ids: both should replace (supersede) the
            // originals, and only the successors remain visible.
            let second = vec![
                upsert_record("a2", "ext-a", 2.0),
                upsert_record("b2", "ext-b", 3.0),
            ];
            let results = store.upsert_many_by_external_id(second).await.unwrap();

            assert!(results.iter().all(|r| !r.inserted));
            assert_eq!(results[0].replaced_id.as_deref(), Some("a1"));
            assert_eq!(results[1].replaced_id.as_deref(), Some("b1"));
            assert_eq!(results[0].record.supersedes_id.as_deref(), Some("a1"));

            let visible = store.list(None, None).await.unwrap();
            assert_eq!(visible.len(), 2);
            let visible_ids: HashSet<&str> = visible.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(
                visible_ids,
                HashSet::from(["a2", "b2"]),
                "only the successors should be visible"
            );

            // Idempotent re-application again still leaves one visible per key.
            let third = vec![
                upsert_record("a3", "ext-a", 4.0),
                upsert_record("b3", "ext-b", 5.0),
            ];
            store.upsert_many_by_external_id(third).await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 2);
        });
    }

    #[test]
    fn upsert_many_handles_mixed_insert_and_replace() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            store
                .upsert_many_by_external_id(vec![upsert_record("a1", "ext-a", 0.0)])
                .await
                .unwrap();

            let batch = vec![
                upsert_record("a2", "ext-a", 1.0), // replace
                upsert_record("c1", "ext-c", 2.0), // insert
            ];
            let results = store.upsert_many_by_external_id(batch).await.unwrap();

            assert_eq!(results.len(), 2);
            assert!(!results[0].inserted);
            assert_eq!(results[0].replaced_id.as_deref(), Some("a1"));
            assert!(results[1].inserted);
            assert!(results[1].replaced_id.is_none());

            let visible_ids: HashSet<String> = store
                .list(None, None)
                .await
                .unwrap()
                .into_iter()
                .map(|r| r.id)
                .collect();
            assert_eq!(
                visible_ids,
                HashSet::from(["a2".to_string(), "c1".to_string()])
            );
        });
    }

    #[test]
    fn upsert_many_rejects_within_batch_duplicate_external_id() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let batch = vec![
                upsert_record("a", "dup", 0.0),
                upsert_record("b", "dup", 1.0),
            ];
            let err = store.upsert_many_by_external_id(batch).await.unwrap_err();
            assert!(
                err.to_string()
                    .contains("duplicate external_id 'dup' in batch"),
                "unexpected error: {err}"
            );
            // Nothing was written (all-or-nothing).
            assert_eq!(store.list(None, None).await.unwrap().len(), 0);
        });
    }

    #[test]
    fn upsert_many_rejects_within_batch_duplicate_id() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let batch = vec![
                upsert_record("same", "ext-a", 0.0),
                upsert_record("same", "ext-b", 1.0),
            ];
            let err = store.upsert_many_by_external_id(batch).await.unwrap_err();
            assert!(
                err.to_string().contains("duplicate id 'same' in batch"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn upsert_many_rejects_missing_external_id() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();

            let no_ext = vec![text_record("a", 0.0)];
            let err = store.upsert_many_by_external_id(no_ext).await.unwrap_err();
            assert!(err.to_string().contains("external_id"), "unexpected: {err}");

            let mut empty = text_record("b", 0.0);
            empty.external_id = Some(String::new());
            let err = store
                .upsert_many_by_external_id(vec![empty])
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains("non-empty external_id"),
                "unexpected: {err}"
            );
        });
    }

    #[test]
    fn upsert_many_rejects_id_collision_with_store() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            store.add(&[text_record("taken", 0.0)]).await.unwrap();

            let batch = vec![upsert_record("taken", "ext-a", 1.0)];
            let err = store.upsert_many_by_external_id(batch).await.unwrap_err();
            assert!(
                err.to_string().contains("id 'taken'")
                    && err.to_string().contains("already exists"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn upsert_many_empty_batch_is_noop() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let results = store.upsert_many_by_external_id(Vec::new()).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn upsert_many_matches_single_upsert_with_btree_index() {
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

            // Seed one record via the single-record path.
            store
                .upsert_by_external_id(upsert_record("a1", "ext-a", 0.0))
                .await
                .unwrap();

            // Batch replaces ext-a and inserts ext-b through the indexed path.
            let results = store
                .upsert_many_by_external_id(vec![
                    upsert_record("a2", "ext-a", 1.0),
                    upsert_record("b1", "ext-b", 2.0),
                ])
                .await
                .unwrap();
            assert_eq!(results[0].replaced_id.as_deref(), Some("a1"));
            assert!(results[1].inserted);

            assert_eq!(
                store.get_by_external_id("ext-a").await.unwrap().unwrap().id,
                "a2"
            );
        });
    }

    #[test]
    fn update_by_external_id_patches_mutable_fields_and_preserves_payload() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();

            let mut record = text_record("stable", 0.0);
            record.external_id = Some("doc-123#chunk-1".to_string());
            record.metadata = Some(serde_json::json!({"revision": 1}));
            store.add(std::slice::from_ref(&record)).await.unwrap();

            let patch = RecordPatch {
                bot_id: Some("bot-a".to_string()),
                session_id: Some("session-a".to_string()),
                metadata: Some(serde_json::json!({"revision": 2, "confidence": 0.9})),
                relationships: Some(vec![Relationship {
                    target_id: "doc-123".to_string(),
                    relation: "derived_from".to_string(),
                    weight: None,
                }]),
                ..Default::default()
            };
            let updated = store
                .update_by_external_id("doc-123#chunk-1", patch)
                .await
                .unwrap()
                .unwrap();

            assert_eq!(updated.replaced_id, record.id);
            assert_ne!(updated.record.id, record.id);
            assert_eq!(updated.record.external_id, record.external_id);
            assert_eq!(updated.record.text_payload, record.text_payload);
            assert_eq!(updated.record.embedding, record.embedding);
            assert_eq!(updated.record.bot_id.as_deref(), Some("bot-a"));
            assert_eq!(updated.record.session_id.as_deref(), Some("session-a"));
            assert_eq!(
                updated.record.metadata,
                Some(serde_json::json!({"revision": 2, "confidence": 0.9}))
            );
            assert_eq!(updated.record.relationships.len(), 1);
            assert_eq!(
                updated.record.supersedes_id.as_deref(),
                Some(record.id.as_str())
            );

            let visible = store
                .get_by_external_id("doc-123#chunk-1")
                .await
                .unwrap()
                .unwrap();
            assert_eq!(visible.id, updated.record.id);

            let history = store
                .list_with_options(None, None, LifecycleQueryOptions::new(false, true))
                .await
                .unwrap();
            assert_eq!(history.len(), 2);
            assert!(history.iter().any(|item| item.id == record.id));
            assert!(history.iter().any(|item| item.id == updated.record.id));
        });
    }

    #[test]
    fn deferred_embedding_patch_makes_raw_record_searchable() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();

            // Raw-first capture: append source chunks without embeddings.
            let mut by_ext = text_record("raw-ext", 0.0);
            by_ext.embedding = None;
            by_ext.external_id = Some("doc-1#chunk-1".to_string());
            let mut by_id = text_record("raw-id", 0.0);
            by_id.embedding = None;
            by_id.external_id = None;
            store.add(&[by_ext.clone(), by_id.clone()]).await.unwrap();

            // Records without an embedding are invisible to vector search.
            let query = make_embedding(1.0);
            assert!(store.search(&query, Some(10)).await.unwrap().is_empty());

            // Enrich-later: patch the embedding by external_id...
            let enriched_ext = store
                .update_by_external_id(
                    "doc-1#chunk-1",
                    RecordPatch {
                        embedding: Some(make_embedding(1.0)),
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(enriched_ext.record.embedding, Some(make_embedding(1.0)));
            // Raw payload is carried forward onto the superseding record.
            assert_eq!(enriched_ext.record.text_payload, by_ext.text_payload);

            // ...and by internal id.
            let enriched_id = store
                .update_by_id(
                    &by_id.id,
                    RecordPatch {
                        embedding: Some(make_embedding(0.0)),
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(enriched_id.record.embedding, Some(make_embedding(0.0)));

            // Both records now participate in vector search.
            let results = store.search(&query, Some(10)).await.unwrap();
            let ids: Vec<&str> = results.iter().map(|r| r.record.id.as_str()).collect();
            assert!(ids.contains(&enriched_ext.record.id.as_str()));
            assert!(ids.contains(&enriched_id.record.id.as_str()));
            // The query matches the external_id record exactly (distance 0).
            assert_eq!(results[0].record.id, enriched_ext.record.id);
        });
    }

    #[test]
    fn relationships_roundtrip_and_support_related_lookup() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let mut related = text_record("related", 0.0);
            related.relationships = vec![
                Relationship {
                    target_id: "doc-1#chunk-1".to_string(),
                    relation: "cites".to_string(),
                    weight: Some(0.75),
                },
                Relationship {
                    target_id: "service-a".to_string(),
                    relation: "mentions".to_string(),
                    weight: None,
                },
            ];
            let unrelated = text_record("unrelated", 1.0);
            store.add(&[related.clone(), unrelated]).await.unwrap();

            let listed = store.list(None, None).await.unwrap();
            let roundtrip = listed
                .iter()
                .find(|record| record.id == related.id)
                .unwrap();
            assert_eq!(roundtrip.relationships, related.relationships);

            let by_target = store
                .list_related("doc-1#chunk-1", None, None)
                .await
                .unwrap();
            assert_eq!(by_target.len(), 1);
            assert_eq!(by_target[0].id, related.id);

            let by_relation = store
                .list_related("doc-1#chunk-1", Some("cites"), None)
                .await
                .unwrap();
            assert_eq!(by_relation.len(), 1);
            assert_eq!(by_relation[0].id, related.id);

            let wrong_relation = store
                .list_related("doc-1#chunk-1", Some("mentions"), None)
                .await
                .unwrap();
            assert!(wrong_relation.is_empty());
        });
    }

    #[test]
    fn migrate_relationships_column_adds_missing_column() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let schema = Arc::new(ContextStore::schema_with_options(
                &HashSet::new(),
                true,
                true,
                false,
                true,
                true,
                DEFAULT_EMBEDDING_DIM,
                DistanceMetric::default(),
            ));
            let empty_batch = RecordBatch::new_empty(schema.clone());
            let batches = RecordBatchIterator::new(
                vec![Ok::<RecordBatch, ArrowError>(empty_batch)].into_iter(),
                schema,
            );
            Dataset::write(
                batches,
                &uri,
                Some(WriteParams {
                    mode: WriteMode::Create,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            let mut store = ContextStore::open(&uri).await.unwrap();
            assert!(!store.has_relationships_column());

            let mut record = text_record("with-relationships", 0.0);
            record.relationships.push(Relationship {
                target_id: "target".to_string(),
                relation: "mentions".to_string(),
                weight: None,
            });
            let err = store.add(std::slice::from_ref(&record)).await.unwrap_err();
            assert!(
                err.to_string().contains("migrate_relationships_column"),
                "unexpected error: {err}"
            );

            assert!(store.migrate_relationships_column().await.unwrap());
            assert!(store.has_relationships_column());
            assert!(!store.migrate_relationships_column().await.unwrap());

            store.add(std::slice::from_ref(&record)).await.unwrap();
            let roundtrip = store.get_by_id(&record.id).await.unwrap().unwrap();
            assert_eq!(roundtrip.relationships, record.relationships);
        });
    }

    #[test]
    fn add_rejects_duplicate_external_id() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
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
            let store = ContextStore::open(&uri).await.unwrap();
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
    fn add_rejects_duplicate_id_against_existing() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            store.add(&[text_record("dup", 0.0)]).await.unwrap();

            let err = store.add(&[text_record("dup", 1.0)]).await.unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("id 'dup'") && message.contains("already exists"),
                "unexpected error message: {message}"
            );
        });
    }

    #[test]
    fn add_rejects_duplicate_id_within_batch() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let err = store
                .add(&[text_record("same", 0.0), text_record("same", 1.0)])
                .await
                .unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("duplicate id 'same' in batch"),
                "unexpected error message: {message}"
            );
        });
    }

    #[test]
    fn add_rejects_duplicate_external_id_within_batch() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let mut first = text_record("a", 0.0);
            first.external_id = Some("ext".to_string());
            let mut second = text_record("b", 1.0);
            second.external_id = Some("ext".to_string());

            let err = store.add(&[first, second]).await.unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("duplicate external_id 'ext' in batch"),
                "unexpected error message: {message}"
            );
        });
    }

    /// A record removed via tombstone frees its `external_id` for reuse:
    /// validation skips tombstones, so a later insert with the same
    /// `external_id` succeeds.
    #[test]
    fn add_allows_external_id_reuse_after_delete() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let mut first = text_record("a", 0.0);
            first.external_id = Some("ext".to_string());
            store.add(std::slice::from_ref(&first)).await.unwrap();
            assert!(store.delete_by_external_id("ext").await.unwrap());

            let mut reused = text_record("b", 1.0);
            reused.external_id = Some("ext".to_string());
            store
                .add(std::slice::from_ref(&reused))
                .await
                .expect("external_id should be reusable after delete");

            let visible = store.get_by_external_id("ext").await.unwrap().unwrap();
            assert_eq!(visible.id, reused.id);
        });
    }

    /// A record removed via tombstone frees its `id` for reuse.
    #[test]
    fn add_allows_id_reuse_after_delete() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let first = text_record("dup", 0.0);
            store.add(std::slice::from_ref(&first)).await.unwrap();
            assert!(store.delete_by_id("dup").await.unwrap());

            store
                .add(&[text_record("dup", 1.0)])
                .await
                .expect("id should be reusable after delete");

            let visible = store.get_by_id("dup").await.unwrap().unwrap();
            assert_eq!(visible.id, "dup");
        });
    }

    /// A superseded (non-tombstone) record still reserves its `external_id`,
    /// so a plain `add` with that `external_id` is rejected.
    #[test]
    fn add_rejects_external_id_after_supersede() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            let mut first = text_record("a", 0.0);
            first.external_id = Some("ext".to_string());
            store.upsert_by_external_id(first).await.unwrap();

            let mut successor = text_record("b", 1.0);
            successor.external_id = Some("ext".to_string());
            store.upsert_by_external_id(successor).await.unwrap();

            // The original is now superseded (non-tombstone) and still present,
            // so its external_id is taken.
            let mut conflict = text_record("c", 2.0);
            conflict.external_id = Some("ext".to_string());
            let err = store.add(&[conflict]).await.unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("external_id 'ext'") && message.contains("already exists"),
                "unexpected error message: {message}"
            );
        });
    }

    /// Uniqueness validation must behave identically when an `id` scalar index
    /// is configured (the indexed-lookup path), covering both a rejected
    /// collision and a permitted reuse-after-delete.
    #[test]
    fn validate_uniqueness_with_btree_index() {
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

            let mut first = text_record("idx-a", 0.0);
            first.external_id = Some("ext".to_string());
            store.add(std::slice::from_ref(&first)).await.unwrap();

            // Duplicate id rejected via the indexed lookup.
            let dup_id = store.add(&[text_record("idx-a", 1.0)]).await.unwrap_err();
            assert!(
                dup_id.to_string().contains("id 'idx-a'")
                    && dup_id.to_string().contains("already exists")
            );

            // Duplicate external_id rejected.
            let mut dup_ext = text_record("idx-b", 1.0);
            dup_ext.external_id = Some("ext".to_string());
            let dup_ext_err = store.add(&[dup_ext]).await.unwrap_err();
            assert!(
                dup_ext_err.to_string().contains("external_id 'ext'")
                    && dup_ext_err.to_string().contains("already exists")
            );

            // After delete, both keys become reusable.
            assert!(store.delete_by_id("idx-a").await.unwrap());
            let mut reused = text_record("idx-a", 2.0);
            reused.external_id = Some("ext".to_string());
            store
                .add(std::slice::from_ref(&reused))
                .await
                .expect("keys should be reusable after delete with index configured");
        });
    }

    /// Validation stays correct as the store grows: a duplicate is rejected and
    /// a fresh key accepted against a store with many historical records,
    /// exercising the projected/filtered scan path over a larger dataset.
    #[test]
    fn validate_uniqueness_against_large_store() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let options = ContextStoreOptions {
                id_index_type: IdIndexType::BTree,
                ..Default::default()
            };
            let store = ContextStore::open_with_options(&uri, options)
                .await
                .unwrap();

            for i in 0..300 {
                let mut record = text_record(&format!("rec-{i}"), i as f32);
                record.external_id = Some(format!("ext-{i}"));
                store.add(std::slice::from_ref(&record)).await.unwrap();
            }

            // Existing id and external_id both rejected.
            let mut dup = text_record("rec-150", 0.0);
            dup.external_id = Some("ext-999".to_string());
            assert!(store
                .add(&[dup])
                .await
                .unwrap_err()
                .to_string()
                .contains("id 'rec-150'"));

            let mut dup_ext = text_record("rec-new", 0.0);
            dup_ext.external_id = Some("ext-42".to_string());
            assert!(store
                .add(&[dup_ext])
                .await
                .unwrap_err()
                .to_string()
                .contains("external_id 'ext-42'"));

            // A fresh record still inserts cleanly.
            let mut fresh = text_record("rec-300", 0.0);
            fresh.external_id = Some("ext-300".to_string());
            store.add(std::slice::from_ref(&fresh)).await.unwrap();
            assert!(store.get_by_id("rec-300").await.unwrap().is_some());
        });
    }

    /// Benchmark guarding against the O(N) regression: with an `id` index,
    /// per-append validation cost should not grow proportionally to store size.
    /// Ignored by default (timing-sensitive); run with
    /// `cargo test -p lance-context-core -- --ignored append_cost`.
    #[test]
    #[ignore = "timing-sensitive benchmark; run explicitly with --ignored"]
    fn append_cost_does_not_grow_linearly() {
        use std::time::Instant;

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

            // Time a window of single-record appends at a given store size,
            // compacting first so the cost reflects validation against the base
            // table rather than accumulated MemWAL generations.
            async fn time_window(store: &mut ContextStore, tag: &str, window: usize) -> f64 {
                store.compact(None).await.unwrap();
                let start = Instant::now();
                for i in 0..window {
                    let id = format!("{tag}-probe-{i}");
                    store.add(&[text_record(&id, i as f32)]).await.unwrap();
                }
                start.elapsed().as_secs_f64() / window as f64
            }

            // Grow the store to `count` rows using batched appends (few commits)
            // so the benchmark isolates per-call validation cost, not raw write
            // throughput.
            async fn seed(store: &mut ContextStore, tag: &str, count: usize) {
                let chunk = 100;
                let mut i = 0;
                while i < count {
                    let batch: Vec<ContextRecord> = (i..(i + chunk).min(count))
                        .map(|j| text_record(&format!("{tag}-seed-{j}"), j as f32))
                        .collect();
                    store.add(&batch).await.unwrap();
                    i += chunk;
                }
                store.compact(None).await.unwrap();
            }

            let window = 30;
            seed(&mut store, "small", 100).await;
            let small = time_window(&mut store, "small", window).await;

            seed(&mut store, "big", 2000).await;
            let large = time_window(&mut store, "big", window).await;

            let ratio = large / small.max(f64::EPSILON);
            eprintln!(
                "append per-call: small={small:.6}s large={large:.6}s ratio={ratio:.2} (store grew ~20x)"
            );
            assert!(
                ratio < 8.0,
                "append cost appears to scale with store size (ratio {ratio:.2}); \
                 expected roughly constant per-call validation"
            );
        });
    }

    /// `external_id` values are caller-supplied and flow into a SQL `IN (...)`
    /// filter, so embedded single quotes must be escaped rather than break the
    /// scan. Both insertion and duplicate detection must handle them.
    #[test]
    fn validation_handles_external_id_with_single_quote() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let tricky = "o'brien#chunk-1";

            let mut first = text_record("a", 0.0);
            first.external_id = Some(tricky.to_string());
            store.add(std::slice::from_ref(&first)).await.unwrap();

            // Re-using the same quoted external_id is still detected as a dup.
            let mut dup = text_record("b", 1.0);
            dup.external_id = Some(tricky.to_string());
            let err = store.add(&[dup]).await.unwrap_err();
            assert!(
                err.to_string().contains("already exists"),
                "unexpected error message: {err}"
            );

            // A different quoted external_id inserts cleanly.
            let mut other = text_record("c", 2.0);
            other.external_id = Some("d'angelo#chunk-2".to_string());
            store.add(std::slice::from_ref(&other)).await.unwrap();
            assert!(store
                .get_by_external_id("d'angelo#chunk-2")
                .await
                .unwrap()
                .is_some());
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
    fn add_is_visible_on_return_by_default() {
        // The default `seal_on_add: true` contract. `ContextStore`'s uniqueness
        // validation, upsert and tombstones all read the table back, so an `add`
        // that returned before sealing would silently break them.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            store.add(&[text_record("r1", 0.0)]).await.unwrap();

            // No flush: the row must already be readable.
            assert_eq!(store.list(None, None).await.unwrap().len(), 1);
            assert!(store.get_by_id("r1").await.unwrap().is_some());
        });
    }

    #[test]
    fn deferred_seal_makes_add_durable_but_invisible_until_flush() {
        // Opting into `RolloutStore`'s write profile: `add` is a durable WAL
        // append only, and visibility waits for an explicit flush. Mirrors
        // `rollout_store::tests::add_is_durable_but_not_visible_until_flush`.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open_with_options(
                &uri,
                ContextStoreOptions {
                    seal_on_add: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

            store.add(&[text_record("r1", 0.0)]).await.unwrap();
            assert_eq!(
                store.list(None, None).await.unwrap().len(),
                0,
                "deferred seal must not publish the row"
            );

            store.flush().await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 1);
            assert!(store.get_by_id("r1").await.unwrap().is_some());
        });
    }

    #[test]
    fn wal_generations_merge_into_the_base_table() {
        // This store previously never merged: flushed generations accumulated
        // under `_mem_wal/` forever and every read unioned all of them. Each
        // sealing `add` emits one generation, so after N adds the merge must
        // reclaim them and leave every row readable from the base table.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = ContextStore::open(&uri).await.unwrap();
            for i in 0..3 {
                store
                    .add(&[text_record(&format!("r{i}"), i as f32)])
                    .await
                    .unwrap();
            }

            assert!(
                store.pending_wal_generations().await.unwrap() > 0,
                "sealing adds should leave generations pending"
            );

            let reclaimed = store.cleanup_wal().await.unwrap();
            assert!(reclaimed > 0, "cleanup must merge the pending generations");
            assert_eq!(store.pending_wal_generations().await.unwrap(), 0);

            // Rows survive the merge and are still readable.
            let ids: Vec<String> = store
                .list(None, None)
                .await
                .unwrap()
                .iter()
                .map(|r| r.id.clone())
                .collect();
            assert_eq!(ids.len(), 3);
            for i in 0..3 {
                assert!(ids.contains(&format!("r{i}")));
            }
        });
    }

    // `test_region_id_derivation_explicit` was deleted with the per-conversation
    // MemWAL sharding it covered. Writes now go to the shard owned by this
    // writer instance (see `ContextStoreOptions::shard_id`), matching
    // `RolloutStore` -- one instance owns exactly one shard, so the epoch-fencing
    // invariant holds by construction. `test_add_multiple_regions` below still
    // pins the user-visible behavior: records with differing bot/session ids
    // round-trip through a single store.

    #[test]
    fn test_add_multiple_regions() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();

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
            let store = ContextStore::open_with_options(&uri, options)
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
            let store = ContextStore::open_with_options(&uri, options)
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
            let store = ContextStore::open_with_options(&uri, options)
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
            let indices = store.base.dataset.load_indices().await.unwrap();
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
            let indices = store.base.dataset.load_indices().await.unwrap();
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
            let indices = store.base.dataset.load_indices().await.unwrap();
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

            let indices = store.base.dataset.load_indices().await.unwrap();
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

            let indices = store.base.dataset.load_indices().await.unwrap();
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

    #[test]
    fn projection_excludes_binary_but_keeps_metadata() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let mut record = text_record("img", 0.0);
            record.content_type = "image/png".to_string();
            record.binary_payload = Some(vec![1, 2, 3, 4]);
            store.add(std::slice::from_ref(&record)).await.unwrap();

            // Default read still includes the bytes.
            let full = store.list(None, None).await.unwrap();
            assert_eq!(full[0].binary_payload.as_deref(), Some(&[1, 2, 3, 4][..]));
            assert!(full[0].embedding.is_some());

            // Projected read drops binary, keeps metadata + embedding.
            let projected = store
                .list_filtered_projected(
                    None,
                    None,
                    None,
                    LifecycleQueryOptions::default(),
                    ReadProjection::without_binary(),
                )
                .await
                .unwrap();
            assert_eq!(projected.len(), 1);
            assert!(projected[0].binary_payload.is_none());
            assert_eq!(projected[0].id, "img");
            assert_eq!(projected[0].content_type, "image/png");
            assert!(projected[0].embedding.is_some());

            // metadata_only drops embedding too.
            let meta = store
                .list_filtered_projected(
                    None,
                    None,
                    None,
                    LifecycleQueryOptions::default(),
                    ReadProjection::metadata_only(),
                )
                .await
                .unwrap();
            assert!(meta[0].binary_payload.is_none());
            assert!(meta[0].embedding.is_none());
            assert_eq!(meta[0].id, "img");

            // get_blob fetches the bytes on demand.
            let blob = store.get_blob("img").await.unwrap();
            assert_eq!(blob.as_deref(), Some(&[1, 2, 3, 4][..]));
            assert!(store.get_blob("missing").await.unwrap().is_none());
        });
    }

    #[test]
    fn search_projection_excludes_binary_keeps_ranking() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = ContextStore::open(&uri).await.unwrap();
            let mut a = text_record("a", 0.0);
            a.binary_payload = Some(vec![9, 9, 9]);
            let mut b = text_record("b", 1.0);
            b.binary_payload = Some(vec![8, 8, 8]);
            store.add(&[a, b]).await.unwrap();

            let query = make_embedding(0.0);
            let results = store
                .search_filtered_projected(
                    &query,
                    Some(5),
                    None,
                    LifecycleQueryOptions::default(),
                    ReadProjection::without_binary(),
                )
                .await
                .unwrap();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0].record.id, "a"); // closest to query pivot 0.0
            assert!(results.iter().all(|r| r.record.binary_payload.is_none()));
            // embedding kept by default projection (without_binary keeps embedding)
            assert!(results[0].record.embedding.is_some());
        });
    }
}
