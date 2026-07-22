//! Storage surface for the RL rollout schema.
//!
//! [`RolloutStore`] is a second, independent first-class store alongside
//! [`crate::store::ContextStore`]. It owns its own Lance dataset and Arrow
//! schema (see [`crate::rollout::RolloutRecord`] and spec §5) but reuses the
//! schema-agnostic infrastructure: MemWAL ingest and the relationship graph.
//!
//! # Artifact bytes are stored inline, not blob-v2 offloaded
//!
//! `binary_payload` holds artifact bytes (spec §6) as a plain inline
//! `LargeBinary` column, *not* a blob-v2 offloaded column. Rollout reads go
//! exclusively through the MemWAL LSM scanner
//! ([`RolloutStore::lsm_scanner`]), which has no blob-materialization step: a
//! blob-v2 (`lance-encoding:blob`) column reads back as `None` through it, so
//! [`RolloutStore::get_blob`] could never return the bytes. Inline storage is
//! therefore the only encoding that round-trips. To keep the "learner doesn't
//! pay for artifacts" property (spec §2), list-style scans project the column
//! out (see [`RolloutStore::list`]) rather than relying on physical offload.
//!
//! # Distributed writes: server-id sharding
//!
//! High fan-in rollout ingest (spec §2: thousands of workers) is served by
//! writing through Lance's **MemWAL** rather than a plain [`Dataset::append`].
//! Each REST server instance owns exactly one MemWAL shard, keyed by its
//! server/instance id (see [`RolloutStoreOptions::shard_id`]). Because a shard
//! has a single active writer per instance and no two instances share a shard,
//! the MemWAL epoch-fencing invariant holds *by construction* — there is never
//! a write war, no matter how a load balancer spreads worker requests across
//! instances. See `specs/rollout-deployment.md`.
//!
//! `MemWAL close-per-append` makes each write durable on object storage before
//! `add` returns, and the read path ([`RolloutStore::lsm_scanner`]) rebuilds
//! purely from object storage (base table ∪ every shard's flushed
//! generations). So any instance reads every instance's writes — reads are not
//! pinned to the writer node.
//!
//! # Reproducibility without `checkout`
//!
//! MemWAL writes land in a separate `_mem_wal/` manifest namespace and do not
//! bump the base dataset version, so per-`add` [`RolloutStore::checkout`] no
//! longer isolates a single append (it did under the old atomic-append path).
//! This is intentional: rollout rows are **append-only and immutable**, so "the
//! exact rollouts that trained checkpoint N" is a *filter over immutable rows*
//! (e.g. by `policy_version`), not a table snapshot — reproducible because the
//! rows never change. `checkout` remains available for base-table time-travel.

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
    StringArray, TimestampMicrosecondArray, UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use chrono::{DateTime, Utc};
use futures::{stream, StreamExt, TryStreamExt};
use lance::dataset::mem_wal::{
    DatasetMemWalExt, LsmScanner, ShardManifestStore, ShardSnapshot, ShardWriter, ShardWriterConfig,
};
use lance::dataset::optimize::{compact_files, CompactionMetrics, CompactionOptions};
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};
use lance_index::mem_wal::{ShardManifest, MEM_WAL_INDEX_NAME};
use lance_index::scalar::ScalarIndexParams;
use lance_index::IndexType;
use serde_json::Value;
use tracing::{info, warn};
use uuid::Uuid;

use crate::rollout::RolloutRecord;
use crate::store::{
    column_as, column_as_optional, relationship_field, relationship_list_item_field,
    relationship_struct_builder, relationships_from_list, timestamp_from_micros, CompactionConfig,
    CompactionStats, RELATIONSHIPS_COLUMN,
};

/// Number of shard manifest files to scan per batch when discovering the latest
/// shard state (mirrors the constant used by `ContextStore`).
const DEFAULT_MANIFEST_SCAN_BATCH_SIZE: usize = 16;

/// Maximum number of shard manifests or flushed-generation datasets opened
/// concurrently while collecting observability metrics.
const DEFAULT_OBSERVE_CONCURRENCY: usize = 16;

/// Name of the scalar index on the base table's `id` column. Kept in sync with
/// `ContextStore`'s `ID_INDEX_NAME` so both tables index `id` under one name.
const ROLLOUT_ID_INDEX_NAME: &str = "id_idx";

/// Read-only observability snapshot of a rollout store.
///
/// Produced by [`RolloutStore::observe`] from base-table and MemWAL metadata.
/// Consumed by the control-plane stats scanner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolloutObservation {
    /// Logical row count across the base table and every flushed MemWAL shard.
    pub row_count: i64,
    /// Number of fragments in the base table.
    pub fragment_count: i64,
    /// Current base dataset manifest version.
    pub version: u64,
    /// Manifest timestamp, Unix milliseconds — when the base table last changed.
    pub last_updated: i64,
    /// Flushed MemWAL generations pending merge across all shards.
    pub pending_wal_generations: i64,
}

/// Exact-match filters for rollout record browsing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RolloutFilters {
    pub id: Option<String>,
    pub rollout_id: Option<String>,
    pub problem_id: Option<String>,
    pub dataset: Option<String>,
    pub role: Option<String>,
    pub content_type: Option<String>,
    pub policy_version: Option<String>,
    pub artifact_type: Option<String>,
    pub include_in_training: Option<bool>,
}

impl RolloutFilters {
    pub fn from_json_value(value: Value) -> Result<Self, String> {
        let Value::Object(object) = value else {
            return Err("rollout filters must be a JSON object".to_string());
        };

        let mut filters = Self::default();
        for (key, value) in object {
            let string_value = || {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("rollout filter '{key}' must be a string"))
            };
            match key.as_str() {
                "rollout_id" => filters.rollout_id = Some(string_value()?),
                "problem_id" => filters.problem_id = Some(string_value()?),
                "policy_version" => filters.policy_version = Some(string_value()?),
                "role" => filters.role = Some(string_value()?),
                "include_in_training" => {
                    filters.include_in_training = Some(value.as_bool().ok_or_else(|| {
                        "rollout filter 'include_in_training' must be a boolean".to_string()
                    })?);
                }
                "artifact_type" => filters.artifact_type = Some(string_value()?),
                _ => return Err(format!("unsupported rollout filter '{key}'")),
            }
        }
        Ok(filters)
    }

    fn expression(&self) -> Option<String> {
        let mut clauses = Vec::new();
        for (column, value) in [
            ("id", self.id.as_deref()),
            ("rollout_id", self.rollout_id.as_deref()),
            ("problem_id", self.problem_id.as_deref()),
            ("dataset", self.dataset.as_deref()),
            ("role", self.role.as_deref()),
            ("content_type", self.content_type.as_deref()),
            ("policy_version", self.policy_version.as_deref()),
            ("artifact_type", self.artifact_type.as_deref()),
        ] {
            if let Some(value) = value.filter(|value| !value.is_empty()) {
                clauses.push(format!("{column} = '{}'", value.replace('\'', "''")));
            }
        }
        if let Some(value) = self.include_in_training {
            clauses.push(format!("include_in_training = {value}"));
        }
        (!clauses.is_empty()).then(|| clauses.join(" AND "))
    }
}

/// One server-side paginated rollout query result.
#[derive(Debug, Clone)]
pub struct RolloutPage {
    pub records: Vec<RolloutRecord>,
    pub has_more: bool,
}

/// Which data source a rollout list scan reads.
///
/// A rollout store's rows live in two tiers: the compacted **base table**
/// (`self.dataset`) and the pending **MemWAL** generations that have been
/// flushed but not yet merged into the base table. The default browse path
/// reads only the base table so list latency is independent of WAL backlog
/// (each pending generation is a separate object-store open); callers that need
/// the not-yet-merged tail or full cross-tier consistency opt into `Wal`/`All`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ListSource {
    /// Scan only the base table (`self.dataset`), skipping all MemWAL
    /// generations. Fast and bounded; may lag the most recent (un-merged)
    /// writes. This is the default.
    #[default]
    Fragments,
    /// Scan only the flushed MemWAL generations (the not-yet-merged tail),
    /// excluding the base table.
    Wal,
    /// Scan the base table unioned with every flushed MemWAL generation — fully
    /// consistent, and the behavior of the historical union read path.
    All,
}

/// Configuration for opening a [`RolloutStore`].
#[derive(Debug, Clone, Default)]
pub struct RolloutStoreOptions {
    /// Object-store credentials/config (e.g. S3), forwarded to Lance.
    pub storage_options: Option<HashMap<String, String>>,
    /// Stable identity of the writing server instance (e.g. a StatefulSet
    /// ordinal hostname like `rollout-0`). Rollout writes go to the MemWAL
    /// shard derived from this id, so each instance owns exactly one shard and
    /// no two instances ever contend for the same shard. `None` falls back to a
    /// single fixed shard (`"default"`), which is correct for single-instance
    /// deployments but must be set per-instance when running multiple writers.
    /// See `specs/rollout-deployment.md`.
    pub shard_id: Option<String>,
    /// Count-triggered self-merge threshold. After an append flushes a new
    /// generation, if this instance's own shard has accumulated at least this
    /// many un-merged flushed generations, `add` synchronously merges them into
    /// the base table and drains the shard's `flushed_generations` back to empty
    /// (see [`RolloutStore::merge_own_shard`]). This bounds read amplification:
    /// without it, `_mem_wal/{shard}/` generations accumulate forever and every
    /// read unions all of them (spec §6).
    ///
    /// The merge runs on the instance that owns the shard, reusing its own
    /// writer epoch, so it never fences a concurrent writer — each instance
    /// merges only the shard it writes. It is synchronous, so the append that
    /// crosses the threshold pays the merge latency (a periodic tail-latency
    /// spike; see the deployment doc).
    ///
    /// `None` or `0` disables self-merge (the 0.6.0 behavior: generations
    /// accumulate and are unioned at read time).
    pub merge_after_generations: Option<usize>,
}

/// A Lance-backed store for RL rollout trajectories.
pub struct RolloutStore {
    dataset: Dataset,
    /// MemWAL shard this instance writes to (derived from `shard_id`).
    write_shard: Uuid,
    /// Object-store options, retained so a self-merge can re-append flushed
    /// generation data into the base table with the same credentials.
    storage_options: Option<HashMap<String, String>>,
    /// Self-merge threshold; see [`RolloutStoreOptions::merge_after_generations`].
    /// `0` (or `None` normalized to 0) disables it.
    merge_after_generations: usize,
    /// Timestamp of the last successful [`Self::compact`] on this handle.
    last_compaction: Option<DateTime<Utc>>,
    /// Number of successful compactions performed by this handle.
    total_compactions: u64,
    /// Error message from the most recent failed compaction on this handle.
    last_compaction_error: Option<String>,
    /// Resident MemWAL writer for this instance's shard. Opened lazily on the
    /// first [`Self::add`] and reused across appends, so the shard epoch is
    /// claimed once and the object-store connection (TCP/TLS + DNS resolution)
    /// is pooled instead of being torn down and cold-rebuilt on every append.
    ///
    /// Cleared (after an explicit `close`) whenever [`Self::merge_own_shard`]
    /// claims the shard epoch — that claim would fence this writer — so the next
    /// `add` transparently reopens with the fresh epoch. [`ShardWriter`] has no
    /// `Drop`, so it must be closed explicitly to drain its background tasks;
    /// see [`Self::close`] and the `Drop` impl.
    write_writer: Option<ShardWriter>,
}

impl RolloutStore {
    /// Open an existing rollout dataset or create a new one with default
    /// options (`binary_payload` stored inline; see [`RolloutStoreOptions`]).
    ///
    /// Uses the fallback single-shard identity; for a multi-instance deployment
    /// open with [`RolloutStoreOptions::shard_id`] set per instance.
    pub async fn open(uri: &str) -> LanceResult<Self> {
        Self::open_with_options(uri, RolloutStoreOptions::default()).await
    }

    /// Open a rollout dataset with explicit storage and shard configuration.
    /// Creates the dataset if it does not exist.
    pub async fn open_with_options(uri: &str, options: RolloutStoreOptions) -> LanceResult<Self> {
        Self::open_inner(uri, options, true).await
    }

    /// Open an **existing** rollout dataset. Unlike [`Self::open_with_options`],
    /// this does **not** create the dataset when it is absent — it returns the
    /// underlying [`LanceError::DatasetNotFound`] instead.
    ///
    /// This is the read/write path used by the server's lazy cache-fill: a
    /// cache miss must load a store that genuinely exists on object storage
    /// (create-on-absence would silently materialize an empty table for a
    /// mistyped name, masking the 404). Store creation goes exclusively through
    /// [`Self::open_with_options`] on the explicit `create` route.
    pub async fn open_existing_with_options(
        uri: &str,
        options: RolloutStoreOptions,
    ) -> LanceResult<Self> {
        Self::open_inner(uri, options, false).await
    }

    async fn open_inner(
        uri: &str,
        options: RolloutStoreOptions,
        create_if_missing: bool,
    ) -> LanceResult<Self> {
        let storage_options = options.storage_options.clone();
        let write_shard = derive_shard_id(options.shard_id.as_deref());
        let dataset = match Self::load_with_options(uri, storage_options.clone()).await {
            Ok(dataset) => dataset,
            Err(LanceError::DatasetNotFound { .. }) if create_if_missing => {
                Self::create_with_options(uri, storage_options.clone()).await?
            }
            Err(err) => return Err(err),
        };

        Ok(Self {
            dataset,
            write_shard,
            storage_options,
            merge_after_generations: options.merge_after_generations.unwrap_or(0),
            last_compaction: None,
            total_compactions: 0,
            last_compaction_error: None,
            write_writer: None,
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

    /// Refresh this handle to the latest base-table manifest while retaining
    /// its session and metadata caches.
    ///
    /// Long-lived read handles call this before a new request so compaction or
    /// WAL merges committed by another process become visible without paying
    /// the cost of reopening the dataset and rebuilding all session caches.
    pub async fn refresh_latest(&mut self) -> LanceResult<()> {
        self.dataset.checkout_latest().await
    }

    /// Append rollout rows through this instance's MemWAL shard; returns the
    /// current base dataset version.
    ///
    /// The write is routed to the shard derived from the configured
    /// `shard_id`, so concurrent appends from other server instances (each
    /// owning a distinct shard) never contend. Each append is sealed and its
    /// flushed generation committed to the shard manifest before returning, so
    /// the rows are immediately visible to reads on **any** instance — reads
    /// rebuild their view from the shard manifests on object storage (see
    /// [`Self::lsm_scanner`]).
    ///
    /// Unlike the previous close-per-append path, this reuses a single resident
    /// [`ShardWriter`] (see [`Self::ensure_write_writer`]): the shard epoch is
    /// claimed once and the object-store connection is pooled, so an append no
    /// longer pays a cold DNS resolution + TCP/TLS handshake + epoch claim every
    /// time. The per-append work is `put` (WAL-durable) → `force_seal_active`
    /// (freeze this append's memtable) → `wait_for_flush_drain` (await the
    /// generation's manifest commit), which keeps the same cross-instance
    /// read-after-write visibility while dropping the per-append connection
    /// churn that dominated latency.
    ///
    /// The returned value is the base dataset version, which MemWAL appends do
    /// **not** advance; it is retained for API compatibility, not as a per-append
    /// snapshot handle (see the module docs on reproducibility).
    pub async fn add(&mut self, records: &[RolloutRecord]) -> LanceResult<u64> {
        if records.is_empty() {
            return Ok(self.dataset.manifest.version);
        }

        let batch = self.records_to_batch(records)?;

        self.ensure_mem_wal().await?;

        // Reuse the resident writer, retrying once on a fence. A fence means
        // another claimer (in practice only this instance's own merge, which
        // takes the writer down first) bumped the shard epoch out from under us;
        // dropping and reopening the writer re-claims the current epoch. Rollout
        // rows are immutable and de-duplicated by `id` at read time, so a retried
        // append can never double-count.
        self.write_with_resident_writer(&batch).await?;

        // Count-triggered self-merge: if this instance's shard has accumulated
        // enough un-merged flushed generations, fold them into the base table
        // now (spec §6). Bounds read amplification. Runs on the shard's own
        // owner so it never fences a concurrent writer.
        if self.merge_after_generations > 0 {
            self.maybe_merge_own_shard().await?;
        }

        Ok(self.dataset.manifest.version)
    }

    /// Put one batch through the resident writer, seal it into a generation, and
    /// await its manifest commit. Reopens the writer once and retries if the
    /// first attempt is fenced.
    async fn write_with_resident_writer(&mut self, batch: &RecordBatch) -> LanceResult<()> {
        match self.put_seal_drain(batch).await {
            Ok(()) => Ok(()),
            Err(err) if is_fenced_error(&err) => {
                // Drop the fenced writer without close() — its epoch is already
                // dead, so a graceful drain is neither possible nor useful — and
                // reopen against the current epoch for a single retry.
                self.write_writer = None;
                self.put_seal_drain(batch).await
            }
            Err(err) => Err(err),
        }
    }

    /// One append against the resident writer: ensure it is open, `put` the
    /// batch (WAL-durable), `force_seal_active` to freeze it into a generation,
    /// then `wait_for_flush_drain` so the generation is committed to the shard
    /// manifest (and therefore visible to reads on every instance) before we
    /// return.
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

    /// Ensure a resident [`ShardWriter`] for this instance's shard is open,
    /// opening one on first use. Reused across appends: the shard epoch is
    /// claimed once at open and the object-store connection is pooled,
    /// eliminating the per-append open/claim/reconnect that dominated append
    /// latency (a cold DNS resolution + TCP/TLS handshake on every append).
    async fn ensure_write_writer(&mut self) -> LanceResult<()> {
        if self.write_writer.is_some() {
            return Ok(());
        }
        let config = ShardWriterConfig {
            shard_id: self.write_shard,
            ..Default::default()
        };
        let writer = self
            .dataset
            .mem_wal_writer(self.write_shard, config)
            .await?;
        self.write_writer = Some(writer);
        Ok(())
    }

    /// Gracefully close the resident writer, draining its background tasks.
    ///
    /// [`ShardWriter`] has no `Drop`, so its background tasks are only reclaimed
    /// by an explicit `close().await`. Call this before dropping a store on a
    /// path that can `await` (e.g. an LRU eviction that owns the last handle).
    /// Idempotent: a no-op when no writer is resident.
    pub async fn close(&mut self) -> LanceResult<()> {
        if let Some(writer) = self.write_writer.take() {
            writer.close().await?;
        }
        Ok(())
    }

    /// If this instance's own shard has at least `merge_after_generations`
    /// flushed generations, merge them into the base table. No-op otherwise.
    async fn maybe_merge_own_shard(&mut self) -> LanceResult<()> {
        self.merge_own_shard_if_ready(self.merge_after_generations)
            .await
            .map(|_| ())
    }

    /// Merge this instance's flushed MemWAL generations into the base table when
    /// the shard has accumulated at least `threshold` of them. Returns the
    /// number of generations reclaimed (`0` if the threshold was not met or the
    /// shard has no manifest yet).
    ///
    /// This is the shared core of both triggers: the synchronous count trigger
    /// in [`Self::add`] (with `threshold = merge_after_generations`) and the
    /// caller-driven time trigger via [`Self::cleanup_own_shard`] (with
    /// `threshold = 1`, i.e. merge whatever is pending). Both merge only
    /// the shard this instance owns and writes, so the epoch claim never fences
    /// another writer.
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

    /// Run one periodic WAL-cleanup pass over this instance's own shard: fold
    /// **every** flushed generation into the base table. This is the *time* half
    /// of the "time OR count" trigger, so it is deliberately *not* gated by any
    /// generation-count threshold: once the interval elapses, whatever is pending
    /// gets merged even if the count trigger
    /// ([`RolloutStoreOptions::merge_after_generations`]) never fired. Returns the
    /// number of generations reclaimed (`0` only when nothing was pending).
    ///
    /// Exposed for callers that drive cleanup on their own schedule (the
    /// server's process-wide sweeper and the master's fan-out `MergeWal`).
    /// Like every merge path here it operates
    /// only on the shard this instance owns, so it is safe to call concurrently
    /// with this instance's own appends but must not target another instance's
    /// shard.
    pub async fn cleanup_own_shard(&mut self) -> LanceResult<usize> {
        // Threshold `1`: merge whenever at least one generation is pending. The
        // time trigger must not depend on the count threshold — that is what
        // makes the two triggers a true OR.
        self.merge_own_shard_if_ready(1).await
    }

    /// Fold this instance's flushed MemWAL generations into the base table and
    /// drain them from the shard manifest.
    ///
    /// This is the "external compactor" path that Lance's MemWAL LSM design
    /// anticipates: each flushed generation at `_mem_wal/{shard}/{path}` is a
    /// self-contained Lance dataset. We read every generation's rows, append
    /// them to the base table (`Dataset::append`), then `commit_update` the
    /// shard manifest to remove *exactly the generations we merged* — leaving
    /// `replay_after_wal_entry_position` untouched so a reopened writer does
    /// not re-replay already-merged WAL entries.
    ///
    /// # Concurrency: surgical drain, not blanket clear
    ///
    /// The drain removes only the generation ids this call actually merged,
    /// retaining any generation that appears in the manifest afterwards. Under
    /// the single-writer-per-shard model this instance is the only writer, so in
    /// practice nothing new lands during a merge — but not depending on that is
    /// what keeps the drain correct: a blanket `flushed_generations = []` would
    /// silently discard any generation flushed between reading the manifest and
    /// committing the drain, *without merging it* (data loss). `commit_update`
    /// re-reads the latest manifest and applies this closure to it, so the
    /// retain filter runs against the current state, not the stale snapshot.
    ///
    /// # Safety of the epoch claim
    ///
    /// `claim_epoch` bumps the shard's writer epoch, which would fence any
    /// *other* live writer of this shard. That is safe here precisely because
    /// each instance merges only the shard it owns and writes: there is no
    /// other live writer of `self.write_shard` to fence. After the merge this
    /// instance's next `add` opens a fresh `mem_wal_writer`, which re-claims
    /// the (now-current) epoch.
    ///
    /// Rollout rows are immutable and de-duplicated by `id` at read time, so
    /// even if a crash interrupts the sequence (data appended to base but
    /// manifest not yet drained), a subsequent read simply sees the rows via
    /// both the base table and the still-listed generation and de-dups them —
    /// no double counting. The next merge attempt then drains the manifest.
    async fn merge_own_shard(
        &mut self,
        manifest_store: &ShardManifestStore,
        manifest: &ShardManifest,
    ) -> LanceResult<()> {
        if manifest.flushed_generations.is_empty() {
            return Ok(());
        }

        // This merge is about to `claim_epoch`, which fences any live writer of
        // this shard — including our own resident writer. Close it (draining its
        // background tasks; `ShardWriter` has no `Drop`) and clear it so the next
        // `add` transparently reopens against the freshly-claimed epoch.
        self.close().await?;

        // Resolve each flushed generation to its absolute dataset path and read
        // all its rows into memory. Record which generation ids we merge so the
        // drain can remove exactly these and nothing else.
        let base_uri = self.dataset.uri().trim_end_matches('/').to_string();
        let mut merged_generations: HashSet<u64> = HashSet::new();
        // Remember each merged generation's on-storage folder name so we can
        // delete the blob directory after the manifest drain (see below).
        let mut merged_paths: Vec<String> = Vec::new();
        let mut batches: Vec<RecordBatch> = Vec::new();
        for flushed in &manifest.flushed_generations {
            let gen_uri = format!(
                "{}/_mem_wal/{}/{}",
                base_uri, self.write_shard, flushed.path
            );
            let gen_dataset =
                Self::load_with_options(&gen_uri, self.storage_options.clone()).await?;
            let mut stream = gen_dataset.scan().try_into_stream().await?;
            while let Some(batch) = stream.try_next().await? {
                if batch.num_rows() > 0 {
                    batches.push(batch);
                }
            }
            merged_generations.insert(flushed.generation);
            merged_paths.push(flushed.path.clone());
        }

        // Append the merged rows to the base table.
        if !batches.is_empty() {
            let schema = Arc::new(rollout_schema());
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

        // Drain the merged generations from the shard manifest. Claim the
        // shard's epoch (safe: we own it) and commit a manifest that retains
        // every generation except the ones we just folded into the base table.
        // Removing only the merged ids (rather than clearing the vec) is what
        // makes this safe against a generation that lands after we read the
        // manifest: it is preserved for the next merge instead of being dropped.
        let (epoch, _) = manifest_store.claim_epoch(manifest.shard_spec_id).await?;
        manifest_store
            .commit_update(epoch, |current| ShardManifest {
                version: current.version + 1,
                flushed_generations: current
                    .flushed_generations
                    .iter()
                    .filter(|fg| !merged_generations.contains(&fg.generation))
                    .cloned()
                    .collect(),
                ..current.clone()
            })
            .await?;

        // Delete the merged generations' blob directories now that no manifest
        // references them. Ordering matters: the drain above already removed
        // these ids from `flushed_generations`, so a reader can no longer resolve
        // them — deleting the data second (never before) keeps the sequence
        // crash-safe. If the process dies between the drain and here, the rows
        // are already in the base table and the manifest no longer lists these
        // generations, so nothing reads them; they simply become storage that a
        // sweep can reclaim later.
        //
        // Best-effort: a delete failure must NOT fail the merge — the merge has
        // logically succeeded (data appended, manifest drained). A failed delete
        // only leaks one directory, which the same reclamation path handles.
        // Skipping this deletion is exactly the historical storage leak: every
        // merged generation left its `_mem_wal/{shard}/{gen}/` directory behind
        // forever.
        let object_store = self.dataset.object_store(None).await?;
        let branch_path = self.dataset.branch_location().path.clone();
        for path in &merged_paths {
            let gen_dir = branch_path
                .clone()
                .join("_mem_wal")
                .join(self.write_shard.to_string().as_str())
                .join(path.as_str());
            if let Err(err) = object_store.remove_dir_all(gen_dir.clone()).await {
                tracing::warn!(
                    shard = %self.write_shard,
                    generation_path = %path,
                    error = %err,
                    "failed to delete merged MemWAL generation directory; \
                     it will remain until reclaimed"
                );
            }
        }

        Ok(())
    }

    /// Compact the base table's small fragments into larger ones.
    ///
    /// Every WAL merge ([`Self::merge_own_shard`]) `append`s a new fragment to
    /// the base table, so a long-running store accumulates many small fragments
    /// that slow scans. This folds them together via Lance's `compact_files`,
    /// exactly like [`crate::store::ContextStore::compact`].
    ///
    /// # Distributed use: run from ONE compactor, not every worker
    ///
    /// Unlike WAL merge — where each worker touches only its own shard and can
    /// never contend — compaction rewrites the *shared* base table. Lance treats
    /// two concurrent `Rewrite` commits as a retryable conflict (one wins, the
    /// other must redo its work), so N workers each compacting the same table
    /// degenerates into a thundering herd of wasted rewrites. This method is
    /// therefore intended to be driven by a *single* external trigger (a cron
    /// job, a k8s CronJob, or one designated instance) rather than a per-worker
    /// background timer. It is safe to call while other workers are appending or
    /// WAL-merging: `Append` vs `Rewrite` is non-conflicting in Lance's matrix,
    /// so a concurrent append simply rebases on the compaction (or vice-versa).
    ///
    /// Returns the Lance [`CompactionMetrics`] (fragments/files added & removed).
    pub async fn compact(
        &mut self,
        options: Option<CompactionConfig>,
    ) -> LanceResult<CompactionMetrics> {
        let config = options.unwrap_or_default();

        let lance_options = CompactionOptions {
            target_rows_per_fragment: config.target_rows_per_fragment,
            max_rows_per_group: config.max_rows_per_group,
            materialize_deletions: config.materialize_deletions,
            materialize_deletions_threshold: config.materialize_deletions_threshold,
            num_threads: config.num_threads,
            // The rollout base table carries a MemWAL index, which is fieldless
            // (it tracks shard/generation bookkeeping, not a data column). Lance's
            // inline index remap panics on a fieldless index ("An index existed
            // with no fields"), so defer remapping: compaction records a
            // fragment-reuse index and remaps lazily instead of touching the
            // MemWAL index during the rewrite.
            defer_index_remap: true,
            ..Default::default()
        };

        match compact_files(&mut self.dataset, lance_options, None).await {
            Ok(metrics) => {
                // Reload the handle so the caller (and subsequent reads on this
                // instance) observe the compacted version.
                let uri = self.dataset.uri().to_string();
                self.dataset = Self::load_with_options(&uri, self.storage_options.clone()).await?;
                self.last_compaction = Some(Utc::now());
                self.total_compactions += 1;
                self.last_compaction_error = None;
                info!(
                    fragments_removed = metrics.fragments_removed,
                    fragments_added = metrics.fragments_added,
                    "rollout base-table compaction completed"
                );
                Ok(metrics)
            }
            Err(e) => {
                warn!(error = %e, "rollout base-table compaction failed");
                self.last_compaction_error = Some(e.to_string());
                Err(e)
            }
        }
    }

    /// Build a ZoneMap scalar index on the base table's `id` column.
    ///
    /// `id` is the rollout table's (unenforced) primary key, so a lightweight
    /// per-fragment min/max index accelerates id point-lookups and range scans
    /// on the already-flushed base table. `replace(true)` makes this idempotent:
    /// re-running simply rebuilds the index in place.
    ///
    /// # MemWAL interaction
    ///
    /// The rollout base table carries a fieldless MemWAL index, and Lance's
    /// MemWAL does not *maintain* ZoneMap indices across WAL flushes (it only
    /// keeps the indices named in `maintained_indexes`). That does not affect
    /// correctness here: rollout rows are immutable and de-duplicated by `id` at
    /// read time, so the ZoneMap only ever needs to describe the base table's
    /// already-merged fragments — rows still living in unmerged WAL generations
    /// are found by the normal full scan of those generations. Creating the
    /// index is therefore safe alongside an existing MemWAL index.
    pub async fn create_id_zonemap_index(&mut self) -> LanceResult<()> {
        info!("Creating ZoneMap index on rollout id column");
        self.dataset
            .create_index_builder(&["id"], IndexType::ZoneMap, &ScalarIndexParams::default())
            .name(ROLLOUT_ID_INDEX_NAME.to_string())
            .replace(true)
            .await?;
        // Reload the handle so subsequent reads on this instance observe the
        // new index (mirrors the reload done after `compact`).
        let uri = self.dataset.uri().to_string();
        self.dataset = Self::load_with_options(&uri, self.storage_options.clone()).await?;
        Ok(())
    }

    /// Whether the base table has accumulated at least `min_fragments`
    /// fragments (and is thus worth compacting). Quiet-hours gating from
    /// [`CompactionConfig`] is honored so an external scheduler can pass the
    /// same config it would pass to [`Self::compact`].
    pub fn should_compact(&self, config: &CompactionConfig) -> bool {
        if self.dataset.count_fragments() < config.min_fragments {
            return false;
        }
        if !config.quiet_hours.is_empty() {
            use chrono::Timelike;
            let hour = Utc::now().hour() as u8;
            for (start, end) in &config.quiet_hours {
                if hour >= *start && hour < *end {
                    return false;
                }
            }
        }
        true
    }

    /// Number of flushed MemWAL generations pending merge into the base table
    /// across all shards. `0` when no shard has a manifest yet.
    ///
    /// Read-only: unlike [`Self::cleanup_own_shard`] it never merges. Used by the
    /// control-plane stats scanner to surface read-amplification pressure.
    pub async fn pending_wal_generations(&self) -> LanceResult<usize> {
        Ok(self
            .wal_shard_snapshots()
            .await?
            .iter()
            .map(|snapshot| snapshot.flushed_generations.len())
            .sum())
    }

    /// Snapshot read-only observability metrics for the rollout store.
    ///
    /// `row_count` combines base-table fragment metadata with row counts from
    /// every flushed generation in every MemWAL shard. Generation datasets are
    /// opened concurrently and counted from their fragment metadata; rollout
    /// IDs are append-only, so summing those immutable generations preserves the
    /// store's logical row count without scanning payload columns.
    ///
    /// `fragment_count` intentionally remains the base-table fragment count:
    /// master-driven compaction only rewrites the shared base table, while each
    /// writer owns and merges its own WAL shard.
    pub async fn observe(&self) -> LanceResult<RolloutObservation> {
        let shard_snapshots = self.wal_shard_snapshots().await?;
        let base_rows = self.dataset.count_rows(None).await? as u64;
        let pending_rows = self.pending_wal_rows(&shard_snapshots).await?;
        let row_count = (base_rows + pending_rows) as i64;
        let fragment_count = self.dataset.count_fragments() as i64;
        let version = self.dataset.manifest.version;
        let last_updated = self.dataset.manifest.timestamp().timestamp_millis();
        let pending_wal_generations = shard_snapshots
            .iter()
            .map(|snapshot| snapshot.flushed_generations.len() as i64)
            .sum();
        Ok(RolloutObservation {
            row_count,
            fragment_count,
            version,
            last_updated,
            pending_wal_generations,
        })
    }

    /// Current compaction statistics for the base table.
    ///
    /// `is_compacting` is always `false`: compaction here runs synchronously
    /// under the caller's `&mut self` (there is no background compactor thread),
    /// so a stats read cannot observe an in-flight compaction on this handle.
    #[must_use]
    pub fn compaction_stats(&self) -> CompactionStats {
        CompactionStats {
            total_fragments: self.dataset.count_fragments(),
            is_compacting: false,
            last_compaction: self.last_compaction,
            last_error: self.last_compaction_error.clone(),
            total_compactions: self.total_compactions,
        }
    }

    /// Initialize the (unsharded) MemWAL index on first write, exactly once.
    /// Subsequent writes see the index already present and skip this. The shard
    /// a write targets is chosen by the writer (`shard_id`), independent of the
    /// index's declared sharding strategy.
    ///
    /// # Concurrent first-writers
    ///
    /// `initialize_mem_wal` commits a `CreateIndex` transaction, and Lance treats
    /// two concurrent `CreateIndex` commits as a hard conflict (not an
    /// auto-retried one): when two instances take their very first write at the
    /// same time, both observe no index, both try to create it, and the loser
    /// gets `RetryableCommitConflict`. That is benign here — the winner created
    /// exactly the index we wanted — so we reload and treat "index now present"
    /// as success rather than surfacing the conflict to the caller. Any other
    /// error propagates.
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
            Err(err) => {
                // A concurrent first-writer may have created the index between
                // our check and our commit. Reload and accept it if so.
                let uri = self.dataset.uri().to_string();
                self.dataset = Self::load_with_options(&uri, self.storage_options.clone()).await?;
                if self.mem_wal_index_present().await? {
                    Ok(())
                } else {
                    Err(err)
                }
            }
        }
    }

    /// Whether the MemWAL index has been initialized on the current dataset
    /// handle.
    async fn mem_wal_index_present(&self) -> LanceResult<bool> {
        let indices = self.dataset.load_indices().await?;
        Ok(indices.iter().any(|i| i.name == MEM_WAL_INDEX_NAME))
    }

    /// List rollout rows (base table ∪ every instance's flushed MemWAL rows).
    ///
    /// `binary_payload` is projected out so artifact bytes are never
    /// materialized on a list scan (spec §2); fetch them on demand via
    /// [`Self::get_blob`]. Ordering is not guaranteed to match append order
    /// because the LSM read path dedups by `id` across generations.
    pub async fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> LanceResult<Vec<RolloutRecord>> {
        self.list_with_filters(limit, offset, None).await
    }

    /// List rollout rows matching exact field filters.
    ///
    /// The filter is pushed into every LSM source before merge and
    /// deduplication. Rollout rows are immutable by id, so filtering each
    /// generation cannot reveal an older state of a mutable record.
    pub async fn list_with_filters(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<&RolloutFilters>,
    ) -> LanceResult<Vec<RolloutRecord>> {
        let columns = self.non_blob_columns();
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let mut scanner = self.lsm_scanner().await?.project(&refs);
        if let Some(predicate) = filters.and_then(RolloutFilters::expression) {
            scanner = scanner.filter(&predicate)?;
        }
        let post_scan_offset = if limit.is_none() { offset } else { None };
        if let Some(limit) = limit {
            scanner = scanner.limit(limit, offset);
        }

        let mut stream = scanner.try_into_stream().await?;
        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            results.extend(batch_to_rollout_records(&batch)?);
        }

        if let Some(offset) = post_scan_offset {
            results = results.into_iter().skip(offset).collect();
        }
        Ok(results)
    }

    /// Return one complete trajectory in deterministic message order.
    pub async fn get_trajectory(&self, rollout_id: &str) -> LanceResult<Vec<RolloutRecord>> {
        if rollout_id.is_empty() {
            return Err(LanceError::from(ArrowError::InvalidArgumentError(
                "rollout_id must not be empty".to_string(),
            )));
        }

        let filters = RolloutFilters {
            rollout_id: Some(rollout_id.to_string()),
            ..Default::default()
        };
        let mut records = self.list_with_filters(None, None, Some(&filters)).await?;
        records.sort_by(|left, right| {
            left.sequence_order
                .cmp(&right.sequence_order)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(records)
    }

    /// Filter and page rollout rows over the full base ∪ WAL union.
    ///
    /// Thin wrapper over [`Self::list_filtered_source`] with [`ListSource::All`],
    /// preserving the historical union semantics for existing callers.
    pub async fn list_filtered(
        &self,
        filters: &RolloutFilters,
        limit: usize,
        offset: usize,
    ) -> LanceResult<RolloutPage> {
        self.list_filtered_source(filters, limit, offset, ListSource::All)
            .await
    }

    /// Filter and page rollout rows from a chosen [`ListSource`].
    ///
    /// Reads one row beyond the requested page to report `has_more`, avoiding
    /// an unbounded full-table count on every UI request. Pagination is
    /// deliberately late-materialized in two scans:
    ///
    /// 1. scan, sort, and deduplicate only `id` to select the page;
    /// 2. fetch the complete non-blob columns for those page ids in one query.
    ///
    /// [`LsmScanner`] sorts every source by primary key before applying its
    /// global limit. Keeping wide token/logprob/metadata columns out of that
    /// full-source sort makes browsing large rollout tables substantially
    /// cheaper while preserving the same LSM deduplication semantics.
    ///
    /// [`ListSource::Fragments`] skips MemWAL manifest discovery entirely, so its
    /// latency is independent of how far the merge backlog has grown.
    pub async fn list_filtered_source(
        &self,
        filters: &RolloutFilters,
        limit: usize,
        offset: usize,
        source: ListSource,
    ) -> LanceResult<RolloutPage> {
        // Fragments never touches the WAL, so skip the per-shard manifest reads.
        let shard_snapshots = match source {
            ListSource::Fragments => Vec::new(),
            ListSource::Wal | ListSource::All => self.wal_shard_snapshots().await?,
        };
        let filter = filters.expression();

        let mut page_scanner = self
            .lsm_scanner_for_source(source, shard_snapshots.clone())
            .project(&["id"]);
        if let Some(filter) = &filter {
            page_scanner = page_scanner.filter(filter)?;
        }
        page_scanner = page_scanner.limit(limit.saturating_add(1), Some(offset));

        let mut stream = page_scanner.try_into_stream().await?;
        let mut page_ids = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let ids = column_as::<StringArray>(&batch, "id")?;
            page_ids.extend((0..batch.num_rows()).map(|row| ids.value(row).to_string()));
        }
        let has_more = page_ids.len() > limit;
        page_ids.truncate(limit);
        if page_ids.is_empty() {
            return Ok(RolloutPage {
                records: Vec::new(),
                has_more,
            });
        }

        let columns = self.non_blob_columns();
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let id_refs: Vec<&str> = page_ids.iter().map(String::as_str).collect();
        let id_filter = format!("id IN ({})", sql_quoted_list(&id_refs));
        let record_scanner = self
            .lsm_scanner_for_source(source, shard_snapshots)
            .project(&refs)
            .filter(&id_filter)?;

        let mut stream = record_scanner.try_into_stream().await?;
        let mut records_by_id = HashMap::with_capacity(page_ids.len());
        while let Some(batch) = stream.try_next().await? {
            for record in batch_to_rollout_records(&batch)? {
                records_by_id.insert(record.id.clone(), record);
            }
        }
        let records = page_ids
            .into_iter()
            .filter_map(|id| records_by_id.remove(&id))
            .collect();
        Ok(RolloutPage { records, has_more })
    }

    /// Retrieve a single rollout row by its unique id, including any freshly
    /// appended (MemWAL-flushed) row on any instance. `binary_payload` is
    /// projected out (fetch bytes via [`Self::get_blob`]).
    ///
    /// # Base-table-first
    ///
    /// This queries the immutable base table first and returns immediately on a
    /// hit, opening **zero** MemWAL generations for the common case of an
    /// already-merged record. Only a base miss falls back to the flushed-WAL
    /// union, so latency is decoupled from the pending-generation backlog
    /// (which can run into the hundreds/thousands on a high-write experiment)
    /// instead of paying one object-store open per pending generation.
    ///
    /// This is a pure optimization, not a semantic change: rollout rows are
    /// immutable and an `id` is never re-appended, so a row present in the base
    /// table is identical to any (necessarily absent) WAL copy, and a row that
    /// is *only* in the WAL is still found by the fallback. The fetchability
    /// invariant holds — any row visible in a `wal`/`all` list is returned here.
    pub async fn get_by_id(&self, id: &str) -> LanceResult<Option<RolloutRecord>> {
        self.get_by_id_source(id, ListSource::All).await
    }

    /// [`Self::get_by_id`] with an explicit [`ListSource`]:
    /// - `Fragments`: base table only — never opens a WAL generation (fastest,
    ///   but misses un-merged rows);
    /// - `Wal`: flushed MemWAL generations only (excludes the base table);
    /// - `All`: base-table-first with a WAL fallback — the fetch-anything
    ///   default, fast whenever the row is already merged.
    pub async fn get_by_id_source(
        &self,
        id: &str,
        source: ListSource,
    ) -> LanceResult<Option<RolloutRecord>> {
        match source {
            ListSource::Fragments => self.scan_one_by_id(id, ListSource::Fragments).await,
            ListSource::Wal => self.scan_one_by_id(id, ListSource::Wal).await,
            ListSource::All => {
                // Base-table-first: try the immutable base with no manifest
                // reads, then fall back to the flushed-WAL union only on a miss.
                if let Some(record) = self.scan_one_by_id(id, ListSource::Fragments).await? {
                    return Ok(Some(record));
                }
                self.scan_one_by_id(id, ListSource::Wal).await
            }
        }
    }

    /// Run an id-equality point scan against a single [`ListSource`] and return
    /// the first matching record. `Fragments` passes an empty snapshot set so it
    /// performs no MemWAL manifest discovery; `Wal`/`All` discover flushed
    /// generations first.
    async fn scan_one_by_id(
        &self,
        id: &str,
        source: ListSource,
    ) -> LanceResult<Option<RolloutRecord>> {
        let shard_snapshots = match source {
            ListSource::Fragments => Vec::new(),
            ListSource::Wal | ListSource::All => self.wal_shard_snapshots().await?,
        };
        let escaped_id = id.replace('\'', "''");
        let columns = self.non_blob_columns();
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let scanner = self
            .lsm_scanner_for_source(source, shard_snapshots)
            .project(&refs)
            .filter(&format!("id = '{}'", escaped_id))?;
        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            for record in batch_to_rollout_records(&batch)? {
                if record.id == id {
                    return Ok(Some(record));
                }
            }
        }
        Ok(None)
    }

    /// Fetch a single artifact row's `binary_payload` bytes on demand.
    ///
    /// `list`/`get_by_id` project `binary_payload` out, so it reads back as
    /// `None` there. This method first locates the row using only `id`, then
    /// takes `binary_payload` for that exact row id. Keeping the large inline
    /// column out of the filtered scan prevents a point lookup from
    /// materializing an entire fragment's payloads. Returns `None` if the row
    /// or its payload is absent.
    ///
    /// # Base-table-first, then bounded-parallel WAL fallback
    ///
    /// The immutable base table is queried first and, on a hit, returned
    /// immediately without opening any MemWAL generation — the common
    /// already-merged case pays zero per-generation opens. Only a base miss
    /// falls back to the flushed generations, which are then opened
    /// **concurrently** (bounded by [`DEFAULT_OBSERVE_CONCURRENCY`]) with the
    /// first match winning, rather than one-at-a-time newest-first.
    ///
    /// Because rollout rows are immutable and an `id` is never re-appended, a
    /// row lives in exactly one place (the base table *or* one generation, never
    /// both), so base-first cannot shadow a newer WAL copy and the parallel
    /// fallback needs no ordering.
    ///
    /// The fallback is tolerant of a generation that a concurrent merge drained
    /// and deleted between snapshot and open: such an open fails with a
    /// not-found error, which is skipped (the row's data is already covered by
    /// the base table we checked first). This removes the transient 500s that
    /// the previous fail-fast fallback surfaced under concurrent auto-merge.
    pub async fn get_blob(&self, id: &str) -> LanceResult<Option<Vec<u8>>> {
        // Base-table-first: an already-merged row is found here with no MemWAL
        // manifest reads and no per-generation opens.
        if let Some(payload) = Self::get_blob_from_dataset(&self.dataset, id).await? {
            return Ok(payload);
        }

        // Base miss: fall back to the flushed generations. Open them with
        // bounded concurrency and take the first hit; tolerate a generation that
        // was concurrently merged away (NotFound) instead of failing the request.
        let snapshots = self.wal_shard_snapshots().await?;
        let uris: Vec<String> = snapshots
            .iter()
            .flat_map(|snapshot| {
                snapshot.flushed_generations.iter().map(|generation| {
                    self.flushed_generation_uri(snapshot.shard_id, &generation.path)
                })
            })
            .collect();

        let mut hits = stream::iter(uris)
            .map(|uri| async move {
                match self.open_flushed_dataset(&uri).await {
                    Ok(dataset) => Self::get_blob_from_dataset(&dataset, id).await,
                    // A generation drained + deleted by a concurrent merge
                    // between snapshot and open: its rows are already in the base
                    // table (checked above), so skip it rather than 500.
                    Err(err) if is_not_found_error(&err) => Ok(None),
                    Err(err) => Err(err),
                }
            })
            .buffer_unordered(DEFAULT_OBSERVE_CONCURRENCY);

        while let Some(result) = hits.next().await {
            if let Some(payload) = result? {
                // Outer Some = row found in this generation; inner is
                // payload-or-null. Return the inner value directly.
                return Ok(payload);
            }
        }

        Ok(None)
    }

    /// Locate `id` without projecting payload bytes, then take only the
    /// matching row's payload. The outer option distinguishes a missing row
    /// from a present row whose payload is null, which matters when a newer WAL
    /// row shadows a base-table row.
    async fn get_blob_from_dataset(
        dataset: &Dataset,
        id: &str,
    ) -> LanceResult<Option<Option<Vec<u8>>>> {
        let escaped_id = id.replace('\'', "''");
        let mut scanner = dataset.scan();
        scanner
            .project(&["id"])?
            .filter(&format!("id = '{}'", escaped_id))?
            .with_row_id()
            .limit(Some(1), None)?;

        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            let id_array = column_as::<StringArray>(&batch, "id")?;
            let row_id_array = column_as::<UInt64Array>(&batch, "_rowid")?;
            for row in 0..batch.num_rows() {
                if id_array.value(row) != id {
                    continue;
                }

                let projection = dataset.schema().project(&["binary_payload"])?;
                let payload_batch = dataset
                    .take_rows(&[row_id_array.value(row)], projection)
                    .await?;
                let binary_array =
                    column_as_optional::<LargeBinaryArray>(&payload_batch, "binary_payload");
                return Ok(Some(match binary_array {
                    Some(arr) if !arr.is_null(0) => Some(arr.value(0).to_vec()),
                    _ => None,
                }));
            }
        }
        Ok(None)
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

    /// Top-level column names excluding `binary_payload`, so list-style scans
    /// never materialize artifact bytes.
    fn non_blob_columns(&self) -> Vec<String> {
        self.dataset
            .schema()
            .fields
            .iter()
            .map(|field| field.name.clone())
            .filter(|name| name != "binary_payload")
            .collect()
    }

    /// Build an LSM scanner over the base table unioned with every shard's
    /// flushed MemWAL generations, discovered from object storage. Because the
    /// snapshot is rebuilt from shard manifests on each call, one instance sees
    /// every other instance's flushed appends — reads are not pinned to the
    /// writing instance. Deduplicates by `id`.
    async fn lsm_scanner(&self) -> LanceResult<LsmScanner> {
        let shard_snapshots = self.wal_shard_snapshots().await?;

        Ok(self.lsm_scanner_with_snapshots(shard_snapshots))
    }

    fn lsm_scanner_with_snapshots(&self, shard_snapshots: Vec<ShardSnapshot>) -> LsmScanner {
        LsmScanner::new(
            Arc::new(self.dataset.clone()),
            shard_snapshots,
            vec!["id".to_string()],
        )
    }

    /// Build a paginating scanner for the requested [`ListSource`], deduplicating
    /// by `id`:
    /// - `Fragments`: base table only (`shard_snapshots` is ignored — callers
    ///   pass an empty vec so no manifest reads happen);
    /// - `All`: base table ∪ the flushed generations in `shard_snapshots`;
    /// - `Wal`: only the flushed generations, via
    ///   [`LsmScanner::without_base_table`], resolving relative generation paths
    ///   against the dataset root (matching [`Self::flushed_generation_uri`]).
    fn lsm_scanner_for_source(
        &self,
        source: ListSource,
        shard_snapshots: Vec<ShardSnapshot>,
    ) -> LsmScanner {
        match source {
            ListSource::Fragments => LsmScanner::new(
                Arc::new(self.dataset.clone()),
                Vec::new(),
                vec!["id".to_string()],
            ),
            ListSource::All => LsmScanner::new(
                Arc::new(self.dataset.clone()),
                shard_snapshots,
                vec!["id".to_string()],
            ),
            ListSource::Wal => {
                let arrow_schema: Schema = self.dataset.schema().into();
                LsmScanner::without_base_table(
                    Arc::new(arrow_schema),
                    self.dataset.uri().trim_end_matches('/').to_string(),
                    shard_snapshots,
                    vec!["id".to_string()],
                )
                .with_session(self.dataset.session())
            }
        }
    }

    /// Read the latest manifest for every MemWAL shard. Manifest reads are
    /// bounded-concurrent so stores with many writer instances do not pay one
    /// object-store round trip per shard serially.
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
            .buffer_unordered(DEFAULT_OBSERVE_CONCURRENCY)
            .try_collect()
            .await?;

        Ok(snapshots.into_iter().flatten().collect())
    }

    /// Count rows in all immutable flushed-generation datasets using metadata
    /// reads rather than a payload scan.
    async fn pending_wal_rows(&self, snapshots: &[ShardSnapshot]) -> LanceResult<u64> {
        let generation_paths: Vec<String> = snapshots
            .iter()
            .flat_map(|snapshot| {
                snapshot.flushed_generations.iter().map(|generation| {
                    self.flushed_generation_uri(snapshot.shard_id, &generation.path)
                })
            })
            .collect();
        let session = self.dataset.session();
        let storage_options = self.storage_options.clone();

        stream::iter(generation_paths)
            .map(|path| {
                let session = session.clone();
                let storage_options = storage_options.clone();
                async move {
                    let mut builder = DatasetBuilder::from_uri(&path).with_session(session);
                    if let Some(options) = storage_options {
                        builder = builder.with_storage_options(options);
                    }
                    let dataset = builder.load().await?;
                    Ok::<_, LanceError>(dataset.count_rows(None).await? as u64)
                }
            })
            .buffer_unordered(DEFAULT_OBSERVE_CONCURRENCY)
            .try_fold(0_u64, |total, rows| async move { Ok(total + rows) })
            .await
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
        let schema = Arc::new(rollout_schema());
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
        let mut model_input_string_builder = LargeStringBuilder::new();
        let mut model_output_string_builder = LargeStringBuilder::new();
        let mut rationale_builder = LargeStringBuilder::new();
        let mut problem_text_builder = LargeStringBuilder::new();
        let mut user_metadata_builder = LargeStringBuilder::new();
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
        let mut artifact_type_builder = StringBuilder::new();
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
            model_input_string_builder.append_option(record.model_input_string.as_deref());
            model_output_string_builder.append_option(record.model_output_string.as_deref());
            rationale_builder.append_option(record.rationale.as_deref());
            problem_text_builder.append_option(record.problem_text.as_deref());
            user_metadata_builder.append_option(record.user_metadata.as_deref());
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
            artifact_type_builder.append_option(record.artifact_type.as_deref());
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
            "model_input_string".to_string(),
            Arc::new(model_input_string_builder.finish()),
        );
        arrays_by_name.insert(
            "model_output_string".to_string(),
            Arc::new(model_output_string_builder.finish()),
        );
        arrays_by_name.insert(
            "rationale".to_string(),
            Arc::new(rationale_builder.finish()),
        );
        arrays_by_name.insert(
            "problem_text".to_string(),
            Arc::new(problem_text_builder.finish()),
        );
        arrays_by_name.insert(
            "user_metadata".to_string(),
            Arc::new(user_metadata_builder.finish()),
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
        arrays_by_name.insert(
            "artifact_type".to_string(),
            Arc::new(artifact_type_builder.finish()),
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

impl Drop for RolloutStore {
    /// Best-effort drain of a still-resident writer's background tasks.
    ///
    /// [`ShardWriter`] has no `Drop`, so dropping it without `close().await`
    /// leaks its background tasks. The graceful path is [`Self::close`], but a
    /// store can also be dropped without an `await` (e.g. LRU eviction). When a
    /// Tokio runtime is available we move the writer into a detached task that
    /// closes it; otherwise (no runtime, e.g. some teardown paths) we can only
    /// drop it. Callers that can `await` should prefer [`Self::close`].
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

/// Whether a Lance error is a MemWAL writer-fence error — i.e. this writer's
/// shard epoch was superseded by a later claimer. Matched on the error text
/// because Lance surfaces the fence as a generic error rather than a dedicated
/// variant (see `ShardWriter::check_fenced`, which reports "Writer fenced").
fn is_fenced_error(err: &LanceError) -> bool {
    let text = err.to_string();
    text.contains("fenced") || text.contains("Fenced")
}

/// Whether a Lance error means the target dataset/path does not exist. Used by
/// the [`RolloutStore::get_blob`] WAL fallback to skip a flushed generation that
/// a concurrent merge drained from the shard manifest and deleted between the
/// manifest snapshot and the open — the row's data is already in the base table,
/// so skipping is safe and avoids a transient failure. Both the structured
/// not-found variants and the object-store "not found" surface are matched; the
/// latter is text-based because it arrives as a generic wrapped IO error.
fn is_not_found_error(err: &LanceError) -> bool {
    if matches!(
        err,
        LanceError::DatasetNotFound { .. } | LanceError::NotFound { .. }
    ) {
        return true;
    }
    let text = err.to_string();
    text.contains("was not found") || text.contains("not found") || text.contains("NotFound")
}

/// Derive the MemWAL shard UUID a server instance writes to from its stable
/// instance id. Deterministic (UUID v5), so the same instance id always maps to
/// the same shard across restarts and reopens — the losing/gaining semantics of
/// StatefulSet rescheduling therefore keep writing the same physical shard.
/// `None` maps to a single fixed `"default"` shard for single-instance use.
#[must_use]
pub fn derive_shard_id(instance_id: Option<&str>) -> Uuid {
    let input = instance_id.unwrap_or("default");
    Uuid::new_v5(&Uuid::NAMESPACE_OID, input.as_bytes())
}

/// Arrow schema for a rollout dataset (spec §5). `binary_payload` is a plain
/// inline `LargeBinary` column: rollout reads go through the MemWAL LSM scanner,
/// which has no blob-materialization step, so a blob-v2 offloaded column would
/// read back as `None`. List-style scans project the column out instead (see
/// [`RolloutStore::list`]).
#[must_use]
pub fn rollout_schema() -> Schema {
    let mut id_metadata = HashMap::new();
    id_metadata.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );

    let binary_field = Field::new("binary_payload", DataType::LargeBinary, true);

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
        // Claim-check offloaded message fields.
        Field::new("model_input_string", DataType::LargeUtf8, true),
        Field::new("model_output_string", DataType::LargeUtf8, true),
        Field::new("rationale", DataType::LargeUtf8, true),
        Field::new("problem_text", DataType::LargeUtf8, true),
        Field::new("user_metadata", DataType::LargeUtf8, true),
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
        Field::new("artifact_type", DataType::Utf8, true),
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
    let model_input_string_array =
        column_as_optional::<LargeStringArray>(batch, "model_input_string");
    let model_output_string_array =
        column_as_optional::<LargeStringArray>(batch, "model_output_string");
    let rationale_array = column_as_optional::<LargeStringArray>(batch, "rationale");
    let problem_text_array = column_as_optional::<LargeStringArray>(batch, "problem_text");
    let user_metadata_array = column_as_optional::<LargeStringArray>(batch, "user_metadata");
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
    let artifact_type_array = column_as_optional::<StringArray>(batch, "artifact_type");
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
            model_input_string: optional_large_string(model_input_string_array, row),
            model_output_string: optional_large_string(model_output_string_array, row),
            rationale: optional_large_string(rationale_array, row),
            problem_text: optional_large_string(problem_text_array, row),
            user_metadata: optional_large_string(user_metadata_array, row),
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
            artifact_type: optional_string(artifact_type_array, row),
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
    use crate::record::Relationship;
    use crate::rollout::{ROLE_ARTIFACT, ROLE_ASSISTANT};
    use chrono::{TimeZone, Utc};
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn rollout_filters_parse_supported_fields() {
        let filters = RolloutFilters::from_json_value(json!({
            "rollout_id": "traj-1",
            "problem_id": "problem-7",
            "policy_version": "ckpt-42",
            "role": "assistant",
            "include_in_training": false,
            "artifact_type": "screenshot"
        }))
        .unwrap();

        assert_eq!(filters.rollout_id.as_deref(), Some("traj-1"));
        assert_eq!(filters.problem_id.as_deref(), Some("problem-7"));
        assert_eq!(filters.policy_version.as_deref(), Some("ckpt-42"));
        assert_eq!(filters.role.as_deref(), Some("assistant"));
        assert_eq!(filters.include_in_training, Some(false));
        assert_eq!(filters.artifact_type.as_deref(), Some("screenshot"));
    }

    #[test]
    fn rollout_filters_reject_unknown_and_wrong_types() {
        assert!(RolloutFilters::from_json_value(json!({"reward": 1.0})).is_err());
        assert!(RolloutFilters::from_json_value(json!({"policy_version": 42})).is_err());
        assert!(RolloutFilters::from_json_value(json!({"include_in_training": "yes"})).is_err());
        assert!(RolloutFilters::from_json_value(json!([])).is_err());
    }

    #[test]
    fn rollout_filter_expression_escapes_strings_and_fields() {
        let filters = RolloutFilters {
            policy_version: Some("worker's-ckpt".to_string()),
            role: Some("assistant".to_string()),
            include_in_training: Some(true),
            ..Default::default()
        };

        assert_eq!(
            filters.expression().as_deref(),
            Some(
                "role = 'assistant' AND policy_version = 'worker''s-ckpt' AND include_in_training = true"
            )
        );
    }

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
            model_input_string: None,
            model_output_string: None,
            rationale: None,
            problem_text: None,
            user_metadata: None,
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
            artifact_type: None,
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
            model_input_string: None,
            model_output_string: None,
            rationale: None,
            problem_text: None,
            user_metadata: None,
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
            artifact_type: Some("excel_grade_screenshot".to_string()),
            metadata: Some(json!({"filename": "trace.bin"})),
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
        // `binary_payload` is projected out of `list`/`get_by_id` scans, so
        // they read it back as `None` (byte materialization is verified
        // separately through `get_blob`). The inline sidecar columns still
        // round-trip.
        assert_eq!(actual.payload_size, expected.payload_size);
        assert_eq!(actual.payload_checksum, expected.payload_checksum);
        assert_eq!(actual.artifact_type, expected.artifact_type);
        assert_eq!(actual.metadata, expected.metadata);
    }

    #[test]
    fn append_list_and_fetch_roundtrip() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let assistant = assistant_record("row-0");
        let artifact_bytes = b"\x00\x01\x02trace-bytes";
        let artifact = artifact_record("row-1", artifact_bytes);
        let quoted_artifact_bytes = b"quoted-id-bytes";
        let quoted_artifact = artifact_record("row-'quoted'", quoted_artifact_bytes);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            // MemWAL appends land in the `_mem_wal` namespace and do not advance
            // the base dataset version; `add` returns it unchanged.
            store
                .add(&[assistant.clone(), artifact.clone(), quoted_artifact.clone()])
                .await
                .unwrap();

            // The LSM read path dedups by `id` across generations and does not
            // guarantee append order, so look rows up by id rather than by
            // position.
            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 3);
            let listed_assistant = listed.iter().find(|r| r.id == "row-0").unwrap();
            let listed_artifact = listed.iter().find(|r| r.id == "row-1").unwrap();
            assert_records_eq(listed_assistant, &assistant);
            assert_records_eq(listed_artifact, &artifact);
            // Offloaded blob column is projected out of a list scan.
            assert_eq!(listed_artifact.binary_payload, None);

            let fetched = store.get_by_id("row-1").await.unwrap().unwrap();
            assert_records_eq(&fetched, &artifact);
            assert!(fetched.is_artifact());

            // The bytes are recoverable on demand from the sidecar file.
            let blob = store.get_blob("row-1").await.unwrap();
            assert_eq!(blob.as_deref(), Some(&artifact_bytes[..]));
            let quoted_blob = store.get_blob("row-'quoted'").await.unwrap();
            assert_eq!(quoted_blob.as_deref(), Some(&quoted_artifact_bytes[..]));
            // The assistant row carries no payload.
            assert_eq!(store.get_blob("row-0").await.unwrap(), None);

            assert!(store.get_by_id("missing").await.unwrap().is_none());
            assert_eq!(store.get_blob("missing").await.unwrap(), None);
        });
    }

    #[test]
    fn observe_reports_cheap_metrics() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            // Empty store: no rows, no pending generations.
            let obs = store.observe().await.unwrap();
            assert_eq!(obs.row_count, 0);
            assert_eq!(obs.pending_wal_generations, 0);

            store
                .add(&[assistant_record("a"), assistant_record("b")])
                .await
                .unwrap();

            // Two MemWAL-appended rows are visible via the LSM read path; the
            // base table itself has not been merged, so they surface as a
            // pending flushed generation rather than base rows.
            let obs = store.observe().await.unwrap();
            assert!(obs.fragment_count >= 0);
            assert!(obs.pending_wal_generations >= 1);
            assert_eq!(obs.row_count, 2);
            assert!(obs.last_updated > 0);
        });
    }

    #[test]
    fn filtered_list_matches_fields_before_pagination() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open(&uri).await.unwrap();

            let mut first = assistant_record("row-a");
            first.rollout_id = "traj-a".to_string();
            first.problem_id = "problem-a".to_string();
            first.policy_version = Some("ckpt-1".to_string());
            first.include_in_training = Some(true);

            let mut second = assistant_record("row-b");
            second.rollout_id = "traj-b".to_string();
            second.problem_id = "problem-b".to_string();
            second.policy_version = Some("ckpt-2".to_string());
            second.include_in_training = Some(false);

            let mut artifact = artifact_record("row-c", b"bytes");
            artifact.rollout_id = "traj-b".to_string();
            artifact.problem_id = "problem-b".to_string();
            artifact.policy_version = Some("ckpt-2".to_string());
            artifact.include_in_training = Some(false);

            store.add(&[first, second, artifact]).await.unwrap();

            let filters = RolloutFilters {
                policy_version: Some("ckpt-2".to_string()),
                include_in_training: Some(false),
                ..Default::default()
            };
            let all = store
                .list_with_filters(None, None, Some(&filters))
                .await
                .unwrap();
            assert_eq!(all.len(), 2);
            assert!(all
                .iter()
                .all(|row| row.policy_version.as_deref() == Some("ckpt-2")));
            assert!(all.iter().all(|row| row.include_in_training == Some(false)));

            let page = store
                .list_with_filters(Some(1), Some(1), Some(&filters))
                .await
                .unwrap();
            assert_eq!(page.len(), 1);
            assert_eq!(page[0].policy_version.as_deref(), Some("ckpt-2"));
        });
    }

    #[test]
    fn filtered_list_supports_dictionary_and_nullable_string_columns() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            store
                .add(&[
                    assistant_record("assistant"),
                    artifact_record("artifact", b"bytes"),
                ])
                .await
                .unwrap();

            let filters = RolloutFilters {
                role: Some(ROLE_ARTIFACT.to_string()),
                artifact_type: Some("excel_grade_screenshot".to_string()),
                ..Default::default()
            };
            let rows = store
                .list_with_filters(None, None, Some(&filters))
                .await
                .unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].id, "artifact");
            assert!(rows[0].binary_payload.is_none());
        });
    }

    #[test]
    fn appends_accumulate_and_are_immutable() {
        // MemWAL appends are append-only: successive `add`s accumulate, and a
        // row is never mutated. Reproducing "the rollouts that trained
        // checkpoint N" is therefore a filter over immutable rows (here, by id),
        // not a per-append table snapshot — so this replaces the old
        // `checkout_recovers_earlier_version` test, whose per-add snapshot
        // semantics MemWAL intentionally drops.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();

        let artifact_bytes = b"\x00\x01\x02checkpoint-trace";
        let artifact = artifact_record("row-0", artifact_bytes);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            store.add(std::slice::from_ref(&artifact)).await.unwrap();

            // A second append accumulates rather than replacing the first.
            store.add(&[assistant_record("row-1")]).await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 2);

            // The first-appended row is still present and unchanged after the
            // later append (immutability), and its inline artifact bytes remain
            // materializable.
            let recovered = store.get_by_id("row-0").await.unwrap().unwrap();
            assert_records_eq(&recovered, &artifact);
            let blob = store.get_blob("row-0").await.unwrap();
            assert_eq!(blob.as_deref(), Some(&artifact_bytes[..]));
        });
    }

    #[test]
    fn distinct_shards_share_one_dataset() {
        // Two instances writing distinct shards of the same dataset both
        // contribute to reads, and neither fences the other. This models the
        // server-id sharding deployment: a load balancer may route appends to
        // either instance, and every reader sees the union.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let shard_a_blob = b"shard-a-blob";
            let shard_b_blob = b"shard-b-blob";
            let options = |shard: &str| RolloutStoreOptions {
                storage_options: None,
                shard_id: Some(shard.to_string()),
                merge_after_generations: None,
            };

            let mut instance_a = RolloutStore::open_with_options(&uri, options("rollout-0"))
                .await
                .unwrap();
            instance_a
                .add(&[artifact_record("a-0", shard_a_blob)])
                .await
                .unwrap();

            let mut instance_b = RolloutStore::open_with_options(&uri, options("rollout-1"))
                .await
                .unwrap();
            instance_b
                .add(&[artifact_record("b-0", shard_b_blob)])
                .await
                .unwrap();

            // Distinct instance ids derive distinct shards.
            assert_ne!(
                derive_shard_id(Some("rollout-0")),
                derive_shard_id(Some("rollout-1"))
            );

            // Either instance's reader sees both shards' rows.
            let seen = instance_b.list(None, None).await.unwrap();
            assert_eq!(seen.len(), 2);
            assert!(seen.iter().any(|r| r.id == "a-0"));
            assert!(seen.iter().any(|r| r.id == "b-0"));
            assert_eq!(
                instance_b.get_blob("a-0").await.unwrap().as_deref(),
                Some(&shard_a_blob[..])
            );
            assert_eq!(
                instance_a.get_blob("b-0").await.unwrap().as_deref(),
                Some(&shard_b_blob[..])
            );

            // Observability is independent of the reader's own write shard:
            // both shards contribute rows and pending generations.
            let reader = RolloutStore::open(&uri).await.unwrap();
            let obs = reader.observe().await.unwrap();
            assert_eq!(obs.row_count, 2);
            assert_eq!(obs.pending_wal_generations, 2);
            assert_eq!(obs.fragment_count, 0);
        });
    }

    #[test]
    fn filtered_listing_pages_across_shards_and_escapes_values() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let options = |shard: &str| RolloutStoreOptions {
                shard_id: Some(shard.to_string()),
                ..Default::default()
            };
            let mut instance_a = RolloutStore::open_with_options(&uri, options("filter-a"))
                .await
                .unwrap();
            let mut quoted = assistant_record("row-'quoted");
            quoted.rollout_id = "rollout-alpha".to_string();
            quoted.policy_version = Some("policy-a".to_string());
            instance_a
                .add(&[quoted, assistant_record("row-a")])
                .await
                .unwrap();

            let mut instance_b = RolloutStore::open_with_options(&uri, options("filter-b"))
                .await
                .unwrap();
            let artifact = artifact_record("row-b", b"blob");
            instance_b.add(&[artifact]).await.unwrap();

            let reader = RolloutStore::open(&uri).await.unwrap();
            let quoted_page = reader
                .list_filtered(
                    &RolloutFilters {
                        id: Some("row-'quoted".to_string()),
                        ..Default::default()
                    },
                    25,
                    0,
                )
                .await
                .unwrap();
            assert!(!quoted_page.has_more);
            assert_eq!(quoted_page.records[0].id, "row-'quoted");
            assert_eq!(quoted_page.records[0].input_tokens, Some(vec![10, 11, 12]));
            assert_eq!(
                quoted_page.records[0].output_logprobs,
                Some(vec![-0.5, -1.25])
            );
            assert_eq!(
                quoted_page.records[0].metadata,
                Some(json!({"harness": "verifiers"}))
            );

            let policy_page = reader
                .list_filtered(
                    &RolloutFilters {
                        rollout_id: Some("rollout-alpha".to_string()),
                        role: Some(ROLE_ASSISTANT.to_string()),
                        policy_version: Some("policy-a".to_string()),
                        ..Default::default()
                    },
                    25,
                    0,
                )
                .await
                .unwrap();
            assert!(!policy_page.has_more);
            assert_eq!(policy_page.records[0].id, "row-'quoted");

            let page = reader
                .list_filtered(&RolloutFilters::default(), 1, 1)
                .await
                .unwrap();
            assert!(page.has_more);
            assert_eq!(page.records.len(), 1);
        });
    }

    #[test]
    fn refresh_latest_keeps_cached_reader_current() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut writer = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    shard_id: Some("refresh-writer".to_string()),
                    merge_after_generations: Some(1),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            writer.add(&[assistant_record("row-0")]).await.unwrap();

            let mut cached_reader =
                RolloutStore::open_existing_with_options(&uri, RolloutStoreOptions::default())
                    .await
                    .unwrap();
            assert_eq!(cached_reader.list(None, None).await.unwrap().len(), 1);

            writer.add(&[assistant_record("row-1")]).await.unwrap();
            cached_reader.refresh_latest().await.unwrap();

            let ids: HashSet<_> = cached_reader
                .list(None, None)
                .await
                .unwrap()
                .into_iter()
                .map(|record| record.id)
                .collect();
            assert_eq!(
                ids,
                HashSet::from(["row-0".to_string(), "row-1".to_string()])
            );
        });
    }

    #[test]
    fn trajectory_rows_are_filtered_and_sorted_across_fragments() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    shard_id: Some("trajectory-test".to_string()),
                    merge_after_generations: Some(1),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

            let mut later = assistant_record("row-a");
            later.rollout_id = "target".to_string();
            later.sequence_order = 2;
            let mut unrelated = assistant_record("other");
            unrelated.rollout_id = "other".to_string();
            store.add(&[later, unrelated]).await.unwrap();

            let mut earlier = assistant_record("row-b");
            earlier.rollout_id = "target".to_string();
            earlier.sequence_order = 0;
            store.add(&[earlier]).await.unwrap();

            assert_eq!(store.observe().await.unwrap().fragment_count, 2);
            let rows = store.get_trajectory("target").await.unwrap();
            let ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
            assert_eq!(ids, vec!["row-b", "row-a"]);
        });
    }

    #[test]
    fn trajectory_read_rejects_an_empty_rollout_id() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = RolloutStore::open(&uri).await.unwrap();
            let err = store.get_trajectory("").await.unwrap_err();
            assert!(err.to_string().contains("rollout_id must not be empty"));
        });
    }

    /// Read the number of un-merged flushed generations recorded for a store's
    /// own write shard. Used by merge tests to assert the manifest drains.
    async fn flushed_generation_count(store: &RolloutStore) -> usize {
        let object_store = store.dataset.object_store(None).await.unwrap();
        let branch_location = store.dataset.branch_location();
        let manifest_store = ShardManifestStore::new(
            object_store,
            &branch_location.path,
            store.write_shard,
            DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
        );
        manifest_store
            .read_latest()
            .await
            .unwrap()
            .map(|m| m.flushed_generations.len())
            .unwrap_or(0)
    }

    /// Read the current writer epoch recorded for a store's own write shard.
    /// Used to assert the resident writer claims the epoch once instead of
    /// bumping it on every append.
    async fn shard_writer_epoch(store: &RolloutStore) -> u64 {
        let object_store = store.dataset.object_store(None).await.unwrap();
        let branch_location = store.dataset.branch_location();
        let manifest_store = ShardManifestStore::new(
            object_store,
            &branch_location.path,
            store.write_shard,
            DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
        );
        manifest_store
            .read_latest()
            .await
            .unwrap()
            .map(|m| m.writer_epoch)
            .unwrap_or(0)
    }

    #[test]
    fn add_reuses_resident_writer_epoch_stable() {
        // The resident writer must claim the shard epoch exactly once at open
        // and reuse it across appends (that reuse is what pools the object-store
        // connection instead of cold-rebuilding it per append). Every append is
        // still immediately visible cross-instance because it commits a flushed
        // generation to the manifest before returning.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: None, // no merge → epoch never reclaimed
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            let epoch_after_first = shard_writer_epoch(&store).await;

            // Read-after-write: the first append is visible right away.
            assert!(store.get_by_id("a-0").await.unwrap().is_some());

            for i in 1..5 {
                store
                    .add(&[assistant_record(&format!("a-{i}"))])
                    .await
                    .unwrap();
                // Each append is immediately visible.
                assert!(
                    store.get_by_id(&format!("a-{i}")).await.unwrap().is_some(),
                    "append a-{i} should be visible immediately"
                );
            }

            // The epoch did NOT advance per append: the writer was opened once.
            assert_eq!(
                shard_writer_epoch(&store).await,
                epoch_after_first,
                "resident writer should claim the epoch once, not per append"
            );

            // Every append landed as its own flushed generation.
            assert_eq!(flushed_generation_count(&store).await, 5);
            assert_eq!(store.list(None, None).await.unwrap().len(), 5);
        });
    }

    #[test]
    fn merge_reopens_writer_after_epoch_claim() {
        // A merge claims the shard epoch, which fences the resident writer. The
        // next append must transparently reopen the writer against the fresh
        // epoch rather than failing with a fence error.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    // Merge on every append → epoch is reclaimed each time, so
                    // the following append always hits the reopen path.
                    merge_after_generations: Some(1),
                },
            )
            .await
            .unwrap();

            for i in 0..4 {
                store
                    .add(&[assistant_record(&format!("a-{i}"))])
                    .await
                    .unwrap_or_else(|e| panic!("append a-{i} after merge must not be fenced: {e}"));
                assert!(store.get_by_id(&format!("a-{i}")).await.unwrap().is_some());
            }

            // Merges folded the generations into the base table.
            assert_eq!(flushed_generation_count(&store).await, 0);
            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 4);
        });
    }

    #[test]
    fn close_drains_resident_writer_and_add_reopens() {
        // close() must gracefully drop the resident writer, and a subsequent
        // add() must reopen one and keep working.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: None,
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            assert!(store.write_writer.is_some());

            // Idempotent close drops the resident writer.
            store.close().await.unwrap();
            assert!(store.write_writer.is_none());
            store.close().await.unwrap(); // second close is a no-op

            // add() after close reopens the writer and stays visible.
            store.add(&[assistant_record("a-1")]).await.unwrap();
            assert!(store.write_writer.is_some());
            assert_eq!(store.list(None, None).await.unwrap().len(), 2);
        });
    }

    #[test]
    fn compact_reduces_fragments_and_preserves_reads() {
        // Each WAL merge appends a fragment to the base table, so several merges
        // leave many small fragments. compact() folds them into fewer fragments
        // while every row stays readable exactly once and inline artifact bytes
        // remain fetchable.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let artifact_bytes = b"\x00\x01\x02compacted";
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    // Merge every append into base so each forms its own fragment.
                    merge_after_generations: Some(1),
                },
            )
            .await
            .unwrap();

            for i in 0..6 {
                store
                    .add(&[assistant_record(&format!("a-{i}"))])
                    .await
                    .unwrap();
            }
            store
                .add(&[artifact_record("a-6", artifact_bytes)])
                .await
                .unwrap();

            let before = store.dataset.count_fragments();
            assert!(before > 1, "expected several fragments, got {before}");
            assert!(store.should_compact(&CompactionConfig {
                min_fragments: 2,
                ..CompactionConfig::default()
            }));

            let metrics = store.compact(None).await.unwrap();
            assert!(metrics.fragments_removed > 0);

            let after = store.dataset.count_fragments();
            assert!(
                after < before,
                "compaction should reduce fragments: {before} -> {after}"
            );

            // All rows conserved, each exactly once.
            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 7);
            let mut ids: Vec<_> = listed.iter().map(|r| r.id.clone()).collect();
            ids.sort();
            assert_eq!(ids, vec!["a-0", "a-1", "a-2", "a-3", "a-4", "a-5", "a-6"]);
            // Inline artifact bytes survive compaction.
            assert_eq!(
                store.get_blob("a-6").await.unwrap().as_deref(),
                Some(&artifact_bytes[..])
            );

            // Stats reflect the successful compaction.
            let stats = store.compaction_stats();
            assert_eq!(stats.total_compactions, 1);
            assert!(stats.last_compaction.is_some());
            assert!(stats.last_error.is_none());
        });
    }

    #[test]
    fn should_compact_respects_min_fragments() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: Some(1),
                },
            )
            .await
            .unwrap();
            store.add(&[assistant_record("a-0")]).await.unwrap();

            let frags = store.dataset.count_fragments();
            // A threshold above the current fragment count says "don't compact".
            assert!(!store.should_compact(&CompactionConfig {
                min_fragments: frags + 1,
                ..CompactionConfig::default()
            }));
            // At/below it says "compact".
            assert!(store.should_compact(&CompactionConfig {
                min_fragments: frags,
                ..CompactionConfig::default()
            }));
        });
    }

    #[test]
    fn compact_composes_with_concurrent_wal_merge() {
        // A base-table compaction (Rewrite) and a WAL merge (Append) are
        // non-conflicting in Lance's commit matrix: running them concurrently
        // must not fail, and no rows are lost. Instance A compacts while
        // instance B (a different shard) merges its own generations into the
        // same base table.
        use tokio::sync::RwLock;

        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            // Seed the base table via A with several fragments to compact.
            let mut a = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: Some(1),
                },
            )
            .await
            .unwrap();
            for i in 0..5 {
                a.add(&[assistant_record(&format!("a-{i}"))]).await.unwrap();
            }

            // B accumulates its own shard's generations (not yet merged).
            let mut b = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-1".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            for i in 0..3 {
                b.add(&[assistant_record(&format!("b-{i}"))]).await.unwrap();
            }

            let a = Arc::new(RwLock::new(a));
            let b = Arc::new(RwLock::new(b));

            // A compacts the base table; B merges its shard into it — concurrently.
            let (ca, mb) = tokio::join!(
                async {
                    let mut g = a.write().await;
                    g.compact(None).await
                },
                async {
                    let mut g = b.write().await;
                    g.cleanup_own_shard().await
                },
            );
            ca.expect("compaction should not fail against a concurrent append");
            assert_eq!(mb.expect("wal merge should not fail"), 3);

            // A fresh reader sees all 8 rows exactly once.
            let reader = RolloutStore::open(&uri).await.unwrap();
            let listed = reader.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 8);
        });
    }

    #[test]
    fn below_threshold_no_merge() {
        // With a threshold of 3, two appends must NOT trigger a merge: the
        // flushed generations stay in `_mem_wal/` and the base table version
        // does not advance from appends.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: Some(3),
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            store.add(&[assistant_record("a-1")]).await.unwrap();

            // Two generations accumulated, threshold (3) not reached: no merge.
            assert_eq!(flushed_generation_count(&store).await, 2);
            // All rows still readable via the LSM union.
            assert_eq!(store.list(None, None).await.unwrap().len(), 2);
        });
    }

    #[test]
    fn merge_at_threshold_drains_shard_and_preserves_reads() {
        // At the threshold, `add` folds the shard's flushed generations into
        // the base table and drains `flushed_generations` to empty. Reads still
        // return every row exactly once (base ∪ empty shard, dedup by id), and
        // inline artifact bytes remain fetchable after the merge.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let artifact_bytes = b"\x00\x01\x02merged-trace";
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: Some(3),
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            store.add(&[assistant_record("a-1")]).await.unwrap();
            // The third append reaches the threshold and triggers the merge.
            store
                .add(&[artifact_record("a-2", artifact_bytes)])
                .await
                .unwrap();

            // Shard drained: no un-merged generations remain.
            assert_eq!(flushed_generation_count(&store).await, 0);

            // Every row is still present exactly once (no duplication despite
            // the base table now holding what the shard used to).
            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 3);
            let mut ids: Vec<_> = listed.iter().map(|r| r.id.clone()).collect();
            ids.sort();
            assert_eq!(ids, vec!["a-0", "a-1", "a-2"]);

            // Point lookup and inline artifact bytes survive the merge into base.
            let fetched = store.get_by_id("a-2").await.unwrap().unwrap();
            assert!(fetched.is_artifact());
            assert_eq!(
                store.get_blob("a-2").await.unwrap().as_deref(),
                Some(&artifact_bytes[..])
            );
        });
    }

    #[test]
    fn merge_is_visible_to_a_fresh_reader_instance() {
        // After a merge folds rows into the base table and drains the shard, a
        // freshly opened store (new process / reader) that re-reads manifests
        // from object storage sees exactly the merged rows — proving the data
        // truly landed in the base table, not just this handle's memory.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            {
                let mut store = RolloutStore::open_with_options(
                    &uri,
                    RolloutStoreOptions {
                        storage_options: None,
                        shard_id: Some("rollout-0".to_string()),
                        merge_after_generations: Some(2),
                    },
                )
                .await
                .unwrap();
                store.add(&[assistant_record("a-0")]).await.unwrap();
                store.add(&[assistant_record("a-1")]).await.unwrap();
                assert_eq!(flushed_generation_count(&store).await, 0);
            }

            // A brand-new reader opens the same dataset.
            let reader = RolloutStore::open(&uri).await.unwrap();
            let listed = reader.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 2);
            assert!(listed.iter().any(|r| r.id == "a-0"));
            assert!(listed.iter().any(|r| r.id == "a-1"));
        });
    }

    #[test]
    fn cleanup_own_shard_merges_whatever_is_pending() {
        // The periodic-cleanup entry point (`cleanup_own_shard`) is the time
        // trigger's per-pass body. Time and count are a strict OR, so the time
        // trigger is NOT gated by any generation count: with count self-merge
        // disabled, a single pending generation is merged the moment the pass
        // runs. A pass with nothing pending is a no-op.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: None, // count trigger off
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            // One generation pending: the time trigger merges it immediately —
            // it does not wait for a count threshold.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 1);
            assert_eq!(flushed_generation_count(&store).await, 0);

            store.add(&[assistant_record("a-1")]).await.unwrap();
            // Next generation is likewise merged on the following pass.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 1);
            assert_eq!(flushed_generation_count(&store).await, 0);

            // Rows survive the merge, readable exactly once.
            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 2);

            // Nothing pending: a further pass reclaims nothing.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 0);
        });
    }

    #[test]
    fn create_id_zonemap_index_builds_and_is_idempotent() {
        // Building the ZoneMap index on `id` must succeed even though the
        // rollout table also carries a (fieldless) MemWAL index, and calling it
        // twice must not error (replace(true) rebuilds in place).
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            // Add rows and fold them into the base table so there is data (and a
            // MemWAL index) present when we build the scalar index.
            store.add(&[assistant_record("a-0")]).await.unwrap();
            store.add(&[assistant_record("a-1")]).await.unwrap();
            store.cleanup_own_shard().await.unwrap();

            store.create_id_zonemap_index().await.unwrap();
            let has_id_index = |s: &RolloutStore| {
                let dataset = s.dataset.clone();
                async move {
                    dataset
                        .load_indices()
                        .await
                        .unwrap()
                        .iter()
                        .any(|i| i.name == ROLLOUT_ID_INDEX_NAME)
                }
            };
            assert!(has_id_index(&store).await, "id index should exist");

            // Idempotent: a second build replaces in place without erroring.
            store.create_id_zonemap_index().await.unwrap();
            assert!(has_id_index(&store).await, "id index should still exist");

            // Rows remain readable exactly once after indexing.
            assert_eq!(store.list(None, None).await.unwrap().len(), 2);
        });
    }

    #[test]
    fn concurrent_merges_from_two_shards_into_one_base_table() {
        // Case 2 of the concurrency hardening: two instances (distinct shards)
        // fold their flushed generations into the SAME base table at the same
        // time. Lance's optimistic concurrency retries the second append on the
        // latest version (Append vs Append is non-conflicting), so no commit is
        // lost. Each instance drains only its own shard manifest. Afterwards
        // every row is present exactly once and both shards are empty.
        use tokio::sync::RwLock;

        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let make = |shard: &str| {
                let uri = uri.clone();
                let shard = shard.to_string();
                async move {
                    RolloutStore::open_with_options(
                        &uri,
                        RolloutStoreOptions {
                            storage_options: None,
                            shard_id: Some(shard),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap()
                }
            };

            // Two instances, each accumulates a few generations on its own shard.
            let mut a = make("rollout-0").await;
            let mut b = make("rollout-1").await;
            for i in 0..4 {
                a.add(&[assistant_record(&format!("a-{i}"))]).await.unwrap();
                b.add(&[assistant_record(&format!("b-{i}"))]).await.unwrap();
            }
            assert_eq!(flushed_generation_count(&a).await, 4);
            assert_eq!(flushed_generation_count(&b).await, 4);

            let a = Arc::new(RwLock::new(a));
            let b = Arc::new(RwLock::new(b));

            // Both merge into the shared base table concurrently.
            let (ra, rb) = tokio::join!(
                async {
                    let mut g = a.write().await;
                    g.cleanup_own_shard().await
                },
                async {
                    let mut g = b.write().await;
                    g.cleanup_own_shard().await
                },
            );
            assert_eq!(ra.unwrap(), 4);
            assert_eq!(rb.unwrap(), 4);

            // Both shards drained; a fresh reader sees all 8 rows exactly once.
            assert_eq!(flushed_generation_count(&*a.read().await).await, 0);
            assert_eq!(flushed_generation_count(&*b.read().await).await, 0);

            let reader = RolloutStore::open(&uri).await.unwrap();
            let listed = reader.list(None, None).await.unwrap();
            let mut ids: Vec<_> = listed.iter().map(|r| r.id.clone()).collect();
            ids.sort();
            let expected: Vec<String> = (0..4)
                .flat_map(|i| [format!("a-{i}"), format!("b-{i}")])
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            assert_eq!(ids, expected);
            assert_eq!(listed.len(), 8);
        });
    }

    #[test]
    fn merge_drains_only_merged_generations() {
        // Regression guard for the surgical drain: the manifest commit removes
        // exactly the generations that were merged, so if a generation were to
        // appear that wasn't part of this merge it would be retained, not
        // silently discarded. We assert the row count is conserved end to end:
        // every appended row is readable exactly once after a merge, proving no
        // generation was dropped without being folded into the base table.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

            for i in 0..3 {
                store
                    .add(&[assistant_record(&format!("g-{i}"))])
                    .await
                    .unwrap();
            }
            assert_eq!(flushed_generation_count(&store).await, 3);

            // Merge all three. Surgical drain removes exactly generations {merged}.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 3);
            assert_eq!(flushed_generation_count(&store).await, 0);

            // Append a fourth AFTER the drain: it forms a new generation that the
            // prior merge must not have wiped. A second merge folds just that one.
            store.add(&[assistant_record("g-3")]).await.unwrap();
            assert_eq!(flushed_generation_count(&store).await, 1);
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 1);

            // All four rows conserved, each exactly once.
            let listed = store.list(None, None).await.unwrap();
            let mut ids: Vec<_> = listed.iter().map(|r| r.id.clone()).collect();
            ids.sort();
            assert_eq!(ids, vec!["g-0", "g-1", "g-2", "g-3"]);
        });
    }

    #[test]
    fn list_source_splits_base_and_wal() {
        // Rows appended but not yet merged live only in the WAL, not the base
        // table. `Fragments` must omit them; `Wal` must show exactly them; `All`
        // is the union. After a merge the rows move to the base table and the WAL
        // empties, flipping which source sees them.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

            async fn list_ids(store: &RolloutStore, source: ListSource) -> Vec<String> {
                let page = store
                    .list_filtered_source(&RolloutFilters::default(), 25, 0, source)
                    .await
                    .unwrap();
                let mut ids: Vec<_> = page.records.iter().map(|r| r.id.clone()).collect();
                ids.sort();
                ids
            }

            // Two un-merged rows sit in the WAL only.
            store.add(&[assistant_record("g-0")]).await.unwrap();
            store.add(&[assistant_record("g-1")]).await.unwrap();
            assert_eq!(flushed_generation_count(&store).await, 2);

            assert!(
                list_ids(&store, ListSource::Fragments).await.is_empty(),
                "fragments must not see un-merged WAL rows"
            );
            assert_eq!(list_ids(&store, ListSource::Wal).await, vec!["g-0", "g-1"]);
            assert_eq!(list_ids(&store, ListSource::All).await, vec!["g-0", "g-1"]);
            // The default wrapper preserves the historical union semantics.
            let default_page = store
                .list_filtered(&RolloutFilters::default(), 25, 0)
                .await
                .unwrap();
            let mut default_ids: Vec<_> =
                default_page.records.iter().map(|r| r.id.clone()).collect();
            default_ids.sort();
            assert_eq!(default_ids, vec!["g-0", "g-1"]);

            // Merge folds the WAL into the base table and drains it.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 2);
            assert_eq!(flushed_generation_count(&store).await, 0);

            assert_eq!(
                list_ids(&store, ListSource::Fragments).await,
                vec!["g-0", "g-1"]
            );
            assert!(
                list_ids(&store, ListSource::Wal).await.is_empty(),
                "wal must be empty after a merge"
            );
            assert_eq!(list_ids(&store, ListSource::All).await, vec!["g-0", "g-1"]);
        });
    }

    /// Reproduces the master data-browser workload at the reported scale and
    /// compares the former wide-row pagination plan with the late-materialized
    /// implementation.
    ///
    /// Run explicitly with:
    /// `cargo test -p lance-context-core bench_master_pagination_90k_52_fragments -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark; creates 90k rows across 52 fragments"]
    fn bench_master_pagination_90k_52_fragments() {
        use std::time::Instant;

        const ROWS: usize = 90_000;
        const FRAGMENTS: usize = 52;
        const PAGE_SIZE: usize = 25;

        async fn legacy_wide_page(
            store: &RolloutStore,
            limit: usize,
        ) -> LanceResult<Vec<RolloutRecord>> {
            let shard_snapshots = store.wal_shard_snapshots().await?;
            let columns = store.non_blob_columns();
            let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
            let scanner = store
                .lsm_scanner_with_snapshots(shard_snapshots)
                .project(&refs)
                .limit(limit.saturating_add(1), Some(0));
            let mut stream = scanner.try_into_stream().await?;
            let mut records = Vec::new();
            while let Some(batch) = stream.try_next().await? {
                records.extend(batch_to_rollout_records(&batch)?);
            }
            records.truncate(limit);
            Ok(records)
        }

        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut writer = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    shard_id: Some("pagination-benchmark".to_string()),
                    merge_after_generations: Some(1),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

            let rows_per_fragment = ROWS.div_ceil(FRAGMENTS);
            let write_start = Instant::now();
            for fragment in 0..FRAGMENTS {
                let start = fragment * rows_per_fragment;
                let end = ((fragment + 1) * rows_per_fragment).min(ROWS);
                if start >= end {
                    break;
                }
                let records: Vec<_> = (start..end)
                    .map(|row| assistant_record(&format!("row-{row:06}")))
                    .collect();
                writer.add(&records).await.unwrap();
            }
            writer.close().await.unwrap();

            let observation = writer.observe().await.unwrap();
            assert_eq!(observation.row_count, ROWS as i64);
            assert_eq!(observation.fragment_count, FRAGMENTS as i64);

            let legacy_reader =
                RolloutStore::open_existing_with_options(&uri, RolloutStoreOptions::default())
                    .await
                    .unwrap();
            let legacy_start = Instant::now();
            let legacy_rows = legacy_wide_page(&legacy_reader, PAGE_SIZE).await.unwrap();
            let legacy_elapsed = legacy_start.elapsed();
            assert_eq!(legacy_rows.len(), PAGE_SIZE);

            let mut optimized_reader =
                RolloutStore::open_existing_with_options(&uri, RolloutStoreOptions::default())
                    .await
                    .unwrap();
            let optimized_start = Instant::now();
            let optimized_page = optimized_reader
                .list_filtered(&RolloutFilters::default(), PAGE_SIZE, 0)
                .await
                .unwrap();
            let optimized_elapsed = optimized_start.elapsed();
            assert_eq!(optimized_page.records.len(), PAGE_SIZE);
            assert!(optimized_page.has_more);
            assert_eq!(
                optimized_page
                    .records
                    .iter()
                    .map(|record| record.id.as_str())
                    .collect::<Vec<_>>(),
                legacy_rows
                    .iter()
                    .map(|record| record.id.as_str())
                    .collect::<Vec<_>>()
            );

            optimized_reader.refresh_latest().await.unwrap();
            let cached_start = Instant::now();
            let cached_page = optimized_reader
                .list_filtered(&RolloutFilters::default(), PAGE_SIZE, PAGE_SIZE)
                .await
                .unwrap();
            let cached_elapsed = cached_start.elapsed();
            assert_eq!(cached_page.records.len(), PAGE_SIZE);
            assert!(cached_page.has_more);

            println!("\n=== master pagination benchmark ===");
            println!(
                "  dataset                              : {ROWS} rows / {FRAGMENTS} fragments"
            );
            println!(
                "  dataset construction                 : {:?}",
                write_start.elapsed()
            );
            println!("  former wide-row LSM page             : {legacy_elapsed:?}");
            println!("  late-materialized ID + row fetch page: {optimized_elapsed:?}");
            println!("  cached-handle next page              : {cached_elapsed:?}");
            println!(
                "  speedup                              : {:.1}x",
                legacy_elapsed.as_secs_f64() / optimized_elapsed.as_secs_f64().max(1e-9)
            );
        });
    }

    /// Micro-benchmark (run with `cargo test -- --ignored --nocapture
    /// bench_merge_read_amplification`): quantifies the read-amplification that
    /// self-merge removes. Appends N generations, times a `list` scan over the
    /// accumulated `_mem_wal/` generations, then merges them into the base
    /// table and times an equivalent scan. Also reports the tail latency of the
    /// single append that triggers the merge.
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn bench_merge_read_amplification() {
        use std::time::Instant;

        const N: usize = 200;
        const READ_ITERS: usize = 20;

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            // --- Un-merged: accumulate N generations, never merge. ---
            let dir_no_merge = TempDir::new().unwrap();
            let uri = dir_no_merge.path().to_string_lossy().to_string();
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: None, // disabled
                },
            )
            .await
            .unwrap();
            for i in 0..N {
                store
                    .add(&[assistant_record(&format!("row-{i}"))])
                    .await
                    .unwrap();
            }
            assert_eq!(flushed_generation_count(&store).await, N);

            let start = Instant::now();
            for _ in 0..READ_ITERS {
                let rows = store.list(None, None).await.unwrap();
                assert_eq!(rows.len(), N);
            }
            let unmerged_read = start.elapsed() / READ_ITERS as u32;

            // --- Merged: same N rows, but self-merge folds each batch into
            // base as soon as it lands (threshold 1). ---
            let dir_merge = TempDir::new().unwrap();
            let uri_m = dir_merge.path().to_string_lossy().to_string();
            let mut merge_store = RolloutStore::open_with_options(
                &uri_m,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: Some(1),
                },
            )
            .await
            .unwrap();
            let mut max_add = std::time::Duration::ZERO;
            for i in 0..N {
                let t = Instant::now();
                merge_store
                    .add(&[assistant_record(&format!("row-{i}"))])
                    .await
                    .unwrap();
                max_add = max_add.max(t.elapsed());
            }
            assert_eq!(flushed_generation_count(&merge_store).await, 0);

            let start = Instant::now();
            for _ in 0..READ_ITERS {
                let rows = merge_store.list(None, None).await.unwrap();
                assert_eq!(rows.len(), N);
            }
            let merged_read = start.elapsed() / READ_ITERS as u32;

            println!("\n=== merge read-amplification benchmark (N={N} generations) ===");
            println!("  list scan, {N} un-merged generations : {unmerged_read:?}");
            println!("  list scan, merged into base table     : {merged_read:?}");
            println!(
                "  read speedup from merge               : {:.1}x",
                unmerged_read.as_secs_f64() / merged_read.as_secs_f64().max(1e-9)
            );
            println!("  slowest single add() (merge on every append) : {max_add:?}");
        });
    }

    #[test]
    fn point_lookup_base_first_finds_unmerged_and_merged_rows() {
        // get_by_id / get_blob default to base-table-first with a WAL fallback.
        // The fallback must still find a row that is only in the (un-merged)
        // WAL, and after a merge the same row must be found via the base table.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let bytes = b"\x00\x01payload-x";
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            // merge_after_generations = None: appended rows stay in the WAL,
            // un-merged, so this exercises the base-miss -> WAL-fallback path.
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let artifact = artifact_record("row-x", bytes);
            store.add(std::slice::from_ref(&artifact)).await.unwrap();

            // Row is only in the WAL (nothing merged): base-first misses, the
            // fallback finds it.
            assert_eq!(flushed_generation_count(&store).await, 1);
            let got = store.get_by_id("row-x").await.unwrap().unwrap();
            assert_records_eq(&got, &artifact);
            assert_eq!(
                store.get_blob("row-x").await.unwrap().as_deref(),
                Some(&bytes[..]),
                "get_blob must find an un-merged WAL row via the fallback"
            );

            // Fragments source (base only) must NOT see the un-merged row;
            // Wal source must.
            assert!(store
                .get_by_id_source("row-x", ListSource::Fragments)
                .await
                .unwrap()
                .is_none());
            assert!(store
                .get_by_id_source("row-x", ListSource::Wal)
                .await
                .unwrap()
                .is_some());

            // Merge folds the row into the base table and drains the WAL.
            store.cleanup_own_shard().await.unwrap();
            assert_eq!(flushed_generation_count(&store).await, 0);

            // Now base-first hits the base table with zero generations open.
            let got = store.get_by_id("row-x").await.unwrap().unwrap();
            assert_records_eq(&got, &artifact);
            assert_eq!(
                store.get_blob("row-x").await.unwrap().as_deref(),
                Some(&bytes[..])
            );
            // After merge the row is in the base table (Fragments sees it) and
            // no longer in the WAL.
            assert!(store
                .get_by_id_source("row-x", ListSource::Fragments)
                .await
                .unwrap()
                .is_some());
            assert!(store
                .get_by_id_source("row-x", ListSource::Wal)
                .await
                .unwrap()
                .is_none());

            // A miss returns None on every source (no panic, no error).
            assert!(store.get_by_id("nope").await.unwrap().is_none());
            assert!(store.get_blob("nope").await.unwrap().is_none());
        });
    }

    #[test]
    fn point_lookup_immutability_contract_returns_base_version() {
        // Contract guard: rollout ids are immutable and never re-appended, so a
        // row present in the base table is authoritative. This pins that
        // base-first returns the base row and does not require scanning the WAL
        // to be correct — if someone later violates the no-overwrite contract by
        // re-appending the same id, this test documents that base-first would
        // return the base (merged) version, surfacing the contract break here
        // rather than as silent stale reads in production.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let bytes = b"base-version-bytes";
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let base = artifact_record("dup", bytes);
            store.add(std::slice::from_ref(&base)).await.unwrap();
            store.cleanup_own_shard().await.unwrap();
            assert_eq!(flushed_generation_count(&store).await, 0);

            // Base table holds the authoritative row; base-first returns it
            // without opening any WAL generation.
            let got = store.get_by_id("dup").await.unwrap().unwrap();
            assert_records_eq(&got, &base);
            assert_eq!(
                store.get_blob("dup").await.unwrap().as_deref(),
                Some(&bytes[..])
            );
        });
    }
}
