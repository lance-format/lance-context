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
    StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use futures::TryStreamExt;
use lance::dataset::mem_wal::{
    DatasetMemWalExt, LsmScanner, ShardManifestStore, ShardSnapshot, ShardWriterConfig,
};
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};
use lance_index::mem_wal::{ShardManifest, MEM_WAL_INDEX_NAME};
use tokio::task::JoinHandle;
use tracing::{info, warn};
use uuid::Uuid;

use crate::rollout::RolloutRecord;
use crate::store::{
    column_as, column_as_optional, relationship_field, relationship_list_item_field,
    relationship_struct_builder, relationships_from_list, timestamp_from_micros,
    RELATIONSHIPS_COLUMN,
};

/// Number of shard manifest files to scan per batch when discovering the latest
/// shard state (mirrors the constant used by `ContextStore`).
const DEFAULT_MANIFEST_SCAN_BATCH_SIZE: usize = 16;

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
    /// Interval, in seconds, for the periodic per-shard WAL cleanup task. When
    /// set (and non-zero), a background timer wakes up every `interval` seconds
    /// and folds this instance's flushed MemWAL generations into the base table
    /// — the *time* half of the "time + count" trigger. It complements the
    /// synchronous, count-triggered [`Self::merge_after_generations`]: even a
    /// low-traffic shard that never crosses the count threshold still gets its
    /// generations reclaimed on a timer, so read amplification stays bounded and
    /// object storage does not accumulate stale generation datasets.
    ///
    /// The cleanup runs on the shard's own owner (reusing its writer epoch), so
    /// it never fences a concurrent writer — exactly like the count-triggered
    /// merge. `None` or `0` disables the periodic task.
    ///
    /// Spawn the task with [`RolloutStore::spawn_periodic_cleanup`] once the
    /// store is behind an `Arc<RwLock<_>>` (the server's ownership model).
    pub cleanup_interval_secs: Option<u64>,
    /// The *count* half of the periodic cleanup trigger: the timer only merges
    /// when this instance's shard has at least this many flushed generations,
    /// skipping the pass otherwise (avoids rewriting the base table to reclaim a
    /// single small generation). `None` defaults to `1` (clean up whenever any
    /// generation is present on a tick).
    pub cleanup_min_generations: Option<usize>,
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
    /// Periodic-cleanup interval in seconds; `0` disables the timer. See
    /// [`RolloutStoreOptions::cleanup_interval_secs`].
    cleanup_interval_secs: u64,
    /// Minimum flushed generations before a periodic-cleanup tick merges.
    /// Normalized to at least `1`. See
    /// [`RolloutStoreOptions::cleanup_min_generations`].
    cleanup_min_generations: usize,
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
    pub async fn open_with_options(uri: &str, options: RolloutStoreOptions) -> LanceResult<Self> {
        let storage_options = options.storage_options.clone();
        let write_shard = derive_shard_id(options.shard_id.as_deref());
        let dataset = match Self::load_with_options(uri, storage_options.clone()).await {
            Ok(dataset) => dataset,
            Err(LanceError::DatasetNotFound { .. }) => {
                Self::create_with_options(uri, storage_options.clone()).await?
            }
            Err(err) => return Err(err),
        };

        Ok(Self {
            dataset,
            write_shard,
            storage_options,
            merge_after_generations: options.merge_after_generations.unwrap_or(0),
            cleanup_interval_secs: options.cleanup_interval_secs.unwrap_or(0),
            cleanup_min_generations: options.cleanup_min_generations.unwrap_or(1).max(1),
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

    /// Append rollout rows through this instance's MemWAL shard; returns the
    /// current base dataset version.
    ///
    /// The write is routed to the shard derived from the configured
    /// `shard_id`, so concurrent appends from other server instances (each
    /// owning a distinct shard) never contend. `close`-per-append flushes the
    /// rows to object storage before returning, so they are immediately visible
    /// to reads on any instance (see [`Self::lsm_scanner`]).
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

        let config = ShardWriterConfig {
            shard_id: self.write_shard,
            ..Default::default()
        };
        let writer = self
            .dataset
            .mem_wal_writer(self.write_shard, config)
            .await?;
        writer.put(vec![batch]).await?;
        writer.close().await?;

        // Count-triggered self-merge: if this instance's shard has accumulated
        // enough un-merged flushed generations, fold them into the base table
        // now (spec §6). Bounds read amplification. Runs on the shard's own
        // owner so it never fences a concurrent writer.
        if self.merge_after_generations > 0 {
            self.maybe_merge_own_shard().await?;
        }

        Ok(self.dataset.manifest.version)
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
    /// periodic timer in [`Self::spawn_periodic_cleanup`] (with
    /// `threshold = cleanup_min_generations`). Both merge only the shard this
    /// instance owns and writes, so the epoch claim never fences another writer.
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

    /// Run one periodic WAL-cleanup pass over this instance's own shard: fold any
    /// flushed generations into the base table once at least
    /// `cleanup_min_generations` have accumulated. Returns the number of
    /// generations reclaimed (`0` if below the threshold or nothing pending).
    ///
    /// Exposed for callers that drive cleanup on their own schedule instead of
    /// (or in addition to) the built-in timer from
    /// [`Self::spawn_periodic_cleanup`]. Like every merge path here it operates
    /// only on the shard this instance owns, so it is safe to call concurrently
    /// with this instance's own appends but must not target another instance's
    /// shard.
    pub async fn cleanup_own_shard(&mut self) -> LanceResult<usize> {
        self.merge_own_shard_if_ready(self.cleanup_min_generations)
            .await
    }

    /// Spawn a background timer that periodically reclaims this instance's
    /// flushed MemWAL generations into the base table (the *time* trigger).
    ///
    /// Every `cleanup_interval_secs` seconds the task acquires the write lock and
    /// calls [`Self::cleanup_own_shard`], which merges only when at least
    /// `cleanup_min_generations` are pending. This bounds read amplification and
    /// reclaims stale generation datasets even on shards that never cross the
    /// synchronous count threshold ([`RolloutStoreOptions::merge_after_generations`]).
    ///
    /// Returns `None` (spawning nothing) when `cleanup_interval_secs` is `0`
    /// (disabled). Otherwise returns the task handle.
    ///
    /// The task holds only a [`std::sync::Weak`] reference to the store, so it
    /// never keeps a deleted store alive: once the caller drops the last strong
    /// `Arc` (e.g. the server removes it from its store map on delete), the next
    /// tick fails to upgrade and the loop exits on its own. `abort`ing the
    /// returned handle also stops it immediately.
    pub fn spawn_periodic_cleanup(store: Arc<tokio::sync::RwLock<Self>>) -> Option<JoinHandle<()>> {
        let interval_secs = {
            // Read the interval without holding the lock across the await loop.
            let guard = store.try_read().ok()?;
            guard.cleanup_interval_secs
        };
        if interval_secs == 0 {
            return None;
        }

        let weak = Arc::downgrade(&store);
        Some(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            // Skip the immediate first tick so we don't merge the instant the
            // task starts; wait a full interval before the first pass.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                // Stop once the store has been dropped by its owner.
                let Some(store) = weak.upgrade() else {
                    return;
                };
                let mut guard = store.write().await;
                match guard.cleanup_own_shard().await {
                    Ok(0) => {}
                    Ok(n) => {
                        info!(
                            shard = %guard.write_shard,
                            reclaimed = n,
                            "periodic WAL cleanup merged flushed generations"
                        );
                    }
                    Err(e) => {
                        warn!(
                            shard = %guard.write_shard,
                            error = %e,
                            "periodic WAL cleanup failed"
                        );
                    }
                }
            }
        }))
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

        // Resolve each flushed generation to its absolute dataset path and read
        // all its rows into memory. Record which generation ids we merge so the
        // drain can remove exactly these and nothing else.
        let base_uri = self.dataset.uri().trim_end_matches('/').to_string();
        let mut merged_generations: HashSet<u64> = HashSet::new();
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

        Ok(())
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
        let columns = self.non_blob_columns();
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let scanner = self.lsm_scanner().await?.project(&refs);
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

    /// Retrieve a single rollout row by its unique id, including any freshly
    /// appended (MemWAL-flushed) row on any instance. `binary_payload` is
    /// projected out (fetch bytes via [`Self::get_blob`]).
    pub async fn get_by_id(&self, id: &str) -> LanceResult<Option<RolloutRecord>> {
        let escaped_id = id.replace('\'', "''");
        let columns = self.non_blob_columns();
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let scanner = self
            .lsm_scanner()
            .await?
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
    /// `None` there. This method projects it in and reads the inline bytes
    /// directly. Returns `None` if the row or its payload is absent.
    pub async fn get_blob(&self, id: &str) -> LanceResult<Option<Vec<u8>>> {
        let escaped_id = id.replace('\'', "''");
        let scanner = self
            .lsm_scanner()
            .await?
            .project(&["id", "binary_payload"])
            .filter(&format!("id = '{}'", escaped_id))?;
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

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open(&uri).await.unwrap();
            // MemWAL appends land in the `_mem_wal` namespace and do not advance
            // the base dataset version; `add` returns it unchanged.
            store
                .add(&[assistant.clone(), artifact.clone()])
                .await
                .unwrap();

            // The LSM read path dedups by `id` across generations and does not
            // guarantee append order, so look rows up by id rather than by
            // position.
            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 2);
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
            // The assistant row carries no payload.
            assert_eq!(store.get_blob("row-0").await.unwrap(), None);

            assert!(store.get_by_id("missing").await.unwrap().is_none());
            assert_eq!(store.get_blob("missing").await.unwrap(), None);
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
            let options = |shard: &str| RolloutStoreOptions {
                storage_options: None,
                shard_id: Some(shard.to_string()),
                merge_after_generations: None,
                ..Default::default()
            };

            let mut instance_a = RolloutStore::open_with_options(&uri, options("rollout-0"))
                .await
                .unwrap();
            instance_a.add(&[assistant_record("a-0")]).await.unwrap();

            let mut instance_b = RolloutStore::open_with_options(&uri, options("rollout-1"))
                .await
                .unwrap();
            instance_b.add(&[assistant_record("b-0")]).await.unwrap();

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
                    ..Default::default()
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
                    ..Default::default()
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
                        ..Default::default()
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
    fn cleanup_own_shard_merges_only_at_min_generations() {
        // The periodic-cleanup entry point (`cleanup_own_shard`) is the time
        // trigger's per-pass body. With count self-merge disabled, generations
        // accumulate; a cleanup pass below `cleanup_min_generations` is a no-op,
        // and one at/above the threshold drains the shard into the base table.
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
                    cleanup_min_generations: Some(2),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            // One generation pending, below min (2): cleanup is a no-op.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 0);
            assert_eq!(flushed_generation_count(&store).await, 1);

            store.add(&[assistant_record("a-1")]).await.unwrap();
            // Two generations pending, at min: cleanup merges both.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 2);
            assert_eq!(flushed_generation_count(&store).await, 0);

            // Rows survive the merge, readable exactly once.
            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), 2);

            // Nothing pending: a further pass reclaims nothing.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 0);
        });
    }

    #[test]
    fn spawn_periodic_cleanup_reclaims_on_a_timer() {
        // The background timer folds accumulated generations into the base table
        // without any count-triggered append. A short interval lets the test
        // observe one cleanup pass drain the shard.
        use std::time::Duration;
        use tokio::sync::RwLock;

        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: None, // only the timer merges
                    cleanup_interval_secs: Some(1),
                    cleanup_min_generations: Some(1),
                },
            )
            .await
            .unwrap();
            let store = Arc::new(RwLock::new(store));

            // Accumulate generations that no count trigger will reclaim.
            {
                let mut guard = store.write().await;
                guard.add(&[assistant_record("a-0")]).await.unwrap();
                guard.add(&[assistant_record("a-1")]).await.unwrap();
                assert_eq!(flushed_generation_count(&guard).await, 2);
            }

            let handle =
                RolloutStore::spawn_periodic_cleanup(store.clone()).expect("timer enabled");

            // Wait for at least one tick (interval 1s, first tick skipped) to run
            // the cleanup pass and drain the shard.
            let drained = async {
                loop {
                    {
                        let guard = store.read().await;
                        if flushed_generation_count(&guard).await == 0 {
                            return true;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            };
            let ok = tokio::time::timeout(Duration::from_secs(10), drained)
                .await
                .unwrap_or(false);
            handle.abort();
            assert!(ok, "periodic cleanup did not drain the shard in time");

            let guard = store.read().await;
            assert_eq!(guard.list(None, None).await.unwrap().len(), 2);
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
                            cleanup_min_generations: Some(1),
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
                    cleanup_min_generations: Some(1),
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
    fn spawn_periodic_cleanup_disabled_returns_none() {
        // With `cleanup_interval_secs` unset (0), no timer task is spawned.
        use tokio::sync::RwLock;

        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    shard_id: Some("rollout-0".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let store = Arc::new(RwLock::new(store));
            assert!(RolloutStore::spawn_periodic_cleanup(store).is_none());
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
                    ..Default::default()
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
                    ..Default::default()
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
}
