//! Shared low-level storage layer for every fixed-schema store.
//!
//! [`StorageBase`] owns everything about talking to Lance that is not
//! schema-specific: opening/creating the dataset, the resident MemWAL
//! [`ShardWriter`] lifecycle, the durable append, the seal/flush, the
//! WAL→base-table merge, compaction, scalar-index management, and the LSM read
//! scanner. A concrete store (`RolloutStore`, `DatagenStore`, `ContextStore`)
//! is then only a *schema* plus an encode/decode pair on top of it.
//!
//! # Rollout semantics are the contract
//!
//! The behavior here is `RolloutStore`'s, verbatim — it was the only one of the
//! three fixed-schema stores with a complete and correct implementation of all
//! of the above. Every store that adopts this base therefore gets the same
//! answer to the questions the three used to answer differently:
//!
//! - **`add` is a durable `put` and nothing else.** It reuses a *resident*
//!   writer (epoch claimed once, object-store connection pooled) and retries
//!   exactly once on a fence. It does **not** seal, so concurrent appends are
//!   not serialized behind one seal+drain — and correspondingly it offers **no
//!   read-your-write guarantee**. Visibility is [`StorageBase::flush`]'s job.
//! - **Merge is surgical.** It drains only the generations it actually merged,
//!   reusing the shard's current epoch (so it never fences the live writer),
//!   and deletes generation directories only *after* the manifest drain.
//! - **Compaction defers index remap**, because the MemWAL index is fieldless
//!   and Lance's inline remap panics on it.
//! - **Every dataset open goes through [`StorageBase::load_with_options`]**, so
//!   storage options and the shared session are never silently dropped.
//!
//! # What stays in the concrete store
//!
//! Anything that needs to know the schema: the Arrow schema itself,
//! record↔`RecordBatch` conversion, filter expressions, projections, and
//! typed read APIs. The base is handed the two things it needs to be
//! schema-agnostic — the merge/primary key column name and (optionally) the
//! latest schema to evolve a base table to — via [`StorageBaseOptions`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{new_null_array, RecordBatch, RecordBatchIterator};
use arrow_schema::{ArrowError, Schema};
use chrono::{DateTime, Utc};
use futures::{stream, StreamExt, TryStreamExt};
use lance::dataset::mem_wal::{
    DatasetMemWalExt, LsmScanner, ShardManifestStore, ShardSnapshot, ShardWriter, ShardWriterConfig,
};
use lance::dataset::optimize::{compact_files, CompactionMetrics, CompactionOptions};
use lance::dataset::{
    builder::DatasetBuilder, Dataset, NewColumnTransform, WriteMode, WriteParams,
};
use lance::index::DatasetIndexExt;
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::session::Session;
use lance::{Error as LanceError, Result as LanceResult};
use lance_index::mem_wal::{ShardManifest, MEM_WAL_INDEX_NAME};
use lance_index::scalar::ScalarIndexParams;
use lance_index::IndexType;
use tracing::{info, warn};
use uuid::Uuid;

use crate::metrics::{count, observe_duration, observe_phase, timer_elapsed, timer_start};
use crate::store::{CompactionConfig, CompactionStats};

/// Number of shard manifest files to scan per batch when discovering the latest
/// shard state.
pub(crate) const DEFAULT_MANIFEST_SCAN_BATCH_SIZE: usize = 16;

/// Maximum number of shard manifests or flushed-generation datasets opened
/// concurrently while collecting observability metrics.
pub(crate) const DEFAULT_OBSERVE_CONCURRENCY: usize = 16;

/// Name of the scalar index on the base table's key column. One name across
/// every store, so all tables index their primary key identically.
pub(crate) const ID_INDEX_NAME: &str = "id_idx";

/// What a [`StorageBase::flush`] actually did, for the `outcome` metric label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlushOutcome {
    /// A memtable was sealed and drained into a flushed generation.
    Sealed,
    /// No resident writer, so nothing to seal (the common case).
    Noop,
    /// The writer's epoch was superseded by a merge; nothing to flush.
    Fenced,
}

/// Which physical sources a read scans.
///
/// A store's rows live in two places: the **base table** fragments
/// (`self.dataset`) and the pending **MemWAL** generations that have been
/// flushed but not yet merged into it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ListSource {
    /// Scan only the base table, skipping all MemWAL generations. Fast and
    /// bounded; may lag the most recent (un-merged) writes. This is the default.
    #[default]
    Fragments,
    /// Scan only the flushed MemWAL generations (the not-yet-merged tail),
    /// excluding the base table.
    Wal,
    /// Scan the base table unioned with every flushed MemWAL generation — fully
    /// consistent, and the behavior of the historical union read path.
    All,
}

/// Rows read out of the flushed generations, ready to be appended to the base
/// table and drained from the shard manifest.
///
/// Produced by `StorageBase::prepare_merge_if_ready` under `&self` (so appends
/// keep running while it reads object storage) and consumed by
/// `StorageBase::commit_prepared_merge` under `&mut self`. `PreparedMerge` is
/// public because it appears in [`RolloutStore`]'s prepare/commit split, but
/// its fields are opaque.
///
/// [`RolloutStore`]: crate::RolloutStore
pub struct PreparedMerge {
    merged_generations: HashSet<u64>,
    merged_paths: Vec<String>,
    batches: Vec<RecordBatch>,
    merge_schema: Arc<Schema>,
}

impl PreparedMerge {
    /// Number of generations this merge will reclaim.
    #[must_use]
    pub fn generation_count(&self) -> usize {
        self.merged_generations.len()
    }
}

/// Configuration for opening a [`StorageBase`].
#[derive(Clone)]
pub(crate) struct StorageBaseOptions {
    /// Object-store credentials/config (e.g. S3), forwarded to Lance.
    pub storage_options: Option<HashMap<String, String>>,
    /// Stable identity of the writing server instance. Writes go to the MemWAL
    /// shard derived from this id, so each instance owns exactly one shard and
    /// no two instances ever contend for the same shard. `None` falls back to a
    /// single fixed `"default"` shard.
    pub shard_id: Option<String>,
    /// Count-triggered self-merge threshold; `None`/`0` disables it.
    pub merge_after_generations: Option<usize>,
    /// Shared, capacity-bounded Lance session. `None` preserves Lance's
    /// per-open default (a fresh 6 GiB index + 1 GiB metadata session *per
    /// store*, which is the source of unbounded per-append RSS growth).
    pub session: Option<Arc<Session>>,
    /// Schema used to create the dataset when it does not exist.
    pub schema: Arc<Schema>,
    /// Merge/primary key column. Used as the LSM dedup key and as the column
    /// the scalar index is built on. Must exist in `schema`.
    pub key_column: String,
    /// Latest schema this store expects. When set, a WAL merge first evolves an
    /// older base table to it by adding missing nullable columns as all-nulls
    /// (see [`StorageBase::ensure_latest_schema`]). `None` disables evolution.
    pub latest_schema: Option<Arc<Schema>>,
    /// Whether [`StorageBase::put`] seals the memtable before returning, making
    /// the rows immediately readable.
    ///
    /// This is the one deliberate behavioral difference between the stores, and
    /// it is a **visibility/throughput trade, not a durability one** — a `put`
    /// is durable either way.
    ///
    /// - `false` (rollout): `put` is a durable WAL append only. Concurrent
    ///   appends are not serialized behind a per-append seal, but there is **no
    ///   read-your-write guarantee** — visibility is bounded by whatever drives
    ///   [`StorageBase::flush`] (the server's flush sweeper, 30s by default), by
    ///   Lance's own memtable-size thresholds, or by an explicit per-request
    ///   flush.
    /// - `true` (context): `put` seals before returning, so a subsequent read
    ///   sees the rows. Required by any store whose writes read the table back —
    ///   `ContextStore`'s id/external-id uniqueness validation, upsert, and
    ///   tombstones all depend on it.
    ///
    /// Sealing per write produces a flushed generation per call, which is
    /// exactly the read amplification the high-fan-in rollout path avoids.
    pub seal_on_put: bool,
}

/// Schema-agnostic Lance storage: dataset handle, MemWAL write path, WAL merge,
/// compaction, indexing, and LSM reads. See the module docs.
pub(crate) struct StorageBase {
    /// The base table. `pub(crate)` because concrete stores build their own
    /// schema-specific scans and projections directly against it.
    pub dataset: Dataset,
    /// MemWAL shard this instance writes to (derived from `shard_id`).
    pub write_shard: Uuid,
    /// Object-store options, retained so a self-merge can re-append flushed
    /// generation data into the base table with the same credentials.
    pub storage_options: Option<HashMap<String, String>>,
    /// Shared Lance session used for the base dataset and every reload
    /// (compact, index, generation reads), so all resident stores share one
    /// capacity-bounded cache.
    pub session: Option<Arc<Session>>,
    /// Merge/primary key column; the LSM dedup key and indexed column.
    pub key_column: String,
    /// Latest expected schema for merge-time evolution; see the option.
    latest_schema: Option<Arc<Schema>>,
    /// Whether `put` seals before returning; see [`StorageBaseOptions::seal_on_put`].
    seal_on_put: bool,
    /// Self-merge threshold; `0` disables it.
    merge_after_generations: usize,
    /// Timestamp of the last successful [`Self::compact`] on this handle.
    last_compaction: Option<DateTime<Utc>>,
    /// Number of successful compactions performed by this handle.
    total_compactions: u64,
    /// Error message from the most recent failed compaction on this handle.
    last_compaction_error: Option<String>,
    /// Resident MemWAL writer for this instance's shard, wrapped for `&self`
    /// concurrent access. The [`tokio::sync::Mutex`] is held only to
    /// fetch-or-open and clone the `Arc` (see [`Self::resident_writer`]) and to
    /// invalidate a fenced writer (see [`Self::invalidate_writer`]); it is
    /// **never** held across `put`, so steady-state appends run concurrently.
    write_writer: tokio::sync::Mutex<Option<Arc<ShardWriter>>>,
}

impl StorageBase {
    /// Build a shared, capacity-bounded Lance [`Session`].
    ///
    /// Split the total cache budget across Lance's two caches:
    /// `index_cache_bytes` bounds opened-index data, `metadata_cache_bytes`
    /// bounds file/dataset metadata. Both are byte-weighted LRUs keyed by
    /// dataset URI, so one session shared across every resident store caps the
    /// process's *total* Lance cache at this budget.
    #[must_use]
    pub fn build_session(index_cache_bytes: usize, metadata_cache_bytes: usize) -> Arc<Session> {
        Arc::new(Session::new(
            index_cache_bytes,
            metadata_cache_bytes,
            Arc::default(),
        ))
    }

    /// Open the dataset at `uri`, creating it from `options.schema` when absent
    /// and `create_if_missing` is set, then initialize the MemWAL index.
    ///
    /// The MemWAL index is initialized up front (idempotent, cheap when already
    /// present) so the hot `add(&self)` path never needs `&mut self` to lazily
    /// create it.
    pub async fn open(
        uri: &str,
        options: StorageBaseOptions,
        create_if_missing: bool,
    ) -> LanceResult<Self> {
        let StorageBaseOptions {
            storage_options,
            shard_id,
            merge_after_generations,
            session,
            schema,
            key_column,
            latest_schema,
            seal_on_put,
        } = options;

        let dataset =
            match Self::load_with_options(uri, storage_options.clone(), session.clone()).await {
                Ok(dataset) => dataset,
                Err(LanceError::DatasetNotFound { .. }) if create_if_missing => {
                    Self::create_with_options(
                        uri,
                        schema.clone(),
                        storage_options.clone(),
                        session.clone(),
                    )
                    .await?
                }
                Err(err) => return Err(err),
            };

        Self::from_dataset(
            dataset,
            StorageBaseOptions {
                storage_options,
                shard_id,
                merge_after_generations,
                session,
                schema,
                key_column,
                latest_schema,
                seal_on_put,
            },
        )
        .await
    }

    /// Wrap an already-open [`Dataset`].
    ///
    /// For stores that must inspect the dataset before they can describe
    /// themselves: `ContextStore` reads the embedding width and distance metric
    /// out of the existing schema (and validates them against the caller's
    /// request) before it knows what schema to hand the base. [`Self::open`] is
    /// the simpler path when the schema is known up front.
    pub async fn from_dataset(dataset: Dataset, options: StorageBaseOptions) -> LanceResult<Self> {
        let StorageBaseOptions {
            storage_options,
            shard_id,
            merge_after_generations,
            session,
            schema,
            key_column,
            latest_schema,
            seal_on_put,
        } = options;

        if schema.field_with_name(&key_column).is_err() {
            return Err(ArrowError::SchemaError(format!(
                "key column '{key_column}' is not present in the store schema"
            ))
            .into());
        }

        let mut base = Self {
            dataset,
            write_shard: derive_shard_id(shard_id.as_deref()),
            storage_options,
            session,
            key_column,
            latest_schema,
            seal_on_put,
            merge_after_generations: merge_after_generations.unwrap_or(0),
            last_compaction: None,
            total_compactions: 0,
            last_compaction_error: None,
            write_writer: tokio::sync::Mutex::new(None),
        };
        // `ensure_mem_wal` may reload the dataset on a concurrent first-writer
        // race, which is why it must run here where we hold `&mut`.
        base.ensure_mem_wal().await?;
        Ok(base)
    }

    /// URI of the underlying Lance dataset.
    #[must_use]
    pub fn uri(&self) -> &str {
        self.dataset.uri()
    }

    /// Current base dataset manifest version.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.dataset.manifest.version
    }

    /// Check out a specific base dataset version (time travel).
    pub async fn checkout(&mut self, version_id: u64) -> LanceResult<()> {
        self.dataset = self.dataset.checkout_version(version_id).await?;
        Ok(())
    }

    /// Refresh this handle to the latest base-table manifest while retaining its
    /// session and metadata caches.
    ///
    /// Long-lived read handles call this before a new request so compaction or
    /// WAL merges committed by another process become visible without paying the
    /// cost of reopening the dataset and rebuilding all session caches.
    pub async fn refresh_latest(&mut self) -> LanceResult<()> {
        self.dataset.checkout_latest().await
    }

    // ---------------------------------------------------------------- writes

    /// Durably append `batches` through this instance's MemWAL shard.
    ///
    /// # Always durable on return; visible on return only if `seal_on_put`
    ///
    /// `put` returns once the WAL entry has been PUT to object storage. The rows
    /// are then **durable** — they survive a crash and are replayed on reopen.
    /// Whether they are also **readable** depends on
    /// [`StorageBaseOptions::seal_on_put`]:
    ///
    /// - `false` (rollout): the rows sit in the active memtable, invisible to
    ///   every reader including this one, until something seals them into a
    ///   flushed generation and commits it to the shard manifest — [`Self::flush`]
    ///   (driven by the server's flush sweeper), [`Self::close`], the merge path,
    ///   or Lance's own memtable-size thresholds. Callers get **no
    ///   read-your-write guarantee**.
    /// - `true` (context): this call seals before returning, so a subsequent
    ///   read sees the rows.
    ///
    /// Deferring the seal is what lets concurrent appends run without
    /// serializing behind one seal+drain, and it avoids emitting a flushed
    /// generation per call. Either way a single resident [`ShardWriter`] is
    /// reused, so the shard epoch is claimed once and the object-store
    /// connection is pooled, rather than paying a cold DNS resolution + TCP/TLS
    /// handshake + epoch claim per append.
    ///
    /// Retried exactly once on a fence: a fence means a merge superseded our
    /// epoch, and invalidating + reopening re-claims the current one. Rows are
    /// de-duplicated by the key column at read time, so a retried append can
    /// never double-count.
    pub async fn put(&self, batches: Vec<RecordBatch>) -> LanceResult<()> {
        if batches.is_empty() {
            return Ok(());
        }
        let started = timer_start!();
        let result = self.put_sealing(batches).await;
        // Success-path latency only. A failed append's *duration* is not
        // actionable, but its *rate* is, so errors increment a flat counter
        // rather than doubling this histogram's series count.
        match &result {
            Ok(()) => observe_duration!(
                crate::metrics::ROLLOUT_ADD_DURATION,
                timer_elapsed!(started),
            ),
            Err(_) => count!(crate::metrics::ROLLOUT_ADD_ERRORS),
        }
        result
    }

    /// [`Self::put_inner`], then seal if this store wants read-your-write.
    ///
    /// The seal is inside the timed region because for a `seal_on_put` store it
    /// is part of what the caller is waiting for; excluding it would understate
    /// the write latency such a store actually sees.
    async fn put_sealing(&self, batches: Vec<RecordBatch>) -> LanceResult<()> {
        self.put_inner(batches).await?;
        if self.seal_on_put {
            self.flush().await?;
        }
        Ok(())
    }

    async fn put_inner(&self, batches: Vec<RecordBatch>) -> LanceResult<()> {
        let writer = self.resident_writer().await?;
        match writer.put(batches.clone()).await {
            Ok(_) => Ok(()),
            Err(err) if is_fenced_error(&err) => {
                // Drop the fenced writer without close() — its epoch is already
                // dead — and reopen against the current epoch for a single retry.
                self.invalidate_writer(&writer).await;
                let writer = self.resident_writer().await?;
                writer.put(batches).await?;
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    /// Fetch the resident [`ShardWriter`], opening one on first use. The mutex is
    /// held only to read-or-open and clone the `Arc`; it is released before the
    /// caller `put`s, so steady-state appends run concurrently. Opening under the
    /// lock serializes only the (rare) first open / post-fence reopen.
    async fn resident_writer(&self) -> LanceResult<Arc<ShardWriter>> {
        let mut guard = self.write_writer.lock().await;
        if let Some(writer) = guard.as_ref() {
            return Ok(writer.clone());
        }
        let config = ShardWriterConfig {
            shard_id: self.write_shard,
            ..Default::default()
        };
        let writer = Arc::new(
            self.dataset
                .mem_wal_writer(self.write_shard, config)
                .await?,
        );
        *guard = Some(writer.clone());
        Ok(writer)
    }

    /// Invalidate a fenced writer so the next [`Self::resident_writer`] reopens.
    /// Identity-checked: only clears the slot if it still holds `stale`, so when
    /// N concurrent appends all fence on the same writer, the first
    /// clears+reopens and the rest observe a fresh writer and reuse it — exactly
    /// one reopen, and no append installs a writer another append just
    /// invalidated.
    async fn invalidate_writer(&self, stale: &Arc<ShardWriter>) {
        let mut guard = self.write_writer.lock().await;
        if guard.as_ref().is_some_and(|cur| Arc::ptr_eq(cur, stale)) {
            *guard = None;
        }
    }

    /// Materialize the active memtable into a flushed, queryable generation:
    /// `force_seal_active` freezes it and hands it to the background flusher,
    /// then `wait_for_flush_drain` blocks until every frozen memtable has landed
    /// in the shard manifest (and is therefore visible to reads on every
    /// instance).
    ///
    /// This is the visibility half of the write path, decoupled from the durable
    /// append in [`Self::put`]. A no-op when no writer is resident.
    pub async fn flush(&self) -> LanceResult<()> {
        let started = timer_start!();
        let result = self.flush_inner().await;
        // `outcome` is worth its cardinality: the three paths differ by orders of
        // magnitude, and `noop` is by far the most common, so a blended histogram
        // would be dominated by near-zero samples. Failures are counted, not timed.
        match &result {
            Ok(outcome) => {
                let label = match outcome {
                    FlushOutcome::Sealed => "sealed",
                    FlushOutcome::Noop => "noop",
                    FlushOutcome::Fenced => "fenced",
                };
                observe_duration!(
                    crate::metrics::ROLLOUT_FLUSH_DURATION,
                    timer_elapsed!(started),
                    "outcome" => label,
                );
            }
            Err(_) => count!(crate::metrics::ROLLOUT_FLUSH_ERRORS),
        }
        result.map(|_| ())
    }

    async fn flush_inner(&self) -> LanceResult<FlushOutcome> {
        let writer = {
            let guard = self.write_writer.lock().await;
            guard.as_ref().cloned()
        };
        let Some(writer) = writer else {
            return Ok(FlushOutcome::Noop);
        };
        match writer.force_seal_active().await {
            Ok(()) => {}
            // A fence means a merge superseded our epoch; the next `put` reopens
            // and replays. Nothing to flush against the dead epoch.
            Err(err) if is_fenced_error(&err) => {
                self.invalidate_writer(&writer).await;
                return Ok(FlushOutcome::Fenced);
            }
            Err(err) => return Err(err),
        }
        writer.wait_for_flush_drain().await?;
        Ok(FlushOutcome::Sealed)
    }

    /// Gracefully close the resident writer, draining its background tasks.
    ///
    /// [`ShardWriter`] has no `Drop`, so its background tasks are only reclaimed
    /// by an explicit `close().await`. Call this before dropping a store on a
    /// path that can `await` (e.g. an LRU eviction that owns the last handle).
    /// Idempotent: a no-op when no writer is resident.
    pub async fn close(&mut self) -> LanceResult<()> {
        // `&mut self` gives exclusive access, so `get_mut` avoids an async lock.
        if let Some(writer) = self.write_writer.get_mut().take() {
            match Arc::try_unwrap(writer) {
                // Sole owner: drain the writer's background tasks gracefully.
                Ok(writer) => writer.close().await?,
                // Another handle (an in-flight append that cloned the Arc) still
                // holds it; it will be dropped when that append completes.
                Err(_shared) => {}
            }
        }
        Ok(())
    }

    /// Rows buffered in this instance's resident writer that have not yet been
    /// sealed into a flushed generation.
    ///
    /// Best-effort and non-failing: no resident writer, a WAL-only writer with
    /// no memtable, or a fenced writer all report `0` rather than erroring, so
    /// an observability read never fails because of writer state.
    pub async fn unflushed_rows(&self) -> i64 {
        let writer = {
            let guard = self.write_writer.lock().await;
            guard.as_ref().cloned()
        };
        let Some(writer) = writer else {
            return 0;
        };
        writer
            .memtable_stats()
            .await
            .map(|stats| stats.row_count as i64)
            .unwrap_or(0)
    }

    // ----------------------------------------------------------- WAL merging

    /// Merge this instance's flushed generations into the base table **if** the
    /// shard has accumulated at least `merge_after_generations` of them (the
    /// count trigger; `0` disables it). No-op otherwise.
    pub async fn maybe_merge_own_shard(&mut self) -> LanceResult<usize> {
        if self.merge_after_generations == 0 {
            return Ok(0);
        }
        self.merge_own_shard_if_ready(self.merge_after_generations)
            .await
    }

    /// Run one periodic WAL-cleanup pass over this instance's own shard: fold
    /// **every** flushed generation into the base table. This is the *time* half
    /// of the "time OR count" trigger, so it is deliberately *not* gated by any
    /// generation-count threshold: once the interval elapses, whatever is
    /// pending gets merged even if the count trigger never fired. Returns the
    /// number of generations reclaimed.
    ///
    /// # Seals first
    ///
    /// This flushes the active memtable before looking at the manifest. Without
    /// that, a deployment with the periodic flush sweeper disabled could never
    /// make progress: nothing would seal the memtable, so `flushed_generations`
    /// would stay empty, so the threshold check would return `0` and never reach
    /// the merge — leaving rows durable but permanently invisible until a
    /// process restart replayed the WAL.
    pub async fn cleanup_own_shard(&mut self) -> LanceResult<usize> {
        self.flush().await?;
        // Threshold `1`: merge whenever at least one generation is pending. The
        // time trigger must not depend on the count threshold — that is what
        // makes the two triggers a true OR.
        self.merge_own_shard_if_ready(1).await
    }

    async fn merge_own_shard_if_ready(&mut self, threshold: usize) -> LanceResult<usize> {
        let Some((manifest_store, manifest, prepared)) =
            self.prepare_merge_if_ready(threshold).await?
        else {
            return Ok(0);
        };
        let pending = prepared.generation_count();
        self.commit_merge(&manifest_store, &manifest, prepared)
            .await?;
        Ok(pending)
    }

    /// The shared-lock half of a merge: decide whether one is due and read the
    /// flushed generations into memory.
    ///
    /// Takes `&self`, so a caller holding a *read* lock can run the expensive
    /// part while appends continue, then take the write lock only to hand the
    /// result to [`Self::commit_prepared_merge`]. Returns `None` when nothing is
    /// due.
    ///
    /// ```ignore
    /// let prepared = { store.read().await.prepare_merge_if_ready(1).await? };
    /// if let Some((manifest_store, manifest, prepared)) = prepared {
    ///     store.write().await.commit_prepared_merge(&manifest_store, &manifest, prepared).await?;
    /// }
    /// ```
    pub async fn prepare_merge_if_ready(
        &self,
        threshold: usize,
    ) -> LanceResult<Option<(ShardManifestStore, ShardManifest, PreparedMerge)>> {
        self.prepare_merge_if_ready_inner(threshold, false).await
    }

    /// [`Self::prepare_merge_if_ready`], but seals the active memtable *before*
    /// consulting the manifest — the time-triggered (`threshold = 1`) behavior
    /// of [`Self::cleanup_own_shard`]. See that method for why the ordering is
    /// load-bearing.
    pub async fn prepare_cleanup_merge(
        &self,
    ) -> LanceResult<Option<(ShardManifestStore, ShardManifest, PreparedMerge)>> {
        self.prepare_merge_if_ready_inner(1, true).await
    }

    async fn prepare_merge_if_ready_inner(
        &self,
        threshold: usize,
        seal_first: bool,
    ) -> LanceResult<Option<(ShardManifestStore, ShardManifest, PreparedMerge)>> {
        if seal_first {
            // Materialize anything buffered so it is eligible for this pass.
            self.flush().await?;
        }
        let object_store = self.dataset.object_store(None).await?;
        let branch_location = self.dataset.branch_location();
        let manifest_store = ShardManifestStore::new(
            object_store,
            &branch_location.path,
            self.write_shard,
            DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
        );
        let Some(manifest) = manifest_store.read_latest().await? else {
            return Ok(None);
        };
        let pending = manifest.flushed_generations.len();
        if pending == 0 || pending < threshold.max(1) {
            return Ok(None);
        }
        let Some(prepared) = self.prepare_merge(&manifest).await? else {
            return Ok(None);
        };
        Ok(Some((manifest_store, manifest, prepared)))
    }

    /// Commit a merge prepared by [`Self::prepare_merge_if_ready`].
    pub async fn commit_prepared_merge(
        &mut self,
        manifest_store: &ShardManifestStore,
        manifest: &ShardManifest,
        prepared: PreparedMerge,
    ) -> LanceResult<usize> {
        let pending = prepared.generation_count();
        self.commit_merge(manifest_store, manifest, prepared)
            .await?;
        Ok(pending)
    }

    /// The `&self` half of a merge: everything that can run while appends
    /// continue — sealing the memtable and reading every flushed generation
    /// into memory.
    ///
    /// # Concurrency: the expensive phase does not need exclusive access
    ///
    /// A merge only ever touches *sealed* generations — history — while a `put`
    /// writes the active memtable at the WAL tail. They operate on disjoint
    /// data, which is the whole point of an LSM, so a merge must not stop the
    /// write path. Notably the merge does **not** `claim_epoch`: the epoch is an
    /// *ownership* token, not a per-commit token, and
    /// [`ShardManifestStore::commit_update`] only rejects a writer whose epoch is
    /// **older** than the stored one. Reusing the shard's current epoch commits
    /// the drain and leaves the live writer untouched.
    async fn prepare_merge(&self, manifest: &ShardManifest) -> LanceResult<Option<PreparedMerge>> {
        if manifest.flushed_generations.is_empty() {
            return Ok(None);
        }

        // Seal the active memtable so its rows are in a generation rather than
        // buffered. Uses `&self` (the writer lives behind a Mutex) and does not
        // close the writer, so appends continue throughout.
        observe_phase!("seal", self.flush().await)?;

        // The expensive phase: pull every flushed generation out of object
        // storage. Buffered in memory, so this is the part that must not hold an
        // exclusive lock.
        let (merged_generations, merged_paths, batches, merge_schema) =
            observe_phase!("read", self.read_flushed_generations(manifest).await)?;

        Ok(Some(PreparedMerge {
            merged_generations,
            merged_paths,
            batches,
            merge_schema,
        }))
    }

    /// The `&mut self` half of a merge: append the prepared rows to the base
    /// table, drain the merged generations from the manifest, and delete their
    /// directories.
    ///
    /// # Surgical drain, not blanket clear
    ///
    /// The drain removes only the generation ids this call actually merged,
    /// retaining anything else present. This is load-bearing now that a
    /// concurrent flush can seal a new generation mid-merge: `commit_update`
    /// re-runs the closure against a freshly-read manifest on every CAS retry,
    /// so a *relative* edit (retain-not-in-set) composes with a concurrent
    /// flush's append, while an absolute `flushed_generations = []` would
    /// silently discard a generation that was never merged — data loss. For the
    /// same reason the closure must preserve `current_generation`,
    /// `replay_after_wal_entry_position` and `wal_entry_position_last_seen`,
    /// which a concurrent flush advances; `..current.clone()` carries them.
    ///
    /// Rows are de-duplicated by the key column at read time, so even if a crash
    /// interrupts the sequence (data appended to base but manifest not yet
    /// drained), a subsequent read simply sees the rows via both the base table
    /// and the still-listed generation and de-dups them — no double counting.
    /// The next merge attempt then drains the manifest.
    async fn commit_merge(
        &mut self,
        manifest_store: &ShardManifestStore,
        manifest: &ShardManifest,
        prepared: PreparedMerge,
    ) -> LanceResult<()> {
        let PreparedMerge {
            merged_generations,
            merged_paths,
            batches,
            merge_schema,
        } = prepared;

        self.ensure_latest_schema().await?;

        if !batches.is_empty() {
            observe_phase!(
                "append",
                self.append_merged_batches(batches, merge_schema).await
            )?;
        }

        // Reuse the shard's *current* epoch rather than claiming a new one:
        // claiming would fence our own live writer. `commit_update` still fails
        // cleanly if a genuinely new writer has claimed the shard meanwhile —
        // its epoch would exceed ours — which is the correct outcome, since that
        // writer now owns the shard.
        let epoch = manifest.writer_epoch;

        observe_phase!(
            "drain",
            manifest_store
                .commit_update(epoch, |current| ShardManifest {
                    version: current.version + 1,
                    // Relative edit: retain everything we did not merge. Must
                    // never become an absolute assignment — see the doc comment.
                    flushed_generations: current
                        .flushed_generations
                        .iter()
                        .filter(|fg| !merged_generations.contains(&fg.generation))
                        .cloned()
                        .collect(),
                    ..current.clone()
                })
                .await
        )?;

        self.delete_merged_generation_dirs(&merged_paths).await
    }

    /// Delete the merged generations' directories now that no manifest
    /// references them.
    ///
    /// Ordering matters: the drain already removed these ids from
    /// `flushed_generations`, so a reader can no longer resolve them — deleting
    /// the data second (never before) keeps the sequence crash-safe.
    ///
    /// Best-effort: a delete failure must NOT fail the merge — the merge has
    /// logically succeeded (data appended, manifest drained). A failed delete
    /// only leaks one directory.
    async fn delete_merged_generation_dirs(&self, merged_paths: &[String]) -> LanceResult<()> {
        let phase = timer_start!();
        let object_store = self.dataset.object_store(None).await?;
        let branch_path = self.dataset.branch_location().path.clone();
        for path in merged_paths {
            let gen_dir = branch_path
                .clone()
                .join("_mem_wal")
                .join(self.write_shard.to_string().as_str())
                .join(path.as_str());
            if let Err(err) = object_store.remove_dir_all(gen_dir.clone()).await {
                warn!(
                    shard = %self.write_shard,
                    generation_path = %path,
                    error = %err,
                    "failed to delete merged MemWAL generation directory; \
                     it will remain until reclaimed"
                );
            }
        }
        // Best-effort by contract: a delete failure is logged above and does not
        // fail the merge, so there is no error counter for this phase.
        observe_duration!(
            crate::metrics::ROLLOUT_WAL_MERGE_DURATION,
            timer_elapsed!(phase),
            "phase" => "delete",
        );
        Ok(())
    }

    /// Read every flushed generation listed in `manifest` into memory, aligned
    /// to the base table's current schema.
    ///
    /// Returns the merged generation ids, their on-storage folder names (needed
    /// to delete the directories after the manifest drain), the batches, and the
    /// schema they were aligned to.
    #[allow(clippy::type_complexity)]
    async fn read_flushed_generations(
        &self,
        manifest: &ShardManifest,
    ) -> LanceResult<(HashSet<u64>, Vec<String>, Vec<RecordBatch>, Arc<Schema>)> {
        let base_uri = self.dataset.uri().trim_end_matches('/').to_string();
        let mut merged_generations: HashSet<u64> = HashSet::new();
        let mut merged_paths: Vec<String> = Vec::new();
        let mut batches: Vec<RecordBatch> = Vec::new();
        let merge_schema: Arc<Schema> = Arc::new(self.dataset.schema().into());
        for flushed in &manifest.flushed_generations {
            let gen_uri = format!(
                "{}/_mem_wal/{}/{}",
                base_uri, self.write_shard, flushed.path
            );
            let gen_dataset = Self::load_with_options(
                &gen_uri,
                self.storage_options.clone(),
                self.session.clone(),
            )
            .await?;
            let mut stream = gen_dataset.scan().try_into_stream().await?;
            while let Some(batch) = stream.try_next().await? {
                if batch.num_rows() > 0 {
                    batches.push(align_batch_to_schema(batch, merge_schema.clone())?);
                }
            }
            merged_generations.insert(flushed.generation);
            merged_paths.push(flushed.path.clone());
        }
        Ok((merged_generations, merged_paths, batches, merge_schema))
    }

    /// Append merged WAL rows into the base table with this store's credentials.
    async fn append_merged_batches(
        &mut self,
        batches: Vec<RecordBatch>,
        merge_schema: Arc<Schema>,
    ) -> LanceResult<()> {
        let reader = RecordBatchIterator::new(
            batches.into_iter().map(Ok::<RecordBatch, ArrowError>),
            merge_schema,
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
        Ok(())
    }

    /// Evolve an older base table to the store's latest additive schema.
    ///
    /// Missing nullable columns are added as all-null arrays. Existing unknown
    /// columns, type changes, and missing required columns remain hard errors.
    /// A no-op when the store declared no `latest_schema`.
    pub async fn ensure_latest_schema(&mut self) -> LanceResult<()> {
        let Some(latest_schema) = self.latest_schema.clone() else {
            return Ok(());
        };
        self.dataset.checkout_latest().await?;

        let base_schema: Arc<Schema> = Arc::new(self.dataset.schema().into());
        align_batch_to_schema(
            RecordBatch::new_empty(base_schema.clone()),
            latest_schema.clone(),
        )?;

        let missing_fields = latest_schema
            .fields()
            .iter()
            .filter(|field| base_schema.field_with_name(field.name()).is_err())
            .cloned()
            .collect::<Vec<_>>();
        if !missing_fields.is_empty() {
            self.dataset
                .add_columns(
                    NewColumnTransform::AllNulls(Arc::new(Schema::new(missing_fields))),
                    None,
                    None,
                )
                .await?;
        }
        Ok(())
    }

    // ------------------------------------------------- compaction & indexing

    /// Compact the base table's small fragments into larger ones.
    ///
    /// Every WAL merge `append`s a new fragment to the base table, so a
    /// long-running store accumulates many small fragments that slow scans.
    ///
    /// # Distributed use: run from ONE compactor, not every worker
    ///
    /// Unlike WAL merge — where each worker touches only its own shard and can
    /// never contend — compaction rewrites the *shared* base table. Lance treats
    /// two concurrent `Rewrite` commits as a retryable conflict, so N workers
    /// each compacting the same table degenerates into a thundering herd of
    /// wasted rewrites. Drive this from a *single* external trigger. It is safe
    /// to call while other workers are appending or WAL-merging: `Append` vs
    /// `Rewrite` is non-conflicting in Lance's matrix.
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
            // Every base table here carries a MemWAL index, which is fieldless
            // (it tracks shard/generation bookkeeping, not a data column).
            // Lance's inline index remap panics on a fieldless index ("An index
            // existed with no fields"), so defer remapping: compaction records a
            // fragment-reuse index and remaps lazily instead of touching the
            // MemWAL index during the rewrite.
            defer_index_remap: true,
            ..Default::default()
        };

        match compact_files(&mut self.dataset, lance_options, None).await {
            Ok(metrics) => {
                // Reload the handle so the caller (and subsequent reads on this
                // instance) observe the compacted version.
                self.reload().await?;
                self.last_compaction = Some(Utc::now());
                self.total_compactions += 1;
                self.last_compaction_error = None;
                info!(
                    fragments_removed = metrics.fragments_removed,
                    fragments_added = metrics.fragments_added,
                    "base-table compaction completed"
                );
                Ok(metrics)
            }
            Err(e) => {
                warn!(error = %e, "base-table compaction failed");
                self.last_compaction_error = Some(e.to_string());
                Err(e)
            }
        }
    }

    /// Build a ZoneMap scalar index on the base table's key column.
    ///
    /// The key column is the table's (unenforced) primary key, so a lightweight
    /// per-fragment min/max index accelerates point lookups and range scans on
    /// the already-flushed base table. `replace(true)` makes this idempotent.
    ///
    /// # MemWAL interaction
    ///
    /// The base table carries a fieldless MemWAL index, and Lance's MemWAL does
    /// not *maintain* ZoneMap indices across WAL flushes (it only keeps the
    /// indices named in `maintained_indexes`). That does not affect correctness:
    /// rows are de-duplicated by the key column at read time, so the ZoneMap
    /// only ever needs to describe the base table's already-merged fragments —
    /// rows still living in unmerged WAL generations are found by the normal
    /// scan of those generations.
    pub async fn create_key_zonemap_index(&mut self) -> LanceResult<()> {
        info!(column = %self.key_column, "creating ZoneMap index on key column");
        self.dataset
            .create_index_builder(
                &[self.key_column.as_str()],
                IndexType::ZoneMap,
                &ScalarIndexParams::default(),
            )
            .name(ID_INDEX_NAME.to_string())
            .replace(true)
            .await?;
        // Reload the handle so subsequent reads on this instance observe the new
        // index (mirrors the reload done after `compact`).
        self.reload().await
    }

    /// Whether the base table has accumulated at least `min_fragments`
    /// fragments (and is thus worth compacting). Quiet-hours gating from
    /// [`CompactionConfig`] is honored so an external scheduler can pass the
    /// same config it would pass to [`Self::compact`].
    #[must_use]
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

    /// Current compaction statistics for the base table.
    ///
    /// `is_compacting` is always `false`: compaction runs synchronously under
    /// the caller's `&mut self`, so a stats read cannot observe an in-flight
    /// compaction on this handle.
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

    /// Reload the base dataset handle through [`Self::load_with_options`], so
    /// the shared session and storage options are never dropped.
    pub async fn reload(&mut self) -> LanceResult<()> {
        let uri = self.dataset.uri().to_string();
        self.dataset =
            Self::load_with_options(&uri, self.storage_options.clone(), self.session.clone())
                .await?;
        Ok(())
    }

    // -------------------------------------------------------- MemWAL plumbing

    /// Initialize the (unsharded) MemWAL index on first write, exactly once.
    ///
    /// # Concurrent first-writers
    ///
    /// `initialize_mem_wal` commits a `CreateIndex` transaction, and Lance
    /// treats two concurrent `CreateIndex` commits as a hard conflict: when two
    /// instances take their very first write at the same time, both observe no
    /// index, both try to create it, and the loser gets
    /// `RetryableCommitConflict`. That is benign here — the winner created
    /// exactly the index we wanted — so we reload and treat "index now present"
    /// as success. Any other error propagates.
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
                self.reload().await?;
                if self.mem_wal_index_present().await? {
                    Ok(())
                } else {
                    Err(err)
                }
            }
        }
    }

    async fn mem_wal_index_present(&self) -> LanceResult<bool> {
        let indices = self.dataset.load_indices().await?;
        Ok(indices.iter().any(|i| i.name == MEM_WAL_INDEX_NAME))
    }

    /// Absolute URI of one flushed generation's dataset.
    #[must_use]
    pub fn flushed_generation_uri(&self, shard_id: Uuid, path: &str) -> String {
        format!(
            "{}/_mem_wal/{shard_id}/{path}",
            self.dataset.uri().trim_end_matches('/')
        )
    }

    /// Open a flushed generation dataset, inheriting the base dataset's session
    /// and this store's storage options.
    pub async fn open_flushed_dataset(&self, uri: &str) -> LanceResult<Dataset> {
        let mut builder = DatasetBuilder::from_uri(uri).with_session(self.dataset.session());
        if let Some(options) = self.storage_options.clone() {
            builder = builder.with_storage_options(options);
        }
        builder.load().await
    }

    /// Read the latest manifest for every MemWAL shard. Manifest reads are
    /// bounded-concurrent so stores with many writer instances do not pay one
    /// object-store round trip per shard serially.
    pub async fn wal_shard_snapshots(&self) -> LanceResult<Vec<ShardSnapshot>> {
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

    /// Number of flushed MemWAL generations pending merge into the base table
    /// across all shards. Read-only: unlike [`Self::cleanup_own_shard`] it never
    /// merges.
    pub async fn pending_wal_generations(&self) -> LanceResult<usize> {
        Ok(self
            .wal_shard_snapshots()
            .await?
            .iter()
            .map(|snapshot| snapshot.flushed_generations.len())
            .sum())
    }

    /// Count rows in all immutable flushed-generation datasets using metadata
    /// reads rather than a payload scan.
    pub async fn pending_wal_rows(&self, snapshots: &[ShardSnapshot]) -> LanceResult<u64> {
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

    // ------------------------------------------------------------ LSM reads

    /// Build an LSM scanner over the base table unioned with every shard's
    /// flushed MemWAL generations, discovered from object storage. Because the
    /// snapshot is rebuilt from shard manifests on each call, one instance sees
    /// every other instance's flushed appends — reads are not pinned to the
    /// writing instance. Deduplicates by the key column.
    pub async fn lsm_scanner(&self) -> LanceResult<LsmScanner> {
        let shard_snapshots = self.wal_shard_snapshots().await?;
        Ok(self.lsm_scanner_for_source(ListSource::All, shard_snapshots))
    }

    /// Build a paginating scanner for the requested [`ListSource`],
    /// deduplicating by the key column:
    /// - `Fragments`: base table only (`shard_snapshots` is ignored — callers
    ///   pass an empty vec so no manifest reads happen);
    /// - `All`: base table ∪ the flushed generations in `shard_snapshots`;
    /// - `Wal`: only the flushed generations, resolving relative generation
    ///   paths against the dataset root.
    pub fn lsm_scanner_for_source(
        &self,
        source: ListSource,
        shard_snapshots: Vec<ShardSnapshot>,
    ) -> LsmScanner {
        let merge_key = vec![self.key_column.clone()];
        match source {
            ListSource::Fragments => {
                LsmScanner::new(Arc::new(self.dataset.clone()), Vec::new(), merge_key)
            }
            ListSource::All => {
                LsmScanner::new(Arc::new(self.dataset.clone()), shard_snapshots, merge_key)
            }
            ListSource::Wal => {
                let arrow_schema: Schema = self.dataset.schema().into();
                LsmScanner::without_base_table(
                    Arc::new(arrow_schema),
                    self.dataset.uri().trim_end_matches('/').to_string(),
                    shard_snapshots,
                    merge_key,
                )
                .with_session(self.dataset.session())
            }
        }
    }

    // ----------------------------------------------------------- open/create

    /// Open a Lance dataset with this store's storage options and session.
    ///
    /// Every open in the storage layer goes through here. A plain
    /// `Dataset::open` builds a fresh per-store session (Lance's 6 GiB index +
    /// 1 GiB metadata default) and silently drops storage options, which breaks
    /// on credentialed object stores.
    pub async fn load_with_options(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
    ) -> LanceResult<Dataset> {
        let mut builder = DatasetBuilder::from_uri(uri);
        if let Some(options) = storage_options {
            builder = builder.with_storage_options(options);
        }
        if let Some(session) = session {
            builder = builder.with_session(session);
        }
        builder.load().await
    }

    /// Create an empty Lance dataset with `schema` at `uri`.
    pub async fn create_with_options(
        uri: &str,
        schema: Arc<Schema>,
        storage_options: Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
    ) -> LanceResult<Dataset> {
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let batches = RecordBatchIterator::new(
            vec![Ok::<RecordBatch, ArrowError>(empty_batch)].into_iter(),
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
        params.session = session;

        Dataset::write(batches, uri, Some(params)).await
    }
}

impl Drop for StorageBase {
    /// Best-effort drain of a still-resident writer's background tasks.
    ///
    /// [`ShardWriter`] has no `Drop`, so dropping it without `close().await`
    /// leaks its background tasks. The graceful path is [`Self::close`], but a
    /// store can also be dropped without an `await` (e.g. LRU eviction). When a
    /// Tokio runtime is available we move the writer into a detached task that
    /// closes it; otherwise we can only drop it.
    ///
    /// `ShardWriter::close` seals the active memtable, so this path is also what
    /// keeps an evicted-but-unflushed store's rows from being stranded. Each way
    /// it can fail to do that is logged rather than swallowed: a stranded
    /// memtable leaves rows durable in the WAL but invisible to reads until the
    /// next process restart replays it, which is near-impossible to diagnose
    /// without a signal here.
    fn drop(&mut self) {
        // `&mut self` in drop → exclusive access, so `get_mut` avoids a lock.
        if let Some(writer) = self.write_writer.get_mut().take() {
            let shard = self.write_shard;
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    // Only the sole owner can close; if an append still shares
                    // the Arc it will drop last. Best-effort either way.
                    match Arc::try_unwrap(writer) {
                        Ok(writer) => {
                            if let Err(err) = writer.close().await {
                                warn!(
                                    shard = %shard,
                                    error = %err,
                                    "detached close of a dropped writer failed; unflushed \
                                     rows stay durable in the WAL but remain invisible \
                                     until the next replay"
                                );
                            }
                        }
                        Err(_shared) => {
                            tracing::debug!(
                                shard = %shard,
                                "dropped writer is still shared by an in-flight append; \
                                 close deferred to the last Arc holder"
                            );
                        }
                    }
                });
            } else {
                warn!(
                    shard = %shard,
                    "writer dropped with no Tokio runtime available; its background tasks \
                     leak and unflushed rows remain invisible until the next WAL replay"
                );
            }
        }
    }
}

/// Whether a Lance error is a MemWAL writer-fence error — i.e. this writer's
/// shard epoch was superseded by a later claimer. Matched on the error text
/// because Lance surfaces the fence as a generic error rather than a dedicated
/// variant (see `ShardWriter::check_fenced`, which reports "Writer fenced").
pub(crate) fn is_fenced_error(err: &LanceError) -> bool {
    let text = err.to_string();
    text.contains("fenced") || text.contains("Fenced")
}

/// Whether a Lance error means the target dataset/path does not exist.
///
/// Used by WAL read fallbacks to skip a flushed generation that a concurrent
/// merge drained from the shard manifest and deleted between the manifest
/// snapshot and the open — the row's data is already in the base table, so
/// skipping is safe and avoids a transient failure. Both the structured
/// not-found variants and the object-store "not found" surface are matched; the
/// latter is text-based because it arrives as a generic wrapped IO error.
pub(crate) fn is_not_found_error(err: &LanceError) -> bool {
    if matches!(
        err,
        LanceError::DatasetNotFound { .. } | LanceError::NotFound { .. }
    ) {
        return true;
    }
    let text = err.to_string();
    text.contains("was not found") || text.contains("not found") || text.contains("NotFound")
}

/// Align a flushed-generation batch with the base table schema before append.
///
/// Nullable columns added after the generation was written are materialized as
/// null arrays. Missing required columns, type changes, and generation-only
/// columns remain hard errors so a merge cannot silently corrupt or discard
/// data.
pub(crate) fn align_batch_to_schema(
    batch: RecordBatch,
    target_schema: Arc<Schema>,
) -> LanceResult<RecordBatch> {
    let source_schema = batch.schema();

    for source_field in source_schema.fields() {
        if target_schema.field_with_name(source_field.name()).is_err() {
            return Err(ArrowError::SchemaError(format!(
                "WAL generation column '{}' does not exist in the base table schema",
                source_field.name()
            ))
            .into());
        }
    }

    let columns = target_schema
        .fields()
        .iter()
        .map(
            |target_field| match source_schema.column_with_name(target_field.name()) {
                Some((index, source_field)) => {
                    if source_field.data_type() != target_field.data_type() {
                        return Err(ArrowError::SchemaError(format!(
                            "WAL generation column '{}' has type {}, expected {}",
                            target_field.name(),
                            source_field.data_type(),
                            target_field.data_type()
                        ))
                        .into());
                    }
                    Ok(batch.column(index).clone())
                }
                None if target_field.is_nullable() => {
                    Ok(new_null_array(target_field.data_type(), batch.num_rows()))
                }
                None => Err(ArrowError::SchemaError(format!(
                    "required base table column '{}' is missing from the WAL generation",
                    target_field.name()
                ))
                .into()),
            },
        )
        .collect::<LanceResult<Vec<_>>>()?;

    Ok(RecordBatch::try_new(target_schema, columns)?)
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
