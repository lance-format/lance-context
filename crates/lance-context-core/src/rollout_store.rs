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
//! instances. See `docs/src/specs/rollout-deployment.md`.
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

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float32Builder, Int32Builder, Int64Builder, Int8Builder, LargeBinaryBuilder,
    LargeStringBuilder, ListBuilder, StringBuilder, StringDictionaryBuilder,
    TimestampMicrosecondBuilder,
};
use arrow_array::types::Int8Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, Float32Array, Int32Array, Int64Array,
    Int8Array, LargeBinaryArray, LargeStringArray, ListArray, RecordBatch, StringArray,
    TimestampMicrosecondArray, UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema, TimeUnit};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use datafusion::sql::parser::{DFParser, Statement as DFStatement};
use datafusion::sql::sqlparser::ast::Statement as SqlStatement;
use futures::{stream, StreamExt, TryStreamExt};
use lance::dataset::mem_wal::{LsmScanner, ShardManifestStore, ShardSnapshot};
use lance::dataset::optimize::CompactionMetrics;
use lance::dataset::Dataset;
use lance::session::Session;
use lance::{Error as LanceError, Result as LanceResult};
use lance_index::mem_wal::ShardManifest;
use serde_json::Value;
use uuid::Uuid;

use crate::rollout::RolloutRecord;
use crate::store::{
    column_as, column_as_optional, relationship_field, relationship_list_item_field,
    relationship_struct_builder, relationships_from_list, timestamp_from_micros, CompactionConfig,
    CompactionStats, RELATIONSHIPS_COLUMN,
};
use crate::store_base::{
    is_not_found_error, StorageBase, StorageBaseOptions, DEFAULT_OBSERVE_CONCURRENCY,
};

// `ListSource` and `PreparedMerge` are schema-agnostic and now live in
// [`crate::store_base`]; re-exported here so the public rollout API is
// unchanged.
#[allow(unused_imports)]
pub use crate::store_base::{derive_shard_id, ListSource, PreparedMerge};

const CLAIM_CHECK_COLUMNS: [&str; 5] = [
    "model_input_string",
    "model_output_string",
    "rationale",
    "problem_text",
    "user_metadata",
];

/// Read-only observability snapshot of a rollout store.
///
/// Produced by [`RolloutStore::observe`] from base-table and MemWAL metadata.
/// Consumed by the control-plane stats scanner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolloutObservation {
    /// Logical row count across the base table and every flushed MemWAL shard.
    ///
    /// **Excludes rows that are durable but not yet flushed.** An `add` returns
    /// once the WAL entry is persisted; the row only reaches a flushed
    /// generation when the memtable is sealed. In steady state this count
    /// therefore lags recent writes by up to the flush interval. Use
    /// [`Self::unflushed_rows`] to see the gap, or `row_count +
    /// unflushed_rows` for every row this instance has durably accepted.
    pub row_count: i64,
    /// Number of fragments in the base table.
    pub fragment_count: i64,
    /// Current base dataset manifest version.
    pub version: u64,
    /// Manifest timestamp, Unix milliseconds — when the base table last changed.
    pub last_updated: i64,
    /// Flushed MemWAL generations pending merge across all shards.
    pub pending_wal_generations: i64,
    /// Rows durably accepted by *this instance's* resident writer but not yet
    /// sealed into a flushed generation, and so not yet counted by
    /// [`Self::row_count`] or visible to any reader.
    ///
    /// Instance-local by nature: it reads this process's in-memory writer, so
    /// it cannot see another instance's unflushed memtable. `0` when no writer
    /// is resident (nothing buffered) — which is also the steady state right
    /// after a flush.
    ///
    /// A persistently non-zero value means the flush sweeper is not keeping up
    /// (or is disabled); that is the signal that reads are lagging writes.
    pub unflushed_rows: i64,
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

/// Table name the ad-hoc SQL console binds an experiment's records to.
pub const SQL_TABLE_NAME: &str = "records";

/// Upper bound on rows materialized into the in-memory `records` table for an
/// ad-hoc SQL query. Guards master memory against a huge experiment.
pub const SQL_MAX_SCAN_ROWS: usize = 200_000;

/// Upper bound on rows returned by an ad-hoc SQL query. Hitting it flags the
/// result `truncated` rather than silently dropping rows.
pub const SQL_MAX_RESULT_ROWS: usize = 10_000;

/// Result of [`RolloutStore::query_sql`]: column names plus JSON-encoded rows.
#[derive(Debug, Clone)]
pub struct SqlQueryResult {
    /// Output column names, in select order.
    pub columns: Vec<String>,
    /// Rows as JSON values (one inner vec per row, aligned to `columns`).
    pub rows: Vec<Vec<serde_json::Value>>,
    /// True when the result was capped at [`SQL_MAX_RESULT_ROWS`].
    pub truncated: bool,
}

/// Reject anything that is not a single read-only `SELECT` (or CTE) statement.
///
/// Parsing with DataFusion's SQL parser is more robust than string matching:
/// it rejects trailing/multiple statements, DML (`INSERT`/`UPDATE`/`DELETE`),
/// DDL (`CREATE`/`DROP`/…), `COPY`, and `EXPLAIN ANALYZE` side effects. Because
/// the console only registers one fixed in-memory table, there is no catalog
/// surface to mutate even if a statement slipped through.
fn ensure_select_only(sql: &str) -> LanceResult<()> {
    let statements = DFParser::parse_sql(sql)
        .map_err(|err| LanceError::invalid_input(format!("could not parse SQL: {err}")))?;
    if statements.len() != 1 {
        return Err(LanceError::invalid_input(
            "exactly one SQL statement is allowed".to_string(),
        ));
    }
    match &statements[0] {
        DFStatement::Statement(stmt) if matches!(stmt.as_ref(), SqlStatement::Query(_)) => Ok(()),
        _ => Err(LanceError::invalid_input(
            "only read-only SELECT queries are allowed".to_string(),
        )),
    }
}

/// Convert DataFusion result batches into a JSON [`SqlQueryResult`], capping at
/// [`SQL_MAX_RESULT_ROWS`] and flagging `truncated` when the cap is reached.
/// `columns` is taken from the query plan schema so it is populated even for a
/// zero-row result.
fn sql_batches_to_result(
    columns: Vec<String>,
    batches: Vec<RecordBatch>,
) -> LanceResult<SqlQueryResult> {
    let mut rows: Vec<Vec<serde_json::Value>> = Vec::new();
    let mut truncated = false;
    'outer: for batch in &batches {
        // arrow-json encodes each row as a JSON object keyed by column name;
        // re-key to a positional array so duplicate/expression column names are
        // preserved in select order.
        let mut writer = arrow_json::ArrayWriter::new(Vec::<u8>::new());
        writer.write(batch).map_err(LanceError::from)?;
        writer.finish().map_err(LanceError::from)?;
        let json_rows: Vec<serde_json::Map<String, serde_json::Value>> =
            serde_json::from_slice(&writer.into_inner()).map_err(|err| {
                LanceError::from(ArrowError::InvalidArgumentError(err.to_string()))
            })?;
        for obj in json_rows {
            if rows.len() >= SQL_MAX_RESULT_ROWS {
                truncated = true;
                break 'outer;
            }
            let row = columns
                .iter()
                .map(|name| obj.get(name).cloned().unwrap_or(serde_json::Value::Null))
                .collect();
            rows.push(row);
        }
    }

    Ok(SqlQueryResult {
        columns,
        rows,
        truncated,
    })
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
    /// See `docs/src/specs/rollout-deployment.md`.
    pub shard_id: Option<String>,
    /// Count-triggered self-merge threshold. After an append flushes a new
    /// generation, if this instance's own shard has accumulated at least this
    /// many un-merged flushed generations, `add` synchronously merges them into
    /// the base table and drains the shard's `flushed_generations` back to empty
    /// (see [`RolloutStore::cleanup_own_shard`]). This bounds read amplification:
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
    /// Shared Lance [`Session`] used to open this store's base dataset (and,
    /// transitively, every flushed MemWAL generation it reads — those inherit
    /// the base dataset's session).
    ///
    /// When `None`, Lance builds a *fresh* per-store session whose index and
    /// metadata caches default to **6 GiB / 1 GiB** and are keyed by dataset
    /// URI. Each flushed generation is a distinct URI, so a busy store's read
    /// path (`observe`, list, stats) feeds an ever-growing set of keys into that
    /// cache until it approaches the 6 GiB cap — worker RSS then grows linearly
    /// with cumulative appends and is never released across merge/compact cycles
    /// (the session outlives the merged generations). Passing a single shared,
    /// capacity-bounded session across all resident stores bounds this: the
    /// deployment's total Lance cache is the session's capacity, shared, rather
    /// than 6 GiB *per store*. Build one with [`RolloutStore::build_session`].
    pub session: Option<Arc<Session>>,
}

/// A Lance-backed store for RL rollout trajectories.
///
/// All storage mechanics — dataset handle, the resident MemWAL writer, the
/// durable append, flush, WAL merge, compaction, indexing and the LSM read
/// scanner — live in the crate-internal `StorageBase`. What remains here is the rollout
/// *schema*: [`rollout_schema`], record encoding/decoding, filters, and the
/// typed read APIs built on top of the base's scanners.
pub struct RolloutStore {
    base: StorageBase,
}

impl RolloutStore {
    /// Build a shared, capacity-bounded Lance [`Session`] for opening rollout
    /// stores.
    ///
    /// Split the total cache budget across Lance's two caches: `index_cache_bytes`
    /// bounds opened-index data and `metadata_cache_bytes` bounds file/dataset
    /// metadata. Both are byte-weighted LRUs keyed by dataset URI, so a single
    /// session shared across every resident [`RolloutStore`] caps the process's
    /// *total* Lance cache at this budget — instead of Lance's default of a fresh
    /// 6 GiB + 1 GiB session per store (the source of the per-append RSS growth;
    /// see [`RolloutStoreOptions::session`]).
    #[must_use]
    pub fn build_session(index_cache_bytes: usize, metadata_cache_bytes: usize) -> Arc<Session> {
        StorageBase::build_session(index_cache_bytes, metadata_cache_bytes)
    }

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
        let RolloutStoreOptions {
            storage_options,
            shard_id,
            merge_after_generations,
            session,
        } = options;
        let base = StorageBase::open(
            uri,
            StorageBaseOptions {
                storage_options,
                shard_id,
                merge_after_generations,
                session,
                schema: Arc::new(rollout_schema()),
                key_column: "id".to_string(),
                // Rollout's schema is additive across releases (e.g. the
                // claim-check columns), so a WAL merge first evolves an older
                // base table to the current schema before appending.
                latest_schema: Some(Arc::new(rollout_schema())),
                // Rollout defers the seal: high fan-in appends must not
                // serialize behind a per-append seal, and rollout rows are
                // immutable so nothing reads back before writing. The server's
                // flush sweeper (and `?flush=true`) provide visibility.
                seal_on_put: false,
            },
            create_if_missing,
        )
        .await?;
        Ok(Self { base })
    }
    /// URI of the underlying Lance dataset.
    #[must_use]
    pub fn uri(&self) -> &str {
        self.base.uri()
    }

    /// Current dataset version.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.base.version()
    }

    /// Checkout a specific dataset version — recovers the exact rollout set that
    /// trained a checkpoint (spec §3, reproducibility).
    pub async fn checkout(&mut self, version_id: u64) -> LanceResult<()> {
        self.base.checkout(version_id).await
    }

    /// Refresh this handle to the latest base-table manifest while retaining
    /// its session and metadata caches.
    ///
    /// Long-lived read handles call this before a new request so compaction or
    /// WAL merges committed by another process become visible without paying
    /// the cost of reopening the dataset and rebuilding all session caches.
    pub async fn refresh_latest(&mut self) -> LanceResult<()> {
        self.base.refresh_latest().await
    }

    /// Append rollout rows through this instance's MemWAL shard; returns the
    /// current base dataset version.
    ///
    /// The write is routed to the shard derived from the configured
    /// `shard_id`, so concurrent appends from other server instances (each
    /// owning a distinct shard) never contend.
    ///
    /// # Durable on return, *not* visible on return
    ///
    /// The only per-append work is `put`, which returns once the WAL entry has
    /// been PUT to object storage. The rows are then **durable** — they survive
    /// a crash and are replayed on reopen — but they are **not yet readable**,
    /// by this instance or any other. A row becomes visible only after its
    /// memtable is sealed into a flushed generation and committed to the shard
    /// manifest, which happens in [`Self::flush`] (also performed by
    /// [`Self::close`], and by the merge path via its internal close).
    ///
    /// Callers therefore get **no read-your-write guarantee**. In the server the
    /// gap is bounded by the periodic flush sweeper's interval
    /// (`ROLLOUT_FLUSH_INTERVAL_SECS`, default 30s); a caller that needs the row
    /// readable immediately must `add(..).await` then `flush().await`.
    ///
    /// This decoupling is deliberate: sealing on the append path serialized
    /// concurrent appends behind one seal+drain. Keeping only the durable `put`
    /// here lets appends run concurrently, and reuses a single resident
    /// `ShardWriter` (see `StorageBase::resident_writer`) so the shard epoch is
    /// claimed once and the object-store connection is pooled, rather than
    /// paying a cold DNS resolution + TCP/TLS handshake + epoch claim per
    /// append.
    ///
    /// Note that because visibility is asynchronous,
    /// [`RolloutObservation::row_count`] does not count rows that are durable
    /// but not yet flushed.
    ///
    /// # The return value carries no information about this append
    ///
    /// It is the base dataset version, which MemWAL appends do **not** advance,
    /// so it is a constant unrelated to the rows just written — the same value
    /// before and after. It does not identify a snapshot containing them, and
    /// (since the seal moved off this path) it does not indicate they are
    /// visible either. Do not treat it as a write handle or use it to poll for
    /// visibility; call [`Self::flush`] instead. Retained only for API
    /// compatibility — see the module docs on reproducibility.
    pub async fn add(&self, records: &[RolloutRecord]) -> LanceResult<u64> {
        if records.is_empty() {
            return Ok(self.base.version());
        }
        // Encoding is schema-specific and stays here; everything after it —
        // the resident writer, the fence retry, and the latency/error metrics —
        // is `StorageBase::put`.
        let batch = self.records_to_batch(records)?;
        self.base.put(vec![batch]).await?;
        Ok(self.base.version())
    }

    /// Materialize the active memtable into a flushed, queryable generation, so
    /// previously added rows become readable on every instance. A no-op when no
    /// writer is resident. See `StorageBase::flush`.
    pub async fn flush(&self) -> LanceResult<()> {
        self.base.flush().await
    }

    /// Gracefully close the resident writer, draining its background tasks.
    /// Idempotent. See `StorageBase::close`.
    pub async fn close(&mut self) -> LanceResult<()> {
        self.base.close().await
    }

    /// Merge this instance's flushed generations into the base table **if** the
    /// shard has accumulated at least `merge_after_generations` of them (the
    /// count trigger; `0` disables it). No-op otherwise.
    pub async fn maybe_merge_own_shard(&mut self) -> LanceResult<usize> {
        self.base.maybe_merge_own_shard().await
    }

    /// The shared-lock half of a merge; see `StorageBase::prepare_merge_if_ready`
    /// for the intended read-lock/write-lock split.
    pub async fn prepare_merge_if_ready(
        &self,
        threshold: usize,
    ) -> LanceResult<Option<(ShardManifestStore, ShardManifest, PreparedMerge)>> {
        self.base.prepare_merge_if_ready(threshold).await
    }

    /// [`Self::prepare_merge_if_ready`], but seals the active memtable first —
    /// the time-triggered behavior of [`Self::cleanup_own_shard`].
    pub async fn prepare_cleanup_merge(
        &self,
    ) -> LanceResult<Option<(ShardManifestStore, ShardManifest, PreparedMerge)>> {
        self.base.prepare_cleanup_merge().await
    }

    /// Commit a merge prepared by [`Self::prepare_merge_if_ready`].
    pub async fn commit_prepared_merge(
        &mut self,
        manifest_store: &ShardManifestStore,
        manifest: &ShardManifest,
        prepared: PreparedMerge,
    ) -> LanceResult<usize> {
        self.base
            .commit_prepared_merge(manifest_store, manifest, prepared)
            .await
    }

    /// Run one periodic WAL-cleanup pass over this instance's own shard: seal,
    /// then fold **every** pending flushed generation into the base table. This
    /// is the *time* half of the "time OR count" trigger and is deliberately not
    /// gated by the count threshold. See `StorageBase::cleanup_own_shard`.
    pub async fn cleanup_own_shard(&mut self) -> LanceResult<usize> {
        self.base.cleanup_own_shard().await
    }

    /// Compact the base table's small fragments into larger ones.
    ///
    /// Must be driven by a *single* external trigger, not a per-worker timer:
    /// compaction rewrites the shared base table and two concurrent `Rewrite`
    /// commits conflict. See `StorageBase::compact`.
    pub async fn compact(
        &mut self,
        options: Option<CompactionConfig>,
    ) -> LanceResult<CompactionMetrics> {
        self.base.compact(options).await
    }

    /// Build a ZoneMap scalar index on the base table's `id` column. Idempotent.
    /// See `StorageBase::create_key_zonemap_index`.
    pub async fn create_id_zonemap_index(&mut self) -> LanceResult<()> {
        self.base.create_key_zonemap_index().await
    }

    /// Whether the base table has accumulated at least `min_fragments`
    /// fragments (and is thus worth compacting), honoring quiet hours.
    #[must_use]
    pub fn should_compact(&self, config: &CompactionConfig) -> bool {
        self.base.should_compact(config)
    }

    /// Number of flushed MemWAL generations pending merge into the base table
    /// across all shards. Read-only; never merges.
    pub async fn pending_wal_generations(&self) -> LanceResult<usize> {
        self.base.pending_wal_generations().await
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
        let base_rows = self.base.dataset.count_rows(None).await? as u64;
        let pending_rows = self.base.pending_wal_rows(&shard_snapshots).await?;
        let row_count = (base_rows + pending_rows) as i64;
        let fragment_count = self.base.dataset.count_fragments() as i64;
        let version = self.base.dataset.manifest.version;
        let last_updated = self.base.dataset.manifest.timestamp().timestamp_millis();
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
            unflushed_rows: self.unflushed_rows().await,
        })
    }

    /// Rows buffered in this instance's resident writer that have not yet been
    /// sealed into a flushed generation. See `StorageBase::unflushed_rows`.
    async fn unflushed_rows(&self) -> i64 {
        self.base.unflushed_rows().await
    }

    /// Current compaction statistics for the base table.
    #[must_use]
    pub fn compaction_stats(&self) -> CompactionStats {
        self.base.compaction_stats()
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

    /// Run a read-only `SELECT` against this experiment's rollout records.
    ///
    /// The merged view ([`ListSource::All`]: base table ∪ pending MemWAL
    /// generations) is materialized into an in-memory table named `records`,
    /// then queried with DataFusion. Only a single `SELECT`/CTE statement is
    /// accepted — any DML/DDL/multi-statement input is rejected before
    /// execution (see [`ensure_select_only`]). The `binary_payload` column is
    /// excluded (blob bytes are fetched via [`Self::get_blob`]).
    ///
    /// Two bounds keep a query from exhausting master memory:
    /// - [`SQL_MAX_SCAN_ROWS`] caps how many rows are materialized into the
    ///   `records` table; exceeding it is a hard error asking the user to work
    ///   on a smaller experiment.
    /// - [`SQL_MAX_RESULT_ROWS`] caps returned rows; hitting it sets
    ///   `truncated = true` rather than silently dropping rows.
    ///
    /// A syntactically invalid or non-`SELECT` query returns
    /// [`LanceError::InvalidInput`] so callers can surface it as a 400.
    pub async fn query_sql(&self, sql: &str) -> LanceResult<SqlQueryResult> {
        ensure_select_only(sql)?;

        // Materialize the merged (base ∪ WAL) non-blob rows, bounded.
        let shard_snapshots = self.wal_shard_snapshots().await?;
        let columns = self.non_blob_columns();
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let scanner = self
            .lsm_scanner_for_source(ListSource::All, shard_snapshots)
            .project(&refs);
        let mut stream = scanner.try_into_stream().await?;

        let mut batches: Vec<RecordBatch> = Vec::new();
        let mut scanned_rows = 0usize;
        let mut table_schema: Option<Arc<Schema>> = None;
        while let Some(batch) = stream.try_next().await? {
            if table_schema.is_none() {
                table_schema = Some(batch.schema());
            }
            scanned_rows += batch.num_rows();
            if scanned_rows > SQL_MAX_SCAN_ROWS {
                return Err(LanceError::invalid_input(format!(
                    "experiment has more than {SQL_MAX_SCAN_ROWS} rows, which is too large \
                     for ad-hoc SQL; use the record browser filters instead"
                )));
            }
            batches.push(batch);
        }

        // Empty experiment: fall back to the projected dataset schema so
        // `SELECT`s still resolve column names against an empty `records` table.
        let schema = match table_schema {
            Some(schema) => schema,
            None => {
                let full: Schema = self.base.dataset.schema().into();
                let projected: Vec<FieldRef> = full
                    .fields()
                    .iter()
                    .filter(|f| f.name() != "binary_payload")
                    .cloned()
                    .collect();
                Arc::new(Schema::new(projected))
            }
        };

        let ctx = SessionContext::new();
        let provider = MemTable::try_new(schema, vec![batches])
            .map_err(|err| LanceError::from(ArrowError::from_external_error(Box::new(err))))?;
        ctx.register_table(SQL_TABLE_NAME, Arc::new(provider))
            .map_err(|err| LanceError::from(ArrowError::from_external_error(Box::new(err))))?;

        // DataFusion planning/exec errors (unknown column, bad function, …) are
        // user errors → InvalidInput so the API returns 400, not 500.
        let df = ctx
            .sql(sql)
            .await
            .map_err(|err| LanceError::invalid_input(err.to_string()))?;
        let df = df
            .limit(0, Some(SQL_MAX_RESULT_ROWS + 1))
            .map_err(|err| LanceError::invalid_input(err.to_string()))?;
        // Capture the output columns from the plan schema so they are known even
        // when the query returns zero rows.
        let columns: Vec<String> = df
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let result_batches = df
            .collect()
            .await
            .map_err(|err| LanceError::invalid_input(err.to_string()))?;

        sql_batches_to_result(columns, result_batches)
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

    /// Fetch a row *together with* its `binary_payload` in a single base-first
    /// scan, returning `(record, payload)`.
    ///
    /// Callers that need both the row metadata (e.g. `content_type` for a
    /// download's `Content-Type`/filename) and the artifact bytes would
    /// otherwise call [`Self::get_by_id`] then [`Self::get_blob`] — two
    /// independent point scans over the same shard. This folds them into one
    /// scan: the row is located once and `binary_payload` is materialized for
    /// only that row (via a projected `take`), so it never reads back an entire
    /// fragment's payloads. `payload` is `None` when the row carries no blob.
    ///
    /// Base-table-first with the same immutable-row reasoning as
    /// [`Self::get_by_id`]: a hit in the base table returns immediately with no
    /// MemWAL generation opened; only a base miss falls back to the flushed WAL.
    pub async fn get_record_with_blob(
        &self,
        id: &str,
    ) -> LanceResult<Option<(RolloutRecord, Option<Vec<u8>>)>> {
        // Base table first — no manifest reads, no per-generation opens.
        if let Some(record) = self.scan_one_by_id(id, ListSource::Fragments).await? {
            let payload = Self::get_blob_from_dataset(&self.base.dataset, id)
                .await?
                .flatten();
            return Ok(Some((record, payload)));
        }

        // Base miss: locate the row in the flushed generations. Reuse the
        // NotFound-tolerant, bounded-parallel blob fallback and pair it with the
        // WAL-sourced metadata scan.
        let Some(record) = self.scan_one_by_id(id, ListSource::Wal).await? else {
            return Ok(None);
        };
        let payload = self.get_blob(id).await?;
        Ok(Some((record, payload)))
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
        let columns = self
            .non_blob_columns()
            .into_iter()
            .filter(|column| {
                source == ListSource::Fragments || !CLAIM_CHECK_COLUMNS.contains(&column.as_str())
            })
            .collect::<Vec<_>>();
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
        if let Some(payload) = Self::get_blob_from_dataset(&self.base.dataset, id).await? {
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
        self.base.flushed_generation_uri(shard_id, path)
    }

    async fn open_flushed_dataset(&self, uri: &str) -> LanceResult<Dataset> {
        self.base.open_flushed_dataset(uri).await
    }

    /// Top-level column names excluding `binary_payload`, so list-style scans
    /// never materialize artifact bytes.
    fn non_blob_columns(&self) -> Vec<String> {
        self.base
            .dataset
            .schema()
            .fields
            .iter()
            .map(|field| field.name.clone())
            .filter(|name| name != "binary_payload")
            .collect()
    }

    /// Build an LSM scanner over the base table unioned with every shard's
    /// flushed MemWAL generations. Deduplicates by `id`. See
    /// `StorageBase::lsm_scanner`.
    async fn lsm_scanner(&self) -> LanceResult<LsmScanner> {
        self.base.lsm_scanner().await
    }

    /// Build a paginating scanner for the requested [`ListSource`]. See
    /// `StorageBase::lsm_scanner_for_source`.
    fn lsm_scanner_for_source(
        &self,
        source: ListSource,
        shard_snapshots: Vec<ShardSnapshot>,
    ) -> LsmScanner {
        self.base.lsm_scanner_for_source(source, shard_snapshots)
    }

    /// Read the latest manifest for every MemWAL shard.
    async fn wal_shard_snapshots(&self) -> LanceResult<Vec<ShardSnapshot>> {
        self.base.wal_shard_snapshots().await
    }

    fn records_to_batch(&self, records: &[RolloutRecord]) -> LanceResult<RecordBatch> {
        let field_paths = self.base.dataset.schema().field_paths();
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

        let schema: Arc<Schema> = Arc::new(self.base.dataset.schema().into());
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

    use std::collections::HashSet;

    use arrow_array::RecordBatchIterator;
    use lance::dataset::NewColumnTransform;
    use lance::index::DatasetIndexExt;

    use crate::record::Relationship;
    use crate::rollout::{ROLE_ARTIFACT, ROLE_ASSISTANT};
    use crate::store_base::{
        align_batch_to_schema, DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
        ID_INDEX_NAME as ROLLOUT_ID_INDEX_NAME,
    };
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

    fn pre_claim_check_schema() -> Arc<Schema> {
        Arc::new(Schema::new(
            rollout_schema()
                .fields()
                .iter()
                .filter(|field| !CLAIM_CHECK_COLUMNS.contains(&field.name().as_str()))
                .cloned()
                .collect::<Vec<_>>(),
        ))
    }

    async fn create_empty_dataset(uri: &str, schema: Arc<Schema>) {
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let batches = RecordBatchIterator::new(
            vec![Ok::<RecordBatch, ArrowError>(empty_batch)].into_iter(),
            schema,
        );
        Dataset::write(batches, uri, None).await.unwrap();
    }

    async fn store_with_legacy_base_and_wal(evolve_base: bool) -> (TempDir, RolloutStore) {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let legacy_schema = pre_claim_check_schema();
        create_empty_dataset(&uri, legacy_schema.clone()).await;

        let mut store = RolloutStore::open_with_options(
            &uri,
            RolloutStoreOptions {
                shard_id: Some("pre-claim-check".to_string()),
                merge_after_generations: None,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let base_batch = store
            .records_to_batch(&[assistant_record("legacy-base")])
            .unwrap();
        let base_schema = base_batch.schema();
        let base_reader = RecordBatchIterator::new(
            vec![Ok::<RecordBatch, ArrowError>(base_batch)].into_iter(),
            base_schema,
        );
        store.base.dataset.append(base_reader, None).await.unwrap();

        store.add(&[assistant_record("legacy-wal")]).await.unwrap();
        store.flush().await.unwrap();
        assert_eq!(flushed_generation_count(&store).await, 1);
        store.close().await.unwrap();

        if evolve_base {
            let claim_check_fields = rollout_schema()
                .fields()
                .iter()
                .filter(|field| legacy_schema.field_with_name(field.name()).is_err())
                .cloned()
                .collect::<Vec<_>>();
            store
                .base
                .dataset
                .add_columns(
                    NewColumnTransform::AllNulls(Arc::new(Schema::new(claim_check_fields))),
                    None,
                    None,
                )
                .await
                .unwrap();
        }

        (dir, store)
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
        assert_eq!(actual.model_input_string, expected.model_input_string);
        assert_eq!(actual.model_output_string, expected.model_output_string);
        assert_eq!(actual.rationale, expected.rationale);
        assert_eq!(actual.problem_text, expected.problem_text);
        assert_eq!(actual.user_metadata, expected.user_metadata);
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
            let store = RolloutStore::open(&uri).await.unwrap();
            // MemWAL appends land in the `_mem_wal` namespace and do not advance
            // the base dataset version; `add` returns it unchanged.
            store
                .add(&[assistant.clone(), artifact.clone(), quoted_artifact.clone()])
                .await
                .unwrap();
            store.flush().await.unwrap();

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
            let store = RolloutStore::open(&uri).await.unwrap();
            // Empty store: no rows, no pending generations.
            let obs = store.observe().await.unwrap();
            assert_eq!(obs.row_count, 0);
            assert_eq!(obs.pending_wal_generations, 0);

            store
                .add(&[assistant_record("a"), assistant_record("b")])
                .await
                .unwrap();
            store.flush().await.unwrap();

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
            let store = RolloutStore::open(&uri).await.unwrap();

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
            store.flush().await.unwrap();

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
            let store = RolloutStore::open(&uri).await.unwrap();
            store
                .add(&[
                    assistant_record("assistant"),
                    artifact_record("artifact", b"bytes"),
                ])
                .await
                .unwrap();
            store.flush().await.unwrap();

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
            let store = RolloutStore::open(&uri).await.unwrap();
            store.add(std::slice::from_ref(&artifact)).await.unwrap();

            // A second append accumulates rather than replacing the first.
            store.add(&[assistant_record("row-1")]).await.unwrap();
            store.flush().await.unwrap();
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
    fn cleanup_own_shard_seals_before_merging() {
        // Regression guard for the `ROLLOUT_FLUSH_INTERVAL_SECS=0` trap: with no
        // periodic flush, nothing seals the active memtable, so
        // `flushed_generations` stays empty and the threshold check in
        // `merge_own_shard_if_ready` used to return 0 without ever merging —
        // leaving rows durable but permanently invisible.
        //
        // `cleanup_own_shard` now flushes first, making it a genuine standalone
        // fallback, which is what the config docs already claimed.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    session: None,
                    shard_id: Some("cleanup-seal-0".to_string()),
                    // Count trigger disabled: cleanup is the only path that can
                    // make this row visible, exactly as with flush interval 0.
                    merge_after_generations: Some(0),
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("c-0")]).await.unwrap();

            // Nothing sealed yet: invisible, and no generation pending.
            assert!(store.list(None, None).await.unwrap().is_empty());
            assert_eq!(
                store.observe().await.unwrap().pending_wal_generations,
                0,
                "precondition: the memtable is unsealed, so no generation exists"
            );

            // A single cleanup pass must seal, merge, and expose the row.
            let reclaimed = store.cleanup_own_shard().await.unwrap();
            assert_eq!(reclaimed, 1, "cleanup must seal then merge the generation");

            let seen = store.list(None, None).await.unwrap();
            assert_eq!(seen.len(), 1, "row must be visible after cleanup alone");
            assert_eq!(seen[0].id, "c-0");
        });
    }

    #[test]
    fn dropping_a_store_seals_its_unflushed_rows() {
        // The server's flush sweeper only visits LRU-resident stores, so a store
        // evicted while holding an unsealed memtable depends entirely on `Drop`
        // to seal it. `Drop` spawns a detached `ShardWriter::close`, which does
        // seal — this asserts that end to end, so the eviction path cannot
        // silently regress into stranding rows.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let options = || RolloutStoreOptions {
                storage_options: None,
                session: None,
                shard_id: Some("evicted-0".to_string()),
                merge_after_generations: None,
            };

            {
                let store = RolloutStore::open_with_options(&uri, options())
                    .await
                    .unwrap();
                store.add(&[assistant_record("e-0")]).await.unwrap();
                // Deliberately no flush() and no close(): this models an LRU
                // eviction dropping the last handle.
                assert!(store.list(None, None).await.unwrap().is_empty());
            }

            // The detached close is spawned, not awaited, so poll for the row
            // rather than assuming it has landed by now.
            let mut seen = 0;
            for _ in 0..100 {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                let reader = RolloutStore::open_with_options(&uri, options())
                    .await
                    .unwrap();
                seen = reader.list(None, None).await.unwrap().len();
                if seen > 0 {
                    break;
                }
            }
            assert_eq!(seen, 1, "Drop must seal the memtable of an evicted store");
        });
    }

    #[test]
    fn observe_reports_unflushed_rows_excluded_from_row_count() {
        // `row_count` counts base ∪ flushed generations, so durable-but-unsealed
        // rows are invisible to it. That undercount is inherent to the async
        // flush design; `unflushed_rows` is what makes it observable, and
        // `row_count + unflushed_rows` is every row durably accepted.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    session: None,
                    shard_id: Some("observe-0".to_string()),
                    merge_after_generations: None,
                },
            )
            .await
            .unwrap();

            // No writer resident yet: nothing buffered.
            assert_eq!(store.observe().await.unwrap().unflushed_rows, 0);

            store
                .add(&[assistant_record("o-0"), assistant_record("o-1")])
                .await
                .unwrap();

            let obs = store.observe().await.unwrap();
            assert_eq!(obs.row_count, 0, "row_count must not count unsealed rows");
            assert_eq!(
                obs.unflushed_rows, 2,
                "the durable-but-invisible rows must be observable"
            );

            store.flush().await.unwrap();

            let obs = store.observe().await.unwrap();
            assert_eq!(obs.row_count, 2, "flushed rows join row_count");
            assert_eq!(
                obs.unflushed_rows, 0,
                "nothing remains buffered after a flush"
            );
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
                session: None,
                shard_id: Some(shard.to_string()),
                merge_after_generations: None,
            };

            let instance_a = RolloutStore::open_with_options(&uri, options("rollout-0"))
                .await
                .unwrap();
            instance_a
                .add(&[artifact_record("a-0", shard_a_blob)])
                .await
                .unwrap();
            instance_a.flush().await.unwrap();

            let instance_b = RolloutStore::open_with_options(&uri, options("rollout-1"))
                .await
                .unwrap();
            instance_b
                .add(&[artifact_record("b-0", shard_b_blob)])
                .await
                .unwrap();
            instance_b.flush().await.unwrap();

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
            let instance_a = RolloutStore::open_with_options(&uri, options("filter-a"))
                .await
                .unwrap();
            let mut quoted = assistant_record("row-'quoted");
            quoted.rollout_id = "rollout-alpha".to_string();
            quoted.policy_version = Some("policy-a".to_string());
            instance_a
                .add(&[quoted, assistant_record("row-a")])
                .await
                .unwrap();
            instance_a.flush().await.unwrap();

            let instance_b = RolloutStore::open_with_options(&uri, options("filter-b"))
                .await
                .unwrap();
            let artifact = artifact_record("row-b", b"blob");
            instance_b.add(&[artifact]).await.unwrap();
            instance_b.flush().await.unwrap();

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
            let writer = RolloutStore::open_with_options(
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
            writer.flush().await.unwrap();

            let mut cached_reader =
                RolloutStore::open_existing_with_options(&uri, RolloutStoreOptions::default())
                    .await
                    .unwrap();
            assert_eq!(cached_reader.list(None, None).await.unwrap().len(), 1);

            writer.add(&[assistant_record("row-1")]).await.unwrap();
            writer.flush().await.unwrap();
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
            store.flush().await.unwrap();
            store.maybe_merge_own_shard().await.unwrap();

            let mut earlier = assistant_record("row-b");
            earlier.rollout_id = "target".to_string();
            earlier.sequence_order = 0;
            store.add(&[earlier]).await.unwrap();
            store.flush().await.unwrap();
            store.maybe_merge_own_shard().await.unwrap();

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
        let object_store = store.base.dataset.object_store(None).await.unwrap();
        let branch_location = store.base.dataset.branch_location();
        let manifest_store = ShardManifestStore::new(
            object_store,
            &branch_location.path,
            store.base.write_shard,
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
        let object_store = store.base.dataset.object_store(None).await.unwrap();
        let branch_location = store.base.dataset.branch_location();
        let manifest_store = ShardManifestStore::new(
            object_store,
            &branch_location.path,
            store.base.write_shard,
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
            let store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: None, // no merge → epoch never reclaimed
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            store.flush().await.unwrap();
            let epoch_after_first = shard_writer_epoch(&store).await;

            // Read-after-write: the first append is visible right away.
            assert!(store.get_by_id("a-0").await.unwrap().is_some());

            for i in 1..5 {
                store
                    .add(&[assistant_record(&format!("a-{i}"))])
                    .await
                    .unwrap();
                store.flush().await.unwrap();
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
                    session: None,
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
                store.flush().await.unwrap();
                store.maybe_merge_own_shard().await.unwrap();
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
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: None,
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            store.flush().await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 1);

            // Idempotent close drops the resident writer.
            store.close().await.unwrap();
            store.close().await.unwrap(); // second close is a no-op

            // add() after close reopens the writer and stays visible.
            store.add(&[assistant_record("a-1")]).await.unwrap();
            store.flush().await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 2);
        });
    }

    #[test]
    fn add_is_durable_but_not_visible_until_flush() {
        // Core semantic of the concurrent-write design: `add` returns once the
        // record is durable (WAL-persisted), but the row is NOT visible to reads
        // until `flush` seals the memtable into a queryable generation. This is
        // the accepted read-after-write asynchrony that lets appends run without
        // a per-append seal.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = RolloutStore::open(&uri).await.unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            // Durable, but not yet flushed: the read path (base ∪ flushed
            // generations) does not see it.
            assert_eq!(store.list(None, None).await.unwrap().len(), 0);
            assert!(store.get_by_id("a-0").await.unwrap().is_none());

            // After a flush the row is visible, and nothing was lost.
            store.flush().await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 1);
            assert!(store.get_by_id("a-0").await.unwrap().is_some());
        });
    }

    #[test]
    fn concurrent_adds_share_one_store_and_none_are_lost() {
        // `add` is `&self`, so many appends can run concurrently against one
        // shared store handle (no RwLock, no per-append serialization). All must
        // succeed and every record must survive to be readable after a flush.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let store = Arc::new(RolloutStore::open(&uri).await.unwrap());

            const N: usize = 50;
            let mut handles = Vec::with_capacity(N);
            for i in 0..N {
                let store = store.clone();
                handles.push(tokio::spawn(async move {
                    store
                        .add(&[assistant_record(&format!("row-{i}"))])
                        .await
                        .unwrap();
                }));
            }
            for h in handles {
                h.await.unwrap();
            }

            store.flush().await.unwrap();
            let listed = store.list(None, None).await.unwrap();
            assert_eq!(listed.len(), N, "every concurrent append must be readable");
            for i in 0..N {
                assert!(
                    store
                        .get_by_id(&format!("row-{i}"))
                        .await
                        .unwrap()
                        .is_some(),
                    "row-{i} was lost"
                );
            }
        });
    }

    #[test]
    fn flush_is_noop_without_writes() {
        // flush() with no resident writer / nothing buffered must be a harmless
        // no-op (the periodic sweeper calls it on every resident store each tick).
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = RolloutStore::open(&uri).await.unwrap();
            // No writer opened yet, nothing buffered.
            store.flush().await.unwrap();
            store.flush().await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 0);

            // A flush after an add makes the row visible; a second flush is a
            // no-op (memtable already drained).
            store.add(&[assistant_record("a-0")]).await.unwrap();
            store.flush().await.unwrap();
            store.flush().await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 1);
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
                    session: None,
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
                store.flush().await.unwrap();
                store.maybe_merge_own_shard().await.unwrap();
            }
            store
                .add(&[artifact_record("a-6", artifact_bytes)])
                .await
                .unwrap();
            store.flush().await.unwrap();
            store.maybe_merge_own_shard().await.unwrap();

            let before = store.base.dataset.count_fragments();
            assert!(before > 1, "expected several fragments, got {before}");
            assert!(store.should_compact(&CompactionConfig {
                min_fragments: 2,
                ..CompactionConfig::default()
            }));

            let metrics = store.compact(None).await.unwrap();
            assert!(metrics.fragments_removed > 0);

            let after = store.base.dataset.count_fragments();
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
            let store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: Some(1),
                },
            )
            .await
            .unwrap();
            store.add(&[assistant_record("a-0")]).await.unwrap();

            let frags = store.base.dataset.count_fragments();
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
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: Some(1),
                },
            )
            .await
            .unwrap();
            for i in 0..5 {
                a.add(&[assistant_record(&format!("a-{i}"))]).await.unwrap();
                a.flush().await.unwrap();
                a.maybe_merge_own_shard().await.unwrap();
            }

            // B accumulates its own shard's generations (not yet merged).
            let b = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    session: None,
                    shard_id: Some("rollout-1".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            for i in 0..3 {
                b.add(&[assistant_record(&format!("b-{i}"))]).await.unwrap();
                b.flush().await.unwrap();
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
            let store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: Some(3),
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            store.flush().await.unwrap();
            store.add(&[assistant_record("a-1")]).await.unwrap();
            store.flush().await.unwrap();

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
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: Some(3),
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            store.flush().await.unwrap();
            store.add(&[assistant_record("a-1")]).await.unwrap();
            store.flush().await.unwrap();
            // The third append reaches the threshold and triggers the merge.
            store
                .add(&[artifact_record("a-2", artifact_bytes)])
                .await
                .unwrap();
            store.flush().await.unwrap();
            store.maybe_merge_own_shard().await.unwrap();

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
                        session: None,
                        shard_id: Some("rollout-0".to_string()),
                        merge_after_generations: Some(2),
                    },
                )
                .await
                .unwrap();
                store.add(&[assistant_record("a-0")]).await.unwrap();
                store.flush().await.unwrap();
                store.add(&[assistant_record("a-1")]).await.unwrap();
                store.flush().await.unwrap();
                store.maybe_merge_own_shard().await.unwrap();
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
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    merge_after_generations: None, // count trigger off
                },
            )
            .await
            .unwrap();

            store.add(&[assistant_record("a-0")]).await.unwrap();
            store.flush().await.unwrap();
            // One generation pending: the time trigger merges it immediately —
            // it does not wait for a count threshold.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 1);
            assert_eq!(flushed_generation_count(&store).await, 0);

            store.add(&[assistant_record("a-1")]).await.unwrap();
            store.flush().await.unwrap();
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
    fn cleanup_merges_pre_claim_check_generations_after_schema_evolution() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let (_dir, mut store) = store_with_legacy_base_and_wal(false).await;

            assert_eq!(store.cleanup_own_shard().await.unwrap(), 1);
            assert_eq!(flushed_generation_count(&store).await, 0);
            let field_paths = store.base.dataset.schema().field_paths();
            for column in CLAIM_CHECK_COLUMNS {
                assert!(field_paths.iter().any(|path| path == column));
            }

            let mut rows = store.list(None, None).await.unwrap();
            rows.sort_by(|left, right| left.id.cmp(&right.id));
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].id, "legacy-base");
            assert_eq!(rows[1].id, "legacy-wal");
            for row in rows {
                assert!(row.model_input_string.is_none());
                assert!(row.model_output_string.is_none());
                assert!(row.rationale.is_none());
                assert!(row.problem_text.is_none());
                assert!(row.user_metadata.is_none());
            }
        });
    }

    #[test]
    fn latest_schema_alignment_preserves_claim_check_values() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let legacy_dir = TempDir::new().unwrap();
            let legacy_uri = legacy_dir.path().to_string_lossy().to_string();
            create_empty_dataset(&legacy_uri, pre_claim_check_schema()).await;
            let mut legacy_store = RolloutStore::open(&legacy_uri).await.unwrap();

            let current_dir = TempDir::new().unwrap();
            let current_uri = current_dir.path().to_string_lossy().to_string();
            let current_store = RolloutStore::open(&current_uri).await.unwrap();
            let mut record = assistant_record("current-generation");
            record.model_input_string = Some("input".to_string());
            record.model_output_string = Some("output".to_string());
            record.rationale = Some("reason".to_string());
            record.problem_text = Some("problem".to_string());
            record.user_metadata = Some(r#"{"source":"worker"}"#.to_string());
            let generation_batch = current_store.records_to_batch(&[record]).unwrap();

            legacy_store.base.ensure_latest_schema().await.unwrap();
            let merge_schema: Arc<Schema> = Arc::new(legacy_store.base.dataset.schema().into());
            let aligned = align_batch_to_schema(generation_batch, merge_schema.clone()).unwrap();
            let reader = RecordBatchIterator::new(
                vec![Ok::<RecordBatch, ArrowError>(aligned)].into_iter(),
                merge_schema,
            );
            legacy_store
                .base
                .dataset
                .append(reader, None)
                .await
                .unwrap();

            let merged = legacy_store
                .get_by_id_source("current-generation", ListSource::Fragments)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(merged.model_input_string.as_deref(), Some("input"));
            assert_eq!(merged.model_output_string.as_deref(), Some("output"));
            assert_eq!(merged.rationale.as_deref(), Some("reason"));
            assert_eq!(merged.problem_text.as_deref(), Some("problem"));
            assert_eq!(
                merged.user_metadata.as_deref(),
                Some(r#"{"source":"worker"}"#)
            );
        });
    }

    #[test]
    fn get_by_id_reads_pre_claim_check_base_and_wal_after_schema_evolution() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let (_dir, store) = store_with_legacy_base_and_wal(true).await;

            let mut base = store.get_by_id("legacy-base").await.unwrap().unwrap();
            let wal = store.get_by_id("legacy-wal").await.unwrap().unwrap();

            assert!(store
                .get_by_id_source("legacy-base", ListSource::Fragments)
                .await
                .unwrap()
                .is_some());
            assert!(store
                .get_by_id_source("legacy-wal", ListSource::Wal)
                .await
                .unwrap()
                .is_some());

            for row in [&base, &wal] {
                assert!(row.model_input_string.is_none());
                assert!(row.model_output_string.is_none());
                assert!(row.rationale.is_none());
                assert!(row.problem_text.is_none());
                assert!(row.user_metadata.is_none());
            }

            base.id.clone_from(&wal.id);
            assert_records_eq(&base, &wal);
            assert_eq!(base.binary_payload, wal.binary_payload);
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
            store.flush().await.unwrap();
            store.cleanup_own_shard().await.unwrap();

            store.create_id_zonemap_index().await.unwrap();
            let has_id_index = |s: &RolloutStore| {
                let dataset = s.base.dataset.clone();
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
                            session: None,
                            shard_id: Some(shard),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap()
                }
            };

            // Two instances, each accumulates a few generations on its own shard.
            let a = make("rollout-0").await;
            let b = make("rollout-1").await;
            for i in 0..4 {
                a.add(&[assistant_record(&format!("a-{i}"))]).await.unwrap();
                a.flush().await.unwrap();
                b.add(&[assistant_record(&format!("b-{i}"))]).await.unwrap();
                b.flush().await.unwrap();
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
                    session: None,
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
                store.flush().await.unwrap();
            }
            assert_eq!(flushed_generation_count(&store).await, 3);

            // Merge all three. Surgical drain removes exactly generations {merged}.
            assert_eq!(store.cleanup_own_shard().await.unwrap(), 3);
            assert_eq!(flushed_generation_count(&store).await, 0);

            // Append a fourth AFTER the drain: it forms a new generation that the
            // prior merge must not have wiped. A second merge folds just that one.
            store.add(&[assistant_record("g-3")]).await.unwrap();
            store.flush().await.unwrap();
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
                    session: None,
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
            store.flush().await.unwrap();
            store.add(&[assistant_record("g-1")]).await.unwrap();
            store.flush().await.unwrap();
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
                .base
                .lsm_scanner_for_source(ListSource::All, shard_snapshots)
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
            let store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    session: None,
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
            let merge_store = RolloutStore::open_with_options(
                &uri_m,
                RolloutStoreOptions {
                    storage_options: None,
                    session: None,
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
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let artifact = artifact_record("row-x", bytes);
            store.add(std::slice::from_ref(&artifact)).await.unwrap();
            store.flush().await.unwrap();

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
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let base = artifact_record("dup", bytes);
            store.add(std::slice::from_ref(&base)).await.unwrap();
            store.flush().await.unwrap();
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

    #[test]
    fn get_record_with_blob_returns_row_and_payload_in_one_scan() {
        // get_record_with_blob folds the metadata point lookup and the payload
        // fetch into a single base-first scan. It must return both the row and
        // its bytes for an un-merged WAL row (base-miss -> fallback) and for a
        // merged base row, and None for a missing id.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let bytes = b"\x00\x01record-with-blob";
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    storage_options: None,
                    session: None,
                    shard_id: Some("rollout-0".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let artifact = artifact_record("row-rw", bytes);
            store.add(std::slice::from_ref(&artifact)).await.unwrap();
            store.flush().await.unwrap();

            // Un-merged: found via the WAL fallback, row + payload paired.
            assert_eq!(flushed_generation_count(&store).await, 1);
            let (record, payload) = store
                .get_record_with_blob("row-rw")
                .await
                .unwrap()
                .expect("row present in WAL");
            assert_records_eq(&record, &artifact);
            assert_eq!(payload.as_deref(), Some(&bytes[..]));

            // After merge: found via the base table, still paired.
            store.cleanup_own_shard().await.unwrap();
            assert_eq!(flushed_generation_count(&store).await, 0);
            let (record, payload) = store
                .get_record_with_blob("row-rw")
                .await
                .unwrap()
                .expect("row present in base");
            assert_records_eq(&record, &artifact);
            assert_eq!(payload.as_deref(), Some(&bytes[..]));

            // Missing id: None, not an error.
            assert!(store.get_record_with_blob("nope").await.unwrap().is_none());
        });
    }

    #[test]
    fn ensure_select_only_accepts_select_and_cte() {
        assert!(ensure_select_only("SELECT * FROM records").is_ok());
        assert!(ensure_select_only("  select id from records where reward > 0 ").is_ok());
        assert!(ensure_select_only("WITH t AS (SELECT id FROM records) SELECT * FROM t").is_ok());
    }

    #[test]
    fn ensure_select_only_rejects_mutations_and_multi_statements() {
        for sql in [
            "DELETE FROM records",
            "UPDATE records SET reward = 1",
            "INSERT INTO records (id) VALUES ('x')",
            "DROP TABLE records",
            "CREATE TABLE t (a INT)",
            "SELECT 1; SELECT 2",
            "not sql at all",
        ] {
            assert!(
                ensure_select_only(sql).is_err(),
                "expected rejection for: {sql}"
            );
        }
    }

    #[test]
    fn query_sql_runs_select_over_merged_records() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = RolloutStore::open(&uri).await.unwrap();
            store
                .add(&[
                    assistant_record("a-0"),
                    assistant_record("a-1"),
                    assistant_record("a-2"),
                ])
                .await
                .unwrap();
            store.flush().await.unwrap();

            // Aggregate over the merged (base + pending WAL) view.
            let result = store
                .query_sql("SELECT count(*) AS n FROM records")
                .await
                .unwrap();
            assert_eq!(result.columns, vec!["n".to_string()]);
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], serde_json::json!(3));
            assert!(!result.truncated);

            // GROUP BY on a real column returns the expected shape.
            let grouped = store
                .query_sql("SELECT role, count(*) AS n FROM records GROUP BY role")
                .await
                .unwrap();
            assert_eq!(grouped.columns, vec!["role".to_string(), "n".to_string()]);
            assert_eq!(grouped.rows.len(), 1);
            assert_eq!(grouped.rows[0][0], serde_json::json!(ROLE_ASSISTANT));
            assert_eq!(grouped.rows[0][1], serde_json::json!(3));

            // The blob column is not exposed to SQL.
            let err = store
                .query_sql("SELECT binary_payload FROM records")
                .await
                .unwrap_err();
            assert!(matches!(err, LanceError::InvalidInput { .. }));

            // A non-SELECT is rejected as invalid input (→ 400 at the API).
            let rejected = store.query_sql("DELETE FROM records").await.unwrap_err();
            assert!(matches!(rejected, LanceError::InvalidInput { .. }));
        });
    }

    #[test]
    fn query_sql_on_empty_experiment_returns_zero_rows() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let store = RolloutStore::open(&uri).await.unwrap();
            let result = store.query_sql("SELECT id FROM records").await.unwrap();
            assert_eq!(result.columns, vec!["id".to_string()]);
            assert!(result.rows.is_empty());
            assert!(!result.truncated);
        });
    }

    /// `add` and `flush` must land on separate histograms, and `flush` must
    /// report which of its three paths it took.
    ///
    /// This is the core of "split the latency": at the HTTP layer these two are
    /// one route (flush is a query param), so if they are not distinguished here
    /// they cannot be distinguished anywhere.
    #[test]
    fn add_and_flush_emit_distinct_labelled_histograms() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};
        use metrics_util::MetricKind;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let dir = TempDir::new().unwrap();
        let uri = dir
            .path()
            .join("rollouts.lance")
            .to_string_lossy()
            .into_owned();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let store = RolloutStore::open(&uri).await.unwrap();
                // A flush with no resident writer: the `noop` fast path.
                store.flush().await.unwrap();
                store.add(&[assistant_record("row-0")]).await.unwrap();
                // Now there is a writer with buffered rows: the `sealed` path.
                store.flush().await.unwrap();
            });
        });

        let mut add_samples = 0usize;
        let mut flush_outcomes: Vec<String> = Vec::new();
        for (key, _, _, value) in snapshotter.snapshot().into_vec() {
            if key.kind() != MetricKind::Histogram {
                continue;
            }
            let name = key.key().name().to_string();
            let labels: Vec<(String, String)> = key
                .key()
                .labels()
                .map(|l| (l.key().to_string(), l.value().to_string()))
                .collect();
            let count = match value {
                DebugValue::Histogram(v) => v.len(),
                _ => 0,
            };
            if name == crate::metrics::ROLLOUT_ADD_DURATION {
                add_samples += count;
                // No `result` label: an error label would double this
                // histogram's series count (one per bucket, twice) to describe
                // the latency of a rare event. Errors are counted instead.
                assert!(
                    labels.is_empty(),
                    "add latency must stay unlabelled to bound cardinality; got {labels:?}"
                );
            } else if name == crate::metrics::ROLLOUT_FLUSH_DURATION {
                assert_eq!(
                    labels.len(),
                    1,
                    "flush should carry exactly `outcome`, no result label; got {labels:?}"
                );
                let outcome = labels
                    .iter()
                    .find(|(k, _)| k == "outcome")
                    .map(|(_, v)| v.clone())
                    .expect("flush must carry an outcome label");
                for _ in 0..count {
                    flush_outcomes.push(outcome.clone());
                }
            }
        }

        assert_eq!(add_samples, 1, "one add should record exactly one sample");
        flush_outcomes.sort();
        // The two flushes took genuinely different paths; collapsing them into
        // one series is what makes the flush histogram unreadable in production,
        // since `noop` is by far the common case and is near-zero.
        assert_eq!(
            flush_outcomes,
            vec!["noop".to_string(), "sealed".to_string()],
            "flush should distinguish the no-op fast path from real sealing"
        );
    }

    /// Guards the cardinality contract: latency histograms must never carry a
    /// label whose domain is unbounded (a store URI, shard id, experiment name)
    /// or redundant with a counter (`result`).
    ///
    /// Every label combination times every bucket is a separate exported series,
    /// and in Datadog a separately-billed custom metric, so this is a cost
    /// regression test as much as a correctness one.
    #[test]
    fn latency_histograms_carry_only_bounded_labels() {
        use metrics_util::debugging::DebuggingRecorder;
        use metrics_util::MetricKind;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let dir = TempDir::new().unwrap();
        let uri = dir
            .path()
            .join("rollouts.lance")
            .to_string_lossy()
            .into_owned();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let store = RolloutStore::open(&uri).await.unwrap();
                store.add(&[assistant_record("row-0")]).await.unwrap();
                store.flush().await.unwrap();
            });
        });

        // Closed sets only. `result` is intentionally absent: it belongs on a
        // counter, where it costs one series instead of one per bucket.
        let allowed: &[(&str, &[&str])] = &[
            ("outcome", &["sealed", "noop", "fenced"]),
            (
                "phase",
                &["seal", "read", "append", "claim_epoch", "drain", "delete"],
            ),
        ];

        for (key, _, _, _) in snapshotter.snapshot().into_vec() {
            if key.kind() != MetricKind::Histogram {
                continue;
            }
            for label in key.key().labels() {
                let (k, v) = (label.key(), label.value());
                let allowed_values = allowed
                    .iter()
                    .find(|(name, _)| *name == k)
                    .map(|(_, vs)| *vs)
                    .unwrap_or_else(|| {
                        panic!(
                            "histogram {} carries unexpected label `{k}`; latency labels must \
                             come from a closed, documented set",
                            key.key().name()
                        )
                    });
                assert!(
                    allowed_values.contains(&v),
                    "histogram {} label {k}={v} is outside its documented domain {allowed_values:?}",
                    key.key().name()
                );
            }
        }
    }
}
