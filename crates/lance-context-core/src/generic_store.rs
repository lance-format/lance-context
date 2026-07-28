//! A store over a user-declared schema.
//!
//! [`GenericStore`] is the fourth store alongside the three fixed-schema ones,
//! and the only one whose columns come from the caller rather than from a
//! hard-coded `fn schema()`. It is deliberately thin: the schema comes from a
//! [`SchemaSpec`], encode/decode from [`crate::generic_codec`], and *all*
//! storage behavior from [`StorageBase`] — the same code path the built-in
//! stores use.
//!
//! That last point is the design constraint, not an implementation detail.
//! `add` and the WAL merge behave identically here to `RolloutStore`, because
//! they are literally the same functions: the resident writer, the fence retry,
//! the surgical generation drain, compaction with `defer_index_remap`, and the
//! LSM read path are all inherited, not reimplemented.
//!
//! # The schema is persisted, not re-declared
//!
//! The spec is written into the dataset's schema metadata at creation and read
//! back on open, so callers pass it once. Reopening with a *different* spec is
//! an error rather than a silent reinterpretation of existing data.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema};
use futures::TryStreamExt;
use lance::dataset::optimize::CompactionMetrics;
use lance::session::Session;
use lance::{Error as LanceError, Result as LanceResult};

use crate::generic_codec::{batch_to_rows, rows_to_batch, Row};
use crate::store::{CompactionConfig, CompactionStats};
use crate::store_base::{ListSource, StorageBase, StorageBaseOptions};
use lance_context_api::schema_spec::{SchemaSpec, ID_COLUMN};

/// Schema-metadata key holding the serialized [`SchemaSpec`], so a store can be
/// reopened without the caller re-declaring its columns.
const SCHEMA_SPEC_KEY: &str = "lance-context:schema_spec";

/// Schema-metadata key holding the store's seal-on-add mode.
///
/// Persisted for the same reason as the schema: it is a property *of the
/// store*, chosen at creation, not of whoever happens to open it. Without this
/// a store created with `seal_on_add: true` silently loses read-your-write the
/// first time it is reopened (after an LRU eviction, or a process restart) —
/// no error, no data loss, just "sometimes I can't read what I just wrote",
/// which is close to undiagnosable in production.
const SEAL_ON_ADD_KEY: &str = "lance-context:seal_on_add";

/// Configuration for opening a [`GenericStore`].
#[derive(Debug, Clone, Default)]
pub struct GenericStoreOptions {
    /// Object-store credentials/config, forwarded to Lance.
    pub storage_options: Option<HashMap<String, String>>,
    /// Stable identity of this writer instance, mapped to a MemWAL shard. Each
    /// instance owns exactly one shard, so no two contend. `None` uses a single
    /// `"default"` shard — correct for one writer, wrong for several.
    pub shard_id: Option<String>,
    /// Fold this instance's flushed generations into the base table once it has
    /// accumulated this many. `None`/`0` disables the count trigger.
    pub merge_after_generations: Option<usize>,
    /// Shared, capacity-bounded Lance session.
    pub session: Option<Arc<Session>>,
    /// Whether [`GenericStore::add`] seals before returning, making the rows it
    /// wrote immediately readable.
    ///
    /// Defaults to `false` — [`crate::RolloutStore`]'s profile: `add` is a
    /// durable WAL append, concurrent appends do not serialize behind a
    /// per-append seal, and no flushed generation is emitted per call. There is
    /// then **no read-your-write guarantee**; drive [`GenericStore::flush`]
    /// periodically, or set this to `true` to seal on every `add`.
    pub seal_on_add: bool,
}

/// A Lance-backed store over a user-declared schema.
pub struct GenericStore {
    base: StorageBase,
    spec: SchemaSpec,
    /// Arrow schema as it exists on disk. Encoding assembles against *this*,
    /// not against `spec.to_arrow()`, so a dataset whose physical field order
    /// differs still writes correctly.
    schema: Arc<Schema>,
    /// Whether `add` seals before returning. Resolved from the dataset on a
    /// reopen, from the caller on a create.
    seal_on_add: bool,
}

impl GenericStore {
    /// Create a store with `spec`, or open it if it already exists.
    ///
    /// # Errors
    ///
    /// - `spec` violates [`SchemaSpec::validate`] (no `id`, reserved names, …)
    /// - the store exists and its persisted spec differs from `spec`
    pub async fn open(
        uri: &str,
        spec: SchemaSpec,
        options: GenericStoreOptions,
    ) -> LanceResult<Self> {
        Self::open_inner(uri, Some(spec), options, true).await
    }

    /// Open an existing store, reading its schema back from the dataset.
    ///
    /// # Errors
    ///
    /// Returns [`LanceError::DatasetNotFound`] when the store does not exist,
    /// rather than creating an empty one — a mistyped name must not silently
    /// materialize a new store.
    pub async fn open_existing(uri: &str, options: GenericStoreOptions) -> LanceResult<Self> {
        Self::open_inner(uri, None, options, false).await
    }

    async fn open_inner(
        uri: &str,
        spec: Option<SchemaSpec>,
        options: GenericStoreOptions,
        create_if_missing: bool,
    ) -> LanceResult<Self> {
        if let Some(spec) = &spec {
            spec.validate()
                .map_err(|error| LanceError::from(ArrowError::SchemaError(error)))?;
        }

        // Creating needs a schema up front; opening reads it back from the
        // dataset. When no schema was supplied the store must already exist, so
        // read its schema first and hand *that* to the base — it validates the
        // key column against whatever it is given, and an empty placeholder
        // would fail that check.
        let create_schema = match &spec {
            Some(spec) => Arc::new(schema_with_spec(spec, options.seal_on_add)?),
            None if create_if_missing => {
                return Err(ArrowError::SchemaError(
                    "a schema is required to create a store".to_string(),
                )
                .into())
            }
            None => {
                let existing = StorageBase::load_with_options(
                    uri,
                    options.storage_options.clone(),
                    options.session.clone(),
                )
                .await?;
                Arc::new(Schema::from(existing.schema()))
            }
        };

        // A reopen honors the mode the store was *created* with; only a create
        // takes it from the caller. Read it before opening, since the base
        // needs it up front.
        let seal_on_add = match &spec {
            Some(_) => options.seal_on_add,
            None => seal_on_add_from_schema(&create_schema).unwrap_or(options.seal_on_add),
        };

        let base = StorageBase::open(
            uri,
            StorageBaseOptions {
                storage_options: options.storage_options,
                shard_id: options.shard_id,
                merge_after_generations: options.merge_after_generations,
                session: options.session,
                schema: create_schema,
                // Always `id`: the LSM merge key, which `SchemaSpec::validate`
                // guarantees exists as a non-nullable string.
                key_column: ID_COLUMN.to_string(),
                // User schemas are immutable in v1, so there is nothing to
                // evolve an older base table to.
                latest_schema: None,
                seal_on_put: seal_on_add,
            },
            create_if_missing,
        )
        .await?;

        let schema: Arc<Schema> = Arc::new(base.dataset.schema().into());
        let persisted = spec_from_schema(&schema)?;

        // Reopening with a different schema would reinterpret existing data.
        if let Some(requested) = spec {
            if requested != persisted {
                return Err(ArrowError::SchemaError(format!(
                    "store at '{uri}' was created with a different schema; \
                     open it without a schema to use the persisted one"
                ))
                .into());
            }
        }

        Ok(Self {
            base,
            spec: persisted,
            schema,
            seal_on_add,
        })
    }

    /// The schema this store was created with.
    #[must_use]
    pub fn spec(&self) -> &SchemaSpec {
        &self.spec
    }

    /// Whether `add` seals before returning, so its rows are immediately
    /// readable. Persisted with the store, so a reopen preserves it.
    #[must_use]
    pub fn seal_on_add(&self) -> bool {
        self.seal_on_add
    }

    /// URI of the underlying Lance dataset.
    #[must_use]
    pub fn uri(&self) -> &str {
        self.base.uri()
    }

    /// Current base dataset version.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.base.version()
    }

    /// Append rows.
    ///
    /// Rows are matched to columns by name; an undeclared key is an error, and
    /// an omitted nullable column is written as null. Visibility on return is
    /// governed by [`GenericStoreOptions::seal_on_add`].
    ///
    /// # Errors
    ///
    /// Propagates encoding failures (missing required column, type mismatch,
    /// wrong vector length) before anything is written.
    pub async fn add(&self, rows: &[Row]) -> LanceResult<u64> {
        if rows.is_empty() {
            return Ok(self.base.version());
        }
        let batch = rows_to_batch(&self.spec, self.schema.clone(), rows)?;
        self.base.put(vec![batch]).await?;
        Ok(self.base.version())
    }

    /// Read rows, newest-generation-wins by `id`.
    ///
    /// Blob columns are projected out (see [`SchemaSpec::scan_columns`]), so a
    /// list never materializes large payloads. Use [`Self::get`] with explicit
    /// columns to fetch them.
    pub async fn list(&self, limit: Option<usize>, offset: Option<usize>) -> LanceResult<Vec<Row>> {
        self.scan(None, limit, offset, &self.spec.scan_columns())
            .await
    }

    /// [`Self::list`], filtered by a SQL predicate over the store's columns.
    pub async fn list_filtered(
        &self,
        filter: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> LanceResult<Vec<Row>> {
        self.scan(Some(filter), limit, offset, &self.spec.scan_columns())
            .await
    }

    /// Fetch one row by `id`, or `None` if absent.
    ///
    /// `columns` selects what to read: `None` reads everything *except* blob
    /// columns, matching [`Self::list`]. Pass an explicit list to fetch blobs —
    /// this is the intended way to read a large payload, one row at a time.
    pub async fn get(&self, id: &str, columns: Option<&[String]>) -> LanceResult<Option<Row>> {
        let columns = columns.map_or_else(|| self.spec.scan_columns(), <[String]>::to_vec);
        // Always read `id` so a match can be confirmed.
        let mut columns = columns;
        if !columns.iter().any(|column| column == ID_COLUMN) {
            columns.push(ID_COLUMN.to_string());
        }

        let filter = format!("{ID_COLUMN} = '{}'", escape_sql_literal(id));
        let rows = self.scan(Some(&filter), Some(1), None, &columns).await?;
        Ok(rows.into_iter().next())
    }

    async fn scan(
        &self,
        filter: Option<&str>,
        limit: Option<usize>,
        offset: Option<usize>,
        columns: &[String],
    ) -> LanceResult<Vec<Row>> {
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let mut scanner = self.base.lsm_scanner().await?.project(&refs);
        if let Some(filter) = filter {
            scanner = scanner.filter(filter)?;
        }
        if limit.is_some() || offset.is_some() {
            scanner = scanner.limit(limit.unwrap_or(usize::MAX), offset);
        }

        let mut stream = scanner.try_into_stream().await?;
        let mut rows = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            rows.extend(batch_to_rows(&self.spec, &batch)?);
        }
        Ok(rows)
    }

    /// Read rows from a chosen [`ListSource`] — base table, pending WAL
    /// generations, or their union.
    pub async fn list_from(&self, source: ListSource) -> LanceResult<Vec<Row>> {
        let snapshots = match source {
            ListSource::Fragments => Vec::new(),
            ListSource::Wal | ListSource::All => self.base.wal_shard_snapshots().await?,
        };
        let columns = self.spec.scan_columns();
        let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
        let scanner = self
            .base
            .lsm_scanner_for_source(source, snapshots)
            .project(&refs);

        let mut stream = scanner.try_into_stream().await?;
        let mut rows = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            rows.extend(batch_to_rows(&self.spec, &batch)?);
        }
        Ok(rows)
    }

    /// Seal the active memtable so previously added rows become readable
    /// everywhere. A no-op when `seal_on_add` is set, since `add` already seals.
    pub async fn flush(&self) -> LanceResult<()> {
        self.base.flush().await
    }

    /// Close the resident writer, draining its background tasks. Idempotent.
    pub async fn close(&mut self) -> LanceResult<()> {
        self.base.close().await
    }

    /// Merge flushed generations into the base table once the count trigger is
    /// met. Returns how many were reclaimed.
    pub async fn maybe_merge_wal(&mut self) -> LanceResult<usize> {
        self.base.maybe_merge_own_shard().await
    }

    /// Seal, then merge **every** pending generation into the base table — the
    /// time half of the "time OR count" trigger.
    pub async fn cleanup_wal(&mut self) -> LanceResult<usize> {
        self.base.cleanup_own_shard().await
    }

    /// Generations pending merge across all shards. Read-only.
    pub async fn pending_wal_generations(&self) -> LanceResult<usize> {
        self.base.pending_wal_generations().await
    }

    /// Compact the base table's small fragments. Drive from a single external
    /// trigger, not per worker — see [`StorageBase::compact`].
    pub async fn compact(
        &mut self,
        options: Option<CompactionConfig>,
    ) -> LanceResult<CompactionMetrics> {
        self.base.compact(options).await
    }

    /// Whether the base table is fragmented enough to be worth compacting.
    #[must_use]
    pub fn should_compact(&self, config: &CompactionConfig) -> bool {
        self.base.should_compact(config)
    }

    /// Compaction statistics for the base table.
    #[must_use]
    pub fn compaction_stats(&self) -> CompactionStats {
        self.base.compaction_stats()
    }

    /// Build a ZoneMap scalar index on `id`. Idempotent.
    pub async fn create_id_index(&mut self) -> LanceResult<()> {
        self.base.create_key_zonemap_index().await
    }

    /// Row count of the base table. Excludes rows still in unmerged
    /// generations or buffered in the writer.
    pub async fn count_base_rows(&self) -> LanceResult<usize> {
        self.base.dataset.count_rows(None).await
    }
}

/// Attach the serialized spec and the seal mode to the Arrow schema, so both
/// round-trip on open.
fn schema_with_spec(spec: &SchemaSpec, seal_on_add: bool) -> Result<Schema, ArrowError> {
    let schema = spec.to_arrow()?;
    let encoded = serde_json::to_string(spec)
        .map_err(|error| ArrowError::SchemaError(format!("could not serialize schema: {error}")))?;

    let mut metadata = schema.metadata().clone();
    metadata.insert(SCHEMA_SPEC_KEY.to_string(), encoded);
    metadata.insert(SEAL_ON_ADD_KEY.to_string(), seal_on_add.to_string());
    Ok(schema.with_metadata(metadata))
}

/// Read the persisted seal mode. `None` for a store written before this was
/// persisted, which then falls back to the caller's option.
fn seal_on_add_from_schema(schema: &Schema) -> Option<bool> {
    schema
        .metadata()
        .get(SEAL_ON_ADD_KEY)
        .and_then(|raw| raw.parse().ok())
}

/// Read the spec back out of a dataset's schema metadata.
fn spec_from_schema(schema: &Schema) -> Result<SchemaSpec, ArrowError> {
    let encoded = schema.metadata().get(SCHEMA_SPEC_KEY).ok_or_else(|| {
        ArrowError::SchemaError(
            "dataset has no stored schema spec: it was not created as a generic store".to_string(),
        )
    })?;
    serde_json::from_str(encoded)
        .map_err(|error| ArrowError::SchemaError(format!("stored schema spec is invalid: {error}")))
}

/// Escape single quotes for a SQL string literal.
fn escape_sql_literal(value: &str) -> String {
    value.replace('\'', "''")
}

/// Row batches, for callers that already have Arrow data.
impl GenericStore {
    /// Append pre-built [`RecordBatch`]es, bypassing row encoding.
    ///
    /// For bulk ingest where the caller already holds Arrow data. Each batch
    /// must match the store's schema exactly.
    ///
    /// # Errors
    ///
    /// Returns an error if a batch's schema differs from the store's.
    pub async fn add_batches(&self, batches: Vec<RecordBatch>) -> LanceResult<u64> {
        if batches.is_empty() {
            return Ok(self.base.version());
        }
        for batch in &batches {
            if batch.schema().fields() != self.schema.fields() {
                return Err(ArrowError::SchemaError(
                    "record batch schema does not match the store schema".to_string(),
                )
                .into());
            }
        }
        self.base.put(batches).await?;
        Ok(self.base.version())
    }
}

#[cfg(test)]
// `GenericStore` owns a `Dataset` and is not `Debug`, so `expect_err` (which
// formats the Ok value) is unavailable; `.err().expect(..)` is the alternative.
#[allow(clippy::err_expect)]
mod tests {
    use super::*;
    use lance_context_api::schema_spec::{ColumnSpec, ColumnType};
    use serde_json::json;
    use tempfile::TempDir;

    fn spec() -> SchemaSpec {
        SchemaSpec::new(vec![
            (
                ID_COLUMN.to_string(),
                ColumnSpec::required(ColumnType::String { large: false }),
            ),
            (
                "user_id".to_string(),
                ColumnSpec::new(ColumnType::String { large: false }),
            ),
            ("score".to_string(), ColumnSpec::new(ColumnType::Float32)),
            (
                "payload".to_string(),
                ColumnSpec::new(ColumnType::Binary { blob: true }),
            ),
        ])
    }

    fn row(value: serde_json::Value) -> Row {
        value.as_object().unwrap().clone()
    }

    fn sealing() -> GenericStoreOptions {
        // Seal on add so each test reads its own writes without an explicit
        // flush; the deferred default is exercised separately.
        GenericStoreOptions {
            seal_on_add: true,
            ..Default::default()
        }
    }

    #[test]
    fn user_schema_round_trips_add_and_read() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = GenericStore::open(&uri, spec(), sealing()).await.unwrap();
            store
                .add(&[
                    row(json!({"id": "r1", "user_id": "u1", "score": 0.5})),
                    row(json!({"id": "r2", "user_id": "u2"})),
                ])
                .await
                .unwrap();

            let rows = store.list(None, None).await.unwrap();
            assert_eq!(rows.len(), 2);

            let found = store.get("r1", None).await.unwrap().unwrap();
            assert_eq!(found["user_id"], json!("u1"));
            assert!(store.get("missing", None).await.unwrap().is_none());
        });
    }

    #[test]
    fn schema_is_persisted_and_reopens_without_being_redeclared() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            {
                let store = GenericStore::open(&uri, spec(), sealing()).await.unwrap();
                store.add(&[row(json!({"id": "r1"}))]).await.unwrap();
            }

            let reopened = GenericStore::open_existing(&uri, sealing()).await.unwrap();
            assert_eq!(reopened.spec(), &spec());
            assert_eq!(reopened.list(None, None).await.unwrap().len(), 1);
        });
    }

    #[test]
    fn seal_mode_survives_a_reopen() {
        // The mode is a property of the store, not of whoever opens it. Before
        // it was persisted, a reopen silently reverted to the caller's default
        // and the store lost read-your-write with no error.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            {
                let store = GenericStore::open(&uri, spec(), sealing()).await.unwrap();
                assert!(store.seal_on_add());
            }

            // Reopen with the *opposite* default: the persisted value wins.
            let reopened = GenericStore::open_existing(&uri, GenericStoreOptions::default())
                .await
                .unwrap();
            assert!(
                reopened.seal_on_add(),
                "a store created with seal_on_add must keep it across a reopen"
            );

            // And it is behavioral, not just a flag: the write is visible
            // without an explicit flush.
            reopened.add(&[row(json!({"id": "r1"}))]).await.unwrap();
            assert_eq!(reopened.list(None, None).await.unwrap().len(), 1);
        });
    }

    #[test]
    fn deferred_seal_mode_also_survives_a_reopen() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            {
                GenericStore::open(&uri, spec(), GenericStoreOptions::default())
                    .await
                    .unwrap();
            }

            // Reopen with the opposite default; the persisted `false` wins.
            let reopened = GenericStore::open_existing(&uri, sealing()).await.unwrap();
            assert!(!reopened.seal_on_add());

            reopened.add(&[row(json!({"id": "r1"}))]).await.unwrap();
            assert_eq!(
                reopened.list(None, None).await.unwrap().len(),
                0,
                "a deferred-seal store must not start sealing after a reopen"
            );
            reopened.flush().await.unwrap();
            assert_eq!(reopened.list(None, None).await.unwrap().len(), 1);
        });
    }

    #[test]
    fn reopening_with_a_conflicting_schema_is_rejected() {
        // Silently reinterpreting existing data under a new schema is the
        // failure mode worth preventing here.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            GenericStore::open(&uri, spec(), sealing()).await.unwrap();

            let mut other = spec();
            other
                .columns
                .push(("extra".to_string(), ColumnSpec::new(ColumnType::Int64)));
            let err = GenericStore::open(&uri, other, sealing())
                .await
                .err()
                .expect("conflicting schema must be rejected");
            assert!(err.to_string().contains("different schema"), "{err}");
        });
    }

    #[test]
    fn invalid_schema_is_rejected_before_anything_is_created() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // No `id` column.
            let bad = SchemaSpec::new(vec![(
                "name".to_string(),
                ColumnSpec::new(ColumnType::String { large: false }),
            )]);
            let err = GenericStore::open(&uri, bad, sealing())
                .await
                .err()
                .expect("invalid schema must be rejected");
            assert!(err.to_string().contains("must declare an 'id'"), "{err}");
        });
    }

    #[test]
    fn blob_columns_are_excluded_from_list_but_fetchable_per_row() {
        // The cost model for large payloads: bulk reads never materialize them,
        // point reads can ask for them explicitly.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = GenericStore::open(&uri, spec(), sealing()).await.unwrap();
            store
                .add(&[row(json!({"id": "r1", "payload": [1, 2, 3]}))])
                .await
                .unwrap();

            let listed = store.list(None, None).await.unwrap();
            assert!(
                !listed[0].contains_key("payload"),
                "list must not materialize blob columns"
            );

            let fetched = store
                .get("r1", Some(&["payload".to_string()]))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(fetched["payload"], json!("AQID"));
        });
    }

    #[test]
    fn large_blob_round_trips_and_survives_wal_merge() {
        // Multi-megabyte inline payloads are the point of this store, so pin
        // that they survive both the write path and a WAL merge.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut store = GenericStore::open(&uri, spec(), sealing()).await.unwrap();
            let payload: Vec<u8> = (0..4 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
            store
                .add(&[row(json!({"id": "big", "payload": payload}))])
                .await
                .unwrap();

            store.cleanup_wal().await.unwrap();
            assert_eq!(store.pending_wal_generations().await.unwrap(), 0);

            let fetched = store
                .get("big", Some(&["payload".to_string()]))
                .await
                .unwrap()
                .unwrap();
            use base64::Engine;
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(fetched["payload"].as_str().unwrap())
                .unwrap();
            assert_eq!(decoded.len(), 4 * 1024 * 1024);
        });
    }

    #[test]
    fn deferred_seal_defers_visibility_like_the_rollout_store() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = GenericStore::open(&uri, spec(), GenericStoreOptions::default())
                .await
                .unwrap();
            store.add(&[row(json!({"id": "r1"}))]).await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 0);

            store.flush().await.unwrap();
            assert_eq!(store.list(None, None).await.unwrap().len(), 1);
        });
    }

    #[test]
    fn wal_generations_merge_into_the_base_table() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut store = GenericStore::open(&uri, spec(), sealing()).await.unwrap();
            for i in 0..3 {
                store
                    .add(&[row(json!({"id": format!("r{i}")}))])
                    .await
                    .unwrap();
            }
            assert!(store.pending_wal_generations().await.unwrap() > 0);

            let reclaimed = store.cleanup_wal().await.unwrap();
            assert!(reclaimed > 0);
            assert_eq!(store.pending_wal_generations().await.unwrap(), 0);
            assert_eq!(store.count_base_rows().await.unwrap(), 3);
            assert_eq!(store.list(None, None).await.unwrap().len(), 3);
        });
    }

    #[test]
    fn duplicate_ids_dedup_to_the_newest_write() {
        // The LSM merge key is `id`, so a re-added id supersedes the old row.
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = GenericStore::open(&uri, spec(), sealing()).await.unwrap();
            store
                .add(&[row(json!({"id": "r1", "user_id": "first"}))])
                .await
                .unwrap();
            store
                .add(&[row(json!({"id": "r1", "user_id": "second"}))])
                .await
                .unwrap();

            let rows = store.list(None, None).await.unwrap();
            assert_eq!(rows.len(), 1, "id is the merge key, so rows dedup");
            assert_eq!(rows[0]["user_id"], json!("second"));
        });
    }

    #[test]
    fn filters_and_paging_work_over_user_columns() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = GenericStore::open(&uri, spec(), sealing()).await.unwrap();
            let rows: Vec<Row> = (0..6)
                .map(|i| {
                    row(json!({
                        "id": format!("r{i}"),
                        "user_id": if i % 2 == 0 { "even" } else { "odd" },
                        "score": i as f32,
                    }))
                })
                .collect();
            store.add(&rows).await.unwrap();

            let evens = store
                .list_filtered("user_id = 'even'", None, None)
                .await
                .unwrap();
            assert_eq!(evens.len(), 3);
            assert!(evens.iter().all(|row| row["user_id"] == json!("even")));

            let page = store.list(Some(2), Some(1)).await.unwrap();
            assert_eq!(page.len(), 2);
        });
    }

    #[test]
    fn open_existing_does_not_create() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("absent").to_string_lossy().to_string();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let err = GenericStore::open_existing(&uri, sealing())
                .await
                .err()
                .expect("open_existing must not create");
            assert!(matches!(err, LanceError::DatasetNotFound { .. }), "{err}");
        });
    }
}
