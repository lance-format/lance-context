//! Persistent stats table for the control-plane master.
//!
//! Mirrors [`lance_context_core::RolloutRegistry`] in shape and idioms: a single
//! small Lance dataset (`_stats.rollout.lance`), one row per experiment, written
//! delete-then-append for idempotency. Where the registry answers *which* stores
//! exist, this table caches the *periodically-scanned metrics* of each store so
//! the UI list endpoint can render row/fragment counts without opening every
//! dataset on each request.
//!
//! Master replicas coordinate a single writer through the configured task-store
//! backend; the data-plane never touches this table.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};

use lance_context_api::ExperimentSummary;

/// One row of scanned experiment stats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatRow {
    pub name: String,
    pub uri: String,
    pub row_count: i64,
    pub fragment_count: i64,
    pub last_updated: i64,
    pub pending_wal_generations: i64,
    /// Unix ms of the last successful compaction, or a negative sentinel
    /// (`-1`) meaning "never", since the column is stored non-nullable.
    pub last_compaction: i64,
    pub total_compactions: i64,
    pub scanned_at: i64,
}

impl StatRow {
    /// Sentinel stored in `last_compaction` when no compaction has run.
    pub const NO_COMPACTION: i64 = -1;

    /// Convert to the wire DTO, mapping the `-1` sentinel back to `None`.
    #[must_use]
    pub fn into_summary(self) -> ExperimentSummary {
        ExperimentSummary {
            name: self.name,
            uri: self.uri,
            row_count: self.row_count,
            fragment_count: self.fragment_count,
            last_updated: self.last_updated,
            pending_wal_generations: self.pending_wal_generations,
            last_compaction: (self.last_compaction >= 0).then_some(self.last_compaction),
            total_compactions: self.total_compactions,
            scanned_at: self.scanned_at,
        }
    }
}

/// Durable directory of per-experiment stats, backed by a single Lance dataset.
///
/// Mutations (`upsert`, `remove`) take `&mut self` and must be serialized by the
/// caller (the master wraps this in a lock). Reads take `&self`.
pub struct StatsStore {
    dataset: Dataset,
    uri: String,
    storage_options: Option<HashMap<String, String>>,
}

fn stats_schema() -> Schema {
    Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("uri", DataType::Utf8, false),
        Field::new("row_count", DataType::Int64, false),
        Field::new("fragment_count", DataType::Int64, false),
        Field::new("last_updated", DataType::Int64, false),
        Field::new("pending_wal_generations", DataType::Int64, false),
        Field::new("last_compaction", DataType::Int64, false),
        Field::new("total_compactions", DataType::Int64, false),
        Field::new("scanned_at", DataType::Int64, false),
    ])
}

impl StatsStore {
    /// Open the stats dataset at `uri`, creating an empty one if absent.
    pub async fn open_or_create(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> LanceResult<Self> {
        let dataset = match Self::load(uri, storage_options.clone()).await {
            Ok(dataset) => dataset,
            Err(LanceError::DatasetNotFound { .. }) => {
                Self::create(uri, storage_options.clone()).await?
            }
            Err(err) => return Err(err),
        };
        Ok(Self {
            dataset,
            uri: uri.to_string(),
            storage_options,
        })
    }

    async fn load(
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

    async fn create(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> LanceResult<Dataset> {
        let schema = Arc::new(stats_schema());
        let empty = RecordBatch::new_empty(schema.clone());
        let batches = RecordBatchIterator::new(
            vec![Ok::<RecordBatch, ArrowError>(empty)].into_iter(),
            schema.clone(),
        );
        let params = Self::write_params(WriteMode::Create, storage_options);
        Dataset::write(batches, uri, Some(params)).await
    }

    fn write_params(
        mode: WriteMode,
        storage_options: Option<HashMap<String, String>>,
    ) -> WriteParams {
        let mut params = WriteParams {
            mode,
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
        params
    }

    fn row_to_batch(row: &StatRow) -> LanceResult<RecordBatch> {
        let schema = Arc::new(stats_schema());
        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![row.name.as_str()])),
                Arc::new(StringArray::from(vec![row.uri.as_str()])),
                Arc::new(Int64Array::from(vec![row.row_count])),
                Arc::new(Int64Array::from(vec![row.fragment_count])),
                Arc::new(Int64Array::from(vec![row.last_updated])),
                Arc::new(Int64Array::from(vec![row.pending_wal_generations])),
                Arc::new(Int64Array::from(vec![row.last_compaction])),
                Arc::new(Int64Array::from(vec![row.total_compactions])),
                Arc::new(Int64Array::from(vec![row.scanned_at])),
            ],
        )?)
    }

    /// Insert or replace the stats row for `row.name` (delete-then-append,
    /// idempotent). Callers must serialize mutations.
    pub async fn upsert(&mut self, row: &StatRow) -> LanceResult<()> {
        self.dataset.checkout_latest().await?;
        self.delete_row(&row.name).await?;
        let batch = Self::row_to_batch(row)?;
        let schema = Arc::new(stats_schema());
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        let params = Self::write_params(WriteMode::Append, self.storage_options.clone());
        self.dataset.append(reader, Some(params)).await?;
        Ok(())
    }

    /// Remove the stats row for `name`, if present. No-op when absent.
    pub async fn remove(&mut self, name: &str) -> LanceResult<()> {
        self.dataset.checkout_latest().await?;
        self.delete_row(name).await
    }

    async fn delete_row(&mut self, name: &str) -> LanceResult<()> {
        let escaped = name.replace('\'', "''");
        self.dataset
            .delete(&format!("name = '{}'", escaped))
            .await?;
        Ok(())
    }

    /// Fetch a single experiment's stats row by name.
    pub async fn get(&mut self, name: &str) -> LanceResult<Option<StatRow>> {
        self.dataset.checkout_latest().await?;
        let escaped = name.replace('\'', "''");
        let mut scanner = self.dataset.scan();
        scanner.filter(&format!("name = '{}'", escaped))?;
        scanner.limit(Some(1), None)?;
        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            let rows = Self::batch_to_rows(&batch)?;
            if let Some(first) = rows.into_iter().next() {
                return Ok(Some(first));
            }
        }
        Ok(None)
    }

    /// Total number of rows matching an optional case-sensitive substring on
    /// `name`, ignoring pagination.
    pub async fn count(&mut self, search: Option<&str>) -> LanceResult<i64> {
        self.dataset.checkout_latest().await?;
        let mut scanner = self.dataset.scan();
        scanner.project(&["name"])?;
        if let Some(q) = search {
            scanner.filter(&Self::like_filter(q))?;
        }
        let mut stream = scanner.try_into_stream().await?;
        let mut total = 0i64;
        while let Some(batch) = stream.try_next().await? {
            total += batch.num_rows() as i64;
        }
        Ok(total)
    }

    /// List stats rows, optionally filtered by a `name` substring, ordered by
    /// name, with `limit`/`offset` pagination applied in memory.
    pub async fn list(
        &mut self,
        search: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> LanceResult<Vec<StatRow>> {
        self.dataset.checkout_latest().await?;
        let mut scanner = self.dataset.scan();
        if let Some(q) = search {
            scanner.filter(&Self::like_filter(q))?;
        }
        let mut stream = scanner.try_into_stream().await?;
        let mut rows = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            rows.extend(Self::batch_to_rows(&batch)?);
        }
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(rows.into_iter().skip(offset).take(limit).collect())
    }

    fn like_filter(search: &str) -> String {
        let escaped = search.replace('\'', "''");
        format!("name LIKE '%{}%'", escaped)
    }

    fn batch_to_rows(batch: &RecordBatch) -> LanceResult<Vec<StatRow>> {
        let names = Self::str_col(batch, "name")?;
        let uris = Self::str_col(batch, "uri")?;
        let row_count = Self::i64_col(batch, "row_count")?;
        let fragment_count = Self::i64_col(batch, "fragment_count")?;
        let last_updated = Self::i64_col(batch, "last_updated")?;
        let pending = Self::i64_col(batch, "pending_wal_generations")?;
        let last_compaction = Self::i64_col(batch, "last_compaction")?;
        let total_compactions = Self::i64_col(batch, "total_compactions")?;
        let scanned_at = Self::i64_col(batch, "scanned_at")?;
        let mut out = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            out.push(StatRow {
                name: names.value(i).to_string(),
                uri: uris.value(i).to_string(),
                row_count: row_count.value(i),
                fragment_count: fragment_count.value(i),
                last_updated: last_updated.value(i),
                pending_wal_generations: pending.value(i),
                last_compaction: last_compaction.value(i),
                total_compactions: total_compactions.value(i),
                scanned_at: scanned_at.value(i),
            });
        }
        Ok(out)
    }

    fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> LanceResult<&'a StringArray> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                LanceError::from(ArrowError::InvalidArgumentError(format!(
                    "stats column '{}' missing or not Utf8",
                    name
                )))
            })
    }

    fn i64_col<'a>(batch: &'a RecordBatch, name: &str) -> LanceResult<&'a Int64Array> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| {
                LanceError::from(ArrowError::InvalidArgumentError(format!(
                    "stats column '{}' missing or not Int64",
                    name
                )))
            })
    }

    /// The stats dataset URI.
    #[must_use]
    pub fn uri(&self) -> &str {
        &self.uri
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample(name: &str, rows: i64) -> StatRow {
        StatRow {
            name: name.to_string(),
            uri: format!("/data/{name}.rollout.lance"),
            row_count: rows,
            fragment_count: 3,
            last_updated: 1_700_000_000_000,
            pending_wal_generations: 0,
            last_compaction: StatRow::NO_COMPACTION,
            total_compactions: 0,
            scanned_at: 1_700_000_001_000,
        }
    }

    async fn new_store(dir: &TempDir) -> StatsStore {
        let uri = dir.path().join("_stats.rollout.lance");
        StatsStore::open_or_create(uri.to_str().unwrap(), None)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn upsert_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let mut s = new_store(&dir).await;
        s.upsert(&sample("exp-a", 10)).await.unwrap();
        s.upsert(&sample("exp-a", 20)).await.unwrap();
        let rows = s.list(None, 100, 0).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].row_count, 20);
    }

    #[tokio::test]
    async fn get_and_remove() {
        let dir = TempDir::new().unwrap();
        let mut s = new_store(&dir).await;
        assert!(s.get("missing").await.unwrap().is_none());
        s.upsert(&sample("exp-b", 5)).await.unwrap();
        assert_eq!(s.get("exp-b").await.unwrap().unwrap().row_count, 5);
        s.remove("exp-b").await.unwrap();
        assert!(s.get("exp-b").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn search_like_and_pagination() {
        let dir = TempDir::new().unwrap();
        let mut s = new_store(&dir).await;
        for i in 0..5 {
            s.upsert(&sample(&format!("alpha-{i}"), i)).await.unwrap();
        }
        s.upsert(&sample("beta-0", 99)).await.unwrap();

        // LIKE substring narrows to the alpha family.
        assert_eq!(s.count(Some("alpha")).await.unwrap(), 5);
        let page = s.list(Some("alpha"), 2, 0).await.unwrap();
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].name, "alpha-0");
        let page2 = s.list(Some("alpha"), 2, 2).await.unwrap();
        assert_eq!(page2[0].name, "alpha-2");

        // Full listing plus the beta row.
        assert_eq!(s.count(None).await.unwrap(), 6);
    }

    #[tokio::test]
    async fn survives_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let mut s = new_store(&dir).await;
            s.upsert(&sample("persist", 7)).await.unwrap();
        }
        let mut s = new_store(&dir).await;
        assert_eq!(s.get("persist").await.unwrap().unwrap().row_count, 7);
    }

    #[tokio::test]
    async fn no_compaction_sentinel_maps_to_none() {
        let row = sample("x", 1);
        assert_eq!(row.into_summary().last_compaction, None);
        let mut with = sample("y", 1);
        with.last_compaction = 123;
        assert_eq!(with.into_summary().last_compaction, Some(123));
    }
}
