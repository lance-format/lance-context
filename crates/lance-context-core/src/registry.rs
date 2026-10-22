//! Persistent directory of rollout stores.
//!
//! # Why this exists
//!
//! With `experiment_name` used as the physical partition key, a deployment may
//! hold **tens or hundreds of thousands** of independent rollout datasets — one
//! per experiment. The server can no longer keep every opened store resident in
//! memory (it uses a bounded LRU), so "is this store in the in-memory map?" is
//! no longer a valid existence check, and `list` can no longer enumerate stores
//! by walking the cache.
//!
//! [`RolloutRegistry`] is a single, small Lance dataset — one row per rollout
//! store (`name`, `uri`, `created_at`) — that serves as the durable source of
//! truth for *which* stores exist. It is consulted on a cache miss to decide
//! whether to lazily open a store (vs. return 404), and it backs the `list`
//! endpoint without touching object storage for every experiment.
//!
//! It deliberately holds only cheap directory metadata, never rollout rows.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use chrono::Utc;
use futures::TryStreamExt;
use lance::dataset::cleanup::RemovalStats;
use lance::dataset::optimize::{compact_files, CompactionMetrics, CompactionOptions};
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};

/// One entry in the rollout-store directory.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RegistryEntry {
    /// Logical store name (the experiment name); unique key.
    pub name: String,
    /// Physical dataset URI, as produced by the server's `rollout_uri`.
    pub uri: String,
    /// Creation time, Unix milliseconds.
    pub created_at: i64,
}

/// A directory of stores: `name -> (uri, created_at)`.
///
/// The name says which stores exist; the dataset on object storage is the
/// data. Implementations must be safe to share across tasks (`&self`).
#[allow(
    clippy::double_must_use,
    reason = "async_trait adds must_use to boxed futures"
)]
#[async_trait::async_trait]
pub trait StoreRegistry: Send + Sync {
    /// Whether a store named `name` exists.
    async fn contains(&self, name: &str) -> LanceResult<bool>;
    /// One directory entry, or `None` when absent.
    async fn get(&self, name: &str) -> LanceResult<Option<RegistryEntry>>;
    /// Every entry, in unspecified order.
    async fn list(&self) -> LanceResult<Vec<RegistryEntry>>;
    /// Insert or replace the entry for `name`. Idempotent.
    async fn upsert(&self, name: &str, uri: &str) -> LanceResult<()>;
    /// Remove the entry for `name`. No-op when absent.
    async fn remove(&self, name: &str) -> LanceResult<()>;
    /// Insert every entry whose name is not already registered; returns how
    /// many were inserted. Existing rows are left unchanged.
    async fn insert_missing(&self, entries: &[(String, String)]) -> LanceResult<usize>;
    /// The Lance table behind this registry, if any, so its owner can run
    /// compaction and version pruning on it. `None` for backends that need no
    /// maintenance (etcd).
    fn lance_table(&self) -> Option<&tokio::sync::Mutex<RolloutRegistry>> {
        None
    }
    /// For downcasting to a concrete backend (the master's migration routes
    /// need the two halves of a [`MirroredRegistry`]).
    fn as_any(&self) -> &dyn std::any::Any;
}

/// [`StoreRegistry`] over a [`RolloutRegistry`]. The Lance handle must check
/// out the latest manifest before every read, hence `&mut self` inside and a
/// mutex here; every call is one manifest read on object storage.
pub struct LanceRegistry(pub tokio::sync::Mutex<RolloutRegistry>);

impl LanceRegistry {
    pub fn new(inner: RolloutRegistry) -> Self {
        Self(tokio::sync::Mutex::new(inner))
    }
}

#[async_trait::async_trait]
impl StoreRegistry for LanceRegistry {
    async fn contains(&self, name: &str) -> LanceResult<bool> {
        self.0.lock().await.contains(name).await
    }
    async fn get(&self, name: &str) -> LanceResult<Option<RegistryEntry>> {
        self.0.lock().await.get(name).await
    }
    async fn list(&self) -> LanceResult<Vec<RegistryEntry>> {
        self.0.lock().await.list().await
    }
    async fn upsert(&self, name: &str, uri: &str) -> LanceResult<()> {
        self.0.lock().await.upsert(name, uri).await
    }
    async fn remove(&self, name: &str) -> LanceResult<()> {
        self.0.lock().await.remove(name).await
    }
    async fn insert_missing(&self, entries: &[(String, String)]) -> LanceResult<usize> {
        self.0.lock().await.insert_missing(entries).await
    }
    fn lance_table(&self) -> Option<&tokio::sync::Mutex<RolloutRegistry>> {
        Some(&self.0)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Lance primary with a versioned etcd mirror. Failed mirror writes do not
/// fail the primary mutation; a complete snapshot reconciliation repairs them.
/// Reverse mirroring is deliberately unsupported.
pub struct MirroredRegistry {
    pub primary: Arc<dyn StoreRegistry>,
    pub mirror: Arc<dyn StoreRegistry>,
    pub label: &'static str,
}

impl MirroredRegistry {
    async fn sync_name(&self, name: &str) -> LanceResult<()> {
        let (source, target) = migration_pair(&*self.primary, &*self.mirror)?;
        let (uri, version, entry) = {
            let mut source = source.0.lock().await;
            let entry = source.get(name).await?;
            (source.uri.clone(), source.dataset.version().version, entry)
        };
        target
            .apply_source(&uri, version, name, entry.as_ref())
            .await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl StoreRegistry for MirroredRegistry {
    async fn contains(&self, name: &str) -> LanceResult<bool> {
        self.primary.contains(name).await
    }
    async fn get(&self, name: &str) -> LanceResult<Option<RegistryEntry>> {
        self.primary.get(name).await
    }
    async fn list(&self) -> LanceResult<Vec<RegistryEntry>> {
        self.primary.list().await
    }
    async fn upsert(&self, name: &str, uri: &str) -> LanceResult<()> {
        self.primary.upsert(name, uri).await?;
        if let Err(error) = self.sync_name(name).await {
            tracing::warn!(registry = self.label, name, %error, "registry mirror upsert failed");
        }
        Ok(())
    }
    async fn remove(&self, name: &str) -> LanceResult<()> {
        self.primary.remove(name).await?;
        if let Err(error) = self.sync_name(name).await {
            tracing::warn!(registry = self.label, name, %error, "registry mirror remove failed");
        }
        Ok(())
    }
    async fn insert_missing(&self, entries: &[(String, String)]) -> LanceResult<usize> {
        let inserted = self.primary.insert_missing(entries).await?;
        if let Err(error) = backfill_registry(&*self.primary, &*self.mirror).await {
            tracing::warn!(registry = self.label, %error, "registry mirror reconciliation failed");
        }
        Ok(inserted)
    }
    fn lance_table(&self) -> Option<&tokio::sync::Mutex<RolloutRegistry>> {
        self.primary.lance_table()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn migration_pair<'a>(
    from: &'a dyn StoreRegistry,
    to: &'a dyn StoreRegistry,
) -> LanceResult<(&'a LanceRegistry, &'a crate::EtcdRegistry)> {
    match (
        from.as_any().downcast_ref::<LanceRegistry>(),
        to.as_any().downcast_ref::<crate::EtcdRegistry>(),
    ) {
        (Some(source), Some(target)) => Ok((source, target)),
        _ => Err(LanceError::io(
            "registry reconciliation supports only Lance -> etcd",
        )),
    }
}

/// Reconcile one consistent Lance snapshot, including changed values and
/// deletions. The source version fences delayed older snapshots and point
/// updates at the destination. The return value counts changed entries.
pub async fn backfill_registry(
    from: &dyn StoreRegistry,
    to: &dyn StoreRegistry,
) -> LanceResult<usize> {
    let (source, target) = migration_pair(from, to)?;
    let (uri, version, entries) = {
        let mut source = source.0.lock().await;
        let entries = source.list().await?;
        (
            source.uri.clone(),
            source.dataset.version().version,
            entries,
        )
    };
    target.reconcile_source(&uri, version, &entries).await
}

#[derive(Debug, serde::Serialize)]
pub struct RegistryMismatch {
    pub name: String,
    pub primary: RegistryEntry,
    pub mirror: RegistryEntry,
}

/// A metadata comparison, not authorization to switch live writers.
#[derive(Debug, serde::Serialize)]
pub struct RegistryDiff {
    pub only_in_primary: Vec<String>,
    pub only_in_mirror: Vec<String>,
    pub mismatched: Vec<RegistryMismatch>,
}

impl RegistryDiff {
    pub fn is_empty(&self) -> bool {
        self.only_in_primary.is_empty()
            && self.only_in_mirror.is_empty()
            && self.mismatched.is_empty()
    }
}

pub async fn diff_registries(
    a: &dyn StoreRegistry,
    b: &dyn StoreRegistry,
) -> LanceResult<RegistryDiff> {
    let a: std::collections::BTreeMap<_, _> = a
        .list()
        .await?
        .into_iter()
        .map(|e| (e.name.clone(), e))
        .collect();
    let b: std::collections::BTreeMap<_, _> = b
        .list()
        .await?
        .into_iter()
        .map(|e| (e.name.clone(), e))
        .collect();
    Ok(RegistryDiff {
        only_in_primary: a.keys().filter(|n| !b.contains_key(*n)).cloned().collect(),
        only_in_mirror: b.keys().filter(|n| !a.contains_key(*n)).cloned().collect(),
        mismatched: a
            .iter()
            .filter_map(|(name, primary)| {
                b.get(name)
                    .filter(|mirror| *mirror != primary)
                    .map(|mirror| RegistryMismatch {
                        name: name.clone(),
                        primary: primary.clone(),
                        mirror: mirror.clone(),
                    })
            })
            .collect(),
    })
}

/// Durable directory of rollout stores, backed by a single Lance dataset.
///
/// All operations take `&mut self` because Lance dataset handles are snapshots:
/// reads and writes first check out the latest manifest so commits made by
/// another process are visible. Callers are expected to serialize access (the
/// server and master wrap this in a lock).
pub struct RolloutRegistry {
    dataset: Dataset,
    uri: String,
    storage_options: Option<HashMap<String, String>>,
}

fn registry_schema() -> Schema {
    Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("uri", DataType::Utf8, false),
        Field::new("created_at", DataType::Int64, false),
    ])
}

impl RolloutRegistry {
    /// Open the registry dataset at `uri`, creating an empty one if it does not
    /// yet exist. Idempotent across process restarts.
    pub async fn open_or_create(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> LanceResult<Self> {
        let dataset = match Self::load(uri, storage_options.clone()).await {
            Ok(dataset) => dataset,
            Err(LanceError::DatasetNotFound { .. }) => {
                Self::create_or_load(uri, storage_options.clone()).await?
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

    /// Create the registry, or load it if another caller won the creation race.
    async fn create_or_load(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> LanceResult<Dataset> {
        match Self::create(uri, storage_options.clone()).await {
            Ok(dataset) => Ok(dataset),
            Err(LanceError::DatasetAlreadyExists { .. }) => Self::load(uri, storage_options).await,
            Err(err) => Err(err),
        }
    }

    async fn create(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> LanceResult<Dataset> {
        let schema = Arc::new(registry_schema());
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

    /// Insert or replace the directory entry for `name`.
    ///
    /// Implemented as delete-same-name-then-append so it is **idempotent**: a
    /// `create` retried after a crash (dataset on disk, registry row possibly
    /// present) converges to exactly one row. Callers must serialize mutations.
    pub async fn upsert(&mut self, name: &str, uri: &str) -> LanceResult<()> {
        self.reload().await?;
        self.delete_row(name).await?;
        let schema = Arc::new(registry_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![name])),
                Arc::new(StringArray::from(vec![uri])),
                Arc::new(Int64Array::from(vec![Utc::now().timestamp_millis()])),
            ],
        )?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        let params = Self::write_params(WriteMode::Append, self.storage_options.clone());
        self.dataset.append(reader, Some(params)).await?;
        Ok(())
    }

    /// Insert every entry whose name is not already registered in one append.
    ///
    /// This is intended for migration/backfill jobs where many pre-existing
    /// rollout datasets need directory entries. Existing rows are left
    /// unchanged, duplicate names in `entries` are ignored, and the return
    /// value is the number of rows inserted.
    pub async fn insert_missing(&mut self, entries: &[(String, String)]) -> LanceResult<usize> {
        if entries.is_empty() {
            return Ok(0);
        }

        self.reload().await?;
        let mut known: HashSet<String> = self
            .list()
            .await?
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        let missing: Vec<&(String, String)> = entries
            .iter()
            .filter(|(name, _)| known.insert(name.clone()))
            .collect();
        if missing.is_empty() {
            return Ok(0);
        }

        let schema = Arc::new(registry_schema());
        let created_at = Utc::now().timestamp_millis();
        let names: Vec<&str> = missing.iter().map(|(name, _)| name.as_str()).collect();
        let uris: Vec<&str> = missing.iter().map(|(_, uri)| uri.as_str()).collect();
        let created = vec![created_at; missing.len()];
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(names)),
                Arc::new(StringArray::from(uris)),
                Arc::new(Int64Array::from(created)),
            ],
        )?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        let params = Self::write_params(WriteMode::Append, self.storage_options.clone());
        self.dataset.append(reader, Some(params)).await?;
        Ok(missing.len())
    }

    /// Remove the directory entry for `name`, if present. No-op when absent.
    pub async fn remove(&mut self, name: &str) -> LanceResult<()> {
        self.reload().await?;
        self.delete_row(name).await
    }

    async fn delete_row(&mut self, name: &str) -> LanceResult<()> {
        let escaped = name.replace('\'', "''");
        self.dataset
            .delete(&format!("name = '{}'", escaped))
            .await?;
        Ok(())
    }

    /// Refresh this handle to the registry's latest committed version.
    pub async fn reload(&mut self) -> LanceResult<()> {
        self.dataset.checkout_latest().await
    }

    /// Whether a store named `name` exists in the latest registry version.
    pub async fn contains(&mut self, name: &str) -> LanceResult<bool> {
        self.reload().await?;
        let escaped = name.replace('\'', "''");
        let mut scanner = self.dataset.scan();
        scanner.project(&["name"])?;
        scanner.filter(&format!("name = '{}'", escaped))?;
        scanner.limit(Some(1), None)?;
        let mut stream = scanner.try_into_stream().await?;
        while let Some(batch) = stream.try_next().await? {
            if batch.num_rows() > 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Return one directory entry from the latest registry version.
    pub async fn get(&mut self, name: &str) -> LanceResult<Option<RegistryEntry>> {
        self.reload().await?;
        let escaped = name.replace('\'', "''");
        let mut scanner = self.dataset.scan();
        scanner.project(&["name", "uri", "created_at"])?;
        scanner.filter(&format!("name = '{}'", escaped))?;
        scanner.limit(Some(1), None)?;
        let mut stream = scanner.try_into_stream().await?;
        let Some(batch) = stream.try_next().await? else {
            return Ok(None);
        };
        if batch.num_rows() == 0 {
            return Ok(None);
        }

        let names = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                LanceError::from(ArrowError::InvalidArgumentError(
                    "registry 'name' column is not Utf8".into(),
                ))
            })?;
        let uris = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                LanceError::from(ArrowError::InvalidArgumentError(
                    "registry 'uri' column is not Utf8".into(),
                ))
            })?;
        let created = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                LanceError::from(ArrowError::InvalidArgumentError(
                    "registry 'created_at' column is not Int64".into(),
                ))
            })?;
        Ok(Some(RegistryEntry {
            name: names.value(0).to_string(),
            uri: uris.value(0).to_string(),
            created_at: created.value(0),
        }))
    }

    /// All directory entries from the latest registry version, ordered as
    /// stored (unspecified). The registry is a narrow three-column table, so
    /// even hundreds of thousands of rows scan quickly; pagination can be
    /// layered on later if needed.
    pub async fn list(&mut self) -> LanceResult<Vec<RegistryEntry>> {
        self.reload().await?;
        let mut scanner = self.dataset.scan();
        scanner.project(&["name", "uri", "created_at"])?;
        let mut stream = scanner.try_into_stream().await?;
        let mut out = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let names = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    LanceError::from(ArrowError::InvalidArgumentError(
                        "registry 'name' column is not Utf8".into(),
                    ))
                })?;
            let uris = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    LanceError::from(ArrowError::InvalidArgumentError(
                        "registry 'uri' column is not Utf8".into(),
                    ))
                })?;
            let created = batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    LanceError::from(ArrowError::InvalidArgumentError(
                        "registry 'created_at' column is not Int64".into(),
                    ))
                })?;
            for i in 0..batch.num_rows() {
                out.push(RegistryEntry {
                    name: names.value(i).to_string(),
                    uri: uris.value(i).to_string(),
                    created_at: created.value(i),
                });
            }
        }
        Ok(out)
    }

    /// The registry dataset URI.
    #[must_use]
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Current dataset version (manifest chain head).
    pub fn version(&self) -> u64 {
        self.dataset.version().version
    }

    /// Fold the one-row append fragments produced by [`Self::upsert`] into a
    /// few, then drop manifest versions older than `older_than`.
    ///
    /// Every `create` is a delete plus an append, so the registry gains two
    /// versions and one fragment per store. Lance keeps every manifest until
    /// cleaned, and every manifest lists every fragment, so an unmaintained
    /// registry grows quadratically: 34k versions of ~840 KB each (29 GB of
    /// manifests) for a 10k-row, three-column table was observed in
    /// production, and each `contains`/`get` re-reads the head manifest.
    ///
    /// Callers that share this registry behind a lock should prefer
    /// [`Self::compact`] followed by [`RegistryCleaner::cleanup`] so the lock
    /// is not held across the cleanup's object-store deletes.
    pub async fn maintain(
        &mut self,
        older_than: Duration,
    ) -> LanceResult<(CompactionMetrics, RemovalStats)> {
        let (compaction, cleaner) = self.compact().await?;
        let removal = cleaner.cleanup(older_than).await?;
        self.reload().await?;
        Ok((compaction, removal))
    }

    /// The compaction half of [`Self::maintain`]. Returns a
    /// [`RegistryCleaner`] that prunes old versions without borrowing the
    /// registry.
    pub async fn compact(&mut self) -> LanceResult<(CompactionMetrics, RegistryCleaner)> {
        self.reload().await?;
        let options = CompactionOptions {
            target_rows_per_fragment: 1_048_576,
            materialize_deletions: true,
            materialize_deletions_threshold: 0.0,
            ..Default::default()
        };
        let compaction = compact_files(&mut self.dataset, options, None).await?;
        self.reload().await?;
        Ok((
            compaction,
            RegistryCleaner {
                dataset: self.dataset.clone(),
            },
        ))
    }
}

/// A detached handle for the old-version cleanup half of registry
/// maintenance. See `StatsCleaner` in the master for why this is split off:
/// cleanup deletes objects no live version references, so it needs no
/// exclusive access, only time.
pub struct RegistryCleaner {
    dataset: Dataset,
}

impl RegistryCleaner {
    /// Drop manifest versions older than `older_than` and the files only they
    /// referenced.
    pub async fn cleanup(self, older_than: Duration) -> LanceResult<RemovalStats> {
        let grace = chrono::TimeDelta::from_std(older_than)
            .map_err(|e| LanceError::io(format!("invalid registry history TTL: {e}")))?;
        self.dataset.cleanup_old_versions(grace, None, None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn new_registry(dir: &TempDir) -> RolloutRegistry {
        let uri = dir.path().join("_registry.rollout.lance");
        RolloutRegistry::open_or_create(uri.to_str().unwrap(), None)
            .await
            .unwrap()
    }

    /// Maintenance folds the per-upsert fragments and prunes old manifests
    /// while keeping every live row; the registry keeps working afterwards.
    #[tokio::test]
    async fn maintain_bounds_versions_and_preserves_rows() {
        let dir = TempDir::new().unwrap();
        let mut r = new_registry(&dir).await;
        for i in 0..20 {
            r.upsert(&format!("exp-{i}"), &format!("/data/exp-{i}.lance"))
                .await
                .unwrap();
        }
        let before = r.version();
        assert!(
            before >= 40,
            "expected two versions per upsert, got {before}"
        );

        let (compaction, removal) = r.maintain(Duration::from_secs(0)).await.unwrap();
        assert!(compaction.fragments_removed > 0, "nothing compacted");
        assert!(removal.old_versions > 0, "no versions reclaimed");

        let mut names: Vec<String> = r
            .list()
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        names.sort();
        assert_eq!(names.len(), 20);
        assert_eq!(names[0], "exp-0");
        assert!(r.contains("exp-19").await.unwrap());
        r.upsert("exp-new", "/data/exp-new.lance").await.unwrap();
        assert!(r.contains("exp-new").await.unwrap());
    }

    /// A second handle on the same URI (a worker) keeps creating stores while
    /// another (the master) maintains. Nothing the worker wrote is lost and
    /// both handles see every row afterwards.
    #[tokio::test]
    async fn maintenance_does_not_lose_concurrent_worker_upserts() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("_registry.lance");
        let uri = uri.to_str().unwrap();
        let mut master = RolloutRegistry::open_or_create(uri, None).await.unwrap();
        let mut worker = RolloutRegistry::open_or_create(uri, None).await.unwrap();
        for i in 0..20 {
            worker.upsert(&format!("old-{i}"), "/x").await.unwrap();
        }

        let (_, cleaner) = master.compact().await.unwrap();
        // Worker writes land between the compaction and the cleanup ...
        for i in 0..5 {
            worker.upsert(&format!("mid-{i}"), "/x").await.unwrap();
        }
        cleaner.cleanup(Duration::from_secs(0)).await.unwrap();
        master.reload().await.unwrap();
        // ... and after it.
        worker.upsert("late", "/x").await.unwrap();

        assert_eq!(master.list().await.unwrap().len(), 26);
        assert_eq!(worker.list().await.unwrap().len(), 26);
        assert!(master.contains("mid-3").await.unwrap());
        assert!(worker.contains("late").await.unwrap());
    }

    /// The cleanup half runs on a detached handle: the registry stays usable
    /// (reads and writes) while the cleaner is outstanding.
    #[tokio::test]
    async fn cleanup_runs_detached_from_the_registry() {
        let dir = TempDir::new().unwrap();
        let mut r = new_registry(&dir).await;
        for i in 0..20 {
            r.upsert(&format!("exp-{i}"), "/x").await.unwrap();
        }
        let (_, cleaner) = r.compact().await.unwrap();
        r.upsert("during", "/x").await.unwrap();
        assert!(r.contains("during").await.unwrap());
        let removal = cleaner.cleanup(Duration::from_secs(0)).await.unwrap();
        assert!(removal.old_versions > 0);
        r.reload().await.unwrap();
        assert_eq!(r.list().await.unwrap().len(), 21);
    }

    #[tokio::test]
    async fn create_or_load_recovers_when_another_caller_wins() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("_registry.rollout.lance");
        let uri = uri.to_str().unwrap();

        // Force the check-then-create interleaving: both callers observe that
        // the registry is absent before either one attempts to create it.
        assert!(matches!(
            RolloutRegistry::load(uri, None).await,
            Err(LanceError::DatasetNotFound { .. })
        ));
        assert!(matches!(
            RolloutRegistry::load(uri, None).await,
            Err(LanceError::DatasetNotFound { .. })
        ));

        let winner = RolloutRegistry::create(uri, None).await.unwrap();
        let mut winner = RolloutRegistry {
            dataset: winner,
            uri: uri.to_string(),
            storage_options: None,
        };
        winner
            .upsert("winner", "/data/winner.rollout.lance")
            .await
            .unwrap();

        let loser = RolloutRegistry::create_or_load(uri, None).await.unwrap();

        // In this late-loser interleaving, the caller reopens the winner's
        // dataset instead of losing its rows or propagating DatasetAlreadyExists.
        assert_eq!(loser.manifest.version, winner.dataset.manifest.version);
        let mut loser = RolloutRegistry {
            dataset: loser,
            uri: uri.to_string(),
            storage_options: None,
        };
        assert!(loser.contains("winner").await.unwrap());
    }

    #[tokio::test]
    async fn upsert_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let mut reg = new_registry(&dir).await;
        reg.upsert("exp-a", "/data/exp-a.rollout.lance")
            .await
            .unwrap();
        // Re-upserting the same name must not create a duplicate row.
        reg.upsert("exp-a", "/data/exp-a.rollout.lance")
            .await
            .unwrap();
        let entries = reg.list().await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "exp-a");
        assert!(reg.contains("exp-a").await.unwrap());
    }

    #[tokio::test]
    async fn contains_and_remove() {
        let dir = TempDir::new().unwrap();
        let mut reg = new_registry(&dir).await;
        assert!(!reg.contains("missing").await.unwrap());
        reg.upsert("exp-b", "/data/exp-b.rollout.lance")
            .await
            .unwrap();
        assert!(reg.contains("exp-b").await.unwrap());
        assert_eq!(
            reg.get("exp-b").await.unwrap().unwrap().uri,
            "/data/exp-b.rollout.lance"
        );
        assert!(reg.get("missing").await.unwrap().is_none());
        reg.remove("exp-b").await.unwrap();
        assert!(!reg.contains("exp-b").await.unwrap());
        assert!(reg.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_returns_all_entries() {
        let dir = TempDir::new().unwrap();
        let mut reg = new_registry(&dir).await;
        for i in 0..5 {
            reg.upsert(&format!("exp-{i}"), &format!("/data/exp-{i}.rollout.lance"))
                .await
                .unwrap();
        }
        let mut names: Vec<String> = reg
            .list()
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["exp-0", "exp-1", "exp-2", "exp-3", "exp-4"]);
    }

    #[tokio::test]
    async fn survives_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let mut reg = new_registry(&dir).await;
            reg.upsert("persist", "/data/persist.rollout.lance")
                .await
                .unwrap();
        }
        // Reopen the same path: the entry must still be present.
        let mut reg = new_registry(&dir).await;
        assert!(reg.contains("persist").await.unwrap());
    }

    #[tokio::test]
    async fn reads_see_commits_from_another_handle() {
        let dir = TempDir::new().unwrap();
        let mut reader = new_registry(&dir).await;
        let mut writer = new_registry(&dir).await;

        writer
            .upsert("external", "/data/external.rollout.lance")
            .await
            .unwrap();

        assert!(reader.contains("external").await.unwrap());
        assert_eq!(reader.list().await.unwrap()[0].name, "external");
    }

    #[tokio::test]
    async fn insert_missing_batches_new_entries() {
        let dir = TempDir::new().unwrap();
        let mut reg = new_registry(&dir).await;
        reg.upsert("existing", "/data/existing.rollout.lance")
            .await
            .unwrap();

        let entries = vec![
            (
                "existing".to_string(),
                "/other/existing.rollout.lance".to_string(),
            ),
            ("new-a".to_string(), "/data/new-a.rollout.lance".to_string()),
            ("new-b".to_string(), "/data/new-b.rollout.lance".to_string()),
            (
                "new-a".to_string(),
                "/duplicate/new-a.rollout.lance".to_string(),
            ),
        ];
        assert_eq!(reg.insert_missing(&entries).await.unwrap(), 2);
        assert_eq!(reg.insert_missing(&entries).await.unwrap(), 0);

        let mut listed = reg.list().await.unwrap();
        listed.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(listed.len(), 3);
        assert_eq!(listed[0].uri, "/data/existing.rollout.lance");
        assert_eq!(listed[1].uri, "/data/new-a.rollout.lance");
        assert_eq!(listed[2].uri, "/data/new-b.rollout.lance");
    }
}
