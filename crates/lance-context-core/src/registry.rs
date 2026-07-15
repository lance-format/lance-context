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

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use chrono::Utc;
use futures::TryStreamExt;
use lance::dataset::{builder::DatasetBuilder, Dataset, WriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance::{Error as LanceError, Result as LanceResult};

/// One entry in the rollout-store directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryEntry {
    /// Logical store name (the experiment name); unique key.
    pub name: String,
    /// Physical dataset URI, as produced by the server's `rollout_uri`.
    pub uri: String,
    /// Creation time, Unix milliseconds.
    pub created_at: i64,
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
