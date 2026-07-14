use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use lance_context_core::{ContextStore, ContextStoreOptions, RolloutStore, RolloutStoreOptions};
use tokio::sync::RwLock;

use crate::config::ServerConfig;
use crate::error::AppError;

pub struct AppState {
    pub stores: RwLock<HashMap<String, Arc<RwLock<ContextStore>>>>,
    pub rollout_stores: RwLock<HashMap<String, Arc<RwLock<RolloutStore>>>>,
    pub base_path: PathBuf,
    /// Stable identity of this server instance, used as the MemWAL shard key for
    /// rollout writes so each instance owns exactly one shard. `None` falls back
    /// to a single shared shard (single-instance deployments only).
    pub instance_id: Option<String>,
    /// Count-triggered self-merge threshold for rollout MemWAL shards; `0`
    /// disables it. See `RolloutStoreOptions::merge_after_generations`.
    pub rollout_merge_after_generations: usize,
    /// Periodic per-shard WAL-cleanup interval in seconds; `0` disables the
    /// background timer. See `RolloutStoreOptions::cleanup_interval_secs`.
    pub rollout_cleanup_interval_secs: u64,
    /// Minimum flushed generations before a periodic cleanup tick merges. See
    /// `RolloutStoreOptions::cleanup_min_generations`.
    pub rollout_cleanup_min_generations: usize,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        let instance_id = config.resolved_instance_id();
        Self {
            stores: RwLock::new(HashMap::new()),
            rollout_stores: RwLock::new(HashMap::new()),
            base_path: PathBuf::from(&config.data_dir),
            instance_id,
            rollout_merge_after_generations: config.rollout_merge_after_generations,
            rollout_cleanup_interval_secs: config.rollout_cleanup_interval_secs,
            rollout_cleanup_min_generations: config.rollout_cleanup_min_generations,
        }
    }

    pub fn context_uri(&self, name: &str) -> String {
        self.base_path
            .join(format!("{}.lance", name))
            .to_string_lossy()
            .to_string()
    }

    /// Rollout datasets live under a distinct `.rollout.lance` suffix so a
    /// rollout store and a context store may share the same name without
    /// colliding on disk.
    pub fn rollout_uri(&self, name: &str) -> String {
        self.base_path
            .join(format!("{}.rollout.lance", name))
            .to_string_lossy()
            .to_string()
    }

    /// Look up a rollout store by name, lazily loading it from object storage on
    /// a local cache miss.
    ///
    /// The server caches each opened store in this process's memory. In a
    /// multi-replica deployment a store `create`d on pod A only enters A's map,
    /// so a read/write that the load balancer routes to pod B would otherwise
    /// 404 even though the dataset exists on shared object storage. This helper
    /// closes that gap: on a miss it *loads* (never creates — see
    /// [`RolloutStore::open_existing_with_options`], which returns a
    /// `DatasetNotFound`/404 for a genuinely-absent store rather than silently
    /// materializing an empty table) and caches the handle for subsequent hits.
    pub async fn get_or_open_rollout_store(
        &self,
        name: &str,
    ) -> Result<Arc<RwLock<RolloutStore>>, AppError> {
        // Fast path: already cached in this process.
        if let Some(store) = self.rollout_stores.read().await.get(name) {
            return Ok(store.clone());
        }

        // Slow path: load from object storage WITHOUT holding the map lock, so a
        // slow open does not block other stores' requests.
        let uri = self.rollout_uri(name);
        let options = RolloutStoreOptions {
            // No request body on the read path: object-store credentials come
            // from the pod's workload-identity environment, exactly as they do
            // for the `create` route in production.
            storage_options: None,
            shard_id: self.instance_id.clone(),
            merge_after_generations: (self.rollout_merge_after_generations > 0)
                .then_some(self.rollout_merge_after_generations),
            cleanup_interval_secs: (self.rollout_cleanup_interval_secs > 0)
                .then_some(self.rollout_cleanup_interval_secs),
            cleanup_min_generations: Some(self.rollout_cleanup_min_generations),
        };
        let opened = RolloutStore::open_existing_with_options(&uri, options)
            .await
            .map_err(AppError::from_lance)?;
        let opened = Arc::new(RwLock::new(opened));

        // Insert under the write lock, re-checking for a store another request
        // may have opened concurrently while we were loading.
        let mut stores = self.rollout_stores.write().await;
        if let Some(existing) = stores.get(name) {
            return Ok(existing.clone());
        }
        // A lazily-opened store must also get the background WAL-cleanup timer
        // that the `create` route starts, or these stores would never merge.
        let _cleanup = RolloutStore::spawn_periodic_cleanup(opened.clone());
        stores.insert(name.to_string(), opened.clone());
        Ok(opened)
    }

    /// Look up a context store by name, lazily loading it from object storage on
    /// a local cache miss. See [`Self::get_or_open_rollout_store`] for the
    /// multi-replica rationale; the same in-memory-cache-as-existence-check bug
    /// affects context stores.
    pub async fn get_or_open_context_store(
        &self,
        name: &str,
    ) -> Result<Arc<RwLock<ContextStore>>, AppError> {
        if let Some(store) = self.stores.read().await.get(name) {
            return Ok(store.clone());
        }

        let uri = self.context_uri(name);
        // Load-only: existing schema (embedding dim, blob columns, distance
        // metric) is read from the persisted dataset, so no create-time options
        // are needed. Credentials come from the pod environment.
        let opened = ContextStore::open_existing_with_options(&uri, ContextStoreOptions::default())
            .await
            .map_err(AppError::from_lance)?;
        let opened = Arc::new(RwLock::new(opened));

        let mut stores = self.stores.write().await;
        if let Some(existing) = stores.get(name) {
            return Ok(existing.clone());
        }
        stores.insert(name.to_string(), opened.clone());
        Ok(opened)
    }
}
