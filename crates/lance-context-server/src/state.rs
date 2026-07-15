use std::num::NonZeroUsize;
#[cfg(test)]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use lance_context_core::{
    join_uri, validate_store_name, ContextStore, ContextStoreOptions, RolloutRegistry,
    RolloutStore, RolloutStoreOptions,
};
use lru::LruCache;
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

use crate::config::ServerConfig;
use crate::error::AppError;

/// Default upper bound on resident rollout-store handles when the config does
/// not specify one. Sized for peak *concurrent* experiments, not the total
/// number of experiments (which may be hundreds of thousands).
pub const DEFAULT_ROLLOUT_CACHE_CAPACITY: usize = 2000;

pub struct AppState {
    pub stores: RwLock<std::collections::HashMap<String, Arc<RwLock<ContextStore>>>>,
    /// Bounded LRU of resident rollout-store handles.
    ///
    /// With one physical dataset per experiment, the deployment may hold
    /// hundreds of thousands of stores; keeping them all resident would exhaust
    /// memory and (formerly) spawn one background timer each. This cache bounds
    /// residency: on overflow the least-recently-used handle is evicted and its
    /// `Arc<RwLock<RolloutStore>>` dropped. Existence is tracked durably by
    /// [`Self::rollout_registry`], not by membership in this cache.
    pub rollout_stores: Mutex<LruCache<String, Arc<RwLock<RolloutStore>>>>,
    /// Durable directory of which rollout stores exist. Consulted on a cache
    /// miss (existence check) and to back the list endpoint. Guarded by a lock
    /// because every operation refreshes the snapshot and therefore takes
    /// `&mut`.
    pub rollout_registry: RwLock<RolloutRegistry>,
    pub base_uri: String,
    /// Stable identity of this server instance, used as the MemWAL shard key for
    /// rollout writes so each instance owns exactly one shard. `None` falls back
    /// to a single shared shard (single-instance deployments only).
    pub instance_id: Option<String>,
    /// Count-triggered self-merge threshold for rollout MemWAL shards; `0`
    /// disables it. See `RolloutStoreOptions::merge_after_generations`.
    pub rollout_merge_after_generations: usize,
    /// Periodic per-shard WAL-cleanup interval in seconds; `0` disables the
    /// global sweeper. See [`Self::spawn_global_sweeper`].
    pub rollout_cleanup_interval_secs: u64,
    /// Minimum flushed generations before a periodic cleanup tick merges. See
    /// `RolloutStoreOptions::cleanup_min_generations`.
    pub rollout_cleanup_min_generations: usize,
}

impl AppState {
    /// Build the shared server state, opening (or creating) the rollout registry
    /// under `data_dir`. Async because opening the registry touches storage.
    pub async fn new(config: ServerConfig) -> Result<Self, AppError> {
        let instance_id = config.resolved_instance_id();
        let base_uri = config.data_dir.clone();
        let registry_uri = join_uri(&base_uri, "_registry.rollout.lance");
        let registry = RolloutRegistry::open_or_create(&registry_uri, None)
            .await
            .map_err(AppError::from_lance)?;
        let capacity = NonZeroUsize::new(config.rollout_cache_capacity)
            .unwrap_or_else(|| NonZeroUsize::new(DEFAULT_ROLLOUT_CACHE_CAPACITY).unwrap());
        Ok(Self {
            stores: RwLock::new(std::collections::HashMap::new()),
            rollout_stores: Mutex::new(LruCache::new(capacity)),
            rollout_registry: RwLock::new(registry),
            base_uri,
            instance_id,
            rollout_merge_after_generations: config.rollout_merge_after_generations,
            rollout_cleanup_interval_secs: config.rollout_cleanup_interval_secs,
            rollout_cleanup_min_generations: config.rollout_cleanup_min_generations,
        })
    }

    pub fn context_uri(&self, name: &str) -> String {
        join_uri(&self.base_uri, &format!("{}.lance", name))
    }

    /// Build a default-configured `AppState` rooted at `base_path`, for tests.
    /// Opens a fresh registry under the directory; cleanup interval disabled.
    #[cfg(test)]
    pub async fn new_for_test(base_path: PathBuf) -> Self {
        Self::new_for_test_with_instance(base_path, None).await
    }

    /// Like [`Self::new_for_test`] but with an explicit MemWAL instance id, so a
    /// second in-process "instance" can present its own shard.
    #[cfg(test)]
    pub async fn new_for_test_with_instance(
        base_path: PathBuf,
        instance_id: Option<String>,
    ) -> Self {
        let base_uri = base_path.to_string_lossy().to_string();
        let registry_uri = join_uri(&base_uri, "_registry.rollout.lance");
        let registry = RolloutRegistry::open_or_create(&registry_uri, None)
            .await
            .expect("open test registry");
        Self {
            stores: RwLock::new(std::collections::HashMap::new()),
            rollout_stores: Mutex::new(LruCache::new(
                NonZeroUsize::new(DEFAULT_ROLLOUT_CACHE_CAPACITY).unwrap(),
            )),
            rollout_registry: RwLock::new(registry),
            base_uri,
            instance_id,
            rollout_merge_after_generations: 0,
            rollout_cleanup_interval_secs: 0,
            rollout_cleanup_min_generations: 1,
        }
    }

    /// Rollout datasets live under a distinct `.rollout.lance` suffix so a
    /// rollout store and a context store may share the same name without
    /// colliding on disk.
    pub fn rollout_uri(&self, name: &str) -> String {
        join_uri(&self.base_uri, &format!("{}.rollout.lance", name))
    }

    fn rollout_store_options(&self) -> RolloutStoreOptions {
        RolloutStoreOptions {
            // No request body on the read path: object-store credentials come
            // from the pod's workload-identity environment.
            storage_options: None,
            shard_id: self.instance_id.clone(),
            merge_after_generations: (self.rollout_merge_after_generations > 0)
                .then_some(self.rollout_merge_after_generations),
            cleanup_interval_secs: (self.rollout_cleanup_interval_secs > 0)
                .then_some(self.rollout_cleanup_interval_secs),
            cleanup_min_generations: Some(self.rollout_cleanup_min_generations),
        }
    }

    /// Record that a rollout store exists, in both the durable registry and the
    /// in-memory LRU. Called by the create route after the dataset is written.
    pub async fn register_rollout(
        &self,
        name: &str,
        uri: &str,
        store: Arc<RwLock<RolloutStore>>,
    ) -> Result<(), AppError> {
        Self::validate_name(name)?;
        self.rollout_registry
            .write()
            .await
            .upsert(name, uri)
            .await
            .map_err(AppError::from_lance)?;
        // Insertion may evict the LRU's least-recently-used entry; dropping the
        // returned handle releases it (no per-store timer to abort — cleanup is
        // now global).
        self.rollout_stores
            .lock()
            .await
            .put(name.to_string(), store);
        Ok(())
    }

    /// Remove a rollout store from the durable registry and evict any resident
    /// handle. Returns whether the store existed.
    pub async fn unregister_rollout(&self, name: &str) -> Result<bool, AppError> {
        Self::validate_name(name)?;
        let existed = self
            .rollout_registry
            .write()
            .await
            .contains(name)
            .await
            .map_err(AppError::from_lance)?;
        if !existed {
            return Ok(false);
        }
        self.rollout_registry
            .write()
            .await
            .remove(name)
            .await
            .map_err(AppError::from_lance)?;
        self.rollout_stores.lock().await.pop(name);
        Ok(true)
    }

    /// Look up a rollout store by name, lazily loading it from object storage on
    /// a local cache miss.
    ///
    /// Unlike the previous implementation, a cache miss is **not** a 404: the
    /// bounded LRU may have evicted a store that still exists. Existence is
    /// resolved against the durable [`RolloutRegistry`]; only a store absent
    /// from the registry yields [`AppError::NotFound`]. This closes the
    /// multi-replica gap (create on pod A, read on pod B) *and* the
    /// eviction-induced false-404 that per-experiment partitioning introduces.
    pub async fn get_or_open_rollout_store(
        &self,
        name: &str,
    ) -> Result<Arc<RwLock<RolloutStore>>, AppError> {
        Self::validate_name(name)?;
        // Fast path: resident in this process's LRU (updates recency).
        if let Some(store) = self.rollout_stores.lock().await.get(name) {
            metrics::counter!("rollout_store_cache_hits_total").increment(1);
            return Ok(store.clone());
        }
        metrics::counter!("rollout_store_cache_misses_total").increment(1);

        // Existence is the registry's job, not the cache's.
        let exists = self
            .rollout_registry
            .write()
            .await
            .contains(name)
            .await
            .map_err(AppError::from_lance)?;
        if !exists {
            return Err(AppError::NotFound(format!(
                "Rollout store '{}' does not exist",
                name
            )));
        }

        // Load from storage WITHOUT holding the LRU lock, so a slow open does
        // not block other stores' requests.
        let uri = self.rollout_uri(name);
        let opened = RolloutStore::open_existing_with_options(&uri, self.rollout_store_options())
            .await
            .map_err(AppError::from_lance)?;
        let opened = Arc::new(RwLock::new(opened));

        // Insert under the lock, re-checking for a store another request may
        // have opened concurrently while we were loading.
        let mut cache = self.rollout_stores.lock().await;
        if let Some(existing) = cache.get(name) {
            return Ok(existing.clone());
        }
        cache.put(name.to_string(), opened.clone());
        metrics::gauge!("rollout_stores_resident").set(cache.len() as f64);
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
        Self::validate_name(name)?;
        if let Some(store) = self.stores.read().await.get(name) {
            return Ok(store.clone());
        }

        let uri = self.context_uri(name);
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

    pub fn validate_name(name: &str) -> Result<(), AppError> {
        validate_store_name(name).map_err(AppError::InvalidRequest)
    }

    /// Spawn the single, process-wide WAL-cleanup sweeper.
    ///
    /// This replaces the former one-timer-per-store model, which does not scale
    /// to hundreds of thousands of per-experiment datasets. On each tick the
    /// sweeper snapshots the *resident* rollout stores (those in the LRU) and
    /// folds each one's flushed MemWAL generations into its base table, guarding
    /// every pass with a timeout so a wedged store cannot stall the sweeper (see
    /// [`RolloutStore::cleanup_own_shard`] and the timeout added in the periodic
    /// cleanup hardening).
    ///
    /// Cold stores (evicted from the LRU) are not swept: having no new writes,
    /// they accumulate no new generations, and are swept again the moment a
    /// request re-opens them. Returns `None` when the interval is `0`.
    pub fn spawn_global_sweeper(self: &Arc<Self>) -> Option<JoinHandle<()>> {
        let interval_secs = self.rollout_cleanup_interval_secs;
        if interval_secs == 0 {
            return None;
        }
        let interval = Duration::from_secs(interval_secs);
        // Abandon any single pass that outruns five intervals (min 30s) so one
        // stuck store cannot wedge cleanup for every other store.
        let pass_timeout = interval.saturating_mul(5).max(Duration::from_secs(30));
        let weak = Arc::downgrade(self);
        Some(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await; // skip the immediate first tick
            loop {
                ticker.tick().await;
                let Some(state) = weak.upgrade() else {
                    return;
                };
                // Snapshot resident stores without holding the LRU lock across
                // the awaits below.
                let resident: Vec<(String, Arc<RwLock<RolloutStore>>)> = {
                    let cache = state.rollout_stores.lock().await;
                    cache
                        .iter()
                        .map(|(name, store)| (name.clone(), store.clone()))
                        .collect()
                };
                for (name, store) in resident {
                    let mut guard = store.write().await;
                    match tokio::time::timeout(pass_timeout, guard.cleanup_own_shard()).await {
                        Ok(Ok(0)) => {}
                        Ok(Ok(n)) => {
                            metrics::counter!("rollout_wal_cleanup_total", "result" => "merged")
                                .increment(1);
                            metrics::counter!("rollout_wal_generations_reclaimed_total")
                                .increment(n as u64);
                            tracing::info!(
                                store = %name,
                                reclaimed = n,
                                "global sweeper merged flushed generations"
                            )
                        }
                        Ok(Err(e)) => {
                            metrics::counter!("rollout_wal_cleanup_total", "result" => "failed")
                                .increment(1);
                            tracing::warn!(
                                store = %name,
                                error = %e,
                                "global sweeper WAL cleanup failed"
                            )
                        }
                        Err(_elapsed) => {
                            metrics::counter!("rollout_wal_cleanup_total", "result" => "timeout")
                                .increment(1);
                            tracing::warn!(
                                store = %name,
                                "global sweeper WAL cleanup timed out; abandoning this store this tick"
                            )
                        }
                    }
                }
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn state_with_interval(dir: &TempDir, secs: u64) -> Arc<AppState> {
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        state.rollout_cleanup_interval_secs = secs;
        Arc::new(state)
    }

    /// The sweeper is a single detached task, gated on the cleanup interval:
    /// spawned when non-zero, `None` when disabled. This is the whole-process
    /// replacement for the former one-timer-per-store model.
    #[tokio::test]
    async fn global_sweeper_is_gated_on_interval() {
        let dir = TempDir::new().unwrap();
        let disabled = state_with_interval(&dir, 0).await;
        assert!(disabled.spawn_global_sweeper().is_none());

        let dir2 = TempDir::new().unwrap();
        let enabled = state_with_interval(&dir2, 3600).await;
        let handle = enabled
            .spawn_global_sweeper()
            .expect("sweeper spawns when interval > 0");
        handle.abort();
    }
}
