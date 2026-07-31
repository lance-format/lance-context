//! Shared master state: durable registry + stats table.

use std::num::NonZeroUsize;
use std::sync::Arc;

use lance_context_core::{join_uri, RolloutRegistry, RolloutStore, RolloutStoreOptions, Session};
use lru::LruCache;
use tokio::sync::{Mutex, RwLock};

use crate::config::MasterConfig;
use crate::discovery;
use crate::stats_store::StatsStore;
use crate::task_store::TaskStore;

/// Bounded number of rollout handles retained by the master data browser.
///
/// Each handle reuses the process-wide Lance session whose fragment/file
/// metadata caches are valuable across pagination requests. The bound prevents
/// a master managing a very large registry from retaining one handle per
/// experiment indefinitely.
const RECORD_STORE_CACHE_CAPACITY: usize = 128;

/// Build the process-wide rollout session from one total cache budget.
///
/// The 6:1 index:metadata split mirrors Lance's default cache ratio.
fn build_rollout_session(cache_bytes: usize) -> Option<Arc<Session>> {
    if cache_bytes == 0 {
        return None;
    }
    let metadata_bytes = cache_bytes / 7;
    let index_bytes = cache_bytes - metadata_bytes;
    Some(RolloutStore::build_session(index_bytes, metadata_bytes))
}

/// Shared state for the master process.
///
/// The data-plane owns steady-state registry writes (store create/delete); the
/// master only adds missing legacy rows during startup discovery, then reads it
/// to enumerate experiments. Master replicas coordinate stats-table writes
/// through the task store. The Lance handles are wrapped in locks because their
/// mutating methods take `&mut self`.
pub struct MasterState {
    /// Durable directory of which rollout stores exist.
    pub registry: RwLock<RolloutRegistry>,
    /// Periodically-refreshed per-experiment metrics (master-owned).
    pub stats: Mutex<StatsStore>,
    /// Read handles used by the records browser, retained to reuse Lance
    /// session and fragment metadata caches across requests.
    record_stores: Mutex<LruCache<String, Arc<RwLock<RolloutStore>>>>,
    /// Process-wide Lance cache session attached to every rollout store opened
    /// by this master. `None` is only used when the configured budget is `0`.
    rollout_session: Option<Arc<Session>>,
    /// Shared data directory / object-store prefix.
    pub base_uri: String,
    /// Effective configuration.
    pub config: MasterConfig,
    /// Durable scheduler queue in etcd, providing shared CAS/lease semantics
    /// for stateless HA master replicas.
    pub task_store: TaskStore,
    /// Shared HTTP client for fanning `MergeWal` tasks out to worker endpoints.
    pub http: reqwest::Client,
    /// Whether this process has already run one `_stats` maintenance pass.
    ///
    /// The first pass runs without a timeout so a deployment carrying a version
    /// chain from the old per-row write path can actually reclaim it; a bounded
    /// first pass times out forever and never recovers. See
    /// [`crate::scanner::maintain_stats`].
    pub stats_maintenance_done: std::sync::atomic::AtomicBool,
    /// Consecutive `_stats` maintenance failures, for alerting.
    ///
    /// Failure was previously silent: the success counter simply stopped
    /// incrementing, which is indistinguishable from "nothing to reclaim".
    pub stats_maintenance_failures: std::sync::atomic::AtomicU64,
    /// `_stats` version at the last successful maintenance pass.
    ///
    /// The gap between this and the current version is how many versions are
    /// going unreclaimed -- the number worth alerting on. The raw version
    /// number is not: it climbs by design and says nothing about disk usage.
    pub stats_last_reclaimed_version: std::sync::atomic::AtomicU64,
}

impl MasterState {
    /// Open the registry, stats dataset, and configured durable task store.
    pub async fn new(config: MasterConfig) -> lance::Result<Arc<Self>> {
        let task_store = TaskStore::open(&config).await?;
        // Serialize first-time registry/stats creation and legacy backfill in
        // etcd mode. Followers wait briefly rather than racing Lance creates.
        let init_guard = loop {
            if let Some(guard) = task_store.try_coordination_lock("state-init").await? {
                break guard;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        };
        let base_uri = config.data_dir.clone();
        let rollout_session = build_rollout_session(config.rollout_cache_bytes);
        let registry_uri = join_uri(&base_uri, "_registry.rollout.lance");
        let stats_uri = join_uri(&base_uri, "_stats.rollout.lance");
        let mut registry = RolloutRegistry::open_or_create(&registry_uri, None).await?;
        let backfilled = discovery::backfill_registry(&config.data_dir, &mut registry).await?;
        if backfilled > 0 {
            tracing::info!(
                experiments = backfilled,
                "backfilled rollout registry from data directory"
            );
        }
        let stats = StatsStore::open_or_create(&stats_uri, None).await?;
        task_store.release_coordination_lock(init_guard).await?;
        let state = Arc::new(Self {
            registry: RwLock::new(registry),
            stats: Mutex::new(stats),
            record_stores: Mutex::new(LruCache::new(
                NonZeroUsize::new(RECORD_STORE_CACHE_CAPACITY).unwrap(),
            )),
            rollout_session,
            base_uri,
            config,
            task_store,
            http: reqwest::Client::new(),
            stats_maintenance_done: std::sync::atomic::AtomicBool::new(false),
            stats_maintenance_failures: std::sync::atomic::AtomicU64::new(0),
            stats_last_reclaimed_version: std::sync::atomic::AtomicU64::new(0),
        });
        Ok(state)
    }

    /// Physical rollout dataset URI for `name`, matching the data-plane's
    /// `rollout_uri` convention (`{name}.rollout.lance`).
    pub fn rollout_uri(&self, name: &str) -> String {
        join_uri(&self.base_uri, &format!("{}.rollout.lance", name))
    }

    /// Options for every rollout store opened by the master.
    pub(crate) fn rollout_store_options(&self) -> RolloutStoreOptions {
        RolloutStoreOptions {
            session: self.rollout_session.clone(),
            ..Default::default()
        }
    }

    /// Return a cached rollout handle for the master records browser, opening
    /// it without holding the cache lock on a miss.
    pub async fn get_or_open_record_store(
        &self,
        name: &str,
        uri: &str,
    ) -> lance::Result<Arc<RwLock<RolloutStore>>> {
        if let Some(store) = self.record_stores.lock().await.get(name) {
            metrics::counter!("master_record_store_cache_hits_total").increment(1);
            return Ok(store.clone());
        }
        metrics::counter!("master_record_store_cache_misses_total").increment(1);

        let opened = Arc::new(RwLock::new(
            RolloutStore::open_existing_with_options(uri, self.rollout_store_options()).await?,
        ));

        let mut cache = self.record_stores.lock().await;
        if let Some(existing) = cache.get(name) {
            return Ok(existing.clone());
        }
        cache.put(name.to_string(), opened.clone());
        metrics::gauge!("master_record_stores_resident").set(cache.len() as f64);
        Ok(opened)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_api::{TaskKind, TaskState};
    use lance_context_core::{generate_id, RolloutStore};
    use tempfile::TempDir;

    fn test_config(dir: &TempDir) -> MasterConfig {
        MasterConfig {
            data_dir: dir.path().to_string_lossy().to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            stats_scan_interval_secs: 0,
            scan_concurrency: 4,
            rollout_cache_bytes: 2 * 1024 * 1024 * 1024,
            stats_maintenance_every_n_scans: 0,
            stats_history_ttl_secs: 3_600,
            stats_cold_retire_secs: 0,
            compaction_interval_secs: 0,
            min_fragments: 16,
            target_rows_per_fragment: 1_048_576,
            merge_wal_interval_secs: 0,
            merge_wal_min_generations: 8,
            worker_endpoints: vec![],
            task_concurrency: 4,
            etcd_endpoints: std::env::var("ETCD_TEST_ENDPOINTS")
                .map(|value| value.split(',').map(str::to_string).collect())
                .unwrap_or_default(),
            etcd_prefix: format!("/lance-context/test/{}", generate_id()),
            etcd_username: None,
            etcd_password: None,
            etcd_ca_cert: None,
            etcd_client_cert: None,
            etcd_client_key: None,
            etcd_lease_ttl_secs: 5,
            task_history_limit: 1_000,
            task_history_ttl_secs: 86_400,
            ui_dir: None,
        }
    }

    #[test]
    fn rollout_session_can_be_disabled() {
        assert!(build_rollout_session(0).is_none());
        assert!(build_rollout_session(7).is_some());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn startup_backfills_legacy_rollout_datasets() {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().join("legacy.rollout.lance");
        RolloutStore::open(uri.to_str().unwrap()).await.unwrap();

        let state = MasterState::new(test_config(&dir)).await.unwrap();
        let entries = state.registry.write().await.list().await.unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "legacy");
        assert_eq!(entries[0].uri, uri.to_string_lossy().to_string());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn startup_requeues_interrupted_local_tasks() {
        let dir = TempDir::new().unwrap();
        let config = test_config(&dir);
        let first = MasterState::new(config.clone()).await.unwrap();
        let task = first
            .task_store
            .enqueue(TaskKind::Compact, "legacy", Vec::new())
            .await
            .unwrap();
        let claim = first.task_store.claim_next().await.unwrap().unwrap();
        assert_eq!(claim.task.state, TaskState::Running);
        drop(claim);
        drop(first);

        let state = MasterState::new(config).await.unwrap();
        let recovered = state.task_store.get(&task.id).await.unwrap().unwrap();
        assert_eq!(recovered.state, TaskState::Queued);
        assert_eq!(recovered.started_at, None);
    }
}
