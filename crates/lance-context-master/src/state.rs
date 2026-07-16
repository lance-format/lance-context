//! Shared master state: durable registry + stats table.

use std::sync::Arc;

use lance_context_core::{join_uri, RolloutRegistry};
use tokio::sync::{Mutex, RwLock};

use crate::config::MasterConfig;
use crate::discovery;
use crate::stats_store::StatsStore;
use crate::task_store::TaskStore;

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
    /// Shared data directory / object-store prefix.
    pub base_uri: String,
    /// Effective configuration.
    pub config: MasterConfig,
    /// Durable scheduler queue in etcd, providing shared CAS/lease semantics
    /// for stateless HA master replicas.
    pub task_store: TaskStore,
    /// Shared HTTP client for fanning `MergeWal` tasks out to worker endpoints.
    pub http: reqwest::Client,
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
            base_uri,
            config,
            task_store,
            http: reqwest::Client::new(),
        });
        Ok(state)
    }

    /// Physical rollout dataset URI for `name`, matching the data-plane's
    /// `rollout_uri` convention (`{name}.rollout.lance`).
    pub fn rollout_uri(&self, name: &str) -> String {
        join_uri(&self.base_uri, &format!("{}.rollout.lance", name))
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
            compaction_interval_secs: 0,
            min_fragments: 16,
            target_rows_per_fragment: 1_048_576,
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
            ui_dir: None,
        }
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
