//! Shared master state: durable registry + stats table.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lance_context_api::TaskRecord;
use lance_context_core::{join_uri, RolloutRegistry};
use tokio::sync::{mpsc, Mutex, RwLock};

use crate::config::MasterConfig;
use crate::discovery;
use crate::stats_store::StatsStore;

/// Shared state for the master process.
///
/// The data-plane owns steady-state registry writes (store create/delete); the
/// master only adds missing legacy rows during startup discovery, then reads it
/// to enumerate experiments. The `stats_store` is owned and written exclusively
/// by the master. Both are wrapped in locks because their mutating methods take
/// `&mut self`.
pub struct MasterState {
    /// Durable directory of which rollout stores exist.
    pub registry: RwLock<RolloutRegistry>,
    /// Periodically-refreshed per-experiment metrics (master-owned).
    pub stats: Mutex<StatsStore>,
    /// Shared data directory / object-store prefix.
    pub base_uri: String,
    /// Effective configuration.
    pub config: MasterConfig,
    /// Enqueues a task id for the scheduler. Both automatic sweeps and manual
    /// API triggers push here; the scheduler drains it with bounded concurrency
    /// (see [`crate::scheduler`]).
    pub task_tx: mpsc::UnboundedSender<String>,
    /// Receiver half, taken once by [`crate::scheduler::spawn_scheduler`].
    pub task_rx: Mutex<Option<mpsc::UnboundedReceiver<String>>>,
    /// Every task (queued / running / terminal) keyed by task id, for the queue
    /// API and UI. Retained after completion so the UI can show recent history.
    pub tasks: Mutex<HashMap<String, TaskRecord>>,
    /// Experiment names with a base-table write currently executing — the
    /// per-name serial gate. A `Compact` or `IndexId` task must not run while
    /// another base-table-mutating task on the same experiment is in flight:
    /// two `Rewrite`s (or a `Rewrite` and a `CreateIndex`) on one dataset can
    /// conflict in Lance's commit matrix, forcing wasteful retries.
    pub inflight_dataset_writes: Mutex<HashSet<String>>,
    /// Shared HTTP client for fanning `MergeWal` tasks out to worker endpoints.
    pub http: reqwest::Client,
}

impl MasterState {
    /// Open (or create) the registry and stats datasets under `data_dir`.
    pub async fn new(config: MasterConfig) -> lance::Result<Arc<Self>> {
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
        let (task_tx, task_rx) = mpsc::unbounded_channel();
        Ok(Arc::new(Self {
            registry: RwLock::new(registry),
            stats: Mutex::new(stats),
            base_uri,
            config,
            task_tx,
            task_rx: Mutex::new(Some(task_rx)),
            tasks: Mutex::new(HashMap::new()),
            inflight_dataset_writes: Mutex::new(HashSet::new()),
            http: reqwest::Client::new(),
        }))
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
    use lance_context_core::RolloutStore;
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
            ui_dir: None,
        }
    }

    #[tokio::test]
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
}
