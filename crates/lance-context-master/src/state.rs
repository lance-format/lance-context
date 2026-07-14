//! Shared master state: durable registry + stats table.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use lance_context_api::CompactJobStatus;
use lance_context_core::RolloutRegistry;
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
    pub base_path: PathBuf,
    /// Effective configuration.
    pub config: MasterConfig,
    /// Enqueues an experiment name for (serial) compaction. Both automatic
    /// sweeps and manual API triggers push here, so compaction has a single
    /// serial driver — never two concurrent `Rewrite`s.
    pub compact_tx: mpsc::UnboundedSender<String>,
    /// Receiver half, taken once by [`crate::scheduler::spawn_scheduler`].
    pub compact_rx: Mutex<Option<mpsc::UnboundedReceiver<String>>>,
    /// Last-known compaction job state per experiment, for the status endpoint.
    pub jobs: Mutex<HashMap<String, CompactJobStatus>>,
}

impl MasterState {
    /// Open (or create) the registry and stats datasets under `data_dir`.
    pub async fn new(config: MasterConfig) -> lance::Result<Arc<Self>> {
        let base_path = PathBuf::from(&config.data_dir);
        let registry_uri = base_path
            .join("_registry.rollout.lance")
            .to_string_lossy()
            .to_string();
        let stats_uri = base_path
            .join("_stats.rollout.lance")
            .to_string_lossy()
            .to_string();
        let mut registry = RolloutRegistry::open_or_create(&registry_uri, None).await?;
        let backfilled = discovery::backfill_registry(&config.data_dir, &mut registry).await?;
        if backfilled > 0 {
            tracing::info!(
                experiments = backfilled,
                "backfilled rollout registry from data directory"
            );
        }
        let stats = StatsStore::open_or_create(&stats_uri, None).await?;
        let (compact_tx, compact_rx) = mpsc::unbounded_channel();
        Ok(Arc::new(Self {
            registry: RwLock::new(registry),
            stats: Mutex::new(stats),
            base_path,
            config,
            compact_tx,
            compact_rx: Mutex::new(Some(compact_rx)),
            jobs: Mutex::new(HashMap::new()),
        }))
    }

    /// Physical rollout dataset URI for `name`, matching the data-plane's
    /// `rollout_uri` convention (`{name}.rollout.lance`).
    pub fn rollout_uri(&self, name: &str) -> String {
        self.base_path
            .join(format!("{}.rollout.lance", name))
            .to_string_lossy()
            .to_string()
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
