//! Shared master state: durable registry (read-only) + stats table (read/write).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use lance_context_api::CompactJobStatus;
use lance_context_core::RolloutRegistry;
use tokio::sync::{mpsc, Mutex, RwLock};

use crate::config::MasterConfig;
use crate::stats_store::StatsStore;

/// Shared state for the master process.
///
/// The `registry` is written only by the data-plane (store create/delete); the
/// master reads it to enumerate experiments. The `stats_store` is owned and
/// written exclusively by the master. Both are wrapped in locks because their
/// mutating methods take `&mut self`.
pub struct MasterState {
    /// Durable directory of which rollout stores exist (data-plane-owned).
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
        let registry = RolloutRegistry::open_or_create(&registry_uri, None).await?;
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
