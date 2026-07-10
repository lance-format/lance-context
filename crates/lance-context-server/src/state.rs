use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use lance_context_core::{ContextStore, RolloutStore};
use tokio::sync::RwLock;

use crate::config::ServerConfig;

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
}
