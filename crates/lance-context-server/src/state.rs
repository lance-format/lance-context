use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use lance_context_core::ContextStore;
use tokio::sync::RwLock;

use crate::config::ServerConfig;

pub struct AppState {
    pub stores: RwLock<HashMap<String, Arc<RwLock<ContextStore>>>>,
    pub base_path: PathBuf,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            stores: RwLock::new(HashMap::new()),
            base_path: PathBuf::from(&config.data_dir),
        }
    }

    pub fn context_uri(&self, name: &str) -> String {
        self.base_path
            .join(format!("{}.lance", name))
            .to_string_lossy()
            .to_string()
    }
}
