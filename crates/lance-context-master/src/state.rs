//! Shared master state: durable registry + stats table.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lance_context_api::{TaskRecord, TaskState};
use lance_context_core::{join_uri, RolloutRegistry};
use tokio::sync::{mpsc, Mutex, RwLock};

use crate::config::MasterConfig;
use crate::discovery;
use crate::stats_store::StatsStore;
use crate::task_store::TaskStore;

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
    /// Durable task records on the master-local RocksDB PVC.
    pub task_store: TaskStore,
    /// Every task (queued / running / terminal) keyed by task id, for the queue
    /// API, de-duplication, and UI. This is a cache of `task_store`, rebuilt at
    /// startup. Terminal history is bounded by `task_history_limit`.
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
    /// Open the registry, stats dataset, and local durable task store.
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
        let task_store = TaskStore::open(&config.task_db_path)?;
        let mut tasks = task_store
            .list()?
            .into_iter()
            .map(|task| (task.id.clone(), task))
            .collect::<HashMap<_, _>>();

        let mut recovered_running = 0usize;
        for task in tasks.values_mut() {
            if task.state == TaskState::Running {
                task.state = TaskState::Queued;
                task.started_at = None;
                task.finished_at = None;
                task.error = None;
                task.detail = None;
                task_store.put(task)?;
                recovered_running += 1;
            }
        }
        let pruned = prune_terminal_history(&task_store, &mut tasks, config.task_history_limit)?;
        let mut pending = tasks
            .values()
            .filter(|task| task.state == TaskState::Queued)
            .map(|task| (task.enqueued_at, task.id.clone()))
            .collect::<Vec<_>>();
        pending.sort();
        let pending_ids = pending.into_iter().map(|(_, id)| id).collect::<Vec<_>>();

        let (task_tx, task_rx) = mpsc::unbounded_channel();
        let state = Arc::new(Self {
            registry: RwLock::new(registry),
            stats: Mutex::new(stats),
            base_uri,
            config,
            task_tx,
            task_rx: Mutex::new(Some(task_rx)),
            task_store,
            tasks: Mutex::new(tasks),
            inflight_dataset_writes: Mutex::new(HashSet::new()),
            http: reqwest::Client::new(),
        });
        for id in &pending_ids {
            let _ = state.task_tx.send(id.clone());
        }
        if !pending_ids.is_empty() || recovered_running > 0 || pruned > 0 {
            tracing::info!(
                queued = pending_ids.len(),
                recovered_running,
                pruned,
                "loaded durable scheduler tasks"
            );
        }
        Ok(state)
    }

    /// Physical rollout dataset URI for `name`, matching the data-plane's
    /// `rollout_uri` convention (`{name}.rollout.lance`).
    pub fn rollout_uri(&self, name: &str) -> String {
        join_uri(&self.base_uri, &format!("{}.rollout.lance", name))
    }
}

pub(crate) fn prune_terminal_history(
    task_store: &TaskStore,
    tasks: &mut HashMap<String, TaskRecord>,
    limit: usize,
) -> lance::Result<usize> {
    // Keep at least the latest terminal task so legacy per-experiment status
    // polling can observe completion even if the configured limit is zero.
    let limit = limit.max(1);
    let mut terminal = tasks
        .values()
        .filter(|task| matches!(task.state, TaskState::Done | TaskState::Failed))
        .map(|task| {
            (
                task.finished_at.unwrap_or(task.enqueued_at),
                task.id.clone(),
            )
        })
        .collect::<Vec<_>>();
    terminal.sort_by_key(|(finished_at, _)| std::cmp::Reverse(*finished_at));
    let ids = terminal
        .into_iter()
        .skip(limit)
        .map(|(_, id)| id)
        .collect::<Vec<_>>();
    task_store.delete_many(ids.iter().map(String::as_str))?;
    for id in &ids {
        tasks.remove(id);
    }
    Ok(ids.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_api::TaskKind;
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
            task_db_path: dir
                .path()
                .join("master-tasks")
                .to_string_lossy()
                .to_string(),
            task_history_limit: 1_000,
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

    #[tokio::test]
    async fn startup_requeues_queued_and_interrupted_tasks() {
        let dir = TempDir::new().unwrap();
        let config = test_config(&dir);
        {
            let state = MasterState::new(config.clone()).await.unwrap();
            for (id, task_state) in [
                ("queued-task", TaskState::Queued),
                ("running-task", TaskState::Running),
                ("done-task", TaskState::Done),
            ] {
                let task = TaskRecord {
                    id: id.to_string(),
                    kind: TaskKind::Compact,
                    target: "legacy".to_string(),
                    state: task_state,
                    error: None,
                    detail: None,
                    enqueued_at: 1,
                    started_at: (task_state == TaskState::Running).then_some(2),
                    finished_at: (task_state == TaskState::Done).then_some(3),
                    depends_on: Vec::new(),
                };
                state.task_store.put(&task).unwrap();
            }
        }

        let state = MasterState::new(config).await.unwrap();
        let tasks = state.tasks.lock().await;
        assert_eq!(tasks["queued-task"].state, TaskState::Queued);
        assert_eq!(tasks["running-task"].state, TaskState::Queued);
        assert_eq!(tasks["running-task"].started_at, None);
        assert_eq!(tasks["done-task"].state, TaskState::Done);
        drop(tasks);

        let mut rx = state.task_rx.lock().await.take().unwrap();
        let mut recovered = Vec::new();
        while let Ok(id) = rx.try_recv() {
            recovered.push(id);
        }
        recovered.sort();
        assert_eq!(recovered, ["queued-task", "running-task"]);
    }

    #[test]
    fn terminal_history_pruning_keeps_active_tasks() {
        let dir = TempDir::new().unwrap();
        let store = TaskStore::open(dir.path().join("tasks")).unwrap();
        let mut tasks = HashMap::new();
        for (id, state, timestamp) in [
            ("queued", TaskState::Queued, 1),
            ("old", TaskState::Done, 2),
            ("new", TaskState::Failed, 3),
        ] {
            let task = TaskRecord {
                id: id.to_string(),
                kind: TaskKind::Compact,
                target: "experiment".to_string(),
                state,
                error: None,
                detail: None,
                enqueued_at: timestamp,
                started_at: None,
                finished_at: matches!(state, TaskState::Done | TaskState::Failed)
                    .then_some(timestamp),
                depends_on: Vec::new(),
            };
            store.put(&task).unwrap();
            tasks.insert(id.to_string(), task);
        }

        assert_eq!(prune_terminal_history(&store, &mut tasks, 1).unwrap(), 1);
        assert!(tasks.contains_key("queued"));
        assert!(tasks.contains_key("new"));
        assert!(!tasks.contains_key("old"));
    }
}
