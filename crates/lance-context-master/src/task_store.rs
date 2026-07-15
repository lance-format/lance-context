//! Durable scheduler task storage backed by RocksDB.

use std::path::Path;

use lance_context_api::TaskRecord;
use rocksdb::{Direction, IteratorMode, Options, WriteBatch, WriteOptions, DB};

const TASK_KEY_PREFIX: &[u8] = b"task/";

/// Single-process RocksDB store for scheduler task records.
///
/// Every write uses RocksDB's WAL with `sync=true`: the scheduler never
/// acknowledges an enqueue or lifecycle transition before it is durable on
/// the master PVC.
pub struct TaskStore {
    db: DB,
}

impl TaskStore {
    pub fn open(path: impl AsRef<Path>) -> lance::Result<Self> {
        let path = path.as_ref();
        let text = path.to_string_lossy();
        if text.contains("://") {
            return Err(lance::Error::io(format!(
                "task DB path must be local, got '{text}'"
            )));
        }
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|err| {
                lance::Error::io(format!(
                    "failed to create task DB parent '{}': {err}",
                    parent.display()
                ))
            })?;
        }

        let mut options = Options::default();
        options.create_if_missing(true);
        options.set_max_open_files(64);
        options.set_keep_log_file_num(4);
        let db = DB::open(&options, path).map_err(|err| {
            lance::Error::io(format!(
                "failed to open task DB '{}': {err}",
                path.display()
            ))
        })?;
        Ok(Self { db })
    }

    pub fn put(&self, task: &TaskRecord) -> lance::Result<()> {
        let value = serde_json::to_vec(task).map_err(|err| {
            lance::Error::io(format!("failed to encode task '{}': {err}", task.id))
        })?;
        self.db
            .put_opt(task_key(&task.id), value, &sync_write_options())
            .map_err(|err| lance::Error::io(format!("failed to persist task '{}': {err}", task.id)))
    }

    pub fn list(&self) -> lance::Result<Vec<TaskRecord>> {
        let mut tasks = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(TASK_KEY_PREFIX, Direction::Forward))
        {
            let (key, value) =
                item.map_err(|err| lance::Error::io(format!("failed to iterate task DB: {err}")))?;
            if !key.starts_with(TASK_KEY_PREFIX) {
                break;
            }
            let task = serde_json::from_slice(&value).map_err(|err| {
                lance::Error::io(format!(
                    "failed to decode task record '{}': {err}",
                    String::from_utf8_lossy(&key)
                ))
            })?;
            tasks.push(task);
        }
        Ok(tasks)
    }

    pub fn delete_many<'a>(&self, ids: impl IntoIterator<Item = &'a str>) -> lance::Result<()> {
        let mut batch = WriteBatch::default();
        let mut count = 0usize;
        for id in ids {
            batch.delete(task_key(id));
            count += 1;
        }
        if count == 0 {
            return Ok(());
        }
        self.db
            .write_opt(batch, &sync_write_options())
            .map_err(|err| lance::Error::io(format!("failed to prune task history: {err}")))
    }
}

fn task_key(id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(TASK_KEY_PREFIX.len() + id.len());
    key.extend_from_slice(TASK_KEY_PREFIX);
    key.extend_from_slice(id.as_bytes());
    key
}

fn sync_write_options() -> WriteOptions {
    let mut options = WriteOptions::default();
    options.set_sync(true);
    options
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_api::{TaskKind, TaskState};
    use tempfile::TempDir;

    fn task(id: &str, state: TaskState, enqueued_at: i64) -> TaskRecord {
        TaskRecord {
            id: id.to_string(),
            kind: TaskKind::Compact,
            target: "experiment".to_string(),
            state,
            error: None,
            detail: None,
            enqueued_at,
            started_at: None,
            finished_at: None,
        }
    }

    #[test]
    fn survives_reopen_and_deletes_in_batch() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tasks");
        {
            let store = TaskStore::open(&path).unwrap();
            store.put(&task("a", TaskState::Queued, 1)).unwrap();
            store.put(&task("b", TaskState::Done, 2)).unwrap();
        }

        let store = TaskStore::open(&path).unwrap();
        let mut tasks = store.list().unwrap();
        tasks.sort_by_key(|task| task.enqueued_at);
        assert_eq!(
            tasks
                .iter()
                .map(|task| task.id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );

        store.delete_many(["a"]).unwrap();
        let tasks = store.list().unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "b");
    }

    #[test]
    fn rejects_object_store_uri() {
        let error = match TaskStore::open("s3://bucket/tasks") {
            Ok(_) => panic!("object-store URI should be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("must be local"));
    }
}
