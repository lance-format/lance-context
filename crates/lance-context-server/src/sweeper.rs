//! Periodic MemWAL maintenance, uniformly across every store kind.
//!
//! Two things have to happen on a timer for a MemWAL-backed store:
//!
//! - **flush** — seal the active memtable so durable-but-invisible rows become
//!   readable. Only matters for a store that defers the seal.
//! - **merge** — fold flushed generations back into the base table. Matters for
//!   *every* store: without it, `_mem_wal/` generations accumulate forever and
//!   every read unions all of them.
//!
//! Both sweepers used to hardcode `rollout_stores`, so datagen and generic
//! stores were never swept — datagen's generations grew without bound, and a
//! generic store with the default deferred seal stayed invisible until someone
//! called `/flush` by hand. Rather than copy the loop a third time (the exact
//! duplication issue #214 removed one layer down), the traversal is generic
//! over [`Sweepable`] and each store kind supplies a thin impl.

use std::sync::Arc;
use std::time::Duration;

use lance_context_core::{DatagenStore, GenericStore, RolloutStore};
use lru::LruCache;
use tokio::sync::{Mutex, RwLock};

/// A store the sweepers can maintain.
///
/// Implemented on `Arc<RwLock<Store>>` rather than on the store itself so each
/// kind decides its own locking. That is load-bearing for rollout, whose merge
/// deliberately splits into a shared-lock prepare (the expensive object-storage
/// reads, during which appends keep flowing) and a brief exclusive-lock commit.
/// A trait over `&mut Store` would have forced the exclusive lock across the
/// whole merge and quietly stalled the write path.
pub(crate) trait Sweepable: Send + Sync + 'static {
    /// Human-readable kind, for log and metric labels.
    fn kind() -> &'static str;

    /// Seal the active memtable. A no-op for a store that seals on write.
    fn flush(&self) -> impl std::future::Future<Output = Result<(), String>> + Send;

    /// Fold this instance's pending generations into the base table **if** the
    /// count trigger is configured and met; returns how many were reclaimed.
    /// Rides the flush timer so read amplification is bounded between the
    /// slower time-triggered passes. Default: no count trigger.
    fn merge_if_due(&self) -> impl std::future::Future<Output = Result<usize, String>> + Send {
        async { Ok(0) }
    }

    /// Fold **every** pending flushed generation into the base table; returns
    /// how many were reclaimed.
    fn merge_wal(&self) -> impl std::future::Future<Output = Result<usize, String>> + Send;
}

impl Sweepable for Arc<RwLock<RolloutStore>> {
    fn kind() -> &'static str {
        "rollout"
    }

    async fn flush(&self) -> Result<(), String> {
        // Read lock: `flush` is `&self`, so concurrent appends are not blocked.
        let guard = self.read().await;
        guard.flush().await.map_err(|e| e.to_string())
    }

    async fn merge_if_due(&self) -> Result<usize, String> {
        let prepared = {
            let guard = self.read().await;
            guard
                .prepare_count_merge()
                .await
                .map_err(|e| e.to_string())?
        };
        // Only now take the write lock, and only for the short commit.
        let Some((manifest_store, manifest, prepared)) = prepared else {
            return Ok(0);
        };
        let mut guard = self.write().await;
        guard
            .commit_prepared_merge(&manifest_store, &manifest, prepared)
            .await
            .map_err(|e| e.to_string())
    }

    async fn merge_wal(&self) -> Result<usize, String> {
        let prepared = {
            let guard = self.read().await;
            guard
                .prepare_cleanup_merge()
                .await
                .map_err(|e| e.to_string())?
        };
        // Only now take the write lock, and only for the short commit.
        let Some((manifest_store, manifest, prepared)) = prepared else {
            return Ok(0);
        };
        let mut guard = self.write().await;
        guard
            .commit_prepared_merge(&manifest_store, &manifest, prepared)
            .await
            .map_err(|e| e.to_string())
    }
}

impl Sweepable for Arc<RwLock<DatagenStore>> {
    fn kind() -> &'static str {
        "datagen"
    }

    async fn flush(&self) -> Result<(), String> {
        // Datagen seals on every append, so this is a no-op in steady state.
        // Kept for symmetry, and it still drains anything a fenced writer left.
        let guard = self.read().await;
        guard.flush().await.map_err(|e| e.to_string())
    }

    async fn merge_wal(&self) -> Result<usize, String> {
        let mut guard = self.write().await;
        guard.cleanup_own_shard().await.map_err(|e| e.to_string())
    }
}

impl Sweepable for Arc<RwLock<GenericStore>> {
    fn kind() -> &'static str {
        "generic"
    }

    async fn flush(&self) -> Result<(), String> {
        let guard = self.read().await;
        guard.flush().await.map_err(|e| e.to_string())
    }

    async fn merge_if_due(&self) -> Result<usize, String> {
        // Generic stores default to a deferred seal and take the same
        // one-row-per-append traffic as rollout, so the count trigger rides
        // this timer too. Without it a hot generic store depends entirely on
        // the slower cleanup sweeper reaching it.
        let prepared = {
            let guard = self.read().await;
            guard
                .prepare_count_merge()
                .await
                .map_err(|e| e.to_string())?
        };
        // Only now take the write lock, and only for the short commit.
        let Some((manifest_store, manifest, prepared)) = prepared else {
            return Ok(0);
        };
        let mut guard = self.write().await;
        guard
            .commit_prepared_merge(&manifest_store, &manifest, prepared)
            .await
            .map_err(|e| e.to_string())
    }

    async fn merge_wal(&self) -> Result<usize, String> {
        let prepared = {
            let guard = self.read().await;
            guard
                .prepare_cleanup_merge()
                .await
                .map_err(|e| e.to_string())?
        };
        // Only now take the write lock, and only for the short commit.
        let Some((manifest_store, manifest, prepared)) = prepared else {
            return Ok(0);
        };
        let mut guard = self.write().await;
        guard
            .commit_prepared_merge(&manifest_store, &manifest, prepared)
            .await
            .map_err(|e| e.to_string())
    }
}

/// Snapshot a cache's resident entries without holding its lock across the
/// awaits that follow.
pub(crate) async fn resident<S: Clone>(cache: &Mutex<LruCache<String, S>>) -> Vec<(String, S)> {
    cache
        .lock()
        .await
        .iter()
        .map(|(name, store)| (name.clone(), store.clone()))
        .collect()
}

/// Flush every resident store of one kind, bounding each by `pass_timeout` so a
/// single wedged store cannot stall the rest.
///
/// Metric names keep their historical `rollout_` prefix so existing dashboards
/// and alerts keep working; the new `kind` label is what distinguishes the
/// store types. Renaming them would be a silent breakage for anyone graphing
/// these today.
pub(crate) async fn flush_pass<S: Sweepable>(stores: Vec<(String, S)>, pass_timeout: Duration) {
    let kind = S::kind();
    for (name, store) in stores {
        match tokio::time::timeout(pass_timeout, store.flush()).await {
            Ok(Ok(())) => {
                metrics::counter!("rollout_wal_flush_total", "result" => "ok", "kind" => kind)
                    .increment(1);
                // The count-triggered merge rides this timer, but it is a merge,
                // not a flush: its outcome is reported under the cleanup
                // counters so a failing merge cannot masquerade as a failing
                // flush on the dashboards.
                report_merge(
                    &name,
                    kind,
                    tokio::time::timeout(pass_timeout, store.merge_if_due()).await,
                );
            }
            Ok(Err(error)) => {
                metrics::counter!("rollout_wal_flush_total", "result" => "failed", "kind" => kind)
                    .increment(1);
                tracing::warn!(store = %name, kind, %error, "flush sweeper failed");
            }
            Err(_elapsed) => {
                metrics::counter!("rollout_wal_flush_total", "result" => "timeout", "kind" => kind)
                    .increment(1);
                tracing::warn!(store = %name, kind, "flush sweeper timed out");
            }
        }
    }
}

/// Merge every resident store's pending generations, same timeout discipline.
pub(crate) async fn merge_pass<S: Sweepable>(stores: Vec<(String, S)>, pass_timeout: Duration) {
    let kind = S::kind();
    for (name, store) in stores {
        report_merge(
            &name,
            kind,
            tokio::time::timeout(pass_timeout, store.merge_wal()).await,
        );
    }
}

/// Record one merge attempt's outcome on the cleanup counters and log.
fn report_merge(
    name: &str,
    kind: &'static str,
    outcome: Result<Result<usize, String>, tokio::time::error::Elapsed>,
) {
    match outcome {
        Ok(Ok(0)) => {}
        Ok(Ok(reclaimed)) => {
            metrics::counter!("rollout_wal_cleanup_total", "result" => "merged", "kind" => kind)
                .increment(1);
            metrics::counter!("rollout_wal_generations_reclaimed_total", "kind" => kind)
                .increment(reclaimed as u64);
            tracing::info!(store = %name, kind, reclaimed, "sweeper merged flushed generations");
        }
        Ok(Err(error)) => {
            metrics::counter!("rollout_wal_cleanup_total", "result" => "failed", "kind" => kind)
                .increment(1);
            tracing::warn!(store = %name, kind, %error, "sweeper WAL cleanup failed");
        }
        Err(_elapsed) => {
            metrics::counter!("rollout_wal_cleanup_total", "result" => "timeout", "kind" => kind)
                .increment(1);
            tracing::warn!(
                store = %name,
                kind,
                "sweeper WAL cleanup timed out; abandoning this store this tick"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_api::{ColumnSpec, ColumnType, SchemaSpec, ID_COLUMN};
    use lance_context_core::GenericStoreOptions;
    use serde_json::json;
    use tempfile::TempDir;

    fn spec() -> SchemaSpec {
        SchemaSpec::new(vec![(
            ID_COLUMN.to_string(),
            ColumnSpec::required(ColumnType::String { large: false }),
        )])
    }

    async fn generic_with_pending(
        dir: &TempDir,
        merge_after_generations: usize,
        pending: usize,
    ) -> Arc<RwLock<GenericStore>> {
        let uri = dir.path().to_string_lossy().to_string();
        let store = GenericStore::open(
            &uri,
            spec(),
            GenericStoreOptions {
                merge_after_generations: Some(merge_after_generations),
                seal_on_add: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        for i in 0..pending {
            store
                .add(&[json!({"id": format!("r{i}")}).as_object().unwrap().clone()])
                .await
                .unwrap();
        }
        assert_eq!(store.pending_wal_generations().await.unwrap(), pending);
        Arc::new(RwLock::new(store))
    }

    /// The count trigger is what bounds read amplification between the slower
    /// time-triggered passes; a generic store at the threshold must merge on a
    /// flush tick. This is the path #256 added and #257 fixed the locking of.
    #[tokio::test]
    async fn generic_merge_if_due_folds_pending_generations_at_threshold() {
        let dir = TempDir::new().unwrap();
        let store = generic_with_pending(&dir, 3, 3).await;

        let reclaimed = store.merge_if_due().await.unwrap();

        assert_eq!(reclaimed, 3);
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            0
        );
        assert_eq!(store.read().await.count_base_rows().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn generic_merge_if_due_is_a_noop_below_threshold() {
        let dir = TempDir::new().unwrap();
        let store = generic_with_pending(&dir, 5, 2).await;

        assert_eq!(store.merge_if_due().await.unwrap(), 0);
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            2
        );
    }

    /// `merge_after_generations = 0` means "count trigger off". It must not be
    /// read as "threshold 0, merge every tick" — which is what the underlying
    /// `threshold.max(1)` would do if the zero were passed straight through.
    #[tokio::test]
    async fn generic_merge_if_due_respects_disabled_count_trigger() {
        let dir = TempDir::new().unwrap();
        let store = generic_with_pending(&dir, 0, 4).await;

        assert_eq!(store.merge_if_due().await.unwrap(), 0);
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            4
        );
        // The time trigger is unaffected and still drains everything.
        assert_eq!(store.merge_wal().await.unwrap(), 4);
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            0
        );
    }

    /// The expensive half of a merge must not hold the exclusive lock: a
    /// concurrent reader takes the shared lock while `prepare` is in flight.
    /// If `merge_if_due` held the write lock across the read, this would
    /// deadlock (the reader waits on the merge, the merge holds the lock).
    #[tokio::test]
    async fn generic_merge_if_due_does_not_block_readers_during_prepare() {
        let dir = TempDir::new().unwrap();
        let store = generic_with_pending(&dir, 3, 3).await;

        // Hold a read lock for the whole merge. With the prepare/commit split,
        // prepare proceeds under a second shared lock and commit waits only for
        // this guard to drop; without the split the merge would need the write
        // lock up front and this test would hang.
        let reader = store.read().await;
        let merge = tokio::spawn({
            let store = store.clone();
            async move { store.merge_if_due().await }
        });
        // Give the merge time to reach the commit phase (it needs the write
        // lock there), proving prepare completed under the shared lock.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!merge.is_finished(), "commit must wait for the live reader");
        assert_eq!(reader.pending_wal_generations().await.unwrap(), 3);
        drop(reader);

        let reclaimed = tokio::time::timeout(Duration::from_secs(30), merge)
            .await
            .expect("merge must finish once the reader releases the lock")
            .unwrap()
            .unwrap();
        assert_eq!(reclaimed, 3);
    }
}
