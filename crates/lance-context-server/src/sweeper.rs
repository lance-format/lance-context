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
/// kind decides its own locking. Merge/flush/commit are `&self` on the store
/// (dataset handle is ArcSwap), so these impls only need a shared lock —
/// concurrent appends keep flowing.
pub(crate) trait Sweepable: Send + Sync + 'static {
    /// Human-readable kind, for log and metric labels.
    fn kind() -> &'static str;

    /// Seal the active memtable. A no-op for a store that seals on write.
    fn flush(&self) -> impl std::future::Future<Output = Result<(), String>> + Send;

    /// Fold pending flushed generations into the base table; returns how many
    /// were reclaimed.
    fn merge_wal(&self) -> impl std::future::Future<Output = Result<usize, String>> + Send;
}

impl Sweepable for Arc<RwLock<RolloutStore>> {
    fn kind() -> &'static str {
        "rollout"
    }

    async fn flush(&self) -> Result<(), String> {
        let guard = self.read().await;
        let result = guard.flush().await.map_err(|e| e.to_string());
        if result.is_ok() {
            // The count-triggered merge rides this timer; it is a no-op unless
            // the threshold is configured and met.
            drop(guard);
            let guard = self.read().await;
            guard
                .maybe_merge_own_shard()
                .await
                .map_err(|e| e.to_string())?;
        }
        result
    }

    async fn merge_wal(&self) -> Result<usize, String> {
        let prepared = {
            let guard = self.read().await;
            guard
                .prepare_cleanup_merge()
                .await
                .map_err(|e| e.to_string())?
        };
        let Some((manifest_store, manifest, prepared)) = prepared else {
            return Ok(0);
        };
        let guard = self.read().await;
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
        let guard = self.read().await;
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

    async fn merge_wal(&self) -> Result<usize, String> {
        let guard = self.read().await;
        guard.cleanup_wal().await.map_err(|e| e.to_string())
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
        match tokio::time::timeout(pass_timeout, store.merge_wal()).await {
            Ok(Ok(0)) => {}
            Ok(Ok(reclaimed)) => {
                metrics::counter!("rollout_wal_cleanup_total", "result" => "merged", "kind" => kind)
                    .increment(1);
                metrics::counter!("rollout_wal_generations_reclaimed_total", "kind" => kind)
                    .increment(reclaimed as u64);
                tracing::info!(
                    store = %name,
                    kind,
                    reclaimed,
                    "sweeper merged flushed generations"
                );
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
}
