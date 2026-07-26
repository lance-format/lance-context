use std::num::NonZeroUsize;
#[cfg(test)]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use lance_context_core::{
    join_uri, validate_store_name, ContextStore, ContextStoreOptions, RolloutRegistry,
    RolloutStore, RolloutStoreOptions, Session,
};
use lru::LruCache;
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

use crate::config::ServerConfig;
use crate::error::AppError;

/// Default upper bound on resident rollout-store handles when the config does
/// not specify one. Sized for peak *concurrent* experiments, not the total
/// number of experiments (which may be hundreds of thousands).
pub const DEFAULT_ROLLOUT_CACHE_CAPACITY: usize = 2000;

pub struct AppState {
    pub stores: RwLock<std::collections::HashMap<String, Arc<RwLock<ContextStore>>>>,
    /// Bounded LRU of resident rollout-store handles.
    ///
    /// With one physical dataset per experiment, the deployment may hold
    /// hundreds of thousands of stores; keeping them all resident would exhaust
    /// memory and (formerly) spawn one background timer each. This cache bounds
    /// residency: on overflow the least-recently-used handle is evicted and its
    /// `Arc<RwLock<RolloutStore>>` dropped. Existence is tracked durably by
    /// [`Self::rollout_registry`], not by membership in this cache.
    ///
    /// # Eviction does not strand unflushed rows
    ///
    /// The flush sweeper only visits *resident* stores, so it is worth being
    /// explicit about why evicting a store that still has an unsealed memtable
    /// is safe: dropping the last handle runs `RolloutStore`'s `Drop`, which
    /// spawns a detached `ShardWriter::close`, and that seals the active
    /// memtable before draining. An evicted store's rows therefore still become
    /// visible without the sweeper ever seeing it.
    ///
    /// The residual risk is that the detached close is best-effort — it cannot
    /// be awaited from `drop`, and it no-ops if an in-flight append still shares
    /// the `Arc`. Both cases are logged (see that `Drop` impl) rather than
    /// silently stranding rows.
    pub rollout_stores: Mutex<LruCache<String, Arc<RwLock<RolloutStore>>>>,
    /// Durable directory of which rollout stores exist. Consulted on a cache
    /// miss (existence check) and to back the list endpoint. Guarded by a lock
    /// because every operation refreshes the snapshot and therefore takes
    /// `&mut`.
    pub rollout_registry: RwLock<RolloutRegistry>,
    pub base_uri: String,
    /// Stable identity of this server instance, used as the MemWAL shard key for
    /// rollout writes so each instance owns exactly one shard. `None` falls back
    /// to a single shared shard (single-instance deployments only).
    pub instance_id: Option<String>,
    /// Count-triggered self-merge threshold for rollout MemWAL shards; `0`
    /// disables it. See `RolloutStoreOptions::merge_after_generations`.
    pub rollout_merge_after_generations: usize,
    /// Periodic per-shard WAL-cleanup interval in seconds; `0` disables the
    /// global sweeper. See [`Self::spawn_global_sweeper`].
    pub rollout_cleanup_interval_secs: u64,
    /// Periodic MemWAL flush interval in seconds; `0` disables periodic flush.
    /// Bounds rollout read-after-write latency (appends are durable but not
    /// visible until flushed). See [`Self::spawn_global_sweeper`].
    pub rollout_flush_interval_secs: u64,
    /// Admission budget for in-flight artifact-blob bytes across concurrent
    /// uploads/downloads. `None` disables the budget (unbounded). See
    /// [`BlobBudget`].
    pub blob_budget: Option<Arc<BlobBudget>>,
    /// Shared Lance cache session attached to every resident rollout store, so
    /// the process's total metadata/index cache is bounded by one budget rather
    /// than Lance's default 6 GiB *per store*. `None` restores the per-store
    /// default session (leak-prone; only when the budget is configured to `0`).
    pub rollout_session: Option<Arc<Session>>,
}

/// Process-wide admission control for the total artifact-blob payload held in
/// memory across concurrent rollout uploads and downloads.
///
/// Each blob request materializes its whole payload as an in-memory buffer, so
/// unbounded concurrency of large requests can OOM the worker. A request calls
/// [`BlobBudget::try_acquire`] with its byte size before allocating; the guard
/// returned holds the reservation and releases it on drop (i.e. when the
/// request completes). When the budget cannot fit the request the caller
/// rejects it with `503` rather than proceeding to allocate.
///
/// This bounds *concurrency*, not maximum blob size: a single request larger
/// than the entire budget is still admitted when nothing else is in flight (it
/// transiently reserves the full budget), so a lone big download never
/// deadlocks against its own limit.
#[derive(Debug)]
pub struct BlobBudget {
    limit: usize,
    used: std::sync::atomic::AtomicUsize,
}

/// RAII reservation returned by [`BlobBudget::try_acquire`]. Releases its bytes
/// back to the budget when dropped.
#[derive(Debug)]
pub struct BlobReservation {
    budget: Arc<BlobBudget>,
    bytes: usize,
}

impl BlobBudget {
    /// Create a budget admitting up to `limit` concurrent in-flight blob bytes.
    #[must_use]
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Reserve `bytes` if they fit alongside what is already in flight. Returns
    /// `None` (request should be rejected with `503`) when admitting the request
    /// would exceed the limit, *unless* nothing is currently in flight — in that
    /// case an over-limit request is admitted so a lone large blob is never
    /// permanently rejected.
    pub fn try_acquire(self: &Arc<Self>, bytes: usize) -> Option<BlobReservation> {
        use std::sync::atomic::Ordering;
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            // Admit when it fits, or when the instance is otherwise idle (so a
            // single request bigger than the whole budget can still proceed).
            let fits = current + bytes <= self.limit;
            if !fits && current != 0 {
                return None;
            }
            match self.used.compare_exchange_weak(
                current,
                current + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(BlobReservation {
                        budget: Arc::clone(self),
                        bytes,
                    })
                }
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for BlobReservation {
    fn drop(&mut self) {
        self.budget
            .used
            .fetch_sub(self.bytes, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Build the process-wide shared Lance cache session for rollout stores from a
/// single total byte budget, or `None` when the budget is `0` (which restores
/// Lance's per-store default session).
///
/// The budget is split 6:1 index:metadata to mirror Lance's own default ratio
/// (`DEFAULT_INDEX_CACHE_SIZE` = 6 GiB, `DEFAULT_METADATA_CACHE_SIZE` = 1 GiB).
fn build_rollout_session(cache_bytes: usize) -> Option<Arc<Session>> {
    if cache_bytes == 0 {
        return None;
    }
    let metadata_bytes = cache_bytes / 7;
    let index_bytes = cache_bytes - metadata_bytes;
    Some(RolloutStore::build_session(index_bytes, metadata_bytes))
}

impl AppState {
    /// Build the shared server state, opening (or creating) the rollout registry
    /// under `data_dir`. Async because opening the registry touches storage.
    pub async fn new(config: ServerConfig) -> Result<Self, AppError> {
        let instance_id = config.resolved_instance_id();
        let base_uri = config.data_dir.clone();
        let registry_uri = join_uri(&base_uri, "_registry.rollout.lance");
        let registry = RolloutRegistry::open_or_create(&registry_uri, None)
            .await
            .map_err(AppError::from_lance)?;
        let capacity = NonZeroUsize::new(config.rollout_cache_capacity)
            .unwrap_or_else(|| NonZeroUsize::new(DEFAULT_ROLLOUT_CACHE_CAPACITY).unwrap());
        let blob_budget = (config.rollout_max_inflight_blob_bytes > 0)
            .then(|| BlobBudget::new(config.rollout_max_inflight_blob_bytes));
        let rollout_session = build_rollout_session(config.rollout_cache_bytes);
        Ok(Self {
            stores: RwLock::new(std::collections::HashMap::new()),
            rollout_stores: Mutex::new(LruCache::new(capacity)),
            rollout_registry: RwLock::new(registry),
            base_uri,
            instance_id,
            rollout_merge_after_generations: config.rollout_merge_after_generations,
            rollout_cleanup_interval_secs: config.rollout_cleanup_interval_secs,
            rollout_flush_interval_secs: config.rollout_flush_interval_secs,
            blob_budget,
            rollout_session,
        })
    }

    pub fn context_uri(&self, name: &str) -> String {
        join_uri(&self.base_uri, &format!("{}.lance", name))
    }

    /// Build a default-configured `AppState` rooted at `base_path`, for tests.
    /// Opens a fresh registry under the directory; cleanup interval disabled.
    #[cfg(test)]
    pub async fn new_for_test(base_path: PathBuf) -> Self {
        Self::new_for_test_with_instance(base_path, None).await
    }

    /// Like [`Self::new_for_test`] but with an explicit MemWAL instance id, so a
    /// second in-process "instance" can present its own shard.
    #[cfg(test)]
    pub async fn new_for_test_with_instance(
        base_path: PathBuf,
        instance_id: Option<String>,
    ) -> Self {
        let base_uri = base_path.to_string_lossy().to_string();
        let registry_uri = join_uri(&base_uri, "_registry.rollout.lance");
        let registry = RolloutRegistry::open_or_create(&registry_uri, None)
            .await
            .expect("open test registry");
        Self {
            stores: RwLock::new(std::collections::HashMap::new()),
            rollout_stores: Mutex::new(LruCache::new(
                NonZeroUsize::new(DEFAULT_ROLLOUT_CACHE_CAPACITY).unwrap(),
            )),
            rollout_registry: RwLock::new(registry),
            base_uri,
            instance_id,
            rollout_merge_after_generations: 0,
            rollout_cleanup_interval_secs: 0,
            rollout_flush_interval_secs: 0,
            blob_budget: None,
            rollout_session: build_rollout_session(2 * 1024 * 1024 * 1024),
        }
    }

    /// Rollout datasets live under a distinct `.rollout.lance` suffix so a
    /// rollout store and a context store may share the same name without
    /// colliding on disk.
    pub fn rollout_uri(&self, name: &str) -> String {
        join_uri(&self.base_uri, &format!("{}.rollout.lance", name))
    }

    fn rollout_store_options(&self) -> RolloutStoreOptions {
        RolloutStoreOptions {
            // No request body on the read path: object-store credentials come
            // from the pod's workload-identity environment.
            storage_options: None,
            shard_id: self.instance_id.clone(),
            merge_after_generations: (self.rollout_merge_after_generations > 0)
                .then_some(self.rollout_merge_after_generations),
            session: self.rollout_session.clone(),
        }
    }

    /// Record that a rollout store exists, in both the durable registry and the
    /// in-memory LRU. Called by the create route after the dataset is written.
    pub async fn register_rollout(
        &self,
        name: &str,
        uri: &str,
        store: Arc<RwLock<RolloutStore>>,
    ) -> Result<(), AppError> {
        Self::validate_name(name)?;
        self.rollout_registry
            .write()
            .await
            .upsert(name, uri)
            .await
            .map_err(AppError::from_lance)?;
        // Insertion may evict the LRU's least-recently-used entry; dropping the
        // returned handle releases it (no per-store timer to abort — cleanup is
        // now global).
        self.rollout_stores
            .lock()
            .await
            .put(name.to_string(), store);
        Ok(())
    }

    /// Remove a rollout store from the durable registry and evict any resident
    /// handle. Returns whether the store existed.
    pub async fn unregister_rollout(&self, name: &str) -> Result<bool, AppError> {
        Self::validate_name(name)?;
        let existed = self
            .rollout_registry
            .write()
            .await
            .contains(name)
            .await
            .map_err(AppError::from_lance)?;
        if !existed {
            return Ok(false);
        }
        self.rollout_registry
            .write()
            .await
            .remove(name)
            .await
            .map_err(AppError::from_lance)?;
        self.rollout_stores.lock().await.pop(name);
        Ok(true)
    }

    /// Look up a rollout store by name, lazily loading it from object storage on
    /// a local cache miss.
    ///
    /// Unlike the previous implementation, a cache miss is **not** a 404: the
    /// bounded LRU may have evicted a store that still exists. Existence is
    /// resolved against the durable [`RolloutRegistry`]; only a store absent
    /// from the registry yields [`AppError::NotFound`]. This closes the
    /// multi-replica gap (create on pod A, read on pod B) *and* the
    /// eviction-induced false-404 that per-experiment partitioning introduces.
    pub async fn get_or_open_rollout_store(
        &self,
        name: &str,
    ) -> Result<Arc<RwLock<RolloutStore>>, AppError> {
        Self::validate_name(name)?;
        // Fast path: resident in this process's LRU (updates recency).
        if let Some(store) = self.rollout_stores.lock().await.get(name) {
            metrics::counter!("rollout_store_cache_hits_total").increment(1);
            return Ok(store.clone());
        }
        metrics::counter!("rollout_store_cache_misses_total").increment(1);

        // Existence is the registry's job, not the cache's.
        let exists = self
            .rollout_registry
            .write()
            .await
            .contains(name)
            .await
            .map_err(AppError::from_lance)?;
        if !exists {
            return Err(AppError::NotFound(format!(
                "Rollout store '{}' does not exist",
                name
            )));
        }

        // Load from storage WITHOUT holding the LRU lock, so a slow open does
        // not block other stores' requests.
        let uri = self.rollout_uri(name);
        let opened = RolloutStore::open_existing_with_options(&uri, self.rollout_store_options())
            .await
            .map_err(AppError::from_lance)?;
        let opened = Arc::new(RwLock::new(opened));

        // Insert under the lock, re-checking for a store another request may
        // have opened concurrently while we were loading.
        let mut cache = self.rollout_stores.lock().await;
        if let Some(existing) = cache.get(name) {
            return Ok(existing.clone());
        }
        cache.put(name.to_string(), opened.clone());
        metrics::gauge!("rollout_stores_resident").set(cache.len() as f64);
        Ok(opened)
    }

    /// Look up a context store by name, lazily loading it from object storage on
    /// a local cache miss. See [`Self::get_or_open_rollout_store`] for the
    /// multi-replica rationale; the same in-memory-cache-as-existence-check bug
    /// affects context stores.
    pub async fn get_or_open_context_store(
        &self,
        name: &str,
    ) -> Result<Arc<RwLock<ContextStore>>, AppError> {
        Self::validate_name(name)?;
        if let Some(store) = self.stores.read().await.get(name) {
            return Ok(store.clone());
        }

        let uri = self.context_uri(name);
        let opened = ContextStore::open_existing_with_options(&uri, ContextStoreOptions::default())
            .await
            .map_err(AppError::from_lance)?;
        let opened = Arc::new(RwLock::new(opened));

        let mut stores = self.stores.write().await;
        if let Some(existing) = stores.get(name) {
            return Ok(existing.clone());
        }
        stores.insert(name.to_string(), opened.clone());
        Ok(opened)
    }

    pub fn validate_name(name: &str) -> Result<(), AppError> {
        validate_store_name(name).map_err(AppError::InvalidRequest)
    }

    /// Spawn the single, process-wide WAL-cleanup sweeper.
    ///
    /// This replaces the former one-timer-per-store model, which does not scale
    /// to hundreds of thousands of per-experiment datasets. On each tick the
    /// sweeper snapshots the *resident* rollout stores (those in the LRU) and
    /// folds each one's flushed MemWAL generations into its base table, guarding
    /// every pass with a timeout so a wedged store cannot stall the sweeper (see
    /// [`RolloutStore::cleanup_own_shard`] and the timeout added in the periodic
    /// cleanup hardening).
    ///
    /// Cold stores (evicted from the LRU) are not swept: having no new writes,
    /// they accumulate no new generations, and are swept again the moment a
    /// request re-opens them. Returns `None` when the interval is `0`.
    pub fn spawn_global_sweeper(self: &Arc<Self>) -> Option<JoinHandle<()>> {
        let interval_secs = self.rollout_cleanup_interval_secs;
        if interval_secs == 0 {
            return None;
        }
        let interval = Duration::from_secs(interval_secs);
        // Abandon any single pass that outruns five intervals (min 30s) so one
        // stuck store cannot wedge cleanup for every other store.
        let pass_timeout = interval.saturating_mul(5).max(Duration::from_secs(30));
        let weak = Arc::downgrade(self);
        Some(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await; // skip the immediate first tick
            loop {
                ticker.tick().await;
                let Some(state) = weak.upgrade() else {
                    return;
                };
                // Snapshot resident stores without holding the LRU lock across
                // the awaits below.
                let resident: Vec<(String, Arc<RwLock<RolloutStore>>)> = {
                    let cache = state.rollout_stores.lock().await;
                    cache
                        .iter()
                        .map(|(name, store)| (name.clone(), store.clone()))
                        .collect()
                };
                for (name, store) in resident {
                    // Same read-lock/write-lock split as the flush sweeper: seal
                    // and read under the shared lock so appends keep flowing,
                    // then commit under the exclusive lock. Threshold 1 — merge
                    // whatever is pending, which is what makes this the *time*
                    // half of the "time OR count" trigger.
                    let prepared = {
                        let guard = store.read().await;
                        match tokio::time::timeout(pass_timeout, guard.prepare_cleanup_merge())
                            .await
                        {
                            Ok(Ok(prepared)) => prepared,
                            Ok(Err(e)) => {
                                metrics::counter!("rollout_wal_cleanup_total", "result" => "failed")
                                    .increment(1);
                                tracing::warn!(
                                    store = %name,
                                    error = %e,
                                    "global sweeper WAL cleanup failed"
                                );
                                continue;
                            }
                            Err(_elapsed) => {
                                metrics::counter!("rollout_wal_cleanup_total", "result" => "timeout")
                                    .increment(1);
                                tracing::warn!(
                                    store = %name,
                                    "global sweeper WAL cleanup timed out; abandoning this store this tick"
                                );
                                continue;
                            }
                        }
                    };
                    let Some((manifest_store, manifest, prepared)) = prepared else {
                        continue;
                    };
                    let mut guard = store.write().await;
                    match tokio::time::timeout(
                        pass_timeout,
                        guard.commit_prepared_merge(&manifest_store, &manifest, prepared),
                    )
                    .await
                    {
                        Ok(Ok(n)) => {
                            metrics::counter!("rollout_wal_cleanup_total", "result" => "merged")
                                .increment(1);
                            metrics::counter!("rollout_wal_generations_reclaimed_total")
                                .increment(n as u64);
                            tracing::info!(
                                store = %name,
                                reclaimed = n,
                                "global sweeper merged flushed generations"
                            )
                        }
                        Ok(Err(e)) => {
                            metrics::counter!("rollout_wal_cleanup_total", "result" => "failed")
                                .increment(1);
                            tracing::warn!(
                                store = %name,
                                error = %e,
                                "global sweeper WAL cleanup failed"
                            )
                        }
                        Err(_elapsed) => {
                            metrics::counter!("rollout_wal_cleanup_total", "result" => "timeout")
                                .increment(1);
                            tracing::warn!(
                                store = %name,
                                "global sweeper WAL cleanup timed out; abandoning this store this tick"
                            )
                        }
                    }
                }
            }
        }))
    }

    /// Spawn the process-wide MemWAL flush sweeper.
    ///
    /// Rollout appends are durable on return (the WAL entry is persisted) but not
    /// visible to reads until the active memtable is flushed into a queryable
    /// generation. Rather than flush on every append — which would serialize
    /// concurrent writes behind a per-append seal — this sweeper flushes each
    /// resident store on a fixed interval, bounding read-after-write latency
    /// while keeping the append path concurrent.
    ///
    /// After flushing a store it also runs the count-triggered merge
    /// ([`RolloutStore::maybe_merge_own_shard`]): the read-amplification bound
    /// that formerly lived on the append path now rides this timer. The heavier
    /// time-based cleanup/merge remains on [`Self::spawn_global_sweeper`].
    ///
    /// Returns `None` when the flush interval is `0`.
    pub fn spawn_flush_sweeper(self: &Arc<Self>) -> Option<JoinHandle<()>> {
        let interval_secs = self.rollout_flush_interval_secs;
        if interval_secs == 0 {
            return None;
        }
        let interval = Duration::from_secs(interval_secs);
        // Abandon any single store's flush that outruns five intervals (min 30s)
        // so one wedged store cannot stall flushing for the rest.
        let pass_timeout = interval.saturating_mul(5).max(Duration::from_secs(30));
        let weak = Arc::downgrade(self);
        Some(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await; // skip the immediate first tick
            loop {
                ticker.tick().await;
                let Some(state) = weak.upgrade() else {
                    return;
                };
                let resident: Vec<(String, Arc<RwLock<RolloutStore>>)> = {
                    let cache = state.rollout_stores.lock().await;
                    cache
                        .iter()
                        .map(|(name, store)| (name.clone(), store.clone()))
                        .collect()
                };
                for (name, store) in resident {
                    // Flush under a read lock so concurrent appends are not blocked.
                    {
                        let guard = store.read().await;
                        match tokio::time::timeout(pass_timeout, guard.flush()).await {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => {
                                metrics::counter!("rollout_wal_flush_total", "result" => "failed")
                                    .increment(1);
                                tracing::warn!(store = %name, error = %e, "flush sweeper failed");
                                continue;
                            }
                            Err(_elapsed) => {
                                metrics::counter!("rollout_wal_flush_total", "result" => "timeout")
                                    .increment(1);
                                tracing::warn!(store = %name, "flush sweeper timed out");
                                continue;
                            }
                        }
                        metrics::counter!("rollout_wal_flush_total", "result" => "ok").increment(1);
                    }
                    // Count-triggered merge (no-op unless the threshold is set
                    // and met).
                    //
                    // Split by lock scope: the expensive half (sealing the
                    // memtable and reading every flushed generation out of
                    // object storage) runs under the *read* lock, so appends
                    // keep flowing through it. Only the short commit — append to
                    // the base table, drain the manifest, delete the merged
                    // dirs — takes the write lock. Previously the whole merge
                    // held the write lock, which stalled every concurrent append
                    // for its full duration.
                    let threshold = state.rollout_merge_after_generations;
                    if threshold == 0 {
                        continue;
                    }
                    let prepared = {
                        let guard = store.read().await;
                        match tokio::time::timeout(
                            pass_timeout,
                            guard.prepare_merge_if_ready(threshold),
                        )
                        .await
                        {
                            Ok(Ok(prepared)) => prepared,
                            Ok(Err(e)) => {
                                tracing::warn!(store = %name, error = %e, "flush sweeper merge failed");
                                continue;
                            }
                            Err(_elapsed) => {
                                tracing::warn!(store = %name, "flush sweeper merge timed out");
                                continue;
                            }
                        }
                    };
                    let Some((manifest_store, manifest, prepared)) = prepared else {
                        continue;
                    };
                    let mut guard = store.write().await;
                    match tokio::time::timeout(
                        pass_timeout,
                        guard.commit_prepared_merge(&manifest_store, &manifest, prepared),
                    )
                    .await
                    {
                        Ok(Ok(n)) => {
                            metrics::counter!("rollout_wal_generations_reclaimed_total")
                                .increment(n as u64);
                            tracing::info!(
                                store = %name,
                                reclaimed = n,
                                "flush sweeper count-merged flushed generations"
                            );
                        }
                        Ok(Err(e)) => {
                            tracing::warn!(store = %name, error = %e, "flush sweeper merge failed");
                        }
                        Err(_elapsed) => {
                            tracing::warn!(store = %name, "flush sweeper merge timed out");
                        }
                    }
                }
            }
        }))
    }
    ///
    /// [`RolloutStore`]'s writer ([`ShardWriter`]) has no `Drop`, so its
    /// background tasks are only reclaimed by an explicit `close().await`. On the
    /// normal serving path an LRU eviction owning the last handle drops it, and
    /// the `Drop` impl spawns a detached best-effort close — but at process
    /// shutdown the runtime is about to stop, so a detached task may never run.
    /// This walks the resident cache and awaits [`RolloutStore::close`] on each
    /// handle so writer background tasks are drained deterministically before the
    /// runtime tears down. Idempotent and safe to call once after the server
    /// stops accepting connections.
    pub async fn shutdown(&self) {
        // Snapshot resident stores without holding the LRU lock across the awaits.
        let resident: Vec<(String, Arc<RwLock<RolloutStore>>)> = {
            let cache = self.rollout_stores.lock().await;
            cache
                .iter()
                .map(|(name, store)| (name.clone(), store.clone()))
                .collect()
        };
        for (name, store) in resident {
            if let Err(e) = store.write().await.close().await {
                tracing::warn!(
                    store = %name,
                    error = %e,
                    "failed to close rollout writer during shutdown"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn blob_budget_admits_until_full_then_releases_on_drop() {
        let budget = BlobBudget::new(1000);
        // Two reservations that together fit.
        let a = budget.try_acquire(600).expect("first fits");
        let b = budget.try_acquire(300).expect("second fits (900 <= 1000)");
        // A third that would overflow is rejected while others are in flight.
        assert!(
            budget.try_acquire(200).is_none(),
            "over-limit request rejected while budget is occupied"
        );
        // Dropping frees the bytes back.
        drop(b);
        let c = budget.try_acquire(200).expect("fits again after release");
        drop(a);
        drop(c);
        // Fully drained: a request equal to the whole budget now fits.
        assert!(budget.try_acquire(1000).is_some());
    }

    #[test]
    fn blob_budget_admits_a_lone_oversized_request() {
        // A single request larger than the entire budget is admitted when the
        // instance is idle, so a lone big blob is never permanently rejected —
        // the budget bounds concurrency, not maximum blob size.
        let budget = BlobBudget::new(100);
        let big = budget
            .try_acquire(500)
            .expect("lone oversized request admitted");
        // But while it holds the (over-)reservation, nothing else gets in.
        assert!(budget.try_acquire(1).is_none());
        drop(big);
        assert!(budget.try_acquire(50).is_some());
    }

    async fn state_with_interval(dir: &TempDir, secs: u64) -> Arc<AppState> {
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        state.rollout_cleanup_interval_secs = secs;
        Arc::new(state)
    }

    /// The sweeper is a single detached task, gated on the cleanup interval:
    /// spawned when non-zero, `None` when disabled. This is the whole-process
    /// replacement for the former one-timer-per-store model.
    #[tokio::test]
    async fn global_sweeper_is_gated_on_interval() {
        let dir = TempDir::new().unwrap();
        let disabled = state_with_interval(&dir, 0).await;
        assert!(disabled.spawn_global_sweeper().is_none());

        let dir2 = TempDir::new().unwrap();
        let enabled = state_with_interval(&dir2, 3600).await;
        let handle = enabled
            .spawn_global_sweeper()
            .expect("sweeper spawns when interval > 0");
        handle.abort();
    }

    /// `shutdown` drains resident rollout writers by awaiting `close()` on each
    /// (idempotent: `close` is a no-op when no writer is resident), so
    /// `ShardWriter` background tasks are reclaimed deterministically instead of
    /// relying on the detached best-effort `Drop`. It must also be a no-op when
    /// the cache is empty.
    #[tokio::test]
    async fn shutdown_closes_resident_writers() {
        use lance_context_core::RolloutStore;

        let dir = TempDir::new().unwrap();
        let state = state_with_interval(&dir, 0).await;

        // Empty cache: shutdown is a no-op and returns promptly.
        state.shutdown().await;

        // Register a resident store, then shut down. The store's writer (if any)
        // is closed under the write lock; the call must complete without error
        // and leave the handle usable/droppable.
        let uri = state.rollout_uri("exp");
        let store = RolloutStore::open_with_options(&uri, state.rollout_store_options())
            .await
            .unwrap();
        let store = Arc::new(RwLock::new(store));
        state
            .register_rollout("exp", &uri, store.clone())
            .await
            .unwrap();

        state.shutdown().await;

        // The handle survives shutdown (shutdown only drains the writer); a
        // fresh close is still a no-op.
        store.write().await.close().await.unwrap();
    }
}
