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
use tokio::sync::{Mutex, RwLock, Semaphore};

/// A store the sweepers can maintain.
///
/// Implemented on `Arc<RwLock<Store>>` rather than on the store itself so each
/// kind decides its own locking. That is load-bearing for rollout, whose merge
/// deliberately splits into a shared-lock prepare (the expensive object-storage
/// reads, during which appends keep flowing) and a brief exclusive-lock commit.
/// A trait over `&mut Store` would have forced the exclusive lock across the
/// whole merge and quietly stalled the write path.
pub(crate) trait Sweepable: Clone + Send + Sync + 'static {
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

    /// Cheap metadata-only predicate used when the master owns execution.
    fn count_merge_due(&self) -> impl std::future::Future<Output = Result<bool, String>> + Send {
        async { Ok(false) }
    }

    /// This writer's shard watermark, metadata only. `None` when the store
    /// has no shard manifest yet or does not publish demand.
    fn own_shard_watermark(
        &self,
    ) -> impl std::future::Future<Output = Result<Option<lance_context_core::ShardWatermark>, String>>
           + Send {
        async { Ok(None) }
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

    async fn count_merge_due(&self) -> Result<bool, String> {
        self.read()
            .await
            .count_merge_due()
            .await
            .map_err(|e| e.to_string())
    }

    async fn own_shard_watermark(
        &self,
    ) -> Result<Option<lance_context_core::ShardWatermark>, String> {
        self.read()
            .await
            .own_shard_watermark()
            .await
            .map_err(|e| e.to_string())
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

    async fn count_merge_due(&self) -> Result<bool, String> {
        self.read()
            .await
            .count_merge_due()
            .await
            .map_err(|e| e.to_string())
    }

    async fn own_shard_watermark(
        &self,
    ) -> Result<Option<lance_context_core::ShardWatermark>, String> {
        self.read()
            .await
            .own_shard_watermark()
            .await
            .map_err(|e| e.to_string())
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
pub(crate) async fn flush_pass_coordinated<S: Sweepable>(
    stores: Vec<(String, S)>,
    pass_timeout: Duration,
    merge_slots: Option<Arc<Semaphore>>,
    state: Option<Arc<crate::state::AppState>>,
) {
    let kind = S::kind();
    for (name, store) in stores {
        match tokio::time::timeout(pass_timeout, store.flush()).await {
            Ok(Ok(())) => {
                metrics::counter!("rollout_wal_flush_total", "result" => "ok", "kind" => kind)
                    .increment(1);
                // The count-triggered merge rides this timer, but it is a merge,
                // not a flush: its outcome is reported under the cleanup
                // counters so a failing merge cannot masquerade as a failing
                // flush on the dashboards. It is also a merge for memory
                // purposes: it shares the per-worker slot with the master's
                // merge requests (ROLLOUT_MERGE_CONCURRENCY). The slot is only
                // *tried*, never awaited: this pass flushes every resident
                // store in sequence, and blocking on a busy slot here would
                // hold up the flush of every store behind this one. When the
                // slots are full the merge is skipped; the next pass (30 s)
                // tries again, and the master's sweep covers the store anyway.
                if let Some(state) = &state {
                    match tokio::time::timeout(
                        pass_timeout,
                        route_merge(state, &name, &store, true),
                    )
                    .await
                    {
                        Ok(Ok(false)) => {}
                        Ok(Ok(true)) => continue,
                        outcome => {
                            tracing::warn!(store = %name, ?outcome, "coordinated count merge request failed");
                            continue;
                        }
                    }
                }
                let slot = match &merge_slots {
                    Some(slots) => match slots.clone().try_acquire_owned() {
                        Ok(permit) => Some(permit),
                        Err(_) => {
                            metrics::counter!(
                                "rollout_wal_self_merge_skipped_total",
                                "kind" => kind
                            )
                            .increment(1);
                            continue;
                        }
                    },
                    None => None,
                };
                let merged = async {
                    let _slot = slot;
                    observe_legacy_merge(
                        state.as_deref(),
                        &name,
                        kind,
                        "count_sweeper",
                        store.merge_if_due(),
                    )
                    .await
                };
                report_merge(
                    &name,
                    kind,
                    tokio::time::timeout(pass_timeout, merged).await,
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
pub(crate) async fn merge_pass_coordinated<S: Sweepable>(
    stores: Vec<(String, S)>,
    pass_timeout: Duration,
    state: Option<Arc<crate::state::AppState>>,
) {
    let kind = S::kind();
    for (name, store) in stores {
        if let Some(state) = &state {
            match tokio::time::timeout(pass_timeout, route_merge(state, &name, &store, false)).await
            {
                Ok(Ok(false)) => {}
                Ok(Ok(true)) => continue,
                outcome => {
                    tracing::warn!(store = %name, ?outcome, "coordinated cleanup request failed");
                    continue;
                }
            }
        }
        let slot = match state.as_ref().and_then(|s| s.merge_slots.clone()) {
            Some(slots) => match slots.try_acquire_owned() {
                Ok(slot) => Some(slot),
                Err(_) => continue,
            },
            None => None,
        };
        let _slot = slot;
        report_merge(
            &name,
            kind,
            tokio::time::timeout(
                pass_timeout,
                observe_legacy_merge(
                    state.as_deref(),
                    &name,
                    kind,
                    "cleanup_sweeper",
                    store.merge_wal(),
                ),
            )
            .await,
        );
    }
}

/// The scope covers only merge work, never the independent ingestion flush.
async fn observe_legacy_merge<T, E>(
    state: Option<&crate::state::AppState>,
    name: &str,
    kind: &str,
    origin: &'static str,
    work: impl std::future::Future<Output = Result<T, E>>,
) -> Result<T, E> {
    if let Some(state) = state {
        let target = if kind == "rollout" {
            name.to_string()
        } else {
            format!("{kind}:{name}")
        };
        state
            .merge_executions
            .legacy
            .run(&target, origin, false, |_| work)
            .await
    } else {
        work.await
    }
}

/// Hard bound on one detached demand publish (manifest read + etcd put).
/// Longer than any healthy path, far shorter than a flush pass timeout.
const DEMAND_PUBLISH_TIMEOUT: Duration = Duration::from_secs(5);

/// Bounded concurrency for detached publishes across every store in the
/// process. When the slots are full the flush is not delayed; the publish is
/// dropped and counted. The next flush republishes an absolute watermark,
/// so a dropped publish loses nothing.
static DEMAND_PUBLISH_SLOTS: std::sync::LazyLock<Arc<Semaphore>> =
    std::sync::LazyLock::new(|| Arc::new(Semaphore::new(4)));

/// Publish this writer's shard watermark off the flush path. Never awaited
/// by the caller; bounded by `DEMAND_PUBLISH_TIMEOUT` and
/// `DEMAND_PUBLISH_SLOTS`. Every outcome is counted under
/// `rollout_wal_demand_events_total{result}`.
fn spawn_publish_demand<S: Sweepable>(
    state: Arc<crate::state::AppState>,
    target: String,
    store: S,
) {
    let Ok(permit) = DEMAND_PUBLISH_SLOTS.clone().try_acquire_owned() else {
        metrics::counter!("rollout_wal_demand_events_total", "result" => "dropped").increment(1);
        return;
    };
    tokio::spawn(async move {
        let _permit = permit;
        let result = tokio::time::timeout(DEMAND_PUBLISH_TIMEOUT, async {
            let watermark = match store.own_shard_watermark().await? {
                Some(watermark) => watermark,
                None => return Ok(None),
            };
            let event = lance_context_merge::demand::DemandEvent {
                v: lance_context_merge::demand::SCHEMA_VERSION,
                shard: watermark.shard_id.to_string(),
                sealed_through: watermark.sealed_through,
                sealed_bytes_through: 0,
                merged_through: None,
                flushed_at_ms: chrono::Utc::now().timestamp_millis(),
                writer_epoch: watermark.writer_epoch,
                source: lance_context_merge::demand::EventSource::Writer,
                merged_epoch: None,
                sealed_times: Default::default(),
            };
            state
                .merge_executions
                .publish_demand_event(&target, &event)
                .await
                .map(Some)
        })
        .await;
        let label = match result {
            Ok(Ok(Some(true))) => "written",
            Ok(Ok(Some(false))) => "unchanged",
            Ok(Ok(None)) => "no_manifest",
            Ok(Err(error)) => {
                tracing::debug!(target = %target, %error, "demand event publish failed");
                "failed"
            }
            Err(_) => {
                tracing::debug!(target = %target, "demand event publish timed out");
                "timeout"
            }
        };
        metrics::counter!("rollout_wal_demand_events_total", "result" => label).increment(1);
    });
}

/// Return true when routing handled the target (including drain/no-op).
async fn route_merge<S: Sweepable>(
    state: &Arc<crate::state::AppState>,
    name: &str,
    store: &S,
    count_trigger: bool,
) -> Result<bool, String> {
    let target = match S::kind() {
        "rollout" => name.to_string(),
        "generic" => format!("generic:{name}"),
        _ => return Ok(false),
    };
    if state.merge_executions.rollout.draining(&target) {
        return Ok(true);
    }
    // Demand is published on every flush, below any count threshold, so the
    // planner sees low-count tails without a sweep. It is detached and
    // bounded: the flush pass never waits on a manifest read or an etcd put
    // for telemetry, so a stuck publish cannot delay this table's merge
    // request or the next table's flush. Additive: `merge-requests` is
    // unchanged.
    spawn_publish_demand(state.clone(), target.clone(), S::clone(store));
    if !state.merge_executions.owned(&target) {
        match state.merge_executions.fencing_mode_known() {
            Some(false) => return Ok(false),
            None if !state.merge_executions.rollout.fence_legacy => {
                crate::merge_execution::Executions::resolve_mode_in_background(state.clone());
                return Ok(true);
            }
            _ => {}
        }
        if !count_trigger || store.count_merge_due().await? {
            crate::merge_execution::publish_background_request(state.clone(), target.clone());
            match crate::merge_execution::spawn_background(state.clone(), target).await {
                Ok(_reply) => {} // Executor is retained independently of this sweep.
                Err(crate::error::AppError::Overloaded(_)) => {
                    metrics::counter!("rollout_wal_self_merge_skipped_total", "kind" => S::kind())
                        .increment(1);
                }
                Err(error) => return Err(format!("{error:?}")),
            }
        }
        return Ok(true);
    }
    if !count_trigger || store.count_merge_due().await? {
        state.merge_executions.request_merge(&target).await?;
        metrics::counter!("rollout_wal_coordinated_merge_requests_total", "kind" => S::kind())
            .increment(1);
    }
    Ok(true)
}

#[cfg(test)]
pub(crate) async fn flush_pass<S: Sweepable>(
    stores: Vec<(String, S)>,
    timeout: Duration,
    slots: Option<Arc<Semaphore>>,
) {
    flush_pass_coordinated(stores, timeout, slots, None).await;
}

#[cfg(test)]
pub(crate) async fn merge_pass<S: Sweepable>(stores: Vec<(String, S)>, timeout: Duration) {
    merge_pass_coordinated(stores, timeout, None).await;
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

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn owned_count_trigger_enqueues_without_bypassing_slots_or_preparing_payloads() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let coordinator = lance_context_merge::Coordinator::new(
            client,
            format!("/owned-sweeper/{}", uuid::Uuid::new_v4()),
        );
        let state_dir = TempDir::new().unwrap();
        let mut state = crate::state::AppState::new_for_test(state_dir.path().to_path_buf()).await;
        state.merge_executions =
            crate::merge_execution::Executions::new(Some(coordinator.clone()), 3600);
        state
            .merge_executions
            .rollout
            .owned_targets
            .push("generic:s".into());
        let state = Arc::new(state);
        let dir = TempDir::new().unwrap();
        let store = generic_with_pending(&dir, 2, 2).await;
        let slots = Arc::new(Semaphore::new(1));
        let _held = slots.clone().acquire_owned().await.unwrap();
        flush_pass_coordinated(
            vec![("s".into(), store.clone())],
            Duration::from_secs(5),
            Some(slots),
            Some(state),
        )
        .await;
        assert_eq!(
            coordinator
                .request_page(None)
                .await
                .unwrap()
                .0
                .into_iter()
                .map(|r| r.target)
                .collect::<Vec<_>>(),
            ["generic:s"]
        );
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            2
        );
        assert_eq!(store.read().await.count_base_rows().await.unwrap(), 0);
    }

    /// A store whose watermark read hangs forever. The flush pass must still
    /// complete promptly and still issue the merge request: demand telemetry
    /// is detached and bounded, never on the flush or merge path.
    #[derive(Clone)]
    struct HangingWatermark(Arc<RwLock<GenericStore>>);

    impl Sweepable for HangingWatermark {
        fn kind() -> &'static str {
            "generic"
        }
        async fn flush(&self) -> Result<(), String> {
            self.0.flush().await
        }
        async fn count_merge_due(&self) -> Result<bool, String> {
            self.0.count_merge_due().await
        }
        async fn own_shard_watermark(
            &self,
        ) -> Result<Option<lance_context_core::ShardWatermark>, String> {
            std::future::pending().await
        }
        async fn merge_wal(&self) -> Result<usize, String> {
            self.0.merge_wal().await
        }
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn hung_demand_publish_never_delays_flush_or_merge_request() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let coordinator = lance_context_merge::Coordinator::new(
            client,
            format!("/hung-demand/{}", uuid::Uuid::new_v4()),
        );
        let state_dir = TempDir::new().unwrap();
        let mut state = crate::state::AppState::new_for_test(state_dir.path().to_path_buf()).await;
        state.merge_executions =
            crate::merge_execution::Executions::new(Some(coordinator.clone()), 3600);
        state
            .merge_executions
            .rollout
            .owned_targets
            .push("generic:s".into());
        let state = Arc::new(state);
        let dir = TempDir::new().unwrap();
        let store = HangingWatermark(generic_with_pending(&dir, 2, 2).await);
        let started = std::time::Instant::now();
        flush_pass_coordinated(
            vec![("s".into(), store.clone())],
            Duration::from_secs(30),
            None,
            Some(state.clone()),
        )
        .await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "flush pass waited on demand publish: {elapsed:?}"
        );
        // The merge request still went out despite the hung telemetry.
        assert_eq!(
            coordinator
                .request_page(None)
                .await
                .unwrap()
                .0
                .into_iter()
                .map(|r| r.target)
                .collect::<Vec<_>>(),
            ["generic:s"]
        );
        // Flushing a second time while the first publish is still hung must
        // also return immediately (slots or drop, never wait).
        let started = std::time::Instant::now();
        for _ in 0..8 {
            flush_pass_coordinated(
                vec![("s".into(), store.clone())],
                Duration::from_secs(30),
                None,
                Some(state.clone()),
            )
            .await;
        }
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    /// Publishes are detached; poll for the expected record with a bound.
    async fn await_event<F>(
        client: &etcd_client::Client,
        key: &str,
        mut accept: F,
    ) -> lance_context_merge::demand::DemandEvent
    where
        F: FnMut(&lance_context_merge::demand::DemandEvent) -> bool,
    {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(kv) = client.clone().get(key, None).await.unwrap().kvs().first() {
                let event: lance_context_merge::demand::DemandEvent =
                    serde_json::from_slice(kv.value()).unwrap();
                if accept(&event) {
                    return event;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "demand event never arrived"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Demand is published on every flush, below the count threshold and for
    /// targets this master does not own, and is monotonic across flushes.
    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn flush_publishes_shard_demand_below_threshold_and_without_ownership() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let prefix = format!("/demand-sweeper/{}", uuid::Uuid::new_v4());
        let coordinator = lance_context_merge::Coordinator::new(client.clone(), prefix.clone());
        let state_dir = TempDir::new().unwrap();
        let mut state = crate::state::AppState::new_for_test(state_dir.path().to_path_buf()).await;
        state.merge_executions =
            crate::merge_execution::Executions::new(Some(coordinator.clone()), 3600);
        // Deliberately NOT owned: demand must still be visible to a planner.
        let state = Arc::new(state);
        let dir = TempDir::new().unwrap();
        // Threshold 8, only 2 pending: no merge request is expected.
        let store = generic_with_pending(&dir, 8, 2).await;
        flush_pass_coordinated(
            vec![("s".into(), store.clone())],
            Duration::from_secs(5),
            None,
            Some(state.clone()),
        )
        .await;
        assert!(coordinator.request_page(None).await.unwrap().0.is_empty());
        let encoded: String = "generic:s".bytes().map(|b| format!("{b:02x}")).collect();
        let shard = store
            .read()
            .await
            .own_shard_watermark()
            .await
            .unwrap()
            .unwrap()
            .shard_id;
        let key = format!("{prefix}/demand-events/{encoded}/{shard}");
        let first = await_event(&client, &key, |_| true).await;
        assert_eq!(
            first.source,
            lance_context_merge::demand::EventSource::Writer
        );
        assert!(first.sealed_through >= 2, "{first:?}");
        let mut table = lance_context_merge::demand::TableDemand::default();
        table.fold(&first, 1, 0).unwrap();
        assert_eq!(table.pending_generations(), first.sealed_through);
        // A second flush with no new generation changes nothing: wait long
        // enough for a detached publish to have landed, then check the
        // revision is unchanged.
        let revision =
            client.clone().get(key.as_str(), None).await.unwrap().kvs()[0].mod_revision();
        flush_pass_coordinated(
            vec![("s".into(), store.clone())],
            Duration::from_secs(5),
            None,
            Some(state.clone()),
        )
        .await;
        tokio::time::sleep(Duration::from_millis(500)).await;
        let again = client.clone().get(key.as_str(), None).await.unwrap();
        assert_eq!(
            again.kvs()[0].mod_revision(),
            revision,
            "unchanged watermark not rewritten"
        );
        // New generation advances it.
        store
            .read()
            .await
            .add(&[json!({"id": "r9"}).as_object().unwrap().clone()])
            .await
            .unwrap();
        flush_pass_coordinated(
            vec![("s".into(), store.clone())],
            Duration::from_secs(5),
            None,
            Some(state),
        )
        .await;
        let second = await_event(&client, &key, |e| e.sealed_through > first.sealed_through).await;
        assert_eq!(second.writer_epoch, first.writer_epoch);
    }

    fn spec() -> SchemaSpec {
        SchemaSpec::new(vec![(
            ID_COLUMN.to_string(),
            ColumnSpec::required(ColumnType::String { large: false }),
        )])
    }

    #[tokio::test]
    async fn legacy_merge_observes_real_reclamation_without_tracking_ingest_flush() {
        let state_dir = TempDir::new().unwrap();
        let state =
            Arc::new(crate::state::AppState::new_for_test(state_dir.path().to_path_buf()).await);
        let dir = TempDir::new().unwrap();
        let store = generic_with_pending(&dir, 3, 3).await;
        flush_pass_coordinated(
            vec![("s".into(), store.clone())],
            Duration::from_secs(10),
            None,
            Some(state.clone()),
        )
        .await;
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            0
        );
        assert_eq!(store.read().await.count_base_rows().await.unwrap(), 3);
        let axum::Json(report) = crate::legacy_merge::status(
            axum::extract::State(state),
            axum::extract::Query(Default::default()),
        )
        .await;
        let executions = report["snapshot"]["executions"].as_array().unwrap();
        assert_eq!(
            executions.len(),
            1,
            "flush itself must not be a maintenance execution"
        );
        let execution = &executions[0];
        assert_eq!(execution["target"], "generic:s");
        assert_eq!(execution["origin"], "count_sweeper");
        assert_eq!(execution["phase"], "returned_ok");
        assert_eq!(execution["work"]["generations_reclaimed"], 3);
        assert!(execution["work"]["base_commits"].as_u64().unwrap() > 0);
        assert_eq!(execution["writes"]["active_commits"], 0);
        assert_eq!(report["snapshot"]["takeover_authorized"], false);
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

    /// The flush pass must flush every store even when every merge slot is
    /// taken: the self-merge only *tries* the slot and skips, it never waits.
    /// Two stores, count trigger off, slots exhausted — both must flush, and
    /// nothing must merge.
    #[tokio::test]
    async fn flush_pass_never_waits_on_a_busy_merge_slot() {
        let dir_a = TempDir::new().unwrap();
        let dir_b = TempDir::new().unwrap();
        // seal_on_add is on in `generic_with_pending`, so build stores with
        // unsealed rows by hand: count trigger off, rows left in the memtable.
        async fn unsealed(dir: &TempDir) -> Arc<RwLock<GenericStore>> {
            let uri = dir.path().to_string_lossy().to_string();
            let store = GenericStore::open(
                &uri,
                spec(),
                GenericStoreOptions {
                    merge_after_generations: Some(0),
                    seal_on_add: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            store
                .add(&[json!({"id": "r0"}).as_object().unwrap().clone()])
                .await
                .unwrap();
            assert_eq!(store.pending_wal_generations().await.unwrap(), 0);
            Arc::new(RwLock::new(store))
        }
        let a = unsealed(&dir_a).await;
        let b = unsealed(&dir_b).await;
        let slots = Arc::new(Semaphore::new(1));
        let _held = slots.clone().acquire_owned().await.unwrap();

        tokio::time::timeout(
            Duration::from_secs(10),
            flush_pass(
                vec![("a".to_string(), a.clone()), ("b".to_string(), b.clone())],
                Duration::from_secs(5),
                Some(slots),
            ),
        )
        .await
        .expect("flush pass must not block on the merge slot");

        // Both flushed: each now has exactly one sealed generation, unmerged.
        for store in [&a, &b] {
            assert_eq!(
                store.read().await.pending_wal_generations().await.unwrap(),
                1
            );
        }
    }

    /// With a free slot and the count trigger set, the pass still merges.
    #[tokio::test]
    async fn flush_pass_merges_when_a_slot_is_free() {
        let dir = TempDir::new().unwrap();
        let store = generic_with_pending(&dir, 2, 2).await;
        flush_pass(
            vec![("s".to_string(), store.clone())],
            Duration::from_secs(5),
            Some(Arc::new(Semaphore::new(1))),
        )
        .await;
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            0
        );
    }
}
