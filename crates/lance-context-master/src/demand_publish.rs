//! Detached, bounded publication of demand watermarks from master-side
//! sources (`docs/design/scheduler-p1-data-model.md` §1.3): the stats scan
//! (source `scan`) and executors after a successful WAL catch-up release
//! (source `executor`).
//!
//! One hard timeout and a process-wide concurrency cap. Background zero-WAL
//! observations drop-and-count when full. Positive observations and transitions
//! to zero wait fairly for a slot inside the scanner's existing bounded stream;
//! they cannot be silently lost just because four background reads are active.
//! Every publish is a per-shard monotonic join. Transport failures remain visible
//! and are reconciled by later writer/scan events.

use std::sync::Arc;
use std::time::Duration;

use lance_context_core::{GenericStore, GenericStoreOptions, RolloutStore, RolloutStoreOptions};
use lance_context_merge::demand::{DemandEvent, EventSource, ShardMerged, SCHEMA_VERSION};
use lance_context_merge::Coordinator;
use tokio::sync::Semaphore;

use crate::scanner::ScanKind;
use crate::state::MasterState;

/// Manifest reads, bounded batches of etcd reads, and per-changed-shard CAS.
const SCAN_PUBLISH_TIMEOUT: Duration = Duration::from_secs(15);

static SCAN_PUBLISH_SLOTS: std::sync::LazyLock<Arc<Semaphore>> =
    std::sync::LazyLock::new(|| Arc::new(Semaphore::new(4)));

/// Waiters live in the scanner's bounded observation stream, not in detached
/// per-table tasks or an unbounded side queue. Tokio's semaphore preserves FIFO
/// order, including against subsequent background try-acquire attempts.
async fn scan_slot(
    slots: Arc<Semaphore>,
    positive_or_clearing: bool,
) -> Option<tokio::sync::OwnedSemaphorePermit> {
    if positive_or_clearing {
        slots.acquire_owned().await.ok()
    } else {
        slots.try_acquire_owned().ok()
    }
}

pub(crate) async fn spawn_scan_demand(
    state: Arc<MasterState>,
    target: String,
    uri: String,
    kind: ScanKind,
    rollout_options: RolloutStoreOptions,
    generic_options: GenericStoreOptions,
    positive_or_clearing: bool,
) {
    let permit = tokio::select! {
        permit = scan_slot(SCAN_PUBLISH_SLOTS.clone(), positive_or_clearing) => permit,
        _ = state.admission.wait_closed() => {
            metrics::counter!("master_scan_demand_events_total", "result" => "draining").increment(1);
            return;
        }
    };
    let Some(permit) = permit else {
        metrics::counter!("master_scan_demand_events_total", "result" => "dropped").increment(1);
        return;
    };
    // Drain may begin while this admitted observation is waiting. Release the
    // slot instead of starting more detached work on the retiring executor.
    if !state.admission.is_accepting() {
        metrics::counter!("master_scan_demand_events_total", "result" => "draining").increment(1);
        return;
    }
    tokio::spawn(async move {
        let _permit = permit;
        let result = tokio::time::timeout(SCAN_PUBLISH_TIMEOUT, async {
            let marks = match kind {
                ScanKind::Rollout => {
                    RolloutStore::open_existing_with_options(&uri, rollout_options)
                        .await
                        .map_err(|e| e.to_string())?
                        .all_shard_watermarks()
                        .await
                        .map_err(|e| e.to_string())?
                }
                ScanKind::Generic => GenericStore::open_existing(&uri, generic_options)
                    .await
                    .map_err(|e| e.to_string())?
                    .all_shard_watermarks()
                    .await
                    .map_err(|e| e.to_string())?,
            };
            let coordinator = state.task_store.merge_coordinator();
            let now_ms = chrono::Utc::now().timestamp_millis();
            let events: Vec<_> = marks
                .into_iter()
                .map(|mark| DemandEvent {
                    v: SCHEMA_VERSION,
                    shard: mark.shard_id.to_string(),
                    sealed_through: mark.sealed_through,
                    sealed_bytes_through: 0,
                    merged_through: mark.merged_through,
                    flushed_at_ms: now_ms,
                    writer_epoch: mark.writer_epoch,
                    source: EventSource::Scan,
                    merged_epoch: None,
                    sealed_times: Default::default(),
                })
                .collect();
            coordinator.publish_demand_events(&target, &events).await
        })
        .await;
        let label = match result {
            Ok(Ok(n)) if n > 0 => "written",
            Ok(Ok(_)) => "unchanged",
            Ok(Err(error)) => {
                tracing::debug!(target = %target, %error, "scan demand publish failed");
                "failed"
            }
            Err(_) => {
                tracing::debug!(target = %target, "scan demand publish timed out");
                "timeout"
            }
        };
        metrics::counter!("master_scan_demand_events_total", "result" => label).increment(1);
    });
}

/// Base index and bounded shard-manifest reads, plus one etcd join per shard.
const MERGED_PUBLISH_TIMEOUT: Duration = Duration::from_secs(15);

static MERGED_PUBLISH_SLOTS: std::sync::LazyLock<Arc<Semaphore>> =
    std::sync::LazyLock::new(|| Arc::new(Semaphore::new(4)));

#[cfg(test)]
pub(crate) async fn hold_merged_publications_for_test() -> tokio::sync::OwnedSemaphorePermit {
    MERGED_PUBLISH_SLOTS.clone().acquire_many_owned(4).await.unwrap()
}

/// Prepared metadata remains inert until the exact commit future acknowledges
/// success. This is advisory demand progress, never a storage/ownership fence.
pub(crate) struct CommittedMergeDemand {
    coordinator: Coordinator,
    target: String,
    shards: Vec<ShardMerged>,
}

impl CommittedMergeDemand {
    pub(crate) fn prepare(
        state: &MasterState,
        target: &str,
        prepared: &lance_context_core::PreparedMerge,
    ) -> Option<Self> {
        if !Self::selected(state, target) {
            return None;
        }
        let (shard, merged_through) = prepared.prepared_watermark()?;
        Some(Self {
            coordinator: state.task_store.merge_coordinator(),
            target: target.to_owned(),
            shards: vec![ShardMerged {
                shard: shard.to_string(),
                merged_through,
            }],
        })
    }

    pub(crate) fn prepare_group(
        state: &Arc<MasterState>,
        target: &str,
        prepared: &[lance_context_core::PreparedShardMerge],
    ) -> Option<Self> {
        if !Self::selected(state, target) || prepared.len() > 32 {
            return None;
        }
        let shards: Vec<_> = prepared
            .iter()
            .filter_map(|part| part.prepared_watermark())
            .map(|(shard, merged_through)| ShardMerged {
                shard: shard.to_string(),
                merged_through,
            })
            .collect();
        (!shards.is_empty()).then(|| Self {
            coordinator: state.task_store.merge_coordinator(),
            target: target.to_owned(),
            shards,
        })
    }

    fn selected(state: &MasterState, target: &str) -> bool {
        crate::local_merge::selected(&state.config, target)
            && state
                .config
                .local_merge
                .master_local_merge_committed_demand_targets
                .iter()
                .any(|allowed| allowed == target)
    }

    /// Poll the original commit in its owning task/scope. Dropping this future,
    /// an error, or a zero-result commit never publishes its prepared progress.
    /// Publication cannot delay commit completion or ownership release.
    pub(crate) async fn observe<F>(observer: Option<Self>, commit: F) -> Result<usize, String>
    where
        F: std::future::Future<Output = Result<usize, String>>,
    {
        let result = commit.await?;
        if result > 0 {
            if let Some(observer) = observer {
                observer.publish();
            }
        }
        Ok(result)
    }

    fn publish(self) {
        // Share the existing process-wide bound; never queue unbounded work.
        let Ok(permit) = MERGED_PUBLISH_SLOTS.clone().try_acquire_owned() else {
            metrics::counter!("master_demand_committed_watermarks_total", "result" => "dropped")
            .increment(1);
            return;
        };
        tokio::spawn(async move {
            let _permit = permit;
            let result = tokio::time::timeout(MERGED_PUBLISH_TIMEOUT, async {
                // A group occupies one bounded publication slot. Shard
                // joins stay sequential and retain their monotonic CAS;
                // no detached task per shard or unbounded queue is added.
                let mut written = false;
                for shard in &self.shards {
                    written |= self
                        .coordinator
                        .publish_merged_watermark(&self.target, shard)
                        .await?;
                }
                Ok::<_, String>(written)
            })
            .await;
            let label = match result {
                Ok(Ok(true)) => "written",
                Ok(Ok(false)) => "unchanged",
                Ok(Err(error)) => {
                    tracing::warn!(target = %self.target, %error, "committed watermark publish failed");
                    "failed"
                }
                Err(_) => {
                    tracing::warn!(target = %self.target, "committed watermark publish timed out");
                    "timeout"
                }
            };
            metrics::counter!("master_demand_committed_watermarks_total", "result" => label)
                .increment(1);
        });
    }
}

/// After a successful WAL catch-up has released ownership, publish every
/// shard's merged watermark. Detached and bounded; never on the release
/// path. Each shard is a separate monotonic put, so a table with hundreds of
/// shards never approaches the etcd transaction op limit and a concurrent
/// writer flush on one shard cannot fail the others.
pub(crate) fn spawn_merged_demand(coordinator: Coordinator, target: String, uri: String) {
    let Ok(permit) = MERGED_PUBLISH_SLOTS.clone().try_acquire_owned() else {
        metrics::counter!("master_demand_merged_watermarks_total", "result" => "dropped")
            .increment(1);
        return;
    };
    tokio::spawn(async move {
        let _permit = permit;
        let result = tokio::time::timeout(MERGED_PUBLISH_TIMEOUT, async {
            let marks = lance_context_core::read_demand_watermarks(&uri)
                .await
                .map_err(|e| e.to_string())?;
            let merged: Vec<ShardMerged> = marks
                .into_iter()
                .filter_map(|mark| {
                    mark.merged_through.map(|generation| ShardMerged {
                        shard: mark.shard_id.to_string(),
                        merged_through: generation,
                    })
                })
                .collect();
            let mut written = 0usize;
            for shard in &merged {
                if coordinator.publish_merged_watermark(&target, shard).await? {
                    written += 1;
                }
            }
            Ok::<(usize, usize), String>((written, merged.len()))
        })
        .await;
        let (label, count) = match result {
            Ok(Ok((written, total))) => (if written > 0 { "written" } else { "unchanged" }, total),
            Ok(Err(error)) => {
                tracing::warn!(target = %target, %error, "merged watermark publish failed");
                ("failed", 0)
            }
            Err(_) => {
                tracing::warn!(target = %target, "merged watermark publish timed out");
                ("timeout", 0)
            }
        };
        metrics::counter!("master_demand_merged_watermarks_total", "result" => label)
            .increment(count.max(1) as u64);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::Poll;

    #[tokio::test]
    #[ignore = "requires isolated loopback ETCD_TEST_ENDPOINTS and ETCD_TEST_EXCLUSIVE=1"]
    async fn committed_demand_requires_success_and_preserves_newer_writer_progress() {
        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").unwrap(), "1");
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        assert!(endpoint.starts_with("http://127.0.0.1:") && !endpoint.contains(','));
        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let prefix = format!(
            "/committed-demand-test/{}",
            lance_context_core::generate_id()
        );
        let coordinator = Coordinator::new(client.clone(), &prefix);
        let key = format!("{prefix}/demand-events/74/shard");
        let observer = || {
            Some(CommittedMergeDemand {
                coordinator: coordinator.clone(),
                target: "t".into(),
                shards: vec![ShardMerged {
                    shard: "shard".into(),
                    merged_through: 1,
                }],
            })
        };
        let failed =
            CommittedMergeDemand::observe(observer(), async { Err("uncertain commit".into()) })
                .await;
        assert!(failed.is_err());
        assert_eq!(
            CommittedMergeDemand::observe(observer(), async { Ok(0) })
                .await
                .unwrap(),
            0
        );
        let mut canceled = Box::pin(CommittedMergeDemand::observe(
            observer(),
            std::future::pending(),
        ));
        assert!(matches!(futures::poll!(canceled.as_mut()), Poll::Pending));
        drop(canceled);
        assert!(client
            .get(key.as_str(), None)
            .await
            .unwrap()
            .kvs()
            .is_empty());

        let writer = DemandEvent {
            v: SCHEMA_VERSION,
            shard: "shard".into(),
            sealed_through: 2,
            sealed_bytes_through: 100,
            merged_through: None,
            flushed_at_ms: 1234,
            writer_epoch: 7,
            source: EventSource::Writer,
            merged_epoch: None,
            sealed_times: Default::default(),
        };
        coordinator
            .publish_demand_event("t", &writer)
            .await
            .unwrap();
        // Saturation drops only optional publication; a successful commit is
        // immediately returned. Existing fallback reconciliation is retained.
        let permits = MERGED_PUBLISH_SLOTS
            .clone()
            .acquire_many_owned(4)
            .await
            .unwrap();
        assert_eq!(
            CommittedMergeDemand::observe(observer(), async { Ok(1) })
                .await
                .unwrap(),
            1
        );
        let before: DemandEvent =
            serde_json::from_slice(client.get(key.as_str(), None).await.unwrap().kvs()[0].value())
                .unwrap();
        assert_eq!(before.merged_through, None);
        drop(permits);
        assert_eq!(
            CommittedMergeDemand::observe(observer(), async { Ok(1) })
                .await
                .unwrap(),
            1
        );
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let row: DemandEvent = serde_json::from_slice(
                    client.get(key.as_str(), None).await.unwrap().kvs()[0].value(),
                )
                .unwrap();
                if row.merged_through == Some(1) {
                    assert_eq!(row.sealed_through, 2);
                    assert_eq!(row.writer_epoch, 7);
                    assert_eq!(row.flushed_at_ms, 1234);
                    assert_eq!(row.sealed_bytes_through, 100);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn positive_observation_waits_instead_of_dropping_when_all_four_slots_are_full() {
        let slots = Arc::new(Semaphore::new(4));
        let mut active = Vec::new();
        for _ in 0..4 {
            active.push(scan_slot(slots.clone(), false).await.unwrap());
        }
        assert!(scan_slot(slots.clone(), false).await.is_none());
        let mut positive = Box::pin(scan_slot(slots.clone(), true));
        assert!(matches!(futures::poll!(positive.as_mut()), Poll::Pending));
        drop(active.pop());
        // The released credit belongs to the already queued positive table.
        assert!(scan_slot(slots.clone(), false).await.is_none());
        let permit = positive.await.unwrap();
        assert_eq!(slots.available_permits(), 0);
        drop(permit);
        assert!(scan_slot(slots.clone(), false).await.is_some());
    }

    #[tokio::test]
    async fn coexisting_positive_tables_get_fifo_turns_without_extra_capacity() {
        let slots = Arc::new(Semaphore::new(1));
        let active = scan_slot(slots.clone(), false).await.unwrap();
        let mut first = Box::pin(scan_slot(slots.clone(), true));
        let mut second = Box::pin(scan_slot(slots.clone(), true));
        assert!(matches!(futures::poll!(first.as_mut()), Poll::Pending));
        assert!(matches!(futures::poll!(second.as_mut()), Poll::Pending));
        drop(active);
        let first = first.await.unwrap();
        assert!(matches!(futures::poll!(second.as_mut()), Poll::Pending));
        assert!(scan_slot(slots.clone(), false).await.is_none());
        drop(first);
        assert!(second.await.is_some());
    }

    #[tokio::test]
    async fn cancelled_observation_returns_its_place_without_stranding_other_tables() {
        let slots = Arc::new(Semaphore::new(1));
        let active = scan_slot(slots.clone(), false).await.unwrap();
        let mut cancelled = Box::pin(scan_slot(slots.clone(), true));
        let mut next = Box::pin(scan_slot(slots.clone(), true));
        assert!(matches!(futures::poll!(cancelled.as_mut()), Poll::Pending));
        assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));
        drop(cancelled);
        drop(active);
        assert!(next.await.is_some());
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS; run publication tests serially"]
    async fn real_scan_publishes_small_tail_and_clear_despite_busy_slots_and_yields_to_drain() {
        use crate::config::MasterConfig;
        use clap::Parser;
        use etcd_client::GetOptions;
        use lance_context_api::{ColumnSpec, ColumnType, SchemaSpec, ID_COLUMN};

        async fn blocked(
            state: &MasterState,
            scan: &tokio::task::JoinHandle<lance::Result<usize>>,
        ) {
            tokio::time::timeout(Duration::from_secs(20), async {
                loop {
                    assert!(
                        !scan.is_finished(),
                        "positive scan observation was dropped instead of waiting"
                    );
                    if state.admission.status().operations.values().any(|op| {
                        op.progress
                            .as_ref()
                            .is_some_and(|p| p.unit == "tables_observed" && p.completed > 0)
                    }) {
                        tokio::task::yield_now().await;
                        assert!(!scan.is_finished());
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
        }

        async fn pending(state: &MasterState) -> u64 {
            let prefix = format!(
                "{}/demand-events/",
                state.config.etcd.prefix().trim_end_matches('/')
            );
            let result = state
                .task_store
                .etcd_client()
                .clone()
                .get(prefix, Some(GetOptions::new().with_prefix()))
                .await
                .unwrap();
            result
                .kvs()
                .iter()
                .map(|kv| {
                    let event: DemandEvent = serde_json::from_slice(kv.value()).unwrap();
                    event
                        .sealed_through
                        .saturating_sub(event.merged_through.unwrap_or(0))
                })
                .sum()
        }

        async fn await_pending(state: &MasterState, count: u64) {
            tokio::time::timeout(Duration::from_secs(20), async {
                while pending(state).await != count {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let mut config = MasterConfig::parse_from(["test"]);
        config.data_dir = dir.path().to_string_lossy().into_owned();
        config.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        config.etcd.etcd_prefix =
            format!("/positive-scan-test/{}", lance_context_core::generate_id());
        config.merge_wal_interval_secs = 0;
        config.stats_scan_interval_secs = 0;
        let state = MasterState::new(config).await.unwrap();
        let uri = dir
            .path()
            .join("positive.generic.lance")
            .to_string_lossy()
            .into_owned();
        let spec = SchemaSpec::new(vec![(
            ID_COLUMN.to_string(),
            ColumnSpec::required(ColumnType::String { large: false }),
        )]);
        let mut writer = GenericStore::open(
            &uri,
            spec,
            GenericStoreOptions {
                seal_on_add: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        for i in 0..3 {
            writer
                .add(&[serde_json::json!({"id": format!("r{i}")})
                    .as_object()
                    .unwrap()
                    .clone()])
                .await
                .unwrap();
        }
        state
            .generic_registry
            .upsert("positive", &uri)
            .await
            .unwrap();
        assert_eq!(pending(&state).await, 0);

        let busy = SCAN_PUBLISH_SLOTS
            .clone()
            .acquire_many_owned(4)
            .await
            .unwrap();
        let scanning = state.clone();
        let scan = tokio::spawn(async move { crate::scanner::scan_once(&scanning).await });
        blocked(&state, &scan).await;
        assert_eq!(pending(&state).await, 0);
        drop(busy);
        tokio::time::timeout(Duration::from_secs(20), scan)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        await_pending(&state, 3).await;
        assert_eq!(state.stats_cache.read().await[0].pending_wal_generations, 3);

        // A zero result still has to clear the previous positive observation.
        writer.cleanup_wal().await.unwrap();
        let busy = SCAN_PUBLISH_SLOTS
            .clone()
            .acquire_many_owned(4)
            .await
            .unwrap();
        let scanning = state.clone();
        let scan = tokio::spawn(async move { crate::scanner::scan_once(&scanning).await });
        blocked(&state, &scan).await;
        assert_eq!(pending(&state).await, 3);
        drop(busy);
        tokio::time::timeout(Duration::from_secs(20), scan)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        await_pending(&state, 0).await;
        assert_eq!(state.stats_cache.read().await[0].pending_wal_generations, 0);

        writer
            .add(&[serde_json::json!({"id": "later"})
                .as_object()
                .unwrap()
                .clone()])
            .await
            .unwrap();
        let busy = SCAN_PUBLISH_SLOTS
            .clone()
            .acquire_many_owned(4)
            .await
            .unwrap();
        let scanning = state.clone();
        let scan = tokio::spawn(async move { crate::scanner::scan_once(&scanning).await });
        blocked(&state, &scan).await;
        state.admission.begin_drain();
        // Drain must not wait for a publishing slot, publish a partial stats
        // snapshot, or start new detached metadata work on the retired actor.
        assert!(tokio::time::timeout(Duration::from_secs(5), scan)
            .await
            .unwrap()
            .unwrap()
            .is_err());
        assert!(state.admission.status().drained());
        assert_eq!(pending(&state).await, 0);
        assert_eq!(state.stats_cache.read().await[0].pending_wal_generations, 0);
        drop(busy);
    }
}
