//! Detached, bounded publication of demand watermarks from master-side
//! sources (`docs/design/scheduler-p1-data-model.md` §1.3): the stats scan
//! (source `scan`) and executors after a successful WAL catch-up release
//! (source `executor`).
//!
//! Same discipline as the writer-side publish: never awaited by the caller,
//! one hard timeout, a process-wide concurrency cap, drop-and-count when the
//! cap is full. Every publish is a per-shard monotonic join, so a dropped or
//! late publish cannot misstate demand; the next scan reconciles it.

use std::sync::Arc;
use std::time::Duration;

use lance_context_core::{GenericStore, GenericStoreOptions, RolloutStore, RolloutStoreOptions};
use lance_context_merge::demand::{DemandEvent, EventSource, ShardMerged, SCHEMA_VERSION};
use lance_context_merge::Coordinator;
use tokio::sync::Semaphore;

use crate::scanner::ScanKind;
use crate::state::MasterState;

/// Manifest reads for every shard plus one etcd put per shard.
const SCAN_PUBLISH_TIMEOUT: Duration = Duration::from_secs(15);

static SCAN_PUBLISH_SLOTS: std::sync::LazyLock<Arc<Semaphore>> =
    std::sync::LazyLock::new(|| Arc::new(Semaphore::new(4)));

pub(crate) fn spawn_scan_demand(
    state: Arc<MasterState>,
    target: String,
    uri: String,
    kind: ScanKind,
    rollout_options: RolloutStoreOptions,
    generic_options: GenericStoreOptions,
) {
    let Ok(permit) = SCAN_PUBLISH_SLOTS.clone().try_acquire_owned() else {
        metrics::counter!("master_scan_demand_events_total", "result" => "dropped").increment(1);
        return;
    };
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
            let mut written = 0usize;
            for mark in marks {
                let event = DemandEvent {
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
                };
                if coordinator.publish_demand_event(&target, &event).await? {
                    written += 1;
                }
            }
            Ok::<usize, String>(written)
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
