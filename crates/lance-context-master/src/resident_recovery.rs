//! Resident catch-up demand discovery, independent of the full stats scanner.
//! A durable fleet cursor bounds metadata reads even when masters are scaled up.
use crate::{config::MasterConfig, state::MasterState, stats_store::StatRow};
use etcd_client::{Compare, CompareOp, Txn, TxnOp};
use futures::{stream, StreamExt};
use lance_context_api::TaskKind;
use lance_context_core::rollout_append::AppendCoordinator;
use serde::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
    time::Duration,
};

const BATCH: usize = 4;
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Default, Serialize, Deserialize)]
struct Cursor {
    after: Option<String>,
    next_batch_ms: i64,
}

fn scope(config: &MasterConfig) -> u64 {
    // Namespace only, not an ownership token. A Rust hash-version change merely
    // restarts discovery; the separate fleet gate still limits total probes.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    for names in [
        &config.merge_rollout.owned_targets,
        &config.merge_rollout.drain_targets,
        &config.append.rollout_append_targets,
    ] {
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.hash(&mut hash);
    }
    hash.finish()
}

fn enabled(config: &MasterConfig) -> bool {
    config.append.rollout_append_local
        && config.append.rollout_append_reconcile_interval_secs > 0
        && config.merge_wal_interval_secs > 0
}

fn candidates(config: &MasterConfig, snapshot: &[StatRow], after: Option<&str>) -> Vec<String> {
    // Explicitly owned tables do not depend on scanner freshness or existence.
    // Wildcard installations additionally use known names from the stats cache;
    // its counts are never used as evidence of current WAL demand.
    let mut names: Vec<_> = config
        .merge_rollout
        .owned_targets
        .iter()
        .chain(config.append.rollout_append_targets.iter())
        .map(String::as_str)
        .chain(snapshot.iter().map(|row| row.name.as_str()))
        .filter(|name| {
            *name != "*"
                && config.append.local(name)
                && config.merge_rollout.owned(name)
                && !config.merge_rollout.draining(name)
                && after.is_none_or(|after| *name > after)
        })
        .map(str::to_owned)
        .collect();
    names.sort_unstable();
    names.dedup();
    names.truncate(BATCH + 1);
    names
}

pub(crate) fn spawn(state: &Arc<MasterState>) {
    if !enabled(&state.config) {
        return;
    }
    let weak = Arc::downgrade(state);
    tokio::spawn(async move {
        loop {
            let Some(state) = weak.upgrade() else { return };
            // Also bound etcd waits. Dropping this read-only discovery never
            // cancels a writer or authorizes a replacement execution.
            match tokio::time::timeout(
                Duration::from_secs(25),
                tick(&state, chrono::Utc::now().timestamp_millis()),
            )
            .await
            {
                Ok(Ok(_)) => {}
                outcome => tracing::warn!(?outcome, "resident WAL demand discovery incomplete"),
            }
            drop(state);
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });
}

async fn tick(state: &Arc<MasterState>, now: i64) -> Result<usize, String> {
    if !enabled(&state.config) {
        return Ok(0);
    }
    let Some(_admission) = state.admission.try_admit() else {
        return Ok(0);
    };
    let key = format!(
        "{}/resident-wal-discovery",
        state.config.etcd.etcd_prefix.trim_end_matches('/')
    );
    let cursor_key = format!("{key}/cursor/{:016x}", scope(&state.config));
    let mut client = state.task_store.etcd_client().clone();
    let prior = client
        .get(key.as_str(), None)
        .await
        .map_err(|e| e.to_string())?;
    let old = prior.kvs().first();
    let next_batch_ms: i64 = old
        .map(|kv| serde_json::from_slice(kv.value()))
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    if next_batch_ms > now {
        return Ok(0);
    }
    let previous_cursor = client
        .get(cursor_key.as_str(), None)
        .await
        .map_err(|e| e.to_string())?;
    let old_cursor = previous_cursor.kvs().first();
    let cursor: Cursor = old_cursor
        .map(|kv| serde_json::from_slice(kv.value()))
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let snapshot = state.stats_cache.read().await.clone();
    if candidates(&state.config, &snapshot, None).is_empty() {
        return Ok(0);
    }
    let mut names = candidates(&state.config, &snapshot, cursor.after.as_deref());
    if names.is_empty() && cursor.after.is_none() {
        return Ok(0);
    }
    let more = names.len() > BATCH;
    names.truncate(BATCH);
    let next = Cursor {
        after: if more { names.last().cloned() } else { None },
        next_batch_ms: now.saturating_add(
            state
                .config
                .append
                .rollout_append_reconcile_interval_secs
                .clamp(30, 86_400) as i64
                * 1000,
        ),
    };
    // Reserve before object-store reads. Lost responses/crashes defer one page;
    // other replicas cannot duplicate its probes. No lease expiry enables writes.
    let compare = old.map_or_else(
        || Compare::version(key.as_str(), CompareOp::Equal, 0),
        |kv| Compare::mod_revision(key.as_str(), CompareOp::Equal, kv.mod_revision()),
    );
    let compare_cursor = old_cursor.map_or_else(
        || Compare::version(cursor_key.as_str(), CompareOp::Equal, 0),
        |kv| Compare::mod_revision(cursor_key.as_str(), CompareOp::Equal, kv.mod_revision()),
    );
    if !client
        .txn(Txn::new().when([compare, compare_cursor]).and_then([
            TxnOp::put(
                key.as_str(),
                serde_json::to_vec(&next.next_batch_ms).map_err(|e| e.to_string())?,
                None,
            ),
            TxnOp::put(
                cursor_key.as_str(),
                serde_json::to_vec(&next).map_err(|e| e.to_string())?,
                None,
            ),
        ]))
        .await
        .map_err(|e| e.to_string())?
        .succeeded()
    {
        return Ok(0);
    }
    let mut probes = stream::iter(names.into_iter().map(|target| async move {
        let outcome = tokio::time::timeout(PROBE_TIMEOUT, probe(state, &target)).await;
        match outcome {
            Ok(Ok(requested)) => requested,
            outcome => {
                metrics::counter!("master_resident_wal_probes_total", "result" => "failed").increment(1);
                tracing::warn!(%target, ?outcome, "resident WAL metadata probe failed; continuing other targets");
                false
            }
        }
    })).buffer_unordered(2);
    let mut requested = 0;
    while let Some(true_or_false) = probes.next().await {
        requested += usize::from(true_or_false);
    }
    Ok(requested)
}

async fn probe(state: &Arc<MasterState>, target: &str) -> Result<bool, String> {
    if state
        .task_store
        .get_active_id(TaskKind::MergeWal, target)
        .await
        .map_err(|e| e.to_string())?
        .is_some()
    {
        return Ok(false);
    }
    // Merge retry policy lives in the durable per-executor failure ledger,
    // not the generic task cooldown (which deliberately excludes MergeWal).
    if state
        .task_store
        .merge_coordinator()
        .failure(
            target,
            lance_context_merge::MaintenanceKind::Catchup.endpoint(),
        )
        .await?
        .is_some_and(|failure| failure.next_retry_ms > lance_context_merge::failure::now_ms())
    {
        return Ok(false);
    }
    let owner = crate::catchup::store::active_key(&state.config.etcd.etcd_prefix, target);
    if !state
        .task_store
        .etcd_client()
        .clone()
        .get(owner, None)
        .await
        .map_err(|e| e.to_string())?
        .kvs()
        .is_empty()
    {
        return Ok(false);
    }
    let coordinator = AppendCoordinator::open(
        &state.rollout_uri(target),
        // Reuse this master's small local-staging session, never default caches.
        state
            .local_append
            .as_ref()
            .map(|local| local.session.clone()),
    )
    .await
    .map_err(|e| e.to_string())?;
    let pending = coordinator
        .has_sealed_wal(&state.config.catchup.shards)
        .await
        .map_err(|e| e.to_string())?;
    if pending {
        state
            .task_store
            .merge_coordinator()
            .request_merge(target)
            .await?;
    }
    metrics::counter!("master_resident_wal_probes_total", "result" => if pending { "requested" } else { "empty" }).increment(1);
    Ok(pending)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use lance_context_core::{RolloutStore, RolloutStoreOptions};

    async fn fixture() -> (tempfile::TempDir, Arc<MasterState>) {
        let dir = tempfile::tempdir().unwrap();
        let mut config = MasterConfig::parse_from(["test"]);
        config.data_dir = dir.path().to_string_lossy().into();
        config.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        config.etcd.etcd_prefix = format!(
            "/resident-recovery-test/{}",
            lance_context_core::generate_id()
        );
        config.append.rollout_append_local = true;
        config.append.rollout_append_targets = vec!["*".into()];
        config.append.rollout_append_max_generations = 1;
        config.merge_rollout.owned_targets = ["a", "b", "c", "d", "e"].map(str::to_owned).to_vec();
        config.catchup.shards = vec!["writer".into()];
        config.worker_endpoints.clear();
        config.stats_scan_interval_secs = 0;
        let state = MasterState::new(config).await.unwrap();
        (dir, state)
    }

    async fn write(state: &MasterState, target: &str, rows: usize) -> RolloutStore {
        let writer = RolloutStore::open_with_options(
            &state.rollout_uri(target),
            RolloutStoreOptions {
                shard_id: Some("writer".into()),
                merge_after_generations: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        for id in 0..rows {
            let dto = serde_json::from_value(
                serde_json::json!({"id":id.to_string(), "rollout_id":"r", "content":"test"}),
            )
            .unwrap();
            writer
                .add(&[lance_context_core::rollout_record_from_add_request(&dto)])
                .await
                .unwrap();
            writer.flush().await.unwrap();
        }
        writer
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn replicas_discover_without_stats_and_restart_rotates_past_missing_tables() {
        let (_dir, state) = fixture().await;
        let _b = write(&state, "b", 1).await;
        let _e = write(&state, "e", 1).await;
        let second = MasterState::new(state.config.clone()).await.unwrap();
        assert!(state.stats_cache.read().await.is_empty());
        let now = chrono::Utc::now().timestamp_millis();
        let (one, two) = tokio::join!(tick(&state, now), tick(&second, now));
        assert_eq!(one.unwrap() + two.unwrap(), 1);
        let coordinator = state.task_store.merge_coordinator();
        assert_eq!(
            coordinator
                .request_page(None)
                .await
                .unwrap()
                .0
                .iter()
                .map(|r| r.target.as_str())
                .collect::<Vec<_>>(),
            ["b"]
        );
        assert_eq!(tick(&second, now + 1000).await.unwrap(), 0);
        let restarted = MasterState::new(state.config.clone()).await.unwrap();
        assert_eq!(tick(&restarted, now + 30_000).await.unwrap(), 1);
        assert_eq!(coordinator.request_page(None).await.unwrap().0.len(), 2);
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn discovery_preserves_external_owner_and_existing_canonical_task() {
        let (_dir, state) = fixture().await;
        let _a = write(&state, "a", 1).await;
        let _b = write(&state, "b", 1).await;
        let owner = crate::catchup::store::active_key(&state.config.etcd.etcd_prefix, "a");
        let mut client = state.task_store.etcd_client().clone();
        client
            .put(owner.as_str(), "external-publisher", None)
            .await
            .unwrap();
        let task = state
            .task_store
            .enqueue(TaskKind::MergeWal, "b", vec![])
            .await
            .unwrap();
        assert!(!probe(&state, "a").await.unwrap());
        assert!(!probe(&state, "b").await.unwrap());
        assert!(state
            .task_store
            .merge_coordinator()
            .request_page(None)
            .await
            .unwrap()
            .0
            .is_empty());
        assert_eq!(
            client.get(owner, None).await.unwrap().kvs()[0].value(),
            b"external-publisher"
        );
        assert_eq!(
            state
                .task_store
                .get_active_id(TaskKind::MergeWal, "b")
                .await
                .unwrap(),
            Some(task.id)
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn discovery_does_not_reset_or_enqueue_through_failure_backoff() {
        use crate::task_store::TaskKinds;
        let (_dir, state) = fixture().await;
        let _a = write(&state, "a", 1).await;
        state
            .task_store
            .enqueue(TaskKind::MergeWal, "a", vec![])
            .await
            .unwrap();
        let claim = state
            .task_store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let coordinator = state.task_store.merge_coordinator();
        coordinator
            .record_failure(
                &state.task_store.merge_claim(&claim),
                "a",
                "master:catchup",
                "invalid rollout schema",
            )
            .await
            .unwrap();
        state
            .task_store
            .finish(claim, Err("invalid rollout schema".into()))
            .await
            .unwrap();
        let before = coordinator
            .failure("a", "master:catchup")
            .await
            .unwrap()
            .unwrap();
        assert!(before.needs_attention);
        assert!(!probe(&state, "a").await.unwrap());
        let after = coordinator
            .failure("a", "master:catchup")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.consecutive_attempts, before.consecutive_attempts);
        assert_eq!(after.next_retry_ms, before.next_retry_ms);
        assert!(coordinator.request_page(None).await.unwrap().0.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn bounded_slice_leaves_durable_demand_and_successor_finishes_without_scanner() {
        use crate::task_store::TaskKinds;
        let (_dir, state) = fixture().await;
        let writer = write(&state, "a", 17).await;
        state
            .task_store
            .enqueue(TaskKind::MergeWal, "a", vec![])
            .await
            .unwrap();
        let claim = state
            .task_store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let outcome = crate::merge_execution::run_merge_wal(&state, &claim).await;
        assert!(outcome.as_ref().unwrap().contains("16 generations"));
        let coordinator = state.task_store.merge_coordinator();
        let request = coordinator.request_page(None).await.unwrap().0.remove(0);
        // A consumer visiting the request while this task runs cannot erase it.
        crate::scheduler::enqueue_merge_request(&state, &coordinator, &request)
            .await
            .unwrap();
        assert_eq!(coordinator.request_page(None).await.unwrap().0.len(), 1);
        state.task_store.finish(claim, outcome).await.unwrap();
        let successor = MasterState::new(state.config.clone()).await.unwrap();
        crate::scheduler::enqueue_merge_request(&successor, &coordinator, &request)
            .await
            .unwrap();
        assert!(coordinator.request_page(None).await.unwrap().0.is_empty());
        let next = successor
            .task_store
            .claim_next_of_kinds(TaskKinds::MERGE_WAL)
            .await
            .unwrap()
            .unwrap();
        let result = crate::merge_execution::run_merge_wal(&successor, &next).await;
        assert!(result.as_ref().unwrap().contains("1 generations"));
        successor.task_store.finish(next, result).await.unwrap();
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
        let reader = lance::Dataset::open(&state.rollout_uri("a")).await.unwrap();
        assert_eq!(reader.count_rows(None).await.unwrap(), 17);
        assert!(!probe(&successor, "a").await.unwrap());
    }

    #[test]
    fn explicit_owned_targets_rotate_without_any_stats_and_exclude_legacy() {
        let mut config = MasterConfig::parse_from(["test"]);
        config.append.rollout_append_local = true;
        config.append.rollout_append_targets = vec!["*".into()];
        config.merge_rollout.owned_targets = ["f", "a", "b", "c", "d", "e", "generic:x"]
            .map(str::to_owned)
            .to_vec();
        config.merge_rollout.drain_targets = vec!["b".into()];
        assert_eq!(candidates(&config, &[], None), ["a", "c", "d", "e", "f"]);
        assert_eq!(candidates(&config, &[], Some("e")), ["f"]);
        config.merge_wal_interval_secs = 0;
        assert!(!enabled(&config));
    }
}
