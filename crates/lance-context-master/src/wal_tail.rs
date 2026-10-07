//! Bounded metadata-only service for WAL tails below the normal count trigger.
//! A service interval is not a claim about the creation age of a generation.
use crate::{config::MasterConfig, scheduler, state::MasterState, stats_store::StatRow};
use etcd_client::{Compare, CompareOp, Txn, TxnOp};
use lance_context_api::TaskKind;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};

#[derive(Clone, Debug, clap::Args)]
pub struct WalTailConfig {
    /// Revisit low-count WAL tails after this interval; 0 disables this sweep.
    /// Requires automatic WAL merge and either workers or resident local append.
    #[arg(long, env = "MERGE_WAL_TAIL_INTERVAL_SECS", default_value_t = 0)]
    pub wal_tail_interval_secs: u64,
    /// At most this many candidates per fleet-wide 30-second batch (hard cap 64).
    #[arg(long, env = "MERGE_WAL_TAIL_BATCH_SIZE", default_value_t = 16)]
    pub wal_tail_batch_size: usize,
    #[arg(long, env = "MERGE_WAL_TAIL_STATS_MAX_AGE_SECS", default_value_t = 900)]
    pub wal_tail_stats_max_age_secs: u64,
}

impl Default for WalTailConfig {
    fn default() -> Self {
        Self {
            wal_tail_interval_secs: 0,
            wal_tail_batch_size: 16,
            wal_tail_stats_max_age_secs: 900,
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Cursor {
    after: Option<String>,
    next_batch_ms: i64,
}

fn enabled(config: &MasterConfig) -> bool {
    config.wal_tail.wal_tail_interval_secs > 0
        && config.merge_wal_interval_secs > 0
        && (!config.worker_endpoints.is_empty() || config.append.rollout_append_local)
        && config.wal_tail.wal_tail_stats_max_age_secs > 0
}

fn candidates(
    config: &MasterConfig,
    snapshot: &[StatRow],
    after: Option<&str>,
    now: i64,
) -> Vec<String> {
    let age = config.wal_tail.wal_tail_stats_max_age_secs.min(86_400) as i64 * 1000;
    let mut names: Vec<_> = snapshot
        .iter()
        .filter(|row| {
            row.pending_wal_generations > 0
                && row.pending_wal_generations < config.merge_wal_min_generations
                && row.scanned_at <= now
                && now.saturating_sub(row.scanned_at) <= age
                && !config.merge_rollout.draining(&row.name)
                && (!config.worker_endpoints.is_empty()
                    || (config.append.local(&row.name) && config.merge_rollout.owned(&row.name)))
                && after.is_none_or(|after| row.name.as_str() > after)
        })
        .map(|row| row.name.clone())
        .collect();
    names.sort_unstable();
    names.dedup();
    names.truncate(config.wal_tail.wal_tail_batch_size.clamp(1, 64) + 1);
    names
}

pub(crate) fn spawn(state: &Arc<MasterState>) {
    if !enabled(&state.config) {
        return;
    }
    let weak = Arc::downgrade(state);
    tokio::spawn(async move {
        loop {
            let Some(state) = weak.upgrade() else {
                return;
            };
            if let Err(error) = tick(&state, chrono::Utc::now().timestamp_millis()).await {
                tracing::warn!(%error, "WAL tail sweep failed; ordinary scheduling continues");
            }
            drop(state);
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    });
}

async fn tick(state: &Arc<MasterState>, now: i64) -> Result<usize, String> {
    if !enabled(&state.config) {
        return Ok(0);
    }
    let Some(_operation) = state.admission.try_admit() else {
        return Ok(0);
    };
    let key = format!(
        "{}/wal-tail-cursor",
        state.config.etcd.etcd_prefix.trim_end_matches('/')
    );
    let mut client = state.task_store.etcd_client().clone();
    let prior = client
        .get(key.as_str(), None)
        .await
        .map_err(|e| e.to_string())?;
    let old = prior.kvs().first();
    let cursor: Cursor = old
        .map(|kv| serde_json::from_slice(kv.value()))
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    if cursor.next_batch_ms > now {
        return Ok(0);
    }
    // Existing in-memory scalar stats only: no datasets, payloads, WAL listing
    // or stats-writer lock. Followers without a fresh snapshot do no work.
    let snapshot = state.stats_cache.read().await.clone();
    let max_age = state
        .config
        .wal_tail
        .wal_tail_stats_max_age_secs
        .min(86_400) as i64
        * 1000;
    // An uninitialized/stale follower must not consume or reset the shared
    // cursor before a replica with a usable snapshot can finish the rotation.
    if !snapshot
        .iter()
        .any(|row| row.scanned_at <= now && now.saturating_sub(row.scanned_at) <= max_age)
    {
        return Ok(0);
    }
    let mut names = candidates(&state.config, &snapshot, cursor.after.as_deref(), now);
    if names.is_empty() && cursor.after.is_none() {
        return Ok(0);
    }
    let limit = state.config.wal_tail.wal_tail_batch_size.clamp(1, 64);
    let more = names.len() > limit;
    names.truncate(limit);
    let next = Cursor {
        after: if more { names.last().cloned() } else { None },
        next_batch_ms: now.saturating_add(if more {
            30_000
        } else {
            state
                .config
                .wal_tail
                .wal_tail_interval_secs
                .clamp(30, 86_400) as i64
                * 1000
        }),
    };
    // Reserve the bounded page BEFORE any enqueues. A lost response or crash
    // may defer this page until the next rotation; it never authorizes replay
    // or a second replica's burst. The cursor survives master replacement.
    let compare = old.map_or_else(
        || Compare::version(key.as_str(), CompareOp::Equal, 0),
        |kv| Compare::mod_revision(key.as_str(), CompareOp::Equal, kv.mod_revision()),
    );
    if !client
        .txn(Txn::new().when([compare]).and_then([TxnOp::put(
            key.as_str(),
            serde_json::to_vec(&next).map_err(|e| e.to_string())?,
            None,
        )]))
        .await
        .map_err(|e| e.to_string())?
        .succeeded()
    {
        return Ok(0);
    }
    let mut queued = 0;
    for target in names {
        let result = enqueue_tail(state, &target).await;
        match result {
            Ok(true) => queued += 1,
            Ok(false) => {}
            Err(error) => {
                tracing::warn!(%target, %error, "WAL tail enqueue unresolved; no immediate replay")
            }
        }
    }
    metrics::counter!("master_wal_tail_enqueue_requests_total").increment(queued as u64);
    Ok(queued)
}

async fn enqueue_tail(state: &Arc<MasterState>, target: &str) -> Result<bool, String> {
    if state
        .task_store
        .is_cooling_down(TaskKind::MergeWal, target)
        .await
        .map_err(|e| e.to_string())?
        || state
            .task_store
            .get_active_id(TaskKind::MergeWal, target)
            .await
            .map_err(|e| e.to_string())?
            .is_some()
    {
        return Ok(false);
    }
    // Persistent publishers and catch-up Jobs already cover their own tails.
    // This loop must never create a second publisher or change their records.
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
    scheduler::enqueue(state, TaskKind::MergeWal, target)
        .await
        .map_err(|e| e.to_string())?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn config() -> MasterConfig {
        let mut cfg = MasterConfig::parse_from(["master"]);
        cfg.wal_tail.wal_tail_interval_secs = 3600;
        cfg.wal_tail.wal_tail_batch_size = 2;
        cfg.worker_endpoints = vec!["http://unused-test-worker".into()];
        cfg
    }

    fn row(name: &str, pending: i64, at: i64) -> StatRow {
        StatRow {
            name: name.into(),
            uri: format!("/unused/{name}"),
            row_count: 0,
            fragment_count: 0,
            last_updated: at,
            pending_wal_generations: pending,
            last_compaction: -1,
            total_compactions: 0,
            scanned_at: at,
            version: -1,
        }
    }

    #[test]
    fn local_master_services_owned_tails_without_worker_endpoints() {
        let mut cfg = config();
        cfg.worker_endpoints.clear();
        cfg.append.rollout_append_local = true;
        cfg.append.rollout_append_targets = vec!["*".into()];
        cfg.merge_rollout.owned_targets = vec!["hot".into(), "generic:other".into()];
        let now = 2_000_000;
        assert!(enabled(&cfg));
        assert_eq!(
            candidates(
                &cfg,
                &[
                    row("hot", 1, now),
                    row("legacy", 1, now),
                    row("generic:other", 1, now)
                ],
                None,
                now
            ),
            ["hot"]
        );
    }

    #[test]
    fn only_fresh_low_count_tails_enter_the_bounded_rotation() {
        let mut cfg = config();
        cfg.merge_rollout.drain_targets = vec!["draining".into()];
        let now = 2_000_000;
        let rows = vec![
            row("d", 4, now),
            row("a", 1, now),
            row("b", 2, now),
            row("c", 3, now),
            row("empty", 0, now),
            row("hot", 100, now),
            row("stale", 1, now - 901_000),
            row("future", 1, now + 1),
            row("unknown", -1, now),
            row("draining", 1, now),
        ];
        assert_eq!(candidates(&cfg, &rows, None, now), ["a", "b", "c"]);
        assert_eq!(candidates(&cfg, &rows, Some("b"), now), ["c", "d"]);
        assert!(enabled(&cfg));
        cfg.merge_wal_interval_secs = 0;
        assert!(
            !enabled(&cfg),
            "maintenance-only masters must stay disabled"
        );
        cfg.merge_wal_interval_secs = 600;
        cfg.worker_endpoints.clear();
        assert!(!enabled(&cfg));
    }

    async fn fixture() -> (tempfile::TempDir, Arc<MasterState>) {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config();
        cfg.data_dir = dir.path().to_string_lossy().into();
        cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .expect("ETCD_TEST_ENDPOINTS required")
            .split(',')
            .map(str::to_string)
            .collect();
        cfg.etcd.etcd_prefix = format!("/wal-tail-test/{}", lance_context_core::generate_id());
        (dir, MasterState::new(cfg).await.unwrap())
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn replicas_share_a_page_and_restart_continues_the_cursor() {
        let (_dir, state) = fixture().await;
        let now = chrono::Utc::now().timestamp_millis();
        *state.stats_cache.write().await =
            Arc::new(vec![row("c", 1, now), row("a", 1, now), row("b", 1, now)]);
        let (one, two) = tokio::join!(tick(&state, now), tick(&state, now));
        assert_eq!(one.unwrap() + two.unwrap(), 2);
        assert!(state
            .task_store
            .get_active_id(TaskKind::MergeWal, "c")
            .await
            .unwrap()
            .is_none());
        let successor = MasterState::new(state.config.clone()).await.unwrap();
        assert_eq!(
            tick(&successor, now + 30_000).await.unwrap(),
            0,
            "empty follower must not advance the shared cursor"
        );
        *successor.stats_cache.write().await = state.stats_cache.read().await.clone();
        assert_eq!(tick(&successor, now + 1000).await.unwrap(), 0);
        assert_eq!(tick(&successor, now + 30_000).await.unwrap(), 1);
        assert!(successor
            .task_store
            .get_active_id(TaskKind::MergeWal, "c")
            .await
            .unwrap()
            .is_some());
        assert_eq!(tick(&successor, now + 60_000).await.unwrap(), 0);
        // The sweep never opens these intentionally nonexistent table paths.
        assert!(!std::path::Path::new("/unused/a").exists());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn external_owners_and_already_queued_work_are_not_replaced() {
        let (_dir, state) = fixture().await;
        let now = chrono::Utc::now().timestamp_millis();
        *state.stats_cache.write().await =
            Arc::new(vec![row("a", 1, now), row("b", 1, now), row("c", 1, now)]);
        let owner_key = crate::catchup::store::active_key(&state.config.etcd.etcd_prefix, "a");
        state
            .task_store
            .etcd_client()
            .clone()
            .put(owner_key.as_str(), "external-publisher", None)
            .await
            .unwrap();
        let queued = scheduler::enqueue(&state, TaskKind::MergeWal, "b")
            .await
            .unwrap();
        assert_eq!(tick(&state, now).await.unwrap(), 0);
        assert!(state
            .task_store
            .get_active_id(TaskKind::MergeWal, "a")
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            state
                .task_store
                .get_active_id(TaskKind::MergeWal, "b")
                .await
                .unwrap()
                .as_deref(),
            Some(queued.id.as_str())
        );
        assert_eq!(
            state
                .task_store
                .etcd_client()
                .clone()
                .get(owner_key, None)
                .await
                .unwrap()
                .kvs()[0]
                .value(),
            b"external-publisher"
        );
        // Busy rows use a page position, not the whole fleet's remaining budget.
        assert_eq!(tick(&state, now + 30_000).await.unwrap(), 1);
    }
}
