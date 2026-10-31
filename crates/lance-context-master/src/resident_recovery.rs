//! Resident catch-up demand discovery, independent of the full stats scanner.
//! A durable fleet cursor bounds metadata reads even when masters are scaled up.
use crate::{config::MasterConfig, state::MasterState, stats_store::StatRow};
use etcd_client::{Compare, CompareOp, Txn, TxnOp};
use futures::{stream, StreamExt, TryStreamExt};
use lance_context_api::TaskKind;
use lance_context_core::rollout_append::has_sealed_wal;
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
        &config.local_merge.master_local_merge_targets,
    ] {
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.hash(&mut hash);
    }
    config.append.rollout_append_local.hash(&mut hash);
    config.local_merge.master_local_merge_auto.hash(&mut hash);
    config.merge_rollout.fence_legacy.hash(&mut hash);
    hash.finish()
}

fn enabled(config: &MasterConfig) -> bool {
    crate::local_merge::enabled(config)
        && config.append.rollout_append_reconcile_interval_secs > 0
        && config.merge_wal_interval_secs > 0
}

#[cfg(test)]
fn candidates(config: &MasterConfig, snapshot: &[StatRow], after: Option<&str>) -> Vec<String> {
    // Explicitly owned tables do not depend on scanner freshness or existence.
    // Wildcard installations additionally use known names from the stats cache;
    // its counts are never used as evidence of current WAL demand.
    let mut names: Vec<_> = config
        .merge_rollout
        .owned_targets
        .iter()
        .chain(config.append.rollout_append_targets.iter())
        .chain(config.local_merge.master_local_merge_targets.iter())
        .map(String::as_str)
        .chain(snapshot.iter().map(|row| row.name.as_str()))
        .filter(|name| {
            *name != "*"
                && crate::local_merge::selected(config, name)
                && after.is_none_or(|after| *name > after)
        })
        .map(str::to_owned)
        .collect();
    names.sort_unstable();
    names.dedup();
    names.truncate(BATCH + 1);
    names
}

/// Discovery policy carried by the existing leased controller membership.
/// Executors advertise it once; only the elected controller probes WAL metadata.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct Policy {
    root: String,
    owned: Vec<String>,
    draining: Vec<String>,
    append: Vec<String>,
    local: Vec<String>,
    automatic: bool,
    shards: Vec<String>,
    interval_secs: u64,
    cursor: u64,
}

impl Policy {
    pub(crate) fn from_config(config: &MasterConfig) -> Option<Self> {
        let sorted = |values: &[String]| {
            let mut values = values.to_vec();
            values.sort_unstable();
            values.dedup();
            values
        };
        enabled(config).then(|| Self {
            root: config.data_dir.clone(),
            owned: sorted(&config.merge_rollout.owned_targets),
            draining: sorted(&config.merge_rollout.drain_targets),
            append: if config.append.rollout_append_local {
                sorted(&config.append.rollout_append_targets)
            } else {
                Vec::new()
            },
            local: sorted(&config.local_merge.master_local_merge_targets),
            automatic: config.local_merge.master_local_merge_auto
                && config.merge_rollout.fence_legacy,
            shards: sorted(&config.catchup.shards),
            interval_secs: config
                .append
                .rollout_append_reconcile_interval_secs
                .clamp(30, 86_400),
            cursor: scope(config),
        })
    }

    fn selected(&self, target: &str) -> bool {
        let dynamic = self.automatic
            && !target.starts_with("datagen:")
            && lance_context_core::validate_store_name(
                target.strip_prefix("generic:").unwrap_or(target),
            )
            .is_ok();
        (self.owned.iter().any(|t| t == target) || dynamic)
            && !self.draining.iter().any(|t| t == target)
            && (self.local.iter().any(|t| t == "*" || t == target)
                || (!target.starts_with("generic:")
                    && self.append.iter().any(|t| t == "*" || t == target)))
    }

    fn candidates(
        &self,
        snapshot: &[StatRow],
        known: &[String],
        after: Option<&str>,
    ) -> Vec<String> {
        let mut names: Vec<_> = self
            .owned
            .iter()
            .chain(&self.append)
            .chain(&self.local)
            .map(String::as_str)
            .chain(snapshot.iter().map(|r| r.name.as_str()))
            .chain(known.iter().map(String::as_str))
            .filter(|name| *name != "*" && self.selected(name) && after.is_none_or(|a| *name > a))
            .map(str::to_owned)
            .collect();
        names.sort_unstable();
        names.dedup();
        names.truncate(BATCH + 1);
        names
    }
}

/// A leader need not be the stats scanner. Restore names from the persisted
/// stats dataset using a bounded name-only projection, never cached WAL counts.
/// Keep one root at a time; ordinary repeated ticks use this local view.
#[derive(Default)]
struct Catalog {
    root: String,
    checked: Option<tokio::time::Instant>,
    names: Vec<String>,
}

impl Catalog {
    async fn refresh(&mut self, policy: &Policy) -> Result<(), String> {
        if self.root != policy.root {
            self.root = policy.root.clone();
            self.checked = None;
            self.names.clear();
        }
        if !policy.automatic
            || self
                .checked
                .is_some_and(|at| at.elapsed() < Duration::from_secs(300))
        {
            return Ok(());
        }
        self.checked = Some(tokio::time::Instant::now());
        let result = tokio::time::timeout(PROBE_TIMEOUT, catalog_names(&policy.root))
            .await
            .map_err(|_| "resident catalog read timed out".to_string())?;
        self.names = result?;
        Ok(())
    }
}

async fn catalog_names(root: &str) -> Result<Vec<String>, String> {
    let uri = format!("{}/_stats.rollout.lance", root.trim_end_matches('/'));
    let dataset = lance::Dataset::open(&uri)
        .await
        .map_err(|e| e.to_string())?;
    let mut scan = dataset.scan();
    scan.project(&["name"]).map_err(|e| e.to_string())?;
    scan.batch_size(256);
    let mut batches = scan.try_into_stream().await.map_err(|e| e.to_string())?;
    let mut names = std::collections::BTreeSet::new();
    let mut bytes = 0usize;
    while let Some(batch) = batches.try_next().await.map_err(|e| e.to_string())? {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or("invalid resident catalog name column")?;
        for name in column.iter().flatten() {
            if names.insert(name.to_string()) {
                bytes = bytes.saturating_add(name.len());
            }
            if names.len() > 100_000 || bytes > 8 * 1024 * 1024 {
                return Err("resident catalog name budget exceeded".into());
            }
        }
    }
    Ok(names.into_iter().collect())
}

/// Runs inside the demand leader's lifetime, concurrently with dispatch. Slow
/// object storage must not block task placement or consumption of watch events.
pub(crate) async fn run(
    state: &Arc<MasterState>,
    owner: &crate::task_store::CoordinationGuard,
    mut policies: tokio::sync::watch::Receiver<Vec<Policy>>,
) {
    let mut next = 0usize;
    let mut catalog = Catalog::default();
    loop {
        let current = policies.borrow_and_update().clone();
        if !current.is_empty() {
            next %= current.len();
            let policy = &current[next];
            if let Err(error) = catalog.refresh(policy).await {
                tracing::warn!(%error, "resident catalog refresh failed; retaining known names");
            }
            match tokio::time::timeout(
                Duration::from_secs(25),
                tick_policy(
                    state,
                    policy,
                    &catalog.names,
                    Some(owner),
                    chrono::Utc::now().timestamp_millis(),
                ),
            )
            .await
            {
                Ok(Ok(Some(_))) => next += 1,
                Ok(Ok(None)) => {} // The fleet gate is not due; preserve policy fairness.
                outcome => {
                    next += 1;
                    tracing::warn!(?outcome, "leader resident WAL discovery incomplete");
                }
            }
        }
        tokio::select! {
            changed = policies.changed() => if changed.is_err() { return; },
            _ = tokio::time::sleep(Duration::from_secs(5)) => {},
        }
    }
}

#[cfg(test)]
async fn tick(state: &Arc<MasterState>, now: i64) -> Result<usize, String> {
    let Some(policy) = Policy::from_config(&state.config) else {
        return Ok(0);
    };
    tick_policy(state, &policy, &[], None, now)
        .await
        .map(|result| result.unwrap_or(0))
}

async fn tick_policy(
    state: &Arc<MasterState>,
    policy: &Policy,
    known: &[String],
    owner: Option<&crate::task_store::CoordinationGuard>,
    now: i64,
) -> Result<Option<usize>, String> {
    let Some(_admission) = state.admission.try_admit_named("resident_discovery") else {
        return Ok(Some(0));
    };
    let key = format!(
        "{}/resident-wal-discovery",
        state.config.etcd.etcd_prefix.trim_end_matches('/')
    );
    let cursor_key = format!("{key}/cursor/{:016x}", policy.cursor);
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
        return Ok(None);
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
    let rows = if policy.root.trim_end_matches('/') == state.base_uri.trim_end_matches('/') {
        snapshot.as_slice()
    } else {
        &[]
    };
    if policy.candidates(rows, known, None).is_empty() {
        return Ok(Some(0));
    }
    let mut names = policy.candidates(rows, known, cursor.after.as_deref());
    if names.is_empty() && cursor.after.is_none() {
        return Ok(Some(0));
    }
    let more = names.len() > BATCH;
    names.truncate(BATCH);
    let next = Cursor {
        after: if more { names.last().cloned() } else { None },
        next_batch_ms: now.saturating_add(policy.interval_secs as i64 * 1000),
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
    let mut compares = vec![compare, compare_cursor];
    if let Some(owner) = owner {
        compares.push(Compare::value(owner.key(), CompareOp::Equal, owner.token()));
    }
    if !client
        .txn(Txn::new().when(compares).and_then([
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
        return Ok(Some(0));
    }
    let publisher = owner.map(|o| state.task_store.discovery_publisher(o));
    let publisher = publisher.as_ref().unwrap_or(&state.task_store);
    let mut probes = stream::iter(names.into_iter().map(|target| async move {
        let outcome = tokio::time::timeout(PROBE_TIMEOUT, probe_with_policy(state, policy, publisher, &target)).await;
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
    Ok(Some(requested))
}

#[cfg(test)]
async fn probe(state: &Arc<MasterState>, target: &str) -> Result<bool, String> {
    let policy = Policy::from_config(&state.config).expect("resident policy");
    probe_with_policy(state, &policy, &state.task_store, target).await
}

async fn probe_with_policy(
    state: &Arc<MasterState>,
    policy: &Policy,
    publisher: &crate::task_store::TaskStore,
    target: &str,
) -> Result<bool, String> {
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
    let suffix = target.strip_prefix("generic:").map_or_else(
        || format!("{target}.rollout.lance"),
        |name| format!("{name}.generic.lance"),
    );
    let uri = format!("{}/{}", policy.root.trim_end_matches('/'), suffix);
    let pending = has_sealed_wal(
        &uri,
        state
            .local_append
            .as_ref()
            .map(|local| local.session.clone()),
        &policy.shards,
    )
    .await
    .map_err(|e| e.to_string())?;
    if pending {
        let coordinator = state.task_store.merge_coordinator();
        // Persist demand before enqueueing so cancellation or an uncertain
        // enqueue response is still recovered by the paged request scan.
        // Discovery already knows this exact target: waiting for that scan's
        // next 15-second tick only adds idle time before an available executor.
        coordinator.request_merge(target).await?;
        if let Some(request) = coordinator.get_merge_request(target).await? {
            publisher
                .enqueue_merge_request(&request)
                .await
                .map_err(|e| e.to_string())?;
        }
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
        // Neither master runs the scheduler/retry ticker in this fixture.
        // Discovery itself must make the task immediately claimable, with
        // the exact request acknowledged only after a durable queue entry.
        let queued_b = state
            .task_store
            .get_active_id(TaskKind::MergeWal, "b")
            .await
            .unwrap()
            .expect("discovery must enqueue without the retry ticker");
        assert!(coordinator.request_page(None).await.unwrap().0.is_empty());
        assert_eq!(tick(&second, now + 1000).await.unwrap(), 0);
        let restarted = MasterState::new(state.config.clone()).await.unwrap();
        assert_eq!(tick(&restarted, now + 30_000).await.unwrap(), 1);
        assert!(coordinator.request_page(None).await.unwrap().0.is_empty());
        assert_eq!(
            restarted
                .task_store
                .get_active_id(TaskKind::MergeWal, "b")
                .await
                .unwrap(),
            Some(queued_b)
        );
        assert!(restarted
            .task_store
            .get_active_id(TaskKind::MergeWal, "e")
            .await
            .unwrap()
            .is_some());
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

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn native_pass_serves_prepared_index_before_continuing_every_shard() {
        use crate::task_store::TaskKinds;
        let record = |id: String| {
            let dto = serde_json::from_value(serde_json::json!({
                "id": id, "rollout_id": "fixture", "content": "kept",
            }))
            .unwrap();
            lance_context_core::rollout_record_from_add_request(&dto)
        };
        for append in [false, true] {
            let (_dir, original) = fixture().await;
            let mut config = original.config.clone();
            config.append.rollout_append_local = append;
            config.append.rollout_append_targets = if append { vec!["*".into()] } else { vec![] };
            config.local_merge.master_local_merge_targets = vec!["*".into()];
            config.catchup.shards = vec!["early".into(), "late".into()];
            config.catchup.merge_max_generations = 1;
            config.maintenance.index_prepare_targets = vec!["a".into()];
            let state = MasterState::new(config).await.unwrap();
            let mut seed = RolloutStore::open(&state.rollout_uri("a")).await.unwrap();
            seed.add(&[record("seed".into())]).await.unwrap();
            seed.flush().await.unwrap();
            seed.cleanup_own_shard().await.unwrap();
            let prepared = seed.prepare_id_key_index().await.unwrap();
            for shard in ["early", "late"] {
                let writer = RolloutStore::open_with_options(
                    &state.rollout_uri("a"),
                    RolloutStoreOptions {
                        shard_id: Some(shard.into()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
                for generation in 0..3 {
                    writer
                        .add(&[record(format!("{shard}-{generation}"))])
                        .await
                        .unwrap();
                    writer.flush().await.unwrap();
                }
            }
            state
                .task_store
                .enqueue(TaskKind::MergeWal, "a", vec![])
                .await
                .unwrap();
            let merge = state
                .task_store
                .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                .await
                .unwrap()
                .unwrap();
            state
                .task_store
                .enqueue(TaskKind::IndexId, "a", vec![])
                .await
                .unwrap();
            let mut index = state
                .task_store
                .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
                .await
                .unwrap()
                .unwrap();
            assert!(index.preparing_maintenance());
            assert!(!state
                .task_store
                .maintenance_commit_requested("a")
                .await
                .unwrap());
            assert!(state
                .task_store
                .request_maintenance_commit(&index)
                .await
                .unwrap());
            assert!(state
                .task_store
                .maintenance_commit_requested("a")
                .await
                .unwrap());
            assert!(!state
                .task_store
                .maintenance_commit_requested("b")
                .await
                .unwrap());
            let result = crate::merge_execution::run_merge_wal(&state, &merge).await;
            assert!(
                result.as_ref().unwrap().contains("2 generations"),
                "{result:?}, append={append}"
            );
            let reader = RolloutStore::open(&state.rollout_uri("a")).await.unwrap();
            for shard in ["early", "late"] {
                assert!(reader
                    .get_by_id_source(
                        &format!("{shard}-0"),
                        lance_context_core::ListSource::Fragments
                    )
                    .await
                    .unwrap()
                    .is_some());
                assert!(reader
                    .get_by_id_source(
                        &format!("{shard}-1"),
                        lance_context_core::ListSource::Fragments
                    )
                    .await
                    .unwrap()
                    .is_none());
            }
            let coordinator = state.task_store.merge_coordinator();
            let request = coordinator.get_merge_request("a").await.unwrap().unwrap();
            state.task_store.finish(merge, result).await.unwrap();
            crate::scheduler::enqueue_merge_request(&state, &coordinator, &request)
                .await
                .unwrap();
            assert!(state
                .task_store
                .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                .await
                .unwrap()
                .is_none());
            assert!(state
                .task_store
                .promote_maintenance(&mut index)
                .await
                .unwrap());
            let result = crate::maintenance_execution::run(&state, &index, async {
                let mut writer = RolloutStore::open(&state.rollout_uri("a"))
                    .await
                    .map_err(|e| e.to_string())?;
                writer
                    .commit_prepared_id_key_index(prepared)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok("index published".into())
            })
            .await;
            assert!(result.is_ok(), "{result:?}");
            state.task_store.finish(index, result).await.unwrap();
            assert!(!state
                .task_store
                .maintenance_commit_requested("a")
                .await
                .unwrap());
            let next = state
                .task_store
                .claim_next_of_kinds(TaskKinds::MERGE_WAL)
                .await
                .unwrap()
                .unwrap();
            let result = crate::merge_execution::run_merge_wal(&state, &next).await;
            assert!(
                result.as_ref().unwrap().contains("4 generations"),
                "{result:?}"
            );
            state.task_store.finish(next, result).await.unwrap();
            let reader = lance::Dataset::open(&state.rollout_uri("a")).await.unwrap();
            assert_eq!(reader.count_rows(None).await.unwrap(), 7);
        }
    }

    async fn generic_fixture() -> (tempfile::TempDir, Arc<MasterState>) {
        generic_fixture_with_auto(false).await
    }

    async fn generic_fixture_with_auto(auto: bool) -> (tempfile::TempDir, Arc<MasterState>) {
        let dir = tempfile::tempdir().unwrap();
        let mut config = MasterConfig::parse_from(["test"]);
        config.data_dir = dir.path().to_string_lossy().into();
        config.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        config.etcd.etcd_prefix = format!(
            "/resident-generic-test/{}",
            lance_context_core::generate_id()
        );
        config.merge_rollout.owned_targets = vec!["generic:a".into(), "generic:b".into()];
        config.local_merge.master_local_merge_targets = vec!["*".into()];
        if auto {
            config.merge_rollout.owned_targets.clear();
            config.merge_rollout.fence_legacy = true;
            config.local_merge.master_local_merge_auto = true;
        }
        config.catchup.shards = vec!["writer".into()];
        config.catchup.merge_max_generations = 1;
        config.catchup.merge_max_bytes = 1024 * 1024;
        config.append.rollout_append_local_memory_bytes = 2 * 1024 * 1024;
        config.worker_endpoints.clear();
        config.stats_scan_interval_secs = 0;
        (dir, MasterState::new(config).await.unwrap())
    }

    async fn generic_writer(state: &MasterState, name: &str) -> lance_context_core::GenericStore {
        use lance_context_core::{
            ColumnSpec, ColumnType, GenericStore, GenericStoreOptions, SchemaSpec,
        };
        GenericStore::open(
            &state.generic_uri(name),
            SchemaSpec::new(vec![
                (
                    "id".into(),
                    ColumnSpec::required(ColumnType::String { large: false }),
                ),
                (
                    "text".into(),
                    ColumnSpec::new(ColumnType::String { large: true }),
                ),
            ]),
            GenericStoreOptions {
                shard_id: Some("writer".into()),
                merge_after_generations: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap()
    }

    async fn generic_write(writer: &lance_context_core::GenericStore, id: &str, text: &str) {
        writer
            .add(&[serde_json::json!({"id":id, "text":text})
                .as_object()
                .unwrap()
                .clone()])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn empty_native_preflight_preserves_racing_flush_and_successor_catchup() {
        let (_dir, initial) = generic_fixture_with_auto(true).await;
        let mut cfg = initial.config.clone();
        // Exercise the worst-case positive probe: only the final writer has WAL.
        cfg.catchup.shards = (0..19).map(|i| format!("empty-{i}")).collect();
        cfg.catchup.shards.push("writer".into());
        let state = MasterState::new(cfg).await.unwrap();
        let successor = MasterState::new(state.config.clone()).await.unwrap();
        let generic = generic_writer(&state, "preflight").await;
        let rollout = write(&state, "preflight", 0).await;

        for target in ["generic:preflight", "preflight"] {
            let uri = if target.starts_with("generic:") {
                state.generic_uri("preflight")
            } else {
                state.rollout_uri("preflight")
            };
            let before = lance::Dataset::open(&uri).await.unwrap().version().version;
            let probe = lance_context_core::rollout_append::SealedWalProbe::open(&uri, None)
                .await
                .unwrap();
            assert!(!probe.has_sealed_wal("writer").await.unwrap());
            let c = state.task_store.merge_coordinator();
            state
                .task_store
                .enqueue(TaskKind::MergeWal, target, vec![])
                .await
                .unwrap();
            let claim = state
                .task_store
                .claim_resident_merge(&crate::local_merge::targets(&state.config))
                .await
                .unwrap()
                .unwrap();
            let result = crate::merge_execution::run_merge_wal(&state, &claim).await;
            assert_eq!(
                result.as_ref().unwrap(),
                "local WAL merge reclaimed 0 generations"
            );
            assert!(c.get(target).await.unwrap().is_none());
            assert_eq!(
                lance::Dataset::open(&uri).await.unwrap().version().version,
                before,
                "empty execution must not evolve schema or commit a base version"
            );
            assert!(
                c.request_page(None).await.unwrap().0.is_empty(),
                "an empty pass must not schedule its own continuation"
            );

            // A flush arrives after the empty observation but before task finish.
            // The request must survive dedupe against that still-active task.
            if target.starts_with("generic:") {
                generic_write(&generic, "late", "preserved").await;
            } else {
                let dto = serde_json::from_value(serde_json::json!({
                    "id": "late", "rollout_id": "r", "content": "preserved"
                }))
                .unwrap();
                rollout
                    .add(&[lance_context_core::rollout_record_from_add_request(&dto)])
                    .await
                    .unwrap();
                rollout.flush().await.unwrap();
            }
            assert!(
                probe.has_sealed_wal("writer").await.unwrap(),
                "the retained base handle must observe a flush after its empty check"
            );
            c.request_merge(target).await.unwrap();
            let request = c.request_page(None).await.unwrap().0.remove(0);
            crate::scheduler::enqueue_merge_request(&successor, &c, &request)
                .await
                .unwrap();
            assert_eq!(c.request_page(None).await.unwrap().0.len(), 1);
            state.task_store.finish(claim, result).await.unwrap();
            crate::scheduler::enqueue_merge_request(&successor, &c, &request)
                .await
                .unwrap();
            let claim = successor
                .task_store
                .claim_resident_merge(&crate::local_merge::targets(&successor.config))
                .await
                .unwrap()
                .unwrap();
            let result = crate::merge_execution::run_merge_wal(&successor, &claim).await;
            assert_eq!(
                result.as_ref().unwrap(),
                "local WAL merge reclaimed 1 generations"
            );
            successor.task_store.finish(claim, result).await.unwrap();
            assert!(
                !probe.has_sealed_wal("writer").await.unwrap(),
                "the same probe must observe a drain after the base version changes"
            );
            assert_eq!(
                lance::Dataset::open(&uri)
                    .await
                    .unwrap()
                    .count_rows(None)
                    .await
                    .unwrap(),
                1
            );
            assert!(!lance_context_core::rollout_append::has_sealed_wal(
                &uri,
                None,
                &state.config.catchup.shards,
            )
            .await
            .unwrap());
            assert!(c.request_page(None).await.unwrap().0.is_empty());
            assert_eq!(
                successor.local_append.as_ref().unwrap().budget.reserved(),
                0
            );
        }
        assert_eq!(
            generic.list(None, None).await.unwrap()[0]["text"],
            "preserved"
        );
        assert_eq!(
            rollout
                .get_by_id("late")
                .await
                .unwrap()
                .unwrap()
                .content
                .as_deref(),
            Some("preserved")
        );
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn unreadable_native_preflight_is_not_successful_empty_work() {
        let (_dir, state) = generic_fixture_with_auto(true).await;
        let target = "missing-table";
        state
            .task_store
            .enqueue(TaskKind::MergeWal, target, vec![])
            .await
            .unwrap();
        let claim = state
            .task_store
            .claim_resident_merge(&crate::local_merge::targets(&state.config))
            .await
            .unwrap()
            .unwrap();
        let result = crate::merge_execution::run_merge_wal(&state, &claim).await;
        assert!(
            result.is_err(),
            "a failed metadata read must not become zero-work success"
        );
        state.task_store.finish(claim, result).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn automatic_new_table_waits_for_background_then_catches_up_across_masters() {
        use lance_context_merge::{Execution, Phase};
        let (_dir, state) = generic_fixture_with_auto(true).await;
        assert!(state.config.merge_rollout.owned_targets.is_empty());
        assert!(state.config.maintenance.fence_legacy_maintenance);
        // Created after master startup, with no stats row or static allowlist.
        let writer = generic_writer(&state, "new-hot").await;
        for n in 0..17 {
            generic_write(&writer, "shared", &n.to_string()).await;
        }
        let target = "generic:new-hot";
        let c = state.task_store.merge_coordinator();
        let mut e = Execution::new(target, "http://127.0.0.1:1", "live-worker", 60);
        e.background = true;
        assert!(c.reserve_background(&e).await.unwrap());
        let running = c.start(&e).await.unwrap().unwrap();
        assert!(c.publish_progress(&running, 0).await.unwrap());
        let request = c.request_page(None).await.unwrap().0.remove(0);
        crate::scheduler::enqueue_merge_request(&state, &c, &request)
            .await
            .unwrap();
        let claim = state
            .task_store
            .claim_resident_merge(&crate::local_merge::targets(&state.config))
            .await
            .unwrap()
            .unwrap();
        let next_master = MasterState::new(state.config.clone()).await.unwrap();
        assert!(next_master
            .task_store
            .claim_resident_merge(&crate::local_merge::targets(&next_master.config))
            .await
            .unwrap()
            .is_none());
        let first = tokio::spawn({
            let state = state.clone();
            async move {
                let outcome = crate::merge_execution::run_merge_wal(&state, &claim).await;
                (claim, outcome)
            }
        });
        for sequence in 1..=4 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let current = c.get(target).await.unwrap().unwrap();
            assert_eq!(current.id, running.id);
            assert_eq!(
                current.phase,
                Phase::Running,
                "native selection must preserve useful worker progress"
            );
            assert!(c.publish_progress(&running, sequence).await.unwrap());
            assert!(!first.is_finished());
        }
        assert!(c.finish(&running, Ok(0)).await.unwrap());
        let (claim, result) = tokio::time::timeout(Duration::from_secs(30), first)
            .await
            .unwrap()
            .unwrap();
        assert!(
            result.as_ref().unwrap().contains("16 generations"),
            "{result:?}"
        );
        let request = c.request_page(None).await.unwrap().0.remove(0);
        crate::scheduler::enqueue_merge_request(&next_master, &c, &request)
            .await
            .unwrap();
        assert_eq!(c.request_page(None).await.unwrap().0.len(), 1);
        state.task_store.finish(claim, result).await.unwrap();
        crate::scheduler::enqueue_merge_request(&next_master, &c, &request)
            .await
            .unwrap();
        let claim = next_master
            .task_store
            .claim_resident_merge(&crate::local_merge::targets(&next_master.config))
            .await
            .unwrap()
            .unwrap();
        let result = crate::merge_execution::run_merge_wal(&next_master, &claim).await;
        assert!(
            result.as_ref().unwrap().contains("1 generations"),
            "{result:?}"
        );
        next_master.task_store.finish(claim, result).await.unwrap();
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
        let rows = writer.list(None, None).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["text"], "16");
        assert_eq!(state.local_append.as_ref().unwrap().budget.reserved(), 0);
        assert_eq!(
            next_master.local_append.as_ref().unwrap().budget.reserved(),
            0
        );
        generic_write(&writer, "shared", "after-handoff").await;
        assert!(probe(&next_master, target).await.unwrap());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn automatic_rollout_uses_native_merge_and_preserves_external_owner() {
        let (_dir, initial) = generic_fixture_with_auto(true).await;
        let mut cfg = initial.config.clone();
        // Existing append wildcard must not send an unowned new table through
        // the explicit-ownership staged-append protocol.
        cfg.append.rollout_append_targets = vec!["*".into()];
        let state = MasterState::new(cfg).await.unwrap();
        let target = "new-rollout";
        let writer = write(&state, target, 4).await;
        let c = state.task_store.merge_coordinator();
        let owner = crate::catchup::store::active_key(&state.config.etcd.etcd_prefix, target);
        let mut client = state.task_store.etcd_client().clone();
        client
            .put(owner.as_str(), "fixture-helper", None)
            .await
            .unwrap();
        assert!(!probe(&state, target).await.unwrap());
        c.request_merge(target).await.unwrap();
        let request = c.request_page(None).await.unwrap().0.remove(0);
        crate::scheduler::enqueue_merge_request(&state, &c, &request)
            .await
            .unwrap();
        assert!(state
            .task_store
            .claim_resident_merge(&crate::local_merge::targets(&state.config))
            .await
            .unwrap()
            .is_none());
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 4);
        // This fixture has no remote actor or commits. End its known ownership.
        client.delete(owner, None).await.unwrap();
        let claim = state
            .task_store
            .claim_resident_merge(&crate::local_merge::targets(&state.config))
            .await
            .unwrap()
            .unwrap();
        let result = crate::merge_execution::run_merge_wal(&state, &claim).await;
        assert!(
            result
                .as_ref()
                .unwrap()
                .contains("local WAL merge reclaimed 4 generations"),
            "{result:?}"
        );
        state.task_store.finish(claim, result).await.unwrap();
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
        assert_eq!(
            lance::Dataset::open(&state.rollout_uri(target))
                .await
                .unwrap()
                .count_rows(None)
                .await
                .unwrap(),
            4
        );
        assert_eq!(state.local_append.as_ref().unwrap().budget.reserved(), 0);
        assert!(!probe(&state, target).await.unwrap());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn generic_discovery_and_continuation_preserve_updates_without_workers() {
        let (_dir, state) = generic_fixture().await;
        let writer = generic_writer(&state, "a").await;
        for n in 0..17 {
            generic_write(&writer, "shared", &n.to_string()).await;
        }
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 17);
        // A schema-independent manifest probe discovers generic demand without
        // payload reads or a stats scan. A second master cannot duplicate it.
        let second = MasterState::new(state.config.clone()).await.unwrap();
        let now = chrono::Utc::now().timestamp_millis();
        let (a, b) = tokio::join!(tick(&state, now), tick(&second, now));
        assert_eq!(a.unwrap() + b.unwrap(), 1);
        let coordinator = state.task_store.merge_coordinator();
        // Discovery must admit generic work too, without a request-scan tick.
        assert!(coordinator.request_page(None).await.unwrap().0.is_empty());
        let claim = state
            .task_store
            .claim_resident_merge(&crate::local_merge::targets(&state.config))
            .await
            .unwrap()
            .unwrap();
        assert!(second
            .task_store
            .claim_resident_merge(&crate::local_merge::targets(&second.config))
            .await
            .unwrap()
            .is_none());
        let result = crate::merge_execution::run_merge_wal(&state, &claim).await;
        assert!(result.as_ref().unwrap().contains("16 generations"));
        let next = coordinator.request_page(None).await.unwrap().0.remove(0);
        crate::scheduler::enqueue_merge_request(&second, &coordinator, &next)
            .await
            .unwrap();
        assert_eq!(coordinator.request_page(None).await.unwrap().0.len(), 1);
        state.task_store.finish(claim, result).await.unwrap();
        crate::scheduler::enqueue_merge_request(&second, &coordinator, &next)
            .await
            .unwrap();
        let claim = second
            .task_store
            .claim_resident_merge(&crate::local_merge::targets(&second.config))
            .await
            .unwrap()
            .unwrap();
        let result = crate::merge_execution::run_merge_wal(&second, &claim).await;
        assert!(result.as_ref().unwrap().contains("1 generations"));
        second.task_store.finish(claim, result).await.unwrap();
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
        let rows = writer.list(None, None).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["text"], "16");
        assert!(!probe(&second, "generic:a").await.unwrap());
        assert_eq!(state.local_append.as_ref().unwrap().budget.reserved(), 0);
        assert_eq!(second.local_append.as_ref().unwrap().budget.reserved(), 0);
        // No ingestion epoch was acquired by either master.
        generic_write(&writer, "shared", "after").await;
        assert!(probe(&second, "generic:a").await.unwrap());
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn generic_replicas_merge_distinct_tables_during_live_ingestion() {
        let (_dir, state) = generic_fixture().await;
        let a = generic_writer(&state, "a").await;
        let b = generic_writer(&state, "b").await;
        for n in 0..4 {
            generic_write(&a, "shared", &n.to_string()).await;
            generic_write(&b, "shared", &n.to_string()).await;
        }
        for target in ["generic:a", "generic:b"] {
            state
                .task_store
                .enqueue(TaskKind::MergeWal, target, vec![])
                .await
                .unwrap();
        }
        let second = MasterState::new(state.config.clone()).await.unwrap();
        let targets = crate::local_merge::targets(&state.config);
        let claim_a = state
            .task_store
            .claim_resident_merge(&targets)
            .await
            .unwrap()
            .unwrap();
        let claim_b = second
            .task_store
            .claim_resident_merge(&targets)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(claim_a.task.target, claim_b.task.target);
        let budget = state.local_append.as_ref().unwrap().budget.clone();
        let held = budget.reserve(budget.limit()).await;
        let runner = {
            let state = state.clone();
            tokio::spawn(async move {
                let result = crate::merge_execution::run_merge_wal(&state, &claim_a).await;
                assert!(result.is_ok(), "{result:?}");
                state.task_store.finish(claim_a, result).await.unwrap();
            })
        };
        let (rb, ()) = tokio::join!(
            crate::merge_execution::run_merge_wal(&second, &claim_b),
            async {
                for n in 0..8 {
                    generic_write(&a, &format!("during-{n}"), "live").await;
                    generic_write(&b, &format!("during-{n}"), "live").await;
                }
            }
        );
        assert!(rb.is_ok(), "{rb:?}");
        second.task_store.finish(claim_b, rb).await.unwrap();
        assert!(
            !runner.is_finished(),
            "local generic merge must respect the process budget"
        );
        drop(held);
        tokio::time::timeout(Duration::from_secs(30), runner)
            .await
            .unwrap()
            .unwrap();
        for writer in [&a, &b] {
            generic_write(writer, "after", "live").await;
            let rows = writer.list(None, None).await.unwrap();
            assert_eq!(rows.len(), 10);
            assert_eq!(
                rows.iter().find(|row| row["id"] == "shared").unwrap()["text"],
                "3"
            );
        }
        assert_eq!(state.local_append.as_ref().unwrap().budget.reserved(), 0);
        assert_eq!(second.local_append.as_ref().unwrap().budget.reserved(), 0);
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
    #[tokio::test]
    async fn wildcard_catalog_recovers_names_without_a_local_stats_scanner() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_str().unwrap();
        let mut stats = crate::stats_store::StatsStore::open_or_create(
            &format!("{root}/_stats.rollout.lance"),
            None,
        )
        .await
        .unwrap();
        let row = |name: &str| StatRow {
            name: name.into(),
            uri: "unused".into(),
            row_count: 0,
            fragment_count: 0,
            last_updated: 0,
            pending_wal_generations: 0,
            last_compaction: -1,
            total_compactions: 0,
            scanned_at: 0,
            version: -1,
        };
        stats
            .write_snapshot(&[row("cold"), row("generic:cold")])
            .await
            .unwrap();
        let mut cfg = MasterConfig::parse_from(["test"]);
        cfg.data_dir = root.into();
        cfg.merge_wal_interval_secs = 30;
        cfg.local_merge.master_local_merge_auto = true;
        cfg.local_merge.master_local_merge_targets = vec!["*".into()];
        cfg.merge_rollout.fence_legacy = true;
        let policy = Policy::from_config(&cfg).unwrap();
        let mut catalog = Catalog::default();
        catalog.refresh(&policy).await.unwrap();
        assert_eq!(
            policy.candidates(&[], &catalog.names, None),
            ["cold", "generic:cold"]
        );
        stats.write_snapshot(&[row("new")]).await.unwrap();
        catalog.refresh(&policy).await.unwrap();
        assert!(
            !catalog.names.iter().any(|name| name == "new"),
            "ordinary ticks use cached names"
        );
        catalog.checked = None;
        catalog.refresh(&policy).await.unwrap();
        assert!(catalog.names.iter().any(|name| name == "new"));
    }

    #[test]
    fn advertised_policy_preserves_selection_and_deduplicates_reordered_members() {
        for automatic in [false, true] {
            for append in [false, true] {
                let mut config = MasterConfig::parse_from(["test"]);
                config.merge_wal_interval_secs = 30;
                config.append.rollout_append_local = append;
                config.append.rollout_append_targets = vec!["*".into()];
                config.local_merge.master_local_merge_targets = vec!["*".into()];
                config.local_merge.master_local_merge_auto = automatic;
                config.merge_rollout.fence_legacy = automatic;
                config.merge_rollout.owned_targets =
                    vec!["b".into(), "a".into(), "generic:g".into()];
                config.merge_rollout.drain_targets = vec!["drained".into()];
                let policy = Policy::from_config(&config).unwrap();
                for name in [
                    "a",
                    "b",
                    "unknown",
                    "generic:g",
                    "generic:unknown",
                    "datagen:d",
                    "drained",
                    "../bad",
                ] {
                    assert_eq!(
                        policy.selected(name),
                        crate::local_merge::selected(&config, name),
                        "{name}"
                    );
                }
                assert_eq!(
                    policy.candidates(&[], &[], None),
                    candidates(&config, &[], None)
                );
                config.merge_rollout.owned_targets.reverse();
                assert_eq!(Some(policy), Policy::from_config(&config));
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn leader_probes_member_policy_and_successor_resumes_fenced_cursor() {
        let (_dir, member) = fixture().await;
        let _b = write(&member, "b", 1).await;
        let _e = write(&member, "e", 1).await;
        let policy = Policy::from_config(&member.config).unwrap();
        let mut cfg = member.config.clone();
        cfg.append.rollout_append_local = false;
        cfg.merge_rollout.owned_targets.clear();
        let leader = MasterState::new(cfg).await.unwrap();
        assert!(Policy::from_config(&leader.config).is_none());
        let store = &leader.task_store;
        let owner = store.discovery_session(b"{}".to_vec()).await.unwrap();
        assert!(store.elect_discovery_leader(&owner).await.unwrap());
        let now = chrono::Utc::now().timestamp_millis();
        assert_eq!(
            tick_policy(&leader, &policy, &[], Some(&owner), now)
                .await
                .unwrap(),
            Some(1)
        );
        let first = store
            .get_active_id(TaskKind::MergeWal, "b")
            .await
            .unwrap()
            .unwrap();
        let gate = format!("{}/resident-wal-discovery", leader.config.etcd.prefix());
        let cursor_key = format!("{gate}/cursor/{:016x}", policy.cursor);
        let cursor = store
            .etcd_client()
            .clone()
            .get(cursor_key.clone(), None)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Cursor>(cursor.kvs()[0].value())
                .unwrap()
                .after
                .as_deref(),
            Some("d")
        );
        // Replace leadership while retaining the old guard/keepalive: no old
        // timer or lease liveness observation may authorize cursor advancement.
        store
            .etcd_client()
            .clone()
            .delete(owner.key(), None)
            .await
            .unwrap();
        let successor = store.discovery_session(b"{}".to_vec()).await.unwrap();
        assert!(store.elect_discovery_leader(&successor).await.unwrap());
        assert_eq!(
            tick_policy(&leader, &policy, &[], Some(&owner), now + 31_000)
                .await
                .unwrap(),
            Some(0)
        );
        let unchanged = store
            .etcd_client()
            .clone()
            .get(cursor_key, None)
            .await
            .unwrap();
        assert_eq!(
            unchanged.kvs()[0].mod_revision(),
            cursor.kvs()[0].mod_revision()
        );
        let stale_publisher = store.discovery_publisher(&owner);
        assert!(probe_with_policy(&leader, &policy, &stale_publisher, "e")
            .await
            .unwrap());
        assert!(store
            .get_active_id(TaskKind::MergeWal, "e")
            .await
            .unwrap()
            .is_none());
        assert!(store
            .merge_coordinator()
            .get_merge_request("e")
            .await
            .unwrap()
            .is_some());
        assert_eq!(
            tick_policy(&leader, &policy, &[], Some(&successor), now + 31_000)
                .await
                .unwrap(),
            Some(1)
        );
        assert!(store
            .get_active_id(TaskKind::MergeWal, "e")
            .await
            .unwrap()
            .is_some());
        assert_eq!(
            store
                .get_active_id(TaskKind::MergeWal, "b")
                .await
                .unwrap()
                .as_deref(),
            Some(first.as_str())
        );
        store.release_coordination_lock(owner).await.unwrap();
        store.release_coordination_lock(successor).await.unwrap();
    }
    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn running_leader_discovers_new_member_scope_without_becoming_executor() {
        let (_dir, member) = fixture().await;
        let _b = write(&member, "b", 1).await;
        let mut cfg = member.config.clone();
        cfg.append.rollout_append_local = false;
        cfg.merge_rollout.owned_targets.clear();
        let leader = MasterState::new(cfg).await.unwrap();
        let store = &leader.task_store;
        let key = format!(
            "{}/demand-controller/leader-v2",
            leader.config.etcd.prefix()
        );
        let first = crate::demand_controller::spawn(&leader);
        let original = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let response = store
                    .etcd_client()
                    .clone()
                    .get(key.clone(), None)
                    .await
                    .unwrap();
                if let Some(kv) = response.kvs().first() {
                    break kv.value().to_vec();
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .unwrap();
        let follower = crate::demand_controller::spawn(&member);
        let discovered = tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                if store
                    .get_active_id(TaskKind::MergeWal, "b")
                    .await
                    .unwrap()
                    .is_some()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await;
        let current = store.etcd_client().clone().get(key, None).await.unwrap();
        first.abort();
        follower.abort();
        let _ = first.await;
        let _ = follower.await;
        discovered.unwrap();
        assert_eq!(current.kvs()[0].value(), original);
        assert!(!crate::local_merge::enabled(&leader.config));
    }
}
