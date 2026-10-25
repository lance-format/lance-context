//! Capacity-backed task offers. Only the elected controller selects candidates;
//! receiving pools retain all canonical dependency and ownership checks.
use super::{prefix_key, Cache};
use crate::{
    admission,
    config::MasterConfig,
    eligibility,
    state::MasterState,
    task_store::{DispatchAssignment, TaskClaim, TaskKinds, TaskStore},
};
use etcd_client::WatchOptions;
use lance_context_api::{TaskKind, TaskRecord, TaskState};
use lance_context_merge::rollout::MergeRollout;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Ticket {
    v: u32,
    executor: String,
    pool: String,
    kinds: TaskKinds,
    resident_targets: Option<Vec<String>>,
    wal_claim_targets: Vec<String>,
    rollout: MergeRollout,
    auto_local_merge: bool,
    compact_prepare: Vec<String>,
    index_prepare: Vec<String>,
    catchup_maintenance: Vec<String>,
    repair_validate: Vec<String>,
}

impl Ticket {
    pub(crate) fn new(
        config: &MasterConfig,
        executor: &str,
        pool: &str,
        kinds: TaskKinds,
        resident: bool,
    ) -> Self {
        Self {
            v: 1,
            executor: executor.into(),
            pool: pool.into(),
            kinds,
            resident_targets: resident.then(|| crate::local_merge::targets(config)),
            wal_claim_targets: config.maintenance.wal_claim_targets.clone(),
            rollout: config.merge_rollout.clone(),
            auto_local_merge: config.local_merge.master_local_merge_auto,
            compact_prepare: config.maintenance.compaction_prepare_targets.clone(),
            index_prepare: config.maintenance.index_prepare_targets.clone(),
            catchup_maintenance: config.maintenance.maintenance_catchup_targets.clone(),
            repair_validate: config.maintenance.repair_validate_only_targets.clone(),
        }
    }
    fn policy(&self) -> eligibility::Policy<'_> {
        eligibility::Policy {
            auto_local_merge: self.auto_local_merge,
            rollout: &self.rollout,
            compaction_prepare_targets: &self.compact_prepare,
            index_prepare_targets: &self.index_prepare,
            maintenance_catchup_targets: &self.catchup_maintenance,
            repair_validate_only_targets: &self.repair_validate,
        }
    }
    fn eligible(&self, prefix: &str, cache: &Cache, task: &TaskRecord) -> bool {
        if self.v != 1
            || (task.kind == TaskKind::MergeWal
                && !self.wal_claim_targets.is_empty()
                && !self.wal_claim_targets.contains(&task.target))
        {
            return false;
        }
        let policy = self.policy();
        if eligibility::pre_dependency_filter(
            task.kind,
            &task.target,
            |kind| self.kinds.contains(kind),
            self.resident_targets.as_deref(),
            &policy,
        )
        .is_err()
        {
            return false;
        }
        let execution_key = lance_context_merge::execution_key(prefix, &task.target);
        let value = |key: &str| {
            cache
                .entries
                .get(key.as_bytes())
                .map(|entry| entry.value.as_slice())
        };
        let execution_value = value(&execution_key);
        let execution = match execution_value
            .map(serde_json::from_slice::<lance_context_merge::Execution>)
            .transpose()
        {
            Ok(value) => value,
            Err(_) => return false,
        };
        let owner = execution.as_ref().map(lance_context_merge::execution_owner);
        let target_key = lance_context_merge::target_lock_key(prefix, &task.target);
        let active_key = execution_key.replace("/merge-executions/", "/catchup-active/");
        let claim_key = format!("{prefix}/claims/{}", task.id);
        let merge_claim_key = execution_key.replace("/merge-executions/", "/merge-claims/");
        let prep_key = execution_key.replace("/merge-executions/", "/compact-preparations/");
        let commit_key = execution_key.replace("/merge-executions/", "/compact-commit-ready/");
        let active = value(&active_key);
        let shape = eligibility::claim_shape(task.kind, &task.target, &policy, active.is_some());
        let orphan = value(&claim_key).is_none()
            && value(&merge_claim_key).is_none()
            && value(&target_key) == owner.as_deref().map(str::as_bytes)
            && eligibility::orphaned_catchup_candidate(
                task.kind,
                &task.target,
                &policy,
                execution.as_ref(),
                active,
            )
            && value(&execution_key.replace("/merge-executions/", "/catchup-records/")).is_none();
        eligibility::admission_filter(
            task.kind,
            shape,
            &eligibility::Snapshot {
                merge_execution: execution_value,
                target_lock: value(&target_key),
                catchup_active: active,
                merge_claim: value(&merge_claim_key),
                task_claim: value(&claim_key),
                compact_preparation: value(&prep_key),
                compact_commit_ready: value(&commit_key),
            },
            owner.as_deref(),
            execution.as_ref().is_some_and(|e| e.maintenance.is_none()),
            (shape.shared_owner || orphan).then_some(active).flatten(),
        )
        .is_ok()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Offer {
    v: u32,
    task_id: String,
    nonce: String,
}

fn assignment(
    prefix: &str,
    ticket_key: &str,
    ticket_value: &[u8],
    offer: &Offer,
) -> Result<DispatchAssignment, String> {
    let id = ticket_key
        .strip_prefix(&prefix_key(prefix, "demand-controller/tickets"))
        .ok_or("invalid ticket key")?;
    if id.is_empty() || id.contains('/') || offer.v != 1 {
        return Err("invalid dispatch offer".into());
    }
    Ok(DispatchAssignment {
        task_id: offer.task_id.clone(),
        offer_key: format!("{prefix}/demand-controller/offers/{id}"),
        offer_value: serde_json::to_vec(offer).map_err(|e| e.to_string())?,
        ticket_key: ticket_key.into(),
        ticket_value: ticket_value.into(),
        reservation_key: format!("{prefix}/demand-controller/reservations/{}", offer.task_id),
    })
}

/// Caller holds the existing pool/compaction permits for this entire operation.
/// Waiting does no fleet scan. Once an offer arrives, claim completion/ambiguity
/// is joined even if master admission starts draining concurrently.
pub(crate) async fn claim(
    state: &Arc<MasterState>,
    ticket: &Ticket,
    operation: &admission::Operation,
    refresh_compaction: bool,
) -> lance::Result<Option<TaskClaim>> {
    let store = &state.task_store;
    let value = serde_json::to_vec(ticket).map_err(|e| lance::Error::io(e.to_string()))?;
    let guard = store.create_dispatch_ticket(value.clone()).await?;
    let prefix = state.config.etcd.prefix();
    let offer_key = guard.key().replace("/tickets/", "/offers/");
    operation.claim_phase("waiting_for_controller_offer");
    let waiting = async {
        let snapshot = store
            .etcd_client()
            .clone()
            .get(offer_key.clone(), None)
            .await
            .map_err(|e| lance::Error::io(e.to_string()))?;
        if let Some(kv) = snapshot.kvs().first() {
            return serde_json::from_slice::<Offer>(kv.value())
                .map(Some)
                .map_err(|e| lance::Error::io(e.to_string()));
        }
        let revision = snapshot
            .header()
            .ok_or_else(|| lance::Error::io("missing offer snapshot revision"))?
            .revision();
        let mut stream = store
            .etcd_client()
            .clone()
            .watch(
                offer_key.clone(),
                Some(WatchOptions::new().with_start_revision(revision + 1)),
            )
            .await
            .map_err(|e| lance::Error::io(e.to_string()))?;
        while let Some(response) = stream
            .message()
            .await
            .map_err(|e| lance::Error::io(e.to_string()))?
        {
            if response.canceled() {
                return Err(lance::Error::io("offer watch canceled"));
            }
            for event in response.events() {
                if event.event_type() == etcd_client::EventType::Put {
                    return serde_json::from_slice::<Offer>(
                        event
                            .kv()
                            .ok_or_else(|| lance::Error::io("offer event missing value"))?
                            .value(),
                    )
                    .map(Some)
                    .map_err(|e| lance::Error::io(e.to_string()));
                }
            }
        }
        Err(lance::Error::io("offer watch ended"))
    };
    let stopped = async {
        while state.admission.is_accepting()
            && guard.is_live()
            && !(refresh_compaction && state.compaction_permits.available_permits() > 0)
        {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    };
    let received = tokio::select! { result = waiting => result, _ = stopped => Ok(None) };
    let result = match received {
        Ok(Some(offer)) => match assignment(prefix, guard.key(), &value, &offer) {
            Ok(offer) => {
                store
                    .claim_assignment(
                        &offer,
                        ticket.kinds,
                        ticket.resident_targets.as_deref(),
                        operation,
                    )
                    .await
            }
            Err(error) => Err(lance::Error::io(error)),
        },
        Ok(None) => Ok(None),
        Err(error) => Err(error),
    };
    // This lease only owns ticket/offer/reservation, never a running task.
    let cleanup = store.release_coordination_lock(guard).await;
    match (result, cleanup) {
        (Ok(claim), Ok(())) => Ok(claim),
        (Err(error), _) => Err(error),
        (Ok(Some(claim)), Err(error)) => {
            tracing::warn!(%error, "accepted task retained after dispatch ticket cleanup failed");
            Ok(Some(claim))
        }
        (Ok(None), Err(error)) => Err(error),
    }
}

#[derive(Default)]
pub(super) struct Placement {
    cursor: Option<Vec<u8>>,
    ticket_cursor: Option<Vec<u8>>,
    dependencies: BTreeMap<String, (Option<TaskState>, i64)>,
}

impl Placement {
    pub(super) fn observe_dependency(&mut self, key: &[u8], value: Option<&[u8]>, revision: i64) {
        let Some(id) = std::str::from_utf8(key)
            .ok()
            .and_then(|key| key.rsplit('/').next())
        else {
            return;
        };
        if let Some((state, old_revision)) = self.dependencies.get_mut(id) {
            if revision > *old_revision {
                *state = value
                    .and_then(|value| serde_json::from_slice::<TaskRecord>(value).ok())
                    .filter(|task| task.id == id)
                    .map(|task| task.state);
                *old_revision = revision;
            }
        }
    }
    async fn dependencies_ready(
        &mut self,
        publisher: &TaskStore,
        task: &TaskRecord,
    ) -> Result<bool, String> {
        if task.depends_on.len() > 10_000 {
            return Ok(true);
        } // authoritative executor handles long chains
        if self.dependencies.len() + task.depends_on.len() > 10_000 {
            self.dependencies.clear();
        }
        let missing: Vec<_> = task
            .depends_on
            .iter()
            .filter(|id| !self.dependencies.contains_key(*id))
            .cloned()
            .collect();
        for chunk in missing.chunks(64) {
            let observations = publisher
                .dispatch_dependency_observations(chunk)
                .await
                .map_err(|e| e.to_string())?;
            for (id, observation) in chunk.iter().zip(observations) {
                self.dependencies.insert(id.clone(), observation);
            }
        }
        // Missing/failed dependencies propagate failure even when another
        // dependency is still running, matching canonical claim semantics.
        let failed = task.depends_on.iter().any(|id| {
            matches!(
                self.dependencies.get(id),
                Some((None | Some(TaskState::Failed), _))
            )
        });
        Ok(failed
            || task.depends_on.iter().all(|id| {
                !matches!(
                    self.dependencies.get(id),
                    Some((Some(TaskState::Queued | TaskState::Running), _))
                )
            }))
    }
}

/// Bounded in-memory queue slice. No per-executor admission reads. Cached
/// predicates only choose a candidate; the receiving pool rechecks all of them.
pub(super) async fn place(
    publisher: &TaskStore,
    prefix: &str,
    cache: &mut Cache,
) -> Result<bool, String> {
    let ticket_prefix = prefix_key(prefix, "demand-controller/tickets");
    let ticket_start = cache
        .placement
        .ticket_cursor
        .clone()
        .unwrap_or_else(|| ticket_prefix.as_bytes().to_vec());
    let tickets: Vec<_> = cache
        .entries
        .range(ticket_start..)
        .take_while(|(key, _)| key.starts_with(ticket_prefix.as_bytes()))
        .filter(|(key, _)| {
            !cache.entries.contains_key(
                String::from_utf8_lossy(key)
                    .replace("/tickets/", "/offers/")
                    .as_bytes(),
            )
        })
        .take(64)
        .filter_map(|(key, entry)| {
            serde_json::from_slice::<Ticket>(&entry.value)
                .ok()
                .map(|ticket| (key.clone(), entry.clone(), ticket))
        })
        .collect();
    if tickets.is_empty() {
        cache.placement.ticket_cursor = None;
        cache.placement.cursor = None;
        return Ok(false);
    }
    let queue_prefix = prefix_key(prefix, "queue");
    let start = cache
        .placement
        .cursor
        .take()
        .unwrap_or_else(|| queue_prefix.as_bytes().to_vec());
    let queued: Vec<_> = cache
        .entries
        .range(start..)
        .take_while(|(key, _)| key.starts_with(queue_prefix.as_bytes()))
        .take(64)
        .map(|(key, entry)| (key.clone(), entry.clone()))
        .collect();
    let has_more = queued.len() == 64;
    for (key, entry) in queued {
        let mut next = key.clone();
        next.push(0);
        cache.placement.cursor = Some(next);
        let Ok(task) = serde_json::from_slice::<TaskRecord>(&entry.value) else {
            continue;
        };
        if task.state != TaskState::Queued
            || key != format!("{prefix}/queue/{}", task.id).as_bytes()
        {
            continue;
        }
        if cache
            .entries
            .contains_key(format!("{prefix}/demand-controller/reservations/{}", task.id).as_bytes())
        {
            continue;
        }
        let Some((ticket_key, ticket_entry, _)) = tickets
            .iter()
            .find(|(_, entry, ticket)| entry.lease_id > 0 && ticket.eligible(prefix, cache, &task))
        else {
            continue;
        };
        if !cache.placement.dependencies_ready(publisher, &task).await? {
            continue;
        }
        let ticket_key = std::str::from_utf8(ticket_key).map_err(|e| e.to_string())?;
        let offer = Offer {
            v: 1,
            task_id: task.id,
            nonce: lance_context_core::generate_id(),
        };
        let offer = assignment(prefix, ticket_key, &ticket_entry.value, &offer)?;
        if let Some(revision) = publisher
            .publish_assignment(&offer, ticket_entry.lease_id, entry.revision)
            .await
            .map_err(|e| e.to_string())?
        {
            // Publish into our cache immediately: don't offer this capacity twice
            // while waiting for our own writes to echo through the Watch stream.
            cache.apply(
                offer.offer_key.as_bytes(),
                Some(&offer.offer_value),
                revision,
            )?;
            cache.apply(
                offer.reservation_key.as_bytes(),
                Some(&offer.offer_value),
                revision,
            )?;
            metrics::counter!("master_controller_task_offers_total").increment(1);
        }
        return Ok(true);
    }
    if !has_more {
        cache.placement.cursor = None;
        if tickets.len() == 64 {
            let mut next = tickets.last().unwrap().0.clone();
            next.push(0);
            cache.placement.ticket_cursor = Some(next);
            return Ok(true);
        }
        cache.placement.ticket_cursor = None;
    }
    Ok(has_more)
}

pub(super) async fn recover(
    publisher: &TaskStore,
    prefix: &str,
    cache: &Cache,
) -> Result<(), String> {
    let running = prefix_key(prefix, "running");
    for (key, entry) in cache
        .entries
        .range(running.as_bytes().to_vec()..)
        .take_while(|(key, _)| key.starts_with(running.as_bytes()))
    {
        let Ok(task) = serde_json::from_slice::<TaskRecord>(&entry.value) else {
            continue;
        };
        if key.as_slice() != format!("{prefix}/running/{}", task.id).as_bytes()
            || task.state != TaskState::Running
        {
            continue;
        }
        if !cache
            .entries
            .contains_key(format!("{prefix}/claims/{}", task.id).as_bytes())
        {
            publisher
                .recover_observed_task(&task)
                .await
                .map_err(|e| e.to_string())?;
            break; // bounded; own Watch echo advances the recovery inventory
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{list, Policy};
    use super::*;
    use etcd_client::GetOptions;

    async fn setup() -> (
        tempfile::TempDir,
        MasterConfig,
        TaskStore,
        crate::task_store::CoordinationGuard,
    ) {
        let (dir, cfg, store) = super::super::tests::fixture().await;
        let leader = store
            .discovery_session(serde_json::to_vec(&Policy::from_config(&cfg)).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&leader).await.unwrap());
        (dir, cfg, store, leader)
    }

    async fn offer_for(
        store: &TaskStore,
        cfg: &MasterConfig,
        ticket: &crate::task_store::CoordinationGuard,
    ) -> DispatchAssignment {
        let key = ticket.key().replace("/tickets/", "/offers/");
        let response = store.etcd_client().clone().get(key, None).await.unwrap();
        let offer: Offer = serde_json::from_slice(response.kvs()[0].value()).unwrap();
        assignment(
            cfg.etcd.prefix(),
            ticket.key(),
            ticket.token().as_bytes(),
            &offer,
        )
        .unwrap()
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn assignment_is_exact_and_running_task_survives_leader_and_ticket_loss() {
        let (_dir, cfg, store, leader) = setup().await;
        store
            .enqueue(TaskKind::IndexId, "unrelated", vec![])
            .await
            .unwrap();
        let wanted = store
            .enqueue(TaskKind::MergeWal, "wanted", vec![])
            .await
            .unwrap();
        let ticket = store
            .create_dispatch_ticket(
                serde_json::to_vec(&Ticket::new(
                    &cfg,
                    "executor",
                    "merge",
                    TaskKinds::MERGE_WAL,
                    false,
                ))
                .unwrap(),
            )
            .await
            .unwrap();
        let publisher = store.discovery_publisher(&leader);
        let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        while place(&publisher, cfg.etcd.prefix(), &mut cache)
            .await
            .unwrap()
        {}
        let offer = offer_for(&store, &cfg, &ticket).await;
        assert_eq!(offer.task_id, wanted.id);
        assert!(
            store
                .etcd_client()
                .clone()
                .get(
                    lance_context_merge::target_lock_key(cfg.etcd.prefix(), "wanted"),
                    None
                )
                .await
                .unwrap()
                .kvs()
                .is_empty(),
            "offer must not own storage"
        );
        let admission = Arc::new(admission::Admission::default());
        let operation = admission.try_admit_named("merge").unwrap();
        let claim = store
            .claim_assignment(&offer, TaskKinds::MERGE_WAL, None, &operation)
            .await
            .unwrap()
            .unwrap();
        assert!(store
            .claim_assignment(&offer, TaskKinds::MERGE_WAL, None, &operation)
            .await
            .unwrap()
            .is_none());
        store.release_coordination_lock(ticket).await.unwrap();
        store.release_coordination_lock(leader).await.unwrap();
        assert!(!store
            .etcd_client()
            .clone()
            .get(format!("{}/claims/{}", cfg.etcd.prefix(), wanted.id), None)
            .await
            .unwrap()
            .kvs()
            .is_empty());
        store
            .finish(claim, Ok("survived controller loss".into()))
            .await
            .unwrap();
        assert_eq!(
            store.get(&wanted.id).await.unwrap().unwrap().state,
            TaskState::Done
        );
    }

    #[tokio::test]
    #[ignore = "requires exclusive loopback ETCD_TEST_ENDPOINTS"]
    async fn fifty_executors_do_not_repeat_blocked_candidate_reads_and_reach_tail() {
        let (_dir, cfg, store, leader) = setup().await;
        let publisher = store.discovery_publisher(&leader);
        for n in 0..96 {
            let target = format!("blocked-{n:03}");
            store
                .enqueue(TaskKind::MergeWal, &target, vec![])
                .await
                .unwrap();
            store
                .etcd_client()
                .clone()
                .put(
                    lance_context_merge::target_lock_key(cfg.etcd.prefix(), &target),
                    "productive-owner",
                    None,
                )
                .await
                .unwrap();
        }
        let tail = store
            .enqueue(TaskKind::MergeWal, "tail", vec![])
            .await
            .unwrap();
        let mut tickets = Vec::new();
        for n in 0..50 {
            tickets.push(
                store
                    .create_dispatch_ticket(
                        serde_json::to_vec(&Ticket::new(
                            &cfg,
                            &format!("executor-{n}"),
                            "merge",
                            TaskKinds::MERGE_WAL,
                            false,
                        ))
                        .unwrap(),
                    )
                    .await
                    .unwrap(),
            );
        }
        let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let before = super::super::tests::rpc_count(&endpoint, &["Range", "Txn"]).await;
        for _ in 0..10 {
            if !place(&publisher, cfg.etcd.prefix(), &mut cache)
                .await
                .unwrap()
            {
                break;
            }
        }
        let rpcs = super::super::tests::rpc_count(&endpoint, &["Range", "Txn"]).await - before;
        assert_eq!(
            rpcs, 1,
            "only one offer transaction; blocked queue is evaluated in memory"
        );
        let offers = store
            .etcd_client()
            .clone()
            .get(
                prefix_key(cfg.etcd.prefix(), "demand-controller/offers"),
                Some(GetOptions::new().with_prefix()),
            )
            .await
            .unwrap();
        assert_eq!(offers.kvs().len(), 1);
        assert_eq!(
            serde_json::from_slice::<Offer>(offers.kvs()[0].value())
                .unwrap()
                .task_id,
            tail.id
        );
        let before = super::super::tests::rpc_count(&endpoint, &["Range", "Txn"]).await;
        for _ in 0..4 {
            while place(&publisher, cfg.etcd.prefix(), &mut cache)
                .await
                .unwrap()
            {}
        }
        assert_eq!(
            super::super::tests::rpc_count(&endpoint, &["Range", "Txn"]).await - before,
            0
        );
        eprintln!("executors=50 blocked=96 productive_tail_offers=1 placement_range_txn=1 repeated_blocked_range_txn=0");
        for ticket in tickets {
            store.release_coordination_lock(ticket).await.unwrap();
        }
        store.release_coordination_lock(leader).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn expired_offer_and_stale_leader_cannot_claim_or_publish() {
        let (_dir, cfg, store, leader) = setup().await;
        let publisher = store.discovery_publisher(&leader);
        let task = store
            .enqueue(TaskKind::MergeWal, "table", vec![])
            .await
            .unwrap();
        let ticket = store
            .create_dispatch_ticket(
                serde_json::to_vec(&Ticket::new(
                    &cfg,
                    "executor",
                    "merge",
                    TaskKinds::MERGE_WAL,
                    false,
                ))
                .unwrap(),
            )
            .await
            .unwrap();
        let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        assert!(place(&publisher, cfg.etcd.prefix(), &mut cache)
            .await
            .unwrap());
        let offer = offer_for(&store, &cfg, &ticket).await;
        store.release_coordination_lock(ticket).await.unwrap();
        let admission = Arc::new(admission::Admission::default());
        let operation = admission.try_admit_named("merge").unwrap();
        assert!(store
            .claim_assignment(&offer, TaskKinds::MERGE_WAL, None, &operation)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            store.get(&task.id).await.unwrap().unwrap().state,
            TaskState::Queued
        );
        let ticket = store
            .create_dispatch_ticket(
                serde_json::to_vec(&Ticket::new(
                    &cfg,
                    "successor",
                    "merge",
                    TaskKinds::MERGE_WAL,
                    false,
                ))
                .unwrap(),
            )
            .await
            .unwrap();
        let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        store.release_coordination_lock(leader).await.unwrap();
        while place(&publisher, cfg.etcd.prefix(), &mut cache)
            .await
            .unwrap()
        {}
        assert!(store
            .etcd_client()
            .clone()
            .get(ticket.key().replace("/tickets/", "/offers/"), None)
            .await
            .unwrap()
            .kvs()
            .is_empty());
        store.release_coordination_lock(ticket).await.unwrap();
    }
    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn actual_receiver_waits_for_leader_and_orphan_recovery_preserves_live_owner() {
        let (_dir, cfg, store) = super::super::tests::fixture().await;
        let state = crate::state::MasterState::new(cfg.clone()).await.unwrap();
        let ticket = Ticket::new(&cfg, "receiver", "merge", TaskKinds::MERGE_WAL, false);
        let pool = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = pool.clone().acquire_owned().await.unwrap();
        let receiver_state = state.clone();
        let receiver = tokio::spawn(async move {
            let _permit = permit;
            let operation = receiver_state.admission.try_admit_named("merge").unwrap();
            claim(&receiver_state, &ticket, &operation, false)
                .await
                .unwrap()
                .unwrap()
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(pool.available_permits(), 0);
        assert!(
            !receiver.is_finished(),
            "no scanning fallback without a leader"
        );
        let task = store
            .enqueue(TaskKind::MergeWal, "watched", vec![])
            .await
            .unwrap();
        let controller = super::super::spawn(&state);
        let accepted = tokio::time::timeout(Duration::from_secs(10), receiver)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(accepted.task.id, task.id);
        // Lose leadership while a canonical execution remains live.
        controller.abort();
        assert!(controller.await.unwrap_err().is_cancelled());
        let successor_controller = super::super::spawn(&state);
        tokio::time::sleep(Duration::from_secs(7)).await;
        assert_eq!(
            store.get(&task.id).await.unwrap().unwrap().state,
            TaskState::Running
        );
        store.abandon_claim_for_test(accepted).await.unwrap();
        tokio::time::timeout(Duration::from_secs(8), async {
            loop {
                if store.get(&task.id).await.unwrap().unwrap().state == TaskState::Queued {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        successor_controller.abort();
        assert!(successor_controller.await.unwrap_err().is_cancelled());
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn heterogeneous_capacity_beyond_first_page_is_not_starved() {
        let (_dir, cfg, store, leader) = setup().await;
        let wanted = store
            .enqueue(TaskKind::MergeWal, "wanted", vec![])
            .await
            .unwrap();
        let mut guards = Vec::new();
        for n in 0..70 {
            guards.push(
                store
                    .create_dispatch_ticket(
                        serde_json::to_vec(&Ticket::new(
                            &cfg,
                            &format!("executor-{n}"),
                            "general",
                            TaskKinds::GENERAL,
                            false,
                        ))
                        .unwrap(),
                    )
                    .await
                    .unwrap(),
            );
        }
        guards.sort_by(|a, b| a.key().cmp(b.key()));
        let selected = guards.last().unwrap();
        let replacement = serde_json::to_vec(&Ticket::new(
            &cfg,
            "last",
            "merge",
            TaskKinds::MERGE_WAL,
            false,
        ))
        .unwrap();
        let response = store
            .etcd_client()
            .clone()
            .get(selected.key(), None)
            .await
            .unwrap();
        store
            .etcd_client()
            .clone()
            .put(
                selected.key(),
                replacement,
                Some(etcd_client::PutOptions::new().with_lease(response.kvs()[0].lease())),
            )
            .await
            .unwrap();
        let publisher = store.discovery_publisher(&leader);
        let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        let mut finished = false;
        for _ in 0..8 {
            if !place(&publisher, cfg.etcd.prefix(), &mut cache)
                .await
                .unwrap()
            {
                finished = true;
                break;
            }
        }
        assert!(finished);
        let response = store
            .etcd_client()
            .clone()
            .get(selected.key().replace("/tickets/", "/offers/"), None)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Offer>(response.kvs()[0].value())
                .unwrap()
                .task_id,
            wanted.id
        );
        for guard in guards {
            store.release_coordination_lock(guard).await.unwrap();
        }
        store.release_coordination_lock(leader).await.unwrap();
    }
    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn dependency_failure_is_not_hidden_by_waiting_and_old_watch_cannot_regress_point_read() {
        let (_dir, cfg, store, leader) = setup().await;
        let waiting = store
            .enqueue(TaskKind::IndexId, "waiting", vec![])
            .await
            .unwrap();
        let failed_child = store
            .enqueue(
                TaskKind::MergeWal,
                "child",
                vec![waiting.id.clone(), "missing".into()],
            )
            .await
            .unwrap();
        let mut placement = Placement::default();
        assert!(
            placement
                .dependencies_ready(&store, &failed_child)
                .await
                .unwrap(),
            "missing dependency must reach executor failure propagation"
        );
        let only_waiting = store
            .enqueue(TaskKind::MergeWal, "later", vec![waiting.id.clone()])
            .await
            .unwrap();
        assert!(!placement
            .dependencies_ready(&store, &only_waiting)
            .await
            .unwrap());
        let mut complete = waiting.clone();
        complete.state = TaskState::Done;
        let key = format!("{}/tasks/{}", cfg.etcd.prefix(), waiting.id);
        let revision = placement.dependencies[&waiting.id].1;
        placement.observe_dependency(
            key.as_bytes(),
            Some(&serde_json::to_vec(&complete).unwrap()),
            revision + 1,
        );
        placement.observe_dependency(
            key.as_bytes(),
            Some(&serde_json::to_vec(&waiting).unwrap()),
            revision,
        );
        assert!(placement
            .dependencies_ready(&store, &only_waiting)
            .await
            .unwrap());
        let op_admission = Arc::new(admission::Admission::default());
        let op = op_admission.try_admit_named("merge").unwrap();
        let ticket = store
            .create_dispatch_ticket(
                serde_json::to_vec(&Ticket::new(
                    &cfg,
                    "executor",
                    "merge",
                    TaskKinds::MERGE_WAL,
                    false,
                ))
                .unwrap(),
            )
            .await
            .unwrap();
        let publisher = store.discovery_publisher(&leader);
        let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
        while place(&publisher, cfg.etcd.prefix(), &mut cache)
            .await
            .unwrap()
        {}
        let offer = offer_for(&store, &cfg, &ticket).await;
        assert_eq!(offer.task_id, failed_child.id);
        assert!(store
            .claim_assignment(&offer, TaskKinds::MERGE_WAL, None, &op)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            store.get(&failed_child.id).await.unwrap().unwrap().state,
            TaskState::Failed
        );
        assert_eq!(
            store.get(&only_waiting.id).await.unwrap().unwrap().state,
            TaskState::Queued
        );
        store.release_coordination_lock(ticket).await.unwrap();
        store.release_coordination_lock(leader).await.unwrap();
    }
}
