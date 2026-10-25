//! Capacity-backed task offers. Only the elected controller selects candidates;
//! receiving pools retain all canonical dependency and ownership checks.
use super::{prefix_key, Cache};
use crate::{
    admission,
    config::MasterConfig,
    eligibility,
    state::MasterState,
    task_store::{CoordinationGuard, DispatchAssignment, TaskClaim, TaskKinds, TaskStore},
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
    target: String,
    nonce: String,
}

pub(super) fn offer_target(value: &[u8]) -> Result<String, String> {
    let offer: Offer = serde_json::from_slice(value).map_err(|e| e.to_string())?;
    if offer.v != 1 {
        return Err("unsupported dispatch offer".into());
    }
    Ok(offer.target)
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

/// A reserved pool slot in one dispatch phase. All production callers use the
/// fixed scheduler pool names. Unwinding/cancellation pays the gauge back.
struct PoolPhase(metrics::Gauge);
impl PoolPhase {
    fn new(pool: &str, phase: &'static str) -> Self {
        let gauge =
            metrics::gauge!("master_task_pool_in_use", "pool" => pool.to_owned(), "state" => phase);
        gauge.increment(1.0);
        Self(gauge)
    }
}
impl Drop for PoolPhase {
    fn drop(&mut self) {
        self.0.decrement(1.0);
    }
}

/// One session per scheduler pool, retained while tasks run. The ticket is
/// absent when no local permit is reserved, even though the lease/watch live on.
#[derive(Default)]
pub(crate) struct Receiver {
    session: Option<Session>,
    resuming: Arc<crate::merge_capacity::Resuming>,
}

struct Session {
    guard: CoordinationGuard,
    stream: etcd_client::WatchStream,
    pending: Option<Offer>,
    revision: i64,
}

impl Session {
    async fn open(store: &TaskStore, value: Vec<u8>) -> lance::Result<Self> {
        let guard = store.create_dispatch_ticket(value).await?;
        let offer_key = guard.key().replace("/tickets/", "/offers/");
        let setup = async {
            let snapshot = store
                .etcd_client()
                .clone()
                .get(offer_key.clone(), None)
                .await
                .map_err(|e| lance::Error::io(e.to_string()))?;
            let pending = snapshot
                .kvs()
                .first()
                .map(|kv| serde_json::from_slice::<Offer>(kv.value()))
                .transpose()
                .map_err(|e| lance::Error::io(e.to_string()))?;
            let revision = snapshot
                .header()
                .ok_or_else(|| lance::Error::io("missing offer snapshot revision"))?
                .revision();
            let stream = store
                .etcd_client()
                .clone()
                .watch(
                    offer_key,
                    Some(WatchOptions::new().with_start_revision(revision + 1)),
                )
                .await
                .map_err(|e| lance::Error::io(e.to_string()))?;
            Ok::<_, lance::Error>((stream, pending, revision))
        }
        .await;
        match setup {
            Ok((stream, pending, revision)) => Ok(Self {
                guard,
                stream,
                pending,
                revision,
            }),
            Err(error) => {
                let _ = store.release_coordination_lock(guard).await;
                Err(error)
            }
        }
    }
}

impl Receiver {
    pub(crate) fn new(resuming: Arc<crate::merge_capacity::Resuming>) -> Self {
        Self {
            session: None,
            resuming,
        }
    }

    pub(crate) async fn close(&mut self, store: &TaskStore) -> lance::Result<()> {
        if let Some(session) = self.session.take() {
            // No canonical task is attached to this lease.
            store.release_coordination_lock(session.guard).await?;
        }
        Ok(())
    }

    /// Caller holds pool/compaction permits throughout. A successful claim
    /// consumes the ticket atomically, then the next reserved slot reuses this
    /// lease and exact-key Watch. Faults invalidate only the dispatch session.
    pub(crate) async fn claim(
        &mut self,
        state: &Arc<MasterState>,
        ticket: &Ticket,
        operation: &admission::Operation,
        refresh_compaction: bool,
    ) -> lance::Result<Option<(TaskClaim, Duration)>> {
        let result = self
            .receive(state, ticket, operation, refresh_compaction)
            .await;
        if !matches!(&result, Ok(Some(_))) {
            // Also removes any rejected offer/reservation on this lease. Never
            // reuse a session after an ambiguous re-advertisement or claim.
            if let Err(error) = self.close(&state.task_store).await {
                tracing::warn!(%error, "dispatch session cleanup failed; lease will expire");
                if result.is_ok() {
                    // Preserve the scheduler's error pacing if withdrawing
                    // capacity failed, instead of reopening it as an empty poll.
                    return Err(error);
                }
            }
        }
        result
    }

    async fn receive(
        &mut self,
        state: &Arc<MasterState>,
        ticket: &Ticket,
        operation: &admission::Operation,
        refresh_compaction: bool,
    ) -> lance::Result<Option<(TaskClaim, Duration)>> {
        let store = &state.task_store;
        let mut phase = PoolPhase::new(&ticket.pool, "waiting_for_offer");
        operation.phase("waiting_for_offer");
        let wait_start = std::time::Instant::now();
        let value = serde_json::to_vec(ticket).map_err(|e| lance::Error::io(e.to_string()))?;
        if let Some(session) = &mut self.session {
            session.revision = store
                .rearm_dispatch_ticket(&mut session.guard, value.clone())
                .await?;
        } else {
            self.session = Some(Session::open(store, value.clone()).await?);
        }
        let session = self.session.as_mut().expect("dispatch session opened");
        let guard = &session.guard;
        let prefix = state.config.etcd.prefix();
        operation.claim_phase("waiting_for_controller_offer");
        let waiting = async {
            if let Some(offer) = session.pending.take() {
                return Ok(Some(offer));
            }
            while let Some(response) = session
                .stream
                .message()
                .await
                .map_err(|e| lance::Error::io(e.to_string()))?
            {
                if response.canceled() {
                    return Err(lance::Error::io("offer watch canceled"));
                }
                for event in response.events() {
                    if event.event_type() == etcd_client::EventType::Put {
                        let kv = event
                            .kv()
                            .ok_or_else(|| lance::Error::io("offer event missing value"))?;
                        // Ignore buffered events from an earlier slot advertisement.
                        if kv.mod_revision() >= session.revision {
                            return serde_json::from_slice::<Offer>(kv.value())
                                .map(Some)
                                .map_err(|e| lance::Error::io(e.to_string()));
                        }
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
        let received = tokio::select! {
            biased;
            _ = self.resuming.wait() => {
                metrics::counter!("master_controller_idle_reservation_yields_total", "pool" => ticket.pool.clone()).increment(1);
                Ok(None)
            },
            result = waiting => result,
            _ = stopped => Ok(None),
        };
        let claim_start = std::time::Instant::now();
        if matches!(&received, Ok(Some(_))) {
            metrics::histogram!("master_controller_offer_wait_seconds", "pool" => ticket.pool.clone())
            .record(wait_start.elapsed().as_secs_f64());
            drop(phase);
            phase = PoolPhase::new(&ticket.pool, "claiming");
            operation.phase("claiming");
        }
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
        drop(phase);
        let claim_elapsed = claim_start.elapsed();
        result.map(|claim| claim.map(|claim| (claim, claim_elapsed)))
    }
}

#[derive(Default)]
pub(super) struct Placement {
    cursor: Option<Vec<u8>>,
    ticket_cursor: Option<Vec<u8>>,
    dependencies: BTreeMap<String, DependencyEvaluation>,
    more_dependencies: bool,
}

/// One compact evaluation per queued dependency chain. Its dependency IDs are
/// bounded by that queued record, and removed with it; no global eviction sweep.
struct DependencyEvaluation {
    ids: Vec<String>,
    revision: i64,
    changed: i64,
    next: usize,
    waiting: bool,
    failed: bool,
}

impl DependencyEvaluation {
    fn new(ids: &[String]) -> Self {
        Self {
            ids: ids.to_vec(),
            revision: 0,
            changed: 0,
            next: 0,
            waiting: false,
            failed: false,
        }
    }
    fn reset(&mut self) {
        self.revision = 0;
        self.next = 0;
        self.waiting = false;
        self.failed = false;
    }
}

struct DependencyCheck {
    ready: bool,
    read: bool,
}

impl Placement {
    pub(super) fn forget_queue_entry(&mut self, key: &[u8]) {
        if let Some((parent, id)) = std::str::from_utf8(key)
            .ok()
            .and_then(|key| key.rsplit_once('/'))
        {
            if parent.ends_with("/queue") {
                self.dependencies.remove(id);
            }
        }
    }

    pub(super) fn observe_dependency(&mut self, key: &[u8], _value: Option<&[u8]>, revision: i64) {
        let Some(id) = std::str::from_utf8(key)
            .ok()
            .and_then(|key| key.rsplit('/').next())
        else {
            return;
        };
        // Only outstanding dependent tasks participate, not the fleet/history.
        // Retaining per-chain observations avoids cache eviction causing reads
        // on every unrelated queue/ownership event.
        for observation in self.dependencies.values_mut() {
            if revision > observation.revision && observation.ids.iter().any(|dep| dep == id) {
                observation.changed = observation.changed.max(revision);
            }
        }
    }

    async fn dependencies_ready(
        &mut self,
        publisher: &TaskStore,
        task: &TaskRecord,
    ) -> Result<DependencyCheck, String> {
        if task.depends_on.is_empty() {
            return Ok(DependencyCheck {
                ready: true,
                read: false,
            });
        }
        let observation = self
            .dependencies
            .entry(task.id.clone())
            .or_insert_with(|| DependencyEvaluation::new(&task.depends_on));
        if observation.next == observation.ids.len() && observation.changed > observation.revision {
            observation.reset();
        }
        let mut read = false;
        if observation.next < observation.ids.len() {
            // At most one 64-key transaction per placement step, all pages at
            // one revision. Long chains yield to other tasks and Watch events.
            let end = (observation.next + 64).min(observation.ids.len());
            let (states, revision) = publisher
                .dispatch_dependency_observations(
                    &observation.ids[observation.next..end],
                    observation.revision,
                )
                .await
                .map_err(|e| e.to_string())?;
            observation.revision = revision;
            observation.next = end;
            read = true;
            for state in states {
                match state {
                    Some(TaskState::Done) => {}
                    Some(TaskState::Queued | TaskState::Running) => observation.waiting = true,
                    None | Some(TaskState::Failed) => observation.failed = true,
                }
            }
            if observation.failed {
                observation.next = observation.ids.len();
            }
        }
        if observation.next == observation.ids.len() && observation.changed > observation.revision {
            observation.reset();
        }
        let complete = observation.next == observation.ids.len();
        self.more_dependencies |= !complete;
        Ok(DependencyCheck {
            ready: complete && (observation.failed || !observation.waiting),
            read,
        })
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
    // Advance by the raw bounded page, including malformed and occupied
    // tickets. Filtering first can hide every valid executor behind bad data
    // or traverse an unbounded occupied prefix in one scheduler step.
    let page: Vec<_> = cache
        .entries
        .range(ticket_start..)
        .take_while(|(key, _)| key.starts_with(ticket_prefix.as_bytes()))
        .take(64)
        .map(|(key, entry)| (key.clone(), entry.clone()))
        .collect();
    let next_ticket = (page.len() == 64).then(|| {
        let mut next = page.last().unwrap().0.clone();
        next.push(0);
        next
    });
    let mut tickets: Vec<_> = page
        .into_iter()
        .filter(|(key, _)| {
            !cache.entries.contains_key(
                String::from_utf8_lossy(key)
                    .replace("/tickets/", "/offers/")
                    .as_bytes(),
            )
        })
        .filter_map(|(key, entry)| {
            serde_json::from_slice::<Ticket>(&entry.value)
                .ok()
                .map(|ticket| (key, entry, ticket))
        })
        .collect();
    // A resident ticket reserves a slot whose target scope was resolved through
    // local_merge::targets. Prefer that existing capacity over a worker fan-out,
    // but keep the same bounded raw page and first eligible fallback. Stable
    // ordering preserves ticket rotation within each class; an ineligible local
    // ticket cannot delay a task that an ordinary pool can claim.
    tickets.sort_by_key(|(_, _, ticket)| ticket.resident_targets.is_none());
    if tickets.is_empty() {
        let more = next_ticket.is_some();
        cache.placement.ticket_cursor = next_ticket;
        cache.placement.cursor = None;
        return Ok(more || std::mem::take(&mut cache.placement.more_dependencies));
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
        if cache.offered_targets.contains_key(&task.target)
            || cache.entries.contains_key(
                format!("{prefix}/demand-controller/reservations/{}", task.id).as_bytes(),
            )
        {
            continue;
        }
        let Some((ticket_key, ticket_entry, _)) = tickets
            .iter()
            .find(|(_, entry, ticket)| entry.lease_id > 0 && ticket.eligible(prefix, cache, &task))
        else {
            continue;
        };
        let dependencies = cache.placement.dependencies_ready(publisher, &task).await?;
        if !dependencies.ready {
            if dependencies.read {
                return Ok(true);
            }
            continue;
        }
        let ticket_key = std::str::from_utf8(ticket_key).map_err(|e| e.to_string())?;
        let offer = Offer {
            v: 1,
            task_id: task.id,
            target: task.target,
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
        if let Some(next) = next_ticket {
            cache.placement.ticket_cursor = Some(next);
            return Ok(true);
        }
        cache.placement.ticket_cursor = None;
    }
    Ok(has_more || std::mem::take(&mut cache.placement.more_dependencies))
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
    async fn resident_capacity_precedes_fanout_with_immediate_eligible_fallback() {
        // The older ordinary ticket wins before resident preference is applied.
        // Local scope, draining and kind exclusions must still fall back without
        // reserving the task for an executor that cannot run it.
        for exclusion in ["none", "target", "draining", "kind"] {
            let (_dir, cfg, store, leader) = setup().await;
            let wanted = store
                .enqueue(TaskKind::MergeWal, "wanted", vec![])
                .await
                .unwrap();
            let ordinary = store
                .create_dispatch_ticket(
                    serde_json::to_vec(&Ticket::new(
                        &cfg,
                        "ordinary",
                        "merge",
                        TaskKinds::MERGE_WAL,
                        false,
                    ))
                    .unwrap(),
                )
                .await
                .unwrap();
            let mut local_cfg = cfg.clone();
            local_cfg.local_merge.master_local_merge_auto = true;
            local_cfg.local_merge.master_local_merge_targets = vec!["*".into()];
            local_cfg.maintenance.fence_legacy_maintenance = true;
            if exclusion == "target" {
                local_cfg.maintenance.wal_claim_targets = vec!["elsewhere".into()];
            }
            if exclusion == "draining" {
                local_cfg.merge_rollout.drain_targets = vec!["wanted".into()];
            }
            let kinds = if exclusion == "kind" {
                TaskKinds::COMPACT
            } else {
                TaskKinds::MERGE_WAL
            };
            let local = store
                .create_dispatch_ticket(
                    serde_json::to_vec(&Ticket::new(
                        &local_cfg,
                        "local",
                        "resident_merge",
                        kinds,
                        true,
                    ))
                    .unwrap(),
                )
                .await
                .unwrap();
            assert!(ordinary.key() < local.key());
            let publisher = store.discovery_publisher(&leader);
            let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
            for _ in 0..8 {
                if !place(&publisher, cfg.etcd.prefix(), &mut cache)
                    .await
                    .unwrap()
                {
                    break;
                }
            }
            let (chosen, rejected) = if exclusion == "none" {
                (&local, &ordinary)
            } else {
                (&ordinary, &local)
            };
            let offer = offer_for(&store, &cfg, chosen).await;
            assert_eq!(offer.task_id, wanted.id, "{exclusion}");
            assert!(
                store
                    .etcd_client()
                    .clone()
                    .get(rejected.key().replace("/tickets/", "/offers/"), None)
                    .await
                    .unwrap()
                    .kvs()
                    .is_empty(),
                "{exclusion}"
            );
            // A preferred offer remains advisory. An owner appearing after
            // placement must defeat the canonical claim on either route.
            store
                .etcd_client()
                .clone()
                .put(
                    lance_context_merge::target_lock_key(cfg.etcd.prefix(), "wanted"),
                    "concurrent-writer",
                    None,
                )
                .await
                .unwrap();
            let local_store = TaskStore::open(&local_cfg).await.unwrap();
            let (claim_store, targets) = if exclusion == "none" {
                (&local_store, Some(vec!["*".to_string()]))
            } else {
                (&store, None)
            };
            let admission = Arc::new(admission::Admission::default());
            let operation = admission.try_admit_named("merge").unwrap();
            assert!(
                claim_store
                    .claim_assignment(&offer, TaskKinds::MERGE_WAL, targets.as_deref(), &operation,)
                    .await
                    .unwrap()
                    .is_none(),
                "{exclusion}"
            );
            assert_eq!(
                store.get(&wanted.id).await.unwrap().unwrap().state,
                TaskState::Queued
            );
            store.release_coordination_lock(local).await.unwrap();
            store.release_coordination_lock(ordinary).await.unwrap();
            store.release_coordination_lock(leader).await.unwrap();
        }
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
        let receive_start = std::time::Instant::now();
        let receiver = tokio::spawn(async move {
            let _permit = permit;
            let operation = receiver_state.admission.try_admit_named("merge").unwrap();
            Receiver::default()
                .claim(&receiver_state, &ticket, &operation, false)
                .await
                .unwrap()
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if state
                    .admission
                    .status()
                    .operations
                    .values()
                    .any(|op| op.phase == "waiting_for_offer")
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
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
        let (accepted, claim_elapsed) = tokio::time::timeout(Duration::from_secs(10), receiver)
            .await
            .unwrap()
            .unwrap();
        assert!(
            receive_start.elapsed().saturating_sub(claim_elapsed) >= Duration::from_millis(100),
            "idle offer wait must be excluded from ownership claim duration"
        );
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
    #[ignore = "requires exclusive loopback ETCD_TEST_ENDPOINTS"]
    async fn receiver_reuses_lease_and_watch_and_withdraws_busy_capacity() {
        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").as_deref(), Ok("1"));
        let (_dir, cfg, store, leader) = setup().await;
        let state = crate::state::MasterState::new(cfg.clone()).await.unwrap();
        let publisher = store.discovery_publisher(&leader);
        let mut receiver = Receiver::default();
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let leases = super::super::tests::rpc_count(&endpoint, &["LeaseGrant"]).await;
        let watches = super::super::tests::rpc_count(&endpoint, &["Watch"]).await;
        let mut session_key = None;
        let mut last_claim = None;
        for n in 0..16 {
            // Exercise changes to the offered kind policy on the SAME session.
            let kinds = if n % 2 == 0 {
                TaskKinds::MERGE_WAL
            } else {
                TaskKinds::ANY
            };
            let ticket = Ticket::new(&cfg, "persistent", "general", kinds, false);
            let wanted = store
                .enqueue(TaskKind::MergeWal, &format!("reuse-{n}"), vec![])
                .await
                .unwrap();
            let operation = state.admission.try_admit_named("general").unwrap();
            let place_one = async {
                loop {
                    let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
                    if cache.entries.keys().any(|k| {
                        k.starts_with(
                            prefix_key(cfg.etcd.prefix(), "demand-controller/tickets").as_bytes(),
                        )
                    }) {
                        while place(&publisher, cfg.etcd.prefix(), &mut cache)
                            .await
                            .unwrap()
                        {}
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            };
            let (result, ()) = tokio::time::timeout(Duration::from_secs(5), async {
                tokio::join!(
                    receiver.claim(&state, &ticket, &operation, false),
                    place_one
                )
            })
            .await
            .unwrap();
            let (claim, _) = result.unwrap().unwrap();
            assert_eq!(claim.task.id, wanted.id);
            let session = receiver.session.as_ref().unwrap();
            let key = session.guard.key().to_string();
            assert_eq!(session_key.get_or_insert(key.clone()), &key);
            let policy: Ticket = serde_json::from_str(session.guard.token()).unwrap();
            assert_eq!(policy.kinds, kinds);
            // This receiver has no spare slot advertised while its task runs.
            for prefix in ["tickets", "offers", "reservations"] {
                assert!(store
                    .etcd_client()
                    .clone()
                    .get(
                        prefix_key(cfg.etcd.prefix(), &format!("demand-controller/{prefix}")),
                        Some(GetOptions::new().with_prefix())
                    )
                    .await
                    .unwrap()
                    .kvs()
                    .is_empty());
            }
            if let Some(previous) = last_claim.replace(claim) {
                store.finish(previous, Ok("finished".into())).await.unwrap();
            }
        }
        // 16 canonical execution leases + ONE pool lease. No per-task Watch.
        assert_eq!(
            super::super::tests::rpc_count(&endpoint, &["LeaseGrant"]).await - leases,
            17
        );
        assert_eq!(
            super::super::tests::rpc_count(&endpoint, &["Watch"]).await - watches,
            1
        );
        receiver.close(&store).await.unwrap();
        let last = last_claim.unwrap();
        assert_eq!(
            store.get(&last.task.id).await.unwrap().unwrap().state,
            TaskState::Running
        );
        store
            .finish(last, Ok("survived receiver shutdown".into()))
            .await
            .unwrap();
        store.release_coordination_lock(leader).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn receiver_recovers_canceled_watch_and_expired_capacity_without_reclaiming_tasks() {
        let (_dir, cfg, store, leader) = setup().await;
        let state = crate::state::MasterState::new(cfg.clone()).await.unwrap();
        let publisher = store.discovery_publisher(&leader);
        let ticket = Ticket::new(&cfg, "persistent", "merge", TaskKinds::MERGE_WAL, false);
        let mut receiver = Receiver::default();
        let operation = state.admission.try_admit_named("merge").unwrap();
        // Seed a session as though its prior slot were already consumed.
        let mut session = Session::open(&store, serde_json::to_vec(&ticket).unwrap())
            .await
            .unwrap();
        let created = session.stream.message().await.unwrap().unwrap();
        assert!(created.created());
        session.stream.cancel(created.watch_id()).await.unwrap();
        store
            .etcd_client()
            .clone()
            .delete(session.guard.key(), None)
            .await
            .unwrap();
        receiver.session = Some(session);
        assert!(tokio::time::timeout(
            Duration::from_secs(5),
            receiver.claim(&state, &ticket, &operation, false)
        )
        .await
        .unwrap()
        .is_err());
        assert!(receiver.session.is_none());
        for fault in [false, true] {
            let wanted = store
                .enqueue(TaskKind::MergeWal, &format!("after-fault-{fault}"), vec![])
                .await
                .unwrap();
            let place_one = async {
                loop {
                    let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
                    if cache.entries.keys().any(|k| {
                        k.starts_with(
                            prefix_key(cfg.etcd.prefix(), "demand-controller/tickets").as_bytes(),
                        )
                    }) {
                        while place(&publisher, cfg.etcd.prefix(), &mut cache)
                            .await
                            .unwrap()
                        {}
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            };
            let (result, ()) = tokio::time::timeout(Duration::from_secs(5), async {
                tokio::join!(
                    receiver.claim(&state, &ticket, &operation, false),
                    place_one
                )
            })
            .await
            .unwrap();
            let (claim, _) = result.unwrap().unwrap();
            assert_eq!(claim.task.id, wanted.id);
            if !fault {
                // Obtain the pool lease from a re-advertisement, then revoke it.
                let session = receiver.session.as_mut().unwrap();
                store
                    .rearm_dispatch_ticket(&mut session.guard, serde_json::to_vec(&ticket).unwrap())
                    .await
                    .unwrap();
                let kv = store
                    .etcd_client()
                    .clone()
                    .get(session.guard.key(), None)
                    .await
                    .unwrap();
                store
                    .etcd_client()
                    .clone()
                    .lease_revoke(kv.kvs()[0].lease())
                    .await
                    .unwrap();
                assert!(tokio::time::timeout(
                    Duration::from_secs(5),
                    receiver.claim(&state, &ticket, &operation, false)
                )
                .await
                .unwrap()
                .is_err());
                assert!(receiver.session.is_none());
                assert_eq!(
                    store.get(&wanted.id).await.unwrap().unwrap().state,
                    TaskState::Running
                );
            }
            store
                .finish(claim, Ok("survived capacity fault".into()))
                .await
                .unwrap();
        }
        receiver.close(&store).await.unwrap();
        store.release_coordination_lock(leader).await.unwrap();
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
    async fn malformed_or_occupied_ticket_page_does_not_hide_healthy_capacity() {
        for occupied in [false, true] {
            let (_dir, cfg, store, leader) = setup().await;
            let wanted = store
                .enqueue(TaskKind::MergeWal, "wanted", vec![])
                .await
                .unwrap();
            let value = serde_json::to_vec(&Ticket::new(
                &cfg,
                "healthy",
                "merge",
                TaskKinds::MERGE_WAL,
                false,
            ))
            .unwrap();
            let ticket = store.create_dispatch_ticket(value.clone()).await.unwrap();
            let mut client = store.etcd_client().clone();
            for n in 0..64 {
                // These sort before generated ticket IDs. They model a bad
                // writer or already occupied capacity at the start of a page.
                let key = format!(
                    "{}/demand-controller/tickets/!first-{n:03}",
                    cfg.etcd.prefix()
                );
                assert!(key.as_str() < ticket.key());
                client
                    .put(
                        key.clone(),
                        if occupied {
                            value.clone()
                        } else {
                            b"invalid".to_vec()
                        },
                        None,
                    )
                    .await
                    .unwrap();
                if occupied {
                    let offer = Offer {
                        v: 1,
                        task_id: format!("occupied-{n}"),
                        target: format!("other-{n}"),
                        nonce: "test".into(),
                    };
                    client
                        .put(
                            key.replace("/tickets/", "/offers/"),
                            serde_json::to_vec(&offer).unwrap(),
                            None,
                        )
                        .await
                        .unwrap();
                }
            }
            let publisher = store.discovery_publisher(&leader);
            let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
            assert!(place(&publisher, cfg.etcd.prefix(), &mut cache)
                .await
                .unwrap());
            assert!(
                cache.placement.ticket_cursor.is_some(),
                "first step must advance past the raw page"
            );
            let offer_key = ticket.key().replace("/tickets/", "/offers/");
            assert!(client
                .get(offer_key.clone(), None)
                .await
                .unwrap()
                .kvs()
                .is_empty());
            let mut complete = false;
            for _ in 0..8 {
                if !place(&publisher, cfg.etcd.prefix(), &mut cache)
                    .await
                    .unwrap()
                {
                    complete = true;
                    break;
                }
            }
            assert!(complete, "finite queue and ticket sweep must stop");
            let response = client.get(offer_key, None).await.unwrap();
            assert_eq!(
                serde_json::from_slice::<Offer>(response.kvs()[0].value())
                    .unwrap()
                    .task_id,
                wanted.id
            );
            store.release_coordination_lock(ticket).await.unwrap();
            store.release_coordination_lock(leader).await.unwrap();
        }
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
                .unwrap()
                .ready,
            "missing dependency must reach executor failure propagation"
        );
        let only_waiting = store
            .enqueue(TaskKind::MergeWal, "later", vec![waiting.id.clone()])
            .await
            .unwrap();
        assert!(
            !placement
                .dependencies_ready(&store, &only_waiting)
                .await
                .unwrap()
                .ready
        );
        let mut complete = waiting.clone();
        complete.state = TaskState::Done;
        let key = format!("{}/tasks/{}", cfg.etcd.prefix(), waiting.id);
        let revision = placement.dependencies[&only_waiting.id].revision;
        let changed = store
            .etcd_client()
            .clone()
            .put(key.clone(), serde_json::to_vec(&complete).unwrap(), None)
            .await
            .unwrap()
            .header()
            .unwrap()
            .revision();
        placement.observe_dependency(
            key.as_bytes(),
            Some(&serde_json::to_vec(&complete).unwrap()),
            changed,
        );
        placement.observe_dependency(
            key.as_bytes(),
            Some(&serde_json::to_vec(&waiting).unwrap()),
            revision,
        );
        assert!(
            placement
                .dependencies_ready(&store, &only_waiting)
                .await
                .unwrap()
                .ready
        );
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
    #[tokio::test]
    #[ignore = "requires exclusive loopback ETCD_TEST_ENDPOINTS"]
    async fn long_dependency_chain_yields_and_unrelated_changes_do_not_repeat_reads() {
        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").as_deref(), Ok("1"));
        let (_dir, cfg, store, leader) = setup().await;
        let mut record = store
            .enqueue(TaskKind::IndexId, "seed", vec![])
            .await
            .unwrap();
        let ids: Vec<_> = (0..10_001).map(|n| format!("dependency-{n:05}")).collect();
        for (page, ids) in ids.chunks(100).enumerate() {
            let ops = ids
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    record.id = id.clone();
                    record.state = if page == 0 && index == 0 {
                        TaskState::Running
                    } else {
                        TaskState::Done
                    };
                    etcd_client::TxnOp::put(
                        format!("{}/tasks/{id}", cfg.etcd.prefix()),
                        serde_json::to_vec(&record).unwrap(),
                        None,
                    )
                })
                .collect::<Vec<_>>();
            store
                .etcd_client()
                .clone()
                .txn(etcd_client::Txn::new().and_then(ops))
                .await
                .unwrap();
        }
        let task = store
            .enqueue(TaskKind::MergeWal, "long-chain", ids.clone())
            .await
            .unwrap();
        let cold = store
            .enqueue(TaskKind::MergeWal, "cold", vec![])
            .await
            .unwrap();
        let mut placement = Placement::default();
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let before = super::super::tests::rpc_count(&endpoint, &["Range", "Txn"]).await;
        for page in 0..157 {
            let check = placement.dependencies_ready(&store, &task).await.unwrap();
            assert!(!check.ready);
            assert!(check.read);
            assert_eq!(
                super::super::tests::rpc_count(&endpoint, &["Range", "Txn"]).await - before,
                page + 1,
                "each long-chain step may make only one bounded transaction"
            );
            let cold = placement.dependencies_ready(&store, &cold).await.unwrap();
            assert!(
                cold.ready && !cold.read,
                "unrelated work must be runnable between chunks"
            );
        }
        let revision = placement.dependencies[&task.id].revision;
        placement.observe_dependency(
            format!("{}/tasks/unrelated", cfg.etcd.prefix()).as_bytes(),
            None,
            revision + 1,
        );
        for _ in 0..10 {
            let check = placement.dependencies_ready(&store, &task).await.unwrap();
            assert!(!check.ready && !check.read);
        }
        assert_eq!(
            super::super::tests::rpc_count(&endpoint, &["Range", "Txn"]).await - before,
            157
        );
        record.id = ids[0].clone();
        record.state = TaskState::Done;
        let key = format!("{}/tasks/{}", cfg.etcd.prefix(), record.id);
        let response = store
            .etcd_client()
            .clone()
            .put(key.clone(), serde_json::to_vec(&record).unwrap(), None)
            .await
            .unwrap();
        placement.observe_dependency(key.as_bytes(), None, response.header().unwrap().revision());
        let mut ready = false;
        for _ in 0..157 {
            ready = placement
                .dependencies_ready(&store, &task)
                .await
                .unwrap()
                .ready;
        }
        assert!(ready);
        let snapshot = placement.dependencies[&task.id].revision;
        placement.observe_dependency(key.as_bytes(), None, snapshot - 1);
        let check = placement.dependencies_ready(&store, &task).await.unwrap();
        assert!(
            check.ready && !check.read,
            "old watch replay must not cause fresh reads"
        );
        placement.forget_queue_entry(format!("{}/queue/{}", cfg.etcd.prefix(), task.id).as_bytes());
        assert!(
            placement.dependencies.is_empty(),
            "summary lifetime follows queued task lifetime"
        );
        store.release_coordination_lock(leader).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn dependency_change_during_paged_snapshot_is_revisited_after_finite_sweep() {
        let (_dir, cfg, store, leader) = setup().await;
        let mut dep = store
            .enqueue(TaskKind::IndexId, "dependency", vec![])
            .await
            .unwrap();
        // Repeated IDs still exercise the multi-page snapshot without expanding
        // storage inventory. The authoritative semantics of duplicates agree.
        let task = store
            .enqueue(TaskKind::MergeWal, "child", vec![dep.id.clone(); 65])
            .await
            .unwrap();
        let mut placement = Placement::default();
        assert!(
            !placement
                .dependencies_ready(&store, &task)
                .await
                .unwrap()
                .ready
        );
        let first_revision = placement.dependencies[&task.id].revision;
        dep.state = TaskState::Done;
        let key = format!("{}/tasks/{}", cfg.etcd.prefix(), dep.id);
        let response = store
            .etcd_client()
            .clone()
            .put(key.clone(), serde_json::to_vec(&dep).unwrap(), None)
            .await
            .unwrap();
        let changed = response.header().unwrap().revision();
        assert!(changed > first_revision);
        placement.observe_dependency(key.as_bytes(), None, changed);
        assert!(
            !placement
                .dependencies_ready(&store, &task)
                .await
                .unwrap()
                .ready,
            "old snapshot finishes then refreshes, never mixes revisions"
        );
        assert_eq!(placement.dependencies[&task.id].next, 0);
        assert!(
            !placement
                .dependencies_ready(&store, &task)
                .await
                .unwrap()
                .ready
        );
        assert!(placement.dependencies[&task.id].revision >= changed);
        assert!(
            placement
                .dependencies_ready(&store, &task)
                .await
                .unwrap()
                .ready
        );
        store.release_coordination_lock(leader).await.unwrap();
    }
    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn pending_offer_coalesces_one_target_but_prepared_compaction_and_merge_overlap() {
        let (_dir, mut cfg, _store, leader) = setup().await;
        cfg.maintenance.compaction_prepare_targets = vec!["table".into()];
        let store = TaskStore::open_executor(&cfg).await.unwrap();
        let compact = store
            .enqueue(TaskKind::Compact, "table", vec![])
            .await
            .unwrap();
        let merge = store
            .enqueue(TaskKind::MergeWal, "table", vec![])
            .await
            .unwrap();
        let compact_ticket = store
            .create_dispatch_ticket(
                serde_json::to_vec(&Ticket::new(
                    &cfg,
                    "executor",
                    "compact",
                    TaskKinds::COMPACT,
                    false,
                ))
                .unwrap(),
            )
            .await
            .unwrap();
        let merge_ticket = store
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
        assert_eq!(cache.offered_targets.get("table"), Some(&1));
        assert!(
            store
                .etcd_client()
                .clone()
                .get(merge_ticket.key().replace("/tickets/", "/offers/"), None)
                .await
                .unwrap()
                .kvs()
                .is_empty(),
            "a pending offer must not fan out multiple competing claims on the same table"
        );
        let offered = offer_for(&store, &cfg, &compact_ticket).await;
        assert_eq!(offered.task_id, compact.id);
        let admission = Arc::new(admission::Admission::default());
        let operation = admission.try_admit_named("compact").unwrap();
        let mut stream = super::super::watch(store.etcd_client(), cfg.etcd.prefix(), &cache)
            .await
            .unwrap();
        let preparing = store
            .claim_assignment(&offered, TaskKinds::COMPACT, None, &operation)
            .await
            .unwrap()
            .unwrap();
        assert!(preparing.preparing_maintenance());
        super::super::tests::next_events(&mut stream, &mut cache, |cache| {
            !cache.offered_targets.contains_key("table")
                && cache.entries.contains_key(
                    lance_context_merge::execution_key(cfg.etcd.prefix(), "table")
                        .replace("/merge-executions/", "/compact-preparations/")
                        .as_bytes(),
                )
        })
        .await;
        while place(&publisher, cfg.etcd.prefix(), &mut cache)
            .await
            .unwrap()
        {}
        let offered = offer_for(&store, &cfg, &merge_ticket).await;
        assert_eq!(offered.task_id, merge.id);
        let merging = store
            .claim_assignment(&offered, TaskKinds::MERGE_WAL, None, &operation)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store.get(&compact.id).await.unwrap().unwrap().state,
            TaskState::Running
        );
        assert_eq!(
            store.get(&merge.id).await.unwrap().unwrap().state,
            TaskState::Running
        );
        store
            .finish(merging, Ok("merge coexists with preparation".into()))
            .await
            .unwrap();
        store
            .finish(preparing, Ok("preparation completed".into()))
            .await
            .unwrap();
        store
            .release_coordination_lock(compact_ticket)
            .await
            .unwrap();
        store.release_coordination_lock(merge_ticket).await.unwrap();
        store.release_coordination_lock(leader).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn stale_history_controller_reports_rejection_and_live_dependency_remains_protected() {
        let (_dir, mut cfg, store, leader) = setup().await;
        let mut done = Vec::new();
        for n in 0..3 {
            let task = store
                .enqueue(TaskKind::IndexId, &format!("done-{n}"), vec![])
                .await
                .unwrap();
            let claim = store
                .claim_next_of_kinds(TaskKinds::GENERAL)
                .await
                .unwrap()
                .unwrap();
            store
                .finish(claim, Ok("test terminal history".into()))
                .await
                .unwrap();
            done.push(task);
        }
        store
            .enqueue(TaskKind::MergeWal, "child", vec![done[0].id.clone()])
            .await
            .unwrap();
        cfg.task_history_limit = 1;
        let limited = TaskStore::open_executor(&cfg).await.unwrap();
        let stale = limited.discovery_publisher(&leader);
        store.release_coordination_lock(leader).await.unwrap();
        assert!(
            stale.prune_controller_history().await.is_err(),
            "lost fence must not be reported as successful pruning"
        );
        for task in &done {
            assert!(store.get(&task.id).await.unwrap().is_some());
        }
        let next = store
            .discovery_session(serde_json::to_vec(&Policy::from_config(&cfg)).unwrap())
            .await
            .unwrap();
        assert!(store.elect_discovery_leader(&next).await.unwrap());
        assert_eq!(
            limited
                .discovery_publisher(&next)
                .prune_controller_history()
                .await
                .unwrap(),
            1
        );
        assert!(
            store.get(&done[0].id).await.unwrap().is_some(),
            "queued dependency must be retained"
        );
        store.release_coordination_lock(next).await.unwrap();
    }
    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn lost_assignment_response_recovers_only_the_original_canonical_claim() {
        let (_dir, cfg, store, leader) = setup().await;
        let publisher = store.discovery_publisher(&leader);
        let admission = Arc::new(admission::Admission::default());
        let operation = admission.try_admit_named("merge").unwrap();
        for replace in [false, true] {
            let target = format!("lost-response-{replace}");
            let task = store
                .enqueue(TaskKind::MergeWal, &target, vec![])
                .await
                .unwrap();
            let ticket = store
                .create_dispatch_ticket(
                    serde_json::to_vec(&Ticket::new(
                        &cfg,
                        "receiver",
                        "merge",
                        TaskKinds::MERGE_WAL,
                        false,
                    ))
                    .unwrap(),
                )
                .await
                .unwrap();
            let mut cache = list(store.etcd_client(), cfg.etcd.prefix()).await.unwrap();
            while place(&publisher, cfg.etcd.prefix(), &mut cache)
                .await
                .unwrap()
            {}
            let offer = offer_for(&store, &cfg, &ticket).await;
            let key = lance_context_merge::target_lock_key(cfg.etcd.prefix(), &target);
            store.lose_next_claim_response_for_test(
                replace.then(|| (key.clone(), "other-live-owner".into())),
            );
            let result = store
                .claim_assignment(&offer, TaskKinds::MERGE_WAL, None, &operation)
                .await;
            if replace {
                assert!(
                    result.is_err(),
                    "a rejected transaction with lost response must not adopt a foreign owner"
                );
                assert_eq!(
                    store.get(&task.id).await.unwrap().unwrap().state,
                    TaskState::Queued
                );
                assert_eq!(
                    store
                        .etcd_client()
                        .clone()
                        .get(key, None)
                        .await
                        .unwrap()
                        .kvs()[0]
                        .value(),
                    b"other-live-owner"
                );
                store.release_coordination_lock(ticket).await.unwrap();
            } else {
                let accepted = result.unwrap().unwrap();
                assert_eq!(accepted.task.id, task.id);
                assert!(store
                    .etcd_client()
                    .clone()
                    .get(offer.offer_key.as_str(), None)
                    .await
                    .unwrap()
                    .kvs()
                    .is_empty());
                assert!(store
                    .etcd_client()
                    .clone()
                    .get(offer.reservation_key.as_str(), None)
                    .await
                    .unwrap()
                    .kvs()
                    .is_empty());
                store.release_coordination_lock(ticket).await.unwrap();
                let claim_key = format!("{}/claims/{}", cfg.etcd.prefix(), task.id);
                assert_eq!(
                    store
                        .etcd_client()
                        .clone()
                        .get(claim_key, None)
                        .await
                        .unwrap()
                        .kvs()
                        .len(),
                    1,
                    "ticket cleanup must not revoke the recovered canonical task"
                );
                store
                    .finish(
                        accepted,
                        Ok("adopted original claim after response loss".into()),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    store.get(&task.id).await.unwrap().unwrap().state,
                    TaskState::Done
                );
            }
        }
        store.release_coordination_lock(leader).await.unwrap();
    }
}
