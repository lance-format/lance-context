//! Bounded parallel preparation, followed by ordered atomic group publication.
use super::{executor::MERGE_BUDGET_BUSY, Result};
use crate::state::MasterState;
use futures::future::join_all;
use lance_context_core::{
    rollout_append::SealedWalProbe, GenericStore, GenericStoreOptions, MergeMemoryBudget,
    MergeReservation, PreparedShardMerge, RolloutStore, RolloutStoreOptions,
};
use std::{collections::VecDeque, sync::Arc};
use tokio::time::Instant;

enum Store {
    Rollout(RolloutStore),
    Generic(GenericStore),
}

struct Part {
    index: usize,
    store: Store,
    prepared: PreparedShardMerge,
}

async fn prepare(
    state: &Arc<MasterState>,
    target: &str,
    index: usize,
    probe: &SealedWalProbe,
    session: Arc<lance::session::Session>,
    budget: Arc<MergeMemoryBudget>,
    reservation: MergeReservation,
) -> Result<Option<Part>> {
    let shard = state.config.catchup.shards[index].clone();
    if !super::observe_stage("grouped_shard_probe", probe.has_sealed_wal(&shard))
        .await
        .map_err(|e| e.to_string())?
    {
        metrics::counter!("master_catchup_empty_shard_probes_total").increment(1);
        return Ok(None);
    }
    let state = state.clone();
    let target = target.to_owned();
    let (store, prepared) =
        PreparedShardMerge::prepare_owned(reservation, async move |reservation| {
            let config = &state.config.catchup;
            let (store, prepared) = if let Some(name) = target.strip_prefix("generic:") {
                let store = super::observe_stage(
                    "grouped_store_open",
                    GenericStore::open_existing(
                        &state.generic_uri(name),
                        GenericStoreOptions {
                            shard_id: Some(shard),
                            merge_after_generations: Some(0),
                            merge_max_generations: Some(config.merge_max_generations),
                            merge_max_bytes: Some(config.grouped_prepare_byte_cap()),
                            key_index_type: state.config.key_index_type,
                            merge_budget: Some(budget),
                            session: Some(session),
                            pending_generations_max: Some(0),
                            ..Default::default()
                        },
                    ),
                )
                .await?;
                let prepared = super::observe_stage(
                    "grouped_wal_prepare",
                    store.prepare_grouped_merge_reserved(reservation),
                )
                .await?;
                (Store::Generic(store), prepared)
            } else {
                let store = super::observe_stage(
                    "grouped_store_open",
                    RolloutStore::open_existing_with_options(
                        &state.rollout_uri(&target),
                        RolloutStoreOptions {
                            shard_id: Some(shard),
                            merge_after_generations: Some(0),
                            merge_max_generations: Some(config.merge_max_generations),
                            merge_max_bytes: Some(config.grouped_prepare_byte_cap()),
                            key_index_type: state.config.key_index_type,
                            merge_budget: Some(budget),
                            session: Some(session),
                            pending_generations_max: Some(0),
                            ..Default::default()
                        },
                    ),
                )
                .await?;
                let prepared = super::observe_stage(
                    "grouped_wal_prepare",
                    store.prepare_grouped_merge_reserved(reservation),
                )
                .await?;
                (Store::Rollout(store), prepared)
            };
            Ok((store, prepared))
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(prepared.map(|prepared| Part {
        index,
        store,
        prepared,
    }))
}

async fn commit(
    state: &Arc<MasterState>,
    target: &str,
    parts: Vec<Part>,
) -> Result<Vec<(usize, usize)>> {
    let mut parts = parts.into_iter();
    let Some(first) = parts.next() else {
        return Ok(Vec::new());
    };
    let mut outcomes = vec![(first.index, first.prepared.generation_count())];
    let mut group = vec![first.prepared];
    let mut store = first.store;
    for part in parts {
        outcomes.push((part.index, part.prepared.generation_count()));
        group.push(part.prepared);
        // Maintenance opens do not create resident writers. Only the publisher
        // needs to retain its base handle after the immutable reads have joined.
        drop(part.store);
    }
    let started = Instant::now();
    let cap = state.config.catchup.merge_max_bytes;
    if state.config.catchup.grouped_parallel_encoding {
        for shard in &mut group {
            shard.use_parallel_group_encoding();
        }
    }
    let demand =
        crate::demand_publish::CommittedMergeDemand::prepare_group(state, target, &group);
    let reclaimed = crate::demand_publish::CommittedMergeDemand::observe(demand, async {
        let reclaimed = match &mut store {
            Store::Rollout(store) => store.commit_prepared_group(group, cap).await,
            Store::Generic(store) => store.commit_prepared_group(group, cap).await,
        }
        .map_err(|e| e.to_string())?;
        if reclaimed != outcomes.iter().map(|(_, count)| count).sum::<usize>() {
            return Err("grouped merge returned an unexpected reclamation count".into());
        }
        Ok(reclaimed)
    })
    .await?;
    metrics::counter!("master_catchup_generations_reclaimed_total").increment(reclaimed as u64);
    metrics::histogram!("master_catchup_grouped_shards").record(outcomes.len() as f64);
    tracing::info!(%target, reclaimed, shards = outcomes.len(),
    commit_secs = started.elapsed().as_secs_f64(), "grouped WAL publication finished");
    for (index, reclaimed) in &outcomes {
        let shard = &state.config.catchup.shards[*index];
        if state.config.catchup.target.is_some() {
            tracing::info!(%target, %shard, reclaimed, grouped = true, "dedicated catch-up committed");
        } else {
            tracing::info!(%target, %shard, reclaimed, grouped = true, "local WAL merge committed");
        }
    }
    Ok(outcomes)
}

struct OrderedPart {
    index: usize,
    generations_remaining: usize,
    prepared: Option<Part>,
}

fn release_preparations(pending: &mut VecDeque<OrderedPart>) -> bool {
    let mut released = false;
    for entry in pending {
        released |= entry.prepared.take().is_some();
    }
    if released {
        metrics::counter!("master_catchup_grouped_memory_retries_total").increment(1);
    }
    released
}

async fn commit_ordered(
    state: &Arc<MasterState>,
    target: &str,
    parts: Vec<Part>,
    probe: &SealedWalProbe,
    session: Arc<lance::session::Session>,
    budget: Arc<MergeMemoryBudget>,
    deadline: Instant,
) -> Result<Vec<(usize, usize)>> {
    let allowance = state
        .config
        .catchup
        .grouped_prepare_byte_cap()
        .min(budget.limit());
    let mut pending: VecDeque<_> = parts
        .into_iter()
        .map(|part| OrderedPart {
            index: part.index,
            generations_remaining: part.prepared.generation_count()
                + part.prepared.unprepared_generation_count(),
            prepared: Some(part),
        })
        .collect();
    let mut outcomes = Vec::new();
    while !pending.is_empty() {
        if pending[0].prepared.is_none() {
            // Later immutable preparations may remain charged, but never block
            // on admission while retaining them. Only new reads obey the slice
            // boundary; an admitted publication always finishes and joins.
            if Instant::now() >= deadline || crate::local_merge::drain_requested(state, target) {
                break;
            }
            let reservation = match budget.try_reserve(allowance) {
                Some(reservation) => reservation,
                None => {
                    release_preparations(&mut pending);
                    super::observe_stage("grouped_budget_wait", budget.reserve(allowance)).await
                }
            };
            if Instant::now() >= deadline || crate::local_merge::drain_requested(state, target) {
                break;
            }
            let index = pending[0].index;
            let prepared = prepare(
                state,
                target,
                index,
                probe,
                session.clone(),
                budget.clone(),
                reservation,
            )
            .await;
            let part = match prepared {
                Err(error) if error.contains(MERGE_BUDGET_BUSY) => {
                    if release_preparations(&mut pending) {
                        continue;
                    }
                    return Err(error);
                }
                result => result?,
            };
            match part {
                Some(part) => pending[0].prepared = Some(part),
                None => {
                    pending.pop_front();
                    outcomes.push((index, 0));
                    continue;
                }
            }
        }
        let mut group = Vec::new();
        let mut continuation = None;
        while pending
            .front()
            .is_some_and(|entry| entry.prepared.is_some())
        {
            let mut entry = pending.pop_front().unwrap();
            let part = entry.prepared.take().unwrap();
            let count = part.prepared.generation_count();
            if count < entry.generations_remaining {
                entry.generations_remaining -= count;
                continuation = Some(entry);
            }
            group.push(part);
            if continuation.is_some() {
                break;
            }
        }
        outcomes.extend(commit(state, target, group).await?);
        if let Some(entry) = continuation {
            // Finish this shard's captured suffix before any later shard. The
            // fixed count bounds work even if ingestion keeps appending WAL.
            pending.push_front(entry);
        }
    }
    Ok(outcomes)
}

pub(super) async fn run(
    state: &Arc<MasterState>,
    target: &str,
    selected: &[usize],
    probe: &SealedWalProbe,
    session: Arc<lance::session::Session>,
    budget: Arc<MergeMemoryBudget>,
    deadline: Instant,
) -> Result<Vec<(usize, usize)>> {
    let config = &state.config.catchup;
    // Admission and the store's read cap must agree: the core requires the full
    // allowance before opening readers. Publication keeps the aggregate cap.
    let allowance = config.grouped_prepare_byte_cap().min(budget.limit());
    let mut remaining: VecDeque<_> = selected.iter().copied().collect();
    let mut outcomes = Vec::new();
    while let Some(first) = remaining.pop_front() {
        if Instant::now() >= deadline || crate::local_merge::drain_requested(state, target) {
            break;
        }
        // No previous wave holds decoded buffers at this blocking admission.
        let reservation =
            super::observe_stage("grouped_budget_wait", budget.reserve(allowance)).await;
        if Instant::now() >= deadline || crate::local_merge::drain_requested(state, target) {
            break;
        }
        let mut wave = vec![(first, reservation)];
        let mut indices = Vec::new();
        let mut results = Vec::new();
        loop {
            while indices.len() + wave.len() < config.grouped_shards && !remaining.is_empty() {
                let Some(reservation) = budget.try_reserve(allowance) else {
                    break;
                };
                wave.push((remaining.pop_front().unwrap(), reservation));
            }
            if wave.is_empty() {
                break;
            }
            indices.extend(wave.iter().map(|(index, _)| *index));
            // Join every admitted read, including on failure. Successful reads
            // release unused decode allowance only after native children exit.
            results.extend(
                super::observe_stage(
                    "grouped_prepare_wave",
                    join_all(wave.drain(..).map(|(index, reservation)| {
                        prepare(
                            state,
                            target,
                            index,
                            probe,
                            session.clone(),
                            budget.clone(),
                            reservation,
                        )
                    })),
                )
                .await,
            );
            if results.iter().any(Result::is_err)
                || Instant::now() >= deadline
                || crate::local_merge::drain_requested(state, target)
            {
                break;
            }
            // Refill using only immediately available capacity. This lets small
            // joined results share one publication even if their original
            // reservations could not all coexist. Never wait with results held.
        }
        let first_storage_error = results.iter().enumerate().find_map(|(index, result)| {
            result
                .as_ref()
                .err()
                .filter(|e| !e.contains(MERGE_BUDGET_BUSY))
                .map(|e| (index, e.clone()))
        });
        if results
            .iter()
            .any(|r| r.as_ref().is_err_and(|e| e.contains(MERGE_BUDGET_BUSY)))
        {
            // An indivisible generation may outgrow its initial reservation.
            // Release the whole speculative wave, then retry in order with no
            // later batch holding capacity needed by the current publisher.
            drop(results);
            metrics::counter!("master_catchup_grouped_memory_retries_total").increment(1);
            for (position, index) in indices.into_iter().enumerate() {
                if Instant::now() >= deadline || crate::local_merge::drain_requested(state, target)
                {
                    break;
                }
                if let Some((failed, error)) = &first_storage_error {
                    if position == *failed {
                        return Err(error.clone());
                    }
                }
                let reservation =
                    super::observe_stage("grouped_budget_wait", budget.reserve(allowance)).await;
                match prepare(
                    state,
                    target,
                    index,
                    probe,
                    session.clone(),
                    budget.clone(),
                    reservation,
                )
                .await?
                {
                    Some(part) => outcomes.extend(
                        commit_ordered(
                            state,
                            target,
                            vec![part],
                            probe,
                            session.clone(),
                            budget.clone(),
                            deadline,
                        )
                        .await?,
                    ),
                    None => outcomes.push((index, 0)),
                }
            }
        } else {
            let mut parts = Vec::new();
            let mut failed = None;
            for (index, result) in indices.into_iter().zip(results) {
                match result {
                    Ok(Some(part)) => parts.push(part),
                    Ok(None) => outcomes.push((index, 0)),
                    Err(error) => {
                        failed = Some(error);
                        break;
                    }
                }
            }
            // Preserve ordered error delivery: preceding prepared shards may
            // finish; later preparations drop without publication.
            outcomes.extend(
                commit_ordered(
                    state,
                    target,
                    parts,
                    probe,
                    session.clone(),
                    budget.clone(),
                    deadline,
                )
                .await?,
            );
            if let Some(error) = failed {
                return Err(error);
            }
        }
    }
    Ok(outcomes)
}

#[cfg(test)]
mod tests {
    use super::super::CatchupConfig;

    #[test]
    fn grouped_admission_is_opt_in_bounded_and_rejects_conflicting_modes() {
        let mut config = CatchupConfig::default();
        assert!(config.grouped_targets.is_empty());
        assert!(config.validate().is_ok());
        config.grouped_parallel_encoding = true;
        assert!(config.validate().is_err());
        config.grouped_parallel_encoding = false;
        config.grouped_targets = vec!["bench".into()];
        for width in [0, 33] {
            config.grouped_shards = width;
            assert!(config.validate().is_err());
        }
        config.grouped_shards = 4;
        for targets in [vec!["*".into()], vec!["bench".into(), "".into()]] {
            config.grouped_targets = targets;
            assert!(config.validate().is_err());
        }
        config.grouped_targets = vec!["bench".into()];
        config.staging_targets = vec!["bench".into()];
        assert!(config.validate().is_err());
        config.staging_targets.clear();
        config.parallel_upsert_io_targets = vec!["bench".into()];
        assert!(config.validate().is_err());
        config.parallel_upsert_io_targets.clear();
        config.bulk_key_delete_targets = vec!["bench".into()];
        assert!(config.validate().is_err());
        config.bulk_key_delete_targets.clear();
        config.merge_max_bytes = 0;
        assert!(config.validate().is_err());
        config.merge_max_bytes = 1024;
        config.merge_memory_bytes = 512;
        assert!(config.validate().is_err());
        config.merge_memory_bytes = 1024;
        assert!(config.validate().is_ok());
        assert_eq!(config.grouped_prepare_byte_cap(), 1024);
        config.grouped_prepare_max_bytes = 256;
        assert!(config.validate().is_ok());
        assert_eq!(config.grouped_prepare_byte_cap(), 256);
        config.grouped_prepare_max_bytes = 1025;
        assert!(config.validate().is_err());
        config.grouped_prepare_max_bytes = 0;
        assert_eq!(config.grouped_prepare_byte_cap(), 1024);
        config.grouped_targets = vec![String::new()];
        assert!(
            config.validate().is_ok(),
            "explicit empty override disables grouping"
        );
    }

    #[test]
    fn grouped_job_override_clears_inherited_opt_in_and_round_trips_bounds() {
        use crate::config::MasterConfig;
        use clap::Parser;
        use serde_json::json;
        let record = serde_json::from_value(json!({
            "target":"hot", "job":"job", "job_uid":"job-uid", "slot":0,
            "attempt":1, "active":true, "reason":"test", "pending_at_admission":1000,
            "created_at_ms":0, "finished_at_ms":null, "consecutive_failures":0,
            "needs_attention":false, "next_retry_ms":0, "outcome":null
        }))
        .unwrap();
        for (targets, prepare_cap) in [
            (vec![], 0),
            (vec!["hot".to_owned(), "generic:other".to_owned()], 0),
            (vec![], 1 << 20),
            (vec!["hot".to_owned(), "generic:other".to_owned()], 1 << 20),
        ] {
            let mut config = MasterConfig::parse_from(["master"]);
            config.catchup.grouped_targets = targets.clone();
            config.catchup.grouped_shards = 3;
            config.catchup.grouped_prepare_max_bytes = prepare_cap;
            config.catchup.grouped_parallel_encoding = !targets.is_empty();
            let job = super::super::kubernetes::render_job(
                &config,
                &record,
                json!({"containers":[{"name":"executor", "env":[
                    {"name":"CATCHUP_GROUPED_TARGETS", "value":"inherited"},
                    {"name":"CATCHUP_GROUPED_SHARDS", "value":"32"},
                    {"name":"CATCHUP_GROUPED_PREPARE_MAX_BYTES", "value":"999999999"},
                    {"name":"CATCHUP_GROUPED_PARALLEL_ENCODING", "value":"true"}
                ]}]}),
            );
            let env = job["spec"]["template"]["spec"]["containers"][0]["env"]
                .as_array()
                .unwrap();
            let entries = |name: &str| env.iter().filter(|v| v["name"] == name).collect::<Vec<_>>();
            let rendered_targets = entries("CATCHUP_GROUPED_TARGETS");
            let rendered_shards = entries("CATCHUP_GROUPED_SHARDS");
            let rendered_prepare = entries("CATCHUP_GROUPED_PREPARE_MAX_BYTES");
            let rendered_encoding = entries("CATCHUP_GROUPED_PARALLEL_ENCODING");
            assert_eq!(rendered_targets.len(), 1);
            assert_eq!(rendered_shards.len(), 1);
            assert_eq!(rendered_prepare.len(), 1);
            assert_eq!(rendered_encoding.len(), 1);
            let target_value = rendered_targets[0]["value"].as_str().unwrap();
            let shard_value = rendered_shards[0]["value"].as_str().unwrap();
            let prepare_value = rendered_prepare[0]["value"].as_str().unwrap();
            let encoding_value = rendered_encoding[0]["value"].as_str().unwrap();
            assert_eq!(encoding_value, (!targets.is_empty()).to_string());
            assert_eq!(target_value, targets.join(","));
            assert_eq!(shard_value, "3");
            assert_eq!(prepare_value, prepare_cap.to_string());
            let parsed = MasterConfig::try_parse_from([
                "master",
                "--grouped-targets",
                target_value,
                "--grouped-shards",
                shard_value,
                "--grouped-prepare-max-bytes",
                prepare_value,
                "--grouped-parallel-encoding",
                encoding_value,
                "--target",
                "hot",
                "--job-name",
                "job",
                "--shards",
                "writer0",
            ])
            .unwrap();
            parsed.catchup.validate().unwrap();
            assert_eq!(parsed.catchup.grouped_shards, 3);
            assert_eq!(parsed.catchup.grouped_prepare_max_bytes, prepare_cap);
            assert_eq!(
                parsed.catchup.grouped_parallel_encoding,
                !targets.is_empty()
            );
            assert_eq!(
                parsed.catchup.grouped_targets.iter().any(|v| v == "hot"),
                !targets.is_empty()
            );
            assert!(!parsed
                .catchup
                .grouped_targets
                .iter()
                .any(|v| v == "inherited"));
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::super::{execute, store::Inventory, tests::fixture, ExecuteOutcome};
    use lance_context_core::{
        ColumnSpec, ColumnType, GenericStore, GenericStoreOptions, SchemaSpec,
    };
    use serde_json::json;

    #[tokio::test]
    #[ignore = "requires isolated loopback ETCD_TEST_ENDPOINTS"]
    async fn grouped_partial_shards_preserve_configured_order_across_continuations() {
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").expect("isolated etcd required");
        assert!(endpoints
            .split(',')
            .all(|e| e.starts_with("http://127.0.0.1:")));
        for (prepare_cap, budget_bytes, tail_bytes, split_slice, rotated_job) in [
            (0, 32 << 20, 256 << 10, false, false),
            (128 << 10, 32 << 20, 256 << 10, false, false),
            // This tail fits alone but cannot grow while retaining a later head.
            (128 << 10, 1 << 20, 900 << 10, false, false),
            (128 << 10, 32 << 20, 256 << 10, true, false),
            // Both continuations start with identical remaining WAL, then use
            // the next durable attempt's rendered order with different caps.
            (0, 32 << 20, 256 << 10, true, true),
            (128 << 10, 32 << 20, 256 << 10, true, true),
        ] {
            let (_dir, state) = fixture().await.unwrap();
            let target = "generic:hot";
            let uri = state.generic_uri("hot");
            let schema = SchemaSpec::new(vec![
                (
                    "id".into(),
                    ColumnSpec::required(ColumnType::String { large: false }),
                ),
                (
                    "text".into(),
                    ColumnSpec::new(ColumnType::String { large: true }),
                ),
            ]);
            let mut writers = Vec::new();
            for shard in [0, 19] {
                let writer = GenericStore::open(
                    &uri,
                    schema.clone(),
                    GenericStoreOptions {
                        shard_id: Some(format!("worker-{shard}")),
                        merge_after_generations: Some(0),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
                for generation in 0..2 {
                    let mut rows = vec![json!({
                        "id": format!("only-{shard}-{generation}"),
                        "text": "x".repeat(if (shard, generation) == (0, 1) {
                            tail_bytes
                        } else {
                            256 << 10
                        }),
                    })
                    .as_object()
                    .unwrap()
                    .clone()];
                    // The later shard's tail deliberately does not touch this
                    // key: a shared key in every generation masks round-order bugs.
                    if (shard, generation) == (0, 1) || (shard, generation) == (19, 0) {
                        rows.push(
                            json!({
                                "id": "overlap",
                                "text": if shard == 0 { "earlier-tail" } else { "later-head" },
                            })
                            .as_object()
                            .unwrap()
                            .clone(),
                        );
                    }
                    writer.add(&rows).await.unwrap();
                    writer.flush().await.unwrap();
                }
                writers.push(writer);
            }
            let mut cfg = state.config.clone();
            cfg.catchup.enabled = false;
            cfg.catchup.shards = (0..20).map(|i| format!("worker-{i}")).collect();
            cfg.catchup.grouped_targets = vec![target.into()];
            cfg.catchup.grouped_shards = 20;
            cfg.catchup.merge_max_bytes = (8 << 20).min(budget_bytes);
            cfg.catchup.grouped_prepare_max_bytes = prepare_cap;
            cfg.catchup.merge_memory_bytes = budget_bytes;
            cfg.merge_rollout.owned_targets.push(target.into());
            cfg.catchup.target = Some(target.into());
            cfg.catchup.job_name = Inventory::new(&state)
                .reserve(target, "grouped-partial-order", 4, 100)
                .await
                .unwrap()
                .job;
            if split_slice {
                use lance_context_core::{
                    merge_write_scope::MergeWriteScope, rollout_append::SealedWalProbe,
                    MergeMemoryBudget, RolloutStore,
                };
                use std::sync::Arc;
                let mut slice_cfg = cfg.clone();
                if rotated_job {
                    slice_cfg.catchup.grouped_prepare_max_bytes = 128 << 10;
                }
                let slice_prepare_cap = slice_cfg.catchup.grouped_prepare_max_bytes;
                let state = crate::state::MasterState::new(slice_cfg).await.unwrap();
                let budget = MergeMemoryBudget::new(budget_bytes);
                let session = RolloutStore::build_session(1 << 20, 1 << 20);
                let probe = SealedWalProbe::open(&uri, Some(session.clone()))
                    .await
                    .unwrap();
                let scope = MergeWriteScope::with_authorizer(Arc::new(Allow));
                let outcomes = scope
                    .run(async {
                        let mut parts = Vec::new();
                        for index in [0, 19] {
                            let reservation = budget.reserve(slice_prepare_cap).await;
                            let part = super::prepare(
                                &state,
                                target,
                                index,
                                &probe,
                                session.clone(),
                                budget.clone(),
                                reservation,
                            )
                            .await
                            .unwrap()
                            .unwrap();
                            assert_eq!(part.prepared.unprepared_generation_count(), 1);
                            parts.push(part);
                        }
                        // Finish admitted work after the slice, but defer the
                        // later head behind the unprepared earlier suffix.
                        super::commit_ordered(
                            &state,
                            target,
                            parts,
                            &probe,
                            session,
                            budget.clone(),
                            tokio::time::Instant::now(),
                        )
                        .await
                        .unwrap()
                    })
                    .await;
                scope.drain().await;
                assert_eq!(outcomes, vec![(0, 1)]);
                assert_eq!(budget.reserved(), 0);
                assert_eq!(
                    writers[1]
                        .own_shard_watermark()
                        .await
                        .unwrap()
                        .unwrap()
                        .pending_generations,
                    2
                );
            }
            if rotated_job {
                let inventory = Inventory::new(&state);
                let first = inventory.get(target).await.unwrap().unwrap();
                assert_eq!(first.attempt, 1);
                inventory
                    .complete(&first, super::super::outcome::JobOutcome::Succeeded, 101)
                    .await
                    .unwrap();
                // Advance the supplied admission clock beyond the normal
                // successful-attempt cooldown; do not clear failure debt.
                cfg.catchup.job_name = inventory
                    .reserve(target, "grouped-rotated-continuation", 3, 60_102)
                    .await
                    .unwrap()
                    .job;
                let second = inventory.get(target).await.unwrap().unwrap();
                assert_eq!(second.attempt, 2);
                assert_ne!(first.job, second.job);
                let job = super::super::kubernetes::render_job(
                    &cfg,
                    &second,
                    json!({"containers":[{"name":"executor", "env":[]}]}),
                );
                let env = job["spec"]["template"]["spec"]["containers"][0]["env"]
                    .as_array()
                    .unwrap();
                cfg.catchup.shards = env
                    .iter()
                    .find(|entry| entry["name"] == "CATCHUP_SHARDS")
                    .unwrap()["value"]
                    .as_str()
                    .unwrap()
                    .split(',')
                    .map(str::to_owned)
                    .collect();
                assert_eq!(cfg.catchup.shards.first().unwrap(), "worker-1");
                assert_eq!(cfg.catchup.shards.last().unwrap(), "worker-0");
            }
            assert_eq!(
                tokio::time::timeout(std::time::Duration::from_secs(30), execute(cfg, target))
                    .await
                    .expect("ordered continuation must not wait on retained buffers")
                    .unwrap(),
                ExecuteOutcome::Completed
            );
            let reader = GenericStore::open_existing(&uri, GenericStoreOptions::default())
                .await
                .unwrap();
            assert_eq!(reader.pending_wal_generations().await.unwrap(), 0);
            let rows = reader.list(None, None).await.unwrap();
            assert_eq!(rows.len(), 5);
            if rotated_job {
                // Rotation deliberately puts worker-0 last. Matching this
                // existing per-attempt contract does not establish a stable
                // cross-attempt or global event-time winner.
                assert_eq!(
                    rows.iter().find(|r| r["id"] == "overlap").unwrap()["text"],
                    "earlier-tail",
                    "preparation cap {prepare_cap} changed rendered attempt order"
                );
            } else {
                assert_eq!(
                    rows.iter().find(|r| r["id"] == "overlap").unwrap()["text"],
                    "later-head",
                    "preparation cap {prepare_cap} changed cross-shard upsert precedence"
                );
            }
            drop(writers);
        }
    }

    #[tokio::test]
    #[ignore = "requires isolated loopback ETCD_TEST_ENDPOINTS"]
    async fn grouped_native_executor_preserves_order_and_handles_tight_and_oversized_budgets() {
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").expect("isolated etcd required");
        assert!(endpoints
            .split(',')
            .all(|e| e.starts_with("http://127.0.0.1:")));
        for (
            budget_bytes,
            payload_bytes,
            expected_commits,
            cap,
            width,
            all_shards,
            prepare_cap,
            parallel,
        ) in [
            (32 << 20, 64, 2, 1 << 20, 4, false, 0, false),
            (1 << 20, 64, 3, 1 << 20, 4, false, 0, false),
            (4 << 20, 1536 << 10, 3, 1 << 20, 4, false, 0, false),
            // Sixteen initial 8-MiB reservations exactly fill 128 MiB.
            // Joined reads must free scratch and admit the last four shards,
            // allowing all twenty to publish once without reducing headroom.
            (128 << 20, 64, 1, 8 << 20, 20, true, 0, false),
            // Smaller reads still combine all twenty shards under the larger
            // publication cap. Indivisible generations can grow or retry in
            // order under the same total budget, without dropping any rows.
            (32 << 20, 64, 1, 8 << 20, 20, true, 1 << 20, false),
            (4 << 20, 1536 << 10, 3, 1 << 20, 4, false, 128 << 10, false),
            // The same native owner and one publication, with two encoded files.
            (256 << 20, 64, 1, 8 << 20, 20, true, 0, true),
            // A requested second encoder must not block a full budget.
            (4 << 20, 1536 << 10, 3, 1 << 20, 4, false, 128 << 10, true),
        ] {
            let (_dir, state) = fixture().await.unwrap();
            let target = "generic:hot";
            let uri = state.generic_uri("hot");
            let schema = SchemaSpec::new(vec![
                (
                    "id".into(),
                    ColumnSpec::required(ColumnType::String { large: false }),
                ),
                (
                    "text".into(),
                    ColumnSpec::new(ColumnType::String { large: true }),
                ),
            ]);
            let mut writers = Vec::new();
            // A later input shard must win in grouped and serial fallback modes.
            let active_shards: Vec<_> = if all_shards {
                (0..20).collect()
            } else {
                vec![0, 1, 19]
            };
            for &shard in &active_shards {
                let writer = GenericStore::open(
                    &uri,
                    schema.clone(),
                    GenericStoreOptions {
                        shard_id: Some(format!("worker-{shard}")),
                        merge_after_generations: Some(0),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
                let rows = [
                    json!({"id":"shared", "text":format!("shard-{shard}")})
                        .as_object()
                        .unwrap()
                        .clone(),
                    json!({"id":format!("only-{shard}"), "text":"x".repeat(payload_bytes)})
                        .as_object()
                        .unwrap()
                        .clone(),
                ];
                writer.add(&rows).await.unwrap();
                writer.flush().await.unwrap();
                writers.push(writer);
            }
            let before = lance::Dataset::open(&uri).await.unwrap().version().version;
            let mut cfg = state.config.clone();
            cfg.catchup.enabled = false;
            cfg.catchup.shards = (0..20).map(|i| format!("worker-{i}")).collect();
            cfg.catchup.grouped_targets = vec![target.into()];
            cfg.catchup.grouped_shards = width;
            cfg.catchup.grouped_parallel_encoding = parallel;
            cfg.catchup.merge_max_bytes = cap;
            cfg.catchup.grouped_prepare_max_bytes = prepare_cap;
            cfg.catchup.merge_memory_bytes = budget_bytes;
            cfg.merge_rollout.owned_targets.push(target.into());
            cfg.catchup.target = Some(target.into());
            cfg.catchup.job_name = Inventory::new(&state)
                .reserve(target, "grouped-integration", 3, 100)
                .await
                .unwrap()
                .job;
            assert_eq!(
                execute(cfg.clone(), target).await.unwrap(),
                ExecuteOutcome::Completed
            );
            let reader = GenericStore::open_existing(&uri, GenericStoreOptions::default())
                .await
                .unwrap();
            assert_eq!(reader.pending_wal_generations().await.unwrap(), 0);
            let base = lance::Dataset::open(&uri).await.unwrap();
            assert_eq!(base.version().version, before + expected_commits);
            if parallel && all_shards {
                assert_eq!(base.manifest().fragments.len(), 2);
            }
            assert_eq!(
                base.count_rows(None).await.unwrap(),
                active_shards.len() + 1
            );
            let rows = reader.list(None, None).await.unwrap();
            assert_eq!(
                rows.iter().find(|r| r["id"] == "shared").unwrap()["text"],
                "shard-19"
            );
            for &shard in &active_shards {
                assert_eq!(
                    rows.iter()
                        .find(|r| r["id"] == format!("only-{shard}"))
                        .unwrap()["text"],
                    "x".repeat(payload_bytes)
                );
            }
            assert!(state
                .task_store
                .merge_coordinator()
                .get(target)
                .await
                .unwrap()
                .is_none());
            // The maintenance pass did not fence ingestion. A later execution
            // must rediscover WAL instead of inheriting empty-shard observations.
            writers[0]
                .add(&[json!({"id":"new-arrival", "text":"live writer"})
                    .as_object()
                    .unwrap()
                    .clone()])
                .await
                .unwrap();
            writers[0].flush().await.unwrap();
            assert_eq!(
                execute(cfg, target).await.unwrap(),
                ExecuteOutcome::Completed
            );
            let reader = GenericStore::open_existing(&uri, GenericStoreOptions::default())
                .await
                .unwrap();
            assert_eq!(reader.pending_wal_generations().await.unwrap(), 0);
            assert_eq!(
                reader.list(None, None).await.unwrap().len(),
                active_shards.len() + 2
            );
        }
    }

    #[derive(Debug)]
    struct Allow;
    impl lance_context_core::merge_write_scope::CommitAuthorizer for Allow {
        fn authorize<'a>(
            &'a self,
            _: &'a str,
            _: u64,
        ) -> futures::future::BoxFuture<'a, lance::Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    #[ignore = "requires isolated loopback ETCD_TEST_ENDPOINTS"]
    async fn grouped_read_failure_commits_only_preceding_shards_even_after_memory_retry() {
        use lance_context_core::{
            merge_write_scope::MergeWriteScope, rollout_append::SealedWalProbe, MergeMemoryBudget,
            RolloutStore,
        };
        use std::{sync::Arc, time::Duration};
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").expect("isolated etcd required");
        assert!(endpoints
            .split(',')
            .all(|e| e.starts_with("http://127.0.0.1:")));
        for (payload_bytes, prepare_cap) in
            [(64, 0), (7 << 19, 0), (64, 128 << 10), (7 << 19, 128 << 10)]
        {
            let (_dir, initial) = fixture().await.unwrap();
            let target = "generic:hot";
            let uri = initial.generic_uri("hot");
            let schema = SchemaSpec::new(vec![
                (
                    "id".into(),
                    ColumnSpec::required(ColumnType::String { large: false }),
                ),
                (
                    "text".into(),
                    ColumnSpec::new(ColumnType::String { large: true }),
                ),
            ]);
            let mut writers = Vec::new();
            for shard in 0..3 {
                let writer = GenericStore::open(
                    &uri,
                    schema.clone(),
                    GenericStoreOptions {
                        shard_id: Some(format!("worker-{shard}")),
                        merge_after_generations: Some(0),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
                writer.add(&[json!({"id":format!("row-{shard}"), "text":"x".repeat(if shard == 0 { payload_bytes } else { 64 })}).as_object().unwrap().clone()]).await.unwrap();
                writer.flush().await.unwrap();
                writers.push(writer);
            }
            let (_, manifest, preparation) =
                writers[1].prepare_cleanup_merge().await.unwrap().unwrap();
            drop(preparation);
            let generation = std::path::PathBuf::from(&uri)
                .join("_mem_wal")
                .join(manifest.shard_id.to_string())
                .join(&manifest.sstables[0].path);
            assert!(generation.starts_with(_dir.path()));
            let hidden = generation.with_extension("injected-hidden");
            std::fs::rename(&generation, &hidden).unwrap();
            let mut cfg = initial.config.clone();
            cfg.catchup.enabled = false;
            cfg.catchup.shards = (0..3).map(|i| format!("worker-{i}")).collect();
            cfg.catchup.grouped_targets = vec![target.into()];
            cfg.catchup.grouped_shards = 3;
            cfg.catchup.merge_max_bytes = 1 << 20;
            cfg.catchup.grouped_prepare_max_bytes = prepare_cap;
            cfg.catchup.merge_memory_bytes = 4 << 20;
            let state = crate::state::MasterState::new(cfg).await.unwrap();
            let budget = MergeMemoryBudget::new(4 << 20);
            let session = RolloutStore::build_session(1 << 20, 1 << 20);
            let probe = SealedWalProbe::open(&uri, Some(session.clone()))
                .await
                .unwrap();
            let before = lance::Dataset::open(&uri).await.unwrap().version().version;
            let scope = MergeWriteScope::with_authorizer(Arc::new(Allow));
            let result = tokio::time::timeout(
                Duration::from_secs(30),
                scope.run(super::run(
                    &state,
                    target,
                    &[0, 1, 2],
                    &probe,
                    session.clone(),
                    budget.clone(),
                    tokio::time::Instant::now() + Duration::from_secs(25),
                )),
            )
            .await
            .unwrap();
            assert!(result.is_err(), "the missing generation must surface");
            scope.drain().await;
            assert_eq!(budget.reserved(), 0);
            assert_eq!(
                lance::Dataset::open(&uri).await.unwrap().version().version,
                before + 1
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
            for (i, writer) in writers.iter().enumerate() {
                assert_eq!(
                    writer
                        .own_shard_watermark()
                        .await
                        .unwrap()
                        .unwrap()
                        .pending_generations,
                    usize::from(i > 0)
                );
            }
            std::fs::rename(&hidden, &generation).unwrap();
            let recovery = MergeWriteScope::with_authorizer(Arc::new(Allow));
            let outcomes = recovery
                .run(super::run(
                    &state,
                    target,
                    &[0, 1, 2],
                    &probe,
                    session,
                    budget.clone(),
                    tokio::time::Instant::now() + Duration::from_secs(25),
                ))
                .await
                .unwrap();
            recovery.drain().await;
            assert_eq!(outcomes.iter().map(|(_, n)| n).sum::<usize>(), 2);
            assert_eq!(budget.reserved(), 0);
            assert_eq!(writers[0].pending_wal_generations().await.unwrap(), 0);
            let reader = GenericStore::open_existing(&uri, Default::default())
                .await
                .unwrap();
            let rows = reader.list(None, None).await.unwrap();
            assert_eq!(rows.len(), 3);
            assert_eq!(
                rows.iter().find(|r| r["id"] == "row-0").unwrap()["text"],
                "x".repeat(payload_bytes)
            );
        }
    }
}
