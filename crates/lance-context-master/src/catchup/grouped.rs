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
                            merge_max_bytes: Some(config.merge_max_bytes),
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
                            merge_max_bytes: Some(config.merge_max_bytes),
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
    let reclaimed = match &mut store {
        Store::Rollout(store) => store.commit_prepared_group(group, cap).await,
        Store::Generic(store) => store.commit_prepared_group(group, cap).await,
    }
    .map_err(|e| e.to_string())?;
    if reclaimed != outcomes.iter().map(|(_, count)| count).sum::<usize>() {
        return Err("grouped merge returned an unexpected reclamation count".into());
    }
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
    let allowance = config.merge_max_bytes.min(budget.limit());
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
                    Some(part) => outcomes.extend(commit(state, target, vec![part]).await?),
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
            outcomes.extend(commit(state, target, parts).await?);
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
        for targets in [vec![], vec!["hot".to_owned(), "generic:other".to_owned()]] {
            let mut config = MasterConfig::parse_from(["master"]);
            config.catchup.grouped_targets = targets.clone();
            config.catchup.grouped_shards = 3;
            let job = super::super::kubernetes::render_job(
                &config,
                &record,
                json!({"containers":[{"name":"executor", "env":[
                    {"name":"CATCHUP_GROUPED_TARGETS", "value":"inherited"},
                    {"name":"CATCHUP_GROUPED_SHARDS", "value":"32"}
                ]}]}),
            );
            let env = job["spec"]["template"]["spec"]["containers"][0]["env"]
                .as_array()
                .unwrap();
            let entries = |name: &str| env.iter().filter(|v| v["name"] == name).collect::<Vec<_>>();
            let rendered_targets = entries("CATCHUP_GROUPED_TARGETS");
            let rendered_shards = entries("CATCHUP_GROUPED_SHARDS");
            assert_eq!(rendered_targets.len(), 1);
            assert_eq!(rendered_shards.len(), 1);
            let target_value = rendered_targets[0]["value"].as_str().unwrap();
            let shard_value = rendered_shards[0]["value"].as_str().unwrap();
            assert_eq!(target_value, targets.join(","));
            assert_eq!(shard_value, "3");
            let parsed = MasterConfig::try_parse_from([
                "master",
                "--grouped-targets",
                target_value,
                "--grouped-shards",
                shard_value,
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
    async fn grouped_native_executor_preserves_order_and_handles_tight_and_oversized_budgets() {
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").expect("isolated etcd required");
        assert!(endpoints
            .split(',')
            .all(|e| e.starts_with("http://127.0.0.1:")));
        for (budget_bytes, payload_bytes, expected_commits, cap, width, all_shards) in [
            (32 << 20, 64, 2, 1 << 20, 4, false),
            (1 << 20, 64, 3, 1 << 20, 4, false),
            (4 << 20, 1536 << 10, 3, 1 << 20, 4, false),
            // Sixteen initial 8-MiB reservations exactly fill 128 MiB.
            // Joined reads must free scratch and admit the last four shards,
            // allowing all twenty to publish once without reducing headroom.
            (128 << 20, 64, 1, 8 << 20, 20, true),
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
            cfg.catchup.merge_max_bytes = cap;
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
        for payload_bytes in [64, 7 << 19] {
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
