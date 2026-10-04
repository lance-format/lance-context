use super::*;
use clap::Parser;
use lance_context_api::TaskKind;
use serde_json::json;

fn config() -> MasterConfig {
    let mut config = MasterConfig::parse_from(["master"]);
    config.catchup.enabled = true;
    config.catchup.shards = vec!["worker-0".into(), "worker-1".into()];
    config.merge_rollout.owned_targets = vec!["hot".into(), "other".into()];
    config
}
fn row(now: i64) -> StatRow {
    StatRow {
        name: "hot".into(),
        uri: "/hot".into(),
        row_count: 0,
        fragment_count: 0,
        last_updated: now,
        pending_wal_generations: 1000,
        last_compaction: -1,
        total_compactions: 0,
        scanned_at: now,
        version: 1,
    }
}
fn template() -> serde_json::Value {
    json!({"containers":[{"name":"catchup","image":format!("example/native@sha256:{}", "a".repeat(64)),"command":["/usr/local/bin/lance-context-master"],"resources":{"requests":{"cpu":"2","memory":"8Gi"},"limits":{"cpu":"2","memory":"8Gi"}}}]})
}
#[test]
fn admission_requires_fresh_owned_pressure_even_for_manual_requests() {
    let mut c = config();
    let now = 10_000_000;
    let mut r = row(now);
    assert!(eligibility(&c, Some(&r), "hot", now).is_none());
    assert_eq!(
        eligibility(&c, Some(&r), "legacy", now),
        Some("requires_owned_target")
    );
    r.scanned_at = now - 901_000;
    assert_eq!(eligibility(&c, Some(&r), "hot", now), Some("stale_stats"));
    r.scanned_at = now + 1;
    assert_eq!(eligibility(&c, Some(&r), "hot", now), Some("stale_stats"));
    r.scanned_at = now;
    r.pending_wal_generations = 0;
    assert_eq!(
        eligibility(&c, Some(&r), "hot", now),
        Some("below_threshold")
    );
    c.catchup.enabled = false;
    assert_eq!(eligibility(&c, Some(&r), "hot", now), Some("disabled"));
}
#[test]
fn job_is_native_and_bounded_and_overrides_unsafe_inherited_mode() {
    let mut c = config();
    c.key_index_type = lance_context_core::KeyIndexType::Zonemap;
    c.catchup.pipeline_enabled = false;
    c.catchup.merge_max_generations = 32;
    let r = Record {
        target: "hot".into(),
        job: "lc-catchup-test".into(),
        job_uid: None,
        slot: 0,
        attempt: 1,
        active: true,
        termination_requested: false,
        reason: "test".into(),
        pending_at_admission: 1000,
        created_at_ms: 0,
        finished_at_ms: None,
        consecutive_failures: 0,
        needs_attention: false,
        next_retry_ms: 0,
        outcome: None,
        progress: None,
    };
    let mut pod = template();
    pod["containers"][0]["env"] = json!([
        {"name":"CATCHUP_ENABLED","value":"true"},
        {"name":"CATCHUP_PIPELINE_ENABLED","value":"true"},
        {"name":"CATCHUP_MERGE_MAX_GENERATIONS","value":"8"},
        {"name":"ROLLOUT_KEY_INDEX_TYPE","value":"btree"}
    ]);
    pod["activeDeadlineSeconds"] = json!(1);
    let job = kubernetes::render_job(&c, &r, pod);
    assert!(job["spec"].get("activeDeadlineSeconds").is_none());
    assert!(job["spec"]["template"]["spec"]
        .get("activeDeadlineSeconds")
        .is_none());
    assert_eq!(job["spec"]["parallelism"], 1);
    assert_eq!(job["spec"]["backoffLimit"], 0);
    assert_eq!(job["spec"]["template"]["spec"]["restartPolicy"], "Never");
    assert_eq!(
        job["spec"]["template"]["spec"]["automountServiceAccountToken"],
        false
    );
    let env = job["spec"]["template"]["spec"]["containers"][0]["env"]
        .as_array()
        .unwrap();
    for (name, expected) in [
        ("CATCHUP_PIPELINE_ENABLED", "false"),
        ("CATCHUP_MERGE_MAX_GENERATIONS", "32"),
    ] {
        let matches: Vec<_> = env.iter().filter(|e| e["name"] == name).collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["value"], expected);
    }
    assert_eq!(
        env.iter()
            .filter(|e| e["name"] == "ROLLOUT_KEY_INDEX_TYPE")
            .count(),
        1
    );
    assert_eq!(
        env.iter()
            .find(|e| e["name"] == "ROLLOUT_KEY_INDEX_TYPE")
            .unwrap()["value"],
        "zonemap"
    );
    assert_eq!(
        env.iter()
            .filter(|e| e["name"] == "CATCHUP_ENABLED")
            .count(),
        1
    );
    assert_eq!(
        env.iter().find(|e| e["name"] == "CATCHUP_ENABLED").unwrap()["value"],
        "false"
    );
    let mut next = r.clone();
    next.attempt = 2;
    let next_job = kubernetes::render_job(&c, &next, template());
    let env = next_job["spec"]["template"]["spec"]["containers"][0]["env"]
        .as_array()
        .unwrap();
    assert_eq!(
        env.iter().find(|e| e["name"] == "CATCHUP_SHARDS").unwrap()["value"],
        "worker-1,worker-0"
    );
}
async fn fixture() -> Option<(tempfile::TempDir, Arc<MasterState>)> {
    let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").expect("ETCD_TEST_ENDPOINTS is required");
    let dir = tempfile::tempdir().unwrap();
    let mut c = config();
    c.data_dir = dir.path().to_string_lossy().into();
    c.etcd.etcd_endpoints = endpoints.split(',').map(str::to_string).collect();
    c.etcd.etcd_prefix = format!("/catchup-test/{}", lance_context_core::generate_id());
    c.catchup.max_jobs = 1;
    let file = dir.path().join("pod.json");
    std::fs::write(&file, template().to_string()).unwrap();
    c.catchup.pod_template = Some(file.to_string_lossy().into());
    Some((dir, MasterState::new(c).await.unwrap()))
}
#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn replicas_share_one_slot_and_dedupe_duplicate_requests() {
    let Some((_dir, state)) = fixture().await else {
        return;
    };
    let a = Inventory::new(&state);
    let b = Inventory::new(&state);
    let (one, two) = tokio::join!(
        a.reserve("hot", "test", 1000, 100),
        b.reserve("hot", "retry", 1000, 100)
    );
    assert_eq!(
        [one.unwrap(), two.unwrap()]
            .iter()
            .filter(|d| d.decision == "reserved")
            .count(),
        1
    );
    assert_eq!(
        a.reserve("other", "test", 1000, 100)
            .await
            .unwrap()
            .decision,
        "capacity_exhausted"
    );
    assert_eq!(a.active().await.unwrap().len(), 1);
    let record = a.get("hot").await.unwrap().unwrap();
    b.complete(&record, false, 1000).await.unwrap();
    let failed = a.get("hot").await.unwrap().unwrap();
    assert_eq!(failed.consecutive_failures, 1);
    assert!(!failed.active);
    assert_ne!(
        a.reserve("hot", "retry", 1000, 1001)
            .await
            .unwrap()
            .decision,
        "reserved"
    );
    assert_eq!(
        a.reserve("other", "test", 1000, 1001)
            .await
            .unwrap()
            .decision,
        "reserved"
    );
    // Replaying an old completion cannot release the new table's slot.
    b.complete(&record, true, 2000).await.unwrap();
    assert_eq!(a.active().await.unwrap()[0].target, "other");
}
#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn dedicated_claim_excludes_normal_pollers_but_not_other_tables() {
    let Some((_dir, state)) = fixture().await else {
        return;
    };
    let inventory = Inventory::new(&state);
    let reserved = inventory.reserve("hot", "test", 1000, 100).await.unwrap();
    state
        .task_store
        .enqueue(TaskKind::MergeWal, "hot", vec![])
        .await
        .unwrap();
    state
        .task_store
        .enqueue(TaskKind::MergeWal, "other", vec![])
        .await
        .unwrap();
    let ordinary = state.task_store.claim_next().await.unwrap().unwrap();
    assert_eq!(ordinary.task.target, "other");
    assert!(state
        .task_store
        .claim_merge_target("hot", "wrong-job")
        .await
        .unwrap()
        .is_none());
    let dedicated = state
        .task_store
        .claim_merge_target("hot", reserved.job.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(dedicated.task.target, "hot");
    assert!(state.task_store.claim_next().await.unwrap().is_none());
    state
        .task_store
        .finish(dedicated, Ok("test".into()))
        .await
        .unwrap();
    state
        .task_store
        .finish(ordinary, Ok("test".into()))
        .await
        .unwrap();
}
#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn running_native_task_prevents_provisioning_empty_waiters() {
    let Some((_dir, state)) = fixture().await else {
        return;
    };
    state
        .task_store
        .enqueue(TaskKind::MergeWal, "hot", vec![])
        .await
        .unwrap();
    let claim = state.task_store.claim_next().await.unwrap().unwrap();
    let result = Inventory::new(&state)
        .reserve("hot", "test", 1000, 100)
        .await
        .unwrap();
    assert_ne!(result.decision, "reserved");
    state
        .task_store
        .finish(claim, Ok("test".into()))
        .await
        .unwrap();
}
#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn policy_disagreement_cannot_expand_the_cluster_budget() {
    let Some((_dir, state)) = fixture().await else {
        return;
    };
    let inventory = Inventory::new(&state);
    inventory.ensure_policy(&state.config).await.unwrap();
    let mut other = state.config.clone();
    other.catchup.max_jobs += 1;
    assert!(inventory.ensure_policy(&other).await.is_err());
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn native_executor_preserves_live_ingestion_and_drains_sealed_shards() {
    for memory_bytes in [1024 * 1024, 2 * 1024 * 1024] {
        native_executor_with_budget(memory_bytes).await;
    }
}

async fn native_executor_with_budget(memory_bytes: usize) {
    use lance_context_core::{
        ColumnSpec, ColumnType, GenericStore, GenericStoreOptions, SchemaSpec,
    };
    let (_dir, state) = fixture().await.unwrap();
    let target = "generic:hot";
    let uri = state.generic_uri("hot");
    let spec = SchemaSpec::new(vec![
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
    for shard in ["worker-0", "worker-1"] {
        let writer = GenericStore::open(
            &uri,
            spec.clone(),
            GenericStoreOptions {
                shard_id: Some(shard.into()),
                merge_after_generations: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        for generation in 0..4 {
            let row = json!({"id":format!("{shard}-{generation}"),"text":"x".repeat(512 * 1024)})
                .as_object()
                .unwrap()
                .clone();
            let shared = json!({"id":"shared", "text":format!("{shard}-{generation}")})
                .as_object()
                .unwrap()
                .clone();
            writer.add(&[row, shared]).await.unwrap();
            writer.flush().await.unwrap();
        }
        writers.push(writer);
    }
    assert_eq!(writers[0].pending_wal_generations().await.unwrap(), 8);
    let mut cfg = state.config.clone();
    cfg.catchup.enabled = false;
    // Exercise both overlapping reads and a budget that admits one batch.
    // Prefetch must not block the commit that releases its reservation.
    cfg.catchup.merge_max_bytes = 512 * 1024;
    cfg.catchup.merge_memory_bytes = memory_bytes;
    cfg.merge_rollout.owned_targets.push(target.into());
    let decision = Inventory::new(&state)
        .reserve(target, "integration", 8, 100)
        .await
        .unwrap();
    cfg.catchup.target = Some(target.into());
    cfg.catchup.job_name = decision.job;
    let (executed, ()) = tokio::join!(execute(cfg, target), async {
        for n in 0..8 {
            for (i, writer) in writers.iter().enumerate() {
                writer
                    .add(
                        &[json!({"id":format!("during-{i}-{n}"),"text":"continuous"})
                            .as_object()
                            .unwrap()
                            .clone()],
                    )
                    .await
                    .unwrap();
                writer.flush().await.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    });
    executed.unwrap();
    // The old ingestion handles remain usable; the executor must not claim
    // a writer epoch or seal somebody else's active memtable.
    for (i, writer) in writers.iter().enumerate() {
        writer
            .add(&[json!({"id":format!("after-{i}"),"text":"live"})
                .as_object()
                .unwrap()
                .clone()])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    let reader = GenericStore::open_existing(&uri, GenericStoreOptions::default())
        .await
        .unwrap();
    assert!(reader.pending_wal_generations().await.unwrap() <= 18);
    let rows = reader.list(None, None).await.unwrap();
    assert_eq!(rows.len(), 27);
    assert_eq!(
        rows.iter().find(|r| r["id"] == "shared").unwrap()["text"],
        "worker-1-3"
    );
    let ids: std::collections::HashSet<_> =
        rows.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), 27);
    assert!(state
        .task_store
        .merge_coordinator()
        .get(target)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn repeated_failed_jobs_keep_backoff_across_controller_restarts() {
    let (_dir, state) = fixture().await.unwrap();
    let mut now = 1000;
    for attempt in 1..=4 {
        let inventory = Inventory::new(&state);
        assert_eq!(
            inventory
                .reserve("hot", "test", 1000, now)
                .await
                .unwrap()
                .decision,
            "reserved"
        );
        let record = inventory.get("hot").await.unwrap().unwrap();
        assert_eq!(record.attempt, u64::from(attempt));
        inventory.complete(&record, false, now + 1).await.unwrap();
        let persisted = Inventory::new(&state).get("hot").await.unwrap().unwrap();
        assert_eq!(persisted.consecutive_failures, attempt);
        assert_eq!(persisted.needs_attention, attempt >= 3);
        assert!(persisted.next_retry_ms > now + 60_000);
        now = persisted.next_retry_ms;
    }
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn catchup_runtime_ceiling_does_not_cancel_work_within_idle_window() {
    let (_dir, state) = fixture().await.unwrap();
    let mut cfg = state.config.clone();
    cfg.catchup.enabled = false;
    cfg.catchup.target = Some("hot".into());
    let inventory = Inventory::new(&state);
    let admitted = inventory
        .reserve("hot", "runtime test", 1000, 0)
        .await
        .unwrap();
    cfg.catchup.job_name = admitted.job.clone();
    cfg.maintenance.maintenance_timeout_secs = 1;
    cfg.maintenance.maintenance_idle_timeout_secs = 10;
    let dedicated = MasterState::new(cfg).await.unwrap();
    dedicated
        .task_store
        .enqueue(TaskKind::MergeWal, "hot", vec![])
        .await
        .unwrap();
    let claim = dedicated
        .task_store
        .claim_merge_target("hot", admitted.job.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    let result = crate::maintenance_execution::run_as(
        &dedicated,
        &claim,
        lance_context_merge::MaintenanceKind::Catchup,
        async {
            tokio::time::sleep(Duration::from_millis(2200)).await;
            Ok("operation completed beyond old ceiling".into())
        },
    )
    .await;
    assert!(result.is_ok(), "{result:?}");
    dedicated.task_store.finish(claim, result).await.unwrap();
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn stale_progress_revoke_cannot_cancel_advancing_or_replaced_execution() {
    let (_dir, state) = fixture().await.unwrap();
    let inventory = Inventory::new(&state);
    let admitted = inventory
        .reserve("hot", "stall test", 1000, 0)
        .await
        .unwrap();
    state
        .task_store
        .enqueue(TaskKind::MergeWal, "hot", vec![])
        .await
        .unwrap();
    let claim = state
        .task_store
        .claim_merge_target("hot", admitted.job.as_ref().unwrap())
        .await
        .unwrap()
        .unwrap();
    let coordinator = state.task_store.merge_coordinator();
    let mut execution = lance_context_merge::Execution::new(
        "hot",
        "master:catchup",
        admitted.job.as_ref().unwrap(),
        1,
    );
    execution.maintenance = Some(lance_context_merge::MaintenanceKind::Catchup);
    let proof = state.task_store.merge_claim(&claim);
    assert!(coordinator.reserve(&proof, &execution).await.unwrap());
    let record = inventory.get("hot").await.unwrap().unwrap();
    let queued = progress::sample(&state, &record).await.unwrap();
    let running = coordinator.start(&execution).await.unwrap().unwrap();
    // Starting the same execution changes its phase before its first progress
    // publication. The old queued sample must not cancel this fresh executor.
    assert!(!progress::revoke(&state, &record, &queued).await.unwrap());
    assert!(coordinator.publish_progress(&running, 1).await.unwrap());
    let record = inventory.get("hot").await.unwrap().unwrap();
    let old = progress::sample(&state, &record).await.unwrap();
    assert!(coordinator.publish_progress(&running, 2).await.unwrap());
    assert!(!progress::revoke(&state, &record, &old).await.unwrap());
    let fresh = progress::sample(&state, &record).await.unwrap();
    assert!(progress::revoke(&state, &record, &fresh).await.unwrap());
    assert!(!coordinator.publish_progress(&running, 3).await.unwrap());
    assert!(coordinator
        .authorize_commit(&running, "/hot", "base", 2)
        .await
        .is_err());
    let uncertain = coordinator.get("hot").await.unwrap().unwrap();
    assert_eq!(uncertain.phase, lance_context_merge::Phase::Uncertain);
    assert!(coordinator.release(&proof, &uncertain).await.is_err());
    let frozen = coordinator
        .freeze(&proof, &uncertain)
        .await
        .unwrap()
        .unwrap();
    assert!(coordinator.finish_recovery(&proof, &frozen).await.unwrap());
    let recovered = coordinator.get("hot").await.unwrap().unwrap();
    assert!(coordinator.release(&proof, &recovered).await.unwrap());
    state
        .task_store
        .finish(claim, Err("injected stall".into()))
        .await
        .unwrap();
}

/// Full native executor comparison; setup and verification are outside timing.
/// Opt in explicitly so CI's ignored etcd suite does not run a benchmark.
#[tokio::test]
#[ignore = "manual benchmark: CATCHUP_BENCH=1 ETCD_TEST_ENDPOINTS required"]
async fn benchmark_native_catchup_pipeline() {
    if std::env::var("CATCHUP_BENCH").as_deref() != Ok("1") {
        return;
    }
    use lance_context_core::{
        ColumnSpec, ColumnType, GenericStore, GenericStoreOptions, SchemaSpec,
    };
    let workload = std::env::var("CATCHUP_BENCH_WORKLOAD").unwrap_or_else(|_| "updates".into());
    assert!(["updates", "unique"].contains(&workload.as_str()));
    for (pipeline, max_generations) in [
        (false, 8),
        (true, 8),
        (true, 64),
        (true, 64),
        (true, 8),
        (false, 8),
    ] {
        let (_dir, state) = fixture().await.unwrap();
        let target = "generic:hot";
        let uri = state.generic_uri("hot");
        let spec = SchemaSpec::new(vec![
            (
                "id".into(),
                ColumnSpec::required(ColumnType::String { large: false }),
            ),
            (
                "text".into(),
                ColumnSpec::new(ColumnType::String { large: true }),
            ),
        ]);
        let shards: Vec<_> = (0..4).map(|i| format!("worker-{i}")).collect();
        let mut writers = Vec::new();
        for shard in &shards {
            let writer = GenericStore::open(
                &uri,
                spec.clone(),
                GenericStoreOptions {
                    shard_id: Some(shard.clone()),
                    merge_after_generations: Some(0),
                    session: Some(lance_context_core::RolloutStore::build_session(
                        32 << 20,
                        32 << 20,
                    )),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            for generation in 0..32 {
                let rows: Vec<_> = (0..8).map(|id| {
                    json!({"id": if workload == "updates" { format!("{shard}-{id}") } else { format!("{shard}-{generation}-{id}") }, "text": format!("{generation}:{}", "x".repeat(8192))})
                        .as_object().unwrap().clone()
                }).collect();
                writer.add(&rows).await.unwrap();
                writer.flush().await.unwrap();
            }
            writers.push(writer);
        }
        let pending = writers[0].pending_wal_generations().await.unwrap();
        assert_eq!(pending, 128);
        let mut cfg = state.config.clone();
        cfg.catchup.enabled = false;
        cfg.catchup.shards = shards;
        cfg.catchup.pipeline_enabled = pipeline;
        cfg.catchup.merge_max_generations = max_generations;
        cfg.merge_rollout.owned_targets.push(target.into());
        cfg.catchup.target = Some(target.into());
        cfg.catchup.job_name = Inventory::new(&state)
            .reserve(target, "benchmark", pending as i64, 100)
            .await
            .unwrap()
            .job;
        let started = std::time::Instant::now();
        execute(cfg, target).await.unwrap();
        let seconds = started.elapsed().as_secs_f64();
        let reader = GenericStore::open_existing(&uri, GenericStoreOptions::default())
            .await
            .unwrap();
        assert_eq!(reader.pending_wal_generations().await.unwrap(), 0);
        let rows = reader.list(None, None).await.unwrap();
        let expected_rows = if workload == "updates" { 32 } else { 1024 };
        assert_eq!(rows.len(), expected_rows);
        for row in rows {
            let generation = if workload == "updates" {
                "31"
            } else {
                row["id"].as_str().unwrap().split('-').nth(2).unwrap()
            };
            assert_eq!(row["text"], format!("{generation}:{}", "x".repeat(8192)));
        }
        let base = lance::Dataset::open(&uri).await.unwrap();
        assert_eq!(base.count_rows(None).await.unwrap(), expected_rows);
        eprintln!(
            "CATCHUP_BENCH {}",
            json!({"workload":workload,"pipeline":pipeline,"max_generations":max_generations,
            "seconds":seconds,"generations":pending,"generations_per_second":pending as f64 / seconds,
            "base_version":base.version().version})
        );
    }
}
