//! Deterministic drain at a real storage-commit boundary, with durable etcd demand.
use crate::{config::MasterConfig, state::MasterState, task_store::TaskKinds};
use clap::Parser;
use lance_context_api::TaskKind;
use lance_context_core::{
    merge_write_scope::{CommitAuthorizer, MergeWriteScope},
    RolloutStore, RolloutStoreOptions,
};
use std::{future::Future, pin::Pin, sync::Arc, time::Duration};
use tokio::sync::{Notify, Semaphore};

#[derive(Debug)]
struct CommitGate {
    entered: Notify,
    proceed: Semaphore,
}
impl CommitAuthorizer for CommitGate {
    fn authorize<'a>(
        &'a self,
        _resource: &'a str,
        _version: u64,
    ) -> Pin<Box<dyn Future<Output = lance::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.entered.notify_one();
            let permit = self.proceed.acquire().await.unwrap();
            drop(permit);
            Ok(())
        })
    }
}

async fn merge(state: &Arc<MasterState>, append: bool) -> Result<String, String> {
    if append {
        crate::rollout_append::run(state, "hot").await
    } else {
        crate::local_merge::run(state, "hot").await
    }
}

async fn drain_case(append: bool, before_first_batch: bool, pipelined: bool) {
    drain_case_with_upsert_io(append, before_first_batch, pipelined, false, false).await;
}

async fn drain_case_with_upsert_io(
    append: bool,
    before_first_batch: bool,
    pipelined: bool,
    parallel_upsert_io: bool,
    bulk_key_delete: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = MasterConfig::parse_from(["test"]);
    cfg.data_dir = dir.path().to_string_lossy().into();
    cfg.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
        .unwrap()
        .split(',')
        .map(str::to_owned)
        .collect();
    cfg.etcd.etcd_prefix = format!("/native-drain/{}", lance_context_core::generate_id());
    cfg.worker_endpoints.clear();
    cfg.merge_rollout.owned_targets = vec!["hot".into()];
    cfg.local_merge.master_local_merge_targets = vec!["hot".into()];
    cfg.catchup.shards = (0..4).map(|i| format!("writer-{i}")).collect();
    cfg.catchup.slice_secs = 1800;
    cfg.catchup.merge_max_generations = 1;
    cfg.catchup.pipeline_enabled = pipelined;
    if parallel_upsert_io {
        cfg.catchup.parallel_upsert_io_targets = vec!["hot".into()];
    }
    if bulk_key_delete {
        cfg.catchup.bulk_key_delete_targets = vec!["hot".into()];
    }
    if append {
        cfg.append.rollout_append_targets = vec!["hot".into()];
        cfg.append.rollout_append_local = true;
        cfg.append.rollout_append_max_generations = 1;
        cfg.append.rollout_append_concurrency = 2;
    }
    let first = MasterState::new(cfg.clone()).await.unwrap();
    let mut expected = std::collections::BTreeMap::new();
    let mut writers = Vec::new();
    for shard in &cfg.catchup.shards {
        let writer = RolloutStore::open_with_options(
            &first.rollout_uri("hot"),
            RolloutStoreOptions {
                shard_id: Some(shard.clone()),
                merge_after_generations: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        for generation in 0..6 {
            // Upsert exercises real replacement of each writer's previous row.
            let id = if append {
                format!("{shard}-{generation}")
            } else {
                shard.clone()
            };
            let content = format!("value-{generation}");
            let dto = serde_json::from_value(serde_json::json!({
                "id": id, "rollout_id": "r", "content": content,
            }))
            .unwrap();
            writer
                .add(&[lance_context_core::rollout_record_from_add_request(&dto)])
                .await
                .unwrap();
            writer.flush().await.unwrap();
            expected.insert(id, content);
        }
        writers.push(writer);
    }
    if append {
        // Establish the migration metadata before the test gate. Its metadata
        // commit precedes any staging and is not an admitted payload batch.
        let mut coordinator = lance_context_core::rollout_append::AppendCoordinator::open(
            &first.rollout_uri("hot"),
            None,
        )
        .await
        .unwrap();
        let (plans, reclaimed) = coordinator
            .plan(&cfg.catchup.shards, 1, cfg.append.rollout_append_max_bytes)
            .await
            .unwrap();
        assert_eq!(plans.len(), 4);
        assert_eq!(reclaimed, 0);
    }
    let initial_version = lance::Dataset::open(&first.rollout_uri("hot"))
        .await
        .unwrap()
        .version()
        .version;
    assert_eq!(writers[0].pending_wal_generations().await.unwrap(), 24);
    first
        .task_store
        .enqueue(TaskKind::MergeWal, "hot", Vec::new())
        .await
        .unwrap();
    let claim = first
        .task_store
        .claim_next_of_kinds(TaskKinds::MERGE_WAL)
        .await
        .unwrap()
        .unwrap();
    let operation = first.admission.try_admit().unwrap();
    let gate = Arc::new(CommitGate {
        entered: Notify::new(),
        proceed: Semaphore::new(usize::from(before_first_batch)),
    });
    let scope = MergeWriteScope::with_pinned_authorizer(gate.clone());
    if before_first_batch {
        first.admission.begin_drain();
    }
    let runner = {
        let first = first.clone();
        let scope = scope.clone();
        tokio::spawn(async move { scope.run(merge(&first, append)).await })
    };
    if !before_first_batch {
        tokio::time::timeout(Duration::from_secs(30), gate.entered.notified())
            .await
            .unwrap();
        first.admission.begin_drain();
        assert!(
            !runner.is_finished(),
            "accepted commit must be joined, not cancelled"
        );
        assert!(!first.admission.status().drained());
        gate.proceed.add_permits(1);
    }
    let result = tokio::time::timeout(Duration::from_secs(30), runner)
        .await
        .unwrap()
        .unwrap();
    assert!(result.is_ok(), "{result:?}");
    // Use fresh durable manifests: legacy upsert does not populate every
    // logical counter exposed by the append progress scope.
    let remaining = writers[0].pending_wal_generations().await.unwrap();
    let reclaimed = 24 - remaining;
    let base = lance::Dataset::open(&first.rollout_uri("hot"))
        .await
        .unwrap();
    eprintln!("native drain append={append} pipeline={pipelined} before_first={before_first_batch} reclaimed={reclaimed}");
    if before_first_batch {
        assert_eq!(reclaimed, 0);
        assert_eq!(base.version().version, initial_version);
    } else {
        // One active commit, one queued batch, one preparing batch at most.
        // The append buffer may have two staged parts per admitted group.
        assert!(
            (1..=4).contains(&reclaimed),
            "continued new batches after drain: {reclaimed}"
        );
        assert!(base.version().version > initial_version);
        assert!(base.count_rows(None).await.unwrap() > 0);
    }
    assert_eq!(first.local_append.as_ref().unwrap().budget.reserved(), 0);
    assert_eq!(scope.write_status().active_commits, 0);
    // Continuation must already exist while the old canonical task still owns
    // the table. Reconstruct a master to prove it isn't a process-local hint.
    let successor = MasterState::new(cfg).await.unwrap();
    let coordinator = successor.task_store.merge_coordinator();
    let request = coordinator
        .get_merge_request("hot")
        .await
        .unwrap()
        .expect("drain lost continuation");
    assert!(successor
        .task_store
        .claim_next_of_kinds(TaskKinds::MERGE_WAL)
        .await
        .unwrap()
        .is_none());
    first.task_store.finish(claim, result).await.unwrap();
    drop(operation);
    first.admission.wait_drained().await;
    assert!(first.admission.try_admit().is_none());

    // Consume persisted demand without scanning or manually selecting a lost
    // task, then use the normal owned/fenced merge path on the new executor.
    successor
        .task_store
        .enqueue(TaskKind::MergeWal, &request.target, Vec::new())
        .await
        .unwrap();
    coordinator.acknowledge_request(&request).await.unwrap();
    let next = successor
        .task_store
        .claim_next_of_kinds(TaskKinds::MERGE_WAL)
        .await
        .unwrap()
        .unwrap();
    let result = crate::merge_execution::run_merge_wal(&successor, &next).await;
    assert!(result.is_ok(), "{result:?}");
    successor.task_store.finish(next, result).await.unwrap();
    let store = RolloutStore::open_existing_with_options(
        &successor.rollout_uri("hot"),
        RolloutStoreOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        lance::Dataset::open(&successor.rollout_uri("hot"))
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        expected.len()
    );
    for (id, value) in expected {
        assert_eq!(
            store
                .get_by_id(&id)
                .await
                .unwrap()
                .unwrap()
                .content
                .as_deref(),
            Some(value.as_str())
        );
    }
    for writer in &writers {
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
    }
    assert_eq!(
        successor.local_append.as_ref().unwrap().budget.reserved(),
        0
    );
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn native_drain_upsert_joins_commit_and_persists_continuation() {
    for pipelined in [false, true] {
        drain_case(false, false, pipelined).await;
    }
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn native_drain_append_joins_commit_and_persists_continuation() {
    drain_case(true, false, true).await;
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn native_drain_before_first_batch_keeps_durable_demand() {
    for (append, pipelined) in [(false, false), (false, true), (true, true)] {
        drain_case(append, true, pipelined).await;
    }
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn native_drain_parallel_upsert_joins_commit_and_persists_continuation() {
    for pipelined in [false, true] {
        for bulk in [false, true] {
            drain_case_with_upsert_io(false, false, pipelined, true, bulk).await;
        }
    }
}
