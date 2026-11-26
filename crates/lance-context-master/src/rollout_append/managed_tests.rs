use super::*;
use crate::{config::MasterConfig, task_store::TaskKinds};
use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use lance_context_api::TaskKind;
use lance_context_core::{StorageBase, StorageBaseOptions};
use lance_context_merge::{Execution, MaintenanceKind};

fn config(dir: &tempfile::TempDir) -> MasterConfig {
    let mut config = MasterConfig::parse_from([
        "test",
        "--data-dir",
        dir.path().to_str().unwrap(),
        "--managed-store-kind",
        "events",
        "--etcd-endpoints",
        &std::env::var("ETCD_TEST_ENDPOINTS").unwrap(),
        "--etcd-prefix",
        &format!("/managed-append/{}", lance_context_core::generate_id()),
    ]);
    config.worker_endpoints.clear();
    config.append.managed_immutable_append = true;
    config.append.rollout_append_local = true;
    config.append.rollout_append_local_task_concurrency = 0;
    config.append.rollout_append_max_bytes = 1 << 20;
    config.append.rollout_append_local_memory_bytes = 8 << 20;
    config.catchup.shards = vec!["a".into(), "b".into()];
    config.maintenance.fence_legacy_maintenance = true;
    config.merge_wal_min_generations = 1;
    config
}

async fn writer(uri: &str, shard: &str) -> StorageBase {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "event_id",
        DataType::Utf8,
        false,
    )
    .with_metadata(
        [("lance-schema:unenforced-primary-key".into(), "true".into())].into(),
    )]));
    let mut options = StorageBaseOptions::new(schema.clone(), "event_id");
    options.shard_id = Some(shard.into());
    options.seal_on_put = false;
    let store = StorageBase::open(uri, options, true).await.unwrap();
    store
        .put(vec![RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![shard]))],
        )
        .unwrap()])
        .await
        .unwrap();
    store.flush().await.unwrap();
    store
}

#[tokio::test]
#[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
async fn registered_shards_merge_through_the_existing_local_executor() {
    let dir = tempfile::tempdir().unwrap();
    let config = config(&dir);
    assert!(crate::local_merge::selected(&config, "events:table"));
    assert!(!crate::local_merge::selected(&config, "other:table"));
    let mut disabled = config.clone();
    disabled.append.managed_immutable_append = false;
    assert!(!crate::local_merge::selected(&disabled, "events:table"));
    assert!(disabled.validate_registered_store().is_err());
    let state = MasterState::new(config).await.unwrap();
    let uri = dir.path().join("application/checkpoints.lance");
    let uri = uri.to_str().unwrap();
    let mut a = writer(uri, "a").await;
    let mut b = writer(uri, "b").await;
    state
        .managed_registry
        .as_ref()
        .unwrap()
        .upsert("table", uri)
        .await
        .unwrap();
    assert_eq!(
        lance::Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        0
    );
    assert_eq!(crate::scanner::scan_once(&state).await.unwrap(), 1);
    let rows = state
        .stats
        .lock()
        .await
        .list(None, usize::MAX, 0)
        .await
        .unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].pending_wal_generations, 2, "{rows:?}");
    // Hot observations enqueue during the scan; the periodic sweep deduplicates
    // that task rather than creating a second merge for the same table.
    assert!(state
        .task_store
        .get_active_id(TaskKind::MergeWal, "events:table")
        .await
        .unwrap()
        .is_some());
    assert_eq!(
        crate::scheduler::sweep_merge_wal_candidates(&state)
            .await
            .unwrap(),
        0
    );
    let claim = state
        .task_store
        .claim_next_of_kinds(TaskKinds::MERGE_WAL)
        .await
        .unwrap()
        .unwrap();
    let result = crate::merge_execution::run_merge_wal(&state, &claim).await;
    assert!(
        result.as_ref().unwrap().contains("2 generations"),
        "{result:?}"
    );
    state.task_store.finish(claim, result).await.unwrap();
    assert_eq!(
        lance::Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        crate::registered_store::open(&state, uri)
            .await
            .unwrap()
            .pending_wal_generations()
            .await
            .unwrap(),
        0
    );
    assert!(state
        .task_store
        .merge_coordinator()
        .get("events:table")
        .await
        .unwrap()
        .is_none());
    assert!(!dir.path().join("events:table.rollout.lance").exists());
    assert_eq!(state.local_append.as_ref().unwrap().budget.reserved(), 0);
    a.close().await.unwrap();
    b.close().await.unwrap();
}

#[tokio::test]
#[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
async fn successor_recovers_registered_maintenance_but_not_unresolved_worker_writes() {
    let dir = tempfile::tempdir().unwrap();
    let config = config(&dir);
    let first = MasterState::new(config.clone()).await.unwrap();
    let uri = dir.path().join("application/checkpoints.lance");
    let uri = uri.to_str().unwrap();
    let mut writer = writer(uri, "a").await;
    first
        .managed_registry
        .as_ref()
        .unwrap()
        .upsert("table", uri)
        .await
        .unwrap();
    first
        .task_store
        .enqueue(TaskKind::MergeWal, "events:table", Vec::new())
        .await
        .unwrap();
    let claim = first
        .task_store
        .claim_next_of_kinds(TaskKinds::MERGE_WAL)
        .await
        .unwrap()
        .unwrap();
    let proof = first.task_store.merge_claim(&claim);
    let coordinator = first.task_store.merge_coordinator();
    let initial_version = lance::Dataset::open(uri).await.unwrap().version().version;
    let mut execution = Execution::new(
        "events:table",
        MaintenanceKind::Catchup.endpoint(),
        "lost-master",
        60,
    );
    execution.maintenance = Some(MaintenanceKind::Catchup);
    assert!(coordinator.reserve(&proof, &execution).await.unwrap());
    let running = coordinator.start(&execution).await.unwrap().unwrap();
    coordinator
        .authorize_commit(&running, uri, "base", initial_version + 1)
        .await
        .unwrap();
    let successor = MasterState::new(config).await.unwrap();
    crate::merge_execution::recover_execution(&successor, &coordinator, &proof, running.clone())
        .await
        .unwrap();
    assert!(coordinator.get("events:table").await.unwrap().is_none());
    assert!(lance::Dataset::open(uri).await.unwrap().version().version > initial_version + 1);
    assert!(coordinator
        .authorize_commit(&running, uri, "base", initial_version + 4)
        .await
        .is_err());
    let result = crate::merge_execution::run_merge_wal(&successor, &claim).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(
        lance::Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        1
    );

    // A rollout must not erase the ownership of a still-unresolved legacy RPC.
    let old = coordinator
        .admit_legacy(&proof, "events:table", "http://127.0.0.1:1")
        .await
        .unwrap();
    let error =
        crate::merge_execution::recover_execution(&successor, &coordinator, &proof, old.clone())
            .await
            .unwrap_err();
    assert!(
        error.contains("unresolved registered-store execution"),
        "{error}"
    );
    assert!(coordinator.get("events:table").await.unwrap().is_some());
    assert!(coordinator.finish(&old, Ok(0)).await.unwrap());
    assert!(coordinator
        .release(&proof, &old.finished(Ok(0)))
        .await
        .unwrap());
    first.task_store.finish(claim, result).await.unwrap();
    writer.close().await.unwrap();
}
