use super::*;
use lance::index::DatasetIndexExt;
use lance_context_api::{ColumnSpec, ColumnType, SchemaSpec, TaskState, ID_COLUMN};
use lance_context_core::{GenericStore, GenericStoreOptions, KeyIndexType};
use tempfile::TempDir;

fn schema() -> SchemaSpec {
    SchemaSpec::new(vec![
        (
            ID_COLUMN.to_string(),
            ColumnSpec::required(ColumnType::String { large: false }),
        ),
        (
            "custom_value".to_string(),
            ColumnSpec::required(ColumnType::String { large: false }),
        ),
    ])
}

async fn seed(state: &MasterState, name: &str) -> GenericStore {
    let mut store = GenericStore::open(
        &state.generic_uri(name),
        schema(),
        GenericStoreOptions::default(),
    )
    .await
    .unwrap();
    for id in ["a", "b", "c", "d"] {
        store
            .add(&[
                serde_json::json!({"id": id, "custom_value": format!("value-{id}")})
                    .as_object()
                    .unwrap()
                    .clone(),
            ])
            .await
            .unwrap();
        store.cleanup_wal().await.unwrap();
    }
    assert!(store.compaction_stats().total_fragments >= 4);
    store
}

async fn exercise(prepared: bool, owned: bool, index_type: KeyIndexType) {
    let dir = TempDir::new().unwrap();
    let mut cfg = super::tests::config(&dir);
    cfg.key_index_type = index_type;
    cfg.index_after_compaction = true;
    cfg.maintenance.fence_legacy_maintenance = true;
    if !owned {
        cfg.merge_rollout.owned_targets.clear();
    }
    if prepared {
        cfg.maintenance.compaction_prepare_targets = vec!["generic:gs".into()];
        cfg.maintenance.index_prepare_targets = vec!["generic:gs".into()];
    }
    let state = MasterState::new(cfg).await.unwrap();
    let generic = seed(&state, "gs").await;
    let before = generic.compaction_stats().total_fragments;
    // Same bare name: maintenance must leave this other dataset untouched.
    let rollout = RolloutStore::open(&state.rollout_uri("gs")).await.unwrap();
    let rollout_version = rollout.version();
    let compact = enqueue(&state, TaskKind::Compact, "generic:gs")
        .await
        .unwrap();
    let claim = state
        .task_store
        .claim_next_of_kinds(TaskKinds::COMPACT)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(claim.preparing_maintenance(), prepared);
    run_task(&state, claim, TaskClaimTiming::default()).await;
    let done = state.task_store.get(&compact.id).await.unwrap().unwrap();
    assert_eq!(done.state, TaskState::Done, "{done:?}");
    let stats = state
        .stats
        .lock()
        .await
        .get("generic:gs")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stats.uri, state.generic_uri("gs"));
    assert_eq!(stats.row_count, 4);
    assert_eq!(stats.total_compactions, 1);
    assert!(stats.fragment_count < before as i64);
    assert!(state.stats.lock().await.get("gs").await.unwrap().is_none());

    let claim = state
        .task_store
        .claim_next_of_kinds(TaskKinds::GENERAL)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(claim.task.kind, TaskKind::IndexId);
    assert_eq!(claim.task.target, "generic:gs");
    assert_eq!(claim.task.depends_on, vec![compact.id]);
    assert_eq!(claim.preparing_maintenance(), prepared);
    let index_id = claim.task.id.clone();
    run_task(&state, claim, TaskClaimTiming::default()).await;
    let done = state.task_store.get(&index_id).await.unwrap().unwrap();
    assert_eq!(done.state, TaskState::Done, "{done:?}");
    let dataset = lance::Dataset::open(&state.generic_uri("gs"))
        .await
        .unwrap();
    let indexes = dataset.load_indices().await.unwrap();
    let id_index = indexes.iter().find(|index| index.name == "id_idx").unwrap();
    assert_eq!(id_index.fields.len(), 1);
    assert_eq!(
        dataset
            .schema()
            .field_by_id(id_index.fields[0])
            .unwrap()
            .name,
        "id"
    );
    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("id_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_fragments"], 0);
    assert_eq!(
        stats["index_type"].as_str().unwrap().to_ascii_lowercase(),
        index_type.as_str()
    );
    let reader = GenericStore::open_existing(&state.generic_uri("gs"), Default::default())
        .await
        .unwrap();
    for id in ["a", "b", "c", "d"] {
        let row = reader.get(id, None).await.unwrap().unwrap();
        assert_eq!(row["custom_value"], format!("value-{id}"));
    }
    assert_eq!(
        RolloutStore::open_existing_with_options(&state.rollout_uri("gs"), Default::default())
            .await
            .unwrap()
            .version(),
        rollout_version
    );
    assert!(state
        .task_store
        .merge_coordinator()
        .get("generic:gs")
        .await
        .unwrap()
        .is_none());
    if prepared && matches!(index_type, KeyIndexType::Btree) {
        let version = dataset.version().version;
        let task = enqueue(&state, TaskKind::IndexId, "generic:gs")
            .await
            .unwrap();
        let claim = state
            .task_store
            .claim_next_of_kinds(TaskKinds::GENERAL.without_compact())
            .await
            .unwrap()
            .unwrap();
        assert!(claim.preparing_maintenance());
        assert_eq!(claim.task.target, "generic:gs");
        run_task(&state, claim, TaskClaimTiming::default()).await;
        let done = state.task_store.get(&task.id).await.unwrap().unwrap();
        assert_eq!(done.state, TaskState::Done, "{done:?}");
        assert_eq!(
            done.detail.as_deref(),
            Some("existing BTree covers the prepared snapshot; no index publication")
        );
        assert_eq!(
            lance::Dataset::open(&state.generic_uri("gs"))
                .await
                .unwrap()
                .version()
                .version,
            version
        );
        assert!(state
            .task_store
            .merge_coordinator()
            .get("generic:gs")
            .await
            .unwrap()
            .is_none());
    }
    // Compaction no-op hints must use the canonical generic identity/URI.
    if prepared {
        let task = enqueue(&state, TaskKind::Compact, "generic:gs")
            .await
            .unwrap();
        let claim = state
            .task_store
            .claim_next_of_kinds(TaskKinds::COMPACT)
            .await
            .unwrap()
            .unwrap();
        assert!(claim.preparing_maintenance());
        run_task(&state, claim, TaskClaimTiming::default()).await;
        let done = state.task_store.get(&task.id).await.unwrap().unwrap();
        assert_eq!(done.state, TaskState::Done, "{done:?}");
        assert_eq!(
            done.detail.as_deref(),
            Some("compaction snapshot needs no rewrite")
        );
    } else {
        let result = compact_inner(&state, "generic:gs").await.unwrap();
        assert_eq!(result.fragments_added, 0);
    }
    let options = compact_options_key(&state.compaction_config());
    assert!(state
        .task_store
        .compact_is_unchanged(
            "generic:gs",
            &state.generic_uri("gs"),
            reader.version() as i64,
            &options
        )
        .await
        .unwrap());
    assert!(!state
        .task_store
        .compact_is_unchanged(
            "gs",
            &state.rollout_uri("gs"),
            rollout_version as i64,
            &options
        )
        .await
        .unwrap());
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn generic_maintenance_owned_compact_and_index() {
    exercise(false, true, KeyIndexType::Btree).await;
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn generic_maintenance_prepared_owned_compact_and_index() {
    exercise(true, true, KeyIndexType::Zonemap).await;
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn generic_maintenance_prepared_owned_btree_noop() {
    exercise(true, true, KeyIndexType::Btree).await;
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn generic_maintenance_legacy_fenced_compact_and_index() {
    exercise(false, false, KeyIndexType::Zonemap).await;
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn generic_maintenance_prepared_legacy_fenced_compact_and_index() {
    exercise(true, false, KeyIndexType::Btree).await;
}

#[tokio::test]
#[ignore = "requires ETCD_TEST_ENDPOINTS"]
async fn generic_maintenance_preparation_does_not_publish_and_rejects_other_dataset() {
    let dir = TempDir::new().unwrap();
    let state = MasterState::new(super::tests::config(&dir)).await.unwrap();
    let store = seed(&state, "source").await;
    seed(&state, "other").await;
    let before = store.version();
    let scope = lance_context_core::merge_write_scope::MergeWriteScope::with_preparation_authorizer(
        Arc::new(PreparationOnly),
    );
    let (compact, index) = scope
        .run(async {
            let store = MaintenanceStore::open(&state, "generic:source")
                .await
                .unwrap();
            (
                store
                    .prepare_compaction(state.compaction_config())
                    .await
                    .unwrap(),
                store.prepare_id_key_index().await.unwrap(),
            )
        })
        .await;
    assert_eq!(
        MaintenanceStore::open(&state, "generic:source")
            .await
            .unwrap()
            .version(),
        before
    );
    let mut other = MaintenanceStore::open(&state, "generic:other")
        .await
        .unwrap();
    let other_before = other.version();
    assert!(other.commit_prepared_compaction(compact).await.is_err());
    assert!(other.commit_prepared_id_key_index(index).await.is_err());
    assert_eq!(other.version(), other_before);
    let (compact, index) = scope
        .run(async {
            let source = MaintenanceStore::open(&state, "generic:source")
                .await
                .unwrap();
            (
                source
                    .prepare_compaction(state.compaction_config())
                    .await
                    .unwrap(),
                source.prepare_id_key_index().await.unwrap(),
            )
        })
        .await;
    // A competing source rewrite invalidates both preparations. Neither may
    // publish obsolete row addresses or replace the new source fragments.
    let mut source = MaintenanceStore::open(&state, "generic:source")
        .await
        .unwrap();
    source
        .compact(Some(state.compaction_config()))
        .await
        .unwrap();
    let rewritten = source.version();
    assert!(rewritten > before);
    assert!(source.commit_prepared_compaction(compact).await.is_err());
    assert!(source.commit_prepared_id_key_index(index).await.is_err());
    assert_eq!(source.version(), rewritten);
    assert!(MaintenanceStore::open(&state, "generic:absent")
        .await
        .is_err());
    assert!(!dir.path().join("absent.generic.lance").exists());
}
