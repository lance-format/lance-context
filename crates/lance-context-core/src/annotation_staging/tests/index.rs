//! Index publication must not retire a newer annotation generation.
use super::*;
use lance_table::transaction::{Operation, Transaction};

async fn indexed(base: &Dataset) -> Result<Transaction> {
    let mut snapshot = base.clone();
    let index = snapshot
        .create_index_builder(&["id"], IndexType::BTree, &ScalarIndexParams::default())
        .name("turn_id".into())
        .execute_uncommitted()
        .await?;
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::CreateIndex {
            new_indices: vec![index],
            removed_indices: Vec::new(),
        },
    );
    transaction.transaction_properties = base
        .read_transaction()
        .await?
        .unwrap()
        .transaction_properties;
    Ok(transaction)
}

async fn versions(staging: &AnnotationStaging) -> Result<Vec<u64>> {
    Ok(staging
        .protected_bases()
        .await?
        .datasets()
        .map(Dataset::version_id)
        .collect())
}

#[tokio::test]
async fn indexed_base_keeps_active_updates_readers_and_replay_boundary() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (staging, base, _) = compaction_body::initialized(root.path()).await?;
    let held = staging.capture_turn("turn-000000").await?;
    let transaction = indexed(&base).await?;
    let patch = TurnAnnotationPatch {
        status: Set("reviewed".into()),
        score: Set(0.9),
        ..Default::default()
    };
    // The index targets the earlier base; this later update remains active.
    staging.apply(1, 2, "turn-000000", &patch).await?;
    let expected = current(&staging, "turn-000000").await?;
    let before = staging.observation().await;
    let selected = compaction::select(base.clone(), &transaction).await?;
    staging
        .install_indexed(&transaction, selected.clone())
        .await?;
    let mut before = before;
    before.installed_base.version = selected.version_id();
    assert_eq!(staging.observation().await, before);
    assert_eq!(current(&staging, "turn-000000").await?, expected);
    assert_eq!(selected.manifest().fragments, base.manifest().fragments);
    assert_eq!(selected.load_indices().await?.len(), 1);
    assert_eq!(
        held.annotations().resolve().await?.unwrap().score,
        Some(0.4)
    );
    let TurnBodySource::Materialized(old) = held.body_source() else {
        panic!("body is materialized")
    };
    assert_eq!(old.version_id(), base.version_id());
    assert_eq!(
        BodySnapshot::read(&selected).await?,
        BodySnapshot::read(&base).await?
    );
    assert_eq!(
        versions(&staging).await?,
        vec![base.version_id(), selected.version_id()]
    );

    // Reopen the index-only publication and replay only the unmaterialized
    // suffix. Native indexing must not advance the durable annotation cut.
    let reopened = Arc::new(Dataset::open(selected.uri()).await?);
    let properties = reopened
        .read_transaction()
        .await?
        .unwrap()
        .transaction_properties
        .unwrap();
    let (recovered_scope, through): (Scope, u64) =
        serde_json::from_str(&properties["lance-context.annotation-cut"])?;
    assert_eq!(through, 1);
    assert_eq!(properties[RESULTS_PROPERTY], "results-at-1");
    let recovered = AnnotationStaging::new(reopened, recovered_scope, through, Limits::default())?;
    recovered.apply(1, 2, "turn-000000", &patch).await?;
    assert_eq!(current(&recovered, "turn-000000").await?, expected);

    drop(held);
    assert_eq!(versions(&staging).await?, vec![selected.version_id()]);
    let frozen = staging
        .freeze_with_results(uuid::Uuid::new_v4(), Some("results-at-2".into()))
        .await?
        .unwrap();
    let prepared = frozen
        .prepare_with_body(
            &crate::MergeMemoryBudget::new(8 << 20),
            8192,
            BodySnapshot::read(&selected).await?.unwrap(),
        )
        .await?;
    assert_eq!(prepared.base().version_id(), selected.version_id());
    let published = commit(&prepared).await?;
    staging
        .install_published(&prepared, published.clone())
        .await?;
    assert_eq!(staging.observation().await.materialized, 2);
    assert_eq!(current(&staging, "turn-000000").await?, expected);
    let reopened = Dataset::open(published.uri()).await?;
    assert_eq!(
        read_base(&reopened, "turn-000000", 4096).await?,
        Some(expected)
    );
    assert_eq!(reopened.load_indices().await?.len(), 1);
    Ok(())
}

#[tokio::test]
async fn index_install_rejects_changed_recovery_metadata_and_frozen_publication() -> Result<()> {
    for case in 0..8 {
        let root = tempfile::tempdir()?;
        let (staging, base, _) = compaction_body::initialized(root.path()).await?;
        let mut transaction = indexed(&base).await?;
        let properties = Arc::make_mut(transaction.transaction_properties.as_mut().unwrap());
        match case {
            0 => {
                properties.remove("lance-context.annotation-cut");
            }
            1 => {
                properties.insert(
                    "lance-context.annotation-cut".into(),
                    serde_json::to_string(&(scope(), 2))?,
                );
            }
            2 => {
                properties.remove(RESULTS_PROPERTY);
            }
            3 => {
                properties.insert(RESULTS_PROPERTY.into(), "different-results".into());
            }
            4 => {
                properties.remove(BODY_SNAPSHOT_PROPERTY);
            }
            5 => {
                let mut reference = BodySnapshot::read(&base).await?.unwrap();
                reference.version += 1;
                properties.insert(
                    BODY_SNAPSHOT_PROPERTY.into(),
                    serde_json::to_string(&reference)?,
                );
            }
            6 => {
                staging
                    .apply(
                        1,
                        2,
                        "turn-000000",
                        &TurnAnnotationPatch {
                            score: Clear,
                            ..Default::default()
                        },
                    )
                    .await?;
                let _frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
            }
            7 => {
                transaction.operation = Operation::UpdateConfig {
                    config_updates: None,
                    table_metadata_updates: None,
                    schema_metadata_updates: None,
                    field_metadata_updates: Default::default(),
                };
            }
            _ => unreachable!(),
        }
        let selected = compaction::select(base.clone(), &transaction).await?;
        let before = staging.observation().await;
        let expected = current(&staging, "turn-000000").await?;
        assert!(
            staging
                .install_indexed(&transaction, selected)
                .await
                .is_err(),
            "case {case}"
        );
        assert_eq!(staging.observation().await, before);
        assert_eq!(current(&staging, "turn-000000").await?, expected);
        assert_eq!(versions(&staging).await?, vec![base.version_id()]);
    }
    Ok(())
}

#[tokio::test]
async fn index_install_requires_the_exact_selected_transaction() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (staging, base, _) = compaction_body::initialized(root.path()).await?;
    let transaction = indexed(&base).await?;
    let selected = compaction::select(base.clone(), &transaction).await?;
    let mut other = transaction.clone();
    other.uuid = uuid::Uuid::new_v4().to_string();
    let before = staging.observation().await;
    assert!(staging
        .install_indexed(&other, selected.clone())
        .await
        .is_err());
    assert!(staging
        .install_indexed(&transaction, base.clone())
        .await
        .is_err());
    assert_eq!(staging.observation().await, before);
    assert_eq!(versions(&staging).await?, vec![base.version_id()]);
    staging
        .install_indexed(&transaction, selected.clone())
        .await?;
    let mut before = before;
    before.installed_base.version = selected.version_id();
    // An already installed publication is not a new base transition.
    assert!(staging
        .install_indexed(&transaction, selected)
        .await
        .is_err());
    assert_eq!(staging.observation().await, before);
    Ok(())
}
