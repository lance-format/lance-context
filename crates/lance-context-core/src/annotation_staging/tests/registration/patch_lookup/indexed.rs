use super::*;
use lance_table::transaction::{Operation, Transaction};

#[tokio::test]
async fn indexed_publication_preserves_prepared_patch_values_without_io() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened_rows(&root.path().join("narrow.lance"), 8).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    staging.apply(0, 1, "turn-000000", &score(0.1)).await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 32)
        .await?;
    let base = commit(&prepared).await?;
    staging.install_published(&prepared, base.clone()).await?;
    drop((prepared, frozen));

    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let lookup = staging
        .prepare_patch_lookup(&["turn-000001", "turn-000002"], &workspace)
        .await?
        .unwrap();
    let mut snapshot = (*base).clone();
    let index = snapshot
        .create_index_builder(&["status"], IndexType::BTree, &ScalarIndexParams::default())
        .name("annotation_status".into())
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
    let selected = compaction::select(base.clone(), &transaction).await?;
    staging
        .install_indexed(&transaction, selected.clone())
        .await?;
    assert_eq!(base.manifest().fragments, selected.manifest().fragments);
    assert_eq!(staging.protected_bases().await?.datasets().count(), 2);

    // The lookup owns decoded narrow values. An index-only publication must
    // not make later requests fetch those identical values from storage again.
    selected.session().file_metadata_cache().clear().await;
    let before = throttle.ops.load(Ordering::Relaxed);
    throttle.frozen.store(true, Ordering::Relaxed);
    let outcome = tokio::time::timeout(Duration::from_secs(2), async {
        let reviewed = TurnAnnotationPatch {
            status: Set("reviewed".into()),
            ..Default::default()
        };
        for (after, turn, patch) in [
            (1, "turn-000001", reviewed),
            (2, "turn-000001", score(0.9)),
            (3, "turn-000002", score(0.8)),
        ] {
            assert_eq!(
                apply(&staging, after, turn, &patch, &lookup).await?,
                ApplyOutcome::Applied
            );
        }
        Ok::<_, Error>(())
    })
    .await;
    throttle.frozen.store(false, Ordering::Relaxed);
    outcome.map_err(|_| invalid("index-only publication caused redundant patch IO"))??;
    assert_eq!(throttle.ops.load(Ordering::Relaxed), before);
    let value = current(&staging, "turn-000001").await?;
    assert_eq!(value.status.as_deref(), Some("reviewed"));
    assert_eq!(value.score, Some(0.9));
    assert_eq!(value.labels, BTreeSet::from(["imported".into()]));
    assert_eq!(current(&staging, "turn-000002").await?.score, Some(0.8));
    assert_eq!(staging.observation().await.applied, 4);
    assert_eq!(staging.observation().await.materialized, 1);
    drop(lookup);
    assert_eq!(workspace.reserved(), 0);
    assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    Ok(())
}
