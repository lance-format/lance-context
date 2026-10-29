use super::*;
use lance::dataset::optimize::{CompactionMode, CompactionOptions, CompactionTask, TaskData};
use lance_table::transaction::{Operation, RewriteGroup, Transaction};

async fn initialized(path: &std::path::Path) -> Result<(AnnotationStaging, Arc<Dataset>)> {
    let mut base = (*fixture(path).await?).clone();
    base.create_index_builder(&["id"], IndexType::BTree, &ScalarIndexParams::default())
        .name("turn_id".into())
        .await?;
    let staging = AnnotationStaging::new(Arc::new(base), scope(), 0, Limits::default())?;
    staging
        .apply(
            0,
            1,
            "T",
            &TurnAnnotationPatch {
                status: Set("reviewed".into()),
                ..Default::default()
            },
        )
        .await?;
    let frozen = staging
        .freeze_with_results(uuid::Uuid::new_v4(), Some("request-results-at-1".into()))
        .await?
        .unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let base = commit(&prepared).await?;
    staging.install_published(&prepared, base.clone()).await?;
    Ok((staging, base))
}

pub(super) async fn compact(base: &Dataset) -> Result<Transaction> {
    let task = CompactionTask {
        task: TaskData {
            fragments: base.manifest().fragments.as_ref().clone(),
        },
        read_version: base.version_id(),
        options: CompactionOptions {
            target_rows_per_fragment: 3,
            num_threads: Some(1),
            batch_size: Some(1),
            compaction_mode: Some(CompactionMode::TryBinaryCopy),
            ..Default::default()
        },
    };
    let mut rewritten = task.execute(base).await?;
    assert_eq!(rewritten.original_fragments.len(), 2);
    assert_eq!(rewritten.new_fragments.len(), 1);
    // The managed publisher fixes these IDs against its fenced read version.
    for (ordinal, fragment) in rewritten.new_fragments.iter_mut().enumerate() {
        fragment.id = u64::from(base.manifest().max_fragment_id.unwrap()) + 1 + ordinal as u64;
    }
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: rewritten.original_fragments,
                new_fragments: rewritten.new_fragments,
            }],
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    );
    transaction.transaction_properties = base
        .read_transaction()
        .await?
        .unwrap()
        .transaction_properties;
    Ok(transaction)
}

pub(super) async fn select(base: Arc<Dataset>, transaction: &Transaction) -> Result<Arc<Dataset>> {
    Ok(Arc::new(
        CommitBuilder::new(base)
            .with_max_retries(0)
            .with_skip_auto_cleanup(true)
            .execute(transaction.clone())
            .await?,
    ))
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
async fn compacted_base_preserves_active_updates_old_readers_and_later_publication() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (staging, base) = initialized(&root.path().join("turns.lance")).await?;
    let held = staging.capture("U").await?;
    let held_t = staging.capture("T").await?;
    let transaction = compact(&base).await?;
    // The compaction has captured its source. A later successful update must
    // remain current when that older physical rewrite becomes the base.
    staging
        .apply(
            1,
            2,
            "T",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let before = staging.observation().await;
    let selected = select(base.clone(), &transaction).await?;
    staging
        .install_compacted(&transaction, selected.clone())
        .await?;
    assert_eq!(staging.observation().await, before);
    let current_t = current(&staging, "T").await?;
    assert_eq!(current_t.status.as_deref(), Some("reviewed"));
    assert_eq!(current_t.score, Some(0.9));
    assert_eq!(
        held.resolve().await?.unwrap().status.as_deref(),
        Some("pending")
    );
    assert_eq!(held_t.resolve().await?.unwrap().score, None);
    assert_eq!(
        versions(&staging).await?,
        vec![base.version_id(), selected.version_id()]
    );
    // The unchanged native ID index covers the replacement fragment.
    let indices = selected.load_indices().await?;
    assert_eq!(indices.len(), 1);
    assert!(indices[0]
        .fragment_bitmap
        .as_ref()
        .unwrap()
        .contains(selected.manifest().fragments[0].id as u32));
    assert_eq!(read_base(&selected, "T", 4096).await?.unwrap().score, None);
    let old_bodies = base
        .scan()
        .project(&["body"])?
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let bodies = selected
        .scan()
        .project(&["body"])?
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let schema = old_bodies[0].schema();
    assert_eq!(
        arrow_select::concat::concat_batches(&schema, &old_bodies)?,
        arrow_select::concat::concat_batches(&schema, &bodies)?
    );
    drop(held);
    drop(held_t);
    assert_eq!(versions(&staging).await?, vec![selected.version_id()]);
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    assert_eq!(prepared.base().version_id(), selected.version_id());
    let next = commit(&prepared).await?;
    staging.install_published(&prepared, next.clone()).await?;
    assert_eq!(current(&staging, "T").await?, current_t);
    assert_eq!(staging.observation().await.materialized, 2);
    let reopened = Dataset::open(next.uri()).await?;
    assert_eq!(read_base(&reopened, "T", 4096).await?, Some(current_t));
    Ok(())
}

#[tokio::test]
async fn compaction_cannot_drop_or_advance_annotation_recovery_metadata() -> Result<()> {
    let advanced_cut = serde_json::to_string(&(scope(), 2)).unwrap();
    for (key, replacement) in [
        ("lance-context.annotation-cut", None),
        ("lance-context.annotation-cut", Some(advanced_cut.as_str())),
        (RESULTS_PROPERTY, None),
        (RESULTS_PROPERTY, Some("different-results")),
    ] {
        let root = tempfile::tempdir()?;
        let (staging, base) = initialized(&root.path().join("turns.lance")).await?;
        let mut transaction = compact(&base).await?;
        let properties = Arc::make_mut(transaction.transaction_properties.as_mut().unwrap());
        if let Some(value) = replacement {
            properties.insert(key.into(), value.into());
        } else {
            properties.remove(key);
        }
        let selected = select(base, &transaction).await?;
        let before = staging.observation().await;
        assert!(staging
            .install_compacted(&transaction, selected)
            .await
            .is_err());
        assert_eq!(staging.observation().await, before);
        assert_eq!(
            current(&staging, "T").await?.status.as_deref(),
            Some("reviewed")
        );
    }
    Ok(())
}

#[tokio::test]
async fn compaction_cannot_replace_a_base_with_an_outstanding_frozen_generation() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (staging, base) = initialized(&root.path().join("turns.lance")).await?;
    let transaction = compact(&base).await?;
    staging
        .apply(
            1,
            2,
            "T",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let _frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let selected = select(base, &transaction).await?;
    let before = staging.observation().await;
    assert!(staging
        .install_compacted(&transaction, selected)
        .await
        .is_err());
    assert_eq!(staging.observation().await, before);
    assert_eq!(current(&staging, "T").await?.score, Some(0.9));
    Ok(())
}
