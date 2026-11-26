//! A body rewrite changes the selected body, not the annotation generation.
use super::*;
use lance::dataset::optimize::{CompactionMode, CompactionOptions, CompactionTask, TaskData};
use lance_table::transaction::{Operation, RewriteGroup, Transaction};

pub(super) async fn initialized(
    root: &std::path::Path,
) -> Result<(AnnotationStaging, Arc<Dataset>, Arc<Dataset>)> {
    let batch = body::imported(0, 2)?;
    let bodies = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            root.join("body.lance").to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: 1,
                enable_stable_row_ids: true,
                skip_auto_cleanup: true,
                ..Default::default()
            }),
        )
        .await?,
    );
    let columns = ["id", "rollout_id", "labels", "status", "score"]
        .map(|name| batch.schema().index_of(name).unwrap());
    let narrow = batch.project(&columns)?;
    let base = Dataset::write(
        RecordBatchIterator::new(vec![Ok(narrow.clone())], narrow.schema()),
        root.join("annotations.lance").to_str().unwrap(),
        Some(WriteParams {
            enable_stable_row_ids: true,
            skip_auto_cleanup: true,
            ..Default::default()
        }),
    )
    .await?;
    let staging = AnnotationStaging::new(Arc::new(base), scope(), 0, Limits::default())?;
    staging
        .apply(
            0,
            1,
            "turn-000000",
            &TurnAnnotationPatch {
                score: Set(0.4),
                ..Default::default()
            },
        )
        .await?;
    let frozen = staging
        .freeze_with_results(uuid::Uuid::new_v4(), Some("results-at-1".into()))
        .await?
        .unwrap();
    let prepared = frozen
        .prepare_with_body(
            &crate::MergeMemoryBudget::new(8 << 20),
            8192,
            BodySnapshot::capture(scope().table_instance, &bodies).await?,
        )
        .await?;
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    Ok((staging, selected, bodies))
}

async fn read_all(dataset: &Dataset) -> Result<RecordBatch> {
    let batches = dataset
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    Ok(arrow_select::concat::concat_batches(
        &batches[0].schema(),
        &batches,
    )?)
}

async fn rewrite(base: Arc<Dataset>) -> Result<Arc<Dataset>> {
    let mut result = CompactionTask {
        task: TaskData {
            fragments: base.manifest().fragments.as_ref().clone(),
        },
        read_version: base.version_id(),
        options: CompactionOptions {
            target_rows_per_fragment: 16,
            num_threads: Some(1),
            batch_size: Some(2),
            compaction_mode: Some(CompactionMode::TryBinaryCopy),
            ..Default::default()
        },
    }
    .execute(&base)
    .await?;
    assert_eq!(result.original_fragments.len(), 2);
    assert_eq!(result.new_fragments.len(), 1);
    result.new_fragments[0].id = u64::from(base.manifest().max_fragment_id.unwrap()) + 1;
    let transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: result.original_fragments,
                new_fragments: result.new_fragments,
            }],
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    );
    let rewritten = Arc::new(
        CommitBuilder::new(base.clone())
            .with_detached(true)
            .with_max_retries(1)
            .with_skip_auto_cleanup(true)
            .execute(transaction)
            .await?,
    );
    assert_eq!(read_all(&base).await?, read_all(&rewritten).await?);
    Ok(rewritten)
}

async fn metadata(base: &Dataset, body: &Dataset) -> Result<Transaction> {
    let reference = BodySnapshot::capture(scope().table_instance, body).await?;
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: Default::default(),
        },
    );
    transaction.uuid = reference.transaction.clone();
    transaction.transaction_properties = base
        .read_transaction()
        .await?
        .unwrap()
        .transaction_properties;
    Arc::make_mut(transaction.transaction_properties.as_mut().unwrap()).insert(
        BODY_SNAPSHOT_PROPERTY.into(),
        serde_json::to_string(&reference)?,
    );
    Ok(transaction)
}

#[tokio::test]
async fn body_compaction_keeps_active_updates_old_readers_and_later_publication() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (staging, base, body) = initialized(root.path()).await?;
    let held = staging.capture_turn("turn-000000").await?;
    let rewritten = rewrite(body.clone()).await?;
    let transaction = metadata(&base, &rewritten).await?;
    staging
        .apply(
            1,
            2,
            "turn-000000",
            &TurnAnnotationPatch {
                status: Set("reviewed".into()),
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let before = staging.observation().await;
    let selected = compaction::select(base.clone(), &transaction).await?;
    staging
        .install_body_compacted(&transaction, selected.clone())
        .await?;
    let mut before = before;
    before.installed_base.version = selected.version_id();
    assert_eq!(staging.observation().await, before);
    assert_eq!(selected.manifest().fragments, base.manifest().fragments);
    assert_eq!(read_all(&selected).await?, read_all(&base).await?);
    assert_eq!(
        held.annotations().resolve().await?.unwrap().score,
        Some(0.4)
    );
    let TurnBodySource::Materialized(old) = held.body_source() else {
        panic!("body is materialized")
    };
    assert_eq!(
        BodySnapshot::read(old).await?.unwrap().version,
        body.version_id()
    );
    let current = staging.capture_turn("turn-000000").await?;
    assert_eq!(
        current.annotations().resolve().await?.unwrap().score,
        Some(0.9)
    );
    let TurnBodySource::Materialized(new) = current.body_source() else {
        panic!("body is materialized")
    };
    assert_eq!(
        BodySnapshot::read(new).await?.unwrap().version,
        rewritten.version_id()
    );
    let versions = || async {
        staging
            .protected_bases()
            .await
            .map(|p| p.datasets().map(Dataset::version_id).collect::<Vec<_>>())
    };
    assert_eq!(
        versions().await?,
        vec![base.version_id(), selected.version_id()]
    );
    drop(held);
    assert_eq!(versions().await?, vec![selected.version_id()]);
    drop(current);
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
    let published = commit(&prepared).await?;
    staging
        .install_published(&prepared, published.clone())
        .await?;
    let reopened = Dataset::open(published.uri()).await?;
    let value = read_base(&reopened, "turn-000000", 4096).await?.unwrap();
    assert_eq!(value.status.as_deref(), Some("reviewed"));
    assert_eq!(value.score, Some(0.9));
    assert_eq!(
        BodySnapshot::read(&reopened).await?,
        BodySnapshot::read(&selected).await?
    );
    assert_eq!(staging.observation().await.materialized, 2);
    Ok(())
}

#[tokio::test]
async fn body_compaction_rejects_changed_recovery_metadata_and_foreign_snapshots() -> Result<()> {
    for invalid in 0..9 {
        let root = tempfile::tempdir()?;
        let (staging, base, body) = initialized(root.path()).await?;
        let rewritten = rewrite(body.clone()).await?;
        let mut transaction = metadata(&base, &rewritten).await?;
        let properties = Arc::make_mut(transaction.transaction_properties.as_mut().unwrap());
        match invalid {
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
                properties.insert(RESULTS_PROPERTY.into(), "changed-results".into());
            }
            3 => {
                properties.remove(BODY_SNAPSHOT_PROPERTY);
            }
            4..=7 => {
                let mut reference =
                    BodySnapshot::capture(scope().table_instance, &rewritten).await?;
                match invalid {
                    4 => reference.uri.push_str("-foreign"),
                    5 => reference.version = body.version_id(),
                    6 => reference.version = 2,
                    7 => reference.transaction = uuid::Uuid::new_v4().to_string(),
                    _ => unreachable!(),
                }
                properties.insert(
                    BODY_SNAPSHOT_PROPERTY.into(),
                    serde_json::to_string(&reference)?,
                );
            }
            8 => {
                staging
                    .apply(
                        1,
                        2,
                        "turn-000000",
                        &TurnAnnotationPatch {
                            status: Clear,
                            ..Default::default()
                        },
                    )
                    .await?;
                let _frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
            }
            _ => unreachable!(),
        }
        let before = staging.observation().await;
        let expected = current(&staging, "turn-000000").await?;
        let selected = compaction::select(base.clone(), &transaction).await?;
        assert!(
            staging
                .install_body_compacted(&transaction, selected)
                .await
                .is_err(),
            "case {invalid}"
        );
        assert_eq!(staging.observation().await, before);
        assert_eq!(current(&staging, "turn-000000").await?, expected);
        let held = staging.capture_turn("turn-000000").await?;
        let TurnBodySource::Materialized(unchanged) = held.body_source() else {
            panic!("body is materialized")
        };
        assert_eq!(unchanged.version_id(), base.version_id());
    }
    Ok(())
}
