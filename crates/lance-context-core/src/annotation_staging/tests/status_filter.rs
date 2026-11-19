use super::*;
use lance_table::transaction::{Operation, Transaction};

async fn flush(staging: &AnnotationStaging) -> Result<Arc<Dataset>> {
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    publish(staging, &frozen).await
}

async fn publish(staging: &AnnotationStaging, frozen: &FrozenAnnotations) -> Result<Arc<Dataset>> {
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let base = commit(&prepared).await?;
    staging.install_published(&prepared, base.clone()).await?;
    Ok(base)
}

async fn query(staging: &AnnotationStaging, status: Option<&str>) -> Result<StatusMatches> {
    staging
        .capture_status(status.map(str::to_owned), 16, 64 << 10)
        .await?
        .resolve()
        .await
}

#[tokio::test]
async fn status_filter_pins_snapshot_and_reports_lag_through_publication_and_index_invalidation(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let initial = fixture_with_row_ids(&root.path().join("turns.lance"), true).await?;
    let staging = AnnotationStaging::new(initial, scope(), 0, Limits::default())?;
    staging
        .apply(
            0,
            1,
            "T",
            &TurnAnnotationPatch {
                score: Set(0.1),
                ..Default::default()
            },
        )
        .await?;
    let base = flush(&staging).await?;
    let index = base
        .as_ref()
        .clone()
        .create_index_builder(&["status"], IndexType::BTree, &ScalarIndexParams::default())
        .name("test_status".into())
        .execute_uncommitted()
        .await?;
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::CreateIndex {
            new_indices: vec![index],
            removed_indices: vec![],
        },
    );
    transaction.transaction_properties = base
        .read_transaction()
        .await?
        .unwrap()
        .transaction_properties;
    let base = compaction::select(base, &transaction).await?;
    staging.install_indexed(&transaction, base).await?;

    let literal = "reviewed' OR status IS NOT NULL --";
    staging
        .apply(
            1,
            2,
            "T",
            &TurnAnnotationPatch {
                status: Set(literal.into()),
                ..Default::default()
            },
        )
        .await?;
    let held = staging
        .capture_status(Some("pending".into()), 16, 64 << 10)
        .await?;
    let old = held.resolve().await?;
    assert_eq!(old.ids.len(), 3);
    assert!(!old.truncated);
    assert_eq!(
        (old.materialized_through, old.applied_through_at_capture),
        (1, 2)
    );
    assert_eq!(
        (
            old.indexed_fragments,
            old.unindexed_fragments,
            old.unindexed_physical_rows
        ),
        (2, 0, Some(0))
    );
    assert_eq!(
        current(&staging, "T").await?.status.as_deref(),
        Some(literal)
    );

    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    staging
        .apply(
            2,
            3,
            "U",
            &TurnAnnotationPatch {
                status: Clear,
                ..Default::default()
            },
        )
        .await?;
    drop(publish(&staging, &frozen).await?);
    drop(frozen);
    let current_snapshot = query(&staging, Some(literal)).await?;
    assert_eq!(current_snapshot.ids, vec!["T"]);
    assert_eq!(
        (
            current_snapshot.materialized_through,
            current_snapshot.applied_through_at_capture
        ),
        (2, 3)
    );
    assert_eq!(
        (
            current_snapshot.indexed_fragments,
            current_snapshot.unindexed_fragments,
            current_snapshot.unindexed_physical_rows
        ),
        (1, 1, Some(2))
    );
    assert!(query(&staging, None).await?.ids.is_empty());
    assert_eq!(current(&staging, "U").await?.status, None);
    drop(flush(&staging).await?);
    assert_eq!(query(&staging, None).await?.ids, vec!["U"]);
    assert_eq!(query(&staging, Some("pending")).await?.ids, vec!["o'hara"]);
    assert_eq!(held.resolve().await?, old);
    assert!(staging
        .protected_bases()
        .await?
        .datasets()
        .any(|d| d.version_id() == old.base_version));
    drop(held);
    assert!(!staging
        .protected_bases()
        .await?
        .datasets()
        .any(|d| d.version_id() == old.base_version));
    Ok(())
}

#[tokio::test]
async fn status_filter_bounds_rows_escaped_response_bytes_and_retained_readers() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("source.lance")).await?;
    let batches = base
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let mut sequence = 0;
    let batches = batches
        .into_iter()
        .map(|batch| {
            let ids = (0..batch.num_rows())
                .map(|_| {
                    sequence += 1;
                    format!("{}-{sequence}", "\u{0001}".repeat(80))
                })
                .collect::<Vec<_>>();
            let mut columns = batch.columns().to_vec();
            columns[0] = Arc::new(StringArray::from(ids));
            RecordBatch::try_new(batch.schema(), columns)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let schema = batches[0].schema();
    let base = Dataset::write(
        RecordBatchIterator::new(batches.into_iter().map(Ok), schema),
        root.path().join("turns.lance").to_str().unwrap(),
        None,
    )
    .await?;
    let staging = AnnotationStaging::new(
        Arc::new(base),
        scope(),
        0,
        Limits {
            readers: 1,
            ..Default::default()
        },
    )?;
    assert!(staging.capture_status(None, 0, 1024).await.is_err());
    assert!(staging
        .capture_status(None, MAX_STATUS_MATCHES + 1, 1024)
        .await
        .is_err());
    let read = staging
        .capture_status(Some("pending".into()), 16, 1024)
        .await?;
    assert!(staging.capture_status(None, 1, 1024).await.is_err());
    let bounded = read.resolve().await?;
    assert!(bounded.truncated && !bounded.ids.is_empty() && bounded.ids.len() < 3);
    assert!(::serde_json::to_vec(&bounded).unwrap().len() <= 1024);
    drop(read);
    let mut read = staging
        .capture_status(Some("pending".into()), 1, 64 << 10)
        .await?;
    // A small caller budget must still serve the same captured query, while
    // preserving the response bound and refusing an unusable workspace.
    assert!(read.cap_scan_workspace(1024).is_err());
    read.cap_scan_workspace(8 << 20)?;
    assert!(read.workspace_bytes()? <= 8 << 20);
    let limited = read.resolve().await?;
    assert_eq!(limited.ids.len(), 1);
    assert!(limited.truncated);
    Ok(())
}

#[tokio::test]
async fn status_filter_preserves_union_and_global_lookahead_across_tail_fragments() -> Result<()> {
    for stable in [false, true] {
        for indexed in [false, true] {
            let root = tempfile::tempdir()?;
            let seed = super::point_scan::batch(vec!["seed".into()], "shared")?;
            let params = WriteParams {
                enable_stable_row_ids: stable,
                max_rows_per_file: 512,
                ..Default::default()
            };
            let mut base = Dataset::write(
                RecordBatchIterator::new(vec![Ok(seed.clone())], seed.schema()),
                root.path().join("turns.lance").to_str().unwrap(),
                Some(params.clone()),
            )
            .await?;
            if indexed {
                base.create_index_builder(
                    &["status"],
                    IndexType::BTree,
                    &ScalarIndexParams::default(),
                )
                .name("status".into())
                .await?;
            }
            let tail = super::point_scan::batch(
                (0..1536).map(|id| format!("tail-{id}")).collect(),
                "tail",
            )?;
            let last = super::point_scan::batch(vec!["late-shared".into()], "shared")?;
            base.append(
                RecordBatchIterator::new(vec![Ok(tail), Ok(last)], seed.schema()),
                Some(params),
            )
            .await?;
            let staging = AnnotationStaging::new(
                Arc::new(base),
                scope(),
                0,
                Limits {
                    value_bytes: 4096,
                    ..Default::default()
                },
            )?;
            let run = |status: &str, limit, budget| {
                let status = status.to_owned();
                let staging = &staging;
                async move {
                    let mut read = staging
                        .capture_status(Some(status), limit, 64 << 10)
                        .await?;
                    read.cap_scan_workspace(budget)?;
                    assert!(read.workspace_bytes()? <= budget);
                    read.resolve().await
                }
            };
            for budget in [64 << 20, (4096 * 2 * 32) + (128 << 10)] {
                // The smaller budget exercises the original sequential path;
                // the larger budget admits bounded adaptive prefetch.
                let shared = run("shared", 16, budget).await?;
                assert_eq!(shared.indexed_fragments, usize::from(indexed));
                assert!(shared.unindexed_fragments >= 4);
                assert_eq!(
                    shared.ids.into_iter().collect::<BTreeSet<_>>(),
                    BTreeSet::from(["seed".into(), "late-shared".into()])
                );
                assert!(!shared.truncated);
                let capped_shared = run("shared", 1, budget).await?;
                assert_eq!(capped_shared.ids.len(), 1);
                assert!(capped_shared.truncated);
                for limit in [64, 512, MAX_STATUS_MATCHES] {
                    let tail = run("tail", limit, budget).await?;
                    assert_eq!(tail.ids.len(), limit);
                    assert!(tail.truncated);
                    let ids = tail.ids.into_iter().collect::<BTreeSet<_>>();
                    assert_eq!(ids.len(), limit);
                    assert!(ids.iter().all(|id| id.starts_with("tail-")));
                }
                let absent = run("absent", 16, budget).await?;
                assert!(absent.ids.is_empty() && !absent.truncated);
            }
        }
    }
    Ok(())
}
