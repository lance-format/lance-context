use super::*;
use lance_table::transaction::{Operation, Transaction};

#[tokio::test]
async fn mixed_status_tail_bounds_takes_and_keeps_captured_values() -> Result<()> {
    for stable in [false, true] {
        let root = tempfile::tempdir()?;
        let uri = root.path().join("turns.lance");
        let id = |row: usize| format!("{}-{row:03}", "\u{0001}".repeat(80));
        let batch = |start: usize| -> Result<RecordBatch> {
            let mut labels = ListBuilder::new(StringBuilder::new());
            for _ in 0..64 {
                labels.append(true);
            }
            let labels = labels.finish();
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("body", DataType::LargeBinary, false),
                Field::new("labels", labels.data_type().clone(), false),
                Field::new("status", DataType::Utf8, true),
                Field::new("score", DataType::Float32, true),
            ]));
            Ok(RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from_iter_values((start..start + 64).map(id))),
                    Arc::new(LargeBinaryArray::from_iter_values((0..64).map(|_| b"body"))),
                    Arc::new(labels),
                    Arc::new(StringArray::from(vec!["pending"; 64])),
                    Arc::new(Float32Array::from(vec![None; 64])),
                ],
            )?)
        };
        let first = batch(0)?;
        let mut base = Dataset::write(
            RecordBatchIterator::new(vec![Ok(first.clone())], first.schema()),
            uri.to_str().unwrap(),
            Some(WriteParams {
                enable_stable_row_ids: stable,
                max_rows_per_file: 16,
                ..Default::default()
            }),
        )
        .await?;
        base.create_index(
            &["status"],
            IndexType::BTree,
            Some("test_status".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await?;
        let tail = batch(64)?;
        let base = Dataset::write(
            RecordBatchIterator::new(vec![Ok(tail.clone())], tail.schema()),
            uri.to_str().unwrap(),
            Some(WriteParams {
                mode: lance::dataset::WriteMode::Append,
                enable_stable_row_ids: stable,
                max_rows_per_file: 16,
                ..Default::default()
            }),
        )
        .await?;
        let staging = AnnotationStaging::new(Arc::new(base), scope(), 0, Limits::default())?;
        let held = staging
            .capture_status(Some("pending".into()), 128, 128 << 10)
            .await?;
        let before = held.resolve().await?;
        assert_eq!(
            before.ids.iter().cloned().collect::<BTreeSet<_>>(),
            (0..128).map(id).collect()
        );
        assert!(!before.truncated);
        assert!(before.indexed_fragments > 0 && before.unindexed_fragments > 0);

        let mut bounded = staging
            .capture_status(Some("pending".into()), 70, 64 << 10)
            .await?;
        while bounded.reduce_scan_batch() {}
        let (matches, observed) = bounded.resolve_observed().await;
        let matches = matches?;
        assert_eq!(matches.ids.len(), 70);
        assert!(matches.truncated);
        assert_eq!(matches.ids.iter().collect::<BTreeSet<_>>().len(), 70);
        assert!(matches.ids.iter().all(|key| before.ids.contains(key)));
        assert!(observed["id_take_batches"].as_u64().unwrap() >= 3);
        assert!(observed["native_at_eof"].is_null());
        let bytes = staging
            .capture_status(Some("pending".into()), 128, 1024)
            .await?
            .resolve()
            .await?;
        assert!(bytes.truncated && !bytes.ids.is_empty() && bytes.ids.len() < 128);
        assert!(serde_json::to_vec(&bytes).unwrap().len() <= 1024);
        let missing = staging
            .capture_status(Some("missing".into()), 128, 64 << 10)
            .await?;
        let (matches, observed) = missing.resolve_observed().await;
        assert!(matches?.ids.is_empty());
        assert_eq!(observed["id_take_batches"], 0);
        assert!(observed["native_at_eof"].is_object());

        staging
            .apply(
                0,
                1,
                &id(64),
                &TurnAnnotationPatch {
                    status: Clear,
                    ..Default::default()
                },
            )
            .await?;
        drop(flush(&staging).await?);
        let after = staging
            .capture_status(Some("pending".into()), 128, 128 << 10)
            .await?
            .resolve()
            .await?;
        assert_eq!(after.ids.len(), 127);
        assert!(!after.ids.contains(&id(64)));
        assert_eq!(query(&staging, None).await?.ids, vec![id(64)]);
        assert_eq!(held.resolve().await?, before);
        assert!(staging
            .protected_bases()
            .await?
            .datasets()
            .any(|base| base.version_id() == before.base_version));
    }
    Ok(())
}

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
async fn status_scan_observation_preserves_results_and_does_not_drain_truncated_scans() -> Result<()>
{
    let root = tempfile::tempdir()?;
    let base = fixture_with_row_ids(&root.path().join("turns.lance"), true).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    for (status, limit, rows, truncated) in [
        ("pending", 16, 3, false),
        ("absent", 16, 0, false),
        ("pending", 1, 1, true),
    ] {
        let read = staging
            .capture_status(Some(status.into()), limit, 64 << 10)
            .await?;
        let expected = read.resolve().await?;
        let (actual, observation) = read.resolve_observed().await;
        let actual = actual?;
        assert_eq!(actual, expected);
        assert_eq!(actual.ids.len(), rows);
        assert_eq!(actual.truncated, truncated);
        assert_eq!(observation["returned_ids"], rows);
        assert_eq!(observation["truncated"], truncated);
        assert_eq!(observation["native_at_eof"].is_null(), truncated);
        let before = observation["before_open_elapsed_us"].as_u64().unwrap();
        let after = observation["after_open_elapsed_us"].as_u64().unwrap();
        assert!(before <= after);
        assert!(after <= observation["elapsed_us"].as_u64().unwrap());
        // The diagnostic contains only aggregate counts, never turn IDs or
        // predicate values, and it does not alter the captured read view.
        assert!(!observation.to_string().contains("pending"));
        assert!(!observation.to_string().contains("o'hara"));
    }
    Ok(())
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
    let (observed, diagnostic) = held.resolve_observed().await;
    assert_eq!(observed?, old);
    assert!(diagnostic["native_at_eof"].is_object());
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
    let empty = staging.capture_status(None, 16, 64 << 10).await?;
    let (observed, diagnostic) = empty.resolve_observed().await;
    assert!(observed?.ids.is_empty());
    assert_eq!(diagnostic["indexed_fragments"], 1);
    assert_eq!(diagnostic["unindexed_fragments"], 1);
    assert!(diagnostic["native_at_eof"].is_object());
    drop(empty);
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
