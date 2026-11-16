//! Real Lance IO and snapshot/admission checks for concurrent cold points.
use super::*;

#[tokio::test]
async fn distinct_cold_points_share_native_io_and_keep_missing_and_escaped_keys() -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("turns.lance");
    let ids = (0..64)
        .map(|n| format!("turn-{n}"))
        .chain(std::iter::once("o'hara".into()))
        .collect::<Vec<_>>();
    let batch = crate::annotation_staging::tests::point_scan::batch(ids.clone(), "pending")?;
    let mut columns = batch.columns().to_vec();
    columns[2] = Arc::new(StringArray::from(ids));
    let batch = RecordBatch::try_new(batch.schema(), columns)?;
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        path.to_str().unwrap(),
        None,
    )
    .await?;
    let (base, throttle) = open_existing(&path, 8 << 20).await?;
    let keys = (0..16)
        .map(|n| format!("turn-{n}"))
        .chain(["o'hara".into(), "missing".into()])
        .collect::<Vec<_>>();
    // Warm only native metadata. Each staging instance has its own value keys.
    let warmup = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    current(&warmup, &keys[0]).await?;
    let reference = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    let before = throttle.ops.load(Ordering::Relaxed);
    let mut expected = Vec::new();
    for key in &keys {
        expected.push(reference.capture(key).await?.resolve().await?);
    }
    let sequential = throttle.ops.load(Ordering::Relaxed) - before;
    assert_eq!(
        expected[16].as_ref().unwrap().status.as_deref(),
        Some("o'hara")
    );
    assert!(expected[17].is_none());

    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let before = throttle.ops.load(Ordering::Relaxed);
    let actual = futures::future::try_join_all(
        keys.iter()
            .map(|key| async { staging.capture(key).await?.resolve().await }),
    )
    .await?;
    let batched = throttle.ops.load(Ordering::Relaxed) - before;
    assert_eq!(actual, expected);
    println!("cold_points distinct_keys={} sequential_data_operations={sequential} batched_data_operations={batched}", keys.len());
    assert!(batched > 0 && batched * 4 < sequential);

    // The transparent batch loader still warms the original bounded caches.
    throttle.frozen.store(true, Ordering::Relaxed);
    let reused = tokio::time::timeout(Duration::from_secs(2), async {
        futures::future::try_join_all(
            keys.iter()
                .map(|key| async { staging.capture(key).await?.resolve().await }),
        )
        .await
    })
    .await;
    throttle.frozen.store(false, Ordering::Relaxed);
    assert_eq!(
        reused.map_err(|_| invalid("cached batch waited for data IO"))??,
        expected
    );
    assert_eq!(throttle.ops.load(Ordering::Relaxed) - before, batched);
    assert_eq!(staging.observation().await.readers, 0);
    Ok(())
}

#[tokio::test]
async fn cancelled_cold_points_keep_admission_and_gc_roots_through_native_io() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("turns.lance"), 0).await?;
    let staging = Arc::new(AnnotationStaging::new(
        base,
        scope(),
        0,
        Limits {
            readers: 3,
            ..Default::default()
        },
    )?);
    staging
        .apply(
            0,
            1,
            "T",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let selected = commit(&prepared).await?;
    // Cold old bases, not staged T: these reads must stay pinned after install.
    throttle.frozen.store(true, Ordering::Relaxed);
    let mut tasks = Vec::new();
    let check = async {
        for key in ["U", "o'hara"] {
            let before = throttle.ops.load(Ordering::Relaxed);
            let read = staging.capture(key).await?;
            tasks.push(tokio::spawn(async move { read.resolve().await }));
            tokio::time::timeout(Duration::from_secs(5), async {
                while throttle.ops.load(Ordering::Relaxed) == before {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .map_err(|_| invalid("cold scan did not enter data IO"))?;
        }
        let read = staging.capture("missing").await?;
        tasks.push(tokio::spawn(async move { read.resolve().await }));
        tokio::task::yield_now().await;
        for task in &tasks {
            task.abort();
        }
        for task in tasks.drain(..) {
            assert!(task.await.unwrap_err().is_cancelled());
        }
        assert_eq!(staging.observation().await.readers, 3);
        assert!(staging.capture("T").await.is_err());
        staging.install_published(&prepared, selected).await?;
        Ok::<_, Error>(())
    }
    .await;
    drop((frozen, prepared));
    let protected = staging.protected_bases().await?.datasets().count();
    throttle.frozen.store(false, Ordering::Relaxed);
    // A failed entry check must release/join the original tasks too.
    for task in tasks {
        task.abort();
        let _ = task.await;
    }
    check?;
    assert_eq!(
        protected, 2,
        "cancelled native scans lost their old base roots"
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while staging.observation().await.readers != 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .map_err(|_| invalid("completed/cancelled reads leaked admission"))?;
    assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    assert_eq!(current(&staging, "T").await?.score, Some(0.9));
    Ok(())
}

#[tokio::test]
async fn concurrent_cold_points_keep_old_and_new_publications_separate() -> Result<()> {
    for cache_bytes in [0, 8 << 20] {
        let root = tempfile::tempdir()?;
        let (base, _) = opened(&root.path().join("turns.lance"), cache_bytes).await?;
        let session = base.session().clone();
        let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
        let old = staging.capture("T").await?;
        staging
            .apply(
                0,
                1,
                "T",
                &TurnAnnotationPatch {
                    status: Set("reviewed".into()),
                    score: Set(0.9),
                    ..Default::default()
                },
            )
            .await?;
        publish(&staging).await?;
        let new = staging.capture("T").await?;
        session.file_metadata_cache().clear().await;
        let (before, after) = tokio::try_join!(old.resolve(), new.resolve())?;
        assert_eq!(before.unwrap().status.as_deref(), Some("pending"));
        let after = after.unwrap();
        assert_eq!(after.status.as_deref(), Some("reviewed"));
        assert_eq!(after.score, Some(0.9));
        assert_eq!(current(&staging, "T").await?, after);
        drop((old, new));
        assert_eq!(staging.observation().await.readers, 0);
        assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    }
    Ok(())
}

#[tokio::test]
async fn cold_point_batches_reject_duplicate_rows_without_caching_partial_values() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let batches = base
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let duplicate = Dataset::write(
        RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            Arc::new(Schema::from(base.schema())),
        ),
        base.uri(),
        Some(WriteParams {
            mode: lance::dataset::WriteMode::Append,
            ..Default::default()
        }),
    )
    .await?;
    let staging = AnnotationStaging::new(Arc::new(duplicate), scope(), 0, Limits::default())?;
    let results = futures::future::join_all(
        ["T", "U", "missing"]
            .into_iter()
            .map(|key| async { staging.capture(key).await?.resolve().await }),
    )
    .await;
    for result in results {
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("duplicate annotation turn key"));
    }
    assert!(staging.published_values.get("T").await.is_none());
    assert_eq!(staging.observation().await.readers, 0);
    Ok(())
}
