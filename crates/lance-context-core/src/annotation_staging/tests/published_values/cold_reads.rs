//! Repeated reads reuse only values from their captured immutable snapshot.
use super::*;

#[tokio::test]
async fn repeated_cold_point_reads_avoid_data_io_within_the_same_snapshot() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("turns.lance"), 8 << 20).await?;
    let staging = Arc::new(AnnotationStaging::new(base, scope(), 0, Limits::default())?);
    let expected = current(&staging, "T").await?;
    assert!(staging.capture("missing").await?.resolve().await?.is_none());
    let before = throttle.ops.load(Ordering::Relaxed);
    assert!(
        before > 0,
        "first reads must use the actual native data store"
    );

    throttle.frozen.store(true, Ordering::Relaxed);
    let mut reads = tokio::spawn({
        let staging = staging.clone();
        async move {
            for _ in 0..8 {
                assert_eq!(current(&staging, "T").await?, expected);
                assert!(staging.capture("missing").await?.resolve().await?.is_none());
            }
            Ok::<_, Error>(())
        }
    });
    let completed = tokio::time::timeout(Duration::from_secs(1), &mut reads).await;
    throttle.frozen.store(false, Ordering::Relaxed);
    let reused = completed.is_ok();
    // Join the original task even in the negative control; release native IO first.
    let result = match completed {
        Ok(result) => result,
        Err(_) => reads.await,
    };
    result.map_err(|e| invalid(e.to_string()))??;
    let after = throttle.ops.load(Ordering::Relaxed);
    println!(
        "cold_point_repeat reads=16 additional_data_operations={}",
        after - before
    );
    assert!(reused, "repeated point reads waited for blocked data IO");
    assert_eq!(after, before);
    let observed = staging.observation().await;
    assert_eq!(observed.applied, 0);
    assert_eq!(observed.retained_bytes, 0);
    assert_eq!(observed.retained_entries, 0);
    Ok(())
}

#[tokio::test]
async fn delayed_cold_read_cannot_poison_a_newly_published_snapshot() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("turns.lance"), 8 << 20).await?;
    let session = base.session().clone();
    let staging = Arc::new(AnnotationStaging::new(base, scope(), 0, Limits::default())?);
    let old = Arc::new(staging.capture("T").await?);
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
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let selected = commit(&prepared).await?;

    let before = throttle.ops.load(Ordering::Relaxed);
    throttle.frozen.store(true, Ordering::Relaxed);
    let pending = tokio::spawn({
        let old = old.clone();
        async move { old.resolve().await }
    });
    let concurrent = async {
        tokio::time::timeout(Duration::from_secs(5), async {
            while throttle.ops.load(Ordering::Relaxed) == before {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .map_err(|_| invalid("old read did not enter native IO"))?;
        staging.install_published(&prepared, selected).await?;
        assert_eq!(current(&staging, "T").await?.score, Some(0.9));
        // Remove the warmed current value. The delayed old read will finish
        // after this clear, exercising the cold path for the new base too.
        session.file_metadata_cache().clear().await;
        Ok::<_, Error>(())
    }
    .await;
    throttle.frozen.store(false, Ordering::Relaxed);
    let old_value = pending.await.map_err(|e| invalid(e.to_string()))??;
    concurrent?;
    let old_value = old_value.unwrap();
    assert_eq!(old_value.status.as_deref(), Some("pending"));
    assert_eq!(old_value.score, None);
    // Populate after clear even if the cache invalidated the in-flight loader.
    assert_eq!(old.resolve().await?, Some(old_value));
    let current = current(&staging, "T").await?;
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));
    drop((old, prepared, frozen));
    assert_eq!(staging.observation().await.retained_bytes, 0);
    assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    Ok(())
}

#[tokio::test]
async fn cold_read_cache_eviction_and_disabled_cache_keep_native_fallback() -> Result<()> {
    for cache_bytes in [0, 8 << 20] {
        let root = tempfile::tempdir()?;
        let (base, throttle) = opened(&root.path().join("turns.lance"), cache_bytes).await?;
        let session = base.session().clone();
        let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
        let expected = current(&staging, "T").await?;
        let before = throttle.ops.load(Ordering::Relaxed);
        assert_eq!(current(&staging, "T").await?, expected);
        if cache_bytes == 0 {
            assert!(throttle.ops.load(Ordering::Relaxed) > before);
        } else {
            assert_eq!(throttle.ops.load(Ordering::Relaxed), before);
        }
        session.file_metadata_cache().clear().await;
        let before = throttle.ops.load(Ordering::Relaxed);
        assert_eq!(current(&staging, "T").await?, expected);
        assert!(throttle.ops.load(Ordering::Relaxed) > before);
    }
    Ok(())
}

#[tokio::test]
async fn concurrent_cold_reads_share_one_native_lookup() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("turns.lance"), 8 << 20).await?;
    let reference = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    let before = throttle.ops.load(Ordering::Relaxed);
    let expected = current(&reference, "T").await?;
    let single_read_ops = throttle.ops.load(Ordering::Relaxed) - before;
    assert!(single_read_ops > 0);

    // A separate staging instance has a distinct value-cache namespace. Keep
    // the same native Dataset/session so file metadata is already warm.
    let staging = Arc::new(AnnotationStaging::new(base, scope(), 0, Limits::default())?);
    let before = throttle.ops.load(Ordering::Relaxed);
    throttle.frozen.store(true, Ordering::Relaxed);
    let reads = (0..16)
        .map(|_| {
            let staging = staging.clone();
            tokio::spawn(async move { current(&staging, "T").await })
        })
        .collect::<Vec<_>>();
    let entered = tokio::time::timeout(Duration::from_secs(5), async {
        while throttle.ops.load(Ordering::Relaxed) == before
            || staging.observation().await.readers != 16
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    throttle.frozen.store(false, Ordering::Relaxed);
    let results = futures::future::join_all(reads).await;
    entered.map_err(|_| invalid("concurrent reads did not reach actual cold IO"))?;
    for result in results {
        assert_eq!(result.map_err(|e| invalid(e.to_string()))??, expected);
    }
    let concurrent_ops = throttle.ops.load(Ordering::Relaxed) - before;
    println!("cold_point_concurrent readers=16 single_data_operations={single_read_ops} concurrent_data_operations={concurrent_ops}");
    assert!(concurrent_ops <= single_read_ops);
    assert_eq!(staging.observation().await.readers, 0);
    Ok(())
}
