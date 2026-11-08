//! Reuse a known current value without tying its lifetime to unrelated versions.
use super::*;

#[tokio::test]
async fn cold_point_survives_unrelated_publication_without_more_data_io() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("turns.lance"), 8 << 20).await?;
    let staging = Arc::new(AnnotationStaging::new(base, scope(), 0, Limits::default())?);
    let expected = current(&staging, "T").await?;
    let original_version = staging.capture("T").await?.base_version;
    staging
        .apply(
            0,
            1,
            "U",
            &TurnAnnotationPatch {
                score: Set(0.7),
                ..Default::default()
            },
        )
        .await?;
    publish(&staging).await?;
    assert!(staging.capture("T").await?.base_version > original_version);

    let before = throttle.ops.load(Ordering::Relaxed);
    throttle.frozen.store(true, Ordering::Relaxed);
    let mut read = tokio::spawn({
        let staging = staging.clone();
        async move { current(&staging, "T").await }
    });
    let completed = tokio::time::timeout(Duration::from_secs(1), &mut read).await;
    throttle.frozen.store(false, Ordering::Relaxed);
    let reused = completed.is_ok();
    let value = match completed {
        Ok(result) => result,
        Err(_) => read.await,
    }
    .map_err(|e| invalid(e.to_string()))??;
    assert_eq!(value, expected);
    let additional_io = throttle.ops.load(Ordering::Relaxed) - before;
    println!("current_point_after_publication additional_data_operations={additional_io}");
    assert!(reused, "unchanged current point waited for blocked data IO");
    assert_eq!(additional_io, 0);
    assert_eq!(staging.observation().await.retained_bytes, 0);
    Ok(())
}

#[tokio::test]
async fn cold_point_supplies_following_patch_without_more_data_io() -> Result<()> {
    point_then_patch(false).await
}

#[tokio::test]
async fn recovered_cold_point_supplies_following_patch_without_more_data_io() -> Result<()> {
    point_then_patch(true).await
}

async fn point_then_patch(recover: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("turns.lance"), 8 << 20).await?;
    let mut staging = Arc::new(AnnotationStaging::new(base, scope(), 0, Limits::default())?);
    let after = if recover {
        staging
            .apply(
                0,
                1,
                "U",
                &TurnAnnotationPatch {
                    score: Set(0.7),
                    ..Default::default()
                },
            )
            .await?;
        publish(&staging).await?;
        let selected = staging.state.read().await.base.dataset.clone();
        staging = Arc::new(AnnotationStaging::new(
            selected,
            scope(),
            1,
            Limits::default(),
        )?);
        1
    } else {
        0
    };
    let expected = current(&staging, "T").await?;
    let before = throttle.ops.load(Ordering::Relaxed);
    throttle.frozen.store(true, Ordering::Relaxed);
    let mut apply = tokio::spawn({
        let staging = staging.clone();
        async move {
            staging
                .apply(
                    after,
                    after + 1,
                    "T",
                    &TurnAnnotationPatch {
                        score: Set(0.9),
                        ..Default::default()
                    },
                )
                .await
        }
    });
    let completed = tokio::time::timeout(Duration::from_secs(1), &mut apply).await;
    throttle.frozen.store(false, Ordering::Relaxed);
    let reused = completed.is_ok();
    match completed {
        Ok(result) => result,
        Err(_) => apply.await,
    }
    .map_err(|e| invalid(e.to_string()))??;
    let additional_io = throttle.ops.load(Ordering::Relaxed) - before;
    let value = current(&staging, "T").await?;
    assert_eq!(value.status, expected.status);
    assert_eq!(value.labels, expected.labels);
    assert_eq!(value.score, Some(0.9));
    assert_eq!(staging.observation().await.applied, after + 1);
    println!("current_point_then_patch additional_data_operations={additional_io}");
    assert!(reused, "following patch waited for blocked data IO");
    assert_eq!(additional_io, 0);
    Ok(())
}
