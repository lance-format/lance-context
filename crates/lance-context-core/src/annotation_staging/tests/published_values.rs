//! Published-value reuse must preserve fields and release dirty-state capacity.
use super::*;
use crate::preparation_io::test_support::{Throttle, ThrottleWrapper};
use lance::{dataset::builder::DatasetBuilder, dataset::ReadParams, session::Session};
use lance_io::object_store::ObjectStoreParams;
use std::{sync::atomic::Ordering, time::Duration};

mod cold_reads;
mod reads;

async fn opened(path: &std::path::Path, cache_bytes: usize) -> Result<(Arc<Dataset>, Throttle)> {
    fixture(path).await?;
    let throttle = Throttle::new(Duration::ZERO);
    let base = DatasetBuilder::from_uri(format!("file-object-store://{}", path.display()))
        .with_session(Arc::new(Session::new(
            1 << 20,
            cache_bytes,
            Default::default(),
        )))
        .with_read_params(ReadParams {
            store_options: Some(ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(ThrottleWrapper(throttle.clone()))),
                ..Default::default()
            }),
            ..Default::default()
        })
        .load()
        .await?;
    Ok((Arc::new(base), throttle))
}

async fn publish(staging: &AnnotationStaging) -> Result<()> {
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await
}

#[tokio::test]
async fn published_values_avoid_cold_apply_io_without_retaining_staging_permits() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("turns.lance"), 8 << 20).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let reviewed = TurnAnnotationPatch {
        status: Set("reviewed".into()),
        ..Default::default()
    };
    staging
        .apply_batch(
            0,
            1,
            &[
                Mutation {
                    turn: "T",
                    patch: &reviewed,
                },
                Mutation {
                    turn: "U",
                    patch: &reviewed,
                },
            ],
        )
        .await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    staging
        .apply(
            1,
            2,
            "T",
            &TurnAnnotationPatch {
                score: Set(0.7),
                ..Default::default()
            },
        )
        .await?;
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    drop((prepared, frozen));
    assert_eq!(current(&staging, "T").await?.score, Some(0.7));
    publish(&staging).await?;
    let view = staging.observation().await;
    assert_eq!(view.retained_bytes, 0);
    assert_eq!(view.retained_entries, 0);
    assert_eq!(view.active_entries + view.frozen_entries, 0);

    let before = throttle.ops.load(Ordering::Relaxed);
    throttle.frozen.store(true, Ordering::Relaxed);
    let score = TurnAnnotationPatch {
        score: Set(0.9),
        ..Default::default()
    };
    let clear = TurnAnnotationPatch {
        status: Clear,
        ..Default::default()
    };
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        staging.apply_batch(
            2,
            3,
            &[
                Mutation {
                    turn: "T",
                    patch: &score,
                },
                Mutation {
                    turn: "U",
                    patch: &clear,
                },
            ],
        ),
    )
    .await;
    throttle.frozen.store(false, Ordering::Relaxed);
    result.map_err(|_| invalid("published-value apply unexpectedly waited for data IO"))??;
    assert_eq!(throttle.ops.load(Ordering::Relaxed), before);
    let t = current(&staging, "T").await?;
    assert_eq!(t.status.as_deref(), Some("reviewed"));
    assert_eq!(t.score, Some(0.9));
    assert_eq!(t.labels, BTreeSet::from(["original".into()]));
    let u = current(&staging, "U").await?;
    assert_eq!(u.status, None);
    assert_eq!(u.score, None);
    Ok(())
}

#[tokio::test]
async fn eviction_and_disabled_cache_fall_back_to_the_selected_lance_values() -> Result<()> {
    for cache_bytes in [0, 8 << 20] {
        let root = tempfile::tempdir()?;
        let (base, throttle) = opened(&root.path().join("turns.lance"), cache_bytes).await?;
        let session = base.session().clone();
        let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
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
        publish(&staging).await?;
        assert_eq!(
            staging.published_values.get("T").await.is_some(),
            cache_bytes > 0
        );
        session.file_metadata_cache().clear().await;
        assert!(staging.published_values.get("T").await.is_none());
        let before = throttle.ops.load(Ordering::Relaxed);
        let published = current(&staging, "T").await?;
        assert_eq!(published.status.as_deref(), Some("reviewed"));
        assert_eq!(published.score, None);
        assert!(throttle.ops.load(Ordering::Relaxed) > before);
        session.file_metadata_cache().clear().await;
        let before = throttle.ops.load(Ordering::Relaxed);
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
        assert!(throttle.ops.load(Ordering::Relaxed) > before);
        let value = current(&staging, "T").await?;
        assert_eq!(value.status.as_deref(), Some("reviewed"));
        assert_eq!(value.score, Some(0.9));
        assert_eq!(value.labels, BTreeSet::from(["original".into()]));
    }
    Ok(())
}

#[tokio::test]
async fn independent_views_sharing_a_session_do_not_reuse_each_others_values() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, _) = opened(&root.path().join("turns.lance"), 8 << 20).await?;
    let first = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    first
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
    publish(&first).await?;
    let old = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    assert_eq!(current(&old, "T").await?.status.as_deref(), Some("pending"));
    old.apply(
        0,
        1,
        "T",
        &TurnAnnotationPatch {
            score: Set(0.9),
            ..Default::default()
        },
    )
    .await?;
    let value = current(&old, "T").await?;
    assert_eq!(value.status.as_deref(), Some("pending"));
    assert_eq!(value.score, Some(0.9));
    assert_eq!(
        current(&first, "T").await?.status.as_deref(),
        Some("reviewed")
    );
    Ok(())
}

#[tokio::test]
async fn cancelled_cache_warming_preserves_frozen_suffix_and_old_reads() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, _) = opened(&root.path().join("turns.lance"), 8 << 20).await?;
    let original_version = base.version_id();
    let staging = Arc::new(AnnotationStaging::new(base, scope(), 0, Limits::default())?);
    let old_read = staging.capture("T").await?;
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
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = Arc::new(
        frozen
            .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
            .await?,
    );
    let selected = commit(&prepared).await?;
    // Let installation warm the cache, then hold its exclusive view handoff.
    // Aborting here exercises cancellation after a real cache insertion.
    let held = staging.state.read().await;
    let install = tokio::spawn({
        let staging = staging.clone();
        let prepared = prepared.clone();
        let selected = selected.clone();
        async move { staging.install_published(&prepared, selected).await }
    });
    let warmed = tokio::time::timeout(Duration::from_secs(5), async {
        while staging.published_values.get("T").await.is_none() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(!install.is_finished());
    install.abort();
    let cancelled = install.await;
    drop(held);
    warmed.map_err(|_| invalid("installation did not warm the cache"))?;
    assert!(cancelled.unwrap_err().is_cancelled());
    assert_eq!(staging.observation().await.materialized, 0);
    assert_eq!(staging.capture("T").await?.base_version, original_version);
    assert_eq!(
        old_read.resolve().await?.unwrap().status.as_deref(),
        Some("pending")
    );
    assert_eq!(
        current(&staging, "T").await?.status.as_deref(),
        Some("reviewed")
    );

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
    staging.install_published(&prepared, selected).await?;
    drop((prepared, frozen));
    assert_eq!(current(&staging, "T").await?.score, Some(0.9));
    publish(&staging).await?;
    assert_eq!(staging.observation().await.retained_bytes, 0);
    // A later apply must preserve the newer published score, not the first
    // generation's null score or a delayed old point read's pending value.
    assert_eq!(
        old_read.resolve().await?.unwrap().status.as_deref(),
        Some("pending")
    );
    staging
        .apply(
            2,
            3,
            "T",
            &TurnAnnotationPatch {
                status: Clear,
                ..Default::default()
            },
        )
        .await?;
    let value = current(&staging, "T").await?;
    assert_eq!(value.score, Some(0.9));
    assert_eq!(value.status, None);
    Ok(())
}
