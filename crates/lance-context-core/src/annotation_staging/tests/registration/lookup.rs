use super::*;
use crate::preparation_io::test_support::{Throttle, ThrottleWrapper};
use lance::{dataset::builder::DatasetBuilder, dataset::ReadParams, session::Session};
use lance_io::object_store::ObjectStoreParams;
use std::{sync::atomic::Ordering, time::Duration};

async fn opened(path: &std::path::Path) -> Result<(Arc<Dataset>, Throttle)> {
    narrow(path).await?;
    let throttle = Throttle::new(Duration::ZERO);
    let base = DatasetBuilder::from_uri(format!("file-object-store://{}", path.display()))
        .with_session(Arc::new(Session::new(1 << 20, 0, Default::default())))
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

fn register<'a>(turn: &'a str, value: &'a TurnAnnotation) -> Registration<'a> {
    Registration {
        turn,
        rollout_id: "session-new",
        annotation: value,
    }
}

#[tokio::test]
async fn lookup_shares_io_but_keeps_request_order_and_frozen_updates() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("narrow.lance")).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let lookup = staging
        .prepare_registration_lookup(&["turn-000000", "new-a", "new-b", "new-a"], &workspace)
        .await?
        .unwrap();
    assert!(workspace.reserved() > 0);
    assert_eq!(staging.observation().await.readers, 1);
    let value = TurnAnnotation::default();
    let before = throttle.ops.load(Ordering::Relaxed);
    throttle.frozen.store(true, Ordering::Relaxed);
    let result = tokio::time::timeout(Duration::from_secs(2), async {
        for (after, turn) in [(0, "turn-000000"), (1, "new-a")] {
            assert_eq!(
                staging
                    .register_committed_with_lookup(
                        after,
                        after + 1,
                        &[register(turn, &value)],
                        &lookup
                    )
                    .await?,
                ApplyOutcome::Applied
            );
        }
        staging
            .apply(
                2,
                3,
                "new-a",
                &TurnAnnotationPatch {
                    score: Set(0.9),
                    ..Default::default()
                },
            )
            .await?;
        let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
        assert_eq!(
            staging
                .register_committed_with_lookup(
                    3,
                    4,
                    &[register("new-a", &value), register("new-b", &value)],
                    &lookup
                )
                .await?,
            ApplyOutcome::Applied
        );
        staging
            .apply(
                4,
                5,
                "new-a",
                &TurnAnnotationPatch {
                    status: Set("reviewed".into()),
                    ..Default::default()
                },
            )
            .await?;
        Ok::<_, Error>(frozen)
    })
    .await;
    throttle.frozen.store(false, Ordering::Relaxed);
    let frozen =
        result.map_err(|_| invalid("captured registration lookup performed more IO"))??;
    assert_eq!(throttle.ops.load(Ordering::Relaxed), before);
    assert_eq!(staging.observation().await.applied, 5);
    assert_eq!(current(&staging, "new-a").await?.score, Some(0.9));
    assert_eq!(
        current(&staging, "new-a").await?.status.as_deref(),
        Some("reviewed")
    );
    // Publishing the earlier frozen request must leave the later request live.
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 32)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    assert_eq!(current(&staging, "new-a").await?.score, Some(0.9));
    assert_eq!(
        current(&staging, "new-a").await?.status.as_deref(),
        Some("reviewed")
    );
    assert_eq!(current(&staging, "new-b").await?, value);
    drop((prepared, frozen, lookup));
    assert_eq!(workspace.reserved(), 0);
    assert_eq!(staging.observation().await.readers, 0);
    Ok(())
}

#[tokio::test]
async fn changed_base_foreign_staging_and_unqueried_keys_use_native_lookup() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("narrow.lance")).await?;
    let staging = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let lookup = staging
        .prepare_registration_lookup(&["turn-000000", "new"], &workspace)
        .await?
        .unwrap();
    let initial = TurnAnnotation {
        status: Some("first".into()),
        ..Default::default()
    };
    // Identical Dataset handles/versions do not make a lookup transferable.
    let other = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let before = throttle.ops.load(Ordering::Relaxed);
    other
        .register_committed_with_lookup(0, 1, &[register("turn-000000", &initial)], &lookup)
        .await?;
    assert!(throttle.ops.load(Ordering::Relaxed) > before);
    staging
        .register_committed_with_lookup(0, 1, &[register("new", &initial)], &lookup)
        .await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 32)
        .await?;
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    drop((prepared, frozen));
    let protected = staging.protected_bases().await?;
    assert_eq!(protected.datasets().count(), 2);
    drop(protected);
    selected.session().file_metadata_cache().clear().await;
    let before = throttle.ops.load(Ordering::Relaxed);
    let stale = TurnAnnotation {
        status: Some("must-not-reset".into()),
        ..Default::default()
    };
    staging
        .register_committed_with_lookup(1, 2, &[register("new", &stale)], &lookup)
        .await?;
    assert!(throttle.ops.load(Ordering::Relaxed) > before);
    assert_eq!(staging.observation().await.active_entries, 0);
    assert_eq!(current(&staging, "new").await?, initial);
    let current_lookup = staging
        .prepare_registration_lookup(&["unrelated"], &workspace)
        .await?
        .unwrap();
    let before = throttle.ops.load(Ordering::Relaxed);
    staging
        .register_committed_with_lookup(2, 3, &[register("turn-000000", &stale)], &current_lookup)
        .await?;
    assert!(throttle.ops.load(Ordering::Relaxed) > before);
    assert_eq!(staging.observation().await.active_entries, 0);
    drop(lookup);
    assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    Ok(())
}

#[tokio::test]
async fn lookup_admission_and_cancellation_release_workspace_and_base_pins() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened(&root.path().join("narrow.lance")).await?;
    let staging = AnnotationStaging::new(
        base,
        scope(),
        0,
        Limits {
            readers: 1,
            ..Default::default()
        },
    )?;
    let workspace = crate::MergeMemoryBudget::new(1 << 20);
    let held = workspace.try_reserve(workspace.limit()).unwrap();
    assert!(staging
        .prepare_registration_lookup(&["new"], &workspace)
        .await?
        .is_none());
    drop(held);
    let reader = staging.capture("turn-000000").await?;
    let before = throttle.ops.load(Ordering::Relaxed);
    assert!(staging
        .prepare_registration_lookup(&["new"], &workspace)
        .await?
        .is_none());
    assert_eq!(workspace.reserved(), 0);
    assert_eq!(throttle.ops.load(Ordering::Relaxed), before);
    drop(reader);
    assert!(staging
        .prepare_registration_lookup(&vec!["new"; COLD_LOOKUP_KEYS + 1], &workspace)
        .await?
        .is_none());
    assert!(staging
        .prepare_registration_lookup(&[""], &workspace)
        .await?
        .is_none());
    // A speculative window may not use the budget's oversized-merge exception.
    let large = "x".repeat(staging.limits.value_bytes);
    assert!(staging
        .prepare_registration_lookup(&vec![large.as_str(); 32], &workspace)
        .await?
        .is_none());
    throttle.frozen.store(true, Ordering::Relaxed);
    let result = tokio::time::timeout(
        Duration::from_millis(50),
        staging.prepare_registration_lookup(&["new"], &workspace),
    )
    .await;
    throttle.frozen.store(false, Ordering::Relaxed);
    assert!(result.is_err());
    assert_eq!(workspace.reserved(), 0);
    assert_eq!(staging.observation().await.readers, 0);
    assert_eq!(staging.observation().await.applied, 0);
    let lookup = staging
        .prepare_registration_lookup(&["new"], &workspace)
        .await?
        .unwrap();
    let protected = staging.protected_bases().await?;
    assert_eq!(protected.datasets().count(), 1);
    drop((protected, lookup));
    assert_eq!(workspace.reserved(), 0);
    assert_eq!(staging.observation().await.readers, 0);
    Ok(())
}

#[tokio::test]
async fn shared_lookup_preserves_independent_rejections_and_backpressure() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = narrow(&root.path().join("narrow.lance")).await?;
    let staging = AnnotationStaging::new(
        base,
        scope(),
        0,
        Limits {
            retained_entries: 1,
            ..Default::default()
        },
    )?;
    let workspace = crate::MergeMemoryBudget::new(1 << 20);
    let lookup = staging
        .prepare_registration_lookup(&["new-a", "new-b"], &workspace)
        .await?
        .unwrap();
    let invalid_value = TurnAnnotation {
        score: Some(f32::NAN),
        ..Default::default()
    };
    assert_eq!(
        staging
            .register_committed_with_lookup(0, 1, &[register("new-a", &invalid_value)], &lookup)
            .await?,
        ApplyOutcome::Rejected(Rejection {
            row: 0,
            reason: RejectionReason::InvalidPatch
        })
    );
    let value = TurnAnnotation::default();
    assert_eq!(
        staging
            .register_committed_with_lookup(1, 2, &[register("new-a", &value)], &lookup)
            .await?,
        ApplyOutcome::Applied
    );
    assert_eq!(
        staging
            .register_committed_with_lookup(2, 3, &[register("new-b", &value)], &lookup)
            .await?,
        ApplyOutcome::Backpressure
    );
    assert_eq!(staging.observation().await.applied, 2);
    assert_eq!(current(&staging, "new-a").await?, value);
    assert!(staging.capture("new-b").await?.resolve().await?.is_none());
    Ok(())
}
