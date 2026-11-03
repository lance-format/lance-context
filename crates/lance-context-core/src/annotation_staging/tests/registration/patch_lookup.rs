use super::{lookup::opened_rows, *};
use std::{sync::atomic::Ordering, time::Duration};

fn score(value: f32) -> TurnAnnotationPatch {
    TurnAnnotationPatch {
        score: Set(value),
        ..Default::default()
    }
}

async fn apply(
    staging: &AnnotationStaging,
    after: u64,
    turn: &str,
    patch: &TurnAnnotationPatch,
    lookup: &PatchLookup,
) -> Result<ApplyOutcome> {
    staging
        .apply_committed_with_lookup(after, after + 1, &[Mutation { turn, patch }], lookup)
        .await
}

#[tokio::test]
async fn shared_values_avoid_io_and_preserve_active_and_frozen_fields() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened_rows(&root.path().join("narrow.lance"), 8).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let lookup = staging
        .prepare_patch_lookup(&["turn-000000", "turn-000001", "turn-000000"], &workspace)
        .await?
        .unwrap();
    assert!(workspace.reserved() >= 3 * staging.limits.value_bytes);
    assert_eq!(staging.observation().await.readers, 1);
    let before = throttle.ops.load(Ordering::Relaxed);
    throttle.frozen.store(true, Ordering::Relaxed);
    let result = tokio::time::timeout(Duration::from_secs(2), async {
        let reviewed = TurnAnnotationPatch {
            status: Set("reviewed".into()),
            ..Default::default()
        };
        assert_eq!(
            apply(&staging, 0, "turn-000000", &reviewed, &lookup).await?,
            ApplyOutcome::Applied
        );
        let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
        assert_eq!(
            apply(&staging, 1, "turn-000000", &score(0.9), &lookup).await?,
            ApplyOutcome::Applied
        );
        assert_eq!(
            apply(&staging, 2, "turn-000001", &score(0.8), &lookup).await?,
            ApplyOutcome::Applied
        );
        assert_eq!(
            apply(&staging, 3, "turn-000000", &score(0.7), &lookup).await?,
            ApplyOutcome::Applied
        );
        assert!(staging
            .prepare_patch_lookup(&["turn-000000", "turn-000001"], &workspace)
            .await?
            .is_none());
        Ok::<_, Error>(frozen)
    })
    .await;
    throttle.frozen.store(false, Ordering::Relaxed);
    let frozen = result.map_err(|_| invalid("shared patch lookup performed more IO"))??;
    assert_eq!(throttle.ops.load(Ordering::Relaxed), before);
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 32)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    let current = current(&staging, "turn-000000").await?;
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.7));
    assert_eq!(current.labels, BTreeSet::from(["imported".into()]));
    drop((prepared, frozen, lookup));
    assert_eq!(workspace.reserved(), 0);
    assert_eq!(staging.observation().await.readers, 0);
    Ok(())
}

#[tokio::test]
async fn old_base_foreign_staging_and_unqueried_values_fall_back() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened_rows(&root.path().join("narrow.lance"), 8).await?;
    let staging = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    let other = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let lookup = staging
        .prepare_patch_lookup(&["turn-000000"], &workspace)
        .await?
        .unwrap();
    let before = throttle.ops.load(Ordering::Relaxed);
    apply(&other, 0, "turn-000000", &score(0.2), &lookup).await?;
    assert!(throttle.ops.load(Ordering::Relaxed) > before);
    apply(&staging, 0, "turn-000000", &score(0.9), &lookup).await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 32)
        .await?;
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    drop((prepared, frozen));
    assert_eq!(staging.protected_bases().await?.datasets().count(), 2);
    selected.session().file_metadata_cache().clear().await;
    let before = throttle.ops.load(Ordering::Relaxed);
    let patch = TurnAnnotationPatch {
        status: Set("reviewed".into()),
        ..Default::default()
    };
    apply(&staging, 1, "turn-000000", &patch, &lookup).await?;
    assert!(throttle.ops.load(Ordering::Relaxed) > before);
    assert_eq!(current(&staging, "turn-000000").await?.score, Some(0.9));
    let lookup2 = staging
        .prepare_patch_lookup(&["missing"], &workspace)
        .await?
        .unwrap();
    let before = throttle.ops.load(Ordering::Relaxed);
    apply(&staging, 2, "turn-000001", &score(0.8), &lookup2).await?;
    assert!(throttle.ops.load(Ordering::Relaxed) > before);
    drop(lookup);
    assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    Ok(())
}

#[tokio::test]
async fn requests_keep_rejection_backpressure_and_replay_boundaries() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, _) = opened_rows(&root.path().join("narrow.lance"), 8).await?;
    let staging = AnnotationStaging::new(
        base.clone(),
        scope(),
        0,
        Limits {
            retained_entries: 1,
            ..Default::default()
        },
    )?;
    let workspace = crate::MergeMemoryBudget::new(1 << 20);
    let lookup = staging
        .prepare_patch_lookup(&["turn-000000", "turn-000001", "missing"], &workspace)
        .await?
        .unwrap();
    assert_eq!(
        apply(&staging, 0, "turn-000000", &score(f32::NAN), &lookup).await?,
        ApplyOutcome::Rejected(Rejection {
            row: 0,
            reason: RejectionReason::InvalidPatch
        })
    );
    // A multi-row request remains atomic when its later row is missing.
    assert_eq!(
        staging
            .apply_committed_with_lookup(
                1,
                2,
                &[
                    Mutation {
                        turn: "turn-000000",
                        patch: &score(0.5)
                    },
                    Mutation {
                        turn: "missing",
                        patch: &score(0.4)
                    },
                ],
                &lookup
            )
            .await?,
        ApplyOutcome::Rejected(Rejection {
            row: 1,
            reason: RejectionReason::MissingTurn
        })
    );
    assert_eq!(current(&staging, "turn-000000").await?.score, None);
    assert_eq!(
        apply(&staging, 2, "turn-000000", &score(0.9), &lookup).await?,
        ApplyOutcome::Applied
    );
    assert_eq!(
        apply(&staging, 3, "turn-000001", &score(0.8), &lookup).await?,
        ApplyOutcome::Backpressure
    );
    assert_eq!(staging.observation().await.applied, 3);
    assert!(apply(&staging, 2, "turn-000000", &score(0.1), &lookup)
        .await
        .is_err());
    assert_eq!(current(&staging, "turn-000000").await?.score, Some(0.9));
    // Rebuild using ordinary apply, without depending on the temporary lookup.
    let replay = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let requests = [
        vec![("turn-000000", score(f32::NAN))],
        vec![("turn-000000", score(0.5)), ("missing", score(0.4))],
        vec![("turn-000000", score(0.9))],
    ];
    for (after, request) in requests.iter().enumerate() {
        let rows = request
            .iter()
            .map(|(turn, patch)| Mutation { turn, patch })
            .collect::<Vec<_>>();
        replay
            .apply_committed(after as u64, after as u64 + 1, &rows)
            .await?;
    }
    assert_eq!(replay.observation().await.applied, 3);
    assert_eq!(
        current(&replay, "turn-000000").await?,
        current(&staging, "turn-000000").await?
    );
    Ok(())
}

#[tokio::test]
async fn admission_and_cancelled_lookup_release_all_resources() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened_rows(&root.path().join("narrow.lance"), 8).await?;
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
        .prepare_patch_lookup(&["turn-000000"], &workspace)
        .await?
        .is_none());
    drop(held);
    let reader = staging.capture("turn-000000").await?;
    let before = throttle.ops.load(Ordering::Relaxed);
    assert!(staging
        .prepare_patch_lookup(&["turn-000000"], &workspace)
        .await?
        .is_none());
    assert_eq!(workspace.reserved(), 0);
    assert_eq!(throttle.ops.load(Ordering::Relaxed), before);
    drop(reader);
    for keys in [
        vec![""],
        vec!["turn-000000"; COLD_LOOKUP_KEYS + 1],
        vec!["turn-000000"; 32],
    ] {
        assert!(staging
            .prepare_patch_lookup(&keys, &workspace)
            .await?
            .is_none());
    }
    let before = throttle.ops.load(Ordering::Relaxed);
    throttle.frozen.store(true, Ordering::Relaxed);
    let result = {
        let lookup = staging.prepare_patch_lookup(&["turn-000000"], &workspace);
        tokio::pin!(lookup);
        tokio::time::timeout(Duration::from_secs(2), async {
            tokio::select! {
                _ = &mut lookup => Err(invalid("frozen lookup unexpectedly completed")),
                _ = async {
                    while throttle.ops.load(Ordering::Relaxed) == before {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                } => {
                    assert!(workspace.reserved() > 0);
                    assert_eq!(staging.observation().await.readers, 1);
                    Ok::<_, Error>(())
                },
            }
        })
        .await
    };
    throttle.frozen.store(false, Ordering::Relaxed);
    result.map_err(|_| invalid("lookup never entered native IO"))??;
    assert_eq!(workspace.reserved(), 0);
    assert_eq!(staging.observation().await.readers, 0);
    assert_eq!(staging.observation().await.applied, 0);
    Ok(())
}

#[tokio::test]
async fn shared_lookup_reduces_native_io_for_eight_independent_requests() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (base, throttle) = opened_rows(&root.path().join("narrow.lance"), 8).await?;
    let separate = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    let shared = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let turns = (0..8).map(|i| format!("turn-{i:06}")).collect::<Vec<_>>();
    let before = throttle.ops.load(Ordering::Relaxed);
    for (i, turn) in turns.iter().enumerate() {
        separate
            .apply(i as u64, i as u64 + 1, turn, &score(0.9))
            .await?;
    }
    let separate_ops = throttle.ops.load(Ordering::Relaxed) - before;
    let before = throttle.ops.load(Ordering::Relaxed);
    let workspace = crate::MergeMemoryBudget::new(1 << 20);
    let keys = turns.iter().map(String::as_str).collect::<Vec<_>>();
    let lookup = shared
        .prepare_patch_lookup(&keys, &workspace)
        .await?
        .unwrap();
    for (i, turn) in turns.iter().enumerate() {
        assert_eq!(
            apply(&shared, i as u64, turn, &score(0.9), &lookup).await?,
            ApplyOutcome::Applied
        );
    }
    let shared_ops = throttle.ops.load(Ordering::Relaxed) - before;
    assert!(shared_ops > 0);
    assert!(
        shared_ops < separate_ops,
        "shared={shared_ops}, separate={separate_ops}"
    );
    for turn in &turns {
        assert_eq!(
            current(&shared, turn).await?,
            current(&separate, turn).await?
        );
    }
    Ok(())
}
