use super::*;

fn pending(read: &TurnRead) -> NewTurnSource {
    match read.body_source() {
        TurnBodySource::Unmaterialized(source) => source,
        TurnBodySource::Materialized(_) => panic!("expected a pending body source"),
    }
}

fn base_version(read: &TurnRead) -> u64 {
    match read.body_source() {
        TurnBodySource::Materialized(base) => base.version_id(),
        TurnBodySource::Unmaterialized(_) => panic!("expected a materialized body"),
    }
}

#[tokio::test]
async fn first_body_row_survives_duplicates_patches_publication_and_replay() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (staging, initial_body) = super::body::binding_fixture(root.path()).await?;
    let original_base = staging.state.read().await.base.dataset.clone();
    let initial = TurnAnnotation {
        labels: BTreeSet::from(["imported".into()]),
        status: Some("pending".into()),
        score: None,
    };
    let conflicting = TurnAnnotation {
        score: Some(0.1),
        ..initial.clone()
    };
    let registrations = [
        Registration {
            turn: "turn-000000",
            rollout_id: "wrong",
            annotation: &conflicting,
        },
        Registration {
            turn: "turn-000002",
            rollout_id: "session-0",
            annotation: &initial,
        },
        Registration {
            turn: "turn-000002",
            rollout_id: "wrong",
            annotation: &conflicting,
        },
        Registration {
            turn: "turn-000003",
            rollout_id: "session-0",
            annotation: &initial,
        },
    ];
    assert_eq!(
        staging.register_committed(1, 2, &registrations).await?,
        ApplyOutcome::Applied
    );
    let expected = NewTurnSource {
        record_index: 2,
        row: 1,
    };
    assert_eq!(
        pending(&staging.capture_turn("turn-000002").await?),
        expected
    );
    staging
        .apply(
            2,
            3,
            "turn-000002",
            &TurnAnnotationPatch {
                status: Set("reviewed".into()),
                ..Default::default()
            },
        )
        .await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    assert_eq!(
        frozen.new_turns().collect::<Vec<_>>(),
        vec![
            ("turn-000002", expected),
            (
                "turn-000003",
                NewTurnSource {
                    record_index: 2,
                    row: 3
                }
            ),
        ]
    );
    staging.register_committed(3, 4, &registrations).await?;
    staging
        .apply(
            4,
            5,
            "turn-000002",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let held_pending = staging.capture_turn("turn-000002").await?;
    assert_eq!(pending(&held_pending), expected);
    assert_eq!(
        held_pending.annotations().resolve().await?.unwrap().score,
        Some(0.9)
    );
    let body = Arc::new(Dataset::open(&initial_body.uri).await?);
    let body = super::body::detached(body, super::body::imported(2, 2)?).await?;
    let reference = BodySnapshot::capture(scope().table_instance, &body).await?;
    let budget = crate::MergeMemoryBudget::new(8 << 20);
    let prepared = frozen
        .prepare_with_body(&budget, 8192, reference.clone())
        .await?;
    let selected = commit(&prepared).await?;
    let first_version = selected.version_id();
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    drop((prepared, frozen));
    // The later active patch retains a birth marker, but that birth is now
    // covered by the selected base and must no longer select the WAL frame.
    let held_base = staging.capture_turn("turn-000002").await?;
    assert_eq!(base_version(&held_base), first_version);
    assert_eq!(
        held_base.annotations().resolve().await?.unwrap().score,
        Some(0.9)
    );
    assert_eq!(pending(&held_pending), expected);
    drop(held_pending);

    let replay = AnnotationStaging::new(original_base, scope(), 0, Limits::default())?;
    replay
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
    replay.register_committed(1, 2, &registrations).await?;
    assert_eq!(
        pending(&replay.capture_turn("turn-000002").await?),
        expected
    );
    let recovered = AnnotationStaging::new(selected, scope(), 3, Limits::default())?;
    recovered.register_committed(3, 4, &registrations).await?;
    recovered
        .apply(
            4,
            5,
            "turn-000002",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    assert_eq!(
        base_version(&recovered.capture_turn("turn-000002").await?),
        first_version
    );

    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    assert!(frozen.new_turns().next().is_none());
    let prepared = frozen.prepare_with_body(&budget, 8192, reference).await?;
    let selected = commit(&prepared).await?;
    staging.install_published(&prepared, selected).await?;
    drop((prepared, frozen));
    assert_eq!(base_version(&held_base), first_version);
    let versions = || async {
        Ok::<_, Error>(
            staging
                .protected_bases()
                .await?
                .datasets()
                .map(Dataset::version_id)
                .collect::<Vec<_>>(),
        )
    };
    assert!(versions().await?.contains(&first_version));
    drop(held_base);
    assert!(!versions().await?.contains(&first_version));
    Ok(())
}

#[tokio::test]
async fn full_turn_capture_shares_reader_limits_and_rejection_has_no_body_origin() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (seed, _) = super::body::binding_fixture(root.path()).await?;
    let base = seed.state.read().await.base.dataset.clone();
    let staging = AnnotationStaging::new(
        base,
        scope(),
        0,
        Limits {
            readers: 1,
            ..Limits::default()
        },
    )?;
    let held = staging.capture_turn("turn-000000").await?;
    assert!(staging.capture("turn-000001").await.is_err());
    assert!(staging.capture_turn("turn-000001").await.is_err());
    drop(held);
    let annotation = TurnAnnotation::default();
    let rows = [
        Registration {
            turn: "new",
            rollout_id: "session",
            annotation: &annotation,
        },
        Registration {
            turn: "",
            rollout_id: "session",
            annotation: &annotation,
        },
    ];
    assert!(matches!(
        staging.register_committed(0, 1, &rows).await?,
        ApplyOutcome::Rejected(_)
    ));
    let read = staging.capture_turn("new").await?;
    assert!(read.annotations().resolve().await?.is_none());
    assert!(matches!(
        read.body_source(),
        TurnBodySource::Materialized(_)
    ));
    assert!(staging.freeze(uuid::Uuid::new_v4()).await?.is_none());
    Ok(())
}
