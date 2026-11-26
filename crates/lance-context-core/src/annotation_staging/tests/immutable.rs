use super::*;

fn bodies(ids: &[&str]) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("session_id", DataType::Utf8, false),
            Field::new("id", DataType::Utf8, false),
            Field::new("payload", DataType::LargeBinary, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["session"; ids.len()])),
            Arc::new(StringArray::from(ids.to_vec())),
            Arc::new(LargeBinaryArray::from_iter_values(
                ids.iter().map(|id| id.as_bytes()),
            )),
        ],
    )?)
}

async fn empty(root: &std::path::Path) -> Result<Arc<Dataset>> {
    let batch = bodies(&[])?;
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    Ok(Arc::new(
        Dataset::write(reader, root.to_str().unwrap(), None).await?,
    ))
}

fn request<'a>(keys: &'a [&str]) -> Vec<ImmutableRegistration<'a>> {
    keys.iter()
        .map(|key| ImmutableRegistration {
            key,
            exists_in_base: false,
        })
        .collect()
}

fn origin(read: &ImmutableBodyRead) -> NewTurnSource {
    match read.source() {
        TurnBodySource::Unmaterialized(source) => source,
        TurnBodySource::Materialized(_) => panic!("expected a staged body"),
    }
}

#[tokio::test]
async fn first_create_freeze_publish_preserves_suffix_and_rejects_stale_lookup() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = empty(&root.path().join("body.lance")).await?;
    let staging = AnnotationStaging::new_immutable_body(base, scope(), 0, Limits::default())?;
    let lookup = staging.capture_immutable_base().await?;
    let first = bodies(&["a", "a", "b"])?;
    assert_eq!(
        staging
            .register_immutable_committed(0, 1, &request(&["a", "a", "b"]), &lookup)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    let a = staging.capture_immutable_body("a").await?;
    assert_eq!(
        origin(&a),
        NewTurnSource {
            record_index: 1,
            row: 0
        }
    );
    assert_eq!(
        origin(&staging.capture_immutable_body("b").await?),
        NewTurnSource {
            record_index: 1,
            row: 2
        }
    );
    let frozen = staging
        .freeze_with_results(uuid::Uuid::new_v4(), Some("first-results".into()))
        .await?
        .unwrap();
    let second = bodies(&["a", "c"])?;
    // Another producer's duplicate cannot replace a body in the frozen cut.
    assert_eq!(
        staging
            .register_immutable_committed(1, 2, &request(&["a", "c"]), &lookup)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    assert_eq!(
        origin(&staging.capture_immutable_body("a").await?),
        origin(&a)
    );
    let c = staging.capture_immutable_body("c").await?;
    let mut known = BTreeMap::from([
        ("a".to_owned(), false),
        ("b".to_owned(), false),
        ("c".to_owned(), false),
        ("d".to_owned(), true),
    ]);
    let staged_lookup = staging
        .try_capture_immutable_base_with_staged(&mut known)
        .await?
        .unwrap();
    assert_eq!(
        known.values().copied().collect::<Vec<_>>(),
        vec![true, true, true, false]
    );
    assert_eq!(
        origin(&c),
        NewTurnSource {
            record_index: 2,
            row: 1
        }
    );
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let original = Arc::new(first);
    let slices = vec![original.slice(0, 1), original.slice(2, 1)];
    assert_eq!(
        slices[0].column(2).to_data().buffers()[0].as_ptr(),
        original.column(2).to_data().buffers()[0].as_ptr()
    );
    let prepared = frozen
        .prepare_immutable_body(&workspace, slices, original)
        .await?;
    assert!(matches!(
        prepared.merge().transaction.operation,
        lance_table::transaction::Operation::Append { .. }
    ));
    let selected = commit(&prepared).await?;
    assert_eq!(selected.count_rows(None).await?, 2);
    let properties = selected
        .read_transaction()
        .await?
        .unwrap()
        .transaction_properties
        .unwrap();
    assert_eq!(properties[RESULTS_PROPERTY], "first-results");
    assert_eq!(
        serde_json::from_str::<(Scope, u64)>(&properties["lance-context.annotation-cut"]).unwrap(),
        (scope(), 1)
    );
    assert!(!properties.contains_key(BODY_SNAPSHOT_PROPERTY));
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    assert_eq!(
        origin(&staging.capture_immutable_body("c").await?),
        origin(&c)
    );
    match staging.capture_immutable_body("a").await?.source() {
        TurnBodySource::Materialized(base) => assert_eq!(base.version_id(), selected.version_id()),
        _ => panic!("published row must resolve against the new full body"),
    }
    // Old negative lookup must not reinsert a just-published body.
    assert_eq!(
        staging
            .register_immutable_committed(2, 3, &request(&["a"]), &lookup)
            .await?,
        None
    );
    assert_eq!(staging.observation().await.applied, 2);
    // Even staged-positive results retain the same publication/base fence.
    assert_eq!(
        staging
            .register_immutable_committed(
                2,
                3,
                &[ImmutableRegistration {
                    key: "a",
                    exists_in_base: known["a"],
                }],
                &staged_lookup,
            )
            .await?,
        None
    );
    let refreshed = staging
        .try_capture_immutable_base_with_staged(&mut known)
        .await?
        .unwrap();
    assert_eq!(
        known.values().copied().collect::<Vec<_>>(),
        vec![false, false, true, false]
    );
    drop((staged_lookup, refreshed));
    assert_eq!(lookup.dataset().count_rows(None).await?, 0);
    drop((prepared, frozen, lookup, a, c));
    let frozen = staging
        .freeze_with_results(uuid::Uuid::new_v4(), Some("second-results".into()))
        .await?
        .unwrap();
    assert_eq!(
        frozen.new_turns().collect::<Vec<_>>(),
        vec![(
            "c",
            NewTurnSource {
                record_index: 2,
                row: 1
            }
        )]
    );
    let selected = Arc::new(second);
    let prepared = frozen
        .prepare_immutable_body(&workspace, vec![selected.slice(1, 1)], selected)
        .await?;
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    assert_eq!(selected.count_rows(None).await?, 3);
    drop((frozen, prepared));
    let observed = staging.observation().await;
    assert_eq!(
        (
            observed.active_entries,
            observed.frozen_entries,
            observed.retained_entries
        ),
        (0, 0, 0)
    );

    // Recovery starts from the same full body and materialized cut. All-hit
    // replay still publishes request results, without another data fragment.
    let recovered =
        AnnotationStaging::new_immutable_body(selected.clone(), scope(), 2, Limits::default())?;
    let base = recovered.capture_immutable_base().await?;
    let mut duplicates = request(&["a", "b", "c"]);
    for row in &mut duplicates {
        row.exists_in_base = true;
    }
    assert_eq!(
        recovered
            .register_immutable_committed(2, 3, &duplicates, &base)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    let frozen = recovered
        .freeze_with_results(uuid::Uuid::new_v4(), Some("replay-results".into()))
        .await?
        .unwrap();
    let prepared = frozen
        .prepare_immutable_body(&workspace, vec![], Arc::new(()))
        .await?;
    assert!(matches!(
        prepared.merge().transaction.operation,
        lance_table::transaction::Operation::UpdateConfig { .. }
    ));
    let published = commit(&prepared).await?;
    assert_eq!(
        published.manifest().fragments,
        selected.manifest().fragments
    );
    recovered.install_published(&prepared, published).await?;
    assert_eq!(recovered.observation().await.materialized, 3);
    assert!(recovered.capture("a").await.is_err());
    assert!(recovered
        .apply_committed(
            3,
            4,
            &[Mutation {
                turn: "a",
                patch: &TurnAnnotationPatch::default()
            }]
        )
        .await
        .is_err());
    Ok(())
}

#[tokio::test]
async fn frozen_body_handles_keep_admission_bounded() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = empty(&root.path().join("body.lance")).await?;
    let staging = AnnotationStaging::new_immutable_body(
        base,
        scope(),
        0,
        Limits {
            retained_entries: 2,
            ..Default::default()
        },
    )?;
    let lookup = staging.capture_immutable_base().await?;
    assert_eq!(
        staging
            .register_immutable_committed(0, 1, &request(&["a", "b"]), &lookup)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    assert_eq!(
        staging
            .register_immutable_committed(1, 2, &request(&["c"]), &lookup)
            .await?,
        Some(ApplyOutcome::Backpressure)
    );
    assert_eq!(staging.observation().await.applied, 1);
    assert!(staging.freeze(uuid::Uuid::new_v4()).await.is_err());
    let batch = Arc::new(bodies(&["a", "b"])?);
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let prepared = frozen
        .prepare_immutable_body(&workspace, vec![batch.as_ref().clone()], batch)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    let lookup = staging.capture_immutable_base().await?;
    assert_eq!(
        staging
            .register_immutable_committed(1, 2, &request(&["c"]), &lookup)
            .await?,
        Some(ApplyOutcome::Backpressure)
    );
    drop((frozen, prepared));
    assert_eq!(
        staging
            .register_immutable_committed(1, 2, &request(&["c"]), &lookup)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    Ok(())
}
