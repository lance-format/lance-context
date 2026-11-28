use super::*;

async fn publish(staging: &AnnotationStaging, ids: &[&str]) -> Result<Arc<Dataset>> {
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    install(staging, frozen, ids).await
}

async fn install(
    staging: &AnnotationStaging,
    frozen: FrozenAnnotations,
    ids: &[&str],
) -> Result<Arc<Dataset>> {
    let batch = Arc::new(bodies(ids)?);
    let prepared = frozen
        .prepare_immutable_body(
            &crate::MergeMemoryBudget::new(8 << 20),
            vec![batch.as_ref().clone()],
            batch,
        )
        .await?;
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    Ok(selected)
}

#[tokio::test]
async fn retired_body_can_recapture_installed_base_with_one_reader_permit() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = empty(&root.path().join("body.lance")).await?;
    let staging = AnnotationStaging::new_immutable_body(
        base,
        scope(),
        0,
        Limits {
            readers: 1,
            ..Limits::default()
        },
    )?;
    let lookup = staging.capture_immutable_base().await?;
    assert_eq!(
        staging
            .register_immutable_committed(0, 1, &request(&["a"]), &lookup)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    drop(lookup);
    let mut staged = staging.capture_immutable_body("a").await?;
    assert_eq!(origin(&staged).record_index, 1);
    let selected = publish(&staging, &["a"]).await?;

    let cut = staging.recapture_immutable_installed(&mut staged).await?;
    assert_eq!(cut, 1);
    assert_eq!(staging.observation().await.readers, 1);
    let TurnBodySource::Materialized(base) = staged.source() else {
        panic!("retired body must recapture the installed dataset");
    };
    assert_eq!(base.version_id(), selected.version_id());
    assert_eq!(base.count_rows(Some("id = 'a'".into())).await?, 1);
    drop(staged);
    assert_eq!(staging.observation().await.readers, 0);
    Ok(())
}

#[tokio::test]
async fn covered_lookup_survives_two_publications_and_rechecks_later_active_rows() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = empty(&root.path().join("body.lance")).await?;
    let staging = AnnotationStaging::new_immutable_body(base, scope(), 0, Limits::default())?;
    let plain = staging.capture_immutable_base().await?;
    let nonblocking_plain = staging.try_capture_immutable_base().await?.unwrap();
    staging
        .register_immutable_committed(0, 1, &request(&["a"]), &plain)
        .await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    staging
        .register_immutable_committed(1, 2, &request(&["b"]), &plain)
        .await?;

    let mut keys = BTreeMap::from([
        ("a".to_owned(), false),
        ("b".to_owned(), false),
        ("c".to_owned(), false),
        ("later".to_owned(), false),
    ]);
    let covered = staging
        .try_capture_immutable_base_with_staged(&mut keys)
        .await?
        .unwrap();
    assert_eq!(
        keys.values().copied().collect::<Vec<_>>(),
        [true, true, false, false]
    );
    // The historical query's exact captured dataset is empty. Finish two
    // real publications before submitting its results, reproducing the race.
    assert_eq!(covered.dataset().count_rows(None).await?, 0);
    install(&staging, frozen, &["a"]).await?;
    let selected = publish(&staging, &["b"]).await?;
    assert_eq!(selected.count_rows(None).await?, 2);
    for lookup in [&plain, &nonblocking_plain] {
        assert_eq!(
            staging
                .register_immutable_committed(2, 3, &request(&["a"]), lookup)
                .await?,
            None
        );
    }
    // A row applied after capture remains in staging and must be checked again,
    // despite the earlier query's negative answer for this key.
    let fresh = staging.capture_immutable_base().await?;
    assert_eq!(
        staging
            .register_immutable_committed(2, 3, &request(&["later"]), &fresh)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    let later = origin(&staging.capture_immutable_body("later").await?);
    let observed = staging.observation().await;
    assert_eq!(observed.immutable_lookup_reused, 0);
    assert_eq!(observed.immutable_lookup_invalidated, 2);
    let rows = keys
        .iter()
        .map(|(key, present)| ImmutableRegistration {
            key,
            exists_in_base: *present,
        })
        .collect::<Vec<_>>();
    let remaining = staging.entries.available_permits() as u32;
    let held_entries = staging
        .entries
        .clone()
        .acquire_many_owned(remaining)
        .await
        .unwrap();
    let before_pressure = staging.observation().await;
    assert_eq!(
        staging
            .register_immutable_committed(3, 4, &rows, &covered)
            .await?,
        Some(ApplyOutcome::Backpressure)
    );
    assert_eq!(staging.observation().await, before_pressure);
    drop(held_entries);
    assert_eq!(
        staging
            .register_immutable_committed(3, 4, &rows, &covered)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    assert_eq!(
        origin(&staging.capture_immutable_body("later").await?),
        later
    );
    assert_eq!(
        origin(&staging.capture_immutable_body("c").await?).record_index,
        4
    );
    assert_eq!(staging.observation().await.active_entries, 2);
    assert_eq!(staging.observation().await.immutable_lookup_reused, 1);
    for id in ["a", "b"] {
        assert!(matches!(
            staging.capture_immutable_body(id).await?.source(),
            TurnBodySource::Materialized(_)
        ));
    }
    // The lookup still pins the physical old dataset for GC, even though its
    // logical answers survived publication. Dropping it releases that pin.
    let versions = staging.protected_bases().await?;
    assert!(versions
        .datasets()
        .any(|base| base.version_id() == covered.dataset().version_id()));
    drop(versions);
    publish(&staging, &["c", "later"]).await?;
    let mut before = staging.observation().await;
    assert_eq!(before.materialized, 4);
    // Now publication includes writes after capture. The old negative answer
    // must not authorize insertion of an already materialized row.
    assert_eq!(
        staging
            .register_immutable_committed(4, 5, &rows, &covered)
            .await?,
        None
    );
    before.immutable_lookup_invalidated += 1;
    assert_eq!(staging.observation().await, before);
    drop((plain, nonblocking_plain, fresh, covered));
    let observed = staging.observation().await;
    assert_eq!(observed.readers, 0);
    assert_eq!(observed.retained_entries, 0);
    assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    Ok(())
}

#[tokio::test]
async fn covered_lookup_survives_native_maintenance_but_rejects_recovered_instance() -> Result<()> {
    let root = tempfile::tempdir()?;
    let batch = bodies(&[])?;
    let base = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            root.path().join("body.lance").to_str().unwrap(),
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await?,
    );
    let staging = AnnotationStaging::new_immutable_body(base, scope(), 0, Limits::default())?;
    for (after, id) in [(0, "a"), (1, "b")] {
        let lookup = staging.capture_immutable_base().await?;
        assert_eq!(
            staging
                .register_immutable_committed(after, after + 1, &request(&[id]), &lookup)
                .await?,
            Some(ApplyOutcome::Applied)
        );
        publish(&staging, &[id]).await?;
    }
    let mut keys = BTreeMap::from([("a".to_owned(), false), ("new".to_owned(), false)]);
    let covered = staging
        .try_capture_immutable_base_with_staged(&mut keys)
        .await?
        .unwrap();
    let base = covered.dataset().clone();
    let plain = staging.capture_immutable_base().await?;
    assert_eq!(base.count_rows(Some("id = 'a'".into())).await?, 1);
    keys.insert("a".into(), true);
    let rows = keys
        .iter()
        .map(|(key, present)| ImmutableRegistration {
            key,
            exists_in_base: *present,
        })
        .collect::<Vec<_>>();
    let recovered =
        AnnotationStaging::new_immutable_body(base.clone(), scope(), 2, Limits::default())?;
    let mut before = recovered.observation().await;
    assert_eq!(
        recovered
            .register_immutable_committed(2, 3, &request(&["new"]), &covered)
            .await?,
        None
    );
    assert_eq!(before.immutable_lookup_reused, 0);
    assert_eq!(before.immutable_lookup_invalidated, 0);
    before.immutable_lookup_invalidated += 1;
    assert_eq!(recovered.observation().await, before);

    // A real native rewrite preserves membership and the cut. Only the
    // staged-covered query may survive; a plain capture still needs refresh.
    let transaction = super::super::compaction::compact(&base).await?;
    let selected = super::super::compaction::select(base.clone(), &transaction).await?;
    staging
        .install_compacted(&transaction, selected.clone())
        .await?;
    assert_eq!(
        staging
            .register_immutable_committed(2, 3, &rows, &plain)
            .await?,
        None
    );
    assert_eq!(
        staging
            .register_immutable_committed(2, 3, &rows, &covered)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    let first = origin(&staging.capture_immutable_body("new").await?);
    assert_eq!(first.record_index, 3);
    assert_eq!(staging.observation().await.active_entries, 1);
    assert_eq!(staging.observation().await.immutable_lookup_reused, 1);

    // Index-only installation has the same logical proof, without reusing
    // physical rows or losing the old dataset's GC protection.
    let mut indexed = selected.as_ref().clone();
    let index = indexed
        .create_index_builder(
            &["id"],
            IndexType::Scalar,
            &ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::ZoneMap),
        )
        .name("immutable_membership_test".into())
        .execute_uncommitted()
        .await?;
    let mut transaction = lance_table::transaction::Transaction::new_from_version(
        selected.version_id(),
        lance_table::transaction::Operation::CreateIndex {
            new_indices: vec![index],
            removed_indices: vec![],
        },
    );
    transaction.transaction_properties = selected
        .read_transaction()
        .await?
        .unwrap()
        .transaction_properties;
    let indexed = super::super::compaction::select(selected, &transaction).await?;
    staging.install_indexed(&transaction, indexed).await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    assert_eq!(
        staging
            .register_immutable_committed(3, 4, &rows, &covered)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    assert_eq!(origin(&staging.capture_immutable_body("new").await?), first);
    let observed = staging.observation().await;
    assert_eq!(observed.materialized, 2);
    assert_eq!(observed.active_entries, 0);
    assert_eq!(observed.frozen_entries, 1);
    assert_eq!(observed.immutable_lookup_reused, 2);
    assert_eq!(observed.immutable_lookup_invalidated, 1);
    assert!(staging
        .protected_bases()
        .await?
        .datasets()
        .any(|dataset| dataset.version_id() == base.version_id()));
    assert_eq!(covered.dataset().count_rows(None).await?, 2);

    // Materializing an insertion after the captured cut still invalidates the
    // old negative answer, even though the same lineage survived maintenance.
    install(&staging, frozen, &["new"]).await?;
    assert_eq!(
        staging
            .register_immutable_committed(4, 5, &rows, &covered)
            .await?,
        None
    );
    assert_eq!(staging.observation().await.applied, 4);
    drop((covered, plain));
    assert_eq!(staging.observation().await.readers, 0);
    assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    Ok(())
}
