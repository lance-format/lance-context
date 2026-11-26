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
    let rows = keys
        .iter()
        .map(|(key, present)| ImmutableRegistration {
            key,
            exists_in_base: *present,
        })
        .collect::<Vec<_>>();
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
    let before = staging.observation().await;
    assert_eq!(before.materialized, 4);
    // Now publication includes writes after capture. The old negative answer
    // must not authorize insertion of an already materialized row.
    assert_eq!(
        staging
            .register_immutable_committed(4, 5, &rows, &covered)
            .await?,
        None
    );
    assert_eq!(staging.observation().await, before);
    drop((plain, nonblocking_plain, fresh, covered));
    let observed = staging.observation().await;
    assert_eq!(observed.readers, 0);
    assert_eq!(observed.retained_entries, 0);
    assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    Ok(())
}

#[tokio::test]
async fn covered_lookup_rejects_recovered_instance_and_native_maintenance() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = empty(&root.path().join("body.lance")).await?;
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
    let mut keys = BTreeMap::from([("new".to_owned(), false)]);
    let covered = staging
        .try_capture_immutable_base_with_staged(&mut keys)
        .await?
        .unwrap();
    let base = covered.dataset().clone();
    let recovered =
        AnnotationStaging::new_immutable_body(base.clone(), scope(), 2, Limits::default())?;
    let before = recovered.observation().await;
    assert_eq!(
        recovered
            .register_immutable_committed(2, 3, &request(&["new"]), &covered)
            .await?,
        None
    );
    assert_eq!(recovered.observation().await, before);

    // Real native rewrite with the same cut; only verified frozen publications
    // are eligible for reuse, so maintenance conservatively invalidates it.
    let transaction = super::super::compaction::compact(&base).await?;
    let selected = super::super::compaction::select(base, &transaction).await?;
    staging.install_compacted(&transaction, selected).await?;
    let before = staging.observation().await;
    assert_eq!(before.materialized, 2);
    assert_eq!(
        staging
            .register_immutable_committed(2, 3, &request(&["new"]), &covered)
            .await?,
        None
    );
    assert_eq!(staging.observation().await, before);
    let fresh = staging
        .try_capture_immutable_base_with_staged(&mut keys)
        .await?
        .unwrap();
    assert_eq!(
        staging
            .register_immutable_committed(2, 3, &request(&["new"]), &fresh)
            .await?,
        Some(ApplyOutcome::Applied)
    );
    Ok(())
}
