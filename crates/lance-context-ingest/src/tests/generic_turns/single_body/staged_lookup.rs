use super::*;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn staged_duplicates_skip_history_and_publication_restores_base_lookup() -> Result<()> {
    tokio::time::timeout(DEADLINE, exercise()).await?
}

async fn exercise() -> Result<()> {
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(fault_store::Faults::default());
    let storage = fault_store::storage(&root.path().join("body.lance"), faults.clone()).await?;
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &fixture::limits(),
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let encode = |i: usize| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i],
        )?)
    };
    let (service, mut runtime) = owner(&c, &storage, &spec).await?;
    let initial = service.observation().await.0.installed_base;
    assert_eq!(initial.rows, Some(0));
    assert!(initial.immutable_body);
    let lane = producer(&service, 1).await?;
    // Apply without running publication: the second frame has the same keys
    // and different values, and must resolve entirely from active staging.
    for (sequence, body) in [(1, 0), (2, 1)] {
        let prepared = request(&service, &lane, sequence, encode(body)?).await?;
        let submitted = service.submit(&prepared, DEADLINE);
        tokio::pin!(submitted);
        let completed = loop {
            runtime.apply_once_for_test().await?;
            tokio::select! {
                result = &mut submitted => break result?,
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        };
        assert_eq!(completed.result, MutationResult::Applied);
    }
    assert_eq!(service.body_lookup_observation().scans_started, 1);
    assert_eq!(service.body_lookup_observation().staged_keys, 2);
    assert_eq!(service.selected_full_body(DEADLINE).await?.rows, Some(0));
    assert_eq!(service.observation().await.0.installed_base, initial);

    // Pause a real data-object upload after freezing, then apply both an
    // all-duplicate frame and a mixed frame while the selected base is old.
    let (entered, resume) = faults.pause_data_write();
    let (stop, task) = run_owner(runtime);
    entered.await?;
    let duplicate = request(&service, &lane, 3, encode(1)?).await?;
    let result = service.submit(&duplicate, DEADLINE).await?;
    assert_eq!(result.result, MutationResult::Applied);
    assert_eq!(service.body_lookup_observation().scans_started, 1);
    assert_eq!(service.body_lookup_observation().staged_keys, 4);
    let mixed = request(&service, &lane, 4, encode(2)?).await?;
    let result = service.submit(&mixed, DEADLINE).await?;
    assert_eq!(result.result, MutationResult::Applied);
    assert_eq!(service.body_lookup_observation().scans_started, 2);
    assert_eq!(service.body_lookup_observation().staged_keys, 6);
    assert_eq!(service.selected_full_body(DEADLINE).await?.rows, Some(0));
    assert_eq!(service.observation().await.0.installed_base, initial);
    for (row, original) in [
        (0, &input[0]),
        (1, &input[0]),
        (2, &input[2]),
        (3, &input[2]),
    ] {
        let key = spec.row_key(&input[2], row, 4096)?;
        assert_eq!(
            service
                .get_full_body(&key, DEADLINE)
                .await?
                .unwrap()
                .batch(),
            &original.slice(row, 1)
        );
    }
    resume.send(()).unwrap();
    drained(&c, &service, result.index).await?;
    assert_eq!(service.retained_body_frames(), (0, 0));
    let selected = service.selected_full_body(DEADLINE).await?;
    assert_eq!(selected.rows, Some(4));
    let installed = service.observation().await.0;
    assert_eq!(installed.installed_base.rows, selected.rows);
    assert_eq!(installed.installed_base.version, selected.manifest.version);
    assert_eq!(installed.installed_base.source, selected.source);
    assert_eq!(installed.materialized, selected.through);
    let durable =
        annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, 4);
    let before = service.body_lookup_observation();
    let repeated = request(&service, &lane, 5, encode(1)?).await?;
    let result = service.submit(&repeated, DEADLINE).await?;
    let after = service.body_lookup_observation();
    assert_eq!(after.scans_started, before.scans_started + 1);
    assert_eq!(after.staged_keys, before.staged_keys);
    // A materialized key is no longer a staged hit; the original body wins.
    let key = spec.row_key(&input[0], 0, 4096)?;
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    drained(&c, &service, result.index).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
