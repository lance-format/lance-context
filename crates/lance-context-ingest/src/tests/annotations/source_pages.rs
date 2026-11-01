//! Buffered committed metadata must not become visible application progress.
use super::*;

fn completed(service: &Service, phase: &str) -> u64 {
    service
        .phase_observation()
        .into_iter()
        .find(|value| value.phase == phase)
        .unwrap()
        .completed
}

#[tokio::test]
async fn control_only_pages_advance_read_barriers_without_immediate_repoll() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits::default();
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let (service, mut driver) = connect(&c, &storage, &limits).await?;
    assert!(!driver.step_for_test().await?);
    let original = service.get("T", Duration::from_secs(5)).await?;
    assert!(original.is_some());

    for slot in 1..=4 {
        let before = service.observation().await.0.applied;
        // Producer opens are real quorum-committed control records. Each
        // advances the shared group while leaving this shard's values alone.
        producer(&service, slot).await?;
        assert!(!driver.step_for_test().await?);
        assert!(service.observation().await.0.applied > before);
        // No driver is running: the control-only step itself must satisfy the
        // read barrier even though it tells the scheduler to wait before polling.
        assert_eq!(service.get("T", Duration::from_secs(5)).await?, original);
    }
    assert_eq!(completed(&service, "annotation_fetch"), 0);
    assert_eq!(completed(&service, "annotation_decode"), 0);
    drop(driver);
    c.shutdown().await
}

#[tokio::test]
async fn source_pages_preserve_apply_order_pressure_and_concurrent_publication() -> Result<()> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("annotation-source-pages-");
    let root = match std::env::var("INGEST_TEST_ARTIFACTS") {
        Ok(parent) => builder.tempdir_in(parent)?,
        Err(_) => builder.tempdir()?,
    }
    .keep();
    let c = network::Cluster::open_signed(&root, annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits::default();
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let budget = MergeMemoryBudget::new(8 << 20);
    let (service, mut driver) = Service::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        &storage,
        limits.clone(),
        MergeMemoryBudget::new(8 << 20),
        budget.clone(),
    )
    .await?;
    let (stale, mut stale_driver) = connect(&c, &storage, &limits).await?;
    let stale_initial = stale.observation().await.0.applied;
    let producer = producer(&service, 1).await?;
    let mut indices = Vec::new();
    for sequence in 1..=67 {
        let status = format!("status-{sequence}");
        let input = request(
            &service,
            &producer,
            sequence,
            wire(
                &limits,
                &[if sequence == 33 { "missing" } else { "T" }],
                1,
                Some(&status),
                u8::from(sequence == 1),
                (sequence == 1).then_some(0.9),
            )?,
        )
        .await?;
        // Build a committed backlog without returning application ACKs.
        let response = c
            .client
            .request_routed(Request::Append {
                group: 10,
                batch: input.batch.clone(),
            })
            .await?;
        let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = response.reply else {
            bail!("test backlog append failed");
        };
        indices.push(index);
    }
    let initial = service.observation().await.0.applied;
    let held = budget.reserve(budget.limit()).await;
    for _ in 0..3 {
        assert!(!driver.step_for_test().await?);
        assert_eq!(service.observation().await.0.applied, initial);
        assert!(service.observation().await.1);
        assert_eq!(
            service.apply_backpressure(),
            Some(annotations::ApplyBackpressure::ApplyMemory)
        );
        assert_eq!(completed(&service, "annotation_fetch"), 0);
        assert_eq!(completed(&service, "annotation_observe"), 1);
    }
    drop(held);
    assert!(driver.step_for_test().await?);
    assert_eq!(service.observation().await.0.applied, indices[0]);
    assert_eq!(service.apply_backpressure(), None);
    let frozen = service.freeze(Uuid::new_v4()).await?.unwrap();
    assert!(driver.step_for_test().await?);
    assert_eq!(service.observation().await.0.applied, indices[1]);
    assert_eq!(completed(&service, "annotation_observe"), 1);

    // Publish the first update while the second is active and the remainder
    // is still only committed metadata. Retirement cannot discard either.
    let (prepared, selection, descriptors, commit) =
        crate::tests::annotation_checkpoint::prepare(&c, &frozen, 0).await?;
    let published = crate::annotation_publication::publish(
        Arc::new(c.client.manifest_catalog()),
        &annotation_publication::one_shard(),
        &storage,
        commit,
        &selection,
        &descriptors,
        prepared.annotations().base().clone(),
    )
    .await?;
    service
        .install_published(&prepared, Arc::new(published))
        .await?;
    annotation_publication::finish(&c).await?;
    drop((prepared, frozen));

    // A successful metadata read cannot authorize advancing an obsolete base
    // after its required source prefix has been retired. Check this even when
    // the two quorum observations can finish in either order.
    let error = stale_driver.step_for_test().await.unwrap_err();
    assert!(format!("{error:#}").contains("view is behind source closure"));
    assert_eq!(stale.observation().await.0.applied, stale_initial);
    assert_eq!(completed(&stale, "annotation_fetch"), 0);
    drop((stale_driver, stale));

    for (offset, index) in indices.iter().enumerate().skip(2) {
        if offset == 32 {
            let held = budget.reserve(budget.limit()).await;
            assert!(!driver.step_for_test().await?);
            assert_eq!(service.observation().await.0.applied, indices[31]);
            assert_eq!(completed(&service, "annotation_fetch"), 32);
            assert_eq!(completed(&service, "annotation_observe"), 2);
            drop(held);
        }
        assert!(driver.step_for_test().await?);
        assert_eq!(service.observation().await.0.applied, *index);
        assert_eq!(completed(&service, "annotation_fetch"), offset as u64 + 1);
        assert_eq!(
            completed(&service, "annotation_observe"),
            offset as u64 / 32 + 1
        );
    }
    // The final page also contains publication control entries. Apply their
    // boundary only after all 67 data records, including the rejected patch.
    assert!(!driver.step_for_test().await?);
    let value = service.get("T", Duration::from_secs(5)).await?.unwrap();
    assert_eq!(value.status.as_deref(), Some("status-67"));
    assert_eq!(value.score, Some(0.9));
    assert_eq!(service.observation().await.0.materialized, indices[0]);
    assert_eq!(completed(&service, "annotation_fetch"), 67);
    assert_eq!(completed(&service, "annotation_observe"), 3);
    assert_eq!(budget.reserved(), 0);
    drop(driver);
    c.shutdown().await
}
