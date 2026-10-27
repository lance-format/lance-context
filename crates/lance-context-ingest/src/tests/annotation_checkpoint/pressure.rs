use super::*;

#[tokio::test]
async fn annotation_byte_pressure_waits_for_retained_generation_release() -> Result<()> {
    retained_generation(true).await
}

#[tokio::test]
async fn annotation_entry_pressure_waits_for_retained_generation_release() -> Result<()> {
    retained_generation(false).await
}

async fn retained_generation(byte_pressure: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits {
        retained_bytes: if byte_pressure { 3600 } else { 16 << 10 },
        retained_entries: if byte_pressure { 8 } else { 2 },
        value_bytes: 1792,
        readers: 16,
    };
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let (service, driver) = connect(&c, &storage, &limits).await?;
    let (stop, task) = run(driver);
    let producer = producer(&service, 1).await?;
    for (offset, key) in ["T", "U"].into_iter().enumerate() {
        let req = request(
            &service,
            &producer,
            offset as u64 + 1,
            wire(&limits, &[key], 1, Some("reviewed"), 0, None)?,
        )
        .await?;
        assert_eq!(
            service.submit(&req, Duration::from_secs(10)).await?.result,
            MutationResult::Applied
        );
    }
    let frozen = service.freeze(Uuid::new_v4()).await?.unwrap();
    // Byte pressure needs one active value as well as the two frozen values.
    // Entry pressure already fills both entry permits with the frozen batch.
    if byte_pressure {
        let req = request(
            &service,
            &producer,
            3,
            wire(&limits, &["T"], 0, None, 1, Some(0.5))?,
        )
        .await?;
        assert_eq!(
            service.submit(&req, Duration::from_secs(10)).await?.result,
            MutationResult::Applied
        );
    }
    let req = Arc::new(
        request(
            &service,
            &producer,
            if byte_pressure { 4 } else { 3 },
            wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
        )
        .await?,
    );
    let submitted = tokio::spawn({
        let service = service.clone();
        let req = req.clone();
        async move { service.submit(&req, Duration::from_secs(30)).await }
    });
    tokio::time::timeout(Duration::from_secs(10), async {
        while !service.observation().await.1 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let attempts = service.apply_attempts();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        service.apply_attempts(),
        attempts,
        "unchanged capacity must not rebuild the blocked batch"
    );
    assert!(!submitted.is_finished());

    let (prepared, selection, descriptors, commit) = prepare(&c, &frozen, 0).await?;
    let selected = publication::publish(
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
        .install_published(&prepared, Arc::new(selected))
        .await?;
    annotation_publication::finish(&c).await?;
    // Publication changes the base, but retained handles still charge capacity.
    tokio::time::timeout(Duration::from_secs(10), async {
        while service.apply_attempts() == attempts {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let attempts = service.apply_attempts();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(service.apply_attempts(), attempts);
    assert!(!submitted.is_finished());
    let through = frozen.through();
    drop((prepared, frozen));
    // Releasing capacity without another publication must resume the original
    // request. Its field patch retains the earlier successfully applied status.
    let completed = submitted.await??;
    assert_eq!(completed.result, MutationResult::Applied);
    assert_eq!(
        service.submit(&req, Duration::from_secs(10)).await?,
        completed
    );
    let value = service.get("T", Duration::from_secs(10)).await?.unwrap();
    assert_eq!(value.status.as_deref(), Some("reviewed"));
    assert_eq!(value.score, Some(0.9));
    assert_eq!(service.observation().await.0.materialized, through);
    publish(&c, &service, &storage, through).await?;
    let observed = service.observation().await.0;
    assert_eq!(observed.retained_bytes, 0);
    assert_eq!(observed.retained_entries, 0);
    assert_eq!(
        service.get("T", Duration::from_secs(10)).await?,
        Some(value)
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
