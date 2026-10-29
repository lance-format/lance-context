use super::*;

fn configuration(limits: &Limits) -> Configuration {
    Configuration {
        limits: limits.clone(),
        upload_budget: MergeMemoryBudget::new(8 << 20),
        apply_budget: MergeMemoryBudget::new(8 << 20),
        publication_budget: MergeMemoryBudget::new(8 << 20),
        publication: Options {
            max_source_bytes: 1 << 20,
            min_source_bytes: 1 << 20,
            max_records: 2,
            max_batch_wait: Duration::from_millis(20),
            ..Default::default()
        },
    }
}

#[tokio::test]
async fn annotation_owner_validates_then_fences_before_opening_lance() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let path = root.path().join("turns.lance");
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let original = annotation_publication::state(&c).await?;
    let acquisition = Acquisition {
        owner: Uuid::new_v4(),
        request: Uuid::new_v4(),
        expected_epoch: original.authority.epoch,
    };
    let connect = |configuration, acquisition| {
        Runtime::connect(
            c.client.clone(),
            annotation_publication::one_shard(),
            binding(),
            storage.clone(),
            configuration,
            acquisition,
        )
    };

    // Invalid local capacity must not fence an otherwise valid publisher.
    let mut invalid = configuration(&limits);
    invalid.apply_budget = invalid.upload_budget.clone();
    let error = connect(invalid, acquisition.clone())
        .await
        .err()
        .expect("shared apply/upload budget was accepted");
    assert!(error.to_string().contains("independent reserved workspace"));
    assert_eq!(
        annotation_publication::state(&c).await?.authority,
        original.authority
    );

    // Make the actual checkpoint unreadable, without changing catalog state.
    // The replacement must fence its predecessor before attempting this I/O.
    let manifests = original.manifests.as_ref().unwrap();
    let manifest = path.join(&manifests.versions[&manifests.head].object.path);
    let unavailable = root.path().join("unavailable.manifest");
    std::fs::rename(&manifest, &unavailable)?;
    let result = connect(configuration(&limits), acquisition.clone()).await;
    std::fs::rename(&unavailable, &manifest)?;
    let error = result.err().expect("unreadable checkpoint was opened");
    assert!(
        format!("{error:#}").contains(manifest.file_name().unwrap().to_str().unwrap()),
        "expected checkpoint I/O failure, got: {error:#}"
    );
    let acquired = annotation_publication::state(&c).await?.authority.clone();
    assert_eq!(
        acquired.owner, acquisition.owner,
        "checkpoint I/O ran before the predecessor was fenced"
    );
    assert_eq!(acquired.epoch, acquisition.expected_epoch + 1);

    // Resume the persisted acquisition identity after restoring storage. It
    // must reuse the grant, not steal another epoch to hide the failed open.
    let (service, runtime) = connect(configuration(&limits), acquisition).await?;
    assert_eq!(annotation_publication::state(&c).await?.authority, acquired);
    let (stop, task) = run_owner(runtime);
    let producer = producer(&service, 1).await?;
    let update = request(
        &service,
        &producer,
        1,
        wire(&limits, &["T"], 1, Some("reviewed"), 1, Some(0.9))?,
    )
    .await?;
    let completed = service.submit(&update, Duration::from_secs(30)).await?;
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));
    drained(&c, &service, completed.index).await?;
    assert_eq!(
        service.submit(&update, Duration::from_secs(30)).await?,
        completed
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    Ok(())
}
