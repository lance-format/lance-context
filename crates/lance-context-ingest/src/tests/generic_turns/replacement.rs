use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replacement_waits_for_retirement_and_fences_only_its_predecessor() -> Result<()> {
    let root = tempfile::tempdir()?;
    let source = bodies()?.remove(0);
    let spec = TurnIngestionSpec::new(
        source.schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let ids = spec.prepare(source.clone(), 4096)?.keys().to_vec();
    let create = encode(mixed::wire::creates_for(
        &spec,
        &source,
        &fixture::patch(
            &ids.iter().map(String::as_str).collect::<Vec<_>>(),
            Some("pending"),
            None,
        )?,
        &fixture::limits(),
    )?)?;
    let patch = encode(mixed::wire::patches_for(
        &spec,
        &fixture::patch(&[&ids[0]], Some("reviewed"), Some(0.9))?,
        &fixture::limits(),
    )?)?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = fixture::storage(root.path(), "narrow.lance").await?;
    let body = fixture::storage(root.path(), "body.lance").await?;
    provision_turn_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        narrow.clone(),
        body.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        fixture::limits(),
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let (a, driver) = connect(&c, &narrow, &body, &spec).await?;
    let (stop, task) = run(driver);
    let p = producer(&a, 1).await?;
    a.submit(&request(&a, &p, 1, create.clone()).await?, DEADLINE)
        .await?;
    let updated = a
        .submit(&request(&a, &p, 2, patch.clone()).await?, DEADLINE)
        .await?;
    let stale = request(&a, &p, 3, patch).await?;
    let operation = Uuid::new_v4();
    // Apply is running but publication/retirement is deliberately paused.
    for _ in 0..2 {
        let ProducerReplacement::WaitingForRetirement { last } =
            a.ingress().replace_producer(&p, operation).await?
        else {
            panic!("premature reuse")
        };
        assert_eq!(
            last,
            Some(lance_context_wal::replication::producer::Last {
                sequence: 2,
                index: updated.index,
            })
        );
    }
    // The pending replacement and its fence survive an owner/applier restart.
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    let (a, driver) = connect(&c, &narrow, &body, &spec).await?;
    let (stop, task) = run(driver);
    assert!(matches!(
        a.ingress().replace_producer(&p, operation).await?,
        ProducerReplacement::WaitingForRetirement { .. }
    ));
    let error = a.ingress().append(&stale).await.unwrap_err();
    assert_eq!(
        error.downcast_ref::<AppendFailure>(),
        Some(&AppendFailure::ProducerFenced)
    );
    assert_eq!(a.get(&ids[0], DEADLINE).await?.unwrap().score, Some(0.9));
    fixture::pressure::publish(&c, &a, &narrow, &body).await?;
    let selected = a.selected_body(DEADLINE).await?;
    let ProducerReplacement::Opened(next) = a.ingress().replace_producer(&p, operation).await?
    else {
        panic!("retired tail should be reusable")
    };
    assert_eq!(next.generation(), p.generation() + 1);
    // The new executor can replay first-create without resetting later labels.
    a.submit(&request(&a, &next, 1, create).await?, DEADLINE)
        .await?;
    assert_eq!(a.get(&ids[0], DEADLINE).await?.unwrap().score, Some(0.9));
    fixture::pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(selected.body, a.selected_body(DEADLINE).await?.body);
    let ProducerReplacement::Opened(retried) = a.ingress().replace_producer(&p, operation).await?
    else {
        panic!("same replacement must resolve")
    };
    assert_eq!(next.id(), retried.id());
    assert!(a
        .ingress()
        .replace_producer(&p, Uuid::new_v4())
        .await
        .unwrap_err()
        .is::<ProducerReplacementFenced>());
    let ProducerReplacement::Opened(third) =
        a.ingress().replace_producer(&next, Uuid::new_v4()).await?
    else {
        panic!("second tail retired")
    };
    assert_eq!(third.generation(), next.generation() + 1);
    assert!(a
        .ingress()
        .replace_producer(&p, operation)
        .await
        .unwrap_err()
        .is::<ProducerReplacementFenced>());
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    Ok(())
}
