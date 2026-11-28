use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bounded_foreground_coalesces_eight_independent_lanes_after_source_gap() -> Result<()> {
    tokio::time::timeout(DEADLINE, foreground(false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bounded_foreground_deadline_survives_cancellation_and_low_traffic() -> Result<()> {
    tokio::time::timeout(DEADLINE, foreground(true)).await?
}

async fn foreground(low_traffic: bool) -> Result<()> {
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let encode = |i| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i],
        )?)
    };
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = fixture::storage(root.path(), "coalescing.lance").await?;
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
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let (stop, task) = run_owner(runtime);
    let seed_lane = producer(&service, 15).await?;
    let seed = request(&service, &seed_lane, 1, encode(0)?).await?;
    let seeded = service.submit(&seed, DEADLINE).await?;
    drained(&c, &service, seeded.index).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);

    let state = annotation_publication::state(&c).await?;
    let (service, runtime) = Runtime::connect_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Configuration {
            limits: fixture::limits(),
            upload_budget: MergeMemoryBudget::new(64 << 20),
            apply_budget: MergeMemoryBudget::new(64 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 64,
                max_batch_wait: Duration::from_millis(20),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let wait = Duration::from_secs(1);
    let mut runtime = runtime.with_body_lookup_batch_wait(wait)?;
    let mut requests = Vec::new();
    for slot in 0..8 {
        let lane = producer(&service, slot).await?;
        requests.push(request(&service, &lane, 1, encode(2)?).await?);
    }
    let append = |batch| {
        c.client
            .request_routed(Request::Append { group: 10, batch })
    };
    let first = append(requests[0].batch.clone()).await?;
    let Reply::Commit(Outcome::Committed(Applied::Accepted { index: head })) = first.reply else {
        bail!("head not committed")
    };
    let applied = service.observation().await.0.applied;
    let before = service
        .body_lookup_observation()
        .prefetch
        .source_empty_at_local_end;
    {
        let step = runtime.apply_once_for_test();
        tokio::pin!(step);
        loop {
            tokio::select! {
                result = &mut step => panic!("underfilled query escaped coalescing: {result:?}"),
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
            if service
                .body_lookup_observation()
                .prefetch
                .source_empty_at_local_end
                > before
            {
                break;
            }
        }
        assert_eq!(service.observation().await.0.applied, applied);
        assert_eq!(service.body_lookup_observation().completed_lookups, 0);
        // Cancel while waiting for arrivals. The same driver retains admitted
        // bodies, metadata and its original deadline, not a new wait budget.
    }
    let mut through = head;
    if low_traffic {
        tokio::time::sleep(wait + Duration::from_millis(25)).await;
        tokio::time::timeout(Duration::from_millis(750), runtime.apply_once_for_test()).await??;
    } else {
        tokio::time::sleep(Duration::from_millis(25)).await;
        for request in &requests[1..] {
            let response = append(request.batch.clone()).await?;
            let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = response.reply
            else {
                bail!("late independent lane not committed")
            };
            through = index;
        }
        runtime.apply_once_for_test().await?;
    }
    assert_eq!(service.observation().await.0.applied, head);
    let observed = service.body_lookup_observation();
    assert_eq!(observed.completed_lookups, 1);
    assert_eq!(observed.lookup_requests, if low_traffic { 1 } else { 8 });
    eprintln!("foreground coalesced low_traffic={low_traffic}: {observed:?}");
    let (stop, task) = run_owner(runtime);
    drained(&c, &service, through).await?;
    for request in requests.iter().take(if low_traffic { 1 } else { 8 }) {
        assert_eq!(
            service.submit(request, DEADLINE).await?.result,
            MutationResult::Applied
        );
    }
    let selected = service.selected_full_body(DEADLINE).await?;
    let durable =
        annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, input[2].num_rows());
    let original = spec.row_key(&input[0], 0, 4096)?;
    assert_eq!(
        service
            .get_full_body(&original, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    assert_eq!(service.retained_body_frames(), (0, 0));
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
