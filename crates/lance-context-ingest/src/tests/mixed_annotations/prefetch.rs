//! Shared lookups must leave request visibility, publication and recovery ordered.
use super::*;

mod large;

fn completed(service: &Service, phase: &str) -> u64 {
    service
        .phase_observation()
        .into_iter()
        .find(|value| value.phase == phase)
        .unwrap()
        .completed
}

async fn with_budget(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    budget: Arc<MergeMemoryBudget>,
) -> Result<(Service, service::Driver)> {
    Service::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        narrow,
        body.clone(),
        limits(),
        MergeMemoryBudget::new(16 << 20),
        budget,
    )
    .await
}

async fn backlog(
    c: &network::Cluster,
    service: &Service,
    batches: Vec<RecordBatch>,
) -> Result<Vec<(PreparedAppend, u64)>> {
    let mut inputs = Vec::new();
    for (offset, batch) in batches.into_iter().enumerate() {
        // Each producer has one unresolved request, including rejected requests.
        let producer = producer(service, offset as u32 + 1).await?;
        let prepared = tokio::time::timeout(
            Duration::from_secs(5),
            request(service, &producer, 1, encoded(batch)?),
        )
        .await
        .context("test backlog upload admission timed out")??;
        let response = c
            .client
            .request_routed(Request::Append {
                group: 10,
                batch: prepared.batch.clone(),
            })
            .await?;
        let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = response.reply else {
            bail!("test backlog append failed");
        };
        inputs.push((prepared, index));
    }
    Ok(inputs)
}

#[tokio::test]
async fn shared_lookup_keeps_ack_and_frozen_cut_per_request() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed(&c, &narrow, &body).await?;
    let budget = MergeMemoryBudget::new(64 << 20);
    let (a, mut driver) = with_budget(&c, &narrow, &body, budget.clone()).await?;
    let inputs = Arc::new(
        backlog(
            &c,
            &a,
            vec![
                create(&["A"], &["s"], &[1])?,
                mixed::wire::patches(&patch(&["A"], Some("reviewed"), Some(0.9))?, &limits())?,
                create(&["A", "B"], &["wrong", "t"], &[9, 2])?,
                mixed::wire::patches(&patch(&["missing"], Some("bad"), None)?, &limits())?,
                create(&["C"], &["u"], &[3])?,
            ],
        )
        .await?,
    );
    let ack = tokio::spawn({
        let a = a.clone();
        let inputs = inputs.clone();
        async move { a.submit(&inputs[1].0, Duration::from_secs(30)).await }
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while !a
            .phase_observation()
            .iter()
            .any(|p| p.phase == "annotation_apply_wait" && p.started > 0)
        {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    driver.step_for_test().await?;
    assert_eq!(completed(&a, "annotation_decode"), 5);
    assert_eq!(completed(&a, "annotation_registration_lookup"), 1);
    assert_eq!(a.observation().await.0.applied, inputs[0].1);
    assert_eq!(a.source_observation().records, 1);
    assert!(!ack.is_finished(), "prefetch must not ACK a later request");
    let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
    assert_eq!(frozen.through(), inputs[0].1);
    assert_eq!(frozen.sources().records.len(), 1);
    driver.step_for_test().await?;
    assert_eq!(ack.await??.index, inputs[1].1);
    assert_eq!(a.observation().await.0.applied, inputs[1].1);
    pressure::publish_frozen(&c, &a, &narrow, &body, frozen).await?;
    assert_eq!(a.observation().await.0.materialized, inputs[0].1);
    #[cfg(feature = "http")]
    {
        let current = a
            .get_owned("A", inputs[1].1, Duration::from_secs(5))
            .await?
            .unwrap();
        assert_eq!(current.status.as_deref(), Some("reviewed"));
        assert_eq!(current.score, Some(0.9));
    }
    // Publication invalidates the captured lookup; active values still win.
    for (_, index) in inputs.iter().skip(2) {
        driver.step_for_test().await?;
        assert_eq!(a.observation().await.0.applied, *index);
    }
    assert_eq!(completed(&a, "annotation_decode"), 5);
    let (stop, task) = run(driver);
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "B", "t", 2, "pending", None).await?;
    check(&a, "C", "u", 3, "pending", None).await?;
    for (offset, (input, index)) in inputs.iter().enumerate() {
        let result = a.submit(input, Duration::from_secs(5)).await?;
        assert_eq!(result.index, *index);
        if offset == 3 {
            assert!(
                matches!(result.result, service::MutationResult::Rejected(ref r)
                if r.reason == lance_context_core::annotation_staging::RejectionReason::MissingTurn)
            );
        } else {
            assert_eq!(result.result, service::MutationResult::Applied);
        }
    }
    pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(
        annotation_publication::read(&c, &narrow, None)
            .await?
            .count_rows(None)
            .await?,
        3
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[tokio::test]
async fn one_request_budget_drains_without_waiting_for_lookahead_memory() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed(&c, &narrow, &body).await?;
    // Leave one rounded permit available while other work holds the rest.
    let budget = MergeMemoryBudget::new(4 << 20);
    let held = budget.reserve(3 << 20).await;
    let (a, mut driver) = with_budget(&c, &narrow, &body, budget.clone()).await?;
    let inputs = backlog(
        &c,
        &a,
        vec![create(&["A"], &["s"], &[1])?, create(&["B"], &["t"], &[2])?],
    )
    .await?;
    tokio::time::timeout(Duration::from_secs(5), driver.step_for_test()).await??;
    assert_eq!(a.observation().await.0.applied, inputs[0].1);
    assert_eq!(completed(&a, "annotation_decode"), 1);
    assert_eq!(completed(&a, "annotation_registration_lookup"), 0);
    driver.step_for_test().await?;
    assert_eq!(
        a.apply_backpressure(),
        Some(service::ApplyBackpressure::ApplyMemory)
    );
    assert_eq!(a.observation().await.0.applied, inputs[0].1);
    assert!(budget.reserved() <= budget.limit());
    pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(budget.reserved(), held.bytes());
    driver.step_for_test().await?;
    assert_eq!(a.observation().await.0.applied, inputs[1].1);
    assert_eq!(a.apply_backpressure(), None);
    pressure::publish(&c, &a, &narrow, &body).await?;
    let (stop, task) = run(driver);
    check(&a, "A", "s", 1, "pending", None).await?;
    check(&a, "B", "t", 2, "pending", None).await?;
    assert_eq!(budget.reserved(), held.bytes());
    stop.send(()).unwrap();
    task.await??;
    drop(held);
    assert_eq!(budget.reserved(), 0);
    c.shutdown().await
}

#[tokio::test]
async fn bounded_prefetched_suffix_is_rebuilt_after_owner_and_wal_restart() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed(&c, &narrow, &body).await?;
    let budget = MergeMemoryBudget::new(64 << 20);
    let (a, mut driver) = with_budget(&c, &narrow, &body, budget.clone()).await?;
    let ids = (0..10).map(|i| format!("turn-{i}")).collect::<Vec<_>>();
    let batches = ids
        .iter()
        .enumerate()
        .map(|(i, id)| create(&[id], &["s"], &[i as u8]))
        .collect::<Result<Vec<_>>>()?;
    let inputs = backlog(&c, &a, batches).await?;
    driver.step_for_test().await?;
    assert_eq!(completed(&a, "annotation_decode"), 8);
    assert_eq!(completed(&a, "annotation_registration_lookup"), 1);
    assert_eq!(a.observation().await.0.applied, inputs[0].1);
    pressure::publish(&c, &a, &narrow, &body).await?;
    // Seven decoded requests are still unapplied. Dropping the owner releases
    // their memory; recovery must rebuild them from the durable suffix.
    assert!(budget.reserved() > 0);
    drop(driver);
    drop(a);
    assert_eq!(budget.reserved(), 0);
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (a, driver) = with_budget(&c, &narrow, &body, budget.clone()).await?;
    assert_eq!(a.observation().await.0.materialized, inputs[0].1);
    let (stop, task) = run(driver);
    for (i, id) in ids.iter().enumerate() {
        check(&a, id, "s", i as u8, "pending", None).await?;
        let outcome = a.submit(&inputs[i].0, Duration::from_secs(5)).await?;
        assert_eq!(outcome.index, inputs[i].1);
        assert_eq!(outcome.result, service::MutationResult::Applied);
    }
    pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(
        annotation_publication::read(&c, &narrow, None)
            .await?
            .count_rows(None)
            .await?,
        10
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
