//! Bulk bodies share only narrow ID lookups; application remains per request.
use super::*;

fn bulk_binding() -> TableBinding {
    TableBinding {
        max_record_bytes: 8 << 20,
        max_rows: 512,
        ..binding()
    }
}

fn bulk_limits() -> Limits {
    Limits {
        retained_bytes: 16 << 20,
        retained_entries: 4096,
        ..limits()
    }
}

async fn connect_bulk(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    budget: Arc<MergeMemoryBudget>,
) -> Result<(Service, service::Driver)> {
    Service::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        bulk_binding(),
        narrow,
        body.clone(),
        bulk_limits(),
        MergeMemoryBudget::new(256 << 20),
        budget,
    )
    .await
}

fn bulk(first: usize, rows: usize, byte: u8, body_bytes: usize) -> Result<RecordBatch> {
    let ids = (first..first + rows)
        .map(|i| format!("bulk-{i}"))
        .collect::<Vec<_>>();
    let ids = ids.iter().map(String::as_str).collect::<Vec<_>>();
    let batch = create_with_limits(&ids, &vec!["s"; rows], &vec![byte; rows], &bulk_limits())?;
    let mut columns = batch.columns().to_vec();
    columns[batch.schema().index_of("binary_payload")?] = Arc::new(
        LargeBinaryArray::from_iter_values((0..rows).map(|_| vec![byte; body_bytes])),
    );
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

#[tokio::test]
async fn bulk_lookup_preserves_update_after_frozen_publication_and_reimport() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed_with_limits(&c, &narrow, &body, bulk_binding(), &bulk_limits()).await?;
    let budget = MergeMemoryBudget::new(128 << 20);
    let (a, mut driver) = connect_bulk(&c, &narrow, &body, budget.clone()).await?;
    let inputs = backlog(
        &c,
        &a,
        vec![
            bulk(0, 256, 1, 16 << 10)?,
            mixed::wire::patches(
                &patch_with_limits(&["bulk-0"], Some("reviewed"), Some(0.9), &bulk_limits())?,
                &bulk_limits(),
            )?,
            bulk(0, 256, 9, 16 << 10)?,
            bulk(256, 256, 2, 16 << 10)?,
        ],
    )
    .await?;
    driver.step_for_test().await?;
    assert_eq!(completed(&a, "annotation_registration_lookup"), 1);
    assert_eq!(completed(&a, "annotation_decode"), 4);
    assert_eq!(a.observation().await.0.applied, inputs[0].1);
    assert_eq!(a.source_observation().records, 1);
    let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
    assert_eq!(frozen.through(), inputs[0].1);
    driver.step_for_test().await?;
    assert_eq!(a.observation().await.0.applied, inputs[1].1);
    pressure::publish_frozen(&c, &a, &narrow, &body, frozen).await?;
    // The old base lookup is now stale. Reimport must retain the first body
    // and both fields of the patch applied after the captured publication cut.
    for (_, index) in &inputs[2..] {
        driver.step_for_test().await?;
        assert_eq!(a.observation().await.0.applied, *index);
    }
    let (stop, task) = run(driver);
    for (input, index) in &inputs {
        let result = a.submit(input, Duration::from_secs(5)).await?;
        assert_eq!(result.index, *index);
        assert_eq!(result.result, service::MutationResult::Applied);
    }
    check(&a, "bulk-0", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "bulk-255", "s", 1, "pending", None).await?;
    check(&a, "bulk-256", "s", 2, "pending", None).await?;
    pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(
        annotation_publication::read(&c, &narrow, None)
            .await?
            .count_rows(None)
            .await?,
        512
    );
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (a, driver) = connect_bulk(&c, &narrow, &body, budget.clone()).await?;
    let (stop, task) = run(driver);
    check(&a, "bulk-0", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "bulk-511", "s", 2, "pending", None).await?;
    for (input, index) in &inputs {
        assert_eq!(a.submit(input, Duration::from_secs(5)).await?.index, *index);
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[tokio::test]
async fn bulk_lookahead_stops_at_total_payload_and_key_limits() -> Result<()> {
    // Nine ~4 MiB requests cross the byte cap before the request/key caps;
    // six small-body 512-row requests cross the key cap first.
    for (rows, bytes, requests, decoded) in [(256, 16 << 10, 9, 7), (512, 64, 6, 4)] {
        let root = artifact_root()?;
        let c =
            network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
        let narrow = storage(root.path(), "annotations.lance").await?;
        let body = storage(root.path(), "body.lance").await?;
        seed_with_limits(&c, &narrow, &body, bulk_binding(), &bulk_limits()).await?;
        let budget = MergeMemoryBudget::new(256 << 20);
        let (a, mut driver) = connect_bulk(&c, &narrow, &body, budget.clone()).await?;
        let inputs = backlog(
            &c,
            &a,
            (0..requests)
                .map(|i| bulk(i * rows, rows, i as u8, bytes))
                .collect::<Result<Vec<_>>>()?,
        )
        .await?;
        driver.step_for_test().await?;
        assert_eq!(completed(&a, "annotation_decode"), decoded);
        assert_eq!(completed(&a, "annotation_registration_lookup"), 1);
        assert_eq!(a.observation().await.0.applied, inputs[0].1);
        assert_eq!(a.source_observation().records, 1);
        assert!(budget.reserved() <= budget.limit());
        // Decoding the suffix does not make it applied or part of this cut.
        pressure::publish(&c, &a, &narrow, &body).await?;
        assert_eq!(a.observation().await.0.materialized, inputs[0].1);
        drop(driver);
        drop(a);
        assert_eq!(budget.reserved(), 0);
        c.shutdown().await?;

        let c =
            network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
        let (a, mut driver) = connect_bulk(&c, &narrow, &body, budget.clone()).await?;
        // The original durable suffix must replay without any prefetched state.
        tokio::time::timeout(Duration::from_secs(5), async {
            while a.observation().await.0.applied < inputs.last().unwrap().1 {
                driver.step_for_test().await?;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        pressure::publish(&c, &a, &narrow, &body).await?;
        drop(driver);
        assert_eq!(a.retained_body_frames(), (0, 0));
        assert_eq!(budget.reserved(), 0);
        assert_eq!(
            annotation_publication::read(&c, &narrow, None)
                .await?
                .count_rows(None)
                .await?,
            rows * requests
        );
        c.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn bulk_lookup_falls_back_when_only_the_head_fits_memory() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed_with_limits(&c, &narrow, &body, bulk_binding(), &bulk_limits()).await?;
    let budget = MergeMemoryBudget::new(32 << 20);
    let held = budget.reserve(16 << 20).await;
    let (a, mut driver) = connect_bulk(&c, &narrow, &body, budget.clone()).await?;
    let inputs = backlog(
        &c,
        &a,
        vec![bulk(0, 256, 1, 16 << 10)?, bulk(256, 256, 2, 16 << 10)?],
    )
    .await?;
    tokio::time::timeout(Duration::from_secs(5), driver.step_for_test()).await??;
    assert_eq!(completed(&a, "annotation_decode"), 1);
    assert_eq!(completed(&a, "annotation_registration_lookup"), 0);
    assert_eq!(a.observation().await.0.applied, inputs[0].1);
    driver.step_for_test().await?;
    assert_eq!(
        a.apply_backpressure(),
        Some(service::ApplyBackpressure::ApplyMemory)
    );
    assert_eq!(a.observation().await.0.applied, inputs[0].1);
    pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(budget.reserved(), held.bytes());
    tokio::time::timeout(Duration::from_secs(5), driver.step_for_test()).await??;
    assert_eq!(a.observation().await.0.applied, inputs[1].1);
    assert_eq!(a.apply_backpressure(), None);
    pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(budget.reserved(), held.bytes());
    drop(held);
    assert_eq!(budget.reserved(), 0);
    drop(driver);
    c.shutdown().await
}
