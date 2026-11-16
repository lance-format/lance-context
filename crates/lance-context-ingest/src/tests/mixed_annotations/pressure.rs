use super::*;

fn large_create(ids: &[&str]) -> Result<Vec<u8>> {
    let source = create(ids, &vec!["s"; ids.len()], &vec![7; ids.len()])?;
    let mut columns = source.columns().to_vec();
    columns[source.schema().index_of("binary_payload")?] = Arc::new(
        LargeBinaryArray::from_iter_values((0..ids.len()).map(|_| vec![7; 256 << 10])),
    );
    let batch = RecordBatch::try_new(source.schema(), columns)?;
    let mut bytes = Vec::with_capacity(ids.len() * (256 << 10) + (64 << 10));
    let mut writer = FileWriter::try_new(&mut bytes, &batch.schema())?;
    writer.write(&batch)?;
    writer.finish()?;
    drop(writer);
    Ok(bytes)
}

async fn received_request(
    service: &Service,
    producer: &Producer,
    sequence: u64,
    body: Vec<u8>,
) -> Result<PreparedAppend> {
    let upload = service.reserve_upload(body.len()).await?;
    let original = body.as_ptr();
    let prepared = service
        .prepare_owned(upload, body, producer, sequence)
        .await?;
    assert_eq!(original, prepared.body().as_ptr());
    Ok(prepared)
}

pub(super) async fn publish(
    c: &network::Cluster,
    service: &Service,
    narrow: &Storage,
    body: &Storage,
) -> Result<()> {
    let frozen = service
        .freeze(Uuid::new_v4())
        .await?
        .context("no checkpoint to drain")?;
    publish_frozen(c, service, narrow, body, frozen).await
}

pub(super) async fn publish_frozen(
    c: &network::Cluster,
    service: &Service,
    narrow: &Storage,
    body: &Storage,
    frozen: service::checkpoint::FrozenCheckpoint,
) -> Result<()> {
    let selection = annotation_publication::select_range(
        c,
        frozen.sources().records.clone(),
        frozen.through(),
        Uuid::parse_str(frozen.identity())?,
    )
    .await?;
    let prepared = frozen
        .prepare_mixed(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let descriptors = crate::annotation_publication::mixed::persist_descriptors(
        prepared.annotations(),
        &selection,
        narrow,
        body,
        c.client.max_descriptor_bytes(),
    )
    .await?;
    let commit = annotation_publication::save(c, &selection, descriptors.clone()).await?;
    let selected = crate::annotation_publication::mixed::publish(
        Arc::new(c.client.manifest_catalog()),
        &annotation_publication::one_shard(),
        narrow,
        body,
        commit,
        &selection,
        &descriptors,
        prepared.annotations().base().clone(),
    )
    .await?;
    annotation_publication::confirm(c).await?;
    service
        .install_published(&prepared, Arc::new(selected))
        .await?;
    drop((prepared, frozen));
    annotation_publication::finish(c).await
}

#[tokio::test]
async fn mixed_frame_pressure_withholds_ack_until_published_readers_release_capacity() -> Result<()>
{
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed(&c, &narrow, &body).await?;
    // One retained payload plus the next decode fits after transient credit is
    // returned. Two retained payloads plus a third decode must still wait.
    let apply = MergeMemoryBudget::new(4 << 20);
    let (a, driver) = Service::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        &narrow,
        body.clone(),
        limits(),
        MergeMemoryBudget::new(8 << 20),
        apply.clone(),
    )
    .await?;
    let (stop, task) = run(driver);
    let p = producer(&a, 1).await?;
    let first = received_request(&a, &p, 1, large_create(&["A", "B", "C"])?).await?;
    a.submit(&first, Duration::from_secs(10)).await?;
    let held = a.get_turn("A", Duration::from_secs(10)).await?.unwrap();
    let more = received_request(&a, &p, 2, large_create(&["G", "H", "I"])?).await?;
    a.submit(&more, Duration::from_secs(10)).await?;
    let also_held = a.get_turn("G", Duration::from_secs(10)).await?.unwrap();
    let second = Arc::new(received_request(&a, &p, 3, large_create(&["D", "E", "F"])?).await?);
    let submit = tokio::spawn({
        let a = a.clone();
        let second = second.clone();
        async move { a.submit(&second, Duration::from_secs(10)).await }
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if a.observation().await.1 {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await?;
    assert!(
        !submit.is_finished(),
        "a durable but unapplied import must not succeed"
    );
    assert!(apply.reserved() <= apply.limit());
    assert_eq!(a.source_observation().records, 2);
    publish(&c, &a, &narrow, &body).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert!(apply.reserved() > 0);
    assert!(
        !submit.is_finished(),
        "a held read must still consume its frame capacity"
    );
    assert_eq!(
        held.body()
            .column_by_name("binary_payload")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap()
            .value(0),
        vec![7; 256 << 10]
    );
    drop((held, also_held));
    let result = submit.await??;
    assert_eq!(result.result, service::MutationResult::Applied);
    assert!(apply.reserved() <= apply.limit());
    publish(&c, &a, &narrow, &body).await?;
    annotation_runtime::drained(&c, &a, result.index).await?;
    assert_eq!(apply.reserved(), 0);
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert_eq!(a.submit(&second, Duration::from_secs(10)).await?, result);
    assert_eq!(
        annotation_publication::read(&c, &narrow, None)
            .await?
            .count_rows(None)
            .await?,
        9
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    Ok(())
}
