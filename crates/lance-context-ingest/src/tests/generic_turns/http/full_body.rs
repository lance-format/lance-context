use super::*;
use crate::{
    annotations::single_body,
    http::{annotations::ReplaceProducerReply, journal::ProducerControlClient},
};

async fn full_client(
    root: &std::path::Path,
    base: reqwest::Url,
    spec: &TurnIngestionSpec,
    open: ProducerOpen,
) -> Result<DurableMutationClient> {
    DurableMutationClient::open_full_body(
        Config {
            directory: root.join("client-journal"),
            opens: vec![open],
            base,
            request_timeout: DEADLINE,
            journal: Settings {
                state_bytes: 16 << 20,
                queue_bytes: 8 << 20,
                batch_bytes: 2 << 20,
                file_bytes: 64 << 10,
                log_bytes: 96 << 20,
                ..Default::default()
            },
        },
        &annotation_publication::one_shard(),
        binding(),
        spec,
        &fixture::limits(),
        Shard { bucket: 0 },
        reqwest::Client::builder()
            .retry(reqwest::retry::never())
            .build()?,
        MergeMemoryBudget::new(64 << 20),
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_http_reconciles_across_ingresses_and_replaces_producer() -> Result<()> {
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
    let storage = fixture::storage(root.path(), "full-body.lance").await?;
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
    let (service, runtime) = super::super::single_body::owner(&c, &storage, &spec).await?;
    // Both constructors validate reply limits against the actual single-body
    // wire schema; no annotation/body auxiliary table is present.
    let owner = Owner::new(&runtime, keys()?, Default::default())?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let lost = Arc::new(LostReply::default());
    let server = Server::start(
        owner
            .router()
            .layer(middleware::from_fn_with_state(lost.clone(), lose_reply)),
    )
    .await?;
    let forward = || {
        Forwarder::new(
            vec![(service.ingress().clone(), server.base.to_string())],
            keys()?,
            Default::default(),
            "test-only-token-012345678901234567890".into(),
        )
    };
    let first = forward()?;
    let second = forward()?;
    let a = Server::start(first.router()).await?;
    let b = Server::start(second.router()).await?;
    let original_open = ProducerOpen {
        bucket: 0,
        slot: 1,
        expected_generation: 0,
        operation: Uuid::from_u128(101),
    };
    let writer = full_client(root.path(), a.base.clone(), &spec, original_open.clone()).await?;
    assert!(
        writer.append(0, encode(0)?).await.is_err(),
        "injected lost ACK"
    );
    let original = lost
        .completion
        .lock()
        .unwrap()
        .clone()
        .context("missing lost ACK")?;
    writer.shutdown().await?;
    let writer = full_client(root.path(), b.base.clone(), &spec, original_open.clone()).await?;
    assert_eq!(writer.resume(0).await?, Some(original.clone()));
    assert!(writer.resume(0).await?.is_none());
    let key = spec.row_key(&input[0], 0, 4096)?;
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    annotation_runtime::drained(&c, &service, original.completion.index).await?;
    let selected = writer.selected_full_body().await?;
    assert_eq!(selected.uri, storage.uri);
    assert!(selected.through >= original.completion.index);
    assert_eq!(selected.incarnation, binding().incarnation);
    assert!(
        writer.selected_body().await.is_err(),
        "full-body descriptor is not mixed selection"
    );
    assert_eq!(
        reqwest::Client::new()
            .post(b.base.join("api/v1/annotations/rollouts/selected-body")?)
            .json(&serde_json::json!({"bucket":1}))
            .send()
            .await?
            .status(),
        reqwest::StatusCode::BAD_REQUEST
    );
    writer.shutdown().await?;

    let control = |base| {
        ProducerControlClient::new_full_body(
            base,
            reqwest::Client::builder()
                .retry(reqwest::retry::never())
                .build()?,
            DEADLINE,
            &annotation_publication::one_shard(),
            &binding(),
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
        )
    };
    let ctl = control(a.base.clone())?;
    let intent = ctl
        .prepare_replacement(&original_open, Uuid::new_v4())
        .await?;
    let saved = serde_json::to_vec(&intent)?;
    assert!(
        ctl.execute(&intent).await.is_err(),
        "injected lost replacement reply"
    );
    let intent = serde_json::from_slice(&saved)?;
    let ctl = control(b.base.clone())?;
    let ReplaceProducerReply::Opened { open, .. } = ctl.execute(&intent).await? else {
        bail!("retired producer was not replaced")
    };
    assert_eq!(open.expected_generation, 1);
    let successor_root = root.path().join("successor");
    std::fs::create_dir(&successor_root)?;
    let writer = full_client(&successor_root, b.base.clone(), &spec, open.clone()).await?;
    let replayed = writer.append(0, encode(1)?).await?;
    assert_eq!(replayed.completion.result, MutationResult::Applied);
    annotation_runtime::drained(&c, &service, replayed.completion.index).await?;
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    let selected = writer.selected_full_body().await?;
    let durable =
        annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, 2);
    let ReplaceProducerReply::Opened { open: again, .. } = ctl.execute(&intent).await? else {
        bail!("same replacement intent must resolve")
    };
    assert_eq!(again, open);
    writer.shutdown().await?;
    first.shutdown().await;
    second.shutdown().await;
    a.shutdown().await?;
    b.shutdown().await?;
    owner.shutdown().await;
    server.shutdown().await?;
    let _ = stop.send(());
    task.await??;
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replacement_flushes_the_predecessor_tail_before_batch_deadline() -> Result<()> {
    let input = bodies()?.remove(0);
    let spec = TurnIngestionSpec::new(
        input.schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let layout = annotation_publication::one_shard();
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let storage = fixture::storage(root.path(), "full-body.lance").await?;
    let shard = Shard { bucket: 0 };
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        layout.clone(),
        binding(),
        storage.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &fixture::limits(),
        shard,
        &spec,
    )
    .await?;
    let (service, runtime) =
        super::super::single_body::owner_with_wait(&c, &storage, &spec, Duration::from_secs(60))
            .await?;
    let owner = Owner::new(&runtime, keys()?, Default::default())?;
    let server = Server::start(owner.router()).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let original = ProducerOpen {
        bucket: 0,
        slot: 1,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    let writer = full_client(root.path(), server.base.clone(), &spec, original.clone()).await?;
    let payload = fixture::encoded(single_body::batch(
        &spec,
        &fixture::limits(),
        shard,
        &input,
    )?)?;
    let appended = writer.append(0, payload).await?.completion;
    assert_eq!(appended.result, MutationResult::Applied);
    assert_eq!(service.source_observation().records, 1);
    writer.shutdown().await?;

    let control = ProducerControlClient::new_full_body(
        server.base.clone(),
        reqwest::Client::builder()
            .retry(reqwest::retry::never())
            .build()?,
        DEADLINE,
        &layout,
        &binding(),
        &spec,
        &fixture::limits(),
        shard,
    )?;
    let intent = control
        .prepare_replacement(&original, Uuid::new_v4())
        .await?;
    let ReplaceProducerReply::WaitingForRetirement { last: Some(last) } =
        control.execute(&intent).await?
    else {
        bail!("an applied predecessor must wait for signed source retirement");
    };
    assert_eq!(last.index, appended.index);
    tokio::time::timeout(
        Duration::from_secs(10),
        annotation_runtime::drained(&c, &service, last.index),
    )
    .await
    .context("replacement did not trigger publication before the batch deadline")??;
    let ReplaceProducerReply::Opened { open, .. } = control.execute(&intent).await? else {
        bail!("the signed closure must allow the same replacement to open");
    };
    assert_eq!(open.expected_generation, 1);

    owner.shutdown().await;
    server.shutdown().await?;
    let _ = stop.send(());
    task.await??;
    c.shutdown().await
}
