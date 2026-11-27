//! External fences cover acknowledged mutations, not control-only apply progress.
use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn selected_body_covers_ack_without_publishing_control_only_progress() -> Result<()> {
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
    let (service, mut runtime) = owner_with_wait(&c, &storage, &spec, Duration::ZERO).await?;
    let lane = producer(&service, 1).await?;
    let encoded = fixture::encoded(single_body::batch(
        &spec,
        &fixture::limits(),
        Shard { bucket: 0 },
        &input[0],
    )?)?;
    let prepared = request(&service, &lane, 1, encoded).await?;
    let ack = {
        let submitted = service.submit(&prepared, DEADLINE);
        tokio::pin!(submitted);
        loop {
            runtime.apply_once_for_test().await?;
            tokio::select! {
                result = &mut submitted => break result?,
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        }
    };
    assert_eq!(ack.result, MutationResult::Applied);
    // ACK makes the current body readable; selected_full_body is only the
    // durable selected snapshot and must not pretend to cover this new ACK.
    let old = service.selected_full_body(DEADLINE).await?;
    assert!(old.through < ack.index);
    assert_eq!(old.rows, Some(0));
    let key = spec.row_key(&input[0], 0, 4096)?;
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );

    runtime.publish_once_for_test().await?;
    drained(&c, &service, ack.index).await?;
    let selected = service.selected_full_body(DEADLINE).await?;
    assert_eq!(selected.through, ack.index);
    assert_eq!(selected.rows, Some(input[0].num_rows() as u64));
    // Publication itself adds committed control records after the data ACK.
    // The source scanner may apply those cursors without any new table input.
    while service.observation().await.0.applied <= selected.through {
        runtime.apply_once_for_test().await?;
    }
    let control_cut = service.observation().await.0.applied;
    assert_eq!(service.source_observation().records, 0);
    assert!(service.freeze(Uuid::new_v4()).await?.is_none());
    for _ in 0..3 {
        runtime.publish_once_for_test().await?;
        let current = service.selected_full_body(DEADLINE).await?;
        assert_eq!(current.manifest.version, selected.manifest.version);
        assert_eq!(current.through, ack.index);
        assert!(current.through < control_cut);
    }
    assert_eq!(service.submit(&prepared, DEADLINE).await?, ack);
    drop(runtime);
    drop(service);

    // Recovery needs only the selected snapshot and retained WAL. The exact
    // mutation retry and its publication coverage survive without a new write.
    let (recovered, runtime) = owner(&c, &storage, &spec).await?;
    assert_eq!(recovered.submit(&prepared, DEADLINE).await?, ack);
    let current = recovered.selected_full_body(DEADLINE).await?;
    assert_eq!(current.through, ack.index);
    assert_eq!(current.manifest.version, selected.manifest.version);
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        recovered
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    stop.send(()).unwrap();
    task.await??;
    drop(recovered);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn selected_body_covers_selected_pending_publication_but_not_acked_suffix() -> Result<()> {
    tokio::time::timeout(DEADLINE, pending_publication()).await?
}

async fn pending_publication() -> Result<()> {
    use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;

    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
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
    // Hold the real Confirm RPC after native selection. The original source
    // checkpoint has not advanced; selected_body must validate the pending plan.
    let proxy = Proxy::open_confirm(&c, root.path(), false).await?;
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let (stop, task) = run_owner(runtime);
    let lane = producer(&service, 1).await?;
    let suffix_lane = producer(&service, 2).await?;
    let encode = |i: usize| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i],
        )?)
    };
    let first = request(&service, &lane, 1, encode(0)?).await?;
    let ack = service.submit(&first, DEADLINE).await?;
    assert_eq!(ack.result, MutationResult::Applied);
    proxy.committed.notified().await;
    let view = annotation_publication::state(&c).await?;
    let pending = view.progress.pending.as_ref().unwrap();
    assert!(view.progress.prepared.is_some());
    assert!(view.progress.published.is_none());
    assert!(view.progress.checkpoint.get(&10).copied().unwrap_or(0) < ack.index);
    let selected = service.selected_full_body(DEADLINE).await?;
    assert_eq!(selected.manifest.version, pending.plan.target_version()?);
    assert!(selected.through >= ack.index);
    assert_eq!(selected.rows, Some(2));

    let suffix = request(&service, &suffix_lane, 1, encode(2)?).await?;
    let suffix_ack = service.submit(&suffix, DEADLINE).await?;
    assert_eq!(suffix_ack.result, MutationResult::Applied);
    let still_selected = service.selected_full_body(DEADLINE).await?;
    assert_eq!(still_selected.manifest.version, selected.manifest.version);
    assert_eq!(still_selected.through, selected.through);
    assert!(still_selected.through < suffix_ack.index);
    // The later body is already queryable even though this selected manifest
    // cannot yet authorize its external source checkpoint.
    let key = spec.row_key(&input[2], 2, 4096)?;
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[2].slice(2, 1)
    );
    proxy.release.notify_one();
    drained(&c, &service, suffix_ack.index).await?;
    let final_selected = service.selected_full_body(DEADLINE).await?;
    assert!(final_selected.through >= suffix_ack.index);
    assert_eq!(final_selected.rows, Some(4));
    assert_eq!(service.submit(&first, DEADLINE).await?, ack);
    assert_eq!(service.submit(&suffix, DEADLINE).await?, suffix_ack);
    stop.send(()).unwrap();
    task.await??;
    c.client = direct;
    drop(proxy);
    c.shutdown().await
}
