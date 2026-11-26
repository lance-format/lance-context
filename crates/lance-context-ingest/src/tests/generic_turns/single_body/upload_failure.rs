use super::*;
use crate::annotations::runtime::PublicationWriteFailed;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;
use std::sync::atomic::Ordering;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_upload_failure_exits_and_fenced_recovery_drains_original_acked_cut() -> Result<()>
{
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(fault_store::Faults::default());
    let storage =
        fault_store::storage(&root.path().join("full-body.lance"), faults.clone()).await?;
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
    let before = annotation_publication::state(&c).await?;
    let (service, mut runtime) = owner(&c, &storage, &spec).await?;
    let lane = producer(&service, 1).await?;
    let bytes = fixture::encoded(single_body::batch(
        &spec,
        &fixture::limits(),
        Shard { bucket: 0 },
        &input[0],
    )?)?;
    let append = request(&service, &lane, 1, bytes).await?;
    // Drive only ordered apply until ACK, so the subsequent upload failure
    // is deterministically after the request became visible and durable.
    let completed = tokio::time::timeout(DEADLINE, async {
        let submitted = service.submit(&append, DEADLINE);
        tokio::pin!(submitted);
        loop {
            runtime.apply_once_for_test().await?;
            tokio::select! {
                result = &mut submitted => break result,
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        }
    })
    .await??;
    assert_eq!(completed.result, MutationResult::Applied);
    let key = spec.row_key(&input[0], 0, 4096)?;
    let held = service.get_full_body(&key, DEADLINE).await?.unwrap();
    faults.body_write_failure.store(true, Ordering::SeqCst);
    let error = tokio::time::timeout(DEADLINE, runtime.run(std::future::pending()))
        .await?
        .expect_err("failed body writer must notify its supervisor");
    assert!(error.is::<PublicationWriteFailed>(), "{error:#}");
    assert_eq!(held.batch(), &input[0].slice(0, 1));
    let failed = annotation_publication::state(&c).await?;
    assert_eq!(failed.manifests, before.manifests);
    assert!(failed.progress.pending.is_some());
    assert!(failed.progress.prepared.is_none());
    assert!(faults.body_write_attempts.load(Ordering::SeqCst) > 0);
    drop((held, service));
    faults.body_write_failure.store(false, Ordering::SeqCst);
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    drained(&c, &service, completed.index).await?;
    let selected = service.selected_full_body(DEADLINE).await?;
    let durable =
        annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, 2);
    assert_eq!(service.submit(&append, DEADLINE).await?, completed);
    assert_eq!(service.retained_body_frames(), (0, 0));
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
