use super::*;
use crate::annotations::runtime::PublicationWriteFailed;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn failed_body_upload_exits_owner_and_fenced_recovery_preserves_acked_updates() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(60), exercise(false)).await?
}

#[tokio::test]
async fn failed_body_upload_response_recovers_concurrent_acked_suffix() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(true)).await?
}

async fn exercise(concurrent_suffix: bool) -> Result<()> {
    let root = artifact_root()?;
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let faults = Arc::new(fault_store::Faults::default());
    let body = fault_store::storage(&root.path().join("body.lance"), faults.clone()).await?;
    seed(&c, &narrow, &body).await?;
    let base = annotation_publication::state(&c).await?;
    // Apply the original create before starting any publication. The second
    // case acknowledges its patch while this create's upload is in flight.
    let (a, driver) = connect(&c, &narrow, &body).await?;
    let (stop, task) = run(driver);
    let p = producer(&a, 1).await?;
    let updates = producer(&a, 2).await?;
    let first = request(&a, &p, 1, encoded(create(&["A"], &["s"], &[1])?)?).await?;
    let first_result = a.submit(&first, Duration::from_secs(10)).await?;
    let update = request(
        &a,
        &updates,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    let mut update_result = if concurrent_suffix {
        None
    } else {
        Some(a.submit(&update, Duration::from_secs(10)).await?)
    };
    stop.send(()).unwrap();
    task.await??;
    drop(a);

    faults
        .body_write_failure
        .store(!concurrent_suffix, Ordering::SeqCst);
    let paused = concurrent_suffix.then(|| faults.pause_data_write());
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (_stop, task) = annotation_runtime::run_owner(runtime);
    if let Some((entered, resume)) = paused {
        // The data object exists, but its upload has not returned successfully
        // and no new manifest is selected. Dropping resume injects an I/O error
        // into that response after a later mutation has been acknowledged.
        tokio::time::timeout(Duration::from_secs(10), entered).await??;
        update_result = Some(a.submit(&update, Duration::from_secs(10)).await?);
        check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
        drop(resume);
    }
    let update_result = update_result.unwrap();
    let error = tokio::time::timeout(Duration::from_secs(10), task)
        .await??
        .expect_err("failed publication must notify its supervisor");
    assert!(error.is::<PublicationWriteFailed>(), "{error:#}");
    assert!(error.chain().any(|cause| matches!(
        cause.downcast_ref::<lance::Error>(),
        Some(lance::Error::IO { .. })
    )));
    let failed = annotation_publication::state(&c).await?;
    assert_eq!(failed.manifests, base.manifests);
    assert!(failed.progress.pending.is_some());
    assert!(failed.progress.prepared.is_none());
    if concurrent_suffix {
        let sources = &failed.progress.pending.as_ref().unwrap().plan.sources;
        assert_eq!(sources.len(), 1);
        assert!(sources.values().next().unwrap().through < update_result.index);
    }
    assert!(faults.body_write_attempts.load(Ordering::SeqCst) > 0);
    drop(a);
    faults.body_write_failure.store(false, Ordering::SeqCst);
    c.shutdown().await?;
    c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;

    // Recover the original publication cut and any acknowledged suffix from
    // disk after restarting every replica, without reimporting the source.
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    annotation_runtime::drained(&c, &a, update_result.index).await?;
    assert_eq!(
        a.submit(&first, Duration::from_secs(10)).await?,
        first_result
    );
    assert_eq!(
        a.submit(&update, Duration::from_secs(10)).await?,
        update_result
    );
    assert_eq!(a.retained_body_frames(), (0, 0));
    let selected = annotation_publication::read(&c, &narrow, None).await?;
    assert_eq!(selected.count_rows(None).await?, 1);
    let reference = BodySnapshot::read(&selected).await?.unwrap();
    assert_eq!(
        mixed::load_body(&body, &reference, &binding().incarnation.to_string())
            .await?
            .count_rows(None)
            .await?,
        1
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
