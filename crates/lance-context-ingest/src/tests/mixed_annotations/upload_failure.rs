use super::*;
use crate::annotations::runtime::PublicationWriteFailed;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn failed_body_upload_exits_owner_and_fenced_recovery_preserves_acked_updates() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let faults = Arc::new(fault_store::Faults::default());
    let body = fault_store::storage(&root.path().join("body.lance"), faults.clone()).await?;
    seed(&c, &narrow, &body).await?;
    let base = annotation_publication::state(&c).await?;
    // Apply acknowledged creates/patches before starting any publication.
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
    let update_result = a.submit(&update, Duration::from_secs(10)).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(a);

    faults.body_write_failure.store(true, Ordering::SeqCst);
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (_stop, task) = annotation_runtime::run_owner(runtime);
    let error = tokio::time::timeout(Duration::from_secs(10), task)
        .await??
        .expect_err("failed publication must notify its supervisor");
    assert!(error.is::<PublicationWriteFailed>(), "{error:#}");
    let failed = annotation_publication::state(&c).await?;
    assert_eq!(failed.manifests, base.manifests);
    assert!(failed.progress.pending.is_some());
    assert!(failed.progress.prepared.is_none());
    assert!(faults.body_write_attempts.load(Ordering::SeqCst) > 0);
    drop(a);
    faults.body_write_failure.store(false, Ordering::SeqCst);

    // A task supervisor can now restart immediately from the original WAL,
    // without waiting for a stuck-publication watchdog or reimporting a source.
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
