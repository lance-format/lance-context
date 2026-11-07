use super::*;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store::{
    self, Action, Faults,
};
use tokio::sync::oneshot;

async fn abandon_body(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    body_path: &std::path::Path,
) -> Result<(PreparedAppend, service::Completed, PathBuf)> {
    let before = files(&selected_body(c, narrow, body).await?, body_path);
    let (a, driver) = Service::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        compact_binding(),
        narrow,
        body.clone(),
        limits(),
        MergeMemoryBudget::new(8 << 20),
        MergeMemoryBudget::new(8 << 20),
    )
    .await?;
    let (stop, task) = run(driver);
    let p = producer(&a, 2).await?;
    let input = request(&a, &p, 1, encoded(create(&["D"], &["s"], &[4])?)?).await?;
    let result = a.submit(&input, Duration::from_secs(10)).await?;
    let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare_mixed(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let transaction = &prepared.annotations().merge().transaction;
    let reference: BodySnapshot = serde_json::from_str(
        &transaction.transaction_properties.as_ref().unwrap()[BODY_SNAPSHOT_PROPERTY],
    )?;
    let uploaded = mixed::load_body(body, &reference, &binding().incarnation.to_string()).await?;
    let abandoned: Vec<_> = files(&uploaded, body_path)
        .difference(&before)
        .cloned()
        .collect();
    assert_eq!(abandoned.len(), 1);
    assert!(annotation_publication::state(c)
        .await?
        .progress
        .pending
        .is_none());
    // ACKed D remains in WAL, but none of these native files were selected or
    // prepared in quorum. Recovery must rebuild from the durable source.
    stop.send(()).unwrap();
    task.await??;
    drop((uploaded, prepared, frozen, a));
    Ok((input, result, abandoned[0].clone()))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_body_delete_outliving_its_owner_cannot_delete_successor_files() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let path = root.path().join("body.lance");
    let faults = Arc::new(Faults::default());
    let body = fault_store::storage(&path, faults.clone()).await?;
    let (last, last_result) = fragmented(&c, &narrow, &body).await?;
    let (initial, initial_result, abandoned) = abandon_body(&c, &narrow, &body, &path).await?;
    let (entered, reached) = oneshot::channel();
    let (release, resume) = oneshot::channel();
    faults.arm(&abandoned, Action::Pause { entered, resume })?;
    let (old, runtime) = compact_owner(&c, &narrow, &body).await?;
    let (_old_stop, old_task) = annotation_runtime::run_owner(collecting(runtime)?);
    tokio::time::timeout(Duration::from_secs(30), reached).await??;
    assert!(abandoned.exists());

    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = fault_store::storage(&path, Arc::new(Faults::default())).await?;
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    // Keep the new owner from collecting the paused orphan, so the exact old
    // sealed DELETE is the operation that ultimately removes it.
    let (stop, task) = annotation_runtime::run_owner(runtime);
    assert_eq!(
        a.submit(&initial, Duration::from_secs(10)).await?,
        initial_result
    );
    let p = producer(&a, 3).await?;
    let create = request(&a, &p, 1, encoded(create(&["E"], &["s"], &[5])?)?).await?;
    let created = a.submit(&create, Duration::from_secs(10)).await?;
    let q = producer(&a, 4).await?;
    let update = request(
        &a,
        &q,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    let updated = a.submit(&update, Duration::from_secs(10)).await?;
    annotation_runtime::drained(&c, &a, updated.index).await?;
    let live_body = selected_body(&c, &narrow, &body).await?;
    let live = files(&live_body, &path);
    assert!(!live.contains(&abandoned));
    release.send(()).unwrap();
    faults.finish_remote().await?;
    let error = tokio::time::timeout(Duration::from_secs(30), old_task)
        .await??
        .unwrap_err();
    let message = format!("{error:#}");
    // The old publisher may reach Plan or Prepare before observing the
    // successor's source fence. Both replicated commands reject stale authority.
    assert!(
        message.contains("fenced")
            || message.contains("view is behind source closure")
            || message == "publication selection rejected: Publisher(Rejected(Authority))"
            || message == "materialization result rejected: Publisher(Rejected(Authority))",
        "{message}"
    );
    assert!(!abandoned.exists());
    assert!(live.iter().all(|p| p.exists()));
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "D", "s", 4, "pending", None).await?;
    check(&a, "E", "s", 5, "pending", None).await?;
    for (input, result) in [
        (&last, &last_result),
        (&initial, &initial_result),
        (&create, &created),
        (&update, &updated),
    ] {
        assert_eq!(&a.submit(input, Duration::from_secs(10)).await?, result);
    }
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert_eq!(live_body.count_rows(None).await?, 5);
    stop.send(()).unwrap();
    task.await??;
    drop((old, a));
    c.shutdown().await
}
