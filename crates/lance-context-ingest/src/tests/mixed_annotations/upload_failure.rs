use super::*;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn failed_body_upload_keeps_acked_values_readable_and_bounds_staging_until_restart(
) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = artifact_root()?;
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let faults = Arc::new(fault_store::Faults::default());
    let body = fault_store::storage(&root.path().join("body.lance"), faults.clone()).await?;
    seed(&c, &narrow, &body).await?;
    let base = annotation_publication::state(&c).await?;
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (_, _, recovery_cut) = runtime.http_binding();
    let p = producer(&a, 1).await?;
    faults.body_write_failure.store(true, Ordering::SeqCst);
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let first = request(&a, &p, 1, encoded(create(&["A"], &["s"], &[1])?)?).await?;
    let first_result = a.submit(&first, Duration::from_secs(10)).await?;
    tokio::time::timeout(Duration::from_secs(10), async {
        while !a
            .phase_observation()
            .iter()
            .any(|phase| phase.phase == "annotation_body_write" && phase.failed == 1)
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;

    // The failed upload has not selected a new Lance base. Its original
    // frozen cut must survive while later complete annotations remain active.
    let mut latest = None;
    for (sequence, status, score) in [
        (2, Some("reviewed"), None),
        (3, None, Some(0.9)),
        (4, None, Some(0.8)),
        (5, None, Some(0.7)),
    ] {
        let update = request(
            &a,
            &p,
            sequence,
            encoded(mixed::wire::patches(
                &patch(&["A"], status, score)?,
                &limits(),
            )?)?,
        )
        .await?;
        latest = Some(a.submit(&update, Duration::from_secs(10)).await?);
    }
    let waiting = request(
        &a,
        &p,
        6,
        encoded(mixed::wire::patches(
            &patch(&["A"], None, Some(0.6))?,
            &limits(),
        )?)?,
    )
    .await?;
    let accepted = a.ingress().append(&waiting).await?;
    tokio::time::timeout(Duration::from_secs(10), async {
        while a.apply_backpressure().is_none() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert!(
        !task.is_finished(),
        "uncommitted upload failure must not stop point reads"
    );
    let (observed, pressured) = a.observation().await;
    assert!(pressured);
    assert_eq!(observed.applied, latest.unwrap().index);
    assert!(observed.applied < accepted.position.index);
    assert_eq!(a.source_observation().records, 4);
    let (frames, bytes) = a.retained_body_frames();
    assert_eq!(frames, 1);
    assert!(bytes > 0 && bytes <= 8 << 20);
    let current = a
        .get_owned("A", recovery_cut, Duration::from_secs(1))
        .await?
        .unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.7));
    let paused = annotation_publication::state(&c).await?;
    assert_eq!(paused.manifests, base.manifests);
    assert!(paused.progress.pending.is_some());
    assert!(paused.progress.prepared.is_none());
    let attempts = faults.body_write_attempts.load(Ordering::SeqCst);
    assert!(attempts > 0);
    // Completing additional owned reads must not create another upload or
    // invent a retry of the client mutation while publication is paused.
    for _ in 0..8 {
        assert_eq!(
            a.get_owned("A", recovery_cut, Duration::from_secs(1))
                .await?,
            Some(current.clone())
        );
    }
    assert_eq!(faults.body_write_attempts.load(Ordering::SeqCst), attempts);
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    faults.body_write_failure.store(false, Ordering::SeqCst);
    c.shutdown().await?;

    // Fenced recovery replays the original pending cut and its suffix. It
    // neither exposes the failed file nor reimports a second visible turn.
    c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    assert_eq!(
        a.submit(&first, Duration::from_secs(10)).await?,
        first_result
    );
    let completed = a.submit(&waiting, Duration::from_secs(10)).await?;
    assert_eq!(completed.index, accepted.position.index);
    annotation_runtime::drained(&c, &a, completed.index).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.6)).await?;
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
    drop(a);
    c.shutdown().await
}
