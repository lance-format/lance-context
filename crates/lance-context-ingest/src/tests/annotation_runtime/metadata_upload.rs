use super::*;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn metadata_upload_throttle_keeps_acks_readable_bounds_staging_and_drains() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (storage, scope, fault) =
        crate::annotations::metadata_upload::tests::local_storage(&root.path().join("turns.lance"))
            .await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let (service, runtime) = owner(&c, &storage, &limits).await?;
    scope.start();
    let (_, _, recovery_cut) = runtime.http_binding();
    let producer = producer(&service, 1).await?;
    fault.blocked.store(true, Ordering::SeqCst);
    let (stop, task) = run_owner(runtime);
    let first = request(
        &service,
        &producer,
        1,
        wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?,
    )
    .await?;
    service.submit(&first, Duration::from_secs(30)).await?;
    tokio::time::timeout(Duration::from_secs(30), fault.entered.notified()).await?;

    let second = request(
        &service,
        &producer,
        2,
        wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    service.submit(&second, Duration::from_secs(30)).await?;
    // Frozen has its own source range. Fill both slots of the next active
    // window before checking that a further durable proposal cannot be ACKed.
    let third = request(
        &service,
        &producer,
        3,
        wire(&limits, &["T"], 0, None, 1, Some(0.8))?,
    )
    .await?;
    let acknowledged = service.submit(&third, Duration::from_secs(30)).await?;
    let waiting = request(
        &service,
        &producer,
        4,
        wire(&limits, &["T"], 0, None, 1, Some(0.7))?,
    )
    .await?;
    let accepted = service.ingress().append(&waiting).await?;
    tokio::time::timeout(Duration::from_secs(10), async {
        while scope.observation().put_resubmissions < 2 || service.apply_backpressure().is_none() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert!(!task.is_finished());
    assert_eq!(scope.observation().waiting_objects, 1);
    let (state, pressured) = service.observation().await;
    assert!(pressured);
    assert_eq!(state.applied, acknowledged.index);
    assert!(state.applied < accepted.position.index);
    assert_eq!(service.source_observation().records, 2);
    let current = service
        .get_owned("T", recovery_cut, Duration::from_secs(1))
        .await?
        .unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.8));
    {
        let calls = fault.calls.lock().unwrap();
        let attempted = calls.last().unwrap();
        let repeats: Vec<_> = calls
            .iter()
            .filter(|(path, _)| path == &attempted.0)
            .collect();
        assert!(repeats.len() >= 3);
        for (_, parts) in repeats {
            assert_eq!(parts.len(), attempted.1.len());
            for (part, original) in parts.iter().zip(&attempted.1) {
                assert_eq!(part, original);
                assert_eq!(part.as_ptr(), original.as_ptr());
            }
        }
    }
    fault.blocked.store(false, Ordering::SeqCst);
    let completed = service.submit(&waiting, Duration::from_secs(30)).await?;
    assert_eq!(completed.index, accepted.position.index);
    drained(&c, &service, completed.index).await?;
    assert_eq!(scope.observation().waiting_objects, 0);
    let current = service.get("T", Duration::from_secs(10)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.7));
    stop.send(()).unwrap();
    task.await??;
    drop(scope);
    c.shutdown().await
}
