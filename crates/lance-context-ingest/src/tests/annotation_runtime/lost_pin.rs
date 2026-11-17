//! An uncertain source pin holds one frozen generation without killing reads/apply.
use super::*;
use std::sync::atomic::Ordering;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn persistent_unknown_pin_keeps_owner_live_and_drains() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(false, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_unknown_pin_recovers_original_plan_and_acked_suffix() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(true, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rpc_unknown_pin_keeps_owner_live_and_drains() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(false, true)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_rpc_unknown_pin_recovers_original_plan_and_acked_suffix() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(true, true)).await?
}

async fn exercise(restart: bool, rpc_unknown: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let proxy = if rpc_unknown {
        lost_prepare::proxy::Proxy::open_rpc_unknown_pin(&c, root.path(), false, true).await?
    } else {
        lost_prepare::proxy::Proxy::open_pin(&c, root.path(), false, true).await?
    };
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let (service, runtime) = owner(&c, &storage, &limits).await?;
    let (stop, mut task) = run_owner(runtime);
    let a = producer(&service, 1).await?;
    let b = producer(&service, 2).await?;
    let d = producer(&service, 3).await?;
    let first = request(
        &service,
        &a,
        1,
        wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?,
    )
    .await?;
    let first = service.submit(&first, Duration::from_secs(30)).await?;
    assert_eq!(first.result, MutationResult::Applied);
    tokio::time::timeout(Duration::from_secs(10), proxy.committed.notified()).await?;
    let retained = annotation_publication::state(&c).await?;
    let pending = retained.progress.pending.clone().unwrap();
    assert!(retained.progress.prepared.is_none());
    assert!(retained.progress.published.is_none());

    let score = request(
        &service,
        &b,
        1,
        wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let status = request(
        &service,
        &d,
        1,
        wire(&limits, &["T"], 1, Some("approved"), 0, None)?,
    )
    .await?;
    let (score, status) = tokio::join!(
        service.submit(&score, Duration::from_secs(30)),
        service.submit(&status, Duration::from_secs(30)),
    );
    let (score, status) = (score?, status?);
    assert_eq!(score.result, MutationResult::Applied);
    assert_eq!(status.result, MutationResult::Applied);
    assert!(pending.plan.sources[&10].through < score.index.min(status.index));
    let current = service.get("T", Duration::from_secs(10)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("approved"));
    assert_eq!(current.score, Some(0.9));
    proxy.unknown_pins.store(true, Ordering::SeqCst);
    proxy.release.notify_one();
    tokio::select! {
        result = &mut task => bail!("uncertain Pin killed owner: {result:?}"),
        result = tokio::time::timeout(Duration::from_secs(5), async {
            while proxy.pin_requests.lock().unwrap().len() < 3 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }) => result?,
    }
    let view = annotation_publication::state(&c).await?;
    assert_eq!(view.progress.pending, Some(pending.clone()));
    assert!(view.progress.prepared.is_none());
    assert!(view.progress.published.is_none());
    assert_eq!(view.progress.checkpoint, retained.progress.checkpoint);
    assert_eq!(view.manifests, retained.manifests);
    let requests = proxy.pin_requests.lock().unwrap().clone();
    assert!(requests.len() >= 3);
    assert!(requests.iter().all(|r| r == &requests[0]));
    assert_eq!(service.source_observation().records, 2);
    assert_eq!(
        service.get("T", Duration::from_secs(10)).await?.unwrap(),
        current
    );

    let mut through = score.index.max(status.index);
    if !restart {
        let queued = request(
            &service,
            &a,
            2,
            wire(&limits, &["T"], 1, Some("approved"), 0, None)?,
        )
        .await?;
        let queued = service.submit(&queued, Duration::from_secs(30));
        tokio::pin!(queued);
        tokio::select! {
            result = &mut queued => bail!("full source window ACKed another mutation: {result:?}"),
            result = &mut task => bail!("owner exited during pin backpressure: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(250)) => {},
        }
        assert_eq!(service.source_observation().records, 2);
        assert_eq!(
            service
                .get_owned("T", through, Duration::from_secs(10))
                .await?
                .unwrap(),
            current
        );
        proxy.unknown_pins.store(false, Ordering::SeqCst);
        let ack = queued.await?;
        assert_eq!(ack.result, MutationResult::Applied);
        through = through.max(ack.index);
        tokio::select! {
            result = drained(&c, &service, through) => result?,
            result = &mut task => bail!("owner exited before pinned publication drained: {result:?}"),
        }
        assert_eq!(
            service.get("T", Duration::from_secs(10)).await?.unwrap(),
            current
        );
        let requests = proxy.pin_requests.lock().unwrap().clone();
        let original: Vec<_> = requests
            .iter()
            .filter(|r| r.selection().transaction == pending.plan.transaction)
            .collect();
        assert!(!original.is_empty());
        assert!(original.iter().all(|r| *r == original[0]));
        stop.send(()).unwrap();
        task.await??;
        c.client = direct;
        drop(proxy);
        return c.shutdown().await;
    }

    stop.send(()).unwrap();
    task.await??;
    let stopped = annotation_publication::state(&c).await?;
    assert_eq!(stopped.progress.pending, Some(pending));
    assert!(stopped.progress.prepared.is_none());
    assert_eq!(stopped.progress.checkpoint, retained.progress.checkpoint);
    for ack in [&first, &score, &status] {
        assert!(matches!(
            direct
                .request_routed(Request::Fetch {
                    group: 10,
                    index: ack.index
                })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
    }
    drop(service);
    c.client = direct;
    drop(proxy);
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (service, runtime) = owner(&c, &storage, &limits).await?;
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service.get("T", Duration::from_secs(10)).await?.unwrap(),
        current
    );
    drained(&c, &service, through).await?;
    assert_eq!(
        service.get("T", Duration::from_secs(10)).await?.unwrap(),
        current
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
