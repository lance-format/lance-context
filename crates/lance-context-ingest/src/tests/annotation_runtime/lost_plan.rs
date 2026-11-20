//! An uncertain Plan must not discard its frozen prefix or the ACKed suffix.
use super::*;
use futures::TryStreamExt;
use std::sync::atomic::Ordering;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_unknown_plan_keeps_concurrent_updates_and_drains() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(true, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_uncommitted_unknown_plan_preserves_selected_cut_and_suffix() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(120), async {
        exercise(false, false).await?;
        exercise(false, true).await
    })
    .await?
}

async fn exercise(execute: bool, rpc_unknown: bool) -> Result<()> {
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
    let proxy = lost_prepare::proxy::Proxy::open_plan(&c, root.path(), execute).await?;
    proxy.rpc_unknown_plans.store(rpc_unknown, Ordering::SeqCst);
    proxy.pause_plan_retry.store(!execute, Ordering::SeqCst);
    c.client = proxy.client.clone();
    let (service, runtime) = owner(&c, &storage, &limits).await?;
    let (stop, task) = run_owner(runtime);
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
    let original = proxy.plan_requests.lock().unwrap()[0].clone();
    let pending = annotation_publication::state(&c)
        .await?
        .progress
        .pending
        .clone();
    if execute {
        assert_eq!(pending.unwrap().plan, original.plan);
    } else {
        assert!(
            pending.is_none(),
            "original Plan unexpectedly committed before retry"
        );
    }
    assert_eq!(
        original.plan.sources[&10].records,
        vec![(first.index, first.meta.clone())]
    );
    // Hold either the committed reply or the uncommitted exact retry. New
    // different-field and same-field updates must remain visible and survive
    // publication of the previously selected prefix in both cases.
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
        service.submit(&status, Duration::from_secs(30))
    );
    let (score, status) = (score?, status?);
    assert_eq!(score.result, MutationResult::Applied);
    assert_eq!(status.result, MutationResult::Applied);
    let current = service.get("T", Duration::from_secs(10)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("approved"));
    assert_eq!(current.score, Some(0.9));
    proxy.release.notify_one();
    drained(&c, &service, score.index.max(status.index)).await?;
    assert_eq!(
        service.get("T", Duration::from_secs(10)).await?.unwrap(),
        current
    );
    let selected =
        annotation_publication::read(&c, &storage, Some(original.plan.target_version()?)).await?;
    assert_eq!(
        selected.read_transaction().await?.unwrap().uuid,
        original.plan.transaction.to_string()
    );
    // The first published snapshot must contain exactly the frozen value,
    // even though the same turn has newer acknowledged updates in active.
    let mut scan = selected.scan();
    scan.project(&["status", "score"])?;
    scan.filter("id = 'T'")?;
    let batches = scan
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );
    let row = batches.iter().find(|batch| batch.num_rows() > 0).unwrap();
    assert_eq!(
        row.column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap()
            .value(0),
        "reviewed"
    );
    assert!(
        row.column(1).is_null(0),
        "suffix score leaked into the frozen publication"
    );
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(
        proxy.plans.load(Ordering::SeqCst),
        if execute { 2 } else { 3 },
        "expected the original selection, its optional exact retry, and the suffix"
    );
    if !execute {
        let requests = proxy.plan_requests.lock().unwrap();
        assert_eq!(requests[0], original);
        assert_eq!(requests[1], original, "retry reselected the newer suffix");
    }
    stop.send(()).unwrap();
    task.await??;
    drop(proxy);
    c.shutdown().await
}
