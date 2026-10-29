//! A confirmed publication must release dirty capacity before source retirement.
use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn published_annotations_release_staging_while_close_reply_is_pending() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = tempfile::tempdir()?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    // One published value occupies the entire entry budget. The later value
    // can fit only after the installed generation releases its actual permits.
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 1,
        value_bytes: 4096,
        readers: 16,
    };
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let proxy = proxy::Proxy::open_close(&c, root.path()).await?;
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let (service, runtime) = owner(&c, &storage, &limits).await?;
    let (_stop, task) = run_owner(runtime);
    let a = producer(&service, 1).await?;
    let b = producer(&service, 2).await?;
    let first = request(
        &service,
        &a,
        1,
        wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?,
    )
    .await?;
    let first_result = service.submit(&first, Duration::from_secs(30)).await?;
    assert_eq!(first_result.result, MutationResult::Applied);
    tokio::time::timeout(Duration::from_secs(30), proxy.committed.notified()).await?;
    let observed = service.observation().await.0;
    assert_eq!(observed.materialized, first_result.index);
    assert_eq!(observed.frozen_entries, 0);
    // The catalog has not advanced; only the confirmed selected publication
    // and local read view changed. Source retirement is deliberately unfinished.
    let retained = annotation_publication::state(&c).await?;
    assert!(retained.progress.pending.is_some());
    assert!(retained.progress.published.is_some());
    assert!(retained.progress.checkpoint.get(&10).copied().unwrap_or(0) < first_result.index);

    let suffix = request(
        &service,
        &b,
        1,
        wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let suffix_result = tokio::time::timeout(
        Duration::from_secs(3),
        service.submit(&suffix, Duration::from_secs(30)),
    )
    .await;
    // Always release the held reply before reporting a regression failure.
    let current = if matches!(&suffix_result, Ok(Ok(_))) {
        Some(service.get("T", Duration::from_secs(1)).await?.unwrap())
    } else {
        None
    };
    proxy.release.notify_one();
    let error = tokio::time::timeout(Duration::from_secs(30), task)
        .await??
        .unwrap_err();
    let suffix_result = suffix_result
        .context("confirmed publication retained staging capacity while source closure waited")??;
    assert_eq!(suffix_result.result, MutationResult::Applied);
    let current = current.unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));
    assert!(
        format!("{error:#}").contains("outcome unknown"),
        "{error:#}"
    );
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    assert!(service.get("T", Duration::from_secs(1)).await.is_err());
    assert!(matches!(
        direct
            .request_routed(Request::Fetch {
                group: 10,
                index: suffix_result.index,
            })
            .await?
            .reply,
        Reply::Record(Some(_))
    ));
    drop(service);
    c.client = direct;
    drop(proxy);
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (service, runtime) = owner(&c, &storage, &limits).await?;
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service.get("T", Duration::from_secs(30)).await?,
        Some(current.clone())
    );
    drained(&c, &service, suffix_result.index).await?;
    for (request, result) in [(&first, &first_result), (&suffix, &suffix_result)] {
        assert_eq!(
            &service.submit(request, Duration::from_secs(30)).await?,
            result
        );
    }
    assert_eq!(
        service.get("T", Duration::from_secs(30)).await?,
        Some(current)
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
