//! A real quorum commits Prepare, but its successful reply never reaches the owner.
//! Confirm it in place when observable; preserve fenced recovery when it is not.
use super::*;
use std::sync::atomic::Ordering;

mod close;
pub(in crate::tests) mod proxy;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_unknown_prepare_preserves_acked_suffix_and_reuses_original_files() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(60), exercise(false, true)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_unknown_prepare_keeps_owner_live_and_preserves_concurrent_updates() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(60), exercise(true, true)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unrecorded_prepare_resumes_original_files_and_preserves_concurrent_updates() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(60), exercise(true, false)).await?
}

async fn exercise(confirm: bool, recorded: bool) -> Result<()> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("annotation-lost-prepare-");
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        builder.disable_cleanup(true).tempdir_in(parent)?
    } else {
        builder.tempdir()?
    };
    eprintln!("lost Prepare reply fixture: {}", root.path().display());
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
    let base = annotation_publication::read(&c, &storage, None).await?;
    let proxy = if confirm {
        proxy::Proxy::open_prepare(&c, root.path(), recorded, true).await?
    } else {
        proxy::Proxy::open(&c, root.path()).await?
    };
    if !recorded {
        proxy
            .pause_uncommitted_prepare
            .store(true, Ordering::SeqCst);
    }
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
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
    let first_result = service.submit(&first, Duration::from_secs(30)).await?;
    assert_eq!(first_result.result, MutationResult::Applied);
    tokio::time::timeout(Duration::from_secs(30), proxy.committed.notified()).await?;

    // The publisher waits for its Prepare outcome, committed or not yet recorded.
    // Apply remains live, and both later field updates must be ACKed now.
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
    let (score_result, status_result) = tokio::join!(
        service.submit(&score, Duration::from_secs(30)),
        service.submit(&status, Duration::from_secs(30)),
    );
    let score_result = score_result?;
    let status_result = status_result?;
    assert_eq!(score_result.result, MutationResult::Applied);
    assert_eq!(status_result.result, MutationResult::Applied);
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("approved"));
    assert_eq!(current.score, Some(0.9));
    let retained = annotation_publication::state(&c).await?;
    let pending = retained.progress.pending.clone().unwrap();
    let prepared = retained.progress.prepared.clone();
    assert_eq!(prepared.is_some(), recorded);
    let preparation = proxy.prepare_requests.lock().unwrap()[0].clone();
    if let Some(prepared) = &prepared {
        assert_eq!(prepared.descriptors, preparation.descriptors);
    }
    let target = pending.plan.target_version()?;
    let expected_transaction = publication::transaction(
        &base,
        &preparation.descriptors,
        &publication::Selection {
            table: binding().table,
            incarnation: binding().incarnation,
            pending: (*pending).clone(),
        },
    )?;
    assert_eq!(retained.manifests.as_ref().unwrap().head, base.version_id());
    assert_eq!(
        pending.plan.sources[&10].records,
        vec![(first_result.index, first_result.meta.clone())]
    );
    assert!(pending.plan.sources[&10].through < score_result.index.min(status_result.index));

    proxy.release.notify_one();
    if confirm {
        let through = score_result.index.max(status_result.index);
        drained(&c, &service, through).await?;
        assert!(!task.is_finished(), "confirmed Prepare stopped the owner");
        assert_eq!(
            service.get("T", Duration::from_secs(30)).await?.unwrap(),
            current
        );
        let selected = annotation_publication::read(&c, &storage, Some(target)).await?;
        assert_eq!(
            selected.read_transaction().await?,
            Some(expected_transaction),
            "confirmation must reuse the original prepared files and transaction"
        );
        assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
        let preparations = proxy.prepare_requests.lock().unwrap().clone();
        let original: Vec<_> = preparations
            .iter()
            .filter(|p| p.transaction == preparation.transaction)
            .collect();
        assert_eq!(original.len(), if recorded { 1 } else { 2 });
        assert!(original.iter().all(|p| **p == preparation));
        assert_eq!(proxy.blocked_reads.load(Ordering::SeqCst), 0);
        stop.send(()).unwrap();
        task.await??;
        c.client = direct;
        drop(proxy);
        return c.shutdown().await;
    }
    let prepared = prepared.unwrap();
    // The proxy blocks all voter observations for the unchanged 30-second RPC deadline.
    let error = tokio::time::timeout(Duration::from_secs(45), task)
        .await??
        .unwrap_err();
    proxy.restore_observations();
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    assert!(proxy.blocked_reads.load(Ordering::SeqCst) >= 3);
    let context = format!("{error:#}");
    assert!(
        context.contains("WAL RPC prepare") && context.contains("outcome unknown"),
        "{context}"
    );
    assert!(service.get("T", Duration::from_secs(1)).await.is_err());
    let stopped = annotation_publication::state(&c)
        .await
        .context("inspect the retained preparation after losing its reply")?;
    assert_eq!(stopped.progress.prepared.as_ref(), Some(&prepared));
    assert_eq!(stopped.manifests.as_ref().unwrap().head, base.version_id());
    for result in [&first_result, &score_result, &status_result] {
        assert!(
            matches!(
                direct
                    .request_routed(Request::Fetch {
                        group: 10,
                        index: result.index
                    })
                    .await?
                    .reply,
                Reply::Record(Some(_))
            ),
            "an unpublished source must survive the unknown Prepare outcome"
        );
    }
    drop(service);
    c.client = direct;
    drop(proxy);
    c.shutdown()
        .await
        .context("close the original WAL replicas")?;

    // Reopen the original replica disks. A new fenced owner must publish the
    // retained descriptors, replay the suffix, and preserve exact retry results.
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard())
        .await
        .context("reopen the original WAL replicas")?;
    assert_eq!(
        annotation_publication::state(&c)
            .await
            .context("read the reopened preparation")?
            .progress
            .prepared,
        Some(prepared)
    );
    let (service, runtime) = owner(&c, &storage, &limits)
        .await
        .context("recover the owner from its unknown preparation")?;
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service
            .get("T", Duration::from_secs(30))
            .await
            .context("read the acknowledged suffix after recovery")?
            .unwrap(),
        current,
        "the first read after recovery must include the acknowledged suffix"
    );
    let through = score_result.index.max(status_result.index);
    drained(&c, &service, through)
        .await
        .context("drain the acknowledged suffix after recovery")?;
    let selected = annotation_publication::read(&c, &storage, Some(target))
        .await
        .context("reopen the original prepared target")?;
    assert_eq!(
        selected.read_transaction().await?,
        Some(expected_transaction)
    );
    for (request, result) in [
        (&first, &first_result),
        (&score, &score_result),
        (&status, &status_result),
    ] {
        assert_eq!(
            &service
                .submit(request, Duration::from_secs(30))
                .await
                .context("resolve an original request after recovery")?,
            result
        );
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: result.index
                })
                .await?
                .reply,
            Reply::Record(None)
        ));
    }
    assert_eq!(
        service.get("T", Duration::from_secs(30)).await?.unwrap(),
        current
    );
    let untouched = service.get("U", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(untouched.status.as_deref(), Some("pending"));
    assert_eq!(untouched.score, None);

    let next = request(
        &service,
        &d,
        2,
        wire(&limits, &["T"], 1, Some("done"), 0, None)?,
    )
    .await?;
    let next_result = service.submit(&next, Duration::from_secs(30)).await?;
    drained(&c, &service, next_result.index).await?;
    assert_eq!(
        service.submit(&first, Duration::from_secs(30)).await?,
        first_result
    );
    let latest = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(latest.status.as_deref(), Some("done"));
    assert_eq!(latest.score, Some(0.9));
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
