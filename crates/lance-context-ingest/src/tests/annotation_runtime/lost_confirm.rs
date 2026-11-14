//! Confirm's uncertain reply must not erase the ACKed active suffix or guess success.
use super::*;
use std::sync::atomic::Ordering;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_unknown_confirm_keeps_owner_live_and_preserves_concurrent_updates() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(60), exercise(true)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn uncommitted_unknown_confirm_retains_recovery_evidence_and_acked_suffix() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(false)).await?
}

async fn exercise(execute: bool) -> Result<()> {
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
    let base = annotation_publication::read(&c, &storage, None).await?;
    let proxy = lost_prepare::proxy::Proxy::open_confirm(&c, root.path(), execute).await?;
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

    // The real manifest is selected, but the owner has no Confirm reply yet.
    // Apply continues: both different-field and same-field updates must survive.
    let retained = annotation_publication::state(&c).await?;
    let pending = retained.progress.pending.clone().unwrap();
    let prepared = retained.progress.prepared.clone().unwrap();
    assert_eq!(retained.progress.published.is_some(), execute);
    let target = pending.plan.target_version()?;
    assert_eq!(retained.manifests.as_ref().unwrap().head, target);
    let transaction = publication::transaction(
        &base,
        &prepared.descriptors,
        &publication::Selection {
            table: binding().table,
            incarnation: binding().incarnation,
            pending: (*pending).clone(),
        },
    )?;
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
    proxy.release.notify_one();
    let through = score.index.max(status.index);
    if execute {
        tokio::select! {
            result = drained(&c, &service, through) => result?,
            stopped = &mut task => bail!("owner exited before draining confirmed publication: {stopped:?}"),
        }
        assert_eq!(
            service.get("T", Duration::from_secs(10)).await?.unwrap(),
            current
        );
        assert_eq!(
            annotation_publication::read(&c, &storage, Some(target))
                .await?
                .read_transaction()
                .await?,
            Some(transaction),
            "reconciliation must keep the original transaction and prepared files"
        );
        assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(
            proxy.confirms.load(Ordering::SeqCst),
            2,
            "one original Confirm and one suffix Confirm; no resubmission"
        );
        stop.send(()).unwrap();
        task.await??;
        c.client = direct;
        drop(proxy);
        return c.shutdown().await;
    }

    let error = tokio::time::timeout(Duration::from_secs(10), task)
        .await??
        .unwrap_err();
    assert!(format!("{error:#}").contains("outcome unknown"));
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(proxy.confirms.load(Ordering::SeqCst), 1);
    let stopped = annotation_publication::state(&c).await?;
    assert_eq!(stopped.progress.pending, Some(pending));
    assert_eq!(stopped.progress.prepared, Some(prepared));
    assert!(stopped.progress.published.is_none());
    assert_eq!(stopped.progress.checkpoint, retained.progress.checkpoint);
    for ack in [&first, &score, &status] {
        assert!(matches!(
            direct
                .request_routed(Request::Fetch {
                    group: 10,
                    index: ack.index,
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

    // Recover the original disks and selected files under a new fenced owner.
    // No original business mutation is submitted again to make the test pass.
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
    assert_eq!(
        annotation_publication::read(&c, &storage, Some(target))
            .await?
            .read_transaction()
            .await?,
        Some(transaction)
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
