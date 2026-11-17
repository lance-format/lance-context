//! Uncertain catalog fences must withhold deletion without killing the owner.
use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use std::sync::atomic::Ordering;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_retirement_unknown_bounds_gc_and_preserves_live_updates() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), uncertain_fence(true, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn uncommitted_retirement_unknown_bounds_gc_and_preserves_live_updates() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), uncertain_fence(false, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_retirement_rpc_unknown_bounds_gc_and_preserves_live_updates() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), uncertain_fence(true, true)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn uncommitted_retirement_rpc_unknown_bounds_gc_and_preserves_live_updates() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), uncertain_fence(false, true)).await?
}

async fn uncertain_fence(execute: bool, rpc_unknown: bool) -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage(&path, faults.clone()).await?;
    let (first, completed, abandoned) = acknowledged_orphan_count(&c, &storage, &path, 1).await?;
    let proxy = Proxy::open_retire(&c, root.path(), execute, rpc_unknown).await?;
    c.client = proxy.client.clone();
    let (service, runtime) = owned(&c, &storage, &limits()).await?;
    let (_, _, recovery_cut) = runtime.http_binding();
    let (stop, task) = run_owner(whole_page(runtime)?);
    tokio::time::timeout(Duration::from_secs(10), proxy.committed.notified()).await?;
    assert_eq!(
        service.submit(&first, Duration::from_secs(10)).await?,
        completed
    );
    assert!(faults.calls.lock().unwrap().is_empty());
    let producer = producer(&service, 2).await?;
    let suffix = request(
        &service,
        &producer,
        1,
        wire(&limits(), &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let suffix_result = service.submit(&suffix, Duration::from_secs(10)).await?;
    let waiting = request(
        &service,
        &producer,
        2,
        wire(&limits(), &["T"], 0, None, 1, Some(0.8))?,
    )
    .await?;
    let accepted = service.ingress().append(&waiting).await?;
    let before_files = files(&path)?;
    let before_metadata = metadata_files(&path)?;
    // Hold subsequent fences uncertain too: one successful later attempt must
    // not hide an unbounded publication loop or premature immutable DELETE.
    proxy.unknown_retirements.store(true, Ordering::SeqCst);
    proxy.release.notify_one();
    tokio::time::timeout(Duration::from_secs(10), async {
        while proxy.dropped.load(Ordering::SeqCst) < 3 {
            assert!(!task.is_finished(), "uncertain GC fence killed the owner");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert!(!task.is_finished());
    assert!(faults.calls.lock().unwrap().is_empty());
    assert!(abandoned.iter().all(|p| p.exists()));
    assert_eq!(files(&path)?, before_files);
    assert_eq!(metadata_files(&path)?, before_metadata);
    // RPC fault counts do not synchronize with the independent apply loop.
    // Wait for that loop to observe the blocked write before checking its cut.
    let observation = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            assert!(!task.is_finished(), "uncertain GC fence killed the owner");
            let (observation, pressured) = service.observation().await;
            if pressured {
                return observation;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert_eq!(observation.applied, suffix_result.index);
    assert!(observation.applied < accepted.position.index);
    assert_eq!(service.source_observation().records, 2);
    // The deployed owner read uses its recovery barrier, not the standalone
    // helper's fresh WAL tail (which includes the deliberately unapplied write).
    let current = service
        .get_owned("T", recovery_cut, Duration::from_secs(10))
        .await?
        .unwrap();
    assert_eq!(current.status.as_deref(), Some("acknowledged"));
    assert_eq!(current.score, Some(0.9));

    proxy.unknown_retirements.store(false, Ordering::SeqCst);
    drained(&c, &service, accepted.position.index).await?;
    wait_files(&path, 3).await?;
    assert!(abandoned.iter().all(|p| !p.exists()));
    cold_check(&c, &storage, Some(0.8)).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
