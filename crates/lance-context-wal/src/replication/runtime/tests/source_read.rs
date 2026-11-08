use super::*;
use crate::replication::source_read;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn committed_source_retains_completion_credit_and_releases_cancelled_work() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = Cluster::open(
        directory.path(),
        Options {
            control_requests: 1,
            ..Options::default()
        },
    )?;
    c.elect(1, 10).await?;
    let handle = &c.handles[&1];
    let original = batch(1);
    let accepted = commit(handle, 10, original.clone()).await?;
    let request = source_read::Request {
        table: original.meta.stream,
        bucket: original.meta.bucket,
        after: 0,
        limit: 1,
    };
    let pending = handle.read_source(10, request.clone())?;
    wait_for(|| !pending.0.is_empty()).await?;
    assert_eq!(handle.admission().control.used_requests, 1);
    assert!(handle.read_source(10, request.clone()).is_err());
    let observed = pending.await?;
    let source_read::Value::Page { page, .. } = observed.value else {
        anyhow::bail!("expected a committed source page");
    };
    assert_eq!(page.entries, vec![(accepted, original.meta)]);
    assert_eq!(handle.admission().control.used_bytes, 0);

    // Cancel a queued request while the real worker is held. Its admission
    // remains charged until the worker releases the request and response.
    let (entered, arrived) = oneshot::channel();
    let (release, resumed) = mpsc::channel::<()>();
    handle.enqueue(
        handle.route(10)?,
        Work::Local(Box::new(move |_| {
            let _ = entered.send(());
            let _ = resumed.recv();
            Box::new(|| {})
        })),
        false,
    )?;
    timeout(Duration::from_secs(5), arrived).await??;
    drop(handle.read_source(10, request.clone())?);
    assert_eq!(handle.admission().control.used_requests, 1);
    assert!(handle.read_source(10, request.clone()).is_err());
    drop(release);
    wait_for(|| handle.admission().control.used_requests == 0).await?;
    assert_eq!(handle.admission().control.used_bytes, 0);
    assert!(matches!(
        handle.read_source(10, request)?.await?.value,
        source_read::Value::Page { .. }
    ));
    c.shutdown().await
}
