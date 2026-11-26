use super::super::payload_memory::{self as admission, Reservation};
use super::*;
use std::sync::atomic::AtomicUsize;

#[derive(Default)]
struct Counts {
    limit: AtomicUsize,
    held: AtomicUsize,
    backed: AtomicUsize,
    calls: AtomicUsize,
}
struct TestPool(Arc<Counts>);
struct Charge {
    counts: Arc<Counts>,
    bytes: usize,
    backed: usize,
}
impl Reservation for Charge {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn set_backed_bytes(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        self.counts.backed.fetch_sub(self.backed, Ordering::SeqCst);
        self.counts.backed.fetch_add(bytes, Ordering::SeqCst);
        self.backed = bytes;
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.counts.held.fetch_sub(self.bytes, Ordering::SeqCst);
        self.counts.backed.fetch_sub(self.backed, Ordering::SeqCst);
    }
}
impl admission::Pool for TestPool {
    fn try_reserve(&self, bytes: usize) -> Result<Option<Box<dyn Reservation>>> {
        self.0.calls.fetch_add(1, Ordering::SeqCst);
        let accepted = self
            .0
            .held
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |held| {
                held.checked_add(bytes)
                    .filter(|n| *n <= self.0.limit.load(Ordering::SeqCst))
            });
        Ok(accepted.ok().map(|_| {
            Box::new(Charge {
                counts: self.0.clone(),
                bytes,
                backed: 0,
            }) as Box<dyn Reservation>
        }))
    }
}

#[tokio::test]
async fn payload_memory_refusal_reads_nothing_and_charges_follow_shared_backing() -> Result<()> {
    let counts = Arc::new(Counts::default());
    let pool = TestPool(counts.clone());
    let source = b"payload-followed-by-another-frame";
    let mut input = source.as_slice();
    let error = admission::receive(&mut input, 7, Some(&pool))
        .await
        .unwrap_err();
    assert!(error.is::<admission::Unavailable>());
    assert_eq!(input, source);
    counts.limit.store(7, Ordering::SeqCst);
    let body = admission::receive(&mut input, 7, Some(&pool)).await?;
    assert_eq!(&body[..], b"payload");
    assert_eq!(input, &source[7..]);
    assert_eq!(counts.held.load(Ordering::SeqCst), 7);
    assert_eq!(counts.backed.load(Ordering::SeqCst), 7);
    let clone = body.slice(1..5);
    assert_eq!(clone.as_ptr(), body.as_ptr().wrapping_add(1));
    drop(body);
    assert_eq!(counts.held.load(Ordering::SeqCst), 7);
    #[cfg(feature = "ipc")]
    let clone = arrow_buffer::Buffer::from(clone);
    assert_eq!(&clone[..], b"aylo");
    drop(clone);
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    assert_eq!(counts.backed.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn payload_memory_releases_failed_and_cancelled_receive_allocations() -> Result<()> {
    let counts = Arc::new(Counts::default());
    counts.limit.store(4096, Ordering::SeqCst);
    let pool = TestPool(counts.clone());
    assert!(
        admission::receive(&mut b"short".as_slice(), 4096, Some(&pool))
            .await
            .is_err()
    );
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    assert_eq!(counts.backed.load(Ordering::SeqCst), 0);
    let (_writer, mut reader) = tokio::io::duplex(16);
    let mut pending = Box::pin(admission::receive(&mut reader, 4096, Some(&pool)));
    tokio::select! {
        biased;
        _ = &mut pending => panic!("writer is still live without sending a payload"),
        _ = tokio::task::yield_now() => {},
    }
    assert_eq!(counts.held.load(Ordering::SeqCst), 4096);
    assert_eq!(counts.backed.load(Ordering::SeqCst), 4096);
    drop(pending);
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    assert_eq!(counts.backed.load(Ordering::SeqCst), 0);
    let calls = counts.calls.load(Ordering::SeqCst);
    assert!(admission::receive(&mut reader, 0, Some(&pool))
        .await?
        .is_empty());
    assert_eq!(counts.calls.load(Ordering::SeqCst), calls);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payload_memory_rf3_refusal_preserves_control_and_resumes_exact_fetch() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = RpcCluster::open(root.path(), Options::default()).await?;
    let counts = Arc::new(Counts::default());
    c.client
        .install_payload_memory(Arc::new(TestPool(counts.clone())))?;
    let leader = c.leader().await?;
    let batch = batch(1);
    let index = accepted(
        &c.client
            .request(
                leader,
                Request::Append {
                    group: 10,
                    batch: batch.clone(),
                },
            )
            .await?,
    );
    let response = c
        .client
        .request_routed(Request::Fetch { group: 10, index })
        .await;
    let error = response
        .err()
        .context("fetch must refuse without payload memory")?;
    assert!(error.is::<admission::Unavailable>(), "{error:#}");
    assert_eq!(producer(&c.client, leader, &batch).await?, Some(index));
    counts.limit.store(batch.body.len(), Ordering::SeqCst);
    let response = c
        .client
        .request_routed(Request::Fetch { group: 10, index })
        .await?;
    assert_eq!(response.body, batch.body);
    let body = response.body.clone();
    drop(response);
    assert_eq!(counts.held.load(Ordering::SeqCst), batch.body.len());
    drop(body);
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    assert_eq!(counts.backed.load(Ordering::SeqCst), 0);
    c.shutdown().await
}
