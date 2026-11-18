//! Local shared-limiter regression: no network or production storage.
use super::*;
use lance_core::utils::aimd::AimdConfig;
use lance_io::object_store::throttle::{AimdThrottleConfig, AimdThrottledStore};
use object_store::ObjectStore as _;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct CountedBackend {
    inner: object_store::memory::InMemory,
    puts: AtomicUsize,
    registrations: AtomicUsize,
}

impl std::fmt::Display for CountedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CountedBackend")
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for CountedBackend {
    async fn put_opts(
        &self,
        path: &Path,
        payload: object_store::PutPayload,
        options: PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.puts.fetch_add(1, Ordering::SeqCst);
        if path.as_ref().contains("_wal_readers_v1/") {
            self.registrations.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.put_opts(path, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }

    async fn get_opts(
        &self,
        path: &Path,
        options: GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(path, options).await
    }

    fn delete_stream(
        &self,
        paths: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(paths)
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

fn bounded_policy() -> Policy {
    Policy {
        heartbeat: Duration::from_secs(5),
        operation_timeout: Duration::from_millis(100),
        max_silence: Duration::from_secs(15),
        stale_after: Duration::from_secs(45),
        poll: Duration::from_millis(10),
        initial_registration_retry: false,
    }
}

async fn shared_write_case(cancelled: usize) {
    let backend = Arc::new(CountedBackend {
        inner: object_store::memory::InMemory::new(),
        puts: AtomicUsize::new(0),
        registrations: AtomicUsize::new(0),
    });
    let config = AimdThrottleConfig::default()
        .with_write_aimd(
            AimdConfig::default()
                .with_initial_rate(10.0)
                .with_min_rate(10.0)
                .with_max_rate(10.0),
        )
        .with_burst_capacity(1);
    let limiter = Arc::new(AimdThrottledStore::new(backend.clone(), config).unwrap());
    let dir = tempfile::TempDir::new().unwrap();
    let (base, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
        .await
        .unwrap();
    let mut store = base.as_ref().clone();
    store.inner = limiter.clone();
    let store = Arc::new(store);
    let callbacks = Arc::new(AtomicUsize::new(0));

    // Both bulk writes and nonce registrations use this exact limiter instance.
    store
        .inner
        .put(&Path::from("bulk0"), "bulk".into())
        .await
        .unwrap();
    let bulk_path = Path::from("bulk1");
    let mut bulk = Box::pin(store.inner.put(&bulk_path, "bulk".into()));
    assert!(futures::poll!(bulk.as_mut()).is_pending());
    let mut attempts = Vec::new();
    for _ in 0..cancelled {
        let count = callbacks.clone();
        let mut future = Box::pin(protect(
            store.clone(),
            &root,
            bounded_policy(),
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(7)
            },
        ));
        assert!(futures::poll!(future.as_mut()).is_pending());
        attempts.push(future);
    }
    for result in futures::future::join_all(attempts).await {
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }
    bulk.await.unwrap();
    assert_eq!(callbacks.load(Ordering::SeqCst), 0);
    assert_eq!(backend.registrations.load(Ordering::SeqCst), 0);
    assert_eq!(backend.puts.load(Ordering::SeqCst), 2);

    // Legitimate bulk work has completed. A cancelled request must not keep
    // charging the shared write budget after its future has been dropped.
    tokio::time::sleep(Duration::from_millis(60)).await;
    let started = Instant::now();
    let count = callbacks.clone();
    let subsequent = protect(store.clone(), &root, bounded_policy(), async move {
        count.fetch_add(1, Ordering::SeqCst);
        Ok(11)
    })
    .await;
    let elapsed = started.elapsed();
    eprintln!(
        "admission cancelled={cancelled} subsequent_ms={} result={subsequent:?} backend_nonce_puts={} callbacks={}",
        elapsed.as_millis(), backend.registrations.load(Ordering::SeqCst), callbacks.load(Ordering::SeqCst)
    );
    if subsequent.is_err() {
        assert_eq!(backend.registrations.load(Ordering::SeqCst), 0);
        assert_eq!(callbacks.load(Ordering::SeqCst), 0);
        // Establish that the backend remains healthy and debt eventually drains.
        tokio::time::sleep(Duration::from_millis(1100)).await;
        assert_eq!(
            protect(store.clone(), &root, bounded_policy(), async { Ok(13) })
                .await
                .unwrap(),
            13
        );
    }
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let rows = backend
                .inner
                .list(Some(&readers_root(&root)))
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            if rows.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        subsequent.unwrap(),
        11,
        "cancelled registrations retained write admission debt"
    );
}

#[tokio::test]
async fn shared_write_admission_control_without_cancelled_registrations() {
    tokio::time::timeout(Duration::from_secs(5), shared_write_case(0))
        .await
        .unwrap();
}

#[tokio::test]
async fn shared_write_admission_cancelled_registrations_release_capacity() {
    tokio::time::timeout(Duration::from_secs(5), shared_write_case(8))
        .await
        .unwrap();
}
