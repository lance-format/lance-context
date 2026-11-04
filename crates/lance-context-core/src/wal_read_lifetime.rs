//! Cross-process protection for captured WAL files. A reader registers before
//! capturing any WAL metadata; reclamation snapshots registrations after drain.
//! Consequently a later reader cannot reference a retired generation and need
//! not delay its collection. This requires strongly consistent storage listing.
//!
//! Lease expiration uses observed nonce stability and monotonic clocks, never
//! wall-clock timestamps. A reader cannot return data after renewal is overdue.
//! Durable retirement intents live in `wal_retirement`. Bounded recovery
//! scheduling and complete read coverage still precede production activation.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use futures::{stream, stream::BoxStream, StreamExt, TryStreamExt};
use lance_io::object_store::ObjectStore;
use object_store::{path::Path, GetOptions, GetRange, ObjectStoreExt, PutMode, PutOptions};

const READER_BATCH: usize = 128;
const CLEANUP_CONCURRENCY: usize = 32;
const NONCE_BYTES: u64 = 32;

#[cfg(test)]
#[derive(Default)]
pub(crate) struct ReadCapturePause {
    pub(crate) captured: tokio::sync::Notify,
    pub(crate) resume: tokio::sync::Notify,
}

/// Exercise a real API future against its already-captured WAL view while a
/// different handle merges it. Shared by the read API coverage regressions.
#[cfg(test)]
pub(crate) async fn assert_read_survives_merge<T: std::fmt::Debug>(
    reader: &crate::store_base::StorageBase,
    writer: &mut crate::store_base::StorageBase,
    read: impl Future<Output = lance::Result<T>>,
) -> T {
    let store = reader.dataset.object_store(None).await.unwrap();
    let registrations = readers_root(&reader.dataset.branch_location().path);
    // Setup writes can themselves perform protected uniqueness reads. Join
    // their asynchronous cleanup before attributing a key to this API call.
    tokio::time::timeout(Duration::from_secs(5), async {
        while store
            .inner
            .list(Some(&registrations))
            .try_next()
            .await
            .unwrap()
            .is_some()
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let snapshots = writer.wal_shard_snapshots().await.unwrap();
    let paths: Vec<_> = snapshots
        .iter()
        .flat_map(|snapshot| {
            snapshot.sstables.iter().map(|generation| {
                writer.flushed_generation_uri(snapshot.shard_id, &generation.path)
            })
        })
        .collect();
    assert!(!paths.is_empty());
    let pause = Arc::new(ReadCapturePause::default());
    *reader.wal_snapshot_pause.lock().unwrap() = Some(pause.clone());
    writer.wal_collection = crate::wal_gc::Mode::Collect;
    let merging = async {
        pause.captured.notified().await;
        assert!(
            store
                .inner
                .list(Some(&registrations))
                .try_next()
                .await
                .unwrap()
                .is_some(),
            "the API captured WAL without a reader registration"
        );
        assert!(writer.cleanup_own_shard().await.unwrap() > 0);
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
        // Give the collector an opportunity to attempt deletion while the
        // actual read is still paused with its old immutable generation view.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(paths.iter().all(|path| std::path::Path::new(path).exists()));
        pause.resume.notify_one();
    };
    let (result, ()) = tokio::time::timeout(Duration::from_secs(30), async {
        tokio::join!(read, merging)
    })
    .await
    .unwrap();
    let result = result.unwrap();
    tokio::time::timeout(Duration::from_secs(30), async {
        while paths.iter().any(|path| std::path::Path::new(path).exists()) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    result
}

/// Verify a long-lived API handle does not lose a row once another executor
/// has merged and physically reclaimed the WAL that originally contained it.
#[cfg(test)]
pub(crate) async fn assert_read_after_merge<T: std::fmt::Debug>(
    reader: &crate::store_base::StorageBase,
    writer: &mut crate::store_base::StorageBase,
    read: impl Future<Output = lance::Result<T>>,
) -> T {
    let original_version = reader.dataset.manifest.version;
    let snapshots = writer.wal_shard_snapshots().await.unwrap();
    let paths: Vec<_> = snapshots
        .iter()
        .flat_map(|snapshot| {
            snapshot.sstables.iter().map(|generation| {
                writer.flushed_generation_uri(snapshot.shard_id, &generation.path)
            })
        })
        .collect();
    assert!(!paths.is_empty());
    writer.wal_collection = crate::wal_gc::Mode::Collect;
    assert!(writer.cleanup_own_shard().await.unwrap() > 0);
    assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
    assert!(writer.dataset.manifest.version > original_version);
    tokio::time::timeout(Duration::from_secs(30), async {
        while paths.iter().any(|path| std::path::Path::new(path).exists()) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let result = read.await.unwrap();
    assert_eq!(
        reader.dataset.manifest.version, original_version,
        "latest-view reads must not mutate the resident handle's base version"
    );
    result
}

#[derive(Clone, Copy)]
pub(crate) struct Policy {
    heartbeat: Duration,
    operation_timeout: Duration,
    max_silence: Duration,
    stale_after: Duration,
    poll: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            heartbeat: Duration::from_secs(5),
            operation_timeout: Duration::from_secs(5),
            max_silence: Duration::from_secs(15),
            stale_after: Duration::from_secs(45),
            poll: Duration::from_secs(2),
        }
    }
}

fn readers_root(root: &Path) -> Path {
    root.clone().join("_wal_readers_v1")
}

async fn bounded<T>(
    timeout: Duration,
    operation: impl Future<Output = lance::Result<T>>,
) -> lance::Result<T> {
    tokio::time::timeout(timeout, operation)
        .await
        .map_err(|_| lance::Error::io("WAL read protection storage operation timed out"))?
}

// Keep reader registration failures distinguishable from data reads and from
// the collector's storage operations. Never format the backing error here: it
// may contain a signed URL. Return it unchanged to the caller instead.
fn reader_failure(phase: &'static str, category: &'static str, started: Instant, last: Instant) {
    tracing::warn!(
        phase,
        category,
        operation_ms = started.elapsed().as_millis() as u64,
        acknowledged_silence_ms = last.elapsed().as_millis() as u64,
        "WAL reader protection failed"
    );
}

async fn reader_publication(
    timeout: Duration,
    phase: &'static str,
    last: Instant,
    operation: impl Future<Output = lance::Result<()>>,
) -> lance::Result<()> {
    let started = Instant::now();
    match tokio::time::timeout(timeout, operation).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => {
            reader_failure(phase, "storage_error", started, last);
            Err(error)
        }
        Err(_) => {
            reader_failure(phase, "operation_timeout", started, last);
            Err(lance::Error::io(
                "WAL read protection storage operation timed out",
            ))
        }
    }
}

async fn write_nonce(store: &ObjectStore, path: &Path, create: bool) -> lance::Result<()> {
    let nonce = uuid::Uuid::new_v4().simple().to_string();
    store
        .inner
        .put_opts(
            path,
            nonce.into(),
            PutOptions {
                mode: if create {
                    PutMode::Create
                } else {
                    PutMode::Overwrite
                },
                ..Default::default()
            },
        )
        .await?;
    Ok(())
}

struct Registration {
    store: Arc<ObjectStore>,
    path: Path,
    timeout: Duration,
}

fn cleanup_registration(
    store: Arc<ObjectStore>,
    path: Path,
    timeout: Duration,
    slots: Arc<tokio::sync::Semaphore>,
) -> bool {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return false;
    };
    let Ok(permit) = slots.try_acquire_owned() else {
        // Never queue an unbounded task on the read's cancellation path.
        // A collector can expire this abandoned registration instead.
        return false;
    };
    runtime.spawn(async move {
        let _permit = permit;
        let _ = tokio::time::timeout(timeout, store.inner.delete(&path)).await;
    });
    true
}

impl Drop for Registration {
    fn drop(&mut self) {
        static SLOTS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
        // Cancellation must also release the registration. If the runtime or
        // cleanup capacity is unavailable, collectors age the nonce out. They
        // must support arbitrarily many abandoned keys with bounded memory.
        cleanup_registration(
            self.store.clone(),
            self.path.clone(),
            self.timeout,
            SLOTS
                .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(CLEANUP_CONCURRENCY)))
                .clone(),
        );
    }
}

pub(crate) async fn protect<T>(
    store: Arc<ObjectStore>,
    root: &Path,
    policy: Policy,
    read: impl Future<Output = lance::Result<T>>,
) -> lance::Result<T> {
    assert!(policy.stale_after > policy.max_silence + policy.operation_timeout);
    let path = readers_root(root).join(uuid::Uuid::new_v4().simple().to_string());
    // Install cleanup before PUT: a timeout is an ambiguous publication, not
    // proof that the registration was never stored.
    let registration = Registration {
        store,
        path,
        timeout: policy.operation_timeout,
    };
    let publication_started = Instant::now();
    reader_publication(
        policy.operation_timeout,
        "registration",
        publication_started,
        write_nonce(&registration.store, &registration.path, true),
    )
    .await?;
    // Timeout polls the operation first: a runtime paused after remote PUT
    // acceptance can receive a ready success after the timeout has elapsed.
    // The registration may already have expired elsewhere. Never restart its
    // lease clock or begin capturing WAL from that late acknowledgement.
    if publication_started.elapsed() >= policy.operation_timeout {
        reader_failure(
            "registration",
            "acknowledgement_expired",
            publication_started,
            publication_started,
        );
        return Err(lance::Error::io(
            "WAL reader registration acknowledgement expired",
        ));
    }
    let acknowledged = Mutex::new(publication_started);
    let renewal = async {
        loop {
            tokio::time::sleep(policy.heartbeat).await;
            let last = *acknowledged.lock().unwrap();
            let started = Instant::now();
            if last.elapsed() >= policy.max_silence {
                reader_failure("renewal", "silence_expired", started, last);
                return Err(lance::Error::io("WAL reader renewal expired"));
            }
            let renewed = reader_publication(
                policy.operation_timeout,
                "renewal",
                last,
                write_nonce(&registration.store, &registration.path, false),
            )
            .await;
            if let Err(error) = renewed {
                // One failed PUT need not invalidate an acknowledged reader
                // that is still within its original monotonic deadline. Retry
                // once, without refreshing that deadline or extending GC's
                // stale-reader window. A late ambiguous PUT can only retain
                // this unique registration longer; it cannot authorize a read.
                let backoff = policy.heartbeat.min(Duration::from_millis(250));
                if last.elapsed().saturating_add(backoff) >= policy.max_silence {
                    return Err(error);
                }
                tokio::time::sleep(backoff).await;
                let remaining = policy.max_silence.saturating_sub(last.elapsed());
                if remaining.is_zero() {
                    return Err(error);
                }
                reader_publication(
                    policy.operation_timeout.min(remaining),
                    "renewal_retry",
                    last,
                    write_nonce(&registration.store, &registration.path, false),
                )
                .await?;
            }
            let mut last = acknowledged.lock().unwrap();
            // A paused process must not resurrect an expired read by accepting
            // a delayed successful renewal response.
            if last.elapsed() >= policy.max_silence {
                reader_failure("renewal", "acknowledgement_expired", started, *last);
                return Err(lance::Error::io("WAL reader renewal expired"));
            }
            *last = Instant::now();
        }
    };
    tokio::pin!(renewal);
    tokio::pin!(read);
    let result = tokio::select! {
        biased;
        expired = &mut renewal => expired,
        result = &mut read => result,
    };
    let last = *acknowledged.lock().unwrap();
    if last.elapsed() >= policy.max_silence {
        reader_failure("return", "silence_expired", publication_started, last);
        return Err(lance::Error::io(
            "WAL reader renewal expired before returning data",
        ));
    }
    result
}

#[cfg(test)]
async fn captured_readers(store: &ObjectStore, root: &Path) -> lance::Result<Vec<Path>> {
    let prefix = readers_root(root);
    let mut listing = store.inner.list(Some(&prefix));
    let mut readers = Vec::new();
    while let Some(entry) = listing.next().await {
        let entry = entry?;
        readers.push(entry.location);
    }
    Ok(readers)
}

async fn read_nonce(store: &ObjectStore, path: &Path) -> lance::Result<Option<String>> {
    let result = match store
        .inner
        .get_opts(
            path,
            GetOptions {
                range: Some(GetRange::Bounded(0..NONCE_BYTES + 1)),
                ..Default::default()
            },
        )
        .await
    {
        Ok(result) => result,
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let data = result.bytes().await?;
    if data.len() != NONCE_BYTES as usize || !data.iter().all(u8::is_ascii_hexdigit) {
        return Err(lance::Error::io(
            "Invalid WAL reader registration; retain files",
        ));
    }
    Ok(Some(String::from_utf8(data.to_vec()).unwrap()))
}

pub(crate) async fn wait_for_readers(
    store: Arc<ObjectStore>,
    mut readers: Vec<Path>,
    policy: Policy,
) -> lance::Result<()> {
    let mut unchanged: HashMap<Path, (String, Instant)> = HashMap::new();
    while !readers.is_empty() {
        let observations = stream::iter(readers.iter().cloned())
            .map(|path| {
                let store = store.clone();
                async move {
                    let value =
                        bounded(policy.operation_timeout, read_nonce(&store, &path)).await?;
                    Ok::<_, lance::Error>((path, value))
                }
            })
            .buffer_unordered(4)
            .try_collect::<Vec<_>>()
            .await?;
        readers.clear();
        for (path, nonce) in observations {
            let Some(nonce) = nonce else {
                unchanged.remove(&path);
                continue;
            };
            let entry = unchanged
                .entry(path.clone())
                .or_insert_with(|| (nonce.clone(), Instant::now()));
            if entry.0 != nonce {
                *entry = (nonce, Instant::now());
            }
            if entry.1.elapsed() < policy.stale_after {
                readers.push(path);
            } else {
                // Every registration is unique to one read. A valid read has
                // already failed its much shorter monotonic renewal deadline.
                // An in-flight late PUT may recreate this stale key, but cannot
                // authorize another read or make the expired read successful.
                let _ = bounded(policy.operation_timeout, async {
                    store.inner.delete(&path).await?;
                    Ok(())
                })
                .await;
            }
        }
        if !readers.is_empty() {
            tokio::time::sleep(policy.poll).await;
        }
    }
    Ok(())
}

/// A fixed total-key cap could prevent abandoned registrations from ever
/// reaching expiry processing. Keep the actual unordered listing instead and
/// process bounded batches. Previously visited expired keys may be removed;
/// registrations of still-valid readers must remain visible throughout this
/// strongly consistent listing. A failed/incomplete listing never authorizes
/// payload deletion. The retirement attempt also has an outer time bound.
async fn wait_for_listing(
    store: Arc<ObjectStore>,
    mut listing: BoxStream<'static, object_store::Result<object_store::ObjectMeta>>,
    policy: Policy,
    batch_size: usize,
) -> lance::Result<()> {
    assert!(batch_size > 0 && batch_size <= READER_BATCH);
    let mut readers = Vec::with_capacity(batch_size);
    loop {
        let entry = bounded(policy.operation_timeout, async {
            Ok(listing.try_next().await?)
        })
        .await?;
        let finished = entry.is_none();
        if let Some(entry) = entry {
            readers.push(entry.location);
        }
        if readers.len() == batch_size || finished {
            wait_for_readers(store.clone(), std::mem::take(&mut readers), policy).await?;
        }
        if finished {
            return Ok(());
        }
    }
}

/// Both self-merge and grouped append reclamation use this barrier. Manifest
/// publication/draining has already completed and is never delayed by readers.
pub(crate) async fn reclaim(
    store: Arc<ObjectStore>,
    root: Path,
    directories: Vec<Path>,
) -> lance::Result<()> {
    let policy = Policy::default();
    let listing = store.inner.list(Some(&readers_root(&root)));
    wait_for_listing(store.clone(), listing, policy, READER_BATCH).await?;
    stream::iter(directories)
        .map(Ok::<_, lance::Error>)
        .try_for_each_concurrent(16, |path| {
            let store = store.clone();
            async move {
                match store.remove_dir_all(path).await {
                    Err(error) if crate::store_base::is_not_found_error(&error) => Ok(()),
                    result => result,
                }
            }
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::throttle::{ThrottleConfig, ThrottledStore};
    use tracing::instrument::WithSubscriber;

    #[derive(Clone, Default)]
    struct Diagnostics(Arc<Mutex<Vec<HashMap<String, String>>>>);

    impl tracing::Subscriber for Diagnostics {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            struct Fields(HashMap<String, String>);
            impl tracing::field::Visit for Fields {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    self.0.insert(field.name().into(), format!("{value:?}"));
                }
                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    self.0.insert(field.name().into(), value.into());
                }
            }
            let mut fields = Fields(HashMap::new());
            event.record(&mut fields);
            if fields
                .0
                .get("message")
                .is_some_and(|v| v == "WAL reader protection failed")
            {
                self.0.lock().unwrap().push(fields.0);
            }
        }
    }

    impl Diagnostics {
        fn assert_failure(&self, phase: &str, category: &str) {
            let events = self.0.lock().unwrap();
            assert_eq!(events.len(), 1, "{events:?}");
            assert_eq!(events[0]["phase"], phase);
            assert_eq!(events[0]["category"], category);
            assert!(events[0]["operation_ms"].parse::<u64>().is_ok());
            assert!(events[0]["acknowledged_silence_ms"].parse::<u64>().is_ok());
        }
    }

    #[tokio::test]
    async fn publication_diagnostic_preserves_storage_error_without_logging_it() {
        for phase in ["registration", "renewal"] {
            let diagnostics = Diagnostics::default();
            let error = lance::Error::invalid_input("signed-url-secret");
            let original = error.to_string();
            let result = reader_publication(Duration::from_secs(1), phase, Instant::now(), async {
                Err(error)
            })
            .with_subscriber(diagnostics.clone())
            .await
            .unwrap_err();
            assert!(matches!(result, lance::Error::InvalidInput { .. }));
            assert_eq!(result.to_string(), original);
            diagnostics.assert_failure(phase, "storage_error");
            assert!(!format!("{:?}", diagnostics.0.lock().unwrap()).contains("signed-url-secret"));
        }
    }

    #[tokio::test]
    async fn successful_protected_read_emits_no_failure_diagnostic() {
        let diagnostics = Diagnostics::default();
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            protect(store, &root, quick_policy(), async { Ok(7) })
                .with_subscriber(diagnostics.clone())
                .await
                .unwrap(),
            7
        );
        assert!(diagnostics.0.lock().unwrap().is_empty());
    }

    #[derive(Debug)]
    struct PausedPublication {
        inner: Arc<dyn object_store::ObjectStore>,
        pause: Duration,
    }

    impl std::fmt::Display for PausedPublication {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "PausedPublication")
        }
    }

    #[async_trait::async_trait]
    impl object_store::ObjectStore for PausedPublication {
        async fn put_opts(
            &self,
            path: &Path,
            payload: object_store::PutPayload,
            options: PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            let result = self.inner.put_opts(path, payload, options).await?;
            // Model a runtime paused after remote publication, with the key
            // expired by another process before its successful reply is polled.
            std::thread::sleep(self.pause);
            self.inner.delete(path).await?;
            Ok(result)
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

    #[tokio::test]
    async fn a_late_initial_acknowledgement_cannot_start_an_unregistered_read() {
        let diagnostics = Diagnostics::default();
        use std::sync::atomic::{AtomicBool, Ordering};
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let policy = quick_policy();
        let mut paused = store.as_ref().clone();
        paused.inner = Arc::new(PausedPublication {
            inner: Arc::new(object_store::memory::InMemory::new()),
            pause: policy.stale_after + Duration::from_millis(50),
        });
        let store = Arc::new(paused);
        // Establish the timeout corner case through the real publication path:
        // its future returns ready after this runtime's timer expired.
        bounded(
            policy.operation_timeout,
            write_nonce(&store, &readers_root(&root).join("expired-control"), true),
        )
        .await
        .unwrap();
        assert!(captured_readers(&store, &root).await.unwrap().is_empty());
        let started = AtomicBool::new(false);
        let result = protect(store, &root, policy, async {
            started.store(true, Ordering::SeqCst);
            Ok(7)
        })
        .with_subscriber(diagnostics.clone())
        .await;
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("registration acknowledgement expired"));
        assert!(!started.load(Ordering::SeqCst));
        diagnostics.assert_failure("registration", "acknowledgement_expired");
    }

    fn quick_policy() -> Policy {
        Policy {
            heartbeat: Duration::from_millis(20),
            operation_timeout: Duration::from_millis(100),
            max_silence: Duration::from_millis(300),
            stale_after: Duration::from_millis(700),
            poll: Duration::from_millis(10),
        }
    }

    #[test]
    fn bounded_cleanup_survives_runtime_shutdown_and_overflow_expires() {
        let dir = tempfile::TempDir::new().unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let slots = Arc::new(tokio::sync::Semaphore::new(2));
        let (store, root) = runtime.block_on(async {
            let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
                .await
                .unwrap();
            let mut delayed = store.as_ref().clone();
            delayed.inner = Arc::new(ThrottledStore::new(
                store.inner.clone(),
                ThrottleConfig {
                    wait_delete_per_call: Duration::from_secs(30),
                    ..Default::default()
                },
            ));
            let delayed = Arc::new(delayed);
            for i in 0..3 {
                let path = readers_root(&root).join(i.to_string());
                write_nonce(&store, &path, true).await.unwrap();
                assert_eq!(
                    cleanup_registration(
                        delayed.clone(),
                        path,
                        Duration::from_secs(30),
                        slots.clone(),
                    ),
                    i < 2
                );
            }
            tokio::task::yield_now().await;
            assert_eq!(slots.available_permits(), 0);
            (store, root)
        });
        drop(runtime);
        assert_eq!(slots.available_permits(), 2);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            assert_eq!(captured_readers(&store, &root).await.unwrap().len(), 3);
            let listing = store.inner.list(Some(&readers_root(&root)));
            wait_for_listing(store.clone(), listing, quick_policy(), 2)
                .await
                .unwrap();
            assert!(captured_readers(&store, &root).await.unwrap().is_empty());
        });
    }

    #[tokio::test]
    async fn reader_batches_reap_more_than_the_old_total_registration_limit() {
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let mut memory = store.as_ref().clone();
        memory.inner = Arc::new(object_store::memory::InMemory::new());
        let store = Arc::new(memory);
        for i in 0..4097 {
            write_nonce(&store, &readers_root(&root).join(i.to_string()), true)
                .await
                .unwrap();
        }
        let policy = Policy {
            heartbeat: Duration::from_millis(2),
            operation_timeout: Duration::from_millis(20),
            max_silence: Duration::from_millis(30),
            stale_after: Duration::from_millis(60),
            poll: Duration::from_millis(5),
        };
        let listing = store.inner.list(Some(&readers_root(&root)));
        tokio::time::timeout(
            Duration::from_secs(10),
            wait_for_listing(store.clone(), listing, policy, READER_BATCH),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(captured_readers(&store, &root).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn unordered_reader_batches_keep_the_live_tail_protected() {
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let reached = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let reading = {
            let (store, root, reached, release) = (
                store.clone(),
                root.clone(),
                reached.clone(),
                release.clone(),
            );
            tokio::spawn(async move {
                protect(store, &root, quick_policy(), async {
                    reached.notify_one();
                    release.notified().await;
                    Ok(7)
                })
                .await
            })
        };
        reached.notified().await;
        let live = captured_readers(&store, &root).await.unwrap().remove(0);
        let mut entries = Vec::new();
        for name in ["z-abandoned", "a-abandoned"] {
            let path = readers_root(&root).join(name);
            write_nonce(&store, &path, true).await.unwrap();
            entries.push(Ok(store.inner.head(&path).await.unwrap()));
        }
        entries.push(Ok(store.inner.head(&live).await.unwrap()));
        let collecting = tokio::spawn(wait_for_listing(
            store.clone(),
            stream::iter(entries).boxed(),
            quick_policy(),
            2,
        ));
        tokio::time::sleep(Duration::from_millis(900)).await;
        assert!(!collecting.is_finished());
        assert!(read_nonce(&store, &live).await.unwrap().is_some());
        release.notify_one();
        assert_eq!(reading.await.unwrap().unwrap(), 7);
        tokio::time::timeout(Duration::from_secs(2), collecting)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn incomplete_reader_listing_is_not_an_empty_reader_set() {
        let dir = tempfile::TempDir::new().unwrap();
        let (store, _) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let listing = stream::once(async {
            Err(object_store::Error::Generic {
                store: "reader listing fault",
                source: std::io::Error::other("incomplete listing").into(),
            })
        })
        .boxed();
        assert!(wait_for_listing(store, listing, quick_policy(), 2)
            .await
            .unwrap_err()
            .to_string()
            .contains("incomplete listing"));
    }

    #[tokio::test]
    async fn registration_put_timeout_never_starts_the_read() {
        let diagnostics = Diagnostics::default();
        use std::sync::atomic::{AtomicBool, Ordering};
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let mut delayed = store.as_ref().clone();
        delayed.inner = Arc::new(ThrottledStore::new(
            store.inner.clone(),
            ThrottleConfig {
                wait_put_per_call: Duration::from_secs(1),
                ..Default::default()
            },
        ));
        let started = AtomicBool::new(false);
        let result = protect(Arc::new(delayed), &root, quick_policy(), async {
            started.store(true, Ordering::SeqCst);
            Ok(7)
        })
        .with_subscriber(diagnostics.clone())
        .await;
        assert!(result.unwrap_err().to_string().contains("timed out"));
        assert!(!started.load(Ordering::SeqCst));
        diagnostics.assert_failure("registration", "operation_timeout");
    }

    #[tokio::test]
    async fn renewal_timeout_fails_the_read_before_collection_can_finish() {
        let diagnostics = Diagnostics::default();
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let throttle = Arc::new(ThrottledStore::new(store.inner.clone(), Default::default()));
        let mut delayed = store.as_ref().clone();
        delayed.inner = throttle.clone();
        let result = protect(Arc::new(delayed), &root, quick_policy(), async {
            throttle.config_mut(|config| config.wait_put_per_call = Duration::from_secs(1));
            std::future::pending::<lance::Result<()>>().await
        })
        .with_subscriber(diagnostics.clone())
        .await;
        assert!(result.unwrap_err().to_string().contains("timed out"));
        {
            let events = diagnostics.0.lock().unwrap();
            assert_eq!(events.len(), 2, "{events:?}");
            for (event, phase) in events.iter().zip(["renewal", "renewal_retry"]) {
                assert_eq!(event["phase"], phase);
                assert_eq!(event["category"], "operation_timeout");
                assert!(event["operation_ms"].parse::<u64>().is_ok());
                assert!(event["acknowledged_silence_ms"].parse::<u64>().is_ok());
            }
        }
        let listing = store.inner.list(Some(&readers_root(&root)));
        wait_for_listing(store, listing, quick_policy(), 2)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn retirement_waits_for_existing_reader_but_not_later_readers() {
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let reached = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let reading = {
            let (store, root, reached, release) = (
                store.clone(),
                root.clone(),
                reached.clone(),
                release.clone(),
            );
            tokio::spawn(async move {
                protect(store, &root, quick_policy(), async {
                    reached.notify_one();
                    release.notified().await;
                    Ok(7)
                })
                .await
            })
        };
        reached.notified().await;
        let captured = captured_readers(&store, &root).await.unwrap();
        assert_eq!(captured.len(), 1);
        let later_path = readers_root(&root).join(uuid::Uuid::new_v4().simple().to_string());
        write_nonce(&store, &later_path, true).await.unwrap();
        let collecting = tokio::spawn(wait_for_readers(store.clone(), captured, quick_policy()));
        // Keep the first reader alive beyond an abandoned nonce's expiration;
        // successful heartbeats, not merely a long grace timer, must protect it.
        tokio::time::sleep(Duration::from_millis(900)).await;
        if collecting.is_finished() {
            release.notify_one();
            // Distinguish an unsafe successful read from a reader that already
            // failed closed (for example on a storage renewal timeout). Keep
            // the liveness assertion; never turn a failed reader into a pass.
            let reader_result = reading.await;
            let collector_result = collecting.await;
            panic!(
                "collector finished before reader release: reader={reader_result:?}, collector={collector_result:?}"
            );
        }
        release.notify_one();
        assert_eq!(reading.await.unwrap().unwrap(), 7);
        tokio::time::timeout(Duration::from_secs(2), collecting)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(read_nonce(&store, &later_path).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn abandoned_registration_expires_without_a_wall_clock() {
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let path = readers_root(&root).join(uuid::Uuid::new_v4().simple().to_string());
        write_nonce(&store, &path, true).await.unwrap();
        let started = Instant::now();
        wait_for_readers(store.clone(), vec![path.clone()], quick_policy())
            .await
            .unwrap();
        assert!(started.elapsed() >= quick_policy().stale_after);
        assert!(read_nonce(&store, &path).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn malformed_registration_never_authorizes_collection() {
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let path = readers_root(&root).join("malformed");
        store.put(&path, b"invalid").await.unwrap();
        assert!(wait_for_readers(store, vec![path], quick_policy())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn cancellation_releases_a_live_reader_without_waiting_for_expiry() {
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let reached = Arc::new(tokio::sync::Notify::new());
        let reading = {
            let (store, root, reached) = (store.clone(), root.clone(), reached.clone());
            tokio::spawn(async move {
                protect(store, &root, quick_policy(), async {
                    reached.notify_one();
                    std::future::pending::<lance::Result<()>>().await
                })
                .await
            })
        };
        reached.notified().await;
        let captured = captured_readers(&store, &root).await.unwrap();
        assert_eq!(captured.len(), 1);
        reading.abort();
        assert!(reading.await.unwrap_err().is_cancelled());
        tokio::time::timeout(
            Duration::from_millis(500),
            wait_for_readers(store, captured, quick_policy()),
        )
        .await
        .unwrap()
        .unwrap();
    }

    #[tokio::test]
    async fn a_paused_reader_cannot_return_data_after_its_renewal_deadline() {
        let dir = tempfile::TempDir::new().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let policy = quick_policy();
        let result = protect(store, &root, policy, async {
            // Deliberately pause this single-thread test runtime: a timeout
            // future alone cannot enforce the return-time validity check.
            std::thread::sleep(policy.max_silence + Duration::from_millis(50));
            Ok(7)
        })
        .await;
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expired before returning data"));
    }
}

#[cfg(test)]
#[path = "wal_read_lifetime_azure_tests.rs"]
mod azure_tests;

#[cfg(test)]
#[path = "wal_read_renewal_tests.rs"]
mod renewal_tests;
