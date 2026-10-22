use super::*;
use arrow_array::{LargeBinaryArray, RecordBatchIterator};
use async_trait::async_trait;
use futures::stream::BoxStream;
use lance::dataset::{transaction::Transaction, CommitBuilder};
use lance_io::object_store::{
    providers::memory::MemoryStoreProvider, ObjectStore, ObjectStoreParams, ObjectStoreProvider,
    ObjectStoreRegistry,
};
use object_store::{
    path::Path, signer::Url, CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload,
    ObjectMeta, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
    Result as StoreResult,
};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::{watch, Notify};

#[derive(Debug)]
struct Gate {
    armed: AtomicBool,
    fail: AtomicBool,
    started: watch::Sender<bool>,
    release: Notify,
}
impl Gate {
    async fn pause(&self, path: &Path) -> StoreResult<()> {
        if path.as_ref().ends_with(".lance") {
            if self.armed.swap(false, Ordering::AcqRel) {
                self.started.send_replace(true);
                self.release.notified().await;
            }
            if self.fail.load(Ordering::Acquire) {
                return Err(object_store::Error::Generic {
                    store: "injected write failure",
                    source: Box::new(std::io::Error::other("injected write failure")),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PausedStore {
    inner: Arc<dyn object_store::ObjectStore>,
    gate: Arc<Gate>,
}
impl std::fmt::Display for PausedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PausedStore({})", self.inner)
    }
}
#[async_trait]
impl object_store::ObjectStore for PausedStore {
    async fn put_opts(&self, p: &Path, b: PutPayload, o: PutOptions) -> StoreResult<PutResult> {
        self.gate.pause(p).await?;
        self.inner.put_opts(p, b, o).await
    }
    async fn put_multipart_opts(
        &self,
        p: &Path,
        o: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        self.gate.pause(p).await?;
        self.inner.put_multipart_opts(p, o).await
    }
    async fn get_opts(&self, p: &Path, o: GetOptions) -> StoreResult<GetResult> {
        self.inner.get_opts(p, o).await
    }
    fn delete_stream(
        &self,
        p: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        self.inner.delete_stream(p)
    }
    fn list(&self, p: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.inner.list(p)
    }
    async fn list_with_delimiter(&self, p: Option<&Path>) -> StoreResult<ListResult> {
        self.inner.list_with_delimiter(p).await
    }
    async fn copy_opts(&self, from: &Path, to: &Path, o: CopyOptions) -> StoreResult<()> {
        self.inner.copy_opts(from, to, o).await
    }
    async fn rename_opts(&self, from: &Path, to: &Path, o: RenameOptions) -> StoreResult<()> {
        self.inner.rename_opts(from, to, o).await
    }
}

#[derive(Debug)]
struct PausedProvider(Arc<PausedStore>);
#[async_trait]
impl ObjectStoreProvider for PausedProvider {
    async fn new_store(&self, url: Url, params: &ObjectStoreParams) -> lance::Result<ObjectStore> {
        // Use Lance's object-store writer, while keeping the actual objects on
        // disk. A file-scheme store could bypass this hook with its local writer.
        let mut store = MemoryStoreProvider.new_store(url, params).await?;
        store.inner = self.0.clone();
        Ok(store)
    }
    fn extract_path(&self, url: &Url) -> lance::Result<Path> {
        MemoryStoreProvider.extract_path(url)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn source_fetch_progresses_while_real_lance_object_write_is_paused() -> Result<()> {
    write_overlap(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_object_write_stops_blocked_prefetch_and_releases_its_budget() -> Result<()> {
    write_overlap(true).await
}

struct MarkDrop(Arc<AtomicBool>);
impl Drop for MarkDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

async fn write_overlap(fail_write: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let (started, mut writing) = watch::channel(false);
    let gate = Arc::new(Gate {
        armed: AtomicBool::new(false),
        fail: AtomicBool::new(false),
        started,
        release: Notify::new(),
    });
    let backend = object_store::local::LocalFileSystem::new_with_prefix(root.path())?;
    let registry = ObjectStoreRegistry::default();
    registry.insert(
        "memory",
        Arc::new(PausedProvider(Arc::new(PausedStore {
            inner: Arc::new(backend),
            gate: gate.clone(),
        }))),
    );
    let session = Arc::new(lance::session::Session::new(0, 0, Arc::new(registry)));
    let small = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(LargeBinaryArray::from_iter_values([b"small".as_slice()]))
            as arrow_array::ArrayRef,
    )])?;
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(small.clone())], small.schema()),
        "memory://overlap.lance",
        Some(WriteParams {
            session: Some(session.clone()),
            ..Default::default()
        }),
    )
    .await?;
    let original_version = dataset.version().version;
    // Exceed the writer's buffering/first multipart threshold with real data.
    // Deterministic high-entropy bytes prevent compression into a tiny upload.
    let mut payload = vec![0; 128 << 20];
    let mut state = 7u64;
    for bytes in payload.as_chunks_mut::<8>().0 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bytes.copy_from_slice(&state.to_le_bytes());
    }
    let large = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(LargeBinaryArray::from_iter_values([payload.as_slice()])) as arrow_array::ArrayRef,
    )])?;
    drop(payload);
    let expected_large = large.clone();
    let fetched = Arc::new(Notify::new());
    let fetched_next = fetched.clone();
    let source_dropped = Arc::new(AtomicBool::new(false));
    let mark_drop = source_dropped.clone();
    let mut source_writing = writing.clone();
    let second = small.clone();
    let source = stream::iter([Ok(large)]).chain(stream::once(async move {
        let _mark_drop = MarkDrop(mark_drop);
        source_writing
            .wait_for(|v| *v)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        fetched_next.notify_one();
        if fail_write {
            futures::future::pending::<()>().await;
        }
        Ok(second)
    }));
    let streams: Vec<SendableRecordBatchStream> = vec![Box::pin(RecordBatchStreamAdapter::new(
        small.schema(),
        source,
    ))];
    let budget = MergeMemoryBudget::new(512 << 20);
    let reservation = Arc::new(budget.reserve(512 << 20).await);
    gate.armed.store(true, Ordering::Release);
    gate.fail.store(fail_write, Ordering::Release);
    let writer_dataset = dataset.clone();
    let job = tokio::spawn(async move {
        write_streams(writer_dataset, streams, reservation, Arc::default()).await
    });
    let reached = tokio::time::timeout(Duration::from_secs(30), writing.wait_for(|v| *v))
        .await
        .is_ok_and(|result| result.is_ok());
    let overlapped = if reached {
        tokio::time::timeout(Duration::from_secs(2), fetched.notified())
            .await
            .is_ok()
    } else {
        false
    };
    // Release the real writer and join it even when the overlap assertion fails.
    gate.started.send_replace(true);
    gate.release.notify_one();
    let result = tokio::time::timeout(Duration::from_secs(30), job).await??;
    assert_eq!(budget.reserved(), 0);
    assert!(source_dropped.load(Ordering::Acquire));
    if fail_write {
        assert!(overlapped, "blocked source was never polled while writing");
        assert!(result.is_err(), "failed file write returned descriptors");
        let reopened = lance::dataset::builder::DatasetBuilder::from_uri("memory://overlap.lance")
            .with_session(session)
            .load()
            .await?;
        assert_eq!(reopened.version().version, original_version);
        assert_eq!(reopened.count_rows(None).await?, 1);
        return Ok(());
    }
    let descriptors = result?;
    let fragments = serde_json::from_slice(&descriptors.bytes)?;
    let appended = CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            original_version,
            Operation::Append { fragments },
            None,
        ))
        .await?;
    assert_eq!(appended.count_rows(None).await?, 3);
    let expected_bytes = expected_large
        .column(0)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap();
    let mut scan = appended.scan();
    scan.batch_size(1).batch_readahead(1).fragment_readahead(1);
    let mut batches = scan.try_into_stream().await?;
    let (mut large_rows, mut small_rows) = (0, 0);
    while let Some(batch) = batches.try_next().await? {
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        for bytes in values.iter().flatten() {
            if bytes == b"small" {
                small_rows += 1;
            } else {
                assert!(bytes == expected_bytes.value(0), "large payload differs");
                large_rows += 1;
            }
        }
    }
    assert_eq!((large_rows, small_rows), (1, 2));
    assert!(
        overlapped,
        "source fetch stopped while a real object-store write was blocked"
    );
    Ok(())
}
