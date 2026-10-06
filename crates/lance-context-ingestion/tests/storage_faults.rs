use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use lance_context_ingestion::{Binding, Entry, Error, Journal, Position, Transition};
use object_store::{
    memory::InMemory, path::Path, CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMode, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

/// Real conditional-put storage with an injected response loss, not a replacement
/// journal. Tests distinguish an orphan upload from a successful head mutation.
#[derive(Debug)]
struct FaultStore {
    inner: InMemory,
    head_failure: AtomicUsize,
}

impl fmt::Display for FaultStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fault-store")
    }
}

fn injected() -> object_store::Error {
    object_store::Error::Generic {
        store: "fault-store",
        source: std::io::Error::other("injected response loss").into(),
    }
}

#[async_trait]
impl ObjectStore for FaultStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        let failure =
            if path.filename() == Some("head.json") && matches!(opts.mode, PutMode::Update(_)) {
                self.head_failure.swap(0, Ordering::SeqCst)
            } else {
                0
            };
        if failure == 1 {
            return Err(injected());
        }
        let result = self.inner.put_opts(path, payload, opts).await?;
        if failure == 2 {
            return Err(injected());
        }
        Ok(result)
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, opts).await
    }
    async fn get_opts(&self, path: &Path, opts: GetOptions) -> object_store::Result<GetResult> {
        self.inner.get_opts(path, opts).await
    }
    fn delete_stream(
        &self,
        paths: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(paths)
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        opts: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, opts).await
    }
}

fn binding() -> Binding {
    Binding {
        run: "r".into(),
        schema: "s".into(),
        partition: 0,
    }
}

fn entry() -> Entry {
    Entry {
        sequence: 1,
        session: "session".into(),
        receipt: "source-receipt".into(),
        input_digest: "exact-input-digest".into(),
        transition: Transition {
            delta: vec![7],
            records: vec![9],
        },
    }
}

#[tokio::test]
async fn uncertain_head_write_is_reconciled_from_storage_not_from_upload_success() {
    for failure in [1, 2] {
        let store = Arc::new(FaultStore {
            inner: InMemory::new(),
            head_failure: AtomicUsize::new(0),
        });
        let journal = Journal::new(store.clone(), Path::from("wal"), binding(), 65536, 4).unwrap();
        let mut writer = journal.acquire().await.unwrap();
        store.head_failure.store(failure, Ordering::SeqCst);
        assert!(writer.append(vec![entry()]).await.is_err());
        assert!(matches!(
            writer.append(vec![entry()]).await,
            Err(Error::Fenced)
        ));
        let mut recovered = journal.acquire().await.unwrap();
        assert_eq!(
            recovered.position().sequence,
            if failure == 2 { 1 } else { 0 }
        );
        if failure == 1 {
            recovered.append(vec![entry()]).await.unwrap();
        }
        let committed = journal
            .pending(&Position::default(), recovered.position())
            .await
            .unwrap();
        assert_eq!(committed.len(), 1);
        assert_eq!(journal.entries(&committed[0]).await.unwrap(), vec![entry()]);
    }
}

#[tokio::test]
async fn local_filesystem_without_conditional_updates_is_rejected_not_silently_emulated() {
    let dir = tempfile::tempdir().unwrap();
    let store =
        Arc::new(object_store::local::LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    let journal = Journal::new(store, Path::from("wal"), binding(), 65536, 4).unwrap();
    let mut writer = journal.acquire().await.unwrap();
    assert!(writer.append(vec![entry()]).await.is_err());
    assert_eq!(journal.position().await.unwrap(), Position::default());
    assert!(journal.acquire().await.is_err());
}
