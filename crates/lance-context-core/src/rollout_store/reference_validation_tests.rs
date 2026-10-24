//! Faults below the real object-store HEAD path; no validation mocks.
use super::*;
use crate::merge_write_scope::{CommitAuthorizer, MergeWriteScope};
use async_trait::async_trait;
use futures::stream::BoxStream;
use lance_io::object_store::WrappingObjectStore;
use object_store::{path::Path, *};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::Notify;

#[derive(Debug, Default)]
struct HeadGate {
    fail: AtomicBool,
    pause: AtomicBool,
    entered: Notify,
    release: Notify,
    heads: AtomicUsize,
}
#[derive(Debug)]
struct Wrapper(Arc<HeadGate>);
impl WrappingObjectStore for Wrapper {
    fn wrap_paginated(
        &self,
        _: &str,
        store: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        Some(store)
    }
    fn wrap(&self, _: &str, store: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(HeadStore {
            inner: store,
            gate: self.0.clone(),
        })
    }
}
#[derive(Debug)]
struct HeadStore {
    inner: Arc<dyn ObjectStore>,
    gate: Arc<HeadGate>,
}
impl std::fmt::Display for HeadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HeadFault({})", self.inner)
    }
}
#[async_trait]
impl ObjectStore for HeadStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(path, payload, opts).await
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, opts).await
    }
    async fn get_opts(&self, path: &Path, opts: GetOptions) -> Result<GetResult> {
        if path.as_ref().contains("/data/") || path.as_ref().contains("/_deletions/") {
            assert!(
                opts.head,
                "reference validation must not decode row payloads"
            );
            self.gate.heads.fetch_add(1, Ordering::SeqCst);
            if self.gate.pause.swap(false, Ordering::SeqCst) {
                self.gate.entered.notify_one();
                self.gate.release.notified().await;
            }
            if self.gate.fail.load(Ordering::SeqCst) {
                return Err(Error::Generic {
                    store: "validation-test",
                    source: std::io::Error::other("injected HEAD transport failure").into(),
                });
            }
        }
        self.inner.get_opts(path, opts).await
    }
    fn delete_stream(
        &self,
        paths: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(paths)
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }
    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> Result<()> {
        self.inner.copy_opts(from, to, opts).await
    }
    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> Result<()> {
        self.inner.rename_opts(from, to, opts).await
    }
}
#[derive(Debug)]
struct NoCommit;
impl CommitAuthorizer for NoCommit {
    fn authorize<'a>(
        &'a self,
        _: &'a str,
        _: u64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = lance::Result<()>> + Send + 'a>> {
        Box::pin(async { Err(lance::Error::io("validation must never commit")) })
    }
}

async fn fixture() -> (tempfile::TempDir, String, RolloutStore) {
    let dir = tempfile::TempDir::new().unwrap();
    let uri = format!("file-object-store://{}", dir.path().display());
    let mut writer = RolloutStore::open(&uri).await.unwrap();
    writer
        .add(&[super::tests::assistant_record("a")])
        .await
        .unwrap();
    writer.cleanup_own_shard().await.unwrap();
    (dir, uri, writer)
}

#[tokio::test]
async fn reference_validation_transport_error_preserves_all_references() {
    let (_dir, uri, writer) = fixture().await;
    let gate = Arc::new(HeadGate::default());
    gate.fail.store(true, Ordering::SeqCst);
    let scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
        Arc::new(NoCommit),
        Arc::new(Wrapper(gate.clone())),
    );
    let error = scope
        .run(async {
            let mut reader = RolloutStore::open_existing_with_options(&uri, Default::default())
                .await
                .unwrap();
            reader.validate_fragment_references().await.unwrap_err()
        })
        .await
        .to_string();
    assert!(error.contains("injected HEAD transport failure"), "{error}");
    assert!(!error.contains("missing data reference"));
    assert_eq!(gate.heads.load(Ordering::SeqCst), 1);
    assert_eq!(
        scope.completed_steps(),
        0,
        "failed HEAD must not count as progress"
    );
    let reader = RolloutStore::open_existing_with_options(&uri, Default::default())
        .await
        .unwrap();
    assert_eq!(reader.version(), writer.version());
    assert_eq!(reader.base_data_files(), writer.base_data_files());
    assert_eq!(reader.list(None, None).await.unwrap().len(), 1);
}

#[tokio::test]
async fn reference_validation_version_race_fails_without_automatic_rescan() {
    let (_dir, uri, mut writer) = fixture().await;
    let gate = Arc::new(HeadGate::default());
    gate.pause.store(true, Ordering::SeqCst);
    let scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
        Arc::new(NoCommit),
        Arc::new(Wrapper(gate.clone())),
    );
    let other_scope = scope.clone();
    let read_uri = uri.clone();
    let validation = tokio::spawn(async move {
        other_scope
            .run(async {
                let mut reader =
                    RolloutStore::open_existing_with_options(&read_uri, Default::default())
                        .await
                        .unwrap();
                reader.validate_fragment_references().await
            })
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(10), gate.entered.notified())
        .await
        .unwrap();
    assert_eq!(
        scope.completed_steps(),
        0,
        "waiting on HEAD is not progress"
    );
    let old = writer.version();
    writer
        .add(&[super::tests::assistant_record("b")])
        .await
        .unwrap();
    writer.cleanup_own_shard().await.unwrap();
    assert!(writer.version() > old);
    gate.release.notify_one();
    let error = tokio::time::timeout(std::time::Duration::from_secs(10), validation)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err()
        .to_string();
    assert!(error.contains("base version changed"), "{error}");
    assert_eq!(
        gate.heads.load(Ordering::SeqCst),
        1,
        "no unbounded rescan under active writer"
    );
    assert_eq!(scope.completed_steps(), 1);
    let reader = RolloutStore::open_existing_with_options(&uri, Default::default())
        .await
        .unwrap();
    assert_eq!(reader.version(), writer.version());
    assert_eq!(reader.list(None, None).await.unwrap().len(), 2);
}
