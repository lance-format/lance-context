//! One-shot gates around real data-file IO; all other store behavior is delegated.
use super::*;
use futures::stream::BoxStream;
use lance_io::object_store::{
    providers::{ObjectStoreProvider, ObjectStoreRegistry},
    ObjectStore as LanceStore, ObjectStoreParams,
};
use object_store::{
    path::Path, CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Debug)]
struct Gate {
    entered: oneshot::Sender<()>,
    resume: oneshot::Receiver<()>,
}

pub(super) struct Paused {
    entered: oneshot::Receiver<()>,
    resume: Option<oneshot::Sender<()>>,
}
impl Paused {
    pub async fn entered(&mut self) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(5), &mut self.entered)
            .await
            .context("expected native data IO was not reached")??;
        Ok(())
    }
    pub fn release(mut self) {
        let _ = self.resume.take().unwrap().send(());
    }
}
impl Drop for Paused {
    fn drop(&mut self) {
        if let Some(resume) = self.resume.take() {
            let _ = resume.send(());
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct Gates {
    get: Mutex<Option<Gate>>,
    put: Mutex<Option<Gate>>,
    committed_put: Mutex<Option<Gate>>,
    pub(super) failed_upload: Mutex<Option<Path>>,
    pub(super) multipart_parts: AtomicUsize,
    pub(super) multipart_completions: AtomicUsize,
}
impl Gates {
    fn arm(slot: &Mutex<Option<Gate>>) -> Paused {
        let (entered, received) = oneshot::channel();
        let (resume, resumed) = oneshot::channel();
        assert!(slot
            .lock()
            .unwrap()
            .replace(Gate {
                entered,
                resume: resumed
            })
            .is_none());
        Paused {
            entered: received,
            resume: Some(resume),
        }
    }
    pub fn pause_get(&self) -> Paused {
        Self::arm(&self.get)
    }
    pub fn pause_put(&self) -> Paused {
        Self::arm(&self.put)
    }
    pub fn lose_put_reply(&self) -> Paused {
        Self::arm(&self.committed_put)
    }
    async fn after_put(&self, path: &Path) -> object_store::Result<()> {
        if !path.as_ref().ends_with(".lance") {
            return Ok(());
        }
        let gate = self.committed_put.lock().unwrap().take();
        if let Some(gate) = gate {
            assert!(self
                .failed_upload
                .lock()
                .unwrap()
                .replace(path.clone())
                .is_none());
            let _ = gate.entered.send(());
            let _ = gate.resume.await;
            return Err(object_store::Error::Generic {
                store: "annotation-test",
                source: std::io::Error::other("injected lost native PUT reply").into(),
            });
        }
        Ok(())
    }
    async fn wait(slot: &Mutex<Option<Gate>>, path: &Path) {
        if !path.as_ref().ends_with(".lance") {
            return;
        }
        let gate = slot.lock().unwrap().take();
        if let Some(gate) = gate {
            let _ = gate.entered.send(());
            let _ = gate.resume.await;
        }
    }
}

#[derive(Debug)]
struct Provider {
    inner: Arc<dyn ObjectStoreProvider>,
    gates: Arc<Gates>,
}
#[async_trait::async_trait]
impl ObjectStoreProvider for Provider {
    async fn new_store(
        &self,
        uri: url::Url,
        params: &ObjectStoreParams,
    ) -> lance::Result<LanceStore> {
        let mut store = self.inner.new_store(uri, params).await?;
        store.inner = Arc::new(GatedStore {
            inner: store.inner,
            gates: self.gates.clone(),
        });
        Ok(store)
    }
}
pub(super) async fn storage(path: &std::path::Path, gates: Arc<Gates>) -> Result<Storage> {
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert(
        "file-object-store",
        Arc::new(Provider {
            inner: registry.get_provider("file-object-store").unwrap(),
            gates,
        }),
    );
    Storage::resolve(
        format!("file-object-store://{}", path.display()),
        Default::default(),
        Arc::new(lance::session::Session::new(1 << 20, 1 << 20, registry)),
    )
    .await
}

#[derive(Debug)]
struct GatedStore {
    inner: Arc<dyn ObjectStore>,
    gates: Arc<Gates>,
}
impl std::fmt::Display for GatedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gated({})", self.inner)
    }
}
#[async_trait::async_trait]
impl ObjectStore for GatedStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        Gates::wait(&self.gates.put, path).await;
        let result = self.inner.put_opts(path, payload, opts).await?;
        self.gates.after_put(path).await?;
        Ok(result)
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Gates::wait(&self.gates.put, path).await;
        Ok(Box::new(GatedUpload {
            inner: self.inner.put_multipart_opts(path, opts).await?,
            gates: self.gates.clone(),
            path: path.clone(),
        }))
    }
    async fn get_opts(&self, path: &Path, opts: GetOptions) -> object_store::Result<GetResult> {
        if !opts.head {
            Gates::wait(&self.gates.get, path).await;
        }
        self.inner.get_opts(path, opts).await
    }
    async fn get_ranges(
        &self,
        path: &Path,
        ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        Gates::wait(&self.gates.get, path).await;
        self.inner.get_ranges(path, ranges).await
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

#[derive(Debug)]
struct GatedUpload {
    inner: Box<dyn MultipartUpload>,
    gates: Arc<Gates>,
    path: Path,
}
#[async_trait::async_trait]
impl MultipartUpload for GatedUpload {
    fn put_part(&mut self, data: PutPayload) -> object_store::UploadPart {
        self.gates.multipart_parts.fetch_add(1, Ordering::SeqCst);
        self.inner.put_part(data)
    }
    async fn complete(&mut self) -> object_store::Result<PutResult> {
        let result = self.inner.complete().await?;
        self.gates
            .multipart_completions
            .fetch_add(1, Ordering::SeqCst);
        self.gates.after_put(&self.path).await?;
        Ok(result)
    }
    async fn abort(&mut self) -> object_store::Result<()> {
        self.inner.abort().await
    }
}
