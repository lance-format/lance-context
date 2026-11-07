//! Keep admitted file uploads alive through explicit storage throttling.
//! No preparation, catalog mutation, or multipart completion is re-executed.
use crate::materializer::Storage;
use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use futures::stream::BoxStream;
use lance_io::object_store::{
    providers::ObjectStoreProvider, ObjectStore as LanceStore, ObjectStoreParams,
};
use object_store::{
    path::Path, CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMode, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
};
use serde::Serialize;
use std::{
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::watch;
use uuid::Uuid;

mod multipart;

#[derive(Debug, Default)]
struct Counters {
    waiting: AtomicU64,
    resubmissions: AtomicU64,
    waiting_part_retries: AtomicU64,
    part_retry_decisions: AtomicU64,
}

#[derive(Debug, Serialize)]
pub(crate) struct Observation {
    pub(crate) waiting_objects: u64,
    /// Calls to the underlying store, not HTTP attempts inside its SDK.
    pub(crate) put_resubmissions: u64,
    /// Parts currently waiting in the SDK policy's backoff.
    pub(crate) waiting_part_retries: u64,
    /// Policy approvals, not HTTP attempts inside the SDK.
    pub(crate) part_retry_decisions: u64,
}

/// Held by the owner role, never by its detached file writers. Dropping this
/// scope cancels even a writer whose scheduling caller has already disappeared.
pub(crate) struct Scope {
    lifetime: Option<watch::Sender<()>>,
    enabled: Arc<AtomicBool>,
    counters: Arc<Counters>,
}

impl Scope {
    /// Install before opening a dataset, in this owner's private fresh session.
    pub(crate) fn install(storage: &Storage) -> Result<Self> {
        let registry = storage.session.store_registry();
        ensure!(
            registry.active_stores().is_empty(),
            "metadata upload scope requires an unopened owner session"
        );
        let scheme = storage
            .uri
            .split_once("://")
            .map_or_else(|| "file".to_owned(), |(s, _)| s.to_ascii_lowercase());
        let inner = registry
            .get_provider(&scheme)
            .context("metadata upload store provider missing")?;
        let (lifetime, stopped) = watch::channel(());
        let counters = Arc::new(Counters::default());
        let enabled = Arc::new(AtomicBool::new(false));
        registry.insert(
            &scheme,
            Arc::new(Provider {
                inner,
                base: storage.base.clone(),
                stopped,
                enabled: enabled.clone(),
                counters: counters.clone(),
            }),
        );
        Ok(Self {
            lifetime: Some(lifetime),
            enabled,
            counters,
        })
    }

    /// Activate only once the owner can monitor its lease and shutdown. During
    /// connection/recovery the existing bounded SDK failure behavior remains.
    pub(crate) fn start(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    pub(crate) fn cancel(&mut self) {
        self.lifetime.take();
    }

    pub(crate) fn observation(&self) -> Observation {
        Observation {
            waiting_objects: self.counters.waiting.load(Ordering::Relaxed),
            put_resubmissions: self.counters.resubmissions.load(Ordering::Relaxed),
            waiting_part_retries: self.counters.waiting_part_retries.load(Ordering::Relaxed),
            part_retry_decisions: self.counters.part_retry_decisions.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
struct Provider {
    inner: Arc<dyn ObjectStoreProvider>,
    base: Path,
    stopped: watch::Receiver<()>,
    enabled: Arc<AtomicBool>,
    counters: Arc<Counters>,
}

#[async_trait::async_trait]
impl ObjectStoreProvider for Provider {
    async fn new_store(
        &self,
        uri: url::Url,
        params: &ObjectStoreParams,
    ) -> lance::Result<LanceStore> {
        let mut store = self.inner.new_store(uri, params).await?;
        store.inner = Arc::new(UploadStore {
            inner: store.inner,
            base: self.base.clone(),
            stopped: self.stopped.clone(),
            enabled: self.enabled.clone(),
            counters: self.counters.clone(),
        });
        // Listing is unchanged, including the native paginated listing path.
        Ok(store)
    }
}

#[derive(Debug)]
struct UploadStore {
    inner: Arc<dyn ObjectStore>,
    base: Path,
    stopped: watch::Receiver<()>,
    enabled: Arc<AtomicBool>,
    counters: Arc<Counters>,
}

impl std::fmt::Display for UploadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("owner immutable metadata uploads")
    }
}

fn immutable_metadata(base: &Path, path: &Path) -> bool {
    let Some(relative) = path.as_ref().strip_prefix(&format!("{base}/")) else {
        return false;
    };
    if let Some(name) = relative.strip_prefix("_versions/") {
        // ManagedManifest gives each attempt its own UUID-suffixed object.
        // Exclude native version paths, latest pointers, and nested paths.
        return !name.contains('/')
            && name
                .rsplit_once(".manifest-")
                .is_some_and(|(_, id)| Uuid::parse_str(id).is_ok());
    }
    relative
        .strip_prefix("_indices/")
        .and_then(|name| name.split_once('/'))
        .is_some_and(|(id, name)| Uuid::parse_str(id).is_ok() && !name.is_empty())
}

fn cancelled() -> object_store::Error {
    object_store::Error::Generic {
        store: "owner metadata upload",
        source: std::io::Error::new(std::io::ErrorKind::Interrupted, "owner upload scope closed")
            .into(),
    }
}

fn explicit_throttle(error: &object_store::Error) -> bool {
    let object_store::Error::Generic { source, .. } = error else {
        return false;
    };
    // The pinned SDK keeps RequestError private. Inspect its own status prefix
    // in the source chain, not words or numbers in a URL or response body.
    // Unknown formats fail closed to the existing owner recovery path.
    let mut source: &(dyn std::error::Error + 'static) = source.as_ref();
    for _ in 0..16 {
        let text = source.to_string();
        if let Some(status) = text.strip_prefix("Server returned non-2xx status code: ") {
            return matches!(status.split_whitespace().next(), Some("429" | "503"));
        }
        let Some(next) = source.source() else { break };
        source = next;
    }
    false
}

struct Waiting(Arc<Counters>);
impl Drop for Waiting {
    fn drop(&mut self) {
        self.0.waiting.fetch_sub(1, Ordering::Relaxed);
    }
}

#[async_trait::async_trait]
impl ObjectStore for UploadStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        if !self.enabled.load(Ordering::Acquire)
            || !matches!(opts.mode, PutMode::Overwrite)
            || !immutable_metadata(&self.base, path)
        {
            return self.inner.put_opts(path, payload, opts).await;
        }
        let mut stopped = self.stopped.clone();
        let mut waiting = None;
        let mut delay = Duration::from_secs(1);
        loop {
            // PutPayload clones share the existing Bytes. The admitted writer
            // still owns its memory allowance; no queue or new file is created.
            let result = tokio::select! {
                biased;
                _ = stopped.changed() => return Err(cancelled()),
                result = self.inner.put_opts(path, payload.clone(), opts.clone()) => result,
            };
            match result {
                Err(error) if explicit_throttle(&error) => {
                    waiting.get_or_insert_with(|| {
                        self.counters.waiting.fetch_add(1, Ordering::Relaxed);
                        Waiting(self.counters.clone())
                    });
                    tokio::select! {
                        biased;
                        _ = stopped.changed() => return Err(cancelled()),
                        _ = tokio::time::sleep(delay) => {},
                    }
                    delay = (delay * 2).min(Duration::from_secs(30));
                    self.counters.resubmissions.fetch_add(1, Ordering::Relaxed);
                }
                result => return result,
            }
        }
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        if !self.enabled.load(Ordering::Acquire) {
            return self.inner.put_multipart_opts(path, opts).await;
        }
        multipart::start(self, path, opts).await
    }
    async fn get_opts(&self, path: &Path, opts: GetOptions) -> object_store::Result<GetResult> {
        self.inner.get_opts(path, opts).await
    }
    async fn get_ranges(
        &self,
        path: &Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        self.inner.get_ranges(path, ranges).await
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }
    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }
    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        opts: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, opts).await
    }
    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        opts: RenameOptions,
    ) -> object_store::Result<()> {
        self.inner.rename_opts(from, to, opts).await
    }
}

#[cfg(test)]
pub(crate) mod tests;
