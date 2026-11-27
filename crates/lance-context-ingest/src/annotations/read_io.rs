//! Diagnostic read counts by a fixed set of file classes, without changing
//! store/cache identity. Counts API calls and delivered bytes, not HTTP attempts
//! or wire bytes. No path, URI, table ID or credential is emitted.
use crate::materializer::Storage;
use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use futures::{stream::BoxStream, Stream};
use lance_io::object_store::{
    providers::ObjectStoreProvider, ObjectStore as LanceStore, ObjectStoreParams,
};
use object_store::{
    path::Path, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
};
use serde::Serialize;
use std::{
    collections::HashMap,
    ops::Range,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context as TaskContext, Poll},
    time::Instant,
};

mod timing;
use timing::{BodyTiming, Elapsed, ElapsedObservation};

const CLASSES: [&str; 5] = ["narrow_data", "body_data", "index", "metadata", "other"];

#[derive(Debug, Default)]
struct Counts {
    started: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    dropped: AtomicU64,
    bytes: AtomicU64,
    heads: AtomicU64,
    ranges: AtomicU64,
    file_payloads: AtomicU64,
    response: Elapsed,
    body_unpolled: Elapsed,
    body_pending: Elapsed,
    body_poll: Elapsed,
}

/// GET completes at stream EOF; a dropped future/body is counted separately.
/// HEAD and get_ranges complete at return. Local file handles are passed through
/// unchanged and counted in file_payloads: their later reads are not observable.
#[derive(Debug, Serialize)]
pub struct Observation {
    class: &'static str,
    started: u64,
    completed: u64,
    failed: u64,
    dropped: u64,
    bytes: u64,
    heads: u64,
    ranges: u64,
    file_payloads: u64,
    // GET: until response headers; HEAD/ranges: until return. Includes backend
    // retries and executor delay, not just server latency. Includes cancellation.
    response: ElapsedObservation,
    // Per-stream totals recorded at EOF, error or drop. Pending includes source
    // and executor wait; unpolled is caller delay between ready chunks/polls.
    // A local file handle has no observable body timing and remains unchanged.
    body_unpolled: ElapsedObservation,
    body_pending: ElapsedObservation,
    body_poll: ElapsedObservation,
}

#[derive(Clone, Debug)]
struct Target {
    store: String,
    base: String,
    data_class: usize,
}

/// API-level read observations for one owner session; this adds no read-ahead.
pub struct Scope(Arc<[Counts; 5]>);

/// Capture backend identity before the existing upload wrapper is installed.
/// Install counters afterward, preserving the owner's original URI resolution.
pub struct Providers(HashMap<String, Arc<dyn ObjectStoreProvider>>);

impl Providers {
    pub fn capture(narrow: &Storage, body: Option<&str>) -> Result<Self> {
        let registry = narrow.session.store_registry();
        let mut providers = HashMap::new();
        for uri in std::iter::once(narrow.uri.as_str()).chain(body) {
            let uri = lance_io::object_store::uri_to_url(uri)?;
            providers.insert(
                uri.scheme().to_owned(),
                registry
                    .get_provider(uri.scheme())
                    .context("read observation provider missing")?,
            );
        }
        Ok(Self(providers))
    }
}

impl Scope {
    /// Install after upload wrapping and before opening any dataset handles.
    pub fn install(
        narrow: &Storage,
        body: Option<&Storage>,
        identities: Providers,
    ) -> Result<Self> {
        let registry = narrow.session.store_registry();
        ensure!(
            registry.active_stores().is_empty(),
            "read observation requires an unopened session"
        );
        let counts = Arc::new(std::array::from_fn(|_| Counts::default()));
        let mut targets = Vec::new();
        let mut providers = HashMap::new();
        for (data_class, storage) in std::iter::once((0, narrow)).chain(body.map(|b| (1, b))) {
            ensure!(
                Arc::ptr_eq(&narrow.session, &storage.session),
                "read observation requires the shared session"
            );
            let uri = lance_io::object_store::uri_to_url(&storage.uri)?;
            let scheme = uri.scheme().to_owned();
            providers.entry(scheme.clone()).or_insert(
                registry
                    .get_provider(&scheme)
                    .context("read observation provider missing")?,
            );
            targets.push(Target {
                store: identities
                    .0
                    .get(&scheme)
                    .context("read identity provider missing")?
                    .calculate_object_store_prefix(&uri, storage.params.storage_options())?,
                base: if storage.base.as_ref().is_empty() {
                    String::new()
                } else {
                    format!("{}/", storage.base)
                },
                data_class,
            });
        }
        // Nested table paths must match the most specific target first.
        targets.sort_by_key(|t| std::cmp::Reverse(t.base.len()));
        let targets = Arc::new(targets);
        for (scheme, inner) in providers {
            registry.insert(
                &scheme,
                Arc::new(Provider {
                    inner,
                    identity: identities
                        .0
                        .get(&scheme)
                        .context("read identity provider missing")?
                        .clone(),
                    targets: targets.clone(),
                    counts: counts.clone(),
                }),
            );
        }
        Ok(Self(counts))
    }

    /// Serializable per-class counters; API durations include SDK retries.
    pub fn observation(&self) -> Vec<Observation> {
        self.0
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let started = c.started.load(Ordering::Relaxed);
                Observation {
                    class: CLASSES[i],
                    started,
                    completed: c.completed.load(Ordering::Relaxed),
                    failed: c.failed.load(Ordering::Relaxed),
                    dropped: c.dropped.load(Ordering::Relaxed),
                    bytes: c.bytes.load(Ordering::Relaxed),
                    heads: c.heads.load(Ordering::Relaxed),
                    ranges: c.ranges.load(Ordering::Relaxed),
                    file_payloads: c.file_payloads.load(Ordering::Relaxed),
                    response: c.response.observation(),
                    body_unpolled: c.body_unpolled.observation(),
                    body_pending: c.body_pending.observation(),
                    body_poll: c.body_poll.observation(),
                }
            })
            .filter(|row| row.started != 0)
            .collect()
    }
}

#[derive(Debug)]
struct Provider {
    inner: Arc<dyn ObjectStoreProvider>,
    identity: Arc<dyn ObjectStoreProvider>,
    targets: Arc<Vec<Target>>,
    counts: Arc<[Counts; 5]>,
}

#[async_trait::async_trait]
impl ObjectStoreProvider for Provider {
    async fn new_store(
        &self,
        uri: url::Url,
        params: &ObjectStoreParams,
    ) -> lance::Result<LanceStore> {
        let prefix = self
            .identity
            .calculate_object_store_prefix(&uri, params.storage_options())?;
        let mut store = self.inner.new_store(uri, params).await?;
        store.inner = Arc::new(ReadStore {
            inner: store.inner,
            targets: self
                .targets
                .iter()
                .filter(|t| t.store == prefix)
                .cloned()
                .collect(),
            counts: self.counts.clone(),
        });
        Ok(store)
    }

    fn extract_path(&self, uri: &url::Url) -> lance::Result<Path> {
        self.inner.extract_path(uri)
    }

    fn calculate_object_store_prefix(
        &self,
        uri: &url::Url,
        options: Option<&HashMap<String, String>>,
    ) -> lance::Result<String> {
        self.inner.calculate_object_store_prefix(uri, options)
    }
}

#[derive(Debug)]
struct ReadStore {
    inner: Arc<dyn ObjectStore>,
    targets: Vec<Target>,
    counts: Arc<[Counts; 5]>,
}

impl std::fmt::Display for ReadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("owner classified reads")
    }
}

impl ReadStore {
    fn begin(&self, path: &Path) -> Read {
        let class = self
            .targets
            .iter()
            .find_map(|target| {
                let relative = path.as_ref().strip_prefix(&target.base)?;
                Some(if relative.starts_with("data/") {
                    target.data_class
                } else if relative.starts_with("_indices/") {
                    2
                } else if relative.starts_with("_versions/")
                    || relative.starts_with("_transactions/")
                    || relative == "_latest.manifest"
                {
                    3
                } else {
                    4
                })
            })
            .unwrap_or(4);
        self.counts[class].started.fetch_add(1, Ordering::Relaxed);
        Read {
            counts: self.counts.clone(),
            class,
            finished: false,
            response_started: Some(Instant::now()),
        }
    }
}

struct Read {
    counts: Arc<[Counts; 5]>,
    class: usize,
    finished: bool,
    response_started: Option<Instant>,
}

impl Read {
    fn counts(&self) -> &Counts {
        &self.counts[self.class]
    }
    fn response(&mut self) {
        if let Some(start) = self.response_started.take() {
            self.counts().response.record(start.elapsed());
        }
    }
    fn finish(&mut self, failed: bool) {
        self.response();
        if !self.finished {
            let counter = if failed {
                &self.counts().failed
            } else {
                &self.counts().completed
            };
            counter.fetch_add(1, Ordering::Relaxed);
            self.finished = true;
        }
    }
}

impl Drop for Read {
    fn drop(&mut self) {
        self.response();
        if !self.finished {
            self.counts().dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct Body {
    inner: BoxStream<'static, object_store::Result<Bytes>>,
    read: Read,
    timing: BodyTiming,
}

impl Body {
    fn new(inner: BoxStream<'static, object_store::Result<Bytes>>, mut read: Read) -> Self {
        read.response();
        Self {
            inner,
            read,
            timing: BodyTiming::new(Instant::now()),
        }
    }
}

impl Drop for Body {
    fn drop(&mut self) {
        self.timing.finish(Instant::now(), self.read.counts());
    }
}

impl Stream for Body {
    type Item = object_store::Result<Bytes>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.timing.before_poll(Instant::now());
        let result = this.inner.as_mut().poll_next(cx);
        this.timing.after_poll(Instant::now(), result.is_pending());
        match &result {
            Poll::Ready(Some(Ok(bytes))) => {
                this.read
                    .counts()
                    .bytes
                    .fetch_add(bytes.len() as u64, Ordering::Relaxed);
            }
            Poll::Ready(Some(Err(_))) => {
                this.timing.finish(Instant::now(), this.read.counts());
                this.read.finish(true);
            }
            Poll::Ready(None) => {
                this.timing.finish(Instant::now(), this.read.counts());
                this.read.finish(false);
            }
            Poll::Pending => {}
        }
        result
    }
}

#[async_trait::async_trait]
impl ObjectStore for ReadStore {
    async fn get_opts(&self, path: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        let mut read = self.begin(path);
        let head = options.head;
        if head {
            read.counts().heads.fetch_add(1, Ordering::Relaxed);
        }
        let mut result = match self.inner.get_opts(path, options).await {
            Ok(result) => result,
            Err(error) => {
                read.finish(true);
                return Err(error);
            }
        };
        read.response();
        if head {
            read.finish(false);
            return Ok(result);
        }
        result.payload = match result.payload {
            GetResultPayload::Stream(inner) => {
                GetResultPayload::Stream(Box::pin(Body::new(inner, read)))
            }
            payload => {
                read.counts().file_payloads.fetch_add(1, Ordering::Relaxed);
                read.finish(false);
                payload
            }
        };
        Ok(result)
    }

    async fn get_ranges(
        &self,
        path: &Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        let mut read = self.begin(path);
        read.counts().ranges.fetch_add(1, Ordering::Relaxed);
        // Preserve the backend's range coalescing and returned slices. The byte
        // count is the returned ranges' total, not hidden prefetch or wire bytes.
        let result = self.inner.get_ranges(path, ranges).await;
        if let Ok(parts) = &result {
            for part in parts {
                read.counts()
                    .bytes
                    .fetch_add(part.len() as u64, Ordering::Relaxed);
            }
        }
        read.finish(result.is_err());
        result
    }

    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(path, payload, opts).await
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, opts).await
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
mod tests;
