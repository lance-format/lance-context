//! Protect a bounded set of immutable outputs before they can appear in LIST.
use futures::stream::BoxStream;
use lance_io::object_store::WrappingObjectStore;
use object_store::{
    path::Path, CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
};
use std::{
    collections::BTreeMap,
    ops::Range,
    sync::{Arc, Mutex},
};

#[derive(Debug)]
pub(super) struct Outputs {
    base: Path,
    limit: usize,
    paths: Mutex<(BTreeMap<Path, ()>, usize)>,
}

impl Outputs {
    pub(super) fn new(base: Path, limit: usize) -> Arc<Self> {
        Arc::new(Self {
            base,
            limit,
            paths: Mutex::new((BTreeMap::new(), 0)),
        })
    }

    pub(super) fn register(&self, path: &Path) -> object_store::Result<()> {
        let mut state = self.paths.lock().unwrap();
        if state.0.contains_key(path) {
            return Ok(());
        }
        let charge = path
            .as_ref()
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(256));
        let total = charge.and_then(|n| state.1.checked_add(n));
        if !path.prefix_matches(&self.base) || total.is_none_or(|n| n > self.limit) {
            return Err(object_store::Error::Generic {
                store: "compaction outputs",
                source: std::io::Error::other(
                    "compaction output path exceeds its destination or workspace",
                )
                .into(),
            });
        }
        state.0.insert(path.clone(), ());
        state.1 = total.unwrap();
        Ok(())
    }

    pub(super) fn protect(&self, candidates: &mut BTreeMap<Path, u64>) {
        // The candidate page is already closed. A future immutable output cannot
        // be in that page: registration precedes every operation that creates it.
        let state = self.paths.lock().unwrap();
        candidates.retain(|path, _| !state.0.contains_key(path));
    }
}

#[derive(Debug)]
pub(super) struct Wrapper(pub(super) Arc<Outputs>);

impl WrappingObjectStore for Wrapper {
    fn wrap(&self, _prefix: &str, inner: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(Store {
            inner,
            outputs: self.0.clone(),
        })
    }
}

#[derive(Debug)]
struct Store {
    inner: Arc<dyn ObjectStore>,
    outputs: Arc<Outputs>,
}

impl std::fmt::Display for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("compaction output protection")
    }
}

#[async_trait::async_trait]
impl ObjectStore for Store {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.outputs.register(path)?;
        self.inner.put_opts(path, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.outputs.register(path)?;
        self.inner.put_multipart_opts(path, opts).await
    }

    async fn get_opts(&self, path: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        self.inner.get_opts(path, options).await
    }

    async fn get_ranges(
        &self,
        path: &Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
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
        self.outputs.register(to)?;
        self.inner.copy_opts(from, to, opts).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        opts: RenameOptions,
    ) -> object_store::Result<()> {
        self.outputs.register(to)?;
        self.inner.rename_opts(from, to, opts).await
    }
}

#[cfg(test)]
mod tests;
