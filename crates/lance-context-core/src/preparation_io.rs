//! Diagnostics for immutable preparation IO, never write authorization.
use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt};
use lance_io::object_store::WrappingObjectStore;
use object_store::{path::Path, *};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Debug)]
pub(crate) struct ProgressWrapper(pub Arc<AtomicU64>);

impl WrappingObjectStore for ProgressWrapper {
    fn wrap(&self, _: &str, original: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(ProgressStore {
            inner: original,
            steps: self.0.clone(),
        })
    }
}

#[derive(Debug)]
struct ProgressStore {
    inner: Arc<dyn ObjectStore>,
    steps: Arc<AtomicU64>,
}

impl std::fmt::Display for ProgressStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PreparationProgress({})", self.inner)
    }
}

// Metadata/lease polling must not mask stalled data work. Match path segments
// so both bucket-relative and dataset-relative stores work.
fn data_file(path: &Path) -> bool {
    path.as_ref()
        .split('/')
        .any(|s| matches!(s, "data" | "_indices"))
}

#[async_trait]
impl ObjectStore for ProgressStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        let count = data_file(path) && payload.content_length() > 0;
        let result = self.inner.put_opts(path, payload, opts).await?;
        if count {
            self.steps.fetch_add(1, Ordering::Relaxed);
        }
        Ok(result)
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        let inner = self.inner.put_multipart_opts(path, opts).await?;
        if !data_file(path) {
            return Ok(inner);
        }
        Ok(Box::new(ProgressUpload {
            inner,
            steps: self.steps.clone(),
        }))
    }

    async fn get_opts(&self, path: &Path, opts: GetOptions) -> Result<GetResult> {
        let count = data_file(path) && !opts.head;
        let result = self.inner.get_opts(path, opts).await?;
        if !count {
            return Ok(result);
        }
        let meta = result.meta.clone();
        let range = result.range.clone();
        let attributes = result.attributes.clone();
        let steps = self.steps.clone();
        // Count body chunks after they arrive, not successful response headers.
        // into_stream also observes actual local file reads without buffering.
        let payload = GetResultPayload::Stream(
            result
                .into_stream()
                .inspect(move |chunk| {
                    if chunk.as_ref().is_ok_and(|b| !b.is_empty()) {
                        steps.fetch_add(1, Ordering::Relaxed);
                    }
                })
                .boxed(),
        );
        Ok(GetResult {
            payload,
            meta,
            range,
            attributes,
        })
    }

    async fn get_ranges(&self, path: &Path, ranges: &[std::ops::Range<u64>]) -> Result<Vec<Bytes>> {
        let result = self.inner.get_ranges(path, ranges).await?;
        if data_file(path) && result.iter().any(|b| !b.is_empty()) {
            self.steps.fetch_add(1, Ordering::Relaxed);
        }
        Ok(result)
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
struct ProgressUpload {
    inner: Box<dyn MultipartUpload>,
    steps: Arc<AtomicU64>,
}
#[async_trait]
impl MultipartUpload for ProgressUpload {
    fn put_part(&mut self, payload: PutPayload) -> UploadPart {
        let nonempty = payload.content_length() > 0;
        let part = self.inner.put_part(payload);
        let steps = self.steps.clone();
        Box::pin(async move {
            part.await?;
            if nonempty {
                steps.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        })
    }
    async fn complete(&mut self) -> Result<PutResult> {
        self.inner.complete().await
    }
    async fn abort(&mut self) -> Result<()> {
        self.inner.abort().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn counts_data_bodies_and_upload_parts_but_not_metadata_or_failures() {
        let steps = Arc::new(AtomicU64::new(0));
        let store = ProgressStore {
            inner: Arc::new(memory::InMemory::new()),
            steps: steps.clone(),
        };
        let data = Path::from("table/data/part.lance");
        let metadata = Path::from("table/_versions/1.manifest");
        store.put(&metadata, "metadata".into()).await.unwrap();
        store.get(&metadata).await.unwrap().bytes().await.unwrap();
        assert_eq!(steps.load(Ordering::Relaxed), 0);
        store.put(&data, "payload".into()).await.unwrap();
        let result = store.get(&data).await.unwrap();
        store.head(&data).await.unwrap();
        assert_eq!(
            steps.load(Ordering::Relaxed),
            1,
            "headers and HEAD are not data progress"
        );
        assert_eq!(result.bytes().await.unwrap(), "payload");
        assert_eq!(steps.load(Ordering::Relaxed), 2);
        assert!(store.get(&Path::from("table/data/missing")).await.is_err());
        assert!(store
            .put_opts(&data, "conflict".into(), PutOptions::from(PutMode::Create))
            .await
            .is_err());
        assert_eq!(steps.load(Ordering::Relaxed), 2);
        let mut upload = store
            .put_multipart(&Path::from("table/_indices/new/index.idx"))
            .await
            .unwrap();
        upload.put_part("index".into()).await.unwrap();
        upload.complete().await.unwrap();
        assert_eq!(steps.load(Ordering::Relaxed), 3);
        let ranges = store.get_ranges(&data, &[0..3, 3..7]).await.unwrap();
        assert_eq!(
            ranges,
            vec![Bytes::from_static(b"pay"), Bytes::from_static(b"load")]
        );
        assert_eq!(steps.load(Ordering::Relaxed), 4);
    }
}
