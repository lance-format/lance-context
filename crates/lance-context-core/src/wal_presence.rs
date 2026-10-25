//! Fresh, bounded advisory WAL presence. Never authorizes a commit or recovery.
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use lance::dataset::mem_wal::ShardManifestStore;
use lance_io::object_store::{
    ObjectStoreParams, ObjectStoreRegistry, ReadDirOptions, WrappingObjectStore,
};
use object_store::list::{PaginatedListOptions, PaginatedListResult, PaginatedListStore};
use object_store::{path::Path, *};

const MAX_SHARDS: usize = 256;
// Logical object-store calls, including each page. Provider-internal retries
// are not counted here; the caller also imposes a wall-clock budget.
const MAX_OPERATIONS: u64 = 512;
const LIST_PAGE_SIZE: usize = 64;
const MAX_LIST_PAGES: u64 = 8;
const MAX_OBJECT_BYTES: u64 = 1024 * 1024;
const MAX_READ_BYTES: u64 = 4 * 1024 * 1024;

/// Observe every persisted shard directory, including historical writers absent
/// from current worker configuration. Return true as soon as one shard has
/// sealed SSTs. Return false only after all discovered manifests were read.
///
/// No row counts, generation datasets, writer handles, ownership changes, or
/// empty-result caches. Capacity/storage errors are unknown demand, never empty;
/// callers must retain their ordinary admission/recovery path on error.
pub async fn has_pending_wal(uri: &str) -> lance::Result<bool> {
    // These are canonical main-branch storage URIs from the scheduler. Opening
    // only the object store avoids reading the base manifest/schema or listing
    // its version history merely to inspect WAL presence.
    let budget = Arc::new(Budget::default());
    let started = std::time::Instant::now();
    let (store, base) = lance_io::object_store::ObjectStore::from_uri_and_params(
        Arc::new(ObjectStoreRegistry::default()),
        uri,
        &ObjectStoreParams {
            object_store_wrapper: Some(budget.clone()),
            ..Default::default()
        },
    )
    .await?;
    let result = observe_wal(&store, &base).await;
    tracing::debug!(
        uri,
        elapsed_seconds = started.elapsed().as_secs_f64(),
        logical_operations = budget.requests.load(Ordering::Relaxed),
        listing_pages = budget.pages.load(Ordering::Relaxed),
        requested_body_bytes = budget.bytes.load(Ordering::Relaxed),
        outcome = match &result {
            Ok(true) => "pending",
            Ok(false) => "empty",
            Err(_) => "unknown",
        },
        "periodic WAL presence probe completed"
    );
    result
}

async fn observe_wal(
    store: &Arc<lance_io::object_store::ObjectStore>,
    base: &Path,
) -> lance::Result<bool> {
    let mut shards = Vec::new();
    let mut page_token = None;
    let mut entries = 0;
    loop {
        let page = store
            .read_dir_page(
                base.clone().join("_mem_wal"),
                ReadDirOptions {
                    page_token,
                    limit: Some(LIST_PAGE_SIZE),
                },
            )
            .await?;
        entries += page.result.common_prefixes.len() + page.result.objects.len();
        if entries > MAX_SHARDS {
            return Err(lance::Error::io("WAL presence shard budget exceeded"));
        }
        for prefix in page.result.common_prefixes {
            if let Some(id) = prefix
                .filename()
                .and_then(|name| uuid::Uuid::parse_str(name).ok())
            {
                shards.push(id);
            }
        }
        page_token = page.page_token;
        if page_token.is_none() {
            break;
        }
    }
    // No known shard is not evidence of a drained store. The URI might not
    // exist, or point at an uninitialized/moved table. Preserve admission.
    if shards.is_empty() {
        return Err(lance::Error::io("no persisted WAL shards found"));
    }
    for shard in shards {
        // One manifest at a time; one forward HEAD per scan step. A stale hint
        // can exhaust the request budget but cannot turn an error into empty.
        let manifest = ShardManifestStore::new(store.clone(), base, shard, 1)
            .refresh_latest()
            .await?;
        let Some(manifest) = manifest else {
            // A discovered directory with an unreadable/missing latest pointer
            // is not proof that historical generations were all reclaimed.
            return Err(lance::Error::io(
                "discovered WAL shard has no readable manifest",
            ));
        };
        if !manifest.sstables.is_empty() {
            return Ok(true);
        }
    }
    Ok(false)
}

#[derive(Clone, Debug, Default)]
struct Budget {
    requests: Arc<AtomicU64>,
    bytes: Arc<AtomicU64>,
    pages: Arc<AtomicU64>,
}

fn exhausted() -> Error {
    Error::Generic {
        store: "wal-presence-probe",
        source: std::io::Error::other("bounded read-only WAL presence probe exhausted").into(),
    }
}

fn reserve(counter: &AtomicU64, amount: u64, limit: u64) -> Result<()> {
    let mut used = counter.load(Ordering::Relaxed);
    loop {
        let next = used
            .checked_add(amount)
            .filter(|next| *next <= limit)
            .ok_or_else(exhausted)?;
        match counter.compare_exchange_weak(used, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Ok(()),
            Err(actual) => used = actual,
        }
    }
}

impl WrappingObjectStore for Budget {
    fn wrap_paginated(
        &self,
        _: &str,
        inner: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        Some(Arc::new(BoundedLister {
            inner,
            budget: self.clone(),
        }))
    }

    fn wrap(&self, prefix: &str, inner: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(BoundedStore {
            inner,
            budget: self.clone(),
            local: cfg!(unix) && prefix == "file",
        })
    }
}

#[derive(Debug)]
struct BoundedStore {
    inner: Arc<dyn ObjectStore>,
    budget: Budget,
    local: bool,
}

struct BoundedLister {
    inner: Arc<dyn PaginatedListStore>,
    budget: Budget,
}

#[async_trait]
impl PaginatedListStore for BoundedLister {
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        opts: PaginatedListOptions,
    ) -> Result<PaginatedListResult> {
        if opts.delimiter.as_deref() != Some("/")
            || !matches!(opts.max_keys, Some(1..=LIST_PAGE_SIZE))
            || opts.offset.is_some()
        {
            return Err(exhausted());
        }
        reserve(&self.budget.requests, 1, MAX_OPERATIONS)?;
        reserve(&self.budget.pages, 1, MAX_LIST_PAGES)?;
        // Pass the size to the provider: never aggregate a full remote LIST.
        let page = self.inner.list_paginated(prefix, opts).await?;
        if page.result.common_prefixes.len() + page.result.objects.len() > LIST_PAGE_SIZE {
            return Err(exhausted());
        }
        Ok(page)
    }
}

impl std::fmt::Display for BoundedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WalPresence({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for BoundedStore {
    async fn get_opts(&self, path: &Path, opts: GetOptions) -> Result<GetResult> {
        reserve(&self.budget.requests, 1, MAX_OPERATIONS)?;
        let head = opts.head;
        let response = self.inner.get_opts(path, opts).await?;
        if response.meta.size > MAX_OBJECT_BYTES {
            return Err(exhausted());
        }
        if !head {
            reserve(
                &self.budget.bytes,
                response.range.end.saturating_sub(response.range.start),
                MAX_READ_BYTES,
            )?;
        }
        Ok(response)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        reserve(&self.budget.requests, 1, MAX_OPERATIONS)?;
        // Providers without pushed-down pagination are unknown. In particular,
        // never call an unbounded remote list_with_delimiter then truncate it.
        if !self.local {
            return Err(exhausted());
        }
        let prefix = prefix.ok_or_else(exhausted)?;
        let local = local::LocalFileSystem::new().path_to_filesystem(prefix)?;
        let mut dir = tokio::fs::read_dir(local).await.map_err(|_| exhausted())?;
        let mut result = ListResult {
            common_prefixes: vec![],
            objects: vec![],
            extensions: Default::default(),
        };
        let mut entries = 0;
        while let Some(entry) = dir.next_entry().await.map_err(|_| exhausted())? {
            entries += 1;
            if entries > MAX_SHARDS {
                return Err(exhausted());
            }
            let kind = entry.file_type().await.map_err(|_| exhausted())?;
            // Symlinks could hide a shard outside the enumerated directory.
            if kind.is_symlink() {
                return Err(exhausted());
            }
            if kind.is_dir() {
                result
                    .common_prefixes
                    .push(Path::from_absolute_path(entry.path()).map_err(|_| exhausted())?);
            }
        }
        Ok(result)
    }

    fn list(&self, _: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        // Shard manifest refresh must use its pointers and bounded HEADs. A
        // recursive listing of WAL payloads is never an acceptable fallback.
        futures::stream::once(async { Err(exhausted()) }).boxed()
    }

    async fn put_opts(&self, _: &Path, _: PutPayload, _: PutOptions) -> Result<PutResult> {
        Err(exhausted())
    }
    async fn put_multipart_opts(
        &self,
        _: &Path,
        _: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        Err(exhausted())
    }
    fn delete_stream(
        &self,
        _: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        futures::stream::once(async { Err(exhausted()) }).boxed()
    }
    async fn copy_opts(&self, _: &Path, _: &Path, _: CopyOptions) -> Result<()> {
        Err(exhausted())
    }
    async fn rename_opts(&self, _: &Path, _: &Path, _: RenameOptions) -> Result<()> {
        Err(exhausted())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenericStore, GenericStoreOptions};
    use lance_context_api::schema_spec::{ColumnSpec, ColumnType, SchemaSpec};

    async fn writer(uri: &str, shard: &str) -> GenericStore {
        GenericStore::open(
            uri,
            SchemaSpec::new(vec![(
                "id".into(),
                ColumnSpec::required(ColumnType::String { large: false }),
            )]),
            GenericStoreOptions {
                shard_id: Some(shard.into()),
                seal_on_add: true,
                merge_after_generations: Some(0),
                merge_max_generations: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap()
    }

    async fn add(store: &GenericStore, id: &str) {
        store
            .add(&[serde_json::json!({"id":id}).as_object().unwrap().clone()])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn historical_shards_partial_passes_and_new_seals_are_never_hidden() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut old = writer(uri, "historical-no-current-worker").await;
        add(&old, "one").await;
        add(&old, "two").await;
        assert!(has_pending_wal(uri).await.unwrap());
        assert_eq!(old.cleanup_wal().await.unwrap(), 1);
        assert!(
            has_pending_wal(uri).await.unwrap(),
            "bounded pass must retain its tail"
        );
        assert_eq!(old.cleanup_wal().await.unwrap(), 1);
        assert!(!has_pending_wal(uri).await.unwrap());
        assert!(!has_pending_wal(uri).await.unwrap());
        let mut new = writer(uri, "new-unlisted-worker").await;
        add(&new, "new-seal").await;
        assert!(
            has_pending_wal(uri).await.unwrap(),
            "empty observations must not be cached"
        );
        assert_eq!(new.cleanup_wal().await.unwrap(), 1);
        assert!(!has_pending_wal(uri).await.unwrap());
        assert_eq!(new.list(None, None).await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn probe_never_opens_generation_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let store = writer(uri, "writer").await;
        add(&store, "sealed").await;
        // Remove the actual generation data files while retaining WAL manifests.
        // A row scan/count would fail; presence must still report pending work.
        fn remove_data(path: &std::path::Path) -> usize {
            let mut removed = 0;
            for entry in std::fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                if !path.is_dir() {
                    continue;
                }
                if path.file_name().unwrap() == "data" {
                    std::fs::remove_dir_all(&path).unwrap();
                    removed += 1;
                } else {
                    removed += remove_data(&path);
                }
            }
            removed
        }
        assert!(remove_data(&dir.path().join("_mem_wal")) > 0);
        assert!(has_pending_wal(uri).await.unwrap());
    }

    #[tokio::test]
    async fn missing_manifest_is_unknown_not_empty() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let store = writer(uri, "writer").await;
        add(&store, "sealed").await;
        let wal = dir.path().join("_mem_wal");
        for entry in std::fs::read_dir(&wal).unwrap() {
            let path = entry.unwrap().path().join("manifest");
            if path.is_dir() {
                std::fs::remove_dir_all(path).unwrap();
            }
        }
        assert!(has_pending_wal(uri).await.is_err());
    }

    #[test]
    fn concurrent_budget_reservations_never_exceed_the_limit() {
        let counter = Arc::new(AtomicU64::new(0));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let counter = counter.clone();
                std::thread::spawn(move || {
                    (0..100)
                        .filter(|_| reserve(&counter, 3, 512).is_ok())
                        .count()
                })
            })
            .collect();
        let successes: usize = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .sum();
        assert_eq!(successes, 170);
        assert_eq!(counter.load(Ordering::Relaxed), 510);
        let overflow = AtomicU64::new(u64::MAX);
        assert!(reserve(&overflow, 1, u64::MAX).is_err());
        assert_eq!(overflow.load(Ordering::Relaxed), u64::MAX);
    }

    #[tokio::test]
    async fn twenty_historical_shards_remain_visible_after_drain() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut writers = Vec::new();
        for i in 0..20 {
            let store = writer(uri, &format!("historical-{i}")).await;
            add(&store, &format!("row-{i}")).await;
            writers.push(store);
        }
        assert!(has_pending_wal(uri).await.unwrap());
        for store in &mut writers {
            assert_eq!(store.cleanup_wal().await.unwrap(), 1);
        }
        assert!(!has_pending_wal(uri).await.unwrap());
        add(&writers[0], "later-write").await;
        assert!(has_pending_wal(uri).await.unwrap());
    }

    #[tokio::test]
    async fn oversized_local_directory_and_missing_store_are_unknown() {
        let dir = tempfile::tempdir().unwrap();
        assert!(has_pending_wal(dir.path().to_str().unwrap()).await.is_err());
        let wal = dir.path().join("_mem_wal");
        std::fs::create_dir(&wal).unwrap();
        for _ in 0..=MAX_SHARDS {
            std::fs::create_dir(wal.join(uuid::Uuid::new_v4().to_string())).unwrap();
        }
        assert!(has_pending_wal(dir.path().to_str().unwrap()).await.is_err());
    }

    struct RecordingLister {
        calls: std::sync::Mutex<Vec<(Option<String>, Option<String>)>>,
    }

    #[async_trait]
    impl PaginatedListStore for RecordingLister {
        async fn list_paginated(
            &self,
            prefix: Option<&str>,
            opts: PaginatedListOptions,
        ) -> Result<PaginatedListResult> {
            assert_eq!(opts.delimiter.as_deref(), Some("/"));
            assert_eq!(opts.max_keys, Some(LIST_PAGE_SIZE));
            assert!(opts.offset.is_none());
            self.calls
                .lock()
                .unwrap()
                .push((prefix.map(str::to_owned), opts.page_token));
            // Empty pages with a continuation must not be treated as EOF; a
            // repeating backend token must eventually exhaust a page budget.
            Ok(PaginatedListResult {
                result: ListResult {
                    common_prefixes: vec![],
                    objects: vec![],
                    extensions: Default::default(),
                },
                page_token: Some("opaque-next-page".into()),
            })
        }
    }

    #[tokio::test]
    async fn azure_backend_receives_bounded_delimiter_pages_over_http() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let mut requests = Vec::new();
            for _ in 0..MAX_LIST_PAGES {
                let (mut socket, _) =
                    tokio::time::timeout(std::time::Duration::from_secs(5), listener.accept())
                        .await
                        .unwrap()
                        .unwrap();
                let mut request = Vec::new();
                loop {
                    let mut buf = [0u8; 1024];
                    let n = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        socket.read(&mut buf),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                    assert!(n > 0 && request.len() < 16384);
                    request.extend_from_slice(&buf[..n]);
                    if request.windows(4).any(|part| part == b"\r\n\r\n") {
                        break;
                    }
                }
                requests.push(String::from_utf8(request).unwrap());
                let body = "<EnumerationResults><Blobs/><NextMarker>next-page</NextMarker></EnumerationResults>";
                let response = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/xml\r\nConnection: close\r\n\r\n{}", body.len(), body);
                socket.write_all(response.as_bytes()).await.unwrap();
            }
            requests
        });
        let azure = object_store::azure::MicrosoftAzureBuilder::new()
            .with_account("test")
            .with_container_name("wal")
            .with_endpoint(endpoint)
            .with_skip_signature(true)
            .with_allow_http(true)
            .build()
            .unwrap();
        let lister = BoundedLister {
            inner: Arc::new(azure),
            budget: Budget::default(),
        };
        let mut token = None;
        let mut results = Vec::new();
        for _ in 0..=MAX_LIST_PAGES {
            let result = lister
                .list_paginated(
                    Some("root/_mem_wal/"),
                    PaginatedListOptions {
                        delimiter: Some("/".into()),
                        max_keys: Some(LIST_PAGE_SIZE),
                        page_token: token,
                        ..Default::default()
                    },
                )
                .await;
            token = result
                .as_ref()
                .ok()
                .and_then(|page| page.page_token.clone());
            results.push(result);
        }
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), MAX_LIST_PAGES as usize);
        for (i, request) in requests.iter().enumerate() {
            let line = request.lines().next().unwrap();
            assert!(line.starts_with("GET "), "{line}");
            assert!(line.contains("maxresults=64"), "{line}");
            assert!(line.contains("delimiter=%2F"), "{line}");
            assert!(line.contains("prefix=root%2F_mem_wal%2F"), "{line}");
            assert_eq!(line.contains("marker=next-page"), i > 0, "{line}");
            assert!(results[i].is_ok(), "{:?}", results[i]);
            assert_eq!(
                results[i].as_ref().unwrap().page_token.as_deref(),
                Some("next-page")
            );
        }
        assert!(results.last().unwrap().is_err());
    }

    #[tokio::test]
    async fn remote_page_size_tokens_and_budget_are_enforced_before_backend_calls() {
        let inner = Arc::new(RecordingLister {
            calls: Default::default(),
        });
        let lister = BoundedLister {
            inner: inner.clone(),
            budget: Budget::default(),
        };
        assert!(lister
            .list_paginated(Some("root/_mem_wal/"), Default::default())
            .await
            .is_err());
        assert!(inner.calls.lock().unwrap().is_empty());
        let mut token = None;
        for _ in 0..MAX_LIST_PAGES {
            token = lister
                .list_paginated(
                    Some("root/_mem_wal/"),
                    PaginatedListOptions {
                        delimiter: Some("/".into()),
                        max_keys: Some(LIST_PAGE_SIZE),
                        page_token: token,
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
                .page_token;
            assert!(token.is_some());
        }
        assert!(lister
            .list_paginated(
                Some("root/_mem_wal/"),
                PaginatedListOptions {
                    delimiter: Some("/".into()),
                    max_keys: Some(LIST_PAGE_SIZE),
                    page_token: token,
                    ..Default::default()
                }
            )
            .await
            .is_err());
        let calls = inner.calls.lock().unwrap();
        assert_eq!(calls.len(), MAX_LIST_PAGES as usize);
        assert_eq!(calls[0], (Some("root/_mem_wal/".into()), None));
        assert_eq!(calls[1].1.as_deref(), Some("opaque-next-page"));
    }

    #[tokio::test]
    async fn probe_enforces_capacity_and_cannot_mutate_storage() {
        let inner = Arc::new(memory::InMemory::new());
        let path = Path::from("large-manifest");
        inner
            .put(
                &path,
                bytes::Bytes::from(vec![0; MAX_OBJECT_BYTES as usize + 1]).into(),
            )
            .await
            .unwrap();
        let store = BoundedStore {
            inner: inner.clone(),
            budget: Budget::default(),
            local: false,
        };
        assert!(
            store.get(&path).await.is_err(),
            "oversized metadata is unknown before decoding"
        );
        assert!(store
            .put(&Path::from("must-not-exist"), bytes::Bytes::new().into())
            .await
            .is_err());
        assert!(inner.head(&Path::from("must-not-exist")).await.is_err());
        assert!(store.delete(&path).await.is_err());
        assert!(inner.head(&path).await.is_ok());
        store
            .budget
            .requests
            .store(MAX_OPERATIONS, Ordering::Relaxed);
        assert!(store.list_with_delimiter(None).await.is_err());
    }
}
