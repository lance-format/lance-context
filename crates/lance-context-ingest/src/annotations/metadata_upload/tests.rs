use super::*;
use object_store::{memory::InMemory, ObjectStoreExt};
use std::sync::{atomic::AtomicBool, Mutex};
use tokio::sync::Notify;

#[derive(Debug, Clone)]
struct StatusTransport(u16);

impl object_store::client::HttpConnector for StatusTransport {
    fn connect(
        &self,
        _: &object_store::ClientOptions,
    ) -> object_store::Result<object_store::client::HttpClient> {
        Ok(object_store::client::HttpClient::new(self.clone()))
    }
}

#[async_trait::async_trait]
impl object_store::client::HttpService for StatusTransport {
    async fn call(
        &self,
        _: object_store::client::HttpRequest,
    ) -> Result<object_store::client::HttpResponse, object_store::client::HttpError> {
        let mut response = object_store::client::HttpResponse::new(
            object_store::client::HttpResponseBody::from("injected response".to_owned()),
        );
        *response.status_mut() = self.0.try_into().unwrap();
        Ok(response)
    }
}

#[tokio::test]
async fn metadata_upload_classifies_real_sdk_status_without_matching_uri_text() -> Result<()> {
    for code in [429, 503, 403, 500] {
        let store = object_store::azure::MicrosoftAzureBuilder::new()
            .with_account("serverbusy503")
            .with_container_name("container")
            .with_skip_signature(true)
            .with_retry(object_store::RetryConfig {
                max_retries: 0,
                ..Default::default()
            })
            .with_http_connector(StatusTransport(code))
            .build()?;
        let error = store
            .put(
                &Path::from("503/serverbusy"),
                Bytes::from_static(b"x").into(),
            )
            .await
            .unwrap_err();
        assert_eq!(explicit_throttle(&error), matches!(code, 429 | 503));
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct FaultStore {
    inner: Arc<dyn ObjectStore>,
    pub(crate) blocked: AtomicBool,
    only_owned: Option<Path>,
    only_data: bool,
    after_write: bool,
    message: &'static str,
    pub(crate) calls: Mutex<Vec<(Path, Vec<Bytes>)>>,
    pub(crate) entered: Notify,
}

impl std::fmt::Display for FaultStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("metadata fault store")
    }
}

fn fixture(after_write: bool, message: &'static str) -> (Scope, Arc<UploadStore>, Arc<FaultStore>) {
    let inner = Arc::new(FaultStore {
        inner: Arc::new(InMemory::new()),
        blocked: AtomicBool::new(true),
        only_owned: None,
        only_data: false,
        after_write,
        message,
        calls: Mutex::new(Vec::new()),
        entered: Notify::new(),
    });
    let (lifetime, stopped) = watch::channel(());
    let counters = Arc::new(Counters::default());
    let enabled = Arc::new(AtomicBool::new(true));
    let scope = Scope {
        lifetime: Some(lifetime),
        enabled: enabled.clone(),
        counters: counters.clone(),
    };
    let store = Arc::new(UploadStore {
        inner: inner.clone(),
        base: Path::from("table"),
        stopped,
        enabled,
        counters,
    });
    (scope, store, inner)
}

fn data_file() -> Path {
    Path::from(format!(
        "table/data/000000000000000000000000{}.lance",
        &Uuid::new_v4().simple().to_string()[6..]
    ))
}

fn manifest() -> Path {
    Path::from(format!("table/_versions/2.manifest-{}", Uuid::new_v4()))
}

#[tokio::test]
async fn metadata_upload_reuses_original_buffers_and_path_after_503() -> Result<()> {
    for (after_write, path) in [
        (false, manifest()),
        (true, manifest()),
        (false, data_file()),
        (true, data_file()),
    ] {
        let (scope, store, fault) = fixture(
            after_write,
            "Server returned non-2xx status code: 503 Service Unavailable",
        );
        let parts = [Bytes::from(vec![1; 8192]), Bytes::from(vec![2; 4096])];
        let payload = parts.iter().cloned().collect::<PutPayload>();
        let uploading = {
            let store = store.clone();
            let path = path.clone();
            tokio::spawn(async move { store.put(&path, payload).await })
        };
        fault.entered.notified().await;
        assert_eq!(scope.observation().waiting_objects, 1);
        assert!(!uploading.is_finished());
        // A definite error can follow the remote write. Keeping the same
        // immutable object makes both pre-write and post-write failures safe.
        if after_write {
            assert_eq!(store.get(&path).await?.bytes().await?.len(), 12_288);
        }
        fault.blocked.store(false, Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(5), uploading).await???;
        assert_eq!(scope.observation().waiting_objects, 0);
        assert_eq!(scope.observation().put_resubmissions, 1);
        let calls = fault.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        for (observed_path, bytes) in calls.iter() {
            assert_eq!(observed_path, &path);
            for (actual, original) in bytes.iter().zip(&parts) {
                assert_eq!(actual, original);
                assert_eq!(actual.as_ptr(), original.as_ptr());
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn metadata_upload_owner_drop_cancels_wait_without_another_put() -> Result<()> {
    let (scope, store, fault) = fixture(
        false,
        "Server returned non-2xx status code: 503 Service Unavailable",
    );
    let uploading = tokio::spawn(async move {
        store
            .put(&data_file(), Bytes::from_static(b"manifest").into())
            .await
    });
    fault.entered.notified().await;
    assert_eq!(scope.observation().waiting_objects, 1);
    let counters = scope.counters.clone();
    drop(scope);
    assert!(tokio::time::timeout(Duration::from_secs(1), uploading)
        .await??
        .is_err());
    assert_eq!(counters.waiting.load(Ordering::Relaxed), 0);
    assert_eq!(fault.calls.lock().unwrap().len(), 1);
    Ok(())
}

#[tokio::test]
async fn metadata_upload_does_not_repeat_non_throttles_or_other_operations() -> Result<()> {
    for message in [
        "HTTP 403 Forbidden",
        "HTTP 500 Internal Server Error",
        "timeout",
        "Error at https://serverbusy.invalid/503 - Server returned non-2xx status code: 500 Internal Server Error",
    ] {
        let (_scope, store, fault) = fixture(false, message);
        assert!(store
            .put(&manifest(), Bytes::from_static(b"x").into())
            .await
            .is_err());
        assert_eq!(fault.calls.lock().unwrap().len(), 1);
    }
    let (_scope, store, fault) = fixture(
        false,
        "Server returned non-2xx status code: 503 Service Unavailable",
    );
    for (path, mode) in [
        (manifest(), PutMode::Create),
        (data_file(), PutMode::Create),
        (Path::from("table/data/body.lance"), PutMode::Overwrite),
        (Path::from("table/_versions/2.manifest"), PutMode::Overwrite),
        (
            Path::from("table-other/_indices/id/index.lance"),
            PutMode::Overwrite,
        ),
    ] {
        assert!(store
            .put_opts(
                &path,
                Bytes::from_static(b"x").into(),
                PutOptions {
                    mode,
                    ..Default::default()
                }
            )
            .await
            .is_err());
    }
    assert_eq!(fault.calls.lock().unwrap().len(), 5);
    Ok(())
}

#[test]
fn metadata_upload_only_holds_owned_immutable_paths() {
    let base = Path::from("table");
    assert!(immutable_file(&base, &manifest()));
    assert!(immutable_file(&base, &data_file()));
    assert!(immutable_file(
        &base,
        &Path::from(format!("table/_indices/{}/index.lance", Uuid::new_v4()))
    ));
    for path in [
        "table/_versions/2.manifest",
        "table/_latest.manifest",
        "table/_indices/current/index.lance",
        "table/data/data.lance",
        "table/data/nested/00000000000000000000000012345678901234567890123456.lance",
        "table/data/20000000000000000000000012345678901234567890123456.lance",
        "table/data/0000000000000000000000001234567890123456789012345g.lance",
        "table/data/00000000000000000000000012345678901234567890123456.lance-stage",
        "table-other/_versions/2.manifest",
        "table/_indices/",
    ] {
        assert!(!immutable_file(&base, &Path::from(path)));
    }
}

#[test]
fn metadata_upload_install_normalizes_the_uri_scheme() -> Result<()> {
    let storage = Storage {
        uri: "FILE-OBJECT-STORE:///tmp/owner-table".to_owned(),
        base: Path::from("tmp/owner-table"),
        options: Default::default(),
        session: Arc::new(lance::session::Session::default()),
    };
    let scope = Scope::install(&storage)?;
    assert!(!scope.enabled.load(Ordering::Acquire));
    Ok(())
}

#[async_trait::async_trait]
impl ObjectStore for FaultStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.calls
            .lock()
            .unwrap()
            .push((path.clone(), payload.iter().cloned().collect()));
        if self.blocked.load(Ordering::SeqCst)
            && self.only_owned.as_ref().is_none_or(|base| {
                immutable_file(base, path)
                    && (!self.only_data || path.as_ref().starts_with(&format!("{base}/data/")))
            })
        {
            if self.after_write {
                self.inner.put_opts(path, payload, opts).await?;
            }
            self.entered.notify_one();
            return Err(object_store::Error::Generic {
                store: "injected metadata transport",
                source: std::io::Error::other(self.message).into(),
            });
        }
        self.inner.put_opts(path, payload, opts).await
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
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
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
}

#[derive(Debug)]
struct FaultProvider {
    inner: Arc<dyn ObjectStoreProvider>,
    faults: Arc<FaultStore>,
}

#[async_trait::async_trait]
impl ObjectStoreProvider for FaultProvider {
    async fn new_store(
        &self,
        uri: url::Url,
        params: &ObjectStoreParams,
    ) -> lance::Result<LanceStore> {
        let mut store = self.inner.new_store(uri, params).await?;
        store.inner = self.faults.clone();
        Ok(store)
    }
}

pub(crate) async fn local_storage(
    path: &std::path::Path,
    only_data: bool,
) -> Result<(Storage, Scope, Arc<FaultStore>)> {
    let registry = Arc::new(lance_io::object_store::ObjectStoreRegistry::default());
    let faults = Arc::new(FaultStore {
        inner: Arc::new(object_store::local::LocalFileSystem::new()),
        blocked: AtomicBool::new(false),
        only_owned: Some(Path::from_absolute_path(path)?),
        only_data,
        after_write: false,
        message: "Server returned non-2xx status code: 503 Service Unavailable",
        calls: Mutex::new(Vec::new()),
        entered: Notify::new(),
    });
    registry.insert(
        "file-object-store",
        Arc::new(FaultProvider {
            inner: registry.get_provider("file-object-store").unwrap(),
            faults: faults.clone(),
        }),
    );
    let storage = Storage::resolve(
        format!("file-object-store://{}", path.display()),
        Default::default(),
        Arc::new(lance::session::Session::new(1 << 20, 1 << 20, registry)),
    )
    .await?;
    let scope = Scope::install(&storage)?;
    Ok((storage, scope, faults))
}
