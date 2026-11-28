use super::*;
use lance_io::object_store::{
    providers::{ObjectStoreProvider, ObjectStoreRegistry},
    ChainedWrappingObjectStore, ObjectStore, ObjectStoreParams, StorageOptionsAccessor,
    StorageOptionsProvider,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Debug, Default)]
struct Credentials {
    generation: AtomicUsize,
    fetched: AtomicUsize,
}

#[async_trait::async_trait]
impl StorageOptionsProvider for Credentials {
    async fn fetch_storage_options(&self) -> lance::Result<Option<HashMap<String, String>>> {
        self.fetched.fetch_add(1, Ordering::SeqCst);
        Ok(Some(HashMap::from([
            (
                "test_credential".into(),
                self.generation.load(Ordering::SeqCst).to_string(),
            ),
            // Every store opening must renew; there are no wall-clock sleeps.
            ("expires_at_millis".into(), "0".into()),
        ])))
    }

    fn provider_id(&self) -> String {
        "single-body-renewable-test".into()
    }
}

#[derive(Debug)]
struct AuthorizedStore {
    inner: Arc<dyn ObjectStoreProvider>,
    expected: ObjectStoreParams,
    credentials: Arc<Credentials>,
    opened: Arc<Mutex<Vec<usize>>>,
}

#[async_trait::async_trait]
impl ObjectStoreProvider for AuthorizedStore {
    async fn new_store(
        &self,
        uri: url::Url,
        params: &ObjectStoreParams,
    ) -> lance::Result<ObjectStore> {
        // Lance applies its native base scoping before calling the provider.
        // The renewable origin and all other params must survive that step.
        assert_eq!(params, self.expected.scoped_to_base(None).as_ref());
        let accessor = params.storage_options_accessor.as_ref().unwrap();
        assert!(accessor.has_provider());
        let options = accessor.get_storage_options().await?;
        let generation = self.credentials.generation.load(Ordering::SeqCst);
        assert_eq!(options.0["test_credential"], generation.to_string());
        self.opened.lock().unwrap().push(generation);
        self.inner.new_store(uri, params).await
    }
}

async fn storage(
    uri: &str,
    params: &ObjectStoreParams,
    credentials: &Arc<Credentials>,
    opened: &Arc<Mutex<Vec<usize>>>,
) -> Result<Storage> {
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert(
        "file-object-store",
        Arc::new(AuthorizedStore {
            inner: registry.get_provider("file-object-store").unwrap(),
            expected: params.clone(),
            credentials: credentials.clone(),
            opened: opened.clone(),
        }),
    );
    let storage = Storage::resolve_with_params(
        uri.to_owned(),
        params.clone(),
        Arc::new(lance::session::Session::new(1 << 20, 1 << 20, registry)),
    )
    .await?;
    assert!(Arc::ptr_eq(
        storage.params.storage_options_accessor.as_ref().unwrap(),
        params.storage_options_accessor.as_ref().unwrap(),
    ));
    Ok(storage)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn renewable_params_survive_bootstrap_publication_and_recovery() -> Result<()> {
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let uri = format!(
        "file-object-store://{}/renewable.lance",
        root.path().display()
    );
    let credentials = Arc::new(Credentials::default());
    let params = ObjectStoreParams {
        block_size: Some(64 << 10),
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_provider(
            credentials.clone(),
        ))),
        object_store_wrapper: Some(Arc::new(ChainedWrappingObjectStore::new(vec![]))),
        use_constant_size_upload_parts: true,
        list_is_lexically_ordered: Some(true),
        ..Default::default()
    };
    let opened = Arc::new(Mutex::new(Vec::new()));
    credentials.generation.store(1, Ordering::SeqCst);
    let body = storage(&uri, &params, &credentials, &opened).await?;
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        body.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &fixture::limits(),
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let encode = |i| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i],
        )?)
    };
    let (service, runtime) = owner(&c, &body, &spec).await?;
    let (stop, task) = run_owner(runtime);
    let lane = producer(&service, 1).await?;
    let append = request(&service, &lane, 1, encode(0)?).await?;
    let receipt = service.submit(&append, DEADLINE).await?;
    drained(&c, &service, receipt.index).await?;
    stop.send(()).unwrap();
    task.await??;
    drop((service, body));

    // A new session cannot hide lost parameters behind the old store cache.
    // Recover and point-read the selected body before accepting more traffic.
    let fetched = credentials.fetched.load(Ordering::SeqCst);
    credentials.generation.store(2, Ordering::SeqCst);
    let body = storage(&uri, &params, &credentials, &opened).await?;
    let (service, runtime) = owner(&c, &body, &spec).await?;
    let (stop, task) = run_owner(runtime);
    let key = spec.row_key(&input[0], 0, 4096)?;
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    assert!(credentials.fetched.load(Ordering::SeqCst) > fetched);
    assert_eq!(service.submit(&append, DEADLINE).await?, receipt);
    let append = request(&service, &lane, 2, encode(2)?).await?;
    let receipt = service.submit(&append, DEADLINE).await?;
    drained(&c, &service, receipt.index).await?;
    let selected = service.selected_full_body(DEADLINE).await?;
    let dataset = annotation_publication::read(&c, &body, Some(selected.manifest.version)).await?;
    assert_eq!(dataset.count_rows(None).await?, 4);
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    stop.send(()).unwrap();
    task.await??;
    assert!(opened.lock().unwrap().contains(&1));
    assert!(opened.lock().unwrap().contains(&2));
    c.shutdown().await
}
