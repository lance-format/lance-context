use super::*;
use futures::{stream, StreamExt, TryStreamExt};
use object_store::{memory::InMemory, ObjectStoreExt};

fn fixture(inner: Arc<dyn ObjectStore>) -> (Scope, ReadStore) {
    let counts = Arc::new(std::array::from_fn(|_| Counts::default()));
    (
        Scope(counts.clone()),
        ReadStore {
            inner,
            targets: vec![
                Target {
                    store: String::new(),
                    base: "narrow/".into(),
                    data_class: 0,
                },
                Target {
                    store: String::new(),
                    base: "body/".into(),
                    data_class: 1,
                },
            ],
            counts,
        },
    )
}

#[tokio::test]
async fn timing_does_not_poll_a_body_until_the_caller_requests_it() -> Result<()> {
    use std::sync::atomic::AtomicUsize;
    let (scope, store) = fixture(Arc::new(InMemory::new()));
    let polls = Arc::new(AtomicUsize::new(0));
    let observed = polls.clone();
    let bytes = Bytes::from_static(b"same allocation");
    let expected = bytes.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut gated = Box::pin(stream::once(async move {
        rx.await.map_err(|_| object_store::Error::Generic {
            store: "test",
            source: std::io::Error::other("sender dropped").into(),
        })
    }));
    let inner = stream::poll_fn(move |cx| {
        observed.fetch_add(1, Ordering::Relaxed);
        gated.as_mut().poll_next(cx)
    });
    let mut body = Body::new(
        inner.boxed(),
        store.begin(&Path::from("narrow/data/gated.lance")),
    );
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    assert!(futures::poll!(body.next()).is_pending());
    assert_eq!(polls.load(Ordering::Relaxed), 1);
    assert_eq!(scope.observation()[0].body_pending.samples, 0);
    tx.send(bytes).unwrap();
    let chunk = body.next().await.unwrap()?;
    assert_eq!(chunk.as_ptr(), expected.as_ptr());
    assert!(body.next().await.is_none());
    // EOF records once even if the caller retains the exhausted stream.
    let before_drop = serde_json::to_value(scope.observation())?;
    assert_eq!(scope.observation()[0].body_pending.samples, 1);
    drop(body);
    assert_eq!(serde_json::to_value(scope.observation())?, before_drop);
    Ok(())
}

#[tokio::test]
async fn read_io_preserves_ranges_conditions_and_shared_buffers() -> Result<()> {
    let inner = Arc::new(InMemory::new());
    let bytes = Bytes::from_static(b"abcdefghijklmnop");
    let path = Path::from("narrow/data/test.lance");
    let put = inner.put(&path, bytes.clone().into()).await?;
    let (scope, store) = fixture(inner);
    let result = store
        .get_opts(
            &path,
            GetOptions {
                range: Some((2..7).into()),
                if_match: put.e_tag.clone(),
                ..Default::default()
            },
        )
        .await?;
    assert_eq!(result.range, 2..7);
    assert_eq!(result.meta.e_tag, put.e_tag);
    let parts: Vec<_> = result.into_stream().try_collect().await?;
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0], bytes.slice(2..7));
    assert_eq!(parts[0].as_ptr(), bytes.as_ptr().wrapping_add(2));
    let ranges = [8..12, 0..3, 2..5];
    let parts = store.get_ranges(&path, &ranges).await?;
    for (part, range) in parts.iter().zip(&ranges) {
        assert_eq!(part, &bytes.slice(range.start as usize..range.end as usize));
        assert_eq!(
            part.as_ptr(),
            bytes.as_ptr().wrapping_add(range.start as usize)
        );
    }
    store.head(&path).await?;
    assert!(store
        .get_opts(
            &path,
            GetOptions {
                if_match: Some("wrong".into()),
                ..Default::default()
            }
        )
        .await
        .is_err());
    let rows = scope.observation();
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.class, "narrow_data");
    assert_eq!(
        (row.started, row.completed, row.failed, row.dropped),
        (4, 3, 1, 0)
    );
    assert_eq!(
        (row.bytes, row.heads, row.ranges, row.file_payloads),
        (15, 1, 1, 0)
    );
    Ok(())
}

#[tokio::test]
async fn read_io_observes_partial_failed_and_unpolled_bodies_without_read_ahead() -> Result<()> {
    let (scope, store) = fixture(Arc::new(InMemory::new()));
    let bytes = Bytes::from_static(b"shared");
    let path = Path::from("body/data/test.lance");
    let mut body = Box::pin(Body::new(
        stream::iter([Ok(bytes.clone())])
            .chain(stream::pending())
            .boxed(),
        store.begin(&path),
    ));
    assert_eq!(scope.observation()[0].bytes, 0);
    let chunk = body.next().await.unwrap()?;
    assert_eq!(chunk.as_ptr(), bytes.as_ptr());
    drop(body);
    let body = Body::new(
        stream::iter([
            Ok(bytes.clone()),
            Err(object_store::Error::Generic {
                store: "test",
                source: std::io::Error::other("body read failed").into(),
            }),
        ])
        .boxed(),
        store.begin(&path),
    );
    assert!(body.try_collect::<Vec<_>>().await.is_err());
    drop(Body::new(stream::pending().boxed(), store.begin(&path)));
    let rows = scope.observation();
    let row = &rows[0];
    assert_eq!(row.class, "body_data");
    assert_eq!(
        (
            row.started,
            row.completed,
            row.failed,
            row.dropped,
            row.bytes
        ),
        (3, 0, 1, 2, 12)
    );
    assert_eq!(row.response.samples, 3);
    assert_eq!(row.body_unpolled.samples, 3);
    assert_eq!(row.body_pending.samples, 3);
    assert_eq!(row.body_poll.samples, 3);
    Ok(())
}

#[tokio::test]
async fn read_io_keeps_local_file_payload_and_path_boundaries() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let inner = Arc::new(object_store::local::LocalFileSystem::new_with_prefix(
        dir.path(),
    )?);
    let path = Path::from("narrow/data/test.lance");
    inner.put(&path, Bytes::from_static(b"file").into()).await?;
    let (scope, store) = fixture(inner);
    let result = store.get(&path).await?;
    assert!(matches!(&result.payload, GetResultPayload::File(..)));
    assert_eq!(result.bytes().await?, Bytes::from_static(b"file"));
    let row = &scope.observation()[0];
    assert_eq!((row.completed, row.bytes, row.file_payloads), (1, 0, 1));
    assert_eq!(row.response.samples, 1);
    assert_eq!(row.body_pending.samples, 0);
    for (path, expected) in [
        ("narrow-other/data/test.lance", 4),
        ("body/_indices/id/index.lance", 2),
        ("narrow/_versions/1.manifest", 3),
        ("body/_transactions/1.txn", 3),
    ] {
        let read = store.begin(&Path::from(path));
        assert_eq!(read.class, expected);
    }
    Ok(())
}

#[tokio::test]
async fn read_io_native_tables_retain_shared_store_and_classify_both_data_paths() -> Result<()> {
    use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
    use lance::{
        dataset::{builder::DatasetBuilder, WriteParams},
        Dataset,
    };
    let dir = tempfile::tempdir()?;
    let session = Arc::new(lance::session::Session::default());
    let narrow = Storage::resolve(
        format!("file-object-store://{}/narrow", dir.path().display()),
        Default::default(),
        session.clone(),
    )
    .await?;
    let body = Storage::resolve(
        format!("file-object-store://{}/body", dir.path().display()),
        Default::default(),
        session.clone(),
    )
    .await?;
    let registry = session.store_registry();
    let original_prefix =
        registry.calculate_object_store_prefix(&narrow.uri, narrow.params.storage_options())?;
    let identities = Providers::capture(&narrow, Some(&body.uri))?;
    let scope = Scope::install(&narrow, Some(&body), identities)?;
    assert_eq!(
        registry.calculate_object_store_prefix(&narrow.uri, narrow.params.storage_options())?,
        original_prefix
    );
    let _uploads = super::super::metadata_upload::Scope::install(&narrow)?;
    let params = ObjectStoreParams::default();
    let (first, _) =
        LanceStore::from_uri_and_params(registry.clone(), &narrow.uri, &params).await?;
    let (second, _) = LanceStore::from_uri_and_params(registry, &body.uri, &params).await?;
    assert!(
        Arc::ptr_eq(&first, &second),
        "diagnostics split the shared store"
    );
    let batch = RecordBatch::try_from_iter([(
        "value",
        Arc::new(Int64Array::from_iter_values(0..1024)) as Arc<dyn arrow_array::Array>,
    )])?;
    for target in [&narrow, &body] {
        let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        Dataset::write(
            reader,
            &target.uri,
            Some(WriteParams {
                session: Some(session.clone()),
                ..Default::default()
            }),
        )
        .await?;
        let dataset = DatasetBuilder::from_uri(&target.uri)
            .with_session(session.clone())
            .load()
            .await?;
        let rows: Vec<RecordBatch> = dataset
            .scan()
            .try_into_stream()
            .await?
            .try_collect()
            .await?;
        assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1024);
    }
    let rows = scope.observation();
    for class in ["narrow_data", "body_data", "metadata"] {
        assert!(
            rows.iter().any(|r| r.class == class && r.started > 0),
            "missing {class}"
        );
    }
    let serialized = serde_json::to_string(&rows)?;
    assert!(!serialized.contains(dir.path().to_str().unwrap()));
    assert!(!serialized.contains("file-object-store"));
    Ok(())
}

#[derive(Debug)]
struct AccountProvider(Arc<dyn ObjectStoreProvider>);

#[async_trait::async_trait]
impl ObjectStoreProvider for AccountProvider {
    async fn new_store(
        &self,
        uri: url::Url,
        params: &ObjectStoreParams,
    ) -> lance::Result<LanceStore> {
        self.0.new_store(uri, params).await
    }
    fn calculate_object_store_prefix(
        &self,
        _: &url::Url,
        options: Option<&HashMap<String, String>>,
    ) -> lance::Result<String> {
        Ok(format!(
            "memory${}",
            options.unwrap().get("test_account").unwrap()
        ))
    }
}

#[tokio::test]
async fn read_io_distinguishes_equal_paths_in_separate_backend_accounts() -> Result<()> {
    use lance_io::object_store::ObjectStoreRegistry;
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert(
        "memory",
        Arc::new(AccountProvider(registry.get_provider("memory").unwrap())),
    );
    let session = Arc::new(lance::session::Session::new(
        1 << 20,
        1 << 20,
        registry.clone(),
    ));
    let narrow = Storage::resolve(
        "memory://bucket/table".into(),
        HashMap::from([("test_account".into(), "a".into())]),
        session.clone(),
    )
    .await?;
    let body = Storage::resolve(
        narrow.uri.clone(),
        HashMap::from([("test_account".into(), "b".into())]),
        session,
    )
    .await?;
    let identities = Providers::capture(&narrow, Some(&body.uri))?;
    // This wrapper has the default prefix implementation. The diagnostic
    // provider beneath it must still match each backend's original identity.
    let _uploads = super::super::metadata_upload::Scope::install(&narrow)?;
    let before = registry.get_provider("memory").unwrap();
    let scope = Scope::install(&narrow, Some(&body), identities)?;
    let after = registry.get_provider("memory").unwrap();
    let uri = lance_io::object_store::uri_to_url(&narrow.uri)?;
    assert_eq!(before.extract_path(&uri)?, after.extract_path(&uri)?);
    assert_eq!(
        before.calculate_object_store_prefix(&uri, narrow.params.storage_options())?,
        after.calculate_object_store_prefix(&uri, narrow.params.storage_options())?
    );
    for storage in [&narrow, &body] {
        let (store, _) =
            LanceStore::from_uri_and_params(registry.clone(), &storage.uri, &storage.params)
                .await?;
        let path = Path::from("table/data/file.lance");
        store
            .inner
            .put(&path, Bytes::from_static(b"data").into())
            .await?;
        assert_eq!(
            store.inner.get(&path).await?.bytes().await?,
            Bytes::from_static(b"data")
        );
    }
    let rows = scope.observation();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].class, "narrow_data");
    assert_eq!(rows[1].class, "body_data");
    for row in rows {
        assert_eq!((row.started, row.completed, row.bytes), (1, 1, 4));
    }
    Ok(())
}
