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
    let mut body = Box::pin(Body {
        inner: stream::iter([Ok(bytes.clone())])
            .chain(stream::pending())
            .boxed(),
        read: store.begin(&path),
    });
    assert_eq!(scope.observation()[0].bytes, 0);
    let chunk = body.next().await.unwrap()?;
    assert_eq!(chunk.as_ptr(), bytes.as_ptr());
    drop(body);
    let body = Body {
        inner: stream::iter([
            Ok(bytes.clone()),
            Err(object_store::Error::Generic {
                store: "test",
                source: std::io::Error::other("body read failed").into(),
            }),
        ])
        .boxed(),
        read: store.begin(&path),
    };
    assert!(body.try_collect::<Vec<_>>().await.is_err());
    drop(Body {
        inner: stream::pending().boxed(),
        read: store.begin(&path),
    });
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
    let _uploads = super::super::metadata_upload::Scope::install(&narrow)?;
    let body = Storage::resolve(
        format!("file-object-store://{}/body", dir.path().display()),
        Default::default(),
        session.clone(),
    )
    .await?;
    let registry = session.store_registry();
    let original_prefix =
        registry.calculate_object_store_prefix(&narrow.uri, Some(&narrow.options))?;
    let scope = Scope::install(&narrow, Some(&body))?;
    assert_eq!(
        registry.calculate_object_store_prefix(&narrow.uri, Some(&narrow.options))?,
        original_prefix
    );
    let params = ObjectStoreParams::default();
    let (first, _) = LanceStore::from_uri_and_params(registry, &narrow.uri, &params).await?;
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
