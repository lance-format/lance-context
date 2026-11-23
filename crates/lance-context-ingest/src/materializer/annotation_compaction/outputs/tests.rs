use super::*;
use bytes::Bytes;
use object_store::ObjectStoreExt;

#[tokio::test]
async fn output_budget_rejects_before_creating_an_unprotected_object() -> anyhow::Result<()> {
    let inner = Arc::new(object_store::memory::InMemory::new());
    let first = Path::from("table/data/first.lance");
    let second = Path::from("table/data/second.lance");
    let outputs = Outputs::new(Path::from("table"), 2 * first.as_ref().len() + 256);
    let store = Wrapper(outputs.clone()).wrap("memory", inner.clone());
    store
        .put(&first, Bytes::from_static(b"first").into())
        .await?;
    // Repeated calls for the same immutable path must not consume another slot.
    store
        .put(&first, Bytes::from_static(b"first").into())
        .await?;
    assert!(store
        .put(&second, Bytes::from_static(b"second").into())
        .await
        .is_err());
    assert!(matches!(
        inner.head(&second).await,
        Err(object_store::Error::NotFound { .. })
    ));
    let mut candidates = BTreeMap::from([(first, 5), (second.clone(), 6)]);
    outputs.protect(&mut candidates);
    assert_eq!(candidates, BTreeMap::from([(second, 6)]));
    Ok(())
}

#[tokio::test]
async fn multipart_output_is_protected_before_upload_completion() -> anyhow::Result<()> {
    let inner = Arc::new(object_store::memory::InMemory::new());
    let outputs = Outputs::new(Path::from("table"), 4096);
    let store = Wrapper(outputs.clone()).wrap("memory", inner.clone());
    let path = Path::from("table/data/multipart.lance");
    let mut upload = store.put_multipart(&path).await?;
    let mut candidates = BTreeMap::from([(path.clone(), 7)]);
    outputs.protect(&mut candidates);
    assert!(candidates.is_empty());
    upload
        .put_part(Bytes::from_static(b"payload").into())
        .await?;
    upload.complete().await?;
    assert_eq!(
        inner.get(&path).await?.bytes().await?,
        Bytes::from_static(b"payload")
    );
    let foreign = Path::from("table-other/data/foreign.lance");
    assert!(store
        .put(&foreign, Bytes::from_static(b"foreign").into())
        .await
        .is_err());
    assert!(matches!(
        inner.head(&foreign).await,
        Err(object_store::Error::NotFound { .. })
    ));
    Ok(())
}
