use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use std::sync::atomic::Ordering;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn materializer_unknown_confirm_finishes_original_publication() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), exercise(false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn materializer_unknown_confirm_cannot_bypass_owner_fencing() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), exercise(true)).await?
}

async fn exercise(fenced: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("confirm.lance");
    let mut cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let proxy = Proxy::open_confirm(&cluster, root.path(), true).await?;
    let direct = std::mem::replace(&mut cluster.client, proxy.client.clone());
    let ingress = connect(&cluster, binding()).await?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(32 << 20);
    let mut publisher = publisher(&ingress, &storage, &budget, 1).await?;
    let open = ProducerOpen {
        bucket: 0,
        slot: 0,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    let (body, _) = append(&ingress, &open, 1, &["turn"]).await?;
    let mut expected = BTreeMap::new();
    remember(&mut expected, body)?;
    publisher.prepare_next().await?.unwrap();
    let authority = publisher.authority().clone();
    let operation = tokio::spawn(async move {
        let result = publisher.commit_pending().await;
        (publisher, result)
    });
    tokio::time::timeout(Duration::from_secs(10), proxy.committed.notified()).await?;
    if fenced {
        cluster.client = direct.clone();
        let ingress = connect(&cluster, binding()).await?;
        let successor = super::publisher(&ingress, &storage, &budget, authority.epoch).await?;
        assert_ne!(successor.authority(), &authority);
    }
    proxy.release.notify_one();
    let (publisher, result) = operation.await?;
    if fenced {
        let error = result.unwrap_err();
        let context = format!("{error:#}");
        assert!(context.contains("outcome unknown"), "{context}");
        assert!(context.contains("publisher fenced"), "{context}");
    } else {
        let published = result?.unwrap();
        assert_eq!(publisher.state().await?.progress.published, Some(published));
        publisher.finish_pending().await?.unwrap();
        verify(&publisher, &expected, 2).await?;
    }
    assert_eq!(proxy.confirms.load(Ordering::SeqCst), 1);
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    drop(publisher);
    cluster.client = direct;
    drop(proxy);
    cluster.shutdown().await
}
