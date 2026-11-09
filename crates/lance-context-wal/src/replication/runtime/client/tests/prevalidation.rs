use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rpc_prevalidation_rejects_bad_bytes_and_preserves_scope_and_exact_retries() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = RpcCluster::open(root.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let original = batch(1);
    let mut broken = original.clone();
    broken.meta.digest[0] ^= 1;
    let response = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: broken,
            },
        )
        .await?;
    assert!(matches!(response.reply, Reply::Unknown));
    drop(response);
    assert_eq!(producer(&c.client, leader, &original).await?, None);

    let mut foreign = original.clone();
    foreign.meta.bucket = 1;
    let response = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: foreign,
            },
        )
        .await?;
    assert!(matches!(response.reply, Reply::Unknown));
    drop(response);
    assert_eq!(producer(&c.client, leader, &original).await?, None);

    let response = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: original.clone(),
            },
        )
        .await?;
    let selected = accepted(&response);
    drop(response);
    let response = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: original.clone(),
            },
        )
        .await?;
    assert_eq!(accepted(&response), selected);
    drop(response);
    let follower = (1..=3).find(|id| *id != leader).unwrap();
    assert!(c.net.nodes[&follower]
        .handle
        .append(10, batch(2))
        .await?
        .await
        .is_err());
    c.shutdown().await?;
    Ok(())
}
