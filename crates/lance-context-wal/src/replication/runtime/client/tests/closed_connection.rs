use super::*;
use crate::replication::runtime::Work;

// Close the real listener's idle connections and reopen the same endpoint.
// This tests the pool's EOF handling without shortening the next request's
// durability deadline just to make an idle timeout arrive sooner.
pub(super) async fn close_idle(cluster: &mut RpcCluster, node: u64, data: bool) -> Result<()> {
    let index = node as usize - 1;
    cluster.servers.remove(index).shutdown().await?;
    let endpoint = &cluster.endpoints[&node];
    let data_listener = TcpListener::bind(&endpoint.data_address).await?;
    let control_listener = TcpListener::bind(&endpoint.control_address).await?;
    let mut config = (*cluster.net.tls[&node].server).clone();
    config.alpn_protocols = vec![ALPN.to_vec()];
    let server = Server::spawn(
        cluster.net.nodes[&node].handle.clone(),
        data_listener,
        control_listener,
        BTreeSet::from([cluster.net.tls[&4].certificate]),
        Arc::new(config),
        Options::default(),
    )?;
    cluster.servers.insert(index, server);
    let pools = &cluster.client.nodes[&node];
    wait_for_idle_close(if data { &pools.data } else { &pools.control }).await
}

pub(super) async fn delayed_append(
    cluster: &RpcCluster,
    leader: u64,
    batch: Batch,
    expected_index: u64,
) -> Result<Response> {
    let handle = &cluster.net.nodes[&leader].handle;
    let (send_store, receive_store) = tokio::sync::oneshot::channel();
    handle.enqueue(
        handle.route(10)?,
        Work::Local(Box::new(move |node| {
            let store = node.groups[&10].raw.store().durable_log().clone();
            Box::new(move || {
                let _ = send_store.send(store);
            })
        })),
        false,
    )?;
    let store = receive_store.await?;
    let blocking_store = store.clone();
    let (entered, arrived) = tokio::sync::oneshot::channel();
    let (release, resumed) = std::sync::mpsc::channel::<()>();
    let blocked = tokio::task::spawn_blocking(move || {
        let _gate = blocking_store.block_sync_for_test();
        let _ = entered.send(());
        let _ = resumed.recv();
    });
    arrived.await?;
    let client = cluster.client.clone();
    let append = tokio::spawn(async move {
        client
            .request(leader, Request::Append { group: 10, batch })
            .await
    });
    let reached_sync = timeout(Duration::from_secs(5), async {
        while store.appended_index_for_test() != Some(expected_index) {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
    // Hold the real durability boundary longer than the old fixture's 500ms
    // connection timeout. Closing an idle socket must not impose that budget
    // on a new append or let it ACK before its own sync completes.
    sleep(Duration::from_millis(750)).await;
    let completed_before_sync = append.is_finished();
    drop(release);
    blocked.await?;
    let result = append.await?;
    reached_sync.context("append did not reach the real storage sync")?;
    let response = result.context("fresh append failed while its real sync was held")?;
    ensure!(
        !completed_before_sync,
        "append acknowledged before its sync"
    );
    Ok(response)
}
