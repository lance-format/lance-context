use super::*;
use crate::replication::source_read;

fn request(after: u64) -> Request {
    Request::ReadSource {
        group: 10,
        request: source_read::Request {
            table: batch(1).meta.stream,
            bucket: batch(1).meta.bucket,
            after,
            limit: 1,
        },
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_source_routes_over_tls_and_holds_bounded_reply_credit() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = RpcCluster::open(directory.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let original = batch(1);
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
    let index = accepted(&response);
    drop(response);
    let follower = (1..=3).find(|id| *id != leader).unwrap();
    assert!(
        matches!(c.client.request(follower, request(0)).await?.reply,
        Reply::Redirect { leader: observed } if observed == leader)
    );
    c.client.leaders.lock().unwrap().insert(10, follower);
    let available = c.client.budgets.control.available_permits();
    let data_available = c.client.budgets.data.available_permits();
    let response = c.client.request_routed(request(0)).await?;
    let Reply::Source(observed) = &response.reply else {
        anyhow::bail!("expected a typed committed source response");
    };
    assert_eq!(observed.group, 10);
    assert_eq!(observed.cluster, layout().cluster);
    assert!(observed.applied >= index);
    let source_read::Value::Page { page, closure } = &observed.value else {
        anyhow::bail!("expected a committed source page");
    };
    assert_eq!(page.entries, vec![(index, original.meta)]);
    assert!(closure.is_none());
    assert!(response.body.is_empty());
    assert!(c.client.budgets.control.available_permits() < available);
    assert_eq!(c.client.budgets.data.available_permits(), data_available);
    assert_eq!(
        c.net.nodes[&leader]
            .handle
            .observation(10)?
            .status
            .pending_reads,
        0
    );
    let future = observed.applied + 1;
    drop(response);
    assert_eq!(c.client.budgets.control.available_permits(), available);
    let response = c.client.request_routed(request(future)).await?;
    assert!(matches!(
        &response.reply,
        Reply::Source(source_read::Observation {
            value: source_read::Value::CursorAhead,
            ..
        })
    ));
    drop(response);
    assert_eq!(c.client.budgets.control.available_permits(), available);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_committed_source_wait_releases_client_credit() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = RpcCluster::open(directory.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let available = c.client.budgets.control.available_permits();
    let pool = &c.client.nodes[&leader].control;
    let slots = pool.slots.available_permits() as u32;
    let held = pool.slots.clone().acquire_many_owned(slots).await?;
    let client = c.client.clone();
    let task = tokio::spawn(async move { client.request(leader, request(0)).await });
    timeout(Duration::from_secs(5), async {
        while c.client.budgets.control.available_permits() == available {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    task.abort();
    assert!(matches!(task.await, Err(error) if error.is_cancelled()));
    assert_eq!(c.client.budgets.control.available_permits(), available);
    drop(held);
    let response = c.client.request(leader, request(0)).await?;
    assert!(matches!(response.reply, Reply::Source(_)));
    drop(response);
    assert_eq!(pool.slots.available_permits(), slots as usize);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tls_client_cannot_substitute_committed_source_for_fresh_quorum_or_the_reverse(
) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = RpcCluster::open(directory.path(), Options::default()).await?;
    let leader = c.leader().await?;
    for source_request in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let mut endpoints = c.endpoints.clone();
        endpoints.get_mut(&leader).unwrap().control_address = listener.local_addr()?.to_string();
        let mut server_tls = (*c.net.tls[&leader].server).clone();
        server_tls.alpn_protocols = vec![ALPN.to_vec()];
        let acceptor = TlsAcceptor::from(Arc::new(server_tls));
        let mut client_tls = (*c.net.tls[&4].client).clone();
        client_tls.alpn_protocols = vec![ALPN.to_vec()];
        let client = Client::new(
            &layout(),
            &Limits::default(),
            endpoints,
            Arc::new(client_tls),
            Options::default(),
        )?;
        let handle = c.net.nodes[&leader].handle.clone();
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await?;
            let mut stream = acceptor.accept(tcp).await?;
            let (id, meta, body) = prefix(&mut stream, 0).await?;
            assert_eq!(body, 0);
            let metadata = receive(&mut stream, meta).await?;
            let (observed, received): (Scope, WireRequest) = codec().deserialize(&metadata)?;
            assert_eq!(observed, scope(&handle));
            assert_eq!(
                matches!(received, WireRequest::ReadSource { .. }),
                source_request
            );
            // Return the other valid read type, from the actual Raft worker.
            let wrong = if source_request {
                WireRequest::Query {
                    group: 10,
                    query: query::Query::Accepted { after: 0, limit: 1 },
                }
            } else {
                WireRequest::ReadSource {
                    group: 10,
                    request: source_read::Request {
                        table: batch(1).meta.stream,
                        bucket: 0,
                        after: 0,
                        limit: 1,
                    },
                }
            };
            let (reply, payload) = execute(&handle, wrong, Bytes::new()).await?;
            assert!(matches!(
                reply,
                Reply::Query(query::Outcome::Observed { .. }) | Reply::Source(_)
            ));
            send(
                &mut stream,
                id,
                &codec().serialize(&(observed, reply))?,
                &payload,
            )
            .await?;
            Ok::<_, anyhow::Error>(())
        });
        let expected = if source_request {
            request(0)
        } else {
            Request::Query {
                group: 10,
                query: query::Query::Accepted { after: 0, limit: 1 },
            }
        };
        let result = client.request(leader, expected).await;
        timeout(Duration::from_secs(5), server).await???;
        let error = result.err().context("different read type accepted")?;
        assert!(format!("{error:#}").contains("unexpected response type"));
        assert_eq!(
            client.budgets.control.available_permits(),
            client.options.control_bytes
        );
    }
    c.shutdown().await
}
