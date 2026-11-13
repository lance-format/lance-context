use super::*;
use crate::{
    ingest_contract::{IngestContract, WritePolicy},
    replication::publication as p,
};
use uuid::Uuid;

fn client(
    c: &RpcCluster,
    endpoints: BTreeMap<u64, Endpoint>,
    deadline: Duration,
) -> Result<Client> {
    let mut tls = (*c.net.tls[&4].client).clone();
    tls.alpn_protocols = vec![ALPN.to_vec()];
    Client::new(
        &layout(),
        &Limits::default(),
        endpoints,
        Arc::new(tls),
        Options {
            connections: 1,
            timeout: deadline,
            ..Options::default()
        },
    )
}

fn query() -> Request {
    let meta = batch(1).meta;
    Request::Query {
        group: 10,
        query: query::Query::Producer {
            table: meta.stream,
            bucket: meta.bucket,
            producer: meta.producer,
        },
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fetch_moves_past_a_refused_initial_or_cached_voter() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let original = batch(1);
    let response = c
        .client
        .request_routed(Request::Append {
            group: 10,
            batch: original.clone(),
        })
        .await?;
    let index = accepted(&response);
    drop(response);
    // Establish the exact retained record on each real replica before testing
    // transport routing. A local missing record is not a quorum absence proof.
    for node in 1..=3 {
        timeout(Duration::from_secs(10), async {
            loop {
                let response = c
                    .client
                    .request(node, Request::Fetch { group: 10, index })
                    .await?;
                if matches!(&response.reply, Reply::Record(Some(meta)) if meta == &original.meta) {
                    assert_eq!(response.body, original.body);
                    return Ok::<_, anyhow::Error>(());
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await??;
    }
    for (failed, cached) in [(1, false), (3, true)] {
        let unavailable = TcpListener::bind("127.0.0.1:0").await?;
        let mut endpoints = c.endpoints.clone();
        endpoints.get_mut(&failed).unwrap().data_address = unavailable.local_addr()?.to_string();
        let client = client(&c, endpoints, Duration::from_secs(1))?;
        if cached {
            client.leaders.lock().unwrap().insert(10, failed);
        }
        drop(unavailable);
        let response = client
            .request_routed(Request::Fetch { group: 10, index })
            .await?;
        assert!(matches!(&response.reply, Reply::Record(Some(meta)) if meta == &original.meta));
        assert_eq!(response.body, original.body);
        drop(response);
        assert_eq!(
            client.budgets.data.available_permits(),
            client.options.data_bytes
        );
    }
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unavailable_voters_share_the_original_fetch_deadline() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let mut endpoints = c.endpoints.clone();
    let mut unavailable = Vec::new();
    for endpoint in endpoints.values_mut() {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        endpoint.data_address = listener.local_addr()?.to_string();
        unavailable.push(listener);
    }
    let deadline = Duration::from_millis(200);
    let client = client(&c, endpoints, deadline)?;
    drop(unavailable);
    let started = tokio::time::Instant::now();
    let result = timeout(
        Duration::from_secs(2),
        client.request_routed(Request::Fetch {
            group: 10,
            index: 1,
        }),
    )
    .await?;
    let error = result.err().context("unavailable fetch returned success")?;
    assert!(error.to_string().contains("routed exchange timed out"));
    assert!(started.elapsed() >= deadline);
    assert_eq!(
        client.budgets.data.available_permits(),
        client.options.data_bytes
    );
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_moves_past_a_refused_cached_voter() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let failed = (1..=3).find(|id| *id != leader).unwrap();
    let unavailable = TcpListener::bind("127.0.0.1:0").await?;
    let mut endpoints = c.endpoints.clone();
    endpoints.get_mut(&failed).unwrap().control_address = unavailable.local_addr()?.to_string();
    let client = client(&c, endpoints, Duration::from_secs(1))?;
    client.leaders.lock().unwrap().insert(10, failed);
    drop(unavailable);

    let response = client.request_routed(query()).await?;
    let Reply::Query(query::Outcome::Observed {
        cut,
        value: query::Value::Producer(None),
    }) = &response.reply
    else {
        anyhow::bail!(
            "expected a real quorum observation, got {:?}",
            response.reply
        );
    };
    assert_eq!(cut.group, 10);
    assert!(cut.term > 0 && cut.quorum_index > 0 && cut.applied >= cut.quorum_index);
    assert_eq!(client.leaders.lock().unwrap().get(&10), Some(&leader));
    drop(response);
    assert_eq!(
        client.budgets.control.available_permits(),
        client.options.control_bytes
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refused_connections_do_not_replay_append_or_publication() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let failed = (1..=3).find(|id| *id != leader).unwrap();
    let unavailable = TcpListener::bind("127.0.0.1:0").await?;
    let mut endpoints = c.endpoints.clone();
    let endpoint = endpoints.get_mut(&failed).unwrap();
    endpoint.control_address = unavailable.local_addr()?.to_string();
    endpoint.data_address = endpoint.control_address.clone();
    let client = client(&c, endpoints, Duration::from_secs(1))?;
    drop(unavailable);
    let table = Uuid::new_v4();
    let acquire = Command::Acquire(p::Acquire {
        table,
        destination: p::Destination {
            namespace: Uuid::new_v4(),
            dataset: "table".into(),
        },
        ingest: IngestContract {
            tenant: "test".into(),
            qualified_table: "table".into(),
            schema: [7; 32],
            policy: WritePolicy::ImmutableUniqueIds,
            max_record_bytes: 4096,
            max_rows: 1,
        },
        owner: Uuid::new_v4(),
        request: Uuid::new_v4(),
        expected_epoch: 0,
    });
    for request in [
        Request::Append {
            group: 10,
            batch: batch(1),
        },
        Request::Commit(Box::new(acquire)),
    ] {
        client.leaders.lock().unwrap().insert(10, failed);
        let error = client
            .request_routed(request)
            .await
            .err()
            .context("mutation unexpectedly rerouted")?;
        assert_eq!(
            error.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::ConnectionRefused
        );
    }
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, None);
    let response = c
        .client
        .request_routed(Request::Query {
            group: 10,
            query: query::Query::Publisher { table },
        })
        .await?;
    assert!(matches!(
        response.reply,
        Reply::Query(query::Outcome::Observed {
            value: query::Value::Publisher(None),
            ..
        })
    ));
    assert_eq!(
        client.budgets.data.available_permits(),
        client.options.data_bytes
    );
    assert_eq!(
        client.budgets.control.available_permits(),
        client.options.control_bytes
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unavailable_voters_share_the_original_query_deadline() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let mut endpoints = c.endpoints.clone();
    let mut unavailable = Vec::new();
    for endpoint in endpoints.values_mut() {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        endpoint.control_address = listener.local_addr()?.to_string();
        unavailable.push(listener);
    }
    let deadline = Duration::from_millis(200);
    let client = client(&c, endpoints, deadline)?;
    drop(unavailable);
    let started = tokio::time::Instant::now();
    let result = timeout(Duration::from_secs(2), client.request_routed(query())).await?;
    let error = result
        .err()
        .context("unavailable quorum returned success")?;
    assert!(error.to_string().contains("routed exchange timed out"));
    assert!(started.elapsed() >= deadline);
    assert_eq!(
        client.budgets.control.available_permits(),
        client.options.control_bytes
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreign_query_scope_does_not_fall_back_to_healthy_voters() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let failed = (1..=3).find(|id| *id != leader).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let mut endpoints = c.endpoints.clone();
    endpoints.get_mut(&failed).unwrap().control_address = listener.local_addr()?.to_string();
    let client = client(&c, endpoints, Duration::from_secs(1))?;
    client.leaders.lock().unwrap().insert(10, failed);
    let mut tls = (*c.net.tls[&failed].server).clone();
    tls.alpn_protocols = vec![ALPN.to_vec()];
    let acceptor = TlsAcceptor::from(Arc::new(tls));
    let mut foreign = scope(&c.net.nodes[&failed].handle);
    foreign.node += 100;
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await?;
        let mut stream = acceptor.accept(tcp).await?;
        let (id, meta, body) = prefix(&mut stream, 0).await?;
        receive(&mut stream, meta + body).await?;
        send(
            &mut stream,
            id,
            &codec().serialize(&(foreign, Reply::Unknown))?,
            &[],
        )
        .await?;
        Ok::<_, anyhow::Error>(())
    });
    let error = client
        .request_routed(query())
        .await
        .err()
        .context("foreign scope was ignored")?;
    server.await??;
    assert!(format!("{error:#}").contains("foreign client response scope"));
    assert_eq!(
        client.budgets.control.available_permits(),
        client.options.control_bytes
    );
    Ok(())
}
