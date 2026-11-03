use super::*;
use crate::{
    ingest_contract::{IngestContract, WritePolicy},
    replication::publication as p,
};
use uuid::Uuid;

fn binding(table: Uuid) -> Request {
    Request::Query {
        group: 10,
        query: query::Query::PublisherBinding { table },
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn maximum_binding_replies_share_credit_until_consumed() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(
        dir.path(),
        Options {
            control_bytes: 8 << 20,
            ..Options::default()
        },
    )
    .await?;
    let table = Uuid::new_v4();
    let destination = p::Destination {
        namespace: Uuid::new_v4(),
        dataset: "é".repeat(2048),
    };
    let ingest = IngestContract {
        tenant: "é".repeat(512),
        qualified_table: "é".repeat(512),
        schema: [7; 32],
        policy: WritePolicy::TurnIngestion,
        max_record_bytes: 4096,
        max_rows: 1,
    };
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Acquire(p::Acquire {
            table,
            destination: destination.clone(),
            ingest: ingest.clone(),
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: 0,
        }))))
        .await?;
    assert!(matches!(
        response.reply,
        Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Acquired(
            _
        ))))
    ));
    drop(response);
    let leader = c.leader().await?;
    let available = c.client.budgets.control.available_permits();
    let mut replies = Vec::new();
    // Keep responses alive: connection reuse must not release their byte credit.
    // A generic 4 MiB reservation cannot admit even two under this budget.
    let result = timeout(Duration::from_secs(5), async {
        for _ in 0..24 {
            let response = c.client.request(leader, binding(table)).await?;
            let Reply::Query(query::Outcome::Observed {
                cut,
                value: query::Value::PublisherBinding(Some(value)),
            }) = &response.reply
            else {
                anyhow::bail!("binding unavailable");
            };
            assert_eq!(value.destination, destination);
            assert_eq!(value.ingest, ingest);
            // Include the optional incarnation in the maximum wire-size proof.
            let mut maximum = value.clone();
            maximum.incarnation = Some(Uuid::new_v4());
            let metadata = codec().serialize(&(
                scope(&c.net.nodes[&leader].handle),
                Reply::Query(query::Outcome::Observed {
                    cut: cut.clone(),
                    value: query::Value::PublisherBinding(Some(maximum)),
                }),
            ))?;
            assert!(metadata.len() <= MAX_BINDING_META);
            assert!(response.body.is_empty());
            assert!(response._credit.num_permits() < 64 << 10);
            replies.push(response);
        }
        Ok::<_, anyhow::Error>(())
    })
    .await;
    let held = replies
        .iter()
        .map(|r| r._credit.num_permits())
        .sum::<usize>();
    assert_eq!(
        c.client.budgets.control.available_permits(),
        available - held
    );
    drop(replies);
    assert_eq!(c.client.budgets.control.available_permits(), available);
    result.context("retained binding replies exhausted generic reply credit")??;
    let missing = c.client.request(leader, binding(Uuid::new_v4())).await?;
    assert!(matches!(
        missing.reply,
        Reply::Query(query::Outcome::Observed {
            value: query::Value::PublisherBinding(None),
            ..
        })
    ));
    drop(missing);
    let follower = *c.net.nodes.keys().find(|&&id| id != leader).unwrap();
    let redirected = c.client.request(follower, binding(table)).await?;
    assert!(matches!(redirected.reply, Reply::Redirect { .. }));
    drop(redirected);
    assert_eq!(c.client.budgets.control.available_permits(), available);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn validated_binding_queries_release_excess_server_credit_before_execution() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let handle = c.net.nodes[&leader].handle.clone();
    let budgets = Arc::new(Budgets::new(&Options {
        control_bytes: 8 << 20,
        ..Options::default()
    }));
    let available = budgets.control.available_permits();
    // Hold actual query execution so admission cannot appear to succeed simply
    // by finishing each previous request. No mutation is proposed by this test.
    let (entered, arrived) = tokio::sync::oneshot::channel();
    let (release, resumed) = std::sync::mpsc::channel::<()>();
    handle.enqueue(
        handle.route(10)?,
        Work::Local(Box::new(move |_| {
            let _ = entered.send(());
            let _ = resumed.recv();
            Box::new(|| {})
        })),
        false,
    )?;
    arrived.await?;
    let metadata = codec().serialize(&(
        scope(&handle),
        WireRequest::Query {
            group: 10,
            query: query::Query::PublisherBinding {
                table: Uuid::new_v4(),
            },
        },
    ))?;
    let mut tasks = JoinSet::new();
    let mut clients = Vec::new();
    for id in 1..=24 {
        let (client, mut server) = tokio::io::duplex(1);
        clients.push(client);
        let handle = handle.clone();
        let budgets = budgets.clone();
        let meta_len = metadata.len();
        tasks.spawn(async move {
            serve_one(&mut server, false, &handle, &budgets, (id, meta_len, 0)).await
        });
    }
    let retained = 24 * (2 * metadata.len() + 2 * MAX_BINDING_META);
    // Admission can select any spawned handler first. Feed all bounded streams
    // concurrently: sequential writes could wait on a handler whose credit is
    // held by a different handler still waiting for its own metadata.
    let mut writers = JoinSet::new();
    for mut client in clients.drain(..) {
        let metadata = metadata.clone();
        writers.spawn(async move {
            client.write_all(&metadata).await?;
            Ok::<_, anyhow::Error>(client)
        });
    }
    let admitted = timeout(Duration::from_secs(3), async {
        while let Some(result) = writers.join_next().await {
            clients.push(result??);
        }
        while budgets.control.available_permits() != available - retained {
            tokio::task::yield_now().await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await;
    writers.abort_all();
    while writers.join_next().await.is_some() {}
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    let restored = budgets.control.available_permits();
    drop(clients);
    drop(release);
    c.shutdown().await?;
    admitted.context("binding queries retained generic server reply credit")??;
    assert_eq!(restored, available, "cancellation leaked control credit");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn oversized_binding_reply_is_rejected_before_receiving_metadata() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
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
        Options {
            timeout: Duration::from_secs(2),
            ..Options::default()
        },
    )?;
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await?;
        let mut stream = acceptor.accept(tcp).await?;
        let (id, meta_len, _) = prefix(&mut stream, 0).await?;
        receive(&mut stream, meta_len).await?;
        let mut fixed = [0; PREFIX];
        fixed[..8].copy_from_slice(MAGIC);
        fixed[8..16].copy_from_slice(&id.to_le_bytes());
        fixed[16..20].copy_from_slice(&((MAX_BINDING_META + 1) as u32).to_le_bytes());
        stream.write_all(&fixed).await?;
        stream.flush().await?;
        // Send no metadata: a receive attempt would wait until the RPC deadline.
        let mut byte = [0];
        let _ = stream.read(&mut byte).await;
        Ok::<_, anyhow::Error>(())
    });
    let result = client.request(leader, binding(Uuid::new_v4())).await;
    let error = result.err().context("oversized metadata was accepted")?;
    assert!(error
        .chain()
        .any(|e| e.to_string() == "client response metadata exceeds request bound"));
    assert_eq!(
        client.budgets.control.available_permits(),
        client.options.control_bytes
    );
    assert_eq!(
        client.nodes[&leader].control.slots.available_permits(),
        client.options.connections
    );
    server.await??;
    c.shutdown().await
}
