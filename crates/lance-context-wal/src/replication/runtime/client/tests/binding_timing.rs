use super::*;

fn binding_request() -> Request {
    Request::Query {
        group: 10,
        query: query::Query::PublisherBinding {
            table: uuid::Uuid::from_u128(123),
        },
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn publisher_binding_timings_separate_control_admission_and_connection() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(
        dir.path(),
        Options {
            connections: 1,
            ..Options::default()
        },
    )
    .await?;
    let leader = c.leader().await?;
    let appended = c
        .client
        .request_routed(Request::Append {
            group: 10,
            batch: batch(1),
        })
        .await?;
    let index = accepted(&appended);
    drop(appended);
    // Neither append nor another quorum query belongs to the binding counters.
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, Some(index));
    for phase in c.client.publisher_binding_timings().values() {
        assert_eq!(phase.started, 0);
    }
    let byte_permits = c.client.budgets.control.available_permits() as u32;
    let bytes = c
        .client
        .budgets
        .control
        .clone()
        .acquire_many_owned(byte_permits)
        .await?;
    let connection = c.client.nodes[&leader]
        .control
        .slots
        .clone()
        .acquire_owned()
        .await?;
    let client = c.client.clone();
    let request = client.request(leader, binding_request());
    tokio::pin!(request);
    assert!(timeout(Duration::from_millis(30), &mut request)
        .await
        .is_err());
    let blocked = c.client.publisher_binding_timings();
    assert_eq!(blocked["admission"].started, 1);
    assert_eq!(blocked["admission"].finished, 0);
    assert_eq!(blocked["connection"].started, 0);
    drop(bytes);

    assert!(timeout(Duration::from_millis(30), &mut request)
        .await
        .is_err());
    let blocked = c.client.publisher_binding_timings();
    assert_eq!(blocked["admission"].finished, 1);
    assert!(blocked["admission"].elapsed_micros >= 20_000);
    assert_eq!(blocked["connection"].started, 1);
    assert_eq!(blocked["connection"].finished, 0);
    assert_eq!(blocked["send"].started, 0);
    drop(connection);

    let response = request.await?;
    assert!(matches!(
        response.reply,
        Reply::Query(query::Outcome::Observed {
            value: query::Value::PublisherBinding(None),
            ..
        })
    ));
    drop(response);
    let observed = c.client.publisher_binding_timings();
    for phase in observed.values() {
        assert_eq!(phase.started, 1);
        assert_eq!(phase.finished, 1);
    }
    assert!(observed["connection"].elapsed_micros >= 20_000);
    for phase in c.client.append_timings().values() {
        assert_eq!(phase.started, 1);
        assert_eq!(phase.finished, 1);
    }
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn publisher_binding_response_timeout_finishes_timing() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open_with_options(
        dir.path(),
        Options::default(),
        Options {
            timeout: Duration::from_millis(300),
            ..Options::default()
        },
    )
    .await?;
    let leader = c.leader().await?;
    drop(c.client.request(leader, binding_request()).await?);
    let handle = &c.net.nodes[&leader].handle;
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
    let expired = c.client.request(leader, binding_request()).await;
    drop(release);
    assert!(expired.is_err());
    let response = &c.client.publisher_binding_timings()["response"];
    assert_eq!(response.started, 2);
    assert_eq!(response.finished, 2);
    assert!(response.elapsed_micros >= 200_000);
    assert_eq!(
        c.client.budgets.control.available_permits(),
        c.client.options.control_bytes
    );
    assert_eq!(c.client.append_timings()["response"].started, 0);
    c.shutdown().await
}

async fn receive_binding(stream: &mut Stream) -> Result<u64> {
    let (id, meta, body) = prefix(stream, 0).await?;
    assert_eq!(body, 0);
    let metadata = receive(stream, meta).await?;
    let (_, request): (Scope, WireRequest) = codec().deserialize(&metadata)?;
    assert!(matches!(
        request,
        WireRequest::Query {
            group: 10,
            query: query::Query::PublisherBinding { table },
        } if table == uuid::Uuid::from_u128(123)
    ));
    Ok(id)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn publisher_binding_timings_include_reconnect_handshake() -> Result<()> {
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
            connections: 1,
            ..Options::default()
        },
    )?;
    let expected = scope(&c.net.nodes[&leader].handle);
    let (entered, arrived) = tokio::sync::oneshot::channel();
    let (release, resumed) = tokio::sync::oneshot::channel();
    // This authenticated peer injects a disconnect and a held TLS handshake;
    // it returns only Unknown and does not simulate any durable storage result.
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await?;
        let mut stream = acceptor.accept(tcp).await?;
        let reply = codec().serialize(&(&expected, Reply::Unknown))?;
        let id = receive_binding(&mut stream).await?;
        send(&mut stream, id, &reply, &[]).await?;
        // Close only after the next request uses this pooled connection.
        receive_binding(&mut stream).await?;
        drop(stream);
        let (tcp, _) = listener.accept().await?;
        let _ = entered.send(());
        resumed.await?;
        let mut stream = acceptor.accept(tcp).await?;
        let id = receive_binding(&mut stream).await?;
        send(&mut stream, id, &reply, &[]).await?;
        Ok::<_, anyhow::Error>(())
    });
    drop(client.request(leader, binding_request()).await?);
    let request = client.request(leader, binding_request());
    tokio::pin!(request);
    tokio::select! {
        result = &mut request => anyhow::bail!("query finished before held handshake: {}", result.is_ok()),
        result = arrived => result?,
    }
    assert!(timeout(Duration::from_millis(30), &mut request)
        .await
        .is_err());
    let blocked = client.publisher_binding_timings();
    assert_eq!(blocked["admission"].finished, 2);
    assert_eq!(blocked["connection"].started, 3);
    assert_eq!(blocked["connection"].finished, 2);
    assert_eq!(blocked["send"].finished, 2);
    assert_eq!(blocked["response"].finished, 2);
    release.send(()).unwrap();
    let response = request.await?;
    assert!(matches!(response.reply, Reply::Unknown));
    drop(response);
    let observed = client.publisher_binding_timings();
    assert_eq!(observed["admission"].started, 2);
    for name in ["connection", "send", "response"] {
        assert_eq!(observed[name].started, 3);
        assert_eq!(observed[name].finished, 3);
    }
    assert!(observed["connection"].elapsed_micros >= 20_000);
    assert_eq!(client.append_timings()["response"].started, 0);
    assert_eq!(client.nodes[&leader].control.slots.available_permits(), 1);
    server.await??;
    c.shutdown().await
}
