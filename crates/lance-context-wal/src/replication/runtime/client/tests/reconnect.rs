use super::*;

#[derive(Clone, Copy)]
enum Failure {
    Disconnect,
    Stall,
    Scope,
    Correlation,
}

// A real authenticated TLS peer returns one valid reply and then injects a
// transport or protocol failure. No storage state or durable ACK is simulated.
async fn failed_read(failure: Failure) -> Result<()> {
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
            timeout: Duration::from_millis(500),
            ..Options::default()
        },
    )?;
    let connections = Arc::new(AtomicU64::new(0));
    let observed = connections.clone();
    let expected = scope(&c.net.nodes[&leader].handle);
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await?;
        observed.fetch_add(1, Ordering::Relaxed);
        let mut stream = acceptor.accept(tcp).await?;
        let (id, meta, body) = prefix(&mut stream, 0).await?;
        receive(&mut stream, meta + body).await?;
        send(
            &mut stream,
            id,
            &codec().serialize(&(&expected, Reply::Unknown))?,
            &[],
        )
        .await?;
        if matches!(failure, Failure::Disconnect | Failure::Stall) {
            drop(stream);
            let (tcp, _) = listener.accept().await?;
            observed.fetch_add(1, Ordering::Relaxed);
            stream = acceptor.accept(tcp).await?;
        }
        let (id, meta, body) = prefix(&mut stream, 0).await?;
        receive(&mut stream, meta + body).await?;
        match failure {
            Failure::Disconnect => drop(stream),
            Failure::Stall => {
                // The caller's original deadline must still bound recovery.
                sleep(Duration::from_secs(60)).await;
                drop(stream);
            }
            Failure::Scope | Failure::Correlation => {
                let mut scope = expected;
                if matches!(failure, Failure::Scope) {
                    scope.node += 1;
                }
                let id = if matches!(failure, Failure::Correlation) {
                    id + 1
                } else {
                    id
                };
                send(
                    &mut stream,
                    id,
                    &codec().serialize(&(scope, Reply::Unknown))?,
                    &[],
                )
                .await?;
            }
        }
        Ok::<_, anyhow::Error>(())
    });
    drop(
        client
            .request(leader, Request::Status { group: 10 })
            .await?,
    );
    if matches!(failure, Failure::Disconnect | Failure::Stall) {
        wait_for_idle_close(&client.nodes[&leader].control).await?;
    }
    let error = match timeout(
        Duration::from_secs(5),
        client.request(leader, Request::Status { group: 10 }),
    )
    .await?
    {
        Ok(_) => anyhow::bail!("faulty peer unexpectedly returned a successful read"),
        Err(error) => error,
    };
    match failure {
        Failure::Disconnect => assert!(disconnected(&error), "{error:#}"),
        Failure::Stall => assert!(
            error.to_string().contains("exchange timed out"),
            "{error:#}"
        ),
        Failure::Scope => assert!(
            error.to_string().contains("foreign client response scope"),
            "{error:#}"
        ),
        Failure::Correlation => assert!(
            error.to_string().contains("correlation mismatch"),
            "{error:#}"
        ),
    }
    assert_eq!(
        connections.load(Ordering::Relaxed),
        if matches!(failure, Failure::Disconnect | Failure::Stall) {
            2
        } else {
            1
        }
    );
    assert!(client.nodes[&leader]
        .control
        .idle
        .lock()
        .unwrap()
        .is_empty());
    assert_eq!(client.nodes[&leader].control.slots.available_permits(), 1);
    assert_eq!(
        client.budgets.control.available_permits(),
        client.options.control_bytes
    );
    if matches!(failure, Failure::Stall) {
        server.abort();
        assert!(server.await.unwrap_err().is_cancelled());
    } else {
        server.await??;
    }
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fresh_read_disconnect_is_not_retried_and_releases_credits() -> Result<()> {
    failed_read(Failure::Disconnect).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stalled_read_reconnect_respects_deadline_and_releases_credits() -> Result<()> {
    failed_read(Failure::Stall).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn invalid_read_scope_and_correlation_are_not_retried() -> Result<()> {
    failed_read(Failure::Scope).await?;
    failed_read(Failure::Correlation).await
}
