use super::*;

#[derive(Clone, Copy)]
enum Failure {
    Disconnect,
    Stall,
    Scope,
    Correlation,
    Unsolicited,
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
        if matches!(failure, Failure::Unsolicited) {
            stream.write_all(&[1]).await?;
            stream.flush().await?;
            assert!(timeout(Duration::from_millis(200), listener.accept())
                .await
                .is_err());
            return Ok::<_, anyhow::Error>(());
        }
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
            Failure::Unsolicited => unreachable!(),
        }
        Ok::<_, anyhow::Error>(())
    });
    drop(
        client
            .request(leader, Request::Status { group: 10 })
            .await?,
    );
    if matches!(
        failure,
        Failure::Disconnect | Failure::Stall | Failure::Unsolicited
    ) {
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
        Failure::Disconnect => {
            assert!(disconnected(&error), "{error:#}");
            assert!(error.to_string().contains(&format!(
                "WAL RPC status group=10 node={leader} reused=false stage=response_prefix"
            )));
        }
        Failure::Stall => assert!(
            error.to_string().contains("exchange timed out"),
            "{error:#}"
        ),
        Failure::Scope => assert!(
            format!("{error:#}").contains("foreign client response scope"),
            "{error:#}"
        ),
        Failure::Correlation => assert!(
            format!("{error:#}").contains("correlation mismatch"),
            "{error:#}"
        ),
        Failure::Unsolicited => {
            assert_eq!(
                error.to_string(),
                format!("WAL RPC status group=10 node={leader} stage=idle_connection")
            );
            assert!(
                format!("{error:#}")
                    .contains("unexpected application bytes on idle WAL connection"),
                "{error:#}"
            );
        }
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
async fn invalid_read_replies_are_not_retried() -> Result<()> {
    failed_read(Failure::Scope).await?;
    failed_read(Failure::Correlation).await?;
    failed_read(Failure::Unsolicited).await
}

// The peer closes only after receiving the complete append. Pool inspection
// must not turn a failure after dispatch into an automatic mutation replay.
async fn lost_append_response(warm: bool) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let mut endpoints = c.endpoints.clone();
    endpoints.get_mut(&leader).unwrap().data_address = listener.local_addr()?.to_string();
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
    let expected = scope(&c.net.nodes[&leader].handle);
    let original = batch(1);
    let received = original.clone();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await?;
        let mut stream = acceptor.accept(tcp).await?;
        if warm {
            let (id, meta, body) = prefix(&mut stream, 0).await?;
            receive(&mut stream, meta + body).await?;
            send(
                &mut stream,
                id,
                &codec().serialize(&(&expected, Reply::Unknown))?,
                &[],
            )
            .await?;
        }
        let (_, meta, body) = prefix(&mut stream, received.body.len()).await?;
        let metadata = receive(&mut stream, meta).await?;
        let (scope, request): (Scope, WireRequest) = codec().deserialize(&metadata)?;
        assert_eq!(scope, expected);
        assert!(
            matches!(request, WireRequest::Append { group: 10, meta } if meta == received.meta)
        );
        assert_eq!(receive(&mut stream, body).await?, received.body);
        // This is only a transport fault injector, not a durable ACK or a
        // claim about whether a real server committed the received mutation.
        drop(stream);
        assert!(
            timeout(Duration::from_millis(200), listener.accept())
                .await
                .is_err(),
            "mutation reconnected after dispatch"
        );
        Ok::<_, anyhow::Error>(())
    });
    if warm {
        drop(
            client
                .request(
                    leader,
                    Request::Fetch {
                        group: 10,
                        index: 1,
                    },
                )
                .await?,
        );
    }
    let error = client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: original,
            },
        )
        .await
        .err()
        .context("append unexpectedly received a reply")?;
    assert!(disconnected(&error), "{error:#}");
    assert!(error.to_string().contains(&format!(
        "WAL RPC append group=10 node={leader} reused={warm} stage=response_prefix; outcome unknown"
    )));
    assert!(client.nodes[&leader].data.idle.lock().unwrap().is_empty());
    assert_eq!(client.nodes[&leader].data.slots.available_permits(), 1);
    assert_eq!(
        client.budgets.data.available_permits(),
        client.options.data_bytes
    );
    server.await??;
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn append_response_loss_is_not_replayed_on_fresh_or_reused_connections() -> Result<()> {
    lost_append_response(false).await?;
    lost_append_response(true).await
}
