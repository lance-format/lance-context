use super::*;

#[derive(Clone, Copy)]
enum AfterRedirect {
    Execute,
    Unknown,
    Disconnect,
    KeepRedirecting,
}

// One authenticated endpoint temporarily withholds the real leader's hint.
// The other two endpoints are actual followers. Successful requests execute
// against the actual RF3 group; the injector never invents a durable ACK.
async fn routed(append: bool, after: AfterRedirect) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let mut endpoints = c.endpoints.clone();
    let endpoint = endpoints.get_mut(&leader).unwrap();
    if append {
        endpoint.data_address = listener.local_addr()?.to_string();
    } else {
        endpoint.control_address = listener.local_addr()?.to_string();
    }
    let mut server_tls = (*c.net.tls[&leader].server).clone();
    server_tls.alpn_protocols = vec![ALPN.to_vec()];
    let acceptor = TlsAcceptor::from(Arc::new(server_tls));
    let mut client_tls = (*c.net.tls[&4].client).clone();
    client_tls.alpn_protocols = vec![ALPN.to_vec()];
    let deadline = Duration::from_millis(500);
    let client = Client::new(
        &layout(),
        &Limits::default(),
        endpoints,
        Arc::new(client_tls),
        Options {
            connections: 1,
            timeout: deadline,
            ..Options::default()
        },
    )?;
    client.leaders.lock().unwrap().insert(10, leader);
    let calls = Arc::new(AtomicU64::new(0));
    let observed = calls.clone();
    let handle = c.net.nodes[&leader].handle.clone();
    let expected_scope = scope(&handle);
    let original = batch(1);
    let received = original.clone();
    let mut server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await?;
        let mut stream = acceptor.accept(tcp).await?;
        loop {
            let (id, meta, body) = prefix(&mut stream, received.body.len()).await?;
            let metadata = receive(&mut stream, meta).await?;
            let (scope, request): (Scope, WireRequest) = codec().deserialize(&metadata)?;
            assert_eq!(scope, expected_scope);
            let body = receive(&mut stream, body).await?;
            if append {
                assert!(
                    matches!(&request, WireRequest::Append { group: 10, meta } if meta == &received.meta)
                );
                assert_eq!(body, received.body);
            } else {
                assert!(matches!(&request, WireRequest::Query { group: 10, .. }));
                assert!(body.is_empty());
            }
            let first = observed.fetch_add(1, Ordering::SeqCst) == 0;
            let (reply, body) = if first || matches!(after, AfterRedirect::KeepRedirecting) {
                (Reply::Redirect { leader: 0 }, Bytes::new())
            } else {
                match after {
                    AfterRedirect::Execute => execute(&handle, request, body).await?,
                    AfterRedirect::Unknown => (Reply::Unknown, Bytes::new()),
                    AfterRedirect::Disconnect => break,
                    AfterRedirect::KeepRedirecting => unreachable!(),
                }
            };
            send(
                &mut stream,
                id,
                &codec().serialize(&(&expected_scope, reply))?,
                &body,
            )
            .await?;
            if !first && !matches!(after, AfterRedirect::KeepRedirecting) {
                break;
            }
        }
        Ok::<_, anyhow::Error>(())
    });
    let request = if append {
        Request::Append {
            group: 10,
            batch: original.clone(),
        }
    } else {
        Request::Query {
            group: 10,
            query: query::Query::Producer {
                table: original.meta.stream,
                bucket: original.meta.bucket,
                producer: original.meta.producer,
            },
        }
    };
    let started = tokio::time::Instant::now();
    let result = client.request_routed(request).await;
    if matches!(after, AfterRedirect::KeepRedirecting) {
        server.abort();
        let _ = server.await;
        let error = result
            .err()
            .context("unavailable leader returned success")?;
        assert!(format!("{error:#}").contains("routed exchange timed out"));
        assert!(started.elapsed() >= deadline);
        assert!((2..=40).contains(&calls.load(Ordering::SeqCst)));
    } else {
        match timeout(Duration::from_secs(1), &mut server).await {
            Ok(done) => done??,
            Err(_) => {
                server.abort();
                let _ = server.await;
                anyhow::bail!("routing stopped before the second leader probe");
            }
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        match after {
            AfterRedirect::Execute if append => {
                let response = result?;
                let index = accepted(&response);
                drop(response);
                assert_eq!(producer(&c.client, leader, &original).await?, Some(index));
                let fetched = c
                    .client
                    .request(leader, Request::Fetch { group: 10, index })
                    .await?;
                assert_eq!(fetched.body, original.body);
                assert_eq!(
                    c.net.nodes[&leader]
                        .handle
                        .observation(10)?
                        .status
                        .accepted_records,
                    1
                );
            }
            AfterRedirect::Execute => {
                assert!(matches!(
                    result?.reply,
                    Reply::Query(query::Outcome::Observed { .. })
                ));
            }
            AfterRedirect::Unknown => assert!(matches!(result?.reply, Reply::Unknown)),
            AfterRedirect::Disconnect => assert!(result.is_err()),
            AfterRedirect::KeepRedirecting => unreachable!(),
        }
    }
    if !append || !matches!(after, AfterRedirect::Execute) {
        assert_eq!(producer(&c.client, leader, &original).await?, None);
    }
    assert_eq!(
        client.budgets.data.available_permits(),
        client.options.data_bytes
    );
    assert_eq!(
        client.budgets.control.available_permits(),
        client.options.control_bytes
    );
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exhausted_hints_refresh_for_queries_and_original_appends() -> Result<()> {
    routed(false, AfterRedirect::Execute).await?;
    routed(true, AfterRedirect::Execute).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refreshed_routing_stops_on_unknown_or_disconnected_mutations() -> Result<()> {
    routed(true, AfterRedirect::Unknown).await?;
    routed(true, AfterRedirect::Disconnect).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unavailable_leader_keeps_the_original_deadline_and_releases_credits() -> Result<()> {
    routed(false, AfterRedirect::KeepRedirecting).await
}
