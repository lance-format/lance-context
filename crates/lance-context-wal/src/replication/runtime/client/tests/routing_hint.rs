use super::*;

// A real TLS endpoint withholds one mutation result from the actual RF3 group.
// It then executes a different producer's next request through the real leader.
// Unknown is neither a leadership change nor permission to replay a mutation.
async fn unknown_keeps_next_append_on_leader(execute_before_unknown: bool) -> Result<()> {
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
        Options::default(),
    )?;
    client.leaders.lock().unwrap().insert(10, leader);
    let original = batch(1);
    let subsequent = Batch::new(
        original.meta.stream,
        original.meta.bucket,
        uuid::Uuid::new_v4(),
        1,
        Bytes::from(vec![2; 1024 * 1024]),
    );
    let expected = [original.clone(), subsequent.clone()];
    let handle = c.net.nodes[&leader].handle.clone();
    let expected_scope = scope(&handle);
    let mut server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await?;
        let mut stream = acceptor.accept(tcp).await?;
        for (number, batch) in expected.into_iter().enumerate() {
            let (id, meta_bytes, body_bytes) = prefix(&mut stream, 1024 * 1024).await?;
            let metadata = receive(&mut stream, meta_bytes).await?;
            let (request_scope, request): (Scope, WireRequest) = codec().deserialize(&metadata)?;
            assert_eq!(request_scope, expected_scope);
            let body = receive(&mut stream, body_bytes).await?;
            assert!(
                matches!(&request, WireRequest::Append { group: 10, meta } if meta == &batch.meta)
            );
            assert_eq!(body, batch.body);
            let (reply, body) = if number == 0 {
                if execute_before_unknown {
                    let (reply, _) = execute(&handle, request, body).await?;
                    assert!(matches!(
                        reply,
                        Reply::Commit(Outcome::Committed(Applied::Accepted { .. }))
                    ));
                    (Reply::Commit(Outcome::Unknown), Bytes::new())
                } else {
                    (Reply::Unknown, Bytes::new())
                }
            } else {
                execute(&handle, request, body).await?
            };
            send(
                &mut stream,
                id,
                &codec().serialize(&(&expected_scope, reply))?,
                &body,
            )
            .await?;
        }
        Ok::<_, anyhow::Error>(())
    });
    let first = client
        .request_routed(Request::Append {
            group: 10,
            batch: original.clone(),
        })
        .await?;
    assert!(matches!(
        first.reply,
        Reply::Unknown | Reply::Commit(Outcome::Unknown)
    ));
    drop(first);
    let next = client
        .request_routed(Request::Append {
            group: 10,
            batch: subsequent.clone(),
        })
        .await?;
    let index = accepted(&next);
    drop(next);
    match timeout(Duration::from_secs(5), &mut server).await {
        Ok(done) => done??,
        Err(error) => {
            server.abort();
            let _ = server.await;
            return Err(error.into());
        }
    }
    assert_eq!(producer(&c.client, leader, &subsequent).await?, Some(index));
    assert_eq!(
        producer(&c.client, leader, &original).await?.is_some(),
        execute_before_unknown
    );
    let fetched = c
        .client
        .request(leader, Request::Fetch { group: 10, index })
        .await?;
    assert_eq!(fetched.body, subsequent.body);
    drop(fetched);
    let follower_appends: u64 = c
        .net
        .nodes
        .iter()
        .filter(|(id, _)| **id != leader)
        .map(|(_, node)| node.handle.append_observation()["started"])
        .sum();
    let accepted_records = c.net.nodes[&leader]
        .handle
        .observation(10)?
        .status
        .accepted_records;
    c.shutdown().await?;
    assert_eq!(accepted_records, if execute_before_unknown { 2 } else { 1 });
    assert_eq!(
        follower_appends, 0,
        "unknown mutation rerouted the next body through a follower"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_does_not_send_the_next_body_to_a_follower() -> Result<()> {
    unknown_keeps_next_append_on_leader(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_committed_append_keeps_its_identity_and_the_next_leader_hint() -> Result<()> {
    unknown_keeps_next_append_on_leader(true).await
}
