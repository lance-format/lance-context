use super::*;

fn binding() -> Request {
    Request::Query {
        group: 10,
        query: query::Query::PublisherBinding {
            table: uuid::Uuid::from_u128(123),
        },
    }
}

fn assert_deadline(error: &anyhow::Error, node: u64, stage: &str, routed: bool) {
    let message = format!("{error:#}");
    let kind = if routed { "routed" } else { "client" };
    assert!(
        message.contains(&format!("{kind} exchange timed out")),
        "{message}"
    );
    assert!(
        message.contains(&format!(
            "WAL RPC query.publisher_binding group=10 node={node}"
        )),
        "{message}"
    );
    assert!(message.contains(&format!("stage={stage}")), "{message}");
    assert!(message.contains("outcome unknown"), "{message}");
    assert!(error
        .downcast_ref::<tokio::time::error::Elapsed>()
        .is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn timeout_diagnostics_distinguish_byte_and_pool_admission() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open_with_options(
        dir.path(),
        Options::default(),
        Options {
            connections: 1,
            timeout: Duration::from_millis(300),
            ..Options::default()
        },
    )
    .await?;
    let leader = c.leader().await?;
    for routed in [false, true] {
        for (semaphore, stage) in [
            (&c.client.budgets.control, "admission"),
            (&c.client.nodes[&leader].control.slots, "pool_admission"),
        ] {
            let held = semaphore
                .clone()
                .acquire_many_owned(semaphore.available_permits() as u32)
                .await?;
            c.client.leaders.lock().unwrap().insert(10, leader);
            let result = if routed {
                c.client.request_routed(binding()).await
            } else {
                c.client.request(leader, binding()).await
            };
            drop(held);
            let error = result.err().context("blocked RPC unexpectedly completed")?;
            assert_deadline(&error, leader, stage, routed);
            assert_eq!(c.client.publisher_binding_timings()["send"].started, 0);
            assert_eq!(
                c.client.budgets.control.available_permits(),
                c.client.options.control_bytes
            );
            assert_eq!(c.client.nodes[&leader].control.slots.available_permits(), 1);
        }
    }
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn timeout_diagnostics_identify_held_tls_without_replaying() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let listener = Arc::new(TcpListener::bind("127.0.0.1:0").await?);
    let address = listener.local_addr()?.to_string();
    let mut endpoints = c.endpoints.clone();
    endpoints.get_mut(&leader).unwrap().control_address = address.clone();
    let mut tls = (*c.net.tls[&4].client).clone();
    tls.alpn_protocols = vec![ALPN.to_vec()];
    let client = Client::new(
        &layout(),
        &Limits::default(),
        endpoints,
        Arc::new(tls),
        Options {
            connections: 1,
            timeout: Duration::from_millis(300),
            ..Options::default()
        },
    )?;
    for routed in [false, true] {
        let listener = listener.clone();
        let (entered, arrived) = tokio::sync::oneshot::channel();
        let (release, resumed) = tokio::sync::oneshot::channel();
        // Accept TCP but deliberately withhold the TLS handshake. This peer
        // never receives an application request or simulates a durable result.
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await?;
            let _ = entered.send(());
            resumed.await?;
            drop(tcp);
            Ok::<_, anyhow::Error>(())
        });
        client.leaders.lock().unwrap().insert(10, leader);
        let request = async {
            if routed {
                client.request_routed(binding()).await
            } else {
                client.request(leader, binding()).await
            }
        };
        tokio::pin!(request);
        tokio::select! {
            result = &mut request => anyhow::bail!("RPC ended before TCP accept: {}", result.is_ok()),
            result = arrived => result?,
        }
        let error = request
            .await
            .err()
            .context("held TLS unexpectedly completed")?;
        release.send(()).unwrap();
        server.await??;
        assert_deadline(&error, leader, "tls_handshake", routed);
        assert!(!format!("{error:#}").contains(&address));
        assert_eq!(client.publisher_binding_timings()["send"].started, 0);
        assert_eq!(
            client.budgets.control.available_permits(),
            client.options.control_bytes
        );
        assert_eq!(client.nodes[&leader].control.slots.available_permits(), 1);
    }
    assert!(timeout(Duration::from_millis(30), listener.accept())
        .await
        .is_err());
    c.shutdown().await
}
