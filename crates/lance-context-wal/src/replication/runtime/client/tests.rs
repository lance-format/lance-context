use super::*;
use crate::replication::{
    runtime::peer::tests::Cluster,
    tests::{batch, layout},
    Applied,
};
use tokio::time::sleep;

mod deadline;
mod reconnect;

struct RpcCluster {
    net: Cluster,
    servers: Vec<Server>,
    client: Arc<Client>,
    endpoints: BTreeMap<u64, Endpoint>,
}
impl RpcCluster {
    async fn open(root: &std::path::Path, options: Options) -> Result<Self> {
        Self::open_with_options(root, options.clone(), options).await
    }
    async fn open_with_options(
        root: &std::path::Path,
        server_options: Options,
        client_options: Options,
    ) -> Result<Self> {
        let net = Cluster::open(root).await?;
        let mut endpoints = BTreeMap::new();
        let mut servers = Vec::new();
        for id in 1..=3 {
            let data = TcpListener::bind("127.0.0.1:0").await?;
            let control = TcpListener::bind("127.0.0.1:0").await?;
            endpoints.insert(
                id,
                Endpoint {
                    data_address: data.local_addr()?.to_string(),
                    control_address: control.local_addr()?.to_string(),
                    server_name: net.tls[&id].name.clone(),
                    certificate: net.tls[&id].certificate,
                },
            );
            let mut config = (*net.tls[&id].server).clone();
            config.alpn_protocols = vec![ALPN.to_vec()];
            servers.push(Server::spawn(
                net.nodes[&id].handle.clone(),
                data,
                control,
                BTreeSet::from([net.tls[&4].certificate]),
                Arc::new(config),
                server_options.clone(),
            )?);
        }
        let mut config = (*net.tls[&4].client).clone();
        config.alpn_protocols = vec![ALPN.to_vec()];
        let client = Client::new(
            &layout(),
            &Limits::default(),
            endpoints.clone(),
            Arc::new(config),
            client_options,
        )?;
        net.nodes[&1].handle.campaign(10)?.await?;
        net.leader(10).await?;
        Ok(Self {
            net,
            servers,
            client: Arc::new(client),
            endpoints,
        })
    }
    async fn leader(&self) -> Result<u64> {
        self.net.leader(10).await
    }
    async fn raw(&self, node: u64, data: bool) -> Result<Stream> {
        let endpoint = &self.endpoints[&node];
        let tcp = TcpStream::connect(if data {
            &endpoint.data_address
        } else {
            &endpoint.control_address
        })
        .await?;
        tcp.set_nodelay(true)?;
        Ok(self
            .client
            .connector
            .connect(ServerName::try_from(endpoint.server_name.clone())?, tcp)
            .await?)
    }
    async fn shutdown(self) -> Result<()> {
        for server in self.servers {
            server.shutdown().await?;
        }
        self.net.shutdown().await
    }
}
fn accepted(response: &Response) -> u64 {
    match response.reply {
        Reply::Commit(Outcome::Committed(Applied::Accepted { index })) => index,
        ref other => panic!("expected durable append, got {other:?}"),
    }
}
async fn producer(client: &Client, leader: u64, batch: &Batch) -> Result<Option<u64>> {
    let response = client
        .request(
            leader,
            Request::Query {
                group: 10,
                query: query::Query::Producer {
                    table: batch.meta.stream,
                    bucket: batch.meta.bucket,
                    producer: batch.meta.producer,
                },
            },
        )
        .await?;
    match &response.reply {
        Reply::Query(query::Outcome::Observed {
            value: query::Value::Producer(value),
            ..
        }) => Ok(value.as_ref().map(|(_, index)| *index)),
        other => anyhow::bail!("expected producer quorum query, got {other:?}"),
    }
}

// Wait for the real TLS listener to close its idle connection. Readiness does
// not consume the EOF: the next request must encounter and handle it itself.
async fn wait_for_idle_close(pool: &Pool) -> Result<()> {
    let stream = pool.idle.lock().unwrap().pop().context("no idle stream")?;
    timeout(Duration::from_secs(5), stream.get_ref().0.readable()).await??;
    pool.idle.lock().unwrap().push(stream);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quorum_read_recovers_after_server_closes_idle_tls_connection() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open_with_options(
        dir.path(),
        Options {
            timeout: Duration::from_millis(500),
            ..Options::default()
        },
        Options::default(),
    )
    .await?;
    let leader = c.leader().await?;
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, None);
    wait_for_idle_close(&c.client.nodes[&leader].control).await?;
    let observed = producer(&c.client, leader, &batch(1)).await;
    assert_eq!(observed?, None);
    wait_for_idle_close(&c.client.nodes[&leader].control).await?;
    let response = c
        .client
        .request_routed(Request::Query {
            group: 10,
            query: query::Query::Producer {
                table: batch(1).meta.stream,
                bucket: batch(1).meta.bucket,
                producer: batch(1).meta.producer,
            },
        })
        .await?;
    assert!(matches!(
        response.reply,
        Reply::Query(query::Outcome::Observed { .. })
    ));
    drop(response);
    wait_for_idle_close(&c.client.nodes[&leader].control).await?;
    let response = c
        .client
        .request(leader, Request::Status { group: 10 })
        .await?;
    assert!(matches!(response.reply, Reply::Status(_)));
    drop(response);
    assert_eq!(
        c.client.budgets.control.available_permits(),
        c.client.options.control_bytes
    );
    wait_for_idle_close(&c.client.nodes[&leader].control).await?;
    let command = Command::Acquire(crate::replication::publication::Acquire {
        table: batch(1).meta.stream,
        destination: crate::replication::publication::Destination {
            namespace: uuid::Uuid::from_u128(101),
            dataset: "canonical/table".into(),
        },
        ingest: crate::ingest_contract::IngestContract {
            tenant: "tenant".into(),
            qualified_table: "table".into(),
            schema: [3; 32],
            policy: crate::ingest_contract::WritePolicy::ImmutableUniqueIds,
            max_record_bytes: 1024 * 1024,
            max_rows: 1024,
        },
        owner: uuid::Uuid::from_u128(100),
        request: uuid::Uuid::from_u128(102),
        expected_epoch: 0,
    });
    assert!(c
        .client
        .request(leader, Request::Commit(Box::new(command)))
        .await
        .is_err());
    assert!(c.client.nodes[&leader]
        .control
        .idle
        .lock()
        .unwrap()
        .is_empty());
    assert_eq!(
        c.client.budgets.control.available_permits(),
        c.client.options.control_bytes
    );
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn closed_data_connection_recovers_fetch_but_does_not_replay_append() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open_with_options(
        dir.path(),
        Options {
            timeout: Duration::from_millis(500),
            ..Options::default()
        },
        Options::default(),
    )
    .await?;
    let leader = c.leader().await?;
    let first = batch(1);
    let response = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: first.clone(),
            },
        )
        .await?;
    let index = accepted(&response);
    drop(response);
    wait_for_idle_close(&c.client.nodes[&leader].data).await?;
    let response = c
        .client
        .request(leader, Request::Fetch { group: 10, index })
        .await?;
    assert_eq!(response.body, first.body);
    assert!(matches!(&response.reply, Reply::Record(Some(meta)) if meta == &first.meta));
    assert!(c.client.budgets.data.available_permits() < c.client.options.data_bytes);
    drop(response);
    wait_for_idle_close(&c.client.nodes[&leader].data).await?;
    let second = batch(2);
    assert!(c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: second.clone()
            }
        )
        .await
        .is_err());
    assert_eq!(producer(&c.client, leader, &first).await?, Some(index));
    assert!(c.client.nodes[&leader].data.idle.lock().unwrap().is_empty());
    assert_eq!(
        c.client.budgets.data.available_permits(),
        c.client.options.data_bytes
    );
    // The caller explicitly retries the exact original identity after reconciling.
    let response = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: second,
            },
        )
        .await?;
    assert!(accepted(&response) > index);
    drop(response);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lost_response_retries_original_bytes_after_full_three_node_restart() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let base = batch(1);
    let original = Batch::new(
        base.meta.stream,
        0,
        base.meta.producer,
        1,
        Bytes::from(vec![0xa7; 2 << 20]),
    );
    let mut stream = c.raw(leader, true).await?;
    let request = WireRequest::Append {
        group: 10,
        meta: original.meta.clone(),
    };
    let metadata = codec().serialize(&(scope(&c.net.nodes[&leader].handle), request))?;
    send(&mut stream, 1, &metadata, &original.body).await?;
    // Do not consume the reply. Observe the commit through an independent
    // quorum read, then disconnect with the application outcome still unknown.
    let index = timeout(Duration::from_secs(10), async {
        loop {
            if let Some(index) = producer(&c.client, leader, &original).await? {
                return Ok::<_, anyhow::Error>(index);
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await??;
    drop(stream);
    let retry = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: original.clone(),
            },
        )
        .await?;
    assert_eq!(accepted(&retry), index);
    drop(retry);
    let fetched = c
        .client
        .request(leader, Request::Fetch { group: 10, index })
        .await?;
    assert!(matches!(&fetched.reply, Reply::Record(Some(meta)) if meta == &original.meta));
    assert_eq!(fetched.body, original.body);
    drop(fetched);
    c.shutdown().await?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let retry = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: original.clone(),
            },
        )
        .await?;
    assert_eq!(accepted(&retry), index);
    drop(retry);
    for id in 1..=3 {
        let fetched = timeout(Duration::from_secs(10), async {
            loop {
                let value = c
                    .client
                    .request(id, Request::Fetch { group: 10, index })
                    .await?;
                if matches!(value.reply, Reply::Record(Some(_))) {
                    return Ok::<_, anyhow::Error>(value);
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await??;
        assert_eq!(fetched.body, original.body);
    }
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn data_connection_saturation_preserves_control_and_rejects_oversized_frames() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(
        dir.path(),
        Options {
            connections: 1,
            server_connections: 1,
            ..Options::default()
        },
    )
    .await?;
    let leader = c.leader().await?;
    let mut stalled = c.raw(leader, true).await?;
    let response = c
        .client
        .request(
            leader,
            Request::Query {
                group: 10,
                query: query::Query::Accepted {
                    after: 0,
                    limit: 16,
                },
            },
        )
        .await?;
    assert!(matches!(
        response.reply,
        Reply::Query(query::Outcome::Observed { .. })
    ));
    drop(response);
    assert!(c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: batch(1)
            }
        )
        .await
        .is_err());
    let mut fixed = [0u8; PREFIX];
    fixed[..8].copy_from_slice(MAGIC);
    fixed[8..16].copy_from_slice(&1u64.to_le_bytes());
    fixed[16..20].copy_from_slice(&((MAX_META + 1) as u32).to_le_bytes());
    stalled.write_all(&fixed).await?;
    stalled.flush().await?;
    let mut byte = [0];
    let closed = timeout(Duration::from_secs(5), stalled.read(&mut byte)).await?;
    assert!(matches!(closed, Ok(0) | Err(_)));
    drop(stalled);
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, None);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn configured_tls_identity_and_scope_are_required_before_runtime_admission() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    // Trusted by the CA but not in the dedicated client service allowlist.
    let mut config = (*c.net.tls[&1].client).clone();
    config.alpn_protocols = vec![ALPN.to_vec()];
    let foreign = Client::new(
        &layout(),
        &Limits::default(),
        c.endpoints.clone(),
        Arc::new(config),
        Options::default(),
    )?;
    assert!(foreign
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: batch(1)
            }
        )
        .await
        .is_err());
    let mut stream = c.raw(leader, true).await?;
    let mut wrong = scope(&c.net.nodes[&leader].handle);
    wrong.cluster = uuid::Uuid::new_v4();
    let original = batch(1);
    let metadata = codec().serialize(&(
        wrong,
        WireRequest::Append {
            group: 10,
            meta: original.meta.clone(),
        },
    ))?;
    send(&mut stream, 1, &metadata, &original.body).await?;
    assert!(timeout(Duration::from_secs(5), prefix(&mut stream, 0))
        .await?
        .is_err());
    assert_eq!(producer(&c.client, leader, &original).await?, None);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_quorum_cannot_return_a_durable_append_ack() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    for id in (1..=3).filter(|id| *id != leader) {
        c.net.stop(id).await?;
    }
    let result = timeout(
        Duration::from_millis(500),
        c.client.request(
            leader,
            Request::Append {
                group: 10,
                batch: batch(1),
            },
        ),
    )
    .await;
    assert!(!matches!(
        result,
        Ok(Ok(Response {
            reply: Reply::Commit(Outcome::Committed(Applied::Accepted { .. })),
            ..
        }))
    ));
    c.shutdown().await?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let response = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: batch(1),
            },
        )
        .await?;
    assert!(accepted(&response) > 0);
    drop(response);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn routing_uses_only_configured_voters_and_durable_responses_hold_credit() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let follower = (1..=3).find(|id| *id != leader).unwrap();
    c.client.leaders.lock().unwrap().insert(10, follower);
    let original = batch(1);
    let result = c
        .client
        .request_routed(Request::Append {
            group: 10,
            batch: original.clone(),
        })
        .await?;
    let index = accepted(&result);
    drop(result);
    assert_eq!(c.client.leaders.lock().unwrap().get(&10), Some(&leader));
    let value = c
        .client
        .request_routed(Request::Fetch { group: 10, index })
        .await?;
    assert_eq!(value.body, original.body);
    assert!(c.client.budgets.data.available_permits() < c.client.options.data_bytes);
    assert_eq!(producer(&c.client, leader, &original).await?, Some(index));
    drop(value);
    assert_eq!(
        c.client.budgets.data.available_permits(),
        c.client.options.data_bytes
    );
    c.shutdown().await
}

#[cfg(feature = "lance-integration")]
mod lance;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn independent_ingress_pools_share_server_capacity_without_request_mixups() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let mut tasks = JoinSet::new();
    for _ in 0..4 {
        let mut config = (*c.net.tls[&4].client).clone();
        config.alpn_protocols = vec![ALPN.to_vec()];
        let client = Arc::new(Client::new(
            &layout(),
            &Limits::default(),
            c.endpoints.clone(),
            Arc::new(config),
            Options::default(),
        )?);
        for _ in 0..4 {
            let client = client.clone();
            let id = uuid::Uuid::new_v4();
            tasks.spawn(async move {
                let body = Bytes::from(vec![id.as_bytes()[0]; 256 << 10]);
                let original = Batch::new(uuid::Uuid::from_u128(2), 0, id, 1, body);
                let receipt = client
                    .request(
                        leader,
                        Request::Append {
                            group: 10,
                            batch: original.clone(),
                        },
                    )
                    .await?;
                let index = accepted(&receipt);
                drop(receipt);
                let fetched = client
                    .request(leader, Request::Fetch { group: 10, index })
                    .await?;
                ensure!(
                    matches!(&fetched.reply, Reply::Record(Some(meta)) if meta == &original.meta)
                        && fetched.body == original.body,
                    "response mixed independent producer payloads"
                );
                Ok::<_, anyhow::Error>(index)
            });
        }
    }
    let mut indices = BTreeSet::new();
    while let Some(result) = tasks.join_next().await {
        assert!(indices.insert(result??));
    }
    assert_eq!(indices.len(), 16);
    c.shutdown().await
}
