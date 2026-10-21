use super::*;
use crate::{
    record::Batch,
    replication::{
        runtime::{Command, Runtime},
        tests::{batch, layout},
        Applied, Limits, Node, Outcome,
    },
};
use raft_proto::eraftpb::{Entry, Message, MessageType};
use rustls::{
    pki_types::{CertificateDer, PrivatePkcs8KeyDer},
    RootCertStore,
};
use std::{net::SocketAddr, path::Path};

mod process;
mod snapshots;

struct Tls {
    client: Arc<rustls::ClientConfig>,
    server: Arc<rustls::ServerConfig>,
    certificate: [u8; 32],
    name: String,
}
fn certificates() -> Result<BTreeMap<u64, Tls>> {
    // Fresh keys and ordinary verifying TLS, including client certificates.
    // A fourth trusted-but-unconfigured identity exercises the node pin boundary.
    let mut identities = Vec::new();
    let mut roots = RootCertStore::empty();
    for id in 1..=4 {
        let name = format!("wal-{id}.example.invalid");
        let certified = rcgen::generate_simple_self_signed(vec![name.clone()])?;
        let der: CertificateDer<'static> = certified.cert.der().clone();
        roots.add(der.clone())?;
        identities.push((id, name, der, certified.signing_key.serialize_der()));
    }
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let roots = Arc::new(roots);
    let mut configs = BTreeMap::new();
    for (id, name, cert, key) in identities {
        let identity = fingerprint(Some(std::slice::from_ref(&cert)))?;
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            roots.clone(),
            provider.clone(),
        )
        .build()?;
        let mut server = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()?
            .with_client_cert_verifier(verifier)
            .with_single_cert(
                vec![cert.clone()],
                PrivatePkcs8KeyDer::from(key.clone()).into(),
            )?;
        let mut client = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()?
            .with_root_certificates(roots.clone())
            .with_client_auth_cert(vec![cert], PrivatePkcs8KeyDer::from(key).into())?;
        server.alpn_protocols = vec![ALPN.to_vec()];
        client.alpn_protocols = vec![ALPN.to_vec()];
        configs.insert(
            id,
            Tls {
                client: Arc::new(client),
                server: Arc::new(server),
                certificate: identity,
                name,
            },
        );
    }
    Ok(configs)
}

struct Running {
    runtime: Runtime,
    network: Transport,
    handle: Handle,
}
struct Cluster {
    nodes: BTreeMap<u64, Running>,
    tls: BTreeMap<u64, Tls>,
    addresses: BTreeMap<u64, SocketAddr>,
    network_options: Options,
}
impl Cluster {
    async fn open(root: &Path) -> Result<Self> {
        Self::with_options(root, Options::default()).await
    }
    async fn with_options(root: &Path, network_options: Options) -> Result<Self> {
        let mut listeners = BTreeMap::new();
        let mut addresses = BTreeMap::new();
        for id in 1..=3 {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            addresses.insert(id, listener.local_addr()?);
            listeners.insert(id, listener);
        }
        let mut c = Self {
            nodes: BTreeMap::new(),
            tls: certificates()?,
            addresses,
            network_options,
        };
        for (id, listener) in listeners {
            c.start(root, id, listener)?;
        }
        Ok(c)
    }
    fn start(&mut self, root: &Path, id: u64, listener: TcpListener) -> Result<()> {
        let node = Node::open(&root.join(id.to_string()), layout(), id, Limits::default())?
            .with_disk_budget(disk_options())?;
        let (runtime, outbox) = Runtime::spawn(node, super::super::Options::default())?;
        let handle = runtime.handle();
        let peers = self
            .addresses
            .iter()
            .filter(|(peer, _)| **peer != id)
            .map(|(&peer, address)| {
                (
                    peer,
                    Peer {
                        address: address.to_string(),
                        server_name: self.tls[&peer].name.clone(),
                        certificate: self.tls[&peer].certificate,
                    },
                )
            })
            .collect();
        let network = Transport::spawn(
            handle.clone(),
            outbox,
            listener,
            peers,
            self.tls[&id].client.clone(),
            self.tls[&id].server.clone(),
            self.network_options.clone(),
        )?;
        self.nodes.insert(
            id,
            Running {
                runtime,
                network,
                handle,
            },
        );
        Ok(())
    }
    async fn stop(&mut self, id: u64) -> Result<()> {
        let node = self.nodes.remove(&id).context("missing test node")?;
        node.network.shutdown().await?;
        node.runtime.shutdown().await
    }
    async fn leader(&self, group: u64) -> Result<u64> {
        timeout(Duration::from_secs(10), async {
            loop {
                let leaders = self
                    .nodes
                    .values()
                    .map(|n| n.handle.observation(group).unwrap().status.leader)
                    .collect::<BTreeSet<_>>();
                if leaders.len() == 1 && self.nodes.contains_key(leaders.first().unwrap()) {
                    return *leaders.first().unwrap();
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .context("TLS replica election did not settle")
    }
    async fn shutdown(mut self) -> Result<()> {
        let ids: Vec<_> = self.nodes.keys().copied().collect();
        for id in ids {
            self.stop(id).await?;
        }
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_follower_that_stops_reading_cannot_consume_the_healthy_quorums_outbox() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut c = Cluster::with_options(
        dir.path(),
        Options {
            io_timeout: Duration::from_secs(60),
            ..Options::default()
        },
    )
    .await?;
    c.nodes[&1].handle.campaign(10)?.await?;
    let leader = c.leader(10).await?;
    let slow = *c.nodes.keys().rfind(|&&id| id != leader).unwrap();
    c.stop(slow).await?;
    // Complete valid mutual TLS, then retain the sockets without consuming
    // application bytes. Its 60-second I/O timeout must not be what permits the
    // other two durable replicas to finish the test's 20-second deadline.
    let listener = TcpListener::bind(c.addresses[&slow]).await?;
    let acceptor = TlsAcceptor::from(c.tls[&slow].server.clone());
    let expected = c.tls[&leader].certificate;
    let connected = Arc::new(AtomicBool::new(false));
    let observed = connected.clone();
    let blackhole = tokio::spawn(async move {
        let mut streams = Vec::new();
        loop {
            let (tcp, _) = listener.accept().await?;
            let stream = acceptor.accept(tcp).await?;
            if fingerprint(stream.get_ref().1.peer_certificates())? == expected {
                observed.store(true, Ordering::Release);
            }
            streams.push(stream);
        }
        #[allow(unreachable_code)]
        Ok::<_, anyhow::Error>(())
    });
    wait_for(|| connected.load(Ordering::Acquire)).await?;
    let body = Bytes::from(vec![0x63; 2 << 20]);
    let base = batch(1);
    let mut pending = Vec::new();
    for sequence in 1..=24 {
        let batch = Batch::new(
            base.meta.stream,
            0,
            base.meta.producer,
            sequence,
            body.clone(),
        );
        pending.push((
            batch.clone(),
            c.nodes[&leader]
                .handle
                .commit(Command::Append { group: 10, batch })?,
        ));
    }
    let accepted = timeout(Duration::from_secs(20), async {
        let mut accepted = Vec::new();
        for (batch, pending) in pending {
            let Outcome::Committed(Applied::Accepted { index }) = pending.await? else {
                anyhow::bail!("healthy quorum did not commit while follower stopped reading");
            };
            accepted.push((index, batch));
        }
        Ok::<_, anyhow::Error>(accepted)
    })
    .await??;
    assert!(!blackhole.is_finished());
    wait_for(|| {
        c.nodes[&leader]
            .handle
            .shared
            .outgoing
            .bytes
            .available_permits()
            >= 32 << 20
    })
    .await?;
    let latest = accepted.last().unwrap().0;
    wait_for(|| {
        c.nodes
            .values()
            .all(|n| n.handle.observation(10).unwrap().status.applied >= latest)
    })
    .await?;
    let survivors: Vec<_> = c.nodes.keys().copied().collect();
    blackhole.abort();
    let _ = blackhole.await;
    c.shutdown().await?;
    for id in survivors {
        let node = Node::open(
            &dir.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        for (index, batch) in &accepted {
            assert_eq!(node.read_accepted(10, *index)?, Some(batch.clone()));
        }
    }
    Ok(())
}
async fn wait_for(mut predicate: impl FnMut() -> bool) -> Result<()> {
    timeout(Duration::from_secs(5), async {
        while !predicate() {
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    Ok(())
}
async fn append(handle: &Handle, group: u64, batch: Batch) -> Result<u64> {
    match timeout(
        Duration::from_secs(10),
        handle.commit(Command::Append { group, batch })?,
    )
    .await??
    {
        Outcome::Committed(Applied::Accepted { index }) => Ok(index),
        other => anyhow::bail!("unexpected append result {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mutual_tls_replication_reconnects_and_retains_exact_payloads_after_all_engines_reopen(
) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut c = Cluster::open(dir.path()).await?;
    c.nodes[&1].handle.campaign(10)?.await?;
    c.nodes[&2].handle.campaign(20)?.await?;
    let leader = c.leader(10).await?;
    let base = batch(1);
    let first = Batch::new(
        base.meta.stream,
        0,
        base.meta.producer,
        1,
        Bytes::from(vec![0xa5; 2 << 20]),
    );
    let i = append(&c.nodes[&leader].handle, 10, first.clone()).await?;
    let other_leader = c.leader(20).await?;
    let mut other = batch(1);
    other.meta.bucket = 1;
    let j = append(&c.nodes[&other_leader].handle, 20, other.clone()).await?;
    wait_for(|| {
        c.nodes.values().all(|n| {
            n.handle.observation(10).unwrap().status.applied >= i
                && n.handle.observation(20).unwrap().status.applied >= j
        })
    })
    .await?;
    c.stop(leader).await?;
    let successor = c.leader(10).await?;
    assert_eq!(
        append(&c.nodes[&successor].handle, 10, first.clone()).await?,
        i
    );
    let second = Batch::new(
        base.meta.stream,
        0,
        base.meta.producer,
        2,
        Bytes::from(vec![0x5a; 2 << 20]),
    );
    let k = append(&c.nodes[&successor].handle, 10, second.clone()).await?;
    let listener = TcpListener::bind(c.addresses[&leader]).await?;
    c.start(dir.path(), leader, listener)?;
    wait_for(|| {
        c.nodes
            .values()
            .all(|n| n.handle.observation(10).unwrap().status.applied >= k)
    })
    .await?;
    assert_eq!(
        append(&c.nodes[&successor].handle, 10, second.clone()).await?,
        k
    );
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &dir.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        assert_eq!(node.read_accepted(10, i)?, Some(first.clone()));
        assert_eq!(node.read_accepted(10, k)?, Some(second.clone()));
        assert_eq!(node.read_accepted(20, j)?, Some(other.clone()));
    }
    Ok(())
}

async fn connect(
    c: &Cluster,
    client: u64,
    server: u64,
) -> Result<tokio_rustls::client::TlsStream<TcpStream>> {
    Ok(TlsConnector::from(c.tls[&client].client.clone())
        .connect(
            ServerName::try_from(c.tls[&server].name.clone())?,
            TcpStream::connect(c.addresses[&server]).await?,
        )
        .await?)
}
fn packet(handle: &Handle, from: u64, body: Bytes) -> Packet {
    let mut message = Message {
        from,
        to: handle.id,
        term: 100,
        ..Default::default()
    };
    message.set_msg_type(MessageType::MsgAppend);
    message.set_entries(
        vec![Entry {
            index: 1,
            term: 100,
            data: body,
            ..Default::default()
        }]
        .into(),
    );
    Packet {
        cluster: handle.cluster,
        layout: handle.layout,
        policy: handle.policy,
        group: 10,
        message,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tls_identity_cannot_impersonate_another_voter_or_use_an_unconfigured_trusted_certificate(
) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = Cluster::open(dir.path()).await?;
    c.nodes[&1].handle.campaign(10)?.await?;
    c.leader(10).await?;
    let stats = c.nodes[&1].network.statistics();
    let before = stats.rejected_connections.load(Ordering::Relaxed);
    let mut unknown = connect(&c, 4, 1).await?;
    let _ = unknown.write_all(&[0; 4]).await;
    wait_for(|| stats.rejected_connections.load(Ordering::Relaxed) > before).await?;
    let term = c.nodes[&1].handle.observation(10)?.status.term;
    let before = stats.failed_exchanges.load(Ordering::Relaxed);
    let mut impostor = connect(&c, 2, 1).await?;
    let forged = packet(
        &c.nodes[&1].handle,
        3,
        Bytes::from_static(b"not a valid log"),
    );
    write_packet(&mut impostor, &forged).await?;
    wait_for(|| stats.failed_exchanges.load(Ordering::Relaxed) > before).await?;
    assert_eq!(c.nodes[&1].handle.observation(10)?.status.term, term);
    c.shutdown().await
}

#[tokio::test]
async fn framing_shares_outbound_payload_and_rejects_unfunded_bodies_before_reading_them(
) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let node = Node::open(dir.path(), layout(), 1, Limits::default())?;
    let (runtime, _outbox) = Runtime::spawn(node, super::super::Options::default())?;
    let handle = runtime.handle();
    let payload = Bytes::from(vec![37; 2 << 20]);
    let p = packet(&handle, 2, payload.clone());
    let (header, bodies) = encode(&p)?;
    assert_eq!(bodies[0].as_ptr(), payload.as_ptr());
    assert!(
        header.len() < 1024,
        "payload was copied into the metadata frame"
    );
    let (mut write, mut read) = tokio::io::duplex(2048);
    crate::transport_io::write_frame(&mut write, &header, &[], None).await?;
    let credits = Arc::new(Semaphore::new(header.len()));
    let result = timeout(
        Duration::from_millis(100),
        read_packet(&mut read, &handle, 2, &credits),
    )
    .await?;
    assert!(result.is_err(), "body was read before reserving space");
    assert_eq!(credits.available_permits(), header.len());
    drop(read);
    drop(write);

    let (mut write, mut read) = tokio::io::duplex(4096);
    let output = tokio::spawn(async move { write_packet(&mut write, &p).await });
    let credits = Arc::new(Semaphore::new(4 << 20));
    let decoded = read_packet(&mut read, &handle, 2, &credits).await?;
    assert_eq!(decoded.packet.message.entries[0].data, payload);
    assert!(credits.available_permits() < 2 << 20);
    output.await??;
    drop(decoded);
    assert_eq!(credits.available_permits(), 4 << 20);
    runtime.shutdown().await
}

fn disk_options() -> crate::raft_log::DiskOptions {
    crate::raft_log::DiskOptions {
        max_bytes: 512 << 20,
        data_bytes: 128 << 20,
        headroom_bytes: 256 << 20,
    }
}
