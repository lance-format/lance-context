use super::super::snapshot;
use super::*;
use crate::replication::checkpoint::Image;
use std::sync::atomic::AtomicU64;

// Conservative fixture accounting for image files only. Charges never return
// during one incarnation, including failed receptions. Reopen scans original
// image files. Production also needs log/rewrite headroom and collection.
pub(super) struct Quota {
    used: AtomicU64,
    limit: u64,
}
impl Quota {
    pub(super) fn open(directory: &Path) -> Result<Self> {
        let mut used = 0;
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with("context-application-")
            {
                used += entry.metadata()?.len();
            }
        }
        Ok(Self {
            used: AtomicU64::new(used),
            limit: 256 << 20,
        })
    }
}
impl snapshot::Admission for Quota {
    fn reserve(&self, incoming: &snapshot::Incoming) -> Result<Arc<dyn Send + Sync>> {
        let bytes = incoming.bytes()?;
        self.used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|n| *n <= self.limit)
            })
            .map_err(|_| anyhow::anyhow!("fixture image disk budget exhausted"))?;
        Ok(Arc::new(bytes))
    }
}
struct Images {
    net: Cluster,
    services: BTreeMap<u64, snapshot::Transport>,
    addresses: BTreeMap<u64, SocketAddr>,
    quotas: BTreeMap<u64, Arc<Quota>>,
}
impl Images {
    async fn open(root: &Path) -> Result<Self> {
        let net = Cluster::open(root).await?;
        let mut listeners = BTreeMap::new();
        let mut addresses = BTreeMap::new();
        let mut quotas = BTreeMap::new();
        for id in 1..=3 {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            addresses.insert(id, listener.local_addr()?);
            listeners.insert(id, listener);
            quotas.insert(id, Arc::new(Quota::open(&root.join(id.to_string()))?));
        }
        let mut c = Self {
            net,
            services: BTreeMap::new(),
            addresses,
            quotas,
        };
        for (id, listener) in listeners {
            c.attach(id, listener)?;
        }
        Ok(c)
    }
    fn attach(&mut self, id: u64, listener: TcpListener) -> Result<()> {
        self.attach_with(id, listener, self.quotas[&id].clone())
    }
    fn attach_with(
        &mut self,
        id: u64,
        listener: TcpListener,
        admission: Arc<dyn snapshot::Admission>,
    ) -> Result<()> {
        let (client_tls, server_tls) = image_tls(&self.net.tls[&id]);
        let peers = self
            .addresses
            .iter()
            .filter(|(peer, _)| **peer != id)
            .map(|(&peer, address)| {
                (
                    peer,
                    Peer {
                        address: address.to_string(),
                        server_name: self.net.tls[&peer].name.clone(),
                        certificate: self.net.tls[&peer].certificate,
                    },
                )
            })
            .collect();
        let service = snapshot::Transport::spawn(
            self.net.nodes[&id].handle.clone(),
            listener,
            snapshot::Settings {
                peers,
                client_tls,
                server_tls,
                options: snapshot::Options {
                    transfer_timeout: Duration::from_secs(60),
                    ..Default::default()
                },
                admission,
            },
        )?;
        self.net
            .nodes
            .get_mut(&id)
            .unwrap()
            .network
            .attach_snapshots(service.route())?;
        self.services.insert(id, service);
        Ok(())
    }
    async fn stop(&mut self, id: u64) -> Result<()> {
        if let Some(service) = self.services.remove(&id) {
            service.shutdown().await?;
        }
        self.net.stop(id).await
    }
    async fn start(&mut self, root: &Path, id: u64) -> Result<()> {
        self.net
            .start(root, id, TcpListener::bind(self.net.addresses[&id]).await?)?;
        self.attach(id, TcpListener::bind(self.addresses[&id]).await?)
    }
    async fn shutdown(mut self) -> Result<()> {
        while let Some((_, service)) = self.services.pop_first() {
            service.shutdown().await?;
        }
        self.net.shutdown().await
    }
}
fn image_tls(tls: &Tls) -> (Arc<rustls::ClientConfig>, Arc<rustls::ServerConfig>) {
    let mut client = (*tls.client).clone();
    let mut server = (*tls.server).clone();
    client.alpn_protocols = vec![snapshot::ALPN.to_vec()];
    server.alpn_protocols = vec![snapshot::ALPN.to_vec()];
    (Arc::new(client), Arc::new(server))
}
async fn checkpoint(handle: &Handle, group: u64) -> Result<Arc<Image>> {
    let task = handle
        .prepare_checkpoint(group, uuid::Uuid::new_v4())?
        .await?;
    let prepared = tokio::task::spawn_blocking(move || task.run()).await??;
    let image = prepared.image();
    handle.install_checkpoint(prepared)?.await?;
    Ok(image)
}
async fn append(handle: &Handle, group: u64, sequence: u64) -> Result<(u64, Batch)> {
    let base = batch(sequence);
    let body = Bytes::from(vec![(group + sequence) as u8; 1 << 20]);
    let b = Batch::new(
        base.meta.stream,
        u32::from(group == 20),
        base.meta.producer,
        sequence,
        body,
    );
    let result = timeout(
        Duration::from_secs(10),
        handle.commit(Command::Append {
            group,
            batch: b.clone(),
        })?,
    )
    .await??;
    match result {
        Outcome::Committed(Applied::Accepted { index }) => Ok((index, b)),
        other => anyhow::bail!("unexpected append {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tls_snapshots_catch_up_two_compacted_groups_and_reopen_original_payloads() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut c = Images::open(root.path()).await?;
    for group in [10, 20] {
        c.net.nodes[&1].handle.campaign(group)?.await?;
        assert_eq!(c.net.leader(group).await?, 1);
    }
    c.stop(3).await?;
    let mut expected = Vec::new();
    for group in [10, 20] {
        for sequence in 1..=4 {
            let (index, batch) = append(&c.net.nodes[&1].handle, group, sequence).await?;
            expected.push((group, index, batch));
        }
        drop(checkpoint(&c.net.nodes[&1].handle, group).await?);
    }
    c.start(root.path(), 3).await?;
    wait_for(|| {
        [10, 20].iter().all(|g| {
            c.net.nodes[&3]
                .handle
                .observation(*g)
                .unwrap()
                .status
                .accepted_records
                == 4
        })
    })
    .await?;
    assert!(
        c.services[&3]
            .statistics()
            .installed
            .load(Ordering::Acquire)
            >= 2
    );
    assert!(c.quotas[&3].used.load(Ordering::Acquire) >= 8 << 20);
    for group in [10, 20] {
        let (index, batch) = append(&c.net.nodes[&1].handle, group, 5).await?;
        expected.push((group, index, batch));
    }
    wait_for(|| {
        c.net.nodes.values().all(|n| {
            [10, 20]
                .iter()
                .all(|g| n.handle.observation(*g).unwrap().status.accepted_records == 5)
        })
    })
    .await?;
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        for (group, index, batch) in &expected {
            assert_eq!(node.read_accepted(*group, *index)?, Some(batch.clone()));
        }
    }
    let c = Images::open(root.path()).await?;
    assert!(c.quotas[&3].used.load(Ordering::Acquire) >= 8 << 20);
    for group in [10, 20] {
        // Reopening every engine may already have elected another voter.
        // A campaign is not a leadership receipt; retry through the actual
        // settled leader without perturbing that election.
        let leader = c.net.leader(group).await?;
        let (index, _) = append(&c.net.nodes[&leader].handle, group, 5).await?;
        assert_eq!(
            index,
            expected
                .iter()
                .find(|(g, _, b)| *g == group && b.meta.sequence == 5)
                .unwrap()
                .1
        );
    }
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stalled_tls_snapshot_does_not_block_healthy_quorum_and_recovers_after_disconnect(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut c = Images::open(root.path()).await?;
    c.net.nodes[&1].handle.campaign(10)?.await?;
    assert_eq!(c.net.leader(10).await?, 1);
    c.stop(3).await?;
    let mut expected = Vec::new();
    for sequence in 1..=4 {
        expected.push(append(&c.net.nodes[&1].handle, 10, sequence).await?);
    }
    drop(checkpoint(&c.net.nodes[&1].handle, 10).await?);
    // Node3 handles ordinary Raft messages, but its separate image socket is a
    // correctly authenticated peer that never grants receive admission.
    c.net.start(
        root.path(),
        3,
        TcpListener::bind(c.net.addresses[&3]).await?,
    )?;
    let listener = TcpListener::bind(c.addresses[&3]).await?;
    let acceptor = TlsAcceptor::from(image_tls(&c.net.tls[&3]).1);
    let connected = Arc::new(AtomicBool::new(false));
    let observed = connected.clone();
    let stalled = tokio::spawn(async move {
        let mut streams = Vec::new();
        loop {
            let (tcp, _) = listener.accept().await?;
            streams.push(acceptor.accept(tcp).await?);
            observed.store(true, Ordering::Release);
        }
        #[allow(unreachable_code)]
        Ok::<_, anyhow::Error>(())
    });
    wait_for(|| connected.load(Ordering::Acquire)).await?;
    // Transfers have a60sdeadline; quorum writes must finish much earlier.
    timeout(Duration::from_secs(10), async {
        for sequence in 5..=12 {
            expected.push(append(&c.net.nodes[&1].handle, 10, sequence).await?);
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    assert!(!stalled.is_finished());
    stalled.abort();
    let _ = stalled.await;
    c.attach(3, TcpListener::bind(c.addresses[&3]).await?)?;
    wait_for(|| {
        c.net.nodes[&3]
            .handle
            .observation(10)
            .unwrap()
            .status
            .accepted_records
            == 12
    })
    .await?;
    assert!(
        c.services[&3]
            .statistics()
            .installed
            .load(Ordering::Acquire)
            > 0
    );
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        for (index, batch) in &expected {
            assert_eq!(node.read_accepted(10, *index)?, Some(batch.clone()));
        }
    }
    Ok(())
}

async fn upload_header(
    c: &Images,
    packet: &Packet,
) -> Result<tokio_rustls::client::TlsStream<TcpStream>> {
    use protobuf::Message as _;
    let connector = TlsConnector::from(image_tls(&c.net.tls[&1]).0);
    let tcp = TcpStream::connect(c.addresses[&3]).await?;
    let mut stream = connector
        .connect(ServerName::try_from(c.net.tls[&3].name.clone())?, tcp)
        .await?;
    let mut header = b"LCWSNAP1".to_vec();
    header.extend_from_slice(packet.cluster.as_bytes());
    header.extend_from_slice(&packet.layout);
    header.extend_from_slice(&packet.policy);
    header.extend_from_slice(&packet.group.to_be_bytes());
    header.extend_from_slice(&packet.message.write_to_bytes()?);
    stream.write_u32(header.len() as u32).await?;
    stream.write_all(&header).await?;
    stream.flush().await?;
    Ok(stream)
}
fn image_files(root: &Path) -> Result<usize> {
    Ok(std::fs::read_dir(root)?
        .collect::<std::io::Result<Vec<_>>>()?
        .iter()
        .filter(|p| {
            p.file_name()
                .to_string_lossy()
                .starts_with("context-application-")
        })
        .count())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tls_snapshot_denial_and_interruption_never_install_or_release_disk_charges() -> Result<()>
{
    use crate::replication::checkpoint::{Part, CHUNK_BYTES};
    let root = tempfile::tempdir()?;
    let mut c = Images::open(root.path()).await?;
    c.net.nodes[&1].handle.campaign(10)?.await?;
    assert_eq!(c.net.leader(10).await?, 1);
    // Retain the actual receiver runtime and snapshot service, but remove its
    // normal transport so only the deliberate TLS uploads can reach it.
    let Running {
        network,
        runtime,
        handle,
    } = c.net.nodes.remove(&3).unwrap();
    network.shutdown().await?;
    let (index, expected) = append(&c.net.nodes[&1].handle, 10, 1).await?;
    let image = checkpoint(&c.net.nodes[&1].handle, 10).await?;
    let leader = &c.net.nodes[&1].handle;
    let packet = Packet {
        cluster: leader.cluster,
        layout: leader.layout,
        policy: leader.policy,
        group: 10,
        message: Message {
            from: 1,
            to: 3,
            term: leader.observation(10)?.status.term,
            msg_type: MessageType::MsgSnapshot,
            snapshot: Some(image.snapshot()?).into(),
            ..Default::default()
        },
    };
    let path = root.path().join("3");
    let before = handle.observation(10)?.status.applied;
    let files = image_files(&path)?;
    c.quotas[&3]
        .used
        .store(c.quotas[&3].limit, Ordering::Release);
    let mut denied = upload_header(&c, &packet).await?;
    let mut ack = [0; 8];
    no_response(&mut denied).await?;
    assert_eq!(image_files(&path)?, files);
    assert_eq!(handle.observation(10)?.status.applied, before);
    drop(denied);

    // The rejected request created no files or charge; open a fresh fixture
    // allowance. A partial reception then retains its full conservative charge.
    c.quotas[&3].used.store(0, Ordering::Release);
    let mut partial = upload_header(&c, &packet).await?;
    assert_eq!(partial.read_u8().await?, 1);
    partial.write_all(&[0]).await?;
    partial.shutdown().await?;
    no_response(&mut partial).await?;
    assert_eq!(handle.observation(10)?.status.applied, before);
    let bytes = image.bytes(Part::Metadata) + image.bytes(Part::Payload);
    assert_eq!(c.quotas[&3].used.load(Ordering::Acquire), bytes);
    assert!(image_files(&path)? > files);
    drop(partial);

    let mut complete = upload_header(&c, &packet).await?;
    assert_eq!(complete.read_u8().await?, 1);
    for part in [Part::Metadata, Part::Payload] {
        let mut offset = 0;
        while offset < image.bytes(part) {
            let chunk = image.chunk(
                part,
                offset,
                (image.bytes(part) - offset).min(CHUNK_BYTES as u64) as usize,
            )?;
            complete.write_all(&chunk).await?;
            offset += chunk.len() as u64;
        }
    }
    complete.flush().await?;
    timeout(Duration::from_secs(5), complete.read_exact(&mut ack)).await??;
    assert_eq!(&ack, b"LCWSYNC1");
    assert_eq!(c.quotas[&3].used.load(Ordering::Acquire), bytes * 2);
    wait_for(|| handle.observation(10).unwrap().status.accepted_records == 1).await?;
    drop(complete);
    drop(image);
    c.shutdown().await?;
    runtime.shutdown().await?;
    let node = Node::open(&path, layout(), 3, Limits::default())?;
    assert_eq!(node.read_accepted(10, index)?, Some(expected));
    Ok(())
}

async fn no_response(stream: &mut (impl AsyncRead + Unpin)) -> Result<()> {
    let mut response = [0; 8];
    match timeout(Duration::from_secs(5), stream.read(&mut response)).await? {
        Ok(0) => Ok(()),
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset
            ) =>
        {
            Ok(())
        }
        other => anyhow::bail!("rejected snapshot returned an unexpected response: {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_file_worker_panic_stops_service_and_allows_original_engine_recovery() -> Result<()>
{
    struct PanicAdmission;
    impl snapshot::Admission for PanicAdmission {
        fn reserve(&self, _: &snapshot::Incoming) -> Result<Arc<dyn Send + Sync>> {
            panic!("injected snapshot admission panic")
        }
    }
    let root = tempfile::tempdir()?;
    let mut c = Images::open(root.path()).await?;
    c.net.nodes[&1].handle.campaign(10)?.await?;
    assert_eq!(c.net.leader(10).await?, 1);
    c.stop(3).await?;
    let (index, expected) = append(&c.net.nodes[&1].handle, 10, 1).await?;
    drop(checkpoint(&c.net.nodes[&1].handle, 10).await?);
    c.net.start(
        root.path(),
        3,
        TcpListener::bind(c.net.addresses[&3]).await?,
    )?;
    c.attach_with(
        3,
        TcpListener::bind(c.addresses[&3]).await?,
        Arc::new(PanicAdmission),
    )?;
    wait_for(|| c.services[&3].statistics().failed.load(Ordering::Acquire) > 0).await?;
    let failed = c.services.remove(&3).unwrap();
    assert!(timeout(Duration::from_secs(5), failed.shutdown())
        .await?
        .is_err());
    assert_eq!(
        c.net.nodes[&3]
            .handle
            .observation(10)?
            .status
            .accepted_records,
        0
    );
    c.net.stop(3).await?;
    c.start(root.path(), 3).await?;
    wait_for(|| {
        c.net.nodes[&3]
            .handle
            .observation(10)
            .unwrap()
            .status
            .accepted_records
            == 1
    })
    .await?;
    c.shutdown().await?;
    let node = Node::open(&root.path().join("3"), layout(), 3, Limits::default())?;
    assert_eq!(node.read_accepted(10, index)?, Some(expected));
    Ok(())
}
