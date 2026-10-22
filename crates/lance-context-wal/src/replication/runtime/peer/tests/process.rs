//! Child processes use the actual TLS transport. The parent sends only test
//! control requests over loopback; it never relays consensus packets.
use super::*;
use crate::replication::runtime::client as rpc;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{Read, Write},
    net::TcpStream as ControlStream,
    path::PathBuf,
    process::{Child, Command as ProcessCommand, Stdio},
    time::Instant,
};

#[derive(Serialize, Deserialize)]
struct Address {
    peer: SocketAddr,
    image: SocketAddr,
    control: SocketAddr,
    rpc_data: SocketAddr,
    rpc_control: SocketAddr,
    pid: u32,
}
#[derive(Serialize, Deserialize)]
enum Request {
    Start {
        peers: BTreeMap<u64, SocketAddr>,
        images: BTreeMap<u64, SocketAddr>,
    },
    Checkpoint(u64),
    ImageStatus,
    Campaign(u64),
    Status(u64),
    Append(u64, Batch),
}
#[derive(Serialize, Deserialize)]
enum Reply {
    Ready,
    Images { installed: u64 },
    Status(crate::replication::Status),
    Append(Outcome),
    Failed(String),
}
const FRAME_LIMIT: usize = 16 << 20;

fn provision(root: &Path) -> Result<()> {
    for id in 1..=4 {
        let certified =
            rcgen::generate_simple_self_signed(vec![format!("wal-{id}.example.invalid")])?;
        fs::write(root.join(format!("cert-{id}.der")), certified.cert.der())?;
        fs::write(
            root.join(format!("key-{id}.der")),
            certified.signing_key.serialize_der(),
        )?;
    }
    Ok(())
}
type TlsConfiguration = (
    Arc<rustls::ClientConfig>,
    Arc<rustls::ServerConfig>,
    BTreeMap<u64, [u8; 32]>,
);
fn tls(root: &Path, id: u64) -> Result<TlsConfiguration> {
    let mut roots = RootCertStore::empty();
    let mut pins = BTreeMap::new();
    for peer in 1..=4 {
        let cert = CertificateDer::from(fs::read(root.join(format!("cert-{peer}.der")))?);
        pins.insert(peer, fingerprint(Some(std::slice::from_ref(&cert)))?);
        roots.add(cert)?;
    }
    let roots = Arc::new(roots);
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
        roots.clone(),
        provider.clone(),
    )
    .build()?;
    let cert = CertificateDer::from(fs::read(root.join(format!("cert-{id}.der")))?);
    let key = fs::read(root.join(format!("key-{id}.der")))?;
    let mut server = rustls::ServerConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()?
        .with_client_cert_verifier(verifier)
        .with_single_cert(
            vec![cert.clone()],
            PrivatePkcs8KeyDer::from(key.clone()).into(),
        )?;
    let mut client = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .with_root_certificates(roots)
        .with_client_auth_cert(vec![cert], PrivatePkcs8KeyDer::from(key).into())?;
    server.alpn_protocols = vec![ALPN.to_vec()];
    client.alpn_protocols = vec![ALPN.to_vec()];
    Ok((Arc::new(client), Arc::new(server), pins))
}

async fn receive(stream: &mut TcpStream) -> Result<Request> {
    let size = stream.read_u32().await? as usize;
    ensure!(size <= FRAME_LIMIT, "test control frame too large");
    let mut bytes = vec![0; size];
    stream.read_exact(&mut bytes).await?;
    Ok(bincode::deserialize(&bytes)?)
}
async fn respond(stream: &mut TcpStream, reply: Reply) -> Result<()> {
    let bytes = bincode::serialize(&reply)?;
    ensure!(bytes.len() <= FRAME_LIMIT, "test reply too large");
    stream.write_u32(bytes.len() as u32).await?;
    stream.write_all(&bytes).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "child fixture invoked by the TLS process qualification"]
async fn tls_peer_process_fixture() -> Result<()> {
    let root = PathBuf::from(std::env::var("CONTEXT_TLS_PEER_TEST_ROOT")?);
    let id: u64 = std::env::var("CONTEXT_TLS_PEER_TEST_ID")?.parse()?;
    let address_file = PathBuf::from(std::env::var("CONTEXT_TLS_PEER_TEST_ADDRESS")?);
    let listener = TcpListener::bind(
        std::env::var("CONTEXT_TLS_PEER_BIND").unwrap_or_else(|_| "127.0.0.1:0".into()),
    )
    .await
    .context("bind peer listener")?;
    let image_listener = TcpListener::bind(
        std::env::var("CONTEXT_TLS_IMAGE_BIND").unwrap_or_else(|_| "127.0.0.1:0".into()),
    )
    .await
    .context("bind image listener")?;
    let control = TcpListener::bind("127.0.0.1:0").await?;
    let rpc_data = TcpListener::bind("127.0.0.1:0").await?;
    let rpc_control = TcpListener::bind("127.0.0.1:0").await?;
    let address = Address {
        peer: listener.local_addr()?,
        image: image_listener.local_addr()?,
        control: control.local_addr()?,
        rpc_data: rpc_data.local_addr()?,
        rpc_control: rpc_control.local_addr()?,
        pid: std::process::id(),
    };
    fs::write(
        address_file.with_extension("tmp"),
        bincode::serialize(&address)?,
    )?;
    fs::rename(address_file.with_extension("tmp"), &address_file)?;
    let (mut stream, _) = control.accept().await?;
    let Request::Start {
        peers: addresses,
        images: image_addresses,
    } = receive(&mut stream).await?
    else {
        anyhow::bail!("missing startup configuration");
    };
    let (client, server, pins) = tls(&root, id)?;
    let node = Node::open(
        &root.join(format!("node-{id}")),
        layout(),
        id,
        Limits::default(),
    )?
    .with_disk_budget(super::disk_options())?;
    let (runtime, outbox) = Runtime::spawn(node, super::super::super::Options::default())?;
    let handle = runtime.handle();
    let mut rpc_tls = (*server).clone();
    rpc_tls.alpn_protocols = vec![rpc::ALPN.to_vec()];
    let rpc_service = rpc::Server::spawn(
        handle.clone(),
        rpc_data,
        rpc_control,
        BTreeSet::from([pins[&4]]),
        Arc::new(rpc_tls),
        rpc::Options::default(),
    )?;
    let peers = addresses
        .into_iter()
        .filter(|(peer, _)| *peer != id)
        .map(|(peer, address)| {
            (
                peer,
                super::super::Peer {
                    address: address.to_string(),
                    server_name: format!("wal-{peer}.example.invalid"),
                    certificate: pins[&peer],
                },
            )
        })
        .collect();
    let mut image_client = (*client).clone();
    let mut image_server = (*server).clone();
    image_client.alpn_protocols = vec![super::super::snapshot::ALPN.to_vec()];
    image_server.alpn_protocols = vec![super::super::snapshot::ALPN.to_vec()];
    let mut network = Transport::spawn(
        handle.clone(),
        outbox,
        listener,
        peers,
        client,
        server,
        Options::default(),
    )?;
    let images = super::super::snapshot::Transport::spawn(
        handle.clone(),
        image_listener,
        super::super::snapshot::Settings {
            peers: image_addresses
                .into_iter()
                .filter(|(peer, _)| *peer != id)
                .map(|(peer, address)| {
                    (
                        peer,
                        super::super::Peer {
                            address: address.to_string(),
                            server_name: format!("wal-{peer}.example.invalid"),
                            certificate: pins[&peer],
                        },
                    )
                })
                .collect(),
            client_tls: Arc::new(image_client),
            server_tls: Arc::new(image_server),
            options: super::super::snapshot::Options {
                pause_after_payload_chunk: std::env::var_os("CONTEXT_TLS_IMAGE_PAUSE")
                    .map(PathBuf::from),
                ..Default::default()
            },
            admission: None,
        },
    )?;
    network.attach_snapshots(images.route())?;
    respond(&mut stream, Reply::Ready).await?;
    loop {
        let request = match receive(&mut stream).await {
            Ok(request) => request,
            Err(_) => {
                rpc_service.shutdown().await?;
                images.shutdown().await?;
                network.shutdown().await?;
                return runtime.shutdown().await;
            }
        };
        let result: Result<Reply> = async {
            Ok(match request {
                Request::Start { .. } => anyhow::bail!("duplicate process startup"),
                Request::Checkpoint(group) => {
                    let task = handle
                        .prepare_checkpoint(group, uuid::Uuid::new_v4())?
                        .await?;
                    let image = tokio::task::spawn_blocking(move || task.run()).await??;
                    handle.install_checkpoint(image)?.await?;
                    Reply::Ready
                }
                Request::ImageStatus => Reply::Images {
                    installed: images.statistics().installed.load(Ordering::Acquire),
                },
                Request::Campaign(group) => {
                    handle.campaign(group)?.await?;
                    Reply::Ready
                }
                Request::Status(group) => Reply::Status(handle.observation(group)?.status),
                Request::Append(group, batch) => Reply::Append(
                    timeout(
                        Duration::from_secs(10),
                        handle.commit(Command::Append { group, batch })?,
                    )
                    .await??,
                ),
            })
        }
        .await;
        respond(
            &mut stream,
            result.unwrap_or_else(|e| Reply::Failed(format!("{e:#}"))),
        )
        .await?;
    }
}

struct Process {
    child: Child,
    stream: ControlStream,
    address: SocketAddr,
    image_address: SocketAddr,
    rpc_data: SocketAddr,
    rpc_control: SocketAddr,
}
impl Process {
    fn spawn(root: &Path, id: u64, incarnation: u64) -> Result<Self> {
        Self::spawn_at(root, id, incarnation, None, None)
    }
    fn spawn_at(
        root: &Path,
        id: u64,
        incarnation: u64,
        bind: Option<(SocketAddr, SocketAddr)>,
        pause: Option<&Path>,
    ) -> Result<Self> {
        let address_file = root.join(format!("address-{id}-{incarnation}"));
        let mut command = ProcessCommand::new(std::env::current_exe()?);
        if let Some((peer, image)) = bind {
            command
                .env("CONTEXT_TLS_PEER_BIND", peer.to_string())
                .env("CONTEXT_TLS_IMAGE_BIND", image.to_string());
        }
        if let Some(pause) = pause {
            command.env("CONTEXT_TLS_IMAGE_PAUSE", pause);
        }
        let mut child = command
            .args([
                "--exact",
                "replication::runtime::peer::tests::process::tls_peer_process_fixture",
                "--ignored",
                "--nocapture",
            ])
            .env("CONTEXT_TLS_PEER_TEST_ROOT", root)
            .env("CONTEXT_TLS_PEER_TEST_ID", id.to_string())
            .env("CONTEXT_TLS_PEER_TEST_ADDRESS", &address_file)
            .stdout(Stdio::from(
                File::options()
                    .write(true)
                    .create_new(true)
                    .open(root.join(format!("stdout-{id}-{incarnation}")))?,
            ))
            .stderr(Stdio::from(
                File::options()
                    .write(true)
                    .create_new(true)
                    .open(root.join(format!("stderr-{id}-{incarnation}")))?,
            ))
            .spawn()?;
        let result = (|| -> Result<(ControlStream, Address)> {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !address_file.exists() {
                ensure!(child.try_wait()?.is_none(), "TLS peer exited at startup");
                ensure!(
                    Instant::now() < deadline,
                    "TLS peer startup observation expired"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            let address: Address = bincode::deserialize(&fs::read(&address_file)?)?;
            ensure!(address.pid == child.id(), "wrong child process identity");
            let stream = ControlStream::connect(address.control)?;
            stream.set_nodelay(true)?;
            stream.set_read_timeout(Some(Duration::from_secs(15)))?;
            stream.set_write_timeout(Some(Duration::from_secs(15)))?;
            if let Some((peer, image)) = bind {
                ensure!(
                    address.peer == peer && address.image == image,
                    "original peer endpoints changed"
                );
            }
            Ok((stream, address))
        })();
        match result {
            Ok((stream, address)) => Ok(Self {
                child,
                stream,
                address: address.peer,
                image_address: address.image,
                rpc_data: address.rpc_data,
                rpc_control: address.rpc_control,
            }),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let stdout = fs::read_to_string(root.join(format!("stdout-{id}-{incarnation}")))
                    .unwrap_or_else(|error| format!("cannot read child stdout: {error}"));
                let stderr = fs::read_to_string(root.join(format!("stderr-{id}-{incarnation}")))
                    .unwrap_or_else(|error| format!("cannot read child stderr: {error}"));
                Err(e).with_context(|| {
                    format!(
                        "TLS peer {id} incarnation {incarnation} startup failed; child stdout:\n{stdout}\nchild stderr:\n{stderr}"
                    )
                })
            }
        }
    }
    fn call(&mut self, request: Request) -> Result<Reply> {
        let bytes = bincode::serialize(&request)?;
        ensure!(bytes.len() <= FRAME_LIMIT, "test request too large");
        self.stream.write_all(&(bytes.len() as u32).to_be_bytes())?;
        self.stream.write_all(&bytes)?;
        let mut size = [0; 4];
        self.stream.read_exact(&mut size)?;
        let size = u32::from_be_bytes(size) as usize;
        ensure!(size <= FRAME_LIMIT, "test response too large");
        let mut bytes = vec![0; size];
        self.stream.read_exact(&mut bytes)?;
        match bincode::deserialize(&bytes)? {
            Reply::Failed(e) => anyhow::bail!(e),
            reply => Ok(reply),
        }
    }
    fn kill(&mut self) -> Result<()> {
        self.child.kill()?;
        let status = self.child.wait()?;
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            ensure!(status.signal() == Some(9), "expected SIGKILL");
        }
        #[cfg(not(unix))]
        ensure!(!status.success(), "expected forced termination");
        Ok(())
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct Processes {
    nodes: BTreeMap<u64, Process>,
    root: PathBuf,
    incarnation: u64,
}
impl Processes {
    fn start_all(&mut self) -> Result<()> {
        ensure!(
            self.nodes.is_empty(),
            "original test processes still running"
        );
        self.incarnation += 1;
        for id in 1..=3 {
            self.nodes
                .insert(id, Process::spawn(&self.root, id, self.incarnation)?);
        }
        let addresses: BTreeMap<_, _> = self
            .nodes
            .iter()
            .map(|(&id, node)| (id, node.address))
            .collect();
        let images = self
            .nodes
            .iter()
            .map(|(&id, n)| (id, n.image_address))
            .collect::<BTreeMap<_, _>>();
        for node in self.nodes.values_mut() {
            ensure!(
                matches!(
                    node.call(Request::Start {
                        peers: addresses.clone(),
                        images: images.clone()
                    })?,
                    Reply::Ready
                ),
                "peer not ready"
            );
        }
        Ok(())
    }
    fn stop_all(&mut self) -> Result<()> {
        for (_, mut node) in std::mem::take(&mut self.nodes) {
            node.kill()?;
        }
        Ok(())
    }
    fn status(&mut self, id: u64, group: u64) -> Result<crate::replication::Status> {
        match self
            .nodes
            .get_mut(&id)
            .unwrap()
            .call(Request::Status(group))?
        {
            Reply::Status(status) => Ok(status),
            _ => anyhow::bail!("unexpected status reply"),
        }
    }
    fn leader(&mut self, group: u64) -> Result<u64> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let ids: Vec<_> = self.nodes.keys().copied().collect();
            let leaders: BTreeSet<_> = ids
                .into_iter()
                .map(|id| self.status(id, group).map(|s| s.leader))
                .collect::<Result<_>>()?;
            if leaders.len() == 1 && self.nodes.contains_key(leaders.first().unwrap()) {
                return Ok(*leaders.first().unwrap());
            }
            ensure!(
                Instant::now() < deadline,
                "TLS process election did not settle"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    fn append(&mut self, group: u64, batch: Batch) -> Result<u64> {
        let leader = self.leader(group)?;
        match self
            .nodes
            .get_mut(&leader)
            .unwrap()
            .call(Request::Append(group, batch))?
        {
            Reply::Append(Outcome::Committed(Applied::Accepted { index })) => Ok(index),
            _ => anyhow::bail!("TLS process did not return committed append"),
        }
    }
    fn applied(&mut self, group: u64, index: u64) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let ids: Vec<_> = self.nodes.keys().copied().collect();
            let applied = ids
                .into_iter()
                .map(|id| self.status(id, group).map(|s| s.applied >= index))
                .collect::<Result<Vec<_>>>()?;
            if applied.into_iter().all(|yes| yes) {
                return Ok(());
            }
            ensure!(
                Instant::now() < deadline,
                "TLS process catchup did not finish"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[test]
fn tls_process_leader_crash_full_restart_and_exact_retries_preserve_all_payloads() -> Result<()> {
    let root = tempfile::tempdir()?.keep();
    eprintln!("TLS process qualification artifacts: {}", root.display());
    provision(&root)?;
    let mut c = Processes {
        nodes: BTreeMap::new(),
        root: root.clone(),
        incarnation: 0,
    };
    c.start_all()?;
    c.nodes.get_mut(&1).unwrap().call(Request::Campaign(10))?;
    c.nodes.get_mut(&2).unwrap().call(Request::Campaign(20))?;
    let mut accepted = BTreeMap::new();
    for phase in 0..2 {
        for sequence in phase * 8 + 1..=phase * 8 + 8 {
            for (group, bucket) in [(10, 0), (20, 1)] {
                let base = batch(sequence);
                let body = Batch::new(
                    base.meta.stream,
                    bucket,
                    base.meta.producer,
                    sequence,
                    Bytes::from(vec![(sequence + group) as u8; 64 << 10]),
                );
                let index = c.append(group, body.clone())?;
                accepted.insert((group, sequence), (index, body));
            }
        }
        if phase == 0 {
            let leader = c.leader(10)?;
            c.nodes.remove(&leader).unwrap().kill()?;
        }
    }
    c.stop_all()?;
    c.start_all()?; // Same original engine files and TLS identities, new processes.
    for group in [10, 20] {
        let (index, body) = &accepted[&(group, 16)];
        c.applied(group, *index)?;
        assert_eq!(c.append(group, body.clone())?, *index);
    }
    c.stop_all()?;
    for id in 1..=3 {
        let node = Node::open(
            &root.join(format!("node-{id}")),
            layout(),
            id,
            Limits::default(),
        )?;
        for (&(group, _), (index, body)) in &accepted {
            assert_eq!(node.read_accepted(group, *index)?, Some(body.clone()));
        }
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn tls_snapshot_process_kill_mid_file_preserves_payloads_after_rejoin_and_full_restart(
) -> Result<()> {
    let root = tempfile::tempdir()?.keep();
    eprintln!(
        "TLS snapshot process qualification artifacts: {}",
        root.display()
    );
    provision(&root)?;
    let mut c = Processes {
        nodes: BTreeMap::new(),
        root: root.clone(),
        incarnation: 0,
    };
    c.start_all()?;
    c.nodes.get_mut(&1).unwrap().call(Request::Campaign(10))?;
    let leader = c.leader(10)?;
    let follower = *c.nodes.keys().rfind(|&&id| id != leader).unwrap();
    let mut stopped = c.nodes.remove(&follower).unwrap();
    let original_endpoints = (stopped.address, stopped.image_address);
    stopped.kill()?;
    let mut expected = Vec::new();
    for sequence in 1..=8 {
        let b = process_batch(sequence);
        expected.push((c.append(10, b.clone())?, b));
    }
    let leader = c.leader(10)?;
    assert!(matches!(
        c.nodes
            .get_mut(&leader)
            .unwrap()
            .call(Request::Checkpoint(10))?,
        Reply::Ready
    ));
    let marker = root.join("received-payload-chunk");
    restart_follower(&mut c, follower, original_endpoints, Some(&marker))?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !marker.exists() {
        ensure!(
            c.nodes
                .get_mut(&follower)
                .unwrap()
                .child
                .try_wait()?
                .is_none(),
            "receiver exited before chunk pause"
        );
        ensure!(
            Instant::now() < deadline,
            "actual TLS payload chunk not observed"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(fs::read_to_string(&marker)?.parse::<u64>()?, 1 << 20);
    assert_eq!(c.status(follower, 10)?.accepted_records, 0);
    // The follower's actual file thread is paused after writing a payload
    // chunk. Healthy replicas must continue committing before it is killed.
    let b = process_batch(9);
    expected.push((c.append(10, b.clone())?, b));
    c.nodes.remove(&follower).unwrap().kill()?;
    let partial_bytes: u64 = fs::read_dir(root.join(format!("node-{follower}")))?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|p| {
            p.file_name()
                .to_string_lossy()
                .starts_with("context-application-")
        })
        .map(|p| p.metadata().map(|m| m.len()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .sum();
    assert!(
        partial_bytes >= 1 << 20,
        "test never wrote an actual partial image"
    );
    for sequence in 10..=12 {
        let b = process_batch(sequence);
        expected.push((c.append(10, b.clone())?, b));
    }
    restart_follower(&mut c, follower, original_endpoints, None)?;
    c.applied(10, expected.last().unwrap().0)?;
    assert!(
        matches!(c.nodes.get_mut(&follower).unwrap().call(Request::ImageStatus)?, Reply::Images { installed } if installed > 0)
    );
    let leader = c.leader(10)?;
    c.nodes.remove(&leader).unwrap().kill()?;
    let b = process_batch(13);
    expected.push((c.append(10, b.clone())?, b));
    c.stop_all()?;
    c.start_all()?;
    let (last, batch) = expected.last().unwrap();
    c.applied(10, *last)?;
    assert_eq!(c.append(10, batch.clone())?, *last);
    c.stop_all()?;
    for id in 1..=3 {
        let node = Node::open(
            &root.join(format!("node-{id}")),
            layout(),
            id,
            Limits::default(),
        )?;
        for (index, batch) in &expected {
            assert_eq!(node.read_accepted(10, *index)?, Some(batch.clone()));
        }
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
fn process_batch(sequence: u64) -> Batch {
    let base = batch(sequence);
    Batch::new(
        base.meta.stream,
        0,
        base.meta.producer,
        sequence,
        Bytes::from(vec![sequence as u8; 1 << 20]),
    )
}
fn restart_follower(
    c: &mut Processes,
    id: u64,
    endpoints: (SocketAddr, SocketAddr),
    pause: Option<&Path>,
) -> Result<()> {
    c.incarnation += 1;
    c.nodes.insert(
        id,
        Process::spawn_at(&c.root, id, c.incarnation, Some(endpoints), pause)?,
    );
    let peers = c.nodes.iter().map(|(&id, n)| (id, n.address)).collect();
    let images = c
        .nodes
        .iter()
        .map(|(&id, n)| (id, n.image_address))
        .collect();
    assert!(matches!(
        c.nodes
            .get_mut(&id)
            .unwrap()
            .call(Request::Start { peers, images })?,
        Reply::Ready
    ));
    Ok(())
}

#[test]
fn client_rpc_survives_leader_sigkill_and_full_original_directory_restart() -> Result<()> {
    let directory = tempfile::tempdir()?;
    provision(directory.path())?;
    let mut processes = Processes {
        nodes: BTreeMap::new(),
        root: directory.path().to_owned(),
        incarnation: 0,
    };
    processes.start_all()?;
    let executor = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let make_client = |processes: &Processes| -> Result<rpc::Client> {
        let (config, _, pins) = tls(&processes.root, 4)?;
        let mut config = (*config).clone();
        config.alpn_protocols = vec![rpc::ALPN.to_vec()];
        let endpoints = processes
            .nodes
            .iter()
            .map(|(&id, node)| {
                (
                    id,
                    rpc::Endpoint {
                        data_address: node.rpc_data.to_string(),
                        control_address: node.rpc_control.to_string(),
                        server_name: format!("wal-{id}.example.invalid"),
                        certificate: pins[&id],
                    },
                )
            })
            .collect();
        rpc::Client::new(
            &layout(),
            &Limits::default(),
            endpoints,
            Arc::new(config),
            rpc::Options::default(),
        )
    };
    let client = make_client(&processes)?;
    let mut records = Vec::new();
    for group in [10, 20] {
        let leader = processes.leader(group)?;
        for sequence in 1..=8 {
            let base = batch(sequence);
            let body = Bytes::from(vec![sequence as u8; 2 << 20]);
            let record = Batch::new(
                base.meta.stream,
                if group == 10 { 0 } else { 1 },
                base.meta.producer,
                sequence,
                body,
            );
            let response = executor.block_on(client.request(
                leader,
                rpc::Request::Append {
                    group,
                    batch: record.clone(),
                },
            ))?;
            let rpc::Reply::Commit(Outcome::Committed(Applied::Accepted { index })) =
                response.reply
            else {
                anyhow::bail!("client did not receive durable ACK");
            };
            records.push((group, index, record));
        }
        processes.applied(group, records.last().unwrap().1)?;
    }
    let leader = processes.leader(10)?;
    let old = records
        .iter()
        .rev()
        .find(|(group, _, _)| *group == 10)
        .unwrap();
    // Populate the actual route cache before killing its current destination.
    let receipt = executor.block_on(client.request_routed(rpc::Request::Append {
        group: 10,
        batch: old.2.clone(),
    }))?;
    assert!(
        matches!(receipt.reply, rpc::Reply::Commit(Outcome::Committed(Applied::Accepted { index })) if index == old.1)
    );
    drop(receipt);
    processes.nodes.remove(&leader).unwrap().kill()?;
    processes.leader(10)?;
    let mut recovered = false;
    for _ in 0..3 {
        // Explicit retry retains identity; the transport does not turn unknown
        // outcomes into a fresh sequence or silently replay the operation.
        if let Ok(response) = executor.block_on(client.request_routed(rpc::Request::Append {
            group: 10,
            batch: old.2.clone(),
        })) {
            if matches!(response.reply, rpc::Reply::Commit(Outcome::Committed(Applied::Accepted { index })) if index == old.1)
            {
                recovered = true;
                break;
            }
        }
    }
    ensure!(recovered, "client could not reconcile after leader loss");
    drop(client);
    processes.stop_all()?;
    processes.start_all()?;
    let client = make_client(&processes)?;
    for group in [10, 20] {
        let leader = processes.leader(group)?;
        let (_, index, record) = records.iter().rev().find(|(g, _, _)| *g == group).unwrap();
        let response = executor.block_on(client.request(
            leader,
            rpc::Request::Append {
                group,
                batch: record.clone(),
            },
        ))?;
        ensure!(
            matches!(response.reply, rpc::Reply::Commit(Outcome::Committed(Applied::Accepted { index: actual })) if actual == *index),
            "retry moved the durable position"
        );
        processes.applied(group, *index)?;
    }
    for node in 1..=3 {
        for (group, index, record) in &records {
            let response = executor.block_on(client.request(
                node,
                rpc::Request::Fetch {
                    group: *group,
                    index: *index,
                },
            ))?;
            ensure!(
                matches!(&response.reply, rpc::Reply::Record(Some(meta)) if meta == &record.meta),
                "recovered record metadata changed"
            );
            ensure!(
                response.body == record.body,
                "recovered complete payload changed"
            );
        }
    }
    processes.stop_all()
}
