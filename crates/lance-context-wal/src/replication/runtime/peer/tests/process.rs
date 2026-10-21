//! Child processes use the actual TLS transport. The parent sends only test
//! control requests over loopback; it never relays consensus packets.
use super::*;
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
    control: SocketAddr,
    pid: u32,
}
#[derive(Serialize, Deserialize)]
enum Request {
    Start(BTreeMap<u64, SocketAddr>),
    Campaign(u64),
    Status(u64),
    Append(u64, Batch),
}
#[derive(Serialize, Deserialize)]
enum Reply {
    Ready,
    Status(crate::replication::Status),
    Append(Outcome),
    Failed(String),
}
const FRAME_LIMIT: usize = 16 << 20;

fn provision(root: &Path) -> Result<()> {
    for id in 1..=3 {
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
    for peer in 1..=3 {
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
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let control = TcpListener::bind("127.0.0.1:0").await?;
    let address = Address {
        peer: listener.local_addr()?,
        control: control.local_addr()?,
        pid: std::process::id(),
    };
    fs::write(
        address_file.with_extension("tmp"),
        bincode::serialize(&address)?,
    )?;
    fs::rename(address_file.with_extension("tmp"), &address_file)?;
    let (mut stream, _) = control.accept().await?;
    let Request::Start(addresses) = receive(&mut stream).await? else {
        anyhow::bail!("missing startup configuration");
    };
    let (client, server, pins) = tls(&root, id)?;
    let node = Node::open(
        &root.join(format!("node-{id}")),
        layout(),
        id,
        Limits::default(),
    )?;
    let (runtime, outbox) = Runtime::spawn(node, super::super::super::Options::default())?;
    let handle = runtime.handle();
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
    let network = Transport::spawn(
        handle.clone(),
        outbox,
        listener,
        peers,
        client,
        server,
        Options::default(),
    )?;
    respond(&mut stream, Reply::Ready).await?;
    loop {
        let request = match receive(&mut stream).await {
            Ok(request) => request,
            Err(_) => {
                network.shutdown().await?;
                return runtime.shutdown().await;
            }
        };
        let result: Result<Reply> = async {
            Ok(match request {
                Request::Start(_) => anyhow::bail!("duplicate process startup"),
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
}
impl Process {
    fn spawn(root: &Path, id: u64, incarnation: u64) -> Result<Self> {
        let address_file = root.join(format!("address-{id}-{incarnation}"));
        let mut child = ProcessCommand::new(std::env::current_exe()?)
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
        let result = (|| -> Result<(ControlStream, SocketAddr)> {
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
            Ok((stream, address.peer))
        })();
        match result {
            Ok((stream, address)) => Ok(Self {
                child,
                stream,
                address,
            }),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(e)
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
        for node in self.nodes.values_mut() {
            ensure!(
                matches!(node.call(Request::Start(addresses.clone()))?, Reply::Ready),
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
