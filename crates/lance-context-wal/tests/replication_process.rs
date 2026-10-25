#![cfg(feature = "replication")]
//! Real child processes and files; the parent relays Raft packets over TCP and
//! deliberately drops selected links. No simulated consensus or storage engine.
use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use lance_context_wal::{
    placement::Layout,
    record::Batch,
    replication::{
        checkpoint::{Image, Part},
        publication::{
            Acquire, Descriptors, Destination, Plan, PlanRequest, PrepareRequest,
            Rejection as PublisherRejection, Reply as PublisherReply, SourceRange, Table,
        },
        query::{Completion as ReadCompletion, Outcome as ReadOutcome, Query, Value},
        source::{CatalogSigner, Permit, Reply as SourceReply, SourceSigner},
        Applied, Completion, Events, Limits, Node, Outcome, Packet, Page, PreparedCheckpoint,
        SnapshotReceiver, Status,
    },
};
use protobuf::Message as _;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;

#[derive(Serialize, Deserialize)]
struct WirePacket {
    cluster: Uuid,
    layout: [u8; 32],
    policy: [u8; 32],
    group: u64,
    message: Vec<u8>,
}
impl WirePacket {
    fn encode(packet: Packet) -> Result<Self> {
        Ok(Self {
            cluster: packet.cluster,
            layout: packet.layout,
            policy: packet.policy,
            group: packet.group,
            message: packet.message.write_to_bytes()?,
        })
    }
    fn decode(self) -> Result<Packet> {
        Ok(Packet {
            cluster: self.cluster,
            layout: self.layout,
            policy: self.policy,
            group: self.group,
            message: raft_proto::eraftpb::Message::parse_from_bytes(&self.message)?,
        })
    }
}
#[derive(Serialize, Deserialize)]
enum Request {
    Producer(u64, lance_context_wal::replication::producer::Command),
    RetireManifestHistory(
        u64,
        lance_context_wal::replication::publication::RetireManifestHistory,
    ),
    AdvancePublication(
        u64,
        lance_context_wal::replication::publication::AdvancePublication,
    ),
    CloseSource(u64, lance_context_wal::replication::source::ClosurePermit),
    ConfirmPublication(
        u64,
        lance_context_wal::replication::publication::ConfirmPublication,
    ),
    BootstrapManifest(
        u64,
        lance_context_wal::replication::publication::BootstrapManifest,
    ),
    ReserveManifest(
        u64,
        lance_context_wal::replication::publication::ReserveManifest,
    ),
    Prepare(u64, PrepareRequest),
    Pin(u64, Permit),
    Query(u64, u64, Query),
    Plan(u64, PlanRequest),
    Acquire(u64, Acquire),
    Publisher(Uuid),
    Campaign(u64),
    Propose(u64, u64, Batch),
    Step(WirePacket),
    Tick,
    Status(u64),
    Read(u64, u64),
    Page(u64),
    PrepareCheckpoint(u64, Uuid),
    InstallCheckpoint,
    MaintainLog,
    ImageInfo(Vec<u8>),
    ImageChunk(Vec<u8>, bool, u64, usize),
    BeginSnapshot(WirePacket, Uuid),
    ReceiveChunk(bool, u64, Bytes),
    FinishSnapshot,
    ReportSnapshot(u64, u64, bool),
    CollectImages,
}
#[derive(Serialize, Deserialize)]
enum Response {
    Publisher(Option<Box<Table>>),
    Events(Vec<WirePacket>, Vec<Completion>, Vec<ReadCompletion>),
    Status(Status),
    Record(Option<Batch>),
    Page(Page),
    Error(String),
    Image(Vec<u8>, u64, u64),
    Chunk(Bytes),
    CollectedImages(u64, u64),
    LogMaintained(Vec<u64>),
    Ok,
}
fn send<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    let bytes = bincode::serialize(value)?;
    ensure!(bytes.len() <= 32 * 1024 * 1024, "oversized test frame");
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}
fn receive<T: DeserializeOwned>(stream: &mut TcpStream) -> Result<T> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    ensure!(len <= 32 * 1024 * 1024, "oversized test frame");
    let mut bytes = vec![0; len];
    stream.read_exact(&mut bytes)?;
    Ok(bincode::deserialize(&bytes)?)
}
struct Worker {
    node: Node,
    prepared: Option<PreparedCheckpoint>,
    receiver: Option<SnapshotReceiver>,
    images: BTreeMap<Vec<u8>, Arc<Image>>,
}

fn part(payload: bool) -> Part {
    if payload {
        Part::Payload
    } else {
        Part::Metadata
    }
}

fn run(worker: &mut Worker, request: Request) -> Result<Response> {
    let node = &mut worker.node;
    match request {
        Request::RetireManifestHistory(token, request) => {
            node.retire_manifest_history(token, request)?
        }
        Request::Producer(token, command) => node.producer_command(token, command)?,
        Request::CloseSource(token, permit) => node.close_source(token, permit)?,
        Request::ConfirmPublication(token, request) => node.confirm_publication(token, request)?,
        Request::AdvancePublication(token, request) => node.advance_publication(token, request)?,
        Request::BootstrapManifest(token, request) => node.bootstrap_manifest(token, request)?,
        Request::ReserveManifest(token, request) => node.reserve_manifest(token, request)?,
        Request::Prepare(token, prepared) => node.prepare_publication(token, prepared)?,
        Request::Pin(token, permit) => node.pin_source(token, permit)?,
        Request::Query(group, token, query) => node.query(group, token, query)?,
        Request::Plan(token, plan) => node.plan_publication(token, plan)?,
        Request::Acquire(token, acquire) => node.acquire_publisher(token, acquire)?,
        Request::Publisher(table) => {
            return Ok(Response::Publisher(
                node.publisher(table)?.cloned().map(Box::new),
            ))
        }
        Request::Campaign(group) => node.campaign(group)?,
        Request::Propose(group, token, batch) => node.propose(group, token, batch)?,
        Request::Step(packet) => node.step(packet.decode()?)?,
        Request::Tick => node.tick()?,
        Request::Status(group) => return Ok(Response::Status(node.status(group)?)),
        Request::Read(group, index) => {
            return Ok(Response::Record(node.read_accepted(group, index)?))
        }
        Request::Page(group) => return Ok(Response::Page(node.accepted(group, 0, 128)?)),
        Request::PrepareCheckpoint(group, id) => {
            ensure!(worker.prepared.is_none(), "existing prepared image");
            let prepared = node.prepare_checkpoint(group, id)?.run()?;
            let image = prepared.image();
            let snapshot = image.snapshot()?.write_to_bytes()?;
            let response = Response::Image(
                snapshot.clone(),
                image.bytes(Part::Metadata),
                image.bytes(Part::Payload),
            );
            worker.images.insert(snapshot, image);
            worker.prepared = Some(prepared);
            return Ok(response);
        }
        Request::InstallCheckpoint => {
            node.install_checkpoint(worker.prepared.take().context("no prepared image")?)?
        }
        Request::CollectImages => {
            let mut task = node.prepare_image_collection()?;
            let mut files = 0;
            let mut bytes = 0;
            for _ in 0..4096 {
                let batch = task.run_batch(1)?;
                files += batch.removed_files;
                bytes += batch.removed_bytes;
                if batch.finished {
                    return Ok(Response::CollectedImages(files, bytes));
                }
            }
            anyhow::bail!("test collection did not finish")
        }
        Request::MaintainLog => {
            return Ok(Response::LogMaintained(
                node.prepare_log_maintenance()?.run()?,
            ));
        }
        Request::ImageInfo(snapshot) => {
            let image = worker
                .images
                .get(&snapshot)
                .context("no captured transfer image")?;
            return Ok(Response::Image(
                snapshot,
                image.bytes(Part::Metadata),
                image.bytes(Part::Payload),
            ));
        }
        Request::ImageChunk(snapshot, payload, offset, bytes) => {
            let image = worker
                .images
                .get(&snapshot)
                .context("no captured transfer image")?;
            return Ok(Response::Chunk(image.chunk(
                part(payload),
                offset,
                bytes,
            )?));
        }
        Request::BeginSnapshot(packet, id) => {
            ensure!(worker.receiver.is_none(), "existing receiver");
            worker.receiver = Some(node.receive_snapshot(packet.decode()?, id)?);
            return Ok(Response::Ok);
        }
        Request::ReceiveChunk(payload, offset, bytes) => {
            worker.receiver.as_mut().context("no receiver")?.write(
                part(payload),
                offset,
                &bytes,
            )?;
            return Ok(Response::Ok);
        }
        Request::FinishSnapshot => {
            node.step_snapshot(worker.receiver.take().context("no receiver")?.finish()?)?
        }
        Request::ReportSnapshot(group, to, delivered) => {
            node.report_snapshot(group, to, delivered)?
        }
    }
    let Events {
        messages,
        completions,
        reads,
    } = node.poll()?;
    for packet in &messages {
        if packet.message.get_msg_type() == raft_proto::eraftpb::MessageType::MsgSnapshot {
            let snapshot = packet.message.get_snapshot();
            worker.images.insert(
                snapshot.write_to_bytes()?,
                node.snapshot_image(packet.group, snapshot)?,
            );
        }
    }
    Ok(Response::Events(
        messages
            .into_iter()
            .map(WirePacket::encode)
            .collect::<Result<_>>()?,
        completions,
        reads,
    ))
}

#[test]
#[ignore = "child fixture, invoked by the process qualification"]
fn replication_peer_fixture() -> Result<()> {
    let Ok(root) = std::env::var("CONTEXT_REPLICATION_FIXTURE") else {
        return Ok(());
    };
    let root = PathBuf::from(root);
    let id = std::env::var("CONTEXT_REPLICATION_NODE")?.parse()?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let limits_path = root.join("limits.json");
    let limits: Limits = if limits_path.exists() {
        serde_json::from_slice(&std::fs::read(limits_path)?)?
    } else {
        Limits::default()
    };
    let mut node = Node::open_with_storage_options(
        &root.join(format!("data-{id}")),
        Arc::new(serde_json::from_slice(&std::fs::read(
            root.join("layout.json"),
        )?)?),
        id,
        limits.clone(),
        // Exercise actual segment rotation and physical removal with the small
        // process fixtures; these thresholds are not benchmark configuration.
        lance_context_wal::raft_log::LogOptions {
            target_file_bytes: 4096,
            purge_threshold_bytes: 4096,
        },
    )?;
    let key_path = root.join(format!("catalog-key-{id}.pkcs8"));
    if key_path.exists() {
        node.set_catalog_signer(Arc::new(CatalogSigner::from_pkcs8(&std::fs::read(
            key_path,
        )?)?))?;
    }
    for group in limits.source_keys.keys() {
        let key_path = root.join(format!("source-key-{group}-{id}.pkcs8"));
        if key_path.exists() {
            node.set_source_signer(
                *group,
                Arc::new(SourceSigner::from_pkcs8(&std::fs::read(key_path)?)?),
            )?;
        }
    }
    let mut worker = Worker {
        node,
        prepared: None,
        receiver: None,
        images: BTreeMap::new(),
    };
    let address = PathBuf::from(std::env::var("CONTEXT_REPLICATION_ADDRESS_FILE")?);
    let temporary = address.with_extension("tmp");
    std::fs::write(&temporary, listener.local_addr()?.to_string())?;
    std::fs::rename(temporary, address)?;
    let (mut stream, _) = listener.accept()?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    loop {
        let request = receive(&mut stream)?;
        let response =
            run(&mut worker, request).unwrap_or_else(|error| Response::Error(format!("{error:#}")));
        send(&mut stream, &response)?;
    }
}

struct Peer {
    child: Child,
    stream: TcpStream,
}
impl Peer {
    fn spawn(root: &Path, id: u64, incarnation: u64) -> Result<Self> {
        let address = root.join(format!("address-{id}-{incarnation}"));
        let mut child = Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "replication_peer_fixture",
                "--ignored",
                "--nocapture",
            ])
            .env("CONTEXT_REPLICATION_FIXTURE", root)
            .env("CONTEXT_REPLICATION_NODE", id.to_string())
            .env("CONTEXT_REPLICATION_ADDRESS_FILE", &address)
            .stdout(Stdio::from(File::create(
                root.join(format!("stdout-{id}-{incarnation}")),
            )?))
            .stderr(Stdio::from(File::create(
                root.join(format!("stderr-{id}-{incarnation}")),
            )?))
            .spawn()?;
        let result = (|| -> Result<TcpStream> {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !address.exists() {
                ensure!(child.try_wait()?.is_none(), "peer exited during startup");
                ensure!(Instant::now() < deadline, "peer startup outcome unknown");
                std::thread::sleep(Duration::from_millis(5));
            }
            let stream = TcpStream::connect(std::fs::read_to_string(address)?)?;
            stream.set_nodelay(true)?;
            stream.set_read_timeout(Some(Duration::from_secs(10)))?;
            stream.set_write_timeout(Some(Duration::from_secs(10)))?;
            Ok(stream)
        })();
        match result {
            Ok(stream) => Ok(Self { child, stream }),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(error)
            }
        }
    }
    fn call(&mut self, request: Request) -> Result<Response> {
        send(&mut self.stream, &request)?;
        match receive(&mut self.stream)? {
            Response::Error(error) => anyhow::bail!("peer rejected request: {error}"),
            response => Ok(response),
        }
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct Cluster {
    // Peers must die before TempDir removes their exclusively owned files.
    peers: BTreeMap<u64, Peer>,
    root: tempfile::TempDir,
    incarnations: BTreeMap<u64, u64>,
    queue: VecDeque<Packet>,
    completions: Vec<Completion>,
    reads: Vec<ReadCompletion>,
    transfers: u64,
    interrupt_next_snapshot: bool,
}
impl Cluster {
    fn new() -> Result<Self> {
        let layout = Layout {
            cluster: Uuid::from_u128(50),
            nodes: (1..=3)
                .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
                .collect(),
            groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
            buckets: vec![10, 20],
        };
        Self::with_layout(layout, None)
    }
    fn with_layout(layout: Layout, catalog_key: Option<&[u8]>) -> Result<Self> {
        Self::with_producer_slots(layout, catalog_key, 0)
    }
    fn with_producer_slots(
        layout: Layout,
        catalog_key: Option<&[u8]>,
        producer_slots: u32,
    ) -> Result<Self> {
        let root = tempfile::tempdir()?;
        if let Some(key) = catalog_key {
            let signer = CatalogSigner::from_pkcs8(key)?;
            let mut source_keys = BTreeMap::new();
            for (group, members) in &layout.groups {
                let source_key = ring::signature::Ed25519KeyPair::generate_pkcs8(
                    &ring::rand::SystemRandom::new(),
                )
                .map_err(|_| anyhow::anyhow!("test source key generation failed"))?;
                let source_signer = SourceSigner::from_pkcs8(source_key.as_ref())?;
                source_keys.insert(*group, source_signer.public_key());
                for id in members {
                    std::fs::write(
                        root.path().join(format!("source-key-{group}-{id}.pkcs8")),
                        source_key.as_ref(),
                    )?;
                }
            }
            std::fs::write(
                root.path().join("limits.json"),
                serde_json::to_vec(&Limits {
                    catalog_key: Some(signer.public_key()),
                    source_keys,
                    producer_slots,
                    ..Default::default()
                })?,
            )?;
            for id in &layout.groups[layout.groups.first_key_value().unwrap().0] {
                std::fs::write(root.path().join(format!("catalog-key-{id}.pkcs8")), key)?;
            }
        }
        if catalog_key.is_none() && producer_slots > 0 {
            std::fs::write(
                root.path().join("limits.json"),
                serde_json::to_vec(&Limits {
                    producer_slots,
                    ..Default::default()
                })?,
            )?;
        }
        std::fs::write(
            root.path().join("layout.json"),
            serde_json::to_vec(&layout)?,
        )?;
        let mut this = Self {
            peers: BTreeMap::new(),
            root,
            incarnations: BTreeMap::new(),
            queue: VecDeque::new(),
            completions: Vec::new(),
            reads: Vec::new(),
            transfers: 0,
            interrupt_next_snapshot: false,
        };
        for id in layout.nodes.keys() {
            this.start(*id)?;
        }
        Ok(this)
    }
    fn start(&mut self, id: u64) -> Result<()> {
        ensure!(
            !self.peers.contains_key(&id),
            "cannot launch an existing peer"
        );
        let incarnation = self.incarnations.entry(id).or_default();
        *incarnation += 1;
        self.peers
            .insert(id, Peer::spawn(self.root.path(), id, *incarnation)?);
        Ok(())
    }
    fn command(&mut self, id: u64, request: Request) -> Result<()> {
        let context = match &request {
            Request::Propose(group, token, _) => format!("propose group {group}, token {token}"),
            Request::Query(group, token, _) => format!("query group {group}, token {token}"),
            Request::Acquire(token, _) => format!("acquire token {token}"),
            Request::Plan(token, _) => format!("plan token {token}"),
            Request::Prepare(token, _) => format!("prepare token {token}"),
            Request::Pin(token, _) => format!("pin token {token}"),
            _ => "replication control".to_string(),
        };
        let Response::Events(packets, completions, reads) = self
            .peers
            .get_mut(&id)
            .context("missing peer")?
            .call(request)
            .with_context(|| format!("peer {id}: {context}"))?
        else {
            anyhow::bail!("wrong reply");
        };
        for packet in packets {
            self.queue.push_back(packet.decode()?);
        }
        self.completions.extend(completions);
        self.reads.extend(reads);
        Ok(())
    }
    fn pump(&mut self, allowed: &[u64]) -> Result<()> {
        for _ in 0..5000 {
            let Some(packet) = self.queue.pop_front() else {
                return Ok(());
            };
            if allowed.contains(&packet.message.from)
                && allowed.contains(&packet.message.to)
                && self.peers.contains_key(&packet.message.to)
            {
                if packet.message.get_msg_type() == raft_proto::eraftpb::MessageType::MsgSnapshot {
                    self.transfer(packet)?;
                } else {
                    self.command(
                        packet.message.to,
                        Request::Step(WirePacket::encode(packet)?),
                    )?;
                }
            } else if packet.message.get_msg_type() == raft_proto::eraftpb::MessageType::MsgSnapshot
                && allowed.contains(&packet.message.from)
                && self.peers.contains_key(&packet.message.from)
            {
                self.command(
                    packet.message.from,
                    Request::ReportSnapshot(packet.group, packet.message.to, false),
                )?;
            }
        }
        anyhow::bail!("test message queue did not settle")
    }

    fn transfer(&mut self, packet: Packet) -> Result<()> {
        let from = packet.message.from;
        let to = packet.message.to;
        let group = packet.group;
        let snapshot = packet.message.get_snapshot().write_to_bytes()?;
        let Response::Image(_, metadata, payload) = self
            .peers
            .get_mut(&from)
            .context("missing image source")?
            .call(Request::ImageInfo(snapshot.clone()))?
        else {
            anyhow::bail!("wrong image info");
        };
        self.transfers += 1;
        let response = self
            .peers
            .get_mut(&to)
            .unwrap()
            .call(Request::BeginSnapshot(
                WirePacket::encode(packet)?,
                Uuid::from_u128(10_000 + u128::from(self.transfers)),
            ))?;
        ensure!(
            matches!(response, Response::Ok),
            "snapshot receiver did not start"
        );
        for (is_payload, total) in [(false, metadata), (true, payload)] {
            let mut offset = 0;
            while offset < total {
                let size = if self.interrupt_next_snapshot {
                    8
                } else {
                    (total - offset).min(64 * 1024) as usize
                };
                let Response::Chunk(bytes) =
                    self.peers
                        .get_mut(&from)
                        .unwrap()
                        .call(Request::ImageChunk(
                            snapshot.clone(),
                            is_payload,
                            offset,
                            size,
                        ))?
                else {
                    anyhow::bail!("wrong image chunk");
                };
                ensure!(bytes.len() == size, "short transfer chunk");
                let response = self
                    .peers
                    .get_mut(&to)
                    .unwrap()
                    .call(Request::ReceiveChunk(is_payload, offset, bytes))?;
                ensure!(matches!(response, Response::Ok), "snapshot chunk rejected");
                if self.interrupt_next_snapshot {
                    self.interrupt_next_snapshot = false;
                    // SIGKILL leaves only an incomplete, uninstalled image. The
                    // restart below must use the original durable committed state.
                    drop(self.peers.remove(&to));
                    self.command(from, Request::ReportSnapshot(group, to, false))?;
                    return Ok(());
                }
                offset += size as u64;
            }
        }
        self.command(to, Request::FinishSnapshot)?;
        self.command(from, Request::ReportSnapshot(group, to, true))?;
        Ok(())
    }
    fn outcome(&mut self, token: u64) -> Option<Outcome> {
        self.completions
            .iter()
            .position(|c| c.request == token)
            .map(|position| self.completions.remove(position).outcome)
    }
    fn read_outcome(&mut self, token: u64) -> Option<ReadOutcome> {
        self.reads
            .iter()
            .position(|c| c.request == token)
            .map(|position| self.reads.remove(position).outcome)
    }
    fn elect(&mut self, id: u64, group: u64, allowed: &[u64]) -> Result<u64> {
        self.command(id, Request::Campaign(group))?;
        self.pump(allowed)?;
        for _ in 0..100 {
            let mut views = Vec::new();
            for node in allowed {
                let Response::Status(status) = self
                    .peers
                    .get_mut(node)
                    .unwrap()
                    .call(Request::Status(group))?
                else {
                    anyhow::bail!("wrong status");
                };
                views.push(status);
            }
            let leader = views[0].leader;
            if allowed.contains(&leader)
                && views
                    .iter()
                    .all(|s| s.leader == leader && s.term == views[0].term)
            {
                return Ok(leader);
            }
            // Expire the old election lease; do not assume which randomized
            // timer wins when both eligible replicas have identical logs.
            for node in allowed {
                self.command(*node, Request::Tick)?;
            }
            self.pump(allowed)?;
        }
        anyhow::bail!("majority did not elect a leader")
    }
}

fn input(bucket: u32, sequence: u64, fill: u8) -> Batch {
    Batch::new(
        Uuid::from_u128(80),
        bucket,
        Uuid::from_u128(90),
        sequence,
        Bytes::from(vec![fill; 64 * 1024]),
    )
}

fn verify_records(peer: &mut Peer, group: u64, expected: &[Batch]) -> Result<()> {
    let Response::Page(page) = peer.call(Request::Page(group))? else {
        anyhow::bail!("wrong page");
    };
    assert_eq!(page.entries.len(), expected.len());
    for ((index, meta), expected) in page.entries.into_iter().zip(expected) {
        assert_eq!(&meta, &expected.meta);
        let Response::Record(Some(actual)) = peer.call(Request::Read(group, index))? else {
            anyhow::bail!("missing payload");
        };
        assert_eq!(&actual, expected);
    }
    Ok(())
}

#[test]
fn application_snapshot_transfer_interruption_and_full_process_restart() -> Result<()> {
    let mut c = Cluster::new()?;
    let first = input(0, 1, 31);
    let second = input(0, 2, 32);
    let sibling = input(1, 1, 33);
    c.command(1, Request::Campaign(10))?;
    c.command(1, Request::Campaign(20))?;
    c.pump(&[1, 2, 3])?;
    c.command(1, Request::Propose(20, 100, sibling.clone()))?;
    c.pump(&[1, 2, 3])?;
    c.command(1, Request::Propose(10, 101, first.clone()))?;
    c.pump(&[1, 2])?;
    assert!(matches!(
        c.outcome(101),
        Some(Outcome::Committed(Applied::Accepted { .. }))
    ));
    let response = c
        .peers
        .get_mut(&1)
        .unwrap()
        .call(Request::PrepareCheckpoint(10, Uuid::from_u128(501)))?;
    assert!(matches!(response, Response::Image(_, _, _)));
    // Kill after complete file sync but BEFORE the Raft descriptor transition.
    drop(c.peers.remove(&1));
    c.start(1)?;
    assert!(
        matches!(c.peers.get_mut(&1).unwrap().call(Request::CollectImages)?, Response::CollectedImages(2, bytes) if bytes > 0)
    );
    verify_records(
        c.peers.get_mut(&1).unwrap(),
        10,
        std::slice::from_ref(&first),
    )?;
    let leader = c.elect(1, 10, &[1, 2])?;
    let response = c
        .peers
        .get_mut(&leader)
        .unwrap()
        .call(Request::PrepareCheckpoint(10, Uuid::from_u128(502)))?;
    assert!(matches!(response, Response::Image(_, _, _)));
    c.command(leader, Request::InstallCheckpoint)?;
    c.pump(&[1, 2])?;
    // Keep a real suffix after the packed cut so restart/transfer must use both.
    c.command(leader, Request::Propose(10, 102, second.clone()))?;
    c.pump(&[1, 2])?;
    let second_ack = c.outcome(102).context("second append had no ACK")?;
    assert!(matches!(
        second_ack,
        Outcome::Committed(Applied::Accepted { .. })
    ));
    let directory = c.root.path().join(format!("data-{leader}"));
    let mut old_segments = Vec::new();
    for entry in std::fs::read_dir(&directory)? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "raftlog")
        {
            old_segments.push(path);
        }
    }
    for _ in 0..4 {
        assert!(matches!(
            c.peers
                .get_mut(&leader)
                .unwrap()
                .call(Request::MaintainLog)?,
            Response::LogMaintained(_)
        ));
    }
    assert!(
        old_segments.iter().any(|path| !path.exists()),
        "native maintenance did not remove a physical segment"
    );
    // The application image and suffix must still serve the lagging replica
    // after physical removal, interrupted transfer and every process restarting.
    c.interrupt_next_snapshot = true;
    for _ in 0..30 {
        c.command(leader, Request::Tick)?;
        c.pump(&[1, 2, 3])?;
        if !c.peers.contains_key(&3) {
            break;
        }
    }
    assert!(
        !c.peers.contains_key(&3),
        "partial snapshot kill was not reached"
    );
    assert_eq!(c.transfers, 1);
    // The first chunk can still be buffered when SIGKILL arrives. Check the
    // actual orphan lengths, including zero-byte files, rather than assuming
    // received bytes were flushed to storage.
    let abandoned: Vec<_> = ["state", "pack"]
        .into_iter()
        .map(|part| {
            c.root.path().join("data-3").join(format!(
                "context-application-10-{}.{part}",
                Uuid::from_u128(10_001)
            ))
        })
        .collect();
    let mut abandoned_bytes = 0;
    for path in &abandoned {
        abandoned_bytes += std::fs::metadata(path)?.len();
    }
    c.start(3)?;
    assert!(
        matches!(c.peers.get_mut(&3).unwrap().call(Request::CollectImages)?, Response::CollectedImages(2, bytes) if bytes == abandoned_bytes)
    );
    assert!(abandoned.iter().all(|path| !path.exists()));
    verify_records(c.peers.get_mut(&3).unwrap(), 10, &[])?;
    verify_records(
        c.peers.get_mut(&3).unwrap(),
        20,
        std::slice::from_ref(&sibling),
    )?;
    for _ in 0..30 {
        c.command(leader, Request::Tick)?;
        c.pump(&[1, 2, 3])?;
        let Response::Status(status) = c.peers.get_mut(&3).unwrap().call(Request::Status(10))?
        else {
            anyhow::bail!("wrong status");
        };
        if status.accepted_records == 2 {
            break;
        }
    }
    assert!(c.transfers >= 2, "snapshot retry was not exercised");
    for peer in c.peers.values_mut() {
        verify_records(peer, 10, &[first.clone(), second.clone()])?;
        verify_records(peer, 20, std::slice::from_ref(&sibling))?;
    }
    // Kill all peers after descriptor installation; preserve original volumes.
    c.queue.clear();
    c.peers.clear();
    for id in 1..=3 {
        c.start(id)?;
    }
    for peer in c.peers.values_mut() {
        assert!(matches!(
            peer.call(Request::CollectImages)?,
            Response::CollectedImages(0, 0)
        ));
        verify_records(peer, 10, &[first.clone(), second.clone()])?;
        verify_records(peer, 20, std::slice::from_ref(&sibling))?;
    }
    let leader = c.elect(3, 10, &[1, 2, 3])?;
    c.command(leader, Request::Propose(10, 103, second.clone()))?;
    c.pump(&[1, 2, 3])?;
    assert_eq!(
        c.outcome(103),
        Some(second_ack),
        "snapshot lost the original retry receipt"
    );
    let third = input(0, 3, 34);
    c.command(leader, Request::Propose(10, 104, third.clone()))?;
    c.pump(&[1, 2, 3])?;
    assert!(matches!(
        c.outcome(104),
        Some(Outcome::Committed(Applied::Accepted { .. }))
    ));
    for peer in c.peers.values_mut() {
        verify_records(peer, 10, &[first.clone(), second.clone(), third.clone()])?;
    }
    Ok(())
}

#[test]
fn durable_quorum_retry_leader_loss_suffix_replacement_and_full_restart() -> Result<()> {
    let mut c = Cluster::new()?;
    c.command(1, Request::Campaign(10))?;
    c.pump(&[1, 2, 3])?;
    let first = input(0, 1, 7);
    c.command(1, Request::Propose(10, 1, first.clone()))?;
    assert!(
        c.outcome(1).is_none(),
        "local append was mistaken for quorum ACK"
    );
    c.pump(&[1, 2])?;
    let first_ack = c.outcome(1).context("no majority ACK")?;
    assert!(matches!(
        first_ack,
        Outcome::Committed(Applied::Accepted { .. })
    ));
    // Only one follower held the acknowledged input when its leader was killed.
    drop(c.peers.remove(&1));
    assert_eq!(
        c.elect(2, 10, &[2, 3])?,
        2,
        "only replica 2 has the committed suffix"
    );
    c.command(2, Request::Propose(10, 2, first.clone()))?;
    c.pump(&[2, 3])?;
    assert_eq!(
        c.outcome(2),
        Some(first_ack.clone()),
        "ambiguous client retry lost original receipt"
    );
    c.start(1)?;
    for _ in 0..4 {
        c.command(2, Request::Tick)?;
        c.pump(&[1, 2, 3])?;
    }
    // A still-self-declared but isolated leader cannot acknowledge a local retry.
    c.command(2, Request::Propose(10, 3, first.clone()))?;
    c.pump(&[2])?;
    assert!(c.outcome(3).is_none());
    let abandoned = input(0, 2, 99);
    c.command(2, Request::Propose(10, 4, abandoned))?;
    c.pump(&[2])?;
    assert!(c.outcome(4).is_none());
    let replacement = c.elect(1, 10, &[1, 3])?;
    let second = input(0, 2, 8);
    c.command(replacement, Request::Propose(10, 5, second.clone()))?;
    c.pump(&[1, 3])?;
    assert!(matches!(
        c.outcome(5),
        Some(Outcome::Committed(Applied::Accepted { .. }))
    ));
    for _ in 0..4 {
        c.command(replacement, Request::Tick)?;
        c.pump(&[1, 2, 3])?;
    }
    assert_eq!(c.outcome(3), Some(Outcome::Unknown));
    assert_eq!(
        c.outcome(4),
        Some(Outcome::Unknown),
        "replaced suffix completed the wrong input"
    );
    let other_leader = c.elect(replacement, 20, &[1, 2, 3])?;
    let separate = input(1, 1, 9);
    c.command(other_leader, Request::Propose(20, 6, separate.clone()))?;
    c.pump(&[1, 2, 3])?;
    let other_outcome = c.outcome(6);
    assert!(
        matches!(
            other_outcome,
            Some(Outcome::Committed(Applied::Accepted { .. }))
        ),
        "other group outcome: {other_outcome:?}"
    );
    // Kill every process, retain its original files, then reconstruct metadata
    // and exact payloads solely from its durable committed prefix.
    c.queue.clear();
    c.peers.clear();
    for id in 1..=3 {
        c.start(id)?;
    }
    for peer in c.peers.values_mut() {
        for (group, expected) in [(10, vec![&first, &second]), (20, vec![&separate])] {
            let Response::Page(page) = peer.call(Request::Page(group))? else {
                anyhow::bail!("wrong page");
            };
            assert_eq!(page.entries.len(), expected.len());
            for ((index, meta), batch) in page.entries.into_iter().zip(expected) {
                assert_eq!(&meta, &batch.meta);
                let Response::Record(Some(actual)) = peer.call(Request::Read(group, index))? else {
                    anyhow::bail!("missing record");
                };
                assert_eq!(&actual, batch);
            }
        }
    }
    c.command(1, Request::Campaign(10))?;
    c.pump(&[1, 2, 3])?;
    c.command(1, Request::Propose(10, 7, second))?;
    c.pump(&[1, 2, 3])?;
    assert!(matches!(
        c.outcome(7),
        Some(Outcome::Committed(Applied::Accepted { .. }))
    ));
    println!("verified real RF3 processes: durable majority ACK, latest retry, isolated old leader, replaced uncommitted suffix, two groups and full process restart with exact payloads");
    Ok(())
}

#[test]
fn publisher_epochs_require_quorum_and_survive_snapshot_transfer_and_restart() -> Result<()> {
    use lance_context_wal::ingest_contract::{IngestContract, WritePolicy};
    let mut c = Cluster::new()?;
    let mut acquire = Acquire {
        table: Uuid::from_u128(700),
        destination: Destination {
            namespace: Uuid::from_u128(701),
            dataset: "tables/example".into(),
        },
        ingest: IngestContract {
            tenant: "test".into(),
            qualified_table: "example".into(),
            schema: [9; 32],
            policy: WritePolicy::ImmutableUniqueIds,
            max_record_bytes: 1024 * 1024,
            max_rows: 1024,
        },
        owner: Uuid::from_u128(702),
        request: Uuid::from_u128(703),
        expected_epoch: 0,
    };
    c.command(1, Request::Campaign(10))?;
    c.pump(&[1, 2, 3])?;
    c.command(1, Request::Acquire(700, acquire.clone()))?;
    assert!(
        c.outcome(700).is_none(),
        "local acquisition released authority without quorum"
    );
    c.pump(&[1, 2])?;
    let first = c.outcome(700).context("missing committed authority")?;
    let Outcome::Committed(Applied::Publisher(PublisherReply::Acquired(first_authority))) = &first
    else {
        anyhow::bail!("unexpected acquisition {first:?}")
    };
    assert_eq!(first_authority.epoch, 1);
    drop(c.peers.remove(&1));
    assert_eq!(c.elect(2, 10, &[2, 3])?, 2);
    c.command(2, Request::Acquire(701, acquire.clone()))?;
    c.pump(&[2, 3])?;
    assert_eq!(c.outcome(701), Some(first));

    // A competing owner wins only at the exact epoch. The binding is immutable.
    acquire.owner = Uuid::from_u128(704);
    acquire.request = Uuid::from_u128(705);
    acquire.expected_epoch = 1;
    c.command(2, Request::Acquire(702, acquire.clone()))?;
    c.pump(&[2, 3])?;
    let second = c.outcome(702).context("takeover did not commit")?;
    assert!(
        matches!(&second, Outcome::Committed(Applied::Publisher(PublisherReply::Acquired(a))) if a.epoch == 2)
    );
    let Outcome::Committed(Applied::Publisher(PublisherReply::Acquired(authority))) = &second
    else {
        anyhow::bail!("missing takeover authority");
    };
    // Catalog election ticks all hosted groups. The source group may already
    // have another leader; a campaign request alone does not transfer its lease.
    let source_leader = c.elect(2, 20, &[2, 3])?;
    let mut source = input(1, 1, 99);
    source.meta.stream = acquire.table;
    c.command(source_leader, Request::Propose(20, 710, source.clone()))?;
    c.pump(&[2, 3])?;
    let Some(Outcome::Committed(Applied::Accepted {
        index: source_index,
    })) = c.outcome(710)
    else {
        anyhow::bail!("source did not commit");
    };
    let mut plan = PlanRequest {
        authority: authority.clone(),
        plan: Plan {
            transaction: Uuid::from_u128(720),
            read_version: 7,
            sources: BTreeMap::from([(
                20,
                SourceRange {
                    after: 0,
                    through: source_index,
                    records: vec![(source_index, source.meta.clone())],
                },
            )]),
            next_group: 10,
        },
    };
    c.command(2, Request::Plan(711, plan.clone()))?;
    assert!(
        c.outcome(711).is_none(),
        "local plan completed without quorum"
    );
    c.pump(&[2, 3])?;
    let Some(Outcome::Committed(Applied::Publisher(PublisherReply::Planned { index: plan_index }))) =
        c.outcome(711)
    else {
        anyhow::bail!("plan did not commit");
    };
    let image = c
        .peers
        .get_mut(&2)
        .unwrap()
        .call(Request::PrepareCheckpoint(10, Uuid::from_u128(710)))?;
    assert!(
        matches!(image, Response::Image(_, _, 0)),
        "catalog unexpectedly retained control payloads as data"
    );
    c.command(2, Request::InstallCheckpoint)?;
    c.pump(&[2, 3])?;
    c.start(1)?;
    for _ in 0..30 {
        c.command(2, Request::Tick)?;
        c.pump(&[1, 2, 3])?;
        if c.transfers > 0 {
            break;
        }
    }
    assert!(
        c.transfers > 0,
        "lagging owner did not receive an application image"
    );
    let Response::Publisher(Some(table)) = c
        .peers
        .get_mut(&1)
        .unwrap()
        .call(Request::Publisher(acquire.table))?
    else {
        anyhow::bail!("snapshot omitted publisher")
    };
    assert_eq!(table.authority.epoch, 2);
    assert_eq!(table.destination, acquire.destination);
    assert_eq!(table.ingest, acquire.ingest);
    let saved = table
        .progress
        .pending
        .as_ref()
        .context("snapshot lost plan")?;
    assert_eq!(saved.index, plan_index);
    assert_eq!(saved.plan, plan.plan);
    assert_eq!(saved.plan.target_version()?, 8);

    c.command(2, Request::Plan(712, plan.clone()))?;
    c.pump(&[2])?;
    assert!(
        c.outcome(712).is_none(),
        "isolated leader answered a plan retry from cache"
    );

    c.command(
        2,
        Request::Query(
            10,
            8000,
            Query::Publisher {
                table: acquire.table,
            },
        ),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(ReadOutcome::Observed {
        cut,
        value: Value::Publisher(Some(captured)),
    }) = c.read_outcome(8000)
    else {
        anyhow::bail!("publisher quorum view missing");
    };
    assert_eq!(*captured, *table);
    assert!(cut.quorum_index >= table.authority.grant_index);
    assert!(cut.applied >= cut.quorum_index);

    c.command(2, Request::Acquire(703, acquire.clone()))?;
    c.pump(&[2])?;
    assert!(
        c.outcome(703).is_none(),
        "isolated leader answered an authority retry from cache"
    );
    let leader = c.elect(1, 10, &[1, 3])?;
    let stale = acquire.clone();
    acquire.owner = Uuid::from_u128(706);
    acquire.request = Uuid::from_u128(707);
    acquire.expected_epoch = 2;
    c.command(leader, Request::Acquire(704, acquire.clone()))?;
    c.pump(&[1, 3])?;
    let latest = c.outcome(704).context("third epoch unavailable")?;
    assert!(
        matches!(&latest, Outcome::Committed(Applied::Publisher(PublisherReply::Acquired(a))) if a.epoch == 3)
    );
    c.command(leader, Request::Plan(713, plan.clone()))?;
    c.pump(&[1, 3])?;
    assert_eq!(
        c.outcome(713),
        Some(Outcome::Committed(Applied::Publisher(
            PublisherReply::Rejected(PublisherRejection::Authority)
        )))
    );
    let Outcome::Committed(Applied::Publisher(PublisherReply::Acquired(authority))) = &latest
    else {
        anyhow::bail!("missing third authority");
    };
    plan.authority = authority.clone();
    c.command(leader, Request::Acquire(705, stale))?;
    c.pump(&[1, 3])?;
    assert_eq!(
        c.outcome(705),
        Some(Outcome::Committed(Applied::Publisher(
            PublisherReply::Rejected(PublisherRejection::Epoch { current: 3 })
        )))
    );
    for _ in 0..4 {
        c.command(leader, Request::Tick)?;
        c.pump(&[1, 2, 3])?;
    }
    c.queue.clear();
    c.peers.clear();
    for id in 1..=3 {
        c.start(id)?;
    }
    for peer in c.peers.values_mut() {
        let Response::Publisher(Some(table)) = peer.call(Request::Publisher(acquire.table))? else {
            anyhow::bail!("restart omitted publisher")
        };
        assert_eq!(table.authority.epoch, 3);
        assert_eq!(table.authority.owner, acquire.owner);
        assert_eq!(table.ingest, acquire.ingest);
        let recovered = table
            .progress
            .pending
            .as_ref()
            .context("restart lost plan")?;
        assert_eq!(recovered.index, plan_index);
        assert_eq!(recovered.plan, plan.plan);
        verify_records(peer, 10, &[])?;
        verify_records(peer, 20, std::slice::from_ref(&source))?;
    }
    let leader = c.elect(1, 10, &[1, 2, 3])?;
    c.command(leader, Request::Acquire(706, acquire))?;
    c.pump(&[1, 2, 3])?;
    assert_eq!(
        c.outcome(706),
        Some(latest),
        "restart changed the committed grant index"
    );
    c.command(leader, Request::Plan(714, plan))?;
    c.pump(&[1, 2, 3])?;
    assert_eq!(
        c.outcome(714),
        Some(Outcome::Committed(Applied::Publisher(
            PublisherReply::Planned { index: plan_index }
        )))
    );
    assert_eq!(
        captured.authority.epoch, 2,
        "later takeover mutated a captured read"
    );
    Ok(())
}

#[test]
fn quorum_queries_capture_committed_tails_without_appends_and_reject_isolated_reads() -> Result<()>
{
    let mut c = Cluster::new()?;
    c.command(1, Request::Campaign(10))?;
    c.pump(&[1, 2, 3])?;
    let first = input(0, 1, 70);
    c.command(1, Request::Propose(10, 900, first.clone()))?;
    c.pump(&[1, 2, 3])?;
    let Some(Outcome::Committed(Applied::Accepted { index: first_index })) = c.outcome(900) else {
        anyhow::bail!("first append did not commit");
    };
    let query = Query::Producer {
        table: first.meta.stream,
        bucket: 0,
        producer: first.meta.producer,
    };
    let Response::Status(before) = c.peers.get_mut(&1).unwrap().call(Request::Status(10))? else {
        anyhow::bail!("missing initial status");
    };
    c.command(1, Request::Query(10, 901, query.clone()))?;
    assert!(
        c.read_outcome(901).is_none(),
        "local tail answered without quorum"
    );
    c.pump(&[1, 2])?;
    let Some(ReadOutcome::Observed {
        cut,
        value: Value::Producer(Some((meta, index))),
    }) = c.read_outcome(901)
    else {
        anyhow::bail!("missing quorum producer tail");
    };
    assert_eq!(meta, first.meta);
    assert_eq!(index, first_index);
    assert!(cut.quorum_index >= index && cut.applied >= cut.quorum_index);
    c.command(
        1,
        Request::Query(
            10,
            902,
            Query::Accepted {
                after: 0,
                limit: 128,
            },
        ),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(ReadOutcome::Observed {
        cut,
        value: Value::Accepted(page),
    }) = c.read_outcome(902)
    else {
        anyhow::bail!("missing quorum accepted page");
    };
    assert_eq!(page.entries, vec![(first_index, first.meta.clone())]);
    assert_eq!(page.through, cut.applied);
    let Response::Status(after) = c.peers.get_mut(&1).unwrap().call(Request::Status(10))? else {
        anyhow::bail!("missing post-read status");
    };
    assert_eq!(
        after.committed, before.committed,
        "read-index queries appended log entries"
    );

    c.command(1, Request::Query(10, 903, query.clone()))?;
    c.pump(&[1])?;
    assert!(c.read_outcome(903).is_none());
    let leader = c.elect(2, 10, &[2, 3])?;
    let second = input(0, 2, 71);
    c.command(leader, Request::Propose(10, 904, second.clone()))?;
    c.pump(&[2, 3])?;
    let Some(Outcome::Committed(Applied::Accepted {
        index: second_index,
    })) = c.outcome(904)
    else {
        anyhow::bail!("new leader did not append");
    };
    for _ in 0..25 {
        c.command(1, Request::Tick)?;
        c.pump(&[1])?;
    }
    assert_eq!(c.read_outcome(903), Some(ReadOutcome::Unknown));
    for _ in 0..4 {
        c.command(leader, Request::Tick)?;
        c.pump(&[1, 2, 3])?;
    }
    c.command(leader, Request::Query(10, 905, query.clone()))?;
    c.pump(&[1, 2, 3])?;
    let Some(ReadOutcome::Observed {
        value: Value::Producer(Some((meta, index))),
        ..
    }) = c.read_outcome(905)
    else {
        anyhow::bail!("missing new leader's tail");
    };
    assert_eq!(meta, second.meta);
    assert_eq!(index, second_index);
    c.queue.clear();
    c.peers.clear();
    for id in 1..=3 {
        c.start(id)?;
    }
    let leader = c.elect(1, 10, &[1, 2, 3])?;
    c.command(leader, Request::Query(10, 901, query))?;
    assert!(
        c.read_outcome(901).is_none(),
        "restart reused an old read result"
    );
    c.pump(&[1, 2, 3])?;
    let Some(ReadOutcome::Observed {
        value: Value::Producer(Some((meta, index))),
        ..
    }) = c.read_outcome(901)
    else {
        anyhow::bail!("missing restarted tail");
    };
    assert_eq!(meta, second.meta);
    assert_eq!(index, second_index);
    for peer in c.peers.values_mut() {
        verify_records(peer, 10, &[first.clone(), second.clone()])?;
    }
    Ok(())
}

#[test]
fn disjoint_source_pins_survive_transfer_takeover_and_all_process_restart() -> Result<()> {
    use lance_context_wal::ingest_contract::{IngestContract, WritePolicy};
    let key = ring::signature::Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
        .map_err(|_| anyhow::anyhow!("test key generation failed"))?;
    let layout = Layout {
        cluster: Uuid::from_u128(810),
        nodes: (1..=6)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![4, 5, 6])]),
        buckets: vec![10, 20],
    };
    let mut c = Cluster::with_layout(layout, Some(key.as_ref()))?;
    let catalog = c.elect(1, 10, &[1, 2, 3])?;
    let source = c.elect(4, 20, &[4, 5, 6])?;
    assert_eq!(source, 4);
    drop(c.peers.remove(&6));
    let table = Uuid::from_u128(811);
    let rows: Vec<_> = (1..=3)
        .map(|sequence| {
            Batch::new(
                table,
                1,
                Uuid::from_u128(812),
                sequence,
                Bytes::from(vec![sequence as u8; 64 * 1024]),
            )
        })
        .collect();
    for (n, row) in rows.iter().enumerate() {
        c.command(source, Request::Propose(20, 1000 + n as u64, row.clone()))?;
        c.pump(&[4, 5])?;
        assert!(matches!(
            c.outcome(1000 + n as u64),
            Some(Outcome::Committed(Applied::Accepted { .. }))
        ));
    }
    c.command(
        source,
        Request::Query(
            20,
            1010,
            Query::Accepted {
                after: 0,
                limit: 128,
            },
        ),
    )?;
    c.pump(&[4, 5])?;
    let Some(ReadOutcome::Observed {
        value: Value::Accepted(page),
        ..
    }) = c.read_outcome(1010)
    else {
        anyhow::bail!("source quorum page missing");
    };
    let mut acquire = Acquire {
        table,
        destination: Destination {
            namespace: Uuid::from_u128(813),
            dataset: "tables/pinned".into(),
        },
        ingest: IngestContract {
            tenant: "test".into(),
            qualified_table: "pinned".into(),
            schema: [3; 32],
            policy: WritePolicy::ImmutableUniqueIds,
            max_record_bytes: 1024 * 1024,
            max_rows: 1024,
        },
        owner: Uuid::from_u128(814),
        request: Uuid::from_u128(815),
        expected_epoch: 0,
    };
    c.command(catalog, Request::Acquire(1020, acquire.clone()))?;
    c.pump(&[1, 2, 3])?;
    let Some(Outcome::Committed(Applied::Publisher(PublisherReply::Acquired(authority)))) =
        c.outcome(1020)
    else {
        anyhow::bail!("catalog authority missing");
    };
    let transaction = Uuid::from_u128(816);
    c.command(
        catalog,
        Request::Plan(
            1021,
            PlanRequest {
                authority: authority.clone(),
                plan: Plan {
                    transaction,
                    read_version: 7,
                    next_group: 10,
                    sources: BTreeMap::from([(
                        20,
                        SourceRange {
                            after: 0,
                            through: page.through,
                            records: page.entries,
                        },
                    )]),
                },
            },
        ),
    )?;
    c.pump(&[1, 2, 3])?;
    let Some(Outcome::Committed(Applied::Publisher(PublisherReply::Planned { index: plan_index }))) =
        c.outcome(1021)
    else {
        anyhow::bail!("catalog plan missing");
    };
    let query = Query::Pin {
        table,
        transaction,
        plan_index,
        source_group: 20,
    };
    c.command(catalog, Request::Query(10, 1022, query.clone()))?;
    assert!(
        c.read_outcome(1022).is_none(),
        "attestation preceded quorum confirmation"
    );
    c.pump(&[1, 2, 3])?;
    let Some(ReadOutcome::Observed {
        value: Value::Pin(Some(permit)),
        ..
    }) = c.read_outcome(1022)
    else {
        anyhow::bail!("catalog attestation missing");
    };
    // The attestation crosses two real TCP connections and a process boundary.
    // Source replicas have only the policy's public verification key.
    c.command(source, Request::Pin(1030, *permit))?;
    assert!(
        c.outcome(1030).is_none(),
        "pin preceded durable source quorum"
    );
    c.pump(&[4, 5])?;
    let Some(Outcome::Committed(Applied::Source(SourceReply::Pinned { index }))) = c.outcome(1030)
    else {
        anyhow::bail!("source pin missing");
    };
    drop(c.peers.remove(&3));
    let descriptors = Descriptors::lance_fragments(Bytes::from(vec![19; 192 * 1024]))?;
    let prepared = PrepareRequest {
        authority,
        transaction,
        plan_index,
        descriptors: descriptors.clone(),
    };
    c.command(catalog, Request::Prepare(1031, prepared.clone()))?;
    c.pump(&[1])?;
    assert!(
        c.outcome(1031).is_none(),
        "preparation preceded durable quorum"
    );
    // The isolated pump dropped the first Append packets. Healing the link
    // needs a heartbeat to resume replication, not just an empty queue drain.
    for _ in 0..30 {
        c.command(catalog, Request::Tick)?;
        c.pump(&[1, 2])?;
        if c.completions
            .iter()
            .any(|completion| completion.request == 1031)
        {
            break;
        }
    }
    let prepared_outcome = c.outcome(1031);
    let Some(Outcome::Committed(Applied::Publisher(PublisherReply::Prepared {
        index: prepared_index,
    }))) = prepared_outcome
    else {
        anyhow::bail!("prepared descriptors missing: {prepared_outcome:?}");
    };
    c.peers
        .get_mut(&catalog)
        .unwrap()
        .call(Request::PrepareCheckpoint(10, Uuid::from_u128(820)))?;
    c.command(catalog, Request::InstallCheckpoint)?;
    c.pump(&[1, 2])?;
    let before_catalog_transfer = c.transfers;
    c.start(3)?;
    for _ in 0..40 {
        c.command(catalog, Request::Tick)?;
        c.pump(&[1, 2, 3])?;
        if c.transfers > before_catalog_transfer {
            break;
        }
    }
    assert!(
        c.transfers > before_catalog_transfer,
        "descriptors never crossed a catalog snapshot transfer"
    );
    let before_source_transfer = c.transfers;
    c.peers
        .get_mut(&4)
        .unwrap()
        .call(Request::PrepareCheckpoint(20, Uuid::from_u128(817)))?;
    c.command(4, Request::InstallCheckpoint)?;
    c.pump(&[4, 5])?;
    c.start(6)?;
    for _ in 0..40 {
        c.command(4, Request::Tick)?;
        c.pump(&[4, 5, 6])?;
        if c.transfers > before_source_transfer {
            break;
        }
    }
    assert!(
        c.transfers > before_source_transfer,
        "source pin never crossed a snapshot transfer"
    );
    drop(c.peers.remove(&4));
    drop(c.peers.remove(&1));
    let source = c.elect(6, 20, &[5, 6])?;
    let catalog = c.elect(2, 10, &[2, 3])?;
    acquire.owner = Uuid::from_u128(818);
    acquire.request = Uuid::from_u128(819);
    acquire.expected_epoch = 1;
    c.command(catalog, Request::Acquire(1040, acquire))?;
    c.pump(&[2, 3])?;
    assert!(matches!(
        c.outcome(1040),
        Some(Outcome::Committed(Applied::Publisher(
            PublisherReply::Acquired(_)
        )))
    ));
    let Response::Publisher(Some(taken_over)) = c
        .peers
        .get_mut(&catalog)
        .unwrap()
        .call(Request::Publisher(table))?
    else {
        anyhow::bail!("takeover table missing");
    };
    let mut retry = prepared.clone();
    retry.authority = taken_over.authority;
    c.command(catalog, Request::Prepare(1043, retry.clone()))?;
    c.pump(&[2, 3])?;
    assert_eq!(
        c.outcome(1043),
        Some(Outcome::Committed(Applied::Publisher(
            PublisherReply::Prepared {
                index: prepared_index
            }
        )))
    );
    c.command(catalog, Request::Prepare(1044, prepared))?;
    c.pump(&[2, 3])?;
    assert_eq!(
        c.outcome(1044),
        Some(Outcome::Committed(Applied::Publisher(
            PublisherReply::Rejected(PublisherRejection::Authority)
        )))
    );
    c.command(catalog, Request::Query(10, 1041, query.clone()))?;
    c.pump(&[2, 3])?;
    let Some(ReadOutcome::Observed {
        value: Value::Pin(Some(permit)),
        ..
    }) = c.read_outcome(1041)
    else {
        anyhow::bail!("takeover attestation missing");
    };
    c.command(source, Request::Pin(1042, *permit))?;
    c.pump(&[5, 6])?;
    assert_eq!(
        c.outcome(1042),
        Some(Outcome::Committed(Applied::Source(SourceReply::Pinned {
            index
        })))
    );
    c.queue.clear();
    c.peers.clear();
    for id in 1..=6 {
        c.start(id)?;
    }
    let catalog = c.elect(2, 10, &[1, 2, 3])?;
    let source = c.elect(6, 20, &[4, 5, 6])?;
    c.command(catalog, Request::Query(10, 1050, query))?;
    c.pump(&[1, 2, 3])?;
    let Some(ReadOutcome::Observed {
        value: Value::Pin(Some(permit)),
        ..
    }) = c.read_outcome(1050)
    else {
        anyhow::bail!("restart attestation missing");
    };
    c.command(source, Request::Pin(1051, *permit))?;
    c.pump(&[4, 5, 6])?;
    assert_eq!(
        c.outcome(1051),
        Some(Outcome::Committed(Applied::Source(SourceReply::Pinned {
            index
        })))
    );
    c.command(source, Request::Query(20, 1052, Query::SourcePin { table }))?;
    c.pump(&[4, 5, 6])?;
    let Some(ReadOutcome::Observed {
        value: Value::SourcePin(Some(pin)),
        ..
    }) = c.read_outcome(1052)
    else {
        anyhow::bail!("restart source pin view missing");
    };
    assert_eq!(pin.index, index);
    assert_eq!(pin.permit.selection().plan_index, plan_index);
    for id in 1..=3 {
        let Response::Publisher(Some(recovered)) = c
            .peers
            .get_mut(&id)
            .unwrap()
            .call(Request::Publisher(table))?
        else {
            anyhow::bail!("restarted descriptor table missing");
        };
        let saved = recovered
            .progress
            .prepared
            .context("restart lost descriptors")?;
        assert_eq!(saved.index, prepared_index);
        assert_eq!(saved.plan_index, plan_index);
        assert_eq!(saved.transaction, transaction);
        assert_eq!(saved.descriptors, descriptors);
    }
    c.command(catalog, Request::Prepare(1053, retry))?;
    c.pump(&[1, 2, 3])?;
    assert_eq!(
        c.outcome(1053),
        Some(Outcome::Committed(Applied::Publisher(
            PublisherReply::Prepared {
                index: prepared_index
            }
        )))
    );
    for id in 4..=6 {
        verify_records(c.peers.get_mut(&id).unwrap(), 20, &rows)?;
    }
    Ok(())
}

#[cfg(feature = "lance-integration")]
#[path = "replication_process/lance_publication.rs"]
mod lance_publication;

#[path = "replication_process/producer_slots.rs"]
mod producer_slots;
