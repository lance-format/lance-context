//! Bounded mutual-TLS RPCs from trusted ingress/materializer services to WAL nodes.
//!
//! Separate data/control listeners and physical-node pools preserve control
//! capacity. Append metadata and opaque Bytes travel separately in one exchange;
//! only the durable runtime completion produces a commit response. This is a
//! trusted service boundary, not end-user table authorization or leader routing.
use super::{query, Command, Handle, Outcome, Pending, Work};
use crate::{
    placement::Layout,
    record::{Batch, BatchMeta},
    replication::{Limits, Status},
};
use anyhow::{ensure, Context, Result};
use bincode::Options as _;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{watch, OwnedSemaphorePermit, Semaphore},
    task::{JoinHandle, JoinSet},
    time::timeout,
};
use tokio_rustls::{
    rustls::{self, pki_types::ServerName},
    TlsAcceptor, TlsConnector,
};

pub const ALPN: &[u8] = b"lance-context-wal-client/1";
const MAGIC: &[u8; 8] = b"LCWCLNT1";
const MAX_META: usize = 2 << 20;
const PREFIX: usize = 24;

#[derive(Clone, Debug)]
pub struct Endpoint {
    pub data_address: String,
    pub control_address: String,
    pub server_name: String,
    /// SHA-256 of the configured service's DER leaf certificate.
    pub certificate: [u8; 32],
}
#[derive(Clone, Debug)]
#[cfg_attr(feature = "service", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "service", serde(default, deny_unknown_fields))]
pub struct Options {
    /// Logical live request/response bytes, independent for the two lanes.
    /// Caller allocations and allocator/decoded metadata overhead are separate.
    pub data_bytes: usize,
    pub control_bytes: usize,
    /// Per node AND lane in one client's physical connection pool.
    pub connections: usize,
    /// Per server listener, shared across all independently running clients.
    pub server_connections: usize,
    /// Client: entire exchange, including pool/admission and durable completion.
    /// Server: separate handshake, idle, and active-request deadlines. The active
    /// deadline starts at the first request bytes and includes the remaining
    /// frame, admission, durable completion, and reply transmission.
    #[cfg_attr(feature = "service", serde(with = "crate::service::duration_millis"))]
    pub timeout: Duration,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            data_bytes: 64 << 20,
            control_bytes: 32 << 20,
            connections: 4,
            server_connections: 64,
            timeout: Duration::from_secs(30),
        }
    }
}
impl Options {
    pub(crate) fn validate(&self, max_record: usize) -> Result<()> {
        ensure!(
            (1..=32).contains(&self.connections)
                && (1..=256).contains(&self.server_connections)
                && (2 * max_record + 4 * MAX_META..=256 << 20).contains(&self.data_bytes)
                && (4 * MAX_META..=256 << 20).contains(&self.control_bytes)
                && (Duration::from_millis(100)..=Duration::from_secs(120)).contains(&self.timeout),
            "invalid client transport limits"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Scope {
    cluster: uuid::Uuid,
    layout: [u8; 32],
    policy: [u8; 32],
    node: u64,
}
#[derive(Clone, Debug)]
pub enum Request {
    Append {
        group: u64,
        batch: Batch,
    },
    Commit(Box<Command>),
    Query {
        group: u64,
        query: query::Query,
    },
    /// Local committed data only. Establish a quorum source pin first; None is
    /// not evidence that an uncertain append was rejected or never committed.
    Fetch {
        group: u64,
        index: u64,
    },
    /// Cached local leadership hint, never a quorum result.
    Status {
        group: u64,
    },
}
#[derive(Debug, Serialize, Deserialize)]
enum WireRequest {
    Append { group: u64, meta: BatchMeta },
    Control(Box<Command>),
    Query { group: u64, query: query::Query },
    Fetch { group: u64, index: u64 },
    Status { group: u64 },
}
impl WireRequest {
    fn data(&self) -> bool {
        matches!(self, Self::Append { .. } | Self::Fetch { .. })
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub enum Reply {
    Commit(Outcome),
    Query(query::Outcome),
    Record(Option<BatchMeta>),
    Status(Status),
    /// Reconcile the exact original identity. This does not prove non-commit.
    Unknown,
    /// No command was submitted by this handler; leader is only a local hint.
    Redirect {
        leader: u64,
    },
}
/// Holds receive credit through consumption of metadata and a fetched payload.
/// Cloning body Bytes transfers shared ownership, so callers retaining clones
/// must provide their own application memory budget before dropping this owner.
pub struct Response {
    pub reply: Reply,
    pub body: Bytes,
    _credit: OwnedSemaphorePermit,
}
struct Budgets {
    data: Arc<Semaphore>,
    control: Arc<Semaphore>,
}
impl Budgets {
    fn new(options: &Options) -> Self {
        Self {
            data: Arc::new(Semaphore::new(options.data_bytes)),
            control: Arc::new(Semaphore::new(options.control_bytes)),
        }
    }
    async fn reserve(&self, data: bool, bytes: usize) -> Result<OwnedSemaphorePermit> {
        Ok(if data { &self.data } else { &self.control }
            .clone()
            .acquire_many_owned(u32::try_from(bytes)?)
            .await?)
    }
}
fn codec() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_META as u64)
        .reject_trailing_bytes()
}
fn certificate(certs: Option<&[rustls::pki_types::CertificateDer<'_>]>) -> Result<[u8; 32]> {
    Ok(Sha256::digest(
        certs
            .and_then(|c| c.first())
            .context("client TLS certificate missing")?
            .as_ref(),
    )
    .into())
}
fn scope(handle: &Handle) -> Scope {
    Scope {
        cluster: handle.cluster,
        layout: handle.layout,
        policy: handle.policy,
        node: handle.id,
    }
}
async fn prefix<R: AsyncRead + Unpin>(io: &mut R, max_body: usize) -> Result<(u64, usize, usize)> {
    let mut fixed = [0; PREFIX];
    io.read_exact(&mut fixed).await?;
    decode_prefix(&fixed, max_body)
}
fn decode_prefix(fixed: &[u8; PREFIX], max_body: usize) -> Result<(u64, usize, usize)> {
    ensure!(&fixed[..8] == MAGIC, "foreign client protocol");
    let id = u64::from_le_bytes(fixed[8..16].try_into()?);
    let meta = u32::from_le_bytes(fixed[16..20].try_into()?) as usize;
    let body = u32::from_le_bytes(fixed[20..24].try_into()?) as usize;
    ensure!(
        id > 0 && (1..=MAX_META).contains(&meta) && body <= max_body,
        "invalid client frame bounds"
    );
    Ok((id, meta, body))
}
async fn receive<R: AsyncRead + Unpin>(io: &mut R, bytes: usize) -> Result<Bytes> {
    let mut body = vec![0; bytes];
    io.read_exact(&mut body).await?;
    Ok(body.into())
}
async fn send<W: AsyncWrite + Unpin>(
    io: &mut W,
    id: u64,
    metadata: &[u8],
    body: &[u8],
) -> Result<()> {
    ensure!(
        (1..=MAX_META).contains(&metadata.len()),
        "oversized client metadata"
    );
    let mut fixed = [0; PREFIX];
    fixed[..8].copy_from_slice(MAGIC);
    fixed[8..16].copy_from_slice(&id.to_le_bytes());
    fixed[16..20].copy_from_slice(&u32::try_from(metadata.len())?.to_le_bytes());
    fixed[20..24].copy_from_slice(&u32::try_from(body.len())?.to_le_bytes());
    // TLS owns its encrypted output buffers; no whole-payload concatenation.
    io.write_all(&fixed).await?;
    io.write_all(metadata).await?;
    io.write_all(body).await?;
    io.flush().await?;
    Ok(())
}

impl Handle {
    /// Read on the owner thread with data-lane credit, separate from small
    /// control replies. Returned bytes describe local committed state only.
    /// Complete after validating that already durable payload, without waiting
    /// for the next poll to persist unrelated proposals or peer updates.
    pub fn read_accepted(&self, group: u64, index: u64) -> Result<Pending<Option<Batch>>> {
        ensure!(index > 0 && index < u64::MAX, "invalid accepted index");
        let route = self.route(group)?;
        let credit = self.shared.data.reserve(self.max_record + 1024)?;
        let (reply, receive) = tokio::sync::oneshot::channel();
        let completion = super::Completion {
            reply,
            _credit: credit,
        };
        self.enqueue(route, Work::Fetch(index, completion), true)?;
        Ok(Pending(receive))
    }
}

pub struct Server {
    stop: watch::Sender<bool>,
    tasks: Vec<JoinHandle<Result<()>>>,
}
impl Server {
    pub fn is_finished(&self) -> bool {
        self.tasks.iter().any(|task| task.is_finished())
    }
    /// Trusted service certificates may access this node's configured groups.
    /// End-user authorization and table binding stay at the application ingress.
    pub fn spawn(
        handle: Handle,
        data: TcpListener,
        control: TcpListener,
        clients: BTreeSet<[u8; 32]>,
        tls: Arc<rustls::ServerConfig>,
        options: Options,
    ) -> Result<Self> {
        options.validate(handle.max_record)?;
        ensure!(
            handle.disk_budget().is_some(),
            "client listener requires node disk admission"
        );
        ensure!(
            !clients.is_empty() && clients.len() <= 256 && !clients.contains(&[0; 32]),
            "invalid trusted client pins"
        );
        ensure!(
            tls.alpn_protocols.iter().any(|p| p == ALPN),
            "missing client RPC ALPN"
        );
        let (stop, shutdown) = watch::channel(false);
        let budgets = Arc::new(Budgets::new(&options));
        let clients = Arc::new(clients);
        let mut tasks = Vec::new();
        for (listener, lane) in [(data, true), (control, false)] {
            tasks.push(tokio::spawn(listen(
                listener,
                lane,
                handle.clone(),
                clients.clone(),
                TlsAcceptor::from(tls.clone()),
                budgets.clone(),
                options.clone(),
                shutdown.clone(),
            )));
        }
        Ok(Self { stop, tasks })
    }
    pub async fn shutdown(mut self) -> Result<()> {
        let _ = self.stop.send(true);
        let mut error = None;
        for task in std::mem::take(&mut self.tasks) {
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => error = Some(e),
                Err(e) => error = Some(e.into()),
            }
        }
        error.map_or(Ok(()), Err)
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
    }
}
#[allow(clippy::too_many_arguments)]
async fn listen(
    listener: TcpListener,
    data: bool,
    handle: Handle,
    clients: Arc<BTreeSet<[u8; 32]>>,
    acceptor: TlsAcceptor,
    budgets: Arc<Budgets>,
    options: Options,
    mut stop: watch::Receiver<bool>,
) -> Result<()> {
    let connections = Arc::new(Semaphore::new(options.server_connections));
    let mut tasks = JoinSet::new();
    loop {
        tokio::select! {
            _ = stop.changed() => break,
            result = tasks.join_next(), if !tasks.is_empty() => { if let Some(Err(error)) = result { if error.is_panic() { return Err(error.into()); } } },
            incoming = listener.accept() => {
                let (tcp, _) = incoming?;
                let Ok(permit) = connections.clone().try_acquire_owned() else { continue; };
                let handle = handle.clone(); let clients = clients.clone(); let acceptor = acceptor.clone(); let budgets = budgets.clone(); let options = options.clone();
                tasks.spawn(async move {
                    let _permit = permit;
                    let connection = async {
                        tcp.set_nodelay(true)?; let stream = acceptor.accept(tcp).await?;
                        ensure!(stream.get_ref().1.alpn_protocol() == Some(ALPN) && clients.contains(&certificate(stream.get_ref().1.peer_certificates())?), "unconfigured client identity");
                        // Handshake is independently bounded; each exchange has its own deadline.
                        Ok::<_, anyhow::Error>(stream)
                    };
                    if let Ok(Ok(mut stream)) = timeout(options.timeout, connection).await {
                        loop {
                            // Idle time cannot shorten the following request's
                            // processing budget. Start it on the first bytes,
                            // so a partial header is still deadline-bounded.
                            let mut fixed = [0; PREFIX];
                            let Ok(Ok(count)) = timeout(options.timeout, stream.read(&mut fixed)).await else { break; };
                            if count == 0 { break; }
                            let exchange = async {
                                stream.read_exact(&mut fixed[count..]).await?;
                                let header = decode_prefix(&fixed, if data { handle.max_record } else { 0 })?;
                                serve_one(&mut stream, data, &handle, &budgets, header).await
                            };
                            if !matches!(timeout(options.timeout, exchange).await, Ok(Ok(()))) { break; }
                        }
                    }
                });
            }
        }
    }
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    Ok(())
}
async fn serve_one<S: AsyncRead + AsyncWrite + Unpin>(
    io: &mut S,
    data: bool,
    handle: &Handle,
    budgets: &Budgets,
    (id, meta_len, body_len): (u64, usize, usize),
) -> Result<()> {
    // Reserve reply capacity too, before reading any variable-length allocation.
    let _credit = budgets
        .reserve(
            data,
            2 * meta_len + body_len + 2 * MAX_META + if data { handle.max_record } else { 0 },
        )
        .await?;
    let metadata = receive(io, meta_len).await?;
    let (request_scope, request): (Scope, WireRequest) = codec().deserialize(&metadata)?;
    ensure!(
        request_scope == scope(handle) && request.data() == data,
        "foreign request scope/lane"
    );
    if let WireRequest::Append { meta, .. } = &request {
        meta.check_body_length(body_len, handle.max_record)?;
    } else {
        ensure!(body_len == 0, "unexpected request payload");
    }
    let body = receive(io, body_len).await?;
    let executed = execute(handle, request, body).await;
    let (reply, body) = executed.unwrap_or((Reply::Unknown, Bytes::new()));
    let metadata = codec().serialize(&(scope(handle), reply))?;
    send(io, id, &metadata, &body).await
}
async fn execute(handle: &Handle, request: WireRequest, body: Bytes) -> Result<(Reply, Bytes)> {
    let group = match &request {
        WireRequest::Append { group, .. } | WireRequest::Query { group, .. } => Some(*group),
        WireRequest::Control(command) => Some(command.group(handle.catalog)),
        _ => None,
    };
    if let Some(group) = group {
        let leader = handle.observation(group)?.status.leader;
        if leader != handle.id {
            return Ok((Reply::Redirect { leader }, Bytes::new()));
        }
    }
    let mut output = Bytes::new();
    let reply = match request {
        WireRequest::Append { group, meta } => Reply::Commit(
            handle
                .commit(Command::Append {
                    group,
                    batch: Batch { meta, body },
                })?
                .await?,
        ),
        WireRequest::Control(command) => {
            ensure!(
                !matches!(*command, Command::Append { .. }),
                "append on control lane"
            );
            Reply::Commit(handle.commit(*command)?.await?)
        }
        WireRequest::Query { group, query } => Reply::Query(handle.query(group, query)?.await?),
        WireRequest::Fetch { group, index } => {
            Reply::Record(handle.read_accepted(group, index)?.await?.map(|batch| {
                output = batch.body;
                batch.meta
            }))
        }
        WireRequest::Status { group } => Reply::Status(handle.observation(group)?.status),
    };
    Ok((reply, output))
}

type Stream = tokio_rustls::client::TlsStream<TcpStream>;
struct Pool {
    idle: Mutex<Vec<Stream>>,
    slots: Arc<Semaphore>,
}
impl Pool {
    fn new(count: usize) -> Self {
        Self {
            idle: Mutex::new(Vec::new()),
            slots: Arc::new(Semaphore::new(count)),
        }
    }
}
struct NodePools {
    endpoint: Endpoint,
    data: Pool,
    control: Pool,
}

fn disconnected(error: &anyhow::Error) -> bool {
    use std::io::ErrorKind;
    error.downcast_ref::<std::io::Error>().is_some_and(|e| {
        matches!(
            e.kind(),
            ErrorKind::UnexpectedEof
                | ErrorKind::BrokenPipe
                | ErrorKind::ConnectionReset
                | ErrorKind::ConnectionAborted
                | ErrorKind::NotConnected
        )
    })
}

pub struct Client {
    nodes: BTreeMap<u64, NodePools>,
    connector: TlsConnector,
    options: Options,
    budgets: Budgets,
    cluster: uuid::Uuid,
    layout: [u8; 32],
    policy: [u8; 32],
    max_record: usize,
    max_plan_source: u64,
    max_descriptors: u64,
    producer_slots: u32,
    next: AtomicU64,
    voters: BTreeMap<u64, Vec<u64>>,
    catalog: u64,
    leaders: Mutex<BTreeMap<u64, u64>>,
}
impl Client {
    pub fn identity(&self) -> (uuid::Uuid, [u8; 32]) {
        (self.cluster, self.layout)
    }
    pub fn max_record_bytes(&self) -> usize {
        self.max_record
    }
    pub fn max_plan_source_bytes(&self) -> u64 {
        self.max_plan_source
    }
    pub fn max_descriptor_bytes(&self) -> u64 {
        self.max_descriptors
    }
    pub fn producer_slots(&self) -> u32 {
        self.producer_slots
    }
    pub fn catalog_group(&self) -> u64 {
        self.catalog
    }
    pub fn new(
        layout: &Layout,
        limits: &Limits,
        endpoints: BTreeMap<u64, Endpoint>,
        tls: Arc<rustls::ClientConfig>,
        options: Options,
    ) -> Result<Self> {
        options.validate(limits.max_record_bytes as usize)?;
        ensure!(
            tls.alpn_protocols.iter().any(|p| p == ALPN),
            "missing client RPC ALPN"
        );
        ensure!(
            endpoints.keys().eq(layout.nodes.keys())
                && endpoints.len() * options.connections * 2 <= 1024,
            "client endpoints must cover exactly the bounded layout"
        );
        let mut pins = BTreeSet::new();
        for endpoint in endpoints.values() {
            ensure!(
                !endpoint.data_address.is_empty()
                    && endpoint.data_address.len() <= 1024
                    && !endpoint.control_address.is_empty()
                    && endpoint.control_address.len() <= 1024
                    && endpoint.certificate != [0; 32]
                    && pins.insert(endpoint.certificate),
                "invalid/duplicate client endpoint identity"
            );
            ServerName::try_from(endpoint.server_name.clone())?;
        }
        Ok(Self {
            cluster: layout.cluster,
            layout: layout.fingerprint()?,
            policy: Sha256::digest(limits.identity()?).into(),
            max_record: limits.max_record_bytes as usize,
            max_plan_source: limits.max_plan_source_bytes,
            max_descriptors: limits.max_descriptor_bytes,
            producer_slots: limits.producer_slots,
            connector: TlsConnector::from(tls),
            budgets: Budgets::new(&options),
            nodes: endpoints
                .into_iter()
                .map(|(id, endpoint)| {
                    (
                        id,
                        NodePools {
                            endpoint,
                            data: Pool::new(options.connections),
                            control: Pool::new(options.connections),
                        },
                    )
                })
                .collect(),
            options,
            next: AtomicU64::new(1),
            voters: layout.groups.clone(),
            catalog: super::p::control_group(layout),
            leaders: Mutex::new(BTreeMap::new()),
        })
    }
    /// A failure/cancellation is an unknown outcome. Retain and reconcile the
    /// original producer sequence/control identity. Mutations are never silently
    /// retried. A read on a disconnected pooled stream may reconnect once within
    /// the original deadline; a cached leader hint is never treated as an ACK.
    pub async fn request(&self, node: u64, request: Request) -> Result<Response> {
        timeout(self.options.timeout, self.exchange(node, request))
            .await
            .context("client exchange timed out; outcome unknown")?
    }
    /// Follow configured-voter hints only when the remote confirms it did not
    /// submit the request. Unknown mutation outcomes require explicit exact retry.
    /// Read-only pooled disconnects may reconnect once per target, within this
    /// method's single overall deadline.
    pub async fn request_routed(&self, request: Request) -> Result<Response> {
        timeout(self.options.timeout, async {
            let group = match &request {
                Request::Append { group, .. }
                | Request::Query { group, .. }
                | Request::Fetch { group, .. }
                | Request::Status { group } => *group,
                Request::Commit(command) => command.group(self.catalog),
            };
            let voters = self.voters.get(&group).context("unknown client group")?;
            let mut node = self
                .leaders
                .lock()
                .unwrap()
                .get(&group)
                .copied()
                .unwrap_or(voters[0]);
            let mut visited = BTreeSet::new();
            for _ in 0..voters.len() {
                visited.insert(node);
                let response = match self.exchange(node, request.clone()).await {
                    Ok(response) => response,
                    Err(error) => {
                        let next = voters[(voters.iter().position(|id| *id == node).unwrap() + 1)
                            % voters.len()];
                        self.leaders.lock().unwrap().insert(group, next);
                        return Err(error);
                    }
                };
                if let Reply::Redirect { leader } = response.reply {
                    node = if voters.contains(&leader) && !visited.contains(&leader) {
                        leader
                    } else {
                        *voters
                            .iter()
                            .find(|n| !visited.contains(n))
                            .context("leader hint loop")?
                    };
                    self.leaders.lock().unwrap().insert(group, node);
                } else {
                    if matches!(
                        response.reply,
                        Reply::Unknown
                            | Reply::Commit(Outcome::Unknown)
                            | Reply::Query(query::Outcome::Unknown)
                    ) {
                        let next = voters[(voters.iter().position(|id| *id == node).unwrap() + 1)
                            % voters.len()];
                        self.leaders.lock().unwrap().insert(group, next);
                    } else if matches!(
                        request,
                        Request::Append { .. } | Request::Commit(_) | Request::Query { .. }
                    ) {
                        self.leaders.lock().unwrap().insert(group, node);
                    }
                    return Ok(response);
                }
            }
            anyhow::bail!("leader not resolved; retain the original request identity")
        })
        .await
        .context("routed exchange timed out; outcome unknown")?
    }
    async fn exchange(&self, node: u64, request: Request) -> Result<Response> {
        let pools = self.nodes.get(&node).context("unknown client target")?;
        let (wire, body) = match request {
            Request::Append { group, batch } => {
                batch
                    .meta
                    .check_body_length(batch.body.len(), self.max_record)?;
                (
                    WireRequest::Append {
                        group,
                        meta: batch.meta,
                    },
                    batch.body,
                )
            }
            Request::Commit(command) => {
                ensure!(
                    !matches!(*command, Command::Append { .. }),
                    "use the append request for data"
                );
                (WireRequest::Control(command), Bytes::new())
            }
            Request::Query { group, query } => (WireRequest::Query { group, query }, Bytes::new()),
            Request::Fetch { group, index } => (WireRequest::Fetch { group, index }, Bytes::new()),
            Request::Status { group } => (WireRequest::Status { group }, Bytes::new()),
        };
        let data = wire.data();
        let read_only = matches!(
            wire,
            WireRequest::Query { .. } | WireRequest::Fetch { .. } | WireRequest::Status { .. }
        );
        let fetch = matches!(wire, WireRequest::Fetch { .. });
        let reply_kind = match &wire {
            WireRequest::Append { .. } | WireRequest::Control(_) => 0,
            WireRequest::Query { .. } => 1,
            WireRequest::Fetch { .. } => 2,
            WireRequest::Status { .. } => 3,
        };
        let expected = Scope {
            cluster: self.cluster,
            layout: self.layout,
            policy: self.policy,
            node,
        };
        let size = usize::try_from(codec().serialized_size(&(&expected, &wire))?)?;
        ensure!(size <= MAX_META, "client metadata too large");
        let credit = self
            .budgets
            .reserve(
                data,
                2 * size + body.len() + 2 * MAX_META + if fetch { self.max_record } else { 0 },
            )
            .await?;
        let metadata = codec().serialize(&(&expected, wire))?;
        let pool = if data { &pools.data } else { &pools.control };
        let _slot = pool.slots.clone().acquire_owned().await?;
        let mut idle = pool.idle.lock().unwrap().pop();
        let mut reconnect_read = read_only && idle.is_some();
        loop {
            let mut stream = match idle.take() {
                Some(stream) => stream,
                None => {
                    let address = if data {
                        &pools.endpoint.data_address
                    } else {
                        &pools.endpoint.control_address
                    };
                    let tcp = TcpStream::connect(address).await?;
                    tcp.set_nodelay(true)?;
                    let stream = self
                        .connector
                        .connect(
                            ServerName::try_from(pools.endpoint.server_name.clone())?,
                            tcp,
                        )
                        .await?;
                    ensure!(
                        stream.get_ref().1.alpn_protocol() == Some(ALPN)
                            && certificate(stream.get_ref().1.peer_certificates())?
                                == pools.endpoint.certificate,
                        "foreign WAL service identity"
                    );
                    stream
                }
            };
            let response: Result<(Reply, Bytes)> = async {
                let id = self
                    .next
                    .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                    .map_err(|_| anyhow::anyhow!("client request ids exhausted"))?;
                send(&mut stream, id, &metadata, &body).await?;
                let (received_id, meta_len, body_len) =
                    prefix(&mut stream, if fetch { self.max_record } else { 0 }).await?;
                ensure!(received_id == id, "client response correlation mismatch");
                let metadata = receive(&mut stream, meta_len).await?;
                let (observed, reply): (Scope, Reply) = codec().deserialize(&metadata)?;
                ensure!(observed == expected, "foreign client response scope");
                ensure!(
                    matches!(reply, Reply::Unknown | Reply::Redirect { .. })
                        || matches!(
                            (&reply, reply_kind),
                            (Reply::Commit(_), 0)
                                | (Reply::Query(_), 1)
                                | (Reply::Record(_), 2)
                                | (Reply::Status(_), 3)
                        ),
                    "unexpected response type"
                );
                match &reply {
                    Reply::Record(Some(meta)) if fetch => {
                        meta.check_body_length(body_len, self.max_record)?
                    }
                    Reply::Record(None) | Reply::Unknown if fetch => {
                        ensure!(body_len == 0, "unexpected record payload")
                    }
                    Reply::Record(_) => anyhow::bail!("record reply for non-fetch request"),
                    _ => ensure!(body_len == 0 && !fetch, "unexpected response payload/type"),
                }
                let body = receive(&mut stream, body_len).await?;
                Ok((reply, body))
            }
            .await;
            let (reply, body) = match response {
                Ok(response) => response,
                Err(error) if reconnect_read && disconnected(&error) => {
                    // Only a reused connection gets this one read-only recovery.
                    // Keep the same byte/connection credits; reconnect authenticates
                    // normally, and fresh-stream failures and invalid replies escape.
                    reconnect_read = false;
                    continue;
                }
                Err(error) => return Err(error),
            };
            // The durable owner already validated fetched bytes; checking corruption
            // here belongs off the async executor for large records. TLS authenticates
            // this transmission; consumers still match metadata against their pin.
            pool.idle.lock().unwrap().push(stream);
            return Ok(Response {
                reply,
                body,
                _credit: credit,
            });
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "lance-integration")]
mod catalog;
#[cfg(feature = "lance-integration")]
pub use catalog::Catalog;
