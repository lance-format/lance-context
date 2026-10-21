//! Authenticated physical-node connections for durable replica workers.
//!
//! One outbound connection per physical peer carries every hosted group. TLS
//! verifies certificate chains/names; an explicit leaf-certificate fingerprint
//! binds each connection to its configured Raft node. Use standard verifying
//! rustls configurations with mutual authentication and the required ALPN.
//! Optional snapshots use a separate listener and bounded transfer workers.
//! No client ingestion is exposed here.
use super::{Handle, Outbound, Packet};
use anyhow::{ensure, Context, Result};
use bytes::{Bytes, BytesMut};
use protobuf::Message as _;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, watch, OwnedSemaphorePermit, Semaphore},
    task::{JoinHandle, JoinSet},
    time::{sleep, timeout},
};
use tokio_rustls::{
    rustls::{self, pki_types::ServerName},
    TlsAcceptor, TlsConnector,
};

pub mod snapshot;

pub const ALPN: &[u8] = b"lance-context-wal-peer/1";
const MAGIC: &[u8; 8] = b"LCWPEER1";
const FIXED: usize = 104;
// Node sends approximately 1 MiB Raft messages, except a single larger entry.
// The header also includes one u32 body length for every protobuf entry.
const MAX_HEADER: usize = 2 << 20;
const MAX_BODY: usize = 16 << 20;
const MAX_ENTRIES: usize = 262_144;

#[derive(Clone, Debug)]
pub struct Peer {
    pub address: String,
    pub server_name: String,
    /// SHA-256 of the complete DER leaf certificate, supplied by trusted config.
    pub certificate: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct Options {
    pub receive_bytes: usize,
    pub connections: usize,
    pub peer_queue: usize,
    /// Queued and actively transmitting logical bytes per destination. A slow
    /// follower cannot retain the entire runtime outbox's byte allowance.
    pub peer_bytes: usize,
    /// Deadline for a complete handshake or frame, including bounded admission.
    pub io_timeout: Duration,
    pub reconnect_delay: Duration,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            receive_bytes: 64 << 20,
            connections: 64,
            peer_queue: 32,
            peer_bytes: 16 << 20,
            io_timeout: Duration::from_secs(5),
            reconnect_delay: Duration::from_millis(100),
        }
    }
}
impl Options {
    fn validate(&self) -> Result<()> {
        ensure!(
            (MAX_HEADER + MAX_BODY..=256 << 20).contains(&self.receive_bytes)
                && (1..=128).contains(&self.connections)
                && (1..=256).contains(&self.peer_queue)
                && (4096..=128 << 20).contains(&self.peer_bytes)
                && (Duration::from_millis(100)..=Duration::from_secs(60))
                    .contains(&self.io_timeout)
                && (Duration::from_millis(10)..=Duration::from_secs(5))
                    .contains(&self.reconnect_delay),
            "invalid peer transport options"
        );
        Ok(())
    }
}

/// Network counters, not replicated receipts or durable write acknowledgments.
#[derive(Default)]
pub struct Statistics {
    pub sent_frames: AtomicU64,
    pub received_frames: AtomicU64,
    pub dropped_outbound: AtomicU64,
    pub rejected_connections: AtomicU64,
    pub failed_exchanges: AtomicU64,
    pub rejected_packets: AtomicU64,
}

pub struct Transport {
    stop: watch::Sender<bool>,
    tasks: Vec<JoinHandle<Result<()>>>,
    statistics: Arc<Statistics>,
    snapshots: Arc<OnceLock<snapshot::Route>>,
    owner: Arc<super::Shared>,
}
impl Transport {
    /// The caller owns certificate provisioning/rotation and the already-bound
    /// listener. All hosted groups share these physical peer queues and sockets.
    pub fn spawn(
        handle: Handle,
        outbox: mpsc::Receiver<Outbound>,
        listener: TcpListener,
        peers: BTreeMap<u64, Peer>,
        client_tls: Arc<rustls::ClientConfig>,
        server_tls: Arc<rustls::ServerConfig>,
        options: Options,
    ) -> Result<Self> {
        options.validate()?;
        let largest = handle.max_record.max(1 << 20) + 4096;
        ensure!(
            options.peer_bytes >= largest
                && options.peer_bytes + largest <= handle.shared.outgoing.capacity_bytes
                && largest <= handle.shared.incoming.capacity_bytes,
            "peer credits must fit a full record and leave runtime capacity for another voter"
        );
        let expected: BTreeSet<_> = handle
            .voters
            .values()
            .flatten()
            .copied()
            .filter(|id| *id != handle.id)
            .collect();
        ensure!(
            peers.keys().copied().collect::<BTreeSet<_>>() == expected,
            "peer configuration must cover the hosted groups' other voters exactly"
        );
        let mut fingerprints = BTreeMap::new();
        for (&id, peer) in &peers {
            ensure!(
                !peer.address.is_empty()
                    && peer.address.len() <= 1024
                    && peer.certificate != [0; 32],
                "invalid peer endpoint or identity"
            );
            ServerName::try_from(peer.server_name.clone()).context("invalid peer TLS name")?;
            ensure!(
                fingerprints.insert(peer.certificate, id).is_none(),
                "peer certificates must identify distinct nodes"
            );
        }
        ensure!(
            client_tls.alpn_protocols.iter().any(|p| p == ALPN)
                && server_tls.alpn_protocols.iter().any(|p| p == ALPN),
            "missing WAL peer ALPN"
        );
        let (stop, shutdown) = watch::channel(false);
        let statistics = Arc::new(Statistics::default());
        let snapshots = Arc::new(OnceLock::new());
        let owner = handle.shared.clone();
        let mut tasks = Vec::new();
        let mut routes = BTreeMap::new();
        for (id, peer) in peers {
            let (send, receive) = mpsc::channel(options.peer_queue);
            let ready = Arc::new(AtomicBool::new(false));
            routes.insert(
                id,
                Route {
                    send,
                    ready: ready.clone(),
                    bytes: Arc::new(Semaphore::new(options.peer_bytes)),
                },
            );
            tasks.push(tokio::spawn(send_loop(
                peer,
                receive,
                ready,
                TlsConnector::from(client_tls.clone()),
                options.clone(),
                shutdown.clone(),
                statistics.clone(),
            )));
        }
        tasks.push(tokio::spawn(dispatch(
            outbox,
            routes,
            snapshots.clone(),
            shutdown.clone(),
            statistics.clone(),
        )));
        tasks.push(tokio::spawn(listen(
            handle,
            listener,
            TlsAcceptor::from(server_tls),
            fingerprints,
            options,
            shutdown,
            statistics.clone(),
        )));
        Ok(Self {
            stop,
            tasks,
            statistics,
            snapshots,
            owner,
        })
    }
    /// Attach the same runtime's separate snapshot service before compacting.
    /// Shut down both transports and join their work before reopening storage.
    pub fn attach_snapshots(&mut self, route: snapshot::Route) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&route.owner, &self.owner),
            "foreign snapshot runtime"
        );
        self.snapshots
            .set(route)
            .map_err(|_| anyhow::anyhow!("snapshot route already attached"))
    }
    pub fn statistics(&self) -> &Arc<Statistics> {
        &self.statistics
    }
    /// Close network tasks before shutting down/reopening the worker runtime.
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
impl Drop for Transport {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
        for task in &self.tasks {
            task.abort();
        }
    }
}

struct Queued {
    packet: Outbound,
    _bytes: OwnedSemaphorePermit,
}
struct Route {
    send: mpsc::Sender<Queued>,
    ready: Arc<AtomicBool>,
    bytes: Arc<Semaphore>,
}
impl Route {
    fn enqueue(&self, packet: Outbound) -> bool {
        if !self.ready.load(Ordering::Acquire) {
            return false;
        }
        let size = packet.packet().message.compute_size().saturating_add(256);
        let Ok(bytes) = self.bytes.clone().try_acquire_many_owned(size) else {
            return false;
        };
        self.send
            .try_send(Queued {
                packet,
                _bytes: bytes,
            })
            .is_ok()
    }
}

async fn dispatch(
    mut outbox: mpsc::Receiver<Outbound>,
    routes: BTreeMap<u64, Route>,
    snapshots: Arc<OnceLock<snapshot::Route>>,
    mut stop: watch::Receiver<bool>,
    statistics: Arc<Statistics>,
) -> Result<()> {
    loop {
        tokio::select! {
            _ = stop.changed() => return Ok(()),
            packet = outbox.recv() => {
                let Some(packet) = packet else { return Ok(()); };
                let sent = if packet.snapshot_image().is_some() {
                    snapshots.get().is_some_and(|route| route.enqueue(packet))
                } else {
                    routes.get(&packet.packet().message.to)
                        .is_some_and(|route| route.enqueue(packet))
                };
                if !sent { statistics.dropped_outbound.fetch_add(1, Ordering::Relaxed); }
            }
        }
    }
}

fn fingerprint(certificates: Option<&[rustls::pki_types::CertificateDer<'_>]>) -> Result<[u8; 32]> {
    let leaf = certificates
        .and_then(|c| c.first())
        .context("peer did not present a certificate")?;
    Ok(ring::digest::digest(&ring::digest::SHA256, leaf.as_ref())
        .as_ref()
        .try_into()?)
}

async fn send_loop(
    peer: Peer,
    mut receive: mpsc::Receiver<Queued>,
    ready: Arc<AtomicBool>,
    connector: TlsConnector,
    options: Options,
    mut stop: watch::Receiver<bool>,
    statistics: Arc<Statistics>,
) -> Result<()> {
    let mut connection = None;
    loop {
        if connection.is_none() {
            // Connect independently of a Raft message. Offline destinations do
            // not accumulate a stream of queued payloads during reconnects.
            let connect = async {
                let tcp = TcpStream::connect(&peer.address).await?;
                tcp.set_nodelay(true)?;
                let stream = connector
                    .connect(ServerName::try_from(peer.server_name.clone())?, tcp)
                    .await?;
                ensure!(
                    stream.get_ref().1.alpn_protocol() == Some(ALPN)
                        && fingerprint(stream.get_ref().1.peer_certificates())? == peer.certificate,
                    "remote certificate or protocol differs from configured node"
                );
                Ok::<_, anyhow::Error>(stream)
            };
            let connected = tokio::select! {
                _ = stop.changed() => return Ok(()),
                result = timeout(options.io_timeout, connect) => result,
            };
            match connected {
                Ok(Ok(stream)) => {
                    connection = Some(stream);
                    ready.store(true, Ordering::Release);
                }
                _ => {
                    statistics.failed_exchanges.fetch_add(1, Ordering::Relaxed);
                    // Also release a packet dispatched concurrently with the
                    // preceding transition to disconnected.
                    while receive.try_recv().is_ok() {
                        statistics.dropped_outbound.fetch_add(1, Ordering::Relaxed);
                    }
                    tokio::select! {
                        _ = stop.changed() => return Ok(()),
                        _ = sleep(options.reconnect_delay) => {},
                    }
                    continue;
                }
            }
        }
        let packet = tokio::select! {
            _ = stop.changed() => return Ok(()),
            packet = receive.recv() => match packet { Some(p) => p, None => return Ok(()) },
        };
        // Retain both physical-runtime and per-destination credits through send.
        let exchange = write_packet(connection.as_mut().unwrap(), packet.packet.packet());
        let outcome = tokio::select! {
            _ = stop.changed() => return Ok(()),
            result = timeout(options.io_timeout, exchange) => result,
        };
        if matches!(outcome, Ok(Ok(()))) {
            statistics.sent_frames.fetch_add(1, Ordering::Relaxed);
        } else {
            ready.store(false, Ordering::Release);
            connection = None; // Never reuse an uncertain partial frame.
            statistics.failed_exchanges.fetch_add(1, Ordering::Relaxed);
            statistics.dropped_outbound.fetch_add(1, Ordering::Relaxed);
            drop(packet);
            while receive.try_recv().is_ok() {
                statistics.dropped_outbound.fetch_add(1, Ordering::Relaxed);
            }
            tokio::select! {
                _ = stop.changed() => return Ok(()),
                _ = sleep(options.reconnect_delay) => {},
            }
        }
    }
}

async fn listen(
    handle: Handle,
    listener: TcpListener,
    acceptor: TlsAcceptor,
    fingerprints: BTreeMap<[u8; 32], u64>,
    options: Options,
    mut stop: watch::Receiver<bool>,
    statistics: Arc<Statistics>,
) -> Result<()> {
    let connections = Arc::new(Semaphore::new(options.connections));
    let bytes = Arc::new(Semaphore::new(options.receive_bytes));
    let fingerprints = Arc::new(fingerprints);
    let mut tasks = JoinSet::new();
    loop {
        tokio::select! {
            _ = stop.changed() => break,
            Some(_) = tasks.join_next(), if !tasks.is_empty() => {},
            incoming = listener.accept() => {
                let (tcp, _) = incoming?;
                let Ok(slot) = connections.clone().try_acquire_owned() else {
                    statistics.rejected_connections.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
                tcp.set_nodelay(true)?;
                let (handle, acceptor, fingerprints, bytes, statistics, options) =
                    (handle.clone(), acceptor.clone(), fingerprints.clone(), bytes.clone(), statistics.clone(), options.clone());
                tasks.spawn(async move {
                    let _slot = slot;
                    let connection = async {
                        let stream = acceptor.accept(tcp).await?;
                        ensure!(stream.get_ref().1.alpn_protocol() == Some(ALPN), "foreign peer protocol");
                        let id = *fingerprints.get(&fingerprint(stream.get_ref().1.peer_certificates())?)
                            .context("unconfigured peer certificate")?;
                        Ok::<_, anyhow::Error>((stream, id))
                    };
                    let Ok(Ok((mut stream, peer))) = timeout(options.io_timeout, connection).await else {
                        statistics.rejected_connections.fetch_add(1, Ordering::Relaxed);
                        return;
                    };
                    loop {
                        let result = timeout(options.io_timeout, read_packet(&mut stream, &handle, peer, &bytes)).await;
                        match result {
                            Ok(Ok(frame)) => {
                                statistics.received_frames.fetch_add(1, Ordering::Relaxed);
                                // Runtime admission acquires its own credits before
                                // these framing credits are released (no ownership gap).
                                if handle.step(frame.packet).is_err() {
                                    statistics.rejected_packets.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            _ => {
                                statistics.failed_exchanges.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
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

struct Frame {
    packet: Packet,
    _header: OwnedSemaphorePermit,
    _body: OwnedSemaphorePermit,
}

fn encode(packet: &Packet) -> Result<(Vec<u8>, Vec<Bytes>)> {
    ensure!(
        packet.message.snapshot.is_none(),
        "snapshot transfer is not a peer frame"
    );
    ensure!(
        packet.message.compute_size() as usize <= MAX_BODY,
        "oversized Raft message"
    );
    let mut message = packet.message.clone(); // Entry.data clones share Bytes.
    ensure!(
        message.entries.len() <= MAX_ENTRIES,
        "too many Raft entries"
    );
    let bodies: Vec<_> = message
        .mut_entries()
        .iter_mut()
        .map(|entry| std::mem::take(&mut entry.data))
        .collect();
    let metadata = message.write_to_bytes()?;
    let size = FIXED
        .checked_add(4 * bodies.len())
        .and_then(|n| n.checked_add(metadata.len()))
        .context("peer header overflow")?;
    ensure!(size <= MAX_HEADER, "peer metadata exceeds header bound");
    let mut header = Vec::with_capacity(size);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(packet.cluster.as_bytes());
    header.extend_from_slice(&packet.layout);
    header.extend_from_slice(&packet.policy);
    header.extend_from_slice(&packet.group.to_be_bytes());
    header.extend_from_slice(&(bodies.len() as u32).to_be_bytes());
    header.extend_from_slice(&(metadata.len() as u32).to_be_bytes());
    for body in &bodies {
        header.extend_from_slice(&(body.len() as u32).to_be_bytes());
    }
    header.extend_from_slice(&metadata);
    Ok((header, bodies))
}
async fn write_packet<W: AsyncWrite + Unpin>(writer: &mut W, packet: &Packet) -> Result<()> {
    let (header, bodies) = encode(packet)?;
    crate::transport_io::write_frame(writer, &header, &bodies, None).await?;
    writer.flush().await?;
    Ok(())
}
async fn read_packet<R: AsyncRead + Unpin>(
    reader: &mut R,
    handle: &Handle,
    peer: u64,
    bytes: &Arc<Semaphore>,
) -> Result<Frame> {
    let size = reader.read_u32().await? as usize;
    ensure!(
        (FIXED..=MAX_HEADER).contains(&size),
        "invalid peer header length"
    );
    let header_credit = bytes
        .clone()
        .try_acquire_many_owned(size as u32)
        .context("peer header credits exhausted")?;
    let mut header = vec![0; size];
    reader.read_exact(&mut header).await?;
    ensure!(&header[..8] == MAGIC, "unsupported peer frame");
    ensure!(
        &header[8..24] == handle.cluster.as_bytes()
            && header[24..56] == handle.layout
            && header[56..88] == handle.policy,
        "foreign peer frame scope"
    );
    let group = u64::from_be_bytes(header[88..96].try_into()?);
    ensure!(
        handle
            .voters
            .get(&group)
            .is_some_and(|ids| ids.contains(&peer)),
        "peer does not vote in this group"
    );
    handle.route(group)?;
    let count = u32::from_be_bytes(header[96..100].try_into()?) as usize;
    let metadata = u32::from_be_bytes(header[100..104].try_into()?) as usize;
    ensure!(
        count <= MAX_ENTRIES && FIXED + 4 * count + metadata == size,
        "invalid peer entry framing"
    );
    let sizes: Vec<_> = header[FIXED..FIXED + 4 * count]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_be_bytes(*bytes) as usize)
        .collect();
    let total = sizes
        .iter()
        .try_fold(0usize, |n, len| n.checked_add(*len))
        .context("peer body overflow")?;
    ensure!(total <= MAX_BODY, "peer bodies exceed limit");
    let mut message = raft_proto::eraftpb::Message::parse_from_bytes(&header[FIXED + 4 * count..])?;
    ensure!(
        message.to == handle.id
            && message.from == peer
            && peer != handle.id
            && message.snapshot.is_none()
            && message.entries.len() == count
            && message.entries.iter().all(|entry| entry.data.is_empty()),
        "invalid authenticated peer route or metadata"
    );
    ensure!(
        (message.compute_size() as usize)
            .checked_add(total)
            .is_some_and(|n| n <= MAX_BODY),
        "oversized reconstructed Raft message"
    );
    let body_credit = bytes
        .clone()
        .try_acquire_many_owned(total as u32)
        .context("peer body credits exhausted")?;
    let mut buffer = BytesMut::with_capacity(total);
    // Tokio fills uninitialized spare capacity; no zero-fill or body concat is
    // needed. TLS encryption/decryption and kernel I/O can still copy bytes.
    let mut limited = reader.take(total as u64);
    while limited.limit() > 0 {
        ensure!(
            limited.read_buf(&mut buffer).await? > 0,
            "truncated peer body"
        );
    }
    let mut body = buffer.freeze();
    for (entry, size) in message.mut_entries().iter_mut().zip(sizes) {
        entry.data = body.split_to(size);
    }
    ensure!(
        message.compute_size() as usize <= MAX_BODY,
        "reconstructed Raft frame exceeds limit"
    );
    Ok(Frame {
        packet: Packet {
            cluster: handle.cluster,
            layout: handle.layout,
            policy: handle.policy,
            group,
            message,
        },
        _header: header_credit,
        _body: body_credit,
    })
}

#[cfg(test)]
mod tests;
