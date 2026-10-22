//! Authenticated image streams on separate sockets and one joined file worker.
//! Snapshot I/O never occupies ordinary peer sockets. A configured node disk
//! budget is required; it reserves exact file lengths before creation and keeps
//! installed and orphaned files charged alongside log growth. An optional extra
//! admission policy may impose stricter service limits. This is not an OS quota.
use super::*;
use crate::replication::checkpoint::{Image, Part, CHUNK_BYTES};
use std::{future::Future, thread};
use tokio::sync::oneshot;

pub const ALPN: &[u8] = b"lance-context-wal-snapshot/1";
const MAGIC: &[u8; 8] = b"LCWSNAP1";
const ACK: &[u8; 8] = b"LCWSYNC1";
const FIXED: usize = 96;
const MAX_HEADER: usize = FIXED + 4096;

/// Fully validated descriptor, before any receive file is created.
#[derive(Clone, Debug)]
pub struct Incoming {
    pub group: u64,
    pub from: u64,
    pub index: u64,
    pub metadata_bytes: u64,
    pub payload_bytes: u64,
}
impl Incoming {
    pub fn bytes(&self) -> Result<u64> {
        self.metadata_bytes
            .checked_add(self.payload_bytes)
            .context("snapshot size overflow")
    }
}

/// Optional additional admission, executed on the bounded file worker.
/// The required node budget is enforced independently before file creation.
/// The returned owner spans pending file work, even when its network task is
/// cancelled. Dropping it MUST NOT release space still used by installed or
/// orphaned files. Reclaim that charge only after actual collection. Reopening
/// the service must recover accounting from its existing files. There is no
/// weakening of the node budget when this additional policy is absent.
pub trait Admission: Send + Sync + 'static {
    fn reserve(&self, incoming: &Incoming) -> Result<Arc<dyn Send + Sync>>;
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "service", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "service", serde(default, deny_unknown_fields))]
pub struct Options {
    /// Total queued and active outgoing transfers across all destinations.
    pub outbound_transfers: usize,
    /// Includes TLS handshakes; independent of ordinary peer admission.
    pub connections: usize,
    pub max_image_bytes: u64,
    /// Whole transfer deadline, not reset on every chunk. A filesystem call
    /// already in progress is still joined, not cancelled, at shutdown.
    #[cfg_attr(feature = "service", serde(with = "crate::service::duration_millis"))]
    pub transfer_timeout: Duration,
    #[cfg(test)]
    #[cfg_attr(feature = "service", serde(skip))]
    pub(super) pause_after_payload_chunk: Option<std::path::PathBuf>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            outbound_transfers: 2,
            connections: 4,
            max_image_bytes: 64 << 30,
            transfer_timeout: Duration::from_secs(300),
            #[cfg(test)]
            pause_after_payload_chunk: None,
        }
    }
}

pub struct Settings {
    pub peers: BTreeMap<u64, Peer>,
    pub client_tls: Arc<rustls::ClientConfig>,
    pub server_tls: Arc<rustls::ServerConfig>,
    pub options: Options,
    pub admission: Option<Arc<dyn Admission>>,
}

#[derive(Default)]
pub struct Statistics {
    pub sent: AtomicU64,
    pub installed: AtomicU64,
    pub failed: AtomicU64,
    pub rejected: AtomicU64,
}

struct Queued {
    outgoing: Outbound,
    _slot: OwnedSemaphorePermit,
    _destination: OwnedSemaphorePermit,
}
struct Destination {
    send: mpsc::Sender<Queued>,
    slot: Arc<Semaphore>,
}
#[derive(Clone)]
pub struct Route {
    pub(super) owner: Arc<super::super::Shared>,
    peers: Arc<BTreeMap<u64, Destination>>,
    slots: Arc<Semaphore>,
}
impl Route {
    pub(super) fn enqueue(&self, outgoing: Outbound) -> bool {
        let Some(send) = self.peers.get(&outgoing.packet().message.to) else {
            return false;
        };
        let Ok(destination) = send.slot.clone().try_acquire_owned() else {
            return false;
        };
        let Ok(slot) = self.slots.clone().try_acquire_owned() else {
            return false;
        };
        send.send
            .try_send(Queued {
                outgoing,
                _slot: slot,
                _destination: destination,
            })
            .is_ok()
    }
}

type FileTask = Box<dyn FnOnce() + Send>;
#[derive(Clone)]
struct Files(mpsc::Sender<FileTask>);
impl Files {
    async fn run<T: Send + 'static>(
        &self,
        operation: impl FnOnce() -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (reply, result) = oneshot::channel();
        self.0
            .send(Box::new(move || {
                let _ = reply.send(operation());
            }))
            .await
            .map_err(|_| anyhow::anyhow!("snapshot file worker stopped"))?;
        result
            .await
            .context("snapshot file work failed; outcome unknown")?
    }
}

pub struct Transport {
    stop: watch::Sender<bool>,
    supervisor: Option<JoinHandle<Result<()>>>,
    route: Route,
    statistics: Arc<Statistics>,
}
impl Transport {
    pub fn is_finished(&self) -> bool {
        self.supervisor
            .as_ref()
            .is_none_or(|task| task.is_finished())
    }
    pub fn spawn(handle: Handle, listener: TcpListener, settings: Settings) -> Result<Self> {
        ensure!(
            handle.disk.is_some(),
            "snapshot transport requires configured node disk admission"
        );
        let Settings {
            peers,
            client_tls,
            server_tls,
            options,
            admission,
        } = settings;
        ensure!(
            (1..=16).contains(&options.outbound_transfers)
                && options.outbound_transfers < handle.shared.outgoing.capacity_slots
                && (1..=64).contains(&options.connections)
                && (4096..=1 << 40).contains(&options.max_image_bytes)
                && (Duration::from_millis(100)..=Duration::from_secs(3600))
                    .contains(&options.transfer_timeout),
            "invalid snapshot transfer bounds"
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
            "snapshot peers must match hosted voters"
        );
        let mut fingerprints = BTreeMap::new();
        for (&id, peer) in &peers {
            ensure!(
                !peer.address.is_empty()
                    && peer.address.len() <= 1024
                    && peer.certificate != [0; 32],
                "invalid snapshot peer"
            );
            ServerName::try_from(peer.server_name.clone())?;
            ensure!(
                fingerprints.insert(peer.certificate, id).is_none(),
                "duplicate snapshot certificate"
            );
        }
        ensure!(
            client_tls.alpn_protocols.iter().any(|p| p == ALPN)
                && server_tls.alpn_protocols.iter().any(|p| p == ALPN),
            "missing snapshot ALPN"
        );
        let (stop, shutdown) = watch::channel(false);
        let file_stop = stop.clone();
        let (file_send, mut file_receive) = mpsc::channel::<FileTask>(1);
        let file_worker = thread::Builder::new()
            .name("wal-image-files".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    while let Some(work) = file_receive.blocking_recv() {
                        work();
                    }
                }));
                if let Err(error) = result {
                    let _ = file_stop.send(true);
                    std::panic::resume_unwind(error);
                }
            })?;
        let files = Files(file_send);
        let statistics = Arc::new(Statistics::default());
        let mut tasks = JoinSet::new();
        let mut routes = BTreeMap::new();
        for (id, peer) in peers {
            let (send, receive) = mpsc::channel(1);
            routes.insert(
                id,
                Destination {
                    send,
                    slot: Arc::new(Semaphore::new(1)),
                },
            );
            tasks.spawn(send_loop(
                peer,
                receive,
                TlsConnector::from(client_tls.clone()),
                files.clone(),
                options.clone(),
                shutdown.clone(),
                statistics.clone(),
            ));
        }
        let route = Route {
            owner: handle.shared.clone(),
            peers: Arc::new(routes),
            slots: Arc::new(Semaphore::new(options.outbound_transfers)),
        };
        tasks.spawn(listen(
            handle,
            listener,
            TlsAcceptor::from(server_tls),
            Arc::new(fingerprints),
            files.clone(),
            options,
            admission,
            shutdown,
            statistics.clone(),
        ));
        let signal = stop.clone();
        let supervisor = tokio::spawn(async move {
            let mut error = None;
            while let Some(result) = tasks.join_next().await {
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        error = Some(e);
                        let _ = signal.send(true);
                    }
                    Err(e) => {
                        error = Some(e.into());
                        let _ = signal.send(true);
                    }
                }
            }
            // Every connection task has stopped producing jobs. Queued jobs
            // and any synchronous I/O finish before the native owner is freed.
            drop(files);
            if tokio::task::spawn_blocking(move || file_worker.join())
                .await?
                .is_err()
            {
                error = Some(anyhow::anyhow!("snapshot file worker panicked"));
            }
            error.map_or(Ok(()), Err)
        });
        Ok(Self {
            stop,
            supervisor: Some(supervisor),
            route,
            statistics,
        })
    }
    pub fn route(&self) -> Route {
        self.route.clone()
    }
    pub fn statistics(&self) -> &Arc<Statistics> {
        &self.statistics
    }
    pub async fn shutdown(mut self) -> Result<()> {
        let _ = self.stop.send(true);
        self.supervisor.take().expect("snapshot supervisor").await?
    }
}
impl Drop for Transport {
    fn drop(&mut self) {
        // Never abort the supervisor: it must retain and join all file work.
        // Call shutdown().await before reopening the runtime's directories.
        let _ = self.stop.send(true);
    }
}

async fn stopping(mut stop: watch::Receiver<bool>) {
    if !*stop.borrow() {
        let _ = stop.changed().await;
    }
}
async fn bounded<T>(
    stop: &watch::Receiver<bool>,
    duration: Duration,
    operation: impl Future<Output = Result<T>>,
) -> Result<T> {
    tokio::select! {
        _ = stopping(stop.clone()) => anyhow::bail!("snapshot service stopping"),
        result = timeout(duration, operation) => result.context("snapshot transfer deadline")?,
    }
}

async fn send_loop(
    peer: Peer,
    mut receive: mpsc::Receiver<Queued>,
    connector: TlsConnector,
    files: Files,
    options: Options,
    stop: watch::Receiver<bool>,
    statistics: Arc<Statistics>,
) -> Result<()> {
    loop {
        let queued = tokio::select! {
            _ = stopping(stop.clone()) => return Ok(()),
            queued = receive.recv() => match queued { Some(q) => q, None => return Ok(()) },
        };
        let result = bounded(&stop, options.transfer_timeout, async {
            let tcp = TcpStream::connect(&peer.address).await?;
            tcp.set_nodelay(true)?;
            let mut stream = connector
                .connect(ServerName::try_from(peer.server_name.clone())?, tcp)
                .await?;
            ensure!(
                stream.get_ref().1.alpn_protocol() == Some(ALPN)
                    && fingerprint(stream.get_ref().1.peer_certificates())? == peer.certificate,
                "foreign snapshot peer or protocol"
            );
            send_image(&mut stream, &queued.outgoing, &files, &options).await
        })
        .await;
        if result.is_ok() {
            queued.outgoing.finish_snapshot(true)?;
            statistics.sent.fetch_add(1, Ordering::Relaxed);
        } else {
            // Drop automatically reports failure with the original reserved slot.
            statistics.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// Each connection consumes bounded network memory, but only one filesystem
// operation runs at once. Node's physical image permit also serializes receive,
// build and collection across every group hosted on this engine.
#[allow(clippy::too_many_arguments)]
async fn listen(
    handle: Handle,
    listener: TcpListener,
    acceptor: TlsAcceptor,
    fingerprints: Arc<BTreeMap<[u8; 32], u64>>,
    files: Files,
    options: Options,
    admission: Option<Arc<dyn Admission>>,
    stop: watch::Receiver<bool>,
    statistics: Arc<Statistics>,
) -> Result<()> {
    let slots = Arc::new(Semaphore::new(options.connections));
    let mut tasks = JoinSet::new();
    let outcome = loop {
        tokio::select! {
            _ = stopping(stop.clone()) => break Ok(()),
            Some(_) = tasks.join_next(), if !tasks.is_empty() => {},
            incoming = listener.accept() => {
                let (tcp, _) = match incoming { Ok(v) => v, Err(e) => break Err(e.into()) };
                let Ok(slot) = slots.clone().try_acquire_owned() else {
                    statistics.rejected.fetch_add(1, Ordering::Relaxed); continue;
                };
                let (handle, acceptor, fingerprints, files, options, admission, stop, statistics) =
                    (handle.clone(), acceptor.clone(), fingerprints.clone(), files.clone(), options.clone(),
                        admission.clone(), stop.clone(), statistics.clone());
                tasks.spawn(async move {
                    let _slot = slot;
                    let result = bounded(&stop, options.transfer_timeout, async {
                        tcp.set_nodelay(true)?;
                        let mut stream = acceptor.accept(tcp).await?;
                        ensure!(stream.get_ref().1.alpn_protocol() == Some(ALPN), "foreign snapshot protocol");
                        let peer = *fingerprints.get(&fingerprint(stream.get_ref().1.peer_certificates())?)
                            .context("unconfigured snapshot certificate")?;
                        receive_image(&mut stream, &handle, peer, &files, &options, admission).await
                    }).await;
                    if result.is_ok() { statistics.installed.fetch_add(1, Ordering::Relaxed); }
                    else { statistics.failed.fetch_add(1, Ordering::Relaxed); }
                });
            }
        }
    };
    // On listener failure these children still have their finite deadlines.
    // On shutdown they observe the same stop signal. Do not abort their owner.
    while tasks.join_next().await.is_some() {}
    outcome
}

fn encode(packet: &Packet) -> Result<Vec<u8>> {
    ensure!(
        packet.message.get_msg_type() == raft_proto::eraftpb::MessageType::MsgSnapshot
            && packet.message.entries.is_empty()
            && packet.message.has_snapshot()
            && packet.message.compute_size() <= 4096,
        "invalid outgoing snapshot message"
    );
    let mut header = Vec::with_capacity(MAX_HEADER);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(packet.cluster.as_bytes());
    header.extend_from_slice(&packet.layout);
    header.extend_from_slice(&packet.policy);
    header.extend_from_slice(&packet.group.to_be_bytes());
    header.extend_from_slice(&packet.message.write_to_bytes()?);
    Ok(header)
}
async fn read_header<R: AsyncRead + Unpin>(
    reader: &mut R,
    handle: &Handle,
    peer: u64,
) -> Result<Packet> {
    let size = reader.read_u32().await? as usize;
    ensure!(
        (FIXED..=MAX_HEADER).contains(&size),
        "invalid snapshot header length"
    );
    let mut header = vec![0; size];
    reader.read_exact(&mut header).await?;
    ensure!(
        &header[..8] == MAGIC
            && &header[8..24] == handle.cluster.as_bytes()
            && header[24..56] == handle.layout
            && header[56..88] == handle.policy,
        "foreign snapshot scope"
    );
    let group = u64::from_be_bytes(header[88..96].try_into()?);
    ensure!(
        handle.voters.get(&group).is_some_and(|v| v.contains(&peer)),
        "foreign snapshot voter"
    );
    let message = raft_proto::eraftpb::Message::parse_from_bytes(&header[FIXED..])?;
    ensure!(
        message.from == peer
            && message.to == handle.id
            && peer != handle.id
            && message.get_msg_type() == raft_proto::eraftpb::MessageType::MsgSnapshot
            && message.entries.is_empty()
            && message.has_snapshot()
            && message.compute_size() <= 4096,
        "invalid authenticated snapshot route"
    );
    Ok(Packet {
        cluster: handle.cluster,
        layout: handle.layout,
        policy: handle.policy,
        group,
        message,
    })
}

async fn send_image<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    outgoing: &Outbound,
    files: &Files,
    options: &Options,
) -> Result<()> {
    let image = outgoing
        .snapshot_image()
        .context("snapshot has no captured image")?;
    ensure!(
        image.snapshot()? == *outgoing.packet().message.get_snapshot(),
        "snapshot capture changed"
    );
    ensure!(
        image
            .bytes(Part::Metadata)
            .checked_add(image.bytes(Part::Payload))
            .is_some_and(|bytes| bytes <= options.max_image_bytes),
        "snapshot exceeds transfer budget"
    );
    let header = encode(outgoing.packet())?;
    stream.write_u32(header.len() as u32).await?;
    stream.write_all(&header).await?;
    stream.flush().await?;
    ensure!(
        stream.read_u8().await? == 1,
        "snapshot receive admission denied"
    );
    for part in [Part::Metadata, Part::Payload] {
        let mut offset = 0;
        while offset < image.bytes(part) {
            let size = (image.bytes(part) - offset).min(CHUNK_BYTES as u64) as usize;
            let input: Arc<Image> = image.clone();
            let chunk = files.run(move || input.chunk(part, offset, size)).await?;
            stream.write_all(&chunk).await?;
            offset += chunk.len() as u64;
        }
    }
    stream.flush().await?;
    let mut ack = [0; 8];
    stream.read_exact(&mut ack).await?;
    ensure!(&ack == ACK, "snapshot durable response missing");
    Ok(())
}

async fn receive_image<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    handle: &Handle,
    peer: u64,
    files: &Files,
    options: &Options,
    admission: Option<Arc<dyn Admission>>,
) -> Result<()> {
    let packet = read_header(stream, handle, peer).await?;
    let sizes = handle.snapshot_sizes(packet.clone())?.await?;
    let incoming = Incoming {
        group: packet.group,
        from: peer,
        index: packet.message.get_snapshot().get_metadata().index,
        metadata_bytes: sizes[0],
        payload_bytes: sizes[1],
    };
    ensure!(
        incoming.bytes()? <= options.max_image_bytes,
        "snapshot exceeds receive budget"
    );
    let reservation = files
        .run(move || admission.map(|a| a.reserve(&incoming)).transpose())
        .await?;
    let mut receiver = handle
        .receive_snapshot(packet, uuid::Uuid::new_v4())?
        .await?;
    stream.write_u8(1).await?;
    stream.flush().await?;
    for (part, length) in [Part::Metadata, Part::Payload].into_iter().zip(sizes) {
        let mut offset = 0;
        while offset < length {
            let size = (length - offset).min(CHUNK_BYTES as u64) as usize;
            let mut buffer = BytesMut::with_capacity(size);
            let mut limited = (&mut *stream).take(size as u64);
            while limited.limit() > 0 {
                ensure!(
                    limited.read_buf(&mut buffer).await? > 0,
                    "truncated snapshot file"
                );
            }
            let bytes = buffer.freeze();
            let keep = reservation.clone();
            #[cfg(test)]
            let pause = options
                .pause_after_payload_chunk
                .clone()
                .filter(|_| part == Part::Payload && offset == 0);
            receiver = files
                .run(move || {
                    let _keep = keep;
                    receiver.write(part, offset, &bytes)?;
                    #[cfg(test)]
                    if let Some(marker) = pause {
                        std::fs::write(&marker, (offset + bytes.len() as u64).to_string())?;
                        let deadline = std::time::Instant::now() + Duration::from_secs(15);
                        while !marker.with_extension("resume").exists() {
                            ensure!(
                                std::time::Instant::now() < deadline,
                                "test chunk pause expired"
                            );
                            thread::sleep(Duration::from_millis(5));
                        }
                    }
                    Ok(receiver)
                })
                .await?;
            offset += size as u64;
        }
    }
    let keep = reservation.clone();
    let prepared = files
        .run(move || {
            let _keep = keep;
            receiver.finish()
        })
        .await?;
    handle.step_snapshot(prepared)?.await?;
    stream.write_all(ACK).await?;
    stream.flush().await?;
    Ok(())
}
