//! Test-only TLS proxy: forward the actual command, read its quorum result,
//! withhold one completion reply or return an uncertain control outcome.
use super::*;
use bincode::Options as _;
use std::path::Path as FsPath;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::Notify;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinSet,
};
use tokio_rustls::{
    rustls::{
        self,
        pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, ServerName},
        RootCertStore,
    },
    TlsAcceptor, TlsConnector,
};

#[derive(serde::Serialize, serde::Deserialize)]
struct WireScope {
    _cluster: Uuid,
    _layout: [u8; 32],
    _policy: [u8; 32],
    _node: u64,
}

pub(in crate::tests) struct Proxy {
    pub client: Arc<Client>,
    pub dropped: Arc<AtomicUsize>,
    pub plans: Arc<AtomicUsize>,
    pub prepares: Arc<AtomicUsize>,
    pub blocked_reads: Arc<AtomicUsize>,
    pub advances: Arc<AtomicUsize>,
    pub appends: Arc<AtomicUsize>,
    pub closes: Arc<AtomicUsize>,
    pub unknown_closes: Arc<AtomicBool>,
    pub close_permits:
        Arc<std::sync::Mutex<Vec<lance_context_wal::replication::source::ClosurePermit>>>,
    pub committed: Arc<Notify>,
    pub release: Arc<Notify>,
    _tasks: JoinSet<Result<()>>,
}

fn codec() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(2 << 20)
}

fn certificates(path: &FsPath) -> Result<Vec<CertificateDer<'static>>> {
    Ok(CertificateDer::pem_slice_iter(&std::fs::read(path)?)
        .collect::<std::result::Result<_, _>>()?)
}

struct Frame {
    fixed: [u8; 24],
    meta: Vec<u8>,
    body: Vec<u8>,
}

#[derive(Clone, Copy)]
enum Point {
    Prepare {
        execute: bool,
        observe: bool,
        structured: bool,
    },
    Append {
        execute: bool,
    },
    Close {
        execute: bool,
    },
    Plan {
        execute: bool,
    },
    Advance {
        execute: bool,
    },
}
impl Frame {
    async fn read(io: &mut (impl AsyncRead + Unpin), data: bool) -> Result<Option<Self>> {
        let mut fixed = [0; 24];
        match io.read_exact(&mut fixed[..1]).await {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            result => {
                result?;
            }
        }
        io.read_exact(&mut fixed[1..]).await?;
        ensure!(&fixed[..8] == b"LCWCLNT1", "foreign proxy protocol");
        let size = u32::from_le_bytes(fixed[16..20].try_into()?) as usize;
        ensure!(
            (1..=2 << 20).contains(&size)
                && if data {
                    u32::from_le_bytes(fixed[20..24].try_into()?) <= 16 << 20
                } else {
                    fixed[20..24] == [0; 4]
                },
            "invalid control frame"
        );
        let mut meta = vec![0; size];
        io.read_exact(&mut meta).await?;
        Ok(Some(Self {
            fixed,
            meta,
            body: Vec::new(),
        }))
    }
    async fn write(&self, io: &mut (impl AsyncWrite + Unpin)) -> Result<()> {
        io.write_all(&self.fixed).await?;
        io.write_all(&self.meta).await?;
        io.write_all(&self.body).await?;
        io.flush().await?;
        Ok(())
    }
    async fn payload(&mut self, io: &mut (impl AsyncRead + Unpin), request: bool) -> Result<()> {
        let (_, kind): (WireScope, u32) = codec().allow_trailing_bytes().deserialize(&self.meta)?;
        let bytes = if request && kind == 0 {
            let (_, _, _, meta): (WireScope, u32, u64, BatchMeta) =
                codec().reject_trailing_bytes().deserialize(&self.meta)?;
            meta.body_bytes
        } else if !request && kind == 2 {
            let (_, _, meta): (WireScope, u32, Option<BatchMeta>) =
                codec().reject_trailing_bytes().deserialize(&self.meta)?;
            meta.map_or(0, |m| m.body_bytes)
        } else {
            0
        };
        ensure!(
            bytes <= 16 << 20
                && bytes == u64::from(u32::from_le_bytes(self.fixed[20..24].try_into()?)),
            "invalid test proxy body length"
        );
        self.body.resize(bytes as usize, 0);
        io.read_exact(&mut self.body).await?;
        Ok(())
    }
    fn matches_command(&self, point: Point) -> Result<bool> {
        // The fixed-int wire enum's control discriminant is 1. Decode all of
        // that command, forwarding original bytes without re-encoding them.
        let (_, kind): (WireScope, u32) = codec().allow_trailing_bytes().deserialize(&self.meta)?;
        if matches!(point, Point::Append { .. }) {
            return Ok(kind == 0);
        }
        if kind != 1 {
            return Ok(false);
        }
        let (_, _, command): (WireScope, u32, Command) =
            codec().reject_trailing_bytes().deserialize(&self.meta)?;
        Ok(matches!(
            (point, command),
            (Point::Prepare { .. }, Command::Prepare(_))
                | (Point::Close { .. }, Command::Close(_))
                | (Point::Plan { .. }, Command::Plan(_))
                | (Point::Advance { .. }, Command::Advance(_))
        ))
    }
    fn publisher_query(&self) -> Result<bool> {
        let (_, kind): (WireScope, u32) = codec().allow_trailing_bytes().deserialize(&self.meta)?;
        if kind != 2 {
            return Ok(false);
        }
        let (_, _, _, query): (
            WireScope,
            u32,
            u64,
            lance_context_wal::replication::query::Query,
        ) = codec().reject_trailing_bytes().deserialize(&self.meta)?;
        Ok(matches!(
            query,
            lance_context_wal::replication::query::Query::Publisher { .. }
        ))
    }
    fn unknown_reply(&self) -> Result<Self> {
        let (scope, _): (WireScope, u32) =
            codec().allow_trailing_bytes().deserialize(&self.meta)?;
        let meta = codec().serialize(&(scope, Reply::Commit(Outcome::Unknown)))?;
        let mut fixed = self.fixed;
        fixed[16..20].copy_from_slice(&u32::try_from(meta.len())?.to_le_bytes());
        fixed[20..24].fill(0);
        Ok(Self {
            fixed,
            meta,
            body: Vec::new(),
        })
    }
    fn successful_reply(&self, point: Point) -> Result<bool> {
        let (_, reply): (WireScope, Reply) =
            codec().reject_trailing_bytes().deserialize(&self.meta)?;
        Ok(matches!(
            (point, reply),
            (
                Point::Append { .. },
                Reply::Commit(Outcome::Committed(Applied::Accepted { .. }))
            ) | (
                Point::Prepare { .. },
                Reply::Commit(Outcome::Committed(Applied::Publisher(
                    p::Reply::Prepared { .. }
                )))
            ) | (
                Point::Plan { .. },
                Reply::Commit(Outcome::Committed(Applied::Publisher(
                    p::Reply::Planned { .. }
                )))
            ) | (
                Point::Advance { .. },
                Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Advanced(
                    _
                ))))
            ) | (
                Point::Close { .. },
                Reply::Commit(Outcome::Committed(Applied::Source(
                    lance_context_wal::replication::source::Reply::Closed { .. }
                )))
            )
        ))
    }
}

impl Proxy {
    pub async fn open_append(c: &network::Cluster, root: &FsPath, execute: bool) -> Result<Self> {
        Self::open_at(c, root, Point::Append { execute }).await
    }
    pub async fn open(c: &network::Cluster, root: &FsPath) -> Result<Self> {
        Self::open_at(
            c,
            root,
            Point::Prepare {
                execute: true,
                observe: false,
                structured: false,
            },
        )
        .await
    }

    pub async fn open_prepare(
        c: &network::Cluster,
        root: &FsPath,
        execute: bool,
        observe: bool,
    ) -> Result<Self> {
        Self::open_at(
            c,
            root,
            Point::Prepare {
                execute,
                observe,
                structured: true,
            },
        )
        .await
    }

    pub async fn open_close(c: &network::Cluster, root: &FsPath) -> Result<Self> {
        Self::open_at(c, root, Point::Close { execute: true }).await
    }

    pub async fn close_fault(
        c: &network::Cluster,
        root: &FsPath,
        execute: bool,
        timeout: Duration,
    ) -> Result<Self> {
        Self::open_at_with_timeout(c, root, Point::Close { execute }, timeout).await
    }

    pub async fn open_plan(c: &network::Cluster, root: &FsPath, execute: bool) -> Result<Self> {
        Self::open_at(c, root, Point::Plan { execute }).await
    }

    pub async fn open_advance(c: &network::Cluster, root: &FsPath, execute: bool) -> Result<Self> {
        Self::open_at(c, root, Point::Advance { execute }).await
    }

    async fn open_at(c: &network::Cluster, root: &FsPath, point: Point) -> Result<Self> {
        Self::open_at_with_timeout(c, root, point, Duration::from_secs(30)).await
    }

    async fn open_at_with_timeout(
        c: &network::Cluster,
        root: &FsPath,
        point: Point,
        timeout: Duration,
    ) -> Result<Self> {
        let mut roots = RootCertStore::empty();
        for cert in certificates(&root.join("roots.pem"))? {
            roots.add(cert)?;
        }
        let roots = Arc::new(roots);
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut outbound = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()?
            .with_root_certificates(roots.clone())
            .with_client_auth_cert(
                certificates(&root.join("tls-4.pem"))?,
                PrivateKeyDer::from_pem_slice(&std::fs::read(root.join("tls-4.key"))?)?,
            )?;
        outbound.alpn_protocols =
            vec![lance_context_wal::replication::runtime::client::ALPN.to_vec()];
        let outbound = Arc::new(outbound);
        let connector = TlsConnector::from(outbound.clone());
        let mut config = c.application.clone();
        let armed = Arc::new(AtomicBool::new(true));
        let dropped = Arc::new(AtomicUsize::new(0));
        let plans = Arc::new(AtomicUsize::new(0));
        let prepares = Arc::new(AtomicUsize::new(0));
        let blocked_reads = Arc::new(AtomicUsize::new(0));
        let advances = Arc::new(AtomicUsize::new(0));
        let appends = Arc::new(AtomicUsize::new(0));
        let closes = Arc::new(AtomicUsize::new(0));
        let unknown_closes = Arc::new(AtomicBool::new(false));
        let close_permits = Arc::new(std::sync::Mutex::new(Vec::new()));
        let committed = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let mut tasks = JoinSet::new();
        for (&id, endpoint) in &mut config.endpoints {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let destination = if matches!(point, Point::Append { .. }) {
                &mut endpoint.data_address
            } else {
                &mut endpoint.control_address
            };
            let address = destination.clone();
            *destination = listener.local_addr()?.to_string();
            let name = ServerName::try_from(endpoint.server_name.clone())?;
            let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
                roots.clone(),
                provider.clone(),
            )
            .build()?;
            let mut inbound = rustls::ServerConfig::builder_with_provider(provider.clone())
                .with_safe_default_protocol_versions()?
                .with_client_cert_verifier(verifier)
                .with_single_cert(
                    certificates(&root.join(format!("tls-{id}.pem")))?,
                    PrivateKeyDer::from_pem_slice(&std::fs::read(
                        root.join(format!("tls-{id}.key")),
                    )?)?,
                )?;
            inbound.alpn_protocols =
                vec![lance_context_wal::replication::runtime::client::ALPN.to_vec()];
            let acceptor = TlsAcceptor::from(Arc::new(inbound));
            let connector = connector.clone();
            let armed = armed.clone();
            let dropped = dropped.clone();
            let plans = plans.clone();
            let prepares = prepares.clone();
            let blocked_reads = blocked_reads.clone();
            let advances = advances.clone();
            let appends = appends.clone();
            let closes = closes.clone();
            let unknown_closes = unknown_closes.clone();
            let close_permits = close_permits.clone();
            let committed = committed.clone();
            let release = release.clone();
            tasks.spawn(async move {
                let mut connections = JoinSet::new();
                loop {
                    tokio::select! {
                        accepted = listener.accept() => {
                            let (socket, _) = accepted?;
                            let (acceptor, connector, name, address, armed, dropped, committed, release) =
                                (acceptor.clone(), connector.clone(), name.clone(), address.clone(), armed.clone(), dropped.clone(), committed.clone(), release.clone());
                            let plans = plans.clone();
                            let prepares = prepares.clone();
                            let blocked_reads = blocked_reads.clone();
                            let advances = advances.clone();
                            let appends = appends.clone();
                            let closes = closes.clone();
                            let unknown_closes = unknown_closes.clone();
                            let close_permits = close_permits.clone();
                            connections.spawn(async move {
                                let mut downstream = acceptor.accept(socket).await?;
                                let mut upstream = connector.connect(name, TcpStream::connect(address).await?).await?;
                                let mut used = false;
                                while let Some(mut request) = Frame::read(&mut downstream, matches!(point, Point::Append { .. })).await? {
                                    if matches!(point, Point::Append { .. }) { request.payload(&mut downstream, true).await?; }
                                    if matches!(point, Point::Prepare { observe: false, .. }) && dropped.load(Ordering::SeqCst) > 0 && request.publisher_query()? && blocked_reads.load(Ordering::SeqCst) == 0 {
                                        // A pooled read may reconnect once. Keep the fault armed
                                        // until its fresh connection also fails, so one logical
                                        // confirmation is unavailable even with a warm pool.
                                        if !used { blocked_reads.store(1, Ordering::SeqCst); }
                                        return Ok(());
                                    }
                                    used = true;
                                    let hold = request.matches_command(point)?;
                                    if hold && matches!(point, Point::Prepare { .. }) {
                                        prepares.fetch_add(1, Ordering::SeqCst);
                                        if matches!(point, Point::Prepare { execute: false, .. }) && armed.swap(false, Ordering::SeqCst) {
                                            dropped.fetch_add(1, Ordering::SeqCst);
                                            request.unknown_reply()?.write(&mut downstream).await?;
                                            continue;
                                        }
                                    }
                                    if hold && matches!(point, Point::Close { .. }) {
                                        closes.fetch_add(1, Ordering::SeqCst);
                                        let (_, _, command): (WireScope, u32, Command) = codec().reject_trailing_bytes().deserialize(&request.meta)?;
                                        let Command::Close(permit) = command else { bail!("expected a Close command"); };
                                        close_permits.lock().unwrap().push(permit);
                                        if unknown_closes.load(Ordering::SeqCst) {
                                            dropped.fetch_add(1, Ordering::SeqCst);
                                            request.unknown_reply()?.write(&mut downstream).await?;
                                            continue;
                                        }
                                    }
                                    if hold && matches!(point, Point::Append { execute: false } | Point::Close { execute: false }) && armed.swap(false, Ordering::SeqCst) {
                                        committed.notify_one(); release.notified().await;
                                        dropped.fetch_add(1, Ordering::SeqCst);
                                        request.unknown_reply()?.write(&mut downstream).await?;
                                        continue;
                                    }
                                    if hold && matches!(point, Point::Plan { .. } | Point::Advance { .. }) {
                                        let submissions = if matches!(point, Point::Plan { .. }) { &plans } else { &advances };
                                        submissions.fetch_add(1, Ordering::SeqCst);
                                        if matches!(point, Point::Plan { execute: false } | Point::Advance { execute: false }) && armed.swap(false, Ordering::SeqCst) {
                                            dropped.fetch_add(1, Ordering::SeqCst);
                                            request.unknown_reply()?.write(&mut downstream).await?;
                                            continue;
                                        }
                                    }
                                    request.write(&mut upstream).await?;
                                    let mut response = Frame::read(&mut upstream, matches!(point, Point::Append { .. })).await?.context("upstream response missing")?;
                                    if matches!(point, Point::Append { .. }) {
                                        response.payload(&mut upstream, false).await?;
                                        if hold && response.successful_reply(point)? { appends.fetch_add(1, Ordering::SeqCst); }
                                    }
                                    ensure!(response.fixed[8..16] == request.fixed[8..16], "proxy response ID changed");
                                    if hold && response.successful_reply(point)? && armed.swap(false, Ordering::SeqCst) {
                                        committed.notify_one();
                                        release.notified().await;
                                        dropped.fetch_add(1, Ordering::SeqCst);
                                        if matches!(point, Point::Plan { .. } | Point::Advance { .. } | Point::Append { .. } | Point::Prepare { structured: true, .. }) {
                                            request.unknown_reply()?.write(&mut downstream).await?;
                                            continue;
                                        }
                                        return Ok::<_, anyhow::Error>(());
                                    }
                                    response.write(&mut downstream).await?;
                                }
                                Ok(())
                            });
                        }
                        finished = connections.join_next(), if !connections.is_empty() => {
                            // A peer can retire another pooled connection after
                            // the injected disconnect. Keep the listener alive;
                            // this fixture drops one reply, not an entire node.
                            if let Err(error) = finished.unwrap()? {
                                eprintln!("lost-prepare proxy connection ended: {error:#}");
                            }
                        }
                    }
                }
            });
        }
        let endpoints = config
            .endpoints
            .into_iter()
            .map(|(id, e)| {
                (
                    id,
                    lance_context_wal::replication::runtime::client::Endpoint {
                        data_address: e.data_address,
                        control_address: e.control_address,
                        server_name: e.server_name,
                        certificate: e.certificate,
                    },
                )
            })
            .collect();
        Ok(Self {
            client: Arc::new(Client::new(
                &config.layout,
                &config.limits,
                endpoints,
                outbound,
                lance_context_wal::replication::runtime::client::Options {
                    timeout,
                    ..Default::default()
                },
            )?),
            dropped,
            plans,
            prepares,
            blocked_reads,
            advances,
            appends,
            closes,
            unknown_closes,
            close_permits,
            committed,
            release,
            _tasks: tasks,
        })
    }
}
