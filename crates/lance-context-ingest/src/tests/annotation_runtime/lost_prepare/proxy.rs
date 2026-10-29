//! Test-only TLS proxy: forward the actual command, read its quorum result,
//! hold one successful Prepare or Close reply until released, then disconnect.
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

#[derive(serde::Deserialize)]
struct WireScope {
    _cluster: Uuid,
    _layout: [u8; 32],
    _policy: [u8; 32],
    _node: u64,
}

pub(super) struct Proxy {
    pub client: Arc<Client>,
    pub dropped: Arc<AtomicUsize>,
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
}

#[derive(Clone, Copy)]
enum Point {
    Prepare,
    Close,
}
impl Frame {
    async fn read(io: &mut (impl AsyncRead + Unpin)) -> Result<Option<Self>> {
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
            (1..=2 << 20).contains(&size) && fixed[20..24] == [0; 4],
            "invalid control frame"
        );
        let mut meta = vec![0; size];
        io.read_exact(&mut meta).await?;
        Ok(Some(Self { fixed, meta }))
    }
    async fn write(&self, io: &mut (impl AsyncWrite + Unpin)) -> Result<()> {
        io.write_all(&self.fixed).await?;
        io.write_all(&self.meta).await?;
        io.flush().await?;
        Ok(())
    }
    fn matches_command(&self, point: Point) -> Result<bool> {
        // The fixed-int wire enum's control discriminant is 1. Decode all of
        // that command, forwarding original bytes without re-encoding them.
        let (_, kind): (WireScope, u32) = codec().allow_trailing_bytes().deserialize(&self.meta)?;
        if kind != 1 {
            return Ok(false);
        }
        let (_, _, command): (WireScope, u32, Command) =
            codec().reject_trailing_bytes().deserialize(&self.meta)?;
        Ok(matches!(
            (point, command),
            (Point::Prepare, Command::Prepare(_)) | (Point::Close, Command::Close(_))
        ))
    }
    fn successful_reply(&self, point: Point) -> Result<bool> {
        let (_, reply): (WireScope, Reply) =
            codec().reject_trailing_bytes().deserialize(&self.meta)?;
        Ok(matches!(
            (point, reply),
            (
                Point::Prepare,
                Reply::Commit(Outcome::Committed(Applied::Publisher(
                    p::Reply::Prepared { .. }
                )))
            ) | (
                Point::Close,
                Reply::Commit(Outcome::Committed(Applied::Source(
                    lance_context_wal::replication::source::Reply::Closed { .. }
                )))
            )
        ))
    }
}

impl Proxy {
    pub async fn open(c: &network::Cluster, root: &FsPath) -> Result<Self> {
        Self::open_at(c, root, Point::Prepare).await
    }

    pub async fn open_close(c: &network::Cluster, root: &FsPath) -> Result<Self> {
        Self::open_at(c, root, Point::Close).await
    }

    async fn open_at(c: &network::Cluster, root: &FsPath, point: Point) -> Result<Self> {
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
        let committed = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let mut tasks = JoinSet::new();
        for (&id, endpoint) in &mut config.endpoints {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let address = endpoint.control_address.clone();
            endpoint.control_address = listener.local_addr()?.to_string();
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
                            connections.spawn(async move {
                                let mut downstream = acceptor.accept(socket).await?;
                                let mut upstream = connector.connect(name, TcpStream::connect(address).await?).await?;
                                while let Some(request) = Frame::read(&mut downstream).await? {
                                    let hold = request.matches_command(point)?;
                                    request.write(&mut upstream).await?;
                                    let response = Frame::read(&mut upstream).await?.context("upstream response missing")?;
                                    ensure!(response.fixed[8..16] == request.fixed[8..16], "proxy response ID changed");
                                    if hold && response.successful_reply(point)? && armed.swap(false, Ordering::SeqCst) {
                                        committed.notify_one();
                                        release.notified().await;
                                        dropped.fetch_add(1, Ordering::SeqCst);
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
                lance_context_wal::replication::runtime::client::Options::default(),
            )?),
            dropped,
            committed,
            release,
            _tasks: tasks,
        })
    }
}
