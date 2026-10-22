use super::*;
use lance_context_wal::{
    placement::Layout,
    raft_log::{DiskOptions, LogOptions},
    replication::{
        runtime::{
            self,
            peer::{self, snapshot},
        },
        source::{CatalogSigner, SourceSigner},
        Limits,
    },
    service::{
        config::{Endpoint, Listen, SigningFiles, TlsFiles},
        maintenance, Config,
    },
};
use ring::{rand::SystemRandom, signature::Ed25519KeyPair};
use std::{
    collections::BTreeSet,
    fs,
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command, Stdio},
};
use tokio_rustls::rustls::{
    self,
    pki_types::{CertificateDer, PrivatePkcs8KeyDer},
    RootCertStore,
};

pub struct Fixture {
    pub root: tempfile::TempDir,
    pub configs: BTreeMap<u64, Config>,
    pub client: Arc<Client>,
    pub layout: Arc<Layout>,
}
impl Fixture {
    pub fn new() -> Result<Self> {
        let root = tempfile::tempdir()?;
        let mut leases = Vec::new();
        let mut listens = BTreeMap::new();
        for node in 1..=6 {
            let mut ports = Vec::new();
            for _ in 0..5 {
                let listener = TcpListener::bind("127.0.0.1:0")?;
                ports.push(listener.local_addr()?);
                leases.push(listener);
            }
            listens.insert(
                node,
                Listen {
                    peer: ports[0],
                    snapshot: ports[1],
                    data: ports[2],
                    control: ports[3],
                    health: ports[4],
                },
            );
        }
        let layout = Arc::new(Layout {
            cluster: Uuid::from_u128(1),
            nodes: listens
                .iter()
                .map(|(&id, l)| (id, l.peer.to_string()))
                .collect(),
            groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![4, 5, 6])]),
            buckets: vec![10, 20, 10, 20],
        });
        let mut roots = RootCertStore::empty();
        let mut roots_pem = String::new();
        let mut certs = BTreeMap::new();
        let mut keys = BTreeMap::new();
        let mut fingerprints = BTreeMap::new();
        for id in 1..=7 {
            let certified =
                rcgen::generate_simple_self_signed(vec![format!("wal-{id}.example.invalid")])?;
            let der: CertificateDer<'static> = certified.cert.der().clone();
            roots.add(der.clone())?;
            roots_pem.push_str(&certified.cert.pem());
            fs::write(root.path().join(format!("{id}.crt")), certified.cert.pem())?;
            fs::write(
                root.path().join(format!("{id}.key")),
                certified.signing_key.serialize_pem(),
            )?;
            fingerprints.insert(
                id,
                ring::digest::digest(&ring::digest::SHA256, der.as_ref())
                    .as_ref()
                    .try_into()?,
            );
            certs.insert(id, der);
            keys.insert(id, certified.signing_key.serialize_der());
        }
        fs::write(root.path().join("roots.pem"), roots_pem)?;
        let catalog = Ed25519KeyPair::generate_pkcs8(&SystemRandom::new())
            .map_err(|_| anyhow::anyhow!("generate fixture catalog key"))?;
        fs::write(root.path().join("catalog.pk8"), catalog.as_ref())?;
        let mut source_keys = BTreeMap::new();
        for group in [10, 20] {
            let key = Ed25519KeyPair::generate_pkcs8(&SystemRandom::new())
                .map_err(|_| anyhow::anyhow!("generate fixture source key"))?;
            fs::write(
                root.path().join(format!("source-{group}.pk8")),
                key.as_ref(),
            )?;
            source_keys.insert(group, SourceSigner::from_pkcs8(key.as_ref())?.public_key());
        }
        let limits = Limits {
            catalog_key: Some(CatalogSigner::from_pkcs8(catalog.as_ref())?.public_key()),
            source_keys,
            producer_slots: 16,
            max_record_bytes: 1 << 20,
            ..Limits::default()
        };
        let endpoints: BTreeMap<_, _> = listens
            .iter()
            .map(|(&id, l)| {
                (
                    id,
                    Endpoint {
                        peer: l.peer.to_string(),
                        snapshot: l.snapshot.to_string(),
                        data: l.data.to_string(),
                        control: l.control.to_string(),
                        server_name: format!("wal-{id}.example.invalid"),
                        certificate: fingerprints[&id],
                    },
                )
            })
            .collect();
        let mut configs = BTreeMap::new();
        for (&node, listen) in &listens {
            let groups: Vec<_> = layout
                .groups
                .iter()
                .filter(|(_, voters)| voters.contains(&node))
                .map(|(&group, _)| group)
                .collect();
            let config = Config {
                node,
                directory: root.path().join(format!("node-{node}")),
                layout: (*layout).clone(),
                limits: limits.clone(),
                endpoints: endpoints.clone(),
                listen: listen.clone(),
                tls: TlsFiles {
                    certificate: root.path().join(format!("{node}.crt")),
                    private_key: root.path().join(format!("{node}.key")),
                    roots: root.path().join("roots.pem"),
                },
                trusted_clients: BTreeSet::from([fingerprints[&7]]),
                signing: SigningFiles {
                    catalog: groups
                        .contains(&10)
                        .then(|| root.path().join("catalog.pk8")),
                    sources: groups
                        .iter()
                        .map(|&group| (group, root.path().join(format!("source-{group}.pk8"))))
                        .collect(),
                },
                disk: DiskOptions {
                    max_bytes: 512 << 20,
                    data_bytes: 128 << 20,
                    headroom_bytes: 256 << 20,
                },
                storage: LogOptions {
                    target_file_bytes: 1 << 20,
                    purge_threshold_bytes: 2 << 20,
                },
                runtime: runtime::Options::default(),
                peer: peer::Options::default(),
                snapshot: snapshot::Options {
                    max_image_bytes: 64 << 20,
                    transfer_timeout: Duration::from_secs(30),
                    ..snapshot::Options::default()
                },
                client: client::Options {
                    timeout: Duration::from_secs(2),
                    ..client::Options::default()
                },
                maintenance: maintenance::Options {
                    interval_ms: 25,
                    checkpoint_entries: 4,
                    checkpoint_interval_ms: 100,
                    log_interval_ms: 100,
                    scan_entries: 8,
                },
            };
            config.validate()?;
            fs::write(
                root.path().join(format!("{node}.json")),
                serde_json::to_vec_pretty(&config)?,
            )?;
            configs.insert(node, config);
        }
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut tls = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()?
            .with_root_certificates(roots)
            .with_client_auth_cert(
                vec![certs[&7].clone()],
                PrivatePkcs8KeyDer::from(keys[&7].clone()).into(),
            )?;
        tls.alpn_protocols = vec![client::ALPN.to_vec()];
        let endpoints = endpoints
            .iter()
            .map(|(&id, e)| {
                (
                    id,
                    client::Endpoint {
                        data_address: e.data.clone(),
                        control_address: e.control.clone(),
                        server_name: e.server_name.clone(),
                        certificate: e.certificate,
                    },
                )
            })
            .collect();
        let client = Arc::new(Client::new(
            &layout,
            &limits,
            endpoints,
            Arc::new(tls),
            client::Options {
                timeout: Duration::from_secs(2),
                ..client::Options::default()
            },
        )?);
        drop(leases);
        Ok(Self {
            root,
            configs,
            client,
            layout,
        })
    }
    pub fn spawn(&self, node: u64, incarnation: u64) -> Result<ChildNode> {
        let log = self
            .root
            .path()
            .join(format!("node-{node}-{incarnation}.log"));
        let child = Command::new(env!("CARGO_BIN_EXE_context-wal-node"))
            .args([
                "--config",
                self.root
                    .path()
                    .join(format!("{node}.json"))
                    .to_str()
                    .unwrap(),
                "--io-threads",
                "2",
            ])
            .stdin(Stdio::null())
            .stdout(fs::File::create(&log)?)
            .stderr(fs::OpenOptions::new().append(true).open(&log)?)
            .spawn()?;
        Ok(ChildNode {
            child: Some(child),
            log,
        })
    }
    pub async fn status(&self, node: u64) -> Result<serde_json::Value> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        tokio::time::timeout(Duration::from_secs(3), async {
            let mut tcp = tokio::net::TcpStream::connect(self.configs[&node].listen.health).await?;
            tcp.write_all(b"GET /status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                .await?;
            let mut bytes = Vec::new();
            tcp.take(64 << 10).read_to_end(&mut bytes).await?;
            let start = bytes
                .windows(4)
                .position(|p| p == b"\r\n\r\n")
                .context("health response header")?
                + 4;
            Ok(serde_json::from_slice(&bytes[start..])?)
        })
        .await?
    }
}
pub struct ChildNode {
    child: Option<Child>,
    log: PathBuf,
}
impl ChildNode {
    pub fn assert_live(&mut self) -> Result<()> {
        if let Some(status) = self.child.as_mut().context("child missing")?.try_wait()? {
            anyhow::bail!(
                "node exited {status}: {}",
                fs::read_to_string(&self.log).unwrap_or_default()
            );
        }
        Ok(())
    }
    pub fn kill(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            child.kill()?;
            child.wait()?;
        }
        Ok(())
    }
    pub async fn terminate(&mut self) -> Result<()> {
        let Some(child) = self.child.as_mut() else {
            return Ok(());
        };
        #[cfg(unix)]
        unsafe {
            ensure!(
                libc::kill(child.id() as i32, libc::SIGTERM) == 0,
                "signal child"
            );
        }
        #[cfg(not(unix))]
        child.kill()?;
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if let Some(status) = self.child.as_mut().unwrap().try_wait()? {
                    ensure!(
                        status.success(),
                        "node shutdown failed {status}: {}",
                        fs::read_to_string(&self.log).unwrap_or_default()
                    );
                    self.child.take();
                    return Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await??;
        Ok(())
    }
}
impl Drop for ChildNode {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
