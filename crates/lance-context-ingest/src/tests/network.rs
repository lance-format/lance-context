use super::*;
use lance_context_wal::{
    raft_log::DiskOptions,
    replication::{
        runtime::{self, client, peer, Runtime},
        Limits, Node,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use tokio::net::TcpListener;
use tokio_rustls::rustls::{
    self,
    pki_types::{CertificateDer, PrivatePkcs8KeyDer},
    RootCertStore,
};

struct Tls {
    client: rustls::ClientConfig,
    server: rustls::ServerConfig,
    certificate: [u8; 32],
    name: String,
    #[cfg(feature = "service")]
    files: crate::application::config::TlsFiles,
}
fn certificates(_root: &Path) -> Result<BTreeMap<u64, Tls>> {
    let mut identities = Vec::new();
    let mut roots = RootCertStore::empty();
    #[cfg(feature = "service")]
    let mut root_pem = String::new();
    #[cfg(feature = "service")]
    let mut files = BTreeMap::new();
    for id in 1..=4 {
        let name = format!("wal-{id}.example.invalid");
        let certified = rcgen::generate_simple_self_signed(vec![name.clone()])?;
        let der: CertificateDer<'static> = certified.cert.der().clone();
        roots.add(der.clone())?;
        #[cfg(feature = "service")]
        {
            let identity = crate::application::config::TlsFiles {
                certificate: _root.join(format!("tls-{id}.pem")),
                private_key: _root.join(format!("tls-{id}.key")),
                roots: _root.join("roots.pem"),
            };
            std::fs::write(&identity.certificate, certified.cert.pem())?;
            std::fs::write(&identity.private_key, certified.signing_key.serialize_pem())?;
            root_pem.push_str(&certified.cert.pem());
            files.insert(id, identity);
        }
        identities.push((id, name, der, certified.signing_key.serialize_der()));
    }
    #[cfg(feature = "service")]
    std::fs::write(_root.join("roots.pem"), root_pem)?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let roots = Arc::new(roots);
    let mut configs = BTreeMap::new();
    for (id, name, cert, key) in identities {
        let certificate = ring::digest::digest(&ring::digest::SHA256, cert.as_ref())
            .as_ref()
            .try_into()?;
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            roots.clone(),
            provider.clone(),
        )
        .build()?;
        let server = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()?
            .with_client_cert_verifier(verifier)
            .with_single_cert(
                vec![cert.clone()],
                PrivatePkcs8KeyDer::from(key.clone()).into(),
            )?;
        let client = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()?
            .with_root_certificates(roots.clone())
            .with_client_auth_cert(vec![cert], PrivatePkcs8KeyDer::from(key).into())?;
        configs.insert(
            id,
            Tls {
                client,
                server,
                certificate,
                name,
                #[cfg(feature = "service")]
                files: files.remove(&id).unwrap(),
            },
        );
    }
    Ok(configs)
}

pub(super) struct Cluster {
    pub client: Arc<Client>,
    nodes: Vec<(Runtime, peer::Transport, client::Server)>,
    #[cfg(feature = "service")]
    pub application: crate::application::Config,
}
type Signers = (
    Arc<lance_context_wal::replication::source::CatalogSigner>,
    BTreeMap<u64, Arc<lance_context_wal::replication::source::SourceSigner>>,
);
impl Cluster {
    pub async fn open(root: &Path, layout: Arc<Layout>) -> Result<Self> {
        Self::open_inner(root, layout, None).await
    }
    #[cfg(feature = "materializer")]
    pub async fn open_signed(root: &Path, layout: Arc<Layout>) -> Result<Self> {
        use lance_context_wal::replication::source::{CatalogSigner, SourceSigner};
        fn key(path: &Path) -> Result<Vec<u8>> {
            if !path.exists() {
                let key = ring::signature::Ed25519KeyPair::generate_pkcs8(
                    &ring::rand::SystemRandom::new(),
                )
                .map_err(|_| anyhow::anyhow!("test signing key generation"))?;
                std::fs::write(path, key.as_ref())?;
            }
            Ok(std::fs::read(path)?)
        }
        let catalog = Arc::new(CatalogSigner::from_pkcs8(&key(&root.join("catalog.pk8"))?)?);
        let sources = layout
            .groups
            .keys()
            .map(|&group| {
                Ok((
                    group,
                    Arc::new(SourceSigner::from_pkcs8(&key(
                        &root.join(format!("source-{group}.pk8"))
                    )?)?),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Self::open_inner(root, layout, Some((catalog, sources))).await
    }
    async fn open_inner(
        root: &Path,
        layout: Arc<Layout>,
        signers: Option<Signers>,
    ) -> Result<Self> {
        let tls = certificates(root)?;
        let limits = Limits {
            producer_slots: 16,
            catalog_key: signers.as_ref().map(|s| s.0.public_key()),
            source_keys: signers
                .as_ref()
                .map(|s| {
                    s.1.iter()
                        .map(|(&group, key)| (group, key.public_key()))
                        .collect()
                })
                .unwrap_or_default(),
            ..Limits::default()
        };
        let mut peers = BTreeMap::new();
        let mut listeners = Vec::new();
        let mut endpoints = BTreeMap::new();
        for id in 1..=3 {
            let peer = TcpListener::bind("127.0.0.1:0").await?;
            peers.insert(
                id,
                peer::Peer {
                    address: peer.local_addr()?.to_string(),
                    server_name: tls[&id].name.clone(),
                    certificate: tls[&id].certificate,
                },
            );
            let data = TcpListener::bind("127.0.0.1:0").await?;
            let control = TcpListener::bind("127.0.0.1:0").await?;
            endpoints.insert(
                id,
                client::Endpoint {
                    data_address: data.local_addr()?.to_string(),
                    control_address: control.local_addr()?.to_string(),
                    server_name: tls[&id].name.clone(),
                    certificate: tls[&id].certificate,
                },
            );
            listeners.push((id, peer, data, control));
        }
        let mut nodes = Vec::new();
        for (id, listener, data, control) in listeners {
            let mut node = Node::open(
                &root.join(id.to_string()),
                layout.clone(),
                id,
                limits.clone(),
            )?
            .with_disk_budget(DiskOptions {
                max_bytes: 512 << 20,
                data_bytes: 128 << 20,
                headroom_bytes: 256 << 20,
            })?;
            if let Some((catalog, sources)) = &signers {
                node.set_catalog_signer(catalog.clone())?;
                for (&group, key) in sources {
                    node.set_source_signer(group, key.clone())?;
                }
            }
            let (runtime, outbox) = Runtime::spawn(node, runtime::Options::default())?;
            let mut server = tls[&id].server.clone();
            server.alpn_protocols = vec![peer::ALPN.to_vec()];
            let mut client = tls[&id].client.clone();
            client.alpn_protocols = vec![peer::ALPN.to_vec()];
            let transport = peer::Transport::spawn(
                runtime.handle(),
                outbox,
                listener,
                peers
                    .iter()
                    .filter(|(n, _)| **n != id)
                    .map(|(n, p)| (*n, p.clone()))
                    .collect(),
                Arc::new(client),
                Arc::new(server),
                peer::Options::default(),
            )?;
            let mut server = tls[&id].server.clone();
            server.alpn_protocols = vec![client::ALPN.to_vec()];
            let rpc = client::Server::spawn(
                runtime.handle(),
                data,
                control,
                BTreeSet::from([tls[&4].certificate]),
                Arc::new(server),
                client::Options::default(),
            )?;
            nodes.push((runtime, transport, rpc));
        }
        let mut config = tls[&4].client.clone();
        config.alpn_protocols = vec![client::ALPN.to_vec()];
        #[cfg(feature = "service")]
        let application = crate::application::Config {
            layout: (*layout).clone(),
            limits: limits.clone(),
            endpoints: endpoints
                .iter()
                .map(|(&id, e)| {
                    (
                        id,
                        crate::application::config::Endpoint {
                            data_address: e.data_address.clone(),
                            control_address: e.control_address.clone(),
                            server_name: e.server_name.clone(),
                            certificate: e.certificate,
                        },
                    )
                })
                .collect(),
            tls: tls[&4].files.clone(),
            rpc: Default::default(),
            tables: vec![],
            ingress_memory_bytes: 32 << 20,
            session_read_concurrency: 1,
            session_scan_batch_rows: 1,
            session_scan_fragment_readahead: 1,
            materializer_memory_bytes: 64 << 20,
            http: None,
            etcd: None,
            annotations: None,
            preparation: None,
            worker: crate::worker::Options {
                publication: crate::materializer::Options {
                    max_source_bytes: 1 << 20,
                    max_records: 8,
                    ..Default::default()
                },
                ..Default::default()
            },
        };
        let client = Arc::new(Client::new(
            &layout,
            &limits,
            endpoints,
            Arc::new(config),
            client::Options::default(),
        )?);
        for &group in layout.groups.keys() {
            nodes[0].0.handle().campaign(group)?.await?;
            tokio::time::timeout(Duration::from_secs(15), async {
                loop {
                    let leaders = nodes
                        .iter()
                        .map(|n| n.0.handle().observation(group).unwrap().status.leader)
                        .collect::<BTreeSet<_>>();
                    if leaders.len() == 1 && leaders.first().is_some_and(|v| (1..=3).contains(v)) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await?;
        }
        Ok(Self {
            client,
            nodes,
            #[cfg(feature = "service")]
            application,
        })
    }
    pub async fn shutdown(self) -> Result<()> {
        for (runtime, peer, rpc) in self.nodes {
            rpc.shutdown().await?;
            peer.shutdown().await?;
            runtime.shutdown().await?;
        }
        Ok(())
    }
}
