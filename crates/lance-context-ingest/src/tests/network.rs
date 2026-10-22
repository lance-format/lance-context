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
}
fn certificates() -> Result<BTreeMap<u64, Tls>> {
    let mut identities = Vec::new();
    let mut roots = RootCertStore::empty();
    for id in 1..=4 {
        let name = format!("wal-{id}.example.invalid");
        let certified = rcgen::generate_simple_self_signed(vec![name.clone()])?;
        let der: CertificateDer<'static> = certified.cert.der().clone();
        roots.add(der.clone())?;
        identities.push((id, name, der, certified.signing_key.serialize_der()));
    }
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
            },
        );
    }
    Ok(configs)
}

pub(super) struct Cluster {
    pub client: Arc<Client>,
    nodes: Vec<(Runtime, peer::Transport, client::Server)>,
}
impl Cluster {
    pub async fn open(root: &Path, layout: Arc<Layout>) -> Result<Self> {
        let tls = certificates()?;
        let limits = Limits {
            producer_slots: 16,
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
            let node = Node::open(
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
        Ok(Self { client, nodes })
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
