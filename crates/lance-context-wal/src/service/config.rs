use super::*;
use crate::{
    placement::Layout,
    raft_log::{DiskOptions, LogOptions},
    replication::{source, Limits},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub peer: String,
    pub snapshot: String,
    pub data: String,
    pub control: String,
    pub server_name: String,
    pub certificate: [u8; 32],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Listen {
    pub peer: SocketAddr,
    pub snapshot: SocketAddr,
    pub data: SocketAddr,
    pub control: SocketAddr,
    /// Local diagnostic endpoint, not an authorization or quorum oracle.
    pub health: SocketAddr,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TlsFiles {
    pub certificate: PathBuf,
    pub private_key: PathBuf,
    pub roots: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SigningFiles {
    pub catalog: Option<PathBuf>,
    pub sources: BTreeMap<u64, PathBuf>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub node: u64,
    pub directory: PathBuf,
    pub layout: Layout,
    pub limits: Limits,
    pub endpoints: BTreeMap<u64, Endpoint>,
    pub listen: Listen,
    pub tls: TlsFiles,
    pub trusted_clients: BTreeSet<[u8; 32]>,
    pub signing: SigningFiles,
    pub disk: DiskOptions,
    #[serde(default)]
    pub storage: LogOptions,
    #[serde(default)]
    pub runtime: runtime::Options,
    #[serde(default)]
    pub peer: peer::Options,
    #[serde(default)]
    pub snapshot: snapshot::Options,
    #[serde(default)]
    pub client: client::Options,
    #[serde(default)]
    pub maintenance: maintenance::Options,
}
impl Config {
    pub fn read(path: &Path) -> Result<Self> {
        let mut config: Self = serde_json::from_slice(&read_bounded(path, 1 << 20)?)?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        fn resolve(base: &Path, path: &mut PathBuf) {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        }
        resolve(base, &mut config.directory);
        for path in [
            &mut config.tls.certificate,
            &mut config.tls.private_key,
            &mut config.tls.roots,
        ] {
            resolve(base, path);
        }
        if let Some(path) = &mut config.signing.catalog {
            resolve(base, path);
        }
        for path in config.signing.sources.values_mut() {
            resolve(base, path);
        }
        config.validate()?;
        Ok(config)
    }
    pub fn groups(&self) -> Vec<u64> {
        self.layout
            .groups
            .iter()
            .filter(|(_, voters)| voters.contains(&self.node))
            .map(|(&group, _)| group)
            .collect()
    }
    pub fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        self.limits.identity()?;
        self.disk.validate()?;
        self.runtime.validate()?;
        self.peer.validate()?;
        self.client
            .validate(self.limits.max_record_bytes as usize)?;
        self.maintenance.validate()?;
        ensure!(
            !self.directory.as_os_str().is_empty() && self.layout.nodes.contains_key(&self.node),
            "invalid node or directory"
        );
        ensure!(
            self.endpoints.keys().eq(self.layout.nodes.keys()),
            "endpoints must cover exactly the layout nodes"
        );
        let mut certificates = BTreeSet::new();
        let mut addresses = BTreeSet::new();
        for (&node, endpoint) in &self.endpoints {
            ensure!(
                endpoint.peer == self.layout.nodes[&node],
                "advertised peer address differs from durable layout"
            );
            for value in [
                &endpoint.peer,
                &endpoint.snapshot,
                &endpoint.data,
                &endpoint.control,
            ] {
                ensure!(
                    !value.is_empty() && value.len() <= 1024 && addresses.insert(value),
                    "invalid or duplicate advertised address"
                );
            }
            tokio_rustls::rustls::pki_types::ServerName::try_from(endpoint.server_name.clone())?;
            ensure!(
                endpoint.certificate != [0; 32] && certificates.insert(endpoint.certificate),
                "invalid or duplicate node certificate"
            );
        }
        ensure!(
            !self.trusted_clients.is_empty()
                && self.trusted_clients.len() <= 256
                && !self.trusted_clients.contains(&[0; 32]),
            "invalid application certificate pins"
        );
        let mut listens = BTreeSet::new();
        for address in [
            self.listen.peer,
            self.listen.snapshot,
            self.listen.data,
            self.listen.control,
            self.listen.health,
        ] {
            ensure!(
                address.port() > 0 && listens.insert(address),
                "listener addresses must be distinct fixed ports"
            );
        }
        ensure!(
            self.storage.target_file_bytes > 0
                && self.storage.purge_threshold_bytes >= self.storage.target_file_bytes
                && usize::try_from(self.storage.purge_threshold_bytes).is_ok(),
            "invalid physical storage thresholds"
        );
        let largest = (self.limits.max_record_bytes as usize).max(1 << 20) + 4096;
        ensure!(
            self.peer.peer_bytes >= largest
                && self.peer.peer_bytes + largest <= self.runtime.peer_bytes,
            "peer credits must fit a record and reserve capacity for another voter"
        );
        ensure!(
            (1..=16).contains(&self.snapshot.outbound_transfers)
                && self.snapshot.outbound_transfers < self.runtime.peer_requests
                && (1..=64).contains(&self.snapshot.connections)
                && (4096..=1 << 40).contains(&self.snapshot.max_image_bytes)
                && (Duration::from_millis(100)..=Duration::from_secs(3600))
                    .contains(&self.snapshot.transfer_timeout),
            "invalid snapshot options"
        );
        ensure!(
            self.limits.producer_slots > 0
                && self.limits.catalog_key.is_some()
                && self.limits.source_keys.keys().eq(self.layout.groups.keys()),
            "service requires bounded producers and complete publication signing policy"
        );
        let catalog = *self
            .layout
            .groups
            .keys()
            .next()
            .context("no catalog group")?;
        let hosted = self.groups();
        ensure!(
            self.signing.catalog.is_some() == hosted.contains(&catalog),
            "catalog secret must be provisioned exactly on catalog replicas"
        );
        ensure!(
            self.signing.sources.keys().copied().collect::<Vec<_>>() == hosted,
            "source secrets must cover exactly this node's hosted groups"
        );
        Ok(())
    }
    pub(super) fn peers(&self, images: bool) -> BTreeMap<u64, peer::Peer> {
        let ids: BTreeSet<_> = self
            .groups()
            .iter()
            .flat_map(|group| self.layout.groups[group].iter().copied())
            .filter(|id| *id != self.node)
            .collect();
        ids.into_iter()
            .map(|id| {
                let endpoint = &self.endpoints[&id];
                (
                    id,
                    peer::Peer {
                        address: if images {
                            endpoint.snapshot.clone()
                        } else {
                            endpoint.peer.clone()
                        },
                        server_name: endpoint.server_name.clone(),
                        certificate: endpoint.certificate,
                    },
                )
            })
            .collect()
    }
    pub(super) fn open(&self) -> Result<(crate::replication::Node, tls::Identity)> {
        self.validate()?;
        let identity = tls::Identity::load(&self.tls, &self.endpoints[&self.node].server_name)?;
        ensure!(
            identity.fingerprint == self.endpoints[&self.node].certificate,
            "local TLS identity differs from configured node pin"
        );
        let catalog = self
            .signing
            .catalog
            .as_ref()
            .map(|path| source::CatalogSigner::from_pkcs8(&read_bounded(path, 16 << 10)?))
            .transpose()?;
        let sources = self
            .signing
            .sources
            .iter()
            .map(|(&group, path)| {
                Ok((
                    group,
                    source::SourceSigner::from_pkcs8(&read_bounded(path, 16 << 10)?)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        ensure!(
            catalog
                .as_ref()
                .is_none_or(|key| Some(key.public_key()) == self.limits.catalog_key)
                && sources
                    .iter()
                    .all(|(group, key)| self.limits.source_keys.get(group)
                        == Some(&key.public_key())),
            "private signing keys differ from immutable public policy"
        );
        let mut node = crate::replication::Node::open_with_storage_options(
            &self.directory,
            Arc::new(self.layout.clone()),
            self.node,
            self.limits.clone(),
            self.storage,
        )?
        .with_disk_budget(self.disk)?;
        if let Some(signer) = catalog {
            node.set_catalog_signer(Arc::new(signer))?;
        }
        for (group, signer) in sources {
            node.set_source_signer(group, Arc::new(signer))?;
        }
        Ok((node, identity))
    }
}
pub(super) fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= limit,
        "configuration file exceeds size limit"
    );
    Ok(bytes)
}
