use super::*;
use lance_context_wal::replication::{runtime::client, Limits};
use std::{
    io::Read,
    net::SocketAddr,
    path::{Path, PathBuf},
};
use tokio_rustls::rustls::{
    self,
    pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer},
    RootCertStore,
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TlsFiles {
    pub certificate: PathBuf,
    pub private_key: PathBuf,
    pub roots: PathBuf,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub data_address: String,
    pub control_address: String,
    pub server_name: String,
    pub certificate: [u8; 32],
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Rpc {
    pub data_bytes: usize,
    pub control_bytes: usize,
    pub connections_per_node: usize,
    pub timeout: Duration,
}
impl Default for Rpc {
    fn default() -> Self {
        let v = client::Options::default();
        Self {
            data_bytes: v.data_bytes,
            control_bytes: v.control_bytes,
            connections_per_node: v.connections,
            timeout: v.timeout,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Table {
    pub binding: TableBinding,
    pub uri: String,
    #[serde(default)]
    pub storage_options: HashMap<String, String>,
    /// Persisted before bootstrap; identical on every process/restart.
    pub bootstrap: bootstrap::Identity,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Etcd {
    pub endpoints: Vec<String>,
    pub prefix: String,
    pub ttl_seconds: i64,
    pub timeout: Duration,
    pub tls: Option<EtcdTls>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EtcdTls {
    pub identity: TlsFiles,
    pub domain: Option<String>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Http {
    pub listen: SocketAddr,
    /// Workload token authorizes this complete configured table set. The edge
    /// proxy provides TLS and any narrower end-user authorization.
    pub bearer_token_file: PathBuf,
    pub receipt_key_file: PathBuf,
    #[serde(default)]
    pub options: http::Options,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Annotations {
    /// Stable deployment-authorized service address; never supplied by a client.
    pub owner_url: String,
    /// Explicit physical bucket for this table. Only fresh mixed bindings opt in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<crate::annotations::Shard>,
    /// Complete authorized owner map for a shared ingress. Owners may omit it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub routes: Vec<AnnotationRoute>,
    /// Additional independent WAL clusters used by shared-ingress routes.
    /// Each has its own existing publication catalog and connection pool.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub wal_clusters: BTreeMap<String, WalConnection>,
    pub apply_memory_bytes: usize,
    #[serde(default)]
    pub staging: lance_context_core::annotation_staging::Limits,
    /// Fixed, quiesced source snapshot for the explicit annotation-bootstrap role.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<bootstrap::annotations::Snapshot>,
    /// Create a separate narrow table from a quiesced source. Mutually exclusive
    /// with the same-table `source` handoff; only annotation-bootstrap uses it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<bootstrap::annotations::projection::Source>,
    /// Authorized immutable-body destination. Explicit annotation-bootstrap
    /// creates a fresh body/narrow pair; owner/ingress only open its mixed
    /// binding. Existing generations are never migrated by role startup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Body>,
    /// Opt into physical collection for a table served exclusively by its
    /// annotation owner. Raw Dataset readers, other writers, tags, branches and
    /// URI reuse are outside this contract. Forwarders never run collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclusive_reclamation: Option<crate::annotations::runtime::reclamation::Options>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationRoute {
    pub binding: TableBinding,
    pub shard: crate::annotations::Shard,
    pub owner_url: String,
    /// Omitted routes use the application's primary WAL connection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wal_cluster: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WalConnection {
    pub layout: Layout,
    pub limits: Limits,
    pub endpoints: BTreeMap<u64, Endpoint>,
    pub tls: TlsFiles,
    #[serde(default)]
    pub rpc: Rpc,
}

impl Annotations {
    fn validate_shards(&self, layout: &Layout, local: &TableBinding) -> Result<()> {
        ensure!(
            self.wal_clusters.len() <= self.routes.len(),
            "annotation WAL clusters require owner routes"
        );
        let mut clusters = HashSet::from([layout.cluster]);
        for (name, wal) in &self.wal_clusters {
            wal.layout.validate()?;
            ensure!(
                !name.is_empty()
                    && name.len() <= 128
                    && clusters.insert(wal.layout.cluster)
                    && wal.layout.buckets.len() == layout.buckets.len(),
                "annotation WAL clusters require distinct identities and the same bucket count"
            );
        }
        let Some(shard) = self.shard else {
            ensure!(
                layout.buckets.len() == 1 && self.routes.is_empty(),
                "legacy annotation configuration requires one shard"
            );
            return Ok(());
        };
        layout.group_for_bucket(shard.bucket)?;
        ensure!(
            self.body.is_some(),
            "explicit annotation shards require mixed body storage"
        );
        if self.routes.is_empty() {
            return Ok(());
        }
        ensure!(
            self.routes.len() <= 4096 && self.routes.len() == layout.buckets.len(),
            "incomplete annotation owner routes"
        );
        let contract = local.contract()?;
        let (mut buckets, mut tables, mut destinations) =
            (HashSet::new(), HashSet::new(), HashSet::new());
        let mut includes_local = false;
        let mut used_clusters = HashSet::new();
        for route in &self.routes {
            let route_layout = if let Some(name) = &route.wal_cluster {
                used_clusters.insert(name);
                &self
                    .wal_clusters
                    .get(name)
                    .context("unknown annotation WAL cluster")?
                    .layout
            } else {
                layout
            };
            route_layout.group_for_bucket(route.shard.bucket)?;
            ensure!(
                route.binding.contract()? == contract
                    && !route.binding.table.is_nil()
                    && !route.binding.incarnation.is_nil()
                    && !route.owner_url.is_empty()
                    && route.owner_url.len() <= 8192
                    && buckets.insert(route.shard.bucket)
                    && tables.insert(route.binding.table)
                    && destinations.insert((
                        &route.binding.destination.namespace,
                        &route.binding.destination.dataset
                    )),
                "inconsistent or duplicate annotation owner route"
            );
            if route.shard == shard {
                ensure!(
                    route.wal_cluster.is_none()
                        && route.binding.table == local.table
                        && route.binding.incarnation == local.incarnation
                        && route.binding.destination == local.destination
                        && route.owner_url == self.owner_url,
                    "local annotation route differs from its physical binding"
                );
                includes_local = true;
            }
        }
        ensure!(includes_local, "local annotation route missing");
        ensure!(
            used_clusters.len() == self.wal_clusters.len(),
            "unused annotation WAL cluster"
        );
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    pub uri: String,
    #[serde(default)]
    pub storage_options: HashMap<String, String>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Preparation {
    pub listen: Option<SocketAddr>,
    pub bearer_token_file: PathBuf,
    /// Empty on a preparation-only process; worker processes require 1–8 peers.
    pub endpoints: Vec<String>,
    pub timeout: Duration,
    pub active: usize,
    pub retained_results: usize,
    /// Opt-in bucket locality within each publication; hot buckets remain whole.
    #[serde(skip_serializing_if = "crate::preparation::Partitioning::is_record_stripe")]
    pub partitioning: crate::preparation::Partitioning,
}
impl Default for Preparation {
    fn default() -> Self {
        Self {
            listen: None,
            bearer_token_file: PathBuf::new(),
            endpoints: vec![],
            timeout: Duration::from_secs(120),
            active: 4,
            retained_results: 32,
            partitioning: Default::default(),
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub layout: Layout,
    pub limits: Limits,
    pub endpoints: BTreeMap<u64, Endpoint>,
    pub tls: TlsFiles,
    #[serde(default)]
    pub rpc: Rpc,
    /// Deployment-controlled authorized registry, never supplied by an upload.
    pub tables: Vec<Table>,
    pub ingress_memory_bytes: usize,
    #[serde(default = "crate::reader::default_concurrency")]
    pub session_read_concurrency: usize,
    #[serde(default = "crate::reader::default_scan_batch_rows")]
    pub session_scan_batch_rows: usize,
    #[serde(default = "crate::reader::default_scan_fragment_readahead")]
    pub session_scan_fragment_readahead: usize,
    pub materializer_memory_bytes: usize,
    pub http: Option<Http>,
    pub etcd: Option<Etcd>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Annotations>,
    #[serde(default)]
    pub preparation: Option<Preparation>,
    #[serde(default)]
    pub worker: worker::Options,
}
impl Config {
    pub fn read(path: &Path) -> Result<Self> {
        let mut v: Self = serde_json::from_slice(&read(path, 4 << 20)?)?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        fn resolve(base: &Path, path: &mut PathBuf) {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        }
        fn tls(base: &Path, files: &mut TlsFiles) {
            for p in [
                &mut files.certificate,
                &mut files.private_key,
                &mut files.roots,
            ] {
                resolve(base, p);
            }
        }
        tls(base, &mut v.tls);
        if let Some(a) = &mut v.annotations {
            for wal in a.wal_clusters.values_mut() {
                tls(base, &mut wal.tls);
            }
        }
        if let Some(h) = &mut v.http {
            resolve(base, &mut h.bearer_token_file);
            resolve(base, &mut h.receipt_key_file);
        }
        if let Some(p) = &mut v.preparation {
            resolve(base, &mut p.bearer_token_file);
        }
        if let Some(e) = v.etcd.as_mut().and_then(|e| e.tls.as_mut()) {
            tls(base, &mut e.identity);
        }
        v.validate()?;
        Ok(v)
    }
    pub fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        crate::reader::validate_concurrency(self.session_read_concurrency)?;
        crate::reader::validate_scan_batch_rows(self.session_scan_batch_rows)?;
        crate::reader::validate_scan_fragment_readahead(self.session_scan_fragment_readahead)?;
        // Client::new validates the complete WAL policy before any RPC.
        ensure!(
            (1..=4096).contains(&self.tables.len()),
            "invalid application table count"
        );
        ensure!(
            self.ingress_memory_bytes > 0 && self.materializer_memory_bytes > 0,
            "application memory budgets must be nonzero"
        );
        if let Some(a) = &self.annotations {
            ensure!(
                self.tables.len() == 1,
                "annotation application requires exactly one local physical table"
            );
            a.validate_shards(&self.layout, &self.tables[0].binding)?;
            ensure!(
                a.apply_memory_bytes > 0
                    && !a.owner_url.is_empty()
                    && a.owner_url.len() <= 8192
                    && self.preparation.is_none()
                    && self.worker.publication.session_index.is_none()
                    && !self.worker.publication.pipeline_preparation,
                "invalid annotation application options"
            );
            if let Some(reclamation) = &a.exclusive_reclamation {
                reclamation.validate(self.materializer_memory_bytes)?;
            }
            if let Some(projection) = &a.projection {
                ensure!(a.source.is_none(), "choose source or projection, not both");
                projection.validate(self.materializer_memory_bytes)?;
            }
            if let Some(body) = &a.body {
                ensure!(
                    a.source.is_none()
                        && a.projection.is_none()
                        && !body.uri.is_empty()
                        && body.uri.len() <= 8192
                        && body.uri != self.tables[0].uri
                        && body.storage_options.len() <= 128,
                    "mixed annotations require a separate authorized body destination"
                );
            }
        }
        let (mut names, mut ids, mut destinations) =
            (HashSet::new(), HashSet::new(), HashSet::new());
        for t in &self.tables {
            t.binding.contract()?;
            ensure!(
                names.insert(&t.binding.qualified_table)
                    && ids.insert(t.binding.table)
                    && destinations.insert((
                        &t.binding.destination.namespace,
                        &t.binding.destination.dataset
                    )),
                "duplicate authorized table generation or destination"
            );
            ensure!(
                !t.uri.is_empty() && t.uri.len() <= 8192 && t.storage_options.len() <= 128,
                "invalid table storage configuration"
            );
        }
        Ok(())
    }
    pub fn client(&self) -> Result<Arc<Client>> {
        self.validate()?;
        wal_client(
            &self.layout,
            &self.limits,
            &self.endpoints,
            &self.tls,
            &self.rpc,
        )
    }
}
impl WalConnection {
    pub(super) fn client(&self) -> Result<Arc<Client>> {
        wal_client(
            &self.layout,
            &self.limits,
            &self.endpoints,
            &self.tls,
            &self.rpc,
        )
    }
}
fn wal_client(
    layout: &Layout,
    limits: &Limits,
    endpoints: &BTreeMap<u64, Endpoint>,
    files: &TlsFiles,
    rpc: &Rpc,
) -> Result<Arc<Client>> {
    let roots = CertificateDer::pem_slice_iter(&read(&files.roots, 1 << 20)?)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        !roots.is_empty() && roots.len() <= 128,
        "invalid TLS root count"
    );
    let mut store = RootCertStore::empty();
    for certificate in roots {
        store.add(certificate)?;
    }
    let certificates = CertificateDer::pem_slice_iter(&read(&files.certificate, 1 << 20)?)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        !certificates.is_empty() && certificates.len() <= 16,
        "invalid TLS chain"
    );
    let key = PrivateKeyDer::from_pem_slice(&read(&files.private_key, 64 << 10)?)?;
    let mut tls = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_root_certificates(store)
    .with_client_auth_cert(certificates, key)?;
    tls.alpn_protocols = vec![client::ALPN.to_vec()];
    let endpoints = endpoints
        .iter()
        .map(|(&id, e)| {
            (
                id,
                client::Endpoint {
                    data_address: e.data_address.clone(),
                    control_address: e.control_address.clone(),
                    server_name: e.server_name.clone(),
                    certificate: e.certificate,
                },
            )
        })
        .collect();
    Ok(Arc::new(Client::new(
        layout,
        limits,
        endpoints,
        Arc::new(tls),
        client::Options {
            data_bytes: rpc.data_bytes,
            control_bytes: rpc.control_bytes,
            connections: rpc.connections_per_node,
            timeout: rpc.timeout,
            ..Default::default()
        },
    )?))
}

pub(super) fn read(path: &Path, max: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take((max + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= max, "configuration file exceeds size limit");
    Ok(bytes)
}
impl Etcd {
    pub async fn claims(&self) -> Result<worker::Claims> {
        ensure!(
            !self.endpoints.is_empty() && self.endpoints.len() <= 16,
            "invalid etcd endpoints"
        );
        let mut options = etcd_client::ConnectOptions::new()
            .with_connect_timeout(self.timeout)
            .with_timeout(self.timeout);
        if let Some(tls) = &self.tls {
            ensure!(
                self.endpoints.iter().all(|e| e.starts_with("https://")),
                "TLS etcd requires HTTPS endpoints"
            );
            let f = &tls.identity;
            let mut config = etcd_client::TlsOptions::new()
                .ca_certificate(etcd_client::Certificate::from_pem(read(&f.roots, 1 << 20)?))
                .identity(etcd_client::Identity::from_pem(
                    read(&f.certificate, 1 << 20)?,
                    read(&f.private_key, 64 << 10)?,
                ));
            if let Some(domain) = &tls.domain {
                config = config.domain_name(domain);
            }
            options = options.with_tls(config);
        }
        let client = tokio::time::timeout(
            self.timeout,
            etcd_client::Client::connect(self.endpoints.clone(), Some(options)),
        )
        .await??;
        worker::Claims::new(client, self.prefix.clone(), self.ttl_seconds, self.timeout)
    }
}
