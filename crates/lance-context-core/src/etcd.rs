//! Shared etcd connection settings and client construction.
//!
//! The master has always needed etcd (task queue, locks); with the etcd store
//! registry the workers need it too. Both parse the same flags, so the flags
//! and the connect logic live here once.

use std::time::Duration;

use etcd_client::{Certificate, Client, ConnectOptions, Identity, TlsOptions};
use lance::{Error as LanceError, Result as LanceResult};

/// etcd connection settings, one field per `ETCD_*` flag.
#[derive(Debug, Clone, Default, clap::Args)]
pub struct EtcdConfig {
    /// Comma-separated etcd v3 endpoints.
    #[arg(long, env = "ETCD_ENDPOINTS", value_delimiter = ',')]
    pub etcd_endpoints: Vec<String>,

    /// Namespace for every lance-context key in etcd.
    #[arg(long, env = "ETCD_PREFIX", default_value = "/lance-context/master")]
    pub etcd_prefix: String,

    /// Optional etcd username. `ETCD_PASSWORD` must also be set.
    #[arg(long, env = "ETCD_USERNAME")]
    pub etcd_username: Option<String>,

    /// Optional etcd password. `ETCD_USERNAME` must also be set.
    #[arg(long, env = "ETCD_PASSWORD")]
    pub etcd_password: Option<String>,

    /// Optional PEM CA certificate path for etcd TLS.
    #[arg(long, env = "ETCD_CA_CERT")]
    pub etcd_ca_cert: Option<String>,

    /// Optional PEM client certificate path for etcd mutual TLS.
    #[arg(long, env = "ETCD_CLIENT_CERT")]
    pub etcd_client_cert: Option<String>,

    /// Optional PEM client private-key path for etcd mutual TLS.
    #[arg(long, env = "ETCD_CLIENT_KEY")]
    pub etcd_client_key: Option<String>,
}

impl EtcdConfig {
    /// Whether any endpoint is configured.
    pub fn is_configured(&self) -> bool {
        !self.etcd_endpoints.is_empty()
    }

    /// `etcd_prefix` without a trailing slash, ready for `format!("{p}/…")`.
    pub fn prefix(&self) -> &str {
        self.etcd_prefix.trim_end_matches('/')
    }

    /// Connect with the configured auth and TLS.
    pub async fn connect(&self) -> LanceResult<Client> {
        if self.etcd_endpoints.is_empty() {
            return Err(LanceError::io("ETCD_ENDPOINTS is required"));
        }
        let mut options = ConnectOptions::new()
            .with_connect_timeout(Duration::from_secs(5))
            .with_timeout(Duration::from_secs(10))
            .with_keep_alive(Duration::from_secs(10), Duration::from_secs(3))
            .with_require_leader(true);
        match (&self.etcd_username, &self.etcd_password) {
            (Some(username), Some(password)) => {
                options = options.with_user(username, password);
            }
            (None, None) => {}
            _ => {
                return Err(LanceError::io(
                    "ETCD_USERNAME and ETCD_PASSWORD must be configured together",
                ))
            }
        }
        if let Some(path) = &self.etcd_ca_cert {
            let pem = std::fs::read(path).map_err(|err| {
                LanceError::io(format!("failed to read ETCD_CA_CERT '{path}': {err}"))
            })?;
            let mut tls = TlsOptions::new().ca_certificate(Certificate::from_pem(pem));
            match (&self.etcd_client_cert, &self.etcd_client_key) {
                (Some(cert), Some(key)) => {
                    let cert_pem = std::fs::read(cert).map_err(|err| {
                        LanceError::io(format!("failed to read ETCD_CLIENT_CERT '{cert}': {err}"))
                    })?;
                    let key_pem = std::fs::read(key).map_err(|err| {
                        LanceError::io(format!("failed to read ETCD_CLIENT_KEY '{key}': {err}"))
                    })?;
                    tls = tls.identity(Identity::from_pem(cert_pem, key_pem));
                }
                (None, None) => {}
                _ => {
                    return Err(LanceError::io(
                        "ETCD_CLIENT_CERT and ETCD_CLIENT_KEY must be configured together",
                    ))
                }
            }
            options = options.with_tls(tls);
        } else if self.etcd_client_cert.is_some() || self.etcd_client_key.is_some() {
            return Err(LanceError::io(
                "ETCD_CA_CERT is required when configuring an etcd client certificate",
            ));
        }
        Client::connect(self.etcd_endpoints.clone(), Some(options))
            .await
            .map_err(|err| LanceError::io(format!("connect to etcd: {err}")))
    }
}

/// Map an etcd client error into a Lance IO error with context.
pub fn etcd_error(context: &'static str) -> impl Fn(etcd_client::Error) -> LanceError {
    move |err| LanceError::io(format!("{context}: {err}"))
}

/// Which backend a store registry reads from, and which (if any) it mirrors
/// writes to. See `docs/design-registry-etcd.md` for the migration this drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum RegistryBackend {
    /// The Lance table under `data_dir` (`_registry.<kind>.lance`).
    #[default]
    Lance,
    /// etcd, under `<ETCD_PREFIX>/registry/<kind>/`.
    Etcd,
}

/// Registry backend selection flags, shared by the server and master.
#[derive(Debug, Clone, Default, clap::Args)]
pub struct RegistryConfig {
    /// Backend the registries read from and write to.
    #[arg(long, env = "REGISTRY_BACKEND", value_enum, default_value_t = RegistryBackend::Lance)]
    pub registry_backend: RegistryBackend,

    /// Versioned etcd mirror for a Lance primary. Reverse mirroring is unsupported.
    #[arg(long, env = "REGISTRY_MIRROR", value_enum)]
    pub registry_mirror: Option<RegistryBackend>,
}

/// Open the registry for one store kind per `RegistryConfig`.
///
/// `lance_uri` is the Lance table's location; `etcd` is a connected client when
/// `EtcdConfig` is configured. Returns the mirrored composite when a mirror is
/// requested. Fails when a requested backend is not available.
pub async fn open_registry(
    kind: &str,
    lance_uri: &str,
    etcd: Option<(&Client, &str)>,
    config: &RegistryConfig,
) -> LanceResult<std::sync::Arc<dyn crate::StoreRegistry>> {
    use crate::{EtcdRegistry, LanceRegistry, MirroredRegistry, RolloutRegistry, StoreRegistry};
    use std::sync::Arc;

    async fn build(
        backend: RegistryBackend,
        kind: &str,
        lance_uri: &str,
        etcd: Option<(&Client, &str)>,
    ) -> LanceResult<Arc<dyn StoreRegistry>> {
        Ok(match backend {
            RegistryBackend::Lance => Arc::new(LanceRegistry::new(
                RolloutRegistry::open_or_create(lance_uri, None).await?,
            )),
            RegistryBackend::Etcd => {
                let (client, prefix) = etcd.ok_or_else(|| {
                    LanceError::io(format!(
                        "REGISTRY_BACKEND/REGISTRY_MIRROR=etcd for the {kind} registry but ETCD_ENDPOINTS is required"
                    ))
                })?;
                EtcdRegistry::new(client.clone(), prefix, kind)
            }
        })
    }

    if config.registry_mirror.is_some()
        && (config.registry_backend != RegistryBackend::Lance
            || config.registry_mirror != Some(RegistryBackend::Etcd))
    {
        return Err(LanceError::io("REGISTRY_MIRROR supports only Lance primary -> etcd mirror; reverse mirroring/rolling rollback is unsupported"));
    }
    let primary = build(config.registry_backend, kind, lance_uri, etcd).await?;
    if let Some(etcd) = primary.as_any().downcast_ref::<EtcdRegistry>() {
        etcd.activate_after_validation(lance_uri).await?;
    }
    let Some(mirror) = config.registry_mirror else {
        return Ok(primary);
    };
    if mirror == config.registry_backend {
        return Err(LanceError::io(
            "REGISTRY_MIRROR must differ from REGISTRY_BACKEND",
        ));
    }
    let mirror = build(mirror, kind, lance_uri, etcd).await?;
    Ok(Arc::new(MirroredRegistry {
        primary,
        mirror,
        label: kind.to_string(),
    }))
}
