//! Process composition for the durable ingestion path. Configuration is a fixed,
//! deployment-authorized table registry; dynamic namespace discovery is separate.
use super::*;
use crate::{
    bootstrap,
    http::{self, Gateway},
    materializer::{self, Storage},
    worker::{self, Assignment, Worker},
};
use axum::{
    extract::{Request, State},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use ring::hmac;
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
};
use tokio::net::TcpListener;

pub mod config;
pub use config::Config;

pub struct Registry {
    tables: BTreeMap<String, Assignment>,
}
impl Registry {
    /// All tables share one physical WAL connection pool and one ingress budget.
    /// Startup observes initialized generations; it never takes publisher epochs.
    pub async fn connect(config: &Config) -> Result<Self> {
        let client = config.client()?;
        Self::connect_with_client(config, client).await
    }
    pub(crate) async fn connect_with_client(config: &Config, client: Arc<Client>) -> Result<Self> {
        config.validate()?;
        let layout = Arc::new(config.layout.clone());
        let budget = MergeMemoryBudget::new(config.ingress_memory_bytes);
        let session = Arc::new(lance::session::Session::default());
        let mut tables = BTreeMap::new();
        for t in &config.tables {
            let storage =
                Storage::resolve(t.uri.clone(), t.storage_options.clone(), session.clone()).await?;
            let ingress = Ingress::connect(
                client.clone(),
                layout.clone(),
                t.binding.clone(),
                budget.clone(),
            )
            .await?;
            // Validate both durable catalog scope and the configured blob path.
            materializer::read_dataset(&ingress, &storage, None).await?;
            tables.insert(
                t.binding.qualified_table.clone(),
                Assignment { ingress, storage },
            );
        }
        Ok(Self { tables })
    }
    pub fn gateway(&self, options: &config::Http) -> Result<Gateway> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Keys {
            active: u32,
            keys: BTreeMap<u32, [u8; 32]>,
        }
        let keys: Keys =
            serde_json::from_slice(&config::read(&options.receipt_key_file, 64 << 10)?)?;
        Gateway::new(
            self.tables
                .iter()
                .map(|(name, a)| (name.clone(), a.ingress.clone()))
                .collect(),
            receipt::ReceiptKeys::new(keys.active, keys.keys)?,
            options.options.clone(),
        )
    }
    pub async fn worker(&self, config: &Config) -> Result<Worker> {
        let claims = config
            .etcd
            .as_ref()
            .context("worker requires etcd scheduling configuration")?
            .claims()
            .await?;
        Worker::start(
            claims,
            self.tables.values().cloned().collect(),
            config.worker.clone(),
            MergeMemoryBudget::new(config.materializer_memory_bytes),
        )
    }
    pub async fn read(&self, table: &str, version: Option<u64>) -> Result<lance::Dataset> {
        let assignment = self
            .tables
            .get(table)
            .context("table outside authorized registry")?;
        materializer::read_dataset(&assignment.ingress, &assignment.storage, version).await
    }
}
/// Run explicitly during provisioning, using the SAME persisted operation IDs
/// after an ambiguous result. Normal ingress/worker startup never bootstraps.
pub async fn provision(config: &Config) -> Result<()> {
    let client = config.client()?;
    let layout = Arc::new(config.layout.clone());
    let session = Arc::new(lance::session::Session::default());
    for table in &config.tables {
        let storage = Storage::resolve(
            table.uri.clone(),
            table.storage_options.clone(),
            session.clone(),
        )
        .await?;
        bootstrap::provision(
            client.clone(),
            layout.clone(),
            table.binding.clone(),
            storage,
            table.bootstrap.clone(),
        )
        .await?;
    }
    Ok(())
}

struct Authorization {
    key: hmac::Key,
    tag: hmac::Tag,
}
impl Authorization {
    fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = config::read(path, 4096)?;
        let token = std::str::from_utf8(&bytes)?.trim();
        ensure!(
            (32..=1024).contains(&token.len()) && token.bytes().all(|b| b.is_ascii_graphic()),
            "invalid workload bearer token"
        );
        // A fixed domain is sufficient for constant-time token comparison. The
        // bearer token is the secret; neither it nor its tag is logged/exposed.
        let key = hmac::Key::new(hmac::HMAC_SHA256, b"lance-context-ingest-workload-auth-v1");
        let tag = hmac::sign(&key, token.as_bytes());
        Ok(Self { key, tag })
    }
}
async fn authorize(
    State(auth): State<Arc<Authorization>>,
    request: Request,
    next: Next,
) -> Response {
    let headers = request.headers();
    let token = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if headers
        .get_all(axum::http::header::AUTHORIZATION)
        .iter()
        .count()
        != 1
        || !token.is_some_and(|v| {
            v.len() <= 1024 && hmac::verify(&auth.key, v.as_bytes(), auth.tag.as_ref()).is_ok()
        })
    {
        return axum::http::StatusCode::UNAUTHORIZED.into_response();
    }
    next.run(request).await
}
/// The caller binds first, making the actual listener available for deployment
/// readiness/testing. All admitted gateway handlers are joined before return.
pub async fn serve_http(
    registry: &Registry,
    config: &config::Http,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let auth = Arc::new(Authorization::load(&config.bearer_token_file)?);
    let gateway = registry.gateway(config)?;
    let app = gateway
        .router()
        .layer(middleware::from_fn_with_state(auth, authorize));
    let draining = gateway.clone();
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown.await;
            draining.shutdown().await;
        })
        .await;
    // Also drain on a listener failure. Disconnected admitted HTTP handlers may
    // outlive their socket and retain reservations until durable outcomes finish.
    gateway.shutdown().await;
    result?;
    Ok(())
}
/// A supervisor failure ends the process after all admitted file work is joined.
pub async fn serve_worker(
    registry: &Registry,
    config: &Config,
    shutdown: impl Future<Output = ()>,
) -> Result<()> {
    let worker = registry.worker(config).await?;
    tokio::pin!(shutdown);
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            _ = tick.tick() => {
                let observation = worker.observation();
                eprintln!("{}", serde_json::json!({"worker": observation}));
                if !observation.running { break; }
            },
        }
    }
    worker.shutdown().await
}
