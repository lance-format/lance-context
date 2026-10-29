//! Authenticated process roles for an already provisioned annotation generation.
use super::*;
use crate::{
    annotations::{self, runtime},
    http::annotations::{Forwarder, Owner},
};
use tokio::{sync::oneshot, task::JoinSet};

/// Explicit offline handoff; normal owner/ingress startup never calls this.
pub async fn provision(config: &Config) -> Result<()> {
    config.validate()?;
    let a = config
        .annotations
        .as_ref()
        .context("annotation configuration missing")?;
    let table = &config.tables[0];
    let storage = Storage::resolve(
        table.uri.clone(),
        table.storage_options.clone(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    if let Some(source) = &a.projection {
        bootstrap::annotations::projection::provision(
            config.client()?,
            Arc::new(config.layout.clone()),
            table.binding.clone(),
            storage,
            table.bootstrap.clone(),
            source,
            a.staging.clone(),
            config.materializer_memory_bytes,
        )
        .await?;
        return Ok(());
    }
    let source = a
        .source
        .clone()
        .context("annotation handoff requires a persisted source snapshot")?;
    bootstrap::annotations::provision(
        config.client()?,
        Arc::new(config.layout.clone()),
        table.binding.clone(),
        storage,
        table.bootstrap.clone(),
        source,
        a.staging.clone(),
    )
    .await?;
    Ok(())
}

async fn connect(config: &Config) -> Result<Ingress> {
    config.validate()?;
    let options = config
        .annotations
        .as_ref()
        .context("annotation configuration missing")?;
    annotations::connect_ingress(
        config.client()?,
        Arc::new(config.layout.clone()),
        config.tables[0].binding.clone(),
        &options.staging,
        MergeMemoryBudget::new(config.ingress_memory_bytes),
    )
    .await
}

/// Ingress opens neither Lance nor a staging view, and never takes an owner lease.
pub async fn serve_ingress(
    config: &Config,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let ingress = connect(config).await?;
    let options = config
        .http
        .as_ref()
        .context("annotation HTTP configuration missing")?;
    let auth = Arc::new(Authorization::load(&options.bearer_token_file)?);
    let token = config::read(&options.bearer_token_file, 4096)?;
    let gateway = Forwarder::new(
        vec![(
            ingress,
            config.annotations.as_ref().unwrap().owner_url.clone(),
        )],
        receipt_keys(options)?,
        options.options.clone(),
        std::str::from_utf8(&token)?.trim().to_owned(),
    )?;
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
    gateway.shutdown().await;
    result?;
    Ok(())
}

/// One process owns one table/shard. Competing starts must acquire the existing
/// etcd claim before a WAL epoch; lease loss stops this role instead of stealing
/// a new epoch or retrying a mutation. A supervisor can start a replacement.
pub async fn serve_owner(
    config: &Config,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let ingress = connect(config).await?;
    let options = config
        .http
        .as_ref()
        .context("annotation HTTP configuration missing")?;
    let auth = Arc::new(Authorization::load(&options.bearer_token_file)?);
    let keys = receipt_keys(options)?;
    let claims = config
        .etcd
        .as_ref()
        .context("annotation owner requires etcd")?
        .claims()
        .await?;
    let claim = claims
        .take(&ingress)
        .await?
        .context("annotation shard already claimed")?;
    let result = async {
        let acquisition = claim.acquisition(&ingress).await?.context("annotation claim expired")?;
        let table = &config.tables[0];
        let a = config.annotations.as_ref().unwrap();
        // Repeated publications leave file metadata cached after staging drains.
        // Bound that cache separately from live annotation generations.
        let storage = Storage::resolve(
            table.uri.clone(),
            table.storage_options.clone(),
            Arc::new(lance::session::Session::new(
                lance::dataset::DEFAULT_INDEX_CACHE_SIZE,
                128 << 20,
                Arc::default(),
            )),
        ).await?;
        let (service, runtime) = runtime::Runtime::connect(
            ingress.client.clone(),
            ingress.layout.clone(),
            table.binding.clone(),
            storage,
            runtime::Configuration {
                limits: a.staging.clone(),
                upload_budget: ingress.budget.clone(),
                apply_budget: MergeMemoryBudget::new(a.apply_memory_bytes),
                publication_budget: MergeMemoryBudget::new(config.materializer_memory_bytes),
                publication: config.worker.publication.clone(),
            },
            acquisition,
        ).await?;
        let runtime = match &a.exclusive_reclamation {
            Some(options) => runtime.with_exclusive_reclamation(options.clone())?,
            None => runtime,
        };
        let owner = Owner::new(&runtime, keys, options.options.clone())?;
        let app = owner.router().layer(middleware::from_fn_with_state(auth, authorize));
        let (stop_http, stopped_http) = oneshot::channel();
        let (stop_runtime, stopped_runtime) = oneshot::channel();
        // JoinSet also aborts both role tasks if the parent is cancelled/panics.
        let mut tasks = JoinSet::new();
        tasks.spawn(async move {
            axum::serve(listener, app).with_graceful_shutdown(async {
                let _ = stopped_http.await;
            }).await.context("annotation owner listener failed")
        });
        tasks.spawn(runtime.run(async { let _ = stopped_runtime.await; }));
        let monitor = async {
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                if !claim.valid().await? {
                    break Err(anyhow::anyhow!("annotation owner lease lost"));
                }
                let (view, pressured) = service.observation().await;
                let source = service.source_observation();
                eprintln!("{}", serde_json::json!({"annotation": {
                    "observed_unix_ms": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis()),
                    "table": table.binding.table,
                    "applied": view.applied, "materialized": view.materialized,
                    "active_entries": view.active_entries, "frozen_entries": view.frozen_entries,
                    "retained_bytes": view.retained_bytes, "source_records": source.records,
                    "source_bytes": source.bytes, "backpressured": pressured,
                    "reclamation": service.reclamation_observation(),
                    "phases": service.phase_observation(),
                }}));
            }
        };
        let mut outcome = tokio::select! {
            _ = shutdown => Ok(()),
            result = monitor => result,
            result = tasks.join_next() => match result {
                Some(Ok(Err(error))) => Err(error),
                Some(Err(error)) => Err(error.into()),
                _ => Err(anyhow::anyhow!("annotation role stopped before shutdown")),
            },
        };
        // Keep the applier alive while admitted HTTP operations finish. No new
        // operations enter after shutdown starts; response buffers drain too.
        owner.shutdown().await;
        let _ = stop_http.send(());
        let _ = stop_runtime.send(());
        while let Some(result) = tasks.join_next().await {
            outcome = outcome.and(result.map_err(anyhow::Error::from).and_then(|r| r));
        }
        outcome
    }.await;
    let released = claim.release().await;
    result.and(released)
}
