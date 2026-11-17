//! Runs the real annotation owner with existing object-store metrics enabled.
//! The additional listener is loopback-only and serves diagnostic reads.
//! Owner status also reports annotation_read_io with fixed file classes: narrow
//! data, body data, indices, metadata and other. These are process-wide store API
//! calls and delivered bytes, not caller attribution or HTTP/wire accounting.
//! Failed and dropped reads are separate; local file payload bytes are unknown.
//! The existing scheme-level metrics and shared store/cache identity are retained.
use anyhow::{bail, Context, Result};
use lance_context_ingest::application::{annotations, Config};
use std::{future::IntoFuture, net::SocketAddr, path::Path};
use tokio::{net::TcpListener, sync::oneshot};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        bail!("usage: annotation_io_owner CONFIG LOOPBACK_METRICS_ADDRESS");
    }
    let address: SocketAddr = args[1].parse()?;
    if !address.ip().is_loopback() {
        bail!("diagnostic metrics address must be loopback");
    }
    let config = Config::read(Path::new(&args[0]))?;
    config.validate()?;
    let options = config
        .http
        .as_ref()
        .context("annotation owner requires HTTP configuration")?;
    let listener = TcpListener::bind(options.listen).await?;
    let metrics_listener = TcpListener::bind(address).await?;
    let metrics = lance_context_metrics::install_recorder();
    eprintln!(
        "{}",
        serde_json::json!({"annotation_io_metrics": {
            "address": metrics_listener.local_addr()?,
            "scope": "process-wide object-store operations, including concurrent reads"
        }})
    );
    let (stop_owner, stopped_owner) = oneshot::channel();
    let (stop_metrics, stopped_metrics) = oneshot::channel();
    let owner = annotations::serve_owner(&config, listener, async {
        let _ = stopped_owner.await;
    });
    let metrics = axum::serve(
        metrics_listener,
        lance_context_metrics::metrics_router(metrics),
    )
    .with_graceful_shutdown(async {
        let _ = stopped_metrics.await;
    })
    .into_future();
    tokio::pin!(owner, metrics);
    let signal = shutdown_signal()?;
    tokio::pin!(signal);
    tokio::select! {
        result = &mut owner => {
            let _ = stop_metrics.send(());
            metrics.await.context("diagnostic metrics listener failed")?;
            result
        }
        result = &mut metrics => {
            let _ = stop_owner.send(());
            owner.await?;
            result.context("diagnostic metrics listener failed")?;
            bail!("diagnostic metrics listener stopped before shutdown");
        }
        () = &mut signal => {
            let _ = stop_owner.send(());
            let result = owner.await;
            let _ = stop_metrics.send(());
            metrics.await.context("diagnostic metrics listener failed")?;
            result
        }
    }
}

#[cfg(unix)]
fn shutdown_signal() -> Result<impl std::future::Future<Output = ()>> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut terminate = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    Ok(async move {
        tokio::select! { _ = terminate.recv() => {}, _ = interrupt.recv() => {} }
    })
}

#[cfg(not(unix))]
fn shutdown_signal() -> Result<impl std::future::Future<Output = ()>> {
    Ok(async {
        let _ = tokio::signal::ctrl_c().await;
    })
}
