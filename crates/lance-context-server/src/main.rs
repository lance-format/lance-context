mod config;
mod error;
mod routes;
mod state;

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use lance_context_core::create_local_dir_if_needed;

use crate::config::ServerConfig;
use crate::state::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let config = ServerConfig::parse();
    let addr = format!("{}:{}", config.host, config.port);

    if let Err(e) = create_local_dir_if_needed(&config.data_dir) {
        tracing::error!(
            "Failed to create local data directory '{}': {}",
            config.data_dir,
            e
        );
        std::process::exit(1);
    }

    let state = match AppState::new(config).await {
        Ok(state) => Arc::new(state),
        Err(e) => {
            tracing::error!("Failed to initialize server state: {:?}", e);
            std::process::exit(1);
        }
    };

    // Single process-wide WAL-cleanup sweeper (replaces one timer per store).
    // The handle is detached: it runs for the server's lifetime and stops when
    // the last `Arc<AppState>` is dropped.
    let _sweeper = state.spawn_global_sweeper();

    // Periodic MemWAL flush sweeper: bounds rollout read-after-write latency
    // without serializing concurrent appends. Detached for the server lifetime.
    let _flush_sweeper = state.spawn_flush_sweeper();

    // Install the Prometheus recorder once, before any metrics are emitted.
    let metrics_handle = lance_context_metrics::install_recorder();

    let app = routes::router()
        .with_state(state.clone())
        .merge(lance_context_metrics::metrics_router(metrics_handle))
        .layer(axum::middleware::from_fn(
            lance_context_metrics::http_metrics_layer,
        ))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    tracing::info!("Starting lance-context-server on {}", addr);

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    // Connections have drained. Deterministically close resident rollout writers
    // (whose `ShardWriter` background tasks need an explicit `close().await`)
    // before the runtime tears down.
    state.shutdown().await;
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    tracing::info!("Shutting down");
}
