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

use crate::config::ServerConfig;
use crate::state::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let config = ServerConfig::parse();
    let addr = format!("{}:{}", config.host, config.port);

    if let Err(e) = std::fs::create_dir_all(&config.data_dir) {
        tracing::error!(
            "Failed to create data directory '{}': {}",
            config.data_dir,
            e
        );
        std::process::exit(1);
    }

    let state = Arc::new(AppState::new(config));

    let app = routes::router()
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    tracing::info!("Starting lance-context-server on {}", addr);

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    tracing::info!("Shutting down");
}
