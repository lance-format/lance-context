use axum::Router;
use clap::Parser;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use lance_context_master::config::MasterConfig;
use lance_context_master::routes;
use lance_context_master::scanner;
use lance_context_master::scheduler;
use lance_context_master::state::MasterState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let config = MasterConfig::parse();
    let addr = format!("{}:{}", config.host, config.port);

    if let Err(e) = std::fs::create_dir_all(&config.data_dir) {
        tracing::error!(
            "Failed to create data directory '{}': {}",
            config.data_dir,
            e
        );
        std::process::exit(1);
    }

    let ui_dir = config.ui_dir.clone();

    let state = match MasterState::new(config).await {
        Ok(state) => state,
        Err(e) => {
            tracing::error!("Failed to initialize master state: {:?}", e);
            std::process::exit(1);
        }
    };

    // Background stats scanner (detached; stops when the last Arc drops).
    let _scanner = scanner::spawn_scanner(&state);

    // Single serial compaction worker (+ optional periodic auto-sweep).
    let _scheduler = scheduler::spawn_scheduler(&state);

    let mut app = Router::new().nest("/api/v1", routes::api_router());

    // Serve the built UI (SPA) as a fallback when a `--ui-dir` is configured.
    if let Some(dir) = ui_dir {
        let index = format!("{}/index.html", dir.trim_end_matches('/'));
        let serve = ServeDir::new(&dir).fallback(tower_http::services::ServeFile::new(index));
        app = app.fallback_service(serve);
    }

    let app = app
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    tracing::info!("Starting lance-context-master on {}", addr);

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
