#![recursion_limit = "256"]

use axum::Router;
use clap::Parser;
use tokio::net::TcpListener;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use lance_context_core::create_local_dir_if_needed;
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
    if let Some(path) = &config.legacy_retirement_request {
        let result = async {
            let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
            if bytes.len() > 8192 {
                return Err("retirement request exceeds metadata bound".to_string());
            }
            let request = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            let coordinator = lance_context_merge::Coordinator::new(
                config.etcd.connect().await.map_err(|e| e.to_string())?,
                config.etcd.etcd_prefix.clone(),
            );
            lance_context_master::legacy_retirement::execute_command(&config, &coordinator, request)
                .await
        }
        .await;
        match result {
            Ok(response) => println!("{}", serde_json::to_string(&response).unwrap()),
            Err(error) => {
                tracing::error!(%error, "legacy retirement incomplete; ownership retained");
                std::process::exit(1);
            }
        }
        return;
    }
    let addr = format!("{}:{}", config.host, config.port);

    // Install the recorder before state recovery and scheduler startup so
    // recovered queue depth is represented immediately.
    let metrics_handle = lance_context_metrics::install_recorder();

    if let Err(e) = create_local_dir_if_needed(&config.data_dir) {
        tracing::error!(
            "Failed to create local data directory '{}': {}",
            config.data_dir,
            e
        );
        std::process::exit(1);
    }

    if let Some(target) = config.catchup.target.clone() {
        use lance_context_master::catchup::ExecuteOutcome;
        match lance_context_master::catchup::execute(config, &target).await {
            Ok(ExecuteOutcome::Completed) => {}
            Ok(ExecuteOutcome::AdmissionBusy) => {
                // Machine-readable, attempt-bound contract with the supervisor.
                // Emitted only before a task claim; never infer this from error text.
                println!(
                    "{}",
                    serde_json::json!({
                        "event": "catchup_admission_deferred",
                        "version": 1,
                        "reason": "busy_before_claim",
                        "target": target,
                        "attempt_id": std::env::var("CATCHUP_ATTEMPT_ID").ok(),
                    })
                );
                std::process::exit(75);
            }
            Err(error) => {
                tracing::error!(%target, %error, "dedicated catch-up failed");
                std::process::exit(1);
            }
        }
        return;
    }

    let ui_dir = config.ui_dir.clone();

    let state = match MasterState::new(config).await {
        Ok(state) => state,
        Err(e) => {
            tracing::error!("Failed to initialize master state: {:?}", e);
            std::process::exit(1);
        }
    };

    let backlog_metrics = lance_context_master::backlog_metrics::BacklogMetrics::new(
        std::time::Duration::from_secs(state.config.backlog_metrics_max_age_secs),
    );
    let _backlog_reader = backlog_metrics.spawn(state.task_store.clone());
    let metrics_handle = metrics_handle.with_extra_metrics(move || backlog_metrics.render());

    // Background stats scanner (detached; stops when the last Arc drops).
    let _scanner = scanner::spawn_scanner(&state);

    // Durable scheduler poller (+ optional coordinated auto-sweep).
    let _scheduler = scheduler::spawn_scheduler(&state);
    let _catchup = lance_context_master::catchup::spawn(&state);

    let mut app = Router::new().nest("/api/v1", routes::api_router());

    // Serve the built UI (SPA) as a fallback when a `--ui-dir` is configured.
    if let Some(dir) = ui_dir {
        let index = format!("{}/index.html", dir.trim_end_matches('/'));
        let serve = ServeDir::new(&dir).fallback(tower_http::services::ServeFile::new(index));
        app = app.fallback_service(serve);
    }

    let admission = state.admission.clone();
    let app = app
        .with_state(state)
        .merge(lance_context_metrics::metrics_router(metrics_handle))
        .layer(axum::middleware::from_fn(
            lance_context_metrics::http_metrics_layer,
        ))
        .layer(TraceLayer::new_for_http());

    tracing::info!("Starting lance-context-master on {}", addr);

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            wait_for_shutdown_signal().await;
            let status = admission.begin_drain();
            tracing::info!(executor_id = %status.executor_id, active = status.active_operations,
                "stopped local admission; waiting for admitted work");
            admission.wait_drained().await;
            tracing::info!("local executor drained");
        })
        .await
        .unwrap();
}

#[cfg(unix)]
async fn wait_for_shutdown_signal() {
    use tokio::signal::unix::{signal as unix_signal, SignalKind};

    // Kubernetes and other process managers stop containers with SIGTERM.
    let mut sigterm =
        unix_signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");

    tokio::select! {
        result = tokio::signal::ctrl_c() => {
            result.expect("failed to install Ctrl+C handler");
            tracing::info!(signal = "SIGINT", "Shutting down");
        }
        _ = sigterm.recv() => {
            tracing::info!(signal = "SIGTERM", "Shutting down");
        }
    }
}

#[cfg(not(unix))]
async fn wait_for_shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    tracing::info!(signal = "Ctrl+C", "Shutting down");
}
