//! Shared Prometheus `/metrics` wiring for lance-context binaries.
//!
//! Each binary installs a single process-wide [`PrometheusRecorder`] via
//! [`install_recorder`], mounts the [`metrics_router`] on its main HTTP port,
//! and wraps its app with [`http_metrics_layer`] to record request counts and
//! latency. Domain metrics are emitted by each binary through the `metrics`
//! facade macros (`counter!`, `gauge!`, `histogram!`).
//!
//! Master exposes only its *own* metrics; it does not scrape workers.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{MatchedPath, State},
    http::Request,
    middleware::Next,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use metrics_process::Collector;

/// Handle used to render the Prometheus exposition text on demand, plus a
/// process-resource collector that is refreshed on each scrape.
#[derive(Clone)]
pub struct MetricsHandle {
    prometheus: PrometheusHandle,
    process: Arc<Collector>,
}

/// Install the global Prometheus recorder for this process.
///
/// Must be called exactly once, before any metrics are emitted. Panics if a
/// recorder was already installed (mirrors the metrics ecosystem contract).
pub fn install_recorder() -> MetricsHandle {
    let prometheus = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    let process = Collector::default();
    // Register the process metric descriptions once up front.
    process.describe();

    MetricsHandle {
        prometheus,
        process: Arc::new(process),
    }
}

/// Router exposing `GET /metrics` (relative to wherever it is nested/merged).
pub fn metrics_router(handle: MetricsHandle) -> Router {
    Router::new()
        .route("/metrics", get(render))
        .with_state(handle)
}

async fn render(State(handle): State<MetricsHandle>) -> impl IntoResponse {
    // Refresh process CPU/memory/FD/thread gauges just before rendering.
    handle.process.collect();
    let body = handle.prometheus.render();
    ([("content-type", "text/plain; version=0.0.4")], body)
}

/// Axum middleware that records `http_requests_total` and
/// `http_request_duration_seconds` for every handled request, labelled by
/// method, route template (low cardinality), and status code.
pub async fn http_metrics_layer(req: Request<axum::body::Body>, next: Next) -> Response {
    let start = Instant::now();
    let method = req.method().clone();
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());

    let response = next.run(req).await;

    let status = response.status().as_u16().to_string();
    let elapsed = start.elapsed().as_secs_f64();
    let labels = [
        ("method", method.to_string()),
        ("path", path),
        ("status", status),
    ];
    metrics::counter!("http_requests_total", &labels).increment(1);
    metrics::histogram!("http_request_duration_seconds", &labels).record(elapsed);

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn metrics_endpoint_renders_prometheus_text() {
        // install_recorder can only run once per process; guard so parallel
        // tests don't double-install.
        let handle = install_recorder();
        metrics::counter!("test_counter_total").increment(7);

        let app = metrics_router(handle);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("test_counter_total 7"), "body: {text}");
    }
}
