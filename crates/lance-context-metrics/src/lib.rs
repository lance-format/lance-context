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
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use metrics_process::Collector;

/// Explicit histogram buckets (upper bounds, seconds) for request-scale latency
/// metrics — anything matching the `_duration_seconds` suffix. Tuned for
/// sub-second-to-tens-of-seconds work like HTTP requests and rollout scans.
const REQUEST_LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// Coarser buckets (upper bounds, seconds) for long-running background jobs
/// (compaction, WAL merge, index builds) whose latency can reach minutes.
/// Without the extended tail every sample over 60s would fall into `+Inf`,
/// pinning `histogram_quantile` for high percentiles at the last finite bucket.
const JOB_LATENCY_BUCKETS: &[f64] = &[
    0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0,
];

/// Metric names whose latency is job-scale rather than request-scale. A `Full`
/// matcher outranks the `_duration_seconds` `Suffix` matcher (the exporter
/// applies Full > Prefix > Suffix), so these get [`JOB_LATENCY_BUCKETS`].
const JOB_LATENCY_METRICS: &[&str] = &[
    "master_task_duration_seconds",
    "rollout_compaction_duration_seconds",
];

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
///
/// Latency histograms (`*_duration_seconds`) are configured with explicit
/// buckets so they export as true Prometheus histograms (`_bucket{le="..."}`)
/// rather than the exporter's default rolling summaries. This lets downstream
/// compute `histogram_quantile()` over an arbitrary window at query time
/// instead of reading an exporter-internal, slow-decaying quantile.
pub fn install_recorder() -> MetricsHandle {
    let mut builder = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Suffix("_duration_seconds".to_string()),
            REQUEST_LATENCY_BUCKETS,
        )
        .expect("request latency buckets are non-empty");
    for name in JOB_LATENCY_METRICS {
        builder = builder
            .set_buckets_for_metric(Matcher::Full((*name).to_string()), JOB_LATENCY_BUCKETS)
            .expect("job latency buckets are non-empty");
    }
    let prometheus = builder
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
        // A request-scale and a job-scale latency sample, to assert both bucket
        // sets render as true histograms rather than summaries.
        metrics::histogram!("http_request_duration_seconds").record(0.3);
        metrics::histogram!("master_task_duration_seconds").record(45.0);

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

        // Latency metrics must export as bucketed histograms, not summaries.
        assert!(
            text.contains("# TYPE http_request_duration_seconds histogram"),
            "request latency should be a histogram, not a summary; body: {text}"
        );
        assert!(
            text.contains("http_request_duration_seconds_bucket{le=\"1\"}"),
            "request latency should expose _bucket series; body: {text}"
        );
        assert!(
            !text.contains("http_request_duration_seconds{quantile="),
            "request latency must not export summary quantiles; body: {text}"
        );

        // Job-scale metric gets the extended tail (a 300s bucket exists), so a
        // 45s sample is not lumped straight into +Inf.
        assert!(
            text.contains("# TYPE master_task_duration_seconds histogram"),
            "job latency should be a histogram; body: {text}"
        );
        assert!(
            text.contains("master_task_duration_seconds_bucket{le=\"300\"}"),
            "job latency should use the extended (job) bucket set; body: {text}"
        );
    }
}
