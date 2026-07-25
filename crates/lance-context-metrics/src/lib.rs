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
///
/// Note the `_lock_wait_seconds` entries: they do not end in `_duration_seconds`
/// and so match no suffix rule, meaning without an explicit entry here they
/// would fall back to the exporter's default summary rather than a histogram.
const JOB_LATENCY_METRICS: &[&str] = &[
    "master_task_duration_seconds",
    "master_task_phase_duration_seconds",
    "master_merge_wal_worker_duration_seconds",
    "rollout_compaction_duration_seconds",
    "rollout_compaction_lock_wait_seconds",
    "rollout_wal_merge_duration_seconds",
    "rollout_wal_merge_request_duration_seconds",
    "rollout_wal_merge_lock_wait_seconds",
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
    describe_metrics();

    MetricsHandle {
        prometheus,
        process: Arc::new(process),
    }
}

/// Register HELP text for the application metrics.
///
/// Purely descriptive, but without it `/metrics` exposes no `# HELP`/`# TYPE`
/// for any application metric, so a scraped series is uninterpretable without
/// reading the source.
fn describe_metrics() {
    use metrics::{describe_counter, describe_histogram, Unit};

    describe_histogram!(
        "http_request_duration_seconds",
        Unit::Seconds,
        "End-to-end HTTP handler latency, including body parsing and admission control."
    );

    // Rollout write path. `rollout_add_request_duration_seconds` is labelled by
    // `flush` because flush is a query param on the add route, so the HTTP
    // metric's `path` label cannot distinguish the two.
    describe_histogram!(
        "rollout_add_request_duration_seconds",
        Unit::Seconds,
        "Store time for an add request (add + optional flush), excluding body parsing. \
         Labels: flush, result."
    );
    describe_histogram!(
        "rollout_add_duration_seconds",
        Unit::Seconds,
        "RolloutStore::add — the durable WAL append only. Label: result."
    );
    describe_histogram!(
        "rollout_flush_duration_seconds",
        Unit::Seconds,
        "RolloutStore::flush — sealing the memtable so added rows become readable. \
         Labels: result, outcome (sealed|noop|fenced)."
    );
    describe_histogram!(
        "rollout_wal_merge_duration_seconds",
        Unit::Seconds,
        "Per-phase WAL self-merge latency. \
         Labels: phase (seal|read|append|claim_epoch|drain|delete), result."
    );
    describe_histogram!(
        "rollout_wal_merge_request_duration_seconds",
        Unit::Seconds,
        "Worker-side handling of a master-driven WAL merge. Label: result."
    );
    describe_histogram!(
        "rollout_wal_merge_lock_wait_seconds",
        Unit::Seconds,
        "Time waiting for the store write lock before a WAL merge (blocks all ingest)."
    );
    describe_histogram!(
        "rollout_compaction_lock_wait_seconds",
        Unit::Seconds,
        "Time waiting for the store write lock before compaction."
    );

    // Master task lifecycle.
    describe_histogram!(
        "master_task_duration_seconds",
        Unit::Seconds,
        "Task work window only (excludes claim, permit wait, and commit). \
         Labels: kind, result."
    );
    describe_histogram!(
        "master_task_phase_duration_seconds",
        Unit::Seconds,
        "Task latency broken down by phase. \
         Labels: kind, phase (claim|permit_wait|work|commit)."
    );
    describe_histogram!(
        "master_merge_wal_worker_duration_seconds",
        Unit::Seconds,
        "Per-worker round trip of a WAL-merge fan-out; the slowest worker sets task latency. \
         Label: result."
    );
    describe_counter!(
        "master_merge_wal_workers_total",
        "WAL-merge fan-out outcomes per worker. \
         Labels: result (ok|not_found|http_error|transport_error)."
    );
    describe_counter!(
        "master_merge_wal_generations_reclaimed_total",
        "MemWAL generations folded into base tables by master-driven merges."
    );
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

        // Per-operation write-path metrics: add and flush must be separable, and
        // an add is separable by whether it flushed (they share one HTTP route,
        // so the `path` label cannot distinguish them).
        metrics::histogram!("rollout_add_duration_seconds", "result" => "ok").record(0.02);
        metrics::histogram!(
            "rollout_flush_duration_seconds",
            "result" => "ok",
            "outcome" => "sealed",
        )
        .record(0.4);
        metrics::histogram!(
            "rollout_add_request_duration_seconds",
            "flush" => "true",
            "result" => "ok",
        )
        .record(0.5);
        metrics::histogram!(
            "rollout_add_request_duration_seconds",
            "flush" => "false",
            "result" => "ok",
        )
        .record(0.01);
        // Job-scale: WAL merge phases and per-worker fan-out.
        metrics::histogram!(
            "rollout_wal_merge_duration_seconds",
            "phase" => "append",
            "result" => "ok",
        )
        .record(120.0);
        metrics::histogram!("master_task_phase_duration_seconds",
            "kind" => "merge_wal", "phase" => "claim")
        .record(90.0);
        metrics::histogram!("master_merge_wal_worker_duration_seconds", "result" => "ok")
            .record(200.0);

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

        // add and flush must be distinct series, not one blended number.
        assert!(
            text.contains("# TYPE rollout_add_duration_seconds histogram")
                && text.contains("# TYPE rollout_flush_duration_seconds histogram"),
            "add and flush must be separate histograms; body: {text}"
        );
        // flush's `outcome` separates real sealing from the no-op fast path,
        // which otherwise dominates the distribution with near-zero samples.
        assert!(
            text.contains("outcome=\"sealed\""),
            "flush should carry an outcome label; body: {text}"
        );
        // The two add paths differ only by query param, so the `flush` label is
        // the only thing that can separate them.
        assert!(
            text.contains("flush=\"true\"") && text.contains("flush=\"false\""),
            "flushing and non-flushing adds must be separable; body: {text}"
        );

        // Job-scale bucket set must apply to the new long-running metrics too,
        // otherwise a 120s merge phase lands in +Inf and p99 is unusable.
        for name in [
            "rollout_wal_merge_duration_seconds",
            "master_task_phase_duration_seconds",
            "master_merge_wal_worker_duration_seconds",
        ] {
            assert!(
                text.contains(&format!("# TYPE {name} histogram")),
                "{name} should be a histogram; body: {text}"
            );
            // Must assert the 300s bucket on *this* metric's own series: a bare
            // `le="300"` substring check passes off any other job-scale metric
            // in the same body and silently tolerates a missing bucket config.
            assert!(
                text.lines().any(|line| {
                    line.starts_with(&format!("{name}_bucket{{")) && line.contains("le=\"300\"")
                }),
                "{name} should use the extended (job) bucket set; body: {text}"
            );
        }

        // Every application metric should carry HELP so a scrape is
        // interpretable without reading the source.
        assert!(
            text.contains("# HELP rollout_add_duration_seconds"),
            "new metrics should be described; body: {text}"
        );
    }
}
