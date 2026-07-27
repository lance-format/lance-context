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
/// metrics — anything matching the `_duration_seconds` suffix.
///
/// # Why only 9 buckets
///
/// Every bucket is a separate exported series, and in Datadog every series is a
/// separately-billed custom metric. 9 buckets keeps adjacent ratios at or below
/// 6x, which bounds `histogram_quantile` interpolation error to roughly that
/// factor *within the straddling bucket only* — ample for latency SLOs, while
/// costing a third less than a denser ladder. Boundaries sit on the values
/// people actually alert on (10ms, 100ms, 250ms, 1s).
const REQUEST_LATENCY_BUCKETS: &[f64] = &[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 10.0, 60.0];

/// Coarser buckets (upper bounds, seconds) for long-running background jobs
/// (compaction, WAL merge, index builds) whose latency can reach minutes.
/// Without the extended tail every sample over 60s would fall into `+Inf`,
/// pinning `histogram_quantile` for high percentiles at the last finite bucket.
///
/// Same cardinality reasoning as [`REQUEST_LATENCY_BUCKETS`]; 7 buckets spanning
/// 0.5s..30min, since job latency is acted on at order-of-magnitude granularity.
const JOB_LATENCY_BUCKETS: &[f64] = &[0.5, 2.0, 10.0, 30.0, 120.0, 600.0, 1800.0];

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

/// Register HELP text and units for the application metrics.
///
/// Not merely cosmetic: Datadog's OpenMetrics check uses the exported `# TYPE`
/// to decide whether a series becomes a gauge or a monotonic count, and the
/// declared `Unit` is what makes latency render as a duration rather than a bare
/// number. Without these, `/metrics` exposes no `# HELP`/`# TYPE` for any
/// application metric.
///
/// Latency histograms are deliberately **unlabelled by result**. Failures are
/// counted (`*_errors_total`, `*_total{result=...}`), which costs one series per
/// value, instead of labelled onto a histogram, which costs one series per
/// bucket per value.
fn describe_metrics() {
    use metrics::{describe_counter, describe_gauge, describe_histogram, Unit};

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
        "Store time for a successful add request (add + optional flush), excluding body \
         parsing. Label: flush (true|false)."
    );
    describe_histogram!(
        "rollout_add_duration_seconds",
        Unit::Seconds,
        "RolloutStore::add — the durable WAL append only, success path."
    );
    describe_counter!(
        "rollout_add_errors_total",
        "Failed RolloutStore::add calls."
    );
    describe_histogram!(
        "rollout_flush_duration_seconds",
        Unit::Seconds,
        "RolloutStore::flush — sealing the memtable so added rows become readable. \
         Label: outcome (sealed|noop|fenced); the paths differ by orders of magnitude."
    );
    describe_counter!(
        "rollout_flush_errors_total",
        "Failed RolloutStore::flush calls."
    );
    describe_histogram!(
        "rollout_wal_merge_duration_seconds",
        Unit::Seconds,
        "Per-phase WAL self-merge latency. \
         Label: phase (seal|read|append|claim_epoch|drain|delete)."
    );
    describe_counter!(
        "rollout_wal_merge_errors_total",
        "Failed WAL self-merge phases; the phase label identifies where the merge died. \
         Label: phase."
    );
    describe_histogram!(
        "rollout_wal_merge_request_duration_seconds",
        Unit::Seconds,
        "Worker-side handling of a master-driven WAL merge."
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
        "Task work window only (excludes claim, permit wait, and commit). Label: kind. \
         Success/failure is on master_tasks_total."
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
        "Per-worker round trip of a WAL-merge fan-out; the slowest worker sets task latency."
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

    // Gauges. These deliberately carry no `_total` suffix: that suffix is the
    // counter convention, and Datadog infers a monotonic count from it, which
    // turns "how many exist right now" into a meaningless per-second rate.
    describe_gauge!(
        "master_experiments",
        "Live experiments seen by the last scan."
    );

    // `_stats` table upkeep. The alerting signal is
    // `master_stats_unreclaimed_versions` (and consecutive failures), not
    // `master_stats_version`: the version number climbs by design and says
    // nothing about disk usage, while unreclaimed versions are manifests still
    // sitting on storage.
    describe_histogram!(
        "master_stats_maintenance_duration_seconds",
        Unit::Seconds,
        "One _stats maintenance pass (compaction plus old-version cleanup)."
    );
    describe_counter!(
        "master_stats_versions_removed_total",
        "Old _stats manifest versions physically reclaimed by cleanup."
    );
    describe_counter!(
        "master_stats_maintenance_failures_total",
        "Failed _stats maintenance passes."
    );
    describe_gauge!(
        "master_stats_version",
        "Current _stats Lance version number. Monotonic by design; not an alerting signal."
    );
    describe_gauge!(
        "master_stats_maintenance_consecutive_failures",
        "Consecutive failed _stats maintenance passes; 0 after any success."
    );
    describe_gauge!(
        "master_stats_unreclaimed_versions",
        "_stats versions created since the last successful maintenance pass. \
         Sustained growth means old manifests are accumulating on storage."
    );
    describe_gauge!(
        "master_stats_hot_experiments",
        "Experiments currently held in the stats table. Bounded by retirement, \
         unlike the registry, which lists every experiment ever created."
    );
    describe_gauge!(
        "master_scan_experiments_skipped",
        "Experiments a scan round skipped because their base version had not moved. \
         A ratio near 1 of total is the healthy state at scale."
    );
    describe_gauge!(
        "master_scan_experiments_observed",
        "Experiments a scan round observed in full."
    );
    describe_counter!(
        "master_stats_experiments_retired_total",
        "Cold experiments merged, compacted, verified quiescent, and dropped from \
         the stats table."
    );
    describe_counter!(
        "master_stats_retire_failures_total",
        "Retirement attempts that failed; the experiment stays in the table and retries."
    );
    describe_gauge!(
        "master_rollout_rows",
        "Total rollout rows across all experiments as of the last scan."
    );
    describe_gauge!(
        "master_rollout_fragments",
        "Total fragments across all experiments as of the last scan."
    );
    describe_gauge!(
        "master_experiments_total",
        "DEPRECATED alias for master_experiments; a gauge despite the _total suffix."
    );
    describe_gauge!(
        "master_rollout_rows_total",
        "DEPRECATED alias for master_rollout_rows; a gauge despite the _total suffix."
    );
    describe_gauge!(
        "master_rollout_fragments_total",
        "DEPRECATED alias for master_rollout_fragments; a gauge despite the _total suffix."
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
        // so the `path` label cannot distinguish them). None carry `result` --
        // failures are counted, not timed.
        metrics::histogram!("rollout_add_duration_seconds").record(0.02);
        metrics::histogram!("rollout_flush_duration_seconds", "outcome" => "sealed").record(0.4);
        metrics::histogram!("rollout_add_request_duration_seconds", "flush" => "true").record(0.5);
        metrics::histogram!("rollout_add_request_duration_seconds", "flush" => "false")
            .record(0.01);
        metrics::counter!("rollout_add_errors_total").increment(1);
        // Job-scale: WAL merge phases and per-worker fan-out.
        metrics::histogram!("rollout_wal_merge_duration_seconds", "phase" => "append")
            .record(120.0);
        metrics::histogram!("master_task_phase_duration_seconds",
            "kind" => "merge_wal", "phase" => "claim")
        .record(90.0);
        metrics::histogram!("master_merge_wal_worker_duration_seconds").record(200.0);
        metrics::counter!("master_merge_wal_workers_total", "result" => "http_error").increment(1);
        metrics::gauge!("master_experiments").set(3.0);

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

        // Job-scale metric gets the extended tail (a 600s bucket exists), so a
        // 45s sample is not lumped straight into +Inf.
        assert!(
            text.contains("# TYPE master_task_duration_seconds histogram"),
            "job latency should be a histogram; body: {text}"
        );
        assert!(
            text.contains("master_task_duration_seconds_bucket{le=\"600\"}"),
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
            // Must assert the bucket on *this* metric's own series: a bare
            // `le="600"` substring check passes off any other job-scale metric
            // in the same body and silently tolerates a missing bucket config.
            assert!(
                text.lines().any(|line| {
                    line.starts_with(&format!("{name}_bucket{{")) && line.contains("le=\"600\"")
                }),
                "{name} should use the extended (job) bucket set; body: {text}"
            );
        }

        // Every application metric should carry HELP so a scrape is
        // interpretable without reading the source, and so Datadog's
        // OpenMetrics check can infer the right type.
        assert!(
            text.contains("# HELP rollout_add_duration_seconds"),
            "new metrics should be described; body: {text}"
        );

        // Latency histograms must not carry a `result` label. Failures belong on
        // counters: `result` on a histogram costs one series *per bucket* per
        // value, which in Datadog is one billed custom metric each.
        for line in text.lines() {
            if line.contains("_duration_seconds_bucket{") || line.contains("_seconds_sum{") {
                assert!(
                    !line.contains("result=\""),
                    "latency histograms must not carry a `result` label \
                     (use an errors counter instead): {line}"
                );
            }
        }

        // Gauges must not be named `_total`: Datadog infers a monotonic count
        // from that suffix, turning "how many exist now" into a nonsense rate.
        // The deprecated aliases are the documented exception.
        assert!(
            text.contains("# TYPE master_experiments gauge"),
            "gauge should be exported without a _total suffix; body: {text}"
        );

        // Cardinality budget. Every series is a billed custom metric in Datadog,
        // so a label added without thought is a cost regression. The bound is
        // deliberately close to the actual count (112 at the time of writing):
        // adding one two-valued label to a job-scale histogram costs ~9 series
        // and trips this, forcing the tradeoff to be made explicitly rather than
        // discovered on an invoice.
        let series = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .count();
        assert!(
            series <= 125,
            "metric cardinality regressed to {series} series (budget 125). Every series is a \
             billed custom metric in Datadog — prefer a counter label over a histogram label."
        );
    }
}
