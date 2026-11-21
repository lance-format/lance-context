//! Reconcile narrowly eligible worker failures after a native takeover.
//!
//! The recovery leader keeps retrying every retained failure revision. Native
//! catch-up finishes `master:catchup`, so it otherwise cannot resolve debt under
//! the old worker endpoint, even after it drains the whole table successfully.
use std::time::Duration;

use lance_context_merge::{failure::FailureObservation, ClaimProof, Coordinator, Execution};

const OBSERVE_TIMEOUT: Duration = Duration::from_secs(3);

pub(crate) async fn snapshot(
    coordinator: &Coordinator,
    target: &str,
    endpoints: &[String],
) -> Vec<FailureObservation> {
    let result = tokio::time::timeout(
        OBSERVE_TIMEOUT,
        coordinator.target_failure_observations(target),
    )
    .await;
    match result {
        Ok(Ok(rows)) => rows
            .into_iter()
            .filter(|row| {
                endpoints.contains(&row.failure.endpoint)
                    && row
                        .failure
                        .is_worker_failure_reconcilable_after_native_drain()
            })
            .collect(),
        _ => {
            // An unavailable snapshot is not permission to clear anything and
            // must not prevent the original native merge from making progress.
            metrics::counter!("master_native_failure_reconciliation_total", "result" => "snapshot_unknown").increment(1);
            Vec::new()
        }
    }
}

/// Called only after successful native work, joined commit leaves, and no
/// uncertain commit, while the same canonical task/execution is still owned.
pub(crate) async fn reconcile(
    coordinator: &Coordinator,
    proof: &ClaimProof,
    running: &Execution,
    uri: &str,
    observed: Vec<FailureObservation>,
) {
    if observed.is_empty() {
        return;
    }
    // Include historical shards absent from current worker configuration. The
    // bounded metadata check alone cannot authorize recovery: it supplements
    // successful fenced execution above. A partial slice/error/timeout retains
    // all old debt. New writer hints and all ownership records remain intact.
    if !matches!(
        tokio::time::timeout(
            OBSERVE_TIMEOUT,
            lance_context_core::wal_presence::has_pending_wal(uri),
        )
        .await,
        Ok(Ok(false))
    ) {
        metrics::counter!("master_native_failure_reconciliation_total", "result" => "not_drained")
            .increment(1);
        return;
    }
    for row in observed {
        // Do not wrap a mutating RPC in a cancellation timeout or replay an
        // ambiguous result. The etcd client bounds the original request; a new
        // recovery pass will observe whatever actually committed.
        let result = coordinator
            .clear_observed_failure_after_native_recovery(proof, running, &row)
            .await;
        let label = match result {
            Ok(true) => {
                tracing::info!(target = %running.target, endpoint = %row.failure.endpoint,
                    failure_revision = row.revision, execution = %running.id,
                    "native recovery retired prior worker failure");
                "retired"
            }
            Ok(false) => "changed",
            Err(error) => {
                tracing::warn!(target = %running.target, endpoint = %row.failure.endpoint,
                    %error, "native failure reconciliation deferred");
                "unknown"
            }
        };
        metrics::counter!("master_native_failure_reconciliation_total", "result" => label)
            .increment(1);
    }
}
