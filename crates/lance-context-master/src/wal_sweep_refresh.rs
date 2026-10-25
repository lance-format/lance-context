//! Advisory preflight for persisted periodic stats, independent of recovery.
use std::time::{Duration, Instant};

const MAX_PROBES: usize = 64;
const PASS_TIME: Duration = Duration::from_secs(10);
const PROBE_TIME: Duration = Duration::from_secs(3);

pub(crate) struct Refresh {
    remaining: usize,
    time_left: Duration,
}

impl Refresh {
    pub(crate) fn new() -> Self {
        Self {
            remaining: MAX_PROBES,
            time_left: PASS_TIME,
        }
    }

    /// Only a completed fresh all-shard observation can suppress this sweep's
    /// candidate. Budget exhaustion/errors retain ordinary admission. In
    /// particular, no negative observation survives a leader change or tick.
    pub(crate) async fn confirmed_empty(&mut self, target: &str, uri: &str) -> bool {
        let time = self.time_left.min(PROBE_TIME);
        if self.remaining == 0 || time.is_zero() {
            metrics::counter!("master_periodic_wal_refresh_total", "outcome" => "budget")
                .increment(1);
            return false;
        }
        self.remaining -= 1;
        let started = Instant::now();
        let result =
            tokio::time::timeout(time, lance_context_core::wal_presence::has_pending_wal(uri))
                .await;
        let elapsed = started.elapsed();
        self.time_left = self.time_left.saturating_sub(elapsed);
        let outcome = match result {
            Ok(Ok(false)) => "empty",
            Ok(Ok(true)) => "pending",
            Ok(Err(error)) => {
                tracing::debug!(target, %error, "WAL preflight unresolved; ordinary admission retained");
                "unknown"
            }
            Err(_) => "unknown",
        };
        metrics::counter!("master_periodic_wal_refresh_total", "outcome" => outcome).increment(1);
        metrics::histogram!("master_periodic_wal_refresh_duration_seconds", "outcome" => outcome)
            .record(elapsed.as_secs_f64());
        if outcome == "empty" {
            tracing::info!(
                target,
                "fresh all-shard metadata is empty; periodic WAL candidate skipped"
            );
            true
        } else {
            false
        }
    }
}
