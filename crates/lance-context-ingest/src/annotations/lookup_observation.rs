//! Completed immutable-key lookup windows for one service lifetime.
use super::*;

/// A native metric may be absent from a successful scan's execution plan.
/// Keep its coverage separate from its total: a measured zero is a sample.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LookupMetricObservation {
    pub samples: u64,
    pub total: u64,
}

impl LookupMetricObservation {
    fn record(&mut self, value: Option<usize>) {
        if let Some(value) = value {
            self.samples = self.samples.saturating_add(1);
            self.total = self.total.saturating_add(value as u64);
        }
    }
}

/// Cumulative successful lookup windows, including probes whose captured base
/// was subsequently replaced. Cached reuse and admission refusals add nothing.
/// Failed/cancelled attempts remain visible in the existing lookup phase.
/// A window entirely resolved by staging starts no native scan.
#[derive(Clone, Debug, Default, Serialize)]
pub struct BodyLookupObservation {
    pub completed_lookups: u64,
    pub requested_keys: u64,
    /// Keys resolved from the same captured active/frozen view, without history I/O.
    pub staged_keys: u64,
    pub scans_started: u64,
    /// Returned key columns, including candidates removed by composite matching.
    /// These are not physical storage bytes or all scanned source rows.
    pub projected_rows: u64,
    pub projected_array_bytes: u64,
    pub elapsed_us: u64,
    pub max_elapsed_us: u64,
    /// Completed lookups with native plan metrics sampled at completion,
    /// including all-hit early exits. Totals exclude later background work
    /// and staging-only resolutions; they are not all object-store I/O.
    pub native_summaries: u64,
    pub lance_bytes_read: u64,
    pub lance_requests: u64,
    pub lance_indices_loaded: u64,
    pub fragments_scanned: LookupMetricObservation,
    /// Native range count; an approximation when decoded batches finish out of order.
    pub ranges_scanned: LookupMetricObservation,
    /// Native source rows scanned, distinct from returned projected rows.
    pub rows_scanned: LookupMetricObservation,
    /// Nanoseconds waiting on native task streams, not CPU time. Concurrent
    /// native tasks can overlap; do not subtract this from wall-clock elapsed.
    pub task_wait_time_ns: LookupMetricObservation,
}

impl BodyLookupObservation {
    pub(super) fn record(&mut self, value: &crate::turn_ingestion::TurnLookupObservation) {
        self.completed_lookups = self.completed_lookups.saturating_add(1);
        self.requested_keys = self
            .requested_keys
            .saturating_add(value.requested_keys as u64);
        self.staged_keys = self.staged_keys.saturating_add(value.staged_keys as u64);
        self.scans_started = self.scans_started.saturating_add(value.scans_started);
        self.projected_rows = self.projected_rows.saturating_add(value.projected_rows);
        self.projected_array_bytes = self
            .projected_array_bytes
            .saturating_add(value.projected_array_bytes);
        self.elapsed_us = self.elapsed_us.saturating_add(value.elapsed_us);
        self.max_elapsed_us = self.max_elapsed_us.max(value.elapsed_us);
        self.fragments_scanned.record(value.fragments_scanned);
        self.ranges_scanned.record(value.ranges_scanned);
        self.rows_scanned.record(value.rows_scanned);
        self.task_wait_time_ns.record(value.task_wait_time_ns);
        if let (Some(bytes), Some(requests), Some(indices)) = (
            value.lance_bytes_read,
            value.lance_requests,
            value.lance_indices_loaded,
        ) {
            self.native_summaries = self.native_summaries.saturating_add(1);
            self.lance_bytes_read = self.lance_bytes_read.saturating_add(bytes as u64);
            self.lance_requests = self.lance_requests.saturating_add(requests as u64);
            self.lance_indices_loaded = self.lance_indices_loaded.saturating_add(indices as u64);
        }
    }
}

impl Service {
    /// Totals for this table/shard owner only; replacement owners start at zero.
    pub fn body_lookup_observation(&self) -> BodyLookupObservation {
        self.shared.body_lookups.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn_ingestion::TurnLookupObservation;

    #[test]
    fn scan_metrics_distinguish_missing_zero_and_partial_coverage() {
        let mut totals = BodyLookupObservation::default();
        totals.record(&TurnLookupObservation {
            requested_keys: 2,
            staged_keys: 2,
            ..Default::default()
        });
        assert_eq!(totals.fragments_scanned.samples, 0);
        assert_eq!(totals.native_summaries, 0);
        for rows in [None, Some(0), Some(7)] {
            totals.record(&TurnLookupObservation {
                scans_started: 1,
                lance_bytes_read: Some(0),
                lance_requests: Some(0),
                lance_indices_loaded: Some(0),
                fragments_scanned: Some(1),
                rows_scanned: rows,
                task_wait_time_ns: Some(11),
                ..Default::default()
            });
        }
        assert_eq!(totals.completed_lookups, 4);
        assert_eq!(totals.native_summaries, 3);
        assert_eq!(
            totals.fragments_scanned,
            LookupMetricObservation {
                samples: 3,
                total: 3
            }
        );
        assert_eq!(
            totals.rows_scanned,
            LookupMetricObservation {
                samples: 2,
                total: 7
            }
        );
        assert_eq!(totals.ranges_scanned, LookupMetricObservation::default());
        assert_eq!(
            totals.task_wait_time_ns,
            LookupMetricObservation {
                samples: 3,
                total: 33
            }
        );
    }
}
