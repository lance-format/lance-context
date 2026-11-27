//! Completed immutable-key lookup windows for one service lifetime.
use super::*;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefetchStop {
    PageExhausted,
    RequestBound,
    ByteBound,
    Admission,
}

/// Completed prefetch decisions, before lookup admission. These are selected
/// inputs, not applied requests or completed native scans. A full-body window
/// can span committed source pages; an empty response ends it without polling
/// for future writes.
#[derive(Clone, Debug, Default, Serialize)]
pub struct BodyPrefetchObservation {
    pub preparations: u64,
    pub requests: u64,
    pub payload_bytes: u64,
    pub last_requests: u64,
    pub last_payload_bytes: u64,
    pub last_stop: Option<PrefetchStop>,
    pub page_exhausted: u64,
    pub request_bound: u64,
    pub byte_bound: u64,
    pub admission: u64,
    /// Validated empty ReadSource responses with more locally applied metadata
    /// beyond the page cursor. That metadata need not match this table/bucket.
    pub source_empty_with_remaining_metadata: u64,
    /// Empty responses at this replica's observed end, not quorum catch-up.
    pub source_empty_at_local_end: u64,
    /// Empty responses at the consumer's fixed publication-recovery cut.
    pub source_empty_at_replay_cut: u64,
    /// ReadSource responses from a replica behind the requested cursor.
    pub source_cursor_ahead: u64,
}

impl BodyPrefetchObservation {
    pub(super) fn record(&mut self, requests: usize, bytes: u64, stop: PrefetchStop) {
        self.preparations = self.preparations.saturating_add(1);
        self.requests = self.requests.saturating_add(requests as u64);
        self.payload_bytes = self.payload_bytes.saturating_add(bytes);
        self.last_requests = requests as u64;
        self.last_payload_bytes = bytes;
        self.last_stop = Some(stop);
        let count = match stop {
            PrefetchStop::PageExhausted => &mut self.page_exhausted,
            PrefetchStop::RequestBound => &mut self.request_bound,
            PrefetchStop::ByteBound => &mut self.byte_bound,
            PrefetchStop::Admission => &mut self.admission,
        };
        *count = count.saturating_add(1);
    }
}

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

/// Lookup totals count successful windows, including probes whose captured base
/// was subsequently replaced. Reusing a completed window or refusing admission
/// adds no lookup totals; prefetch and fallback attempts have separate counters.
/// Failed/cancelled lookups remain visible in the existing lookup phase.
/// A window resolved by staging or committed membership starts no native scan.
#[derive(Clone, Debug, Default, Serialize)]
pub struct BodyLookupObservation {
    pub completed_lookups: u64,
    /// Original requests/wire bytes covered by successful lookup windows.
    /// Re-probes count again; these are not unique writes or publication ACKs.
    pub lookup_requests: u64,
    pub lookup_payload_bytes: u64,
    pub last_lookup_requests: u64,
    pub last_lookup_payload_bytes: u64,
    /// Attempts releasing prefetched bodies before a head-only lookup. The
    /// replacement lookup can still fail admission; this is not a success count.
    pub lookahead_admission_fallbacks: u64,
    /// Successor windows whose native lookup was observed active while the
    /// current native lookup was also active. Not a count of physical I/O.
    pub history_query_overlaps: u64,
    pub prefetched_lookups_started: u64,
    pub prefetched_lookups_completed: u64,
    /// Completed speculative lookups accepted by ordered apply at least once.
    pub prefetched_lookups_reused: u64,
    /// Speculative results rejected by the existing captured-base check.
    pub prefetched_lookups_stale: u64,
    /// Prepared speculative queries dropped before completion, including
    /// admission pressure and owner cancellation. Their WAL positions remain.
    pub prefetched_lookups_cancelled: u64,
    pub prefetch: BodyPrefetchObservation,
    pub requested_keys: u64,
    /// Keys resolved from the same captured active/frozen view, without history I/O.
    pub staged_keys: u64,
    /// Committed positive membership reused without native history I/O.
    pub cached_keys: u64,
    /// Exact distinct keys per successful window, after composite matching.
    /// Re-probes count again; these are not applied or globally unique rows.
    pub history_hits: u64,
    pub history_misses: u64,
    pub scans_started: u64,
    /// Returned key columns, including candidates removed by composite matching.
    /// These are not physical storage bytes or all scanned source rows.
    pub projected_rows: u64,
    pub projected_array_bytes: u64,
    pub elapsed_us: u64,
    pub max_elapsed_us: u64,
    pub first_result_us: LookupMetricObservation,
    pub parallel_scan_queries: u64,
    pub last_scan_partitions: usize,
    /// Native fragment task producers, not physical I/O or decoded batches.
    pub last_fragment_streams_peak: Option<usize>,
    pub max_fragment_streams_peak: usize,
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
    pub(super) fn record_window(
        &mut self,
        value: &crate::turn_ingestion::TurnLookupObservation,
        requests: usize,
        bytes: u64,
    ) {
        self.record(value);
        self.lookup_requests = self.lookup_requests.saturating_add(requests as u64);
        self.lookup_payload_bytes = self.lookup_payload_bytes.saturating_add(bytes);
        self.last_lookup_requests = requests as u64;
        self.last_lookup_payload_bytes = bytes;
    }

    pub(super) fn record(&mut self, value: &crate::turn_ingestion::TurnLookupObservation) {
        self.completed_lookups = self.completed_lookups.saturating_add(1);
        self.requested_keys = self
            .requested_keys
            .saturating_add(value.requested_keys as u64);
        self.staged_keys = self.staged_keys.saturating_add(value.staged_keys as u64);
        self.cached_keys = self.cached_keys.saturating_add(value.cached_keys as u64);
        self.history_hits = self.history_hits.saturating_add(value.history_hits as u64);
        self.history_misses = self
            .history_misses
            .saturating_add(value.history_misses as u64);
        self.scans_started = self.scans_started.saturating_add(value.scans_started);
        self.projected_rows = self.projected_rows.saturating_add(value.projected_rows);
        self.projected_array_bytes = self
            .projected_array_bytes
            .saturating_add(value.projected_array_bytes);
        self.elapsed_us = self.elapsed_us.saturating_add(value.elapsed_us);
        self.max_elapsed_us = self.max_elapsed_us.max(value.elapsed_us);
        self.first_result_us.record(value.first_result_us);
        self.parallel_scan_queries = self
            .parallel_scan_queries
            .saturating_add(u64::from(value.scan_partitions > 1));
        self.last_scan_partitions = value.scan_partitions;
        self.last_fragment_streams_peak = value.fragment_streams_peak;
        self.max_fragment_streams_peak = self
            .max_fragment_streams_peak
            .max(value.fragment_streams_peak.unwrap_or(0));
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
