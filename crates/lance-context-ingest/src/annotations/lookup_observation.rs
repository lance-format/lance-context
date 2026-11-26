//! Completed immutable-key lookup windows for one service lifetime.
use super::*;

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
    /// Number of completed lookups with a native scan summary. The following
    /// totals cover only these summaries; zero summaries means unavailable.
    pub native_summaries: u64,
    pub lance_bytes_read: u64,
    pub lance_requests: u64,
    pub lance_indices_loaded: u64,
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
