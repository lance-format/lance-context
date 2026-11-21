//! Opt-in observations of the existing scan, without an extra poll or query.
use super::StatusMatches;
use lance::dataset::scanner::{ExecutionSummaryCounts, Scanner};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

pub(super) struct ScanObservation {
    started: Instant,
    milestones: Mutex<(Option<u64>, Option<u64>)>,
    native: Arc<Mutex<Option<ExecutionSummaryCounts>>>,
}

impl ScanObservation {
    pub(super) fn new() -> Self {
        Self {
            started: Instant::now(),
            milestones: Mutex::new((None, None)),
            native: Arc::default(),
        }
    }

    fn elapsed_us(&self) -> u64 {
        self.started.elapsed().as_micros().min(u64::MAX as u128) as u64
    }

    pub(super) fn attach(&self, scan: &mut Scanner) {
        self.milestones.lock().unwrap().0 = Some(self.elapsed_us());
        let native = self.native.clone();
        scan.scan_stats_callback(Arc::new(move |summary| {
            *native.lock().unwrap() = Some(summary.clone());
        }));
    }

    pub(super) fn opened(&self) {
        self.milestones.lock().unwrap().1 = Some(self.elapsed_us());
    }

    pub(super) fn snapshot(&self, result: &lance::Result<StatusMatches>) -> serde_json::Value {
        let elapsed_us = self.elapsed_us();
        let (before_open_us, after_open_us) = *self.milestones.lock().unwrap();
        let native = self.native.lock().unwrap();
        let native = native.as_ref().map(|s| {
            serde_json::json!({
                "bytes_read": s.bytes_read, "iops": s.iops, "requests": s.requests,
                "indices_loaded": s.indices_loaded, "parts_loaded": s.parts_loaded,
                "index_comparisons": s.index_comparisons,
                "index_cache_hits": s.index_cache_hits(),
                "index_cache_misses": s.index_cache_misses(),
                "operator_times_ns": s.all_times,
            })
        });
        let matches = result.as_ref().ok();
        serde_json::json!({
            "outcome": if result.is_ok() { "success" } else { "error" },
            "elapsed_us": elapsed_us,
            "before_open_elapsed_us": before_open_us,
            "after_open_elapsed_us": after_open_us,
            "base_version": matches.map(|m| m.base_version),
            "materialized_through": matches.map(|m| m.materialized_through),
            "indexed_fragments": matches.map(|m| m.indexed_fragments),
            "unindexed_fragments": matches.map(|m| m.unindexed_fragments),
            "unindexed_physical_rows": matches.and_then(|m| m.unindexed_physical_rows),
            "returned_ids": matches.map(|m| m.ids.len()),
            "truncated": matches.map(|m| m.truncated),
            // Lance reports its summary at EOF. A bounded early return need
            // not reach EOF: null means unavailable, never zero work.
            "native_at_eof": native,
        })
    }

    pub(super) fn emit(&self, result: &lance::Result<StatusMatches>) {
        let data = self.snapshot(result);
        tracing::debug!(target: "lance_context::annotation_status_scan", observation = %data);
    }
}
