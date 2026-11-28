//! Lock-free producer counters; `snapshot` derives rates since open.
use std::{
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    time::{Duration, Instant},
};

use serde::Serialize;

pub struct Metrics {
    opened: Instant,
    buffer_ram_bytes: AtomicU64,
    buffer_spill_bytes: AtomicU64,
    turns_buffered: AtomicU64,
    spilled_entries: AtomicU64,
    spilled_bytes_total: AtomicU64,
    spill_errors: AtomicU64,
    backpressure_waits: AtomicU64,
    backpressure_nanos: AtomicU64,
    lane_idle_nanos: AtomicU64,
    records: AtomicU64,
    record_bytes: AtomicU64,
    record_rows: AtomicU64,
    commit_nanos: AtomicU64,
    commit_max_nanos: AtomicU64,
    retries: AtomicU64,
    oversize_turns: AtomicU64,
    oversize_slices: AtomicU64,
    applied_waits: AtomicU64,
    applied_wait_failures: AtomicU64,
    applied_wait_nanos: AtomicU64,
    dropped_known: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            opened: Instant::now(),
            buffer_ram_bytes: AtomicU64::new(0),
            buffer_spill_bytes: AtomicU64::new(0),
            turns_buffered: AtomicU64::new(0),
            spilled_entries: AtomicU64::new(0),
            spilled_bytes_total: AtomicU64::new(0),
            spill_errors: AtomicU64::new(0),
            backpressure_waits: AtomicU64::new(0),
            backpressure_nanos: AtomicU64::new(0),
            lane_idle_nanos: AtomicU64::new(0),
            records: AtomicU64::new(0),
            record_bytes: AtomicU64::new(0),
            record_rows: AtomicU64::new(0),
            commit_nanos: AtomicU64::new(0),
            commit_max_nanos: AtomicU64::new(0),
            retries: AtomicU64::new(0),
            oversize_turns: AtomicU64::new(0),
            oversize_slices: AtomicU64::new(0),
            applied_waits: AtomicU64::new(0),
            applied_wait_failures: AtomicU64::new(0),
            applied_wait_nanos: AtomicU64::new(0),
            dropped_known: AtomicU64::new(0),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricsSnapshot {
    pub buffer_ram_bytes: u64,
    pub buffer_spill_bytes: u64,
    pub turns_buffered: u64,
    pub spilled_entries: u64,
    pub spilled_bytes_total: u64,
    pub spill_errors: u64,
    pub backpressure_waits: u64,
    pub backpressure_seconds: f64,
    /// Summed over lanes: time a lane waited for turns of its shard.
    pub lane_idle_seconds: f64,
    pub records: u64,
    pub records_per_second: f64,
    pub bytes_per_record: f64,
    pub rows_per_record: f64,
    pub commit_ack_seconds: f64,
    pub commit_ack_max_seconds: f64,
    pub retries: u64,
    pub oversize_turns: u64,
    pub oversize_slices: u64,
    pub applied_waits: u64,
    pub applied_wait_failures: u64,
    pub applied_wait_seconds: f64,
    /// Rows dropped before buffering because the key was already known.
    pub dropped_known_rows: u64,
}

pub(super) struct Backpressure<'a>(&'a Metrics, Instant);

impl Drop for Backpressure<'_> {
    fn drop(&mut self) {
        self.0
            .backpressure_nanos
            .fetch_add(nanos(self.1.elapsed()), Relaxed);
    }
}

fn nanos(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

impl Metrics {
    pub(super) fn buffered(&self, ram: usize, spill: u64, turns: usize) {
        self.buffer_ram_bytes.store(ram as u64, Relaxed);
        self.buffer_spill_bytes.store(spill, Relaxed);
        self.turns_buffered.store(turns as u64, Relaxed);
    }
    pub(super) fn spilled(&self, bytes: u64) {
        self.spilled_entries.fetch_add(1, Relaxed);
        self.spilled_bytes_total.fetch_add(bytes, Relaxed);
    }
    pub(super) fn spill_error(&self) {
        self.spill_errors.fetch_add(1, Relaxed);
    }
    pub(super) fn backpressure(&self) -> Backpressure<'_> {
        self.backpressure_waits.fetch_add(1, Relaxed);
        Backpressure(self, Instant::now())
    }
    pub(super) fn lane_idle(&self, d: Duration) {
        self.lane_idle_nanos.fetch_add(nanos(d), Relaxed);
    }
    pub(super) fn committed(&self, d: Duration, bytes: usize, rows: usize) {
        self.records.fetch_add(1, Relaxed);
        self.record_bytes.fetch_add(bytes as u64, Relaxed);
        self.record_rows.fetch_add(rows as u64, Relaxed);
        self.commit_nanos.fetch_add(nanos(d), Relaxed);
        self.commit_max_nanos.fetch_max(nanos(d), Relaxed);
    }
    pub(super) fn retry(&self) {
        self.retries.fetch_add(1, Relaxed);
    }
    pub(super) fn oversize_turn(&self) {
        self.oversize_turns.fetch_add(1, Relaxed);
    }
    pub(super) fn oversize_slice(&self) {
        self.oversize_slices.fetch_add(1, Relaxed);
    }
    pub(super) fn applied_wait(&self, d: Duration, ok: bool) {
        self.applied_waits.fetch_add(1, Relaxed);
        self.applied_wait_failures
            .fetch_add(u64::from(!ok), Relaxed);
        self.applied_wait_nanos.fetch_add(nanos(d), Relaxed);
    }
    pub(super) fn dropped_known(&self, rows: usize) {
        self.dropped_known.fetch_add(rows as u64, Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let secs = |a: &AtomicU64| a.load(Relaxed) as f64 / 1e9;
        let records = self.records.load(Relaxed);
        let per = |a: &AtomicU64| {
            if records == 0 {
                0.0
            } else {
                a.load(Relaxed) as f64 / records as f64
            }
        };
        let elapsed = self.opened.elapsed().as_secs_f64().max(1e-9);
        MetricsSnapshot {
            buffer_ram_bytes: self.buffer_ram_bytes.load(Relaxed),
            buffer_spill_bytes: self.buffer_spill_bytes.load(Relaxed),
            turns_buffered: self.turns_buffered.load(Relaxed),
            spilled_entries: self.spilled_entries.load(Relaxed),
            spilled_bytes_total: self.spilled_bytes_total.load(Relaxed),
            spill_errors: self.spill_errors.load(Relaxed),
            backpressure_waits: self.backpressure_waits.load(Relaxed),
            backpressure_seconds: secs(&self.backpressure_nanos),
            lane_idle_seconds: secs(&self.lane_idle_nanos),
            records,
            records_per_second: records as f64 / elapsed,
            bytes_per_record: per(&self.record_bytes),
            rows_per_record: per(&self.record_rows),
            commit_ack_seconds: secs(&self.commit_nanos),
            commit_ack_max_seconds: secs(&self.commit_max_nanos),
            retries: self.retries.load(Relaxed),
            oversize_turns: self.oversize_turns.load(Relaxed),
            oversize_slices: self.oversize_slices.load(Relaxed),
            applied_waits: self.applied_waits.load(Relaxed),
            applied_wait_failures: self.applied_wait_failures.load(Relaxed),
            applied_wait_seconds: secs(&self.applied_wait_nanos),
            dropped_known_rows: self.dropped_known.load(Relaxed),
        }
    }
}
