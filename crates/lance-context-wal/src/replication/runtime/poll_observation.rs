//! Cached diagnostics only: never a storage barrier, quorum proof, or ACK.
use serde::Serialize;
use std::time::Duration;

fn micros(value: Duration) -> u64 {
    value.as_micros().min(u64::MAX as u128) as u64
}

/// Completed calls to the worker's durable Raft poll, including failures.
/// Excludes lane queueing, proposal validation, and response delivery. A poll
/// can process many requests, or no client request; these are not ACK timings.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PollObservation {
    pub completed: u64,
    pub failed: u64,
    pub elapsed_micros: u64,
    pub max_elapsed_micros: u64,
    pub engine: EngineWriteObservation,
}

/// Native thread-local write totals captured after the last successful poll.
/// Includes all engine work on this group worker, including local maintenance
/// outside poll, but excludes other threads and time waiting for wrapper locks.
/// Native write groups copy shared write/sync durations to participating writers:
/// summing across workers does not measure unique physical disk time or CPU time.
/// Native phases can also overlap (for example, rotation is within write).
/// These fields overlap with poll elapsed time and must not be added to it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct EngineWriteObservation {
    pub populate_micros: u64,
    pub group_wait_micros: u64,
    pub write_micros: u64,
    pub rotate_micros: u64,
    pub sync_micros: u64,
    pub apply_micros: u64,
}

impl PollObservation {
    pub(super) fn record(&mut self, elapsed: Duration, success: bool) {
        let elapsed = micros(elapsed);
        self.completed = self.completed.saturating_add(1);
        self.elapsed_micros = self.elapsed_micros.saturating_add(elapsed);
        self.max_elapsed_micros = self.max_elapsed_micros.max(elapsed);
        if success {
            let native = raft_engine::get_perf_context();
            self.engine = EngineWriteObservation {
                populate_micros: micros(native.log_populating_duration),
                group_wait_micros: micros(native.write_wait_duration),
                write_micros: micros(native.log_write_duration),
                rotate_micros: micros(native.log_rotate_duration),
                sync_micros: micros(native.log_sync_duration),
                apply_micros: micros(native.apply_duration),
            };
        } else {
            self.failed = self.failed.saturating_add(1);
            // A native write panic can interrupt restoration of its TLS perf
            // context. Keep the last successful snapshot; the worker stops on
            // this error and must not present partial native timings as totals.
        }
    }
}
