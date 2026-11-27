use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use tokio::time::Instant;

/// Per-wrapper counters, shared by clones. Snapshots are not atomic across
/// fields. Durations sum completed or cancelled intervals across concurrent
/// calls; a currently running interval is included when it ends.
#[derive(Debug, Serialize)]
pub struct Observation {
    pub active_calls: u64,
    pub backing_off: u64,
    /// Origin-provider invocations, including the first; not inner HTTP attempts.
    pub provider_rounds: u64,
    /// Origin-provider invocations after a transient failure and backoff.
    pub retry_rounds: u64,
    /// Transient failures which started backoff; cancellation may prevent retry.
    pub retry_decisions: u64,
    pub provider_elapsed_micros: u64,
    pub backoff_elapsed_micros: u64,
    /// Successful calls that encountered at least one transient failure.
    pub recovered_calls: u64,
    /// Errors returned without retry, including unknown or external failures.
    pub failed_calls: u64,
    pub cancelled_calls: u64,
}

#[derive(Debug, Default)]
pub(super) struct Counters {
    active_calls: AtomicU64,
    backing_off: AtomicU64,
    provider_rounds: AtomicU64,
    retry_rounds: AtomicU64,
    retry_decisions: AtomicU64,
    provider_elapsed_micros: AtomicU64,
    backoff_elapsed_micros: AtomicU64,
    recovered_calls: AtomicU64,
    failed_calls: AtomicU64,
    cancelled_calls: AtomicU64,
}

impl Counters {
    pub(super) fn observation(&self) -> Observation {
        Observation {
            active_calls: self.active_calls.load(Relaxed),
            backing_off: self.backing_off.load(Relaxed),
            provider_rounds: self.provider_rounds.load(Relaxed),
            retry_rounds: self.retry_rounds.load(Relaxed),
            retry_decisions: self.retry_decisions.load(Relaxed),
            provider_elapsed_micros: self.provider_elapsed_micros.load(Relaxed),
            backoff_elapsed_micros: self.backoff_elapsed_micros.load(Relaxed),
            recovered_calls: self.recovered_calls.load(Relaxed),
            failed_calls: self.failed_calls.load(Relaxed),
            cancelled_calls: self.cancelled_calls.load(Relaxed),
        }
    }

    pub(super) fn call(&self) -> Call<'_> {
        self.active_calls.fetch_add(1, Relaxed);
        Call {
            counters: self,
            retried: false,
            finished: false,
        }
    }

    pub(super) fn phase(&self, backoff: bool) -> Phase<'_> {
        if backoff {
            self.backing_off.fetch_add(1, Relaxed);
        }
        Phase {
            counters: self,
            started: Instant::now(),
            backoff,
        }
    }
}

pub(super) struct Call<'a> {
    counters: &'a Counters,
    retried: bool,
    finished: bool,
}

impl Call<'_> {
    pub(super) fn provider_round(&self) -> Phase<'_> {
        self.counters.provider_rounds.fetch_add(1, Relaxed);
        if self.retried {
            self.counters.retry_rounds.fetch_add(1, Relaxed);
        }
        self.counters.phase(false)
    }

    pub(super) fn retry(&mut self) {
        self.retried = true;
        self.counters.retry_decisions.fetch_add(1, Relaxed);
    }

    pub(super) fn finish(&mut self, success: bool) {
        self.finished = true;
        if !success {
            self.counters.failed_calls.fetch_add(1, Relaxed);
        } else if self.retried {
            self.counters.recovered_calls.fetch_add(1, Relaxed);
        }
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.counters.cancelled_calls.fetch_add(1, Relaxed);
        }
        self.counters.active_calls.fetch_sub(1, Relaxed);
    }
}

pub(super) struct Phase<'a> {
    counters: &'a Counters,
    started: Instant,
    backoff: bool,
}

impl Drop for Phase<'_> {
    fn drop(&mut self) {
        let micros = self.started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        if self.backoff {
            self.counters
                .backoff_elapsed_micros
                .fetch_add(micros, Relaxed);
            self.counters.backing_off.fetch_sub(1, Relaxed);
        } else {
            self.counters
                .provider_elapsed_micros
                .fetch_add(micros, Relaxed);
        }
    }
}
