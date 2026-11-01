//! Fixed-size elapsed-time counters. An interval ending is not an ACK: errors
//! and cancelled futures also finish intervals, while in-flight time is absent.
use serde::Serialize;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

#[derive(Default)]
pub(in crate::replication::runtime) struct Counter {
    started: AtomicU64,
    finished: AtomicU64,
    elapsed_micros: AtomicU64,
    max_elapsed_micros: AtomicU64,
}

/// Independently sampled counters, not an atomic snapshot. Concurrent intervals
/// overlap; their sum is neither CPU time nor a request latency percentile.
#[derive(Debug, Clone, Serialize)]
pub struct Observation {
    pub started: u64,
    pub finished: u64,
    pub elapsed_micros: u64,
    pub max_elapsed_micros: u64,
}

impl Counter {
    pub(in crate::replication::runtime) fn start(&self) -> Guard<'_> {
        self.started.fetch_add(1, Ordering::Relaxed);
        Guard {
            counter: self,
            start: Instant::now(),
        }
    }

    /// Record an already completed interval, such as residence in a queue.
    /// Pending intervals are absent from both counts for this kind of sample.
    pub(in crate::replication::runtime) fn record(&self, elapsed: Duration) {
        self.started.fetch_add(1, Ordering::Relaxed);
        self.finish(elapsed);
    }

    fn finish(&self, elapsed: Duration) {
        let micros = elapsed.as_micros().min(u64::MAX as u128) as u64;
        self.elapsed_micros.fetch_add(micros, Ordering::Relaxed);
        self.max_elapsed_micros.fetch_max(micros, Ordering::Relaxed);
        self.finished.fetch_add(1, Ordering::Relaxed);
    }

    pub(in crate::replication::runtime) fn snapshot(&self) -> Observation {
        Observation {
            started: self.started.load(Ordering::Relaxed),
            finished: self.finished.load(Ordering::Relaxed),
            elapsed_micros: self.elapsed_micros.load(Ordering::Relaxed),
            max_elapsed_micros: self.max_elapsed_micros.load(Ordering::Relaxed),
        }
    }
}

pub(in crate::replication::runtime) struct Guard<'a> {
    counter: &'a Counter,
    start: Instant,
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.counter.finish(self.start.elapsed());
    }
}
