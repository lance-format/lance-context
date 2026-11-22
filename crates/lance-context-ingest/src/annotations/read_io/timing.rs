//! Elapsed API timing, not server latency. Pending includes executor delay;
//! unpolled includes caller work/backpressure. Concurrent sums are not wall time.
use super::*;
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
pub(super) struct Elapsed {
    samples: AtomicU64,
    total_us: AtomicU64,
    max_us: AtomicU64,
}

#[derive(Debug, Serialize)]
pub(super) struct ElapsedObservation {
    pub(super) samples: u64,
    pub(super) total_us: u64,
    pub(super) max_us: u64,
}

impl Elapsed {
    pub(super) fn record(&self, duration: Duration) {
        let us = duration.as_micros().min(u64::MAX as u128) as u64;
        self.total_us.fetch_add(us, Ordering::Relaxed);
        self.max_us.fetch_max(us, Ordering::Relaxed);
        self.samples.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn observation(&self) -> ElapsedObservation {
        ElapsedObservation {
            samples: self.samples.load(Ordering::Relaxed),
            total_us: self.total_us.load(Ordering::Relaxed),
            max_us: self.max_us.load(Ordering::Relaxed),
        }
    }
}

pub(super) struct BodyTiming {
    last: Instant,
    pending: bool,
    recorded: bool,
    unpolled: Duration,
    waiting: Duration,
    polling: Duration,
}

impl BodyTiming {
    pub(super) fn new(now: Instant) -> Self {
        Self {
            last: now,
            pending: false,
            recorded: false,
            unpolled: Duration::ZERO,
            waiting: Duration::ZERO,
            polling: Duration::ZERO,
        }
    }

    pub(super) fn before_poll(&mut self, now: Instant) {
        if self.recorded {
            return;
        }
        let elapsed = now.duration_since(self.last);
        if self.pending {
            self.waiting += elapsed;
        } else {
            self.unpolled += elapsed;
        }
        self.last = now;
    }

    pub(super) fn after_poll(&mut self, now: Instant, pending: bool) {
        if !self.recorded {
            self.polling += now.duration_since(self.last);
            self.last = now;
            self.pending = pending;
        }
    }

    pub(super) fn finish(&mut self, now: Instant, counts: &Counts) {
        if self.recorded {
            return;
        }
        self.before_poll(now);
        counts.body_unpolled.record(self.unpolled);
        counts.body_pending.record(self.waiting);
        counts.body_poll.record(self.polling);
        self.recorded = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_delay_pending_delay_and_poll_work_are_disjoint() {
        let counts = Counts::default();
        let start = Instant::now();
        let at = |ms| start + Duration::from_millis(ms);
        let mut timing = BodyTiming::new(start);
        // Headers returned at 0; caller first requests the body at 20.
        timing.before_poll(at(20));
        timing.after_poll(at(22), true);
        // Pending includes both source wait and time until the next poll.
        timing.before_poll(at(52));
        timing.after_poll(at(55), false);
        // One chunk was delivered; caller waits 40ms before dropping it.
        timing.finish(at(95), &counts);
        timing.finish(at(195), &counts);
        assert_eq!(counts.body_unpolled.observation().total_us, 60_000);
        assert_eq!(counts.body_pending.observation().total_us, 30_000);
        assert_eq!(counts.body_poll.observation().total_us, 5_000);
        assert_eq!(counts.body_pending.observation().samples, 1);
    }

    #[test]
    fn entirely_unpolled_and_dropped_pending_bodies_stay_distinct() {
        let counts = Counts::default();
        let start = Instant::now();
        let at = |ms| start + Duration::from_millis(ms);
        let mut unpolled = BodyTiming::new(start);
        unpolled.finish(at(80), &counts);
        let mut pending = BodyTiming::new(start);
        pending.before_poll(start);
        pending.after_poll(start, true);
        pending.finish(at(70), &counts);
        assert_eq!(counts.body_unpolled.observation().total_us, 80_000);
        assert_eq!(counts.body_unpolled.observation().max_us, 80_000);
        assert_eq!(counts.body_pending.observation().total_us, 70_000);
        assert_eq!(counts.body_pending.observation().samples, 2);
        assert_eq!(counts.body_poll.observation().total_us, 0);
    }
}
