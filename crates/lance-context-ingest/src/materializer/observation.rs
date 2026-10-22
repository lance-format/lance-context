//! Bounded cumulative phase observations, independent of publication authority.
use serde::Serialize;
use std::{
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Phase {
    Claim,
    Acquire,
    State,
    Prepare,
    Commit,
    Retire,
    Fetch,
    DecodeValidate,
    BlobWrite,
    Admission,
}
const NAMES: [&str; 10] = [
    "claim",
    "acquire",
    "state",
    "prepare",
    "commit",
    "retire",
    "fetch",
    "decode_validate",
    "blob_write",
    "admission",
];

#[derive(Default)]
struct Counter {
    started: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    elapsed_micros: AtomicU64,
}

#[derive(Default)]
pub struct Counters([Counter; 10]);

/// Each field is sampled independently. Parent phases contain child phases;
/// concurrent phase times overlap and must not be summed into elapsed or CPU time.
#[derive(Clone, Debug, Serialize)]
pub struct Observation {
    pub phase: &'static str,
    pub started: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    /// Includes failed/cancelled work, but not time still in flight.
    pub elapsed_micros: u64,
}

impl Counters {
    pub fn observation(&self) -> [Observation; 10] {
        std::array::from_fn(|i| {
            let c = &self.0[i];
            Observation {
                phase: NAMES[i],
                started: c.started.load(Ordering::Relaxed),
                completed: c.completed.load(Ordering::Relaxed),
                failed: c.failed.load(Ordering::Relaxed),
                cancelled: c.cancelled.load(Ordering::Relaxed),
                elapsed_micros: c.elapsed_micros.load(Ordering::Relaxed),
            }
        })
    }

    pub async fn measure<T, E>(
        self: &Arc<Self>,
        phase: Phase,
        future: impl Future<Output = Result<T, E>>,
    ) -> Result<T, E> {
        let mut guard = self.enter(phase);
        let result = future.await;
        guard.success = Some(result.is_ok());
        result
    }

    pub(super) fn blocking<T, E>(
        self: &Arc<Self>,
        phase: Phase,
        work: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let mut guard = self.enter(phase);
        let result = work();
        guard.success = Some(result.is_ok());
        result
    }

    fn enter(self: &Arc<Self>, phase: Phase) -> Guard {
        self.0[phase as usize]
            .started
            .fetch_add(1, Ordering::Relaxed);
        Guard {
            counters: self.clone(),
            phase,
            start: Instant::now(),
            success: None,
        }
    }
}

struct Guard {
    counters: Arc<Counters>,
    phase: Phase,
    start: Instant,
    success: Option<bool>,
}
impl Drop for Guard {
    fn drop(&mut self) {
        let c = &self.counters.0[self.phase as usize];
        c.elapsed_micros.fetch_add(
            self.start.elapsed().as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        match self.success {
            Some(success) => {
                if !success {
                    c.failed.fetch_add(1, Ordering::Relaxed);
                }
                c.completed.fetch_add(1, Ordering::Relaxed);
            }
            None => {
                c.cancelled.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancelled_observation_is_not_successful_work() {
        let counters = Arc::new(Counters::default());
        let owned = counters.clone();
        let (entered, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            owned
                .measure(Phase::Fetch, async {
                    entered.send(()).unwrap();
                    std::future::pending::<Result<(), ()>>().await
                })
                .await
        });
        ready.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let cancelled = &counters.observation()[Phase::Fetch as usize];
        assert_eq!(cancelled.started, 1);
        assert_eq!(cancelled.cancelled, 1);
        assert_eq!(cancelled.completed, 0);
        assert_eq!(cancelled.failed, 0);
        assert!(counters
            .measure(Phase::Fetch, async { Err::<(), _>(()) })
            .await
            .is_err());
        let failed = &counters.observation()[Phase::Fetch as usize];
        assert_eq!(failed.started, 2);
        assert_eq!(failed.completed, 1);
        assert_eq!(failed.failed, 1);
        assert_eq!(failed.cancelled, 1);
    }
}
