// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Consumer-supplied sink for MemWAL write-path events.

use std::fmt::Debug;
use std::future::Future;
use std::time::{Duration, Instant};

use lance_core::Error;

/// Sink for individual write-path events, supplied by the consumer via
/// [`ShardWriterConfig::observer`](super::write::ShardWriterConfig::observer).
///
/// Cumulative counts stay on
/// [`WriteStats`](super::write::WriteStats), which an embedder polls: a total
/// loses nothing to aggregation. A duration does — an average reconstructed
/// from a running total cannot show a tail — so each flush is reported here as
/// it completes and the consumer decides how to aggregate it.
///
/// Observers run inline on the flush task. Do the aggregation, not the export.
///
/// Every method defaults to a no-op, so adding an event is not a breaking
/// change for existing implementors.
pub trait WalObserver: Send + Sync + Debug {
    /// One actual flush phase started. Skipped work on a retry emits no event.
    fn on_memtable_flush_phase_start(&self, _phase: MemtableFlushPhase) {}

    /// A phase finished or its local future was abandoned. Abandonment does not
    /// establish whether an object-store operation completed remotely.
    fn on_memtable_flush_phase_finish(
        &self,
        _phase: MemtableFlushPhase,
        _duration: Duration,
        _outcome: FlushPhaseOutcome,
    ) {
    }

    /// A non-empty WAL buffer flush landed in object storage. Includes flush
    /// work but excludes its preceding queue; a durable put may also wait for
    /// admission and index visibility.
    fn on_wal_flush(&self, _duration: Duration, _bytes: usize) {}

    /// A frozen memtable became an SSTable. Total attempt work includes WAL
    /// and index waits, preparation and publication, but excludes queueing
    /// before the attempt and retry backoff.
    fn on_memtable_flush(&self, _duration: Duration, _rows: usize) {}

    /// A MemTable flush attempt failed. Called once per attempt, so a
    /// generation that keeps failing is reported on every retry.
    fn on_memtable_flush_failure(&self, _generation: u64, _error: &Error) {}

    /// Coarse durable checkpoints for one generation. `started` is only an
    /// attempt notification; `wal_durable` and `prepared` report completed work.
    /// Repeating a checkpoint on retry must not be interpreted as new progress.
    fn on_memtable_flush_stage(&self, _generation: u64, _stage: &'static str) {}

    /// A generation flush attempt failed (including background retries).
    fn on_memtable_flush_error(&self) {}

    /// An already-failed generation is being retried.
    fn on_memtable_flush_retry(&self) {}

    /// Current retained retry queue, including successors blocked behind its
    /// oldest failure. Bytes are resident payload/index bytes, not WAL bytes.
    /// A generation remains counted while its retry is in flight.
    fn on_memtable_flush_retry_state(&self, _generations: usize, _bytes: usize) {}
}

/// Non-overlapping work within a memtable flush attempt. Queueing before the
/// attempt and retry backoff are excluded. A retry may reuse durable/prepared
/// work and therefore emit only the phases it actually executes.
#[derive(Clone, Copy, Debug)]
pub enum MemtableFlushPhase {
    /// Wait for this frozen generation's WAL completion and receipt.
    WalWait,
    /// Wait for the frozen generation's in-memory indexes.
    IndexWait,
    /// Refresh the fence, write data and index files, and prepare metadata.
    Prepare,
    /// Publish the prepared generation in the shard manifest.
    Publish,
}

impl MemtableFlushPhase {
    /// Bounded label suitable for metrics; never a table or generation ID.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WalWait => "wal_wait",
            Self::IndexWait => "index_wait",
            Self::Prepare => "prepare",
            Self::Publish => "publish",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FlushPhaseOutcome {
    Ok,
    Error,
    Abandoned,
}

struct FlushPhaseGuard<'a> {
    observer: &'a dyn WalObserver,
    phase: MemtableFlushPhase,
    start: Instant,
    outcome: FlushPhaseOutcome,
}

impl Drop for FlushPhaseGuard<'_> {
    fn drop(&mut self) {
        self.observer.on_memtable_flush_phase_finish(
            self.phase,
            self.start.elapsed(),
            self.outcome,
        );
    }
}

pub(crate) async fn observe_flush_phase<T, E>(
    observer: Option<&dyn WalObserver>,
    phase: MemtableFlushPhase,
    work: impl Future<Output = Result<T, E>>,
) -> Result<T, E> {
    let mut guard = observer.map(|observer| {
        observer.on_memtable_flush_phase_start(phase);
        FlushPhaseGuard {
            observer,
            phase,
            start: Instant::now(),
            outcome: FlushPhaseOutcome::Abandoned,
        }
    });
    let result = work.await;
    if let Some(guard) = &mut guard {
        guard.outcome = if result.is_ok() {
            FlushPhaseOutcome::Ok
        } else {
            FlushPhaseOutcome::Error
        };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct Observer(Mutex<(usize, Vec<FlushPhaseOutcome>)>);

    impl WalObserver for Observer {
        fn on_memtable_flush_phase_start(&self, _: MemtableFlushPhase) {
            self.0.lock().unwrap().0 += 1;
        }
        fn on_memtable_flush_phase_finish(
            &self,
            _: MemtableFlushPhase,
            _: Duration,
            outcome: FlushPhaseOutcome,
        ) {
            let mut state = self.0.lock().unwrap();
            state.0 -= 1;
            state.1.push(outcome);
        }
    }

    #[tokio::test]
    async fn phase_observer_settles_success_error_and_abandoned_futures() {
        let observer = Observer::default();
        let result = observe_flush_phase(Some(&observer), MemtableFlushPhase::Prepare, async {
            Ok::<_, &str>(42)
        })
        .await;
        assert_eq!(result, Ok(42));
        let result = observe_flush_phase(Some(&observer), MemtableFlushPhase::Publish, async {
            Err::<(), _>("publication failed")
        })
        .await;
        assert_eq!(result, Err("publication failed"));
        let mut pending = Box::pin(observe_flush_phase(
            Some(&observer),
            MemtableFlushPhase::WalWait,
            std::future::pending::<Result<(), ()>>(),
        ));
        assert!(futures::poll!(&mut pending).is_pending());
        assert_eq!(observer.0.lock().unwrap().0, 1);
        drop(pending);
        let state = observer.0.lock().unwrap();
        assert_eq!(state.0, 0);
        assert!(matches!(
            state.1.as_slice(),
            [
                FlushPhaseOutcome::Ok,
                FlushPhaseOutcome::Error,
                FlushPhaseOutcome::Abandoned
            ]
        ));
    }
}
