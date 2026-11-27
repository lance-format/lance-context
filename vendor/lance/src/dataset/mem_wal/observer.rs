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
    /// Begin one put phase. The returned token is private to this observer and
    /// pairs concurrent starts with their finish; never use it as a metric label.
    fn on_write_phase_start(&self, _phase: WritePhase) -> u64 {
        0
    }

    /// Finish one put phase, including errors and locally abandoned futures.
    /// Abandonment does not establish whether the write persisted remotely.
    fn on_write_phase_finish(
        &self,
        _token: u64,
        _phase: WritePhase,
        _duration: Duration,
        _outcome: FlushPhaseOutcome,
    ) {
    }

    /// The cursors this put is waiting for. Both can advance concurrently;
    /// these are in-memory observations, not additional storage requests.
    fn on_write_visibility_progress(&self, _token: u64, _progress: WriteVisibilityProgress) {}

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

/// Work on the MemTable put path. Background WAL and SSTable work have their
/// own observations; visibility waits can include both WAL and index progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WritePhase {
    WriterOpen,
    Admission,
    StateLock,
    Prepare,
    Insert,
    Visibility,
}

impl WritePhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WriterOpen => "writer_open",
            Self::Admission => "admission",
            Self::StateLock => "state_lock",
            Self::Prepare => "prepare",
            Self::Insert => "insert",
            Self::Visibility => "visibility",
        }
    }
}

/// Batch counts use two coordinate spaces: durability is writer-global,
/// indexing is local to this put's memtable. Compare each only with its target.
#[derive(Clone, Copy, Debug)]
pub struct WriteVisibilityProgress {
    pub durable: usize,
    pub target_durable: usize,
    pub indexed: usize,
    pub target_indexed: usize,
    pub requires_durable: bool,
}

pub(crate) struct WritePhaseGuard<'a> {
    observer: Option<&'a dyn WalObserver>,
    token: u64,
    phase: WritePhase,
    start: Option<Instant>,
    outcome: FlushPhaseOutcome,
}

impl<'a> WritePhaseGuard<'a> {
    pub(crate) fn new(observer: Option<&'a dyn WalObserver>, phase: WritePhase) -> Self {
        Self {
            observer,
            token: observer.map_or(0, |observer| observer.on_write_phase_start(phase)),
            phase,
            start: observer.map(|_| Instant::now()),
            outcome: FlushPhaseOutcome::Abandoned,
        }
    }

    pub(crate) fn finish<T, E>(&mut self, result: &Result<T, E>) {
        self.outcome = if result.is_ok() {
            FlushPhaseOutcome::Ok
        } else {
            FlushPhaseOutcome::Error
        };
    }

    pub(crate) fn progress(&self, progress: WriteVisibilityProgress) {
        if let Some(observer) = self.observer {
            observer.on_write_visibility_progress(self.token, progress);
        }
    }
}

impl Drop for WritePhaseGuard<'_> {
    fn drop(&mut self) {
        if let (Some(observer), Some(start)) = (self.observer, self.start) {
            observer.on_write_phase_finish(self.token, self.phase, start.elapsed(), self.outcome);
        }
    }
}

/// Observe one phase without changing its result, polling or cancellation behavior.
pub async fn observe_write_phase<T, E>(
    observer: Option<&dyn WalObserver>,
    phase: WritePhase,
    work: impl Future<Output = Result<T, E>>,
) -> Result<T, E> {
    let mut observation = WritePhaseGuard::new(observer, phase);
    let result = work.await;
    observation.finish(&result);
    result
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
    struct WriteObserver {
        next: std::sync::atomic::AtomicU64,
        active: Mutex<std::collections::BTreeSet<u64>>,
        finished: Mutex<Vec<FlushPhaseOutcome>>,
    }

    impl WalObserver for WriteObserver {
        fn on_write_phase_start(&self, _: WritePhase) -> u64 {
            let token = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            assert!(self.active.lock().unwrap().insert(token));
            token
        }

        fn on_write_phase_finish(
            &self,
            token: u64,
            _: WritePhase,
            _: Duration,
            outcome: FlushPhaseOutcome,
        ) {
            assert!(self.active.lock().unwrap().remove(&token));
            self.finished.lock().unwrap().push(outcome);
        }
    }

    #[tokio::test]
    async fn concurrent_write_phases_settle_independently_on_cancellation_and_error() {
        let observer = WriteObserver::default();
        let mut pending = Box::pin(observe_write_phase(
            Some(&observer),
            WritePhase::Admission,
            std::future::pending::<Result<(), ()>>(),
        ));
        assert!(futures::poll!(&mut pending).is_pending());
        let result = observe_write_phase(Some(&observer), WritePhase::Admission, async {
            Ok::<_, ()>(42)
        })
        .await;
        assert_eq!(result, Ok(42));
        assert_eq!(
            observer.active.lock().unwrap().len(),
            1,
            "another put finishing must not clear the pending put"
        );
        let result = observe_write_phase(Some(&observer), WritePhase::Prepare, async {
            Err::<(), _>("prepare failed")
        })
        .await;
        assert_eq!(result, Err("prepare failed"));
        drop(pending);
        assert!(observer.active.lock().unwrap().is_empty());
        assert!(matches!(
            observer.finished.lock().unwrap().as_slice(),
            [
                FlushPhaseOutcome::Ok,
                FlushPhaseOutcome::Error,
                FlushPhaseOutcome::Abandoned
            ]
        ));
    }

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
