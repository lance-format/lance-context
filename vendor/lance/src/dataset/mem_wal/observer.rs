// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Consumer-supplied sink for MemWAL write-path events.

use std::fmt::Debug;
use std::time::Duration;

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
    /// A WAL buffer flush landed in object storage. This is the latency a
    /// `durable_write` put waits on.
    fn on_wal_flush(&self, _duration: Duration, _bytes: usize) {}

    /// A frozen memtable became an SSTable. Orders of magnitude longer
    /// than a WAL flush.
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
