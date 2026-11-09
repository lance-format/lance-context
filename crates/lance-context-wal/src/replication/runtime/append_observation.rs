//! Completed append intervals on one group worker. No payloads or request IDs.
use super::Outcome;
use crate::replication::Applied;
use anyhow::Result;
use serde::Serialize;
use std::time::{Duration, Instant};

/// Same-request, nonoverlapping wall intervals for completed appends only.
/// Concurrent requests overlap. Maxima are not percentiles and cannot be added.
/// Excludes RPC transport, runtime credit admission, pre-enqueue payload
/// validation, reply delivery, and owner staging application. The RPC execution
/// timing also includes pre-enqueue validation. Queued/in-flight work dropped on shutdown is absent;
/// caller cancellation alone does not remove a worker's eventual completion.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AppendIntervals {
    pub completed: u64,
    /// Enqueue attempt to dequeue, including time blocked behind a durable poll.
    pub queue_micros: u64,
    pub queue_max_micros: u64,
    /// Worker-side scope checks, disk reservation, and local Raft proposal.
    /// Raw submissions also validate the payload here; prevalidated RPCs do not.
    pub proposal_micros: u64,
    pub proposal_max_micros: u64,
    /// Proposal return to worker completion: scheduling, local persistence,
    /// replication, quorum commit and WAL application; NOT isolated disk time.
    pub completion_micros: u64,
    pub completion_max_micros: u64,
}

/// Group-local counters sampled under the same observation lock. An accepted
/// sample is the worker result, not evidence that the caller received its ACK.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AppendObservation {
    pub accepted: AppendIntervals,
    pub committed_other: AppendIntervals,
    pub unknown: AppendIntervals,
    pub proposal_error: AppendIntervals,
}

pub(super) struct AppendTiming {
    pub queued: Instant,
    pub dequeued: Instant,
    pub proposed: Instant,
}

fn record(total: &mut u64, maximum: &mut u64, elapsed: Duration) {
    let micros = elapsed.as_micros().min(u64::MAX as u128) as u64;
    *total = total.saturating_add(micros);
    *maximum = (*maximum).max(micros);
}

impl AppendObservation {
    pub(super) fn record(
        &mut self,
        timing: AppendTiming,
        finished: Instant,
        result: &Result<Outcome>,
    ) {
        let intervals = match result {
            Ok(Outcome::Committed(Applied::Accepted { .. })) => &mut self.accepted,
            Ok(Outcome::Committed(_)) => &mut self.committed_other,
            Ok(Outcome::Unknown) => &mut self.unknown,
            Err(_) => &mut self.proposal_error,
        };
        intervals.completed = intervals.completed.saturating_add(1);
        record(
            &mut intervals.queue_micros,
            &mut intervals.queue_max_micros,
            timing.dequeued.duration_since(timing.queued),
        );
        record(
            &mut intervals.proposal_micros,
            &mut intervals.proposal_max_micros,
            timing.proposed.duration_since(timing.dequeued),
        );
        record(
            &mut intervals.completion_micros,
            &mut intervals.completion_max_micros,
            finished.duration_since(timing.proposed),
        );
    }
}
