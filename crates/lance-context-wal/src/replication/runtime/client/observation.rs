//! Fixed-cardinality append execution diagnostics, independent of wire replies.
//! Starts after the complete RPC body is received. Excludes TLS/frame errors,
//! time waiting for transport memory, and reply delivery. A cancelled execution
//! may still commit; none of these counters changes the caller's retry contract.
use super::{Bytes, Outcome, Reply};
use crate::{
    admission::DiskBudgetExhausted,
    replication::{runtime::SubmissionFailure, Applied, ProposalRejection},
};
use anyhow::Result;
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
};

#[derive(Clone, Copy)]
enum Kind {
    Started,
    Accepted,
    Redirect,
    CommittedCapacity,
    SequenceRejected,
    ProducerFenced,
    LeadershipUnknown,
    DiskBudget,
    RequestCredits,
    ByteCredits,
    QueueUnavailable,
    WorkerStopped,
    NotLeader,
    PendingCapacity,
    ExecutionOther,
    Cancelled,
    UnexpectedReply,
}
const LABELS: [&str; 17] = [
    "started",
    "accepted",
    "redirect",
    "committed_capacity",
    "sequence_rejected",
    "producer_fenced",
    "leadership_unknown",
    "disk_budget",
    "request_credits",
    "byte_credits",
    "queue_unavailable",
    "worker_stopped",
    "not_leader",
    "pending_capacity",
    "execution_other",
    "cancelled",
    "unexpected_reply",
];

#[derive(Default)]
pub(in crate::replication::runtime) struct Counters {
    outcomes: [AtomicU64; LABELS.len()],
    last_disk_refusal: Mutex<Option<DiskBudgetExhausted>>,
    pub(super) execution: super::timing::Counter,
    pub(super) receive_admission: super::timing::Counter,
    pub(super) receive_body: super::timing::Counter,
}
impl Counters {
    fn increment(&self, kind: Kind) {
        super::super::increment_rejection(&self.outcomes[kind as usize]);
    }
    pub(in crate::replication::runtime) fn snapshot(&self) -> BTreeMap<&'static str, u64> {
        LABELS
            .iter()
            .zip(&self.outcomes)
            .map(|(&label, counter)| (label, counter.load(Ordering::Relaxed)))
            .collect()
    }
    pub(super) fn start(&self) -> Guard<'_> {
        self.increment(Kind::Started);
        Guard {
            counters: self,
            finished: false,
        }
    }
    pub(super) fn last_disk_refusal(&self) -> Option<DiskBudgetExhausted> {
        self.last_disk_refusal.lock().unwrap().clone()
    }
}
pub(super) struct Guard<'a> {
    counters: &'a Counters,
    finished: bool,
}
impl Guard<'_> {
    pub(super) fn finish(mut self, result: &Result<(Reply, Bytes)>) {
        let kind = match result {
            Ok((Reply::Redirect { .. }, _)) => Kind::Redirect,
            Ok((Reply::Commit(Outcome::Unknown), _)) => Kind::LeadershipUnknown,
            Ok((Reply::Commit(Outcome::Committed(applied)), _)) => match applied {
                Applied::Accepted { .. } => Kind::Accepted,
                Applied::Capacity => Kind::CommittedCapacity,
                Applied::Rejected { .. } => Kind::SequenceRejected,
                Applied::ProducerFenced => Kind::ProducerFenced,
                _ => Kind::UnexpectedReply,
            },
            Ok(_) => Kind::UnexpectedReply,
            Err(error) => {
                if let Some(decision) = error.downcast_ref::<DiskBudgetExhausted>() {
                    // Copy the failed decision before its wire reply discards
                    // the byte values. Successful appends never take this lock.
                    *self.counters.last_disk_refusal.lock().unwrap() = Some(decision.clone());
                }
                classify(error)
            }
        };
        self.counters.increment(kind);
        self.finished = true;
    }
}
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.counters.increment(Kind::Cancelled);
        }
    }
}
fn classify(error: &anyhow::Error) -> Kind {
    if error.is::<DiskBudgetExhausted>() {
        Kind::DiskBudget
    } else if let Some(error) = error.downcast_ref::<SubmissionFailure>() {
        match error {
            SubmissionFailure::RequestCredits => Kind::RequestCredits,
            SubmissionFailure::ByteCredits => Kind::ByteCredits,
            SubmissionFailure::QueueUnavailable => Kind::QueueUnavailable,
            SubmissionFailure::WorkerStopped => Kind::WorkerStopped,
        }
    } else if let Some(error) = error.downcast_ref::<ProposalRejection>() {
        match error {
            ProposalRejection::NotLeader => Kind::NotLeader,
            ProposalRejection::PendingCapacity => Kind::PendingCapacity,
        }
    } else {
        Kind::ExecutionOther
    }
}
