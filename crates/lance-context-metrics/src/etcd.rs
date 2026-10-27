//! Client-call accounting. No request contents or dynamic identifiers become labels.
//!
//! Counts awaited client method calls, not wire attempts: transport retries and
//! watch subscription messages are not separate calls here. A dropped future
//! has an unknown remote outcome and must never be treated as a failed commit.

use std::future::Future;

/// The complete, static label vocabulary. Add coverage explicitly; never label
/// metrics with keys, targets, endpoint URLs, execution IDs or error messages.
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    MergeRead,
    MergeScan,
    MergeReserve,
    MergeStart,
    MergeTerminal,
    MergeRelease,
    MergeProgressInitial,
    MergeProgressChanged,
    MergeWorkProgress,
    MergeCommitRead,
    MergeCommitWrite,
    MergeFailureRead,
    MergeOther,
    MergeOwnerWatch,
    MergeProgressWatch,
    DemandRead,
    DemandPublish,
    DemandScan,
    TaskRead,
    TaskReadTxn,
    TaskScan,
    TaskEnqueue,
    TaskClaim,
    TaskFinish,
    TaskOther,
    DispatchRead,
    DispatchWrite,
    DispatchWatch,
    LeaseGrant,
    LeaseStreamOpen,
    LeaseRevoke,
}

impl Operation {
    fn label(self) -> &'static str {
        match self {
            Self::MergeRead => "merge_read",
            Self::MergeScan => "merge_scan",
            Self::MergeReserve => "merge_reserve",
            Self::MergeStart => "merge_start",
            Self::MergeTerminal => "merge_terminal",
            Self::MergeRelease => "merge_release",
            Self::MergeProgressInitial => "merge_progress_initial",
            Self::MergeProgressChanged => "merge_progress_changed",
            Self::MergeWorkProgress => "merge_work_progress",
            Self::MergeCommitRead => "merge_commit_read",
            Self::MergeCommitWrite => "merge_commit_write",
            Self::MergeFailureRead => "merge_failure_read",
            Self::MergeOther => "merge_other",
            Self::MergeOwnerWatch => "merge_owner_watch",
            Self::MergeProgressWatch => "merge_progress_watch",
            Self::DemandRead => "demand_read",
            Self::DemandPublish => "demand_publish",
            Self::DemandScan => "demand_scan",
            Self::TaskRead => "task_read",
            Self::TaskReadTxn => "task_read_txn",
            Self::TaskScan => "task_scan",
            Self::TaskEnqueue => "task_enqueue",
            Self::TaskClaim => "task_claim",
            Self::TaskFinish => "task_finish",
            Self::TaskOther => "task_other",
            Self::DispatchRead => "dispatch_read",
            Self::DispatchWrite => "dispatch_write",
            Self::DispatchWatch => "dispatch_watch",
            Self::LeaseGrant => "lease_grant",
            Self::LeaseStreamOpen => "lease_stream_open",
            Self::LeaseRevoke => "lease_revoke",
        }
    }
}

struct Attempt {
    operation: &'static str,
    finished: bool,
}

impl Attempt {
    fn start(operation: Operation) -> Self {
        let operation = operation.label();
        metrics::counter!("etcd_operation_attempts_total", "operation" => operation).increment(1);
        Self {
            operation,
            finished: false,
        }
    }

    fn finish(mut self, ok: bool) {
        self.finished = true;
        metrics::counter!("etcd_operation_completions_total",
            "operation" => self.operation, "outcome" => if ok { "ok" } else { "error" })
        .increment(1);
    }
}

impl Drop for Attempt {
    fn drop(&mut self) {
        if !self.finished {
            metrics::counter!("etcd_operation_completions_total",
                "operation" => self.operation, "outcome" => "abandoned")
            .increment(1);
        }
    }
}

/// Observe exactly one client future without retries, allocation, spawning,
/// timeout changes or error conversion. Unpolled futures emit no metrics.
pub trait ObserveEtcd<T, E>: Future<Output = Result<T, E>> + Sized {
    fn observed(self, operation: Operation) -> impl Future<Output = Result<T, E>> {
        async move {
            let attempt = Attempt::start(operation);
            let result = self.await;
            attempt.finish(result.is_ok());
            result
        }
    }
}

impl<F, T, E> ObserveEtcd<T, E> for F where F: Future<Output = Result<T, E>> {}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics_exporter_prometheus::PrometheusBuilder;
    use std::{
        future::{pending, ready},
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        task::{Context, Poll, Waker},
    };

    fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        future.poll(&mut Context::from_waker(Waker::noop()))
    }

    #[test]
    fn unpolled_calls_are_absent_and_drop_preserves_unknown_remote_outcome() {
        struct PendingCall(Arc<AtomicBool>);
        impl Future for PendingCall {
            type Output = Result<(), ()>;
            fn poll(self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
                Poll::Pending
            }
        }
        impl Drop for PendingCall {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            drop(pending::<Result<(), ()>>().observed(Operation::TaskClaim));
            assert!(!handle.render().contains("etcd_operation_attempts_total"));
            let dropped = Arc::new(AtomicBool::new(false));
            let mut call = Box::pin(PendingCall(dropped.clone()).observed(Operation::TaskClaim));
            assert!(poll_once(call.as_mut()).is_pending());
            drop(call);
            assert!(dropped.load(Ordering::SeqCst));
        });
        let text = handle.render();
        assert!(text.contains("etcd_operation_attempts_total{operation=\"task_claim\"} 1"));
        assert!(text.contains("outcome=\"abandoned\""));
        assert!(!text.contains("outcome=\"error\""));
    }

    #[test]
    fn responses_retries_and_interleaved_calls_keep_exact_attribution() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            // A rejected comparison is an OK RPC response; preserve its value.
            let mut rejected =
                Box::pin(ready(Ok::<_, &'static str>(false)).observed(Operation::TaskClaim));
            assert_eq!(poll_once(rejected.as_mut()), Poll::Ready(Ok(false)));
            for _ in 0..2 {
                let mut failed =
                    Box::pin(ready(Err::<(), _>("same error")).observed(Operation::MergeRead));
                assert_eq!(poll_once(failed.as_mut()), Poll::Ready(Err("same error")));
            }
            let mut first =
                Box::pin(pending::<Result<(), ()>>().observed(Operation::DemandPublish));
            assert!(poll_once(first.as_mut()).is_pending());
            let mut second = Box::pin(ready(Ok::<_, ()>(7)).observed(Operation::MergeReserve));
            assert_eq!(poll_once(second.as_mut()), Poll::Ready(Ok(7)));
            drop(first);
        });
        let text = handle.render();
        for (operation, count) in [
            ("task_claim", 1),
            ("merge_read", 2),
            ("demand_publish", 1),
            ("merge_reserve", 1),
        ] {
            assert!(text.contains(&format!(
                "etcd_operation_attempts_total{{operation=\"{operation}\"}} {count}"
            )));
        }
        assert!(text.contains("operation=\"merge_read\",outcome=\"error\"} 2"));
        assert!(text.contains("operation=\"task_claim\",outcome=\"ok\"} 1"));
        assert!(text.contains("operation=\"demand_publish\",outcome=\"abandoned\"} 1"));
        assert!(text.contains("operation=\"merge_reserve\",outcome=\"ok\"} 1"));
    }
}
