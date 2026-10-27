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

#[derive(Clone, Copy)]
enum Outcome {
    Ok,
    Error,
    Abandoned,
}

impl Operation {
    fn record(self, outcome: Option<Outcome>) {
        // Static keys avoid a label-vector allocation on every call. Register
        // against the current recorder; caching handles across recorders would
        // break local recorder isolation.
        use metrics::{Key, Label, Level, Metadata};
        static META: Metadata<'static> =
            Metadata::new(module_path!(), Level::INFO, Some(module_path!()));
        macro_rules! record {
            ($label:literal) => {{
                static ATTEMPT: Key = {
                    static LABELS: [Label; 1] = [Label::from_static_parts("operation", $label)];
                    Key::from_static_parts("etcd_operation_attempts_total", &LABELS)
                };
                static OK: Key = {
                    static LABELS: [Label; 2] = [
                        Label::from_static_parts("operation", $label),
                        Label::from_static_parts("outcome", "ok"),
                    ];
                    Key::from_static_parts("etcd_operation_completions_total", &LABELS)
                };
                static ERROR: Key = {
                    static LABELS: [Label; 2] = [
                        Label::from_static_parts("operation", $label),
                        Label::from_static_parts("outcome", "error"),
                    ];
                    Key::from_static_parts("etcd_operation_completions_total", &LABELS)
                };
                static ABANDONED: Key = {
                    static LABELS: [Label; 2] = [
                        Label::from_static_parts("operation", $label),
                        Label::from_static_parts("outcome", "abandoned"),
                    ];
                    Key::from_static_parts("etcd_operation_completions_total", &LABELS)
                };
                let key = match outcome {
                    None => &ATTEMPT,
                    Some(Outcome::Ok) => &OK,
                    Some(Outcome::Error) => &ERROR,
                    Some(Outcome::Abandoned) => &ABANDONED,
                };
                metrics::with_recorder(|recorder| {
                    recorder.register_counter(key, &META).increment(1)
                });
            }};
        }
        match self {
            Self::MergeRead => record!("merge_read"),
            Self::MergeScan => record!("merge_scan"),
            Self::MergeReserve => record!("merge_reserve"),
            Self::MergeStart => record!("merge_start"),
            Self::MergeTerminal => record!("merge_terminal"),
            Self::MergeRelease => record!("merge_release"),
            Self::MergeProgressInitial => record!("merge_progress_initial"),
            Self::MergeProgressChanged => record!("merge_progress_changed"),
            Self::MergeWorkProgress => record!("merge_work_progress"),
            Self::MergeCommitRead => record!("merge_commit_read"),
            Self::MergeCommitWrite => record!("merge_commit_write"),
            Self::MergeFailureRead => record!("merge_failure_read"),
            Self::MergeOther => record!("merge_other"),
            Self::MergeOwnerWatch => record!("merge_owner_watch"),
            Self::MergeProgressWatch => record!("merge_progress_watch"),
            Self::DemandRead => record!("demand_read"),
            Self::DemandPublish => record!("demand_publish"),
            Self::DemandScan => record!("demand_scan"),
            Self::TaskRead => record!("task_read"),
            Self::TaskReadTxn => record!("task_read_txn"),
            Self::TaskScan => record!("task_scan"),
            Self::TaskEnqueue => record!("task_enqueue"),
            Self::TaskClaim => record!("task_claim"),
            Self::TaskFinish => record!("task_finish"),
            Self::TaskOther => record!("task_other"),
            Self::DispatchRead => record!("dispatch_read"),
            Self::DispatchWrite => record!("dispatch_write"),
            Self::DispatchWatch => record!("dispatch_watch"),
            Self::LeaseGrant => record!("lease_grant"),
            Self::LeaseStreamOpen => record!("lease_stream_open"),
            Self::LeaseRevoke => record!("lease_revoke"),
        }
    }
}

struct Attempt {
    operation: Operation,
    finished: bool,
}

impl Attempt {
    fn start(operation: Operation) -> Self {
        operation.record(None);
        Self {
            operation,
            finished: false,
        }
    }

    fn finish(mut self, ok: bool) {
        self.finished = true;
        self.operation
            .record(Some(if ok { Outcome::Ok } else { Outcome::Error }));
    }
}

impl Drop for Attempt {
    fn drop(&mut self) {
        if !self.finished {
            self.operation.record(Some(Outcome::Abandoned));
        }
    }
}

/// Observe exactly one client future without boxing, retries, spawning,
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
