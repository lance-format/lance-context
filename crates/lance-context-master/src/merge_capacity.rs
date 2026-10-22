//! Bounded legacy response waits which keep the original task and claim alive.
//!
//! This is scheduler fairness, not a worker timeout. The HTTP future is never
//! cancelled, and it must regain execution capacity before continuing fan-out.
use crate::admission::Operation;
use std::{
    future::Future,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

tokio::task_local! {
    static CAPACITY: Arc<Capacity>;
}

pub(crate) struct Capacity {
    permit: Mutex<Option<OwnedSemaphorePermit>>,
    pool: Arc<Semaphore>,
    waiters: Arc<Semaphore>,
    foreground: Duration,
    running: metrics::Gauge,
    operation: Arc<Operation>,
}

impl Capacity {
    pub(crate) fn new(
        permit: OwnedSemaphorePermit,
        pool: Arc<Semaphore>,
        waiters: Arc<Semaphore>,
        foreground: Duration,
        running: metrics::Gauge,
        operation: Arc<Operation>,
    ) -> Arc<Self> {
        running.increment(1.0);
        Arc::new(Self {
            permit: Mutex::new(Some(permit)),
            pool,
            waiters,
            foreground,
            running,
            operation,
        })
    }

    async fn observe<F: Future>(&self, request: F) -> F::Output {
        tokio::pin!(request);
        // Slow successful merges remain valid. This timer only allows another
        // table to use the local slot; it changes no ownership/task state.
        let result = tokio::time::timeout(self.foreground, &mut request).await;
        if let Ok(result) = result {
            return result;
        }
        let waiter = tokio::select! {
            biased;
            result = &mut request => return result,
            permit = self.waiters.clone().acquire_owned() => permit.expect("wait budget remains open"),
        };
        let waiting = Waiting::new(waiter);
        let permit = self.permit.lock().unwrap().take().expect("running permit");
        self.running.decrement(1.0);
        drop(permit);
        self.operation.phase("waiting_legacy_response");
        metrics::counter!("master_merge_wal_response_yields_total").increment(1);
        let result = request.await;
        self.operation.phase("awaiting_merge_slot");
        // Register in the fair execution queue BEFORE releasing wait capacity:
        // new claims cannot jump ahead and inflate the number of retained tasks.
        // Then release wait capacity BEFORE blocking, so full execution/wait
        // pools cannot deadlock each other.
        let acquire = self.pool.clone().acquire_owned();
        tokio::pin!(acquire);
        let first_poll = futures::poll!(&mut acquire);
        drop(waiting);
        let permit = match first_poll {
            std::task::Poll::Ready(permit) => permit,
            std::task::Poll::Pending => acquire.await,
        }
        .expect("scheduler pool remains open");
        *self.permit.lock().unwrap() = Some(permit);
        self.running.increment(1.0);
        self.operation.phase("running");
        result
    }
}

impl Drop for Capacity {
    fn drop(&mut self) {
        if self.permit.get_mut().unwrap().is_some() {
            self.running.decrement(1.0);
        }
    }
}

struct Waiting {
    _permit: OwnedSemaphorePermit,
    gauge: metrics::Gauge,
}
impl Waiting {
    fn new(permit: OwnedSemaphorePermit) -> Self {
        let gauge = metrics::gauge!("master_legacy_response_waiters");
        gauge.increment(1.0);
        Self {
            _permit: permit,
            gauge,
        }
    }
}
impl Drop for Waiting {
    fn drop(&mut self) {
        self.gauge.decrement(1.0);
    }
}

pub(crate) async fn scope<F: Future>(capacity: Arc<Capacity>, work: F) -> F::Output {
    CAPACITY.scope(capacity, work).await
}

/// Only the legacy HTTP response path calls this. Fenced executions, local
/// merges, compaction and claim RPCs retain their existing execution budgets.
pub(crate) async fn observe<F: Future>(request: F) -> F::Output {
    match CAPACITY.try_with(Arc::clone) {
        Ok(capacity) => capacity.observe(request).await,
        Err(_) => request.await, // Direct execution outside the scheduler.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::Admission;

    #[tokio::test]
    async fn response_wait_yields_capacity_but_keeps_drain_and_reacquires_before_continuing() {
        let admission = Arc::new(Admission::default());
        let pool = Arc::new(Semaphore::new(1));
        let waiters = Arc::new(Semaphore::new(1));
        let operation = Arc::new(admission.try_admit_named("merge").unwrap());
        let capacity = Capacity::new(
            pool.clone().acquire_owned().await.unwrap(),
            pool.clone(),
            waiters.clone(),
            Duration::from_millis(10),
            metrics::gauge!("test_capacity_running"),
            operation,
        );
        let (reply, receive) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(scope(capacity, async {
            observe(async { receive.await.unwrap() }).await
        }));
        let other = tokio::time::timeout(Duration::from_secs(5), pool.clone().acquire_owned())
            .await
            .unwrap()
            .unwrap();
        admission.begin_drain();
        assert_eq!(waiters.available_permits(), 0);
        assert_eq!(admission.status().active_operations, 1);
        reply.send(7).unwrap();
        tokio::task::yield_now().await;
        assert!(
            !task.is_finished(),
            "late response cannot continue without an execution slot"
        );
        assert!(!admission.status().drained());
        drop(other);
        assert_eq!(task.await.unwrap(), 7);
        admission.wait_drained().await;
        assert_eq!(pool.available_permits(), 1);
        assert_eq!(waiters.available_permits(), 1);
    }

    #[tokio::test]
    async fn full_pools_allow_a_finished_waiter_to_resume_before_the_next_response() {
        let admission = Arc::new(Admission::default());
        let pool = Arc::new(Semaphore::new(1));
        let waiters = Arc::new(Semaphore::new(1));
        let make = |permit| {
            Capacity::new(
                permit,
                pool.clone(),
                waiters.clone(),
                Duration::from_millis(5),
                metrics::gauge!("test_capacity_running"),
                Arc::new(admission.try_admit_named("merge").unwrap()),
            )
        };
        let (reply_a, receive_a) = tokio::sync::oneshot::channel();
        let a = tokio::spawn(scope(
            make(pool.clone().acquire_owned().await.unwrap()),
            async { observe(async { receive_a.await.unwrap() }).await },
        ));
        let permit_b = tokio::time::timeout(Duration::from_secs(5), pool.clone().acquire_owned())
            .await
            .unwrap()
            .unwrap();
        let (reply_b, receive_b) = tokio::sync::oneshot::channel();
        let b = tokio::spawn(scope(make(permit_b), async {
            observe(async { receive_b.await.unwrap() }).await
        }));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), pool.clone().acquire_owned())
                .await
                .is_err()
        );
        reply_a.send(1).unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), a)
                .await
                .unwrap()
                .unwrap(),
            1,
            "a completed response must not deadlock behind a full response-wait pool"
        );
        assert!(!b.is_finished());
        reply_b.send(2).unwrap();
        assert_eq!(b.await.unwrap(), 2);
        admission.begin_drain();
        admission.wait_drained().await;
        assert_eq!(pool.available_permits(), 1);
        assert_eq!(waiters.available_permits(), 1);
    }

    #[tokio::test]
    async fn full_response_wait_budget_keeps_original_slot_and_request() {
        let admission = Arc::new(Admission::default());
        let pool = Arc::new(Semaphore::new(1));
        let waiters = Arc::new(Semaphore::new(1));
        let _occupied = waiters.clone().acquire_owned().await.unwrap();
        let capacity = Capacity::new(
            pool.clone().acquire_owned().await.unwrap(),
            pool.clone(),
            waiters,
            Duration::from_millis(5),
            metrics::gauge!("test_capacity_running"),
            Arc::new(admission.try_admit_named("merge").unwrap()),
        );
        let (reply, receive) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(scope(capacity, async {
            observe(async { receive.await.unwrap() }).await
        }));
        assert!(
            tokio::time::timeout(Duration::from_millis(30), pool.clone().acquire_owned())
                .await
                .is_err()
        );
        assert_eq!(admission.status().active_operations, 1);
        reply.send(9).unwrap();
        assert_eq!(task.await.unwrap(), 9);
        assert_eq!(pool.available_permits(), 1);
    }
}
