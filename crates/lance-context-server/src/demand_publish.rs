//! Bounded, fair metadata publication without retaining cached datasets.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::FutureExt;

const CONCURRENCY: usize = 4;
type Job = Pin<Box<dyn Future<Output = ()> + Send>>;

#[derive(Default)]
struct Pending {
    order: VecDeque<String>,
    jobs: HashMap<String, Job>,
    workers: usize,
}

/// One queue per server, shared by rollout and generic stores. Pending jobs
/// must contain only weak store/state references; an active probe may retain
/// its store for the existing bounded metadata timeout.
pub(crate) struct DemandPublisher {
    capacity: usize,
    pending: Mutex<Pending>,
}

impl DemandPublisher {
    pub(crate) fn new(store_cache_capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            // Both resident caches plus one requeued notification for each
            // active probe. This bounds metadata, not merge payload memory.
            capacity: store_cache_capacity
                .saturating_mul(2)
                .saturating_add(CONCURRENCY),
            pending: Mutex::new(Pending::default()),
        })
    }

    /// Never await capacity on the flush path. Replacing a queued observation
    /// keeps its FIFO position and observes the newest resident handle when
    /// executed. A hot table cannot move itself ahead of already queued work.
    pub(crate) fn submit(self: &Arc<Self>, target: String, job: Job) -> bool {
        let mut pending = self.pending.lock().unwrap();
        if let Some(prior) = pending.jobs.get_mut(&target) {
            *prior = job;
            metrics::counter!("rollout_wal_demand_queue_total", "result" => "coalesced")
                .increment(1);
            return true;
        }
        if pending.jobs.len() >= self.capacity {
            metrics::counter!("rollout_wal_demand_queue_total", "result" => "full").increment(1);
            return false;
        }
        pending.order.push_back(target.clone());
        pending.jobs.insert(target, job);
        metrics::counter!("rollout_wal_demand_queue_total", "result" => "queued").increment(1);
        if pending.workers < CONCURRENCY {
            pending.workers += 1;
            let publisher = self.clone();
            tokio::spawn(async move {
                loop {
                    let job = {
                        let mut pending = publisher.pending.lock().unwrap();
                        let Some(target) = pending.order.pop_front() else {
                            pending.workers -= 1;
                            return;
                        };
                        pending
                            .jobs
                            .remove(&target)
                            .expect("queued target has a job")
                    };
                    if std::panic::AssertUnwindSafe(job)
                        .catch_unwind()
                        .await
                        .is_err()
                    {
                        // A faulty observation must not permanently consume a
                        // publisher or prevent the remaining tables from running.
                        metrics::counter!("rollout_wal_demand_events_total", "result" => "panicked")
                            .increment(1);
                    }
                }
            });
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::{mpsc, Semaphore};
    use tokio::time::{timeout, Duration};

    async fn occupy(publisher: &Arc<DemandPublisher>) -> Arc<Semaphore> {
        let gate = Arc::new(Semaphore::new(0));
        let (started, mut observed) = mpsc::unbounded_channel();
        for i in 0..CONCURRENCY {
            let gate = gate.clone();
            let started = started.clone();
            assert!(publisher.submit(
                format!("active-{i}"),
                Box::pin(async move {
                    started.send(()).unwrap();
                    let _permit = gate.acquire().await.unwrap();
                })
            ));
        }
        timeout(Duration::from_secs(1), async {
            for _ in 0..CONCURRENCY {
                observed.recv().await.unwrap();
            }
        })
        .await
        .unwrap();
        gate
    }

    #[tokio::test]
    async fn hot_updates_keep_fifo_position_and_replace_stale_handles() {
        let publisher = DemandPublisher::new(8);
        let gate = occupy(&publisher).await;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let stale = Arc::new(());
        let weak = Arc::downgrade(&stale);
        assert!(publisher.submit(
            "hot".into(),
            Box::pin(async move {
                drop(stale);
                panic!("superseded observation must never execute");
            })
        ));
        for (target, value) in [("cold", 2), ("hot", 1), ("tail", 3)] {
            let tx = tx.clone();
            assert!(publisher.submit(
                target.into(),
                Box::pin(async move {
                    tx.send(value).unwrap();
                })
            ));
        }
        assert!(
            weak.upgrade().is_none(),
            "coalescing retained the old handle"
        );
        assert!(
            rx.try_recv().is_err(),
            "more than four probes ran concurrently"
        );
        gate.add_permits(1);
        let order = timeout(Duration::from_secs(1), async {
            let mut order = Vec::new();
            for _ in 0..3 {
                order.push(rx.recv().await.unwrap());
            }
            order
        })
        .await
        .unwrap();
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn bounded_queue_recovers_after_overflow_and_probe_panic() {
        let publisher = DemandPublisher::new(1);
        let gate = occupy(&publisher).await;
        let (tx, mut rx) = mpsc::unbounded_channel();
        assert!(publisher.submit("panic".into(), Box::pin(async { panic!("bad probe") })));
        for i in 1..publisher.capacity {
            let tx = tx.clone();
            assert!(publisher.submit(
                format!("queued-{i}"),
                Box::pin(async move {
                    tx.send(()).unwrap();
                })
            ));
        }
        assert!(!publisher.submit(
            "overflow".into(),
            Box::pin(async {
                panic!("overflow must not execute");
            })
        ));
        gate.add_permits(4);
        timeout(Duration::from_secs(1), async {
            for _ in 1..publisher.capacity {
                rx.recv().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert!(publisher.submit(
            "after".into(),
            Box::pin(async move {
                tx.send(()).unwrap();
            })
        ));
        timeout(Duration::from_secs(1), rx.recv())
            .await
            .unwrap()
            .unwrap();
    }
}
