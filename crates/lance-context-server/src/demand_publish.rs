//! Bounded, fair metadata publication without retaining cached datasets.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use lance_context_merge::demand::{DemandEvent, EventSource, SCHEMA_VERSION};
use lru::LruCache;

const CONCURRENCY: usize = 4;
const MAX_ACKNOWLEDGEMENTS: usize = 16_384;
const MAX_ACK_KEY_BYTES: usize = 1024;
type Job = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Only the simple writer observation is cached. Additional age/merge evidence
/// must always reach the durable join, even if the sealed watermark is equal.
#[derive(Clone, Copy)]
struct WriterAcknowledgement {
    generation: u64,
    epoch: u64,
    bytes: u64,
    flushed_at_ms: i64,
}

impl WriterAcknowledgement {
    fn from_event(event: &DemandEvent) -> Option<Self> {
        (event.v == SCHEMA_VERSION
            && event.source == EventSource::Writer
            && event.merged_through.is_none()
            && event.merged_epoch.is_none()
            && event.sealed_times.is_empty())
        .then_some(Self {
            generation: event.sealed_through,
            epoch: event.writer_epoch,
            bytes: event.sealed_bytes_through,
            flushed_at_ms: event.flushed_at_ms,
        })
    }

    fn covers(self, next: Self) -> bool {
        self.generation == next.generation
            && self.epoch == next.epoch
            && self.bytes == next.bytes
            && self.flushed_at_ms <= next.flushed_at_ms
    }

    fn join(&mut self, next: Self) {
        // Jobs for a target can overlap. An old completion must not replace a
        // newer watermark; an earlier time for the SAME generation is useful.
        let earliest = self.flushed_at_ms.min(next.flushed_at_ms);
        let same_generation = self.generation == next.generation;
        if (next.generation, next.epoch, next.bytes) > (self.generation, self.epoch, self.bytes) {
            *self = next;
        }
        if same_generation {
            self.flushed_at_ms = earliest;
        }
    }
}

/// The pending manifest returns generation zero after a shard is drained.
/// Keep that successfully acknowledged observation independently: the positive
/// high-watermark must survive late completions, but cannot cover generation
/// zero (or any other older event) without first observing its durable join.
/// Two fixed slots per LRU entry keep memory bounded without retaining stores.
#[derive(Default)]
struct ShardAcknowledgements {
    positive: Option<WriterAcknowledgement>,
    empty: Option<WriterAcknowledgement>,
}

impl ShardAcknowledgements {
    fn covers(&self, next: WriterAcknowledgement) -> bool {
        let prior = if next.generation == 0 {
            self.empty
        } else {
            self.positive
        };
        prior.is_some_and(|prior| prior.covers(next))
    }

    fn join(&mut self, next: WriterAcknowledgement) {
        let slot = if next.generation == 0 {
            &mut self.empty
        } else {
            &mut self.positive
        };
        if let Some(prior) = slot {
            prior.join(next);
        } else {
            *slot = Some(next);
        }
    }
}

impl From<WriterAcknowledgement> for ShardAcknowledgements {
    fn from(next: WriterAcknowledgement) -> Self {
        let mut acknowledgements = Self::default();
        acknowledgements.join(next);
        acknowledgements
    }
}

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
    acknowledged: Mutex<LruCache<(String, String), ShardAcknowledgements>>,
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
            acknowledged: Mutex::new(LruCache::new(
                NonZeroUsize::new(
                    store_cache_capacity
                        .saturating_mul(2)
                        .saturating_add(CONCURRENCY)
                        .min(MAX_ACKNOWLEDGEMENTS),
                )
                .unwrap(),
            )),
        })
    }

    /// Avoid a durable GET only after this server observed a successful join
    /// of the same writer watermark. Errors, dropped futures and lost replies
    /// never acknowledge an event. Eviction/restart safely replays the join.
    /// This cache must not cover the independent merge-request publication.
    pub(crate) async fn publish_writer_event<E>(
        &self,
        target: &str,
        event: &DemandEvent,
        publish: impl Future<Output = Result<bool, E>>,
    ) -> Result<&'static str, E> {
        let acknowledgement = WriterAcknowledgement::from_event(event)
            .filter(|_| target.len().saturating_add(event.shard.len()) <= MAX_ACK_KEY_BYTES);
        let key = acknowledgement.map(|_| (target.to_owned(), event.shard.clone()));
        if let (Some(key), Some(next)) = (&key, acknowledgement) {
            if self
                .acknowledged
                .lock()
                .unwrap()
                .get(key)
                .is_some_and(|prior| prior.covers(next))
            {
                return Ok("cached");
            }
        }
        let written = publish.await?;
        if let (Some(key), Some(next)) = (key, acknowledgement) {
            let mut acknowledged = self.acknowledged.lock().unwrap();
            if let Some(prior) = acknowledged.get_mut(&key) {
                prior.join(next);
            } else {
                acknowledged.put(key, next.into());
            }
        }
        Ok(if written { "written" } else { "unchanged" })
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

    fn event() -> DemandEvent {
        DemandEvent {
            v: SCHEMA_VERSION,
            shard: "shard".into(),
            sealed_through: 7,
            sealed_bytes_through: 40,
            merged_through: None,
            merged_epoch: None,
            flushed_at_ms: 100,
            writer_epoch: 2,
            source: EventSource::Writer,
            sealed_times: Default::default(),
        }
    }

    async fn must_skip(publisher: &DemandPublisher, target: &str, event: &DemandEvent) {
        let result: Result<_, ()> = publisher
            .publish_writer_event(target, event, async {
                panic!("acknowledged observation issued another durable read")
            })
            .await;
        assert_eq!(result, Ok("cached"));
    }

    #[tokio::test]
    async fn acknowledged_writer_skips_reads_but_preserves_new_evidence() {
        let publisher = DemandPublisher::new(8);
        let mut observation = event();
        // A server restart may see an already durable event. Ok(false) is
        // just as strong an acknowledgement as a successful write.
        assert_eq!(
            publisher
                .publish_writer_event("t", &observation, async { Ok::<_, ()>(false) })
                .await,
            Ok("unchanged")
        );
        for _ in 0..100 {
            observation.flushed_at_ms += 1;
            must_skip(&publisher, "t", &observation).await;
        }
        let mut changes = Vec::new();
        let mut e = observation.clone();
        e.flushed_at_ms = 99;
        changes.push(e);
        let mut e = observation.clone();
        e.sealed_through -= 1; // Older events can add missing age evidence.
        changes.push(e);
        let mut e = observation.clone();
        e.writer_epoch += 1;
        changes.push(e);
        let mut e = observation.clone();
        e.sealed_bytes_through += 1;
        changes.push(e);
        let mut e = observation.clone();
        e.sealed_through += 1;
        changes.push(e);
        let mut e = observation.clone();
        e.shard = "another-shard".into();
        changes.push(e);
        for e in changes {
            assert_eq!(
                publisher
                    .publish_writer_event("t", &e, async { Ok::<_, ()>(true) })
                    .await,
                Ok("written")
            );
        }
        assert_eq!(
            publisher
                .publish_writer_event("other-table", &observation, async { Ok::<_, ()>(true) })
                .await,
            Ok("written")
        );
    }

    #[tokio::test]
    async fn drained_shard_stops_republishing_empty_without_losing_positive_ack() {
        let publisher = DemandPublisher::new(8);
        let full = event();
        publisher
            .publish_writer_event("t", &full, async { Ok::<_, ()>(true) })
            .await
            .unwrap();
        let mut empty = full.clone();
        empty.sealed_through = 0;
        empty.sealed_bytes_through = 0;
        empty.flushed_at_ms += 1;
        assert_eq!(
            publisher
                .publish_writer_event("t", &empty, async { Ok::<_, ()>(false) })
                .await,
            Ok("unchanged")
        );
        for _ in 0..100 {
            empty.flushed_at_ms += 1;
            must_skip(&publisher, "t", &empty).await;
            must_skip(&publisher, "t", &full).await;
        }
        // Nonempty lower generations can carry useful age evidence; the empty
        // acknowledgement must not accidentally suppress them.
        let mut old = full.clone();
        old.sealed_through -= 1;
        assert_eq!(
            publisher
                .publish_writer_event("t", &old, async { Ok::<_, ()>(false) })
                .await,
            Ok("unchanged")
        );
        let mut fresh = full.clone();
        fresh.sealed_through += 1;
        assert_eq!(
            publisher
                .publish_writer_event("t", &fresh, async { Ok::<_, ()>(true) })
                .await,
            Ok("written")
        );
        must_skip(&publisher, "t", &fresh).await;
        must_skip(&publisher, "t", &empty).await;
    }

    #[tokio::test]
    async fn empty_ack_requires_success_and_preserves_new_evidence() {
        let publisher = DemandPublisher::new(8);
        let full = event();
        publisher
            .publish_writer_event("t", &full, async { Ok::<_, ()>(true) })
            .await
            .unwrap();
        let mut empty = full.clone();
        empty.sealed_through = 0;
        for _ in 0..2 {
            assert_eq!(
                publisher
                    .publish_writer_event("t", &empty, async { Err::<bool, _>("lost") })
                    .await,
                Err("lost")
            );
        }
        let blocked = publisher.publish_writer_event("t", &empty, async {
            futures::future::pending::<()>().await;
            Ok::<_, ()>(false)
        });
        assert!(timeout(Duration::from_millis(10), blocked).await.is_err());
        assert_eq!(
            publisher
                .publish_writer_event("t", &empty, async { Ok::<_, ()>(false) })
                .await,
            Ok("unchanged")
        );
        must_skip(&publisher, "t", &empty).await;
        for dimension in 0..3 {
            let mut changed = empty.clone();
            match dimension {
                0 => changed.writer_epoch += 1,
                1 => changed.sealed_bytes_through += 1,
                _ => changed.flushed_at_ms -= 1,
            }
            assert_eq!(
                publisher
                    .publish_writer_event("t", &changed, async { Ok::<_, ()>(false) })
                    .await,
                Ok("unchanged")
            );
        }
        must_skip(&publisher, "t", &full).await;
    }

    #[tokio::test]
    async fn empty_and_positive_acknowledgements_survive_out_of_order_completion() {
        for empty_finishes_last in [false, true] {
            let publisher = DemandPublisher::new(8);
            let full = event();
            let mut empty = full.clone();
            empty.sealed_through = 0;
            let (old, new) = if empty_finishes_last {
                (&empty, &full)
            } else {
                (&full, &empty)
            };
            let (release, blocked) = tokio::sync::oneshot::channel();
            let older = publisher.publish_writer_event("t", old, async {
                blocked.await.unwrap();
                Ok::<_, ()>(false)
            });
            tokio::pin!(older);
            assert!(futures::poll!(&mut older).is_pending());
            publisher
                .publish_writer_event("t", new, async { Ok::<_, ()>(false) })
                .await
                .unwrap();
            release.send(()).unwrap();
            older.await.unwrap();
            must_skip(&publisher, "t", &full).await;
            must_skip(&publisher, "t", &empty).await;
        }
    }

    #[tokio::test]
    async fn failure_and_cancelled_reply_never_acknowledge_writer() {
        let publisher = DemandPublisher::new(8);
        let observation = event();
        assert_eq!(
            publisher
                .publish_writer_event("t", &observation, async { Err::<bool, _>("unavailable") })
                .await,
            Err("unavailable")
        );
        let lost_reply = publisher.publish_writer_event("t", &observation, async {
            futures::future::pending::<()>().await;
            Ok::<_, ()>(true)
        });
        assert!(timeout(Duration::from_millis(10), lost_reply)
            .await
            .is_err());
        let panicked = std::panic::AssertUnwindSafe(publisher.publish_writer_event(
            "t",
            &observation,
            async {
                panic!("failed transport");
                #[allow(unreachable_code)]
                Ok::<_, ()>(true)
            },
        ))
        .catch_unwind()
        .await;
        assert!(panicked.is_err());
        assert_eq!(
            publisher
                .publish_writer_event("t", &observation, async { Ok::<_, ()>(true) })
                .await,
            Ok("written")
        );
        must_skip(&publisher, "t", &observation).await;
    }

    #[tokio::test]
    async fn old_completion_cannot_regress_newer_acknowledgement() {
        let publisher = DemandPublisher::new(8);
        let old = event();
        let mut new = old.clone();
        new.sealed_through += 1;
        new.flushed_at_ms += 10;
        let (release, blocked) = tokio::sync::oneshot::channel();
        let older = publisher.publish_writer_event("t", &old, async {
            blocked.await.unwrap();
            Ok::<_, ()>(true)
        });
        tokio::pin!(older);
        assert!(futures::poll!(&mut older).is_pending());
        assert_eq!(
            publisher
                .publish_writer_event("t", &new, async { Ok::<_, ()>(true) })
                .await,
            Ok("written")
        );
        release.send(()).unwrap();
        assert_eq!(older.await, Ok("written"));
        must_skip(&publisher, "t", &new).await;
        // An older generation must still reach the join for its flush time.
        assert_eq!(
            publisher
                .publish_writer_event("t", &old, async { Ok::<_, ()>(false) })
                .await,
            Ok("unchanged")
        );
        must_skip(&publisher, "t", &new).await;
        let mut earlier = new.clone();
        earlier.flushed_at_ms -= 5;
        publisher
            .publish_writer_event("t", &earlier, async { Ok::<_, ()>(true) })
            .await
            .unwrap();
        must_skip(&publisher, "t", &earlier).await;
    }

    #[tokio::test]
    async fn bounded_lru_eviction_and_new_server_replay_safely() {
        let publisher = DemandPublisher::new(0);
        let observation = event();
        for i in 0..CONCURRENCY {
            publisher
                .publish_writer_event(&i.to_string(), &observation, async { Ok::<_, ()>(true) })
                .await
                .unwrap();
        }
        must_skip(&publisher, "0", &observation).await;
        publisher
            .publish_writer_event("new", &observation, async { Ok::<_, ()>(true) })
            .await
            .unwrap();
        assert_eq!(publisher.acknowledged.lock().unwrap().len(), CONCURRENCY);
        must_skip(&publisher, "0", &observation).await;
        assert_eq!(
            publisher
                .publish_writer_event("1", &observation, async { Ok::<_, ()>(false) })
                .await,
            Ok("unchanged")
        );
        let weak = Arc::downgrade(&publisher);
        drop(publisher);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            DemandPublisher::new(0)
                .publish_writer_event("0", &observation, async { Ok::<_, ()>(false) })
                .await,
            Ok("unchanged")
        );
    }

    #[tokio::test]
    async fn empty_ack_is_bounded_and_replays_after_eviction_or_restart() {
        let publisher = DemandPublisher::new(0);
        let full = event();
        let mut empty = full.clone();
        empty.sealed_through = 0;
        for i in 0..CONCURRENCY {
            for observation in [&full, &empty] {
                publisher
                    .publish_writer_event(&i.to_string(), observation, async { Ok::<_, ()>(false) })
                    .await
                    .unwrap();
            }
        }
        assert_eq!(publisher.acknowledged.lock().unwrap().len(), CONCURRENCY);
        must_skip(&publisher, "0", &empty).await;
        publisher
            .publish_writer_event("extra", &empty, async { Ok::<_, ()>(false) })
            .await
            .unwrap();
        assert_eq!(publisher.acknowledged.lock().unwrap().len(), CONCURRENCY);
        assert_eq!(
            publisher
                .publish_writer_event("1", &empty, async { Ok::<_, ()>(false) })
                .await,
            Ok("unchanged")
        );
        let restarted = DemandPublisher::new(0);
        assert_eq!(
            restarted
                .publish_writer_event("0", &empty, async { Ok::<_, ()>(false) })
                .await,
            Ok("unchanged")
        );
    }

    #[tokio::test]
    async fn extended_events_and_large_keys_always_reach_durable_join() {
        let publisher = DemandPublisher::new(8);
        let observation = event();
        publisher
            .publish_writer_event("t", &observation, async { Ok::<_, ()>(true) })
            .await
            .unwrap();
        let mut variants = Vec::new();
        let mut e = observation.clone();
        e.merged_through = Some(3);
        variants.push(e);
        let mut e = observation.clone();
        e.merged_epoch = Some(1);
        variants.push(e);
        let mut e = observation.clone();
        e.source = EventSource::Scan;
        variants.push(e);
        let mut e = observation.clone();
        e.sealed_times.insert(1, 10);
        variants.push(e);
        let mut e = observation.clone();
        e.v += 1;
        variants.push(e);
        for e in variants {
            for _ in 0..2 {
                assert_eq!(
                    publisher
                        .publish_writer_event("t", &e, async { Ok::<_, ()>(true) })
                        .await,
                    Ok("written")
                );
            }
        }
        let large = "x".repeat(MAX_ACK_KEY_BYTES + 1);
        for _ in 0..2 {
            assert_eq!(
                publisher
                    .publish_writer_event(&large, &observation, async { Ok::<_, ()>(true) })
                    .await,
                Ok("written")
            );
        }
        assert_eq!(publisher.acknowledged.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn durable_join_replays_lost_reply_and_skips_only_acknowledged_event() {
        let client =
            etcd_client::Client::connect([std::env::var("ETCD_TEST_ENDPOINTS").unwrap()], None)
                .await
                .unwrap();
        let prefix = format!("/writer-ack/{}", uuid::Uuid::new_v4());
        let coordinator = lance_context_merge::Coordinator::new(client.clone(), prefix.clone());
        let publisher = DemandPublisher::new(8);
        let mut observation = event();
        // The write really commits, but no acknowledgement reaches our cache.
        let lost = publisher.publish_writer_event("t", &observation, async {
            assert!(coordinator.publish_demand_event("t", &observation).await?);
            futures::future::pending::<()>().await;
            Ok::<_, String>(true)
        });
        assert!(timeout(Duration::from_secs(1), lost).await.is_err());
        assert_eq!(
            publisher
                .publish_writer_event(
                    "t",
                    &observation,
                    coordinator.publish_demand_event("t", &observation),
                )
                .await
                .unwrap(),
            "unchanged"
        );
        for _ in 0..100 {
            must_skip(&publisher, "t", &observation).await;
        }
        observation.sealed_through += 1;
        assert_eq!(
            publisher
                .publish_writer_event(
                    "t",
                    &observation,
                    coordinator.publish_demand_event("t", &observation),
                )
                .await
                .unwrap(),
            "written"
        );
        let stored = client
            .clone()
            .get(format!("{prefix}/demand-events/74/shard"), None)
            .await
            .unwrap();
        let stored: DemandEvent = serde_json::from_slice(stored.kvs()[0].value()).unwrap();
        assert_eq!(stored.sealed_through, observation.sealed_through);
        must_skip(&publisher, "t", &observation).await;
        coordinator
            .publish_merged_watermark(
                "t",
                &lance_context_merge::demand::ShardMerged {
                    shard: observation.shard.clone(),
                    merged_through: observation.sealed_through,
                },
            )
            .await
            .unwrap();
        let mut empty = observation.clone();
        empty.sealed_through = 0;
        empty.sealed_bytes_through = 0;
        // A durable no-op whose reply is lost must not populate the empty slot.
        let lost_empty = publisher.publish_writer_event("t", &empty, async {
            assert!(!coordinator.publish_demand_event("t", &empty).await?);
            futures::future::pending::<()>().await;
            Ok::<_, String>(false)
        });
        assert!(timeout(Duration::from_millis(50), lost_empty)
            .await
            .is_err());
        assert_eq!(
            publisher
                .publish_writer_event("t", &empty, coordinator.publish_demand_event("t", &empty))
                .await
                .unwrap(),
            "unchanged"
        );
        for _ in 0..100 {
            empty.flushed_at_ms += 1;
            must_skip(&publisher, "t", &empty).await;
        }
        let stored = client
            .clone()
            .get(format!("{prefix}/demand-events/74/shard"), None)
            .await
            .unwrap();
        let stored: DemandEvent = serde_json::from_slice(stored.kvs()[0].value()).unwrap();
        assert_eq!(stored.sealed_through, observation.sealed_through);
        assert_eq!(stored.merged_through, Some(observation.sealed_through));
        observation.sealed_through += 1;
        assert_eq!(
            publisher
                .publish_writer_event(
                    "t",
                    &observation,
                    coordinator.publish_demand_event("t", &observation)
                )
                .await
                .unwrap(),
            "written"
        );
        let stored = client
            .clone()
            .get(format!("{prefix}/demand-events/74/shard"), None)
            .await
            .unwrap();
        let stored: DemandEvent = serde_json::from_slice(stored.kvs()[0].value()).unwrap();
        assert_eq!(stored.sealed_through, observation.sealed_through);
        assert_eq!(stored.merged_through, Some(observation.sealed_through - 1));
    }

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
