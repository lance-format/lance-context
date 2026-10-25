use super::*;
use crate::demand_controller::aggregate::{frontier_key, Admission};
use metrics::{
    Counter, CounterFn, Gauge, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
};
use std::collections::BTreeMap;
use std::sync::{atomic::AtomicU64, Mutex};

#[derive(Default)]
struct Count(AtomicU64);
impl CounterFn for Count {
    fn increment(&self, value: u64) {
        self.0.fetch_add(value, Ordering::Relaxed);
    }
    fn absolute(&self, value: u64) {
        self.0.fetch_max(value, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct Capture(Mutex<HashMap<Key, Arc<Count>>>);
impl Recorder for Capture {
    fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn register_counter(&self, key: &Key, _: &Metadata<'_>) -> Counter {
        Counter::from_arc(
            self.0
                .lock()
                .unwrap()
                .entry(key.clone())
                .or_default()
                .clone(),
        )
    }
    fn register_gauge(&self, _: &Key, _: &Metadata<'_>) -> Gauge {
        Gauge::noop()
    }
    fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> Histogram {
        Histogram::noop()
    }
}
impl Capture {
    async fn run<F: std::future::Future>(&self, future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        futures::future::poll_fn(|cx| {
            metrics::with_local_recorder(self, || future.as_mut().poll(cx))
        })
        .await
    }

    fn outcomes(&self, origin: &str) -> BTreeMap<String, u64> {
        let mut result = BTreeMap::new();
        for (key, count) in self.0.lock().unwrap().iter() {
            if key.name() != "master_task_admissions_total" {
                continue;
            }
            let labels: BTreeMap<_, _> = key.labels().map(|l| (l.key(), l.value())).collect();
            assert_eq!(
                labels.keys().copied().collect::<Vec<_>>(),
                ["kind", "origin", "outcome"]
            );
            if labels["origin"] == origin {
                *result.entry(labels["outcome"].to_owned()).or_default() +=
                    count.0.load(Ordering::Relaxed);
            }
        }
        result
    }

    fn created_legacy(&self) -> u64 {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|(key, _)| key.name() == "master_task_enqueued_total")
            .map(|(_, count)| count.0.load(Ordering::Relaxed))
            .sum()
    }
}

fn counts(entries: &[(&str, u64)]) -> BTreeMap<String, u64> {
    entries.iter().map(|(k, v)| ((*k).into(), *v)).collect()
}

#[tokio::test]
#[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
async fn concurrent_admissions_count_one_creation_and_preserve_task_wire() {
    let (_dir, store) = bounded_scheduler_store(vec![]).await;
    let capture = Capture::default();
    let tasks = capture
        .run(async {
            futures::future::join_all(
                (0..16).map(|_| store.enqueue(TaskKind::MergeWal, "same-target", vec![])),
            )
            .await
        })
        .await;
    let tasks: Vec<_> = tasks.into_iter().map(Result::unwrap).collect();
    assert!(tasks.iter().all(|task| task.id == tasks[0].id));
    assert_eq!(store.list().await.unwrap().len(), 1);
    assert_eq!(
        capture.outcomes("explicit"),
        counts(&[("created", 1), ("deduplicated", 15)])
    );
    assert_eq!(capture.created_legacy(), 1);
    let mut client = store.etcd_client().clone();
    let value = client
        .get(store.inner.task_key(&tasks[0].id), None)
        .await
        .unwrap();
    assert_eq!(value.kvs()[0].value(), encode_task(&tasks[0]).unwrap());
    let claim = store.claim_next().await.unwrap().unwrap();
    capture
        .run(store.enqueue(TaskKind::MergeWal, "same-target", vec![]))
        .await
        .unwrap();
    store
        .finish(claim, Ok("completed pass".into()))
        .await
        .unwrap();
    let next = capture
        .run(store.enqueue(TaskKind::MergeWal, "same-target", vec![]))
        .await
        .unwrap();
    assert_ne!(next.id, tasks[0].id);
    assert_eq!(
        capture.outcomes("explicit"),
        counts(&[("created", 2), ("deduplicated", 16)])
    );
    assert_eq!(capture.created_legacy(), 2);
}

#[tokio::test]
#[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
async fn request_admissions_preserve_running_demand_and_count_cached_ack_once() {
    let (_dir, store) = bounded_scheduler_store(vec![]).await;
    let c = store.merge_coordinator();
    let capture = Capture::default();
    c.request_merge("hot").await.unwrap();
    let first = c.get_merge_request("hot").await.unwrap().unwrap();
    let task = capture
        .run(store.enqueue_merge_request(&first))
        .await
        .unwrap()
        .unwrap();
    assert!(c.get_merge_request("hot").await.unwrap().is_none());
    c.request_merge("hot").await.unwrap();
    let queued = c.get_merge_request("hot").await.unwrap().unwrap();
    let wire = encode_task(&task).unwrap();
    assert!(capture
        .run(store.acknowledge_cached_merge_request(&queued, &wire))
        .await
        .unwrap());
    // A failed cache fast path emits no outcome; the normal fallback owns it.
    assert!(!capture
        .run(store.acknowledge_cached_merge_request(&queued, &wire))
        .await
        .unwrap());
    assert_eq!(
        capture.outcomes("writer_request"),
        counts(&[("created", 1), ("deduplicated", 1)])
    );
    assert!(capture
        .run(store.enqueue_merge_request(&queued))
        .await
        .unwrap()
        .is_none());
    c.request_merge("hot").await.unwrap();
    let uncached = c.get_merge_request("hot").await.unwrap().unwrap();
    assert_eq!(
        capture
            .run(store.enqueue_merge_request(&uncached))
            .await
            .unwrap()
            .unwrap()
            .id,
        task.id
    );
    assert!(c.get_merge_request("hot").await.unwrap().is_none());
    c.request_merge("hot").await.unwrap();
    let running = c.get_merge_request("hot").await.unwrap().unwrap();
    let claim = store.claim_next().await.unwrap().unwrap();
    let same = capture
        .run(store.enqueue_merge_request(&running))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(same.id, task.id);
    assert_eq!(
        c.get_merge_request("hot").await.unwrap().unwrap().revision,
        running.revision
    );
    store.finish(claim, Ok("pass done".into())).await.unwrap();
    let next = capture
        .run(store.enqueue_merge_request(&running))
        .await
        .unwrap()
        .unwrap();
    assert_ne!(next.id, task.id);
    assert_eq!(
        capture.outcomes("writer_request"),
        counts(&[("created", 2), ("deduplicated", 3), ("superseded", 1)])
    );
    assert_eq!(capture.created_legacy(), 2);
}

#[tokio::test]
#[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
async fn aggregate_admissions_report_cas_result_without_consuming_blocked_frontier() {
    let (_dir, store) = bounded_scheduler_store(vec![]).await;
    let session = store
        .discovery_session(b"admission-metrics".to_vec())
        .await
        .unwrap();
    assert!(store.elect_discovery_leader(&session).await.unwrap());
    let publisher = store.discovery_publisher(&session);
    let capture = Capture::default();
    let frontier = br#"{"v":1,"shards":{"writer":7}}"#;
    let admitted = capture
        .run(publisher.enqueue_aggregate_wal("hot", 0, frontier))
        .await
        .unwrap();
    let Admission::Admitted {
        revision,
        frontier: stored,
        task,
    } = admitted
    else {
        panic!("expected admission");
    };
    assert!(
        matches!(capture.run(publisher.enqueue_aggregate_wal("hot", revision, frontier)).await.unwrap(), Admission::Blocked {id,..} if id == task)
    );
    let mut client = store.etcd_client().clone();
    let checkpoint = client
        .get(frontier_key(&store.inner.prefix, "hot"), None)
        .await
        .unwrap();
    assert_eq!(checkpoint.kvs()[0].value(), stored);
    assert_eq!(checkpoint.kvs()[0].mod_revision(), revision);
    assert!(matches!(
        capture
            .run(publisher.enqueue_aggregate_wal("hot", 0, frontier))
            .await
            .unwrap(),
        Admission::Changed
    ));
    store.release_coordination_lock(session).await.unwrap();
    assert!(matches!(
        capture
            .run(publisher.enqueue_aggregate_wal("other", 0, frontier))
            .await
            .unwrap(),
        Admission::Changed
    ));
    assert_eq!(
        capture.outcomes("aggregate_frontier"),
        counts(&[("created", 1), ("deduplicated", 1), ("superseded", 2)])
    );
    assert_eq!(capture.created_legacy(), 1);
}

#[tokio::test]
#[ignore = "requires exclusive loopback ETCD_TEST_ENDPOINTS"]
async fn lost_aggregate_reply_is_not_reported_as_a_definitive_admission() {
    let (_dir, proxy, store, mut observer) = lease_proxy_store().await;
    let session = store
        .discovery_session(b"lost-admission-reply".to_vec())
        .await
        .unwrap();
    assert!(store.elect_discovery_leader(&session).await.unwrap());
    let publisher = store.discovery_publisher(&session);
    let capture = Capture::default();
    let key = frontier_key(&store.inner.prefix, "hot");
    proxy.responses.send(false).unwrap();
    let committed = async {
        loop {
            if !observer
                .get(key.as_str(), None)
                .await
                .unwrap()
                .kvs()
                .is_empty()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    {
        let call = capture.run(publisher.enqueue_aggregate_wal(
            "hot",
            0,
            br#"{"v":1,"shards":{"writer":7}}"#,
        ));
        tokio::pin!(call);
        tokio::select! {
            _ = &mut call => panic!("reply should be withheld"),
            result = tokio::time::timeout(Duration::from_secs(5), committed) => result.unwrap(),
        }
    }
    // The real etcd commit happened, but this caller never observed its reply.
    assert!(capture.outcomes("aggregate_frontier").is_empty());
    assert_eq!(capture.created_legacy(), 0);
    proxy.responses.send(true).unwrap();
    assert_eq!(store.list().await.unwrap().len(), 1);
    store.release_coordination_lock(session).await.unwrap();
}

#[tokio::test]
#[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
async fn recovery_admission_rejects_replaced_revision_without_erasing_new_demand() {
    let (_dir, store) = bounded_scheduler_store(vec![]).await;
    let c = store.merge_coordinator();
    let mut client = store.etcd_client().clone();
    let key = lance_context_merge::execution_key(&store.inner.prefix, "hot");
    let execution = lance_context_merge::Execution::new("hot", "worker", "old", 60);
    let value = serde_json::to_vec(&execution).unwrap();
    client.put(key.as_str(), value.clone(), None).await.unwrap();
    let old = c
        .execution_observation_page(None)
        .await
        .unwrap()
        .0
        .remove(0);
    let capture = Capture::default();
    let one = capture
        .run(store.enqueue_execution_recovery(TaskKind::MergeWal, &old))
        .await
        .unwrap()
        .unwrap();
    let two = capture
        .run(store.enqueue_execution_recovery(TaskKind::MergeWal, &old))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(one.id, two.id);
    client.put(key.as_str(), value, None).await.unwrap();
    c.request_merge("hot").await.unwrap();
    let request = c.get_merge_request("hot").await.unwrap().unwrap();
    assert!(capture
        .run(store.enqueue_execution_recovery(TaskKind::MergeWal, &old))
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        c.get_merge_request("hot").await.unwrap().unwrap().revision,
        request.revision
    );
    assert_eq!(
        store
            .get_active_id(TaskKind::MergeWal, "hot")
            .await
            .unwrap(),
        Some(one.id)
    );
    assert_eq!(
        capture.outcomes("execution_recovery"),
        counts(&[("created", 1), ("deduplicated", 1), ("superseded", 1)])
    );
    assert_eq!(capture.created_legacy(), 1);
}

#[test]
fn admission_evidence_is_logged_without_unbounded_metric_labels() {
    #[derive(Clone)]
    struct Log(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Log {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let capture = Capture::default();
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let log = Log(bytes.clone());
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_writer(move || log.clone())
        .finish();
    let row = crate::stats_store::StatRow {
        name: "private-table-name".into(),
        uri: "/unused".into(),
        row_count: 0,
        fragment_count: 0,
        last_updated: 9,
        pending_wal_generations: 11,
        last_compaction: -1,
        total_compactions: 0,
        scanned_at: 123,
        version: 7,
    };
    metrics::with_local_recorder(&capture, || {
        tracing::subscriber::with_default(subscriber, || {
            for origin in [
                Origin::Explicit,
                Origin::PeriodicStats,
                Origin::Scan,
                Origin::AggregateFrontier,
                Origin::WriterRequest,
                Origin::ExecutionRecovery,
                Origin::FailureRecovery,
                Origin::WalTail,
            ] {
                AdmissionContext::stats(origin, &row).record(
                    TaskKind::MergeWal,
                    &row.name,
                    Some("private-task-id"),
                    Outcome::Created,
                );
            }
            AdmissionContext::revision(Origin::WriterRequest, 42).record(
                TaskKind::MergeWal,
                &row.name,
                None,
                Outcome::Superseded,
            );
        })
    });
    let output = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
    for value in [
        "private-table-name",
        "private-task-id",
        "observed_base_version=7",
        "scanned_at_ms=123",
        "pending_wal_generations=11",
        "observation_revision=42",
    ] {
        assert!(output.contains(value), "missing {value}: {output}");
    }
    let entries = capture.0.lock().unwrap();
    for key in entries.keys() {
        assert!(key.labels().all(|label| !label.value().contains("private")));
        assert_eq!(
            key.labels().count(),
            if key.name() == "master_task_admissions_total" {
                3
            } else {
                1
            }
        );
    }
    assert_eq!(
        entries
            .keys()
            .filter(|key| key.name() == "master_task_admissions_total")
            .count(),
        9
    );
}

#[tokio::test]
#[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
async fn dependency_admissions_remain_distinct_and_count_each_committed_task() {
    let (_dir, store) = bounded_scheduler_store(vec![]).await;
    let capture = Capture::default();
    let dependency = store
        .enqueue(TaskKind::IndexId, "hot", vec![])
        .await
        .unwrap();
    let first = capture
        .run(store.enqueue(TaskKind::MergeWal, "hot", vec![dependency.id.clone()]))
        .await
        .unwrap();
    let second = capture
        .run(store.enqueue(TaskKind::MergeWal, "hot", vec![dependency.id.clone()]))
        .await
        .unwrap();
    assert_ne!(first.id, second.id);
    assert_eq!(
        first.depends_on.as_slice(),
        std::slice::from_ref(&dependency.id)
    );
    assert_eq!(second.depends_on, [dependency.id]);
    assert_eq!(capture.outcomes("explicit"), counts(&[("created", 2)]));
    assert_eq!(capture.created_legacy(), 2);
}

#[tokio::test]
#[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
async fn failure_admissions_keep_the_current_budget_and_count_observation_outcomes() {
    use lance_context_merge::failure::{failure_key, FailureClass, ShardFailure};
    let (_dir, store) = bounded_scheduler_store(vec![]).await;
    let c = store.merge_coordinator();
    let mut client = store.etcd_client().clone();
    let key = failure_key(&store.inner.prefix, "hot", "worker");
    let mut failure = ShardFailure {
        target: "hot".into(),
        endpoint: "worker".into(),
        consecutive_attempts: 1,
        class: FailureClass::Retryable,
        last_error: "transport failure".into(),
        last_failure_ms: 1,
        next_retry_ms: 0,
        needs_attention: false,
    };
    client
        .put(key.as_str(), serde_json::to_vec(&failure).unwrap(), None)
        .await
        .unwrap();
    let old = c
        .failure_observation_page(None, 256)
        .await
        .unwrap()
        .0
        .remove(0);
    let capture = Capture::default();
    let one = capture
        .run(store.enqueue_failure_recovery(TaskKind::MergeWal, &old))
        .await
        .unwrap()
        .unwrap();
    let two = capture
        .run(store.enqueue_failure_recovery(TaskKind::MergeWal, &old))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(one.id, two.id);
    failure.consecutive_attempts = 15;
    failure.next_retry_ms = lance_context_merge::failure::now_ms() + 60_000;
    let latest = serde_json::to_vec(&failure).unwrap();
    client
        .put(key.as_str(), latest.clone(), None)
        .await
        .unwrap();
    assert!(capture
        .run(store.enqueue_failure_recovery(TaskKind::MergeWal, &old))
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        client.get(key, None).await.unwrap().kvs()[0].value(),
        latest
    );
    assert_eq!(
        store
            .get_active_id(TaskKind::MergeWal, "hot")
            .await
            .unwrap(),
        Some(one.id)
    );
    assert_eq!(
        capture.outcomes("failure_recovery"),
        counts(&[("created", 1), ("deduplicated", 1), ("superseded", 1)])
    );
    assert_eq!(capture.created_legacy(), 1);
}
