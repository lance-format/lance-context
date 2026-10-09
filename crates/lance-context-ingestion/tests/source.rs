use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use lance_context_ingestion::{
    Aligner, BatchPolicy, Binding, Entry, HistoryLoader, Journal, PipelineConfig, Position,
    Request, Result, SourceConfig, SourcePartition, SourceRequest, Transition,
};
use object_store::{memory::InMemory, path::Path};
use tokio::{sync::Semaphore, time::timeout};

struct EmptyHistory;
#[async_trait]
impl HistoryLoader for EmptyHistory {
    async fn load(&self, _: &Request, _: usize) -> Result<Vec<u8>> {
        Ok(vec![])
    }
}

struct Counter {
    states: HashMap<String, u64>,
    observed: Arc<Mutex<Vec<String>>>,
    slow: Arc<Semaphore>,
    fast_done: Arc<Semaphore>,
}
#[async_trait]
impl Aligner for Counter {
    async fn restore(&mut self, _: &Binding) -> Result<Position> {
        Ok(Position::default())
    }
    async fn replay(&mut self, entry: &Entry) -> Result<()> {
        self.states.insert(
            entry.session.clone(),
            serde_json::from_slice(&entry.transition.delta)?,
        );
        Ok(())
    }
    async fn align(&mut self, request: &Request, _: &[u8]) -> Result<Transition> {
        if request.session == "slow" {
            self.slow.acquire().await.unwrap().forget();
        }
        let count = self.states.entry(request.session.clone()).or_default();
        *count += 1;
        self.observed.lock().unwrap().push(request.receipt.clone());
        if request.session == "fast" {
            self.fast_done.add_permits(1);
        }
        Ok(Transition {
            delta: serde_json::to_vec(count)?,
            records: request.receipt.as_bytes().to_vec(),
        })
    }
}

fn journal() -> Journal {
    Journal::new(
        Arc::new(InMemory::new()),
        Path::from("source"),
        Binding {
            run: "source-run".into(),
            schema: "counter".into(),
            partition: 0,
        },
        1 << 20,
        2,
    )
    .unwrap()
}

fn config() -> PipelineConfig {
    PipelineConfig {
        queue_entries: 8,
        load_concurrency: 4,
        memory_bytes: 1 << 20,
        max_input_bytes: 1024,
        max_transition_bytes: 1024,
        max_history_bytes: 0,
        wal: BatchPolicy {
            max_entries: 8,
            max_bytes: 16 << 10,
            max_delay: Duration::from_millis(2),
        },
    }
}

async fn start(
    journal: &Journal,
    observed: Arc<Mutex<Vec<String>>>,
    slow: Arc<Semaphore>,
    fast_done: Arc<Semaphore>,
    config: PipelineConfig,
) -> SourcePartition {
    SourcePartition::start(
        journal.acquire().await.unwrap(),
        (0..2)
            .map(|_| Counter {
                states: HashMap::new(),
                observed: observed.clone(),
                slow: slow.clone(),
                fast_done: fast_done.clone(),
            })
            .collect(),
        EmptyHistory,
        config,
        SourceConfig {
            max_batch_requests: 16,
            max_batch_bytes: 16 << 10,
            receipt_read_concurrency: 2,
        },
    )
    .await
    .unwrap()
}

fn request(receipt: &str, session: &str) -> SourceRequest {
    SourceRequest {
        receipt: receipt.into(),
        session: session.into(),
        payload: receipt.as_bytes().to_vec(),
    }
}

#[tokio::test]
async fn in_flight_source_retries_do_not_realign_or_block_later_dispatch() {
    let journal = journal();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let slow = Arc::new(Semaphore::new(0));
    let fast_done = Arc::new(Semaphore::new(0));
    let mut source = start(
        &journal,
        observed.clone(),
        slow.clone(),
        fast_done.clone(),
        config(),
    )
    .await;
    let a = request("a", "slow");
    let first = source
        .enqueue_many(vec![a.clone(), a.clone(), request("b", "fast")])
        .await
        .unwrap();
    timeout(Duration::from_secs(2), fast_done.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    let second = source
        .enqueue_many(vec![a.clone(), request("c", "fast")])
        .await
        .unwrap();
    timeout(Duration::from_secs(2), fast_done.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    assert_eq!(journal.position().await.unwrap().sequence, 0);
    let mut changed = a;
    changed.payload.push(9);
    assert!(source.enqueue_many(vec![changed]).await.is_err());
    slow.add_permits(1);
    let mut sequences = Vec::new();
    for ack in first.into_iter().chain(second) {
        let commit = timeout(Duration::from_secs(2), ack.wait())
            .await
            .unwrap()
            .unwrap();
        assert!(commit.through.sequence >= commit.sequence);
        sequences.push(commit.sequence);
    }
    assert_eq!(sequences, [1, 1, 2, 1, 3]);
    source.shutdown().await.unwrap();
    let mut aligned = observed.lock().unwrap().clone();
    aligned.sort();
    assert_eq!(aligned, ["a", "b", "c"]);
    assert_eq!(journal.position().await.unwrap().sequence, 3);
}

#[tokio::test]
async fn lost_response_and_restart_recover_receipts_without_realigning_or_resetting_sequences() {
    let journal = journal();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let slow = Arc::new(Semaphore::new(0));
    let fast_done = Arc::new(Semaphore::new(0));
    let mut source = start(
        &journal,
        observed.clone(),
        slow.clone(),
        fast_done.clone(),
        config(),
    )
    .await;
    let original = request("a", "fast");
    drop(source.enqueue_many(vec![original.clone()]).await.unwrap());
    source.shutdown().await.unwrap();
    let before = journal.position().await.unwrap();
    let mut source = start(&journal, observed.clone(), slow, fast_done, config()).await;
    let mut changed = original.clone();
    changed.session = "different-session".into();
    assert!(source.enqueue_many(vec![changed]).await.is_err());
    assert_eq!(journal.position().await.unwrap(), before);
    let acks = source
        .enqueue_many(vec![original, request("b", "fast")])
        .await
        .unwrap();
    let mut sequences = Vec::new();
    for ack in acks {
        sequences.push(ack.wait().await.unwrap().sequence);
    }
    assert_eq!(sequences, [1, 2]);
    let mut consumer = source.open_receipt_consumer().await.unwrap();
    let mut sink = source.receipt_index().sink(2).unwrap();
    while consumer.consume(&mut sink, 2, 1 << 20).await.unwrap() != 0 {}
    assert_eq!(consumer.position().sequence, 2);
    let retry = source
        .enqueue_many(vec![request("a", "fast")])
        .await
        .unwrap();
    assert_eq!(
        retry
            .into_iter()
            .next()
            .unwrap()
            .wait()
            .await
            .unwrap()
            .sequence,
        1
    );
    source.shutdown().await.unwrap();
    assert_eq!(*observed.lock().unwrap(), ["a", "b"]);
    assert_eq!(journal.position().await.unwrap().sequence, 2);
}

#[tokio::test]
async fn cancelled_partial_admission_requires_recovery_and_retries_exactly_once() {
    let journal = journal();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let slow = Arc::new(Semaphore::new(0));
    let fast_done = Arc::new(Semaphore::new(0));
    let mut limited = config();
    limited.memory_bytes = 60_000;
    let mut source = start(
        &journal,
        observed.clone(),
        slow.clone(),
        fast_done.clone(),
        limited,
    )
    .await;
    let first = request("a", "slow");
    drop(source.enqueue_many(vec![first.clone()]).await.unwrap());
    let later = (0..10)
        .map(|n| request(&format!("later-{n}"), "fast"))
        .collect::<Vec<_>>();
    let mut admission = Box::pin(source.enqueue_many(later.clone()));
    timeout(Duration::from_secs(2), async {
        tokio::select! {
            _ = &mut admission => panic!("admission must wait for bounded capacity"),
            ready = fast_done.acquire() => ready.unwrap().forget(),
        }
    })
    .await
    .unwrap();
    assert!(timeout(Duration::from_millis(20), &mut admission)
        .await
        .is_err());
    drop(admission);
    assert!(source.enqueue_many(vec![first.clone()]).await.is_err());
    slow.add_permits(1);
    timeout(Duration::from_secs(2), source.shutdown())
        .await
        .unwrap()
        .unwrap();
    let prefix = journal.position().await.unwrap();
    assert!(prefix.sequence > 1 && prefix.sequence < 11);
    let mut source = start(&journal, observed.clone(), slow, fast_done, config()).await;
    let acks = source
        .enqueue_many(std::iter::once(first).chain(later).collect())
        .await
        .unwrap();
    for (n, ack) in acks.into_iter().enumerate() {
        assert_eq!(ack.wait().await.unwrap().sequence, n as u64 + 1);
    }
    source.shutdown().await.unwrap();
    assert_eq!(journal.position().await.unwrap().sequence, 11);
    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), 11);
    let mut unique = observed.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 11);
}

#[tokio::test]
async fn invalid_batch_cannot_allocate_a_sequence_or_poison_valid_admission() {
    let journal = journal();
    let mut source = start(
        &journal,
        Arc::default(),
        Arc::new(Semaphore::new(0)),
        Arc::new(Semaphore::new(0)),
        config(),
    )
    .await;
    let original = request("a", "fast");
    let mut changed = original.clone();
    changed.payload.push(1);
    assert!(source
        .enqueue_many(vec![original.clone(), changed])
        .await
        .is_err());
    let mut oversized = original.clone();
    oversized.payload = vec![0; 1025];
    assert!(source.enqueue_many(vec![oversized]).await.is_err());
    assert!(source.enqueue_many(Vec::new()).await.is_err());
    assert_eq!(journal.position().await.unwrap().sequence, 0);
    let ack = source
        .enqueue_many(vec![original])
        .await
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(ack.wait().await.unwrap().sequence, 1);
    source.shutdown().await.unwrap();
}

struct Failing {
    release: Arc<Semaphore>,
    fail_wal: bool,
}

#[async_trait]
impl Aligner for Failing {
    async fn restore(&mut self, _: &Binding) -> Result<Position> {
        Ok(Position::default())
    }
    async fn replay(&mut self, _: &Entry) -> Result<()> {
        Ok(())
    }
    async fn align(&mut self, _: &Request, _: &[u8]) -> Result<Transition> {
        self.release.acquire().await.unwrap().forget();
        if self.fail_wal {
            // Fits the transition budget but exceeds the journal object budget.
            Ok(Transition {
                delta: vec![1],
                records: vec![1; 256],
            })
        } else {
            Err(lance_context_ingestion::Error::Stage(
                "injected alignment failure".into(),
            ))
        }
    }
}

#[tokio::test]
async fn original_and_duplicate_waiters_fail_together_on_alignment_or_wal_error() {
    for fail_wal in [false, true] {
        let journal = Journal::new(
            Arc::new(InMemory::new()),
            Path::from("failure"),
            Binding {
                run: "run".into(),
                schema: "counter".into(),
                partition: 0,
            },
            512,
            2,
        )
        .unwrap();
        let release = Arc::new(Semaphore::new(0));
        let mut source = SourcePartition::start(
            journal.acquire().await.unwrap(),
            vec![Failing {
                release: release.clone(),
                fail_wal,
            }],
            EmptyHistory,
            config(),
            SourceConfig {
                max_batch_requests: 4,
                max_batch_bytes: 16 << 10,
                receipt_read_concurrency: 2,
            },
        )
        .await
        .unwrap();
        let input = request("original", "session");
        let acks = source
            .enqueue_many(vec![input.clone(), input.clone()])
            .await
            .unwrap();
        release.add_permits(1);
        for ack in acks {
            assert!(timeout(Duration::from_secs(2), ack.wait())
                .await
                .unwrap()
                .is_err());
        }
        assert!(source.enqueue_many(vec![input]).await.is_err());
        assert!(timeout(Duration::from_secs(2), source.shutdown())
            .await
            .unwrap()
            .is_err());
        assert_eq!(journal.position().await.unwrap().sequence, 0);
    }
}

async fn start_bulk(
    journal: &Journal,
    observed: Arc<Mutex<Vec<String>>>,
    slow: Arc<Semaphore>,
    fast_done: Arc<Semaphore>,
    pipeline: PipelineConfig,
) -> SourcePartition {
    SourcePartition::start_batched(
        journal.acquire().await.unwrap(),
        (0..2)
            .map(|_| Counter {
                states: HashMap::new(),
                observed: observed.clone(),
                slow: slow.clone(),
                fast_done: fast_done.clone(),
            })
            .collect(),
        EmptyHistory,
        pipeline,
        SourceConfig {
            max_batch_requests: 256,
            max_batch_bytes: 64 << 10,
            receipt_read_concurrency: 2,
        },
        lance_context_ingestion::BatchFlush::new(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn bulk_batch_preserves_pending_dedup_and_restarts_without_checkpoint_or_realigning() {
    let journal = journal();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let slow = Arc::new(Semaphore::new(0));
    let fast = Arc::new(Semaphore::new(0));
    let mut pipeline = config();
    pipeline.wal.max_entries = 128;
    let mut source = start_bulk(
        &journal,
        observed.clone(),
        slow.clone(),
        fast.clone(),
        pipeline.clone(),
    )
    .await;
    let mut requests: Vec<_> = (0..39)
        .map(|i| request(&format!("batch-{i}"), "fast"))
        .collect();
    requests.push(request("batch-last", "slow"));
    requests.push(request("batch-0", "fast")); // Duplicate within the same uncommitted batch.
    let mut acks = source.enqueue_many(requests.clone()).await.unwrap();
    timeout(Duration::from_secs(5), fast.acquire_many(39))
        .await
        .unwrap()
        .unwrap()
        .forget();
    let first = acks.remove(0).wait();
    tokio::pin!(first);
    // The 2ms streaming timer must not split this batch while its final call is
    // still aligning. All earlier calls have actually finished alignment.
    assert!(timeout(Duration::from_millis(30), &mut first)
        .await
        .is_err());
    assert_eq!(journal.position().await.unwrap(), Position::default());
    slow.add_permits(1);
    timeout(Duration::from_secs(5), &mut first)
        .await
        .unwrap()
        .unwrap();
    for ack in acks {
        ack.wait().await.unwrap();
    }
    let head = journal.position().await.unwrap();
    assert_eq!((head.sequence, head.generation), (40, 1));
    let entries = journal.entries(&head).await.unwrap();
    assert_eq!(entries.len(), 40);
    for (index, entry) in entries[..39].iter().enumerate() {
        assert_eq!(
            serde_json::from_slice::<u64>(&entry.transition.delta).unwrap(),
            index as u64 + 1
        );
    }
    // Stop after a durable ACK without checkpoint/receipt consumers. Recovery
    // must use the committed WAL, including same-batch source identity mappings.
    drop(source);
    let mut source = start_bulk(&journal, observed.clone(), slow, fast, pipeline).await;
    for ack in source.enqueue_many(requests).await.unwrap() {
        ack.wait().await.unwrap();
    }
    assert_eq!(journal.position().await.unwrap(), head);
    assert_eq!(observed.lock().unwrap().len(), 40);
    let mut changed = request("batch-0", "fast");
    changed.payload = b"different contents".to_vec();
    assert!(source.enqueue_many(vec![changed]).await.is_err());
    for ack in source
        .enqueue_many(vec![request("next", "fast")])
        .await
        .unwrap()
    {
        ack.wait().await.unwrap();
    }
    let last = journal.position().await.unwrap();
    assert_eq!(last.sequence, 41);
    assert_eq!(
        serde_json::from_slice::<u64>(&journal.entries(&last).await.unwrap()[0].transition.delta)
            .unwrap(),
        40
    );
    source.shutdown().await.unwrap();
}

#[tokio::test]
async fn bulk_batch_splits_at_size_or_memory_headroom_without_waiting_for_a_timer() {
    for limit in ["entries", "bytes", "memory"] {
        let journal = journal();
        let mut pipeline = config();
        pipeline.wal.max_delay = Duration::from_secs(3600);
        pipeline.wal.max_entries = if limit == "entries" { 7 } else { 128 };
        if limit == "memory" {
            pipeline.memory_bytes = pipeline.reservation_bytes().unwrap() as usize;
        }
        if limit == "bytes" {
            pipeline.wal.max_bytes = 1024;
        }
        let mut source = start_bulk(
            &journal,
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Semaphore::new(0)),
            Arc::new(Semaphore::new(0)),
            pipeline,
        )
        .await;
        let requests = (0..40)
            .map(|i| request(&format!("row-{i}"), "fast"))
            .collect();
        timeout(Duration::from_secs(5), async {
            for ack in source.enqueue_many(requests).await.unwrap() {
                ack.wait().await.unwrap();
            }
        })
        .await
        .unwrap();
        let head = journal.position().await.unwrap();
        assert_eq!(head.sequence, 40);
        assert!(head.generation > 1);
        if limit == "entries" {
            assert_eq!(head.generation, 6);
        }
        let mut cursor = Position::default();
        let mut entries = Vec::new();
        while cursor != head {
            for position in journal.pending(&cursor, &head).await.unwrap() {
                entries.extend(journal.entries(&position).await.unwrap());
                cursor = position;
            }
        }
        assert_eq!(entries.len(), 40);
        assert!(entries
            .iter()
            .enumerate()
            .all(|(i, e)| e.sequence == i as u64 + 1));
        source.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn bulk_recovery_prefix_is_not_hidden_by_a_later_batch_boundary() {
    let journal = journal();
    let slow = Arc::new(Semaphore::new(0));
    let fast = Arc::new(Semaphore::new(0));
    let flush = lance_context_ingestion::BatchFlush::new();
    let mut source = SourcePartition::start_batched(
        journal.acquire().await.unwrap(),
        vec![Counter {
            states: HashMap::new(),
            observed: Arc::new(Mutex::new(Vec::new())),
            slow: slow.clone(),
            fast_done: fast.clone(),
        }],
        EmptyHistory,
        config(),
        SourceConfig {
            max_batch_requests: 16,
            max_batch_bytes: 16384,
            receipt_read_concurrency: 2,
        },
        flush.clone(),
    )
    .await
    .unwrap();
    let mut acks = source
        .enqueue_many(vec![request("first", "fast"), request("last", "slow")])
        .await
        .unwrap();
    fast.acquire().await.unwrap().forget();
    // enqueue_many already requested the full batch (sequence 2). The pending
    // alignment needs sequence 1 committed before it can recover evicted state.
    flush.request_prefix(1);
    timeout(Duration::from_secs(2), acks.remove(0).wait())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(journal.position().await.unwrap().sequence, 1);
    let last = acks.remove(0).wait();
    tokio::pin!(last);
    assert!(timeout(Duration::from_millis(20), &mut last).await.is_err());
    slow.add_permits(1);
    timeout(Duration::from_secs(2), &mut last)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(journal.position().await.unwrap().sequence, 2);
    source.shutdown().await.unwrap();
}

#[tokio::test]
async fn bulk_cancelled_partial_admission_recovers_without_duplicate_rows() {
    let journal = journal();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let slow = Arc::new(Semaphore::new(0));
    let fast_done = Arc::new(Semaphore::new(0));
    let mut limited = config();
    limited.memory_bytes = 60_000;
    let mut source = start_bulk(
        &journal,
        observed.clone(),
        slow.clone(),
        fast_done.clone(),
        limited,
    )
    .await;
    let first = request("a", "slow");
    drop(source.enqueue_many(vec![first.clone()]).await.unwrap());
    let later = (0..10)
        .map(|n| request(&format!("later-{n}"), "fast"))
        .collect::<Vec<_>>();
    let mut admission = Box::pin(source.enqueue_many(later.clone()));
    timeout(Duration::from_secs(2), async {
        tokio::select! {
            _ = &mut admission => panic!("admission must wait for bounded capacity"),
            ready = fast_done.acquire() => ready.unwrap().forget(),
        }
    })
    .await
    .unwrap();
    assert!(timeout(Duration::from_millis(20), &mut admission)
        .await
        .is_err());
    drop(admission);
    assert!(source.enqueue_many(vec![first.clone()]).await.is_err());
    slow.add_permits(1);
    timeout(Duration::from_secs(2), source.shutdown())
        .await
        .unwrap()
        .unwrap();
    let prefix = journal.position().await.unwrap();
    assert!(prefix.sequence > 1 && prefix.sequence < 11);
    let mut source = start_bulk(&journal, observed.clone(), slow, fast_done, config()).await;
    let acks = source
        .enqueue_many(std::iter::once(first).chain(later).collect())
        .await
        .unwrap();
    for (n, ack) in acks.into_iter().enumerate() {
        assert_eq!(ack.wait().await.unwrap().sequence, n as u64 + 1);
    }
    source.shutdown().await.unwrap();
    assert_eq!(journal.position().await.unwrap().sequence, 11);
    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), 11);
    let mut unique = observed.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 11);
}
