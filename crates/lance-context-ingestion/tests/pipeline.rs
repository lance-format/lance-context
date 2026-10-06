use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lance_context_ingestion::{
    Aligner, BacklogPolicy, BatchPolicy, Binding, Consumer, Entry, Error, HistoryLoader, Journal,
    Partition, PipelineConfig, Position, Reducer, Request, Result, SessionCheckpoints, Sink,
    Transition,
};

use object_store::{memory::InMemory, path::Path, ObjectStore, ObjectStoreExt};
use tokio::sync::Semaphore;
use tokio::time::timeout;

fn backlog(consumers: &[&str], max_segments: u64) -> BacklogPolicy {
    BacklogPolicy {
        consumers: consumers.iter().map(|name| (*name).into()).collect(),
        max_segments,
        poll_interval: Duration::from_millis(1),
    }
}

fn wal_entry(sequence: u64) -> Entry {
    Entry {
        sequence,
        session: "session-a".into(),
        receipt: format!("receipt-{sequence}"),
        input_digest: "digest".into(),
        transition: Transition {
            delta: serde_json::to_vec(&sequence).unwrap(),
            records: vec![],
        },
    }
}

#[tokio::test]
async fn backlog_waits_for_every_required_consumer_and_reopens_without_reset() {
    let journal = journal(Arc::new(InMemory::new()), 0);
    let mut writer = journal.acquire().await.unwrap();
    for n in 1..=3 {
        writer.append(vec![wal_entry(n)]).await.unwrap();
    }
    // Enabling the limit on an existing backlog must preserve all its data.
    let mut writer = journal
        .acquire()
        .await
        .unwrap()
        .with_backlog(backlog(&["table", "checkpoint"], 2))
        .unwrap();
    assert_eq!(
        journal.consumer_position("table").await.unwrap(),
        Position::default()
    );
    let mut append = Box::pin(writer.append(vec![wal_entry(4)]));
    assert!(timeout(Duration::from_millis(20), &mut append)
        .await
        .is_err());
    let mut table = Consumer::open(journal.clone(), "table").await.unwrap();
    let mut checkpoint = Consumer::open(journal.clone(), "checkpoint").await.unwrap();
    let mut table_sink = Collect::new();
    let mut checkpoint_sink = Collect::new();
    assert_eq!(table.consume(&mut table_sink, 3, 16384).await.unwrap(), 3);
    assert!(timeout(Duration::from_millis(20), &mut append)
        .await
        .is_err());
    assert_eq!(
        checkpoint
            .consume(&mut checkpoint_sink, 1, 16384)
            .await
            .unwrap(),
        1
    );
    assert!(timeout(Duration::from_millis(20), &mut append)
        .await
        .is_err());
    assert_eq!(
        checkpoint
            .consume(&mut checkpoint_sink, 1, 16384)
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        timeout(Duration::from_secs(2), append)
            .await
            .unwrap()
            .unwrap()
            .sequence,
        4
    );
    assert_eq!(journal.position().await.unwrap().generation, 4);
    assert_eq!(
        journal
            .consumer_position("checkpoint")
            .await
            .unwrap()
            .generation,
        2
    );
}

#[tokio::test]
async fn cancelling_or_reassigning_a_backlog_paused_writer_fences_it() {
    let journal = journal(Arc::new(InMemory::new()), 0);
    let mut writer = journal
        .acquire()
        .await
        .unwrap()
        .with_backlog(backlog(&["table"], 1))
        .unwrap();
    writer.append(vec![wal_entry(1)]).await.unwrap();
    // This timeout drops the actual append future, unlike the retained futures above.
    assert!(
        timeout(Duration::from_millis(20), writer.append(vec![wal_entry(2)]))
            .await
            .is_err()
    );
    assert!(matches!(
        writer.append(vec![wal_entry(2)]).await,
        Err(Error::Fenced)
    ));
    let mut writer = journal
        .acquire()
        .await
        .unwrap()
        .with_backlog(backlog(&["table"], 1))
        .unwrap();
    let mut append = Box::pin(writer.append(vec![wal_entry(2)]));
    assert!(timeout(Duration::from_millis(20), &mut append)
        .await
        .is_err());
    let mut replacement = journal
        .acquire()
        .await
        .unwrap()
        .with_backlog(backlog(&["table"], 1))
        .unwrap();
    assert!(matches!(
        timeout(Duration::from_secs(2), append).await.unwrap(),
        Err(Error::Fenced)
    ));
    let mut table = Consumer::open(journal.clone(), "table").await.unwrap();
    assert_eq!(
        table.consume(&mut Collect::new(), 1, 16384).await.unwrap(),
        1
    );
    replacement.append(vec![wal_entry(2)]).await.unwrap();
    assert_eq!(journal.position().await.unwrap().sequence, 2);
}

#[tokio::test]
async fn pipeline_backpressure_keeps_retries_live_and_other_partitions_independent() {
    let store = Arc::new(InMemory::new());
    let j = journal(store.clone(), 0);
    let mut cfg = config();
    cfg.wal.max_entries = 1;
    let pipeline = Partition::start(
        j.acquire()
            .await
            .unwrap()
            .with_backlog(backlog(&["table"], 1))
            .unwrap(),
        Counter::new(Arc::default()),
        cfg,
    )
    .await
    .unwrap();
    pipeline
        .enqueue(request(1))
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    let mut next = Box::pin(pipeline.enqueue(request(2)).await.unwrap().wait());
    assert!(timeout(Duration::from_millis(30), &mut next).await.is_err());
    // Already durable retries must not wait for a table consumer to catch up.
    timeout(
        Duration::from_secs(2),
        pipeline.enqueue(request(1)).await.unwrap().wait(),
    )
    .await
    .unwrap()
    .unwrap();
    let other = journal(store, 1);
    other
        .acquire()
        .await
        .unwrap()
        .with_backlog(backlog(&["table"], 1))
        .unwrap()
        .append(vec![wal_entry(1)])
        .await
        .unwrap();
    assert_eq!(j.position().await.unwrap().sequence, 1);
    let mut consumer = Consumer::open(j.clone(), "table").await.unwrap();
    consumer
        .consume(&mut Collect::new(), 1, 16384)
        .await
        .unwrap();
    timeout(Duration::from_secs(2), &mut next)
        .await
        .unwrap()
        .unwrap();
    pipeline.shutdown().await.unwrap();
    assert_eq!(j.position().await.unwrap().sequence, 2);
}

#[tokio::test]
async fn backlog_rejects_forged_cursor_even_when_its_generation_would_release_capacity() {
    let store = Arc::new(InMemory::new());
    let j = journal(store.clone(), 0);
    let mut writer = j
        .acquire()
        .await
        .unwrap()
        .with_backlog(backlog(&["table"], 1))
        .unwrap();
    let committed = writer.append(vec![wal_entry(1)]).await.unwrap();
    let forged = Position {
        segment: Some(uuid::Uuid::new_v4().to_string()),
        ..committed.clone()
    };
    let path = Path::from("run/partition-0/consumers/table.json");
    store
        .put(
            &path,
            serde_json::to_vec(&serde_json::json!({"binding":binding(0),"position":forged}))
                .unwrap()
                .into(),
        )
        .await
        .unwrap();
    assert!(matches!(
        writer.append(vec![wal_entry(2)]).await,
        Err(Error::Invalid(_))
    ));
    assert_eq!(j.position().await.unwrap(), committed);
    assert!(matches!(
        writer.append(vec![wal_entry(2)]).await,
        Err(Error::Fenced)
    ));
}

#[tokio::test]
async fn lagging_recovery_and_consumers_page_without_loading_entire_wal() {
    let store = Arc::new(InMemory::new());
    let journal = Journal::new(store, Path::from("paged"), binding(0), 1 << 20, 3).unwrap();
    let mut writer = journal.acquire().await.unwrap();
    for sequence in 1..=35 {
        writer
            .append(vec![Entry {
                sequence,
                session: "s".into(),
                receipt: format!("r-{sequence}"),
                input_digest: "digest".into(),
                transition: Transition {
                    delta: serde_json::to_vec(&sequence).unwrap(),
                    records: vec![],
                },
            }])
            .await
            .unwrap();
    }
    let head = writer.position().clone();
    let mut position = Position::default();
    let mut seen = Vec::new();
    while position != head {
        let page = journal.pending(&position, &head).await.unwrap();
        assert!(!page.is_empty() && page.len() <= 3);
        seen.extend(page.iter().map(|p| p.sequence));
        position = page.last().unwrap().clone();
    }
    assert_eq!(seen, (1..=35).collect::<Vec<_>>());
    let mut consumer = Consumer::open(journal.clone(), "table").await.unwrap();
    let mut sink = Collect::new();
    while consumer.consume(&mut sink, 2, 16384).await.unwrap() != 0 {}
    assert_eq!(sink.entries.lock().unwrap().len(), 35);
    let observed: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let pipeline = Partition::start(
        journal.acquire().await.unwrap(),
        Counter::new(observed.clone()),
        config(),
    )
    .await
    .unwrap();
    let mut next = request(36);
    next.session = "s".into();
    pipeline.enqueue(next).await.unwrap().wait().await.unwrap();
    assert_eq!(*observed.lock().unwrap(), vec![(36, 36)]);
    pipeline.shutdown().await.unwrap();
}

struct AddDelta {
    fail_b: Arc<std::sync::atomic::AtomicBool>,
    a_calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Reducer for AddDelta {
    async fn apply(&self, session: &str, state: &[u8], delta: &[u8]) -> Result<Vec<u8>> {
        if session == "b" && self.fail_b.load(Ordering::SeqCst) {
            return Err(Error::Stage("injected checkpoint failure".into()));
        }
        if session == "a" {
            self.a_calls.fetch_add(1, Ordering::SeqCst);
        }
        let value = if state.is_empty() {
            0
        } else {
            serde_json::from_slice::<u64>(state)?
        };
        Ok(serde_json::to_vec(
            &(value + serde_json::from_slice::<u64>(delta)?),
        )?)
    }
}

#[tokio::test]
async fn partial_checkpoint_batch_recovery_skips_already_applied_session_deltas() {
    let journal = journal(Arc::new(InMemory::new()), 0);
    let mut writer = journal.acquire().await.unwrap();
    for (sequence, session) in [(1, "a"), (2, "b"), (3, "a")] {
        writer
            .append(vec![Entry {
                sequence,
                session: session.into(),
                receipt: format!("r-{sequence}"),
                input_digest: "digest".into(),
                transition: Transition {
                    delta: b"1".to_vec(),
                    records: vec![],
                },
            }])
            .await
            .unwrap();
    }
    let fail_b = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let a_calls = Arc::new(AtomicUsize::new(0));
    let checkpoints = SessionCheckpoints::new(journal.clone(), 1024).unwrap();
    let mut sink = checkpoints
        .sink(
            AddDelta {
                fail_b: fail_b.clone(),
                a_calls: a_calls.clone(),
            },
            1,
        )
        .unwrap();
    let mut consumer = Consumer::open(journal.clone(), "checkpoint").await.unwrap();
    assert!(consumer.consume(&mut sink, 10, 16384).await.is_err());
    assert_eq!(consumer.position().sequence, 0);
    let a = checkpoints.load("a").await.unwrap().unwrap();
    assert_eq!(a.through_sequence, 3);
    assert_eq!(a.value, b"2");
    assert!(checkpoints.load("b").await.unwrap().is_none());
    fail_b.store(false, Ordering::SeqCst);
    let mut consumer = Consumer::open(journal.clone(), "checkpoint").await.unwrap();
    assert_eq!(consumer.consume(&mut sink, 10, 16384).await.unwrap(), 3);
    assert_eq!(consumer.position().sequence, 3);
    assert_eq!(checkpoints.load("a").await.unwrap().unwrap(), a);
    assert_eq!(
        a_calls.load(Ordering::SeqCst),
        2,
        "retry must not apply delta to a twice"
    );
    assert_eq!(checkpoints.load("b").await.unwrap().unwrap().value, b"1");
}

fn binding(partition: u32) -> Binding {
    Binding {
        run: "run-1".into(),
        schema: "test-delta-v1".into(),
        partition,
    }
}

fn journal(store: Arc<dyn ObjectStore>, partition: u32) -> Journal {
    Journal::new(
        store,
        Path::from(format!("run/partition-{partition}")),
        binding(partition),
        1 << 20,
        100,
    )
    .unwrap()
}

fn config() -> PipelineConfig {
    PipelineConfig {
        queue_entries: 4,
        load_concurrency: 4,
        memory_bytes: 1 << 20,
        max_input_bytes: 1024,
        max_transition_bytes: 1024,
        max_history_bytes: 0,
        wal: BatchPolicy {
            max_entries: 3,
            max_bytes: 16 * 1024,
            max_delay: Duration::from_millis(10),
        },
    }
}

fn request(sequence: u64) -> Request {
    Request {
        sequence,
        session: "session-a".into(),
        receipt: format!("receipt-{sequence}"),
        payload: vec![sequence as u8],
    }
}

struct Counter {
    states: BTreeMap<String, u64>,
    observed: Arc<Mutex<Vec<(u64, u64)>>>,
    align_gate: Option<Arc<Semaphore>>,
}

struct ReorderedLoads {
    first: Arc<Semaphore>,
    second_loaded: Arc<Semaphore>,
}

#[async_trait]
impl HistoryLoader for ReorderedLoads {
    async fn load(&self, request: &Request, _max_bytes: usize) -> Result<Vec<u8>> {
        if request.sequence == 1 {
            self.first.acquire().await.unwrap().forget();
        } else {
            self.second_loaded.add_permits(1);
        }
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn history_prefetch_is_concurrent_but_same_session_alignment_keeps_input_order() {
    let first = Arc::new(Semaphore::new(0));
    let second_loaded = Arc::new(Semaphore::new(0));
    let observed: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let journal = journal(Arc::new(InMemory::new()), 0);
    let pipeline = Partition::start_with_loader(
        journal.acquire().await.unwrap(),
        Counter::new(observed.clone()),
        ReorderedLoads {
            first: first.clone(),
            second_loaded: second_loaded.clone(),
        },
        config(),
    )
    .await
    .unwrap();
    let a = pipeline.enqueue(request(1)).await.unwrap();
    let b = pipeline.enqueue(request(2)).await.unwrap();
    timeout(Duration::from_secs(2), second_loaded.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    assert!(
        observed.lock().unwrap().is_empty(),
        "prefetch finishing out of order cannot reorder alignment"
    );
    first.add_permits(1);
    a.wait().await.unwrap();
    b.wait().await.unwrap();
    assert_eq!(*observed.lock().unwrap(), vec![(1, 1), (2, 2)]);
    pipeline.shutdown().await.unwrap();
}

#[tokio::test]
async fn dropping_partition_discards_uncommitted_suffix_before_recovery() {
    let journal = journal(Arc::new(InMemory::new()), 0);
    let mut aligner = Counter::new(Arc::default());
    aligner.align_gate = Some(Arc::new(Semaphore::new(0)));
    let pipeline = Partition::start(journal.acquire().await.unwrap(), aligner, config())
        .await
        .unwrap();
    let ack = pipeline.enqueue(request(1)).await.unwrap();
    drop(pipeline);
    assert!(timeout(Duration::from_secs(2), ack.wait())
        .await
        .unwrap()
        .is_err());
    assert_eq!(journal.position().await.unwrap(), Position::default());
    let observed: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let pipeline = Partition::start(
        journal.acquire().await.unwrap(),
        Counter::new(observed.clone()),
        config(),
    )
    .await
    .unwrap();
    pipeline
        .enqueue(request(1))
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(*observed.lock().unwrap(), vec![(1, 1)]);
    pipeline.shutdown().await.unwrap();
}

impl Counter {
    fn new(observed: Arc<Mutex<Vec<(u64, u64)>>>) -> Self {
        Self {
            states: BTreeMap::new(),
            observed,
            align_gate: None,
        }
    }
}

#[async_trait]
impl Aligner for Counter {
    async fn restore(&mut self, _binding: &Binding) -> Result<Position> {
        Ok(Position::default())
    }

    async fn replay(&mut self, entry: &Entry) -> Result<()> {
        self.states.insert(
            entry.session.clone(),
            serde_json::from_slice(&entry.transition.delta)?,
        );
        Ok(())
    }

    async fn align(&mut self, request: &Request, _history: &[u8]) -> Result<Transition> {
        if let Some(gate) = &self.align_gate {
            gate.acquire().await.unwrap().forget();
        }
        let state = self.states.entry(request.session.clone()).or_default();
        *state += 1;
        self.observed
            .lock()
            .unwrap()
            .push((request.sequence, *state));
        Ok(Transition {
            delta: serde_json::to_vec(state)?,
            records: request.payload.clone(),
        })
    }
}

struct Collect {
    entries: Arc<Mutex<BTreeMap<u64, Entry>>>,
    gate: Option<Arc<Semaphore>>,
    calls: Arc<AtomicUsize>,
    fail_after_apply: bool,
}

impl Collect {
    fn new() -> Self {
        Self {
            entries: Arc::default(),
            gate: None,
            calls: Arc::default(),
            fail_after_apply: false,
        }
    }
}

#[async_trait]
impl Sink for Collect {
    async fn apply(&mut self, _binding: &Binding, entries: &[Entry]) -> Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = &self.gate {
            gate.acquire().await.unwrap().forget();
        }
        for entry in entries {
            if let Some(previous) = self
                .entries
                .lock()
                .unwrap()
                .insert(entry.sequence, entry.clone())
            {
                assert_eq!(previous, *entry);
            }
        }
        if self.fail_after_apply {
            return Err(Error::Stage("injected lost sink response".into()));
        }
        Ok(())
    }
}

#[tokio::test]
async fn durable_ack_and_alignment_continue_while_checkpoint_is_blocked() {
    let journal = journal(Arc::new(InMemory::new()), 0);
    let observed = Arc::default();
    let pipeline = Partition::start(
        journal.acquire().await.unwrap(),
        Counter::new(observed),
        config(),
    )
    .await
    .unwrap();
    let first = pipeline.enqueue(request(1)).await.unwrap();
    assert_eq!(first.wait().await.unwrap().sequence, 1);

    let gate = Arc::new(Semaphore::new(0));
    let mut checkpoint = Consumer::open(journal.clone(), "checkpoint").await.unwrap();
    let mut sink = Collect::new();
    sink.gate = Some(gate.clone());
    let calls = sink.calls.clone();
    let stalled = tokio::spawn(async move {
        checkpoint.consume(&mut sink, 10, 64 * 1024).await.unwrap();
        checkpoint.position().clone()
    });
    timeout(Duration::from_secs(2), async {
        while calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    let second = pipeline.enqueue(request(2)).await.unwrap();
    assert_eq!(
        timeout(Duration::from_secs(2), second.wait())
            .await
            .unwrap()
            .unwrap()
            .sequence,
        2
    );
    assert!(!stalled.is_finished());
    let mut merge = Consumer::open(journal.clone(), "table").await.unwrap();
    let mut table = Collect::new();
    assert_eq!(merge.consume(&mut table, 10, 64 * 1024).await.unwrap(), 2);
    assert_eq!(
        table.calls.load(Ordering::SeqCst),
        1,
        "merge independently coalesces two WAL batches"
    );
    assert_eq!(merge.position().sequence, 2);
    gate.add_permits(1);
    assert_eq!(stalled.await.unwrap().sequence, 1);
    pipeline.shutdown().await.unwrap();
}

#[tokio::test]
async fn restart_replays_committed_suffix_and_retry_does_not_align_twice() {
    let journal = journal(Arc::new(InMemory::new()), 0);
    let observed = Arc::default();
    let pipeline = Partition::start(
        journal.acquire().await.unwrap(),
        Counter::new(observed),
        config(),
    )
    .await
    .unwrap();
    let ack = pipeline.enqueue(request(1)).await.unwrap();
    drop(ack); // client disappeared; admitted work still commits
    pipeline.shutdown().await.unwrap();
    assert_eq!(journal.position().await.unwrap().sequence, 1);

    let observed: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let pipeline = Partition::start(
        journal.acquire().await.unwrap(),
        Counter::new(observed.clone()),
        config(),
    )
    .await
    .unwrap();
    assert_eq!(
        pipeline
            .enqueue(request(1))
            .await
            .unwrap()
            .wait()
            .await
            .unwrap()
            .sequence,
        1
    );
    let mut changed = request(1);
    changed.payload = vec![99];
    assert!(pipeline
        .enqueue(changed)
        .await
        .unwrap()
        .wait()
        .await
        .is_err());
    assert!(observed.lock().unwrap().is_empty());
    assert!(
        pipeline
            .enqueue(request(3))
            .await
            .unwrap()
            .wait()
            .await
            .is_err(),
        "cannot skip missing source receipt"
    );
    assert_eq!(
        pipeline
            .enqueue(request(2))
            .await
            .unwrap()
            .wait()
            .await
            .unwrap()
            .sequence,
        2
    );
    assert_eq!(
        *observed.lock().unwrap(),
        vec![(2, 2)],
        "restored state determines next aligned ID"
    );
    pipeline.shutdown().await.unwrap();
}

#[tokio::test]
async fn independent_partition_progresses_while_another_alignment_is_stalled() {
    let store = Arc::new(InMemory::new());
    let gate = Arc::new(Semaphore::new(0));
    let mut slow = Counter::new(Arc::default());
    slow.align_gate = Some(gate.clone());
    let a = Partition::start(
        journal(store.clone(), 0).acquire().await.unwrap(),
        slow,
        config(),
    )
    .await
    .unwrap();
    let b = Partition::start(
        journal(store, 1).acquire().await.unwrap(),
        Counter::new(Arc::default()),
        config(),
    )
    .await
    .unwrap();
    let blocked = a.enqueue(request(1)).await.unwrap();
    let fast = b.enqueue(request(1)).await.unwrap();
    assert_eq!(
        timeout(Duration::from_secs(2), fast.wait())
            .await
            .unwrap()
            .unwrap()
            .sequence,
        1
    );
    assert_eq!(a.durable_position().sequence, 0);
    gate.add_permits(1);
    assert_eq!(blocked.wait().await.unwrap().sequence, 1);
    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}

#[tokio::test]
async fn stale_writer_upload_is_not_recoverable_or_acknowledged() {
    let journal = journal(Arc::new(InMemory::new()), 0);
    let mut old = journal.acquire().await.unwrap();
    let mut new = journal.acquire().await.unwrap();
    let entry = Entry {
        sequence: 1,
        session: "s".into(),
        receipt: "r".into(),
        input_digest: "hash".into(),
        transition: Transition {
            delta: vec![1],
            records: vec![2],
        },
    };
    assert!(old.append(vec![entry.clone()]).await.is_err());
    assert!(matches!(
        old.append(vec![entry.clone()]).await,
        Err(Error::Fenced)
    ));
    assert_eq!(journal.position().await.unwrap().sequence, 0);
    let committed = new.append(vec![entry.clone()]).await.unwrap();
    let positions = journal
        .pending(&Position::default(), &committed)
        .await
        .unwrap();
    assert_eq!(positions, vec![committed.clone()]);
    assert_eq!(journal.entries(&committed).await.unwrap(), vec![entry]);
}

#[tokio::test]
async fn consumer_uncertain_apply_replays_without_skipping_or_duplicating_output() {
    let journal = journal(Arc::new(InMemory::new()), 0);
    let pipeline = Partition::start(
        journal.acquire().await.unwrap(),
        Counter::new(Arc::default()),
        config(),
    )
    .await
    .unwrap();
    pipeline
        .enqueue(request(1))
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    let mut consumer = Consumer::open(journal.clone(), "table").await.unwrap();
    let mut sink = Collect::new();
    sink.fail_after_apply = true;
    assert!(consumer.consume(&mut sink, 5, 32 * 1024).await.is_err());
    assert_eq!(consumer.position().sequence, 0);
    assert!(matches!(
        consumer.consume(&mut sink, 5, 32 * 1024).await,
        Err(Error::Fenced)
    ));
    let mut recovered = Consumer::open(journal.clone(), "table").await.unwrap();
    assert_eq!(recovered.position().sequence, 0);
    sink.fail_after_apply = false;
    recovered.consume(&mut sink, 1, 32 * 1024).await.unwrap();
    assert_eq!(sink.entries.lock().unwrap().len(), 1);
    assert_eq!(sink.calls.load(Ordering::SeqCst), 2);
    pipeline.shutdown().await.unwrap();
}

#[tokio::test]
async fn byte_admission_is_bounded_and_sparse_wal_flushes_on_timer() {
    let mut cfg = config();
    // Exactly one maximum-sized input/output reservation fits.
    cfg.memory_bytes = ((cfg.max_input_bytes + cfg.max_transition_bytes) * 16 + 4096) as u32;
    cfg.wal.max_delay = Duration::from_millis(50);
    let gate = Arc::new(Semaphore::new(0));
    let mut aligner = Counter::new(Arc::default());
    aligner.align_gate = Some(gate.clone());
    let pipeline = Partition::start(
        journal(Arc::new(InMemory::new()), 0)
            .acquire()
            .await
            .unwrap(),
        aligner,
        cfg,
    )
    .await
    .unwrap();
    let first = pipeline.enqueue(request(1)).await.unwrap();
    assert!(
        timeout(Duration::from_millis(30), pipeline.enqueue(request(2)))
            .await
            .is_err()
    );
    gate.add_permits(1);
    assert_eq!(
        timeout(Duration::from_secs(2), first.wait())
            .await
            .unwrap()
            .unwrap()
            .sequence,
        1
    );
    let second = pipeline.enqueue(request(2)).await.unwrap();
    gate.add_permits(1);
    assert_eq!(second.wait().await.unwrap().sequence, 2);
    pipeline.shutdown().await.unwrap();
}

#[tokio::test]
async fn corrupt_committed_segment_fails_recovery_and_binding_cannot_change() {
    let store = Arc::new(InMemory::new());
    let journal = journal(store.clone(), 0);
    let pipeline = Partition::start(
        journal.acquire().await.unwrap(),
        Counter::new(Arc::default()),
        config(),
    )
    .await
    .unwrap();
    let position = pipeline
        .enqueue(request(1))
        .await
        .unwrap()
        .wait()
        .await
        .unwrap();
    pipeline.shutdown().await.unwrap();
    let wrong = Journal::new(
        store.clone(),
        Path::from("run/partition-0"),
        binding(1),
        1 << 20,
        100,
    )
    .unwrap();
    assert!(wrong.acquire().await.is_err());
    let segment = Path::from(format!(
        "run/partition-0/segments/{}.json",
        position.segment.unwrap()
    ));
    store
        .put(&segment, b"corrupt".to_vec().into())
        .await
        .unwrap();
    assert!(Partition::start(
        journal.acquire().await.unwrap(),
        Counter::new(Arc::default()),
        config()
    )
    .await
    .is_err());
}
