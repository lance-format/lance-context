//! Couple real WAL receive backing to native scan admission. Releasing decoded
//! lookahead must recover the first applied record without a publisher prefix.
use super::*;
use futures::future::BoxFuture;
use lance_context_wal::replication::runtime::client::payload_memory::{Pool, Reservation};
use lance_core::memory::{NativeMemoryKind, NativeMemoryPool, NativeMemoryReservation};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug, Default)]
struct Memory {
    payload: Arc<AtomicUsize>,
    peak: AtomicUsize,
    fetches: AtomicUsize,
    limit: AtomicUsize,
    refuse: AtomicBool,
    native_refusals: AtomicUsize,
}

struct Payload(Arc<AtomicUsize>, usize);
impl Reservation for Payload {
    fn bytes(&self) -> usize {
        self.1
    }
    fn set_backed_bytes(&mut self, bytes: usize) {
        assert!(bytes <= self.1);
    }
}
impl Drop for Payload {
    fn drop(&mut self) {
        self.0.fetch_sub(self.1, Ordering::SeqCst);
    }
}
impl Pool for Memory {
    fn try_reserve(&self, bytes: usize) -> Result<Option<Box<dyn Reservation>>> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        let held = self.payload.fetch_add(bytes, Ordering::SeqCst) + bytes;
        self.peak.fetch_max(held, Ordering::SeqCst);
        Ok(Some(Box::new(Payload(self.payload.clone(), bytes))))
    }
}

#[derive(Debug)]
struct Native(usize);
impl NativeMemoryReservation for Native {
    fn bytes(&self) -> usize {
        self.0
    }
    fn set_backed_bytes(&mut self, bytes: usize) {
        assert!(bytes <= self.0);
    }
}
impl NativeMemoryPool for Memory {
    fn reserve(
        &self,
        kind: NativeMemoryKind,
        bytes: usize,
    ) -> BoxFuture<'_, lance_core::Result<Box<dyn NativeMemoryReservation>>> {
        Box::pin(async move { NativeMemoryPool::try_reserve(self, kind, bytes) })
    }
    fn try_reserve(
        &self,
        kind: NativeMemoryKind,
        bytes: usize,
    ) -> lance_core::Result<Box<dyn NativeMemoryReservation>> {
        if kind == NativeMemoryKind::Decode
            && (self.refuse.load(Ordering::SeqCst)
                || self.payload.load(Ordering::SeqCst) > self.limit.load(Ordering::SeqCst))
        {
            self.native_refusals.fetch_add(1, Ordering::SeqCst);
            return Err(lance_core::Error::backpressure("retained WAL input"));
        }
        Ok(Box::new(Native(bytes)))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn zero_prefix_pressure_releases_lookahead_and_replays_its_metadata() -> Result<()> {
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let memory = Arc::new(Memory::default());
    memory.limit.store(usize::MAX, Ordering::SeqCst);
    let storage = Storage::resolve(
        root.path().join("full-body.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default().with_native_memory(memory.clone())),
    )
    .await?;
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &fixture::limits(),
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let encode = |body: &RecordBatch| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            body,
        )?)
    };
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let (stop, task) = run_owner(runtime);
    let lane = producer(&service, 1).await?;
    let first = request(&service, &lane, 1, encode(&input[0])?).await?;
    let first_result = service.submit(&first, DEADLINE).await?;
    drained(&c, &service, first_result.index).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);

    let mut later = input[3].columns().to_vec();
    later[4] = Arc::new(Float64Array::from(vec![99.; input[3].num_rows()]));
    let later = RecordBatch::try_new(input[3].schema(), later)?;
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let mut backlog = Vec::new();
    for (i, body) in [&input[2], &input[3], &later].into_iter().enumerate() {
        let lane = producer(&service, i as u32 + 2).await?;
        let prepared = request(&service, &lane, 1, encode(body)?).await?;
        let response = c
            .client
            .request_routed(Request::Append {
                group: 10,
                batch: prepared.batch.clone(),
            })
            .await?;
        let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = response.reply else {
            bail!("backlog was not quorum durable")
        };
        backlog.push((prepared, index));
    }
    drop(runtime);
    drop(service);
    let head_bytes = backlog[0].0.body().len();
    let total_bytes = backlog.iter().map(|(p, _)| p.body().len()).sum::<usize>();
    memory.limit.store(
        backlog.iter().map(|(p, _)| p.body().len()).max().unwrap(),
        Ordering::SeqCst,
    );
    c.client.install_payload_memory(memory.clone())?;
    let (service, mut runtime) = owner(&c, &storage, &spec).await?;
    memory.refuse.store(true, Ordering::SeqCst);
    let before = service.observation().await.0.applied;
    runtime.apply_once_for_test().await?;
    assert!(memory.native_refusals.load(Ordering::SeqCst) > 0);
    assert_eq!(memory.peak.load(Ordering::SeqCst), total_bytes);
    assert_eq!(
        memory.payload.load(Ordering::SeqCst),
        head_bytes,
        "lookup refusal must release trailing decoded payloads"
    );
    assert_eq!(service.observation().await.0.applied, before);
    assert_eq!(service.source_observation().records, 0);
    assert_eq!(
        service.apply_backpressure(),
        Some(crate::annotations::ApplyBackpressure::ApplyMemory)
    );
    runtime.publish_once_for_test().await?;
    assert_eq!(service.observation().await.0.applied, before);
    // Repeated refusal must retain the head without immediately refetching the
    // released suffix. No publisher prefix exists to free these inputs for us.
    let fetches = memory.fetches.load(Ordering::SeqCst);
    for _ in 0..3 {
        runtime.apply_once_for_test().await?;
        assert_eq!(memory.fetches.load(Ordering::SeqCst), fetches);
        assert_eq!(memory.payload.load(Ordering::SeqCst), head_bytes);
        assert_eq!(service.observation().await.0.applied, before);
    }
    assert!(service
        .submit(&backlog[0].0, Duration::from_millis(50))
        .await
        .is_err());

    // Keep the payload-dependent native limit. Only the unrelated external
    // refusal ends; this same owner must drain all requeued committed records.
    memory.refuse.store(false, Ordering::SeqCst);
    let (stop, task) = run_owner(runtime);
    drained(&c, &service, backlog.last().unwrap().1).await?;
    for (prepared, index) in &backlog {
        assert_eq!(service.submit(prepared, DEADLINE).await?.index, *index);
    }
    assert_eq!(service.submit(&first, DEADLINE).await?, first_result);
    let selected = service.selected_full_body(DEADLINE).await?;
    let durable =
        annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, 6);
    for (body, row) in [(&input[0], 0), (&input[2], 2), (&input[3], 0)] {
        let key = spec.row_key(body, row, 4096)?;
        assert_eq!(
            service
                .get_full_body(&key, DEADLINE)
                .await?
                .unwrap()
                .batch(),
            &body.slice(row, 1)
        );
    }
    assert_eq!(service.retained_body_frames(), (0, 0));
    assert_eq!(memory.payload.load(Ordering::SeqCst), 0);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
