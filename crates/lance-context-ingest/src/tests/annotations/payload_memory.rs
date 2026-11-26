use super::*;
use lance_context_wal::replication::runtime::client::payload_memory::{
    Pool, Reservation, Unavailable,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Default)]
struct Admission {
    refuse: AtomicBool,
    held: Arc<AtomicUsize>,
}

struct Charge(Arc<AtomicUsize>, usize);

impl Reservation for Charge {
    fn bytes(&self) -> usize {
        self.1
    }

    fn set_backed_bytes(&mut self, bytes: usize) {
        assert!(bytes <= self.1);
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.0.fetch_sub(self.1, Ordering::SeqCst);
    }
}

impl Pool for Admission {
    fn try_reserve(&self, bytes: usize) -> Result<Option<Box<dyn Reservation>>> {
        if self.refuse.load(Ordering::SeqCst) {
            return Ok(None);
        }
        self.held.fetch_add(bytes, Ordering::SeqCst);
        Ok(Some(Box::new(Charge(self.held.clone(), bytes))))
    }
}

#[tokio::test]
async fn upload_backing_admission_survives_apply_and_detached_arrow_views() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits::default();
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let memory = Arc::new(Admission::default());
    c.client.install_payload_memory(memory.clone())?;
    let (service, driver) = connect(&c, &storage, &limits).await?;
    let producer = producer(&service, 1).await?;
    let (stop, task) = run(driver);
    let bytes = wire(&limits, &["T"], 1, Some("kept"), 1, Some(0.9))?;
    memory.refuse.store(true, Ordering::SeqCst);
    let refused = service
        .prepare_owned(
            service.reserve_upload(bytes.len()).await?,
            bytes.to_vec(),
            &producer,
            1,
        )
        .await;
    assert!(matches!(refused, Err(ref error) if error.is::<Unavailable>()));
    assert_eq!(memory.held.load(Ordering::SeqCst), 0);
    assert_eq!(service.source_observation().records, 0);
    memory.refuse.store(false, Ordering::SeqCst);

    let mut owned = bytes.to_vec();
    owned.reserve_exact(257);
    let capacity = owned.capacity();
    let pointer = owned.as_ptr();
    let prepared = service
        .prepare_owned(
            service.reserve_upload(owned.len()).await?,
            owned,
            &producer,
            1,
        )
        .await?;
    assert_eq!(prepared.body().as_ptr(), pointer);
    assert_eq!(memory.held.load(Ordering::SeqCst), capacity);
    let decoded = ipc::decode(prepared.body().clone(), annotations::wire::schema(&limits))?;
    let retained = prepared.body().slice(..1);
    let result = service.submit(&prepared, Duration::from_secs(5)).await?;
    assert_eq!(result.result, MutationResult::Applied);
    let value = service.get("T", Duration::from_secs(5)).await?.unwrap();
    assert_eq!(value.status.as_deref(), Some("kept"));
    assert_eq!(value.score, Some(0.9));
    drop(prepared);
    drop(retained);
    assert_eq!(memory.held.load(Ordering::SeqCst), capacity);
    drop(decoded);
    assert_eq!(memory.held.load(Ordering::SeqCst), 0);

    // A slice cannot prove its parent's allocation capacity. Preparation owns
    // and charges one bounded copy, without retaining the larger parent.
    let mut parent = vec![0; 4096];
    parent.extend_from_slice(&bytes);
    parent.extend_from_slice(&[0; 4096]);
    let parent = Bytes::from(parent);
    let input = parent.slice(4096..4096 + bytes.len());
    let pointer = input.as_ptr();
    let borrowed = service
        .prepare(
            service.reserve_upload(input.len()).await?,
            input,
            &producer,
            2,
        )
        .await?;
    assert_ne!(borrowed.body().as_ptr(), pointer);
    assert_eq!(borrowed.body(), &bytes);
    assert_eq!(memory.held.load(Ordering::SeqCst), bytes.len());
    drop(borrowed);
    assert_eq!(memory.held.load(Ordering::SeqCst), 0);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
