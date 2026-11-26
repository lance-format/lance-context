use super::*;
use lance_context_wal::replication::runtime::client::payload_memory::{Pool, Reservation};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Default)]
struct Counts {
    allow: AtomicBool,
    attempts: AtomicUsize,
    held: AtomicUsize,
}
struct Admission(Arc<Counts>);
struct Charge(Arc<Counts>, usize);
impl Pool for Admission {
    fn try_reserve(&self, bytes: usize) -> Result<Option<Box<dyn Reservation>>> {
        self.0.attempts.fetch_add(1, Ordering::SeqCst);
        if !self.0.allow.load(Ordering::SeqCst) {
            return Ok(None);
        }
        self.0.held.fetch_add(bytes, Ordering::SeqCst);
        Ok(Some(Box::new(Charge(self.0.clone(), bytes))))
    }
}
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
        self.0.held.fetch_sub(self.1, Ordering::SeqCst);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payload_memory_pressure_preserves_acked_replay_and_reader_charge() -> Result<()> {
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = fixture::storage(root.path(), "full-body.lance").await?;
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
    let (service, runtime) = owner_with_wait(&c, &storage, &spec, Duration::from_secs(60)).await?;
    let (stop, task) = run_owner(runtime);
    let lane = producer(&service, 1).await?;
    let payload = fixture::encoded(single_body::batch(
        &spec,
        &fixture::limits(),
        Shard { bucket: 0 },
        &input[0],
    )?)?;
    let bytes = payload.len();
    let append = request(&service, &lane, 1, payload).await?;
    let receipt = service.submit(&append, DEADLINE).await?;
    assert_eq!(receipt.result, MutationResult::Applied);
    assert!(service.observation().await.0.materialized < receipt.index);
    stop.send(()).unwrap();
    task.await??;
    drop(service);

    let counts = Arc::new(Counts::default());
    c.client
        .install_payload_memory(Arc::new(Admission(counts.clone())))?;
    let (service, mut runtime) =
        owner_with_wait(&c, &storage, &spec, Duration::from_secs(60)).await?;
    tokio::time::timeout(DEADLINE, async {
        while counts.attempts.load(Ordering::SeqCst) == 0 {
            runtime.apply_once_for_test().await?;
            tokio::task::yield_now().await;
        }
        anyhow::Ok(())
    })
    .await??;
    assert_eq!(
        service.apply_backpressure(),
        Some(crate::annotations::ApplyBackpressure::ApplyMemory)
    );
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    counts.allow.store(true, Ordering::SeqCst);
    let key = spec.row_key(&input[0], 0, 4096)?;
    // Apply without running publication so the read owns a recovered WAL frame.
    let held = tokio::time::timeout(DEADLINE, async {
        let read = service.get_full_body(&key, DEADLINE);
        tokio::pin!(read);
        loop {
            runtime.apply_once_for_test().await?;
            tokio::select! {
                result = &mut read => break result,
                _ = tokio::task::yield_now() => {},
            }
        }
    })
    .await??
    .context("recovered body missing")?;
    assert_eq!(held.batch(), &input[0].slice(0, 1));
    assert_eq!(counts.held.load(Ordering::SeqCst), bytes);
    // A second record hits the existing batch threshold. Hold the original
    // recovered read across publication and verify its backing stays charged.
    let (stop, task) = run_owner(runtime);
    let payload = fixture::encoded(single_body::batch(
        &spec,
        &fixture::limits(),
        Shard { bucket: 0 },
        &input[1],
    )?)?;
    let other = producer(&service, 2).await?;
    let duplicate = request(&service, &other, 1, payload).await?;
    let result = service.submit(&duplicate, DEADLINE).await?;
    drained(&c, &service, result.index).await?;
    assert_eq!(counts.held.load(Ordering::SeqCst), bytes);
    assert_eq!(held.batch(), &input[0].slice(0, 1));
    // The original exact request result remains available after materialization.
    assert_eq!(service.submit(&append, DEADLINE).await?, receipt);
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    assert_eq!(counts.held.load(Ordering::SeqCst), bytes);
    drop(held);
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    c.shutdown().await
}
