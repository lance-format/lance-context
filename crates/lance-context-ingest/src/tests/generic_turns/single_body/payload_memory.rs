use super::*;
use lance_context_wal::replication::runtime::client::payload_memory::{Pool, Reservation};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counts {
    allow: AtomicBool,
    attempts: AtomicUsize,
    held: AtomicUsize,
    limit: AtomicUsize,
    peak: AtomicUsize,
}
impl Default for Counts {
    fn default() -> Self {
        Self {
            allow: AtomicBool::new(false),
            attempts: AtomicUsize::new(0),
            held: AtomicUsize::new(0),
            limit: AtomicUsize::new(usize::MAX),
            peak: AtomicUsize::new(0),
        }
    }
}
struct Admission(Arc<Counts>);
struct Charge(Arc<Counts>, usize);
impl Pool for Admission {
    fn try_reserve(&self, bytes: usize) -> Result<Option<Box<dyn Reservation>>> {
        self.0.attempts.fetch_add(1, Ordering::SeqCst);
        if !self.0.allow.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let Ok(held) = self
            .0
            .held
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |held| {
                held.checked_add(bytes)
                    .filter(|n| *n <= self.0.limit.load(Ordering::SeqCst))
            })
        else {
            return Ok(None);
        };
        self.0.peak.fetch_max(held + bytes, Ordering::SeqCst);
        Ok(Some(Box::new(Charge(self.0.clone(), bytes))))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bounded_payload_pressure_flushes_below_batch_target() -> Result<()> {
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
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let mut backlog = Vec::new();
    for (i, body) in input.iter().enumerate() {
        let lane = producer(&service, i as u32 + 1).await?;
        let payload = fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            body,
        )?)?;
        let prepared = request(&service, &lane, 1, payload).await?;
        // Quorum acceptance alone is not a successful applied mutation ACK.
        // Queue durable source records without local payload sharing, as when
        // recovering writes submitted by a different ingress.
        let response = c
            .client
            .request_routed(Request::Append {
                group: 10,
                batch: prepared.batch.clone(),
            })
            .await?;
        let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = response.reply else {
            bail!("backlog append failed");
        };
        backlog.push((prepared, index));
    }
    drop(runtime);
    drop(service);
    let capacity = backlog.iter().map(|(p, _)| p.body().len()).max().unwrap();
    assert!(capacity < backlog[0].0.body().len() + backlog[1].0.body().len());
    let counts = Arc::new(Counts::default());
    counts.allow.store(true, Ordering::SeqCst);
    counts.limit.store(capacity, Ordering::SeqCst);
    c.client
        .install_payload_memory(Arc::new(Admission(counts.clone())))?;
    let (service, mut runtime) =
        owner_with_wait(&c, &storage, &spec, Duration::from_secs(60)).await?;
    tokio::time::timeout(DEADLINE, async {
        while service.apply_backpressure()
            != Some(crate::annotations::ApplyBackpressure::ApplyMemory)
        {
            runtime.apply_once_for_test().await?;
        }
        anyhow::Ok(())
    })
    .await??;
    assert_eq!(service.observation().await.0.applied, backlog[0].1);
    assert_eq!(service.source_observation().records, 1);
    assert!(service.source_observation().bytes < 1 << 20);
    // The publisher has not run, so its 60-second batching timer has not even
    // started. Memory pressure must publish this single below-target record.
    tokio::time::timeout(Duration::from_secs(5), runtime.publish_once_for_test()).await??;
    assert_eq!(service.observation().await.0.materialized, backlog[0].1);
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    drop(runtime);
    drop(service);

    // Rebuild the remaining suffix under the same bounded pool. Use the normal
    // short tail timer for its final below-target publication.
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let (stop, task) = run_owner(runtime);
    drained(&c, &service, backlog.last().unwrap().1).await?;
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    assert!(counts.peak.load(Ordering::SeqCst) <= capacity);
    for (prepared, index) in &backlog {
        let result = service.submit(prepared, DEADLINE).await?;
        assert_eq!(result.index, *index);
        assert_eq!(result.result, MutationResult::Applied);
    }
    let key = spec.row_key(&input[0], 0, 4096)?;
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    let selected = service.selected_full_body(DEADLINE).await?;
    let durable =
        annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, 6);
    assert_eq!(service.retained_body_frames(), (0, 0));
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
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
    drop(duplicate);
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_upload_memory_survives_publication_and_detached_arrow_views() -> Result<()> {
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
    let counts = Arc::new(Counts::default());
    c.client
        .install_payload_memory(Arc::new(Admission(counts.clone())))?;
    let (service, runtime) = owner_with_wait(&c, &storage, &spec, Duration::from_secs(60)).await?;
    let (stop, task) = run_owner(runtime);
    let lane = producer(&service, 1).await?;
    let encode = |i| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i],
        )?)
    };
    let body = encode(0)?;
    let bytes = body.len();
    let refused = request(&service, &lane, 1, body.clone()).await;
    assert!(refused
        .err()
        .unwrap()
        .is::<lance_context_wal::replication::runtime::client::payload_memory::Unavailable>());
    assert_eq!(service.source_observation().records, 0);
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    counts.allow.store(true, Ordering::SeqCst);
    let owned = body.to_vec();
    let pointer = owned.as_ptr();
    drop(body);
    let upload = service.reserve_upload(owned.len()).await?;
    let prepared = service.prepare_owned(upload, owned, &lane, 1).await?;
    assert_eq!(
        prepared.body().as_ptr(),
        pointer,
        "known owned payload must not copy"
    );
    assert_eq!(counts.held.load(Ordering::SeqCst), bytes);
    let receipt = service.submit(&prepared, DEADLINE).await?;
    assert_eq!(receipt.result, MutationResult::Applied);
    drop(prepared);
    // Applying a locally registered payload shares its charge; it must not
    // allocate/fetch another backing buffer just to apply the same request.
    assert_eq!(counts.held.load(Ordering::SeqCst), bytes);
    assert_eq!(counts.attempts.load(Ordering::SeqCst), 2);
    let key = spec.row_key(&input[0], 0, 4096)?;
    let read = service.get_full_body(&key, DEADLINE).await?.unwrap();
    let detached = read.batch().clone();
    drop(read);
    let other = producer(&service, 2).await?;
    let prepared = request(&service, &other, 1, encode(1)?).await?;
    let receipt = service.submit(&prepared, DEADLINE).await?;
    drop(prepared);
    drained(&c, &service, receipt.index).await?;
    assert_eq!(service.retained_body_frames(), (0, 0));
    assert_eq!(detached, input[0].slice(0, 1));
    assert_eq!(counts.held.load(Ordering::SeqCst), bytes);
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    assert_eq!(counts.held.load(Ordering::SeqCst), bytes);
    drop(detached);
    assert_eq!(counts.held.load(Ordering::SeqCst), 0);
    c.shutdown().await
}
