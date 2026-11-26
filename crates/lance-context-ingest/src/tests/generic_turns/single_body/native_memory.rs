//! Native scan refusal must not turn a durable, unapplied create into a miss.
use super::*;
use futures::future::BoxFuture;
use lance_core::memory::{NativeMemoryKind, NativeMemoryPool, NativeMemoryReservation};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug, Default)]
struct Admission {
    refuse: AtomicBool,
    typed: AtomicBool,
    refused: AtomicUsize,
    held: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct Charge(Arc<AtomicUsize>, usize);

impl NativeMemoryReservation for Charge {
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

impl NativeMemoryPool for Admission {
    fn reserve(
        &self,
        kind: NativeMemoryKind,
        bytes: usize,
    ) -> BoxFuture<'_, lance_core::Result<Box<dyn NativeMemoryReservation>>> {
        Box::pin(async move { self.try_reserve(kind, bytes) })
    }

    fn try_reserve(
        &self,
        kind: NativeMemoryKind,
        bytes: usize,
    ) -> lance_core::Result<Box<dyn NativeMemoryReservation>> {
        let typed = self.typed.load(Ordering::SeqCst);
        if self.refuse.load(Ordering::SeqCst) && (!typed || kind == NativeMemoryKind::Decode) {
            self.refused.fetch_add(1, Ordering::SeqCst);
            let owner = match kind {
                NativeMemoryKind::Read => "read",
                NativeMemoryKind::Decode => "decode",
                NativeMemoryKind::Index => "index decode",
            };
            let message = format!("executor {owner} memory exhausted: requested {bytes}");
            return Err(if typed {
                lance_core::Error::backpressure(message)
            } else {
                lance_core::Error::execution(message)
            });
        }
        self.held.fetch_add(bytes, Ordering::SeqCst);
        Ok(Box::new(Charge(self.held.clone(), bytes)))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_lookup_refusal_preserves_wal_suffix_and_first_create_on_recovery() -> Result<()> {
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let admission = Arc::new(Admission::default());
    let storage = Storage::resolve(
        root.path().join("full-body.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default().with_native_memory(admission.clone())),
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
    let encode = |i| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i],
        )?)
    };
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let (stop, task) = run_owner(runtime);
    let lane = producer(&service, 1).await?;
    let first = request(&service, &lane, 1, encode(0)?).await?;
    let first_result = service.submit(&first, DEADLINE).await?;
    drained(&c, &service, first_result.index).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);

    let (service, runtime) = owner(&c, &storage, &spec).await?;
    // Contains two existing keys with changed values and two new keys. A
    // failed native scan cannot be interpreted as all four keys being absent.
    let next_lane = producer(&service, 2).await?;
    let next = request(&service, &next_lane, 1, encode(2)?).await?;
    let reply = c
        .client
        .request_routed(Request::Append {
            group: 10,
            batch: next.batch.clone(),
        })
        .await?;
    let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = reply.reply else {
        bail!("suffix was not quorum durable")
    };
    admission.refuse.store(true, Ordering::SeqCst);
    let failure = tokio::time::timeout(DEADLINE, runtime.run(std::future::pending()))
        .await?
        .expect_err("untyped native errors must stop the owner");
    assert_eq!(failure.to_string(), "annotation apply driver stopped");
    assert!(format!("{failure:#}").contains("immutable body historical lookup failed"));
    assert!(format!("{failure:#}").contains("memory exhausted: requested"));
    assert!(admission.refused.load(Ordering::SeqCst) > 0);
    assert!(service.observation().await.0.applied < index);
    assert!(service.observation().await.0.materialized < index);
    assert!(service
        .submit(&next, Duration::from_millis(100))
        .await
        .is_err());
    drop(service);

    // A replacement owner replays the same durable suffix when admission is
    // available. The previously acknowledged values must win first-create.
    admission.refuse.store(false, Ordering::SeqCst);
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let (stop, task) = run_owner(runtime);
    drained(&c, &service, index).await?;
    assert_eq!(service.submit(&first, DEADLINE).await?, first_result);
    let result = service.submit(&next, DEADLINE).await?;
    assert_eq!(result.index, index);
    assert_eq!(result.result, MutationResult::Applied);
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
    assert_eq!(durable.count_rows(None).await?, 4);
    assert_eq!(service.retained_body_frames(), (0, 0));
    stop.send(()).unwrap();
    task.await??;
    drop(durable);
    drop(service);
    drop(storage);
    assert_eq!(admission.held.load(Ordering::SeqCst), 0);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn typed_native_lookup_pressure_drains_prefix_and_resumes_same_owner() -> Result<()> {
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let admission = Arc::new(Admission::default());
    admission.typed.store(true, Ordering::SeqCst);
    let storage = Storage::resolve(
        root.path().join("full-body.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default().with_native_memory(admission.clone())),
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
    let encode = |i| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i],
        )?)
    };
    let (service, runtime) = owner(&c, &storage, &spec).await?;
    let (stop, task) = run_owner(runtime);
    let lane = producer(&service, 1).await?;
    let first = request(&service, &lane, 1, encode(0)?).await?;
    let first_result = service.submit(&first, DEADLINE).await?;
    drained(&c, &service, first_result.index).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);

    // Keep an acknowledged prefix below the ordinary publication threshold.
    // Native admission pressure, rather than a timer or explicit flush, must
    // let that prefix drain while the next committed request remains pending.
    let (service, runtime) = owner_with_wait(&c, &storage, &spec, Duration::from_secs(60)).await?;
    let (stop, task) = run_owner(runtime);
    let lane = producer(&service, 2).await?;
    let prefix = request(&service, &lane, 1, encode(3)?).await?;
    let prefix_result = service.submit(&prefix, DEADLINE).await?;
    assert!(service.observation().await.0.materialized < prefix_result.index);
    let next = request(&service, &lane, 2, encode(2)?).await?;
    admission.refuse.store(true, Ordering::SeqCst);
    let reply = c
        .client
        .request_routed(Request::Append {
            group: 10,
            batch: next.batch.clone(),
        })
        .await?;
    let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = reply.reply else {
        bail!("suffix was not quorum durable")
    };
    tokio::time::timeout(DEADLINE, async {
        while service.apply_backpressure()
            != Some(crate::annotations::ApplyBackpressure::ApplyMemory)
        {
            assert!(!task.is_finished(), "native pressure stopped the owner");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert!(admission.refused.load(Ordering::SeqCst) > 0);
    tokio::time::timeout(DEADLINE, async {
        loop {
            assert!(!task.is_finished(), "publisher stopped under pressure");
            let observed = service.observation().await.0;
            let view = annotation_publication::state(&c).await?;
            if observed.materialized >= prefix_result.index
                && observed.retained_bytes == 0
                && observed.retained_metadata_bytes == 0
                && service.source_observation().records == 0
                && view.progress.pending.is_none()
                && view.progress.checkpoint.get(&10).copied().unwrap_or(0) >= prefix_result.index
            {
                break Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await??;
    assert!(
        !task.is_finished(),
        "publisher could not drain under pressure"
    );
    assert!(service.observation().await.0.applied < index);
    assert!(service.observation().await.0.materialized < index);
    assert_eq!(service.retained_body_frames(), (0, 0));
    assert!(service
        .submit(&next, Duration::from_millis(100))
        .await
        .is_err());
    assert!(!task.is_finished());

    admission.refuse.store(false, Ordering::SeqCst);
    let result = service.submit(&next, DEADLINE).await?;
    assert_eq!(result.index, index);
    assert_eq!(result.result, MutationResult::Applied);
    // Resume remains the same owner and request. The ordinary low-flow timer
    // is deliberately long; apply a duplicate to reach its two-record batch.
    let duplicate = request(&service, &lane, 3, encode(2)?).await?;
    let duplicate_result = service.submit(&duplicate, DEADLINE).await?;
    drained(&c, &service, duplicate_result.index).await?;
    assert_eq!(service.submit(&first, DEADLINE).await?, first_result);
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
    drop(durable);
    drop(service);
    drop(storage);
    assert_eq!(admission.held.load(Ordering::SeqCst), 0);
    c.shutdown().await
}
