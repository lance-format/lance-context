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
    pause_reads: AtomicBool,
    entered: AtomicUsize,
    resume: tokio::sync::Notify,
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
        Box::pin(async move {
            if kind == NativeMemoryKind::Read && self.pause_reads.load(Ordering::SeqCst) {
                self.entered.fetch_add(1, Ordering::SeqCst);
                loop {
                    let resume = self.resume.notified();
                    if !self.pause_reads.load(Ordering::SeqCst) {
                        break;
                    }
                    resume.await;
                }
            }
            self.try_reserve(kind, bytes)
        })
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_window_queries_overlap_and_recheck_staging_in_order() -> Result<()> {
    tokio::time::timeout(DEADLINE, two_windows(false, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_window_old_misses_requery_after_prefix_publication() -> Result<()> {
    tokio::time::timeout(DEADLINE, two_windows(true, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_window_cancelled_native_queries_recover_the_whole_durable_suffix() -> Result<()> {
    tokio::time::timeout(DEADLINE, two_windows(false, true)).await?
}

async fn two_windows(publish: bool, cancel: bool) -> Result<()> {
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
    let seed = request(&service, &lane, 1, encode(0)?).await?;
    let seed_result = service.submit(&seed, DEADLINE).await?;
    drained(&c, &service, seed_result.index).await?;
    // Warm shared fragment metadata while the initial applier is still live.
    // Point reads require its quorum barrier even when all data is published.
    let seed_key = spec.row_key(&input[0], 0, 4096)?;
    assert!(service.get_full_body(&seed_key, DEADLINE).await?.is_some());
    stop.send(()).unwrap();
    task.await??;
    drop(service);

    let state = annotation_publication::state(&c).await?;
    let (service, mut runtime) = Runtime::connect_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Configuration {
            limits: fixture::limits(),
            upload_budget: MergeMemoryBudget::new(64 << 20),
            apply_budget: MergeMemoryBudget::new(64 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 64,
                max_batch_wait: Duration::ZERO,
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let initial = service.observation().await.0.applied;
    let mut backlog = Vec::new();
    let mut lanes = Vec::new();
    for slot in 2..16 {
        lanes.push(producer(&service, slot).await?);
    }
    // Two full windows and one successor. B shares A's new keys; a lookup
    // captured before A applies must recheck staging or the published base.
    for i in 0..17 {
        let prepared = request(
            &service,
            &lanes[i % lanes.len()],
            (i / lanes.len() + 1) as u64,
            encode(if i == 16 { 3 } else { 2 })?,
        )
        .await?;
        let response = c
            .client
            .request_routed(Request::Append {
                group: 10,
                batch: prepared.batch.clone(),
            })
            .await?;
        let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = response.reply else {
            bail!("test backlog was not quorum durable")
        };
        backlog.push((prepared, index));
    }
    admission.pause_reads.store(true, Ordering::SeqCst);
    {
        let step = runtime.apply_once_for_test();
        tokio::pin!(step);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            tokio::select! {
                result = &mut step => panic!("native lookup escaped read gate: {result:?}"),
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
            if service.body_lookup_observation().history_query_overlaps == 1
                && admission.entered.load(Ordering::SeqCst) >= 2
            {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "native overlap not observed: reads={} lookup={:?}",
                admission.entered.load(Ordering::SeqCst),
                service.body_lookup_observation()
            );
        }
        assert_eq!(service.observation().await.0.applied, initial);
        assert_eq!(service.body_lookup_observation().prefetch.requests, 16);
        assert_eq!(
            service.body_lookup_observation().prefetched_lookups_started,
            1
        );
        if !cancel {
            admission.pause_reads.store(false, Ordering::SeqCst);
            admission.resume.notify_waiters();
            step.await?;
        }
    }
    if cancel {
        drop(runtime);
        assert_eq!(service.observation().await.0.applied, initial);
        assert_eq!(
            service
                .body_lookup_observation()
                .prefetched_lookups_cancelled,
            1
        );
        admission.pause_reads.store(false, Ordering::SeqCst);
        admission.resume.notify_waiters();
        drop(service);
        let (service, runtime) = owner(&c, &storage, &spec).await?;
        finish_two_windows(&c, &storage, &spec, &input, service, runtime, &backlog).await?;
    } else {
        assert_eq!(service.observation().await.0.applied, backlog[0].1);
        for (_, index) in backlog.iter().take(8).skip(1) {
            runtime.apply_once_for_test().await?;
            assert_eq!(service.observation().await.0.applied, *index);
        }
        if publish {
            while service.observation().await.0.materialized < backlog[7].1 {
                runtime.publish_once_for_test().await?;
            }
            runtime.apply_once_for_test().await?;
            assert_eq!(service.observation().await.0.applied, backlog[7].1);
            assert_eq!(
                service.body_lookup_observation().prefetched_lookups_stale,
                1
            );
        }
        runtime.apply_once_for_test().await?;
        assert_eq!(service.observation().await.0.applied, backlog[8].1);
        if !publish {
            assert_eq!(
                service.body_lookup_observation().prefetched_lookups_reused,
                1
            );
            assert_eq!(
                service.body_lookup_observation().prefetched_lookups_stale,
                0
            );
        }
        finish_two_windows(&c, &storage, &spec, &input, service, runtime, &backlog).await?;
    }
    drop(storage);
    tokio::time::timeout(DEADLINE, async {
        while admission.held.load(Ordering::SeqCst) != 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    c.shutdown().await
}

async fn finish_two_windows(
    c: &network::Cluster,
    storage: &Storage,
    spec: &TurnIngestionSpec,
    input: &[RecordBatch],
    service: Service,
    runtime: Runtime,
    backlog: &[(crate::PreparedAppend, u64)],
) -> Result<()> {
    let (stop, task) = run_owner(runtime);
    drained(c, &service, backlog.last().unwrap().1).await?;
    for (i, (prepared, index)) in backlog.iter().enumerate() {
        // The contract retains the latest result per producer, not permanent
        // request history. Its final ACK covers the earlier ordered prefix.
        if backlog[i + 1..]
            .iter()
            .any(|(later, _)| later.meta().producer == prepared.meta().producer)
        {
            continue;
        }
        assert_eq!(service.submit(prepared, DEADLINE).await?.index, *index);
    }
    let selected = service.selected_full_body(DEADLINE).await?;
    let durable = annotation_publication::read(c, storage, Some(selected.manifest.version)).await?;
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
    stop.send(()).unwrap();
    task.await??;
    Ok(())
}
