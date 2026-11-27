//! A retained short page must not limit the admitted lookup window. Use an
//! actual RF3 page captured under apply pressure, then commit a longer suffix.
use super::*;
use lance_context_wal::replication::runtime::client::payload_memory::{Pool, Reservation};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
struct Gate {
    allow: AtomicBool,
}
struct Charge(usize);
impl Reservation for Charge {
    fn bytes(&self) -> usize {
        self.0
    }
    fn set_backed_bytes(&mut self, bytes: usize) {
        assert!(bytes <= self.0);
    }
}
impl Pool for Gate {
    fn try_reserve(&self, bytes: usize) -> Result<Option<Box<dyn Reservation>>> {
        Ok(self
            .allow
            .load(Ordering::SeqCst)
            .then(|| Box::new(Charge(bytes)) as Box<dyn Reservation>))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_lookup_refills_across_a_retained_short_page() -> Result<()> {
    tokio::time::timeout(DEADLINE, exercise(false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_cancelled_cross_page_lookahead_replays_without_skipping() -> Result<()> {
    tokio::time::timeout(DEADLINE, exercise(true)).await?
}

async fn exercise(cancel: bool) -> Result<()> {
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
    let state = annotation_publication::state(&c).await?;
    let (service, mut runtime) = Runtime::connect_body_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Configuration {
            limits: fixture::limits(),
            // The existing budget charges whole MiB permits. Retaining twelve
            // prepared requests needs twelve upload permits; the frame/lookup
            // partition must independently admit an eight-request window.
            upload_budget: MergeMemoryBudget::new(16 << 20),
            apply_budget: MergeMemoryBudget::new(16 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 64,
                max_batch_wait: Duration::from_millis(20),
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
    let gate = Arc::new(Gate::default());
    let mut backlog = Vec::new();
    for i in 0..12 {
        let lane = producer(&service, i as u32 + 1).await?;
        let encoded = fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i % input.len()],
        )?)?;
        // With a shared payload pool, preparation owns unknown upload backing.
        // Check the original bytes; borrowed-pointer identity is not promised.
        let expected = encoded.clone();
        let upload = service.reserve_upload(encoded.len()).await?;
        let prepared = service.prepare(upload, encoded, &lane, 1).await?;
        assert_eq!(prepared.body(), &expected);
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
        if i == 2 {
            c.client.install_payload_memory(gate.clone())?;
            runtime.apply_once_for_test().await?;
            assert_eq!(service.observation().await.0.applied, initial);
            assert_eq!(service.body_lookup_observation().completed_lookups, 0);
            assert_eq!(
                service.apply_backpressure(),
                Some(crate::annotations::ApplyBackpressure::ApplyMemory)
            );
            // The applier now retains the real three-record metadata page.
            // Subsequent appends cannot retroactively extend that response.
            gate.allow.store(true, Ordering::SeqCst);
        }
    }
    let diagnostic_budget = MergeMemoryBudget::new(1 << 20);
    let captured = runtime
        .body_lookup_capture()?
        .arm(diagnostic_budget.clone())?;
    runtime.apply_once_for_test().await?;
    let observed = service.body_lookup_observation();
    assert_eq!(observed.completed_lookups, 1);
    // The window is no longer capped at eight requests: it spans the whole
    // committed backlog, bounded by key workspace. Only the first eight keep
    // decoded bodies resident; the rest are keys-only until re-read from WAL.
    assert_eq!(observed.last_lookup_requests, 12);
    assert_eq!(observed.prefetch.request_bound, 0);
    assert_eq!(observed.prefetch.page_exhausted, 1);
    assert_eq!(observed.prefetch.keys_only_records, 4);
    assert_eq!(observed.prefetch.body_refetches, 0);
    assert_eq!(observed.scans_started, 1);
    assert_eq!(
        observed.last_lookup_payload_bytes,
        backlog[..12]
            .iter()
            .map(|(p, _)| p.body().len() as u64)
            .sum::<u64>()
    );
    assert_eq!(service.observation().await.0.applied, backlog[0].1);
    assert_eq!(service.source_observation().records, 1);
    let captured = tokio::time::timeout(DEADLINE, captured).await???;
    let window = &captured.data().window;
    assert_eq!(window.first_record, backlog[0].1);
    assert_eq!(window.last_record, backlog[11].1);
    assert_eq!(window.requests, 12);
    assert_eq!(window.payload_bytes, observed.last_lookup_payload_bytes);
    assert_eq!(captured.data().staged_keys, 0);
    assert_eq!(captured.data().keys.len(), 6);
    drop(captured);
    assert_eq!(diagnostic_budget.reserved(), 0);

    if cancel {
        // Eleven admitted requests span two pages, but only the first applied.
        // Dropping the owner must not persist the last observed page cursor.
        drop(runtime);
        drop(service);
        let (service, runtime) = owner(&c, &storage, &spec).await?;
        assert!(service.observation().await.0.applied < backlog[0].1);
        finish(&c, &storage, &spec, &input, service, runtime, backlog).await?;
    } else {
        for (_, index) in backlog.iter().skip(1) {
            runtime.apply_once_for_test().await?;
            assert_eq!(service.observation().await.0.applied, *index);
        }
        // One history scan covered all twelve requests; keys-only records
        // were re-read from the durable WAL instead of held in memory.
        let observed = service.body_lookup_observation();
        assert_eq!(observed.completed_lookups, 1);
        assert_eq!(observed.scans_started, 1);
        assert_eq!(observed.prefetch.body_refetches, 4);
        finish(&c, &storage, &spec, &input, service, runtime, backlog).await?;
    }
    c.shutdown().await
}

async fn finish(
    c: &network::Cluster,
    storage: &Storage,
    spec: &TurnIngestionSpec,
    input: &[RecordBatch],
    service: Service,
    runtime: Runtime,
    backlog: Vec<(crate::PreparedAppend, u64)>,
) -> Result<()> {
    let (stop, task) = run_owner(runtime);
    drained(c, &service, backlog.last().unwrap().1).await?;
    for (prepared, index) in backlog {
        assert_eq!(service.submit(&prepared, DEADLINE).await?.index, index);
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
    stop.send(()).unwrap();
    task.await??;
    Ok(())
}
