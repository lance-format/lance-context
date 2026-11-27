use super::*;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;
use lance_context_wal::replication::runtime::client::payload_memory::{Pool, Reservation};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

#[derive(Default)]
struct CapturePool {
    refuse: AtomicBool,
    held: Arc<AtomicUsize>,
    backed: Arc<AtomicUsize>,
    cancel: Mutex<
        Option<tokio::sync::oneshot::Receiver<Result<crate::annotations::lookup_capture::Capture>>>,
    >,
}

struct CaptureCharge {
    bytes: usize,
    held: Arc<AtomicUsize>,
    backed: Arc<AtomicUsize>,
}

impl Pool for CapturePool {
    fn try_reserve(&self, bytes: usize) -> Result<Option<Box<dyn Reservation>>> {
        // Deterministically cancel after State took the request and checked
        // receiver liveness, but before allocating/copying the diagnostic keys.
        drop(self.cancel.lock().unwrap().take());
        if self.refuse.load(Ordering::SeqCst) {
            return Ok(None);
        }
        self.held.fetch_add(bytes, Ordering::SeqCst);
        Ok(Some(Box::new(CaptureCharge {
            bytes,
            held: self.held.clone(),
            backed: self.backed.clone(),
        })))
    }
}

impl Reservation for CaptureCharge {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn set_backed_bytes(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        assert!(self.held.load(Ordering::SeqCst) >= self.bytes);
        if bytes != 0 {
            self.backed.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl Drop for CaptureCharge {
    fn drop(&mut self) {
        self.held.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn staged_duplicates_skip_history_and_publication_restores_base_lookup() -> Result<()> {
    tokio::time::timeout(DEADLINE, exercise()).await?
}

async fn exercise() -> Result<()> {
    let input = bodies()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(fault_store::Faults::default());
    let storage = fault_store::storage(&root.path().join("body.lance"), faults.clone()).await?;
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
    let encode = |i: usize| {
        fixture::encoded(single_body::batch(
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
            &input[i],
        )?)
    };
    let (service, mut runtime) = owner(&c, &storage, &spec).await?;
    let capture = runtime.body_lookup_capture()?;
    let diagnostic_budget = MergeMemoryBudget::new(1 << 20);
    let initial = service.observation().await.0.installed_base;
    assert_eq!(initial.rows, Some(0));
    assert!(initial.immutable_body);
    let lane = producer(&service, 1).await?;
    // Apply without running publication: the second frame has the same keys
    // and different values, and must resolve entirely from active staging.
    for (sequence, body) in [(1, 0), (2, 1)] {
        let prepared = request(&service, &lane, sequence, encode(body)?).await?;
        let submitted = service.submit(&prepared, DEADLINE);
        tokio::pin!(submitted);
        let completed = loop {
            runtime.apply_once_for_test().await?;
            tokio::select! {
                result = &mut submitted => break result?,
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        };
        assert_eq!(completed.result, MutationResult::Applied);
    }
    assert_eq!(service.body_lookup_observation().scans_started, 1);
    assert_eq!(service.body_lookup_observation().staged_keys, 2);
    assert_eq!(service.body_lookup_observation().history_hits, 0);
    assert_eq!(service.body_lookup_observation().history_misses, 2);
    let applied = service.observation().await.0.immutable_apply;
    assert_eq!((applied.batches, applied.rows), (2, 4));
    assert_eq!((applied.inserted_rows, applied.duplicate_rows), (2, 2));
    assert_eq!(service.selected_full_body(DEADLINE).await?.rows, Some(0));
    assert_eq!(service.observation().await.0.installed_base, initial);

    // Pause a real data-object upload after freezing, then apply both an
    // all-duplicate frame and a mixed frame while the selected base is old.
    let (entered, resume) = faults.pause_data_write();
    let (stop, task) = run_owner(runtime);
    entered.await?;
    let cancelled = capture.arm(diagnostic_budget.clone())?;
    assert!(capture.arm(diagnostic_budget.clone()).is_err());
    drop(cancelled);
    let mut keys = capture.arm(diagnostic_budget.clone())?;
    let duplicate = request(&service, &lane, 3, encode(1)?).await?;
    let result = service.submit(&duplicate, DEADLINE).await?;
    assert_eq!(result.result, MutationResult::Applied);
    assert_eq!(service.body_lookup_observation().scans_started, 1);
    assert_eq!(service.body_lookup_observation().staged_keys, 4);
    // All-staged windows do not consume the one-shot historical capture.
    assert!(matches!(
        keys.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
    assert_eq!(diagnostic_budget.reserved(), 0);
    let mixed = request(&service, &lane, 4, encode(2)?).await?;
    let result = service.submit(&mixed, DEADLINE).await?;
    assert_eq!(result.result, MutationResult::Applied);
    assert_eq!(service.body_lookup_observation().scans_started, 2);
    assert_eq!(service.body_lookup_observation().staged_keys, 6);
    assert_eq!(service.body_lookup_observation().history_hits, 0);
    assert_eq!(service.body_lookup_observation().history_misses, 4);
    let captured = tokio::time::timeout(DEADLINE, keys).await???;
    assert!(diagnostic_budget.reserved() > 0);
    let data = captured.data();
    assert_eq!(
        data.owner,
        annotation_publication::state(&c).await?.authority
    );
    assert_eq!(data.incarnation, binding().incarnation);
    assert_eq!((data.source_group, data.bucket), (10, 0));
    assert_eq!(data.base_version, initial.version);
    assert!(data.base_fragment_ids.is_empty());
    assert_eq!(
        (data.window.first_record, data.window.last_record),
        (result.index, result.index)
    );
    assert_eq!(data.window.requests, 1);
    assert_eq!(data.window.payload_bytes, mixed.body().len() as u64);
    assert!(!data.window.speculative);
    assert_eq!((data.requested_keys, data.staged_keys), (4, 2));
    assert_eq!(
        data.primary_key_columns
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        ["session_id", "id"]
    );
    let mut expected = (2..4)
        .map(|row| spec.row_key(&input[2], row, 4096))
        .collect::<Result<Vec<_>>>()?;
    expected.sort();
    assert_eq!(
        data.keys.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        expected
    );
    // Output contains only identities, not the table URI or any body schema.
    let json = serde_json::to_value(&captured)?;
    assert!(json.get("uri").is_none());
    assert!(json.get("content").is_none());
    assert!(json.get("_reservation").is_none());
    let capture_id = data.capture_id;
    drop(captured);
    assert_eq!(diagnostic_budget.reserved(), 0);
    let applied = service.observation().await.0.immutable_apply;
    assert_eq!((applied.batches, applied.rows), (4, 10));
    assert_eq!((applied.inserted_rows, applied.duplicate_rows), (4, 6));
    assert_eq!(service.selected_full_body(DEADLINE).await?.rows, Some(0));
    assert_eq!(service.observation().await.0.installed_base, initial);
    for (row, original) in [
        (0, &input[0]),
        (1, &input[0]),
        (2, &input[2]),
        (3, &input[2]),
    ] {
        let key = spec.row_key(&input[2], row, 4096)?;
        assert_eq!(
            service
                .get_full_body(&key, DEADLINE)
                .await?
                .unwrap()
                .batch(),
            &original.slice(row, 1)
        );
    }
    resume.send(()).unwrap();
    drained(&c, &service, result.index).await?;
    assert_eq!(service.retained_body_frames(), (0, 0));
    let selected = service.selected_full_body(DEADLINE).await?;
    assert_eq!(selected.rows, Some(4));
    let installed = service.observation().await.0;
    assert_eq!(installed.installed_base.rows, selected.rows);
    assert_eq!(installed.installed_base.version, selected.manifest.version);
    assert_eq!(installed.installed_base.source, selected.source);
    assert_eq!(installed.materialized, selected.through);
    let durable =
        annotation_publication::read(&c, &storage, Some(selected.manifest.version)).await?;
    assert_eq!(durable.count_rows(None).await?, 4);
    let before = service.body_lookup_observation();
    let pool = Arc::new(CapturePool::default());
    let keys = capture.arm_with_pool(diagnostic_budget.clone(), pool.clone())?;
    let repeated = request(&service, &lane, 5, encode(1)?).await?;
    let result = service.submit(&repeated, DEADLINE).await?;
    let after = service.body_lookup_observation();
    let captured = tokio::time::timeout(DEADLINE, keys).await???;
    let data = captured.data();
    assert_ne!(data.capture_id, capture_id);
    assert_eq!(data.base_version, selected.manifest.version);
    assert_eq!(
        data.base_fragment_ids.as_ref(),
        durable.iter_fragments().map(|f| f.id).collect::<Vec<_>>()
    );
    assert_eq!((data.requested_keys, data.staged_keys), (2, 0));
    assert_eq!(data.window.first_record, result.index);
    assert!(pool.held.load(Ordering::SeqCst) > 0);
    drop(captured);
    assert_eq!(pool.held.load(Ordering::SeqCst), 0);
    assert_eq!(diagnostic_budget.reserved(), 0);
    assert_eq!(after.scans_started, before.scans_started + 1);
    assert_eq!(after.native_summaries, before.native_summaries + 1);
    assert_eq!(after.staged_keys, before.staged_keys);
    assert_eq!(after.history_hits, before.history_hits + 2);
    assert_eq!(after.history_misses, before.history_misses);
    assert_eq!(
        after.requested_keys,
        after.staged_keys + after.history_hits + after.history_misses
    );
    let applied = service.observation().await.0.immutable_apply;
    assert_eq!((applied.batches, applied.rows), (5, 12));
    assert_eq!((applied.inserted_rows, applied.duplicate_rows), (4, 8));
    // A materialized key is no longer a staged hit; the original body wins.
    let key = spec.row_key(&input[0], 0, 4096)?;
    assert_eq!(
        service
            .get_full_body(&key, DEADLINE)
            .await?
            .unwrap()
            .batch(),
        &input[0].slice(0, 1)
    );
    drained(&c, &service, result.index).await?;
    // Diagnostic refusal cannot fail or delay durable apply. The snapshot is
    // one-shot even on failure, and releases every partial allocation.
    let held = diagnostic_budget
        .try_reserve(diagnostic_budget.limit())
        .unwrap();
    let keys = capture.arm(diagnostic_budget.clone())?;
    let repeated = request(&service, &lane, 6, encode(1)?).await?;
    let result = service.submit(&repeated, DEADLINE).await?;
    assert_eq!(result.result, MutationResult::Applied);
    let error = tokio::time::timeout(DEADLINE, keys)
        .await??
        .err()
        .context("capture unexpectedly bypassed admission")?;
    assert!(error.to_string().contains("budget unavailable"));
    drop(held);
    assert_eq!(diagnostic_budget.reserved(), 0);
    drained(&c, &service, result.index).await?;
    pool.refuse.store(true, Ordering::SeqCst);
    let keys = capture.arm_with_pool(diagnostic_budget.clone(), pool.clone())?;
    let repeated = request(&service, &lane, 7, encode(1)?).await?;
    let result = service.submit(&repeated, DEADLINE).await?;
    assert_eq!(result.result, MutationResult::Applied);
    let error = tokio::time::timeout(DEADLINE, keys)
        .await??
        .err()
        .context("capture bypassed host admission")?;
    assert!(
        error.is::<lance_context_wal::replication::runtime::client::payload_memory::Unavailable>()
    );
    assert_eq!(pool.held.load(Ordering::SeqCst), 0);
    assert_eq!(diagnostic_budget.reserved(), 0);
    drained(&c, &service, result.index).await?;

    pool.refuse.store(false, Ordering::SeqCst);
    let backed = pool.backed.load(Ordering::SeqCst);
    *pool.cancel.lock().unwrap() =
        Some(capture.arm_with_pool(diagnostic_budget.clone(), pool.clone())?);
    let repeated = request(&service, &lane, 8, encode(1)?).await?;
    let result = service.submit(&repeated, DEADLINE).await?;
    assert_eq!(result.result, MutationResult::Applied);
    assert!(pool.cancel.lock().unwrap().is_none());
    assert_eq!(pool.backed.load(Ordering::SeqCst), backed + 1);
    assert_eq!(
        pool.held.load(Ordering::SeqCst),
        0,
        "failed send releases host charge after copying"
    );
    assert_eq!(diagnostic_budget.reserved(), 0);
    drop(capture.arm_with_pool(diagnostic_budget.clone(), pool)?);
    drained(&c, &service, result.index).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
