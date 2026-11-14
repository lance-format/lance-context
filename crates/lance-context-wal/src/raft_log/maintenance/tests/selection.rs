use super::*;
use raft_engine::internals::{EventListener, FileBlockHandle, LogQueue};

#[derive(Default)]
struct RewriteWrites(AtomicU64);
impl EventListener for RewriteWrites {
    fn on_append_log_file(&self, handle: FileBlockHandle) {
        if handle.id.queue == LogQueue::Rewrite {
            self.0.fetch_add(handle.len as u64, Ordering::Relaxed);
        }
    }
}

fn records() -> Vec<Entry> {
    (1..=64)
        .map(|index| Entry {
            data: vec![index as u8; 128 << 10].into(),
            ..entry(10, index)
        })
        .collect()
}

fn rewritten_engine(root: &Path) -> Result<(Engine, raft_engine::Config, Arc<RewriteWrites>)> {
    let config = raft_engine::Config {
        dir: root.to_str().unwrap().to_owned(),
        target_file_size: raft_engine::ReadableSize(64 << 10),
        purge_threshold: raft_engine::ReadableSize(64 << 10),
        purge_rewrite_threshold: Some(raft_engine::ReadableSize(64 << 10)),
        batch_compression_threshold: raft_engine::ReadableSize(0),
        enable_log_recycle: false,
        ..Default::default()
    };
    let writes = Arc::new(RewriteWrites::default());
    let engine = Engine::open_with_listeners(config.clone(), vec![writes.clone()])?;
    let records = records();
    for record in &records {
        let mut batch = LogBatch::default();
        batch.add_entries::<EntryIndex>(10, std::slice::from_ref(record))?;
        engine.write(&mut batch, true)?;
    }
    let mut batch = LogBatch::default();
    batch.put(30, b"retained".to_vec(), vec![7; 64 << 10])?;
    engine.write(&mut batch, true)?;
    engine.purge_manager().must_rewrite_append_queue(None, None);
    assert!(!engine.purge_manager().has_obsolete_rewrite_entries());
    Ok((engine, config, writes))
}

#[test]
fn an_idle_rewrite_queue_has_live_bytes_but_no_native_copy() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (engine, config, writes) = rewritten_engine(root.path())?;
    let boundary = engine.purge_manager().capture_append_boundary();
    assert!(!boundary.rewrite_queue);
    let allowance = engine
        .estimated_rewrite_bytes_through(boundary.append_watermark)
        .unwrap();
    assert!(allowance > 8 << 20);
    assert!(engine.estimated_rewrite_bytes_for(&boundary).unwrap() < 128 << 10);
    writes.0.store(0, Ordering::Relaxed);
    engine
        .purge_manager()
        .purge_expired_files_before(boundary)?;
    assert_eq!(writes.0.load(Ordering::Relaxed), 0);
    println!("idle rewrite: live allowance={allowance}, actual rewrite bytes=0");
    drop(engine);
    let reopened = Engine::open(config)?;
    for record in records() {
        assert_eq!(
            reopened.get_entry::<EntryIndex>(10, record.index)?,
            Some(record)
        );
    }
    assert_eq!(reopened.get(30, b"retained"), Some(vec![7; 64 << 10]));
    Ok(())
}

#[test]
fn append_work_does_not_reserve_or_copy_an_unselected_rewrite_queue() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (engine, config, writes) = rewritten_engine(root.path())?;
    let appended = records()[..8].to_vec();
    let mut batch = LogBatch::default();
    batch.put(30, b"append".to_vec(), vec![9; 256 << 10])?;
    engine.write(&mut batch, true)?;
    for record in &appended {
        let mut batch = LogBatch::default();
        batch.add_entries::<EntryIndex>(20, std::slice::from_ref(record))?;
        engine.write(&mut batch, true)?;
    }
    let boundary = engine.purge_manager().capture_append_boundary();
    assert!(!boundary.rewrite_queue);
    let allowance = engine.estimated_rewrite_bytes_for(&boundary).unwrap();
    let all_queues = engine
        .estimated_rewrite_bytes_through(boundary.append_watermark)
        .unwrap();
    assert!(allowance < 2 << 20);
    assert!(all_queues > allowance + (8 << 20));
    writes.0.store(0, Ordering::Relaxed);
    engine
        .purge_manager()
        .purge_expired_files_before(boundary)?;
    let actual = writes.0.load(Ordering::Relaxed);
    assert!(actual > 0 && actual <= allowance as u64);
    println!("append-only rewrite: all-queue allowance={all_queues}, selected allowance={allowance}, actual rewrite bytes={actual}");
    drop(engine);
    let reopened = Engine::open(config)?;
    for (group, records) in [(10, records()), (20, appended)] {
        for record in records {
            assert_eq!(
                reopened.get_entry::<EntryIndex>(group, record.index)?,
                Some(record)
            );
        }
    }
    assert_eq!(reopened.get(30, b"retained"), Some(vec![7; 64 << 10]));
    assert_eq!(reopened.get(30, b"append"), Some(vec![9; 256 << 10]));
    Ok(())
}

#[test]
fn later_compaction_cannot_add_an_unreserved_rewrite_queue() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (engine, config, writes) = rewritten_engine(root.path())?;
    let captured = engine.purge_manager().capture_append_boundary();
    assert!(!captured.rewrite_queue);
    let allowance = engine.estimated_rewrite_bytes_for(&captured).unwrap();
    assert!(allowance < 128 << 10);

    // Deterministically exercise foreground compaction between boundary
    // capture and native execution. The newly eligible queue was not reserved.
    engine.compact_to(10, 49);
    engine.sync()?;
    assert!(
        engine
            .purge_manager()
            .capture_append_boundary()
            .rewrite_queue
    );
    writes.0.store(0, Ordering::Relaxed);
    engine
        .purge_manager()
        .purge_expired_files_before(captured)?;
    assert_eq!(writes.0.load(Ordering::Relaxed), 0);

    // The following pass must still reclaim the queue with its own complete
    // allowance. Deferring this work must not silently disable reclamation.
    let next = engine.purge_manager().capture_append_boundary();
    assert!(next.rewrite_queue);
    let next_allowance = engine.estimated_rewrite_bytes_for(&next).unwrap();
    assert!(next_allowance > 2 << 20);
    engine.purge_manager().purge_expired_files_before(next)?;
    let actual = writes.0.load(Ordering::Relaxed);
    assert!(actual > 0 && actual <= next_allowance as u64);
    drop(engine);
    let reopened = Engine::open(config)?;
    assert!(reopened.get_entry::<EntryIndex>(10, 48)?.is_none());
    for record in records().into_iter().skip(48) {
        assert_eq!(
            reopened.get_entry::<EntryIndex>(10, record.index)?,
            Some(record)
        );
    }
    assert_eq!(reopened.get(30, b"retained"), Some(vec![7; 64 << 10]));
    Ok(())
}

#[test]
fn an_idle_rewrite_queue_maintenance_does_not_require_another_live_copy() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    group.bind_application(b"selected-rewrite-test/v1")?;
    let records = records();
    for record in &records {
        group.persist(
            std::slice::from_ref(record),
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: record.index,
                ..Default::default()
            }),
        )?;
    }
    let sibling = log.group(20)?;
    append(&sibling, 1, 4)?;
    group
        .node
        .engine
        .purge_manager()
        .must_rewrite_append_queue(None, None);
    assert!(!group
        .node
        .engine
        .purge_manager()
        .has_obsolete_rewrite_entries());
    // Measure with the production accountant before choosing this test's
    // ceiling. No writes or maintenance run concurrently with this fixture.
    let measured = crate::admission::Budget::new(
        root.path(),
        group.node.written.clone(),
        64 << 20,
        group.node.filesystem.clone(),
    )?
    .snapshot()
    .accounted_disk_bytes;
    let measured = usize::try_from(measured)?;
    let max_bytes = measured + measured / 2;
    group.configure_disk(DiskOptions {
        max_bytes,
        data_bytes: max_bytes - max_bytes / 4,
        headroom_bytes: max_bytes / 4,
    })?;
    // Keep the normal data/control split and more than two-file rotation
    // headroom. This pass copies neither queue, so the remaining space need
    // not fit another complete copy of the unchanged live rewrite queue.
    group.maintenance().run()?;
    let snapshot = group.disk_budget().unwrap().snapshot();
    assert_eq!(snapshot.reserved_rewrite_bytes, 0);
    assert!(snapshot.accounted_disk_bytes <= (measured + (128 << 10)) as u64);
    drop((group, sibling, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    group.bind_application(b"selected-rewrite-test/v1")?;
    assert_eq!(group.initial_state()?.hard_state.commit, 64);
    assert_eq!(
        group.entries(1, 65, None, GetEntriesContext::empty(false))?,
        records
    );
    verify(&reopened.group(20)?, 1, 4)?;
    Ok(())
}
