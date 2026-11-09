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

#[test]
fn an_idle_rewrite_queue_has_live_bytes_but_no_native_copy() -> Result<()> {
    let root = tempfile::tempdir()?;
    let config = raft_engine::Config {
        dir: root.path().to_str().unwrap().to_owned(),
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
    engine.purge_manager().must_rewrite_append_queue(None, None);
    assert!(!engine.purge_manager().has_obsolete_rewrite_entries());
    let boundary = engine.purge_manager().capture_append_boundary();
    let allowance = engine.estimated_rewrite_bytes_through(boundary.2).unwrap();
    assert!(allowance > 8 << 20);
    writes.0.store(0, Ordering::Relaxed);
    engine
        .purge_manager()
        .purge_expired_files_before(boundary)?;
    assert_eq!(writes.0.load(Ordering::Relaxed), 0);
    println!("idle rewrite: live allowance={allowance}, actual rewrite bytes=0");
    drop(engine);
    let reopened = Engine::open(config)?;
    for record in records {
        assert_eq!(
            reopened.get_entry::<EntryIndex>(10, record.index)?,
            Some(record)
        );
    }
    Ok(())
}

#[test]
fn an_idle_rewrite_queue_maintenance_fits_rotation_headroom() -> Result<()> {
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
    group.configure_disk(DiskOptions {
        max_bytes: measured + (256 << 10),
        data_bytes: measured + (128 << 10),
        headroom_bytes: 128 << 10,
    })?;
    // This pass copies neither queue. Retain two-file rotation headroom,
    // but do not require a second copy of the unchanged live rewrite queue.
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
