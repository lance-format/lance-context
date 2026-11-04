use super::*;
use raft_engine::internals::{EventListener, FileBlockHandle, FileId, LogQueue};

#[derive(Default)]
struct RewriteObservation {
    bytes: AtomicU64,
    purged: AtomicU64,
}
impl EventListener for RewriteObservation {
    fn on_append_log_file(&self, handle: FileBlockHandle) {
        if handle.id.queue == LogQueue::Rewrite {
            self.bytes.fetch_add(handle.len as u64, Ordering::Relaxed);
        }
    }
    fn post_purge(&self, file: FileId) {
        let queue = match file.queue {
            LogQueue::Append => 1,
            LogQueue::Rewrite => 2,
        };
        self.purged.fetch_or(queue, Ordering::Relaxed);
    }
}

#[test]
fn captured_prefix_allowance_covers_both_queues_with_later_appends() -> Result<()> {
    let root = tempfile::tempdir()?;
    let config = raft_engine::Config {
        dir: root.path().to_str().unwrap().to_owned(),
        target_file_size: raft_engine::ReadableSize(64 << 10),
        purge_threshold: raft_engine::ReadableSize(64 << 10),
        purge_rewrite_threshold: Some(raft_engine::ReadableSize(64 << 10)),
        purge_rewrite_garbage_ratio: 0.0,
        batch_compression_threshold: raft_engine::ReadableSize(0),
        enable_log_recycle: false,
        ..Default::default()
    };
    let observed = Arc::new(RewriteObservation::default());
    let engine = Engine::open_with_listeners(config.clone(), vec![observed.clone()])?;
    let mut records: Vec<_> = (1..=16)
        .map(|index| Entry {
            data: vec![index as u8; 128 << 10].into(),
            ..entry(10, index)
        })
        .collect();
    for (offset, records) in records.chunks(8).enumerate() {
        for record in records {
            let mut batch = LogBatch::default();
            batch.add_entries::<EntryIndex>(10, std::slice::from_ref(record))?;
            engine.write(&mut batch, true)?;
        }
        let mut batch = LogBatch::default();
        batch.put(20, b"value".to_vec(), vec![offset as u8; 16 << 10])?;
        if offset == 0 {
            batch.put(30, b"old".to_vec(), vec![7; 4096])?;
            engine.write(&mut batch, true)?;
            engine.purge_manager().must_rewrite_append_queue(None, None);
        } else {
            batch.add_command(30, raft_engine::Command::Clean);
            batch.put(30, b"new".to_vec(), vec![9; 4096])?;
            engine.write(&mut batch, true)?;
        }
    }
    engine.compact_to(10, 3);
    engine.sync()?;
    let boundary = engine.purge_manager().capture_append_boundary();
    let allowance = engine.estimated_rewrite_bytes_before(boundary).unwrap() as u64;
    let all_live = engine.estimated_rewrite_bytes().unwrap() as u64;
    assert!(all_live > allowance + (256 << 10));
    // These writes rotate the append queue and move its normal watermark.
    // The captured pass must still fit its original allowance, including when
    // a key or group is replaced after capture.
    let later = Entry {
        data: vec![17; 4 << 20].into(),
        ..entry(10, 17)
    };
    let mut batch = LogBatch::default();
    batch.add_entries::<EntryIndex>(10, std::slice::from_ref(&later))?;
    batch.put(20, b"value".to_vec(), vec![2; 1 << 20])?;
    batch.add_command(30, raft_engine::Command::Clean);
    batch.put(30, b"later".to_vec(), vec![8; 1 << 20])?;
    engine.write(&mut batch, true)?;
    records.push(later);
    observed.bytes.store(0, Ordering::Relaxed);
    observed.purged.store(0, Ordering::Relaxed);
    engine
        .purge_manager()
        .purge_expired_files_before(boundary)?;
    let written = observed.bytes.load(Ordering::Relaxed);
    assert_eq!(observed.purged.load(Ordering::Relaxed), 3);
    assert!(
        written > 0 && written <= allowance,
        "{written} > {allowance}"
    );
    drop(engine);
    let reopened = Engine::open(config)?;
    assert!(reopened.get_entry::<EntryIndex>(10, 2)?.is_none());
    for record in &records[2..] {
        assert_eq!(
            reopened.get_entry::<EntryIndex>(10, record.index)?,
            Some(record.clone())
        );
    }
    assert_eq!(reopened.get(20, b"value"), Some(vec![2; 1 << 20]));
    assert!(reopened.get(30, b"old").is_none());
    assert!(reopened.get(30, b"new").is_none());
    assert_eq!(reopened.get(30, b"later"), Some(vec![8; 1 << 20]));
    Ok(())
}

#[test]
fn one_live_copy_headroom_reclaims_both_queues_and_recovers_every_record() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    let sibling = log.group(20)?;
    group.bind_application(b"maintenance-test/v1")?;
    append(&sibling, 1, 4)?;
    let records: Vec<_> = (1..=16)
        .map(|index| Entry {
            data: vec![index as u8; 128 << 10].into(),
            ..entry(10, index)
        })
        .collect();
    for (offset, batch) in records.chunks(8).enumerate() {
        for record in batch {
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
        if offset == 0 {
            // Put the first half in the rewrite queue, then leave the second
            // half in append files. No logical entry is discarded by this test.
            group
                .node
                .engine
                .purge_manager()
                .must_rewrite_append_queue(None, None);
        }
    }
    let before = files(root.path())?;
    for extension in ["raftlog", "rewrite"] {
        assert!(before
            .keys()
            .any(|path| path.extension().is_some_and(|ext| ext == extension)));
    }
    group.configure_disk(DiskOptions {
        max_bytes: 6 << 20,
        data_bytes: 2 << 20,
        headroom_bytes: 2 << 20,
    })?;
    let budget = group.disk_budget().unwrap().clone();
    assert!(budget.reserve(256 << 10, true).is_err());
    let maintenance = group.maintenance();
    // Both queues share the live records, rather than each containing a copy
    // of all of them. The six-MiB ceiling has room for their actual rewrite.
    for _ in 0..4 {
        maintenance.run()?;
    }
    assert!(budget.snapshot().accounted_disk_bytes < 4 << 20);
    assert!(before.keys().any(|path| !path.exists()));
    assert_eq!(
        group.entries(1, 17, None, GetEntriesContext::empty(false))?,
        records
    );
    verify(&sibling, 1, 4)?;
    drop((budget, maintenance, group, sibling, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    group.bind_application(b"maintenance-test/v1")?;
    assert_eq!(group.initial_state()?.hard_state.commit, 16);
    assert_eq!(
        group.entries(1, 17, None, GetEntriesContext::empty(false))?,
        records
    );
    verify(&reopened.group(20)?, 1, 4)?;
    Ok(())
}

#[test]
fn rewrite_queue_allowance_covers_actual_bytes_and_preserves_both_queues() -> Result<()> {
    let root = tempfile::tempdir()?;
    let config = raft_engine::Config {
        dir: root.path().to_str().unwrap().to_owned(),
        target_file_size: raft_engine::ReadableSize(64 << 10),
        batch_compression_threshold: raft_engine::ReadableSize(0),
        enable_log_recycle: false,
        ..Default::default()
    };
    let observed = Arc::new(RewriteObservation::default());
    let engine = Engine::open_with_listeners(config.clone(), vec![observed.clone()])?;
    let records: Vec<_> = (1..=16)
        .map(|index| Entry {
            data: vec![index as u8; 128 << 10].into(),
            ..entry(10, index)
        })
        .collect();
    for (offset, records) in records.chunks(8).enumerate() {
        for record in records {
            let mut batch = LogBatch::default();
            batch.add_entries::<EntryIndex>(10, std::slice::from_ref(record))?;
            engine.write(&mut batch, true)?;
        }
        let mut batch = LogBatch::default();
        if offset == 0 {
            batch.put(20, b"retained".to_vec(), vec![7; 64 << 10])?;
            batch.put(30, b"removed".to_vec(), vec![9; 64 << 10])?;
            engine.write(&mut batch, true)?;
            engine.purge_manager().must_rewrite_append_queue(None, None);
        } else {
            batch.put(20, b"append".to_vec(), vec![3; 256 << 10])?;
            batch.add_command(30, raft_engine::Command::Clean);
            engine.write(&mut batch, true)?;
        }
    }
    engine.compact_to(10, 5);
    engine.sync()?;
    let allowance = engine.estimated_rewrite_queue_bytes().unwrap() as u64;
    let all_live = engine.estimated_rewrite_bytes().unwrap() as u64;
    assert!(all_live > allowance + (1 << 20));
    observed.bytes.store(0, Ordering::Relaxed);
    engine.purge_manager().reclaim_rewrite_queue()?;
    let written = observed.bytes.load(Ordering::Relaxed);
    assert!(
        written > 0 && written <= allowance,
        "{written} > {allowance}"
    );
    assert_eq!(
        engine
            .purge_manager()
            .capture_append_boundary()
            .cleaned_regions(),
        1
    );
    drop(engine);
    let reopened = Engine::open(config)?;
    assert!(reopened.get_entry::<EntryIndex>(10, 4)?.is_none());
    for record in &records[4..] {
        assert_eq!(
            reopened.get_entry::<EntryIndex>(10, record.index)?,
            Some(record.clone())
        );
    }
    assert_eq!(reopened.get(20, b"retained"), Some(vec![7; 64 << 10]));
    assert_eq!(reopened.get(20, b"append"), Some(vec![3; 256 << 10]));
    assert!(reopened.get(30, b"removed").is_none());
    Ok(())
}

#[test]
fn rewrite_queue_reclamation_does_not_reserve_the_unmodified_append_tail() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    group.bind_application(b"queue-allowance-test/v1")?;
    let records: Vec<_> = (1..=64)
        .map(|index| Entry {
            data: vec![index as u8; 64 << 10].into(),
            ..entry(10, index)
        })
        .collect();
    for record in &records[..16] {
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
    append(&log.group(20)?, 1, 4)?;
    group
        .node
        .engine
        .purge_manager()
        .must_rewrite_append_queue(None, None);
    let snapshot = checkpoint(&group, 12)?;
    for record in &records[16..] {
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
    group.configure_disk(DiskOptions {
        max_bytes: 12 << 20,
        data_bytes: 6 << 20,
        headroom_bytes: 2 << 20,
    })?;
    let budget = group.disk_budget().unwrap().clone();
    assert!(budget.reserve(64 << 10, true).is_ok());
    let pause = group.node.fault.pause_rewrite_sync_for_test();
    let maintenance = group.maintenance();
    let maintain = std::thread::spawn(move || {
        // Exercise the pressure fallback with the same node-wide serialization
        // and filesystem lifecycle that LogMaintenance::run owns.
        let _job = maintenance.node.maintenance_job.lock().unwrap();
        let _files = maintenance.node.filesystem.write().unwrap();
        maintenance.run_pass(true)
    });
    if let Err(error) = pause.wait() {
        drop(pause);
        maintain.join().unwrap()?;
        return Err(error);
    }
    let reserved = budget.snapshot().reserved_proposal_bytes;
    let producer = budget.reserve(64 << 10, true);
    let admitted = producer.is_ok();
    drop(pause);
    maintain.join().unwrap()?;
    drop(producer);
    assert_eq!(budget.snapshot().reserved_proposal_bytes, 0);
    assert_eq!(
        group.entries(13, 65, None, GetEntriesContext::empty(false))?,
        records[12..]
    );
    drop((group, budget, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    assert_eq!(group.snapshot(12, 2)?, snapshot);
    assert_eq!(group.initial_state()?.hard_state.commit, 64);
    assert_eq!(
        group.entries(13, 65, None, GetEntriesContext::empty(false))?,
        records[12..]
    );
    verify(&reopened.group(20)?, 1, 4)?;
    eprintln!("rewrite queue allowance: reserved_bytes={reserved}, producer_admitted={admitted}; all 56 retained records reopened");
    assert!(
        admitted,
        "rewrite-only pass charged the unrelated append tail"
    );
    assert!(reserved < 1 << 20);
    Ok(())
}
