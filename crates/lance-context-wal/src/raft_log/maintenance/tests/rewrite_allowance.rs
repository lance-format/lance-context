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
fn native_two_queue_rewrite_bytes_fit_the_live_allowance() -> Result<()> {
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
    let allowance = engine.estimated_rewrite_bytes().unwrap() as u64;
    observed.bytes.store(0, Ordering::Relaxed);
    observed.purged.store(0, Ordering::Relaxed);
    engine.purge_expired_files()?;
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
    assert_eq!(reopened.get(20, b"value"), Some(vec![1; 16 << 10]));
    assert!(reopened.get(30, b"old").is_none());
    assert_eq!(reopened.get(30, b"new"), Some(vec![9; 4096]));
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
