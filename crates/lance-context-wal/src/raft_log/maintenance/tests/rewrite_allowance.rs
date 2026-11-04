use super::*;

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
