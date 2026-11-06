use super::*;
use std::{sync::mpsc, time::Duration};

#[test]
fn append_watermark_cannot_expand_after_its_rewrite_allowance_is_reserved() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    let sibling = log.group(20)?;
    group.bind_application(b"append-cap-test/v1")?;
    append(&sibling, 1, 4)?;
    let records: Vec<_> = (1..=64)
        .map(|index| Entry {
            data: vec![index as u8; 64 << 10].into(),
            ..entry(10, index)
        })
        .collect();
    for (offset, batch) in records.chunks(32).enumerate() {
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
            group
                .node
                .engine
                .purge_manager()
                .must_rewrite_append_queue(None, None);
            checkpoint(&group, 28)?;
        }
    }
    group.configure_disk(DiskOptions {
        max_bytes: 32 << 20,
        data_bytes: 16 << 20,
        headroom_bytes: 4 << 20,
    })?;
    let budget = group.disk_budget().unwrap().clone();
    let boundary = group.node.engine.purge_manager().capture_append_boundary();
    let allowance = group
        .node
        .engine
        .estimated_rewrite_bytes_through(boundary.2)
        .unwrap();
    assert!(group.node.engine.estimated_rewrite_bytes().unwrap() > allowance + (256 << 10));
    let pause = group.node.fault.pause_rewrite_sync_for_test();
    let maintenance = group.maintenance();
    let maintain = std::thread::spawn(move || maintenance.run());
    if let Err(error) = pause.wait() {
        drop(pause);
        maintain.join().unwrap()?;
        return Err(error.context("fixture did not reach rewrite-queue sync"));
    }
    assert_eq!(
        budget.snapshot().reserved_proposal_bytes,
        (allowance + (128 << 10)) as u64
    );
    let suffix: Vec<_> = (65..=68)
        .map(|index| Entry {
            data: vec![index as u8; 1 << 20].into(),
            ..entry(10, index)
        })
        .collect();
    let writer = log.group(10)?;
    let input = suffix.clone();
    let (sent, received) = mpsc::sync_channel(1);
    let write = std::thread::spawn(move || {
        sent.send(writer.persist(
            &input,
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: 68,
                ..Default::default()
            }),
        ))
        .unwrap();
    });
    let written = received.recv_timeout(Duration::from_secs(2));
    let progressed = written.as_ref().is_ok_and(|r| r.is_ok());
    let expanded = progressed.then(|| {
        group
            .node
            .engine
            .purge_manager()
            .capture_append_boundary()
            .2
    });
    drop(pause);
    maintain.join().unwrap()?;
    write.join().unwrap();
    written.or_else(|_| received.recv_timeout(Duration::from_secs(2)))??;
    assert!(
        progressed,
        "append did not progress while native rewrite sync was paused"
    );
    assert!(expanded.is_some_and(|watermark| watermark > boundary.2));
    assert!(
        group.node.engine.estimated_rewrite_queue_bytes().unwrap() <= allowance,
        "later append growth expanded the already reserved rewrite prefix"
    );
    assert_eq!(budget.snapshot().reserved_proposal_bytes, 0);
    drop((group, sibling, budget, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    assert_eq!(group.initial_state()?.hard_state.commit, 68);
    assert_eq!(group.first_index()?, 29);
    assert_eq!(
        group.entries(29, 65, None, GetEntriesContext::empty(false))?,
        records[28..]
    );
    assert_eq!(
        group.entries(65, 69, None, GetEntriesContext::empty(false))?,
        suffix
    );
    verify(&reopened.group(20)?, 1, 4)?;
    Ok(())
}

#[test]
fn append_prefix_reclamation_fits_even_when_all_live_reservation_does_not() -> Result<()> {
    // First establish the actual copy footprint on an equivalent fresh engine.
    // No writes run concurrently with this storage-level diagnostic. Sync,
    // truncation and unlink observations retain the allocation peak before
    // old segments disappear; the data/retention workload is identical.
    for strict_budget in [false, true] {
        let root = tempfile::tempdir()?;
        let layout = layout();
        let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
        let group = log.group(10)?;
        let sibling = log.group(20)?;
        group.bind_application(b"append-admission-test/v1")?;
        append(&sibling, 1, 4)?;
        let records: Vec<_> = (1..=64)
            .map(|index| Entry {
                data: vec![index as u8; 64 << 10].into(),
                ..entry(10, index)
            })
            .collect();
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
        let snapshot = checkpoint(&group, 32)?;
        assert!(!group
            .node
            .engine
            .purge_manager()
            .has_obsolete_rewrite_entries());
        group.configure_disk(DiskOptions {
            max_bytes: if strict_budget { 6 << 20 } else { 12 << 20 },
            data_bytes: 4 << 20,
            headroom_bytes: 1 << 20,
        })?;
        let budget = group.disk_budget().unwrap().clone();
        let before = budget.snapshot().accounted_disk_bytes;
        let normal = group.node.engine.estimated_rewrite_bytes().unwrap() + (128 << 10);
        assert!(budget.reserve(4096, true).is_err());
        assert!(before + normal as u64 > 6 << 20);
        let peak = group
            .node
            .fault
            .observe_allocated_peak_for_test(root.path());
        let result = group.maintenance().run();
        let after = budget.snapshot().accounted_disk_bytes;
        eprintln!(
            "append admission: strict={strict_budget}, before={before}, after={after}, normal={normal}, actual_peak={}",
            peak.load(Ordering::Relaxed)
        );
        result?;
        assert!(after < before);
        assert!(peak.load(Ordering::Relaxed) < 6 << 20);
        assert!(budget.reserve(4096, true).is_ok());
        assert_eq!(
            group.entries(33, 65, None, GetEntriesContext::empty(false))?,
            records[32..]
        );
        verify(&sibling, 1, 4)?;
        drop((group, sibling, budget, log));
        let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
        let group = reopened.group(10)?;
        assert_eq!(group.snapshot(32, 2)?, snapshot);
        assert_eq!(group.initial_state()?.hard_state.commit, 64);
        assert_eq!(
            group.entries(33, 65, None, GetEntriesContext::empty(false))?,
            records[32..]
        );
        verify(&reopened.group(20)?, 1, 4)?;
    }
    Ok(())
}
