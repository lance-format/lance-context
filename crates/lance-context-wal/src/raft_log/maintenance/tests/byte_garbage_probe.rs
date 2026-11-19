use super::*;

fn populate(data: &GroupLog, sibling: &GroupLog) -> Result<Vec<Entry>> {
    data.bind_application(b"byte-garbage-probe/v1")?;
    append(sibling, 1, 4)?;
    let mut metadata = LogBatch::default();
    for key in 0u32..2000 {
        metadata.put(20, key.to_be_bytes().to_vec(), vec![7; 8])?;
    }
    data.node.write(&mut metadata)?;
    let records: Vec<_> = (1..=16)
        .map(|index| Entry {
            data: vec![index as u8; 256 << 10].into(),
            ..entry(10, index)
        })
        .collect();
    for record in &records {
        data.persist(
            std::slice::from_ref(record),
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: record.index,
                ..Default::default()
            }),
        )?;
    }
    Ok(records)
}

#[test]
fn obsolete_payload_bytes_below_and_above_producer_pressure() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let data = log.group(10)?;
    let sibling = log.group(20)?;
    populate(&data, &sibling)?;
    let maintenance = data.maintenance();
    for _ in 0..4 {
        maintenance.run()?;
    }
    let snapshot = checkpoint(&data, 16)?;
    data.configure_disk(DiskOptions {
        max_bytes: 64 << 20,
        data_bytes: 16 << 20,
        headroom_bytes: 32 << 20,
    })?;
    let budget = data.disk_budget().unwrap().clone();
    let report = |phase: &str| -> Result<u64> {
        let physical = files(root.path())?;
        let rewrite_bytes: u64 = physical
            .iter()
            .filter(|(p, _)| p.extension().is_some_and(|ext| ext == "rewrite"))
            .map(|(_, size)| *size)
            .sum();
        let observed = budget.snapshot();
        eprintln!(
            "byte-garbage-probe phase={phase} accounted={} rewrite_file_bytes={} live_rewrite_allowance={} reserved_file_bytes={} obsolete_rewrite_entries={}",
            observed.accounted_disk_bytes,
            rewrite_bytes,
            data.node.engine.estimated_rewrite_queue_bytes().unwrap(),
            observed.reserved_file_bytes,
            data.node
                .engine
                .purge_manager()
                .has_obsolete_rewrite_entries()
        );
        Ok(observed.accounted_disk_bytes)
    };
    let before = report("below-pressure-before")?;
    assert!(before > 2 << 20, "fixture did not retain the old payloads");
    drop(budget.reserve(256 << 10, true)?);
    assert!(!maintenance.data_under_pressure());
    for _ in 0..4 {
        maintenance.run()?;
    }
    let after_normal = report("below-pressure-after-four-passes")?;
    assert!(
        after_normal < 1 << 20,
        "large obsolete bodies survived below-pressure reclamation: {after_normal}"
    );
    assert!(!maintenance.data_under_pressure());
    drop(budget.reserve(256 << 10, true)?);

    // Hold a real bounded file reservation to cross producer admission without
    // changing any retained entry, checkpoint, disk budget or file contents.
    let file_bytes = budget
        .snapshot()
        .data_limit
        .saturating_sub(after_normal)
        .saturating_add(1 << 20);
    let image = budget.reserve_files(&[file_bytes])?;
    assert!(budget.reserve(256 << 10, true).is_err());
    assert!(maintenance.data_under_pressure());
    report("pressure-before")?;
    maintenance.run()?;
    report("pressure-after-one-pass")?;
    assert_eq!(budget.snapshot().reserved_file_bytes, file_bytes);
    // The held file reservation alone exceeds the data window after early GC.
    // Reclamation must not erase that charge to make producer admission pass.
    assert!(budget.reserve(256 << 10, true).is_err());
    verify(&sibling, 1, 4)?;
    drop((image, budget, maintenance, data, sibling, log));

    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let data = reopened.group(10)?;
    assert_eq!(data.snapshot(16, 2)?, snapshot);
    assert_eq!(data.first_index()?, 17);
    assert_eq!(data.last_index()?, 16);
    verify(&reopened.group(20)?, 1, 4)?;
    for key in 0u32..2000 {
        assert_eq!(
            data.node.engine.get(20, &key.to_be_bytes()),
            Some(vec![7; 8])
        );
    }
    Ok(())
}

#[test]
fn mostly_live_rewrite_payloads_are_not_copied_on_every_pass() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let data = log.group(10)?;
    let sibling = log.group(20)?;
    let records = populate(&data, &sibling)?;
    let maintenance = data.maintenance();
    for _ in 0..4 {
        maintenance.run()?;
    }
    let snapshot = checkpoint(&data, 1)?;
    data.configure_disk(DiskOptions {
        max_bytes: 64 << 20,
        data_bytes: 16 << 20,
        headroom_bytes: 32 << 20,
    })?;
    assert!(data
        .node
        .engine
        .purge_manager()
        .has_obsolete_rewrite_entries());
    assert!(!maintenance.data_under_pressure());
    let mut rewritten_bytes = 0;
    for _ in 0..4 {
        let before = files(root.path())?;
        maintenance.run()?;
        rewritten_bytes += files(root.path())?
            .iter()
            .filter(|(p, _)| p.extension().is_some_and(|ext| ext == "rewrite"))
            .map(|(p, bytes)| bytes.saturating_sub(*before.get(p).unwrap_or(&0)))
            .sum::<u64>();
    }
    assert!(
        rewritten_bytes < 256 << 10,
        "mostly live payloads copied repeatedly: {rewritten_bytes} bytes"
    );
    eprintln!("byte-garbage-probe mostly-live rewritten_bytes={rewritten_bytes}");
    assert_eq!(
        data.entries(2, 17, None, GetEntriesContext::empty(false))?,
        records[1..]
    );
    drop((maintenance, data, sibling, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let data = reopened.group(10)?;
    assert_eq!(data.snapshot(1, 2)?, snapshot);
    assert_eq!(data.initial_state()?.hard_state.commit, 16);
    assert_eq!(
        data.entries(2, 17, None, GetEntriesContext::empty(false))?,
        records[1..]
    );
    verify(&reopened.group(20)?, 1, 4)?;
    for key in 0u32..2000 {
        assert_eq!(
            data.node.engine.get(20, &key.to_be_bytes()),
            Some(vec![7; 8])
        );
    }
    Ok(())
}
