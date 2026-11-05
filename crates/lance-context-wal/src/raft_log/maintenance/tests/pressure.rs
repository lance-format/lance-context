use super::*;

#[test]
fn append_pressure_recovers_when_rewrite_file_numbers_are_ahead() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let catalog = log.group(10)?;
    let data = log.group(20)?;
    data.bind_application(b"maintenance-test/v1")?;
    let records: Vec<_> = (1..=96)
        .map(|index| Entry {
            data: vec![index as u8; 128].into(),
            ..entry(10, index)
        })
        .collect();
    catalog.persist(
        &records[..32],
        Some(&HardState {
            term: 1,
            vote: 1,
            commit: 32,
            ..Default::default()
        }),
    )?;
    // Both queues have independent file counters. A quiet append queue can
    // coexist with frequent rewrite reclamation of a small live prefix.
    let purge = catalog.node.engine.purge_manager();
    purge.must_rewrite_append_queue(None, None);
    for _ in 0..256 {
        purge.must_rewrite_rewrite_queue();
    }
    for index in 1..=64 {
        let large = Entry {
            data: vec![index as u8; 256 << 10].into(),
            ..entry(20, index)
        };
        data.persist(
            &[large],
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: index,
                ..Default::default()
            }),
        )?;
        catalog.persist(
            &records[index as usize + 31..index as usize + 32],
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: index + 32,
                ..Default::default()
            }),
        )?;
    }
    let snapshot = checkpoint(&data, 64)?;
    let native_files = files(root.path())?;
    let last_file = |extension: &str| {
        native_files
            .keys()
            .filter(|path| path.extension().is_some_and(|ext| ext == extension))
            .map(|path| {
                path.file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap()
            })
            .max()
            .unwrap()
    };
    assert!(last_file("rewrite") > last_file("raftlog"));
    catalog.configure_disk(DiskOptions {
        max_bytes: 32 << 20,
        data_bytes: 4 << 20,
        headroom_bytes: 8 << 20,
    })?;
    let budget = catalog.disk_budget().unwrap().clone();
    let before = budget.snapshot().accounted_disk_bytes;
    assert!(
        before > 12 << 20,
        "fixture did not retain obsolete append files: {before}"
    );
    assert!(budget.reserve(4096, true).is_err());
    let maintenance = catalog.maintenance();
    // No new writes or more logical compaction: reclamation must make progress
    // by itself, retaining all catalog entries and the data checkpoint.
    for _ in 0..32 {
        maintenance.run()?;
    }
    let after = budget.snapshot().accounted_disk_bytes;
    eprintln!("append pressure: before={before}, after={after}");
    assert!(
        after < 1 << 20,
        "obsolete append files still occupy {after} bytes"
    );
    assert!(budget.reserve(4096, true).is_ok());
    assert_eq!(
        catalog.entries(1, 97, None, GetEntriesContext::empty(false))?,
        records
    );
    drop((budget, maintenance, catalog, data, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    assert_eq!(
        reopened
            .group(10)?
            .entries(1, 97, None, GetEntriesContext::empty(false))?,
        records
    );
    let data = reopened.group(20)?;
    assert_eq!(data.snapshot(64, 2)?, snapshot);
    assert_eq!(data.first_index()?, 65);
    assert_eq!(data.last_index()?, 64);
    Ok(())
}

#[test]
fn compacted_large_records_release_admission_despite_many_small_live_values() -> Result<()> {
    release_admission(2 << 20, None)
}

#[test]
fn pending_image_files_trigger_collection_without_releasing_their_reservation() -> Result<()> {
    release_admission(8 << 20, Some(6 << 20))
}

#[test]
fn unchanged_live_payloads_are_not_rewritten_on_every_pressure_pass() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    group.bind_application(b"maintenance-test/v1")?;
    let records: Vec<_> = (1..=16)
        .map(|index| Entry {
            data: vec![index as u8; 256 << 10].into(),
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
    let maintenance = group.maintenance();
    for _ in 0..4 {
        maintenance.run()?;
    }
    group.configure_disk(DiskOptions {
        max_bytes: 32 << 20,
        data_bytes: 8 << 20,
        headroom_bytes: 16 << 20,
    })?;
    let budget = group.disk_budget().unwrap().clone();
    let image = budget.reserve_files(&[6 << 20])?;
    assert!(budget.reserve(4096, true).is_err());
    // Let the first pressure pass reclaim any previous garbage. Subsequent
    // passes have no new writes, compaction, or expired payloads to collect.
    maintenance.run()?;
    let mut rewritten_bytes = 0;
    for _ in 0..3 {
        let before = files(root.path())?;
        maintenance.run()?;
        let added: u64 = files(root.path())?
            .iter()
            .filter(|(path, _)| {
                path.extension().is_some_and(|ext| ext == "rewrite") && !before.contains_key(*path)
            })
            .map(|(_, bytes)| *bytes)
            .sum();
        rewritten_bytes += added;
        assert_eq!(budget.snapshot().reserved_file_bytes, 6 << 20);
        assert!(budget.reserve(4096, true).is_err());
    }
    drop(image);
    group.disk_measurement()?.run()?;
    assert!(budget.reserve(4096, true).is_ok());
    drop((budget, maintenance, group, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    assert_eq!(group.initial_state()?.hard_state.commit, 16);
    assert_eq!(
        group.entries(1, 17, None, GetEntriesContext::empty(false))?,
        records
    );
    assert!(
        rewritten_bytes < 256 << 10,
        "unchanged live payloads rewritten under file reservation pressure: {rewritten_bytes} bytes"
    );
    Ok(())
}

fn release_admission(data_bytes: usize, file_bytes: Option<u64>) -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    let sibling = log.group(20)?;
    group.bind_application(b"maintenance-test/v1")?;
    append(&sibling, 1, 4)?;
    // Native rewrite eligibility counts entries, including metadata keys. A
    // small number of large expired entries can dominate disk bytes while
    // these live values keep the entry-count garbage ratio below its trigger.
    let mut metadata = LogBatch::default();
    for key in 0u32..2000 {
        metadata.put(20, key.to_be_bytes().to_vec(), vec![7; 8])?;
    }
    group.node.write(&mut metadata)?;
    for index in 1..=16 {
        let mut record = entry(10, index);
        record.data = vec![index as u8; 256 << 10].into();
        group.persist(
            &[record],
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: index,
                ..Default::default()
            }),
        )?;
    }
    let maintenance = group.maintenance();
    for _ in 0..4 {
        maintenance.run()?;
    }
    assert!(
        files(root.path())?
            .keys()
            .filter(|path| path.extension().is_some_and(|ext| ext == "rewrite"))
            .count()
            > 1
    );
    let snapshot = checkpoint(&group, 16)?;
    group.configure_disk(DiskOptions {
        max_bytes: 32 << 20,
        data_bytes,
        headroom_bytes: 16 << 20,
    })?;
    let budget = group.disk_budget().unwrap().clone();
    // Keep an image build/receive reservation alive throughout collection. The
    // expired payloads alone fit this case's data ceiling; together with the
    // pending file they block producers until physical collection releases them.
    let file = file_bytes
        .map(|bytes| budget.reserve_files(&[bytes]))
        .transpose()?;
    let before = budget.snapshot().accounted_disk_bytes;
    assert!(before > 2 << 20, "fixture did not fill admission: {before}");
    if file_bytes.is_some() {
        assert!(before + 2 * group.node.target_file_bytes < data_bytes as u64);
    }
    assert!(budget.reserve(256 << 10, true).is_err());
    for _ in 0..4 {
        maintenance.run()?;
    }
    let after = budget.snapshot().accounted_disk_bytes;
    eprintln!("compacted payload disk bytes: before={before}, after={after}");
    assert!(
        after < 1 << 20,
        "compacted payloads still occupy {after} bytes"
    );
    assert!(budget.reserve(256 << 10, true).is_ok());
    assert_eq!(
        budget.snapshot().reserved_file_bytes,
        file_bytes.unwrap_or_default()
    );
    verify(&sibling, 1, 4)?;
    drop((file, budget, maintenance, group, sibling, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    assert_eq!(group.snapshot(16, 2)?, snapshot);
    assert_eq!(group.first_index()?, 17);
    assert_eq!(group.last_index()?, 16);
    verify(&reopened.group(20)?, 1, 4)?;
    for key in 0u32..2000 {
        assert_eq!(
            group.node.engine.get(20, &key.to_be_bytes()),
            Some(vec![7; 8])
        );
    }
    Ok(())
}
