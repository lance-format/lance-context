use super::*;

fn layout() -> Layout {
    Layout {
        cluster: uuid::Uuid::from_u128(2202),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3])]),
        buckets: vec![10],
    }
}

fn entries() -> Vec<Entry> {
    [(1, 32), (1, 512), (2, 128), (2, 4096), (3, 256)]
        .into_iter()
        .enumerate()
        .map(|(i, (term, size))| Entry {
            index: i as u64 + 1,
            term,
            data: vec![i as u8; size].into(),
            ..Default::default()
        })
        .collect()
}

fn snapshot(group: &GroupLog, index: u64, term: u64) -> Result<Snapshot> {
    let mut snapshot = Snapshot::default();
    snapshot.mut_metadata().index = index;
    snapshot.mut_metadata().term = term;
    snapshot
        .mut_metadata()
        .set_conf_state(group.initial_state()?.conf_state);
    snapshot.data = format!("test-checkpoint/{index}").into_bytes().into();
    Ok(snapshot)
}

fn open(path: &Path) -> Result<PhysicalLog> {
    PhysicalLog::open(path, &layout(), 1)
}

fn seed(log: &PhysicalLog) -> Result<GroupLog> {
    let group = log.group(10)?;
    group.bind_application(b"retained-tail-test")?;
    group.persist(
        &entries(),
        Some(&HardState {
            term: 3,
            vote: 1,
            commit: 5,
            ..Default::default()
        }),
    )?;
    Ok(group)
}

#[test]
fn retained_tail_uses_encoded_bytes_and_survives_restart_and_further_compaction() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = open(directory.path())?;
    let group = seed(&log)?;
    let entries = entries();
    let allowance = entries[2..4]
        .iter()
        .map(|e| u64::from(e.compute_size()) + 128)
        .sum();
    let cut = snapshot(&group, 4, 2)?;
    group.checkpoint_retaining(&cut, allowance)?;
    assert_eq!(group.first_index()?, 3);
    assert_eq!(group.checkpoint_index()?, 4);
    assert_eq!(group.term(2)?, 1);
    assert_eq!(group.term(3)?, 2);
    assert!(group.term(1).is_err());
    assert_eq!(
        group.entries(3, 6, None, GetEntriesContext::empty(false))?,
        entries[2..]
    );
    group.maintenance().run()?;
    drop((group, log));

    let log = open(directory.path())?;
    let group = log.group(10)?;
    assert_eq!(group.first_index()?, 3);
    assert_eq!(group.snapshot(4, 2)?, cut);
    assert_eq!(group.term(2)?, 1);
    assert_eq!(
        group.entries(3, 6, None, GetEntriesContext::empty(false))?,
        entries[2..]
    );
    // A single oversized last entry does not exceed the requested allowance.
    let newer = snapshot(&group, 5, 3)?;
    group.checkpoint_retaining(&newer, 1)?;
    assert_eq!(group.first_index()?, 6);
    assert_eq!(group.term(5)?, 3);
    assert!(group
        .entries(4, 6, None, GetEntriesContext::empty(false))
        .is_err());
    assert!(group.checkpoint_retaining(&cut, allowance).is_err());
    drop((group, log));
    let log = open(directory.path())?;
    assert_eq!(log.group(10)?.first_index()?, 6);
    assert_eq!(log.group(10)?.snapshot(5, 2)?, newer);
    Ok(())
}

#[test]
fn checkpoint_can_retain_from_index_one_and_zero_allowance_retires_the_tail() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = open(directory.path())?;
    let group = seed(&log)?;
    let cut = snapshot(&group, 5, 3)?;
    group.checkpoint_retaining(&cut, 1 << 20)?;
    assert_eq!(group.first_index()?, 1);
    assert_eq!(group.checkpoint_index()?, 5);
    drop((group, log));
    let log = open(directory.path())?;
    let group = log.group(10)?;
    assert_eq!(group.snapshot(5, 2)?, cut);
    assert_eq!(
        group.entries(1, 6, None, GetEntriesContext::empty(false))?,
        entries()
    );
    group.checkpoint(&cut)?;
    assert_eq!(group.first_index()?, 6);
    drop((group, log));
    assert_eq!(open(directory.path())?.group(10)?.first_index()?, 6);
    Ok(())
}

#[test]
fn received_snapshot_replaces_the_retained_tail_and_its_boundary() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = open(directory.path())?;
    let group = seed(&log)?;
    group.checkpoint_retaining(&snapshot(&group, 5, 3)?, 1 << 20)?;
    let remote = snapshot(&group, 7, 4)?;
    let suffix = Entry {
        index: 8,
        term: 4,
        data: vec![8; 64].into(),
        ..Default::default()
    };
    group.install_snapshot(
        &remote,
        std::slice::from_ref(&suffix),
        &HardState {
            term: 4,
            vote: 1,
            commit: 8,
            ..Default::default()
        },
    )?;
    assert_eq!(group.first_index()?, 8);
    assert_eq!(group.checkpoint_index()?, 7);
    drop((group, log));
    let log = open(directory.path())?;
    let group = log.group(10)?;
    assert_eq!(group.first_index()?, 8);
    assert_eq!(group.snapshot(7, 2)?, remote);
    assert_eq!(
        group.entries(8, 9, None, GetEntriesContext::empty(false))?,
        vec![suffix]
    );
    Ok(())
}

#[test]
fn missing_boundary_metadata_accepts_only_the_legacy_fully_compacted_layout() -> Result<()> {
    for retained in [0, 1 << 20] {
        let directory = tempfile::tempdir()?;
        let log = open(directory.path())?;
        let group = seed(&log)?;
        let cut = snapshot(&group, 5, 3)?;
        group.checkpoint_retaining(&cut, retained)?;
        let mut change = LogBatch::default();
        change.delete(10, TRUNCATED.to_vec());
        group.node.write(&mut change)?;
        drop((group, log));
        if retained == 0 {
            // This is the metadata layout written by previous binaries.
            let log = open(directory.path())?;
            let group = log.group(10)?;
            assert_eq!(group.first_index()?, 6);
            assert_eq!(group.snapshot(5, 2)?, cut);
        } else {
            let error = open(directory.path())
                .err()
                .expect("missing tail boundary accepted");
            assert!(error.to_string().contains("invalid durable term index"));
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "replication")]
fn uncertain_tail_checkpoint_fences_reads_and_reopens_one_complete_boundary() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = open(directory.path())?;
    let group = seed(&log)?;
    let cut = snapshot(&group, 4, 2)?;
    let allowance = entries()[2..4]
        .iter()
        .map(|e| u64::from(e.compute_size()) + 128)
        .sum();
    group.node.fault.fail_sync_for_test();
    assert!(group.checkpoint_retaining(&cut, allowance).is_err());
    assert!(group.first_index().is_err());
    assert!(group.checkpoint_index().is_err());
    assert!(group.persist(&[], None).is_err());
    drop((group, log));
    let log = open(directory.path())?;
    let group = log.group(10)?;
    let first = group.first_index()?;
    assert!(matches!(
        (group.checkpoint_index()?, first),
        (0, 1) | (4, 3)
    ));
    assert_eq!(
        group.entries(first, 6, None, GetEntriesContext::empty(false))?,
        entries()[first as usize - 1..]
    );
    Ok(())
}
