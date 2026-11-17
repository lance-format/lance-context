use super::*;

#[test]
fn pressure_rewrite_yields_before_copying_the_entire_uncheckpointed_prefix() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    group.bind_application(b"bounded-maintenance/v1")?;
    append(&log.group(20)?, 1, 4)?;
    let records: Vec<_> = (1..=64)
        .map(|index| Entry {
            data: vec![index as u8; 4 << 20].into(),
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
    group.configure_disk(DiskOptions {
        max_bytes: 1 << 30,
        data_bytes: 256 << 20,
        headroom_bytes: 128 << 20,
    })?;
    assert!(group.disk_budget().unwrap().reserve(4096, true).is_err());

    // The real maintenance call must return before copying most of this live
    // prefix. Its owner can then checkpoint already materialized data, avoiding
    // copies that become obsolete as soon as the blocking pass finishes.
    group.maintenance().run()?;
    let rewrite_bytes: u64 = files(root.path())?
        .into_iter()
        .filter(|(path, _)| path.extension().is_some_and(|ext| ext == "rewrite"))
        .map(|(_, bytes)| bytes)
        .sum();
    assert_eq!(
        group.entries(1, 65, None, GetEntriesContext::empty(false))?,
        records
    );
    assert!(
        rewrite_bytes < 160 << 20,
        "one pass copied {rewrite_bytes} bytes before allowing a fresh checkpoint"
    );
    checkpoint(&group, 56)?;
    for _ in 0..12 {
        group.maintenance().run()?;
    }
    assert!(group.disk_budget().unwrap().snapshot().accounted_disk_bytes < 64 << 20);
    assert!(group.disk_budget().unwrap().reserve(4096, true).is_ok());
    drop((group, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    assert_eq!(group.first_index()?, 57);
    assert_eq!(group.initial_state()?.hard_state.commit, 64);
    assert_eq!(
        group.entries(57, 65, None, GetEntriesContext::empty(false))?,
        records[56..]
    );
    verify(&reopened.group(20)?, 1, 4)?;
    Ok(())
}
