use super::*;

#[test]
fn pressure_reclamation_progresses_when_normal_rewrite_admission_is_closed() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    group.bind_application(b"pressure-admission-test/v1")?;
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
        max_bytes: 6 << 20,
        data_bytes: 4 << 20,
        headroom_bytes: 1 << 20,
    })?;
    let budget = group.disk_budget().unwrap().clone();
    let before = budget.snapshot().accounted_disk_bytes;
    assert!(budget.reserve(4096, true).is_err());
    let rotation = 2 * usize::try_from(group.node.target_file_bytes)?;
    let normal = group.node.engine.estimated_rewrite_bytes().unwrap() + rotation;
    let pressure = group.node.engine.estimated_rewrite_queue_bytes().unwrap() + rotation;
    assert!(budget.reserve(normal, false).is_err());
    drop(budget.reserve(pressure, false)?);

    // Exercise the public maintenance entry point. No new writes, lower logical
    // retention, released reservations or enlarged disk budget can unblock it.
    group.maintenance().run()?;
    let after = budget.snapshot().accounted_disk_bytes;
    assert!(after < before, "physical reclamation made no progress");
    assert!(budget.reserve(4096, true).is_ok());
    assert_eq!(budget.snapshot().reserved_proposal_bytes, 0);
    assert_eq!(
        group.entries(13, 65, None, GetEntriesContext::empty(false))?,
        records[12..]
    );
    verify(&log.group(20)?, 1, 4)?;
    eprintln!("maintenance admission: before={before}, after={after}, normal={normal}, pressure={pressure}");
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
    Ok(())
}
