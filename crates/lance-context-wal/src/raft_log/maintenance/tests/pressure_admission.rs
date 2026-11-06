use super::*;

#[test]
fn pressure_reclamation_progresses_when_normal_rewrite_admission_is_closed() -> Result<()> {
    exercise(Failure::None)
}

#[test]
fn pressure_fallback_respects_held_reservations_without_mutating_or_fencing() -> Result<()> {
    exercise(Failure::Admission)
}

#[test]
fn pressure_fallback_sync_failure_fences_siblings_and_preserves_recovery() -> Result<()> {
    exercise(Failure::RewriteSync)
}

enum Failure {
    None,
    Admission,
    RewriteSync,
}

fn exercise(failure: Failure) -> Result<()> {
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
    let sibling = log.group(20)?;
    append(&sibling, 1, 4)?;
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

    if matches!(&failure, Failure::Admission) {
        // Keep only 64 KiB free: the pressure pass must obtain its own allowance.
        let held_bytes = budget.snapshot().max_log_bytes - before - (64 << 10);
        let held = budget.reserve(usize::try_from(held_bytes)?, false)?;
        let before_files = files(root.path())?;
        let error = group.maintenance().run().unwrap_err();
        assert!(error.is::<crate::admission::DiskBudgetExhausted>());
        assert_eq!(files(root.path())?, before_files);
        assert_eq!(budget.snapshot().reserved_proposal_bytes, held_bytes);
        assert_eq!(group.initial_state()?.hard_state.commit, 64);
        verify(&sibling, 1, 4)?;
        drop(held);
    }

    // Exercise the public maintenance entry point without new writes, lower
    // logical retention or an enlarged disk budget.
    let fault = matches!(failure, Failure::RewriteSync)
        .then(|| group.node.fault.fail_rewrite_sync_for_test());
    let result = group.maintenance().run();
    let after = budget.snapshot().accounted_disk_bytes;
    if fault.is_some() {
        let error = result.unwrap_err();
        assert!(!error.is::<crate::admission::DiskBudgetExhausted>());
        assert!(group.initial_state().is_err());
        assert!(sibling.initial_state().is_err());
        assert!(group.maintenance().run().is_err());
    } else {
        result?;
        assert!(after < before, "physical reclamation made no progress");
        assert!(budget.reserve(4096, true).is_ok());
        assert_eq!(
            group.entries(13, 65, None, GetEntriesContext::empty(false))?,
            records[12..]
        );
        verify(&sibling, 1, 4)?;
    }
    assert_eq!(budget.snapshot().reserved_proposal_bytes, 0);
    eprintln!("maintenance admission: before={before}, after={after}, normal={normal}, pressure={pressure}");
    drop((fault, group, sibling, budget, log));

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
