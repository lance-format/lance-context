use super::*;

fn budget(data_mib: usize) -> Result<(tempfile::TempDir, Arc<Budget>)> {
    let directory = tempfile::tempdir()?;
    let budget = Budget::new(
        directory.path(),
        Arc::new(AtomicU64::new(0)),
        16 << 20,
        Arc::new(RwLock::new(())),
    )?;
    budget.limit_data(data_mib << 20)?;
    Ok((directory, budget))
}

#[test]
fn native_scratch_preserves_the_data_window_without_hiding_other_charges() -> Result<()> {
    let (directory, budget) = budget(4)?;
    let rewrite = budget.reserve_rewrite(6 << 20)?;
    let data = budget.reserve(4 << 20, true)?;
    assert_eq!(budget.snapshot().reserved_proposal_bytes, 10 << 20);
    assert_eq!(budget.snapshot().reserved_rewrite_bytes, 6 << 20);
    assert!(budget.reserve(1, true).is_err());
    budget.limit_data(2 << 20)?;
    assert!(budget.reserve(1, true).is_err());
    drop(data);
    let smaller = budget.reserve(2 << 20, true)?;
    assert!(budget.reserve(1, true).is_err());
    drop((smaller, rewrite));

    // Ordinary control proposals do not get the scratch exemption.
    let control = budget.reserve(3 << 20, false)?;
    assert!(budget.reserve(1, true).is_err());
    drop(control);
    // Nor do live files: rewriting them does not increase the data allowance.
    std::fs::write(directory.path().join("retained"), vec![7; 3 << 20])?;
    budget.refresh()?;
    let rewrite = budget.reserve_rewrite(1 << 20)?;
    assert!(budget.reserve(1, true).is_err());
    drop(rewrite);
    Ok(())
}

#[test]
fn scratch_never_lends_the_control_window_to_new_data() -> Result<()> {
    let (_directory, budget) = budget(8)?;
    let rewrite = budget.reserve_rewrite(10 << 20)?;
    // The immutable 12-MiB producer ceiling leaves four MiB for control.
    let data = budget.reserve(2 << 20, true)?;
    let error = budget.reserve(1, true).err().unwrap();
    let decision = error.downcast_ref::<DiskBudgetExhausted>().unwrap();
    assert_eq!(decision.limit_bytes, 12 << 20);
    assert_eq!(decision.reserved_rewrite_bytes, 10 << 20);
    let control = budget.reserve(4 << 20, false)?;
    assert!(budget.reserve(1, false).is_err());
    assert!(budget.reserve_rewrite(1).is_err());
    drop((control, data, rewrite));
    assert_eq!(budget.snapshot().reserved_proposal_bytes, 0);
    Ok(())
}

#[test]
fn racing_producers_share_scratch_headroom_and_leave_control_capacity() -> Result<()> {
    let (_directory, budget) = budget(8)?;
    let rewrite = budget.reserve_rewrite(8 << 20)?;
    let gate = std::sync::Barrier::new(32);
    let held = std::thread::scope(|scope| {
        let jobs = (0..32)
            .map(|_| {
                let (budget, gate) = (budget.clone(), &gate);
                scope.spawn(move || {
                    gate.wait();
                    budget.reserve(256 << 10, true).ok()
                })
            })
            .collect::<Vec<_>>();
        jobs.into_iter()
            .filter_map(|job| job.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(held.len(), 16);
    assert_eq!(budget.snapshot().reserved_proposal_bytes, 12 << 20);
    let control = budget.reserve(4 << 20, false)?;
    assert!(budget.reserve(1, true).is_err());
    drop((control, held, rewrite));
    assert_eq!(budget.snapshot().reserved_proposal_bytes, 0);
    assert!(budget.reserve(8 << 20, true).is_ok());
    Ok(())
}
