use super::*;

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
