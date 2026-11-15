use super::*;

#[test]
fn checkpointed_replication_tail_does_not_delay_pressure_reclamation() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    let sibling = log.group(20)?;
    group.bind_application(b"retained-tail-test/v1")?;
    sibling.bind_application(b"retained-tail-test/v1")?;
    let retained: Vec<_> = (1..=64)
        .map(|index| Entry {
            data: vec![index as u8; 64 << 10].into(),
            ..entry(10, index)
        })
        .collect();
    let persist_one = |group: &GroupLog, record: &Entry| {
        group.persist(
            std::slice::from_ref(record),
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: record.index,
                ..Default::default()
            }),
        )
    };
    // Separate native writes rotate files. The small replication tail is
    // checkpointed but must remain readable, so another checkpoint cannot
    // solve its physical retention of the append queue's oldest files.
    for record in &retained {
        persist_one(&group, record)?;
    }
    let mut tail_cut = Snapshot::default();
    tail_cut.mut_metadata().index = 64;
    tail_cut.mut_metadata().term = 1;
    tail_cut
        .mut_metadata()
        .set_conf_state(group.initial_state()?.conf_state);
    tail_cut.data = b"retained-tail-test/64".to_vec().into();
    group.checkpoint_retaining(&tail_cut, 8 << 20)?;
    assert_eq!(group.first_index()?, 1);
    for index in 1..=256 {
        persist_one(
            &sibling,
            &Entry {
                data: vec![index as u8; 64 << 10].into(),
                ..entry(20, index)
            },
        )?;
    }
    let sibling_cut = checkpoint(&sibling, 256)?;
    group.configure_disk(DiskOptions {
        max_bytes: 32 << 20,
        data_bytes: 6 << 20,
        headroom_bytes: 4 << 20,
    })?;
    let budget = group.disk_budget().unwrap().clone();
    assert!(budget.reserve(4096, true).is_err());
    let maintenance = group.maintenance();
    let mut admission_after_first_pass = false;
    // Preserve all observations and verify cold recovery before asserting the
    // prompt-reclamation requirement, including on the baseline failure path.
    let mut first_admitted_pass = None;
    for pass in 1..=12 {
        let hints = maintenance.run()?;
        let admitted = budget.reserve(4096, true).is_ok();
        eprintln!(
            "retained tail: pass={pass}, hints={hints:?}, bytes={}, admitted={admitted}",
            budget.snapshot().accounted_disk_bytes
        );
        if pass == 1 {
            admission_after_first_pass = admitted;
        }
        if admitted {
            first_admitted_pass = Some(pass);
            break;
        }
    }
    assert_eq!(
        group.entries(1, 65, None, GetEntriesContext::empty(false))?,
        retained
    );
    drop((maintenance, group, sibling, budget, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    let sibling = reopened.group(20)?;
    assert_eq!(group.snapshot(64, 2)?, tail_cut);
    assert_eq!(group.initial_state()?.hard_state.commit, 64);
    assert_eq!(group.first_index()?, 1);
    assert_eq!(
        group.entries(1, 65, None, GetEntriesContext::empty(false))?,
        retained
    );
    assert_eq!(sibling.snapshot(256, 2)?, sibling_cut);
    assert_eq!(sibling.first_index()?, 257);
    assert_eq!(sibling.last_index()?, 256);
    eprintln!("retained tail: all values and both durable checkpoints verified after reopen");
    assert!(
        admission_after_first_pass,
        "already-checkpointed tail kept admission closed; first admitted pass={first_admitted_pass:?}"
    );
    Ok(())
}
