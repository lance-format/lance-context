use super::*;
use std::{sync::mpsc, time::Duration};

#[test]
fn slow_background_rewrite_does_not_stop_foreground_append_or_retained_read() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    let records: Vec<_> = (1..=17)
        .map(|index| Entry {
            data: vec![index as u8; 256 << 10].into(),
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
    let maintenance = group.maintenance();
    let pause = group.node.fault.pause_rewrite_sync_for_test();
    let maintain = std::thread::spawn(move || maintenance.run());
    // Pause a real rewrite-file sync, without failing it or manually acquiring
    // the maintenance gate. The production storage path chooses its locks.
    let entered = pause.wait();
    if let Err(error) = entered {
        drop(pause);
        maintain.join().unwrap()?;
        return Err(error);
    }
    let writer = log.group(10)?;
    let reader = log.group(20)?;
    let last = records[16].clone();
    let (written, wait_written) = mpsc::sync_channel(1);
    let (read, wait_read) = mpsc::sync_channel(1);
    let write = std::thread::spawn(move || {
        let result = writer.persist(
            &[last],
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: 17,
                ..Default::default()
            }),
        );
        written.send(result).unwrap();
    });
    let fetch = std::thread::spawn(move || {
        read.send(reader.entries(1, 5, None, GetEntriesContext::empty(false)))
            .unwrap();
    });
    let append_before_release = wait_written.recv_timeout(Duration::from_secs(2));
    let read_before_release = wait_read.recv_timeout(Duration::from_secs(2));
    drop(pause);
    maintain.join().unwrap()?;
    write.join().unwrap();
    fetch.join().unwrap();
    let append_progressed = append_before_release.is_ok();
    let read_progressed = read_before_release.is_ok();
    append_before_release.or_else(|_| wait_written.recv_timeout(Duration::from_secs(2)))??;
    assert_eq!(
        read_before_release.or_else(|_| wait_read.recv_timeout(Duration::from_secs(2)))??,
        (1..=4).map(|index| entry(20, index)).collect::<Vec<_>>()
    );
    drop((group, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    assert_eq!(group.initial_state()?.hard_state.commit, 17);
    assert_eq!(
        group.entries(1, 18, None, GetEntriesContext::empty(false))?,
        records
    );
    verify(&reopened.group(20)?, 1, 4)?;
    eprintln!(
        "rewrite pause: append_before_release={append_progressed}, retained_read_before_release={read_progressed}; all 21 records reopened"
    );
    assert!(
        append_progressed && read_progressed,
        "background rewrite sync excluded foreground append and/or retained read"
    );
    Ok(())
}

#[test]
fn rewrite_allowance_excludes_later_snapshot_entries_and_tombstones() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    group.bind_application(b"late-snapshot-test/v1")?;
    // Rewrite eligibility counts entries, including live metadata and sibling
    // values. Expire enough individual records to cross that threshold; a few
    // large expired records alone need not trigger the native rewrite path.
    for index in 1..=64 {
        group.persist(
            &[Entry {
                data: vec![index as u8; 64 << 10].into(),
                ..entry(10, index)
            }],
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: index,
                ..Default::default()
            }),
        )?;
    }
    append(&log.group(20)?, 1, 4)?;
    for _ in 0..4 {
        group.maintenance().run()?;
    }
    checkpoint(&group, 60)?;
    group.configure_disk(DiskOptions {
        max_bytes: 32 << 20,
        data_bytes: 8 << 20,
        headroom_bytes: 16 << 20,
    })?;
    let budget = group.disk_budget().unwrap().clone();
    let pause = group.node.fault.pause_rewrite_sync_for_test();
    let maintenance = group.maintenance();
    let maintain = std::thread::spawn(move || maintenance.run());
    if let Err(error) = pause.wait() {
        drop(pause);
        maintain.join().unwrap()?;
        return Err(error.context("fixture did not reach rewrite sync"));
    }
    let before = budget.snapshot();
    let reserved_before = before.reserved_proposal_bytes;
    let mut snapshot = Snapshot::default();
    snapshot.mut_metadata().index = 65;
    snapshot.mut_metadata().term = 2;
    snapshot
        .mut_metadata()
        .set_conf_state(group.initial_state()?.conf_state);
    snapshot.data = b"late-snapshot/65".to_vec().into();
    let suffix: Vec<_> = (66..=69)
        .map(|index| Entry {
            term: 2,
            data: vec![index as u8; 1 << 20].into(),
            ..entry(10, index)
        })
        .collect();
    let writer = log.group(10)?;
    let input = snapshot.clone();
    let entries = suffix.clone();
    let (sent, received) = mpsc::sync_channel(1);
    let install = std::thread::spawn(move || {
        sent.send(writer.install_snapshot(
            &input,
            &entries,
            &HardState {
                term: 2,
                vote: 1,
                commit: 69,
                ..Default::default()
            },
        ))
        .unwrap();
    });
    let observed = received.recv_timeout(Duration::from_secs(2));
    let installed_before_release = observed.as_ref().is_ok_and(|r| r.is_ok());
    let pending_before_release = installed_before_release.then(|| {
        group
            .node
            .engine
            .purge_manager()
            .capture_append_boundary()
            .1
    });
    let after = budget.snapshot();
    let reserved_after = after.reserved_proposal_bytes;
    drop(pause);
    maintain.join().unwrap()?;
    install.join().unwrap();
    observed.or_else(|_| received.recv_timeout(Duration::from_secs(2)))??;
    assert!(installed_before_release);
    assert!(reserved_before > 0 && reserved_before < 2 << 20);
    assert_eq!(reserved_after, reserved_before);
    assert_eq!(before.reserved_rewrite_bytes, reserved_before);
    assert_eq!(after.reserved_rewrite_bytes, reserved_before);
    assert_eq!(pending_before_release, Some(1));
    assert_eq!(
        group
            .node
            .engine
            .purge_manager()
            .capture_append_boundary()
            .1,
        1,
        "new snapshot tombstone consumed by an older rewrite allowance"
    );
    let rewrite_bytes: u64 = files(root.path())?
        .into_iter()
        .filter(|(p, _)| p.extension().is_some_and(|e| e == "rewrite"))
        .map(|(_, bytes)| bytes)
        .sum();
    assert!(
        rewrite_bytes < 1 << 20,
        "late 4 MiB suffix entered the old rewrite allowance: {rewrite_bytes}"
    );
    assert_eq!(budget.snapshot().reserved_proposal_bytes, 0);
    assert_eq!(budget.snapshot().reserved_rewrite_bytes, 0);
    group.maintenance().run()?;
    drop((group, budget, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    assert_eq!(group.snapshot(65, 2)?, snapshot);
    assert_eq!(group.initial_state()?.hard_state.commit, 69);
    assert_eq!(group.first_index()?, 66);
    assert_eq!(
        group.entries(66, 70, None, GetEntriesContext::empty(false))?,
        suffix
    );
    verify(&reopened.group(20)?, 1, 4)?;
    Ok(())
}

#[test]
fn removed_memtable_files_survive_an_inflight_native_payload_read() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let log = PhysicalLog::open_with_file_size(root.path(), &layout, 1, 64 << 10)?;
    let group = log.group(10)?;
    group.bind_application(b"read-lifetime-test/v1")?;
    let records: Vec<_> = (1..=4)
        .map(|index| Entry {
            data: vec![index as u8; 128 << 10].into(),
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
    append(&log.group(20)?, 1, 4)?;
    let before = files(root.path())?;
    let pause = group.node.fault.pause_payload_read_for_test();
    let reader = group.node.clone();
    // Exercise native index selection followed by payload I/O directly. The
    // GroupLog state lock would otherwise serialize snapshot installation and
    // hide the removed-memtable lifetime that the native guard must protect.
    let read = std::thread::spawn(move || {
        let mut records = Vec::new();
        reader
            .engine
            .fetch_entries_to::<EntryIndex>(10, 1, 5, None, &mut records)
            .map(|_| records)
    });
    pause.wait()?;
    let mut snapshot = Snapshot::default();
    snapshot.mut_metadata().index = 5;
    snapshot.mut_metadata().term = 2;
    snapshot
        .mut_metadata()
        .set_conf_state(group.initial_state()?.conf_state);
    snapshot.data = b"read-lifetime/5".to_vec().into();
    let suffix = Entry {
        term: 2,
        ..entry(10, 6)
    };
    group.install_snapshot(
        &snapshot,
        std::slice::from_ref(&suffix),
        &HardState {
            term: 2,
            vote: 1,
            commit: 6,
            ..Default::default()
        },
    )?;
    let maintenance = group.maintenance();
    let (started, wait_started) = mpsc::sync_channel(1);
    let (done, wait_done) = mpsc::sync_channel(1);
    let maintain = std::thread::spawn(move || {
        started.send(()).unwrap();
        done.send(maintenance.run()).unwrap();
    });
    wait_started.recv_timeout(Duration::from_secs(2))?;
    let completed = wait_done.recv_timeout(Duration::from_millis(200));
    let retired_before_read = before.keys().any(|path| !path.exists());
    let completed_before_read = completed.is_ok();
    drop(pause);
    let observed = read.join().unwrap();
    maintain.join().unwrap();
    completed.or_else(|_| wait_done.recv_timeout(Duration::from_secs(2)))??;
    assert_eq!(observed?, records);
    assert!(!retired_before_read && !completed_before_read);
    assert!(
        before.keys().any(|path| !path.exists()),
        "fixture must retire old files after the reader finishes"
    );
    drop((group, log));
    let reopened = PhysicalLog::open(root.path(), &layout, 1)?;
    let group = reopened.group(10)?;
    assert_eq!(group.snapshot(5, 2)?, snapshot);
    assert_eq!(group.initial_state()?.hard_state.commit, 6);
    assert_eq!(
        group.entries(6, 7, None, GetEntriesContext::empty(false))?,
        vec![suffix]
    );
    verify(&reopened.group(20)?, 1, 4)?;
    Ok(())
}
