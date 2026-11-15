use super::*;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn layout() -> Layout {
    Layout {
        cluster: uuid::Uuid::from_u128(100),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        buckets: vec![10],
        groups: BTreeMap::from([(10, vec![1, 2, 3])]),
    }
}

fn hard(term: u64, commit: u64) -> HardState {
    HardState {
        term,
        commit,
        ..Default::default()
    }
}

fn entry(index: u64, term: u64, value: u8) -> Entry {
    Entry {
        index,
        term,
        data: vec![value; 512].into(),
        ..Default::default()
    }
}

fn view(log: GroupLog) -> Result<BufferedLog> {
    BufferedLog::new(
        log,
        Window {
            readies: 4,
            entries: 8,
            bytes: 8192,
        },
    )
}

fn read(store: &impl Storage, low: u64, high: u64) -> raft::Result<Vec<Entry>> {
    store.entries(low, high, None, GetEntriesContext::empty(false))
}

#[test]
fn queued_readies_share_one_sync_and_keep_original_receipts() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let physical = PhysicalLog::open(directory.path(), &layout(), 1)?;
    let log = physical.group(10)?;
    let mut buffered = view(log.clone())?;
    let mut tasks = Vec::new();
    for number in 1..=4 {
        tasks.push(buffered.stage(
            number,
            vec![entry(number, 1, number as u8)],
            Some(hard(1, number)),
        )?);
    }
    let before = log.node.fault.sync_calls_for_test();
    let receipts = WriteTask::persist_batch(tasks)?;
    let syncs = log.node.fault.sync_calls_for_test() - before;
    assert_eq!(receipts.len(), 4);
    assert_eq!(buffered.durable_hard_state()?.commit, 0);
    for (offset, receipt) in receipts.into_iter().enumerate() {
        let number = offset as u64 + 1;
        assert_eq!(buffered.complete(receipt)?, number);
        assert_eq!(buffered.durable_hard_state()?.commit, number);
        assert_eq!(buffered.usage().readies, 4 - number as usize);
    }
    assert_eq!(buffered.usage(), Usage::default());
    drop(buffered.into_durable()?);
    drop(log);
    drop(physical);
    let reopened = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    assert_eq!(reopened.initial_state()?.hard_state.commit, 4);
    assert_eq!(
        read(&reopened, 1, 5)?,
        (1..=4).map(|i| entry(i, 1, i as u8)).collect::<Vec<_>>()
    );
    assert_eq!(
        syncs, 1,
        "already queued adjacent Ready writes should share one native sync"
    );
    Ok(())
}

#[test]
fn older_completion_keeps_newer_overlapping_suffix_and_durable_boundary() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let physical = PhysicalLog::open(directory.path(), &layout(), 1)?;
    let log = physical.group(10)?;
    log.persist(&[entry(1, 1, 1), entry(2, 1, 2)], Some(&hard(1, 2)))?;
    let mut buffered = view(log.clone())?;
    let first = buffered.stage(10, vec![entry(3, 1, 3), entry(4, 1, 4)], Some(hard(1, 2)))?;
    let newer = entry(3, 2, 9);
    let pointer = newer.data.as_ptr();
    let second = buffered.stage(11, vec![newer.clone()], Some(hard(2, 3)))?;
    assert_eq!(buffered.initial_state()?.hard_state.commit, 3);
    assert_eq!(buffered.durable_hard_state()?.commit, 2);
    assert_eq!(buffered.complete(first.persist()?)?, 10);
    assert_eq!(buffered.term(3)?, 2);
    assert_eq!(buffered.last_index()?, 3);
    assert!(read(&buffered, 4, 5).is_err());
    let cached = read(&buffered, 3, 4)?;
    assert_eq!(cached, vec![newer.clone()]);
    assert_eq!(cached[0].data.as_ptr(), pointer);
    assert_eq!(buffered.durable_hard_state()?.commit, 2);
    assert_eq!(buffered.usage().readies, 1);
    buffered.complete(second.persist()?)?;
    assert_eq!(buffered.usage(), Usage::default());
    assert!(buffered.entries.is_empty());
    assert_eq!(buffered.durable_hard_state()?.commit, 3);
    assert_eq!(
        read(&buffered, 1, 4)?,
        vec![entry(1, 1, 1), entry(2, 1, 2), newer]
    );
    drop(buffered.into_durable()?);
    drop(log);
    drop(physical);
    let reopened = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    assert_eq!(reopened.initial_state()?.hard_state.commit, 3);
    assert_eq!(
        read(&reopened, 1, 4)?,
        vec![entry(1, 1, 1), entry(2, 1, 2), entry(3, 2, 9)]
    );
    Ok(())
}

#[test]
fn queued_suffix_replacement_keeps_each_native_write_boundary() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let physical = PhysicalLog::open(directory.path(), &layout(), 1)?;
    let log = physical.group(10)?;
    log.persist(&[entry(1, 1, 1)], Some(&hard(1, 1)))?;
    let mut buffered = view(log.clone())?;
    let first = buffered.stage(10, vec![entry(2, 1, 2), entry(3, 1, 3)], Some(hard(1, 1)))?;
    let newer = entry(2, 2, 9);
    let pointer = newer.data.as_ptr();
    let second = buffered.stage(11, vec![newer.clone()], Some(hard(2, 2)))?;
    let before = log.node.fault.sync_calls_for_test();
    let mut receipts = WriteTask::persist_batch(vec![first, second])?.into_iter();
    assert_eq!(log.node.fault.sync_calls_for_test() - before, 2);
    assert_eq!(buffered.complete(receipts.next().unwrap())?, 10);
    let visible = read(&buffered, 2, 3)?;
    assert_eq!(visible, vec![newer.clone()]);
    assert_eq!(visible[0].data.as_ptr(), pointer);
    assert_eq!(buffered.durable_hard_state()?.commit, 1);
    assert_eq!(buffered.complete(receipts.next().unwrap())?, 11);
    assert!(receipts.next().is_none());
    assert_eq!(buffered.usage(), Usage::default());
    drop(buffered.into_durable()?);
    drop(log);
    drop(physical);
    let reopened = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    assert_eq!(reopened.initial_state()?.hard_state, hard(2, 2));
    assert_eq!(read(&reopened, 1, 3)?, vec![entry(1, 1, 1), newer]);
    Ok(())
}

#[test]
fn queued_readies_release_no_receipt_before_the_shared_sync() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    let mut buffered = view(log.clone())?;
    let mut tasks = Vec::new();
    for number in 1..=4 {
        tasks.push(buffered.stage(
            number,
            vec![entry(number, 1, number as u8)],
            Some(hard(1, number)),
        )?);
    }
    let gate = log.block_sync_for_test();
    let writer = thread::spawn(move || WriteTask::persist_batch(tasks));
    let deadline = Instant::now() + Duration::from_secs(2);
    while log.appended_index_for_test() != Some(4) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let appended = log.appended_index_for_test();
    let escaped = writer.is_finished();
    let durable_before = buffered.durable_hard_state()?.commit;
    // Always release and join the original native operation before assertions.
    drop(gate);
    let receipts = writer.join().unwrap()?;
    assert_eq!(appended, Some(4));
    assert!(!escaped, "batch receipts escaped the original sync");
    assert_eq!(durable_before, 0);
    for receipt in receipts {
        buffered.complete(receipt)?;
    }
    assert_eq!(buffered.durable_hard_state()?.commit, 4);
    assert_eq!(buffered.usage(), Usage::default());
    Ok(())
}

#[test]
fn queued_readies_release_no_receipt_after_sync_failure() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    let mut buffered = view(log.clone())?;
    let mut tasks = Vec::new();
    for number in 1..=4 {
        tasks.push(buffered.stage(
            number,
            vec![entry(number, 1, number as u8)],
            Some(hard(1, number)),
        )?);
    }
    log.fail_next_sync_for_test();
    assert!(WriteTask::persist_batch(tasks).is_err());
    assert_eq!(buffered.usage().readies, 4);
    assert!(buffered.durable_hard_state().is_err());
    assert!(log.initial_state().is_err());
    Ok(())
}

#[test]
fn pending_window_refuses_without_mutation_and_reuses_only_completed_capacity() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    for window in [
        Window {
            readies: 1,
            entries: 8,
            bytes: 8192,
        },
        Window {
            readies: 8,
            entries: 1,
            bytes: 8192,
        },
        Window {
            readies: 8,
            entries: 8,
            bytes: 4096,
        },
    ] {
        let mut buffered = BufferedLog::new(log.clone(), window)?;
        let index = buffered.last_index()? + 1;
        let first = buffered.stage(1, vec![entry(index, 1, 1)], Some(hard(1, index)))?;
        let before = buffered.usage();
        let oversized = if window.bytes == 4096 {
            Entry {
                data: vec![2; 4096].into(),
                ..entry(index + 1, 1, 2)
            }
        } else {
            entry(index + 1, 1, 2)
        };
        assert!(!buffered.can_stage(std::slice::from_ref(&oversized))?);
        assert!(buffered
            .stage(2, vec![oversized], Some(hard(1, index + 1)))
            .err()
            .unwrap()
            .to_string()
            .contains("window full"));
        assert_eq!(buffered.usage(), before);
        assert_eq!(buffered.last_index()?, index);
        assert_eq!(buffered.last, 1);
        buffered.complete(first.persist()?)?;
        assert_eq!(buffered.usage(), Usage::default());
        let second = buffered.stage(2, vec![entry(index + 1, 1, 2)], Some(hard(1, index + 1)))?;
        buffered.complete(second.persist()?)?;
    }
    Ok(())
}

#[test]
fn blocked_native_sync_does_not_block_pending_or_stable_prefix_reads() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    log.persist(&[entry(1, 1, 1)], Some(&hard(1, 1)))?;
    let mut buffered = view(log.clone())?;
    let pending = entry(2, 1, 2);
    let pointer = pending.data.as_ptr() as usize;
    let task = buffered.stage(1, vec![pending], Some(hard(1, 2)))?;
    let guard = log.block_sync_for_test();
    let writer = thread::spawn(move || task.persist());
    let deadline = Instant::now() + Duration::from_secs(2);
    while log.appended_index_for_test() != Some(2) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let appended = log.appended_index_for_test();
    let (tx, rx) = mpsc::channel();
    let reader = thread::spawn(move || {
        let result = (|| {
            let entries = read(&buffered, 1, 3)?;
            ensure!(entries == vec![entry(1, 1, 1), entry(2, 1, 2)]);
            ensure!(entries[1].data.as_ptr() as usize == pointer);
            ensure!(buffered.term(2)? == 1);
            ensure!(buffered.initial_state()?.hard_state.commit == 2);
            ensure!(buffered.durable_hard_state()?.commit == 1);
            Ok::<_, anyhow::Error>(())
        })();
        let _ = tx.send((buffered, result));
    });
    let observed = rx.recv_timeout(Duration::from_millis(500));
    // Release/join originals on every path, including an unexpected blocked read.
    drop(guard);
    let receipt = writer.join().unwrap()?;
    reader.join().unwrap();
    assert_eq!(appended, Some(2));
    let (mut buffered, result) = observed.context("Storage read blocked behind native sync")?;
    result?;
    buffered.complete(receipt)?;
    assert_eq!(buffered.durable_hard_state()?.commit, 2);
    assert_eq!(buffered.usage(), Usage::default());
    Ok(())
}

#[test]
fn drained_checkpoint_reopen_preserves_storage_bounds_and_size_limits() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    log.bind_application(b"buffered-test")?;
    let mut buffered = view(log)?;
    let first = buffered.stage(
        1,
        vec![entry(1, 1, 1), entry(2, 1, 2), entry(3, 1, 3)],
        Some(hard(1, 3)),
    )?;
    assert_eq!(
        buffered.entries(1, 4, Some(0), GetEntriesContext::empty(false))?,
        vec![entry(1, 1, 1)]
    );
    assert_eq!(
        buffered
            .entries(
                1,
                4,
                Some(u64::from(entry(1, 1, 1).compute_size())),
                GetEntriesContext::empty(false)
            )?
            .len(),
        1
    );
    buffered.complete(first.persist()?)?;
    let log = buffered.into_durable()?;
    let mut snapshot = Snapshot {
        data: b"snapshot-descriptor".to_vec().into(),
        ..Default::default()
    };
    snapshot.mut_metadata().index = 2;
    snapshot.mut_metadata().term = 1;
    snapshot
        .mut_metadata()
        .set_conf_state(log.initial_state()?.conf_state);
    log.checkpoint(&snapshot)?;
    let buffered = view(log)?;
    assert_eq!(buffered.first_index()?, 3);
    assert_eq!(buffered.term(2)?, 1);
    assert!(read(&buffered, 2, 3).is_err());
    assert_eq!(read(&buffered, 3, 4)?, vec![entry(3, 1, 3)]);
    assert_eq!(buffered.snapshot(2, 2)?, snapshot);
    Ok(())
}

#[test]
fn out_of_order_io_fences_the_view_without_mutating_the_log() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    let mut buffered = view(log.clone())?;
    let first = buffered.stage(1, vec![entry(1, 1, 1)], Some(hard(1, 1)))?;
    let second = buffered.stage(2, vec![entry(2, 1, 2)], Some(hard(1, 2)))?;
    assert!(second
        .persist()
        .err()
        .unwrap()
        .to_string()
        .contains("out-of-order"));
    assert!(first.persist().is_err());
    assert!(buffered.initial_state().is_err());
    assert!(read(&buffered, 1, 2).is_err());
    assert_eq!(log.last_index()?, 0);
    assert_eq!(log.initial_state()?.hard_state.commit, 0);
    Ok(())
}

#[test]
fn out_of_order_completion_never_advances_the_durable_view() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    let mut buffered = view(log.clone())?;
    let first = buffered.stage(1, vec![entry(1, 1, 1)], Some(hard(1, 1)))?;
    let second = buffered.stage(2, vec![entry(2, 1, 2)], Some(hard(1, 2)))?;
    let first = first.persist()?;
    let second = second.persist()?;
    let before = buffered.usage();
    assert!(buffered
        .complete(second)
        .err()
        .unwrap()
        .to_string()
        .contains("out-of-order"));
    assert_eq!(buffered.durable_hard_state()?.commit, 0);
    assert_eq!(buffered.usage(), before);
    assert_eq!(buffered.complete(first)?, 1);
    assert_eq!(buffered.durable_hard_state()?.commit, 1);
    assert_eq!(log.initial_state()?.hard_state.commit, 2);
    // The rejected receipt was not an acknowledgement. The caller must stop;
    // original persisted state remains recoverable after dropping all handles.
    drop(buffered);
    drop(log);
    let reopened = PhysicalLog::open(directory.path(), &layout(), 1)?.group(10)?;
    assert_eq!(reopened.initial_state()?.hard_state.commit, 2);
    assert_eq!(read(&reopened, 1, 3)?, vec![entry(1, 1, 1), entry(2, 1, 2)]);
    Ok(())
}
