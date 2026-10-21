#![cfg(feature = "raft-log")]

use std::collections::BTreeMap;

use anyhow::Result;
use lance_context_wal::{
    placement::Layout,
    raft_log::{GroupLog, PhysicalLog},
};
use protobuf::Message;
use raft::{GetEntriesContext, Storage, StorageError};
use raft_proto::eraftpb::{Entry, HardState, Snapshot};
use uuid::Uuid;

// A concurrent fork briefly inherits other tests' flock descriptors until exec.
// Exclude that window between independent reopen cases, while preserving the
// real subprocess/crash behavior exercised within each case.
static TEST_CASE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn layout() -> Layout {
    Layout {
        cluster: Uuid::from_u128(22),
        nodes: (1..=3)
            .map(|n| (n, format!("wal-{n}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
        buckets: vec![10, 20],
    }
}

fn entry(index: u64, term: u64) -> Entry {
    Entry {
        index,
        term,
        data: vec![index as u8; 8192].into(),
        ..Default::default()
    }
}

fn hard(term: u64, commit: u64) -> HardState {
    HardState {
        term,
        commit,
        vote: 1,
        ..Default::default()
    }
}

fn snapshot(log: &GroupLog, index: u64, term: u64) -> Result<Snapshot> {
    let mut snapshot = Snapshot::default();
    snapshot.mut_metadata().index = index;
    snapshot.mut_metadata().term = term;
    snapshot
        .mut_metadata()
        .set_conf_state(log.initial_state()?.conf_state);
    // Self-contained toy application state; production descriptors additionally
    // need validated immutable files and their own scope/recovery contracts.
    snapshot.data = vec![index as u8; 32].into();
    Ok(snapshot)
}

fn read(log: &GroupLog, low: u64, high: u64) -> raft::Result<Vec<Entry>> {
    log.entries(low, high, None, GetEntriesContext::empty(false))
}

#[test]
fn checkpoint_preserves_suffix_vote_and_other_group_across_reopen() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("log");
    let original: Vec<_> = (1..=6)
        .map(|i| entry(i, if i < 3 { 1 } else { 2 }))
        .collect();
    let saved;
    {
        let log = PhysicalLog::open(&path, &layout(), 1)?;
        let group = log.group(10)?;
        group.bind_application(b"checkpoint-test/v1")?;
        group.persist(&original, Some(&hard(2, 5)))?;
        log.group(20)?.persist(&[entry(1, 7)], Some(&hard(7, 1)))?;
        saved = snapshot(&group, 3, 2)?;
        group.checkpoint(&saved)?;
        group.checkpoint(&saved)?;
        assert_eq!(group.first_index()?, 4);
        assert_eq!(group.last_index()?, 6);
        assert_eq!(group.term(3)?, 2);
        assert!(matches!(
            group.term(2),
            Err(raft::Error::Store(StorageError::Compacted))
        ));
        assert!(matches!(
            read(&group, 3, 4),
            Err(raft::Error::Store(StorageError::Compacted))
        ));
        assert_eq!(read(&group, 4, 7)?, original[3..]);
        assert_eq!(group.snapshot(3, 2)?, saved);
        assert!(matches!(
            group.snapshot(4, 2),
            Err(raft::Error::Store(
                StorageError::SnapshotTemporarilyUnavailable
            ))
        ));
        assert_eq!(group.initial_state()?.hard_state, hard(2, 5));
    }
    {
        let log = PhysicalLog::open(&path, &layout(), 1)?;
        let group = log.group(10)?;
        group.bind_application(b"checkpoint-test/v1")?;
        assert_eq!(group.snapshot(0, 3)?, saved);
        assert_eq!(read(&group, 4, 7)?, original[3..]);
        assert_eq!(read(&log.group(20)?, 1, 2)?, vec![entry(1, 7)]);
        assert_eq!(log.group(20)?.first_index()?, 1);
        // A retained committed entry remains immutable; an uncommitted suffix
        // still permits replacement in a later term.
        assert!(group.persist(&[entry(5, 3)], Some(&hard(3, 5))).is_err());
        group.persist(&[entry(6, 3), entry(7, 3)], Some(&hard(3, 7)))?;
        assert_eq!(
            read(&group, 4, 8)?,
            vec![entry(4, 2), entry(5, 2), entry(6, 3), entry(7, 3)]
        );
        let next = snapshot(&group, 7, 3)?;
        group.checkpoint(&next)?;
        assert_eq!((group.first_index()?, group.last_index()?), (8, 7));
        assert!(read(&group, 8, 8)?.is_empty());
    }
    let log = PhysicalLog::open(&path, &layout(), 1)?;
    let group = log.group(10)?;
    assert_eq!(
        (group.first_index()?, group.last_index()?, group.term(7)?),
        (8, 7, 3)
    );
    assert_eq!(group.initial_state()?.hard_state, hard(3, 7));
    group.persist(&[entry(8, 3)], Some(&hard(3, 8)))?;
    assert_eq!(read(&group, 8, 9)?, vec![entry(8, 3)]);
    let config = raft::Config {
        id: 1,
        applied: 7,
        ..Default::default()
    };
    let mut raw = raft::RawNode::new(
        &config,
        group,
        &slog::Logger::root(slog::Discard, slog::o!()),
    )?;
    let mut ready = raw.ready();
    assert_eq!(ready.take_committed_entries(), vec![entry(8, 3)]);
    Ok(())
}

#[test]
fn invalid_or_changed_checkpoint_never_changes_usable_log() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(&directory.path().join("log"), &layout(), 1)?;
    let group = log.group(10)?;
    group.persist(&[entry(1, 1), entry(2, 2), entry(3, 2)], Some(&hard(2, 2)))?;
    let good = snapshot(&group, 2, 2)?;
    assert!(
        group.checkpoint(&good).is_err(),
        "application must be bound before its first entry"
    );
    // Use a separately bound group to exercise every invalid candidate without
    // mutating or adopting the unbound group's existing log.
    let group = log.group(20)?;
    group.bind_application(b"checkpoint-test/v1")?;
    group.persist(&[entry(1, 1), entry(2, 2), entry(3, 2)], Some(&hard(2, 2)))?;
    let good = snapshot(&group, 2, 2)?;
    let mut cases = vec![
        snapshot(&group, 0, 0)?,
        snapshot(&group, 3, 2)?,
        snapshot(&group, 2, 1)?,
        snapshot(&group, u64::MAX, 2)?,
    ];
    let mut wrong = good.clone();
    wrong.mut_metadata().mut_conf_state().voters = vec![1, 2, 4];
    cases.push(wrong);
    let mut wrong = good.clone();
    wrong.data = vec![0; 4096].into();
    cases.push(wrong);
    let mut wrong = good.clone();
    wrong.data.clear();
    cases.push(wrong);
    for invalid in cases {
        assert!(group.checkpoint(&invalid).is_err());
        assert_eq!(group.first_index()?, 1);
        assert_eq!(
            read(&group, 1, 4)?,
            vec![entry(1, 1), entry(2, 2), entry(3, 2)]
        );
    }
    group.checkpoint(&good)?;
    let mut changed = good.clone();
    changed.data = vec![99].into();
    assert!(group.checkpoint(&changed).is_err());
    assert!(group.checkpoint(&snapshot(&group, 1, 1)?).is_err());
    assert_eq!(group.snapshot(0, 2)?, good);
    assert_eq!(read(&group, 3, 4)?, vec![entry(3, 2)]);
    Ok(())
}

#[test]
fn missing_corrupt_or_inconsistent_checkpoint_fails_reopen() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    for corruption in 0..5 {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("log");
        let saved;
        {
            let log = PhysicalLog::open(&path, &layout(), 1)?;
            let group = log.group(10)?;
            group.bind_application(b"checkpoint-test/v1")?;
            group.persist(&[entry(1, 1), entry(2, 2), entry(3, 2)], Some(&hard(2, 3)))?;
            saved = snapshot(&group, 2, 2)?;
            group.checkpoint(&saved)?;
        }
        {
            let engine = raft_engine::Engine::open(raft_engine::Config {
                dir: path.to_str().unwrap().into(),
                ..Default::default()
            })?;
            let mut batch = raft_engine::LogBatch::default();
            match corruption {
                0 => batch.delete(10, b"snapshot".to_vec()),
                1 => batch.put(10, b"snapshot".to_vec(), vec![255])?,
                2 => {
                    let mut invalid = saved;
                    invalid.mut_metadata().index = 3;
                    batch.put(10, b"snapshot".to_vec(), invalid.write_to_bytes()?)?;
                }
                3 => batch.put(10, b"snapshot".to_vec(), vec![0; 4097])?,
                4 => batch.delete(10, b"application".to_vec()),
                _ => unreachable!(),
            }
            engine.write(&mut batch, true)?;
        }
        let error = match PhysicalLog::open(&path, &layout(), 1) {
            Ok(_) => anyhow::bail!("accepted corruption {corruption}"),
            Err(error) => error,
        };
        // A transient lock failure is not evidence that corrupt state was
        // rejected. Require the actual validation failure for every case.
        match corruption {
            0 | 2 => assert!(
                error.to_string().contains("invalid durable term index"),
                "{error:#}"
            ),
            1 => assert!(
                error.downcast_ref::<protobuf::ProtobufError>().is_some(),
                "{error:#}"
            ),
            3 => assert!(
                error.to_string().contains("oversized durable snapshot"),
                "{error:#}"
            ),
            4 => assert!(
                error.to_string().contains("unbound checkpoint"),
                "{error:#}"
            ),
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[cfg(unix)]
mod process {
    use super::*;
    use std::{
        process::{Child, Command, Stdio},
        time::{Duration, Instant},
    };

    struct Worker(Child);
    impl Drop for Worker {
        fn drop(&mut self) {
            if !matches!(self.0.try_wait(), Ok(Some(_))) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
    }

    #[test]
    #[ignore = "subprocess fixture invoked by checkpoint_survives_sigkill"]
    fn checkpoint_worker() -> Result<()> {
        let directory = std::path::PathBuf::from(std::env::var("CONTEXT_CHECKPOINT_WORKER_ROOT")?);
        let after = std::env::var("CONTEXT_CHECKPOINT_WORKER_PHASE")? == "after";
        let log = PhysicalLog::open(&directory.join("log"), &layout(), 1)?;
        let group = log.group(10)?;
        group.bind_application(b"checkpoint-test/v1")?;
        group.persist(&[entry(1, 1), entry(2, 2), entry(3, 2)], Some(&hard(2, 3)))?;
        log.group(20)?.persist(&[entry(1, 5)], Some(&hard(5, 1)))?;
        if after {
            group.checkpoint(&snapshot(&group, 2, 2)?)?;
        }
        std::fs::write(directory.join("ready"), b"ready")?;
        std::thread::sleep(Duration::from_secs(30));
        anyhow::bail!("parent did not stop checkpoint worker")
    }

    #[test]
    fn checkpoint_survives_sigkill() -> Result<()> {
        let _case = TEST_CASE.lock().unwrap();
        use std::os::unix::process::ExitStatusExt;
        for phase in ["before", "after"] {
            let directory = tempfile::tempdir()?;
            let stderr = directory.path().join("stderr");
            let mut child = Worker(
                Command::new(std::env::current_exe()?)
                    .args([
                        "--ignored",
                        "--exact",
                        "process::checkpoint_worker",
                        "--nocapture",
                    ])
                    .env("CONTEXT_CHECKPOINT_WORKER_ROOT", directory.path())
                    .env("CONTEXT_CHECKPOINT_WORKER_PHASE", phase)
                    .stdout(Stdio::null())
                    .stderr(std::fs::File::create(&stderr)?)
                    .spawn()?,
            );
            let deadline = Instant::now() + Duration::from_secs(20);
            while !directory.path().join("ready").exists() {
                anyhow::ensure!(
                    child.0.try_wait()?.is_none() && Instant::now() < deadline,
                    "checkpoint worker did not reach {phase}: {}",
                    std::fs::read_to_string(&stderr)?
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            child.0.kill()?;
            assert_eq!(child.0.wait()?.signal(), Some(9));
            let log = PhysicalLog::open(&directory.path().join("log"), &layout(), 1)?;
            let group = log.group(10)?;
            assert_eq!(group.initial_state()?.hard_state, hard(2, 3));
            assert_eq!(read(&log.group(20)?, 1, 2)?, vec![entry(1, 5)]);
            if phase == "after" {
                assert_eq!(group.first_index()?, 3);
                assert_eq!(group.snapshot(2, 2)?, snapshot(&group, 2, 2)?);
                assert_eq!(read(&group, 3, 4)?, vec![entry(3, 2)]);
            } else {
                assert_eq!(group.first_index()?, 1);
                assert_eq!(
                    read(&group, 1, 4)?,
                    vec![entry(1, 1), entry(2, 2), entry(3, 2)]
                );
            }
        }
        Ok(())
    }
}
