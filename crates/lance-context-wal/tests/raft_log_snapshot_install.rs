#![cfg(feature = "raft-log")]

use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, Result};
use lance_context_wal::{
    placement::Layout,
    raft_log::{GroupLog, PhysicalLog},
};
use raft::{GetEntriesContext, Storage, StorageError};
use raft_proto::eraftpb::{Entry, HardState, Message, MessageType, Snapshot};
use uuid::Uuid;

// A concurrently spawning child briefly inherits other tests' flock descriptors
// before exec closes them. Keep these independent reopen cases out of that
// window; this does not serialize any concurrency exercised inside a case.
static TEST_CASE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn layout() -> Layout {
    Layout {
        cluster: Uuid::from_u128(23),
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
        data: vec![(index * 17 + term) as u8; 8192].into(),
        ..Default::default()
    }
}
fn hard(term: u64, commit: u64, vote: u64) -> HardState {
    HardState {
        term,
        commit,
        vote,
        ..Default::default()
    }
}
fn read(log: &GroupLog, low: u64, high: u64) -> raft::Result<Vec<Entry>> {
    log.entries(low, high, None, GetEntriesContext::empty(false))
}
fn original() -> Vec<Entry> {
    (1..=7)
        .map(|i| entry(i, if i == 1 { 1 } else { 2 }))
        .collect()
}
fn seed(path: &Path) -> Result<PhysicalLog> {
    let log = PhysicalLog::open(path, &layout(), 2)?;
    let group = log.group(10)?;
    group.bind_application(b"snapshot-install-test/v1")?;
    group.persist(&original(), Some(&hard(2, 1, 2)))?;
    log.group(20)?
        .persist(&[entry(1, 7)], Some(&hard(7, 1, 2)))?;
    Ok(log)
}
fn snapshot(log: &GroupLog, index: u64, term: u64) -> Result<Snapshot> {
    let mut snapshot = Snapshot::default();
    snapshot.mut_metadata().index = index;
    snapshot.mut_metadata().term = term;
    snapshot
        .mut_metadata()
        .set_conf_state(log.initial_state()?.conf_state);
    snapshot.data = b"storage-contract/application-descriptor".to_vec().into();
    Ok(snapshot)
}
fn verify_installed(log: &PhysicalLog, snapshot: &Snapshot) -> Result<()> {
    let group = log.group(10)?;
    group.bind_application(b"snapshot-install-test/v1")?;
    assert!(group.bind_application(b"different-application").is_err());
    assert_eq!(group.snapshot(0, 1)?, *snapshot);
    assert_eq!(group.initial_state()?.hard_state, hard(3, 5, 0));
    assert_eq!(
        (group.first_index()?, group.last_index()?, group.term(4)?),
        (5, 6, 3)
    );
    assert_eq!(read(&group, 5, 7)?, vec![entry(5, 3), entry(6, 3)]);
    assert!(matches!(
        group.term(3),
        Err(raft::Error::Store(StorageError::Compacted))
    ));
    assert!(matches!(
        group.term(7),
        Err(raft::Error::Store(StorageError::Unavailable))
    ));
    assert_eq!(read(&log.group(20)?, 1, 2)?, vec![entry(1, 7)]);
    assert_eq!(log.group(20)?.initial_state()?.hard_state, hard(7, 1, 2));
    Ok(())
}

#[test]
fn received_ready_replaces_divergent_tail_and_recovers_committed_suffix() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("log");
    let saved;
    {
        let log = seed(&path)?;
        let group = log.group(10)?;
        let mut raw = raft::RawNode::new(
            &raft::Config {
                id: 2,
                applied: 1,
                ..Default::default()
            },
            group.clone(),
            &slog::Logger::root(slog::Discard, slog::o!()),
        )?;
        saved = snapshot(&group, 4, 3)?;
        let mut received = Message {
            from: 1,
            to: 2,
            term: 3,
            ..Default::default()
        };
        received.set_msg_type(MessageType::MsgSnapshot);
        received.set_snapshot(saved.clone());
        raw.step(received)?;
        let mut suffix = Message {
            from: 1,
            to: 2,
            term: 3,
            index: 4,
            log_term: 3,
            commit: 5,
            ..Default::default()
        };
        suffix.set_msg_type(MessageType::MsgAppend);
        suffix.set_entries(vec![entry(5, 3), entry(6, 3)].into());
        raw.step(suffix)?;
        let mut ready = raw.ready();
        assert_eq!(ready.snapshot(), &saved);
        assert_eq!(ready.entries(), &[entry(5, 3), entry(6, 3)]);
        assert_eq!(read(&group, 1, 8)?, original());
        group.install_snapshot(
            ready.snapshot(),
            ready.entries(),
            ready.hs().context("missing Ready HardState")?,
        )?;
        verify_installed(&log, &saved)?;
        // Release/apply Ready only after the complete storage transition.
        let mut applied = ready.take_committed_entries();
        let mut light = raw.advance(ready);
        if light.commit_index().is_some() {
            group.persist(&[], Some(&raw.raft.hard_state()))?;
        }
        applied.extend(light.take_committed_entries());
        raw.advance_apply();
        assert_eq!(applied, vec![entry(5, 3)]);
    }
    let log = PhysicalLog::open(&path, &layout(), 2)?;
    verify_installed(&log, &saved)?;
    let group = log.group(10)?;
    let mut raw = raft::RawNode::new(
        &raft::Config {
            id: 2,
            applied: 4,
            ..Default::default()
        },
        group.clone(),
        &slog::Logger::root(slog::Discard, slog::o!()),
    )?;
    assert_eq!(raw.ready().take_committed_entries(), vec![entry(5, 3)]);
    drop(raw);
    group.persist(&[entry(7, 3)], Some(&hard(3, 7, 0)))?;
    let next = snapshot(&group, 9, 4)?;
    group.install_snapshot(&next, &[], &hard(4, 9, 0))?;
    assert_eq!((group.first_index()?, group.last_index()?), (10, 9));
    drop((group, log));
    let log = PhysicalLog::open(&path, &layout(), 2)?;
    let group = log.group(10)?;
    assert_eq!(group.snapshot(9, 1)?, next);
    assert!(read(&group, 10, 10)?.is_empty());
    group.persist(&[entry(10, 4)], Some(&hard(4, 10, 0)))?;
    assert_eq!(read(&group, 10, 11)?, vec![entry(10, 4)]);
    Ok(())
}

#[test]
fn invalid_install_preserves_prior_log_and_vote() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let log = seed(&directory.path().join("log"))?;
    let group = log.group(10)?;
    let good = snapshot(&group, 4, 3)?;
    let suffix = vec![entry(5, 3), entry(6, 3)];
    let mut cases = vec![
        (snapshot(&group, 1, 1)?, suffix.clone(), hard(3, 5, 0)),
        (good.clone(), suffix.clone(), hard(3, 3, 0)),
        (good.clone(), suffix.clone(), hard(3, 7, 0)),
        (good.clone(), suffix.clone(), hard(3, 5, 4)),
        (good.clone(), vec![entry(6, 3)], hard(3, 6, 0)),
        (good.clone(), vec![entry(4, 3)], hard(3, 4, 0)),
        (good.clone(), vec![entry(5, 2)], hard(3, 5, 0)),
        (good.clone(), vec![entry(5, 4)], hard(3, 5, 0)),
        (snapshot(&group, 4, 2)?, vec![], hard(2, 4, 1)),
        (snapshot(&group, 4, 1)?, vec![], hard(1, 4, 1)),
    ];
    for corruption in 0..4 {
        let mut invalid = good.clone();
        match corruption {
            0 => invalid.mut_metadata().mut_conf_state().voters = vec![1, 2, 4],
            1 => invalid.data.clear(),
            2 => invalid.data = vec![0; 4096].into(),
            _ => invalid.mut_metadata().index = u64::MAX,
        }
        cases.push((invalid, suffix.clone(), hard(3, 5, 0)));
    }
    for (snapshot, entries, next_hard) in cases {
        assert!(group
            .install_snapshot(&snapshot, &entries, &next_hard)
            .is_err());
        assert_eq!(group.initial_state()?.hard_state, hard(2, 1, 2));
        assert_eq!(read(&group, 1, 8)?, original());
    }
    let unbound = log.group(20)?;
    assert!(unbound
        .install_snapshot(&snapshot(&unbound, 4, 8)?, &[], &hard(8, 4, 0))
        .is_err());
    group.install_snapshot(&good, &suffix, &hard(3, 5, 0))?;
    assert!(group
        .install_snapshot(&good, &suffix, &hard(3, 5, 0))
        .is_err());
    verify_installed(&log, &good)?;
    let directory = tempfile::tempdir()?;
    let log = seed(&directory.path().join("log"))?;
    let group = log.group(10)?;
    group.persist(&[], Some(&hard(2, 3, 2)))?;
    assert!(group
        .install_snapshot(&snapshot(&group, 4, 1)?, &[], &hard(3, 4, 0))
        .is_err());
    assert_eq!(group.initial_state()?.hard_state, hard(2, 3, 2));
    assert_eq!(read(&group, 1, 8)?, original());
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
    #[ignore = "subprocess fixture invoked by install_survives_sigkill"]
    fn install_worker() -> Result<()> {
        let directory = std::path::PathBuf::from(std::env::var("CONTEXT_INSTALL_TEST_ROOT")?);
        let log = seed(&directory.join("log"))?;
        let group = log.group(10)?;
        if std::env::var("CONTEXT_INSTALL_TEST_PHASE")? == "after" {
            group.install_snapshot(
                &snapshot(&group, 4, 3)?,
                &[entry(5, 3), entry(6, 3)],
                &hard(3, 5, 0),
            )?;
        }
        std::fs::write(directory.join("ready"), b"ready")?;
        std::thread::sleep(Duration::from_secs(30));
        anyhow::bail!("parent did not stop snapshot worker")
    }
    #[test]
    fn install_survives_sigkill() -> Result<()> {
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
                        "process::install_worker",
                        "--nocapture",
                    ])
                    .env("CONTEXT_INSTALL_TEST_ROOT", directory.path())
                    .env("CONTEXT_INSTALL_TEST_PHASE", phase)
                    .stdout(Stdio::null())
                    .stderr(std::fs::File::create(&stderr)?)
                    .spawn()?,
            );
            let deadline = Instant::now() + Duration::from_secs(20);
            while !directory.path().join("ready").exists() {
                anyhow::ensure!(
                    child.0.try_wait()?.is_none() && Instant::now() < deadline,
                    "snapshot worker did not reach {phase}: {}",
                    std::fs::read_to_string(&stderr)?
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            child.0.kill()?;
            assert_eq!(child.0.wait()?.signal(), Some(9));
            let log = PhysicalLog::open(&directory.path().join("log"), &layout(), 2)?;
            let group = log.group(10)?;
            if phase == "after" {
                verify_installed(&log, &snapshot(&group, 4, 3)?)?;
            } else {
                assert_eq!(group.initial_state()?.hard_state, hard(2, 1, 2));
                assert_eq!(read(&group, 1, 8)?, original());
                assert_eq!(read(&log.group(20)?, 1, 2)?, vec![entry(1, 7)]);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "subprocess fixture for the inherited directory lock regression"]
    fn inherited_lock_child() {}

    #[test]
    fn pre_exec_child_does_not_extend_an_unrelated_engine_lock() -> Result<()> {
        use std::{
            io::{Read, Write},
            os::{fd::AsRawFd, unix::net::UnixStream, unix::process::CommandExt},
        };

        let _case = TEST_CASE.lock().unwrap();
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("log");
        let log = seed(&path)?;
        let (mut parent, child) = UnixStream::pair()?;
        parent.set_read_timeout(Some(Duration::from_secs(5)))?;
        parent.set_write_timeout(Some(Duration::from_secs(5)))?;
        let executable = std::env::current_exe()?;
        let spawn = std::thread::spawn(move || -> Result<_> {
            let fd = child.as_raw_fd();
            let mut command = Command::new(executable);
            command
                .args(["--ignored", "--exact", "process::inherited_lock_child"])
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            // Only async-signal-safe syscalls and allocation-free error values
            // run between fork and exec. The deadline also bounds a failed parent.
            unsafe {
                command.pre_exec(move || {
                    let mut byte = 1u8;
                    if libc::write(fd, (&byte as *const u8).cast(), 1) != 1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    let mut poll = libc::pollfd {
                        fd,
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    if libc::poll(&mut poll, 1, 10_000) != 1 {
                        return Err(std::io::Error::from_raw_os_error(libc::ETIMEDOUT));
                    }
                    if libc::read(fd, (&mut byte as *mut u8).cast(), 1) != 1 {
                        return Err(std::io::Error::from_raw_os_error(libc::EPIPE));
                    }
                    Ok(())
                });
            }
            let status = command.spawn()?.wait()?;
            drop(child);
            Ok(status)
        });
        let ready = parent.read_exact(&mut [0]);
        let live_excluded = ready.is_ok() && PhysicalLog::open(&path, &layout(), 2).is_err();
        drop(log);
        let reopened = ready
            .as_ref()
            .ok()
            .map(|_| PhysicalLog::open(&path, &layout(), 2));
        // The child is still parked before exec with the inherited descriptor.
        // The replacement owner must nevertheless acquire and hold a fresh lock.
        let replacement_excluded = reopened.as_ref().is_some_and(Result::is_ok)
            && PhysicalLog::open(&path, &layout(), 2).is_err();
        // Always release and join the bounded child before propagating any error.
        let released = parent.write_all(&[1]);
        let child = spawn
            .join()
            .map_err(|_| anyhow::anyhow!("spawn thread panicked"))?;
        ready?;
        released?;
        assert!(child?.success());
        assert!(live_excluded, "live engine must exclude another owner");
        let reopened = reopened.expect("child ready")?;
        assert!(replacement_excluded, "replacement must own its lock");
        assert!(
            PhysicalLog::open(&path, &layout(), 2).is_err(),
            "child exec must not release the replacement's lock"
        );
        assert_eq!(read(&reopened.group(10)?, 1, 8)?, original());
        assert_eq!(read(&reopened.group(20)?, 1, 2)?, vec![entry(1, 7)]);
        drop(reopened);
        let reopened_again = PhysicalLog::open(&path, &layout(), 2)?;
        assert_eq!(read(&reopened_again.group(10)?, 1, 8)?, original());
        Ok(())
    }
}
