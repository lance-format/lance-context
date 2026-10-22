#![cfg(feature = "raft-log")]

use std::collections::BTreeMap;

use anyhow::Result;
use lance_context_wal::placement::Layout;
use lance_context_wal::raft_log::{GroupLog, PhysicalLog};
use protobuf::Message;
use raft::{GetEntriesContext, Storage};
use raft_proto::eraftpb::{Entry, EntryType, HardState};
use uuid::Uuid;

// A child spawned by another case can briefly inherit this case's flock until
// exec. Exclude that window around open/drop/reopen sequences. Concurrency and
// real child crashes within a case remain part of the test.
static TEST_CASE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn layout() -> Layout {
    Layout {
        cluster: Uuid::from_u128(1),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
        buckets: vec![10, 20],
    }
}

fn hard(term: u64, vote: u64, commit: u64) -> HardState {
    HardState {
        term,
        vote,
        commit,
        ..Default::default()
    }
}

fn entry(index: u64, term: u64, body: Vec<u8>) -> Entry {
    Entry {
        index,
        term,
        data: body.into(),
        ..Default::default()
    }
}

fn read(log: &GroupLog, low: u64, high: u64, limit: Option<u64>) -> raft::Result<Vec<Entry>> {
    log.entries(low, high, limit, GetEntriesContext::empty(false))
}

#[test]
fn recovers_two_groups_and_rejects_layout_changes() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("log");
    let entries = vec![
        entry(1, 1, vec![1; 4096]),
        entry(2, 3, vec![2; 8192]),
        entry(3, 3, vec![3; 2048]),
    ];
    {
        let log = PhysicalLog::open(&path, &layout(), 1)?;
        assert!(log.group(20)?.persist(&[], Some(&hard(0, 1, 0))).is_err());
        log.group(10)?.persist(&entries, Some(&hard(3, 2, 2)))?;
        log.group(20)?
            .persist(&[entry(1, 8, vec![9; 4096])], Some(&hard(8, 3, 1)))?;
        assert!(log.group(99).is_err());
        assert!(
            PhysicalLog::open(&path, &layout(), 1).is_err(),
            "exclusive physical engine lock"
        );
    }
    {
        let log = PhysicalLog::open(&path, &layout(), 1)?;
        let group = log.group(10)?;
        assert_eq!(group.initial_state()?.hard_state, hard(3, 2, 2));
        assert_eq!(group.initial_state()?.conf_state.voters, vec![1, 2, 3]);
        assert_eq!(read(&group, 1, 4, None)?, entries);
        assert_eq!(
            (
                group.term(0)?,
                group.term(1)?,
                group.term(2)?,
                group.term(3)?
            ),
            (0, 1, 3, 3)
        );
        assert_eq!(log.group(20)?.initial_state()?.hard_state, hard(8, 3, 1));
        assert_eq!(
            read(&log.group(20)?, 1, 2, None)?[0].data.as_ref(),
            vec![9; 4096]
        );
    }
    let mut changed = layout();
    changed
        .nodes
        .insert(2, "changed.example.invalid:7800".into());
    assert!(PhysicalLog::open(&path, &changed, 1).is_err());
    assert!(PhysicalLog::open(&path, &layout(), 2).is_err());
    let log = PhysicalLog::open(&path, &layout(), 1)?;
    assert_eq!(read(&log.group(10)?, 1, 4, None)?, entries);
    Ok(())
}

#[test]
fn foreign_engine_and_invalid_metadata_are_rejected_without_rebinding() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    fn engine(path: &std::path::Path) -> Result<raft_engine::Engine> {
        Ok(raft_engine::Engine::open(raft_engine::Config {
            dir: path.to_str().unwrap().to_owned(),
            ..Default::default()
        })?)
    }
    let directory = tempfile::tempdir()?;
    let foreign = directory.path().join("foreign");
    {
        let engine = engine(&foreign)?;
        let mut batch = raft_engine::LogBatch::default();
        batch.put(99, b"foreign".to_vec(), b"retain-me".to_vec())?;
        engine.write(&mut batch, true)?;
    }
    assert!(PhysicalLog::open(&foreign, &layout(), 1).is_err());
    {
        let engine = engine(&foreign)?;
        assert_eq!(engine.get(99, b"foreign"), Some(b"retain-me".to_vec()));
        assert!(engine.get(0, b"identity").is_none());
    }
    let invalid = directory.path().join("invalid");
    {
        let log = PhysicalLog::open(&invalid, &layout(), 1)?;
        log.group(10)?
            .persist(&[entry(1, 1, vec![9; 4096])], Some(&hard(1, 1, 1)))?;
    }
    {
        let engine = engine(&invalid)?;
        let mut batch = raft_engine::LogBatch::default();
        batch.put(10, b"terms".to_vec(), b"invalid".to_vec())?;
        engine.write(&mut batch, true)?;
    }
    assert!(PhysicalLog::open(&invalid, &layout(), 1).is_err());
    let engine = engine(&invalid)?;
    assert_eq!(engine.last_index(10), Some(1));
    assert_eq!(
        engine.get_message::<HardState>(10, b"hard")?,
        Some(hard(1, 1, 1))
    );
    assert_eq!(engine.get(10, b"terms"), Some(b"invalid".to_vec()));
    Ok(())
}

#[test]
fn replaces_only_uncommitted_suffix_and_preserves_vote() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("log");
    let expected = vec![
        entry(1, 1, vec![1]),
        entry(2, 1, vec![2]),
        entry(3, 2, vec![33]),
        entry(4, 2, vec![44]),
    ];
    {
        let log = PhysicalLog::open(&path, &layout(), 1)?;
        let group = log.group(10)?;
        let initial: Vec<_> = (1..=5)
            .map(|index| entry(index, 1, vec![index as u8]))
            .collect();
        group.persist(&initial, Some(&hard(1, 1, 2)))?;
        group.persist(&expected[2..], Some(&hard(2, 2, 3)))?;
        assert_eq!(group.last_index()?, 4);
        assert!(group.term(5).is_err());
        for invalid in [
            hard(1, 1, 3),
            hard(2, 3, 3),
            hard(2, 0, 3),
            hard(2, 2, 2),
            hard(2, 2, 5),
        ] {
            assert!(group.persist(&[], Some(&invalid)).is_err());
        }
        for invalid in [
            vec![entry(3, 2, vec![99])],
            vec![entry(6, 2, vec![])],
            vec![entry(5, 2, vec![]), entry(7, 2, vec![])],
            vec![entry(5, 1, vec![])],
            vec![entry(5, 3, vec![])],
        ] {
            assert!(group.persist(&invalid, None).is_err());
        }
        let mut membership = entry(5, 2, vec![]);
        membership.set_entry_type(EntryType::EntryConfChange);
        assert!(group.persist(&[membership], None).is_err());
        assert_eq!(read(&group, 1, 5, None)?, expected);
        assert_eq!(group.initial_state()?.hard_state, hard(2, 2, 3));
    }
    let log = PhysicalLog::open(&path, &layout(), 1)?;
    assert_eq!(read(&log.group(10)?, 1, 5, None)?, expected);
    assert_eq!(log.group(10)?.initial_state()?.hard_state, hard(2, 2, 3));
    Ok(())
}

#[test]
fn read_size_and_chunk_boundaries_obey_raft_storage_contract() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(&directory.path().join("log"), &layout(), 1)?;
    let group = log.group(10)?;
    let entries: Vec<_> = (1..=300)
        .map(|index| {
            entry(
                index,
                1,
                vec![index as u8; if index == 257 { 4 * 1024 * 1024 } else { 1024 }],
            )
        })
        .collect();
    group.persist(&entries, Some(&hard(1, 1, 300)))?;
    assert_eq!(read(&group, 1, 301, Some(0))?, entries[..1]);
    assert_eq!(read(&group, 257, 301, Some(1))?, entries[256..257]);
    let prefix_size: u64 = entries[..256]
        .iter()
        .map(|entry| u64::from(entry.compute_size()))
        .sum();
    assert_eq!(
        read(&group, 1, 301, Some(prefix_size + 10))?,
        entries[..256]
    );
    assert_eq!(read(&group, 1, 301, None)?, entries);
    assert!(read(&group, 10, 10, Some(0))?.is_empty());
    assert!(read(&group, 0, 1, None).is_err());
    assert!(read(&group, 1, 302, None).is_err());
    assert!(read(&group, 10, 9, None).is_err());
    assert!(group.snapshot(1, 2).is_err());
    Ok(())
}

#[test]
fn concurrent_groups_share_engine_without_sharing_positions() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("log");
    {
        let log = PhysicalLog::open(&path, &layout(), 1)?;
        let writers: Vec<_> = [10, 20]
            .into_iter()
            .map(|id| {
                let group = log.group(id).unwrap();
                std::thread::spawn(move || -> Result<()> {
                    for index in 1..=32 {
                        group.persist(
                            &[entry(index, id, vec![id as u8; 8192])],
                            Some(&hard(id, 1, index)),
                        )?;
                    }
                    Ok(())
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap()?;
        }
    }
    let log = PhysicalLog::open(&path, &layout(), 1)?;
    for id in [10, 20] {
        let group = log.group(id)?;
        assert_eq!(group.initial_state()?.hard_state, hard(id, 1, 32));
        let actual = read(&group, 1, 33, None)?;
        assert_eq!(actual.len(), 32);
        assert!(actual
            .iter()
            .all(|entry| entry.term == id && entry.data.as_ref() == vec![id as u8; 8192]));
    }
    Ok(())
}

#[test]
fn raft_ready_persistence_preserves_committed_data_through_election_and_restart() -> Result<()> {
    let _case = TEST_CASE.lock().unwrap();
    use raft::{Config, RawNode, StateRole};
    use std::path::Path;

    struct Replica {
        raw: RawNode<GroupLog>,
        log: GroupLog,
        applied: Vec<Entry>,
        _engine: PhysicalLog,
    }
    fn open(root: &Path, id: u64) -> Result<Replica> {
        let engine = PhysicalLog::open(&root.join(format!("node-{id}")), &layout(), id)?;
        let log = engine.group(10)?;
        let config = Config {
            id,
            election_tick: 10,
            heartbeat_tick: 1,
            ..Default::default()
        };
        let logger = slog::Logger::root(slog::Discard, slog::o!());
        let raw = RawNode::new(&config, log.clone(), &logger)?;
        Ok(Replica {
            raw,
            log,
            applied: Vec::new(),
            _engine: engine,
        })
    }
    fn apply(replica: &mut Replica, committed: Vec<Entry>) {
        for entry in committed {
            assert_eq!(
                entry.index,
                replica.applied.last().map_or(1, |old| old.index + 1)
            );
            replica.applied.push(entry);
        }
    }
    // Manually initiated elections and deterministic in-process message delivery;
    // only the storage/Ready contract is under test, not failure-detector timing.
    // Each replica has real independent engine files. This is not a network or
    // multiprocess throughput/failover benchmark.
    fn drive(replicas: &mut BTreeMap<u64, Replica>) -> Result<()> {
        for _ in 0..1000 {
            let mut changed = false;
            let mut messages = Vec::new();
            for replica in replicas.values_mut() {
                if !replica.raw.has_ready() {
                    continue;
                }
                changed = true;
                let mut ready = replica.raw.ready();
                assert_eq!(ready.snapshot().get_metadata().index, 0);
                replica.log.persist(ready.entries(), ready.hs())?;
                messages.extend(ready.take_messages());
                messages.extend(ready.take_persisted_messages());
                apply(replica, ready.take_committed_entries());
                let mut light = replica.raw.advance_append(ready);
                if light.commit_index().is_some() {
                    replica
                        .log
                        .persist(&[], Some(&replica.raw.raft.hard_state()))?;
                }
                messages.extend(light.take_messages());
                apply(replica, light.take_committed_entries());
                replica.raw.advance_apply();
            }
            for message in messages {
                if let Some(replica) = replicas.get_mut(&message.to) {
                    replica.raw.step(message)?;
                }
            }
            if !changed {
                return Ok(());
            }
        }
        anyhow::bail!("Ready delivery did not quiesce")
    }
    fn durable_data(replica: &Replica) -> Result<Vec<Vec<u8>>> {
        Ok(read(&replica.log, 1, replica.log.last_index()? + 1, None)?
            .into_iter()
            .filter(|entry| !entry.data.is_empty())
            .map(|entry| entry.data.to_vec())
            .collect())
    }
    let root = tempfile::tempdir()?;
    let mut replicas: BTreeMap<_, _> = (1..=3)
        .map(|id| Ok((id, open(root.path(), id)?)))
        .collect::<Result<_>>()?;
    replicas.get_mut(&1).unwrap().raw.campaign()?;
    drive(&mut replicas)?;
    assert_eq!(replicas[&1].raw.raft.state, StateRole::Leader);
    let first = vec![11; 8192];
    let second = vec![22; 16384];
    replicas
        .get_mut(&1)
        .unwrap()
        .raw
        .propose(Vec::new(), first.clone())?;
    drive(&mut replicas)?;
    assert!(replicas.values().all(|replica| replica
        .applied
        .iter()
        .any(|entry| entry.data.as_ref() == first)));
    replicas.remove(&1);
    replicas.get_mut(&2).unwrap().raw.campaign()?;
    drive(&mut replicas)?;
    assert_eq!(replicas[&2].raw.raft.state, StateRole::Leader);
    replicas
        .get_mut(&2)
        .unwrap()
        .raw
        .propose(Vec::new(), second.clone())?;
    drive(&mut replicas)?;
    replicas.insert(1, open(root.path(), 1)?);
    replicas.get_mut(&2).unwrap().raw.tick();
    drive(&mut replicas)?;
    let expected = vec![first, second];
    for replica in replicas.values() {
        assert_eq!(durable_data(replica)?, expected);
        assert_eq!(
            replica.log.initial_state()?.hard_state.commit,
            replica.log.last_index()?
        );
    }
    let committed = replicas[&2].log.initial_state()?.hard_state.commit;
    replicas.remove(&1);
    replicas.remove(&3);
    replicas
        .get_mut(&2)
        .unwrap()
        .raw
        .propose(Vec::new(), vec![99; 4096])?;
    drive(&mut replicas)?;
    assert_eq!(
        replicas[&2].log.initial_state()?.hard_state.commit,
        committed
    );
    assert!(
        replicas[&2].log.last_index()? > committed,
        "local persistence is not quorum commit"
    );
    drop(replicas);
    let mut replicas: BTreeMap<_, _> = (1..=3)
        .map(|id| Ok((id, open(root.path(), id)?)))
        .collect::<Result<_>>()?;
    replicas.get_mut(&1).unwrap().raw.campaign()?;
    drive(&mut replicas)?;
    assert_eq!(replicas[&1].raw.raft.state, StateRole::Leader);
    for replica in replicas.values() {
        assert_eq!(durable_data(replica)?, expected);
        assert_eq!(
            replica.log.initial_state()?.hard_state.commit,
            replica.log.last_index()?
        );
    }
    let third = vec![33; 4096];
    replicas
        .get_mut(&1)
        .unwrap()
        .raw
        .propose(Vec::new(), third.clone())?;
    drive(&mut replicas)?;
    let mut expected = expected;
    expected.push(third);
    for replica in replicas.values() {
        assert_eq!(durable_data(replica)?, expected);
        assert_eq!(
            replica.log.initial_state()?.hard_state.commit,
            replica.log.last_index()?
        );
        let applied: Vec<_> = replica
            .applied
            .iter()
            .filter(|entry| !entry.data.is_empty())
            .map(|entry| entry.data.to_vec())
            .collect();
        assert_eq!(applied, expected);
    }
    Ok(())
}

#[cfg(target_os = "linux")]
mod process {
    use super::*;
    use std::io::Write;
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::time::{Duration, Instant};

    struct Worker(Child);

    impl Drop for Worker {
        fn drop(&mut self) {
            // Even a failed assertion must not leave the disposable worker alive.
            if !matches!(self.0.try_wait(), Ok(Some(_))) {
                let _ = self.0.kill();
            }
            let _ = self.0.wait();
        }
    }

    impl Worker {
        fn wait(&mut self, deadline: Instant) -> Result<ExitStatus> {
            loop {
                if let Some(status) = self.0.try_wait()? {
                    return Ok(status);
                }
                anyhow::ensure!(Instant::now() < deadline, "worker exit timed out");
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    fn prefix_len(mode: &str) -> u64 {
        if mode == "rotate-kill" {
            272
        } else {
            8
        }
    }

    fn body(index: u64) -> Vec<u8> {
        (0..256 * 1024)
            .map(|offset| (offset as u64).wrapping_mul(17).wrapping_add(index) as u8)
            .collect()
    }

    #[test]
    #[ignore = "invoked in an isolated subprocess by the crash and write-error tests"]
    fn journal_worker() -> Result<()> {
        let path = std::env::var("CONTEXT_WAL_LOG_TEST_DIRECTORY")?;
        let mode = std::env::var("CONTEXT_WAL_LOG_TEST_MODE")?;
        let log = PhysicalLog::open(Path::new(&path), &layout(), 1)?;
        let group = log.group(10)?;
        for index in 1..=prefix_len(&mode) {
            group.persist(&[entry(index, 1, body(index))], Some(&hard(1, 1, index)))?;
        }
        println!("DURABLE_PREFIX_READY");
        std::io::stdout().flush()?;
        if mode == "kill" || mode == "rotate-kill" {
            std::thread::sleep(std::time::Duration::from_secs(60));
            anyhow::bail!("parent did not kill worker");
        }
        assert_eq!(mode, "write-error");
        // Confined to this disposable child. Force an actual file write failure
        // after a synced prefix; restore the limit before closing the engine.
        let mut saved = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        unsafe {
            assert_eq!(libc::getrlimit(libc::RLIMIT_FSIZE, &mut saved), 0);
            libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
            let limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: saved.rlim_max,
            };
            assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &limit), 0);
        }
        let result = group.persist(&[entry(9, 1, body(9))], Some(&hard(1, 1, 9)));
        unsafe {
            assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &saved), 0);
        }
        assert!(
            result.is_err(),
            "file size limit must fail the actual engine write"
        );
        assert!(group.term(1).is_err());
        assert!(group.initial_state().is_err());
        assert!(log.group(20)?.persist(&[], Some(&hard(1, 1, 0))).is_err());
        assert!(log.group(20)?.first_index().is_err());
        println!("ENGINE_HANDLES_FENCED");
        Ok(())
    }

    use std::path::Path;

    fn run_child(mode: &str) -> Result<()> {
        let _case = TEST_CASE.lock().unwrap();
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("log");
        let stdout = directory.path().join("worker.stdout");
        let stderr = directory.path().join("worker.stderr");
        // Regular files avoid pipe backpressure and unbounded blocking reads.
        let mut child = Worker(
            Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "process::journal_worker",
                    "--ignored",
                    "--nocapture",
                ])
                .env("CONTEXT_WAL_LOG_TEST_DIRECTORY", &path)
                .env("CONTEXT_WAL_LOG_TEST_MODE", mode)
                .stdout(Stdio::from(std::fs::File::create(&stdout)?))
                .stderr(Stdio::from(std::fs::File::create(&stderr)?))
                .spawn()?,
        );
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let status = child.0.try_wait()?;
            if std::fs::read_to_string(&stdout)?.contains("DURABLE_PREFIX_READY") {
                break;
            }
            anyhow::ensure!(
                status.is_none() && Instant::now() < deadline,
                "worker failed to report synced prefix: {status:?}; {}",
                std::fs::read_to_string(&stderr)?
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        if mode == "kill" || mode == "rotate-kill" {
            child.0.kill()?;
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(child.wait(deadline)?.signal(), Some(9));
        } else {
            assert!(
                child.wait(deadline)?.success(),
                "{}",
                std::fs::read_to_string(&stderr)?
            );
            assert!(std::fs::read_to_string(&stdout)?.contains("ENGINE_HANDLES_FENCED"));
        }
        if mode == "rotate-kill" {
            let segments = std::fs::read_dir(&path)?
                .map(|entry| {
                    Ok(entry?
                        .path()
                        .extension()
                        .is_some_and(|ext| ext == "raftlog"))
                })
                .collect::<std::io::Result<Vec<_>>>()?;
            assert!(segments.into_iter().filter(|is_log| *is_log).count() > 1);
        }
        let log = PhysicalLog::open(&path, &layout(), 1)?;
        let group = log.group(10)?;
        let state = group.initial_state()?.hard_state;
        assert_eq!((state.term, state.vote), (1, 1));
        let uncertain_suffix = u64::from(mode == "write-error");
        assert!(
            state.commit >= prefix_len(mode) && state.commit <= prefix_len(mode) + uncertain_suffix
        );
        assert_eq!(group.last_index()?, state.commit);
        for start in (1..=state.commit).step_by(16) {
            let end = (start + 16).min(state.commit + 1);
            let entries = read(&group, start, end, None)?;
            assert_eq!(entries.len() as u64, end - start);
            for (index, entry) in (start..end).zip(entries) {
                assert_eq!(entry.index, index);
                assert_eq!(entry.data.as_ref(), body(entry.index));
                assert_eq!(group.term(entry.index)?, 1);
            }
        }
        Ok(())
    }

    #[test]
    fn acknowledged_prefix_survives_sigkill() -> Result<()> {
        run_child("kill")
    }

    #[test]
    fn acknowledged_prefix_survives_rotation_and_sigkill() -> Result<()> {
        run_child("rotate-kill")
    }

    #[test]
    fn write_error_fences_all_groups_and_reopen_preserves_synced_prefix() -> Result<()> {
        run_child("write-error")
    }
}
