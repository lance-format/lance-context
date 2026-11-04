//! Test-only filesystem adapter: real files with a controlled sync error. No
//! fault adapter or injection flag is included in ordinary library/binary builds.
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use raft_engine::env::{DefaultFileSystem, FileSystem, Handle, Permission};

use super::*;

type FileHandle = <DefaultFileSystem as FileSystem>::Handle;

#[derive(Default)]
pub(super) struct FaultControl {
    fail_sync: AtomicBool,
    fail_once: AtomicBool,
    fail_rewrite_sync: AtomicBool,
    pause_rewrite_sync: AtomicBool,
    pause_payload_read: AtomicBool,
    sync_calls: AtomicU64,
    entered: Mutex<Option<mpsc::SyncSender<()>>>,
    release: Mutex<Option<mpsc::Receiver<()>>>,
}

impl FaultControl {
    pub(super) fn pause_rewrite_sync_for_test(&self) -> FilePause {
        self.pause_for_test(&self.pause_rewrite_sync)
    }

    pub(super) fn pause_payload_read_for_test(&self) -> FilePause {
        self.pause_for_test(&self.pause_payload_read)
    }

    fn pause_for_test(&self, armed: &AtomicBool) -> FilePause {
        let (entered, wait_entered) = mpsc::sync_channel(1);
        let (release, wait_release) = mpsc::channel();
        *self.entered.lock().unwrap() = Some(entered);
        *self.release.lock().unwrap() = Some(wait_release);
        armed.store(true, Ordering::Release);
        FilePause {
            entered: wait_entered,
            release,
        }
    }

    pub(super) fn fail_rewrite_sync_for_test(self: &Arc<Self>) -> RewriteSyncFault {
        self.fail_rewrite_sync.store(true, Ordering::Release);
        RewriteSyncFault(self.clone())
    }

    #[cfg(feature = "replication")]
    pub(super) fn fail_sync_for_test(&self) {
        self.fail_once.store(true, Ordering::Release);
    }
}

pub(super) struct FilePause {
    entered: mpsc::Receiver<()>,
    release: mpsc::Sender<()>,
}

impl FilePause {
    pub(super) fn wait(&self) -> Result<()> {
        self.entered.recv_timeout(Duration::from_secs(10))?;
        Ok(())
    }
}

impl Drop for FilePause {
    fn drop(&mut self) {
        let _ = self.release.send(());
    }
}

pub(super) struct RewriteSyncFault(Arc<FaultControl>);

impl Drop for RewriteSyncFault {
    fn drop(&mut self) {
        // Disarm before engine destructors, including if an assertion fails
        // before a native rewrite reaches the intended injection point.
        self.0.fail_rewrite_sync.store(false, Ordering::Release);
    }
}

pub(super) struct FaultHandle {
    inner: Arc<FileHandle>,
    control: Arc<FaultControl>,
    rewrite: bool,
}

impl Handle for FaultHandle {
    fn truncate(&self, offset: usize) -> io::Result<()> {
        self.inner.truncate(offset)
    }

    fn file_size(&self) -> io::Result<usize> {
        self.inner.file_size()
    }

    fn sync(&self) -> io::Result<()> {
        self.control.sync_calls.fetch_add(1, Ordering::Relaxed);
        if self.rewrite
            && self
                .control
                .pause_rewrite_sync
                .swap(false, Ordering::AcqRel)
        {
            if let Some(entered) = self.control.entered.lock().unwrap().take() {
                let _ = entered.send(());
            }
            if let Some(release) = self.control.release.lock().unwrap().take() {
                let _ = release.recv_timeout(Duration::from_secs(10));
            }
        }
        if self.control.fail_once.swap(false, Ordering::AcqRel)
            || self.control.fail_sync.load(Ordering::Acquire)
            || (self.rewrite && self.control.fail_rewrite_sync.swap(false, Ordering::AcqRel))
        {
            if let Some(entered) = self.control.entered.lock().unwrap().take() {
                entered.send(()).unwrap();
            }
            if let Some(release) = self.control.release.lock().unwrap().take() {
                release.recv_timeout(Duration::from_secs(10)).unwrap();
            }
            return Err(io::Error::other("injected file sync failure"));
        }
        self.inner.sync()
    }
}

#[test]
fn concurrent_appends_share_one_real_sync() -> Result<()> {
    let layout = Layout {
        cluster: uuid::Uuid::from_u128(2),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
        buckets: vec![10, 20],
    };
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(&directory.path().join("log"), &layout, 1)?;
    let node = log.group(10)?.node.clone();
    let before = node.fault.sync_calls.load(Ordering::Relaxed);
    let gate = node.sync_gate.lock().unwrap();
    let writers: Vec<_> = [10, 20]
        .into_iter()
        .map(|id| {
            let group = log.group(id).unwrap();
            std::thread::spawn(move || {
                group.persist(
                    &[Entry {
                        index: 1,
                        term: 1,
                        data: vec![id as u8; 4096].into(),
                        ..Default::default()
                    }],
                    Some(&HardState {
                        term: 1,
                        vote: 1,
                        commit: 1,
                        ..Default::default()
                    }),
                )
            })
        })
        .collect();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while node.engine.last_index(10) != Some(1) || node.engine.last_index(20) != Some(1) {
        assert!(
            std::time::Instant::now() < deadline,
            "both appends must reach the real engine before sync"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(node.fault.sync_calls.load(Ordering::Relaxed), before);
    drop(gate);
    for writer in writers {
        writer.join().unwrap()?;
    }
    assert_eq!(node.fault.sync_calls.load(Ordering::Relaxed) - before, 1);
    for id in [10, 20] {
        let group = log.group(id)?;
        assert_eq!(group.initial_state()?.hard_state.commit, 1);
        assert_eq!(
            group.entries(1, 2, None, GetEntriesContext::empty(false))?[0]
                .data
                .as_ref(),
            vec![id as u8; 4096]
        );
    }
    // An append submitted after that barrier needs a new physical sync.
    log.group(10)?.persist(
        &[],
        Some(&HardState {
            term: 2,
            vote: 2,
            commit: 1,
            ..Default::default()
        }),
    )?;
    assert_eq!(node.fault.sync_calls.load(Ordering::Relaxed) - before, 2);
    Ok(())
}

#[derive(Default)]
pub(super) struct FaultFileSystem {
    pub(super) control: Arc<FaultControl>,
}

impl FaultFileSystem {
    fn wrap(&self, inner: FileHandle, path: &Path) -> FaultHandle {
        FaultHandle {
            inner: Arc::new(inner),
            control: self.control.clone(),
            rewrite: path
                .extension()
                .is_some_and(|extension| extension == "rewrite"),
        }
    }
}

impl FileSystem for FaultFileSystem {
    type Handle = FaultHandle;
    type Reader = <DefaultFileSystem as FileSystem>::Reader;
    type Writer = <DefaultFileSystem as FileSystem>::Writer;

    fn create<P: AsRef<Path>>(&self, path: P) -> io::Result<Self::Handle> {
        Ok(self.wrap(DefaultFileSystem.create(path.as_ref())?, path.as_ref()))
    }

    fn open<P: AsRef<Path>>(&self, path: P, permission: Permission) -> io::Result<Self::Handle> {
        Ok(self.wrap(
            DefaultFileSystem.open(path.as_ref(), permission)?,
            path.as_ref(),
        ))
    }

    fn delete<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        DefaultFileSystem.delete(path)
    }

    fn rename<P: AsRef<Path>>(&self, source: P, destination: P) -> io::Result<()> {
        DefaultFileSystem.rename(source, destination)
    }

    fn new_reader(&self, handle: Arc<Self::Handle>) -> io::Result<Self::Reader> {
        if self
            .control
            .pause_payload_read
            .swap(false, Ordering::AcqRel)
        {
            if let Some(entered) = self.control.entered.lock().unwrap().take() {
                let _ = entered.send(());
            }
            if let Some(release) = self.control.release.lock().unwrap().take() {
                // The RAII guard releases this even when the test unwinds.
                let _ = release.recv();
            }
        }
        DefaultFileSystem.new_reader(handle.inner.clone())
    }

    fn new_writer(&self, handle: Arc<Self::Handle>) -> io::Result<Self::Writer> {
        DefaultFileSystem.new_writer(handle.inner.clone())
    }
}

#[test]
fn sync_panic_cannot_ack_another_group() -> Result<()> {
    sync_failure_cannot_ack_another_group(64 * 1024 * 1024)
}

#[test]
fn retained_payload_read_does_not_wait_for_another_groups_sync() -> Result<()> {
    let layout = Layout {
        cluster: uuid::Uuid::from_u128(1201),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
        buckets: vec![10, 20],
    };
    let directory = tempfile::tempdir()?;
    let log = PhysicalLog::open(&directory.path().join("log"), &layout, 1)?;
    let retained = Entry {
        index: 1,
        term: 1,
        data: vec![42; 4096].into(),
        ..Default::default()
    };
    let hard = HardState {
        term: 1,
        vote: 1,
        commit: 1,
        ..Default::default()
    };
    let sibling = log.group(20)?;
    sibling.persist(std::slice::from_ref(&retained), Some(&hard))?;
    let writer = log.group(10)?;
    let control = writer.node.fault.clone();
    let (entered, wait_entered) = mpsc::sync_channel(1);
    let (release, wait_release) = mpsc::channel();
    *control.entered.lock().unwrap() = Some(entered);
    *control.release.lock().unwrap() = Some(wait_release);
    control.fail_once.store(true, Ordering::Release);
    let input = retained.clone();
    let write = std::thread::spawn(move || writer.persist(&[input], Some(&hard)));
    wait_entered.recv_timeout(Duration::from_secs(10))?;

    // The real filesystem sync is held open. A read of previously durable
    // sibling data must finish before that sync is released or fails.
    let (completed, wait_completed) = mpsc::sync_channel(1);
    let read = std::thread::spawn(move || {
        completed
            .send(sibling.entries(1, 2, None, GetEntriesContext::empty(false)))
            .unwrap();
    });
    let observed = wait_completed.recv_timeout(Duration::from_secs(2));
    release.send(())?;
    assert!(write.join().unwrap().is_err());
    read.join().unwrap();
    assert_eq!(
        observed.context("payload read waited on an unrelated sync")??,
        vec![retained]
    );
    assert!(log.group(20)?.initial_state().is_err());
    Ok(())
}

#[test]
fn rotation_sync_panic_cannot_ack_another_group() -> Result<()> {
    // Rotation also calls Handle::sync, even for an engine write(sync=false).
    // Force rotation at every append to exercise that different panic boundary.
    sync_failure_cannot_ack_another_group(1)
}

#[test]
fn snapshot_install_sync_failure_recovers_all_old_or_all_new_state() -> Result<()> {
    let layout = Layout {
        cluster: uuid::Uuid::from_u128(24),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
        buckets: vec![10, 20],
    };
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("log");
    let log = PhysicalLog::open(&path, &layout, 1)?;
    let group = log.group(10)?;
    group.bind_application(b"snapshot-install-fault/v1")?;
    let entry = |index, term| Entry {
        index,
        term,
        data: vec![(index + term) as u8; 4096].into(),
        ..Default::default()
    };
    let old: Vec<_> = (1..=5).map(|i| entry(i, 2)).collect();
    let old_hard = HardState {
        term: 2,
        vote: 1,
        commit: 2,
        ..Default::default()
    };
    group.persist(&old, Some(&old_hard))?;
    let sibling = log.group(20)?;
    let sibling_hard = HardState {
        term: 7,
        vote: 2,
        commit: 1,
        ..Default::default()
    };
    sibling.persist(&[entry(1, 7)], Some(&sibling_hard))?;
    let mut snapshot = Snapshot::default();
    snapshot.mut_metadata().index = 4;
    snapshot.mut_metadata().term = 3;
    snapshot
        .mut_metadata()
        .set_conf_state(group.initial_state()?.conf_state);
    snapshot.data = b"snapshot-install-fault/descriptor".to_vec().into();
    let suffix = vec![entry(5, 3), entry(6, 3)];
    let new_hard = HardState {
        term: 3,
        vote: 2,
        commit: 5,
        ..Default::default()
    };
    group.node.fault.fail_once.store(true, Ordering::Release);
    assert!(group
        .install_snapshot(&snapshot, &suffix, &new_hard)
        .is_err());
    assert!(group.initial_state().is_err());
    assert!(group.snapshot(0, 2).is_err());
    assert!(sibling.persist(&[], None).is_err());
    assert!(sibling.initial_state().is_err());
    drop((group, sibling, log));
    let log = PhysicalLog::open(&path, &layout, 1)?;
    let group = log.group(10)?;
    group.bind_application(b"snapshot-install-fault/v1")?;
    match group.first_index()? {
        1 => {
            assert_eq!(group.initial_state()?.hard_state, old_hard);
            assert_eq!(group.last_index()?, 5);
            assert_eq!(
                group.entries(1, 6, None, GetEntriesContext::empty(false))?,
                old
            );
            assert!(group.snapshot(0, 2).is_err());
        }
        5 => {
            assert_eq!(group.initial_state()?.hard_state, new_hard);
            assert_eq!(group.last_index()?, 6);
            assert_eq!(group.term(4)?, 3);
            assert_eq!(group.snapshot(4, 2)?, snapshot);
            assert_eq!(
                group.entries(5, 7, None, GetEntriesContext::empty(false))?,
                suffix
            );
        }
        index => panic!("partial snapshot install boundary {index}"),
    }
    let sibling = log.group(20)?;
    assert_eq!(sibling.initial_state()?.hard_state, sibling_hard);
    assert_eq!(
        sibling.entries(1, 2, None, GetEntriesContext::empty(false))?,
        vec![entry(1, 7)]
    );
    Ok(())
}

#[test]
fn checkpoint_sync_failure_fences_all_handles_and_recovers_one_complete_cut() -> Result<()> {
    let layout = Layout {
        cluster: uuid::Uuid::from_u128(3),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
        buckets: vec![10, 20],
    };
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("log");
    let log = PhysicalLog::open(&path, &layout, 1)?;
    let group = log.group(10)?;
    group.bind_application(b"checkpoint-fault/v1")?;
    let entries: Vec<_> = (1..=3)
        .map(|index| Entry {
            index,
            term: 1,
            data: vec![index as u8; 4096].into(),
            ..Default::default()
        })
        .collect();
    group.persist(
        &entries,
        Some(&HardState {
            term: 1,
            vote: 1,
            commit: 3,
            ..Default::default()
        }),
    )?;
    let mut snapshot = Snapshot::default();
    snapshot.mut_metadata().index = 2;
    snapshot.mut_metadata().term = 1;
    snapshot
        .mut_metadata()
        .set_conf_state(group.initial_state()?.conf_state);
    snapshot.data = vec![2; 32].into();
    group.node.fault.fail_sync.store(true, Ordering::Release);
    assert!(group.checkpoint(&snapshot).is_err());
    assert!(group.snapshot(0, 2).is_err());
    assert!(group.first_index().is_err());
    assert!(log.group(20)?.persist(&[], None).is_err());
    group.node.fault.fail_sync.store(false, Ordering::Release);
    drop((group, log));
    let log = PhysicalLog::open(&path, &layout, 1)?;
    let group = log.group(10)?;
    assert_eq!(group.initial_state()?.hard_state.commit, 3);
    match group.first_index()? {
        1 => assert_eq!(
            group.entries(1, 4, None, GetEntriesContext::empty(false))?,
            entries
        ),
        3 => {
            assert_eq!(group.snapshot(2, 2)?, snapshot);
            assert_eq!(group.term(2)?, 1);
            assert_eq!(
                group.entries(3, 4, None, GetEntriesContext::empty(false))?,
                entries[2..]
            );
        }
        index => panic!("partial checkpoint boundary {index}"),
    }
    assert_eq!(
        log.group(20)?.initial_state()?.hard_state,
        HardState::default()
    );
    Ok(())
}

fn sync_failure_cannot_ack_another_group(file_size: u64) -> Result<()> {
    let layout = Layout {
        cluster: uuid::Uuid::from_u128(1),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
        buckets: vec![10, 20],
    };
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("log");
    let log = PhysicalLog::open_with_file_size(&path, &layout, 1, file_size)?;
    let first = log.group(10)?;
    let second = log.group(20)?;
    let prefix = Entry {
        index: 1,
        term: 1,
        data: vec![7; 4096].into(),
        ..Default::default()
    };
    first.persist(
        std::slice::from_ref(&prefix),
        Some(&HardState {
            term: 1,
            vote: 1,
            commit: 1,
            ..Default::default()
        }),
    )?;

    let control = first.node.fault.clone();
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    *control.entered.lock().unwrap() = Some(entered_tx);
    *control.release.lock().unwrap() = Some(release_rx);
    control.fail_sync.store(true, Ordering::Release);
    let writer = first.clone();
    let a = std::thread::spawn(move || {
        writer.persist(
            &[Entry {
                index: 2,
                term: 1,
                data: vec![8; 4096].into(),
                ..Default::default()
            }],
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: 2,
                ..Default::default()
            }),
        )
    });
    entered_rx.recv_timeout(Duration::from_secs(10))?;
    let (finished_tx, finished_rx) = mpsc::sync_channel(1);
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let writer = second.clone();
    let b = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let result = writer.persist(
            &[],
            Some(&HardState {
                term: 1,
                vote: 1,
                commit: 0,
                ..Default::default()
            }),
        );
        finished_tx.send(result).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(10))?;
    assert!(finished_rx.recv_timeout(Duration::from_millis(50)).is_err());
    release_tx.send(())?;
    assert!(
        a.join().unwrap().is_err(),
        "sync panic must not become a successful persist"
    );
    assert!(finished_rx.recv_timeout(Duration::from_secs(10))?.is_err());
    b.join().unwrap();
    assert!(first.initial_state().is_err());
    assert!(second.initial_state().is_err());
    assert!(first.persist(&[], None).is_err());
    control.fail_sync.store(false, Ordering::Release);
    drop((first, second, log));
    let reopened = PhysicalLog::open(&path, &layout, 1)?;
    let group = reopened.group(10)?;
    assert!(group.initial_state()?.hard_state.commit >= 1);
    assert_eq!(
        group.entries(1, 2, None, GetEntriesContext::empty(false))?,
        vec![prefix]
    );
    assert_eq!(reopened.group(20)?.initial_state()?.hard_state.term, 0);
    Ok(())
}
