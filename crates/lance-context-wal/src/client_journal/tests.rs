use super::*;
use raft_engine::env::{DefaultFileSystem, Handle, Permission};
use std::io;
use std::sync::{mpsc as blocking, Mutex};
mod pressure_faults;
mod rewrite;
mod skew;

fn settings() -> Settings {
    Settings {
        linger_ms: 2,
        batch_bytes: 1 << 20,
        queue_bytes: 2 << 20,
        state_bytes: 4 << 20,
        file_bytes: 1 << 20,
        log_bytes: 32 << 20,
    }
}

#[test]
fn disk_admission_keeps_room_for_live_state_and_rewrite() {
    let mut config = settings();
    config.log_bytes = config.reserve() + 2 * config.batch_bytes as u64;
    assert!(config.validate().is_err());
    config.log_bytes += config.state_bytes as u64;
    assert!(config.validate().is_ok());
}

#[test]
fn relative_directory_reopens_and_missing_parent_does_not_create_ancestors() -> Result<()> {
    // Avoid changing the process-wide current directory while tests run.
    let dir = tempfile::Builder::new()
        .prefix(".client-journal-relative-")
        .tempdir_in(".")?;
    let relative = Path::new(dir.path().file_name().unwrap());
    let (journal, _) = Journal::open(relative, b"relative", 1, settings())?;
    drop(journal);
    let (journal, _) = Journal::open(relative, b"relative", 1, settings())?;
    drop(journal);

    let root = tempfile::tempdir()?;
    let missing = root.path().join("missing");
    assert!(Journal::open(&missing.join("journal"), b"scope", 1, settings()).is_err());
    assert!(!missing.exists());
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn owner_close_releases_lock_while_a_duplicated_description_remains_open() -> Result<()> {
    use std::os::{fd::BorrowedFd, unix::fs::MetadataExt};
    let dir = tempfile::tempdir()?;
    let (journal, _) = Journal::open(dir.path(), b"inherited-lock", 1, settings())?;
    let lock = std::fs::metadata(dir.path().join("LOCK"))?;
    let mut inherited = None;
    for entry in std::fs::read_dir("/proc/self/fd")? {
        let entry = entry?;
        let Ok(metadata) = std::fs::metadata(entry.path()) else {
            continue;
        };
        if metadata.dev() == lock.dev() && metadata.ino() == lock.ino() {
            let fd: i32 = entry.file_name().to_str().unwrap().parse()?;
            // SAFETY: this unique journal owns the identified lock descriptor
            // until drop(journal) below; its worker never closes that lock. The
            // borrow only duplicates it and does not assume ownership of it.
            inherited = Some(unsafe { BorrowedFd::borrow_raw(fd) }.try_clone_to_owned()?);
            break;
        }
    }
    let inherited = inherited.context("journal lock descriptor missing")?;
    assert!(Journal::open(dir.path(), b"inherited-lock", 1, settings()).is_err());
    // dup and fork share an open file description. Keep it alive deterministically
    // instead of hoping to hit the short fork-to-exec window of another test.
    drop(journal);
    let (replacement, _) = Journal::open(dir.path(), b"inherited-lock", 1, settings())?;
    drop(inherited);
    assert!(Journal::open(dir.path(), b"inherited-lock", 1, settings()).is_err());
    drop(replacement);
    assert!(Journal::open(dir.path(), b"inherited-lock", 1, settings()).is_ok());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_replacements_reclaim_and_reopen_exact_latest_values() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let config = settings();
    let (journal, recovered) = Journal::open(dir.path(), b"scope", 4, config.clone())?;
    assert!(recovered.iter().all(Option::is_none));
    let mut jobs = tokio::task::JoinSet::new();
    for key in 0..4 {
        let writer = journal.clone();
        jobs.spawn(async move {
            for round in 0..128u8 {
                writer
                    .put(
                        key,
                        Bytes::from(vec![key as u8, round]),
                        Bytes::from(vec![round; 65536]),
                    )
                    .await?;
            }
            Ok::<_, anyhow::Error>(())
        });
    }
    while let Some(result) = jobs.join_next().await {
        result??;
    }
    let stats = journal.statistics();
    assert_eq!(stats.requests, 512);
    assert!(stats.groups < 512 && stats.max_group_requests > 1);
    assert!(stats.maintenance_calls > 0 && !stats.fenced);
    assert!(footprint(dir.path())? < config.log_bytes - config.reserve());
    drop(journal);
    let (journal, values) = Journal::open(dir.path(), b"scope", 4, config.clone())?;
    for (key, value) in values.into_iter().enumerate() {
        let value = value.unwrap();
        assert_eq!(&value[..2], &[key as u8, 127]);
        assert_eq!(&value[2..], vec![127; 65536]);
    }
    assert!(Journal::open(dir.path(), b"scope", 4, config.clone()).is_err());
    drop(journal);
    assert!(Journal::open(dir.path(), b"other", 4, config.clone()).is_err());
    assert!(Journal::open(dir.path(), b"scope", 3, config).is_err());
    Ok(())
}

type RealHandle = <DefaultFileSystem as FileSystem>::Handle;
#[derive(Default)]
struct Control {
    fail: AtomicBool,
    fail_rewrite: AtomicBool,
    rewrite_failures: AtomicU64,
    rewrite_skip_syncs: AtomicU64,
    rewrite_sync_calls: AtomicU64,
    calls: AtomicU64,
    entered: Mutex<Option<blocking::SyncSender<()>>>,
    release: Mutex<Option<blocking::Receiver<()>>>,
    rewrite_pause: Mutex<Option<(std::path::PathBuf, bool)>>,
}
struct FaultHandle {
    inner: Arc<RealHandle>,
    control: Arc<Control>,
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
        if self.rewrite {
            let call = self
                .control
                .rewrite_sync_calls
                .fetch_add(1, Ordering::Relaxed)
                + 1;
            let mut skip = self.control.rewrite_skip_syncs.load(Ordering::Acquire);
            while skip > 0 {
                match self.control.rewrite_skip_syncs.compare_exchange_weak(
                    skip,
                    skip - 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return self.inner.sync(),
                    Err(actual) => skip = actual,
                }
            }
            let pause = self.control.rewrite_pause.lock().unwrap().take();
            if let Some((marker, after_sync)) = pause {
                if after_sync {
                    self.inner.sync()?;
                }
                // The parent treats existence as the completed pause receipt.
                // Publish atomically so it cannot observe an empty/partial JSON
                // file between create/truncate and write under coverage load.
                let pending_marker = marker.with_extension("pending");
                std::fs::write(
                    &pending_marker,
                    serde_json::to_vec(
                        &serde_json::json!({"rewrite_sync":call,"file_bytes":self.inner.file_size()?}),
                    )?,
                )?;
                std::fs::rename(pending_marker, marker)?;
                loop {
                    std::thread::park();
                }
            }
            if self.control.fail_rewrite.load(Ordering::Acquire) {
                self.control
                    .rewrite_failures
                    .fetch_add(1, Ordering::Relaxed);
                return Err(io::Error::other("injected pressure rewrite sync failure"));
            }
        }
        self.control.calls.fetch_add(1, Ordering::Relaxed);
        if let Some(tx) = self.control.entered.lock().unwrap().take() {
            tx.send(()).unwrap();
        }
        if let Some(rx) = self.control.release.lock().unwrap().take() {
            rx.recv_timeout(Duration::from_secs(10)).unwrap();
        }
        if self.control.fail.load(Ordering::Acquire) {
            return Err(io::Error::other("injected client sync failure"));
        }
        self.inner.sync()
    }
}
#[derive(Default)]
struct FaultFs {
    control: Arc<Control>,
}
impl FileSystem for FaultFs {
    type Handle = FaultHandle;
    type Reader = <DefaultFileSystem as FileSystem>::Reader;
    type Writer = <DefaultFileSystem as FileSystem>::Writer;
    fn create<P: AsRef<Path>>(&self, path: P) -> io::Result<Self::Handle> {
        let rewrite = path.as_ref().extension().is_some_and(|e| e == "rewrite");
        Ok(FaultHandle {
            inner: Arc::new(DefaultFileSystem.create(path)?),
            control: self.control.clone(),
            rewrite,
        })
    }
    fn open<P: AsRef<Path>>(&self, path: P, permission: Permission) -> io::Result<Self::Handle> {
        let rewrite = path.as_ref().extension().is_some_and(|e| e == "rewrite");
        Ok(FaultHandle {
            inner: Arc::new(DefaultFileSystem.open(path, permission)?),
            control: self.control.clone(),
            rewrite,
        })
    }
    fn delete<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        DefaultFileSystem.delete(path)
    }
    fn rename<P: AsRef<Path>>(&self, src: P, dst: P) -> io::Result<()> {
        DefaultFileSystem.rename(src, dst)
    }
    fn new_reader(&self, h: Arc<Self::Handle>) -> io::Result<Self::Reader> {
        DefaultFileSystem.new_reader(h.inner.clone())
    }
    fn new_writer(&self, h: Arc<Self::Handle>) -> io::Result<Self::Writer> {
        DefaultFileSystem.new_writer(h.inner.clone())
    }
}
type Controlled = (
    Journal,
    Arc<Control>,
    blocking::Receiver<()>,
    blocking::SyncSender<()>,
);
fn controlled(dir: &Path) -> Result<Controlled> {
    let fs = Arc::new(FaultFs::default());
    let control = fs.control.clone();
    let config = Settings {
        linger_ms: 25,
        ..settings()
    };
    let engine = Engine::open_with(config.engine(dir), fs, vec![])?;
    let (journal, _) = Journal::with_engine(engine, b"scope", 2, config)?;
    let (entered, rx) = blocking::sync_channel(1);
    let (tx, release) = blocking::sync_channel(1);
    *control.entered.lock().unwrap() = Some(entered);
    *control.release.lock().unwrap() = Some(release);
    Ok((journal, control, rx, tx))
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_waiter_completes_before_sync_and_failure_fences_all_handles() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (journal, control, entered, release) = controlled(dir.path())?;
    control.fail.store(true, Ordering::Release);
    let mut jobs = tokio::task::JoinSet::new();
    for key in 0..2 {
        let j = journal.clone();
        jobs.spawn(async move {
            j.put(
                key,
                Bytes::from_static(b"intent"),
                Bytes::from(vec![7; 4096]),
            )
            .await
        });
    }
    tokio::task::spawn_blocking(move || entered.recv_timeout(Duration::from_secs(10))).await??;
    assert!(
        jobs.try_join_next().is_none(),
        "success before actual filesystem sync"
    );
    release.send(())?;
    while let Some(result) = jobs.join_next().await {
        assert!(result?.is_err());
    }
    let calls = control.calls.load(Ordering::Relaxed);
    assert!(journal.statistics().fenced);
    assert!(journal
        .put(0, Bytes::new(), Bytes::from_static(b"next"))
        .await
        .is_err());
    assert_eq!(control.calls.load(Ordering::Relaxed), calls);
    assert_eq!(journal.statistics().requests, 0);
    // Pinned engine destruction also syncs. Clear only the test filesystem
    // injection before dropping the permanently fenced journal, never reuse it.
    control.fail.store(false, Ordering::Release);
    drop(journal);
    Ok(())
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancellation_keeps_the_original_durable_outcome_recoverable() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (journal, _control, entered, release) = controlled(dir.path())?;
    let j = journal.clone();
    let task = tokio::spawn(async move {
        j.put(
            0,
            Bytes::from_static(b"intent"),
            Bytes::from_static(b"exact-body"),
        )
        .await
    });
    tokio::task::spawn_blocking(move || entered.recv_timeout(Duration::from_secs(10))).await??;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    release.send(())?;
    journal
        .put(1, Bytes::from_static(b"barrier"), Bytes::new())
        .await?;
    drop(journal);
    let (_reopened, values) = Journal::open(
        dir.path(),
        b"scope",
        2,
        Settings {
            linger_ms: 25,
            ..settings()
        },
    )?;
    assert_eq!(values[0].as_deref(), Some(b"intentexact-body".as_slice()));
    Ok(())
}

#[tokio::test]
async fn state_admission_and_oversized_values_do_not_acknowledge() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let config = Settings {
        state_bytes: 1024,
        ..settings()
    };
    let (journal, _) = Journal::open(dir.path(), b"scope", 2, config)?;
    assert!(journal
        .put(0, Bytes::new(), Bytes::from(vec![3; 1 << 20]))
        .await
        .is_err());
    assert!(journal.put(2, Bytes::new(), Bytes::new()).await.is_err());
    journal
        .put(0, Bytes::new(), Bytes::from(vec![1; 800]))
        .await?;
    assert!(journal
        .put(1, Bytes::new(), Bytes::from(vec![2; 800]))
        .await
        .is_err());
    assert!(journal.statistics().fenced);
    assert_eq!(journal.statistics().requests, 1);
    Ok(())
}
