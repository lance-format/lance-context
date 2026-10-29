use super::*;
use crate::annotations::Completed;
use std::path::{Path as FsPath, PathBuf};
use tokio::sync::oneshot;

mod fault_store;
mod listing;
mod manifest_cache;
mod protected_page;
use fault_store::{Action, Faults};

fn limits() -> Limits {
    Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    }
}

fn root() -> Result<tempfile::TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("annotation-delete-fault-");
    Ok(if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        // Preserve WAL, blob files and original request evidence when requested.
        builder.disable_cleanup(true).tempdir_in(parent)?
    } else {
        builder.tempdir()?
    })
}

async fn acknowledged_orphans(
    c: &network::Cluster,
    storage: &Storage,
    path: &FsPath,
) -> Result<(PreparedAppend, Completed, Vec<PathBuf>)> {
    let limits = limits();
    annotation_publication::seed_contract(c, storage, Service::contract(&binding(), &limits)?)
        .await?;
    let original = files(path)?;
    let mut first = None;
    for _ in 0..2 {
        let (service, driver) = connect(c, storage, &limits).await?;
        let (stop, task) = run(driver);
        let producer = producer(&service, 1).await?;
        let request = request(
            &service,
            &producer,
            1,
            wire(&limits, &["T"], 1, Some("acknowledged"), 0, None)?,
        )
        .await?;
        let completed = service.submit(&request, Duration::from_secs(30)).await?;
        if let Some((_, expected)) = &first {
            assert_eq!(&completed, expected);
        }
        let frozen = service.freeze(Uuid::new_v4()).await?.unwrap();
        let prepared = frozen
            .prepare(&MergeMemoryBudget::new(8 << 20), 8192)
            .await?;
        stop.send(()).unwrap();
        task.await??;
        drop((prepared, frozen, service));
        first = Some((request, completed));
    }
    let abandoned: Vec<_> = files(path)?.difference(&original).cloned().collect();
    assert_eq!(abandoned.len(), 2, "two real abandoned native preparations");
    let (request, completed) = first.unwrap();
    Ok((request, completed, abandoned))
}

fn whole_page(runtime: Runtime) -> Result<Runtime> {
    runtime.with_exclusive_reclamation(crate::annotations::runtime::reclamation::Options {
        max_objects: 16,
        max_bytes: 16 << 10,
        delete_parallelism: 2,
        interval: Duration::from_millis(10),
    })
}

async fn cold_check(c: &network::Cluster, storage: &Storage, score: Option<f32>) -> Result<()> {
    let base = annotation_publication::read(c, storage, None).await?;
    assert!(base.read_transaction().await?.is_some());
    let batches = base
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
    let mut found = false;
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let bodies = batch
            .column_by_name("body")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        let statuses = batch
            .column_by_name("status")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = batch
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let body = bodies.value(row);
            assert_eq!(body.len(), 256 << 10);
            assert!(body
                .iter()
                .enumerate()
                .all(|(i, b)| *b == ((i * 31 + i / 17) % 251) as u8));
            if ids.value(row) == "T" {
                found = true;
                assert_eq!(statuses.value(row), "acknowledged");
                assert_eq!(scores.iter().nth(row).unwrap(), score);
            }
        }
    }
    assert!(found);
    Ok(())
}

async fn wait_files(path: &FsPath, count: usize) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if files(path)?.len() == count {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await??;
    Ok(())
}

#[tokio::test]
async fn annotation_collector_slow_pass_leaves_a_full_publication_cooldown() -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage(&path, faults.clone()).await?;
    let (first, completed, abandoned) = acknowledged_orphans(&c, &storage, &path).await?;
    let (entered, reached) = oneshot::channel();
    let (resume, resumed) = oneshot::channel();
    faults.arm(
        &abandoned[0],
        Action::Pause {
            entered,
            resume: resumed,
        },
    )?;
    let interval = Duration::from_millis(500);
    let (service, runtime) = owned(&c, &storage, &limits()).await?;
    let runtime =
        runtime.with_exclusive_reclamation(crate::annotations::runtime::reclamation::Options {
            interval,
            max_objects: 64,
            max_bytes: 16 << 10,
            delete_parallelism: 2,
        })?;
    faults.collection_starts.lock().unwrap().clear();
    let (stop, task) = run_owner(runtime);
    tokio::time::timeout(Duration::from_secs(30), reached).await??;
    assert_eq!(
        service.submit(&first, Duration::from_secs(30)).await?,
        completed
    );
    let producer = producer(&service, 2).await?;
    let suffix = request(
        &service,
        &producer,
        1,
        wire(&limits(), &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let suffix_result = service.submit(&suffix, Duration::from_secs(30)).await?;
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("acknowledged"));
    assert_eq!(current.score, Some(0.9));
    // A real sealed DELETE keeps the publisher occupied longer than its interval.
    // The next pass must leave a full interval *after* that I/O finishes.
    tokio::time::sleep(interval + Duration::from_millis(50)).await;
    assert_eq!(faults.collection_starts.lock().unwrap().len(), 1);
    let released = tokio::time::Instant::now();
    resume.send(()).unwrap();
    faults.finish_remote().await?;
    drained(&c, &service, suffix_result.index).await?;
    tokio::time::timeout(Duration::from_secs(30), async {
        while faults.collection_starts.lock().unwrap().len() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let next_started = faults.collection_starts.lock().unwrap()[1];
    // Finish cleanup before asserting the regression, so failure preserves a
    // complete protocol shutdown and does not leave background tasks running.
    wait_files(&path, 3).await?;
    cold_check(&c, &storage, Some(0.9)).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    assert!(
        next_started >= released + interval,
        "collection restarted {:?} after releasing slow I/O; expected at least {interval:?}",
        next_started.duration_since(released)
    );
    Ok(())
}

#[tokio::test]
async fn annotation_collector_delayed_delete_survives_owner_takeover() -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage(&path, faults.clone()).await?;
    let (first, completed, abandoned) = acknowledged_orphans(&c, &storage, &path).await?;
    let (entered, reached) = oneshot::channel();
    let (resume, resumed) = oneshot::channel();
    faults.arm(
        &abandoned[0],
        Action::Pause {
            entered,
            resume: resumed,
        },
    )?;
    let (old, runtime) = owned(&c, &storage, &limits()).await?;
    let (_old_stop, old_task) = run_owner(whole_page(runtime)?);
    tokio::time::timeout(Duration::from_secs(30), reached).await??;
    assert!(
        abandoned[0].exists(),
        "DELETE entered but has not removed the object"
    );

    // Acquire/open a new owner while the previous owner's sealed DELETE is in
    // flight. Use a fresh Session so old caches cannot conceal deleted files.
    let fresh = fault_store::storage(&path, Arc::new(Faults::default())).await?;
    let (service, runtime) = owner(&c, &fresh, &limits()).await?;
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service.submit(&first, Duration::from_secs(30)).await?,
        completed
    );
    let producer = producer(&service, 2).await?;
    let suffix = request(
        &service,
        &producer,
        1,
        wire(&limits(), &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let suffix_result = service.submit(&suffix, Duration::from_secs(30)).await?;
    drained(&c, &service, suffix_result.index).await?;
    let live = files(&path)?;
    resume.send(()).unwrap();
    faults.finish_remote().await?;
    let old_error = tokio::time::timeout(Duration::from_secs(30), old_task)
        .await??
        .unwrap_err();
    let reason = format!("{old_error:#}");
    assert!(
        reason.contains("fenced") || reason.contains("view is behind source closure"),
        "{reason}"
    );
    assert!(!abandoned[0].exists());
    for file in live.difference(&abandoned.iter().cloned().collect()) {
        assert!(
            file.exists(),
            "old owner's DELETE removed a successor's file"
        );
    }
    assert_eq!(
        service.submit(&first, Duration::from_secs(30)).await?,
        completed
    );
    assert_eq!(
        service.submit(&suffix, Duration::from_secs(30)).await?,
        suffix_result
    );
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("acknowledged"));
    assert_eq!(current.score, Some(0.9));
    cold_check(&c, &fresh, Some(0.9)).await?;
    stop.send(()).unwrap();
    task.await??;
    drop((old, service));
    c.shutdown().await
}

#[tokio::test]
async fn annotation_collector_partial_delete_restarts_without_losing_values_or_results(
) -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage(&path, faults.clone()).await?;
    let (first, completed, abandoned) = acknowledged_orphans(&c, &storage, &path).await?;
    faults.arm(&abandoned[0], Action::LoseReply)?;
    faults.arm(&abandoned[1], Action::FailBefore)?;
    let (service, runtime) = owned(&c, &storage, &limits()).await?;
    let runtime =
        runtime.with_exclusive_reclamation(crate::annotations::runtime::reclamation::Options {
            max_objects: 16,
            max_bytes: 16 << 10,
            delete_parallelism: 2,
            interval: Duration::from_secs(3600),
        })?;
    let (stop, task) = run_owner(runtime);
    tokio::time::timeout(Duration::from_secs(30), async {
        while service.reclamation_observation().pending_objects != 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    stop.send(()).unwrap();
    task.await??;
    assert!(
        !abandoned[0].exists(),
        "first actual DELETE succeeded but reply was lost"
    );
    assert!(
        abandoned[1].exists(),
        "second actual DELETE failed before removal"
    );
    assert_eq!(
        faults.calls.lock().unwrap().len(),
        2,
        "shutdown precedes the next collection interval"
    );
    // Success counters cannot be used as physical inventory after a lost reply.
    assert_eq!(service.reclamation_observation().deleted, 0);
    drop((service, storage));
    c.shutdown().await?;
    c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let fresh = fault_store::storage(&path, Arc::new(Faults::default())).await?;
    let (service, runtime) = owned(&c, &fresh, &limits()).await?;
    let (stop, task) = run_owner(runtime);
    assert_eq!(
        service.submit(&first, Duration::from_secs(30)).await?,
        completed
    );
    drained(&c, &service, completed.index).await?;
    wait_files(&path, 3).await?;
    assert!(abandoned.iter().all(|p| !p.exists()));
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("acknowledged"));
    assert_eq!(current.score, None);
    cold_check(&c, &fresh, None).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[cfg(unix)]
#[tokio::test]
#[ignore = "subprocess role, launched by annotation_collector_recovers_after_delete_sigkill"]
async fn annotation_delete_process_role() -> Result<()> {
    let root = PathBuf::from(std::env::var("LANCE_CONTEXT_TEST_DELETE_PROCESS_ROOT")?);
    let path = root.join("turns.lance");
    let c = network::Cluster::open_signed(&root, annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage(&path, faults.clone()).await?;
    let (request, completed, abandoned) = acknowledged_orphans(&c, &storage, &path).await?;
    std::fs::write(root.join("request.arrow"), request.body())?;
    std::fs::write(root.join("ack.json"), serde_json::to_vec(&completed)?)?;
    std::fs::write(root.join("abandoned.json"), serde_json::to_vec(&abandoned)?)?;
    // Expose one orphan, then hold the listing across the collector's time
    // slice until SIGKILL. A one-shot failure on the other orphan is not a
    // barrier: if listing splits the files into separate pages, it can be
    // successfully retried before the crash marker is reached.
    let (listed, _listing_entered) = oneshot::channel();
    let (_release_listing, resume) = oneshot::channel();
    faults.pause_listing(&path, Some(&abandoned[0]), listed, resume)?;
    faults.arm(
        &abandoned[0],
        Action::CrashAfter(root.join("delete-entered")),
    )?;
    let (_, runtime) = owned(&c, &storage, &limits()).await?;
    whole_page(runtime)?.run(std::future::pending()).await?;
    bail!("delete role stopped before SIGKILL")
}

#[tokio::test]
async fn annotation_collector_corrupt_delete_reply_stops_without_reissuing() -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage(&path, faults.clone()).await?;
    let (_, _, abandoned) = acknowledged_orphans(&c, &storage, &path).await?;
    // A time-sliced listing can place the two orphans on separate pages. Pin
    // this page to the corrupted response's object; later objects must remain
    // untouched once the runtime rejects that response.
    let (listed, _listing_entered) = oneshot::channel();
    let (_release_listing, resume) = oneshot::channel();
    faults.pause_listing(&path, Some(&abandoned[0]), listed, resume)?;
    faults.arm(&abandoned[0], Action::WrongReply)?;
    let (_, runtime) = owned(&c, &storage, &limits()).await?;
    let error = tokio::time::timeout(
        Duration::from_secs(30),
        whole_page(runtime)?.run(std::future::pending()),
    )
    .await?
    .unwrap_err();
    assert!(format!("{error:#}").contains("invalid sealed annotation deletion response"));
    assert_eq!(
        faults.calls.lock().unwrap().as_slice(),
        &[object_store::path::Path::from_absolute_path(&abandoned[0])?]
    );
    assert!(!abandoned[0].exists());
    assert!(abandoned[1].exists());
    c.shutdown().await
}

#[cfg(unix)]
#[tokio::test]
async fn annotation_collector_recovers_after_delete_sigkill() -> Result<()> {
    use std::os::unix::process::ExitStatusExt;
    struct Child(std::process::Child);
    impl Drop for Child {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let root = root()?;
    let log = std::fs::File::create(root.path().join("child.log"))?;
    let mut child = Child(std::process::Command::new(std::env::current_exe()?)
        .args(["--exact", "tests::annotation_runtime::automatic_reclamation::deletion_faults::annotation_delete_process_role", "--ignored", "--nocapture"])
        .env("LANCE_CONTEXT_TEST_DELETE_PROCESS_ROOT", root.path())
        .stdout(log.try_clone()?).stderr(log).spawn()?);
    tokio::time::timeout(Duration::from_secs(60), async {
        while !root.path().join("delete-entered").exists() {
            if let Some(status) = child.0.try_wait()? {
                bail!(
                    "delete child exited {status}: {}",
                    std::fs::read_to_string(root.path().join("child.log"))?
                );
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    child.0.kill()?;
    assert_eq!(child.0.wait()?.signal(), Some(9));
    let expected: Completed =
        serde_json::from_slice(&std::fs::read(root.path().join("ack.json"))?)?;
    let abandoned: Vec<PathBuf> =
        serde_json::from_slice(&std::fs::read(root.path().join("abandoned.json"))?)?;
    assert_eq!(
        std::fs::read_to_string(root.path().join("delete-entered"))?,
        object_store::path::Path::from_absolute_path(&abandoned[0])?.as_ref()
    );
    assert!(!abandoned[0].exists());
    assert!(abandoned[1].exists());
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let path = root.path().join("turns.lance");
    let storage = fault_store::storage(&path, Arc::new(Faults::default())).await?;
    let (service, runtime) = owned(&c, &storage, &limits()).await?;
    let (stop, task) = run_owner(runtime);
    let producer = producer(&service, 1).await?;
    let request = request(
        &service,
        &producer,
        1,
        Bytes::from(std::fs::read(root.path().join("request.arrow"))?),
    )
    .await?;
    assert_eq!(
        service.submit(&request, Duration::from_secs(30)).await?,
        expected
    );
    drained(&c, &service, expected.index).await?;
    wait_files(&path, 3).await?;
    assert!(abandoned.iter().all(|p| !p.exists()));
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("acknowledged"));
    assert_eq!(current.score, None);
    cold_check(&c, &storage, None).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[cfg(feature = "http")]
mod blocked;
