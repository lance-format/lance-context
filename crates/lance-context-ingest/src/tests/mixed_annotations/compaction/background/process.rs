//! Actual process loss during an unselected background rewrite. The child owns
//! all three WAL replicas; their disks survive. This does not model volume loss.
use super::*;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};

#[derive(serde::Serialize, serde::Deserialize)]
struct RequestOutcome {
    slot: u32,
    sequence: u64,
    body: Vec<u8>,
    completed: service::Completed,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Expected {
    progress: lance_context_wal::replication::publication::Progress,
    manifest_head: u64,
    materialized: u64,
    body_ids: BTreeMap<String, u64>,
    output: PathBuf,
    requests: Vec<RequestOutcome>,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "subprocess role, launched by background_compaction_sigkill_recovers_published_and_pending_updates"]
async fn background_compaction_process_role() -> Result<()> {
    let root = PathBuf::from(std::env::var("BACKGROUND_COMPACTION_PROCESS_ROOT")?);
    prepare_crash(&root).await
}

async fn prepare_crash(root: &Path) -> Result<()> {
    let c = network::Cluster::open_signed(root, annotation_publication::one_shard()).await?;
    let faults = Arc::new(fault_store::Faults::default());
    let narrow = fault_store::storage(&root.join("annotations.lance"), faults.clone()).await?;
    let body = storage(root, "body.lance").await?;
    let (last, completed) = fragmented(&c, &narrow, &body).await?;
    let mut requests = vec![RequestOutcome {
        slot: 1,
        sequence: 3,
        body: last.body().to_vec(),
        completed,
    }];
    drop(last);
    let (a, mut runtime) = compact_owner(&c, &narrow, &body).await?;
    let (entered, _resume) = faults.pause_data_write();
    while runtime.compaction_preparation_finished_for_test().is_none() {
        runtime.publish_once_for_test().await?;
    }
    let path = tokio::time::timeout(Duration::from_secs(20), entered).await??;
    let output = Path::new("/").join(path.as_ref());
    assert!(output.exists());

    let p = producer(&a, 2).await?;
    for sequence in 1..=8 {
        let id = format!("new-{sequence}");
        let bytes = encoded(create(&[&id], &["s"], &[sequence as u8 + 10])?)?;
        let input = request(&a, &p, sequence, bytes.clone()).await?;
        let completed = submit(&mut runtime, &a, input).await?;
        assert_eq!(completed.result, service::MutationResult::Applied);
        while a.observation().await.0.materialized < completed.index {
            runtime.publish_once_for_test().await?;
        }
        assert_eq!(a.source_observation().records, 0);
        assert_eq!(
            runtime.compaction_preparation_finished_for_test(),
            Some(false)
        );
        // Request outcomes retain the latest sequence of each producer.
        if sequence == 8 {
            requests.push(RequestOutcome {
                slot: 2,
                sequence,
                body: bytes.to_vec(),
                completed,
            });
        }
    }
    let materialized = a.observation().await.0.materialized;
    let body_ids = ids(&selected_body(&c, &narrow, &body).await?).await?;
    assert_eq!(body_ids.len(), 11);

    // Leave a mixed suffix acknowledged and visible, but never published. Two
    // independent requests modify different fields of the same existing turn.
    for (slot, batch) in [
        (
            3,
            mixed::wire::patches(&patch(&["A"], Some("reviewed"), None)?, &limits())?,
        ),
        (
            4,
            mixed::wire::patches(&patch(&["A"], None, Some(0.9))?, &limits())?,
        ),
        (5, create(&["D"], &["s"], &[4])?),
    ] {
        let p = producer(&a, slot).await?;
        let bytes = encoded(batch)?;
        let input = request(&a, &p, 1, bytes.clone()).await?;
        let completed = submit(&mut runtime, &a, input).await?;
        assert_eq!(completed.result, service::MutationResult::Applied);
        assert!(completed.index > materialized);
        requests.push(RequestOutcome {
            slot,
            sequence: 1,
            body: bytes.to_vec(),
            completed,
        });
    }
    check_without_publication(&a, &mut runtime).await?;
    assert_eq!(a.observation().await.0.materialized, materialized);
    assert_eq!(a.source_observation().records, 3);
    assert_eq!(
        runtime.compaction_preparation_finished_for_test(),
        Some(false)
    );
    assert!(output.exists());
    let state = annotation_publication::state(&c).await?;
    assert!(state.progress.pending.is_none());
    let expected = Expected {
        progress: state.progress,
        manifest_head: state.manifests.unwrap().head,
        materialized,
        body_ids,
        output,
        requests,
    };
    std::fs::write(root.join("expected.json"), serde_json::to_vec(&expected)?)?;
    std::fs::write(
        root.join("ready"),
        b"background rewrite paused; suffix acknowledged",
    )?;
    // Keep the actual writer, owner and all replicas alive until the parent
    // kills this process. No graceful shutdown or release of the paused write.
    std::future::pending::<Result<()>>().await
}

async fn check_values(a: &Service) -> Result<()> {
    check(a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(a, "B", "s", 2, "pending", None).await?;
    check(a, "C", "s", 3, "pending", None).await?;
    check(a, "D", "s", 4, "pending", None).await?;
    for sequence in 1..=8 {
        check(
            a,
            &format!("new-{sequence}"),
            "s",
            sequence + 10,
            "pending",
            None,
        )
        .await?;
    }
    Ok(())
}

async fn check_without_publication(
    a: &Service,
    runtime: &mut service::runtime::Runtime,
) -> Result<()> {
    let reading = a.clone();
    let task = tokio::spawn(async move { check_values(&reading).await });
    while !task.is_finished() {
        runtime.apply_once_for_test().await?;
        tokio::task::yield_now().await;
    }
    task.await?
}

struct Child(std::process::Child);

impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn background_compaction_sigkill_recovers_published_and_pending_updates() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), killed_recovery()).await?
}

async fn killed_recovery() -> Result<()> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("background-compaction-process-");
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        builder.disable_cleanup(true).tempdir_in(parent)?
    } else {
        builder.tempdir()?
    };
    eprintln!(
        "background compaction SIGKILL fixture: {}",
        root.path().display()
    );
    let log = std::fs::File::create(root.path().join("child.log"))?;
    let mut child = Child(std::process::Command::new(std::env::current_exe()?)
        .args(["--exact", "tests::mixed_annotations::compaction::background::process::background_compaction_process_role", "--ignored", "--nocapture"])
        .env("BACKGROUND_COMPACTION_PROCESS_ROOT", root.path())
        .stdout(log.try_clone()?).stderr(log).spawn()?);
    while !root.path().join("ready").exists() {
        if let Some(status) = child.0.try_wait()? {
            bail!(
                "background compaction child exited {status}: {}",
                std::fs::read_to_string(root.path().join("child.log"))?
            );
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    child.0.kill()?;
    assert_eq!(child.0.wait()?.signal(), Some(9));
    let expected: Expected =
        serde_json::from_slice(&std::fs::read(root.path().join("expected.json"))?)?;
    verify_recovery(root.path(), expected).await
}

async fn verify_recovery(root: &Path, expected: Expected) -> Result<()> {
    // Fresh Lance sessions and replica processes must reconstruct both the
    // published checkpoint and acknowledged suffix without a warm read cache.
    let c = network::Cluster::open_signed(root, annotation_publication::one_shard()).await?;
    let narrow = storage(root, "annotations.lance").await?;
    let body = storage(root, "body.lance").await?;
    let recovered = annotation_publication::state(&c).await?;
    assert_eq!(recovered.progress, expected.progress);
    assert_eq!(recovered.manifests.unwrap().head, expected.manifest_head);
    assert!(expected.output.exists());
    for request in &expected.requests[2..] {
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: request.completed.index,
                })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
    }
    let (a, mut runtime) = compact_owner(&c, &narrow, &body).await?;
    check_without_publication(&a, &mut runtime).await?;
    for retained in &expected.requests {
        let p = producer(&a, retained.slot).await?;
        let retry = request(
            &a,
            &p,
            retained.sequence,
            Bytes::from(retained.body.clone()),
        )
        .await?;
        assert_eq!(submit(&mut runtime, &a, retry).await?, retained.completed);
    }
    assert_eq!(a.observation().await.0.materialized, expected.materialized);
    assert_eq!(
        annotation_publication::state(&c)
            .await?
            .manifests
            .unwrap()
            .head,
        expected.manifest_head
    );
    assert_eq!(
        ids(&selected_body(&c, &narrow, &body).await?).await?,
        expected.body_ids
    );

    let runtime = runtime.with_exclusive_reclamation(service::runtime::reclamation::Options {
        max_objects: 2,
        max_bytes: 16 << 10,
        delete_parallelism: 2,
        interval: Duration::from_millis(5),
    })?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let through = expected.requests.last().unwrap().completed.index;
    annotation_runtime::drained(&c, &a, through).await?;
    while expected.output.exists() {
        ensure!(
            !task.is_finished(),
            "successor stopped before collecting the abandoned output"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    check_values(&a).await?;
    let current_ids = ids(&selected_body(&c, &narrow, &body).await?).await?;
    assert_eq!(current_ids.len(), 12);
    for (id, stable) in expected.body_ids {
        assert_eq!(current_ids[&id], stable);
    }
    assert_eq!(a.retained_body_frames(), (0, 0));
    for retained in &expected.requests {
        let p = producer(&a, retained.slot).await?;
        let retry = request(
            &a,
            &p,
            retained.sequence,
            Bytes::from(retained.body.clone()),
        )
        .await?;
        assert_eq!(
            a.submit(&retry, Duration::from_secs(10)).await?,
            retained.completed
        );
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
