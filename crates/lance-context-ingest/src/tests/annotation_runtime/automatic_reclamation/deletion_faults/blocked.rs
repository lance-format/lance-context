use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[tokio::test]
async fn failed_sealed_deletes_bound_publication_keep_acks_readable_and_resume() -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage(&path, faults.clone()).await?;
    let (first, completed, abandoned) = acknowledged_orphans(&c, &storage, &path).await?;
    let blocked = Arc::new(AtomicBool::new(true));
    faults.arm(&abandoned[0], Action::LoseReply)?;
    faults.arm(&abandoned[1], Action::FailWhile(blocked.clone()))?;
    let (service, runtime) = owned(&c, &storage, &limits()).await?;
    let (_, _, recovery_cut) = runtime.http_binding();
    let runtime =
        runtime.with_exclusive_reclamation(crate::annotations::runtime::reclamation::Options {
            max_objects: 16,
            max_bytes: 16 << 10,
            delete_parallelism: 2,
            interval: Duration::from_millis(100),
        })?;
    let (stop, task) = run_owner(runtime);
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let gc = service.reclamation_observation();
            if gc.delete_failures >= 3 && gc.pending_objects == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert_eq!(
        service.submit(&first, Duration::from_secs(30)).await?,
        completed
    );
    assert!(!abandoned[0].exists());
    assert!(abandoned[1].exists());
    let producer = producer(&service, 2).await?;
    let suffix = request(
        &service,
        &producer,
        1,
        wire(&limits(), &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let suffix_result = service.submit(&suffix, Duration::from_secs(30)).await?;
    let waiting = request(
        &service,
        &producer,
        2,
        wire(&limits(), &["T"], 0, None, 1, Some(0.8))?,
    )
    .await?;
    let accepted = service.ingress().append(&waiting).await?;
    let before_files = files(&path)?;
    let before_metadata = metadata_files(&path)?;
    let before_failures = service.reclamation_observation().delete_failures;
    tokio::time::timeout(Duration::from_secs(30), async {
        while service.reclamation_observation().delete_failures < before_failures + 3 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let observation = service.observation().await;
    assert!(observation.1);
    assert_eq!(observation.0.applied, suffix_result.index);
    assert!(observation.0.applied < accepted.position.index);
    assert_eq!(service.source_observation().records, 2);
    assert_eq!(service.reclamation_observation().pending_objects, 1);
    assert_eq!(
        files(&path)?,
        before_files,
        "blocked GC must not generate more data"
    );
    assert_eq!(metadata_files(&path)?, before_metadata);
    assert!(
        tokio::time::timeout(
            Duration::from_millis(100),
            service.submit(&waiting, Duration::from_secs(30)),
        )
        .await
        .is_err(),
        "an unapplied proposal must not receive an annotation ACK"
    );
    let current = service
        .get_owned("T", recovery_cut, Duration::from_secs(1))
        .await?
        .unwrap();
    assert_eq!(current.status.as_deref(), Some("acknowledged"));
    assert_eq!(current.score, Some(0.9));

    blocked.store(false, Ordering::SeqCst);
    let result = service.submit(&waiting, Duration::from_secs(30)).await?;
    assert_eq!(result.index, accepted.position.index);
    drained(&c, &service, result.index).await?;
    wait_files(&path, 3).await?;
    assert_eq!(service.reclamation_observation().pending_objects, 0);
    assert!(abandoned.iter().all(|p| !p.exists()));
    let current = service
        .get_owned("T", recovery_cut, Duration::from_secs(1))
        .await?
        .unwrap();
    assert_eq!(current.status.as_deref(), Some("acknowledged"));
    assert_eq!(current.score, Some(0.8));
    cold_check(&c, &storage, Some(0.8)).await?;
    // Confirm the absent path is retried once to resolve its lost reply, then
    // excluded from all later attempts of the still-failing sealed page.
    let absent = object_store::path::Path::from_absolute_path(&abandoned[0])?;
    assert_eq!(
        faults
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|p| **p == absent)
            .count(),
        2
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
