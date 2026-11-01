use super::*;
use object_store::path::Path;

#[tokio::test]
async fn delayed_delete_allows_the_bounded_source_window_to_drain() -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage(&path, faults.clone()).await?;
    let (_, completed, abandoned) = acknowledged_orphans(&c, &storage, &path).await?;
    let (entered, reached) = oneshot::channel();
    let (release, held) = oneshot::channel();
    faults.arm(
        &abandoned[0],
        Action::Pause {
            entered,
            resume: held,
        },
    )?;
    let (listing_entered, listing_reached) = oneshot::channel();
    let (listing_release, listing_held) = oneshot::channel();
    faults.pause_listing(&path, None, listing_entered, listing_held)?;
    let (service, runtime) = owned(&c, &storage, &limits()).await?;
    let (stop, task) = run_owner(whole_page(runtime)?);
    tokio::time::timeout(Duration::from_secs(30), listing_reached).await??;
    // A slow initial LIST can span multiple collection slices before any
    // DELETE starts. Exercise that ordering without depending on machine load.
    tokio::time::timeout(Duration::from_secs(30), async {
        while faults.collection_starts.lock().unwrap().len() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    listing_release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(30), reached).await??;
    let collections_at_pause = faults.collection_starts.lock().unwrap().len();
    assert!(collections_at_pause >= 2);
    let producer = producer(&service, 2).await?;
    let mut last = completed.index;
    let mut score = None;
    let progress = async {
        // More updates than the two-record source window, without releasing,
        // cancelling or resending the original DELETE or any mutation.
        for sequence in 1..=4 {
            let update = request(
                &service,
                &producer,
                sequence,
                wire(&limits(), &["T"], 0, None, 1, Some(sequence as f32))?,
            )
            .await?;
            let result = service.submit(&update, Duration::from_secs(5)).await?;
            last = result.index;
            score = Some(sequence as f32);
            ensure!(result.result == MutationResult::Applied);
            let value = service.get("T", Duration::from_secs(5)).await?.unwrap();
            ensure!(value.status.as_deref() == Some("acknowledged") && value.score == score);
            ensure!(service.source_observation().records <= 2);
        }
        tokio::time::timeout(Duration::from_secs(5), drained(&c, &service, last)).await??;
        ensure!(abandoned[0].exists(), "held DELETE completed early");
        let expected = Path::from_absolute_path(&abandoned[0])?;
        ensure!(
            faults
                .calls
                .lock()
                .unwrap()
                .iter()
                .filter(|p| **p == expected)
                .count()
                == 1,
            "pending DELETE was reissued"
        );
        let collections_after_updates = faults.collection_starts.lock().unwrap().len();
        ensure!(
            collections_after_updates == collections_at_pause,
            "started another page while the original DELETE was pending: at_pause={}, after_updates={}",
            collections_at_pause,
            collections_after_updates
        );
        Ok::<_, anyhow::Error>(())
    }
    .await;
    let retained = release.send(()).is_ok();
    faults.finish_remote().await?;
    let drain = drained(&c, &service, last).await;
    let _ = stop.send(());
    let runtime = task.await?;
    let cold = if progress.is_ok() {
        cold_check(&c, &storage, score).await
    } else {
        Ok(())
    };
    c.shutdown().await?;
    ensure!(retained, "original DELETE response was cancelled");
    progress?;
    drain?;
    runtime?;
    cold
}
