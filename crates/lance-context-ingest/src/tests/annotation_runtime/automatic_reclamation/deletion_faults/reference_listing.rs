//! A pending reference check cannot hold the bounded source window full.
use super::*;

#[tokio::test]
async fn annotation_collector_pending_tag_check_allows_publication() -> Result<()> {
    pending_reference("tags").await
}

#[tokio::test]
async fn annotation_collector_pending_branch_check_allows_publication() -> Result<()> {
    pending_reference("branches").await
}

async fn pending_reference(reference: &str) -> Result<()> {
    let root = root()?;
    let path = root.path().join("turns.lance");
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(Faults::default());
    let storage = fault_store::storage(&path, faults.clone()).await?;
    let (first, completed, abandoned) = acknowledged_orphans(&c, &storage, &path).await?;
    let (service, runtime) = owned(&c, &storage, &limits()).await?;
    let (entered, reached) = oneshot::channel();
    let (resume, resumed) = oneshot::channel();
    faults.pause_listing(&path.join("_refs").join(reference), None, entered, resumed)?;
    let (stop, task) = run_owner(whole_page(runtime)?);
    tokio::time::timeout(Duration::from_secs(30), reached).await??;
    let producer = producer(&service, 2).await?;
    let mut last = completed.index;
    let mut score = None;
    let progress = async {
        // Cross several publication generations while the original reference
        // response stays pending. No new candidate page may be classified yet.
        for sequence in 1..=4 {
            let update = request(
                &service,
                &producer,
                sequence,
                wire(&limits(), &["T"], 0, None, 1, Some(sequence as f32))?,
            )
            .await?;
            let result = service.submit(&update, Duration::from_secs(30)).await?;
            last = result.index;
            score = Some(sequence as f32);
            let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
            ensure!(current.status.as_deref() == Some("acknowledged"));
            ensure!(current.score == score);
            tokio::time::timeout(Duration::from_secs(5), drained(&c, &service, last))
                .await
                .context("publication blocked on pending reference LIST")??;
            ensure!(faults.calls.lock().unwrap().is_empty());
            ensure!(faults.data_listings.lock().unwrap().is_empty());
            ensure!(faults.collection_starts.lock().unwrap().len() == 1);
            ensure!(abandoned.iter().all(|p| p.exists()));
        }
        Ok::<_, anyhow::Error>(())
    }
    .await;
    // Settle the original response and all fixture operations even when the
    // parent implementation fails the publication-progress assertion.
    let retained = resume.send(()).is_ok();
    drained(&c, &service, last).await?;
    wait_files(&path, 3).await?;
    assert_eq!(
        service.submit(&first, Duration::from_secs(30)).await?,
        completed
    );
    let _ = stop.send(());
    task.await??;
    cold_check(&c, &storage, score).await?;
    assert!(abandoned.iter().all(|p| !p.exists()));
    c.shutdown().await?;
    ensure!(retained, "original reference response was abandoned");
    progress
}
