use super::*;
use crate::materializer::SessionIndex;
use lance::index::DatasetIndexExt;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn index_recovers_prepared_and_selected_files_then_compacts_with_a_visible_wal_tail(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("indexed.lance");
    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let ingress = connect(&cluster, binding()).await?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(768 << 20);
    let opens = (0..2)
        .map(|bucket| ProducerOpen {
            bucket,
            slot: bucket,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .collect::<Vec<_>>();
    let mut expected = BTreeMap::new();
    let mut writer =
        compaction::acquire(&ingress, &storage, &budget, 1, compaction::options(false)).await?;
    for i in 0..4 {
        let (body, _) = append(
            &ingress,
            &opens[i % 2],
            (i / 2 + 1) as u64,
            &[&format!("row-{i}")],
        )
        .await?;
        remember(&mut expected, body)?;
        assert!(writer.run_once().await?.is_some());
    }
    let old_snapshot = writer.read_dataset(None).await?;
    assert!(old_snapshot.manifest().uses_stable_row_ids());
    assert!(old_snapshot.load_indices().await?.is_empty());
    let checkpoint = writer.state().await?.progress.checkpoint.clone();
    let mut options = compaction::options(true);
    options.session_index = Some(SessionIndex {
        min_fragments: 4,
        max_fragments: 4,
        max_segments: 2,
    });
    let mut indexer = compaction::acquire(&ingress, &storage, &budget, 2, options.clone()).await?;
    assert!(writer.prepare_next().await.is_err());
    let prepared = indexer
        .prepare_next()
        .await?
        .context("index not prepared")?;
    assert_eq!(prepared.descriptors.format, 3);
    let pending = indexer.state().await?.progress.pending.clone().unwrap();
    assert!(pending.plan.is_maintenance());
    assert_eq!(indexer.read_dataset(None).await?.version().version, 5);
    assert!(indexer
        .read_dataset(None)
        .await?
        .load_indices()
        .await?
        .is_empty());
    assert_eq!(indexer.state().await?.progress.checkpoint, checkpoint);
    assert_eq!(budget.reserved(), 0);

    let (body, _) = append(&ingress, &opens[0], 3, &["queued"]).await?;
    remember(&mut expected, body)?;
    compaction::query_all(&ingress, &storage, &expected).await?;
    drop((writer, indexer, ingress));
    cluster.shutdown().await?;

    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&cluster, binding()).await?;
    // Recover the same prepared index even when scheduling new indexes is disabled.
    let mut recovered =
        compaction::acquire(&ingress, &storage, &budget, 3, compaction::options(false)).await?;
    assert_eq!(recovered.prepare_next().await?.unwrap(), prepared);
    let view = recovered.state().await?;
    let predecessor = &view.manifests.as_ref().unwrap().versions[&pending.plan.read_version];
    crate::materializer::stage::publish(
        &recovered,
        &pending,
        &prepared,
        predecessor,
        recovered
            .read_dataset(Some(pending.plan.read_version))
            .await?,
    )
    .await?;
    assert!(recovered.state().await?.progress.published.is_none());
    let selected = recovered.read_dataset(None).await?;
    assert_eq!(selected.version().version, 6);
    assert_eq!(selected.load_indices().await?.len(), 1);
    let index_uuid = selected.load_indices().await?[0].uuid;
    compaction::query_all(&ingress, &storage, &expected).await?;
    assert_eq!(recovered.state().await?.progress.checkpoint, checkpoint);
    drop((recovered, ingress));
    cluster.shutdown().await?;

    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&cluster, binding()).await?;
    let mut recovered =
        compaction::acquire(&ingress, &storage, &budget, 4, options.clone()).await?;
    assert_eq!(recovered.prepare_next().await?.unwrap(), prepared);
    let completed = recovered
        .run_once()
        .await?
        .context("index did not advance")?;
    assert!(completed.sources.is_empty());
    assert_eq!(recovered.state().await?.progress.checkpoint, checkpoint);
    assert_eq!(recovered.read_dataset(None).await?.version().version, 6);
    assert_eq!(
        recovered.read_dataset(None).await?.load_indices().await?[0].uuid,
        index_uuid
    );
    // Publish the queued row, then compact only the four fragments with identical
    // index coverage. The unindexed tail remains visible throughout.
    assert!(!recovered.run_once().await?.unwrap().sources.is_empty());
    let rewrite = recovered
        .prepare_next()
        .await?
        .context("compaction not prepared")?;
    assert_eq!(rewrite.descriptors.format, 2);
    let rewrite_plan = recovered.state().await?.progress.pending.clone().unwrap();
    let mut invalid = rewrite.as_ref().clone();
    let mut result: lance::dataset::optimize::RewriteResult =
        serde_json::from_slice(&invalid.descriptors.bytes)?;
    // The managed index contains stable IDs: compaction updates its fragment
    // coverage while preserving its exact native pages and UUID.
    assert!(result.row_addrs.is_none());
    assert_eq!(result.original_fragments.len(), 4);
    assert_eq!(result.new_fragments.len(), 1);
    result.new_fragments[0].id = 0;
    invalid.descriptors =
        p::Descriptors::lance_compaction(Bytes::from(serde_json::to_vec(&result)?))?;
    let view = recovered.state().await?;
    let predecessor = &view.manifests.as_ref().unwrap().versions[&rewrite_plan.plan.read_version];
    assert!(crate::materializer::stage::publish(
        &recovered,
        &rewrite_plan,
        &invalid,
        predecessor,
        recovered
            .read_dataset(Some(rewrite_plan.plan.read_version))
            .await?,
    )
    .await
    .is_err());
    assert_eq!(recovered.read_dataset(None).await?.version().version, 7);
    compaction::query_all(&ingress, &storage, &expected).await?;
    assert!(recovered.run_once().await?.unwrap().sources.is_empty());
    let compacted = recovered.read_dataset(None).await?;
    assert_eq!(compacted.version().version, 8);
    assert_eq!(compacted.count_rows(None).await?, 5);
    assert_eq!(compacted.get_fragments().len(), 2);
    let indices = compacted.load_indices().await?;
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].uuid, index_uuid);
    assert_eq!(indices[0].index_version, 0);
    let indexed = compacted
        .manifest()
        .fragments
        .iter()
        .find(|f| f.physical_rows == Some(4))
        .unwrap();
    assert!(indices[0]
        .fragment_bitmap
        .as_ref()
        .unwrap()
        .contains(indexed.id as u32));
    compaction::query_all(&ingress, &storage, &expected).await?;
    assert_eq!(old_snapshot.count_rows(None).await?, 4);
    assert!(old_snapshot.load_indices().await?.is_empty());

    // Isolate incremental index/consolidation recovery after indexed compaction
    // has passed. Fair scheduling with both policies enabled is covered below.
    options.compaction = None;
    let mut recovered = compaction::acquire(&ingress, &storage, &budget, 5, options).await?;
    for sequence in 4..=6 {
        let (body, _) = append(
            &ingress,
            &opens[0],
            sequence,
            &[&format!("tail-{sequence}")],
        )
        .await?;
        remember(&mut expected, body)?;
        assert!(!recovered.run_once().await?.unwrap().sources.is_empty());
    }
    let checkpoint = recovered.state().await?.progress.checkpoint.clone();
    let next_index = recovered
        .prepare_next()
        .await?
        .context("incremental index not prepared")?;
    assert_eq!(next_index.descriptors.format, 3);
    assert!(recovered.run_once().await?.unwrap().sources.is_empty());
    assert_eq!(recovered.state().await?.progress.checkpoint, checkpoint);
    let indices = recovered.read_dataset(None).await?.load_indices().await?;
    assert_eq!(indices.len(), 2);
    assert!(indices.iter().any(|i| i.uuid == index_uuid));
    let before_merge = recovered.read_dataset(None).await?;
    assert!(recovered.run_once().await?.unwrap().sources.is_empty());
    assert_eq!(recovered.state().await?.progress.checkpoint, checkpoint);
    let merged = recovered.read_dataset(None).await?.load_indices().await?;
    assert_eq!(merged.len(), 1);
    assert!(indices.iter().all(|i| i.uuid != merged[0].uuid));
    assert_eq!(before_merge.load_indices().await?.len(), 2);
    let fresh = Storage::resolve(storage.uri.clone(), Default::default(), Arc::default()).await?;
    compaction::query_all(&ingress, &fresh, &expected).await?;
    assert_eq!(
        recovered.read_dataset(None).await?.count_rows(None).await?,
        expected.len()
    );
    assert_eq!(budget.reserved(), 0);
    drop((recovered, ingress));
    cluster.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backlog_allows_index_and_compaction_to_progress_across_takeover() -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("fair-maintenance.lance");
    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let ingress = connect(&cluster, binding()).await?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(768 << 20);
    let open = ProducerOpen {
        bucket: 0,
        slot: 0,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    let mut expected = BTreeMap::new();
    let mut writer =
        compaction::acquire(&ingress, &storage, &budget, 1, compaction::options(false)).await?;
    for sequence in 1..=16 {
        let (body, _) = append(&ingress, &open, sequence, &[&format!("row-{sequence}")]).await?;
        remember(&mut expected, body)?;
        assert!(writer.run_once().await?.is_some());
    }
    let mut options = compaction::options(true);
    options.session_index = Some(SessionIndex {
        min_fragments: 4,
        max_fragments: 4,
        max_segments: 16,
    });
    let mut writer = compaction::acquire(&ingress, &storage, &budget, 2, options.clone()).await?;
    let (body, _) = append(&ingress, &open, 17, &["first-tail"]).await?;
    remember(&mut expected, body)?;
    assert_eq!(writer.prepare_next().await?.unwrap().descriptors.format, 3);
    assert!(writer.run_once().await?.unwrap().sources.is_empty());
    assert!(!writer.run_once().await?.unwrap().sources.is_empty());
    let (body, _) = append(&ingress, &open, 18, &["second-tail"]).await?;
    remember(&mut expected, body)?;
    let checkpoint = writer.state().await?.progress.checkpoint.clone();
    // Both policies remain eligible, but a new owner must give compaction its
    // next turn instead of resetting to index-first and indefinitely deferring it.
    let mut writer = compaction::acquire(&ingress, &storage, &budget, 3, options).await?;
    let before = writer.read_dataset(None).await?;
    let prepared = writer.prepare_next().await?.unwrap();
    assert_eq!(prepared.descriptors.format, 2);
    assert!(writer.run_once().await?.unwrap().sources.is_empty());
    let after = writer.read_dataset(None).await?;
    assert!(after.get_fragments().len() < before.get_fragments().len());
    assert_eq!(
        after.count_rows(None).await?,
        before.count_rows(None).await?
    );
    assert_eq!(writer.state().await?.progress.checkpoint, checkpoint);
    compaction::query_all(&ingress, &storage, &expected).await?;
    assert!(!writer.run_once().await?.unwrap().sources.is_empty());
    compaction::query_all(&ingress, &storage, &expected).await?;
    assert_eq!(budget.reserved(), 0);
    drop((writer, ingress));
    cluster.shutdown().await
}
