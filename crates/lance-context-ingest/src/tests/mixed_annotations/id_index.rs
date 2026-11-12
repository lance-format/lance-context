use super::*;
use lance::index::DatasetIndexExt;
use std::collections::BTreeSet;
mod reclamation;

async fn indexed_owner(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    compact: bool,
) -> Result<(Service, service::runtime::Runtime)> {
    indexed_owner_with_history(c, narrow, body, compact, 64).await
}

async fn indexed_owner_with_history(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    compact: bool,
    history: usize,
) -> Result<(Service, service::runtime::Runtime)> {
    let state = annotation_publication::state(c).await?;
    service::runtime::Runtime::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        compaction::compact_binding(),
        narrow.clone(),
        body.clone(),
        service::runtime::Configuration {
            limits: limits(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(1024 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 4,
                max_batch_wait: Duration::from_millis(20),
                retained_manifest_versions: history,
                turn_id_index: Some(crate::materializer::TurnIdIndex { min_rows: 1 }),
                compaction: compact.then_some(crate::materializer::SmallFileCompaction {
                    target_file_bytes: 64 << 10,
                    min_fragments: 2,
                }),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
    )
    .await
}

async fn fully_indexed(
    c: &network::Cluster,
    narrow: &Storage,
    rows: usize,
    segments: usize,
) -> Result<Arc<Dataset>> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let selected = annotation_publication::read(c, narrow, None).await?;
            let indices = selected.load_indices().await?;
            let covered: BTreeSet<_> = indices
                .iter()
                .filter(|i| i.name == "lance_context_turn_id")
                .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
                .collect();
            if selected.count_rows(None).await? == rows
                && indices.len() == segments
                && selected
                    .fragments()
                    .iter()
                    .all(|f| covered.contains(&(f.id as u32)))
                && selected.fragments().len() <= segments
                && annotation_publication::state(c)
                    .await?
                    .progress
                    .pending
                    .is_none()
            {
                return Ok::<_, anyhow::Error>(selected);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_id_index_build_merge_compaction_and_restart_preserve_current_turns() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), indexed_load()).await?
}

async fn indexed_load() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed_with_binding(&c, &narrow, &body, compaction::compact_binding()).await?;
    let initial = annotation_publication::state(&c).await?;
    assert_eq!(initial.progress.minimum_read_version, 1);
    assert!(initial.manifests.as_ref().unwrap().head > 1);
    let (a, runtime) = indexed_owner(&c, &narrow, &body, true)
        .await
        .context("open indexed owner at selected bootstrap version")?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let creator = producer(&a, 1).await?;
    let patcher = producer(&a, 2).await?;
    let mut last_patch = None;
    for row in 0..8 {
        let id = format!("turn-{row}");
        let input = request(
            &a,
            &creator,
            row + 1,
            encoded(create(&[&id], &["s"], &[row as u8])?)?,
        )
        .await?;
        a.submit(&input, Duration::from_secs(10)).await?;
        // New registration and an older turn's update both reach quorum/apply
        // while the publisher independently maintains the native ID index.
        let patch = request(
            &a,
            &patcher,
            row + 1,
            encoded(mixed::wire::patches(
                &patch(&["turn-0"], Some("reviewed"), Some(row as f32 / 10.0))?,
                &limits(),
            )?)?,
        )
        .await?;
        let result = a.submit(&patch, Duration::from_secs(10)).await?;
        annotation_runtime::drained(&c, &a, result.index).await?;
        let count = row as usize + 1;
        fully_indexed(&c, &narrow, count, count / 4 + count % 4).await?;
        last_patch = Some((patch, result));
    }
    // Prove consolidation ran through actual retained native transactions,
    // not merely that the final reader happens to find every ID.
    let state = annotation_publication::state(&c).await?;
    let mut merges = 0;
    let manifests = state.manifests.as_ref().unwrap();
    // Bootstrap identity is retained in the catalog even after its data view
    // is retired. Only subsequent retained versions are readable here.
    for version in manifests
        .versions
        .keys()
        .filter(|v| **v != manifests.initial_version)
    {
        let selected = annotation_publication::read(&c, &narrow, Some(*version)).await?;
        if let Some(transaction) = selected.read_transaction().await? {
            if let Operation::CreateIndex {
                removed_indices, ..
            } = transaction.operation
            {
                if !removed_indices.is_empty() {
                    assert_eq!(removed_indices.len(), 4);
                    merges += 1;
                }
            }
        }
    }
    assert_eq!(merges, 2);
    let again = request(
        &a,
        &creator,
        9,
        encoded(create(&["turn-0"], &["other"], &[99])?)?,
    )
    .await?;
    let repeated = a.submit(&again, Duration::from_secs(10)).await?;
    annotation_runtime::drained(&c, &a, repeated.index).await?;
    check(&a, "turn-0", "s", 0, "reviewed", Some(0.7)).await?;
    for row in 1..8 {
        check(&a, &format!("turn-{row}"), "s", row, "pending", None).await?;
    }
    let (last_patch, patch_result) = last_patch.unwrap();
    assert_eq!(
        a.submit(&last_patch, Duration::from_secs(10)).await?,
        patch_result
    );
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (a, runtime) = indexed_owner(&c, &narrow, &body, true)
        .await
        .context("reopen indexed owner after WAL restart")?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    check(&a, "turn-0", "s", 0, "reviewed", Some(0.7)).await?;
    for row in 1..8 {
        check(&a, &format!("turn-{row}"), "s", row, "pending", None).await?;
    }
    assert_eq!(a.submit(&again, Duration::from_secs(10)).await?, repeated);
    assert_eq!(
        a.submit(&last_patch, Duration::from_secs(10)).await?,
        patch_result
    );
    fully_indexed(&c, &narrow, 8, 2).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[cfg(feature = "service")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_id_index_unknown_prepare_preserves_acked_create_and_patch() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), lost_prepare()).await?
}

#[cfg(feature = "service")]
async fn lost_prepare() -> Result<()> {
    use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
    use std::sync::atomic::Ordering;
    let root = artifact_root()?;
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    let (previous, previous_result) =
        compaction::fragmented_with_slot(&c, &narrow, &body, 1).await?;
    let base = annotation_publication::read(&c, &narrow, None).await?;
    let proxy = Proxy::open(&c, root.path()).await?;
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let (a, runtime) = indexed_owner(&c, &narrow, &body, false).await?;
    let (_stop, task) = annotation_runtime::run_owner(runtime);
    tokio::time::timeout(Duration::from_secs(30), proxy.committed.notified()).await?;
    let p = producer(&a, 2).await?;
    let q = producer(&a, 3).await?;
    let update = request(
        &a,
        &p,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    let insert = request(&a, &q, 1, encoded(create(&["D"], &["s"], &[4])?)?).await?;
    let updated = a.submit(&update, Duration::from_secs(10)).await?;
    let inserted = a.submit(&insert, Duration::from_secs(10)).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "D", "s", 4, "pending", None).await?;
    let retained = annotation_publication::state(&c).await?;
    let pending = retained.progress.pending.clone().unwrap();
    let prepared = retained.progress.prepared.clone().unwrap();
    assert!(pending.plan.is_maintenance());
    assert_eq!(prepared.descriptors.format, 3);
    let expected = crate::materializer::stage::annotation_compaction_transaction(
        &base,
        &pending,
        &prepared.descriptors,
    )
    .await?;
    let through = retained.progress.checkpoint[&10];
    assert!(updated.index > through && inserted.index > through);
    proxy.release.notify_one();
    // The proxy blocks all voter observations for the unchanged 30-second RPC deadline.
    let error = tokio::time::timeout(Duration::from_secs(45), task)
        .await??
        .unwrap_err();
    proxy.restore_observations();
    assert!(format!("{error:#}").contains("outcome unknown"));
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(
        annotation_publication::state(&c).await?.progress.prepared,
        Some(prepared.clone())
    );
    for result in [&updated, &inserted] {
        assert!(matches!(
            direct
                .request_routed(Request::Fetch {
                    group: 10,
                    index: result.index
                })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
    }
    drop(a);
    c.client = direct;
    drop(proxy);
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    assert_eq!(
        annotation_publication::state(&c).await?.progress.prepared,
        Some(prepared)
    );
    let (a, runtime) = indexed_owner(&c, &narrow, &body, false).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    annotation_runtime::drained(&c, &a, inserted.index).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "D", "s", 4, "pending", None).await?;
    for (input, result) in [
        (&previous, &previous_result),
        (&update, &updated),
        (&insert, &inserted),
    ] {
        assert_eq!(&a.submit(input, Duration::from_secs(10)).await?, result);
    }
    let original =
        annotation_publication::read(&c, &narrow, Some(pending.plan.target_version()?)).await?;
    assert_eq!(original.read_transaction().await?, Some(expected));
    assert_eq!(original.manifest().fragments, base.manifest().fragments);
    assert_eq!(original.load_indices().await?.len(), 1);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
