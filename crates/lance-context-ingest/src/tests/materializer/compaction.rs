use super::*;
use crate::{materializer::SmallFileCompaction, reader};

#[cfg(all(unix, feature = "service"))]
mod process;

pub(super) fn options(enabled: bool) -> Options {
    Options {
        max_source_bytes: 1 << 20,
        max_records: 8,
        write_parallelism: 1,
        compaction: enabled.then_some(SmallFileCompaction {
            target_file_bytes: 256 << 10,
            min_fragments: 4,
            ..Default::default()
        }),
        ..Options::default()
    }
}

pub(super) async fn acquire(
    ingress: &Ingress,
    storage: &Storage,
    budget: &Arc<MergeMemoryBudget>,
    epoch: u64,
    options: Options,
) -> Result<Publisher> {
    Publisher::acquire(
        ingress.clone(),
        storage.clone(),
        options,
        budget.clone(),
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: epoch,
        },
    )
    .await
}

pub(super) async fn query_all(
    ingress: &Ingress,
    storage: &Storage,
    expected: &BTreeMap<String, RecordBatch>,
) -> Result<()> {
    // Two sessions in distinct source groups are read concurrently while the
    // same table may contain both published rows and an acknowledged WAL tail.
    futures::future::try_join_all((0..2).map(|bucket| async move {
        let s = session(bucket);
        let mut read = reader::read_session(ingress, storage, &s).await?;
        let mut found = BTreeMap::new();
        while let Some(batch) = read.batches.try_next().await? {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for row in 0..batch.num_rows() {
                let id = ids.value(row);
                assert!(found.insert(id.to_owned(), ()).is_none());
                assert_eq!(batch.slice(row, 1), expected[id]);
            }
        }
        let expected_ids = expected
            .iter()
            .filter(|(_, batch)| {
                batch
                    .column_by_name("rollout_id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value(0)
                    == s
            })
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        assert_eq!(found.keys().collect::<Vec<_>>(), expected_ids);
        Ok::<_, anyhow::Error>(())
    }))
    .await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compaction_preserves_wal_queries_and_resumes_prepared_and_published_output() -> Result<()>
{
    let root = tempfile::tempdir()?;
    let path = root.path().join("compacted.lance");
    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let ingress = connect(&cluster, binding())
        .await?
        .with_session_read_concurrency(2)?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(32 << 20);
    let opens = (0..2)
        .map(|bucket| ProducerOpen {
            bucket,
            slot: bucket,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .collect::<Vec<_>>();
    let mut expected = BTreeMap::new();
    let mut first = acquire(&ingress, &storage, &budget, 1, options(false)).await?;
    for i in 0..4 {
        let (body, _) = append(
            &ingress,
            &opens[i % 2],
            (i / 2 + 1) as u64,
            &[&format!("row-{i}")],
        )
        .await?;
        remember(&mut expected, body)?;
        assert!(first.run_once().await?.is_some());
    }
    verify(&first, &expected, 5).await?;
    let old_snapshot = first.read_dataset(None).await?;
    assert_eq!(old_snapshot.manifest().fragments.len(), 4);
    let checkpoint = first.state().await?.progress.checkpoint.clone();
    let mut compactor = acquire(&ingress, &storage, &budget, 2, options(true)).await?;
    assert!(
        first.prepare_next().await.is_err(),
        "stale owner was not fenced"
    );
    let prepared = compactor
        .prepare_next()
        .await?
        .context("missing compaction")?;
    let pending = compactor.state().await?.progress.pending.clone().unwrap();
    assert!(pending.plan.is_compaction());
    assert_eq!(prepared.descriptors.format, 2);
    assert_eq!(compactor.state().await?.progress.checkpoint, checkpoint);
    verify(&compactor, &expected, 5).await?;
    let (body, _) = append(&ingress, &opens[0], 3, &["queued"]).await?;
    let mut all = expected.clone();
    remember(&mut all, body)?;
    query_all(&ingress, &storage, &all).await?;
    assert_eq!(budget.reserved(), 0);
    drop((first, compactor, ingress));
    cluster.shutdown().await?;

    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&cluster, binding())
        .await?
        .with_session_read_concurrency(2)?;
    // Prepared output must remain recoverable even with new scheduling disabled.
    let mut recovered = acquire(&ingress, &storage, &budget, 3, options(false)).await?;
    assert_eq!(recovered.prepare_next().await?.unwrap(), prepared);
    assert_eq!(
        recovered.state().await?.progress.pending.clone().unwrap(),
        pending
    );
    query_all(&ingress, &storage, &all).await?;
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
    let compacted = recovered.read_dataset(None).await?;
    assert_eq!(compacted.version().version, 6);
    assert!(compacted.manifest().fragments.len() < 4);
    let selected_fragments = compacted.manifest().fragments.clone();
    verify(&recovered, &expected, 6).await?;
    assert_eq!(recovered.state().await?.progress.checkpoint, checkpoint);
    // The rewrite must not remove an append acknowledged after its selection.
    query_all(&ingress, &storage, &all).await?;
    // A reader holding the pre-compaction snapshot can still read its files.
    let mut old_rows = BTreeMap::new();
    let mut old_stream = old_snapshot.scan().try_into_stream().await?;
    while let Some(batch) = old_stream.try_next().await? {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            assert!(old_rows
                .insert(ids.value(row).to_owned(), batch.slice(row, 1))
                .is_none());
        }
    }
    assert_eq!(old_rows, expected);
    drop((recovered, ingress));
    cluster.shutdown().await?;

    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&cluster, binding()).await?;
    let mut recovered = acquire(&ingress, &storage, &budget, 4, options(true)).await?;
    assert_eq!(recovered.prepare_next().await?.unwrap(), prepared);
    recovered.commit_pending().await?;
    assert_eq!(
        recovered.read_dataset(None).await?.manifest().fragments,
        selected_fragments
    );
    drop((recovered, ingress));
    cluster.shutdown().await?;

    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&cluster, binding()).await?;
    let mut recovered = acquire(&ingress, &storage, &budget, 5, options(true)).await?;
    assert_eq!(recovered.prepare_next().await?.unwrap(), prepared);
    recovered.commit_pending().await?;
    let advanced = recovered.finish_pending().await?.unwrap();
    assert!(advanced.sources.is_empty());
    assert_eq!(recovered.state().await?.progress.checkpoint, checkpoint);
    query_all(&ingress, &storage, &all).await?;
    assert!(recovered.run_once().await?.is_some());
    verify(&recovered, &all, 7).await?;
    query_all(&ingress, &storage, &all).await?;
    assert!(
        recovered.run_once().await?.is_none(),
        "compaction created an endless version loop"
    );
    assert_eq!(budget.reserved(), 0);
    drop((recovered, ingress));
    cluster.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compaction_and_appends_progress_with_backlog_across_takeover() -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("backlogged.lance");
    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let ingress = connect(&cluster, binding())
        .await?
        .with_session_read_concurrency(2)?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(32 << 20);
    let opens = (0..2)
        .map(|bucket| ProducerOpen {
            bucket,
            slot: bucket,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .collect::<Vec<_>>();
    let mut expected = BTreeMap::new();
    let mut publisher = acquire(&ingress, &storage, &budget, 1, options(false)).await?;
    for i in 0..24 {
        let (body, _) = append(
            &ingress,
            &opens[i % 2],
            (i / 2 + 1) as u64,
            &[&format!("row-{i}")],
        )
        .await?;
        remember(&mut expected, body)?;
        if i < 16 {
            assert!(publisher.run_once().await?.is_some());
        }
    }
    assert_eq!(
        publisher
            .read_dataset(None)
            .await?
            .manifest()
            .fragments
            .len(),
        16
    );
    let mut checkpoint = publisher.state().await?.progress.checkpoint.clone();
    drop(publisher);
    let bounded = Options {
        max_records: 1,
        compaction: Some(SmallFileCompaction {
            target_file_bytes: 64 << 10,
            min_fragments: 2,
            ..Default::default()
        }),
        ..options(true)
    };
    let mut published_rows = 16;
    let mut published_groups = std::collections::BTreeSet::new();
    for turn in 0..4 {
        // A new owner each turn cannot reset the append/rewrite scheduling.
        let mut publisher = acquire(&ingress, &storage, &budget, 2 + turn, bounded.clone()).await?;
        query_all(&ingress, &storage, &expected).await?;
        let before = publisher
            .read_dataset(None)
            .await?
            .manifest()
            .fragments
            .len();
        let completed = publisher
            .run_once()
            .await?
            .context("backlog made no progress")?;
        let dataset = publisher.read_dataset(None).await?;
        if turn % 2 == 0 {
            assert!(
                completed.sources.is_empty(),
                "WAL backlog starved compaction"
            );
            assert!(dataset.manifest().fragments.len() < before);
            assert_eq!(publisher.state().await?.progress.checkpoint, checkpoint);
        } else {
            assert!(
                !completed.sources.is_empty(),
                "compaction starved WAL publication"
            );
            published_groups.extend(completed.sources.keys().copied());
            published_rows += 1;
            let after = publisher.state().await?.progress.checkpoint.clone();
            assert_ne!(after, checkpoint);
            for (group, position) in &checkpoint {
                assert!(after.get(group).is_some_and(|next| next >= position));
            }
            checkpoint = after;
        }
        assert_eq!(dataset.count_rows(None).await?, published_rows);
        assert!(
            published_rows < expected.len(),
            "test drained the WAL backlog"
        );
        // Verify every column and ID across both source groups and Lance.
        query_all(&ingress, &storage, &expected).await?;
    }
    assert_eq!(published_groups.len(), 2, "rewrites starved a source group");
    assert_eq!(budget.reserved(), 0);
    drop(ingress);
    cluster.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compaction_does_not_flush_pending_append_batch() -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("priority.lance");
    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let ingress = connect(&cluster, binding()).await?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(32 << 20);
    let open = ProducerOpen {
        bucket: 0,
        slot: 0,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    let mut publisher = acquire(&ingress, &storage, &budget, 1, options(false)).await?;
    let mut expected = BTreeMap::new();
    for sequence in 1..=4 {
        let (body, _) = append(&ingress, &open, sequence, &[&format!("row-{sequence}")]).await?;
        remember(&mut expected, body)?;
        publisher.run_once().await?;
    }
    assert_eq!(
        publisher
            .read_dataset(None)
            .await?
            .manifest()
            .fragments
            .len(),
        4
    );
    let mut foreign = binding();
    foreign.table = Uuid::from_u128(71);
    foreign.incarnation = Uuid::from_u128(72);
    foreign.destination.dataset = "other.lance".into();
    foreign.qualified_table = "other".into();
    bootstrap(&cluster, &root.path().join("other.lance"), &foreign).await?;
    let other = connect(&cluster, foreign.clone()).await?;
    let other_producer = other
        .open_producer(ProducerOpen {
            bucket: 0,
            slot: 1,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .await?;
    let other_session = (0..1000)
        .map(|i| format!("other-{i}"))
        .find(|s| {
            layout()
                .session_bucket(&foreign.tenant, &foreign.qualified_table, s)
                .unwrap()
                == 0
        })
        .unwrap();
    // After one rewrite, an incomplete scan or an accumulating append must not
    // trigger another rewrite or force a small append publication.
    for seq in 1..=129 {
        let body = ipc_file(&[batch(&[&format!("foreign-{seq}")], &[&other_session])?])?;
        let prepared = other
            .prepare(
                other.reserve_upload(body.len()).await?,
                body,
                &other_producer,
                seq,
            )
            .await?;
        other.append(&prepared).await?;
    }
    let (body, _) = append(&ingress, &open, 5, &["tail"]).await?;
    remember(&mut expected, body)?;
    let mut publisher = acquire(
        &ingress,
        &storage,
        &budget,
        2,
        Options {
            min_source_bytes: 1 << 20,
            max_batch_wait: Duration::from_secs(60),
            scan_pages_per_group: 1,
            compaction: Some(SmallFileCompaction {
                min_fragments: 2,
                ..options(true).compaction.unwrap()
            }),
            ..options(true)
        },
    )
    .await?;
    let checkpoint = publisher.state().await?.progress.checkpoint.clone();
    let rewrite = publisher.run_once().await?.context("missing rewrite")?;
    assert!(rewrite.sources.is_empty());
    assert_eq!(publisher.state().await?.progress.checkpoint, checkpoint);
    assert_eq!(publisher.read_dataset(None).await?.version().version, 6);
    assert!(publisher.prepare_next().await?.is_none());
    assert!(publisher.state().await?.progress.pending.is_none());
    assert!(
        publisher.prepare_next().await?.is_none(),
        "append batch wait triggered compaction"
    );
    assert!(publisher.state().await?.progress.pending.is_none());
    query_all(&ingress, &storage, &expected).await?;
    // A full record selection bypasses the wait and must publish before rewrite.
    let mut publisher = acquire(
        &ingress,
        &storage,
        &budget,
        3,
        Options {
            max_records: 1,
            min_source_bytes: 1 << 20,
            max_batch_wait: Duration::from_secs(60),
            scan_pages_per_group: 1,
            compaction: Some(SmallFileCompaction {
                min_fragments: 2,
                ..options(true).compaction.unwrap()
            }),
            ..options(true)
        },
    )
    .await?;
    assert!(
        publisher.prepare_next().await?.is_none(),
        "incomplete scan triggered compaction"
    );
    publisher
        .prepare_next()
        .await?
        .context("append was not selected")?;
    assert!(!publisher
        .state()
        .await?
        .progress
        .pending
        .clone()
        .unwrap()
        .plan
        .is_compaction());
    publisher.commit_pending().await?;
    publisher.finish_pending().await?;
    verify(&publisher, &expected, 7).await?;
    // Completing an append permits another bounded rewrite.
    let mut compacted = false;
    for _ in 0..4 {
        if publisher.run_once().await?.is_some() {
            compacted = true;
            break;
        }
    }
    assert!(compacted, "append completion never reached compaction");
    verify(&publisher, &expected, 8).await?;
    assert!(
        publisher
            .read_dataset(None)
            .await?
            .manifest()
            .fragments
            .len()
            < 5
    );
    assert_eq!(budget.reserved(), 0);
    drop((publisher, ingress));
    cluster.shutdown().await
}
