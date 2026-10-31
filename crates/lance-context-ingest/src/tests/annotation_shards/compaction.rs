//! Nonzero shards must compact both selected datasets and preserve request results.
use super::*;
use lance::dataset::builder::DatasetBuilder;
use lance_context_core::annotation_staging::BodySnapshot;

async fn selected(c: &network::Cluster, layout: &Layout, shard: &Physical) -> Result<Dataset> {
    let handler = p::lance::ManagedManifest::reader(
        Arc::new(c.client.manifest_catalog()),
        shard.narrow.base.clone(),
        layout,
        shard.binding.table,
        shard.binding.incarnation,
    )?;
    Ok(DatasetBuilder::from_uri(&shard.narrow.uri)
        .with_commit_handler(Arc::new(handler))
        .load()
        .await?)
}

async fn compact_owner(
    c: &network::Cluster,
    layout: &Arc<Layout>,
    shard: &Physical,
) -> Result<(Service, Runtime)> {
    Runtime::connect_mixed_shard(
        c.client.clone(),
        layout.clone(),
        shard.binding.clone(),
        shard.narrow.clone(),
        shard.body.clone(),
        Configuration {
            limits: fixture::limits(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 4,
                max_batch_wait: Duration::from_millis(20),
                compaction: Some(crate::materializer::SmallFileCompaction {
                    target_file_bytes: 256 << 10,
                    min_fragments: 2,
                }),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: shard.view(c).await?.authority.epoch,
        },
        shard.shard,
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn nonzero_shard_compacts_narrow_and_body_and_recovers_request_results() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut definition = (*layout()).clone();
    definition.buckets = vec![10, 20];
    let layout = Arc::new(definition);
    // Force large rewrite metadata through the native preparation reference,
    // while keeping real RF3 publication and small physical data files.
    let c = network::Cluster::open_signed_with_limits(
        root.path(),
        layout.clone(),
        lance_context_wal::replication::Limits::default().max_records,
        512,
    )
    .await?;
    let other = Physical::provision(&c, &layout, root.path(), 0).await?;
    let other_before = other.view(&c).await?;
    let shard = Physical::provision_with_binding(
        &c,
        &layout,
        root.path(),
        1,
        TableBinding {
            max_record_bytes: 256 << 10,
            ..binding()
        },
    )
    .await?;
    let turns: Vec<_> = (0..1000)
        .map(|n| format!("compact-{n}"))
        .filter(|id| Shard::for_turn(&layout, &shard.binding, id).unwrap() == shard.shard)
        .take(4)
        .collect();
    assert_eq!(turns.len(), 4);
    let (service, owner) = shard.owner(&c, &layout, Duration::from_millis(20)).await?;
    let (stop, task) = annotation_runtime::run_owner(owner);
    let producer = producer(&service).await?;
    for (offset, id) in turns.iter().enumerate() {
        let prepared = request(
            &service,
            &producer,
            offset as u64 + 1,
            shard.encode(fixture::create(&[id], &["same-session"], &[7])?)?,
        )
        .await?;
        let ack = service.submit(&prepared, DEADLINE).await?;
        shard.drained(&c, &layout, &service, ack.index).await?;
    }
    // Retain a field update's exact request/result across physical maintenance.
    let patch = request(
        &service,
        &producer,
        5,
        shard.patch(&turns[0], Some("reviewed"), Some(0.9))?,
    )
    .await?;
    let ack = service.submit(&patch, DEADLINE).await?;
    shard.drained(&c, &layout, &service, ack.index).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    let before = shard.view(&c).await?;
    let base = selected(&c, &layout, &shard).await?;
    assert!(base.manifest().fragments.len() >= 4);
    let reference = BodySnapshot::read(&base).await?.unwrap();
    let body = mixed::load_body(
        &shard.body,
        &reference,
        &shard.binding.incarnation.to_string(),
    )
    .await?;
    assert_eq!(body.manifest().fragments.len(), 4);

    let (service, owner) = compact_owner(&c, &layout, &shard).await?;
    // Reproduce a retained legacy maintenance plan whose scheduling cursor is
    // the layout's first group. Recovery must finish that same transaction;
    // creating correctly scoped new plans alone cannot unblock this table.
    let maintenance = p::Plan {
        transaction: Uuid::new_v4(),
        read_version: before.progress.minimum_read_version,
        sources: BTreeMap::new(),
        next_group: 10,
    };
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Plan(p::PlanRequest {
            authority: shard.view(&c).await?.authority,
            plan: maintenance.clone(),
        }))))
        .await?;
    assert!(matches!(
        response.reply,
        Reply::Commit(Outcome::Committed(_))
    ));
    assert_eq!(
        shard
            .view(&c)
            .await?
            .progress
            .pending
            .as_ref()
            .unwrap()
            .plan,
        maintenance
    );
    let (stop, mut task) = annotation_runtime::run_owner(owner);
    let drained = async {
        loop {
            let base = selected(&c, &layout, &shard).await?;
            let reference = BodySnapshot::read(&base).await?.unwrap();
            let body = mixed::load_body(
                &shard.body,
                &reference,
                &shard.binding.incarnation.to_string(),
            )
            .await?;
            if base.manifest().fragments.len() == 1 && body.manifest().fragments.len() == 1 {
                shard.drained(&c, &layout, &service, ack.index).await?;
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::select! {
        result = drained => result?,
        result = &mut task => {
            result??;
            bail!("maintenance owner exited before compacting both datasets");
        }
    }
    assert_eq!(
        shard.view(&c).await?.progress.checkpoint,
        before.progress.checkpoint
    );
    assert_eq!(other.view(&c).await?, other_before);
    assert_eq!(service.submit(&patch, DEADLINE).await?, ack);
    fixture::check(
        &service,
        &turns[0],
        "same-session",
        7,
        "reviewed",
        Some(0.9),
    )
    .await?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    drop((base, body));
    c.shutdown().await?;

    let c = network::Cluster::open_signed_with_limits(
        root.path(),
        layout.clone(),
        lance_context_wal::replication::Limits::default().max_records,
        512,
    )
    .await?;
    let shard = Physical {
        narrow: fixture::storage(root.path(), "shard-1.lance").await?,
        body: fixture::storage(root.path(), "body-1.lance").await?,
        ..shard
    };
    let (service, owner) = compact_owner(&c, &layout, &shard).await?;
    let (stop, task) = annotation_runtime::run_owner(owner);
    assert_eq!(service.submit(&patch, DEADLINE).await?, ack);
    for (offset, id) in turns.iter().enumerate() {
        let (status, score) = if offset == 0 {
            ("reviewed", Some(0.9))
        } else {
            ("pending", None)
        };
        fixture::check(&service, id, "same-session", 7, status, score).await?;
    }
    assert_eq!(other.view(&c).await?, other_before);
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    c.shutdown().await
}
