//! Real RF3 narrow preparation with a 512-byte catalog budget and concurrent writes.
use super::*;
use crate::annotation_publication as publication;
use crate::materializer;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_narrow_reference_unknown_prepare_recovers_concurrent_suffix() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), exercise(false)).await?
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_narrow_reference_unknown_selection_recovers_original_rewrite() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), exercise(true)).await?
}

async fn cluster(root: &std::path::Path) -> Result<network::Cluster> {
    network::Cluster::open_signed_with_limits(
        root,
        annotation_publication::one_shard(),
        lance_context_wal::replication::Limits::default().max_records,
        512,
    )
    .await
}

async fn exercise(selected: bool) -> Result<()> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("narrow-compaction-metadata-");
    let root = match std::env::var("INGEST_TEST_ARTIFACTS") {
        Ok(parent) => builder.tempdir_in(parent)?,
        Err(_) => builder.tempdir()?,
    }
    .keep();
    let mut c = cluster(&root).await?;
    let narrow = storage(&root, "annotations.lance").await?;
    let body = storage(&root, "body.lance").await?;
    let (last, last_result) = fragmented(&c, &narrow, &body).await?;
    let base = annotation_publication::read(&c, &narrow, None).await?;
    let reference = BodySnapshot::read(&base).await?;
    let proxy = Proxy::open(&c, &root).await?;
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let ingress = mixed::connect_ingress(
        c.client.clone(),
        annotation_publication::one_shard(),
        compact_binding(),
        &limits(),
        MergeMemoryBudget::new(8 << 20),
    )
    .await?;
    let mut publisher = materializer::Publisher::acquire(
        ingress,
        narrow.clone(),
        Options {
            max_source_bytes: 1 << 20,
            min_source_bytes: 1 << 20,
            max_records: 4,
            compaction: Some(materializer::SmallFileCompaction {
                target_file_bytes: 64 << 10,
                min_fragments: 2,
            }),
            ..Default::default()
        },
        MergeMemoryBudget::new(8 << 20),
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: annotation_publication::state(&c).await?.authority.epoch,
        },
    )
    .await?;
    let (a, driver) = Service::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        compact_binding(),
        &narrow,
        body.clone(),
        limits(),
        MergeMemoryBudget::new(8 << 20),
        MergeMemoryBudget::new(8 << 20),
    )
    .await?;
    let (stop, task) = run(driver);
    let plan =
        materializer::compaction::plan(&publisher, &*annotation_publication::state(&c).await?)
            .await?
            .unwrap();
    publisher.record_plan(plan).await?;
    let preparing = tokio::spawn(async move { publisher.prepare_next().await });
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
    let create = request(&a, &q, 1, encoded(create(&["D"], &["s"], &[4])?)?).await?;
    let updated = a.submit(&update, Duration::from_secs(10)).await?;
    let created = a.submit(&create, Duration::from_secs(10)).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "D", "s", 4, "pending", None).await?;
    let state = annotation_publication::state(&c).await?;
    let pending = state.progress.pending.clone().unwrap();
    let prepared = state.progress.prepared.clone().unwrap();
    assert_eq!(prepared.descriptors.format, 5);
    prepared.descriptors.validate(512)?;
    let selection = publication::Selection {
        table: compact_binding().table,
        incarnation: compact_binding().incarnation,
        pending: (*pending).clone(),
    };
    let expected =
        publication::mixed::resolve(&base, &prepared.descriptors, &selection, &narrow, &body)
            .await?;
    assert!(matches!(expected.operation, Operation::Rewrite { .. }));
    use prost::Message;
    let inline = |transaction: &Transaction| {
        p::Descriptors::lance_annotations(Bytes::from(
            lance_table::format::pb::Transaction::try_from(transaction)?.encode_to_vec(),
        ))
    };
    assert_eq!(
        publication::mixed::resolve(&base, &inline(&expected)?, &selection, &narrow, &body).await?,
        expected
    );
    for mutation in 0..5 {
        let mut foreign = expected.clone();
        match mutation {
            0 => foreign.read_version += 1,
            1 => foreign.uuid = Uuid::new_v4().to_string(),
            2 => {
                Arc::make_mut(foreign.transaction_properties.as_mut().unwrap())
                    .remove(BODY_SNAPSHOT_PROPERTY);
            }
            3 => {
                Arc::make_mut(foreign.transaction_properties.as_mut().unwrap())
                    .remove("lance-context.annotation-cut");
            }
            4 => {
                if let Operation::Rewrite { groups, .. } = &mut foreign.operation {
                    groups[0].old_fragments[0].id += 1000;
                }
            }
            _ => unreachable!(),
        }
        assert!(
            publication::mixed::resolve(&base, &inline(&foreign)?, &selection, &narrow, &body)
                .await
                .is_err()
        );
    }

    assert_eq!(
        expected.transaction_properties,
        base.read_transaction()
            .await?
            .unwrap()
            .transaction_properties
    );
    // The patch-only route must never accept a paired maintenance transaction.
    assert!(
        publication::resolve_transaction(&base, &prepared.descriptors, &selection, &narrow)
            .await
            .is_err()
    );
    let metadata = root
        .join("annotations.lance/_annotation_preparations")
        .join(selection.incarnation.to_string())
        .join(pending.plan.transaction.to_string());
    let originals = std::fs::read_dir(&metadata)?
        .map(|e| {
            let path = e?.path();
            Ok((path.clone(), std::fs::read(path)?))
        })
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(originals.len(), 1);
    assert!(originals[0].1.len() > 512);
    proxy.release.notify_one();
    let error = preparing.await?.unwrap_err();
    proxy.restore_observations();
    assert!(format!("{error:#}").contains("outcome unknown"));
    c.client = direct;
    if selected {
        let commit = p::lance::ManifestCommit {
            authority: state.authority,
            incarnation: selection.incarnation,
            transaction: pending.plan.transaction,
            plan_index: pending.index,
            prepared_index: prepared.index,
            descriptor_digest: prepared.descriptors.digest,
            predecessor: state.manifests.as_ref().unwrap().versions[&pending.plan.read_version]
                .identity,
            target: pending.plan.target_version()?,
        };
        assert!(publication::mixed::publish(
            Arc::new(annotation_publication::LostReply(Arc::new(
                c.client.manifest_catalog()
            ))),
            &annotation_publication::one_shard(),
            &narrow,
            &body,
            commit,
            &selection,
            &prepared.descriptors,
            base.clone()
        )
        .await
        .is_err());
    }
    let stopped = annotation_publication::state(&c).await?;
    assert_eq!(stopped.progress.prepared.as_ref(), Some(&prepared));
    assert_eq!(stopped.progress.checkpoint, state.progress.checkpoint);
    assert_eq!(
        stopped
            .manifests
            .as_ref()
            .unwrap()
            .versions
            .contains_key(&pending.plan.target_version()?),
        selected
    );
    for (path, bytes) in &originals {
        assert_eq!(&std::fs::read(path)?, bytes);
    }
    stop.send(()).unwrap();
    task.await??;
    drop((a, proxy));
    c.shutdown().await?;

    let c = cluster(&root).await?;
    let narrow = storage(&root, "annotations.lance").await?;
    let body = storage(&root, "body.lance").await?;
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    annotation_runtime::drained(&c, &a, created.index).await?;
    let original =
        annotation_publication::read(&c, &narrow, Some(pending.plan.target_version()?)).await?;
    assert_eq!(original.read_transaction().await?, Some(expected));
    assert_eq!(BodySnapshot::read(&original).await?, reference);
    for (path, bytes) in &originals {
        assert_eq!(&std::fs::read(path)?, bytes);
    }
    for (input, outcome) in [
        (&last, &last_result),
        (&update, &updated),
        (&create, &created),
    ] {
        assert_eq!(&a.submit(input, Duration::from_secs(10)).await?, outcome);
    }
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "D", "s", 4, "pending", None).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
