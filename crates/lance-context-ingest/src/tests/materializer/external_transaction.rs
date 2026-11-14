//! A real Lance transaction may live outside its manifest once metadata grows.
//! This isolates the adapter boundary, not the cause of an unrelated cloud run.
use super::*;
use lance::dataset::{transaction::Operation, transaction::Transaction, CommitBuilder};
use lance_context_wal::replication::publication as p;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn managed_publication_accepts_valid_external_transaction() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("external-transaction.lance");
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
    let mut publisher = publisher(&ingress, &storage, &budget, 1).await?;
    let (body, _) = append(
        &ingress,
        &ProducerOpen {
            bucket: 0,
            slot: 0,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        },
        1,
        &["turn"],
    )
    .await?;
    let prepared = publisher.prepare_next().await?.unwrap();
    let view = publisher.state().await?;
    let pending = view.progress.pending.as_ref().unwrap();
    let base = publisher
        .read_dataset(Some(pending.plan.read_version))
        .await?;
    let target = pending.plan.target_version()?;
    let mut transaction = Transaction::new(
        pending.plan.read_version,
        Operation::Append {
            fragments: serde_json::from_slice(&prepared.descriptors.bytes)?,
        },
        None,
    );
    transaction.uuid = pending.plan.transaction.to_string();
    // Exceed Lance's actual 20-MiB dependency-build threshold. No mock codec,
    // conditional test-only threshold or body ingestion volume substitutes for it.
    transaction.transaction_properties = Some(Arc::new(HashMap::from([(
        "external-metadata-fixture".into(),
        "x".repeat(21 << 20),
    )])));
    let handler = p::lance::ManagedManifest::new(
        Arc::new(cluster.client.manifest_catalog()),
        storage.base.clone(),
        &layout(),
        p::lance::ManifestCommit {
            authority: publisher.authority().clone(),
            incarnation: binding().incarnation,
            transaction: pending.plan.transaction,
            plan_index: pending.index,
            prepared_index: prepared.index,
            descriptor_digest: prepared.descriptors.digest,
            predecessor: view.manifests.as_ref().unwrap().versions[&base.version_id()].identity,
            target,
        },
    )?;
    let result = CommitBuilder::new(Arc::new(base))
        .with_commit_handler(Arc::new(handler))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(transaction.clone())
        .await;
    if let Err(error) = result {
        cluster.shutdown().await?;
        return Err(error).context("valid external transaction rejected by managed publication");
    }
    let selected = publisher.read_dataset(Some(target)).await?;
    assert!(selected.manifest().transaction_file.is_some());
    assert!(selected.manifest().transaction_section.is_none());
    assert_eq!(selected.read_transaction().await?, Some(transaction));
    publisher.commit_pending().await?.unwrap();
    publisher.finish_pending().await?.unwrap();
    let mut expected = BTreeMap::new();
    remember(&mut expected, body)?;
    verify(&publisher, &expected, target).await?;
    drop(publisher);
    cluster.shutdown().await
}
