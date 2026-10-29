//! Bootstrap receipts must not permanently pin superseded initial data.
use super::*;
use crate::bootstrap::annotations::{self as handoff, Snapshot};
use std::collections::BTreeSet;

fn references(dataset: &Dataset) -> BTreeSet<String> {
    dataset
        .manifest()
        .fragments
        .iter()
        .flat_map(|f| f.referenced_lance_files())
        .map(|file| file.path.clone())
        .collect()
}

async fn collecting_owner(
    c: &network::Cluster,
    storage: &Storage,
    limits: &Limits,
) -> Result<(Service, Runtime)> {
    let (service, owner) = Runtime::connect(
        c.client.clone(),
        one_shard(),
        fixture_binding(),
        storage.clone(),
        Configuration {
            limits: limits.clone(),
            upload_budget: MergeMemoryBudget::new(32 << 20),
            apply_budget: MergeMemoryBudget::new(16 << 20),
            publication_budget: MergeMemoryBudget::new(32 << 20),
            publication: Options {
                retained_manifest_versions: 4,
                ..options()
            },
        },
        acquire(c).await?,
    )
    .await?;
    Ok((service, collector(owner)?))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_collection_releases_bootstrap_data_after_history_and_readers() -> Result<()> {
    collect_bootstrap(true).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_collection_keeps_bootstrap_data_inside_retention_window() -> Result<()> {
    collect_bootstrap(false).await
}

async fn collection_passes(service: &Service) -> Result<()> {
    let passes = service.reclamation_observation().passes;
    tokio::time::timeout(Duration::from_secs(30), async {
        while service.reclamation_observation().passes < passes + 3 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;
    Ok(())
}

async fn collect_bootstrap(hold_reader: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let storage = fixture_storage(root.path()).await?;
    let limits = fixture_limits();
    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    let base = fragmented_base(&storage, 8).await?;
    let original = references(&base);
    let index_id = base.load_indices().await?[0].uuid;
    assert_eq!(original.len(), 8);
    let snapshot = Snapshot {
        version: base.version_id(),
        transaction: Uuid::parse_str(&base.read_transaction().await?.unwrap().uuid)?,
    };
    let identity = crate::tests::bootstrap::identity();
    handoff::provision(
        c.client.clone(),
        one_shard(),
        fixture_binding(),
        storage.clone(),
        identity.clone(),
        snapshot.clone(),
        limits.clone(),
    )
    .await?;
    drop(base);
    let bootstrap = state(&c).await?.manifests.as_ref().unwrap().initial_version;
    let data = root.path().join("turns.lance/data");
    let (service, owner) = collecting_owner(&c, &storage, &limits).await?;
    let (stop, task) = runtime::run_owner(owner);
    // Capture before the first update, without resolving/warming the old read.
    let mut old = Some(service.capture_read_for_test("t00000000").await?);
    assert_eq!(old.as_ref().unwrap().base_version, bootstrap);
    let producer = producer(&service, 1).await?;
    let mut last = None;
    for sequence in 1..=4 {
        let body = wire(
            &limits,
            &["t00000000"],
            1,
            Some("reviewed"),
            1,
            Some(sequence as f32),
        )?;
        let request = request(&service, &producer, sequence, body.clone()).await?;
        let completed = service.submit(&request, Duration::from_secs(30)).await?;
        runtime::drained(&c, &service, completed.index).await?;
        if sequence == 1 {
            tokio::time::timeout(Duration::from_secs(30), async {
                loop {
                    let selected = read(&c, &storage, None).await?;
                    if selected.manifest().fragments.len() == 1 {
                        assert!(original.is_disjoint(&references(&selected)));
                        return Ok::<_, anyhow::Error>(());
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await??;
            // No history retirement yet: a suffix starts at bootstrap + 1
            // even while the initial snapshot is within the configured window.
            let view = state(&c).await?;
            let manifests = view.manifests.as_ref().unwrap();
            assert_eq!(manifests.history_boundary().retained_from, bootstrap + 1);
            if !hold_reader {
                drop(old.take());
            }
            collection_passes(&service).await?;
            assert!(original.iter().all(|p| data.join(p).exists()));
        }
        last = Some((body, completed));
    }
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let view = state(&c).await?;
            let manifests = view.manifests.as_ref().unwrap();
            if manifests.history_boundary().retained_from > bootstrap + 2 {
                // Every retained snapshot other than the receipt has replaced
                // the initial data. Only a captured reader could still need it.
                for version in manifests.versions.keys().filter(|v| **v != bootstrap) {
                    let selected = read(&c, &storage, Some(*version)).await?;
                    assert!(original.is_disjoint(&references(&selected)));
                }
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await??;
    if let Some(old) = old {
        collection_passes(&service).await?;
        assert!(original.iter().all(|p| data.join(p).exists()));
        let old_value = old.resolve().await?.unwrap();
        assert_eq!(old_value.status.as_deref(), Some("pending"));
        assert_eq!(old_value.score, None);
        drop(old);
    }
    tokio::time::timeout(Duration::from_secs(30), async {
        while original.iter().any(|p| data.join(p).exists()) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .context("bootstrap-only data did not drain after reader release")?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    c.shutdown().await?;

    // Fresh replicas and Session: cached pages must not hide missing files.
    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    let storage = fixture_storage(root.path()).await?;
    let before = state(&c).await?;
    handoff::provision(
        c.client.clone(),
        one_shard(),
        fixture_binding(),
        storage.clone(),
        identity,
        snapshot,
        limits.clone(),
    )
    .await?;
    assert_eq!(state(&c).await?, before);
    // The adoption receipt's exact manifest and transaction remain readable.
    let receipt = read(&c, &storage, Some(bootstrap)).await?;
    assert!(receipt.read_transaction().await?.is_some());
    let (service, owner) = collecting_owner(&c, &storage, &limits).await?;
    let (stop, task) = runtime::run_owner(owner);
    let producer = crate::tests::annotations::producer(&service, 1).await?;
    let (body, completed) = last.unwrap();
    let retry = request(&service, &producer, 4, body).await?;
    assert_eq!(
        service.submit(&retry, Duration::from_secs(30)).await?,
        completed
    );
    let value = service
        .get("t00000000", Duration::from_secs(30))
        .await?
        .unwrap();
    assert_eq!(value.status.as_deref(), Some("reviewed"));
    assert_eq!(value.score, Some(4.0));
    let selected = read(&c, &storage, None).await?;
    let indices = selected.load_indices().await?;
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].uuid, index_id);
    assert!(indices[0]
        .fragment_bitmap
        .as_ref()
        .unwrap()
        .contains(selected.manifest().fragments[0].id as u32));
    let batches = selected
        .scan()
        .project(&["body"])?
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 16);
    for batch in batches {
        for body in batch
            .column(0)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap()
            .iter()
            .flatten()
        {
            assert_eq!(body, vec![0x5a; 1024].as_slice());
        }
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
