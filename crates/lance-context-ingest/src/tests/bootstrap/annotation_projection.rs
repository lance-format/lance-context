use super::*;
use crate::{
    bootstrap::annotations::{self as handoff, projection, Snapshot},
    tests::{annotation_publication, annotation_runtime, annotations as requests},
};
use futures::TryStreamExt;
use lance::{dataset::CommitBuilder, index::DatasetIndexExt};
use lance_context_core::annotation_staging::Limits;

fn files(
    root: &std::path::Path,
) -> Result<std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>> {
    let mut result = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            result.extend(files(&entry.path())?);
        } else {
            result.insert(entry.path(), std::fs::read(entry.path())?);
        }
    }
    Ok(result)
}

fn limits() -> Limits {
    Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    }
}

async fn store(root: &std::path::Path, name: &str) -> Result<Storage> {
    Storage::resolve(
        root.join(name).to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await
}

fn scope(layout: &Layout) -> Result<Scope> {
    Ok(Scope {
        cluster: layout.cluster,
        layout: layout.fingerprint()?,
        table: binding().table,
        destination: binding().destination,
        incarnation: binding().incarnation,
        contract: crate::annotations::Service::contract(&binding(), &limits())?,
    })
}

async fn source(storage: &Storage, base: &Dataset) -> Result<projection::Source> {
    Ok(projection::Source {
        uri: storage.uri.clone(),
        storage_options: Default::default(),
        snapshot: Snapshot {
            version: base.version_id(),
            transaction: base.read_transaction().await?.unwrap().uuid.parse()?,
        },
    })
}

#[tokio::test]
async fn annotation_projection_resumes_each_native_boundary_and_keeps_later_updates() -> Result<()>
{
    for stop_after in 0..=3 {
        let root = tempfile::tempdir()?;
        let bodies = store(root.path(), "bodies.lance").await?;
        let target = store(root.path(), "annotations.lance").await?;
        let base = annotations::original(&bodies).await?;
        let source_files = files(std::path::Path::new(&bodies.uri))?;
        let source = source(&bodies, &base).await?;
        let layout = annotation_publication::one_shard();
        let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
        if stop_after > 0 {
            let scope = scope(&layout)?;
            let identity = identity();
            let marker = serde_json::to_string(&(
                &scope,
                &identity,
                &target.uri,
                &source.uri,
                &source.snapshot,
            ))?;
            let initial = projection::project(
                &base,
                &target,
                projection::schema(&base, &scope, &identity, &marker)?,
                &marker,
                1 << 20,
            )
            .await?;
            if stop_after > 1 {
                let indexed = projection::index(&target, &initial, &marker).await?;
                if stop_after > 2 {
                    let snapshot = Snapshot {
                        version: 2,
                        transaction: indexed.read_transaction().await?.unwrap().uuid.parse()?,
                    };
                    let marker = serde_json::to_string(&(&scope, &identity, &snapshot))?;
                    CommitBuilder::new(Arc::new(indexed.clone()))
                        .with_commit_handler(Arc::new(handoff::ExactVersion(3)))
                        .with_max_retries(0)
                        .with_skip_auto_cleanup(true)
                        .execute(handoff::prepare(&indexed, &scope, &identity, &marker)?)
                        .await?;
                }
            }
            assert_eq!(Dataset::open(&target.uri).await?.version_id(), stop_after);
        }
        projection::provision(
            c.client.clone(),
            layout.clone(),
            binding(),
            target.clone(),
            identity(),
            &source,
            limits(),
            1 << 20,
        )
        .await?;
        let selected = Dataset::open(&target.uri).await?;
        assert_eq!(selected.version_id(), 3);
        assert_eq!(selected.schema().fields.len(), 5);
        assert!(selected.schema().field("content").is_none());
        assert_eq!(selected.count_rows(None).await?, 2);
        let index = selected.load_indices().await?;
        assert_eq!(index.len(), 1);
        assert!(!index[0].uuid.is_nil());
        let (service, runtime) = annotation_runtime::owner(&c, &target, &limits()).await?;
        let (stop, task) = annotation_runtime::run_owner(runtime);
        let producer = requests::producer(&service, 1).await?;
        let bytes = requests::wire(&limits(), &["T"], 1, Some("reviewed"), 0, None)?;
        let request = requests::request(&service, &producer, 1, bytes).await?;
        let receipt = service.submit(&request, Duration::from_secs(30)).await?;
        annotation_runtime::drained(&c, &service, receipt.index).await?;
        let materialized = annotation_publication::read(&c, &target, None).await?;
        let kept = materialized.load_indices().await?;
        assert_eq!(kept[0].uuid, index[0].uuid);
        assert!(materialized.fragments().iter().all(|f| kept[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .contains(f.id as u32)));
        let before = view(&c, binding().table).await?;
        projection::provision(
            c.client.clone(),
            layout.clone(),
            binding(),
            target.clone(),
            identity(),
            &source,
            limits(),
            1 << 20,
        )
        .await?;
        assert_eq!(view(&c, binding().table).await?, before);
        let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
        assert_eq!(current.status.as_deref(), Some("reviewed"));
        assert_eq!(current.score, Some(0.5));
        assert!(current.labels.is_empty());
        let replay = service.submit(&request, Duration::from_secs(30)).await?;
        assert_eq!(replay.index, receipt.index);
        let mut changed = source.clone();
        changed.snapshot.transaction = Uuid::new_v4();
        assert!(projection::provision(
            c.client.clone(),
            layout,
            binding(),
            target.clone(),
            identity(),
            &changed,
            limits(),
            1 << 20
        )
        .await
        .is_err());
        assert_eq!(view(&c, binding().table).await?, before);
        stop.send(()).unwrap();
        task.await??;
        c.shutdown().await?;
        let original = Dataset::open(&bodies.uri).await?;
        assert_eq!(original.version_id(), base.version_id());
        assert_eq!(original.manifest().fragments, base.manifest().fragments);
        assert_eq!(files(std::path::Path::new(&bodies.uri))?, source_files);
        let original_rows = original
            .scan()
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        let expected_rows = base
            .scan()
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(original_rows, expected_rows);
    }
    Ok(())
}

#[tokio::test]
async fn annotation_projection_rejects_changed_source_and_foreign_destination() -> Result<()> {
    let root = tempfile::tempdir()?;
    let bodies = store(root.path(), "bodies.lance").await?;
    let target = store(root.path(), "annotations.lance").await?;
    let mut base = annotations::original(&bodies).await?;
    let source = source(&bodies, &base).await?;
    let layout = annotation_publication::one_shard();
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    // A foreign destination is never overwritten, even if its rows match.
    let nested = store(root.path(), "bodies.lance/data/annotations.lance").await?;
    let error = projection::provision(
        c.client.clone(),
        layout.clone(),
        binding(),
        nested,
        identity(),
        &source,
        limits(),
        1 << 20,
    )
    .await
    .err()
    .context("overlapping roots accepted")?;
    assert!(error.to_string().contains("storage roots overlap"));
    let foreign = annotations::original(&target).await?;
    let foreign_files = files(std::path::Path::new(&target.uri))?;
    assert!(projection::provision(
        c.client.clone(),
        layout.clone(),
        binding(),
        target.clone(),
        identity(),
        &source,
        limits(),
        1 << 20
    )
    .await
    .is_err());
    let current = Dataset::open(&target.uri).await?;
    assert_eq!(current.version_id(), foreign.version_id());
    assert_eq!(files(std::path::Path::new(&target.uri))?, foreign_files);
    let fresh = store(root.path(), "fresh.lance").await?;
    base.update_metadata([("source", "advanced")]).await?;
    assert!(projection::provision(
        c.client.clone(),
        layout,
        binding(),
        fresh.clone(),
        identity(),
        &source,
        limits(),
        1 << 20
    )
    .await
    .is_err());
    assert!(Dataset::open(&fresh.uri).await.is_err());
    c.shutdown().await
}

#[tokio::test]
async fn annotation_projection_streams_multiple_batches_without_retaining_bodies() -> Result<()> {
    let root = tempfile::tempdir()?;
    let bodies = store(root.path(), "bodies.lance").await?;
    let target = store(root.path(), "annotations.lance").await?;
    let rows = 65_536;
    let schema = Arc::new(rollout_schema());
    let batches = (0..rows).step_by(256).map(|start| {
        let ids = (start..start + 256)
            .map(|n| format!("turn-{n:06}"))
            .collect::<Vec<_>>();
        batch(
            &ids.iter().map(String::as_str).collect::<Vec<_>>(),
            &vec!["session"; 256],
        )
        .map_err(|e| arrow_schema::ArrowError::ExternalError(e.into()))
    });
    let base = Dataset::write(
        RecordBatchIterator::new(batches, schema),
        &bodies.uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 1024,
            ..Default::default()
        }),
    )
    .await?;
    assert_eq!(base.fragments().len(), rows / 1024);
    let scope = scope(&annotation_publication::one_shard())?;
    let projected = projection::project(
        &base,
        &target,
        projection::schema(&base, &scope, &identity(), "stream-test")?,
        "stream-test",
        1 << 20,
    )
    .await?;
    assert_eq!(projected.count_rows(None).await?, rows);
    assert_eq!(projected.schema().fields.len(), 5);
    let batches = projected
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert!(
        batches
            .iter()
            .map(RecordBatch::get_array_memory_size)
            .sum::<usize>()
            > 1 << 20
    );
    let mut next = 0;
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        for id in ids.iter() {
            assert_eq!(id.unwrap(), format!("turn-{next:06}"));
            next += 1;
        }
    }
    assert_eq!(next, rows);
    assert_eq!(Dataset::open(&bodies.uri).await?.version_id(), 1);
    Ok(())
}

#[tokio::test]
async fn annotation_projection_refuses_an_oversized_batch_without_publishing() -> Result<()> {
    let root = tempfile::tempdir()?;
    let bodies = store(root.path(), "bodies.lance").await?;
    let target = store(root.path(), "annotations.lance").await?;
    let row = batch(&[&"T".repeat(512 << 10)], &["session"])?;
    let base = Dataset::write(
        RecordBatchIterator::new(vec![Ok(row.clone())], row.schema()),
        &bodies.uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await?;
    let scope = scope(&annotation_publication::one_shard())?;
    let error = projection::project(
        &base,
        &target,
        projection::schema(&base, &scope, &identity(), "oversized")?,
        "oversized",
        1 << 20,
    )
    .await
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("projection batch exceeds workspace"));
    assert!(Dataset::open(&target.uri).await.is_err());
    assert_eq!(
        Dataset::open(&bodies.uri).await?.version_id(),
        base.version_id()
    );
    Ok(())
}

#[cfg(feature = "service")]
#[tokio::test]
async fn annotation_projection_application_config_routes_bootstrap_and_rejects_both_sources(
) -> Result<()> {
    use crate::application::{self, config};
    let root = tempfile::tempdir()?;
    let bodies = store(root.path(), "bodies.lance").await?;
    let target = store(root.path(), "annotations.lance").await?;
    let base = annotations::original(&bodies).await?;
    let source = source(&bodies, &base).await?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let mut config = c.application.clone();
    config.tables.push(config::Table {
        binding: binding(),
        uri: target.uri.clone(),
        storage_options: Default::default(),
        bootstrap: identity(),
    });
    config.annotations = Some(config::Annotations {
        owner_url: "http://unused-owner:8080".into(),
        apply_memory_bytes: 8 << 20,
        staging: limits(),
        source: Some(source.snapshot.clone()),
        projection: Some(source),
        exclusive_reclamation: None,
    });
    assert!(application::annotations::provision(&config).await.is_err());
    assert!(Dataset::open(&target.uri).await.is_err());
    config.annotations.as_mut().unwrap().source = None;
    let path = root.path().join("project.json");
    std::fs::write(&path, serde_json::to_vec(&config)?)?;
    let config = application::Config::read(&path)?;
    application::annotations::provision(&config).await?;
    let before = view(&c, binding().table).await?;
    application::annotations::provision(&config).await?;
    assert_eq!(view(&c, binding().table).await?, before);
    assert_eq!(Dataset::open(&target.uri).await?.schema().fields.len(), 5);
    c.shutdown().await
}
