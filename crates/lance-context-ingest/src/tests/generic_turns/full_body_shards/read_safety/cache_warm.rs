use super::*;
use crate::streaming::{
    key_cache::{Config, Scope, WarmLimits},
    AppliedKeyCache,
};
use arrow_array::RecordBatchIterator;
use std::sync::atomic::AtomicBool;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cache_warm_rejects_another_dataset_at_the_selected_version() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let input = body("same-session", "foreign-id", 73)?;
    let spec = TurnIngestionSpec::new(
        input.schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let shard = Shard::for_session(&layout, &binding(), "same-session")?;
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let storage = fixture::storage(root.path(), "bodies.lance").await?;
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        layout.clone(),
        binding(),
        storage.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &fixture::limits(),
        shard,
        &spec,
    )
    .await?;
    let physical = Physical {
        binding: binding(),
        shard,
        storage,
    };
    let (service, runtime) = physical
        .owner(&c, &layout, &spec, MergeMemoryBudget::new(32 << 20))
        .await?;
    let selected = service.selected_full_body(DEADLINE).await?;
    assert_eq!(selected.rows, Some(0));
    let foreign = lance::Dataset::write(
        RecordBatchIterator::new(vec![Ok(input.clone())], input.schema()),
        root.path().join("foreign.lance").to_str().unwrap(),
        None,
    )
    .await?;
    assert_eq!(foreign.version().version, selected.manifest.version);
    let cache = AppliedKeyCache::open(Config {
        scopes: vec![Scope::new("turns", shard.bucket, &physical.binding, &spec)?],
        ram_keys: 16,
        disk_dir: None,
        block_cache_bytes: 0,
        bloom_bits_per_key: 10.0,
    })?;
    let limits = WarmLimits {
        batch_rows: 8,
        io_buffer_bytes: 1 << 20,
        bytes_per_second: 0,
    };
    let stop = AtomicBool::new(false);
    let keys = cache.batch_keys(shard.bucket, &input, spec.primary_key())?;
    let warm = cache
        .warm(
            shard.bucket,
            &selected,
            &foreign,
            spec.primary_key(),
            limits,
            &stop,
        )
        .await;
    assert_eq!(cache.contains_many(&keys)?, vec![false]);
    assert!(warm.is_err());

    let handler = p::lance::PinnedManifest::new(
        physical.storage.base.clone(),
        selected.cluster,
        selected.layout,
        selected.table,
        selected.incarnation,
        &selected.manifest,
    )?;
    let native = lance::dataset::builder::DatasetBuilder::from_uri(&selected.uri)
        .with_version(selected.manifest.version)
        .with_commit_handler(Arc::new(handler))
        .load()
        .await?;
    cache
        .warm(
            shard.bucket,
            &selected,
            &native,
            spec.primary_key(),
            limits,
            &stop,
        )
        .await?;
    let mut stale_selection = selected;
    stale_selection.manifest.transaction = Uuid::new_v4();
    assert!(cache
        .warm(
            shard.bucket,
            &stale_selection,
            &native,
            spec.primary_key(),
            limits,
            &stop
        )
        .await
        .is_err());
    drop((runtime, service));
    c.shutdown().await
}
