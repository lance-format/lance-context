//! Real RF3 publications must reuse mappings only within their manifest generation.
use super::*;
use arrow_array::RecordBatchIterator;
use lance::dataset::{rowids::get_row_id_index, WriteParams};
use std::collections::BTreeSet;

async fn observe(dataset: &Dataset) -> Result<serde_json::Value> {
    assert!(dataset.manifest().uses_stable_row_ids());
    let rows = dataset.count_rows(None).await?;
    let before = dataset.session().metadata_cache_stats().await.num_entries;
    let first = get_row_id_index(dataset)
        .await?
        .context("missing row-ID map")?;
    let second = get_row_id_index(dataset)
        .await?
        .context("missing row-ID map")?;
    let after = dataset.session().metadata_cache_stats().await.num_entries;
    assert!(Arc::ptr_eq(&first, &second));
    let ids = (0..rows as u64).collect::<Vec<_>>();
    let first_addresses = first.get_many(&ids)?;
    assert_eq!(first_addresses, second.get_many(&ids)?);
    let addresses: BTreeSet<u64> = first_addresses
        .into_iter()
        .map(|address| address.expect("a live row must resolve").into())
        .collect();
    assert_eq!(addresses.len(), rows);
    Ok(serde_json::json!({
        "rows": rows,
        "fragments": dataset.fragments().len(),
        "has_etag": dataset.manifest_location().e_tag.is_some(),
        "has_identity": dataset.manifest_location().identity.is_some(),
        "shared_mapping": true,
        "cache_entries_added": after - before,
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_and_managed_rowid_mapping_cache_tracks_publication() -> Result<()> {
    let control = batch(&["a", "b", "c", "d"], &["s", "s", "s", "s"])?;
    let native = Dataset::write(
        RecordBatchIterator::new(vec![Ok(control.clone())], control.schema()),
        "memory://",
        Some(WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await?;
    assert!(native.manifest_location().e_tag.is_some());
    let native_observation = observe(&native).await?;

    let root = tempfile::tempdir()?;
    let path = root.path().join("managed.lance");
    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let ingress = connect(&cluster, binding()).await?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(128 << 20);
    let mut writer = publisher(&ingress, &storage, &budget, 1).await?;
    let open = ProducerOpen {
        bucket: 0,
        slot: 0,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    let mut expected = BTreeMap::new();
    for sequence in 1..=4 {
        let names = [format!("row-{sequence}-a"), format!("row-{sequence}-b")];
        let ids = names.iter().map(String::as_str).collect::<Vec<_>>();
        let (body, _) = append(&ingress, &open, sequence, &ids).await?;
        remember(&mut expected, body)?;
        assert!(writer.run_once().await?.is_some());
    }
    let managed = writer.read_dataset(None).await?;
    assert!(managed.manifest_location().e_tag.is_none());
    assert!(managed.manifest_location().identity.is_some());
    assert_eq!(managed.count_rows(None).await?, 8);
    let managed_observation = observe(&managed).await?;
    let held_map = get_row_id_index(&managed).await?.unwrap();
    assert!(held_map.get(8)?.is_none());
    let (body, _) = append(&ingress, &open, 5, &["row-5-a", "row-5-b"]).await?;
    remember(&mut expected, body)?;
    assert!(writer.run_once().await?.is_some());
    let newer = writer.read_dataset(None).await?;
    assert_eq!(newer.count_rows(None).await?, 10);
    assert_ne!(
        managed.manifest_location().identity,
        newer.manifest_location().identity
    );
    assert!(newer.manifest_location().e_tag.is_none());
    let newer_observation = observe(&newer).await?;
    let newer_map = get_row_id_index(&newer).await?.unwrap();
    assert!(!Arc::ptr_eq(&held_map, &newer_map));
    assert!(newer_map.get(8)?.is_some());
    assert!(held_map.get(8)?.is_none());
    assert!(Arc::ptr_eq(
        &held_map,
        &get_row_id_index(&managed).await?.unwrap()
    ));
    compaction::query_all(&ingress, &storage, &expected).await?;
    println!(
        "{}",
        serde_json::json!({
            "native": native_observation,
            "managed": managed_observation,
            "newer": newer_observation,
            "scope": "Cache identity and complete tiny-value diagnostic, not throughput or large-table memory measurement.",
        })
    );
    drop((managed, newer, writer, ingress));
    cluster.shutdown().await
}
