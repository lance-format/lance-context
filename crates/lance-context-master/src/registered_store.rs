//! Maintenance of registered tables using their persisted schema and indexes.

use std::{collections::BTreeSet, sync::Arc};

use arrow_schema::Schema;
use chrono::Utc;
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance::Result;
use lance_context_core::{KeyIndexType, StorageBase, StorageBaseOptions};
use lance_index::optimize::OptimizeOptions;

use crate::{state::MasterState, stats_store::StatRow};

pub(crate) async fn open(state: &MasterState, uri: &str) -> Result<StorageBase> {
    let dataset = StorageBase::load_with_options(uri, None, state.rollout_session.clone()).await?;
    let schema = Arc::new(Schema::from(dataset.schema()));
    let keys: Vec<_> = schema
        .fields()
        .iter()
        .filter(|field| {
            field
                .metadata()
                .get("lance-schema:unenforced-primary-key")
                .is_some_and(|v| v == "true")
        })
        .collect();
    if keys.len() != 1 {
        return Err(lance::Error::invalid_input(
            "maintenance requires one declared primary key",
        ));
    }
    let key_column = keys[0].name().to_string();
    let mut options = StorageBaseOptions::new(schema, key_column);
    options.session = state.rollout_session.clone();
    options.key_index_type = state.config.key_index_type;
    options.pending_generations_max = Some(0);
    StorageBase::from_dataset(dataset, options).await
}

#[derive(Default)]
pub(crate) struct IndexWork {
    create_key: bool,
    extend: Vec<String>,
}

impl IndexWork {
    pub(crate) fn needed(&self) -> bool {
        self.create_key || !self.extend.is_empty()
    }
}

/// Inspect index metadata, including Lance's fragment-reuse mapping and the union
/// of same-name deltas. This never scans table rows or observes worker WAL.
pub(crate) async fn index_work(store: &StorageBase, key_type: KeyIndexType) -> Result<IndexWork> {
    let dataset = store.dataset();
    let key_id = dataset
        .schema()
        .field(store.key_column())
        .ok_or_else(|| lance::Error::invalid_input("missing declared key column"))?
        .id;
    let key_details = match key_type {
        KeyIndexType::Btree => "BTreeIndexDetails",
        KeyIndexType::Zonemap => "ZoneMapIndexDetails",
    };
    let mut work = IndexWork {
        create_key: true,
        ..Default::default()
    };
    let mut names = BTreeSet::new();
    for index in dataset.load_indices().await?.iter() {
        let Some(details) = &index.index_details else {
            continue;
        };
        let is_key = index
            .fields
            .iter()
            .copied()
            .filter(|id| !index.covering_fields.contains(id))
            .eq([key_id])
            && details.type_url.ends_with(key_details);
        work.create_key &= !is_key;
        if is_key || details.type_url.ends_with("BTreeIndexDetails") {
            names.insert(index.name.clone());
        }
    }
    for name in names {
        if !dataset.unindexed_fragments(&name).await?.is_empty() {
            work.extend.push(name);
        }
    }
    Ok(work)
}

/// Create a missing configured key index and extend only incomplete indexes.
/// Callers hold the existing maintenance fence; a queued task rechecks current
/// coverage so another index task completing first cannot force a rebuild.
pub(crate) async fn maintain_indexes(
    store: &mut StorageBase,
    key_type: KeyIndexType,
) -> Result<()> {
    let mut work = index_work(store, key_type).await?;
    if work.create_key {
        store.create_configured_key_index().await?;
        work = index_work(store, key_type).await?;
    }
    if !work.extend.is_empty() {
        let mut dataset = store.dataset().clone();
        // Merge existing deltas too: leaving a delta per update prevents data
        // compaction from coalescing fragments with different index coverage.
        dataset
            .optimize_indices(&OptimizeOptions::merge(usize::MAX).index_names(work.extend))
            .await?;
        store.reload().await?;
    }
    Ok(())
}

/// Return the exact post-index handle; callers can retain it without opening latest.
pub(crate) async fn indexed_dataset(store: &StorageBase) -> Result<lance::Dataset> {
    let mut dataset = store.dataset().clone();
    let names = dataset
        .load_indices()
        .await?
        .iter()
        .filter(|index| {
            index
                .index_details
                .as_ref()
                .is_some_and(|details| details.type_url.ends_with("BTreeIndexDetails"))
        })
        .map(|index| index.name.clone())
        .collect();
    dataset
        .optimize_indices(&OptimizeOptions::merge(usize::MAX).index_names(names))
        .await?;
    Ok(dataset)
}

/// Observe only metadata; a cold master must not claim any worker's WAL shard.
pub(crate) async fn observe(
    state: &MasterState,
    name: &str,
    uri: &str,
    previous: Option<&StatRow>,
) -> Result<(StatRow, bool, usize)> {
    tokio::time::timeout(std::time::Duration::from_secs(60), async {
        let store = open(state, uri).await?;
        let version = store.version() as i64;
        let (pending, shard_max) = store.pending_wal_generation_counts().await?;
        let pending = pending as i64;
        if let Some(previous) = previous.filter(|row| row.uri == uri && row.version == version) {
            let mut row = previous.clone();
            row.uri = uri.into();
            row.pending_wal_generations = pending;
            row.scanned_at = Utc::now().timestamp_millis();
            return Ok((row, true, shard_max));
        }
        Ok((
            StatRow {
                name: name.into(),
                uri: uri.into(),
                version,
                row_count: store.dataset().count_rows(None).await? as i64,
                fragment_count: store.compaction_stats().total_fragments as i64,
                last_updated: store.dataset().manifest.timestamp().timestamp_millis(),
                pending_wal_generations: pending,
                last_compaction: previous.map_or(StatRow::NO_COMPACTION, |row| row.last_compaction),
                total_compactions: previous.map_or(0, |row| row.total_compactions),
                scanned_at: Utc::now().timestamp_millis(),
            },
            false,
            shard_max,
        ))
    })
    .await
    .map_err(|_| lance::Error::io(format!("observe timed out for store '{name}'")))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field};
    use clap::Parser;
    use lance::dataset::{Dataset, WriteParams};

    #[tokio::test]
    async fn index_maintenance_extends_all_btrees_and_skips_covered_tables() {
        use lance::dataset::WriteMode;
        use lance_index::{scalar::ScalarIndexParams, IndexType};

        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Int32, false),
            Field::new("item_id", DataType::Int32, false),
            Field::new("parent_item_id", DataType::Int32, false),
        ]));
        let rows = |value| {
            RecordBatchIterator::new(
                vec![Ok(RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from(vec![value])),
                        Arc::new(Int32Array::from(vec![value / 2])),
                        Arc::new(Int32Array::from(vec![0])),
                    ],
                )
                .unwrap())],
                schema.clone(),
            )
        };
        let mut dataset = Dataset::write(rows(1), uri, None).await.unwrap();
        for column in ["item_id", "parent_item_id"] {
            dataset
                .create_index_builder(&[column], IndexType::BTree, &ScalarIndexParams::default())
                .await
                .unwrap();
        }
        let mut store =
            StorageBase::from_dataset(dataset, StorageBaseOptions::new(schema.clone(), "event_id"))
                .await
                .unwrap();
        assert!(
            index_work(&store, KeyIndexType::Btree)
                .await
                .unwrap()
                .create_key
        );
        maintain_indexes(&mut store, KeyIndexType::Btree)
            .await
            .unwrap();
        let version = store.version();
        let fragments = store
            .dataset()
            .get_fragments()
            .iter()
            .map(|f| f.id())
            .collect::<Vec<_>>();
        let indices = store.dataset().load_indices().await.unwrap();
        let btree_count = |indices: &[lance_table::format::IndexMetadata]| {
            indices
                .iter()
                .filter(|index| {
                    index
                        .index_details
                        .as_ref()
                        .is_some_and(|details| details.type_url.ends_with("BTreeIndexDetails"))
                })
                .count()
        };
        assert_eq!(btree_count(&indices), 3);
        assert!(!index_work(&store, KeyIndexType::Btree)
            .await
            .unwrap()
            .needed());
        maintain_indexes(&mut store, KeyIndexType::Btree)
            .await
            .unwrap();
        assert_eq!(store.version(), version, "covered table must not commit");
        assert_eq!(store.dataset().load_indices().await.unwrap(), indices);

        Dataset::write(
            rows(2),
            uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        store.reload().await.unwrap();
        let work = index_work(&store, KeyIndexType::Btree).await.unwrap();
        assert!(!work.create_key);
        assert_eq!(
            work.extend.len(),
            3,
            "primary and both secondary indexes need new coverage"
        );
        maintain_indexes(&mut store, KeyIndexType::Btree)
            .await
            .unwrap();
        assert!(!index_work(&store, KeyIndexType::Btree)
            .await
            .unwrap()
            .needed());
        assert_eq!(store.dataset().count_rows(None).await.unwrap(), 2);
        assert_eq!(
            store.dataset().get_fragments().len(),
            2,
            "indexing must not compact data"
        );
        assert_eq!(store.dataset().get_fragments()[0].id(), fragments[0]);

        // Coverage is a union across deltas, not a requirement that each delta
        // cover the entire table. A fully covered multi-delta index is a no-op.
        let mut dataset = Dataset::write(
            rows(3),
            uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert!(btree_count(&dataset.load_indices().await.unwrap()) > 3);
        store.reload().await.unwrap();
        let version = store.version();
        assert!(!index_work(&store, KeyIndexType::Btree)
            .await
            .unwrap()
            .needed());
        maintain_indexes(&mut store, KeyIndexType::Btree)
            .await
            .unwrap();
        assert_eq!(store.version(), version);
    }

    #[tokio::test]
    async fn index_maintenance_honors_configured_key_type() {
        use lance::dataset::WriteMode;
        for key_type in [KeyIndexType::Btree, KeyIndexType::Zonemap] {
            let dir = tempfile::tempdir().unwrap();
            let uri = dir.path().to_str().unwrap();
            let schema = Arc::new(Schema::new(vec![Field::new(
                "sequence",
                DataType::Int32,
                false,
            )]));
            let rows = || {
                RecordBatchIterator::new(
                    vec![Ok(RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(vec![1]))],
                    )
                    .unwrap())],
                    schema.clone(),
                )
            };
            let dataset = Dataset::write(rows(), uri, None).await.unwrap();
            let mut options = StorageBaseOptions::new(schema.clone(), "sequence");
            options.key_index_type = key_type;
            let mut store = StorageBase::from_dataset(dataset, options).await.unwrap();
            maintain_indexes(&mut store, key_type).await.unwrap();
            Dataset::write(
                rows(),
                uri,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            store.reload().await.unwrap();
            let work = index_work(&store, key_type).await.unwrap();
            assert!(!work.create_key);
            assert_eq!(work.extend.len(), 1);
            maintain_indexes(&mut store, key_type).await.unwrap();
            assert!(!index_work(&store, key_type).await.unwrap().needed());
            let version = store.version();
            maintain_indexes(&mut store, key_type).await.unwrap();
            assert_eq!(store.version(), version);
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn registered_store_rejects_missing_or_composite_primary_keys() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::config::MasterConfig::parse_from([
            "test",
            "--data-dir",
            dir.path().to_str().unwrap(),
            "--managed-store-kind",
            "events",
            "--etcd-prefix",
            &format!("/registered-schema/{}", lance_context_core::generate_id()),
            "--etcd-endpoints",
            &std::env::var("ETCD_TEST_ENDPOINTS").unwrap(),
        ]);
        let state = MasterState::new(cfg).await.unwrap();
        for declared_keys in [0, 2] {
            let schema = Arc::new(Schema::new(
                (0..2)
                    .map(|i| {
                        let field = Field::new(format!("key_{i}"), DataType::Int32, false);
                        if i < declared_keys {
                            field.with_metadata(
                                [("lance-schema:unenforced-primary-key".into(), "true".into())]
                                    .into(),
                            )
                        } else {
                            field
                        }
                    })
                    .collect::<Vec<_>>(),
            ));
            let uri = format!("{}/invalid-{declared_keys}.lance", dir.path().display());
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(RecordBatch::new_empty(schema.clone()))], schema),
                &uri,
                None,
            )
            .await
            .unwrap();
            let before = Dataset::open(&uri).await.unwrap().version().version;
            let error = match open(&state, &uri).await {
                Ok(_) => panic!("accepted {declared_keys} primary keys"),
                Err(error) => error,
            };
            assert!(error
                .to_string()
                .contains("requires one declared primary key"));
            assert_eq!(Dataset::open(&uri).await.unwrap().version().version, before);
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn registry_uri_change_refreshes_same_version_statistics() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::config::MasterConfig::parse_from([
            "test",
            "--data-dir",
            dir.path().to_str().unwrap(),
            "--managed-store-kind",
            "events",
            "--etcd-prefix",
            &format!("/registered-observe/{}", lance_context_core::generate_id()),
            "--etcd-endpoints",
            &std::env::var("ETCD_TEST_ENDPOINTS").unwrap(),
        ]);
        let state = MasterState::new(cfg).await.unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)
            .with_metadata(
                [("lance-schema:unenforced-primary-key".into(), "true".into())].into(),
            )]));
        let mut rows = Vec::new();
        for count in [1, 3] {
            let uri = dir
                .path()
                .join(format!("table-{count}"))
                .to_str()
                .unwrap()
                .to_string();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(0..count))],
            )
            .unwrap();
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
                &uri,
                Some(WriteParams {
                    max_rows_per_file: 1,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            state
                .managed_registry
                .as_ref()
                .unwrap()
                .upsert("table", &uri)
                .await
                .unwrap();
            let resolved = state.target_uri("events:table").await.unwrap().unwrap();
            let (row, reused, _) = observe(&state, "events:table", &resolved, rows.last())
                .await
                .unwrap();
            assert!(!reused);
            assert_eq!(row.row_count, i64::from(count));
            assert_eq!(row.fragment_count, i64::from(count));
            assert_eq!(row.uri, uri);
            rows.push(row);
        }
        assert_eq!(rows[0].version, rows[1].version);
        let (_, reused, _) = observe(&state, "events:table", &rows[1].uri, Some(&rows[1]))
            .await
            .unwrap();
        assert!(reused);
    }
}
