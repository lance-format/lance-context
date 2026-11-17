//! Maintenance of registered tables using their persisted schema and indexes.

use std::sync::Arc;

use arrow_schema::Schema;
use chrono::Utc;
use lance::index::DatasetIndexExt;
use lance::Result;
use lance_context_core::{StorageBase, StorageBaseOptions};
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

/// Maintain BTree indexes on a freshly opened registered store, then refresh its manifest.
pub(crate) async fn maintain_indexes(store: &mut StorageBase) -> Result<()> {
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
    store.reload().await
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

/// Compact a registered table without acquiring any worker WAL identity.
pub(crate) async fn compact(
    state: &MasterState,
    target: &str,
) -> std::result::Result<lance::dataset::optimize::CompactionMetrics, String> {
    let uri = state
        .target_uri(target)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("unknown maintenance target '{target}'"))?;
    let mut store = open(state, &uri).await.map_err(|e| e.to_string())?;
    let config = state.compaction_config();
    let options = crate::scheduler::compact_options_key(&config);
    let before = store.version();
    maintain_indexes(&mut store)
        .await
        .map_err(|e| e.to_string())?;
    let metrics = store
        .compact(Some(config))
        .await
        .map_err(|e| e.to_string())?;
    maintain_indexes(&mut store)
        .await
        .map_err(|e| e.to_string())?;
    if metrics.fragments_removed == 0
        && metrics.fragments_added == 0
        && before == store.version()
        && !matches!(
            tokio::time::timeout(
                std::time::Duration::from_secs(3),
                state
                    .task_store
                    .record_compact_noop(target, &uri, before, &options)
            )
            .await,
            Ok(Ok(()))
        )
    {
        tracing::warn!(target, "could not persist registered-store no-op");
    }
    if let Ok(Ok(Some(guard))) = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        state.task_store.try_coordination_lock("stats-writer"),
    )
    .await
    {
        let result = async {
            let (mut row, _, _) = observe(state, target, &uri, None).await?;
            let mut stats = state.stats.lock().await;
            row.total_compactions = stats
                .get(target)
                .await?
                .map_or(1, |old| old.total_compactions + 1);
            row.last_compaction = Utc::now().timestamp_millis();
            stats.upsert(&row).await
        }
        .await;
        if let Err(error) = result {
            tracing::warn!(target, %error, "registered-store statistics refresh failed");
        }
        if let Err(error) = state.task_store.release_coordination_lock(guard).await {
            tracing::warn!(target, %error, "registered-store statistics unlock failed");
        }
    }
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field};
    use clap::Parser;
    use lance::dataset::{Dataset, WriteParams};

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
