//! Storage barriers for a legacy executor whose process has verifiably exited.
//!
//! This is NOT a live-writer fence. Before calling this module, the caller must
//! prove the exact old process terminated and prevent a replacement from running
//! unfenced merges. A timeout, missing Pod, expired lease or changed endpoint is
//! insufficient. Once the process is dead it cannot retry a conflict; already
//! issued immutable manifest writes can target at most latest + 1. Occupying
//! those versions therefore also covers storage replies delayed past process exit.

use lance::{
    dataset::builder::DatasetBuilder, dataset::mem_wal::ShardManifestStore, Error, Result,
};
use std::collections::{BTreeMap, HashMap};

/// Capture once, AFTER verified process termination, and persist before fencing.
/// Reuse this plan after a recovery crash; do not repeatedly advance its bounds.
/// Only metadata is read. A missing shard fails closed: absence is not proof
/// that an old initial shard publication cannot still arrive.
pub async fn snapshot_retired_writer(
    uri: &str,
    storage_options: Option<HashMap<String, String>>,
    writer_id: &str,
) -> Result<BTreeMap<String, u64>> {
    if writer_id.is_empty() || !crate::merge_write_scope::supports_version_fencing(uri) {
        return Err(Error::invalid_input(
            "retirement requires a writer ID and conditional manifests",
        ));
    }
    let mut builder = DatasetBuilder::from_uri(uri);
    if let Some(options) = storage_options {
        builder = builder.with_storage_options(options);
    }
    let dataset = builder.load().await?;
    let shard = crate::store_base::derive_shard_id(Some(writer_id));
    let manifests = ShardManifestStore::new(
        dataset.object_store(None).await?,
        &dataset.branch_location().path,
        shard,
        16,
    );
    let manifest = manifests.refresh_latest().await?.ok_or_else(|| {
        Error::io("retired writer shard is absent; initial publication remains unresolved")
    })?;
    let next = |version: u64| {
        version
            .checked_add(1)
            .ok_or_else(|| Error::io("manifest version overflow"))
    };
    Ok(BTreeMap::from([
        ("base".into(), next(dataset.version().version)?),
        (format!("shard:{shard}"), next(manifest.version)?),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenericStore, GenericStoreOptions};
    use lance_context_api::schema_spec::{ColumnSpec, ColumnType, SchemaSpec};
    use serde_json::json;

    fn spec() -> SchemaSpec {
        SchemaSpec::new(vec![(
            "id".into(),
            ColumnSpec::required(ColumnType::String { large: false }),
        )])
    }

    #[tokio::test]
    async fn missing_retired_shard_cannot_be_treated_as_quiescence() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let _store = GenericStore::open(uri, spec(), GenericStoreOptions::default())
            .await
            .unwrap();
        assert!(snapshot_retired_writer(uri, None, "never-published")
            .await
            .unwrap_err()
            .to_string()
            .contains("absent"));
    }

    #[tokio::test]
    async fn retired_writer_barrier_preserves_pending_rows_and_rejects_delayed_drain() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut store = GenericStore::open(
            uri,
            spec(),
            GenericStoreOptions {
                shard_id: Some("old-worker".into()),
                seal_on_add: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        store
            .add(&[json!({"id":"before"}).as_object().unwrap().clone()])
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
        let shard = crate::store_base::derive_shard_id(Some("old-worker"));
        let manifests = ShardManifestStore::new(
            dataset.object_store(None).await.unwrap(),
            &dataset.branch_location().path,
            shard,
            16,
        );
        let before = manifests.refresh_latest().await.unwrap().unwrap();
        let mut delayed = before.clone();
        delayed.version += 1;
        delayed.sstables.clear();
        let plan = snapshot_retired_writer(uri, None, "old-worker")
            .await
            .unwrap();
        assert_eq!(plan["base"], dataset.version().version + 1);
        assert_eq!(plan[&format!("shard:{shard}")], delayed.version);
        // A retry must reuse the persisted plan, including after partial barriers.
        for _ in 0..2 {
            crate::merge_write_scope::fence_manifest_versions(
                uri,
                None,
                &plan,
                "retired-old-worker",
            )
            .await
            .unwrap();
        }
        assert!(
            crate::merge_write_scope::commit_exact_shard(&manifests, &delayed)
                .await
                .is_err()
        );
        let after = manifests.refresh_latest().await.unwrap().unwrap();
        assert_eq!(after.sstables, before.sstables);
        assert_eq!(after.writer_epoch, before.writer_epoch);
        store
            .add(&[json!({"id":"during"}).as_object().unwrap().clone()])
            .await
            .unwrap();
        store.cleanup_wal().await.unwrap();
        assert_eq!(store.count_base_rows().await.unwrap(), 2);
        assert_eq!(store.pending_wal_generations().await.unwrap(), 0);
        store.close().await.unwrap();
    }
}
