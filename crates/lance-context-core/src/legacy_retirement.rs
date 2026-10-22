//! Storage barriers for a legacy executor whose process has verifiably exited.
//!
//! This is NOT a live-writer fence. Before calling this module, the caller must
//! prove the exact old process terminated and prevent a replacement from running
//! unfenced merges. A timeout, missing Pod, expired lease or changed endpoint is
//! insufficient. Once the process is dead it cannot retry a conflict; already
//! issued immutable manifest writes can target at most latest + 1. Occupying
//! those versions therefore also covers storage replies delayed past process exit.

use futures::TryStreamExt;
use lance::{
    dataset::builder::DatasetBuilder, dataset::mem_wal::ShardManifestStore, Error, Result,
};
use std::collections::{BTreeMap, HashMap};

/// Capture once, AFTER verified process termination, and persist before fencing.
/// Reuse this plan after a recovery crash; do not repeatedly advance its bounds.
/// Only metadata is read. An absent shard records version 1, the initial
/// publication that may still arrive. Completion must occupy that version;
/// absence alone never authorizes releasing the execution.
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
    let manifest = manifests.refresh_latest().await?;
    if manifest.is_none() {
        require_unpublished_shard(&dataset, shard).await?;
    }
    let next = |version: u64| {
        version
            .checked_add(1)
            .ok_or_else(|| Error::io("manifest version overflow"))
    };
    Ok(BTreeMap::from([
        ("base".into(), next(dataset.version().version)?),
        (
            format!("shard:{shard}"),
            match manifest {
                Some(manifest) => next(manifest.version)?,
                None => 1,
            },
        ),
    ]))
}

/// Complete a persisted retirement plan AFTER exact old-process termination.
///
/// A context writer claims its initial manifest before it can append any WAL.
/// If that publication is still absent, occupy version 1 conditionally, then
/// cross the normal barrier. If the old PUT or a successor wins the race, keep
/// its manifest, WAL entries and epoch. This never opens a shard writer.
pub async fn fence_retired_writer(
    uri: &str,
    storage_options: Option<HashMap<String, String>>,
    writer_id: &str,
    versions: &BTreeMap<String, u64>,
    recovery_id: &str,
) -> Result<()> {
    let shard = crate::store_base::derive_shard_id(Some(writer_id));
    let resource = format!("shard:{shard}");
    if writer_id.is_empty()
        || !crate::merge_write_scope::supports_version_fencing(uri)
        || versions.len() != 2
        || !versions.contains_key("base")
        || !versions.contains_key(&resource)
        || versions.values().any(|version| *version == 0)
    {
        return Err(Error::invalid_input("invalid retired writer barrier plan"));
    }
    // Only a plan that originally observed absence may initialize a shard.
    // A missing previously published shard remains an error, not an empty WAL.
    if versions[&resource] == 1 {
        let mut builder = DatasetBuilder::from_uri(uri);
        if let Some(options) = storage_options.clone() {
            builder = builder.with_storage_options(options);
        }
        let dataset = builder.load().await?;
        let manifests = ShardManifestStore::new(
            dataset.object_store(None).await?,
            &dataset.branch_location().path,
            shard,
            16,
        );
        if manifests.refresh_latest().await?.is_none() {
            require_unpublished_shard(&dataset, shard).await?;
            // Context uses manually assigned shards (spec 0, no field values).
            // initialize_shard uses PUT-if-absent and re-reads on a collision.
            // Epoch 0 makes this a metadata barrier, not a new writer claim.
            manifests.initialize_shard(0, HashMap::new()).await?;
        }
    }
    crate::merge_write_scope::fence_manifest_versions(uri, storage_options, versions, recovery_id)
        .await
}

// A broken/missing version hint plus a sparse old manifest history must not
// turn an existing shard into an empty one. Stop at the first object; do not
// collect a shard listing or read any WAL payload. A concurrent publication
// conservatively retries through refresh_latest on the next completion.
async fn require_unpublished_shard(dataset: &lance::Dataset, shard: uuid::Uuid) -> Result<()> {
    let store = dataset.object_store(None).await?;
    let prefix =
        lance::dataset::mem_wal::util::shard_base_path(&dataset.branch_location().path, &shard);
    let mut objects = store.inner.list(Some(&prefix));
    if objects
        .try_next()
        .await
        .map_err(|error| Error::io(format!("cannot verify unpublished shard: {error}")))?
        .is_some()
    {
        return Err(Error::io(
            "retired shard has stored objects but no discoverable manifest; publication unresolved",
        ));
    }
    Ok(())
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
    async fn absent_retired_shard_occupies_initial_version_before_release() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut store = GenericStore::open(uri, spec(), GenericStoreOptions::default())
            .await
            .unwrap();
        store
            .add(&[json!({"id":"other-writer"}).as_object().unwrap().clone()])
            .await
            .unwrap();
        let writer = "never-published";
        let plan = snapshot_retired_writer(uri, None, writer).await.unwrap();
        let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
        let shard = crate::store_base::derive_shard_id(Some(writer));
        let object_store = dataset.object_store(None).await.unwrap();
        let manifests = ShardManifestStore::new(
            object_store.clone(),
            &dataset.branch_location().path,
            shard,
            16,
        );
        assert_eq!(plan[&format!("shard:{shard}")], 1);
        assert!(manifests.refresh_latest().await.unwrap().is_none());
        // Ordinary recovery must still refuse a missing shard. Only the
        // process-verified retirement path can create the initial barrier.
        assert!(
            crate::merge_write_scope::fence_manifest_versions(uri, None, &plan, "ordinary")
                .await
                .is_err()
        );
        for _ in 0..2 {
            fence_retired_writer(uri, None, writer, &plan, "retired")
                .await
                .unwrap();
        }
        let after = manifests.refresh_latest().await.unwrap().unwrap();
        assert!(after.version > 1);
        assert_eq!(after.writer_epoch, 0);
        assert!(after.sstables.is_empty());
        assert_eq!(after.current_generation, 1);
        // A dead process cannot retry a conflict. Any delayed initial PUT,
        // regardless of its serialized bytes, must collide with version 1.
        let initial = lance::dataset::mem_wal::util::shard_manifest_path(
            &dataset.branch_location().path,
            &shard,
        )
        .join(lance::dataset::mem_wal::util::manifest_filename(1));
        assert!(object_store
            .put_if_absent(
                &initial,
                bytes::Bytes::from_static(b"delayed initial PUT").into()
            )
            .await
            .is_err());
        store.cleanup_wal().await.unwrap();
        assert_eq!(store.count_base_rows().await.unwrap(), 1);
        assert!(store.get("other-writer", None).await.unwrap().is_some());
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn absent_plan_preserves_successor_writes_before_and_after_barrier() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut creator = GenericStore::open(uri, spec(), GenericStoreOptions::default())
            .await
            .unwrap();
        creator.close().await.unwrap();
        let writer = "retired-worker";
        let plan = snapshot_retired_writer(uri, None, writer).await.unwrap();
        // The successor wins initial publication after the persisted snapshot.
        // Its pending generation must not be replaced by an empty barrier.
        let mut successor = GenericStore::open(
            uri,
            spec(),
            GenericStoreOptions {
                shard_id: Some(writer.into()),
                seal_on_add: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        successor
            .add(&[json!({"id":"before"}).as_object().unwrap().clone()])
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
        let shard = crate::store_base::derive_shard_id(Some(writer));
        let manifests = ShardManifestStore::new(
            dataset.object_store(None).await.unwrap(),
            &dataset.branch_location().path,
            shard,
            16,
        );
        let before = manifests.refresh_latest().await.unwrap().unwrap();
        fence_retired_writer(uri, None, writer, &plan, "retired")
            .await
            .unwrap();
        let after = manifests.refresh_latest().await.unwrap().unwrap();
        assert_eq!(after.sstables, before.sstables);
        assert_eq!(after.writer_epoch, before.writer_epoch);
        successor
            .add(&[json!({"id":"after"}).as_object().unwrap().clone()])
            .await
            .unwrap();
        fence_retired_writer(uri, None, writer, &plan, "retired")
            .await
            .unwrap();
        successor.cleanup_wal().await.unwrap();
        assert_eq!(successor.count_base_rows().await.unwrap(), 2);
        for id in ["before", "after"] {
            assert!(successor.get(id, None).await.unwrap().is_some());
        }
        successor.close().await.unwrap();
    }

    #[tokio::test]
    async fn published_shard_plan_cannot_recreate_a_missing_shard() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut creator = GenericStore::open(uri, spec(), GenericStoreOptions::default())
            .await
            .unwrap();
        creator.close().await.unwrap();
        let writer = "retired-worker";
        let mut plan = snapshot_retired_writer(uri, None, writer).await.unwrap();
        let shard = crate::store_base::derive_shard_id(Some(writer));
        plan.insert(format!("shard:{shard}"), 2);
        assert!(fence_retired_writer(uri, None, writer, &plan, "retired")
            .await
            .unwrap_err()
            .to_string()
            .contains("missing"));
        let absent = snapshot_retired_writer(uri, None, writer).await.unwrap();
        assert_eq!(absent[&format!("shard:{shard}")], 1);
        assert!(
            fence_retired_writer(uri, None, "another-writer", &plan, "retired")
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn undiscoverable_shard_objects_are_not_replaced_by_empty_state() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut creator = GenericStore::open(uri, spec(), GenericStoreOptions::default())
            .await
            .unwrap();
        creator.close().await.unwrap();
        let writer = "retired-worker";
        let plan = snapshot_retired_writer(uri, None, writer).await.unwrap();
        let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
        let shard = crate::store_base::derive_shard_id(Some(writer));
        let prefix =
            lance::dataset::mem_wal::util::shard_base_path(&dataset.branch_location().path, &shard);
        let store = dataset.object_store(None).await.unwrap();
        store
            .put_if_absent(
                &prefix.join("orphan-wal"),
                bytes::Bytes::from_static(b"must preserve").into(),
            )
            .await
            .unwrap();
        for error in [
            snapshot_retired_writer(uri, None, writer)
                .await
                .unwrap_err(),
            fence_retired_writer(uri, None, writer, &plan, "retired")
                .await
                .unwrap_err(),
        ] {
            assert!(error.to_string().contains("stored objects"), "{error}");
        }
        let manifests = ShardManifestStore::new(store, &dataset.branch_location().path, shard, 16);
        assert!(manifests.refresh_latest().await.unwrap().is_none());
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
            fence_retired_writer(uri, None, "old-worker", &plan, "retired-old-worker")
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
