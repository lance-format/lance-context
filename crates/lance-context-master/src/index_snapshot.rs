//! Retain a registered table's exact post-index version before its task completes.
use std::collections::BTreeSet;

use lance::{index::DatasetIndexExt, Dataset, Error, Result};
use lance_context_core::StorageBase;

use crate::{registered_store, state::MasterState};

pub(crate) async fn prepare(state: &MasterState, target: &str, tag: &str) -> Result<u64> {
    if state.registered_target(target).is_none()
        || !matches!(
            state.config.key_index_type,
            lance_context_core::KeyIndexType::Btree
        )
    {
        return Err(Error::invalid_input(
            "index snapshots require a registered table with BTree key indexes",
        ));
    }
    let uri = state
        .target_uri(target)
        .await?
        .ok_or_else(|| Error::invalid_input("unknown snapshot target"))?;
    let mut store = registered_store::open(state, &uri).await?;
    index_and_tag(&mut store, tag).await
}

// The scheduler retains table ownership throughout indexing and tag publication.
// Business writers must be quiescent: this is not a distributed write barrier.
async fn index_and_tag(store: &mut StorageBase, tag: &str) -> Result<u64> {
    lance::dataset::refs::check_valid_tag(tag)?;
    match store.dataset().tags().get_version(tag).await {
        Ok(version) => {
            validate_coverage(&store.dataset().checkout_version(version).await?).await?;
            return Ok(version); // Recover a lost task response without moving its tag.
        }
        Err(Error::RefNotFound { .. }) => {}
        Err(error) => return Err(error),
    }
    if store.pending_wal_generations().await? != 0 {
        return Err(Error::invalid_input(
            "index snapshot requires all published WAL to be merged",
        ));
    }
    store.create_key_btree_index().await?;
    let dataset = registered_store::indexed_dataset(store).await?;
    validate_coverage(&dataset).await?;
    let version = dataset.version().version;
    // Never reload latest here: it may name fragments this build did not index.
    // Index and tag commits are separate; a failed tag write never reports success.
    match dataset.tags().create(tag, version).await {
        Ok(()) => Ok(version),
        Err(Error::RefConflict { .. }) => {
            let retained = dataset.tags().get_version(tag).await?;
            validate_coverage(&dataset.checkout_version(retained).await?).await?;
            Ok(retained)
        }
        Err(error) => Err(error),
    }
}

async fn validate_coverage(dataset: &Dataset) -> Result<()> {
    let indices = dataset.load_indices().await?;
    let names: BTreeSet<_> = indices
        .iter()
        .filter(|index| {
            index
                .index_details
                .as_ref()
                .is_some_and(|details| details.type_url.ends_with("BTreeIndexDetails"))
        })
        .map(|index| &index.name)
        .collect();
    if names.is_empty() {
        return Err(Error::invalid_input(
            "index snapshot requires a BTree index",
        ));
    }
    for name in names {
        let covered = dataset.get_fragments().iter().all(|fragment| {
            u32::try_from(fragment.id()).ok().is_some_and(|id| {
                indices
                    .iter()
                    .filter(|index| &index.name == name)
                    .any(|index| {
                        index
                            .fragment_bitmap
                            .as_ref()
                            .is_some_and(|bitmap| bitmap.contains(id))
                    })
            })
        });
        if !covered {
            return Err(Error::invalid_input(format!(
                "index snapshot requires full BTree coverage for {name}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance::dataset::{WriteMode, WriteParams};
    use lance_context_core::StorageBaseOptions;
    use std::sync::Arc;

    #[tokio::test]
    async fn index_snapshot_keeps_its_commit_across_later_writes_and_retry() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let rows = |value| {
            RecordBatchIterator::new(
                vec![Ok(RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from(vec![value]))],
                )
                .unwrap())],
                schema.clone(),
            )
        };
        let original = Dataset::write(rows(1), uri, None).await.unwrap();
        let before = original.version().version;
        let mut store =
            StorageBase::from_dataset(original, StorageBaseOptions::new(schema.clone(), "key"))
                .await
                .unwrap();
        let version = index_and_tag(&mut store, "resume-test").await.unwrap();
        assert!(version > before);
        let later = Dataset::write(
            rows(2),
            uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert!(validate_coverage(&later).await.is_err());
        assert_eq!(
            index_and_tag(&mut store, "resume-test").await.unwrap(),
            version
        );
        let pinned = later.checkout_version(version).await.unwrap();
        assert_eq!(pinned.count_rows(None).await.unwrap(), 1);
        assert_eq!(later.count_rows(None).await.unwrap(), 2);
        validate_coverage(&pinned).await.unwrap();
        let rebuilt = index_and_tag(&mut store, "resume-next").await.unwrap();
        assert!(rebuilt > later.version().version);
        assert_eq!(
            later.tags().get_version("resume-test").await.unwrap(),
            version
        );
    }
}
