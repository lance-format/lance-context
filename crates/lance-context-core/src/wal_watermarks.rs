//! Advisory demand watermarks; never used to authorize a commit or WAL replay.

use std::collections::HashSet;
use std::sync::Arc;

use futures::{stream, StreamExt, TryStreamExt};
use lance::dataset::mem_wal::{DatasetMemWalExt, ShardManifestStore};
use lance::dataset::Dataset;
use lance::Result;
use lance_index::mem_wal::ShardManifest;

use crate::store_base::{ShardWatermark, DEFAULT_MANIFEST_SCAN_BATCH_SIZE};

/// Read latest per-shard demand after maintenance without decoding WAL payloads.
/// Reads the base index and at most eight shard manifests concurrently. This
/// includes ordinary upsert merges, which need not write an append watermark.
/// Errors propagate rather than publishing invented zero progress. Generation
/// gaps can overstate demand; these watermarks are not exact pending counts.
pub async fn read_demand_watermarks(uri: &str) -> Result<Vec<ShardWatermark>> {
    from_dataset(&Dataset::open(uri).await?).await
}

pub(crate) async fn from_dataset(dataset: &Dataset) -> Result<Vec<ShardWatermark>> {
    let merged = Arc::new(crate::rollout_append::watermarks(dataset).await?);
    let object_store = dataset.object_store(None).await?;
    let branch_path = dataset.branch_location().path.clone();
    let shard_ids = dataset.list_mem_wal_latest_shard_ids().await?;
    let marks: Vec<Option<ShardWatermark>> = stream::iter(shard_ids)
        .map(|shard_id| {
            let object_store = object_store.clone();
            let branch_path = branch_path.clone();
            let merged = merged.clone();
            async move {
                let store = ShardManifestStore::new(
                    object_store,
                    &branch_path,
                    shard_id,
                    DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
                );
                Ok::<_, lance::Error>(store.refresh_latest().await?.map(|manifest| {
                    from_manifest(&manifest, merged.get(&manifest.shard_id).copied())
                }))
            }
        })
        .buffer_unordered(8)
        .try_collect()
        .await?;
    let mut marks: Vec<_> = marks.into_iter().flatten().collect();
    let seen: HashSet<_> = marks.iter().map(|mark| mark.shard_id).collect();
    // Preserve the completion publisher's existing base-only progress if a
    // shard no longer has a discoverable manifest. No writer epoch is known.
    for (&shard_id, &generation) in merged.iter() {
        if !seen.contains(&shard_id) {
            marks.push(ShardWatermark {
                shard_id,
                writer_epoch: 0,
                sealed_through: generation,
                pending_generations: 0,
                merged_through: Some(generation),
            });
        }
    }
    Ok(marks)
}

fn from_manifest(manifest: &ShardManifest, base_merged: Option<u64>) -> ShardWatermark {
    // Lance advances this durable counter in the same manifest commit that
    // appends a flushed SST. Allocating an in-memory generation does not advance
    // it. Upsert merge drains preserve it, even when the last SST is removed.
    let flushed = manifest.current_generation.saturating_sub(1);
    // A non-prefix drain must not hide an older pending SST. Do not derive the
    // watermark by subtracting the pending count from the highest generation.
    let drained = manifest
        .sstables
        .iter()
        .map(|sst| sst.generation.saturating_sub(1))
        .min()
        .unwrap_or(flushed)
        .min(flushed);
    // Append may commit its base watermark before the shard drain completes.
    // Keep that authoritative progress as well, without altering replay fences.
    let merged = drained.max(base_merged.unwrap_or(0));
    ShardWatermark {
        shard_id: manifest.shard_id,
        writer_epoch: manifest.writer_epoch,
        sealed_through: manifest
            .sstables
            .iter()
            .map(|sst| sst.generation)
            .max()
            .unwrap_or(0)
            .max(flushed)
            .max(merged),
        pending_generations: manifest.sstables.len(),
        merged_through: Some(merged),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_index::mem_wal::{ShardStatus, SsTable};
    use uuid::Uuid;

    fn manifest(current: u64, pending: &[u64]) -> ShardManifest {
        ShardManifest {
            shard_id: Uuid::nil(),
            version: 1,
            shard_spec_id: 0,
            shard_field_values: Default::default(),
            writer_epoch: 1,
            replay_after_wal_entry_position: 0,
            wal_entry_position_last_seen: 0,
            current_generation: current,
            sstables: pending
                .iter()
                .map(|g| SsTable::unmeasured(*g, format!("gen-{g}")))
                .collect(),
            status: ShardStatus::Active,
        }
    }

    #[test]
    fn non_prefix_drain_never_hides_older_pending_generations() {
        // Generation 5 was drained before 3 and 4; highest-minus-count
        // would claim 3 was merged even though it remains pending.
        let mark = from_manifest(&manifest(7, &[6, 3, 4]), None);
        assert_eq!(mark.sealed_through, 6);
        assert_eq!(mark.merged_through, Some(2));
        assert_eq!(mark.pending_generations, 3);
    }

    #[test]
    fn append_commit_before_drain_retains_committed_progress() {
        let mark = from_manifest(&manifest(4, &[1, 2, 3]), Some(2));
        assert_eq!(mark.merged_through, Some(2));
        assert_eq!(mark.sealed_through, 3);
        assert_eq!(mark.pending_generations, 3);
    }

    #[test]
    fn unflushed_wal_and_epoch_changes_do_not_invent_generations() {
        let mut shard = manifest(1, &[]);
        shard.wal_entry_position_last_seen = 100;
        shard.writer_epoch = 7;
        let mark = from_manifest(&shard, None);
        assert_eq!(mark.sealed_through, 0);
        assert_eq!(mark.merged_through, Some(0));
        assert_eq!(mark.writer_epoch, 7);
    }
}
