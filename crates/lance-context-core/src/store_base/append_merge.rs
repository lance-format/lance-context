use super::{encode_merge_files, StagedMergeFiles, StorageBase};
use arrow_array::RecordBatch;
use lance::{Error, Result};
use lance_context_api::MaintenanceErrorCode;
use lance_index::mem_wal::CompactedSsTable;
use std::sync::Arc;
use uuid::Uuid;

impl StorageBase {
    /// Reuse the rollout publisher for application-owned immutable rows. The
    /// ordinary preparation/commit path owns prefix validation and WAL cleanup.
    pub(super) async fn append_prepared_batches(
        &mut self,
        batches: Vec<RecordBatch>,
        watermarks: Vec<(Uuid, u64)>,
        staged: Option<StagedMergeFiles>,
    ) -> Result<()> {
        let files = match staged {
            Some(files) => files,
            None => encode_merge_files(Arc::new(self.dataset.clone()), batches).await?,
        };
        let before = crate::rollout_append::watermarks(&self.dataset).await?;
        let merged = watermarks
            .iter()
            .map(|(shard, high)| CompactedSsTable::new(*shard, *high))
            .collect::<Vec<_>>();
        for attempt in 0..=10 {
            files.validate(&self.dataset)?;
            let result = crate::rollout_append::commit_files(
                &mut self.dataset,
                files.fragments.clone(),
                merged.clone(),
            )
            .await;
            match result {
                Ok(()) => return Ok(()),
                Err(Error::CommitConflict { .. }) if attempt < 10 => {
                    self.dataset.checkout_latest().await?;
                    let after = crate::rollout_append::watermarks(&self.dataset).await?;
                    if watermarks
                        .iter()
                        .any(|(shard, _)| before.get(shard) != after.get(shard))
                    {
                        return Err(Error::io(
                            MaintenanceErrorCode::StalePreparation
                                .tag("shard coverage changed during immutable append; reprepare"),
                        ));
                    }
                    // A definite conflict published nothing. Reuse the files
                    // only while their schema and shard coverage remain valid.
                }
                // An uncertain result or revoked ownership must be reconciled
                // from the atomic watermark by a fresh preparation, not retried.
                Err(error) => return Err(error),
            }
        }
        unreachable!("last commit attempt always returns")
    }
}

#[cfg(test)]
mod tests;
