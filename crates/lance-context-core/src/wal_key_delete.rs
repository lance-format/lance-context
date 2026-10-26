//! Keep the old table's payload out of Lance 13's delete-only merge plan.
use std::sync::Arc;

use arrow_array::RecordBatch;
use lance::dataset::{
    Dataset, MergeInsertBuilder, MergeInsertJob, UncommittedMergeInsert, WhenMatched,
    WhenNotMatched,
};
use lance::index::DatasetIndexExt;
use lance::Result;
use lance_index::IndexCriteria;

async fn job_and_source(
    snapshot: Arc<Dataset>,
    batches: &[RecordBatch],
    key: &str,
    force_scan: bool,
) -> Result<(MergeInsertJob, Vec<RecordBatch>)> {
    // Match Lance's single-key indexed-join selection, on this immutable
    // snapshot. A ZoneMap is not an exact-equality index for this operation.
    let indexed = !force_scan
        && snapshot
            .load_scalar_index(
                IndexCriteria::default()
                    .for_column(key)
                    .supports_exact_equality(),
            )
            .await?
            .is_some();
    let source = if indexed {
        // The indexed path projects the target to the SOURCE schema, including
        // uncovered fragments. It must receive keys only.
        batches
            .iter()
            .map(|batch| batch.project(&[batch.schema().index_of(key)?]))
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        // Lance 13 treats a key-only source as a partial-row rewrite even for
        // Delete: it fills missing columns from the target and retains them in
        // the join. Supplying the full schema avoids that fill. Delete-only
        // projection pruning then removes the source's non-key columns AND
        // keeps the target scan key-only. These clones share Arrow buffers;
        // the caller already retains the batches for immutable file encoding.
        batches.to_vec()
    };
    let job = MergeInsertBuilder::try_new(snapshot, vec![key.to_string()])?
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .use_index(indexed)
        .conflict_retries(0)
        .commit_retries(0)
        .try_build()?;
    Ok((job, source))
}

pub(crate) async fn prepare(
    snapshot: Arc<Dataset>,
    batches: &[RecordBatch],
    key: &str,
    force_scan: bool,
) -> Result<UncommittedMergeInsert> {
    let (job, source) = job_and_source(snapshot, batches, key, force_scan).await?;
    job.execute_uncommitted_batches(source).await
}

#[cfg(test)]
#[path = "wal_key_delete_tests.rs"]
mod tests;
