use super::*;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use lance::dataset::{
    optimize::{CompactionOptions, CompactionTask, TaskData},
    CommitBuilder, WriteParams,
};
use lance_index::{scalar::ScalarIndexParams, IndexType};
use lance_table::transaction::Transaction;
use std::sync::Arc;

#[tokio::test]
async fn zonemap_compaction_withdraws_coverage_and_keeps_old_reads() -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().join("turns.lance");
    let batch = RecordBatch::try_from_iter([(
        "id",
        Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as arrow_array::ArrayRef,
    )])?;
    let mut base = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        uri.to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            enable_stable_row_ids: true,
            skip_auto_cleanup: true,
            ..Default::default()
        }),
    )
    .await?;
    assert_eq!(base.fragments().len(), 2);
    base.create_index_builder(
        &["id"],
        IndexType::ZoneMap,
        &ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::ZoneMap),
    )
    .name("id_zonemap".into())
    .await?;
    let original = base.load_indices().await?[0].clone();
    assert_eq!(original.fragment_bitmap.as_ref().unwrap().len(), 2);
    let mut result = CompactionTask {
        task: TaskData {
            fragments: base.fragments().to_vec(),
        },
        read_version: base.version_id(),
        options: CompactionOptions {
            target_rows_per_fragment: 4,
            ..Default::default()
        },
    }
    .execute(&base)
    .await?;
    assert_eq!(result.new_fragments.len(), 1);
    result.new_fragments[0].id = u64::from(base.manifest().max_fragment_id.unwrap()) + 1;
    let operation = prepare_rewrite(&base, result, 1 << 20).await?;
    assert!(
        matches!(&operation, Operation::Rewrite { rewritten_indices, .. } if rewritten_indices.is_empty())
    );
    validate_index_remaps(&base, &operation).await?;
    let current = CommitBuilder::new(Arc::new(base.clone()))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(Transaction::new_from_version(base.version_id(), operation))
        .await?;
    assert_eq!(current.fragments().len(), 1);
    let indices = current.load_indices().await?;
    assert!(indices
        .iter()
        .all(|index| index.fragment_bitmap.as_ref().unwrap().is_empty()));
    // Both the current unindexed rows and the old index remain queryable. No
    // new fragment may inherit addresses pointing into the retired fragments.
    for snapshot in [&base, &current] {
        assert_eq!(snapshot.count_rows(None).await?, 4);
        assert_eq!(
            snapshot.count_rows(Some("id IN ('a', 'd')".into())).await?,
            2
        );
        assert_eq!(snapshot.count_rows(Some("id = 'missing'".into())).await?, 0);
    }
    assert_eq!(base.load_indices().await?[0], original);
    Ok(())
}
