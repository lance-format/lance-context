use super::*;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use lance::dataset::{WriteMode, WriteParams};

async fn write(uri: &str, id: &str, mode: WriteMode) -> Result<Dataset> {
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(StringArray::from(vec![id])) as arrow_array::ArrayRef,
        ),
        (
            "status",
            Arc::new(StringArray::from(vec!["pending"])) as arrow_array::ArrayRef,
        ),
    ])?;
    Ok(Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        uri,
        Some(WriteParams {
            mode,
            enable_stable_row_ids: true,
            skip_auto_cleanup: true,
            ..Default::default()
        }),
    )
    .await?)
}

async fn index(source: &Dataset, key: Key) -> Result<p::Descriptors> {
    let budget = MergeMemoryBudget::new(WORKSPACE);
    spawn_action(
        source.clone(),
        Action::Build(source.fragments().iter().map(|f| f.id as u32).collect()),
        key,
        budget.reserve(WORKSPACE).await,
        Arc::default(),
    )
    .await?
}

#[tokio::test]
async fn id_rebase_accepts_append_without_covering_new_fragments() -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().join("turns.lance");
    let source = write(uri.to_str().unwrap(), "A", WriteMode::Create).await?;
    let original = index(&source, Key::Turn).await?;
    let target = write(source.uri(), "B", WriteMode::Append).await?;
    // The ordinary decoder remains strict: stale descriptors cannot publish.
    assert!(operation_for(&target, &original, Key::Turn).await.is_err());
    let certified = rebase_turn(&source, &target, &original).await?;
    let Operation::CreateIndex { new_indices, .. } =
        operation_for(&target, &certified, Key::Turn).await?
    else {
        unreachable!()
    };
    assert_eq!(new_indices[0].dataset_version, target.version_id());
    assert_eq!(
        new_indices[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![source.fragments()[0].id as u32]
    );
    assert_eq!(target.fragments().len(), 2);
    Ok(())
}

#[tokio::test]
async fn id_rebase_rejects_replaced_column_and_foreign_table() -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().join("turns.lance");
    let source = write(uri.to_str().unwrap(), "A", WriteMode::Create).await?;
    let original = index(&source, Key::Turn).await?;
    let target = write(source.uri(), "Z", WriteMode::Overwrite).await?;
    assert!(rebase_turn(&source, &target, &original).await.is_err());
    let other = root.path().join("other.lance");
    let other = write(other.to_str().unwrap(), "A", WriteMode::Create).await?;
    assert!(rebase_turn(&source, &other, &original).await.is_err());
    Ok(())
}

#[tokio::test]
async fn id_rebase_rejects_status_result_even_when_rows_are_unchanged() -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().join("turns.lance");
    let source = write(uri.to_str().unwrap(), "A", WriteMode::Create).await?;
    let status = index(&source, Key::Status).await?;
    assert!(rebase_turn(&source, &source, &status).await.is_err());
    Ok(())
}

#[tokio::test]
async fn id_rebase_rejects_competing_index_coverage() -> Result<()> {
    use lance::dataset::{transaction::Transaction, CommitBuilder};
    let root = tempfile::tempdir()?;
    let uri = root.path().join("turns.lance");
    let source = write(uri.to_str().unwrap(), "A", WriteMode::Create).await?;
    let original = index(&source, Key::Turn).await?;
    let competing = index(&source, Key::Turn).await?;
    let operation = operation_for(&source, &competing, Key::Turn).await?;
    let target = CommitBuilder::new(Arc::new(source.clone()))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(Transaction::new_from_version(
            source.version_id(),
            operation,
        ))
        .await?;
    assert!(rebase_turn(&source, &target, &original).await.is_err());
    Ok(())
}
