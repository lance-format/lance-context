use super::*;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use lance::dataset::{
    transaction::Transaction, CommitBuilder, UpdateBuilder, WriteMode, WriteParams,
};

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
    prepare(
        source,
        Action::Build(source.fragments().iter().map(|f| f.id as u32).collect()),
        key,
    )
    .await
}

async fn prepare(source: &Dataset, action: Action, key: Key) -> Result<p::Descriptors> {
    let budget = MergeMemoryBudget::new(WORKSPACE);
    spawn_action(
        source.clone(),
        action,
        key,
        budget.reserve(WORKSPACE).await,
        Arc::default(),
    )
    .await?
}

async fn install(source: &Dataset, descriptors: &p::Descriptors, key: Key) -> Result<Dataset> {
    Ok(CommitBuilder::new(Arc::new(source.clone()))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(Transaction::new_from_version(
            source.version_id(),
            operation_for(source, descriptors, key).await?,
        ))
        .await?)
}

async fn update_a(source: &Dataset, value: &str) -> Result<Dataset> {
    Ok(UpdateBuilder::new(Arc::new(source.clone()))
        .update_where("id = 'A'")?
        .set("status", value)?
        .build()?
        .execute()
        .await?
        .new_dataset
        .as_ref()
        .clone())
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

#[tokio::test]
async fn status_rebase_masks_changed_fragments_and_leaves_appended_rows_uncovered() -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().join("turns.lance");
    let first = write(uri.to_str().unwrap(), "A", WriteMode::Create).await?;
    let source = write(first.uri(), "B", WriteMode::Append).await?;
    let retained = source.fragments()[1].id as u32;
    let original = index(&source, Key::Status).await?;
    let appended = write(source.uri(), "C", WriteMode::Append).await?;
    let target = update_a(&appended, "'accepted'").await?;
    assert!(operation_for(&target, &original, Key::Status)
        .await
        .is_err());
    let certified = rebase_status(&source, &target, &original).await?.unwrap();
    let selected = install(&target, &certified, Key::Status).await?;
    let indices = selected.load_indices().await?;
    assert_eq!(indices.len(), 1);
    assert_eq!(
        indices[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![retained]
    );
    assert_eq!(
        selected
            .count_rows(Some("status = 'pending'".into()))
            .await?,
        2
    );
    assert_eq!(
        selected
            .count_rows(Some("status = 'accepted'".into()))
            .await?,
        1
    );
    Ok(())
}

#[tokio::test]
async fn status_rebase_discards_fully_changed_input_even_when_values_return() -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().join("turns.lance");
    let source = write(uri.to_str().unwrap(), "A", WriteMode::Create).await?;
    let original = index(&source, Key::Status).await?;
    let changed = update_a(&source, "'accepted'").await?;
    let target = update_a(&changed, "'pending'").await?;
    assert_eq!(
        target.count_rows(Some("status = 'pending'".into())).await?,
        1
    );
    assert!(rebase_status(&source, &target, &original).await?.is_none());
    Ok(())
}

#[tokio::test]
async fn status_rebase_rejects_foreign_table_id_result_and_competing_coverage() -> Result<()> {
    let root = tempfile::tempdir()?;
    let source = write(
        root.path().join("a.lance").to_str().unwrap(),
        "A",
        WriteMode::Create,
    )
    .await?;
    let foreign = write(
        root.path().join("b.lance").to_str().unwrap(),
        "A",
        WriteMode::Create,
    )
    .await?;
    let original = index(&source, Key::Status).await?;
    assert!(rebase_status(&source, &foreign, &original).await.is_err());
    let ids = index(&source, Key::Turn).await?;
    assert!(rebase_status(&source, &source, &ids).await.is_err());
    let competing = index(&source, Key::Status).await?;
    let target = install(&source, &competing, Key::Status).await?;
    assert!(rebase_status(&source, &target, &original).await.is_err());
    Ok(())
}

#[tokio::test]
async fn status_merge_rebase_replaces_only_current_surviving_source_coverage() -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().join("turns.lance");
    let first = write(uri.to_str().unwrap(), "A", WriteMode::Create).await?;
    let first = install(&first, &index(&first, Key::Status).await?, Key::Status).await?;
    let appended = write(first.uri(), "B", WriteMode::Append).await?;
    let retained = appended.fragments()[1].id as u32;
    let tail = prepare(&appended, Action::Build(vec![retained]), Key::Status).await?;
    let source = install(&appended, &tail, Key::Status).await?;
    let segments = source.load_indices().await?.as_ref().clone();
    assert_eq!(segments.len(), 2);
    let original = prepare(&source, Action::Merge(segments), Key::Status).await?;
    let target = update_a(&source, "'accepted'").await?;
    let certified = rebase_status(&source, &target, &original).await?.unwrap();
    let selected = install(&target, &certified, Key::Status).await?;
    let indices = selected.load_indices().await?;
    assert_eq!(indices.len(), 1);
    assert_eq!(
        indices[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![retained]
    );
    assert_eq!(
        selected
            .count_rows(Some("status = 'pending'".into()))
            .await?,
        1
    );
    assert_eq!(
        selected
            .count_rows(Some("status = 'accepted'".into()))
            .await?,
        1
    );
    Ok(())
}
