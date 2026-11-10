//! This fixture must come from the Lance 13 builder, not a relabeled 14 index.
use super::*;
use lance::dataset::WriteMode;
use std::path::Path;

pub(super) fn copy_fixture(source: &Path, destination: &Path) -> Result<()> {
    std::fs::create_dir(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_fixture(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn native_lance13_indexes_survive_append_extension_merge_compaction_and_reopen() -> Result<()>
{
    let root = tempfile::tempdir()?;
    let uri = root.path().join("turns.lance");
    copy_fixture(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lance13-btree"),
        &uri,
    )?;
    let mut base = Dataset::open(uri.to_str().unwrap()).await?;
    assert_eq!(base.count_rows(None).await?, 6);
    let originals = base.load_indices().await?;
    assert_eq!(originals.len(), 2);
    assert!(originals.iter().all(|index| index.index_version == 0
        && index
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
            == vec![0]));
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(StringArray::from(vec!["G"])) as arrow_array::ArrayRef,
        ),
        (
            "status",
            Arc::new(StringArray::from(vec![None::<&str>])) as arrow_array::ArrayRef,
        ),
    ])?;
    base = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        base.uri(),
        Some(WriteParams {
            mode: WriteMode::Append,
            skip_auto_cleanup: true,
            ..Default::default()
        }),
    )
    .await?;
    let budget = MergeMemoryBudget::new(WORKSPACE);
    for key in [Key::Turn, Key::Status] {
        let descriptor = spawn_action(
            base.clone(),
            Action::Build(vec![1, 2, 3]),
            key,
            budget.reserve(WORKSPACE).await,
            Arc::default(),
        )
        .await??;
        base = commit(&base, operation_for(&base, &descriptor, key).await?).await?;
        let indices = base.load_indices().await?;
        let segments: Vec<_> = indices
            .iter()
            .filter(|index| index.name == key.name())
            .cloned()
            .collect();
        assert_eq!(segments.len(), 2);
        assert!(segments.iter().all(|index| index.index_version == 0));
        let descriptor = spawn_action(
            base.clone(),
            Action::Merge(segments),
            key,
            budget.reserve(WORKSPACE).await,
            Arc::default(),
        )
        .await??;
        base = commit(&base, operation_for(&base, &descriptor, key).await?).await?;
    }
    let before = base.clone();
    let mut result = CompactionTask {
        task: TaskData {
            fragments: base.manifest().fragments.as_ref().clone(),
        },
        read_version: base.version_id(),
        options: CompactionOptions {
            target_rows_per_fragment: 7,
            num_threads: Some(1),
            ..Default::default()
        },
    }
    .execute(&base)
    .await?;
    assert_eq!(result.new_fragments.len(), 1);
    result.new_fragments[0].id = u64::from(base.manifest().max_fragment_id.unwrap()) + 1;
    let operation =
        lance_context_core::native_compaction::prepare_rewrite(&base, result, 1 << 20).await?;
    base = commit(&base, operation).await?;
    let reopened = Dataset::open(base.uri()).await?;
    for snapshot in [&before, &reopened] {
        query(snapshot, "id IN ('B', 'D', 'G')", &["B", "D", "G"]).await?;
        query(snapshot, "status IS NULL", &["B", "D", "G"]).await?;
        query(snapshot, "status = 'ready'", &["A", "E"]).await?;
        let indices = snapshot.load_indices().await?;
        assert_eq!(indices.len(), 2);
        for index in indices.iter() {
            assert_eq!(index.index_version, 0);
            assert!(snapshot.fragments().iter().all(|fragment| index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .contains(fragment.id as u32)));
        }
    }
    Ok(())
}
