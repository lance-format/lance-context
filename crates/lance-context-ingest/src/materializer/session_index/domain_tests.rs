//! New managed stable-ID indexes and existing native address indexes must both
//! survive segment extension, consolidation, compaction, and snapshot reopening.
use super::*;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use futures::TryStreamExt;
use lance::dataset::{
    optimize::{CompactionOptions, CompactionTask, TaskData},
    transaction::Transaction,
    CommitBuilder, WriteParams,
};

async fn commit(base: &Dataset, operation: Operation) -> Result<Dataset> {
    Ok(CommitBuilder::new(Arc::new(base.clone()))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(Transaction::new_from_version(base.version_id(), operation))
        .await?)
}

async fn managed(base: &Dataset, action: Action, key: Key) -> Result<Dataset> {
    let budget = MergeMemoryBudget::new(WORKSPACE);
    let descriptor = spawn_action(
        base.clone(),
        action,
        key,
        budget.reserve(WORKSPACE).await,
        Arc::default(),
    )
    .await??;
    commit(base, operation_for(base, &descriptor, key).await?).await
}

async fn query(base: &Dataset, predicate: &str, expected: &[&str]) -> Result<()> {
    let mut scan = base.scan();
    scan.project(&["id", "body"])?.filter(predicate)?;
    assert!(scan.explain_plan(false).await?.contains("ScalarIndex"));
    let mut rows = Vec::new();
    let mut stream = scan.try_into_stream().await?;
    while let Some(batch) = stream.try_next().await? {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let bodies = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let id = ids.value(row);
            assert_eq!(bodies.value(row), format!("body-{id}"));
            rows.push(id.to_owned());
        }
    }
    rows.sort();
    assert_eq!(rows, expected);
    Ok(())
}

async fn exercise(existing_address_index: bool) -> Result<()> {
    for key in [Key::Session, Key::Turn, Key::Status] {
        let root = tempfile::tempdir()?;
        let uri = root.path().join("turns.lance");
        let batch = RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(StringArray::from(vec!["A", "B", "C", "D", "E", "F"]))
                    as arrow_array::ArrayRef,
            ),
            (
                "rollout_id",
                Arc::new(StringArray::from(vec!["s0", "s0", "s1", "s1", "s2", "s2"]))
                    as arrow_array::ArrayRef,
            ),
            (
                "status",
                Arc::new(StringArray::from(vec![
                    Some("ready"),
                    None,
                    Some("pending"),
                    None,
                    Some("ready"),
                    Some("pending"),
                ])) as arrow_array::ArrayRef,
            ),
            (
                "body",
                Arc::new(StringArray::from(vec![
                    "body-A", "body-B", "body-C", "body-D", "body-E", "body-F",
                ])) as arrow_array::ArrayRef,
            ),
        ])?;
        let mut base = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            uri.to_str().unwrap(),
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                skip_auto_cleanup: true,
                ..Default::default()
            }),
        )
        .await?;
        assert_eq!(base.fragments().len(), 3);
        if existing_address_index {
            let index = base
                .create_index_builder(
                    &[key.column()],
                    IndexType::BTree,
                    &ScalarIndexParams::default(),
                )
                .name(key.name().into())
                .fragments(vec![0])
                .execute_uncommitted()
                .await?;
            assert_eq!(index.index_version, 1);
            base = commit(
                &base,
                Operation::CreateIndex {
                    new_indices: vec![index],
                    removed_indices: vec![],
                },
            )
            .await?;
        } else {
            base = managed(&base, Action::Build(vec![0]), key).await?;
        }
        let version = i32::from(existing_address_index);
        assert_eq!(base.load_indices().await?[0].index_version, version);
        // A fresh managed index must not first build and discard another index
        // merely to obtain metadata. Every directory is referenced at this cut.
        let directories =
            std::fs::read_dir(uri.join("_indices"))?.collect::<std::io::Result<Vec<_>>>()?;
        assert_eq!(directories.len(), 1);
        let files = std::fs::read_dir(directories[0].path())?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<std::io::Result<BTreeSet<_>>>()?;
        assert_eq!(
            files,
            ["page_data.lance", "page_lookup.lance"]
                .map(std::ffi::OsString::from)
                .into_iter()
                .collect()
        );
        // Fragments 1 and 2 have stable row IDs that differ from their physical
        // addresses; this catches wrongly relabeling an address-trained index.
        base = managed(&base, Action::Build(vec![1, 2]), key).await?;
        let indices = base.load_indices().await?;
        assert_eq!(indices.len(), 2);
        assert!(indices.iter().all(|index| index.index_version == version));
        base = managed(&base, Action::Merge(indices.as_ref().clone()), key).await?;
        let old_index = base.load_indices().await?[0].clone();
        let before = base.clone();
        let mut rewrite = CompactionTask {
            task: TaskData {
                fragments: base.fragments().to_vec(),
            },
            read_version: base.version_id(),
            options: CompactionOptions {
                target_rows_per_fragment: 6,
                num_threads: Some(1),
                ..Default::default()
            },
        }
        .execute(&base)
        .await?;
        assert_eq!(rewrite.new_fragments.len(), 1);
        rewrite.new_fragments[0].id = u64::from(base.manifest().max_fragment_id.unwrap()) + 1;
        let store = base.object_store(None).await?;
        let before_remap = store.io_stats_snapshot();
        let operation =
            lance_context_core::native_compaction::prepare_rewrite(&base, rewrite, 1 << 20).await?;
        let after_remap = store.io_stats_snapshot();
        let Operation::Rewrite {
            rewritten_indices, ..
        } = &operation
        else {
            anyhow::bail!("expected native rewrite");
        };
        if existing_address_index {
            assert_eq!(rewritten_indices.len(), 1);
            assert_eq!(rewritten_indices[0].old_id, old_index.uuid);
            assert_ne!(rewritten_indices[0].new_id, old_index.uuid);
            assert!(after_remap.written_bytes > before_remap.written_bytes);
        } else {
            assert!(rewritten_indices.is_empty());
            assert_eq!(after_remap.written_bytes, before_remap.written_bytes);
            assert_eq!(after_remap.read_bytes, before_remap.read_bytes);
        }
        lance_context_core::native_compaction::validate_index_remaps(&base, &operation).await?;
        assert_eq!(
            Dataset::open(base.uri()).await?.version_id(),
            before.version_id()
        );
        base = commit(&base, operation).await?;
        let reopened = Dataset::open(base.uri()).await?;
        let current = reopened.load_indices().await?[0].clone();
        if !existing_address_index {
            assert_eq!(current.uuid, old_index.uuid);
            assert_eq!(current.files, old_index.files);
        }
        for snapshot in [&before, &reopened] {
            assert_eq!(snapshot.count_rows(None).await?, 6);
            let indices = snapshot.load_indices().await?;
            assert_eq!(indices.len(), 1);
            assert_eq!(indices[0].index_version, version);
            assert!(snapshot.fragments().iter().all(|fragment| indices[0]
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .contains(fragment.id as u32)));
            match key {
                Key::Turn => query(snapshot, "id IN ('B', 'D', 'F')", &["B", "D", "F"]).await?,
                Key::Session => {
                    query(
                        snapshot,
                        "rollout_id IN ('s0', 's2')",
                        &["A", "B", "E", "F"],
                    )
                    .await?
                }
                Key::Status => {
                    query(snapshot, "status IS NULL", &["B", "D"]).await?;
                    query(snapshot, "status = 'ready'", &["A", "E"]).await?;
                }
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn new_managed_indexes_keep_native_pages_across_compaction() -> Result<()> {
    exercise(false).await
}

#[tokio::test]
async fn existing_address_indexes_keep_their_domain_and_remap_on_compaction() -> Result<()> {
    exercise(true).await
}
