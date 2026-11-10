use super::*;
use arrow_array::{RecordBatchIterator, StringArray};
use lance::dataset::{
    optimize::{CompactionOptions, CompactionTask, TaskData},
    transaction::Transaction,
    CommitBuilder, WriteParams,
};

mod cross_version;

async fn commit(base: &Dataset, operation: Operation) -> Result<Dataset> {
    Ok(CommitBuilder::new(Arc::new(base.clone()))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(Transaction::new_from_version(base.version_id(), operation))
        .await?)
}

async fn query(base: &Dataset, filter: &str, expected: &[&str]) -> Result<()> {
    let mut scan = base.scan();
    scan.project(&["id"])?.filter(filter)?;
    assert!(scan.explain_plan(false).await?.contains("ScalarIndex"));
    let batches = scan
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let mut ids = Vec::new();
    for batch in batches {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        ids.extend(array.iter().map(|value| value.unwrap().to_owned()));
    }
    ids.sort();
    assert_eq!(ids, expected);
    Ok(())
}

#[tokio::test]
async fn legacy_extension_merge_compaction_and_reopen_keep_stable_ids_and_nulls() -> Result<()> {
    for key in [Key::Turn, Key::Status] {
        let root = tempfile::tempdir()?;
        let uri = root.path().join("turns.lance");
        let batch = RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(StringArray::from(vec!["A", "B", "C", "D", "E", "F"]))
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
        // Obtain the native definition, then train NEW version-0 pages using
        // real stable IDs. The native version-1 pages are never relabeled.
        let mut template = base
            .create_index_builder(
                &[key.column()],
                IndexType::BTree,
                &ScalarIndexParams::default(),
            )
            .name(key.name().into())
            .execute_uncommitted()
            .await?;
        template.index_version = 0;
        let first = build(&base, key, vec![0], &template).await?;
        base = commit(
            &base,
            Operation::CreateIndex {
                new_indices: vec![first],
                removed_indices: vec![],
            },
        )
        .await?;
        let budget = MergeMemoryBudget::new(WORKSPACE);
        let descriptors = spawn_action(
            base.clone(),
            Action::Build(vec![1, 2]),
            key,
            budget.reserve(WORKSPACE).await,
            Arc::default(),
        )
        .await??;
        let operation = operation_for(&base, &descriptors, key).await?;
        // The persisted decoder rejects a mixed address/ID domain even when
        // fragment coverage and index name would otherwise be valid.
        let mut foreign: Change = serde_json::from_slice(&descriptors.bytes)?;
        let mut metadata =
            IndexMetadata::try_from(pb::IndexMetadata::decode(foreign.index.as_slice())?)?;
        assert_eq!(metadata.index_version, 0);
        metadata.index_version = 1;
        foreign.index = pb::IndexMetadata::from(&metadata).encode_to_vec();
        let foreign = p::Descriptors::lance_index(Bytes::from(serde_json::to_vec(&foreign)?))?;
        assert!(operation_for(&base, &foreign, key).await.is_err());
        base = commit(&base, operation).await?;
        let indices = base.load_indices().await?;
        assert_eq!(indices.len(), 2);
        assert!(indices.iter().all(|index| index.index_version == 0));
        let descriptors = spawn_action(
            base.clone(),
            Action::Merge(indices.as_ref().clone()),
            key,
            budget.reserve(WORKSPACE).await,
            Arc::default(),
        )
        .await??;
        base = commit(&base, operation_for(&base, &descriptors, key).await?).await?;
        let before = base.clone();
        let mut rewritten = CompactionTask {
            task: TaskData {
                fragments: base.manifest().fragments.as_ref().clone(),
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
        assert_eq!(rewritten.new_fragments.len(), 1);
        for (ordinal, fragment) in rewritten.new_fragments.iter_mut().enumerate() {
            fragment.id = u64::from(base.manifest().max_fragment_id.unwrap()) + 1 + ordinal as u64;
        }
        let operation =
            lance_context_core::native_compaction::prepare_rewrite(&base, rewritten, 1 << 20)
                .await?;
        lance_context_core::native_compaction::validate_index_remaps(&base, &operation).await?;
        base = commit(&base, operation).await?;
        let reopened = Dataset::open(base.uri()).await?;
        for snapshot in [&before, &reopened] {
            let index = snapshot.load_indices().await?;
            assert_eq!(index.len(), 1);
            assert_eq!(index[0].index_version, 0);
            for fragment in snapshot.fragments().iter() {
                assert!(index[0]
                    .fragment_bitmap
                    .as_ref()
                    .unwrap()
                    .contains(fragment.id as u32));
            }
            match key {
                Key::Turn => query(snapshot, "id IN ('B', 'D', 'F')", &["B", "D", "F"]).await?,
                Key::Status => {
                    query(snapshot, "status IS NULL", &["B", "D"]).await?;
                    query(snapshot, "status = 'ready'", &["A", "E"]).await?;
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn legacy_status_merge_discards_values_from_withdrawn_fragment_coverage() -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("turns.lance");
    cross_version::copy_fixture(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lance13-btree"),
        &path,
    )?;
    let mut base = Dataset::open(path.to_str().unwrap()).await?;
    let budget = MergeMemoryBudget::new(WORKSPACE);
    // Merge the old fragment-0 pages with fragment 1 before adding fragment 2.
    // A later status rewrite withdraws only part of that first merged segment.
    for action in 0..3 {
        let action = match action {
            0 => Action::Build(vec![1]),
            1 => Action::Merge(
                base.load_indices()
                    .await?
                    .iter()
                    .filter(|index| index.name == Key::Status.name())
                    .cloned()
                    .collect(),
            ),
            _ => Action::Build(vec![2]),
        };
        let descriptor = spawn_action(
            base.clone(),
            action,
            Key::Status,
            budget.reserve(WORKSPACE).await,
            Arc::default(),
        )
        .await??;
        base = commit(&base, operation_for(&base, &descriptor, Key::Status).await?).await?;
    }
    base = lance::dataset::UpdateBuilder::new(Arc::new(base))
        .update_where("id IN ('A', 'B')")?
        .set("status", "'reviewed'")?
        .build()?
        .execute()
        .await?
        .new_dataset
        .as_ref()
        .clone();
    let segments: Vec<_> = base
        .load_indices()
        .await?
        .iter()
        .filter(|index| index.name == Key::Status.name())
        .cloned()
        .collect();
    assert_eq!(segments.len(), 2);
    assert!(segments
        .iter()
        .all(|index| !index.fragment_bitmap.as_ref().unwrap().contains(0)));
    let descriptor = spawn_action(
        base.clone(),
        Action::Merge(segments),
        Key::Status,
        budget.reserve(WORKSPACE).await,
        Arc::default(),
    )
    .await??;
    base = commit(&base, operation_for(&base, &descriptor, Key::Status).await?).await?;
    let reopened = Dataset::open(base.uri()).await?;
    query(&reopened, "status = 'ready'", &["E"]).await?;
    query(&reopened, "status IS NULL", &["D"]).await?;
    query(&reopened, "status = 'reviewed'", &["A", "B"]).await?;
    Ok(())
}
