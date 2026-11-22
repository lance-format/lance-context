//! Prepare incremental index updates with Lance's public optimize stages.

use super::*;
use lance::index::optimize::{commit_optimize_indices, plan_optimize_indices, IndexSegmentResult};
use lance_index::optimize::OptimizeOptions;
use lance_table::format::{Fragment, IndexMetadata};

/// The same stale-preparation checks apply to a new key index and incremental
/// updates. Keep metadata only: preparation-scoped dataset handles must not
/// escape to the publishing phase.
#[derive(Debug)]
pub(super) struct PreparedIndexSource {
    uri: String,
    schema: Schema,
    fragments: Vec<Fragment>,
    names: Vec<String>,
    pub(super) indices: Vec<IndexMetadata>,
}

impl PreparedIndexSource {
    pub(super) async fn capture(dataset: &Dataset, names: &[String]) -> LanceResult<Self> {
        Ok(Self {
            uri: dataset.uri().to_owned(),
            schema: Schema::from(dataset.schema()),
            fragments: dataset.manifest().fragments.as_ref().clone(),
            names: names.to_vec(),
            indices: dataset
                .load_indices()
                .await?
                .iter()
                .filter(|index| names.contains(&index.name))
                .cloned()
                .collect(),
        })
    }

    pub(super) async fn validate(&self, dataset: &Dataset) -> LanceResult<()> {
        if self.uri != dataset.uri() || self.schema != Schema::from(dataset.schema()) {
            return Err(LanceError::invalid_input(
                MaintenanceErrorCode::StalePreparation
                    .tag("index dataset or schema changed; reprepare"),
            ));
        }
        let sources: HashMap<_, _> = dataset
            .manifest()
            .fragments
            .iter()
            .map(|f| (f.id, f))
            .collect();
        if self.fragments.iter().any(|old| {
            let Some(current) = sources.get(&old.id).copied() else {
                return true;
            };
            // Current deletion masks still hide indexed rows. Appended
            // fragments remain uncovered; every other source change is stale.
            let mut expected = old.clone();
            expected.deletion_file = current.deletion_file.clone();
            current != &expected
        }) {
            return Err(LanceError::invalid_input(
                MaintenanceErrorCode::StalePreparation
                    .tag("index source fragment changed; reprepare"),
            ));
        }
        let indices: Vec<_> = dataset
            .load_indices()
            .await?
            .iter()
            .filter(|index| self.names.contains(&index.name))
            .cloned()
            .collect();
        if indices != self.indices {
            return Err(LanceError::invalid_input(
                MaintenanceErrorCode::StalePreparation.tag("index metadata changed; reprepare"),
            ));
        }
        Ok(())
    }
}

/// Immutable index segments prepared without table ownership. Publication uses
/// a separately opened handle and Lance's original read-version transaction.
#[derive(Debug)]
pub struct PreparedIndexOptimization {
    source: PreparedIndexSource,
    results: Vec<IndexSegmentResult>,
}

impl StorageBase {
    /// Run the same plan/build/merge stages as `Dataset::optimize_indices`,
    /// leaving manifest publication to the caller's existing commit protocol.
    /// Open this handle inside a preparation-only write scope.
    pub async fn prepare_index_optimization(
        &self,
        names: Vec<String>,
    ) -> LanceResult<PreparedIndexOptimization> {
        self.ensure_writable()?;
        let source = PreparedIndexSource::capture(&self.dataset, &names).await?;
        let options = OptimizeOptions::merge(usize::MAX).index_names(names);
        let plan = plan_optimize_indices(&self.dataset, &options).await?;
        let mut built = Vec::with_capacity(plan.tasks.len());
        for task in &plan.tasks {
            built.extend(task.execute(&self.dataset).await?);
        }
        let mut results = Vec::with_capacity(plan.merges.len());
        for merge in &plan.merges {
            results.extend(merge.execute(&self.dataset, &built).await?);
        }
        Ok(PreparedIndexOptimization { source, results })
    }

    /// Appends and deletion masks can advance while files are built. Reject
    /// rewritten sources or replaced indexes rather than publish stale files.
    pub async fn commit_prepared_index_optimization(
        &mut self,
        prepared: PreparedIndexOptimization,
    ) -> LanceResult<()> {
        self.ensure_writable()?;
        if prepared.results.is_empty() {
            return Ok(());
        }
        self.dataset.checkout_latest().await?;
        prepared.source.validate(&self.dataset).await?;
        commit_optimize_indices(
            &mut self.dataset,
            prepared.results,
            &OptimizeOptions::merge(usize::MAX).index_names(prepared.source.names),
        )
        .await?;
        self.reload().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field};
    use lance::index::DatasetIndexInternalExt;

    #[derive(Debug)]
    struct NoCommit;
    impl crate::merge_write_scope::CommitAuthorizer for NoCommit {
        fn authorize<'a>(
            &'a self,
            _resource: &'a str,
            _version: u64,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = LanceResult<()>> + Send + 'a>>
        {
            Box::pin(async { Err(LanceError::io("unexpected preparation commit")) })
        }
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "event_id",
            DataType::Int32,
            false,
        )]))
    }

    async fn append(uri: &str, values: Vec<i32>, mode: WriteMode) {
        let batch =
            RecordBatch::try_new(schema(), vec![Arc::new(Int32Array::from(values))]).unwrap();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema()),
            uri,
            Some(WriteParams {
                mode,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    }

    async fn open(uri: &str) -> StorageBase {
        let dataset = StorageBase::load_with_options(uri, None, None)
            .await
            .unwrap();
        StorageBase::from_dataset(dataset, StorageBaseOptions::new(schema(), "event_id"))
            .await
            .unwrap()
    }

    async fn fixture(uri: &str) {
        append(uri, vec![1, 2], WriteMode::Create).await;
        open(uri).await.create_configured_key_index().await.unwrap();
        append(uri, vec![3, 4], WriteMode::Append).await;
    }

    async fn prepare(uri: &str) -> PreparedIndexOptimization {
        let before = Dataset::open(uri).await.unwrap().version().version;
        let scope = crate::merge_write_scope::MergeWriteScope::with_preparation_authorizer(
            Arc::new(NoCommit),
        );
        let prepared = scope
            .run(async {
                open(uri)
                    .await
                    .prepare_index_optimization(vec![ID_INDEX_NAME.into()])
                    .await
                    .unwrap()
            })
            .await;
        assert!(
            scope.completed_steps() > 0,
            "file IO must advance the preparation watchdog"
        );
        assert_eq!(Dataset::open(uri).await.unwrap().version().version, before);
        prepared
    }

    #[tokio::test]
    async fn prepared_optimization_preserves_append_and_partial_delete() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        fixture(uri).await;
        let prepared = prepare(uri).await;
        append(uri, vec![5, 6], WriteMode::Append).await;
        Dataset::open(uri)
            .await
            .unwrap()
            .delete("event_id = 1")
            .await
            .unwrap();
        let mut publisher = open(uri).await;
        publisher
            .commit_prepared_index_optimization(prepared)
            .await
            .unwrap();
        let dataset = publisher.dataset();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);
        assert_eq!(
            dataset
                .unindexed_fragments(ID_INDEX_NAME)
                .await
                .unwrap()
                .len(),
            1
        );
        for value in 1..=6 {
            let mut scan = dataset.scan();
            scan.filter(&format!("event_id = {value}")).unwrap();
            let rows = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert_eq!(
                rows.iter().map(|batch| batch.num_rows()).sum::<usize>(),
                usize::from(value != 1)
            );
        }
        // The next pass covers the append that raced with preparation.
        let prepared = prepare(uri).await;
        open(uri)
            .await
            .commit_prepared_index_optimization(prepared)
            .await
            .unwrap();
        assert!(Dataset::open(uri)
            .await
            .unwrap()
            .unindexed_fragments(ID_INDEX_NAME)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn prepared_optimization_rejects_source_schema_or_index_changes() {
        for change in ["compact", "index", "schema", "delete"] {
            let dir = tempfile::tempdir().unwrap();
            let uri = dir.path().to_str().unwrap();
            fixture(uri).await;
            let prepared = prepare(uri).await;
            let mut publisher = open(uri).await;
            match change {
                "compact" => {
                    // Different index coverage prevents coalescing these two
                    // fragments. Materialize a deletion to force a source rewrite.
                    Dataset::open(uri)
                        .await
                        .unwrap()
                        .delete("event_id = 1")
                        .await
                        .unwrap();
                    let result = publisher
                        .compact(Some(CompactionConfig {
                            target_rows_per_fragment: 1024,
                            ..Default::default()
                        }))
                        .await
                        .unwrap();
                    assert!(result.fragments_removed > 0);
                }
                "index" => publisher.create_configured_key_index().await.unwrap(),
                "schema" => {
                    Dataset::open(uri)
                        .await
                        .unwrap()
                        .add_columns(
                            NewColumnTransform::AllNulls(Arc::new(Schema::new(vec![Field::new(
                                "extra",
                                DataType::Int32,
                                true,
                            )]))),
                            None,
                            None,
                        )
                        .await
                        .unwrap();
                }
                "delete" => {
                    Dataset::open(uri)
                        .await
                        .unwrap()
                        .delete("event_id <= 2")
                        .await
                        .unwrap();
                }
                _ => unreachable!(),
            }
            let version = Dataset::open(uri).await.unwrap().version().version;
            let error = publisher
                .commit_prepared_index_optimization(prepared)
                .await
                .unwrap_err();
            assert!(error.to_string().contains("reprepare"), "{change}: {error}");
            assert_eq!(Dataset::open(uri).await.unwrap().version().version, version);
        }
    }

    #[tokio::test]
    async fn prepared_optimization_rejects_restore_of_deleted_rows() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        fixture(uri).await;
        let before = Dataset::open(uri).await.unwrap();
        let mut deleted = before.clone();
        deleted.delete("event_id = 3").await.unwrap();
        let prepared = prepare(uri).await;
        let mut restored = deleted
            .checkout_version(before.version().version)
            .await
            .unwrap();
        restored.restore().await.unwrap();
        let version = Dataset::open(uri).await.unwrap().version().version;
        open(uri)
            .await
            .commit_prepared_index_optimization(prepared)
            .await
            .unwrap_err();
        assert_eq!(Dataset::open(uri).await.unwrap().version().version, version);
    }

    #[tokio::test]
    async fn prepared_optimization_cannot_publish_with_revoked_authorization() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        fixture(uri).await;
        let prepared = prepare(uri).await;
        let version = Dataset::open(uri).await.unwrap().version().version;
        let error = crate::merge_write_scope::MergeWriteScope::with_preparation_authorizer(
            Arc::new(NoCommit),
        )
        .run(async {
            open(uri)
                .await
                .commit_prepared_index_optimization(prepared)
                .await
                .unwrap_err()
        })
        .await;
        assert!(
            error.to_string().contains("unexpected preparation commit"),
            "{error}"
        );
        assert_eq!(Dataset::open(uri).await.unwrap().version().version, version);
    }
}
