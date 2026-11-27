//! Publish original full-body Arrow slices in the same transaction as the
//! processed WAL cut and bounded request results. No auxiliary annotation file.
use super::*;
use datafusion::physical_plan::{RecordBatchStream, SendableRecordBatchStream};

/// Rows per streamed write group when writing one owner-batch fragment.
const STREAM_ROWS_PER_GROUP: usize = 8192;

impl FrozenAnnotations {
    /// Prepare exactly the first-create rows exposed by `new_turns`, in their
    /// original schema. The caller validates their key/source ordinals when
    /// slicing frames. `source_owner` must retain the input frames' admission
    /// reservations; the writer keeps it even if the awaiting caller cancels.
    pub async fn prepare_immutable_body(
        &self,
        workspace: &Arc<MergeMemoryBudget>,
        batches: Vec<RecordBatch>,
        source_owner: Arc<dyn Send + Sync>,
    ) -> Result<PreparedAnnotations> {
        if !self.0.immutable_body {
            return Err(invalid("not an immutable body generation"));
        }
        let mut prepared = self.0.prepared.lock().await;
        if let Some(prepared) = prepared.as_ref() {
            return Ok(PreparedAnnotations {
                generation: self.0.clone(),
                prepared: prepared.clone(),
            });
        }
        let schema = Schema::from(self.0.base.schema());
        if batches
            .iter()
            .any(|batch| batch.schema().as_ref() != &schema)
        {
            return Err(invalid("immutable input differs from full body schema"));
        }
        let rows = batches
            .iter()
            .try_fold(0usize, |n, batch| n.checked_add(batch.num_rows()))
            .ok_or_else(|| invalid("immutable body row count overflow"))?;
        if rows != self.new_turns().count() || rows != self.rows() {
            return Err(invalid(
                "immutable input differs from frozen first-create rows",
            ));
        }
        let metadata_bytes = self.0.results.as_ref().map_or(0, |s| s.len() * 2) + 1024;
        if metadata_bytes > workspace.limit() / 2 {
            return Err(invalid("immutable publication metadata exceeds workspace"));
        }
        let reservation = workspace.reserve(workspace.limit() / 2).await;
        let generation = self.0.clone();
        let maximum = workspace.limit();
        let result = tokio::spawn(async move {
            let _source_owner = source_owner;
            let mut transaction = if rows == 0 {
                Transaction::new_from_version(
                    generation.base.version_id(),
                    Operation::UpdateConfig {
                        config_updates: None,
                        table_metadata_updates: None,
                        schema_metadata_updates: None,
                        field_metadata_updates: HashMap::new(),
                    },
                )
            } else {
                InsertBuilder::new(generation.base.dataset.clone())
                    .with_params(&WriteParams {
                        mode: WriteMode::Append,
                        max_bytes_per_file: maximum,
                        skip_auto_cleanup: true,
                        file_writer_options: Some(FileWriterOptions {
                            data_cache_bytes: Some((maximum / 8) as u64),
                            max_page_bytes: Some((maximum / 8) as u64),
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .execute_uncommitted(batches)
                    .await?
            };
            if rows > 0 && !matches!(transaction.operation, Operation::Append { .. }) {
                return Err(invalid("immutable body writer did not produce an append"));
            }
            transaction.uuid = generation.id.clone();
            let cut = serde_json::to_string(&(&generation.scope, generation.through))
                .map_err(|error| invalid(error.to_string()))?;
            let mut properties = HashMap::from([("lance-context.annotation-cut".into(), cut)]);
            if let Some(results) = &generation.results {
                properties.insert(RESULTS_PROPERTY.into(), results.clone());
            }
            transaction.transaction_properties = Some(Arc::new(properties));
            let mut merge = UncommittedMergeInsert {
                transaction,
                affected_rows: None,
                stats: Default::default(),
                inserted_rows_filter: None,
            };
            merge.stats.num_inserted_rows = rows as u64;
            Ok::<_, Error>(Arc::new(Prepared {
                merge,
                body: None,
                _workspace: reservation,
            }))
        })
        .await
        .map_err(|error| Error::internal(format!("immutable body writer stopped: {error}")))??;
        *prepared = Some(result.clone());
        Ok(PreparedAnnotations {
            generation: self.0.clone(),
            prepared: result,
        })
    }

    /// Write ONE fragment from a streamed, already deduplicated owner batch.
    /// The stream yields exactly the `new_turns` rows in the full-body schema,
    /// usually refetched from the WAL and clustered by key. Unlike
    /// `prepare_immutable_body`, input is not materialized: only the writer's
    /// page buffers are resident. File/row limits are lifted so the batch
    /// lands as one data file; `rows` must match the frozen first creates.
    pub async fn prepare_immutable_body_stream(
        &self,
        workspace: &Arc<MergeMemoryBudget>,
        rows: usize,
        stream: SendableRecordBatchStream,
        source_owner: Arc<dyn Send + Sync>,
    ) -> Result<PreparedAnnotations> {
        if !self.0.immutable_body {
            return Err(invalid("not an immutable body generation"));
        }
        let mut prepared = self.0.prepared.lock().await;
        if let Some(prepared) = prepared.as_ref() {
            return Ok(PreparedAnnotations {
                generation: self.0.clone(),
                prepared: prepared.clone(),
            });
        }
        let schema = Schema::from(self.0.base.schema());
        if stream.schema().as_ref() != &schema {
            return Err(invalid("immutable input differs from full body schema"));
        }
        if rows != self.new_turns().count() || rows != self.rows() {
            return Err(invalid(
                "immutable input differs from frozen first-create rows",
            ));
        }
        let metadata_bytes = self.0.results.as_ref().map_or(0, |s| s.len() * 2) + 1024;
        if metadata_bytes > workspace.limit() / 2 {
            return Err(invalid("immutable publication metadata exceeds workspace"));
        }
        let reservation = workspace.reserve(workspace.limit() / 2).await;
        let generation = self.0.clone();
        let maximum = workspace.limit();
        let result = tokio::spawn(async move {
            let _source_owner = source_owner;
            let mut transaction = if rows == 0 {
                drop(stream);
                Transaction::new_from_version(
                    generation.base.version_id(),
                    Operation::UpdateConfig {
                        config_updates: None,
                        table_metadata_updates: None,
                        schema_metadata_updates: None,
                        field_metadata_updates: HashMap::new(),
                    },
                )
            } else {
                // Count rows as they stream so a short or long refetch can
                // never publish a fragment that differs from the decisions.
                let counted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let seen = counted.clone();
                let stream_schema = stream.schema();
                let checked = futures::StreamExt::map(stream, move |batch| {
                    if let Ok(batch) = &batch {
                        seen.fetch_add(batch.num_rows(), std::sync::atomic::Ordering::Relaxed);
                    }
                    batch
                });
                let source: SendableRecordBatchStream = Box::pin(
                    datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
                        stream_schema,
                        checked,
                    ),
                );
                let transaction = InsertBuilder::new(generation.base.dataset.clone())
                    .with_params(&WriteParams {
                        mode: WriteMode::Append,
                        // One fragment per owner batch.
                        max_rows_per_file: usize::MAX,
                        max_rows_per_group: STREAM_ROWS_PER_GROUP,
                        max_bytes_per_file: usize::MAX,
                        skip_auto_cleanup: true,
                        file_writer_options: Some(FileWriterOptions {
                            data_cache_bytes: Some((maximum / 8) as u64),
                            max_page_bytes: Some((maximum / 8) as u64),
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .execute_uncommitted_stream(source)
                    .await?;
                if counted.load(std::sync::atomic::Ordering::Relaxed) != rows {
                    return Err(invalid(
                        "streamed body rows differ from frozen first-create rows",
                    ));
                }
                transaction
            };
            if rows > 0 && !matches!(transaction.operation, Operation::Append { .. }) {
                return Err(invalid("immutable body writer did not produce an append"));
            }
            transaction.uuid = generation.id.clone();
            let cut = serde_json::to_string(&(&generation.scope, generation.through))
                .map_err(|error| invalid(error.to_string()))?;
            let mut properties = HashMap::from([("lance-context.annotation-cut".into(), cut)]);
            if let Some(results) = &generation.results {
                properties.insert(RESULTS_PROPERTY.into(), results.clone());
            }
            transaction.transaction_properties = Some(Arc::new(properties));
            let mut merge = UncommittedMergeInsert {
                transaction,
                affected_rows: None,
                stats: Default::default(),
                inserted_rows_filter: None,
            };
            merge.stats.num_inserted_rows = rows as u64;
            Ok::<_, Error>(Arc::new(Prepared {
                merge,
                body: None,
                _workspace: reservation,
            }))
        })
        .await
        .map_err(|error| Error::internal(format!("immutable body writer stopped: {error}")))??;
        *prepared = Some(result.clone());
        Ok(PreparedAnnotations {
            generation: self.0.clone(),
            prepared: result,
        })
    }
}
