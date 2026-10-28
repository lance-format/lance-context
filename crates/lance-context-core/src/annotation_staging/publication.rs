use super::*;
use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    RecordBatch,
};
use arrow_schema::Schema;
use lance::dataset::{
    InsertBuilder, MergeInsertBuilder, MergeInsertWriteMode, UncommittedMergeInsert, WhenMatched,
    WhenNotMatched, WriteMode, WriteParams,
};
use lance_file::writer::FileWriterOptions;
use lance_table::transaction::{Operation, Transaction};
use std::collections::HashMap;

use crate::{MergeMemoryBudget, MergeReservation};

pub const RESULTS_PROPERTY: &str = "lance-context.annotation-results";
pub const MAX_RESULTS_BYTES: usize = 640 << 10;
pub(super) const METADATA_BUDGET: usize = 8 * MAX_RESULTS_BYTES + 1024;

pub(super) struct Generation {
    pub values: Values,
    base: Arc<views::Base>,
    base_cut: u64,
    scope: Scope,
    through: u64,
    id: String,
    prepared: Mutex<Option<Arc<Prepared>>>,
    results: Option<String>,
    _metadata: Option<OwnedSemaphorePermit>,
}
struct Prepared {
    merge: UncommittedMergeInsert,
    _workspace: MergeReservation,
}

/// Holding a retired generation continues charging its entries and values.
/// It cannot silently become an unbounded queue of old snapshots.
#[derive(Clone)]
pub struct FrozenAnnotations(Arc<Generation>);

/// Exact native transaction, prepared once for this frozen generation.
/// Keep this handle until the original publication outcome is resolved.
pub struct PreparedAnnotations {
    generation: Arc<Generation>,
    prepared: Arc<Prepared>,
}
impl PreparedAnnotations {
    /// Submit this transaction and affected-row set through the fenced commit
    /// path. This accessor neither commits nor grants publication authority.
    pub fn merge(&self) -> &UncommittedMergeInsert {
        &self.prepared.merge
    }
    pub fn base(&self) -> &Arc<Dataset> {
        &self.generation.base.dataset
    }
    pub fn through(&self) -> u64 {
        self.generation.through
    }
}

impl AnnotationStaging {
    /// Freeze a processed cut without copying annotation bodies or map entries.
    /// The caller supplies the publication identity so recovery can rebuild
    /// its original cut and freeze with that same identity. It must durably
    /// record this identity/cut before publishing it.
    pub async fn freeze(&self, transaction: uuid::Uuid) -> Result<Option<FrozenAnnotations>> {
        self.freeze_with_results(transaction, None).await
    }

    /// The caller captures result metadata under the same ordered apply gate as
    /// this cut. Result-only generations publish metadata without rewriting any
    /// data file. Retired handles still charge their metadata against staging.
    pub async fn freeze_with_results(
        &self,
        transaction: uuid::Uuid,
        results: Option<String>,
    ) -> Result<Option<FrozenAnnotations>> {
        if transaction.is_nil() {
            return Err(invalid("nil annotation publication identity"));
        }
        let _mutation = self.mutation.lock().await;
        let mut state = self.state.write().await;
        if state.frozen.is_some() {
            return Err(invalid("an annotation generation is already frozen"));
        }
        if state.active.is_empty() && (results.is_none() || state.applied == state.materialized) {
            return Ok(None);
        }
        let metadata = results
            .as_ref()
            .map(|value| {
                if value.is_empty() || value.len() > MAX_RESULTS_BYTES {
                    return Err(invalid("invalid annotation result metadata size"));
                }
                // Also cover the caller's compact decoded result snapshot. Its
                // retained lifetime is tied to this generation by the service.
                let charge = value
                    .capacity()
                    .checked_mul(4)
                    .and_then(|n| n.checked_add(512))
                    .ok_or_else(|| invalid("annotation metadata charge overflow"))?;
                reserve(&self.metadata, charge)
            })
            .transpose()?;
        let generation = Arc::new(Generation {
            values: std::mem::take(&mut state.active),
            base: state.base.clone(),
            base_cut: state.materialized,
            scope: self.scope.clone(),
            through: state.applied,
            id: transaction.to_string(),
            prepared: Mutex::new(None),
            results,
            _metadata: metadata,
        });
        state.frozen = Some(generation.clone());
        Ok(Some(FrozenAnnotations(generation)))
    }

    /// Atomically install the exact selected Lance snapshot and retire only its
    /// frozen generation. The caller must first confirm its fenced publication
    /// outcome; native transaction identity alone is not a quorum/GC proof.
    pub async fn install_published(
        &self,
        prepared: &PreparedAnnotations,
        selected: Arc<Dataset>,
    ) -> Result<()> {
        let generation = &prepared.generation;
        if selected.uri() != generation.base.uri()
            || selected.version_id() != generation.base.version_id() + 1
            || selected.read_transaction().await?.as_ref() != Some(&prepared.merge().transaction)
        {
            return Err(invalid(
                "selected Lance snapshot is not this prepared annotation transaction",
            ));
        }
        let _mutation = self.mutation.lock().await;
        let state = self.state.read().await;
        if !state
            .frozen
            .as_ref()
            .is_some_and(|f| Arc::ptr_eq(f, generation))
        {
            return Err(invalid("annotation frozen generation is no longer current"));
        }
        // These keys are still masked by frozen (or newer active) values.
        // Cancellation during cache insertion therefore leaves the old view
        // valid. Only this installation writes the cache: a delayed read of an
        // old base must never replace a newer published value. Cached copies
        // own no staging permits or base handles.
        self.published_values
            .remember(&generation.values, generation.through)
            .await?;
        drop(state);
        let mut state = self.state.write().await;
        state.base = state.views.install(selected);
        state.materialized = generation.through;
        state.frozen = None;
        // Never remove active entries by turn ID. They belong to the suffix.
        Ok(())
    }
}

impl FrozenAnnotations {
    pub fn through(&self) -> u64 {
        self.0.through
    }
    pub fn identity(&self) -> &str {
        &self.0.id
    }
    pub fn rows(&self) -> usize {
        self.0.values.len()
    }

    /// Prepare existing narrow values with RewriteColumns and append new narrow
    /// rows. Both file sets belong to one native transaction and one frozen cut.
    /// Registration is restricted to tables containing no turn body columns.
    /// File upload does not publish a manifest or remove the frozen generation.
    /// A second call shares the first exact preparation, including file paths.
    pub async fn prepare(
        &self,
        workspace: &Arc<MergeMemoryBudget>,
        batch_rows: usize,
    ) -> Result<PreparedAnnotations> {
        if !(1..=65_536).contains(&batch_rows) {
            return Err(invalid("invalid annotation publication batch rows"));
        }
        let mut prepared = self.0.prepared.lock().await;
        if let Some(prepared) = prepared.as_ref() {
            return Ok(PreparedAnnotations {
                generation: self.0.clone(),
                prepared: prepared.clone(),
            });
        }
        let bytes = self
            .0
            .values
            .iter()
            .try_fold(0usize, |n, (key, value)| {
                n.checked_add(registration::retained_charge(
                    charge(key, &value.value)?,
                    value.pending_turn.as_ref(),
                )?)
                .ok_or_else(|| invalid("publication size overflow"))
            })?
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.0.results.as_ref().map_or(0, |s| s.len() * 2)))
            .ok_or_else(|| invalid("publication workspace overflow"))?;
        if bytes > workspace.limit() {
            return Err(invalid(
                "annotation frozen generation exceeds publication workspace",
            ));
        }
        let reservation = workspace.reserve(bytes).await;
        // RewriteColumns interleaves original rows with updates. Preserve the
        // base's nullability: an additive handoff can leave null label lists in
        // untouched rows even though every update writes a non-null set.
        let batches = self.batches(batch_rows, false)?;
        let inserts = self.batches(batch_rows, true)?;
        let inserted_rows = inserts.iter().map(RecordBatch::num_rows).sum::<usize>();
        let updated_rows = self.rows() - inserted_rows;
        let mut merge = if batches.is_empty() {
            UncommittedMergeInsert {
                transaction: Transaction::new_from_version(
                    self.0.base.version_id(),
                    Operation::UpdateConfig {
                        config_updates: None,
                        table_metadata_updates: None,
                        schema_metadata_updates: None,
                        field_metadata_updates: HashMap::new(),
                    },
                ),
                affected_rows: None,
                stats: Default::default(),
                inserted_rows_filter: None,
            }
        } else {
            MergeInsertBuilder::try_new(self.0.base.dataset.clone(), vec!["id".into()])?
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .write_mode(MergeInsertWriteMode::RewriteColumns)
                .conflict_retries(0)
                .skip_auto_cleanup(true)
                .try_build()?
                .execute_uncommitted_batches(batches)
                .await?
        };
        if merge.stats.num_updated_rows != updated_rows as u64
            || merge.stats.num_inserted_rows != 0
            || merge.stats.num_deleted_rows != 0
        {
            return Err(invalid(
                "annotation publication did not update exactly the frozen turns",
            ));
        }
        if updated_rows != 0 {
            super::preserve_id::preserve_id(&self.0.base, &mut merge.transaction.operation)?;
        }
        if inserted_rows != 0 {
            registration::validate_target(&self.0.base)?;
            let buffer = workspace.limit() / 4;
            let append = InsertBuilder::new(self.0.base.dataset.clone())
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    max_bytes_per_file: workspace.limit(),
                    skip_auto_cleanup: true,
                    file_writer_options: Some(FileWriterOptions {
                        data_cache_bytes: Some(buffer as u64),
                        max_page_bytes: Some(buffer as u64),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .execute_uncommitted(inserts)
                .await?;
            let Operation::Append { fragments } = append.operation else {
                return Err(invalid("new annotations did not prepare append fragments"));
            };
            if updated_rows == 0 {
                merge.transaction.operation = Operation::Append { fragments };
            } else if let Operation::Update { new_fragments, .. } = &mut merge.transaction.operation
            {
                // preserve_id already checked the update-only preparation.
                *new_fragments = fragments;
            } else {
                return Err(invalid("unexpected mixed annotation preparation"));
            }
            merge.stats.num_inserted_rows = inserted_rows as u64;
        }
        merge.transaction.uuid = self.0.id.clone();
        let cut = serde_json::to_string(&(&self.0.scope, self.0.through))
            .map_err(|e| invalid(e.to_string()))?;
        let mut properties = HashMap::from([("lance-context.annotation-cut".into(), cut)]);
        if let Some(results) = &self.0.results {
            properties.insert(RESULTS_PROPERTY.into(), results.clone());
        }
        merge.transaction.transaction_properties = Some(Arc::new(properties));
        let result = Arc::new(Prepared {
            merge,
            _workspace: reservation,
        });
        *prepared = Some(result.clone());
        Ok(PreparedAnnotations {
            generation: self.0.clone(),
            prepared: result,
        })
    }

    fn batches(&self, batch_rows: usize, inserting: bool) -> Result<Vec<RecordBatch>> {
        let mut values = self
            .0
            .values
            .iter()
            .filter(|(_, stored)| {
                stored
                    .pending_turn
                    .as_ref()
                    .is_some_and(|p| p.through > self.0.base_cut)
                    == inserting
            })
            .peekable();
        if values.peek().is_none() {
            return Ok(Vec::new());
        }
        let base_schema = Schema::from(self.0.base.schema());
        let names: &[&str] = if inserting {
            &["id", "rollout_id", "labels", "status", "score"]
        } else {
            &["id", "labels", "status", "score"]
        };
        let fields = names
            .iter()
            .map(|name| base_schema.field_with_name(name).cloned())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let schema = Arc::new(Schema::new(fields));
        let mut batches = Vec::new();
        while values.peek().is_some() {
            let mut ids = StringBuilder::new();
            let mut groups = StringBuilder::new();
            let mut labels = ListBuilder::new(StringBuilder::new());
            let mut status = StringBuilder::new();
            let mut scores = Vec::new();
            for (key, stored) in values.by_ref().take(batch_rows) {
                ids.append_value(key);
                if inserting {
                    groups.append_value(&stored.pending_turn.as_ref().unwrap().rollout_id);
                }
                for label in &stored.value.labels {
                    labels.values().append_value(label);
                }
                labels.append(true);
                status.append_option(stored.value.status.as_deref());
                scores.push(stored.value.score);
            }
            let mut columns: Vec<Arc<dyn Array>> = vec![Arc::new(ids.finish())];
            if inserting {
                columns.push(Arc::new(groups.finish()));
            }
            columns.extend([
                Arc::new(labels.finish()) as Arc<dyn Array>,
                Arc::new(status.finish()),
                Arc::new(Float32Array::from(scores)),
            ]);
            batches.push(RecordBatch::try_new(schema.clone(), columns)?);
        }
        Ok(batches)
    }
}
