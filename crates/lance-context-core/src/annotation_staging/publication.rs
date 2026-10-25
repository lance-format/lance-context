use super::*;
use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    RecordBatch,
};
use arrow_schema::Schema;
use lance::dataset::{
    MergeInsertBuilder, MergeInsertWriteMode, UncommittedMergeInsert, WhenMatched, WhenNotMatched,
};
use lance_table::transaction::{Operation, Transaction};
use std::collections::HashMap;

use crate::{MergeMemoryBudget, MergeReservation};

pub const RESULTS_PROPERTY: &str = "lance-context.annotation-results";
pub const MAX_RESULTS_BYTES: usize = 640 << 10;
pub(super) const METADATA_BUDGET: usize = 8 * MAX_RESULTS_BYTES + 1024;

pub(super) struct Generation {
    pub values: Values,
    base: Arc<Dataset>,
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
        &self.generation.base
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
        let mut state = self.state.lock().await;
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
        let mut state = self.state.lock().await;
        if !state
            .frozen
            .as_ref()
            .is_some_and(|f| Arc::ptr_eq(f, generation))
        {
            return Err(invalid("annotation frozen generation is no longer current"));
        }
        state.base = selected;
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

    /// Prepare complete narrow values in one native merge, explicitly choosing
    /// RewriteColumns so the default strategy cannot rewrite turn bodies.
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
                n.checked_add(charge(key, &value.value)?)
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
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "labels",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
            Field::new("score", DataType::Float32, true),
        ]));
        let mut batches = Vec::new();
        let mut values = self.0.values.iter().peekable();
        while values.peek().is_some() {
            let mut ids = StringBuilder::new();
            let mut labels = ListBuilder::new(StringBuilder::new());
            let mut status = StringBuilder::new();
            let mut scores = Vec::new();
            for (key, stored) in values.by_ref().take(batch_rows) {
                ids.append_value(key);
                for label in &stored.value.labels {
                    labels.values().append_value(label);
                }
                labels.append(true);
                status.append_option(stored.value.status.as_deref());
                scores.push(stored.value.score);
            }
            batches.push(RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(ids.finish()),
                    Arc::new(labels.finish()),
                    Arc::new(status.finish()),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?);
        }
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
            MergeInsertBuilder::try_new(self.0.base.clone(), vec!["id".into()])?
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .write_mode(MergeInsertWriteMode::RewriteColumns)
                .conflict_retries(0)
                .skip_auto_cleanup(true)
                .try_build()?
                .execute_uncommitted_batches(batches)
                .await?
        };
        if merge.stats.num_updated_rows != self.rows() as u64
            || merge.stats.num_inserted_rows != 0
            || merge.stats.num_deleted_rows != 0
        {
            return Err(invalid(
                "annotation publication did not update exactly the frozen turns",
            ));
        }
        if self.rows() != 0 {
            super::preserve_id::preserve_id(&self.0.base, &mut merge.transaction.operation)?;
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
}
