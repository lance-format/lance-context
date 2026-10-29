//! Bounded current-annotation views over a real, fixed Lance snapshot.
//!
//! This is the view/publication component, not a durable mutation endpoint.
//! Its caller must supply ordered, deduplicated committed mutations, persist
//! frozen publication identity/coverage, fence the Lance commit, and establish
//! a read barrier before serving another ingress. A successful `apply` alone
//! is not permission to acknowledge a client or reclaim WAL records.
use std::{collections::BTreeMap, sync::Arc};

use arrow_array::{Array, Float32Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field};
use datafusion::prelude::{col, lit};
use futures::TryStreamExt;
use lance::{Dataset, Error, Result};
use lance_context_api::annotation::{TurnAnnotation, TurnAnnotationPatch};
use tokio::sync::{Mutex, OwnedSemaphorePermit, RwLock, Semaphore};

mod body;
mod compaction;
mod preserve_id;
mod publication;
mod published_values;
mod registration;
mod views;
pub use body::{BodySnapshot, TurnBodySource, TurnRead, BODY_SNAPSHOT_PROPERTY};
pub use publication::{
    FrozenAnnotations, PreparedAnnotations, MAX_RESULTS_BYTES, RESULTS_PROPERTY,
};
use registration::{Batch, Input, PendingTurn};
pub use registration::{NewTurnSource, Registration};
pub use views::ProtectedBases;
#[cfg(test)]
mod tests;

/// Limits for retained annotation values/map nodes and captured point reads.
/// Lance scan/merge workspace and RPC queues require separate admission.
#[derive(Clone, Debug, ::serde::Serialize, ::serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub retained_bytes: usize,
    pub retained_entries: usize,
    pub value_bytes: usize,
    pub readers: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            retained_bytes: 64 << 20,
            retained_entries: 100_000,
            value_bytes: 64 << 10,
            readers: 128,
        }
    }
}
impl Limits {
    pub fn validate(&self) -> Result<()> {
        if self.value_bytes < 512
            || self.value_bytes > self.retained_bytes / 2
            || self.retained_bytes > u32::MAX as usize
            || self.retained_entries == 0
            || self.retained_entries > 1_000_000
            || !(1..=4096).contains(&self.readers)
        {
            return Err(invalid("invalid annotation staging limits"));
        }
        Ok(())
    }
}

/// Each position is scoped to one table instance and one shard in one group.
/// It is not a proof that every other shard in this group has been published.
#[derive(Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize)]
pub struct Scope {
    pub table_instance: String,
    pub source_group: u64,
    pub shard: u32,
}

struct Stored {
    value: TurnAnnotation,
    pending_turn: Option<PendingTurn>,
    _bytes: OwnedSemaphorePermit,
    _entry: Arc<OwnedSemaphorePermit>,
}
type Values = BTreeMap<String, Arc<Stored>>;

/// One mutation within an atomic, already committed WAL batch. Several
/// mutations may target the same turn; their order within the batch matters.
pub struct Mutation<'a> {
    pub turn: &'a str,
    pub patch: &'a TurnAnnotationPatch,
}

/// A deterministic rejection of the whole committed batch. No partial values
/// are installed. Persist this outcome before releasing its recovery evidence.
#[derive(Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize)]
pub struct Rejection {
    pub row: usize,
    pub reason: RejectionReason,
}

#[derive(Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize)]
pub enum RejectionReason {
    InvalidPatch,
    InvalidTurn,
    MissingTurn,
    ValueTooLarge,
    BatchTooLarge,
}

/// Application pressure is not a mutation rejection and must not advance the
/// ordered cursor. Storage/invariant errors are returned separately as errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyOutcome {
    Applied,
    Rejected(Rejection),
    Backpressure,
}

enum PendingBatch {
    Values(Values),
    Rejected(Rejection),
    Backpressure,
}

struct State {
    base: Arc<views::Base>,
    views: views::Registry,
    applied: u64,
    materialized: u64,
    active: Values,
    frozen: Option<Arc<publication::Generation>>,
}

/// One active generation and at most one frozen generation. Published narrow
/// values may share the existing, bounded Lance session cache for later apply.
/// Point reads retain only a selected value or a base handle, never whole maps.
pub struct AnnotationStaging {
    scope: Scope,
    limits: Limits,
    // Writers queue here before requesting exclusive state access. Cold apply
    // holds only shared state access, so installed values remain readable even
    // when another mutation or publication is waiting behind that IO.
    mutation: Mutex<()>,
    state: RwLock<State>,
    bytes: Arc<Semaphore>,
    entries: Arc<Semaphore>,
    readers: Arc<Semaphore>,
    metadata: Arc<Semaphore>,
    root_snapshot: Arc<Semaphore>,
    published_values: published_values::PublishedValues,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub applied: u64,
    pub materialized: u64,
    pub active_entries: usize,
    pub frozen_entries: usize,
    /// Conservative allocation charges, including held values/generations.
    pub retained_bytes: usize,
    pub retained_entries: usize,
    /// Separately bounded publication metadata, including held retired handles.
    /// Data admission cannot occupy this reserve and prevent its own flush.
    pub retained_metadata_bytes: usize,
    pub readers: usize,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid_input(message.into())
}
fn reserve(budget: &Arc<Semaphore>, bytes: usize) -> Result<OwnedSemaphorePermit> {
    budget
        .clone()
        .try_acquire_many_owned(u32::try_from(bytes).map_err(|_| invalid("budget overflow"))?)
        .map_err(|_| invalid("annotation staging capacity exhausted"))
}

// Overcharge map/Arc/BTreeSet nodes rather than counting only UTF-8 payload.
// Each individual label is charged for a whole BTree node. This is admission
// accounting, not a measurement of allocator RSS or Lance's internal workspace.
fn charge(turn: &str, value: &TurnAnnotation) -> Result<usize> {
    let mut bytes = 512usize
        .checked_add(turn.len())
        .and_then(|n| n.checked_add(value.status.as_ref().map_or(0, String::capacity)))
        .ok_or_else(|| invalid("annotation charge overflow"))?;
    for label in &value.labels {
        bytes = bytes
            .checked_add(512)
            .and_then(|n| n.checked_add(label.capacity()))
            .ok_or_else(|| invalid("annotation charge overflow"))?;
    }
    Ok(bytes)
}

fn validate_schema(base: &Dataset) -> Result<()> {
    let fields = [
        ("id", DataType::Utf8),
        (
            "labels",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        ),
        ("status", DataType::Utf8),
        ("score", DataType::Float32),
    ];
    for (name, expected) in fields {
        let field = base
            .schema()
            .field(name)
            .ok_or_else(|| invalid(format!("missing annotation field {name}")))?;
        if field.data_type() != expected {
            return Err(invalid(format!("invalid annotation field {name}")));
        }
    }
    Ok(())
}

impl AnnotationStaging {
    /// `materialized` must come from the selected durable publication, not a
    /// local "flush finished" marker. Reconstruction applies only its suffix.
    pub fn new(
        base: Arc<Dataset>,
        scope: Scope,
        materialized: u64,
        limits: Limits,
    ) -> Result<Self> {
        validate_schema(&base)?;
        limits.validate()?;
        if scope.table_instance.is_empty()
            || scope.table_instance.len() > 256
            || scope.source_group == 0
            || materialized == u64::MAX
        {
            return Err(invalid("invalid annotation staging scope or limits"));
        }
        let published_values = published_values::PublishedValues::new(&base);
        let (base, views) = views::Registry::new(base);
        Ok(Self {
            scope,
            bytes: Arc::new(Semaphore::new(limits.retained_bytes)),
            entries: Arc::new(Semaphore::new(limits.retained_entries)),
            readers: Arc::new(Semaphore::new(limits.readers)),
            metadata: Arc::new(Semaphore::new(publication::METADATA_BUDGET)),
            root_snapshot: Arc::new(Semaphore::new(1)),
            published_values,
            limits,
            mutation: Mutex::new(()),
            state: RwLock::new(State {
                base,
                views,
                applied: materialized,
                materialized,
                active: Values::new(),
                frozen: None,
            }),
        })
    }

    /// Apply one already-deduplicated mutation in the caller's ordered apply
    /// path. `after` must match the previous applied position; `through` may
    /// skip unrelated/control entries only after the caller has processed them.
    /// Every error leaves both the value and applied position unchanged.
    /// Capacity failure must stop/retry application, never discard the WAL item.
    pub async fn apply(
        &self,
        after: u64,
        through: u64,
        turn: &str,
        patch: &TurnAnnotationPatch,
    ) -> Result<()> {
        self.apply_batch(after, through, &[Mutation { turn, patch }])
            .await
    }

    /// Apply all mutations from one WAL batch before advancing its position.
    /// Validation/capacity failure cannot leave a partially visible batch.
    /// Admission must bound each committed batch so it can fit after draining.
    pub async fn apply_batch(
        &self,
        after: u64,
        through: u64,
        mutations: &[Mutation<'_>],
    ) -> Result<()> {
        match self
            .apply_inner(after, through, Batch::Patches(mutations), false)
            .await?
        {
            ApplyOutcome::Applied => Ok(()),
            ApplyOutcome::Rejected(rejection) => Err(invalid(format!(
                "annotation mutation rejected: {rejection:?}"
            ))),
            ApplyOutcome::Backpressure => Err(invalid("annotation staging capacity exhausted")),
        }
    }

    /// Apply one deduplicated, quorum-committed batch. A permanent rejection
    /// advances the cursor without changing any values, allowing later records
    /// to make progress. Backpressure and storage errors leave it unchanged.
    /// The caller must retain the returned result independently of dirty values
    /// and include it in durable recovery state before closing the source WAL.
    /// Limits affecting rejection must remain fixed across replay.
    pub async fn apply_committed(
        &self,
        after: u64,
        through: u64,
        mutations: &[Mutation<'_>],
    ) -> Result<ApplyOutcome> {
        self.apply_inner(after, through, Batch::Patches(mutations), true)
            .await
    }

    /// Advance through already processed control/unrelated entries. This must
    /// never skip an unapplied mutation; a bare quorum index is not that proof.
    pub async fn advance_processed(&self, after: u64, through: u64) -> Result<()> {
        let _mutation = self.mutation.lock().await;
        let mut state = self.state.write().await;
        check_order(&state, after, through)?;
        state.applied = through;
        Ok(())
    }

    async fn apply_inner(
        &self,
        after: u64,
        through: u64,
        mutations: Batch<'_, '_>,
        advance_rejection: bool,
    ) -> Result<ApplyOutcome> {
        let _mutation = self.mutation.lock().await;
        let pending = {
            let state = self.state.read().await;
            check_order(&state, after, through)?;
            self.pending_batch(&state, through, mutations).await?
        };
        // The mutation gate keeps the base, values and applied cut unchanged
        // while upgrading. Cancellation before this point installs nothing.
        let mut state = self.state.write().await;
        match pending {
            PendingBatch::Values(pending) => {
                // No awaits/fallible operations after installing the whole batch.
                state.active.extend(pending);
                state.applied = through;
                Ok(ApplyOutcome::Applied)
            }
            PendingBatch::Rejected(rejection) => {
                if advance_rejection {
                    state.applied = through;
                }
                Ok(ApplyOutcome::Rejected(rejection))
            }
            PendingBatch::Backpressure => Ok(ApplyOutcome::Backpressure),
        }
    }

    async fn pending_batch(
        &self,
        state: &State,
        through: u64,
        mutations: Batch<'_, '_>,
    ) -> Result<PendingBatch> {
        if mutations.len() == 0 || mutations.len() > 65_536 {
            return Ok(reject(0, RejectionReason::BatchTooLarge));
        }
        if matches!(mutations, Batch::Registrations(_)) {
            registration::validate_target(&state.base)?;
        }
        for row in 0..mutations.len() {
            let input = mutations.row(row);
            let result = match input {
                Input::Patch { turn, patch } => self.validate_input(turn, patch),
                Input::Register(value) => registration::validate(value, self.limits.value_bytes),
            };
            if let Err(reason) = result {
                return Ok(reject(row, reason));
            }
        }
        let mut pending = Values::new();
        let mut batch_bytes = 0;
        for start in (0..mutations.len()).step_by(COLD_LOOKUP_KEYS) {
            let end = (start + COLD_LOOKUP_KEYS).min(mutations.len());
            let mut keys = (start..end)
                .map(|row| mutations.row(row).turn())
                .filter(|turn| {
                    !pending.contains_key(*turn)
                        && !state.active.contains_key(*turn)
                        && !state
                            .frozen
                            .as_ref()
                            .is_some_and(|f| f.values.contains_key(*turn))
                })
                .collect::<std::collections::BTreeSet<_>>();
            let mut cached = BTreeMap::new();
            for turn in &keys {
                if let Some(value) = self.published_values.get(turn).await {
                    cached.insert(*turn, value);
                }
            }
            keys.retain(|turn| !cached.contains_key(turn));
            let cold = match mutations {
                Batch::Patches(_) => read_bases(&state.base, &keys).await?,
                Batch::Registrations(_) => read_existing(&state.base, &keys, &["id"]).await?,
            };
            for row in start..end {
                let input = mutations.row(row);
                let turn = input.turn();
                let previous = pending
                    .get(turn)
                    .cloned()
                    .or_else(|| state.active.get(turn).cloned())
                    .or_else(|| {
                        state
                            .frozen
                            .as_ref()
                            .and_then(|f| f.values.get(turn).cloned())
                    });
                let exists =
                    previous.is_some() || cached.contains_key(turn) || cold.contains_key(turn);
                if matches!(input, Input::Register(_)) && exists {
                    // Re-import must neither reset annotations nor create dirty
                    // values/files for a turn already visible at this cut.
                    continue;
                }
                let (current, pending_turn) = match input {
                    Input::Register(value) => (
                        value.annotation.clone(),
                        Some(PendingTurn {
                            rollout_id: value.rollout_id.to_owned(),
                            through,
                            row: row as u32,
                        }),
                    ),
                    Input::Patch { patch, .. } => {
                        let mut current = match &previous {
                            Some(value) => value.value.clone(),
                            None if cached.contains_key(turn) => cached[turn].value.clone(),
                            None => match cold.get(turn) {
                                Some(batch) => {
                                    decode_base_row(batch, 0, turn, self.limits.value_bytes)?
                                }
                                None => return Ok(reject(row, RejectionReason::MissingTurn)),
                            },
                        };
                        patch
                            .apply_to(&mut current)
                            .map_err(|e| invalid(e.to_string()))?;
                        let pending_turn = previous
                            .as_ref()
                            .and_then(|v| v.pending_turn.as_ref())
                            .filter(|v| v.through > state.materialized)
                            .cloned();
                        (current, pending_turn)
                    }
                };
                let value_bytes = charge(turn, &current)?;
                if value_bytes > self.limits.value_bytes {
                    return Ok(reject(row, RejectionReason::ValueTooLarge));
                }
                let bytes = registration::retained_charge(value_bytes, pending_turn.as_ref())?;
                // Patch rejection cannot depend on transient new-row metadata:
                // publication may retire that metadata before suffix replay.
                batch_bytes += if matches!(input, Input::Register(_)) {
                    bytes
                } else {
                    value_bytes
                };
                if let Some(old) = pending.get(turn) {
                    batch_bytes -= charge(turn, &old.value)?;
                }
                // Reject batches that could never fit, even with empty staging.
                // Leave half the byte budget for an old value held during replace.
                if batch_bytes > self.limits.retained_bytes / 2
                    || (!pending.contains_key(turn)
                        && pending.len() == self.limits.retained_entries)
                {
                    return Ok(reject(row, RejectionReason::BatchTooLarge));
                }
                let Ok(allocation) = self.bytes.clone().try_acquire_many_owned(bytes as u32) else {
                    return Ok(PendingBatch::Backpressure);
                };
                let entry = match pending.get(turn).or_else(|| state.active.get(turn)) {
                    Some(old) => old._entry.clone(),
                    None => {
                        let Ok(entry) = self.entries.clone().try_acquire_owned() else {
                            return Ok(PendingBatch::Backpressure);
                        };
                        Arc::new(entry)
                    }
                };
                pending.insert(
                    turn.to_owned(),
                    Arc::new(Stored {
                        value: current,
                        pending_turn,
                        _bytes: allocation,
                        _entry: entry,
                    }),
                );
            }
        }
        Ok(PendingBatch::Values(pending))
    }

    fn validate_input(
        &self,
        turn: &str,
        patch: &TurnAnnotationPatch,
    ) -> std::result::Result<(), RejectionReason> {
        patch
            .validate()
            .map_err(|_| RejectionReason::InvalidPatch)?;
        if turn.is_empty() || turn.len() > self.limits.value_bytes {
            return Err(RejectionReason::InvalidTurn);
        }
        // Reject oversized input before cloning its strings into current state.
        let input_bytes = patch
            .add_labels
            .iter()
            .chain(&patch.remove_labels)
            .try_fold(0usize, |n, v| n.checked_add(512)?.checked_add(v.len()))
            .and_then(|n| match &patch.status {
                lance_context_api::annotation::AnnotationFieldUpdate::Set(v) => {
                    n.checked_add(v.len())
                }
                _ => Some(n),
            })
            .ok_or(RejectionReason::ValueTooLarge)?;
        if input_bytes > self.limits.value_bytes {
            return Err(RejectionReason::ValueTooLarge);
        }
        Ok(())
    }

    /// Capture a compatible view. A base miss carries THIS base into the read,
    /// even if publication retires frozen before the storage read completes.
    /// This does not establish a distributed read barrier by itself.
    pub async fn capture(&self, turn: &str) -> Result<PointRead> {
        Ok(self.capture_inner(turn, false).await?.0)
    }

    /// Capture annotation and body location under the same view lock. The
    /// service still establishes a read barrier and retains the referenced WAL
    /// frame before releasing its ordered source-window gate.
    pub async fn capture_turn(&self, turn: &str) -> Result<TurnRead> {
        let (annotations, source) = self.capture_inner(turn, true).await?;
        Ok(TurnRead {
            annotations,
            source: source.expect("full capture includes body source"),
        })
    }

    async fn capture_inner(
        &self,
        turn: &str,
        body: bool,
    ) -> Result<(PointRead, Option<body::CapturedBody>)> {
        if turn.is_empty() || turn.len() > self.limits.value_bytes {
            return Err(invalid("invalid annotation turn key"));
        }
        let permit = reserve(&self.readers, 1)?;
        let state = self.state.read().await;
        let value = state.active.get(turn).cloned().or_else(|| {
            state
                .frozen
                .as_ref()
                .and_then(|f| f.values.get(turn).cloned())
        });
        let body = body.then(|| {
            value
                .as_ref()
                .and_then(|v| v.pending_turn.as_ref())
                .filter(|p| p.through > state.materialized)
                .map_or_else(
                    || body::CapturedBody::Base(state.base.clone()),
                    |p| body::CapturedBody::Staged(p.source()),
                )
        });
        Ok((
            PointRead {
                applied: state.applied,
                base_version: state.base.version_id(),
                source: value.map_or_else(
                    || PointSource::Base {
                        base: state.base.clone(),
                        turn: turn.to_owned(),
                        published: self.published_values.clone(),
                        materialized: state.materialized,
                    },
                    PointSource::Staged,
                ),
                value_bytes: self.limits.value_bytes,
                _reader: permit,
            },
            body,
        ))
    }

    pub async fn observation(&self) -> Observation {
        let state = self.state.read().await;
        Observation {
            applied: state.applied,
            materialized: state.materialized,
            active_entries: state.active.len(),
            frozen_entries: state.frozen.as_ref().map_or(0, |f| f.values.len()),
            retained_bytes: self.limits.retained_bytes - self.bytes.available_permits(),
            retained_entries: self.limits.retained_entries - self.entries.available_permits(),
            retained_metadata_bytes: publication::METADATA_BUDGET
                - self.metadata.available_permits(),
            readers: self.limits.readers - self.readers.available_permits(),
        }
    }
}

fn check_order(state: &State, after: u64, through: u64) -> Result<()> {
    if after != state.applied || through <= after || through == u64::MAX {
        return Err(invalid("annotation apply is not an ordered suffix"));
    }
    Ok(())
}

fn reject(row: usize, reason: RejectionReason) -> PendingBatch {
    PendingBatch::Rejected(Rejection { row, reason })
}

enum PointSource {
    Staged(Arc<Stored>),
    Base {
        base: Arc<views::Base>,
        turn: String,
        published: published_values::PublishedValues,
        materialized: u64,
    },
}
pub struct PointRead {
    pub applied: u64,
    pub base_version: u64,
    source: PointSource,
    value_bytes: usize,
    _reader: OwnedSemaphorePermit,
}
impl PointRead {
    /// Resolve only id/labels/status/score. No body projection or WAL access.
    /// Hold this handle for the duration of the read; its admission bounds
    /// old base handles and old staged values during publication handoff.
    pub async fn resolve(&self) -> Result<Option<TurnAnnotation>> {
        match &self.source {
            PointSource::Staged(value) => Ok(Some(value.value.clone())),
            PointSource::Base {
                base,
                turn,
                published,
                materialized,
            } => match published.get_at(turn, *materialized).await {
                Some(value) => Ok(Some(value.value.clone())),
                None => read_base(base, turn, self.value_bytes).await,
            },
        }
    }
}

async fn read_base(base: &Dataset, turn: &str, max_bytes: usize) -> Result<Option<TurnAnnotation>> {
    let mut scan = base.scan();
    scan.project(&["labels", "status", "score"])?;
    scan.filter_expr(col("id").eq(lit(turn)));
    scan.limit(Some(2), None)?;
    // The two-row result limit detects duplicates. Scan vectors can be larger
    // without returning more matches or projecting the turn body.
    scan.batch_size(COLD_LOOKUP_ROWS);
    let mut stream = scan.try_into_stream().await?;
    let mut value = None;
    while let Some(batch) = stream.try_next().await? {
        for row in 0..batch.num_rows() {
            if value.is_some() {
                return Err(invalid("duplicate annotation turn key in Lance"));
            }
            value = Some(decode_base_row(&batch, row, turn, max_bytes)?);
        }
    }
    Ok(value)
}

// Scanner vectors and query key windows are independent. A 2048-row mutation
// batch needs at most one cold query, while point reads retain small vectors.
// Values are decoded in mutation order; slices share projected Arrow buffers.
const COLD_LOOKUP_ROWS: usize = 128;
const COLD_LOOKUP_KEYS: usize = 2048;

async fn read_bases<'a>(
    base: &Dataset,
    turns: &std::collections::BTreeSet<&'a str>,
) -> Result<BTreeMap<&'a str, RecordBatch>> {
    read_existing(base, turns, &["labels", "status", "score", "id"]).await
}

// Re-import only needs existence; keep annotations out of that projection.
async fn read_existing<'a>(
    base: &Dataset,
    turns: &std::collections::BTreeSet<&'a str>,
    projection: &[&str],
) -> Result<BTreeMap<&'a str, RecordBatch>> {
    let mut rows = BTreeMap::new();
    if turns.is_empty() {
        return Ok(rows);
    }
    let mut scan = base.scan();
    scan.project(projection)?;
    scan.filter_expr(col("id").in_list(turns.iter().map(|turn| lit(*turn)).collect(), false));
    // A valid table returns at most one row per requested key. One extra row
    // proves a duplicate without retaining an unbounded corrupt result set.
    scan.limit(Some(turns.len() as i64 + 1), None)?;
    // Existence probes decode only IDs. Amortize scanning a growing base over
    // the existing key window without increasing annotation/body projections.
    scan.batch_size(if projection == ["id"] {
        COLD_LOOKUP_KEYS
    } else {
        COLD_LOOKUP_ROWS
    });
    let mut stream = scan.try_into_stream().await?;
    while let Some(batch) = stream.try_next().await? {
        let ids = batch
            .column_by_name("id")
            .ok_or_else(|| invalid("missing annotation id"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| invalid("invalid annotation id"))?;
        for row in 0..batch.num_rows() {
            let turn = ids
                .is_valid(row)
                .then(|| ids.value(row))
                .and_then(|id| turns.get(id))
                .copied()
                .ok_or_else(|| invalid("unexpected annotation lookup key"))?;
            if rows.insert(turn, batch.slice(row, 1)).is_some() {
                return Err(invalid("duplicate annotation turn key in Lance"));
            }
        }
    }
    Ok(rows)
}

fn decode_base_row(
    batch: &RecordBatch,
    row: usize,
    turn: &str,
    max_bytes: usize,
) -> Result<TurnAnnotation> {
    let labels = batch
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| invalid("invalid annotation labels"))?;
    let status = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| invalid("invalid annotation status"))?;
    let score = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| invalid("invalid annotation score"))?;
    let strings = labels.value(row);
    let strings = strings
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| invalid("invalid annotation label values"))?;
    // Additive migration can represent an unset label set as a virtual
    // all-null column. Its logical value is empty; ignore hidden values
    // beneath a null list slot. Published annotations write actual sets.
    let label_count = if labels.is_null(row) {
        0
    } else {
        strings.len()
    };
    // Bound owned-value allocation before copying projected strings.
    // The scanner's Arrow buffers still use its separate Lance budget.
    let mut projected_bytes = 512usize
        .checked_add(turn.len())
        .and_then(|n| {
            n.checked_add(if status.is_null(row) {
                0
            } else {
                status.value(row).len()
            })
        })
        .ok_or_else(|| invalid("base annotation size overflow"))?;
    for label in strings.iter().take(label_count) {
        let label = label
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("null/empty annotation label"))?;
        projected_bytes = projected_bytes
            .checked_add(512)
            .and_then(|n| n.checked_add(label.len()))
            .ok_or_else(|| invalid("base annotation size overflow"))?;
    }
    if projected_bytes > max_bytes {
        return Err(invalid("base annotation exceeds value budget"));
    }
    let mut current = TurnAnnotation {
        status: (!status.is_null(row)).then(|| status.value(row).to_owned()),
        score: (!score.is_null(row)).then(|| score.value(row)),
        ..Default::default()
    };
    for label in strings.iter().take(label_count) {
        let label = label
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("null/empty annotation label"))?;
        current.labels.insert(label.to_owned());
    }
    if current.score.is_some_and(|s| !s.is_finite()) || charge(turn, &current)? > max_bytes {
        return Err(invalid("invalid/oversized base annotation"));
    }
    Ok(current)
}
