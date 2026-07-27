use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use chrono::{DateTime, Utc};
use serde_json::Value;
use uuid::Uuid;

/// Current schema version for the append-only datagen checkpoint log.
pub const DATAGEN_SCHEMA_VERSION: i32 = 2;

/// One lifecycle or field-level event in a datagen item's checkpoint history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DatagenEventType {
    ItemCreated,
    StepStarted,
    FieldSet,
    FieldAppend,
    StepCompleted,
    Failed,
    Terminal,
}

impl DatagenEventType {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ItemCreated => "ITEM_CREATED",
            Self::StepStarted => "STEP_STARTED",
            Self::FieldSet => "FIELD_SET",
            Self::FieldAppend => "FIELD_APPEND",
            Self::StepCompleted => "STEP_COMPLETED",
            Self::Failed => "FAILED",
            Self::Terminal => "TERMINAL",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "ITEM_CREATED" => Ok(Self::ItemCreated),
            "STEP_STARTED" => Ok(Self::StepStarted),
            "FIELD_SET" => Ok(Self::FieldSet),
            "FIELD_APPEND" => Ok(Self::FieldAppend),
            "STEP_COMPLETED" => Ok(Self::StepCompleted),
            "FAILED" => Ok(Self::Failed),
            "TERMINAL" => Ok(Self::Terminal),
            other => Err(format!("unsupported datagen event type '{other}'")),
        }
    }
}

/// The composition kind of a step. Drives two behaviors:
///   - forks a stream: `MapReduce`/`Branch`/`SubPipeline` sub-items get their own `item_id`.
///   - drives a frame: `Sequence`/`Loop` are the `enclosing_step` frames; only these emit STEP_STARTED.
///
/// `Conditional`/`Router` are selectors (wrap one chosen child); `Leaf`/`Root` are the endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DatagenStepKind {
    Root,
    Leaf,
    Sequence,
    Loop,
    MapReduce,
    Branch,
    SubPipeline,
    Conditional,
    Router,
}

impl DatagenStepKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Leaf => "leaf",
            Self::Sequence => "sequence",
            Self::Loop => "loop",
            Self::MapReduce => "map_reduce",
            Self::Branch => "branch",
            Self::SubPipeline => "sub_pipeline",
            Self::Conditional => "conditional",
            Self::Router => "router",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "root" => Ok(Self::Root),
            "leaf" => Ok(Self::Leaf),
            "sequence" => Ok(Self::Sequence),
            "loop" => Ok(Self::Loop),
            "map_reduce" => Ok(Self::MapReduce),
            "branch" => Ok(Self::Branch),
            "sub_pipeline" => Ok(Self::SubPipeline),
            "conditional" => Ok(Self::Conditional),
            "router" => Ok(Self::Router),
            other => Err(format!("unsupported datagen step kind '{other}'")),
        }
    }

    /// A driver frame (`Sequence`/`Loop`) is the only kind that emits STEP_STARTED.
    #[must_use]
    pub fn is_driver(self) -> bool {
        matches!(self, Self::Sequence | Self::Loop)
    }
}

/// An item (stream) identity — a structured value stored as a materialized path string. Root ids are
/// built by the executor from a source key (e.g. `5`); sub-item ids extend a parent with one fan-out
/// segment (`5/expand:0`) via [`DatagenItemId::child`]. The store owns the parse<->format; the client
/// only ever holds the structured form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DatagenItemId {
    root_key: String,
    /// One `(origin_step, branch_idx)` per fan-out hop below the root.
    segments: Vec<(String, i64)>,
}

impl DatagenItemId {
    /// Build a root id from the executor's source key (the base case).
    #[must_use]
    pub fn from_source_key(key: &str) -> Self {
        Self {
            root_key: key.to_string(),
            segments: Vec::new(),
        }
    }

    /// Extend this id with one fan-out segment -> the sub-item's id. Pure, deterministic, no I/O.
    /// The sole id-composition entry point. Valid to call for a branch that was never written.
    #[must_use]
    pub fn child(&self, origin_step: &str, branch_idx: i64) -> Self {
        let mut segments = self.segments.clone();
        segments.push((origin_step.to_string(), branch_idx));
        Self {
            root_key: self.root_key.clone(),
            segments,
        }
    }

    /// The parent stream's id (`None` on a root).
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        if self.segments.is_empty() {
            return None;
        }
        let mut segments = self.segments.clone();
        segments.pop();
        Some(Self {
            root_key: self.root_key.clone(),
            segments,
        })
    }

    /// The root of this id's tree (== self if root).
    #[must_use]
    pub fn root(&self) -> Self {
        Self {
            root_key: self.root_key.clone(),
            segments: Vec::new(),
        }
    }

    /// The fan-out step that created this sub-item (`None` on a root).
    #[must_use]
    pub fn origin_step(&self) -> Option<&str> {
        self.segments.last().map(|(step, _)| step.as_str())
    }

    /// Which branch this sub-item is (`None` on a root).
    #[must_use]
    pub fn branch_idx(&self) -> Option<i64> {
        self.segments.last().map(|(_, idx)| *idx)
    }

    #[must_use]
    pub fn is_root(&self) -> bool {
        self.segments.is_empty()
    }

    /// Parse a stored path string back into a structured id.
    pub fn parse(path: &str) -> Result<Self, String> {
        let mut parts = path.split('/');
        let root_key = parts
            .next()
            .filter(|part| !part.is_empty())
            .ok_or_else(|| format!("empty datagen item id '{path}'"))?
            .to_string();
        let mut segments = Vec::new();
        for part in parts {
            let (step, idx) = part
                .split_once(':')
                .ok_or_else(|| format!("malformed item id segment '{part}' in '{path}'"))?;
            let branch_idx = idx
                .parse::<i64>()
                .map_err(|_| format!("non-integer branch index '{idx}' in '{path}'"))?;
            segments.push((step.to_string(), branch_idx));
        }
        Ok(Self { root_key, segments })
    }
}

impl fmt::Display for DatagenItemId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.root_key)?;
        for (step, idx) in &self.segments {
            write!(formatter, "/{step}:{idx}")?;
        }
        Ok(())
    }
}

/// Terminal outcome of an item (the write-side input to [`DatagenStreamWriter::item_terminal`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatagenTerminal {
    Completed,
    Filtered,
}

impl DatagenTerminal {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Filtered => "filtered",
        }
    }
}

/// The value type of the `status` column. Two read lenses: lifecycle {Running, Completed, Filtered}
/// vs failure {Failed}. A folded item's status is only ever a lifecycle value; `Failed` surfaces only
/// through the failure lens (overview / failure history).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DatagenItemStatus {
    Running,
    Completed,
    Filtered,
    Failed,
}

impl DatagenItemStatus {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Filtered => "filtered",
            Self::Failed => "failed",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "filtered" => Ok(Self::Filtered),
            "failed" => Ok(Self::Failed),
            other => Err(format!("unsupported datagen status '{other}'")),
        }
    }
}

/// Lazy reference to an inline blob event. `bytes` is absent on normal (lazy) fold and trajectory
/// reads; callers materialize it through `DatagenStore::load_blob`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatagenBlobValue {
    pub bytes: Option<Vec<u8>>,
    pub size: i64,
    pub checksum: Option<String>,
}

/// Canonical value stored by a FIELD_SET or FIELD_APPEND event.
#[derive(Debug, Clone, PartialEq)]
pub enum DatagenValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Json(Value),
    Blob(DatagenBlobValue),
}

impl DatagenValue {
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Bool(_) => "bool",
            Self::Str(_) => "str",
            Self::Json(_) => "json",
            Self::Blob(_) => "blob",
        }
    }
}

/// A step's identity: its (globally-unique) name + its kind. Name and kind always travel together.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DatagenStepId {
    pub name: String,
    pub kind: DatagenStepKind,
}

/// A position within one stream's step tree — the coordinate a step write is attributed to. Maps 1:1
/// to the Group C provenance columns. `enclosing`/`selector` are stored as bare step *names* (the log
/// has no enclosing/selector kind column); `None` means "directly under the stream root" / "no
/// selector".
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DatagenStreamPosition {
    pub step: DatagenStepId,
    pub index: i64,
    pub enclosing: Option<String>,
    pub selector: Option<String>,
}

/// How a field folds: FIELD_SET replaces (last-writer-wins); FIELD_APPEND accumulates in order.
#[derive(Debug, Clone, PartialEq)]
pub enum DatagenFieldState {
    Set(DatagenValue),
    Appended(Vec<DatagenValue>),
}

/// A pointer to one completed step position (a single STEP_COMPLETED).
#[derive(Debug, Clone, PartialEq)]
pub struct DatagenStepCursor {
    pub position: DatagenStreamPosition,
    /// The STEP_COMPLETED's `item_seq` — the fold cutoff for "state as of this step".
    pub item_seq: i64,
}

/// The ordered list of cursors an item passed through, plus sets for O(1) skip lookup. `completed`
/// gates STEP_COMPLETED (re-)emission; `started` gates STEP_STARTED. `started \ completed` = frames
/// that were open when the process died.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DatagenTrajectory {
    pub ordered: Vec<DatagenStepCursor>,
    pub completed: HashSet<DatagenStreamPosition>,
    pub started: HashSet<DatagenStreamPosition>,
}

/// Error payload, shared by the write side (input to `item_failed`) and the read side (composed into
/// [`DatagenFailure`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatagenErrorInfo {
    pub error_type: String,
    pub error_dump: Option<String>,
    pub traceback: Option<String>,
}

/// One failure record — a lightweight pointer to a FAILED event (no folded item). An item may have
/// 0..N of these across attempts.
#[derive(Debug, Clone, PartialEq)]
pub struct DatagenFailure {
    pub at: DatagenStepCursor,
    pub run_id: String,
    pub attempt: i32,
    pub error: DatagenErrorInfo,
}

/// One item reconstructed by folding its events (latest state). Carries enough to continue processing
/// and to rebuild a `DatagenStreamWriter` on resume. Does not carry failures.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldedDatagenItem {
    pub item_id: DatagenItemId,
    pub root_item_id: DatagenItemId,
    pub parent_item_id: Option<DatagenItemId>,
    pub status: DatagenItemStatus,
    /// Max `item_seq` -> resume continues at `last_item_seq + 1`.
    pub last_item_seq: i64,
    /// Max `attempt` seen -> resume runs at `last_attempt + 1`.
    pub last_attempt: i32,
    pub fields: BTreeMap<String, DatagenFieldState>,
    pub trajectory: DatagenTrajectory,
    pub query_tags: Option<Value>,
    /// Internal `field_name -> event_id` map for the folded blob fields, so the store can resolve a
    /// lazy blob without the caller handling an `event_id`.
    pub blob_event_ids: BTreeMap<String, String>,
}

/// Result of a resumption fold. `NeverStarted` (no ITEM_CREATED) is the fresh-vs-restore fork the
/// executor acts on; `Found` carries the folded item (whose `status` is the lifecycle status).
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)] // `Found` is the common path; boxing it adds indirection to the hot case.
pub enum DatagenItemLookup {
    NeverStarted,
    Found(FoldedDatagenItem),
}

impl DatagenItemLookup {
    #[must_use]
    pub fn folded(&self) -> Option<&FoldedDatagenItem> {
        match self {
            Self::NeverStarted => None,
            Self::Found(item) => Some(item),
        }
    }
}

/// Bulk startup classification of root items. A missing id means "never started".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DatagenRootItemStatuses {
    inner: HashMap<String, DatagenItemStatus>,
}

impl DatagenRootItemStatuses {
    #[must_use]
    pub fn from_map(inner: HashMap<String, DatagenItemStatus>) -> Self {
        Self { inner }
    }

    /// The classified status of a root item, or `None` if it was never started.
    #[must_use]
    pub fn status(&self, item_id: &DatagenItemId) -> Option<DatagenItemStatus> {
        self.inner.get(&item_id.to_string()).copied()
    }

    /// Whether this item reached a terminal lifecycle state (Completed | Filtered).
    #[must_use]
    pub fn is_terminated(&self, item_id: &DatagenItemId) -> bool {
        matches!(
            self.status(item_id),
            Some(DatagenItemStatus::Completed | DatagenItemStatus::Filtered)
        )
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &DatagenItemStatus)> {
        self.inner.iter()
    }
}

/// One append-only row in `log.lance`. Item/root/parent ids are the stored path strings; fold parses
/// them into [`DatagenItemId`].
#[derive(Debug, Clone, PartialEq)]
pub struct DatagenEvent {
    /// Deterministic idempotency key. The MemWAL read path de-duplicates by it.
    pub event_id: String,
    pub item_id: String,
    pub root_item_id: String,
    pub parent_item_id: Option<String>,
    /// Strictly increasing per item. A collision between different event ids is split-brain corruption.
    pub item_seq: i64,
    /// Shared by every event emitted for one checkpoint boundary.
    pub checkpoint_id: String,
    pub event_type: DatagenEventType,
    pub step_name: Option<String>,
    pub step_kind: Option<DatagenStepKind>,
    pub step_index: Option<i64>,
    pub enclosing_step: Option<String>,
    pub selector_step: Option<String>,
    pub attempt: i32,
    pub run_id: String,
    /// Fencing identity for the writer/lease that owned this item.
    pub writer_epoch: String,
    pub field_name: Option<String>,
    /// Stable codec id, not a Python class name.
    pub field_type: Option<String>,
    pub codec_version: Option<i32>,
    pub value: Option<DatagenValue>,
    /// Query tags captured on ITEM_CREATED. Not part of correctness.
    pub query_tags: Option<Value>,
    /// The stored lifecycle/failure status (populated on ITEM_CREATED / TERMINAL / FAILED).
    pub status: Option<DatagenItemStatus>,
    pub error_type: Option<String>,
    pub error_dump: Option<String>,
    pub traceback: Option<String>,
    pub event_ts: DateTime<Utc>,
    pub schema_version: i32,
}

impl DatagenEvent {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("event_id", self.event_id.as_str()),
            ("item_id", self.item_id.as_str()),
            ("root_item_id", self.root_item_id.as_str()),
            ("checkpoint_id", self.checkpoint_id.as_str()),
            ("run_id", self.run_id.as_str()),
            ("writer_epoch", self.writer_epoch.as_str()),
        ] {
            if value.is_empty() {
                return Err(format!("{name} must not be empty"));
            }
        }
        if self.item_seq < 0 {
            return Err("item_seq must be non-negative".to_string());
        }
        if self.attempt < 0 {
            return Err("attempt must be non-negative".to_string());
        }
        if self.schema_version <= 0 {
            return Err("schema_version must be positive".to_string());
        }

        match self.event_type {
            DatagenEventType::FieldSet | DatagenEventType::FieldAppend => {
                if self.field_name.as_deref().is_none_or(str::is_empty) {
                    return Err("field events require field_name".to_string());
                }
                if self.field_type.as_deref().is_none_or(str::is_empty) {
                    return Err("field events require field_type".to_string());
                }
                if self.codec_version.is_none() {
                    return Err("field events require codec_version".to_string());
                }
                if self.value.is_none() {
                    return Err("field events require a value".to_string());
                }
                self.require_step_provenance("field events")?;
            }
            DatagenEventType::StepStarted | DatagenEventType::StepCompleted => {
                self.require_step_provenance(self.event_type.as_str())?;
            }
            DatagenEventType::Failed => {
                if self.error_type.as_deref().is_none_or(str::is_empty) {
                    return Err("FAILED requires error_type".to_string());
                }
                self.require_step_provenance("FAILED")?;
            }
            DatagenEventType::Terminal => match self.status {
                Some(DatagenItemStatus::Completed | DatagenItemStatus::Filtered) => {}
                _ => return Err("TERMINAL requires status completed or filtered".to_string()),
            },
            DatagenEventType::ItemCreated => {}
        }
        Ok(())
    }

    fn require_step_provenance(&self, context: &str) -> Result<(), String> {
        if self.step_name.as_deref().is_none_or(str::is_empty) {
            return Err(format!("{context} require step_name"));
        }
        if self.step_kind.is_none() {
            return Err(format!("{context} require step_kind"));
        }
        if self.step_index.is_none() {
            return Err(format!("{context} require step_index"));
        }
        Ok(())
    }
}

/// Generate a deterministic event id for retry-safe checkpoint ingestion.
#[must_use]
pub fn datagen_event_id(item_id: &str, checkpoint_id: &str, ordinal: u32) -> String {
    let input = format!("{item_id}\0{checkpoint_id}\0{ordinal}");
    Uuid::new_v5(&Uuid::NAMESPACE_OID, input.as_bytes()).to_string()
}

/// Fold an item's events into its latest state. Returns `None` if there is no ITEM_CREATED (the item
/// was never started).
pub fn fold_datagen_events(events: &[DatagenEvent]) -> Result<Option<FoldedDatagenItem>, String> {
    let ordered = normalize_events(events)?;
    let Some(first) = ordered.first() else {
        return Ok(None);
    };
    if first.event_type != DatagenEventType::ItemCreated {
        return Ok(None);
    }
    let mut item = initial_item(first)?;
    for event in ordered {
        apply_event(&mut item, event)?;
    }
    Ok(Some(item))
}

/// The ordered completed-step cursors of an item, in `item_seq` order.
pub fn datagen_trajectory(events: &[DatagenEvent]) -> Result<Vec<DatagenStepCursor>, String> {
    Ok(fold_datagen_events(events)?
        .map(|item| item.trajectory.ordered)
        .unwrap_or_default())
}

/// The lightweight failure pointers of an item, in `item_seq` order.
pub fn datagen_failures(events: &[DatagenEvent]) -> Result<Vec<DatagenFailure>, String> {
    let ordered = normalize_events(events)?;
    let mut failures = Vec::new();
    for event in ordered {
        if event.event_type == DatagenEventType::Failed {
            failures.push(DatagenFailure {
                at: DatagenStepCursor {
                    position: stream_position(event)?,
                    item_seq: event.item_seq,
                },
                run_id: event.run_id.clone(),
                attempt: event.attempt,
                error: DatagenErrorInfo {
                    error_type: event.error_type.clone().unwrap(),
                    error_dump: event.error_dump.clone(),
                    traceback: event.traceback.clone(),
                },
            });
        }
    }
    Ok(failures)
}

fn normalize_events(events: &[DatagenEvent]) -> Result<Vec<&DatagenEvent>, String> {
    let mut by_id: HashMap<&str, &DatagenEvent> = HashMap::new();
    for event in events {
        event.validate()?;
        match by_id.insert(&event.event_id, event) {
            Some(previous) if previous != event => {
                return Err(format!(
                    "event_id '{}' was reused with different content",
                    event.event_id
                ));
            }
            _ => {}
        }
    }

    let mut ordered: Vec<_> = by_id.into_values().collect();
    ordered.sort_by(|left, right| {
        left.item_seq
            .cmp(&right.item_seq)
            .then_with(|| left.event_id.cmp(&right.event_id))
    });
    for pair in ordered.windows(2) {
        if pair[0].item_id != pair[1].item_id {
            return Err("all events in a fold must belong to one item".to_string());
        }
        if pair[0].item_seq == pair[1].item_seq {
            return Err(format!(
                "item '{}' has conflicting events at item_seq {}",
                pair[0].item_id, pair[0].item_seq
            ));
        }
    }
    Ok(ordered)
}

fn initial_item(first: &DatagenEvent) -> Result<FoldedDatagenItem, String> {
    let parent_item_id = match &first.parent_item_id {
        Some(parent) => Some(DatagenItemId::parse(parent)?),
        None => None,
    };
    Ok(FoldedDatagenItem {
        item_id: DatagenItemId::parse(&first.item_id)?,
        root_item_id: DatagenItemId::parse(&first.root_item_id)?,
        parent_item_id,
        status: DatagenItemStatus::Running,
        last_item_seq: first.item_seq,
        last_attempt: first.attempt,
        fields: BTreeMap::new(),
        trajectory: DatagenTrajectory::default(),
        query_tags: None,
        blob_event_ids: BTreeMap::new(),
    })
}

fn apply_event(item: &mut FoldedDatagenItem, event: &DatagenEvent) -> Result<(), String> {
    if DatagenItemId::parse(&event.item_id)? != item.item_id {
        return Err(format!(
            "event '{}' belongs to item '{}', expected '{}'",
            event.event_id, event.item_id, item.item_id
        ));
    }

    item.last_item_seq = item.last_item_seq.max(event.item_seq);
    item.last_attempt = item.last_attempt.max(event.attempt);

    match event.event_type {
        DatagenEventType::ItemCreated => {
            item.status = DatagenItemStatus::Running;
            if event.query_tags.is_some() {
                item.query_tags = event.query_tags.clone();
            }
        }
        DatagenEventType::StepStarted => {
            item.trajectory.started.insert(stream_position(event)?);
        }
        DatagenEventType::FieldSet => {
            let field_name = event.field_name.clone().unwrap();
            let value = event.value.clone().unwrap();
            record_blob_event_id(item, &field_name, &value, &event.event_id);
            item.fields
                .insert(field_name, DatagenFieldState::Set(value));
        }
        DatagenEventType::FieldAppend => {
            let field_name = event.field_name.clone().unwrap();
            let value = event.value.clone().unwrap();
            record_blob_event_id(item, &field_name, &value, &event.event_id);
            match item.fields.entry(field_name) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(DatagenFieldState::Appended(vec![value]));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => match entry.get_mut() {
                    DatagenFieldState::Appended(values) => values.push(value),
                    DatagenFieldState::Set(_) => {
                        return Err(format!(
                            "field '{}' mixes FIELD_SET and FIELD_APPEND",
                            entry.key()
                        ));
                    }
                },
            }
        }
        DatagenEventType::StepCompleted => {
            let position = stream_position(event)?;
            item.trajectory.completed.insert(position.clone());
            item.trajectory.ordered.push(DatagenStepCursor {
                position,
                item_seq: event.item_seq,
            });
        }
        DatagenEventType::Failed => {
            // Failure lens only: a FAILED row leaves the item Running under the lifecycle lens.
        }
        DatagenEventType::Terminal => {
            item.status = match event.status {
                Some(status @ (DatagenItemStatus::Completed | DatagenItemStatus::Filtered)) => {
                    status
                }
                _ => return Err("TERMINAL event missing completed/filtered status".to_string()),
            };
        }
    }
    Ok(())
}

fn record_blob_event_id(
    item: &mut FoldedDatagenItem,
    field_name: &str,
    value: &DatagenValue,
    event_id: &str,
) {
    if matches!(value, DatagenValue::Blob(_)) {
        item.blob_event_ids
            .insert(field_name.to_string(), event_id.to_string());
    }
}

fn stream_position(event: &DatagenEvent) -> Result<DatagenStreamPosition, String> {
    Ok(DatagenStreamPosition {
        step: DatagenStepId {
            name: event
                .step_name
                .clone()
                .ok_or_else(|| "step event missing step_name".to_string())?,
            kind: event
                .step_kind
                .ok_or_else(|| "step event missing step_kind".to_string())?,
        },
        index: event
            .step_index
            .ok_or_else(|| "step event missing step_index".to_string())?,
        enclosing: event.enclosing_step.clone(),
        selector: event.selector_step.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    fn event(seq: i64, event_type: DatagenEventType) -> DatagenEvent {
        let checkpoint_id = format!("checkpoint-{seq}");
        DatagenEvent {
            event_id: datagen_event_id("5", &checkpoint_id, 0),
            item_id: "5".to_string(),
            root_item_id: "5".to_string(),
            parent_item_id: None,
            item_seq: seq,
            checkpoint_id,
            event_type,
            step_name: None,
            step_kind: None,
            step_index: None,
            enclosing_step: None,
            selector_step: None,
            attempt: 0,
            run_id: "run-1".to_string(),
            writer_epoch: "writer-1".to_string(),
            field_name: None,
            field_type: None,
            codec_version: None,
            value: None,
            query_tags: None,
            status: None,
            error_type: None,
            error_dump: None,
            traceback: None,
            event_ts: Utc.timestamp_micros(1_700_000_000_000_000 + seq).unwrap(),
            schema_version: DATAGEN_SCHEMA_VERSION,
        }
    }

    fn created(seq: i64) -> DatagenEvent {
        let mut created = event(seq, DatagenEventType::ItemCreated);
        created.status = Some(DatagenItemStatus::Running);
        created
    }

    fn leaf_completed(seq: i64, name: &str, index: i64, enclosing: Option<&str>) -> DatagenEvent {
        let mut completed = event(seq, DatagenEventType::StepCompleted);
        completed.step_name = Some(name.to_string());
        completed.step_kind = Some(DatagenStepKind::Leaf);
        completed.step_index = Some(index);
        completed.enclosing_step = enclosing.map(str::to_string);
        completed
    }

    fn driver_started(seq: i64, name: &str, index: i64, enclosing: Option<&str>) -> DatagenEvent {
        let mut started = event(seq, DatagenEventType::StepStarted);
        started.step_name = Some(name.to_string());
        started.step_kind = Some(DatagenStepKind::Sequence);
        started.step_index = Some(index);
        started.enclosing_step = enclosing.map(str::to_string);
        started
    }

    #[test]
    fn item_id_round_trips_and_navigates() {
        let root = DatagenItemId::from_source_key("5");
        let child = root.child("expand", 0).child("enrich", 1);
        assert_eq!(child.to_string(), "5/expand:0/enrich:1");
        assert_eq!(DatagenItemId::parse("5/expand:0/enrich:1").unwrap(), child);
        assert_eq!(child.origin_step(), Some("enrich"));
        assert_eq!(child.branch_idx(), Some(1));
        assert_eq!(child.parent().unwrap().to_string(), "5/expand:0");
        assert_eq!(child.root(), root);
        assert!(root.is_root());
        assert_eq!(root.origin_step(), None);
    }

    #[test]
    fn never_started_folds_to_none() {
        let completed = leaf_completed(1, "gen", 0, Some("main"));
        assert!(fold_datagen_events(&[completed]).unwrap().is_none());
    }

    #[test]
    fn fold_tracks_started_and_completed_positions() {
        let events = [
            created(0),
            driver_started(1, "main", 0, None),
            leaf_completed(2, "gen", 0, Some("main")),
        ];
        let folded = fold_datagen_events(&events).unwrap().unwrap();
        assert_eq!(folded.status, DatagenItemStatus::Running);
        assert_eq!(folded.trajectory.ordered.len(), 1);
        assert_eq!(folded.trajectory.completed.len(), 1);
        assert_eq!(folded.trajectory.started.len(), 1);
        assert_eq!(folded.last_item_seq, 2);
    }

    #[test]
    fn field_set_is_last_writer_wins_and_append_accumulates() {
        let mut set_v1 = leaf_completed(1, "gen", 0, Some("main"));
        set_v1.event_type = DatagenEventType::FieldSet;
        set_v1.field_name = Some("draft".to_string());
        set_v1.field_type = Some("str".to_string());
        set_v1.codec_version = Some(1);
        set_v1.value = Some(DatagenValue::Str("v1".to_string()));

        let mut set_v2 = set_v1.clone();
        set_v2.item_seq = 2;
        set_v2.checkpoint_id = "c2".to_string();
        set_v2.event_id = datagen_event_id("5", "c2", 0);
        set_v2.value = Some(DatagenValue::Str("v2".to_string()));

        let mut append = leaf_completed(3, "b1", 0, Some("body"));
        append.event_type = DatagenEventType::FieldAppend;
        append.field_name = Some("revisions".to_string());
        append.field_type = Some("json".to_string());
        append.codec_version = Some(1);
        append.value = Some(DatagenValue::Json(json!({"n": "a"})));
        let mut append2 = append.clone();
        append2.item_seq = 4;
        append2.checkpoint_id = "c4".to_string();
        append2.event_id = datagen_event_id("5", "c4", 0);
        append2.value = Some(DatagenValue::Json(json!({"n": "b"})));

        let folded = fold_datagen_events(&[created(0), set_v1, set_v2, append, append2])
            .unwrap()
            .unwrap();
        assert_eq!(
            folded.fields.get("draft"),
            Some(&DatagenFieldState::Set(DatagenValue::Str("v2".to_string())))
        );
        assert_eq!(
            folded.fields.get("revisions"),
            Some(&DatagenFieldState::Appended(vec![
                DatagenValue::Json(json!({"n": "a"})),
                DatagenValue::Json(json!({"n": "b"})),
            ]))
        );
    }

    #[test]
    fn terminal_sets_lifecycle_status() {
        let mut terminal = event(2, DatagenEventType::Terminal);
        terminal.status = Some(DatagenItemStatus::Completed);
        let folded = fold_datagen_events(&[created(0), terminal])
            .unwrap()
            .unwrap();
        assert_eq!(folded.status, DatagenItemStatus::Completed);
    }

    #[test]
    fn failed_leaves_item_running_but_surfaces_in_failures() {
        let mut failed = leaf_completed(1, "check", 2, Some("solve"));
        failed.event_type = DatagenEventType::Failed;
        failed.status = Some(DatagenItemStatus::Failed);
        failed.error_type = Some("ValueError".to_string());
        failed.attempt = 0;

        let folded = fold_datagen_events(&[created(0), failed.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(folded.status, DatagenItemStatus::Running);

        let failures = datagen_failures(&[created(0), failed]).unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].error.error_type, "ValueError");
        assert_eq!(failures[0].at.position.step.name, "check");
    }

    #[test]
    fn sequence_collision_is_rejected() {
        let first = leaf_completed(1, "gen", 0, Some("main"));
        let mut second = first.clone();
        second.event_id = "different-event".to_string();
        second.checkpoint_id = "different-checkpoint".to_string();
        let error = fold_datagen_events(&[created(0), first, second]).unwrap_err();
        assert!(error.contains("conflicting events at item_seq 1"));
    }

    #[test]
    fn retry_duplicate_event_is_folded_once() {
        let mut append = leaf_completed(1, "gen", 1, Some("main"));
        append.event_type = DatagenEventType::FieldAppend;
        append.field_name = Some("messages".to_string());
        append.field_type = Some("json".to_string());
        append.codec_version = Some(1);
        append.value = Some(DatagenValue::Json(json!({"role": "assistant"})));

        let folded = fold_datagen_events(&[created(0), append.clone(), append])
            .unwrap()
            .unwrap();
        assert_eq!(
            folded.fields.get("messages"),
            Some(&DatagenFieldState::Appended(vec![DatagenValue::Json(
                json!({"role": "assistant"})
            )]))
        );
    }

    #[test]
    fn step_kind_and_status_parse_round_trip() {
        for kind in [
            DatagenStepKind::Root,
            DatagenStepKind::Leaf,
            DatagenStepKind::Sequence,
            DatagenStepKind::Loop,
            DatagenStepKind::MapReduce,
            DatagenStepKind::Branch,
            DatagenStepKind::SubPipeline,
            DatagenStepKind::Conditional,
            DatagenStepKind::Router,
        ] {
            assert_eq!(DatagenStepKind::parse(kind.as_str()).unwrap(), kind);
        }
        assert!(DatagenStepKind::Sequence.is_driver());
        assert!(DatagenStepKind::Loop.is_driver());
        assert!(!DatagenStepKind::MapReduce.is_driver());
        assert!(!DatagenStepKind::Leaf.is_driver());
        assert!(DatagenStepKind::parse("nope").is_err());

        for status in [
            DatagenItemStatus::Running,
            DatagenItemStatus::Completed,
            DatagenItemStatus::Filtered,
            DatagenItemStatus::Failed,
        ] {
            assert_eq!(DatagenItemStatus::parse(status.as_str()).unwrap(), status);
        }
        assert!(DatagenItemStatus::parse("nope").is_err());
    }

    #[test]
    fn terminal_filtered_sets_filtered_status() {
        let mut terminal = event(2, DatagenEventType::Terminal);
        terminal.status = Some(DatagenItemStatus::Filtered);
        let folded = fold_datagen_events(&[created(0), terminal])
            .unwrap()
            .unwrap();
        assert_eq!(folded.status, DatagenItemStatus::Filtered);
    }

    #[test]
    fn selector_step_is_folded_onto_the_chosen_child_position() {
        // A Conditional/Router writes no row; the chosen leaf records who selected it.
        let mut chosen = leaf_completed(1, "stage2_qa", 0, Some("rubric_generation"));
        chosen.selector_step = Some("if_stage2_qa".to_string());
        let folded = fold_datagen_events(&[created(0), chosen]).unwrap().unwrap();
        let cursor = &folded.trajectory.ordered[0];
        assert_eq!(cursor.position.selector.as_deref(), Some("if_stage2_qa"));
        assert_eq!(cursor.position.step.name, "stage2_qa");
    }

    #[test]
    fn fan_out_sub_item_folds_with_lineage() {
        // A MapReduce/Branch sub-item is its own stream carrying denormalized parent/root ids.
        let root = DatagenItemId::from_source_key("5");
        let child = root.child("solve_twice", 1);
        let mut created = event(0, DatagenEventType::ItemCreated);
        created.item_id = child.to_string();
        created.parent_item_id = Some(root.to_string());
        created.status = Some(DatagenItemStatus::Running);
        created.event_id = datagen_event_id(&child.to_string(), "created", 0);

        let mut solved = leaf_completed(1, "solve", 0, Some("solve_attempt"));
        solved.item_id = child.to_string();
        solved.root_item_id = root.to_string();
        solved.parent_item_id = Some(root.to_string());
        solved.event_id = datagen_event_id(&child.to_string(), "solve-0", 0);

        let folded = fold_datagen_events(&[created, solved]).unwrap().unwrap();
        assert_eq!(folded.item_id, child);
        assert_eq!(folded.root_item_id, root);
        assert_eq!(folded.parent_item_id, Some(root));
        assert_eq!(folded.item_id.origin_step(), Some("solve_twice"));
        assert_eq!(folded.item_id.branch_idx(), Some(1));
    }

    #[test]
    fn field_rejects_mixing_set_and_append() {
        let mut set = leaf_completed(1, "gen", 0, Some("main"));
        set.event_type = DatagenEventType::FieldSet;
        set.field_name = Some("draft".to_string());
        set.field_type = Some("str".to_string());
        set.codec_version = Some(1);
        set.value = Some(DatagenValue::Str("v1".to_string()));

        let mut append = set.clone();
        append.event_type = DatagenEventType::FieldAppend;
        append.item_seq = 2;
        append.checkpoint_id = "c2".to_string();
        append.event_id = datagen_event_id("5", "c2", 0);

        let error = fold_datagen_events(&[created(0), set, append]).unwrap_err();
        assert!(error.contains("mixes FIELD_SET and FIELD_APPEND"));
    }

    #[test]
    fn resume_open_frame_is_started_minus_completed() {
        // A driver frame that opened but never completed = the frame live at crash time.
        let mut main_completed = driver_started(3, "main", 0, None);
        main_completed.event_type = DatagenEventType::StepCompleted;
        main_completed.checkpoint_id = "main-done".to_string();
        main_completed.event_id = datagen_event_id("5", "main-done", 0);
        let events = [
            created(0),
            driver_started(1, "main", 0, None),
            driver_started(2, "refine", 1, Some("main")),
            main_completed,
        ];
        let folded = fold_datagen_events(&events).unwrap().unwrap();
        let open: Vec<_> = folded
            .trajectory
            .started
            .difference(&folded.trajectory.completed)
            .collect();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].step.name, "refine");
    }
}
