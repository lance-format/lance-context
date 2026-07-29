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

/// Per-run write identity, stamped onto every event a writer emits. `run_id` groups a batch job;
/// `writer_epoch` fences a revived zombie writer (a fresh process gets a fresh epoch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatagenWriteContext {
    pub run_id: String,
    pub writer_epoch: String,
}

/// The inputs for a fresh stream (Case 3, the `open_stream` path). `query_tags` is captured onto
/// ITEM_CREATED and is not part of correctness.
#[derive(Debug, Clone, PartialEq)]
pub struct DatagenNewStream {
    pub item_id: DatagenItemId,
    pub parent_item_id: Option<DatagenItemId>,
    pub query_tags: Option<Value>,
}

/// Whether a field write replaces (FIELD_SET) or accumulates (FIELD_APPEND).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOp {
    Set,
    Append,
}

/// One field write within a checkpoint boundary — the typed input a caller hands the writer instead
/// of hand-building a FIELD_* event.
#[derive(Debug, Clone, PartialEq)]
pub struct DatagenFieldChange {
    pub name: String,
    pub field_type: String,
    pub codec_version: i32,
    pub op: FieldOp,
    pub value: DatagenValue,
}

/// A per-stream write handle. Owns the bookkeeping columns the client should never touch
/// (`item_seq`, `attempt`, `checkpoint_id`, `event_id`), stamping them onto every event it emits.
///
/// Each method returns the event batch to persist rather than performing I/O itself: the store layer
/// (embedded or remote) appends it. Two ways to obtain one:
///   - fresh:  [`open_stream_events`] (Case 3) — emits ITEM_CREATED, `next_seq = 1`, `attempt = 0`.
///   - resume: [`FoldedDatagenItem::resuming_writer`] (Case 2) — pure, emits nothing, `next_seq =
///     last_item_seq + 1`, `attempt = last_attempt + 1`.
#[derive(Debug, Clone)]
pub struct DatagenStreamWriter {
    item_id: DatagenItemId,
    root_item_id: DatagenItemId,
    parent_item_id: Option<DatagenItemId>,
    context: DatagenWriteContext,
    next_seq: i64,
    attempt: i32,
    checkpoint_ordinal: u32,
}

/// A fresh stream's ITEM_CREATED event plus the writer positioned to continue after it.
#[derive(Debug, Clone)]
pub struct DatagenOpenStream {
    pub created_event: DatagenEvent,
    pub writer: DatagenStreamWriter,
}

impl DatagenStreamWriter {
    /// The item this writer streams to.
    #[must_use]
    pub fn item_id(&self) -> &DatagenItemId {
        &self.item_id
    }

    /// The attempt number this writer stamps onto its events (0 fresh, `last_attempt + 1` on resume).
    #[must_use]
    pub fn attempt(&self) -> i32 {
        self.attempt
    }

    fn resume(
        item_id: DatagenItemId,
        root_item_id: DatagenItemId,
        parent_item_id: Option<DatagenItemId>,
        context: DatagenWriteContext,
        next_seq: i64,
        attempt: i32,
    ) -> Self {
        Self {
            item_id,
            root_item_id,
            parent_item_id,
            context,
            next_seq,
            attempt,
            checkpoint_ordinal: 0,
        }
    }

    fn compose_checkpoint_id(&mut self, position: &DatagenStreamPosition) -> String {
        let ordinal = self.checkpoint_ordinal;
        self.checkpoint_ordinal += 1;
        // `attempt` is embedded so a resume re-emitting the same step position produces a distinct
        // `checkpoint_id` (and thus a distinct `event_id`); otherwise attempt 0 and attempt 1 would
        // collide on `datagen_event_id` and fold would reject them as reused-with-different-content.
        format!(
            "{}\0{}\0{}\0{}\0{}",
            self.item_id, self.attempt, position.step.name, position.index, ordinal
        )
    }

    fn take_seq(&mut self) -> i64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        seq
    }

    fn base_event(
        &self,
        event_id: String,
        item_seq: i64,
        checkpoint_id: String,
        event_type: DatagenEventType,
    ) -> DatagenEvent {
        DatagenEvent {
            event_id,
            item_id: self.item_id.to_string(),
            root_item_id: self.root_item_id.to_string(),
            parent_item_id: self.parent_item_id.as_ref().map(DatagenItemId::to_string),
            item_seq,
            checkpoint_id,
            event_type,
            step_name: None,
            step_kind: None,
            step_index: None,
            enclosing_step: None,
            selector_step: None,
            attempt: self.attempt,
            run_id: self.context.run_id.clone(),
            writer_epoch: self.context.writer_epoch.clone(),
            field_name: None,
            field_type: None,
            codec_version: None,
            value: None,
            query_tags: None,
            status: None,
            error_type: None,
            error_dump: None,
            traceback: None,
            event_ts: Utc::now(),
            schema_version: DATAGEN_SCHEMA_VERSION,
        }
    }

    fn stamp_position(event: &mut DatagenEvent, position: &DatagenStreamPosition) {
        event.step_name = Some(position.step.name.clone());
        event.step_kind = Some(position.step.kind);
        event.step_index = Some(position.index);
        event.enclosing_step = position.enclosing.clone();
        event.selector_step = position.selector.clone();
    }

    /// Emit STEP_STARTED for a driver frame (`Sequence`/`Loop`). Structural marker, written once.
    pub fn step_started(&mut self, position: &DatagenStreamPosition) -> DatagenEvent {
        let seq = self.take_seq();
        let checkpoint_id = self.compose_checkpoint_id(position);
        let event_id = datagen_event_id(&self.item_id.to_string(), &checkpoint_id, 0);
        let mut event =
            self.base_event(event_id, seq, checkpoint_id, DatagenEventType::StepStarted);
        Self::stamp_position(&mut event, position);
        event
    }

    /// Emit a checkpoint boundary: the step's field writes plus its STEP_COMPLETED, all sharing one
    /// `checkpoint_id`. This is the atomic unit `append_checkpoint` persists.
    pub fn step_completed(
        &mut self,
        position: &DatagenStreamPosition,
        fields: &[DatagenFieldChange],
    ) -> Vec<DatagenEvent> {
        let checkpoint_id = self.compose_checkpoint_id(position);
        let item_id = self.item_id.to_string();
        let mut events = Vec::with_capacity(fields.len() + 1);
        for (ordinal, change) in fields.iter().enumerate() {
            let seq = self.take_seq();
            let event_id = datagen_event_id(&item_id, &checkpoint_id, ordinal as u32 + 1);
            let event_type = match change.op {
                FieldOp::Set => DatagenEventType::FieldSet,
                FieldOp::Append => DatagenEventType::FieldAppend,
            };
            let mut event = self.base_event(event_id, seq, checkpoint_id.clone(), event_type);
            Self::stamp_position(&mut event, position);
            event.field_name = Some(change.name.clone());
            event.field_type = Some(change.field_type.clone());
            event.codec_version = Some(change.codec_version);
            event.value = Some(change.value.clone());
            events.push(event);
        }
        let seq = self.take_seq();
        let event_id = datagen_event_id(&item_id, &checkpoint_id, 0);
        let mut completed = self.base_event(
            event_id,
            seq,
            checkpoint_id,
            DatagenEventType::StepCompleted,
        );
        Self::stamp_position(&mut completed, position);
        events.push(completed);
        events
    }

    /// Emit TERMINAL — the item reached a lifecycle end (completed/filtered).
    pub fn item_terminal(&mut self, terminal: DatagenTerminal) -> DatagenEvent {
        let seq = self.take_seq();
        let checkpoint_id = format!("{}\0terminal\0{}", self.item_id, seq);
        let event_id = datagen_event_id(&self.item_id.to_string(), &checkpoint_id, 0);
        let mut event = self.base_event(event_id, seq, checkpoint_id, DatagenEventType::Terminal);
        event.status = Some(match terminal {
            DatagenTerminal::Completed => DatagenItemStatus::Completed,
            DatagenTerminal::Filtered => DatagenItemStatus::Filtered,
        });
        event
    }

    /// Emit FAILED at a step position. Does not terminate the item (failure lens only).
    pub fn item_failed(
        &mut self,
        position: &DatagenStreamPosition,
        error: &DatagenErrorInfo,
    ) -> DatagenEvent {
        let seq = self.take_seq();
        let checkpoint_id = self.compose_checkpoint_id(position);
        let event_id = datagen_event_id(&self.item_id.to_string(), &checkpoint_id, 0);
        let mut event = self.base_event(event_id, seq, checkpoint_id, DatagenEventType::Failed);
        Self::stamp_position(&mut event, position);
        event.status = Some(DatagenItemStatus::Failed);
        event.error_type = Some(error.error_type.clone());
        event.error_dump = error.error_dump.clone();
        event.traceback = error.traceback.clone();
        event
    }
}

/// Build the ITEM_CREATED event + a writer for a fresh stream. Pure; the store persists the event.
#[must_use]
pub fn open_stream_events(
    stream: &DatagenNewStream,
    context: &DatagenWriteContext,
) -> DatagenOpenStream {
    let mut writer = DatagenStreamWriter {
        item_id: stream.item_id.clone(),
        root_item_id: stream.item_id.root(),
        parent_item_id: stream.parent_item_id.clone(),
        context: context.clone(),
        next_seq: 1,
        attempt: 0,
        checkpoint_ordinal: 0,
    };
    let seq = writer.take_seq();
    let checkpoint_id = format!("{}\0created", stream.item_id);
    let event_id = datagen_event_id(&stream.item_id.to_string(), &checkpoint_id, 0);
    let mut created =
        writer.base_event(event_id, seq, checkpoint_id, DatagenEventType::ItemCreated);
    created.status = Some(DatagenItemStatus::Running);
    created.query_tags = stream.query_tags.clone();
    DatagenOpenStream {
        created_event: created,
        writer,
    }
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

impl FoldedDatagenItem {
    /// Build a resume write handle for this already-folded, Running item. Pure — no I/O, emits no
    /// ITEM_CREATED. The store owns the continuation rules: `next_seq = last_item_seq + 1`,
    /// `attempt = last_attempt + 1`. The resume counterpart to [`open_stream_events`] (the fresh
    /// path). `checkpoint_ordinal` restarts at 0 for the new attempt; `checkpoint_id`s stay unique
    /// across attempts because [`DatagenStreamWriter::compose_checkpoint_id`] embeds `attempt`.
    #[must_use]
    pub fn resuming_writer(&self, context: &DatagenWriteContext) -> DatagenStreamWriter {
        DatagenStreamWriter::resume(
            self.item_id.clone(),
            self.root_item_id.clone(),
            self.parent_item_id.clone(),
            context.clone(),
            self.last_item_seq + 1,
            self.last_attempt + 1,
        )
    }
}

/// One node in a root's inspection tree: a folded item plus the `item_id`s of its direct children.
/// Children are ordered by `item_id` string for a stable, deterministic walk.
#[derive(Debug, Clone, PartialEq)]
pub struct DatagenItemNode {
    pub item: FoldedDatagenItem,
    pub children: Vec<DatagenItemId>,
}

/// A root item and every projected descendant, each folded to latest state and linked parent->child.
/// Built purely from one root's event log — no I/O. `roots` are the entry item_ids (normally one, the
/// source root; more only if a log mixes roots). Use [`node`](Self::node) to walk from any item.
#[derive(Debug, Clone, PartialEq)]
pub struct DatagenItemTree {
    nodes: BTreeMap<String, DatagenItemNode>,
    roots: Vec<DatagenItemId>,
}

impl DatagenItemTree {
    /// Fold every item in a root's event log and link them into a tree. Events for different items may
    /// be interleaved; they are grouped by `item_id` and folded independently. Items with no
    /// ITEM_CREATED (never started) are skipped. A child whose parent is absent becomes an extra root.
    pub fn build(events: &[DatagenEvent]) -> Result<Self, String> {
        let mut by_item: BTreeMap<String, Vec<&DatagenEvent>> = BTreeMap::new();
        for event in events {
            by_item
                .entry(event.item_id.clone())
                .or_default()
                .push(event);
        }

        let mut nodes: BTreeMap<String, DatagenItemNode> = BTreeMap::new();
        for item_events in by_item.values() {
            let owned: Vec<DatagenEvent> =
                item_events.iter().map(|event| (*event).clone()).collect();
            if let Some(item) = fold_datagen_events(&owned)? {
                nodes.insert(
                    item.item_id.to_string(),
                    DatagenItemNode {
                        item,
                        children: Vec::new(),
                    },
                );
            }
        }

        let mut roots: Vec<DatagenItemId> = Vec::new();
        let child_ids: Vec<(String, Option<String>)> = nodes
            .values()
            .map(|node| {
                (
                    node.item.item_id.to_string(),
                    node.item
                        .parent_item_id
                        .as_ref()
                        .map(DatagenItemId::to_string),
                )
            })
            .collect();
        for (child_id, parent_id) in child_ids {
            match parent_id {
                Some(parent) if nodes.contains_key(&parent) => {
                    let child = DatagenItemId::parse(&child_id)?;
                    nodes.get_mut(&parent).unwrap().children.push(child);
                }
                _ => roots.push(DatagenItemId::parse(&child_id)?),
            }
        }
        for node in nodes.values_mut() {
            node.children.sort();
        }
        roots.sort();

        Ok(Self { nodes, roots })
    }

    /// The entry items (normally the single source root).
    #[must_use]
    pub fn roots(&self) -> &[DatagenItemId] {
        &self.roots
    }

    /// The node for an item, or `None` if it is not in this tree.
    #[must_use]
    pub fn node(&self, item_id: &DatagenItemId) -> Option<&DatagenItemNode> {
        self.nodes.get(&item_id.to_string())
    }

    /// Total number of folded items in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every folded item, ordered by `item_id`.
    pub fn items(&self) -> impl Iterator<Item = &FoldedDatagenItem> {
        self.nodes.values().map(|node| &node.item)
    }
}

/// Whole-run aggregation over a datagen log: how many items are in each lifecycle state, how far
/// they got, and where they failed. Built purely from events — no I/O.
///
/// `items` counts *root* items only (fan-out sub-items roll up under their root), so
/// `running + completed + filtered` equals the number of roots the log has seen. `failures` counts
/// FAILED events, which are non-terminal: a failed-then-retried item still counts as `running`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatagenRunOverview {
    /// Root items seen in the log (`running + completed + filtered`).
    pub items: usize,
    pub running: usize,
    pub completed: usize,
    pub filtered: usize,
    /// Total FAILED events across every item, including sub-items and retried attempts.
    pub failures: usize,
    /// FAILED-event count per `error_type`, so the dominant failure mode is one lookup away.
    pub failures_by_error_type: BTreeMap<String, usize>,
    /// Failure roll-up grouped by the `run_id` that emitted the FAILED event. One store spans every
    /// attempt at an experiment, so this separates "this run's failures" from historical ones.
    pub failures_by_run: BTreeMap<String, DatagenFailureBucket>,
    /// STEP_COMPLETED count per step name, across every item — the run's step-level progress.
    pub completed_steps: BTreeMap<String, usize>,
}

/// One `run_id`'s slice of the failure roll-up: how many, of what kind, and a handful of root item
/// ids to open next. The sample is capped at [`FAILURE_SAMPLE_LIMIT`] and is deterministic (roots in
/// id order), so an overview stays small no matter how wide the run is.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatagenFailureBucket {
    pub failures: usize,
    pub failures_by_error_type: BTreeMap<String, usize>,
    /// Up to [`FAILURE_SAMPLE_LIMIT`] distinct root item ids that failed under this run.
    pub sample_root_item_ids: Vec<String>,
}

/// How many root item ids each [`DatagenFailureBucket`] samples.
pub const FAILURE_SAMPLE_LIMIT: usize = 5;

impl DatagenRunOverview {
    /// Aggregate a whole run's events. Events for many items may be interleaved; they are grouped by
    /// `item_id` and folded independently, exactly like [`DatagenItemTree::build`].
    pub fn build(events: &[DatagenEvent]) -> Result<Self, String> {
        let mut by_item: BTreeMap<String, Vec<DatagenEvent>> = BTreeMap::new();
        for event in events {
            by_item
                .entry(event.item_id.clone())
                .or_default()
                .push(event.clone());
        }

        let mut overview = Self::default();
        for item_events in by_item.values() {
            let Some(item) = fold_datagen_events(item_events)? else {
                continue;
            };
            for cursor in &item.trajectory.ordered {
                *overview
                    .completed_steps
                    .entry(cursor.position.step.name.clone())
                    .or_default() += 1;
            }
            for failure in datagen_failures(item_events)? {
                overview.failures += 1;
                *overview
                    .failures_by_error_type
                    .entry(failure.error.error_type.clone())
                    .or_default() += 1;
                let bucket = overview
                    .failures_by_run
                    .entry(failure.run_id.clone())
                    .or_default();
                bucket.failures += 1;
                *bucket
                    .failures_by_error_type
                    .entry(failure.error.error_type.clone())
                    .or_default() += 1;
                // Items are visited in id order, so the sample is the same on every rebuild.
                let root = item.root_item_id.to_string();
                if bucket.sample_root_item_ids.len() < FAILURE_SAMPLE_LIMIT
                    && !bucket.sample_root_item_ids.contains(&root)
                {
                    bucket.sample_root_item_ids.push(root);
                }
            }
            // Only roots are counted as items; a sub-item's outcome rolls up under its root.
            if item.parent_item_id.is_some() {
                continue;
            }
            overview.items += 1;
            match item.status {
                DatagenItemStatus::Running => overview.running += 1,
                DatagenItemStatus::Completed => overview.completed += 1,
                DatagenItemStatus::Filtered => overview.filtered += 1,
                // A folded status is never Failed (failures live in the failure lens), but count it
                // as still running rather than silently dropping the item from `items`.
                DatagenItemStatus::Failed => overview.running += 1,
            }
        }
        Ok(overview)
    }
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

/// Whether a fold keeps blob field bytes it happens to have, or drops them to a pointer.
///
/// The log's `value_blob` column is normally projected away on read, so a folded blob field is a
/// lazy pointer resolved through `blob_event_ids` + `get_blob`. `Eager` keeps whatever bytes the
/// events already carry, which lets a caller that scanned the blob column fold the payload in one
/// pass instead of one `get_blob` round trip per field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DatagenBlobProjection {
    /// Blob fields fold to a pointer: `bytes` is dropped even when present. The default.
    #[default]
    Lazy,
    /// Blob fields keep the bytes the events carry (`None` when the scan projected them away).
    Eager,
}

/// Fold an item's events into its latest state. Returns `None` if there is no ITEM_CREATED (the item
/// was never started). Blob fields fold lazily — see [`fold_datagen_events_with`] to keep bytes.
pub fn fold_datagen_events(events: &[DatagenEvent]) -> Result<Option<FoldedDatagenItem>, String> {
    fold_datagen_events_with(events, DatagenBlobProjection::Lazy)
}

/// Fold an item's events under an explicit blob projection.
///
/// `DatagenBlobProjection::Eager` is only useful when the events were read with the blob column
/// projected in; with the default read path it folds identically to `Lazy`.
pub fn fold_datagen_events_with(
    events: &[DatagenEvent],
    blobs: DatagenBlobProjection,
) -> Result<Option<FoldedDatagenItem>, String> {
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
    if blobs == DatagenBlobProjection::Lazy {
        drop_blob_bytes(&mut item);
    }
    Ok(Some(item))
}

/// Replace every folded blob field's inline bytes with a pointer (`bytes: None`), keeping its
/// `size` / `checksum`. `blob_event_ids` still resolves the payload through `get_blob`.
fn drop_blob_bytes(item: &mut FoldedDatagenItem) {
    fn strip(value: &mut DatagenValue) {
        if let DatagenValue::Blob(blob) = value {
            blob.bytes = None;
        }
    }
    for state in item.fields.values_mut() {
        match state {
            DatagenFieldState::Set(value) => strip(value),
            DatagenFieldState::Appended(values) => values.iter_mut().for_each(strip),
        }
    }
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
            // Group D: a field's value kind is fixed by its first write. A later SET that
            // drifts to another kind means the pipeline wrote the field inconsistently.
            if let Some(DatagenFieldState::Set(existing)) = item.fields.get(&field_name) {
                if existing.kind() != value.kind() {
                    return Err(format!(
                        "field '{}' changes value kind from {} to {}",
                        field_name,
                        existing.kind(),
                        value.kind()
                    ));
                }
            }
            record_blob_event_id(item, &field_name, &value, &event.event_id);
            item.fields
                .insert(field_name, DatagenFieldState::Set(value));
        }
        DatagenEventType::FieldAppend => {
            let field_name = event.field_name.clone().unwrap();
            let value = event.value.clone().unwrap();
            record_blob_event_id(item, &field_name, &value, &event.event_id);
            match item.fields.entry(field_name.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(DatagenFieldState::Appended(vec![value]));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => match entry.get_mut() {
                    DatagenFieldState::Appended(values) => {
                        // Group D: the same kind rule holds within one appended list.
                        if let Some(existing) = values.first() {
                            if existing.kind() != value.kind() {
                                return Err(format!(
                                    "field '{}' changes value kind from {} to {}",
                                    field_name,
                                    existing.kind(),
                                    value.kind()
                                ));
                            }
                        }
                        values.push(value);
                    }
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
    fn dto_projection_carries_started_and_completed_sets() {
        let events = [
            created(0),
            driver_started(1, "main", 0, None),
            leaf_completed(2, "gen", 0, Some("main")),
        ];
        let folded = fold_datagen_events(&events).unwrap().unwrap();
        let dto = crate::api_impl::folded_item_to_dto(&folded);

        // `trajectory` stays the ordered cursor list; the two sets ride alongside it so the
        // caller can gate STEP_STARTED / STEP_COMPLETED re-emission on resume.
        assert_eq!(dto.trajectory.len(), 1);
        assert_eq!(dto.started.len(), 1);
        assert_eq!(dto.started[0].step_name, "main");
        assert_eq!(dto.completed.len(), 1);
        assert_eq!(dto.completed[0].step_name, "gen");
        assert_eq!(dto.completed[0].enclosing_step.as_deref(), Some("main"));
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
    fn field_value_kind_drift_is_rejected() {
        let mut set_str = leaf_completed(1, "gen", 0, Some("main"));
        set_str.event_type = DatagenEventType::FieldSet;
        set_str.field_name = Some("draft".to_string());
        set_str.field_type = Some("str".to_string());
        set_str.codec_version = Some(1);
        set_str.value = Some(DatagenValue::Str("v1".to_string()));

        let mut set_int = set_str.clone();
        set_int.item_seq = 2;
        set_int.checkpoint_id = "c2".to_string();
        set_int.event_id = datagen_event_id("5", "c2", 0);
        set_int.field_type = Some("int".to_string());
        set_int.value = Some(DatagenValue::Int(7));

        let err = fold_datagen_events(&[created(0), set_str, set_int]).unwrap_err();
        assert!(err.contains("changes value kind"), "{err}");

        // Same rule inside an appended list.
        let mut append_str = leaf_completed(3, "b1", 0, Some("body"));
        append_str.event_type = DatagenEventType::FieldAppend;
        append_str.field_name = Some("revisions".to_string());
        append_str.field_type = Some("str".to_string());
        append_str.codec_version = Some(1);
        append_str.value = Some(DatagenValue::Str("a".to_string()));

        let mut append_int = append_str.clone();
        append_int.item_seq = 4;
        append_int.checkpoint_id = "c4".to_string();
        append_int.event_id = datagen_event_id("5", "c4", 0);
        append_int.field_type = Some("int".to_string());
        append_int.value = Some(DatagenValue::Int(1));

        let err = fold_datagen_events(&[created(0), append_str, append_int]).unwrap_err();
        assert!(err.contains("changes value kind"), "{err}");
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
    fn resume_second_attempt_overwrites_field_and_advances_last_attempt() {
        // attempt 0 runs, writes `draft=v1`, then fails. A resume (attempt 1) rewrites the same
        // field at a higher item_seq and reaches TERMINAL. Fold is a flat replay ordered by
        // item_seq, so the later attempt's value wins and last_attempt advances.
        let mut set_a0 = leaf_completed(1, "gen", 0, Some("main"));
        set_a0.event_type = DatagenEventType::FieldSet;
        set_a0.field_name = Some("draft".to_string());
        set_a0.field_type = Some("str".to_string());
        set_a0.codec_version = Some(1);
        set_a0.value = Some(DatagenValue::Str("v1".to_string()));

        let mut failed_a0 = leaf_completed(2, "check", 0, Some("main"));
        failed_a0.event_type = DatagenEventType::Failed;
        failed_a0.status = Some(DatagenItemStatus::Failed);
        failed_a0.error_type = Some("ValueError".to_string());

        // Resume: attempt 1, structural events (ITEM_CREATED/STEP_STARTED) are NOT re-emitted.
        let mut set_a1 = set_a0.clone();
        set_a1.item_seq = 3;
        set_a1.attempt = 1;
        set_a1.checkpoint_id = "c3".to_string();
        set_a1.event_id = datagen_event_id("5", "c3", 0);
        set_a1.value = Some(DatagenValue::Str("v2".to_string()));

        let mut terminal_a1 = event(4, DatagenEventType::Terminal);
        terminal_a1.attempt = 1;
        terminal_a1.status = Some(DatagenItemStatus::Completed);

        let events = [created(0), set_a0, failed_a0.clone(), set_a1, terminal_a1];
        let folded = fold_datagen_events(&events).unwrap().unwrap();
        assert_eq!(folded.status, DatagenItemStatus::Completed);
        assert_eq!(folded.last_attempt, 1);
        assert_eq!(folded.last_item_seq, 4);
        assert_eq!(
            folded.fields.get("draft"),
            Some(&DatagenFieldState::Set(DatagenValue::Str("v2".to_string())))
        );

        // The failure lens still surfaces the attempt-0 failure, tagged with its attempt.
        let failures = datagen_failures(&events).unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].attempt, 0);
    }

    fn write_context() -> DatagenWriteContext {
        DatagenWriteContext {
            run_id: "run-1".to_string(),
            writer_epoch: "writer-1".to_string(),
        }
    }

    fn leaf_position(name: &str, index: i64, enclosing: Option<&str>) -> DatagenStreamPosition {
        DatagenStreamPosition {
            step: DatagenStepId {
                name: name.to_string(),
                kind: DatagenStepKind::Leaf,
            },
            index,
            enclosing: enclosing.map(str::to_string),
            selector: None,
        }
    }

    fn set_field(name: &str, value: DatagenValue) -> DatagenFieldChange {
        DatagenFieldChange {
            name: name.to_string(),
            field_type: "str".to_string(),
            codec_version: 1,
            op: FieldOp::Set,
            value,
        }
    }

    #[test]
    fn open_stream_writer_produces_a_foldable_lifecycle() {
        let stream = DatagenNewStream {
            item_id: DatagenItemId::from_source_key("5"),
            parent_item_id: None,
            query_tags: Some(json!({"lang": "en"})),
        };
        let opened = open_stream_events(&stream, &write_context());
        let mut writer = opened.writer;
        assert_eq!(writer.attempt(), 0);

        let position = leaf_position("gen", 0, Some("main"));
        let checkpoint = writer.step_completed(
            &position,
            &[set_field("draft", DatagenValue::Str("v1".into()))],
        );
        let terminal = writer.item_terminal(DatagenTerminal::Completed);

        let mut events = vec![opened.created_event];
        events.extend(checkpoint);
        events.push(terminal);

        // Contiguous item_seq starting at 1, every event validates.
        for (offset, ev) in events.iter().enumerate() {
            ev.validate().unwrap();
            assert_eq!(ev.item_seq, offset as i64 + 1);
            assert_eq!(ev.attempt, 0);
        }

        let folded = fold_datagen_events(&events).unwrap().unwrap();
        assert_eq!(folded.status, DatagenItemStatus::Completed);
        assert_eq!(
            folded.fields.get("draft"),
            Some(&DatagenFieldState::Set(DatagenValue::Str("v1".into())))
        );
        assert_eq!(folded.query_tags, Some(json!({"lang": "en"})));
    }

    #[test]
    fn resuming_writer_continues_seq_bumps_attempt_and_avoids_event_id_collision() {
        let stream = DatagenNewStream {
            item_id: DatagenItemId::from_source_key("5"),
            parent_item_id: None,
            query_tags: None,
        };
        let context = write_context();
        let opened = open_stream_events(&stream, &context);
        let mut writer = opened.writer;
        let position = leaf_position("gen", 0, Some("main"));
        let attempt0 = writer.step_completed(
            &position,
            &[set_field("draft", DatagenValue::Str("v1".into()))],
        );

        let mut events = vec![opened.created_event];
        events.extend(attempt0.clone());

        // Fold attempt 0, then resume from that folded state.
        let folded0 = fold_datagen_events(&events).unwrap().unwrap();
        let mut resumed = folded0.resuming_writer(&context);
        assert_eq!(resumed.attempt(), 1);

        let attempt1 = resumed.step_completed(
            &position,
            &[set_field("draft", DatagenValue::Str("v2".into()))],
        );
        let terminal = resumed.item_terminal(DatagenTerminal::Completed);

        // Same step position across attempts must not collide on event_id (embeds attempt).
        for a0 in &attempt0 {
            for a1 in &attempt1 {
                assert_ne!(a0.event_id, a1.event_id);
                assert_ne!(a0.checkpoint_id, a1.checkpoint_id);
            }
        }

        events.extend(attempt1);
        events.push(terminal);
        assert!(events[events.len() - 2].item_seq > attempt0.last().unwrap().item_seq);

        let folded = fold_datagen_events(&events).unwrap().unwrap();
        assert_eq!(folded.status, DatagenItemStatus::Completed);
        assert_eq!(folded.last_attempt, 1);
        assert_eq!(
            folded.fields.get("draft"),
            Some(&DatagenFieldState::Set(DatagenValue::Str("v2".into())))
        );
    }

    #[test]
    fn item_tree_links_parent_and_child_items() {
        // Root "5" spawns child "5/expand:0". Build each via its own writer so item_ids/roots are set.
        let context = write_context();
        let root_stream = DatagenNewStream {
            item_id: DatagenItemId::from_source_key("5"),
            parent_item_id: None,
            query_tags: None,
        };
        let root_open = open_stream_events(&root_stream, &context);
        let mut root_writer = root_open.writer;
        let root_terminal = root_writer.item_terminal(DatagenTerminal::Completed);

        let child_id = DatagenItemId::from_source_key("5").child("expand", 0);
        let child_stream = DatagenNewStream {
            item_id: child_id.clone(),
            parent_item_id: Some(DatagenItemId::from_source_key("5")),
            query_tags: None,
        };
        let child_open = open_stream_events(&child_stream, &context);
        let mut child_writer = child_open.writer;
        let child_terminal = child_writer.item_terminal(DatagenTerminal::Completed);

        let events = vec![
            root_open.created_event,
            root_terminal,
            child_open.created_event,
            child_terminal,
        ];
        let tree = DatagenItemTree::build(&events).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.roots(), &[DatagenItemId::from_source_key("5")]);

        let root_node = tree.node(&DatagenItemId::from_source_key("5")).unwrap();
        assert_eq!(root_node.item.status, DatagenItemStatus::Completed);
        assert_eq!(root_node.children, vec![child_id.clone()]);

        let child_node = tree.node(&child_id).unwrap();
        assert_eq!(
            child_node.item.parent_item_id,
            Some(DatagenItemId::from_source_key("5"))
        );
        assert!(child_node.children.is_empty());
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

    /// A blob-valued FIELD_SET on item "5", at `seq`.
    fn blob_set(seq: i64, field: &str, bytes: &[u8]) -> DatagenEvent {
        let mut set = leaf_completed(seq, "gen", 0, Some("main"));
        set.event_type = DatagenEventType::FieldSet;
        set.field_name = Some(field.to_string());
        set.field_type = Some("blob".to_string());
        set.codec_version = Some(1);
        set.value = Some(DatagenValue::Blob(DatagenBlobValue {
            bytes: Some(bytes.to_vec()),
            size: bytes.len() as i64,
            checksum: None,
        }));
        set
    }

    #[test]
    fn blob_projection_selects_lazy_or_eager_bytes() {
        let events = vec![created(0), blob_set(1, "image", b"png-bytes")];

        let lazy = fold_datagen_events_with(&events, DatagenBlobProjection::Lazy)
            .unwrap()
            .unwrap();
        let DatagenFieldState::Set(DatagenValue::Blob(lazy_blob)) =
            lazy.fields.get("image").unwrap()
        else {
            panic!("expected a blob field");
        };
        assert_eq!(lazy_blob.bytes, None);
        assert_eq!(lazy_blob.size, 9);
        // The default fold is the lazy one.
        assert_eq!(fold_datagen_events(&events).unwrap().unwrap(), lazy);

        let eager = fold_datagen_events_with(&events, DatagenBlobProjection::Eager)
            .unwrap()
            .unwrap();
        let DatagenFieldState::Set(DatagenValue::Blob(eager_blob)) =
            eager.fields.get("image").unwrap()
        else {
            panic!("expected a blob field");
        };
        assert_eq!(eager_blob.bytes.as_deref(), Some(&b"png-bytes"[..]));
    }

    #[test]
    fn overview_counts_roots_and_rolls_failures_up_by_run() {
        // Two roots: "5" completes; "9" stays running with two failures, one of them under a
        // second run_id. "5/expand:0" is a sub-item and must not inflate the root counts.
        let mut events = vec![created(0), leaf_completed(1, "gen", 0, Some("main"))];
        let mut terminal = event(2, DatagenEventType::Terminal);
        terminal.status = Some(DatagenItemStatus::Completed);
        events.push(terminal);

        let mut sub_created = created(0);
        sub_created.item_id = "5/expand:0".to_string();
        sub_created.parent_item_id = Some("5".to_string());
        sub_created.event_id = datagen_event_id("5/expand:0", "checkpoint-0", 0);
        events.push(sub_created);

        let mut other_created = created(0);
        other_created.item_id = "9".to_string();
        other_created.root_item_id = "9".to_string();
        other_created.event_id = datagen_event_id("9", "checkpoint-0", 0);
        events.push(other_created);
        for (seq, run_id, error_type) in [(1, "run-1", "ValueError"), (2, "run-2", "KeyError")] {
            let mut failed = leaf_completed(seq, "score", 0, Some("main"));
            failed.item_id = "9".to_string();
            failed.root_item_id = "9".to_string();
            failed.event_id = datagen_event_id("9", &format!("checkpoint-{seq}"), 0);
            failed.event_type = DatagenEventType::Failed;
            failed.run_id = run_id.to_string();
            failed.error_type = Some(error_type.to_string());
            events.push(failed);
        }

        let overview = DatagenRunOverview::build(&events).unwrap();
        assert_eq!(overview.items, 2);
        assert_eq!(overview.completed, 1);
        assert_eq!(overview.running, 1);
        assert_eq!(overview.filtered, 0);
        assert_eq!(overview.failures, 2);
        assert_eq!(overview.failures_by_error_type["ValueError"], 1);
        assert_eq!(overview.completed_steps["gen"], 1);

        // Failures group by the run_id that emitted them, each carrying a root-id sample.
        assert_eq!(overview.failures_by_run.len(), 2);
        let run1 = &overview.failures_by_run["run-1"];
        assert_eq!(run1.failures, 1);
        assert_eq!(run1.failures_by_error_type["ValueError"], 1);
        assert_eq!(run1.sample_root_item_ids, vec!["9".to_string()]);
        assert_eq!(overview.failures_by_run["run-2"].failures, 1);
    }

    #[test]
    fn overview_failure_sample_is_capped() {
        let mut events = Vec::new();
        for idx in 0..(FAILURE_SAMPLE_LIMIT + 3) {
            let item_id = format!("item-{idx:02}");
            let mut item_created = created(0);
            item_created.item_id = item_id.clone();
            item_created.root_item_id = item_id.clone();
            item_created.event_id = datagen_event_id(&item_id, "checkpoint-0", 0);
            events.push(item_created);

            let mut failed = leaf_completed(1, "score", 0, Some("main"));
            failed.item_id = item_id.clone();
            failed.root_item_id = item_id.clone();
            failed.event_id = datagen_event_id(&item_id, "checkpoint-1", 0);
            failed.event_type = DatagenEventType::Failed;
            failed.error_type = Some("ValueError".to_string());
            events.push(failed);
        }

        let overview = DatagenRunOverview::build(&events).unwrap();
        assert_eq!(overview.failures, FAILURE_SAMPLE_LIMIT + 3);
        let bucket = &overview.failures_by_run["run-1"];
        assert_eq!(bucket.sample_root_item_ids.len(), FAILURE_SAMPLE_LIMIT);
        // Items are folded in id order, so the sample is deterministic.
        assert_eq!(bucket.sample_root_item_ids[0], "item-00");
    }
}
