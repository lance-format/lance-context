use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use serde_json::Value;
use uuid::Uuid;

/// Current schema version for the append-only datagen checkpoint log.
pub const DATAGEN_SCHEMA_VERSION: i32 = 1;

/// One lifecycle or field-level event in a datagen item's checkpoint history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatagenEventType {
    ItemCreated,
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
            "FIELD_SET" => Ok(Self::FieldSet),
            "FIELD_APPEND" => Ok(Self::FieldAppend),
            "STEP_COMPLETED" => Ok(Self::StepCompleted),
            "FAILED" => Ok(Self::Failed),
            "TERMINAL" => Ok(Self::Terminal),
            other => Err(format!("unsupported datagen event type '{other}'")),
        }
    }
}

/// Terminal outcome of an item.
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

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "completed" => Ok(Self::Completed),
            "filtered" => Ok(Self::Filtered),
            other => Err(format!("unsupported datagen terminal value '{other}'")),
        }
    }
}

/// Current status derived exclusively by folding the event log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatagenItemStatus {
    Pending,
    Running,
    Completed,
    Filtered,
    Failed,
}

/// Lazy reference to an inline blob event. `bytes` is absent on normal fold and
/// trajectory reads; callers materialize it through `DatagenStore::get_blob`.
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
    String(String),
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
            Self::String(_) => "str",
            Self::Json(_) => "json",
            Self::Blob(_) => "blob",
        }
    }
}

/// A single append-only row in `log.lance`.
#[derive(Debug, Clone, PartialEq)]
pub struct DatagenEvent {
    /// Deterministic idempotency key. The MemWAL read path de-duplicates by it.
    pub event_id: String,
    pub item_id: String,
    pub root_item_id: String,
    pub parent_item_id: Option<String>,
    /// Strictly increasing per item. A collision between different event ids is
    /// treated as split-brain corruption during fold.
    pub item_seq: i64,
    /// Shared by every event emitted for one checkpoint boundary.
    pub checkpoint_id: String,
    pub event_type: DatagenEventType,
    pub step_name: Option<String>,
    pub step_index: Option<i64>,
    pub step_instance_id: Option<String>,
    pub iteration: Option<i64>,
    pub attempt: i32,
    pub run_id: String,
    /// Fencing identity for the writer/lease that owned this item.
    pub writer_epoch: String,
    pub field_name: Option<String>,
    /// Stable codec id, not a Python class name.
    pub field_type: Option<String>,
    pub codec_version: Option<i32>,
    pub value: Option<DatagenValue>,
    /// Query tags captured on ITEM_CREATED. They are not part of correctness.
    pub query_tags: Option<Value>,
    pub terminal: Option<DatagenTerminal>,
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
                if self.step_name.as_deref().is_none_or(str::is_empty)
                    || self.step_index.is_none()
                    || self.step_instance_id.as_deref().is_none_or(str::is_empty)
                {
                    return Err(
                        "field events require step_name, step_index, and step_instance_id"
                            .to_string(),
                    );
                }
            }
            DatagenEventType::StepCompleted => {
                if self.step_name.as_deref().is_none_or(str::is_empty)
                    || self.step_index.is_none()
                    || self.step_instance_id.as_deref().is_none_or(str::is_empty)
                {
                    return Err(
                        "STEP_COMPLETED requires step_name, step_index, and step_instance_id"
                            .to_string(),
                    );
                }
            }
            DatagenEventType::Failed => {
                if self.error_type.as_deref().is_none_or(str::is_empty) {
                    return Err("FAILED requires error_type".to_string());
                }
            }
            DatagenEventType::Terminal => {
                if self.terminal.is_none() {
                    return Err("TERMINAL requires terminal".to_string());
                }
            }
            DatagenEventType::ItemCreated => {}
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DatagenStepCursor {
    pub checkpoint_id: String,
    pub step_name: String,
    pub step_index: i64,
    pub step_instance_id: String,
    pub iteration: Option<i64>,
    pub attempt: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DatagenFieldState {
    Set(DatagenValue),
    Appended(Vec<DatagenValue>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatagenFailure {
    pub event_id: String,
    pub run_id: String,
    pub checkpoint_id: String,
    pub item_seq: i64,
    pub step_name: Option<String>,
    pub error_type: String,
    pub error_dump: Option<String>,
    pub traceback: Option<String>,
    pub failed_at: DateTime<Utc>,
}

/// Current item state reconstructed solely from the append-only log.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldedDatagenItem {
    pub item_id: String,
    pub root_item_id: String,
    pub parent_item_id: Option<String>,
    pub fields: BTreeMap<String, DatagenFieldState>,
    pub completed_steps: BTreeSet<DatagenStepCursor>,
    pub status: DatagenItemStatus,
    pub terminal: Option<DatagenTerminal>,
    pub failure: Option<DatagenFailure>,
    pub query_tags: Option<Value>,
    pub current_run_id: String,
    pub last_item_seq: i64,
    pub last_checkpoint_id: String,
}

/// State captured immediately after a STEP_COMPLETED event.
#[derive(Debug, Clone, PartialEq)]
pub struct DatagenTrajectoryPoint {
    pub cursor: DatagenStepCursor,
    pub item: FoldedDatagenItem,
}

pub fn fold_datagen_events(events: &[DatagenEvent]) -> Result<FoldedDatagenItem, String> {
    let ordered = normalize_events(events)?;
    let first = ordered
        .first()
        .ok_or_else(|| "cannot fold an empty datagen event list".to_string())?;
    let mut item = initial_item(first);
    for event in ordered {
        apply_event(&mut item, event)?;
    }
    Ok(item)
}

pub fn datagen_trajectory(events: &[DatagenEvent]) -> Result<Vec<DatagenTrajectoryPoint>, String> {
    let ordered = normalize_events(events)?;
    let first = ordered
        .first()
        .ok_or_else(|| "cannot build a trajectory from an empty event list".to_string())?;
    let mut item = initial_item(first);
    let mut trajectory = Vec::new();
    for event in ordered {
        apply_event(&mut item, event)?;
        if event.event_type == DatagenEventType::StepCompleted {
            let cursor = step_cursor(event)?;
            trajectory.push(DatagenTrajectoryPoint {
                cursor,
                item: item.clone(),
            });
        }
    }
    Ok(trajectory)
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

fn initial_item(first: &DatagenEvent) -> FoldedDatagenItem {
    FoldedDatagenItem {
        item_id: first.item_id.clone(),
        root_item_id: first.root_item_id.clone(),
        parent_item_id: first.parent_item_id.clone(),
        fields: BTreeMap::new(),
        completed_steps: BTreeSet::new(),
        status: DatagenItemStatus::Pending,
        terminal: None,
        failure: None,
        query_tags: None,
        current_run_id: first.run_id.clone(),
        last_item_seq: first.item_seq,
        last_checkpoint_id: first.checkpoint_id.clone(),
    }
}

fn apply_event(item: &mut FoldedDatagenItem, event: &DatagenEvent) -> Result<(), String> {
    if event.item_id != item.item_id {
        return Err(format!(
            "event '{}' belongs to item '{}', expected '{}'",
            event.event_id, event.item_id, item.item_id
        ));
    }
    if event.root_item_id != item.root_item_id {
        return Err(format!(
            "item '{}' changed root_item_id from '{}' to '{}'",
            item.item_id, item.root_item_id, event.root_item_id
        ));
    }
    if event.parent_item_id != item.parent_item_id {
        return Err(format!(
            "item '{}' changed parent_item_id during its trajectory",
            item.item_id
        ));
    }

    item.current_run_id = event.run_id.clone();
    item.last_item_seq = event.item_seq;
    item.last_checkpoint_id = event.checkpoint_id.clone();

    match event.event_type {
        DatagenEventType::ItemCreated => {
            item.status = DatagenItemStatus::Pending;
            item.terminal = None;
            item.failure = None;
            if event.query_tags.is_some() {
                item.query_tags = event.query_tags.clone();
            }
        }
        DatagenEventType::FieldSet => {
            let field_name = event.field_name.clone().unwrap();
            item.fields.insert(
                field_name,
                DatagenFieldState::Set(event.value.clone().unwrap()),
            );
            item.status = DatagenItemStatus::Running;
            item.terminal = None;
            item.failure = None;
        }
        DatagenEventType::FieldAppend => {
            let field_name = event.field_name.clone().unwrap();
            let value = event.value.clone().unwrap();
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
            item.status = DatagenItemStatus::Running;
            item.terminal = None;
            item.failure = None;
        }
        DatagenEventType::StepCompleted => {
            item.completed_steps.insert(step_cursor(event)?);
            item.status = DatagenItemStatus::Running;
            item.terminal = None;
            item.failure = None;
        }
        DatagenEventType::Failed => {
            item.status = DatagenItemStatus::Failed;
            item.terminal = None;
            item.failure = Some(DatagenFailure {
                event_id: event.event_id.clone(),
                run_id: event.run_id.clone(),
                checkpoint_id: event.checkpoint_id.clone(),
                item_seq: event.item_seq,
                step_name: event.step_name.clone(),
                error_type: event.error_type.clone().unwrap(),
                error_dump: event.error_dump.clone(),
                traceback: event.traceback.clone(),
                failed_at: event.event_ts,
            });
        }
        DatagenEventType::Terminal => {
            let terminal = event.terminal.unwrap();
            item.status = match terminal {
                DatagenTerminal::Completed => DatagenItemStatus::Completed,
                DatagenTerminal::Filtered => DatagenItemStatus::Filtered,
            };
            item.terminal = Some(terminal);
            item.failure = None;
        }
    }
    Ok(())
}

fn step_cursor(event: &DatagenEvent) -> Result<DatagenStepCursor, String> {
    Ok(DatagenStepCursor {
        checkpoint_id: event.checkpoint_id.clone(),
        step_name: event
            .step_name
            .clone()
            .ok_or_else(|| "STEP_COMPLETED missing step_name".to_string())?,
        step_index: event
            .step_index
            .ok_or_else(|| "STEP_COMPLETED missing step_index".to_string())?,
        step_instance_id: event
            .step_instance_id
            .clone()
            .ok_or_else(|| "STEP_COMPLETED missing step_instance_id".to_string())?,
        iteration: event.iteration,
        attempt: event.attempt,
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
            event_id: datagen_event_id("item-1", &checkpoint_id, 0),
            item_id: "item-1".to_string(),
            root_item_id: "item-1".to_string(),
            parent_item_id: None,
            item_seq: seq,
            checkpoint_id,
            event_type,
            step_name: None,
            step_index: None,
            step_instance_id: None,
            iteration: None,
            attempt: 0,
            run_id: "run-1".to_string(),
            writer_epoch: "writer-1".to_string(),
            field_name: None,
            field_type: None,
            codec_version: None,
            value: None,
            query_tags: None,
            terminal: None,
            error_type: None,
            error_dump: None,
            traceback: None,
            event_ts: Utc.timestamp_micros(1_700_000_000_000_000 + seq).unwrap(),
            schema_version: DATAGEN_SCHEMA_VERSION,
        }
    }

    fn completed_step(seq: i64) -> DatagenEvent {
        let mut event = event(seq, DatagenEventType::StepCompleted);
        event.step_name = Some("noop".to_string());
        event.step_index = Some(3);
        event.step_instance_id = Some("loop/2/noop".to_string());
        event.iteration = Some(2);
        event
    }

    #[test]
    fn no_op_step_is_present_in_fold_and_trajectory() {
        let created = event(0, DatagenEventType::ItemCreated);
        let completed = completed_step(1);

        let folded = fold_datagen_events(&[completed.clone(), created]).unwrap();
        assert_eq!(folded.status, DatagenItemStatus::Running);
        assert_eq!(folded.completed_steps.len(), 1);
        assert!(folded.fields.is_empty());

        let trajectory = datagen_trajectory(&[completed]).unwrap();
        assert_eq!(trajectory.len(), 1);
        assert_eq!(trajectory[0].cursor.step_instance_id, "loop/2/noop");
    }

    #[test]
    fn retry_duplicate_event_is_folded_once() {
        let created = event(0, DatagenEventType::ItemCreated);
        let mut append = event(1, DatagenEventType::FieldAppend);
        append.field_name = Some("messages".to_string());
        append.field_type = Some("json".to_string());
        append.codec_version = Some(1);
        append.value = Some(DatagenValue::Json(json!({"role": "assistant"})));
        append.step_name = Some("generate".to_string());
        append.step_index = Some(1);
        append.step_instance_id = Some("generate/0".to_string());

        let folded = fold_datagen_events(&[created, append.clone(), append.clone()]).unwrap();
        assert_eq!(
            folded.fields.get("messages"),
            Some(&DatagenFieldState::Appended(vec![DatagenValue::Json(
                json!({"role": "assistant"})
            )]))
        );
    }

    #[test]
    fn sequence_collision_is_rejected() {
        let created = event(0, DatagenEventType::ItemCreated);
        let first = completed_step(1);
        let mut second = first.clone();
        second.event_id = "different-event".to_string();
        second.checkpoint_id = "different-checkpoint".to_string();

        let error = fold_datagen_events(&[created, first, second]).unwrap_err();
        assert!(error.contains("conflicting events at item_seq 1"));
    }

    #[test]
    fn later_run_can_supersede_a_failure() {
        let created = event(0, DatagenEventType::ItemCreated);
        let mut failed = event(1, DatagenEventType::Failed);
        failed.error_type = Some("RuntimeError".to_string());

        let mut retried = event(2, DatagenEventType::ItemCreated);
        retried.run_id = "run-2".to_string();
        retried.checkpoint_id = "retry-created".to_string();
        retried.event_id = datagen_event_id("item-1", "retry-created", 0);

        let folded = fold_datagen_events(&[created, failed, retried]).unwrap();
        assert_eq!(folded.status, DatagenItemStatus::Pending);
        assert_eq!(folded.current_run_id, "run-2");
        assert!(folded.failure.is_none());
    }
}
