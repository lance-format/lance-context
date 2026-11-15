//! Bounded readable pending entries for raft-rs asynchronous Ready processing.
//!
//! `stage` supplies Storage visibility, never durability. Send its write task to
//! one ordered I/O worker and return the successful receipt to `complete` before
//! calling `RawNode::on_persist_ready`. Client ACKs, application, messages and
//! snapshot publication still belong to the owning Raft driver.
//!
//! The owner must not otherwise mutate the underlying group while this view is
//! live. Local checkpoints use the staged view/receipt below; other snapshot
//! replacement must drain all writes and consume the view before installation.
//! physical maintenance retains its normal native read guards. No pending entry
//! bodies are decoded or copied: the cache and write task share protobuf Bytes.
use super::*;
use std::collections::VecDeque;

/// Bounds on outstanding writes, including overwritten entries still owned by
/// an I/O task. Bytes count encoded entries and owned entry descriptors; backing
/// allocation capacity pinned by shared Bytes must also be bounded by ingress.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub readies: usize,
    pub entries: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub readies: usize,
    pub entries: usize,
    pub bytes: usize,
}

struct Pending {
    number: u64,
    range: Option<(u64, u64)>,
    entries: usize,
    bytes: usize,
    hard: HardState,
}

struct Cached {
    number: u64,
    entry: Entry,
}

struct PendingCheckpoint {
    snapshot: Snapshot,
    previous: u64,
    bytes: usize,
}

#[derive(Default)]
struct Writer {
    last: Mutex<u64>,
    failed: AtomicBool,
}

/// A single-owner Raft Storage view. It retains only pending writes; completed
/// payloads fall back to the existing physical log instead of accumulating here.
pub struct BufferedLog {
    log: GroupLog,
    visible: GroupState,
    durable_hard: HardState,
    entries: BTreeMap<u64, Cached>,
    pending: VecDeque<Pending>,
    usage: Usage,
    window: Window,
    last: u64,
    writer: Arc<Writer>,
    checkpoint: Option<PendingCheckpoint>,
    durable_checkpoint: u64,
}

/// One original persistence operation. Dropping a task is not cancellation of
/// a task already running on a worker, and never supplies a durability receipt.
pub struct WriteTask {
    log: GroupLog,
    writer: Arc<Writer>,
    previous: u64,
    number: u64,
    entries: Vec<Entry>,
    hard: Option<HardState>,
}

/// Only a successful original native append/sync creates this receipt.
pub struct Persisted {
    writer: Arc<Writer>,
    number: u64,
}

/// A local image of already durable, applied state. The owning application must
/// install its readable immutable image before dispatching this task, retain
/// both image generations, and wait for the receipt before publishing success.
pub(crate) struct CheckpointWrite {
    log: GroupLog,
    writer: Arc<Writer>,
    previous: u64,
    snapshot: Snapshot,
    retained_bytes: u64,
    truncated: (u64, u64),
}

pub(crate) struct CheckpointPersisted {
    writer: Arc<Writer>,
    previous: u64,
    snapshot: Snapshot,
}

impl CheckpointWrite {
    pub(crate) fn persist(self) -> Result<CheckpointPersisted> {
        let result = (|| {
            let last = self
                .writer
                .last
                .lock()
                .map_err(|_| anyhow::anyhow!("writer poisoned"))?;
            ensure!(
                !self.writer.failed.load(Ordering::Acquire),
                "buffered writer failed"
            );
            ensure!(*last == self.previous, "out-of-order checkpoint write");
            self.log
                .checkpoint_retaining(&self.snapshot, self.retained_bytes)?;
            ensure!(
                self.log.read()?.truncated == self.truncated,
                "checkpoint truncation differs from staged view"
            );
            Ok(CheckpointPersisted {
                writer: self.writer.clone(),
                previous: self.previous,
                snapshot: self.snapshot.clone(),
            })
        })();
        if result.is_err() {
            self.writer.failed.store(true, Ordering::Release);
        }
        result
    }
}

impl WriteTask {
    pub fn number(&self) -> u64 {
        self.number
    }

    pub fn persist(self) -> Result<Persisted> {
        let result = (|| {
            let mut last = self
                .writer
                .last
                .lock()
                .map_err(|_| anyhow::anyhow!("writer poisoned"))?;
            ensure!(
                !self.writer.failed.load(Ordering::Acquire),
                "buffered writer failed"
            );
            ensure!(*last == self.previous, "out-of-order buffered write");
            self.log.persist(&self.entries, self.hard.as_ref())?;
            *last = self.number;
            Ok(Persisted {
                writer: self.writer.clone(),
                number: self.number,
            })
        })();
        if result.is_err() {
            self.writer.failed.store(true, Ordering::Release);
        }
        result
    }
}

impl BufferedLog {
    pub(crate) fn durable_log(&self) -> &GroupLog {
        &self.log
    }

    pub(crate) fn checkpoint_index(&self) -> Result<u64> {
        self.check()?;
        Ok(self.durable_checkpoint)
    }

    pub(crate) fn checkpoint_pending(&self) -> bool {
        self.checkpoint.is_some()
    }

    /// Speculate only the readable view of an already applied local image. No
    /// checkpoint completion or durable observation advances until its receipt.
    /// Reserve one native queue slot alongside Ready tasks, including on shutdown.
    pub(crate) fn stage_checkpoint(
        &mut self,
        snapshot: Snapshot,
        retained_bytes: u64,
    ) -> Result<CheckpointWrite> {
        self.check()?;
        ensure!(
            self.pending.is_empty() && self.checkpoint.is_none(),
            "checkpoint requires drained writer"
        );
        let next = self.log.checkpoint_view(&snapshot, retained_bytes)?;
        let bytes = 2 * snapshot.compute_size() as usize
            + std::mem::size_of::<PendingCheckpoint>()
            + std::mem::size_of::<CheckpointWrite>();
        ensure!(
            bytes <= self.window.bytes,
            "checkpoint exceeds pending window"
        );
        let task = CheckpointWrite {
            log: self.log.clone(),
            writer: self.writer.clone(),
            previous: self.last,
            snapshot: snapshot.clone(),
            retained_bytes,
            truncated: next.truncated,
        };
        self.checkpoint = Some(PendingCheckpoint {
            snapshot,
            previous: self.last,
            bytes,
        });
        self.usage.bytes += bytes;
        self.visible = next;
        Ok(task)
    }

    pub(crate) fn complete_checkpoint(&mut self, receipt: CheckpointPersisted) -> Result<()> {
        self.check()?;
        ensure!(
            Arc::ptr_eq(&receipt.writer, &self.writer),
            "foreign checkpoint receipt"
        );
        let pending = self
            .checkpoint
            .as_ref()
            .context("unmatched checkpoint receipt")?;
        ensure!(
            pending.previous == receipt.previous && pending.snapshot == receipt.snapshot,
            "checkpoint receipt differs from original task"
        );
        self.durable_checkpoint = receipt.snapshot.get_metadata().index;
        self.usage.bytes -= self.checkpoint.take().unwrap().bytes;
        Ok(())
    }

    pub fn new(log: GroupLog, window: Window) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&window.readies)
                && (1..=1_048_576).contains(&window.entries)
                && (4096..=1 << 30).contains(&window.bytes),
            "invalid pending log window"
        );
        let visible = log.read()?.clone();
        Ok(Self {
            durable_checkpoint: visible.snapshot.get_metadata().index,
            durable_hard: visible.hard.clone(),
            log,
            visible,
            entries: BTreeMap::new(),
            pending: VecDeque::new(),
            usage: Usage::default(),
            window,
            last: 0,
            writer: Arc::new(Writer::default()),
            checkpoint: None,
        })
    }

    fn check(&self) -> Result<()> {
        self.log.node.check()?;
        ensure!(
            !self.writer.failed.load(Ordering::Acquire),
            "buffered writer failed"
        );
        Ok(())
    }

    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Application/status consumers use this boundary, not Storage's visible
    /// HardState. It changes only after accepting a successful ordered receipt.
    pub fn durable_hard_state(&self) -> Result<&HardState> {
        self.check()?;
        Ok(&self.durable_hard)
    }

    fn entry_charge(entries: &[Entry]) -> Result<usize> {
        entries
            .iter()
            .try_fold(std::mem::size_of::<Pending>(), |sum, entry| {
                sum.checked_add(entry.compute_size() as usize)
                    .and_then(|sum| sum.checked_add(2 * std::mem::size_of::<Entry>()))
                    .context("pending entry accounting overflow")
            })
    }

    /// Inspect admission without taking ownership or changing any Ready state.
    /// Only this owner can stage/complete, so a successful check remains valid
    /// until the owner changes the view again.
    pub fn can_stage(&self, entries: &[Entry]) -> Result<bool> {
        self.can_stage_with(entries, &[], 0)
    }

    pub(crate) fn can_stage_with(
        &self,
        entries: &[Entry],
        extra: &[Entry],
        reserved: usize,
    ) -> Result<bool> {
        self.check()?;
        let bytes = Self::entry_charge(entries)?
            .checked_add(Self::entry_charge(extra)? - std::mem::size_of::<Pending>())
            .and_then(|bytes| bytes.checked_add(reserved))
            .context("pending entry accounting overflow")?;
        Ok(
            self.usage.readies + usize::from(self.checkpoint.is_some()) < self.window.readies
                && entries.len() + extra.len()
                    <= self.window.entries.saturating_sub(self.usage.entries)
                && bytes <= self.window.bytes.saturating_sub(self.usage.bytes),
        )
    }

    /// Refuses before changing the view when the pending window is full. The
    /// driver must reserve sufficient capacity before collecting a Ready, since
    /// a collected Ready cannot be discarded or recreated with a new identity.
    pub fn stage(
        &mut self,
        number: u64,
        entries: Vec<Entry>,
        hard: Option<HardState>,
    ) -> Result<WriteTask> {
        self.check()?;
        ensure!(number > self.last, "Ready number did not advance");
        ensure!(self.can_stage(&entries)?, "pending log window full");
        let bytes = Self::entry_charge(&entries)?;
        let next = if entries.is_empty() && hard.is_none() {
            None
        } else {
            Some(self.visible.appended(&entries, hard.as_ref())?)
        };
        if let Some(first) = entries.first() {
            drop(self.entries.split_off(&first.index));
        }
        self.entries.extend(entries.iter().map(|entry| {
            (
                entry.index,
                Cached {
                    number,
                    entry: entry.clone(),
                },
            )
        }));
        self.pending.push_back(Pending {
            number,
            range: entries
                .first()
                .zip(entries.last())
                .map(|(first, last)| (first.index, last.index)),
            entries: entries.len(),
            bytes,
            hard: next.as_ref().unwrap_or(&self.visible).hard.clone(),
        });
        self.usage.readies += 1;
        self.usage.entries += entries.len();
        self.usage.bytes += bytes;
        if let Some(next) = next {
            self.visible = next;
        }
        let previous = self.last;
        self.last = number;
        Ok(WriteTask {
            log: self.log.clone(),
            writer: self.writer.clone(),
            previous,
            number,
            entries,
            hard,
        })
    }

    /// Retire exactly the original completed write. An older completion cannot
    /// evict entries staged by a newer overlapping suffix replacement.
    pub fn complete(&mut self, receipt: Persisted) -> Result<u64> {
        self.check()?;
        ensure!(
            self.checkpoint.is_none(),
            "Ready completion preceded checkpoint receipt"
        );
        ensure!(
            Arc::ptr_eq(&receipt.writer, &self.writer),
            "foreign persistence receipt"
        );
        ensure!(
            self.pending
                .front()
                .is_some_and(|p| p.number == receipt.number),
            "out-of-order persistence receipt"
        );
        let pending = self.pending.pop_front().unwrap();
        if let Some((mut cursor, last)) = pending.range {
            while let Some((&index, cached)) = self.entries.range(cursor..=last).next() {
                let owned = cached.number == pending.number;
                cursor = index + 1;
                if owned {
                    self.entries.remove(&index);
                }
                if cursor > last {
                    break;
                }
            }
        }
        self.usage.readies -= 1;
        self.usage.entries -= pending.entries;
        self.usage.bytes -= pending.bytes;
        self.durable_hard = pending.hard;
        Ok(pending.number)
    }

    /// Only a drained view can hand its group back for checkpoint/install or
    /// synchronous use. Pending writes must retain this view until resolved.
    pub fn into_durable(self) -> Result<GroupLog> {
        self.check()?;
        ensure!(
            self.pending.is_empty() && self.checkpoint.is_none(),
            "pending log writes remain"
        );
        Ok(self.log)
    }
}

impl Storage for BufferedLog {
    fn initial_state(&self) -> raft::Result<RaftState> {
        self.check().map_err(storage_error)?;
        Ok(RaftState {
            hard_state: self.visible.hard.clone(),
            conf_state: self.visible.conf.clone(),
        })
    }

    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        _: GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        self.check().map_err(storage_error)?;
        if low <= self.visible.boundary() {
            return Err(StorageError::Compacted.into());
        }
        if low > high || high > self.visible.last() + 1 {
            return Err(StorageError::Unavailable.into());
        }
        let limit = max_size.into().unwrap_or(u64::MAX);
        let (mut cursor, mut bytes) = (low, 0u64);
        let mut entries = Vec::new();
        while cursor < high {
            if let Some(cached) = self.entries.get(&cursor) {
                let size = u64::from(cached.entry.compute_size());
                if !entries.is_empty() && size > limit.saturating_sub(bytes) {
                    return Ok(entries);
                }
                cursor += 1;
                bytes = bytes.saturating_add(size);
                entries.push(cached.entry.clone());
            } else {
                let end = self
                    .entries
                    .range(cursor..high)
                    .next()
                    .map_or(high, |(&index, _)| index);
                let next = self
                    .log
                    .read_entries(cursor, end, Some(limit.saturating_sub(bytes)))?;
                if next.is_empty() {
                    return Err(StorageError::Unavailable.into());
                }
                for entry in next {
                    let size = u64::from(entry.compute_size());
                    if !entries.is_empty() && size > limit.saturating_sub(bytes) {
                        return Ok(entries);
                    }
                    cursor = entry.index + 1;
                    bytes = bytes.saturating_add(size);
                    entries.push(entry);
                }
            }
            if bytes >= limit {
                break;
            }
        }
        Ok(entries)
    }

    fn term(&self, index: u64) -> raft::Result<u64> {
        self.check().map_err(storage_error)?;
        if index < self.visible.boundary() {
            return Err(StorageError::Compacted.into());
        }
        self.visible
            .term(index)
            .ok_or(StorageError::Unavailable.into())
    }

    fn first_index(&self) -> raft::Result<u64> {
        self.check().map_err(storage_error)?;
        Ok(self.visible.boundary() + 1)
    }

    fn last_index(&self) -> raft::Result<u64> {
        self.check().map_err(storage_error)?;
        Ok(self.visible.last())
    }

    fn snapshot(&self, request_index: u64, _: u64) -> raft::Result<Snapshot> {
        self.check().map_err(storage_error)?;
        let snapshot = &self.visible.snapshot;
        if snapshot.get_metadata().index == 0 || snapshot.get_metadata().index < request_index {
            return Err(StorageError::SnapshotTemporarilyUnavailable.into());
        }
        Ok(snapshot.clone())
    }
}

#[cfg(test)]
mod tests;
