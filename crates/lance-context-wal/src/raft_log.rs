//! Synchronous durable log storage for static session-sharded Raft groups.
//!
//! One physical engine serves every group assigned to a node. A successful
//! `persist` writes entries, the term index and HardState in one engine batch,
//! then waits for a shared physical sync barrier.
//! This is a storage boundary, not a quorum ACK: a Ready driver must still obey
//! raft-rs's message, persistence, commit and application ordering requirements.
//! In particular, do not acknowledge a producer merely because `persist` returned.
//!
//! An application may checkpoint a validated, durable committed cut. The small
//! opaque descriptor and logical compaction cross one physical sync barrier.
//! A received Ready snapshot can atomically replace the log and persist its
//! suffix and HardState. Application snapshot files/transfer, driver integration,
//! maintenance scheduling, membership changes and replica replacement remain
//! higher-level responsibilities. Physical log maintenance is explicit.
//! Only the owning process may modify the files.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
#[cfg(any(test, feature = "replication"))]
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard};

use anyhow::{ensure, Context, Result};
use protobuf::Message;
use raft::{GetEntriesContext, RaftState, Storage, StorageError};
use raft_engine::{Engine, LogBatch, MessageExt};
use raft_proto::eraftpb::{ConfState, Entry, EntryType, HardState, Snapshot};

use crate::placement::Layout;

mod budget;
mod checkpoint;
pub use budget::{DiskMeasurement, DiskOptions};
mod maintenance;
pub use maintenance::LogMaintenance;
#[cfg(test)]
mod fault_tests;

#[cfg(not(test))]
type LogEngine = Engine;
#[cfg(test)]
type LogEngine = Engine<fault_tests::FaultFileSystem>;

const IDENTITY: &[u8] = b"identity";
const HARD: &[u8] = b"hard";
const CONF: &[u8] = b"conf";
const TERMS: &[u8] = b"terms";
const SNAPSHOT: &[u8] = b"snapshot";
const MAX_SNAPSHOT_BYTES: usize = 4096;
// raft-engine preallocates entry descriptors for the requested index interval,
// before applying max_size. Never pass an unbounded interval to that API.
const READ_CHUNK_ENTRIES: u64 = 256;

struct EntryIndex;
impl MessageExt for EntryIndex {
    type Entry = Entry;

    fn index(entry: &Entry) -> u64 {
        entry.index
    }
}

struct NodeEngine {
    engine: LogEngine,
    #[cfg(any(test, feature = "replication"))]
    directory: PathBuf,
    #[cfg(feature = "replication")]
    target_file_bytes: u64,
    written: Arc<AtomicU64>,
    budget: OnceLock<Arc<crate::admission::Budget>>,
    failed: AtomicBool,
    maintenance_gate: Arc<RwLock<()>>,
    filesystem: Arc<RwLock<()>>,
    append_gate: RwLock<()>,
    sync_gate: Mutex<()>,
    synced: AtomicU64,
    #[cfg(test)]
    fault: Arc<fault_tests::FaultControl>,
}

impl NodeEngine {
    fn check(&self) -> Result<()> {
        ensure!(
            !self.failed.load(Ordering::Acquire),
            "uncertain engine write: drop every handle and reopen the node log"
        );
        Ok(())
    }

    fn write(&self, batch: &mut LogBatch) -> Result<()> {
        let _maintenance = self
            .maintenance_gate
            .read()
            .map_err(|_| anyhow::anyhow!("maintenance gate poisoned"))?;
        let result = self.append_and_sync(batch);
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }

    fn append_and_sync(&self, batch: &mut LogBatch) -> Result<()> {
        let observed;
        {
            let _append = self
                .append_gate
                .read()
                .map_err(|_| anyhow::anyhow!("append gate poisoned"))?;
            self.check()?;
            self.engine_write(batch, false)?;
            // Capture while holding the read gate. A later generation can only
            // be published by a sync that began after this append completed.
            observed = self.synced.load(Ordering::Acquire);
        }
        let _sync = self
            .sync_gate
            .lock()
            .map_err(|_| anyhow::anyhow!("sync gate poisoned"))?;
        self.check()?;
        if self.synced.load(Ordering::Acquire) == observed {
            // raft-engine 0.4.2 panics on sync failure. Isolate the syncing write
            // from native write coalescing: another group must not return a
            // successful result while the native group leader is unwinding.
            let _exclusive = self
                .append_gate
                .write()
                .map_err(|_| anyhow::anyhow!("append gate poisoned"))?;
            self.check()?;
            self.sync_locked()?;
        }
        self.check()
    }

    // The caller holds sync_gate and the exclusive append_gate. A maintenance
    // barrier also covers appends waiting for their original sync generation.
    fn sync_locked(&self) -> Result<()> {
        let next = self
            .synced
            .load(Ordering::Acquire)
            .checked_add(1)
            .context("sync generation exhausted")?;
        let mut marker = LogBatch::default();
        marker.put(0, b"sync".to_vec(), next.to_le_bytes().to_vec())?;
        // Engine::sync() is a no-op in the pinned version: its empty batch
        // returns before sync. A nonempty marker is required here.
        self.engine_write(&mut marker, true)?;
        self.synced.store(next, Ordering::Release);
        Ok(())
    }

    fn engine_write(&self, batch: &mut LogBatch, sync: bool) -> Result<()> {
        // The caller retains append_gate until failure is published. No other
        // group can complete a successful sync barrier after an uncertain write.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.engine.write(batch, sync)
        })) {
            Ok(Ok(bytes)) => {
                if self
                    .written
                    .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                        n.checked_add(bytes as u64)
                    })
                    .is_err()
                {
                    self.failed.store(true, Ordering::Release);
                    anyhow::bail!("physical write counter exhausted; reopen the node log");
                }
                Ok(())
            }
            Ok(Err(error)) => {
                self.failed.store(true, Ordering::Release);
                Err(error).context("uncertain engine write; node log must be reopened")
            }
            Err(_) => {
                self.failed.store(true, Ordering::Release);
                anyhow::bail!("engine write panicked; drop every handle and reopen the node log")
            }
        }
    }
}

#[derive(Clone, Copy)]
struct TermRange {
    first: u64,
    last: u64,
    term: u64,
}

struct GroupState {
    hard: HardState,
    conf: ConfState,
    terms: Vec<TermRange>,
    snapshot: Snapshot,
}

impl GroupState {
    fn last(&self) -> u64 {
        self.terms
            .last()
            .map_or(self.boundary(), |range| range.last)
    }

    fn boundary(&self) -> u64 {
        self.snapshot.get_metadata().index
    }

    fn term(&self, index: u64) -> Option<u64> {
        if index == self.boundary() {
            return Some(self.snapshot.get_metadata().term);
        }
        self.terms
            .get(self.terms.partition_point(|range| range.last < index))
            .filter(|range| range.first <= index)
            .map(|range| range.term)
    }

    fn validate(&self) -> Result<()> {
        let metadata = self.snapshot.get_metadata();
        if self.boundary() == 0 {
            ensure!(
                self.snapshot == Snapshot::default(),
                "invalid empty checkpoint"
            );
        } else {
            checkpoint::validate(&self.snapshot, &self.conf)?;
        }
        let (mut previous_last, mut previous_term) = (metadata.index, metadata.term);
        for range in &self.terms {
            ensure!(
                range.first == previous_last + 1
                    && range.last >= range.first
                    && range.last < u64::MAX
                    && range.term > 0
                    && (range.term > previous_term
                        || (previous_last == self.boundary() && range.term == previous_term)),
                "invalid durable term index"
            );
            (previous_last, previous_term) = (range.last, range.term);
        }
        ensure!(
            self.hard.commit >= self.boundary()
                && self.hard.commit <= self.last()
                && self.hard.term >= previous_term
                && (self.hard.term > 0 || self.hard.vote == 0)
                && (self.hard.vote == 0 || self.conf.voters.contains(&self.hard.vote)),
            "HardState does not match the durable log or membership"
        );
        Ok(())
    }
    fn appended(&self, entries: &[Entry], hard: Option<&HardState>) -> Result<Self> {
        let next_hard = hard.unwrap_or(&self.hard);
        ensure!(
            next_hard.term >= self.hard.term
                && next_hard.commit >= self.hard.commit
                && (next_hard.term != self.hard.term
                    || self.hard.vote == 0
                    || next_hard.vote == self.hard.vote),
            "HardState term, vote or commit regression"
        );
        let mut terms = self.terms.clone();
        if let Some(first) = entries.first() {
            ensure!(
                first.index > self.hard.commit && first.index <= self.last() + 1,
                "entries overwrite committed data or leave a gap"
            );
            let mut previous_index = first.index - 1;
            let mut previous_term = self
                .term(previous_index)
                .context("missing preceding term")?;
            for entry in entries {
                ensure!(
                    entry.index == previous_index + 1
                        && entry.index < u64::MAX
                        && entry.term > 0
                        && entry.term >= previous_term
                        && entry.term <= next_hard.term
                        && entry.get_entry_type() == EntryType::EntryNormal,
                    "noncontiguous entries, invalid term or unsupported membership change"
                );
                previous_index = entry.index;
                previous_term = entry.term;
            }
            terms.retain(|range| range.first < first.index);
            if let Some(last) = terms.last_mut() {
                last.last = last.last.min(first.index - 1);
            }
            for entry in entries {
                if let Some(last) = terms.last_mut().filter(|range| range.term == entry.term) {
                    last.last = entry.index;
                } else {
                    terms.push(TermRange {
                        first: entry.index,
                        last: entry.index,
                        term: entry.term,
                    });
                }
            }
        }
        let next = GroupState {
            hard: next_hard.clone(),
            conf: self.conf.clone(),
            terms,
            snapshot: self.snapshot.clone(),
        };
        next.validate()?;
        Ok(next)
    }
}

fn encode_terms(terms: &[TermRange]) -> Result<Vec<u8>> {
    let capacity = terms.len().checked_mul(24).context("term index overflow")?;
    let mut bytes = Vec::with_capacity(capacity);
    for range in terms {
        for value in [range.first, range.last, range.term] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn decode_terms(bytes: &[u8]) -> Result<Vec<TermRange>> {
    ensure!(
        bytes.len().is_multiple_of(24),
        "truncated durable term index"
    );
    Ok(bytes
        .as_chunks::<24>()
        .0
        .iter()
        .map(|bytes| TermRange {
            first: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            last: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            term: u64::from_le_bytes(bytes[16..].try_into().unwrap()),
        })
        .collect())
}

/// A node's exclusively opened physical engine and its configured group handles.
///
/// The layout, node identity and initial membership are bound in one synced batch
/// before any handle is returned. Reopen requires the same complete layout.
pub struct PhysicalLog {
    groups: BTreeMap<u64, GroupLog>,
}

/// Local physical-file tuning, independent of the replicated layout and policy.
/// Thresholds trigger explicit maintenance; they are not disk-admission quotas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "service", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "service", serde(default, deny_unknown_fields))]
pub struct LogOptions {
    pub target_file_bytes: u64,
    /// Native file-count size estimate; oversized batches can exceed it.
    pub purge_threshold_bytes: u64,
}

impl Default for LogOptions {
    fn default() -> Self {
        Self {
            target_file_bytes: 64 * 1024 * 1024,
            purge_threshold_bytes: 256 * 1024 * 1024,
        }
    }
}

impl PhysicalLog {
    pub fn open(directory: &Path, layout: &Layout, node: u64) -> Result<Self> {
        Self::open_with_options(directory, layout, node, LogOptions::default())
    }

    #[cfg(test)]
    fn open_with_file_size(
        directory: &Path,
        layout: &Layout,
        node: u64,
        file_size: u64,
    ) -> Result<Self> {
        Self::open_with_options(
            directory,
            layout,
            node,
            LogOptions {
                target_file_bytes: file_size,
                purge_threshold_bytes: file_size,
            },
        )
    }

    pub fn open_with_options(
        directory: &Path,
        layout: &Layout,
        node: u64,
        options: LogOptions,
    ) -> Result<Self> {
        ensure!(
            options.target_file_bytes > 0
                && options.purge_threshold_bytes >= options.target_file_bytes
                && usize::try_from(options.purge_threshold_bytes).is_ok(),
            "invalid physical log thresholds"
        );
        layout.validate()?;
        ensure!(layout.nodes.contains_key(&node), "unknown local node");
        if !directory.exists() {
            std::fs::create_dir(directory)?;
            let parent = directory
                .parent()
                .filter(|path| !path.as_os_str().is_empty());
            std::fs::File::open(parent.unwrap_or_else(|| Path::new(".")))?.sync_all()?;
        }
        let directory = directory.to_str().context("log directory is not UTF-8")?;
        let config = raft_engine::Config {
            dir: directory.to_owned(),
            batch_compression_threshold: raft_engine::ReadableSize(0),
            target_file_size: raft_engine::ReadableSize(options.target_file_bytes),
            purge_threshold: raft_engine::ReadableSize(options.purge_threshold_bytes),
            purge_rewrite_threshold: Some(raft_engine::ReadableSize(options.target_file_bytes)),
            enable_log_recycle: false,
            ..Default::default()
        };
        #[cfg(not(test))]
        let engine = Engine::open(config)?;
        #[cfg(test)]
        let (engine, fault) = {
            let fs = Arc::new(fault_tests::FaultFileSystem::default());
            let control = fs.control.clone();
            (Engine::open_with_file_system(config, fs)?, control)
        };
        let node_id = node;
        let node = Arc::new(NodeEngine {
            engine,
            #[cfg(any(test, feature = "replication"))]
            directory: PathBuf::from(directory),
            #[cfg(feature = "replication")]
            target_file_bytes: options.target_file_bytes,
            written: Arc::new(AtomicU64::new(0)),
            budget: OnceLock::new(),
            failed: AtomicBool::new(false),
            maintenance_gate: Arc::new(RwLock::new(())),
            filesystem: Arc::new(RwLock::new(())),
            append_gate: RwLock::new(()),
            sync_gate: Mutex::new(()),
            synced: AtomicU64::new(0),
            #[cfg(test)]
            fault,
        });
        let identity = serde_json::to_vec(&("lance-context/raft-log/v1", node_id, layout))?;
        let assigned: BTreeMap<_, _> = layout
            .groups
            .iter()
            .filter(|(_, voters)| voters.contains(&node_id))
            .map(|(group, voters)| (*group, voters))
            .collect();
        if let Some(saved) = node.engine.get(0, IDENTITY) {
            ensure!(
                saved == identity,
                "node identity, layout or log format changed"
            );
        } else {
            ensure!(
                node.engine.is_empty(),
                "cannot adopt a nonempty unbound engine"
            );
            let mut batch = LogBatch::default();
            batch.put(0, IDENTITY.to_vec(), identity)?;
            for (group, voters) in &assigned {
                let mut conf = ConfState::default();
                conf.set_voters((*voters).clone());
                batch.put_message(*group, CONF.to_vec(), &conf)?;
                batch.put_message(*group, HARD.to_vec(), &HardState::default())?;
                batch.put(*group, TERMS.to_vec(), Vec::new())?;
            }
            node.write(&mut batch)?;
        }
        let expected: BTreeSet<_> = assigned.keys().copied().chain([0]).collect();
        ensure!(
            node.engine
                .raft_groups()
                .into_iter()
                .collect::<BTreeSet<_>>()
                == expected,
            "durable group inventory differs from the layout"
        );
        let mut groups = BTreeMap::new();
        for (group, voters) in assigned {
            let mut expected_conf = ConfState::default();
            expected_conf.set_voters(voters.clone());
            let state = GroupState {
                hard: node
                    .engine
                    .get_message(group, HARD)?
                    .context("missing HardState")?,
                conf: node
                    .engine
                    .get_message(group, CONF)?
                    .context("missing ConfState")?,
                terms: decode_terms(
                    &node
                        .engine
                        .get(group, TERMS)
                        .context("missing term index")?,
                )?,
                snapshot: match node.engine.get(group, SNAPSHOT) {
                    Some(bytes) => {
                        ensure!(
                            bytes.len() <= MAX_SNAPSHOT_BYTES,
                            "oversized durable snapshot"
                        );
                        let application = node
                            .engine
                            .get(group, b"application")
                            .context("unbound checkpoint")?;
                        ensure!(
                            !application.is_empty() && application.len() <= 4096,
                            "invalid checkpoint application binding"
                        );
                        Snapshot::parse_from_bytes(&bytes)?
                    }
                    None => Snapshot::default(),
                },
            };
            ensure!(state.conf == expected_conf, "static membership changed");
            state.validate()?;
            ensure!(
                node.engine.last_index(group) == state.terms.last().map(|range| range.last)
                    && node.engine.first_index(group)
                        == state.terms.first().map(|range| range.first),
                "entry bounds differ from durable term index"
            );
            groups.insert(
                group,
                GroupLog {
                    node: node.clone(),
                    group,
                    state: Arc::new(RwLock::new(state)),
                },
            );
        }
        Ok(Self { groups })
    }

    pub fn group(&self, group: u64) -> Result<GroupLog> {
        self.groups
            .get(&group)
            .cloned()
            .context("group is not hosted on this node")
    }
}

/// Cloneable raft-rs `Storage` view of one group in a shared engine.
///
/// Clones share a lock: readers cannot observe entries with an old term index
/// while a write is being synced. Different groups can submit engine writes
/// concurrently. Term queries inspect only the small cached range index; they
/// never decode a large entry payload just to compare log terms.
#[derive(Clone)]
pub struct GroupLog {
    node: Arc<NodeEngine>,
    group: u64,
    state: Arc<RwLock<GroupState>>,
}

impl GroupLog {
    #[cfg(all(test, feature = "replication"))]
    pub(crate) fn fail_next_sync_for_test(&self) {
        self.node.fault.fail_sync_for_test();
    }

    /// Bind an immutable application format/policy before its first entry.
    /// Reopen must supply identical bytes; an unidentified existing log cannot
    /// be silently interpreted as a different state machine.
    pub fn bind_application(&self, identity: &[u8]) -> Result<()> {
        ensure!(
            !identity.is_empty() && identity.len() <= 4096,
            "invalid application identity"
        );
        let state = self
            .state
            .write()
            .map_err(|_| anyhow::anyhow!("group state lock poisoned"))?;
        self.node.check()?;
        if let Some(saved) = self.node.engine.get(self.group, b"application") {
            ensure!(saved == identity, "application format or policy changed");
            return self.node.check();
        }
        ensure!(
            state.last() == 0,
            "cannot identify an existing application log"
        );
        let mut batch = LogBatch::default();
        batch.put(self.group, b"application".to_vec(), identity.to_vec())?;
        self.node.write(&mut batch)
    }

    fn read(&self) -> Result<RwLockReadGuard<'_, GroupState>> {
        self.node.check()?;
        let state = self
            .state
            .read()
            .map_err(|_| anyhow::anyhow!("group state lock poisoned"))?;
        self.node.check()?;
        Ok(state)
    }

    /// Atomically sync a Ready's entries and optional HardState.
    ///
    /// The entries may replace an uncommitted suffix, but cannot alter a committed
    /// prefix, leave a gap, change a vote in the same term or move commit backwards.
    /// Validation errors leave this handle usable. An engine write error has an
    /// uncertain outcome and invalidates every handle for this physical engine;
    /// stop the driver and reopen rather than blindly repeating that Ready.
    /// An empty entries/HardState update is a no-op, not a flush barrier.
    pub fn persist(&self, entries: &[Entry], hard: Option<&HardState>) -> Result<()> {
        if entries.is_empty() && hard.is_none() {
            return self.node.check();
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| anyhow::anyhow!("group state lock poisoned"))?;
        self.node.check()?;
        let next = state.appended(entries, hard)?;
        let mut batch = LogBatch::default();
        if !entries.is_empty() {
            batch.add_entries::<EntryIndex>(self.group, entries)?;
            batch.put(self.group, TERMS.to_vec(), encode_terms(&next.terms)?)?;
        }
        if let Some(hard) = hard {
            batch.put_message(self.group, HARD.to_vec(), hard)?;
        }
        if !batch.is_empty() {
            self.node.write(&mut batch)?;
        }
        *state = next;
        Ok(())
    }
}

fn storage_error(error: impl std::fmt::Display) -> raft::Error {
    StorageError::Other(Box::new(std::io::Error::other(error.to_string()))).into()
}

impl Storage for GroupLog {
    fn initial_state(&self) -> raft::Result<RaftState> {
        let state = self.read().map_err(storage_error)?;
        Ok(RaftState {
            hard_state: state.hard.clone(),
            conf_state: state.conf.clone(),
        })
    }

    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        _: GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        let state = self.read().map_err(storage_error)?;
        // Native maintenance can relocate entries and delete their old files.
        // Keep its exclusive gate out until indexes and payloads are both read.
        let _read = self
            .node
            .maintenance_gate
            .read()
            .map_err(|_| storage_error("maintenance gate poisoned"))?;
        self.node.check().map_err(storage_error)?;
        if low <= state.boundary() {
            return Err(StorageError::Compacted.into());
        }
        if low > high || high > state.last() + 1 {
            return Err(StorageError::Unavailable.into());
        }
        let limit = max_size.into().unwrap_or(u64::MAX);
        let (mut cursor, mut bytes) = (low, 0u64);
        let mut entries = Vec::new();
        while cursor < high {
            let end = cursor.saturating_add(READ_CHUNK_ENTRIES).min(high);
            let start = entries.len();
            let remaining = usize::try_from(limit.saturating_sub(bytes)).unwrap_or(usize::MAX);
            self.node
                .engine
                .fetch_entries_to::<EntryIndex>(
                    self.group,
                    cursor,
                    end,
                    Some(remaining),
                    &mut entries,
                )
                .map_err(storage_error)?;
            if entries.len() == start {
                return Err(StorageError::Unavailable.into());
            }
            for i in start..entries.len() {
                let size = u64::from(entries[i].compute_size());
                if i > 0 && size > limit.saturating_sub(bytes) {
                    entries.truncate(i);
                    return Ok(entries);
                }
                bytes = bytes.saturating_add(size);
            }
            cursor = entries.last().unwrap().index + 1;
            if bytes >= limit || cursor < end {
                break;
            }
        }
        Ok(entries)
    }

    fn term(&self, index: u64) -> raft::Result<u64> {
        let state = self.read().map_err(storage_error)?;
        if index < state.boundary() {
            return Err(StorageError::Compacted.into());
        }
        state.term(index).ok_or(StorageError::Unavailable.into())
    }

    fn first_index(&self) -> raft::Result<u64> {
        Ok(self.read().map_err(storage_error)?.boundary() + 1)
    }

    fn last_index(&self) -> raft::Result<u64> {
        Ok(self.read().map_err(storage_error)?.last())
    }

    fn snapshot(&self, request_index: u64, _: u64) -> raft::Result<Snapshot> {
        let state = self.read().map_err(storage_error)?;
        if state.boundary() == 0 || state.boundary() < request_index {
            return Err(StorageError::SnapshotTemporarilyUnavailable.into());
        }
        Ok(state.snapshot.clone())
    }
}
