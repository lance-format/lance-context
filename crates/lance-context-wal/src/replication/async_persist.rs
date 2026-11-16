//! Ordered bounded native I/O for the actual Node driver.
use super::*;
use crate::raft_log::buffered::{
    CheckpointPersisted, CheckpointWrite, Persisted, Window, WriteTask,
};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Mutex,
    },
    thread::{self, JoinHandle},
};

const READY_BYTES: usize = 1024 * 1024;
const READIES: usize = 128;
const CONTROL_RESERVE: usize = 256 * 1024;
const WRITE_BATCH: usize = 32;

#[derive(Debug)]
pub(super) struct PeerPersistenceCapacity;
impl std::fmt::Display for PeerPersistenceCapacity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("pending persistence capacity")
    }
}
impl std::error::Error for PeerPersistenceCapacity {}

struct Deferred {
    number: u64,
    messages: Vec<Message>,
    reads: Vec<raft::ReadState>,
    apply: Vec<Entry>,
    bytes: usize,
    _reservations: Vec<crate::admission::Reservation>,
}

enum Write {
    Ready(WriteTask),
    Checkpoint(CheckpointWrite),
}
enum Written {
    Ready(Persisted),
    Checkpoint(CheckpointPersisted),
}
impl Write {
    fn persist(
        self,
        queue: &mpsc::Receiver<Write>,
        held: &mut Option<Write>,
    ) -> Result<Vec<Written>> {
        match self {
            Self::Ready(task) => {
                let mut tasks = vec![task];
                while tasks.len() < WRITE_BATCH {
                    match queue.try_recv() {
                        Ok(Self::Ready(task)) => tasks.push(task),
                        Ok(checkpoint @ Self::Checkpoint(_)) => {
                            *held = Some(checkpoint);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                WriteTask::persist_batch(tasks)
                    .map(|receipts| receipts.into_iter().map(Written::Ready).collect())
            }
            Self::Checkpoint(task) => task
                .persist()
                .map(|receipt| vec![Written::Checkpoint(receipt)]),
        }
    }
}

pub(super) struct Persistence {
    window: Window,
    tx: Option<mpsc::SyncSender<Write>>,
    rx: mpsc::Receiver<Result<Written>>,
    writer: Option<JoinHandle<Result<()>>>,
    notified: Arc<AtomicBool>,
    pending: VecDeque<Deferred>,
    deferred_bytes: usize,
    delivered_apply: u64,
    max_application_record: usize,
    io: Arc<Mutex<runtime::PollObservation>>,
    checkpoint_draining: bool,
}

pub(super) fn window(options: &runtime::Options, limits: &Limits) -> Result<Window> {
    let bytes = options.data_bytes + options.peer_bytes + options.control_bytes;
    // One Ready can return a single oversized record or <=1MiB of smaller
    // entries. Every entry has positive term/index and therefore >=4 wire bytes.
    let application = (READY_BYTES / 4 + 1) * std::mem::size_of::<Entry>()
        + (limits.max_record_bytes as usize + 4096).max(READY_BYTES);
    ensure!(
        bytes >= application + limits.max_record_bytes as usize + 4096 + CONTROL_RESERVE,
        "async persistence window cannot cover one Ready and its application batch"
    );
    Ok(Window {
        readies: READIES,
        entries: 1_048_576,
        bytes,
    })
}

impl Persistence {
    fn new(group: u64, window: Window, applied: u64) -> Result<Self> {
        let (tx, work) = mpsc::sync_channel::<Write>(window.readies);
        let (done, rx) = mpsc::sync_channel(window.readies);
        let notified = Arc::new(AtomicBool::new(false));
        let signal = notified.clone();
        let io = Arc::new(Mutex::new(runtime::PollObservation::default()));
        let timing = io.clone();
        let owner = thread::current();
        let writer = thread::Builder::new()
            .name(format!("wal-persist-{group}"))
            .spawn(move || {
                let mut held = None;
                while let Some(task) = held.take().or_else(|| work.recv().ok()) {
                    let started = std::time::Instant::now();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        task.persist(&work, &mut held)
                    }))
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("native persistence worker panicked")));
                    timing
                        .lock()
                        .unwrap()
                        .record(started.elapsed(), result.is_ok());
                    match result {
                        Ok(receipts) => {
                            for receipt in receipts {
                                if done.send(Ok(receipt)).is_err() {
                                    return Ok(());
                                }
                                signal.store(true, Ordering::Release);
                                owner.unpark();
                            }
                        }
                        Err(error) => {
                            let failure = error.to_string();
                            let _ = done.send(Err(error));
                            signal.store(true, Ordering::Release);
                            owner.unpark();
                            anyhow::bail!(failure);
                        }
                    }
                }
                Ok(())
            })?;
        Ok(Self {
            window,
            tx: Some(tx),
            rx,
            writer: Some(writer),
            notified,
            pending: VecDeque::new(),
            deferred_bytes: 0,
            delivered_apply: applied,
            max_application_record: 0,
            io,
            checkpoint_draining: false,
        })
    }

    fn shutdown(&mut self) -> Result<()> {
        self.tx.take();
        if let Some(worker) = self.writer.take() {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("native persistence join panicked"))??;
        }
        Ok(())
    }
}
impl Drop for Persistence {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

impl Group {
    /// A heartbeat in the follower's already durable term carries no new log
    /// payload or vote. Let Raft refresh its election timer even when a local
    /// checkpoint or the bounded persistence window holds ordinary admission.
    pub(super) fn accepts_durable_term_heartbeat(&self, message: &Message) -> Result<bool> {
        if self.raw.raft.state != StateRole::Follower
            || message.get_msg_type() != MessageType::MsgHeartbeat
            || message.term != self.raw.raft.term
            || message.from != self.raw.raft.leader_id
            || !message.entries.is_empty()
        {
            return Ok(false);
        }
        let LogView::Buffered(log) = self.raw.store() else {
            return Ok(false);
        };
        Ok(log.durable_hard_state()?.term == message.term)
    }

    // Stop collecting new Ready records while the original pending writes
    // drain. The owner must keep polling and ticking; its existing safe-message
    // path still sends leader heartbeats without advancing durability.
    pub(super) fn drain_for_checkpoint(&mut self, draining: bool) {
        if let Some(persistence) = &mut self.persistence {
            persistence.checkpoint_draining = draining;
        }
    }

    pub(super) fn resume_persistence(&mut self) -> Result<()> {
        if let Some(persistence) = &self.persistence {
            self.raw.mut_store().buffered(persistence.window)?;
        }
        Ok(())
    }

    /// Includes Ready application/read/message ownership, not just staged I/O.
    pub(super) fn accepts_async(&self, extra: &[Entry]) -> Result<bool> {
        let Some(persistence) = &self.persistence else {
            return Ok(true);
        };
        if persistence.checkpoint_draining {
            return Ok(false);
        }
        let LogView::Buffered(log) = self.raw.store() else {
            return Ok(true);
        };
        let count = self
            .raw
            .raft
            .raft_log
            .persisted
            .saturating_sub(persistence.delivered_apply)
            .min((READY_BYTES / 4 + 1) as u64) as usize;
        // Keep body/control headroom before new input can advance commit. Include
        // persisted-but-uncommitted entries as a conservative descriptor bound;
        // a later quorum response must not strand an already admitted Ready.
        let application = persistence.application_record_bound()
            + count * std::mem::size_of::<Entry>()
            + CONTROL_RESERVE;
        let reads = self
            .raw
            .raft
            .read_states
            .iter()
            .map(|s| s.request_ctx.len() + std::mem::size_of::<raft::ReadState>())
            .sum::<usize>();
        let messages = if self.raw.raft.state == StateRole::Leader {
            0
        } else {
            self.raw
                .raft
                .msgs
                .iter()
                .map(|m| m.compute_size() as usize + std::mem::size_of::<Message>())
                .sum()
        };
        log.can_stage_with(
            &self.raw.raft.raft_log.unstable.entries,
            extra,
            persistence.deferred_bytes + application + reads + messages,
        )
    }

    pub(super) fn persistence_idle(&self) -> bool {
        self.persistence
            .as_ref()
            .is_none_or(|p| p.pending.is_empty())
            && !matches!(self.raw.store(), LogView::Buffered(log) if log.checkpoint_pending())
    }

    pub(super) fn start_checkpoint_write(
        &mut self,
        snapshot: raft_proto::eraftpb::Snapshot,
        retained_bytes: u64,
    ) -> Result<()> {
        ensure!(self.persistence_idle(), "checkpoint writer is not drained");
        let persistence = self
            .persistence
            .as_ref()
            .context("missing persistence owner")?;
        let task = self
            .raw
            .mut_store()
            .buffered(persistence.window)?
            .stage_checkpoint(snapshot, retained_bytes)?;
        persistence
            .tx
            .as_ref()
            .context("native writer stopped")?
            .try_send(Write::Checkpoint(task))
            .map_err(|_| anyhow::anyhow!("native checkpoint queue unavailable"))
    }

    fn finish_io(
        &mut self,
        id: u64,
        limits: &Limits,
        receipt: Written,
        events: &mut Events,
        outbound: &mut Vec<Message>,
    ) -> Result<()> {
        match receipt {
            Written::Ready(receipt) => self.finish_write(id, limits, receipt, events, outbound),
            Written::Checkpoint(receipt) => {
                let window = self
                    .persistence
                    .as_ref()
                    .context("missing persistence owner")?
                    .window;
                self.raw
                    .mut_store()
                    .buffered(window)?
                    .complete_checkpoint(receipt)
            }
        }
    }

    fn finish_write(
        &mut self,
        id: u64,
        limits: &Limits,
        receipt: Persisted,
        events: &mut Events,
        outbound: &mut Vec<Message>,
    ) -> Result<()> {
        let persistence = self
            .persistence
            .as_mut()
            .context("missing persistence owner")?;
        let number = self
            .raw
            .mut_store()
            .buffered(persistence.window)?
            .complete(receipt)?;
        let done = persistence
            .pending
            .pop_front()
            .context("unmatched persistence completion")?;
        ensure!(number == done.number, "out-of-order Ready completion");
        persistence.deferred_bytes -= done.bytes;
        self.raw.on_persist_ready(number);
        outbound.extend(done.messages);
        for read in done.reads {
            self.reads.confirm(&read.request_ctx, read.index)?;
        }
        let applied = done.apply.last().map(|e| e.index);
        apply(self, id, done.apply, limits, events)?;
        if let Some(index) = applied {
            self.raw.advance_apply_to(index);
        }
        Ok(())
    }

    /// Local checkpoint and received snapshot barriers retain completions for
    /// the next poll; waiting for original I/O never invents another operation.
    pub(super) fn quiesce(&mut self, id: u64, limits: &Limits) -> Result<()> {
        let mut events = Events::default();
        let mut messages = Vec::new();
        while !self.persistence_idle() {
            let result = self
                .persistence
                .as_ref()
                .unwrap()
                .rx
                .recv()
                .context("native writer disconnected")??;
            self.finish_io(id, limits, result, &mut events, &mut messages)?;
        }
        self.deferred_completions.extend(events.completions);
        self.deferred_messages.extend(messages);
        self.raw.mut_store().synchronous()
    }

    pub(super) fn drive_async(
        &mut self,
        id: u64,
        limits: &Limits,
        events: &mut Events,
        outbound: &mut Vec<Message>,
    ) -> Result<()> {
        loop {
            let result = match self.persistence.as_ref().unwrap().rx.try_recv() {
                Ok(result) => result?,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    anyhow::bail!("native writer disconnected")
                }
            };
            self.finish_io(id, limits, result, events, outbound)?;
        }
        let persistence = self.persistence.as_ref().unwrap();
        self.raw.mut_store().buffered(persistence.window)?;
        // Snapshot replacement keeps the existing synchronous application-image
        // protocol, after all earlier Ready completions have been consumed.
        if self.raw.raft.raft_log.unstable.snapshot.is_some() {
            if !self.persistence_idle() {
                return Ok(());
            }
            self.raw.mut_store().synchronous()?;
            let mut ready = self.raw.ready();
            images::install_received(self, &ready, id, events)?;
            outbound.extend(ready.take_messages());
            outbound.extend(ready.take_persisted_messages());
            for read in ready.take_read_states() {
                self.reads.confirm(&read.request_ctx, read.index)?;
            }
            apply(self, id, ready.take_committed_entries(), limits, events)?;
            let mut light = self.raw.advance(ready);
            if light.commit_index().is_some() {
                self.raw
                    .store()
                    .persist(&[], Some(&self.raw.raft.hard_state()))?;
            }
            outbound.extend(light.take_messages());
            apply(self, id, light.take_committed_entries(), limits, events)?;
            self.raw.advance_apply();
            self.persistence.as_mut().unwrap().delivered_apply = self.state.applied;
            let window = self.persistence.as_ref().unwrap().window;
            self.raw.mut_store().buffered(window)?;
        }
        if self.raw.raft.state == StateRole::Follower {
            let LogView::Buffered(log) = self.raw.store() else {
                unreachable!()
            };
            let durable = log.durable_hard_state()?;
            let (durable_term, durable_commit) = (durable.term, durable.commit);
            if durable_term == self.raw.raft.term {
                // Heartbeat responses do not advance the peer's matched index.
                // Bound their advisory commit field by our durable boundary,
                // and echo the read context without waiting for native writes. Term
                // changes, votes, Append responses, read results and apply
                // still use their original persisted Ready boundaries.
                // Incoming peer credits cover these messages until the runtime
                // transfers them to its bounded outbound queue in this poll.
                self.raw.raft.msgs.retain_mut(|message| {
                    if message.get_msg_type() == MessageType::MsgHeartbeatResponse
                        && message.term == durable_term
                    {
                        message.commit = message.commit.min(durable_commit);
                        outbound.push(std::mem::take(message));
                        false
                    } else {
                        true
                    }
                });
            }
        }
        if self.raw.raft.state == StateRole::Leader {
            // RawNode::ready classifies a leader's outbound messages as safe
            // before persistence. Send those messages without creating another
            // Ready record just for outgoing heartbeats or replication. This
            // also keeps heartbeats moving when the persistence window is full.
            // It neither advances persisted/applied indices nor releases pending
            // commit or read completions. Follower and candidate messages still
            // require the ordinary Ready boundary.
            outbound.append(&mut self.raw.raft.msgs);
        }
        while self.raw.has_ready() && self.accepts_async(&[])? {
            let mut ready = self.raw.ready();
            ensure!(
                ready.snapshot().is_empty(),
                "unexpected asynchronous snapshot"
            );
            let number = ready.number();
            let entries = ready.take_entries();
            let window = self.persistence.as_ref().unwrap().window;
            let task = self.raw.mut_store().buffered(window)?.stage(
                number,
                entries,
                ready.hs().cloned(),
            )?;
            outbound.extend(ready.take_messages());
            let messages = ready.take_persisted_messages();
            let reads = ready.take_read_states();
            let application = ready.take_committed_entries();
            let bytes = application
                .iter()
                .map(|e| e.compute_size() as usize + std::mem::size_of::<Entry>())
                .sum::<usize>()
                + messages
                    .iter()
                    .map(|m| m.compute_size() as usize + std::mem::size_of::<Message>())
                    .sum::<usize>()
                + reads
                    .iter()
                    .map(|r| r.request_ctx.len() + std::mem::size_of::<raft::ReadState>())
                    .sum::<usize>();
            let persistence = self.persistence.as_mut().unwrap();
            if let Some(entry) = application.last() {
                persistence.delivered_apply = entry.index;
            }
            persistence.deferred_bytes += bytes;
            let LogView::Buffered(log) = self.raw.store() else {
                unreachable!()
            };
            ensure!(
                log.usage().bytes + persistence.deferred_bytes <= window.bytes,
                "Ready reservation underestimated ownership"
            );
            persistence.pending.push_back(Deferred {
                number,
                messages,
                reads,
                apply: application,
                bytes,
                _reservations: std::mem::take(&mut self.peer_reservations),
            });
            self.raw.advance_append_async(ready);
            persistence
                .tx
                .as_ref()
                .context("native writer stopped")?
                .try_send(Write::Ready(task))
                .map_err(|_| anyhow::anyhow!("native writer queue unavailable"))?;
        }
        Ok(())
    }
}

impl Persistence {
    // Configured at enable; stored separately below to keep admission independent
    // from policy lookups and avoid decoding application entries before Ready.
    fn application_record_bound(&self) -> usize {
        self.max_application_record
    }
}

impl Node {
    pub(super) fn persistence_observation(
        &self,
        id: u64,
    ) -> Option<runtime::PersistenceObservation> {
        let group = self.groups.get(&id)?;
        let persistence = group.persistence.as_ref()?;
        let usage = match group.raw.store() {
            LogView::Buffered(log) => log.usage(),
            LogView::Durable(_) => crate::raft_log::buffered::Usage::default(),
        };
        Some(runtime::PersistenceObservation {
            pending_readies: usage.readies,
            pending_checkpoint: matches!(group.raw.store(), LogView::Buffered(log) if log.checkpoint_pending()),
            pending_entries: usage.entries,
            pending_bytes: usage.bytes,
            deferred_bytes: persistence.deferred_bytes,
            io: persistence.io.lock().unwrap().clone(),
        })
    }

    pub(super) fn enable_async_persistence(&mut self, options: &runtime::Options) -> Result<()> {
        let window = window(options, &self.limits)?;
        for (&id, group) in &mut self.groups {
            ensure!(group.persistence.is_none(), "persistence already enabled");
            group.raw.mut_store().buffered(window)?;
            let mut persistence = Persistence::new(id, window, group.state.applied)?;
            persistence.max_application_record =
                (self.limits.max_record_bytes as usize + 4096).max(READY_BYTES);
            group.persistence = Some(persistence);
        }
        Ok(())
    }

    pub(super) fn take_persistence_notification(&self) -> bool {
        let mut ready = false;
        for persistence in self.groups.values().filter_map(|g| g.persistence.as_ref()) {
            ready |= persistence.notified.swap(false, Ordering::AcqRel);
        }
        ready
    }

    pub(super) fn shutdown_persistence(&mut self) -> Result<()> {
        let mut result = Ok(());
        for group in self.groups.values_mut() {
            if let Some(persistence) = group.persistence.as_mut() {
                if let Err(error) = persistence.shutdown() {
                    result = Err(error);
                }
            }
        }
        result
    }
}
