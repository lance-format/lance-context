//! Mechanism experiment for raft-rs's actual asynchronous Ready contract.
//! Real GroupLog files and fsync, with bounded BufferedLog as the readable
//! pending view. This does not enable asynchronous persistence in the runtime, or
//! cover its snapshot/GC, query, admission or application integration.
use super::*;
use std::collections::VecDeque;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::buffered::{BufferedLog, Persisted, Window, WriteTask};
use raft::{RawNode, StateRole};
use raft_proto::eraftpb::{Message as RaftMessage, MessageType};

struct Deferred {
    number: u64,
    messages: Vec<RaftMessage>,
    apply: Vec<Entry>,
    bytes: usize,
}
struct Peer {
    raw: RawNode<BufferedLog>,
    log: GroupLog,
    tx: Option<mpsc::SyncSender<WriteTask>>,
    rx: mpsc::Receiver<(u64, Result<Persisted>)>,
    worker: Option<JoinHandle<()>>,
    pending: VecDeque<Deferred>,
    pending_bytes: usize,
    applied: Vec<Entry>,
}
impl Peer {
    fn open(directory: &Path, layout: &Layout, id: u64) -> Result<Self> {
        let physical = PhysicalLog::open(directory, layout, id)?;
        let log = physical.group(10)?;
        let cache = BufferedLog::new(
            log.clone(),
            Window {
                readies: 64,
                entries: 4096,
                bytes: 8 << 20,
            },
        )?;
        let config = raft::Config {
            id,
            election_tick: 10,
            heartbeat_tick: 2,
            check_quorum: true,
            pre_vote: true,
            max_uncommitted_size: 8 << 20,
            ..Default::default()
        };
        let raw = RawNode::new(
            &config,
            cache,
            &slog::Logger::root(slog::Discard, slog::o!()),
        )?;
        let (tx, writes) = mpsc::sync_channel::<WriteTask>(64);
        let (done, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            while let Ok(write) = writes.recv() {
                let number = write.number();
                let result = write.persist();
                let failed = result.is_err();
                if done.send((number, result)).is_err() || failed {
                    break;
                }
            }
        });
        Ok(Self {
            raw,
            log,
            tx: Some(tx),
            rx,
            worker: Some(worker),
            pending: VecDeque::new(),
            pending_bytes: 0,
            applied: Vec::new(),
        })
    }
    fn poll(&mut self) -> Result<Vec<RaftMessage>> {
        let mut messages = Vec::new();
        while let Ok((number, result)) = self.rx.try_recv() {
            messages.extend(self.complete(number, result)?);
        }
        if self.raw.has_ready() {
            let mut ready = self.raw.ready();
            ensure!(
                ready.snapshot().is_empty(),
                "snapshot outside this mechanism experiment"
            );
            let entries = ready.take_entries();
            let hard = ready.hs().cloned();
            let bytes = entries
                .iter()
                .map(|e| e.compute_size() as usize)
                .sum::<usize>();
            ensure!(
                self.pending.len() < 64 && self.pending_bytes + bytes <= 8 << 20,
                "experiment persistence window exhausted"
            );
            // RawNode must read every appended update BEFORE async advance.
            // Entry::data uses shared Bytes; cache append keeps the same body.
            let number = ready.number();
            let last = entries.last().cloned();
            let write = self.raw.mut_store().stage(number, entries, hard)?;
            if let Some(entry) = last.as_ref() {
                let cached = self.raw.store().entries(
                    entry.index,
                    entry.index + 1,
                    None,
                    GetEntriesContext::empty(false),
                )?;
                ensure!(cached[0] == *entry);
                ensure!(
                    cached[0].data.as_ptr() == entry.data.as_ptr(),
                    "cache copied the body"
                );
            }
            messages.extend(ready.take_messages());
            let deferred = Deferred {
                number,
                messages: ready.take_persisted_messages(),
                apply: ready.take_committed_entries(),
                bytes,
            };
            self.raw.advance_append_async(ready);
            self.pending_bytes += bytes;
            self.pending.push_back(deferred);
            self.tx
                .as_ref()
                .context("writer stopped")?
                .try_send(write)?;
        }
        Ok(messages)
    }

    fn complete(&mut self, number: u64, result: Result<Persisted>) -> Result<Vec<RaftMessage>> {
        // A failed sync cannot advance Raft, release durable messages or
        // make the test application's committed value visible.
        ensure!(self.raw.mut_store().complete(result?)? == number);
        let done = self
            .pending
            .pop_front()
            .context("unmatched persistence completion")?;
        ensure!(done.number == number, "out-of-order persistence completion");
        self.pending_bytes -= done.bytes;
        self.raw.on_persist_ready(number);
        if let Some(entry) = done.apply.last() {
            self.raw.advance_apply_to(entry.index);
        }
        self.applied.extend(done.apply);
        Ok(done.messages)
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
struct Cluster {
    peers: Vec<Peer>,
    next_tick: Instant,
    heartbeats: usize,
}
impl Cluster {
    fn open(root: &Path, layout: &Layout) -> Result<Self> {
        let peers = (1..=3)
            .map(|id| Peer::open(&root.join(id.to_string()), layout, id))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            peers,
            next_tick: Instant::now() + Duration::from_millis(100),
            heartbeats: 0,
        })
    }
    fn pump(&mut self) -> Result<()> {
        if Instant::now() >= self.next_tick {
            for peer in &mut self.peers {
                peer.raw.tick();
            }
            self.next_tick = Instant::now() + Duration::from_millis(100);
        }
        let mut messages = Vec::new();
        for peer in &mut self.peers {
            messages.extend(peer.poll()?);
        }
        for message in messages {
            self.heartbeats += usize::from(message.get_msg_type() == MessageType::MsgHeartbeat);
            self.peers[message.to as usize - 1].raw.step(message)?;
        }
        Ok(())
    }
    fn until(&mut self, predicate: impl Fn(&Self) -> bool) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !predicate(self) {
            ensure!(
                Instant::now() < deadline,
                "mechanism condition did not finish"
            );
            self.pump()?;
            thread::sleep(Duration::from_millis(2));
        }
        Ok(())
    }
    fn run_for(&mut self, duration: Duration, check: impl Fn(&Self) -> Result<()>) -> Result<()> {
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            self.pump()?;
            check(self)?;
            thread::sleep(Duration::from_millis(2));
        }
        Ok(())
    }
    fn applied(&self, peer: usize, index: u64) -> bool {
        self.peers[peer]
            .applied
            .last()
            .is_some_and(|e| e.index >= index)
    }
    fn propose(&mut self, value: u8) -> Result<u64> {
        self.peers[0].raw.propose(Vec::new(), vec![value; 4096])?;
        Ok(self.peers[0].raw.raft.raft_log.last_index())
    }
    fn start(&mut self) -> Result<u64> {
        self.peers[0].raw.campaign()?;
        self.until(|c| c.peers[0].raw.raft.state == StateRole::Leader)?;
        let index = self.propose(1)?;
        self.until(|c| {
            (0..3).all(|i| c.applied(i, index)) && c.peers.iter().all(|p| p.pending.is_empty())
        })?;
        Ok(index)
    }
}
fn layout() -> Layout {
    Layout {
        cluster: uuid::Uuid::from_u128(1),
        nodes: (1..=3)
            .map(|id| (id, format!("wal-{id}.example.invalid:7800")))
            .collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3])]),
        buckets: vec![10],
    }
}
fn reopened(root: &Path, layout: &Layout, first: u64, last: u64) -> Result<()> {
    for id in 1..=3 {
        let log = PhysicalLog::open(&root.join(id.to_string()), layout, id)?.group(10)?;
        ensure!(log.initial_state()?.hard_state.commit >= last);
        for (index, value) in [(first, 1), (last, 2)] {
            let entries = log.entries(index, index + 1, None, GetEntriesContext::empty(false))?;
            ensure!(entries.len() == 1 && entries[0].data.as_ref() == vec![value; 4096]);
        }
    }
    Ok(())
}

#[test]
fn async_ready_leader_keeps_heartbeats_without_early_application() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let mut c = Cluster::open(root.path(), &layout)?;
    let first = c.start()?;
    let term = c.peers[0].raw.raft.term;
    let store = c.peers[0].log.clone();
    let gate = store.block_sync_for_test();
    let checked = (|| {
        let last = c.propose(2)?;
        c.until(|_| store.appended_index_for_test().is_some_and(|i| i >= last))?;
        let before = c.heartbeats;
        c.run_for(Duration::from_millis(2500), |c| {
            ensure!(
                !c.applied(0, last),
                "leader applied before its fsync completed"
            );
            ensure!(
                c.peers
                    .iter()
                    .all(|p| p.raw.raft.leader_id == 1 && p.raw.raft.term == term),
                "blocked leader lost term despite active heartbeat processing"
            );
            Ok(())
        })?;
        ensure!(
            c.heartbeats >= before + 10,
            "no sustained heartbeat traffic"
        );
        Ok::<_, anyhow::Error>(last)
    })();
    drop(gate);
    let last = checked?;
    c.until(|c| (0..3).all(|i| c.applied(i, last)))?;
    drop(c);
    drop(store);
    reopened(root.path(), &layout, first, last)
}

#[test]
fn async_ready_cached_follower_entries_do_not_form_a_durable_quorum() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let mut c = Cluster::open(root.path(), &layout)?;
    let first = c.start()?;
    let second = c.peers[1].log.clone();
    let third = c.peers[2].log.clone();
    let second_gate = second.block_sync_for_test();
    let third_gate = third.block_sync_for_test();
    let checked = (|| {
        let last = c.propose(2)?;
        c.until(|c| {
            c.peers[0]
                .log
                .appended_index_for_test()
                .is_some_and(|i| i >= last)
                && second.appended_index_for_test().is_some_and(|i| i >= last)
                && third.appended_index_for_test().is_some_and(|i| i >= last)
        })?;
        c.run_for(Duration::from_millis(350), |c| {
            ensure!(
                !c.applied(0, last),
                "volatile follower entries counted as quorum"
            );
            Ok(())
        })?;
        Ok::<_, anyhow::Error>(last)
    })();
    drop(second_gate);
    let after = checked.and_then(|last| {
        c.until(|c| c.applied(0, last) && c.applied(1, last))?;
        ensure!(!c.applied(2, last), "blocked minority applied early");
        Ok(last)
    });
    drop(third_gate);
    let last = after?;
    c.until(|c| (0..3).all(|i| c.applied(i, last)))?;
    drop(c);
    drop(second);
    drop(third);
    reopened(root.path(), &layout, first, last)
}

#[test]
fn async_ready_sync_failure_never_advances_application() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let mut c = Cluster::open(root.path(), &layout)?;
    let first = c.start()?;
    let gate = c.peers[0].log.node.fault.pause_sync_for_test();
    c.peers[0].log.fail_next_sync_for_test();
    let last = c.propose(2)?;
    c.pump()?;
    gate.wait()?;
    ensure!(!c.applied(0, last));
    let pending = c.peers[0].pending.len();
    let pending_bytes = c.peers[0].pending_bytes;
    ensure!(pending > 0 && pending_bytes > 0);
    drop(gate);
    // Observe the original completion deterministically. Polling RawNode here
    // races its storage access with the worker fencing the failed engine, which
    // raft-rs treats as a fatal panic before the receipt reaches try_recv().
    // This mechanism fixture checks completion ordering; the runtime separately
    // catches fatal owner panics and resolves outstanding outcomes as unknown.
    let (number, receipt) = c.peers[0].rx.recv_timeout(Duration::from_secs(10))?;
    ensure!(c.peers[0].pending.front().unwrap().number == number);
    let result = c.peers[0].complete(number, receipt);
    let error = result.expect_err("failed fsync produced an applied value");
    ensure!(
        error.to_string().contains("engine write panicked"),
        "unexpected failure: {error:#}"
    );
    ensure!(!c.applied(0, last));
    ensure!(c.peers[0].pending.len() == pending);
    ensure!(c.peers[0].pending_bytes == pending_bytes);
    drop(c);
    let log = PhysicalLog::open(&root.path().join("1"), &layout, 1)?.group(10)?;
    ensure!(log.initial_state()?.hard_state.commit >= first);
    ensure!(
        log.entries(first, first + 1, None, GetEntriesContext::empty(false))?[0]
            .data
            .as_ref()
            == vec![1; 4096]
    );
    Ok(())
}
