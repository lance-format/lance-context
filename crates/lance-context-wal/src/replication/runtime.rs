//! Bounded asynchronous callers over dedicated, durable Raft group workers.
//!
//! Each group has one owner thread; groups retain their shared physical engine
//! and failure fence. Storage and payload validation never run on the caller's
//! async executor. Only Node::poll completions answer commits and quorum reads.
//! Queue admission, cached status and peer enqueue are not durable acknowledgments.
//!
//! A transport must authenticate peers/callers, enforce frame limits before
//! allocation, hold outbound ownership through transmission, and implement
//! snapshot transfer and maintenance. This module is the worker layer, not a
//! network listener, disk admission policy, or complete ingestion service.
use super::{producer, publication as p, query, source, Node, Outcome, Packet, Status};
use crate::record::{Batch, RecordKind};
use anyhow::{ensure, Context, Result};
use protobuf::Message as _;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    task::{Context as TaskContext, Poll},
    thread::{self, JoinHandle, Thread},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc as async_mpsc, oneshot, OwnedSemaphorePermit, Semaphore};

mod images;
#[cfg(feature = "peer-transport")]
pub mod peer;

/// Logical owned frame bytes and requests shared across all physical groups.
/// Counts do not include caller allocations or an oversized allocation pinned
/// by a small Bytes slice. The transport must account for those owners too.
#[derive(Clone, Debug)]
pub struct Options {
    pub data_requests: usize,
    pub data_bytes: usize,
    pub control_requests: usize,
    pub control_bytes: usize,
    pub peer_requests: usize,
    pub peer_bytes: usize,
    pub tick_interval: Duration,
    /// Maximum commands from each lane before polling durable Raft progress.
    pub batch_requests: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            data_requests: 128,
            data_bytes: 64 << 20,
            control_requests: 64,
            control_bytes: 4 << 20,
            peer_requests: 128,
            peer_bytes: 64 << 20,
            tick_interval: Duration::from_millis(100),
            batch_requests: 32,
        }
    }
}
impl Options {
    fn validate(&self) -> Result<()> {
        for (count, bytes) in [
            (self.data_requests, self.data_bytes),
            (self.control_requests, self.control_bytes),
            (self.peer_requests, self.peer_bytes),
        ] {
            ensure!(
                (1..=4096).contains(&count) && (4096..=256 * 1024 * 1024).contains(&bytes),
                "invalid runtime credit limits"
            );
        }
        ensure!(
            (1..=256).contains(&self.batch_requests)
                && self.tick_interval >= Duration::from_millis(10)
                && self.tick_interval <= Duration::from_secs(1),
            "invalid runtime scheduling options"
        );
        Ok(())
    }
}

/// Typed commands preserve Node's existing validation and durable formats.
/// This scheduling enum is not a network or persisted wire format.
#[derive(Debug, Serialize)]
pub enum Command {
    Append { group: u64, batch: Batch },
    Acquire(p::Acquire),
    Plan(p::PlanRequest),
    Prepare(p::PrepareRequest),
    Bootstrap(p::BootstrapManifest),
    Reserve(p::ReserveManifest),
    Confirm(p::ConfirmPublication),
    Advance(p::AdvancePublication),
    RetireHistory(p::RetireManifestHistory),
    Pin(source::Permit),
    Close(source::ClosurePermit),
    Producer(producer::Command),
}
impl Command {
    fn group(&self, catalog: u64) -> u64 {
        match self {
            Self::Append { group, .. } => *group,
            Self::Pin(p) => p.selection().group,
            Self::Close(p) => p.closure().group,
            Self::Producer(p) => p.group,
            _ => catalog,
        }
    }
    fn execute(self, node: &mut Node, token: u64) -> Result<()> {
        match self {
            Self::Append { group, batch } => node.propose(group, token, batch),
            Self::Acquire(p) => node.acquire_publisher(token, p),
            Self::Plan(p) => node.plan_publication(token, p),
            Self::Prepare(p) => node.prepare_publication(token, p),
            Self::Bootstrap(p) => node.bootstrap_manifest(token, p),
            Self::Reserve(p) => node.reserve_manifest(token, p),
            Self::Confirm(p) => node.confirm_publication(token, p),
            Self::Advance(p) => node.advance_publication(token, p),
            Self::RetireHistory(p) => node.retire_manifest_history(token, p),
            Self::Pin(p) => node.pin_source(token, p),
            Self::Close(p) => node.close_source(token, p),
            Self::Producer(p) => node.producer_command(token, p),
        }
    }
}

/// Dropping this future does not revoke a queued or possibly committed command.
/// Credits cover queued, executing and completed-but-unconsumed replies. A
/// cancelled receiver leaves its credits with the worker until completion/stop.
/// Treat every error/cancellation as uncertain and reconcile the original identity.
pub struct Pending<T>(oneshot::Receiver<Completed<T>>);
struct Completed<T> {
    result: Result<T>,
    _credit: Credit,
}
impl<T> Future for Pending<T> {
    type Output = Result<T>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.0).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(value)) => Poll::Ready(value.result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(anyhow::anyhow!(
                "replica worker stopped; outcome unknown"
            ))),
        }
    }
}

struct Credits {
    slots: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
    #[cfg(feature = "peer-transport")]
    capacity_bytes: usize,
    #[cfg(feature = "peer-transport")]
    capacity_slots: usize,
}
struct Credit {
    _slot: OwnedSemaphorePermit,
    _bytes: OwnedSemaphorePermit,
}
impl Credits {
    fn new(slots: usize, bytes: usize) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(slots)),
            bytes: Arc::new(Semaphore::new(bytes)),
            #[cfg(feature = "peer-transport")]
            capacity_bytes: bytes,
            #[cfg(feature = "peer-transport")]
            capacity_slots: slots,
        }
    }
    fn reserve(&self, bytes: usize) -> Result<Credit> {
        let slot = self
            .slots
            .clone()
            .try_acquire_owned()
            .context("runtime request credits exhausted")?;
        let bytes = u32::try_from(
            bytes
                .checked_add(256)
                .context("runtime frame size overflow")?,
        )?;
        let bytes = self
            .bytes
            .clone()
            .try_acquire_many_owned(bytes)
            .context("runtime byte credits exhausted")?;
        Ok(Credit {
            _slot: slot,
            _bytes: bytes,
        })
    }
}
struct Completion<T> {
    reply: oneshot::Sender<Completed<T>>,
    _credit: Credit,
}
impl<T> Completion<T> {
    fn finish(self, result: Result<T>) {
        let _ = self.reply.send(Completed {
            result,
            _credit: self._credit,
        });
    }
}
enum Work {
    // Keep the data lane inline; only larger control metadata needs boxing.
    Append(u64, Batch, Completion<Outcome>),
    Commit(u64, Box<Command>, Completion<Outcome>),
    Query(u64, query::Query, Completion<query::Outcome>),
    Campaign(Completion<()>),
    Local(LocalWork),
}
type AfterPoll = Box<dyn FnOnce() + Send>;
type LocalWork = Box<dyn FnOnce(&mut Node) -> AfterPoll + Send>;
struct PeerWork {
    packet: Packet,
    credit: Credit,
}
struct Route {
    data: mpsc::SyncSender<Work>,
    control: mpsc::SyncSender<Work>,
    peers: mpsc::SyncSender<PeerWork>,
    wake: Thread,
    observation: Arc<Mutex<Observation>>,
}
struct Shared {
    stopped: AtomicBool,
    token: AtomicU64,
    data: Credits,
    control: Credits,
    incoming: Credits,
    outgoing: Credits,
    dropped_outgoing: AtomicU64,
    rejected_incoming: AtomicU64,
}

/// A stale, local observation. Its timestamp does not establish quorum freshness.
#[derive(Clone, Debug)]
pub struct Observation {
    pub status: Status,
    pub observed_at: Instant,
    pub polling: bool,
    pub stopped: bool,
    pub failure: Option<String>,
}

/// Keep this owner alive while encoding/transmitting its packet. A full outbox
/// drops peer traffic without blocking persistence workers; Raft retransmits.
pub struct Outbound {
    packet: Packet,
    _credit: Option<Credit>,
    image: Option<Arc<super::checkpoint::Image>>,
    snapshot_feedback: Option<images::SnapshotFeedback>,
}
impl Outbound {
    pub fn packet(&self) -> &Packet {
        &self.packet
    }
    /// Captured with the outgoing snapshot, before a later checkpoint can
    /// replace it. Retain this owner through transfer and report its outcome.
    pub fn snapshot_image(&self) -> Option<&Arc<super::checkpoint::Image>> {
        self.image.as_ref()
    }
}

#[derive(Clone)]
pub struct Handle {
    routes: Arc<BTreeMap<u64, Route>>,
    shared: Arc<Shared>,
    catalog: u64,
    cluster: uuid::Uuid,
    layout: [u8; 32],
    policy: [u8; 32],
    id: u64,
    voters: Arc<BTreeMap<u64, Vec<u64>>>,
    max_record: usize,
}
impl Handle {
    fn route(&self, group: u64) -> Result<&Route> {
        ensure!(
            !self.shared.stopped.load(Ordering::Acquire),
            "replica runtime stopped; outcome unknown"
        );
        self.routes
            .get(&group)
            .context("group not hosted by runtime")
    }
    fn token(&self) -> Result<u64> {
        let mut current = self.shared.token.load(Ordering::Relaxed);
        loop {
            let next = current
                .checked_add(1)
                .context("runtime request tokens exhausted")?;
            match self.shared.token.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(current),
                Err(observed) => current = observed,
            }
        }
    }
    fn enqueue(&self, route: &Route, work: Work, data: bool) -> Result<()> {
        let queue = if data { &route.data } else { &route.control };
        queue.try_send(work).map_err(|_| {
            anyhow::anyhow!("runtime queue unavailable; reconcile original request")
        })?;
        route.wake.unpark();
        Ok(())
    }
    pub fn commit(&self, command: Command) -> Result<Pending<Outcome>> {
        let group = command.group(self.catalog);
        let route = self.route(group)?;
        let data = matches!(command, Command::Append { .. });
        if let Command::Append { batch, .. } = &command {
            ensure!(
                batch.meta.kind == RecordKind::Data,
                "use typed commands for control records"
            );
            batch
                .meta
                .check_body_length(batch.body.len(), self.max_record)?;
        }
        let size = usize::try_from(bincode::serialized_size(&command)?)?;
        let credit = if data {
            &self.shared.data
        } else {
            &self.shared.control
        }
        // Catalog replies can include a previously selected manifest, larger
        // than this request. Reserve a conservative logical response allowance.
        .reserve(if data { size } else { size.max(16 * 1024) })?;
        let (reply, receive) = oneshot::channel();
        let completion = Completion {
            reply,
            _credit: credit,
        };
        let token = self.token()?;
        let work = match command {
            Command::Append { batch, .. } => Work::Append(token, batch, completion),
            control => Work::Commit(token, Box::new(control), completion),
        };
        self.enqueue(route, work, data)?;
        Ok(Pending(receive))
    }
    pub fn query(&self, group: u64, query: query::Query) -> Result<Pending<query::Outcome>> {
        let route = self.route(group)?;
        let response_bytes = match &query {
            query::Query::Publisher { .. } => p::MAX_TABLE_BYTES + 4096,
            query::Query::Pin { .. } | query::Query::SourcePin { .. } => {
                source::MAX_PIN_BYTES + 4096
            }
            query::Query::Accepted { limit, .. } => {
                ensure!((1..=128).contains(limit), "invalid accepted query bounds");
                256 * limit + 1024
            }
            _ => 16 * 1024,
        };
        let credit = self
            .shared
            .control
            .reserve(response_bytes.max(usize::try_from(bincode::serialized_size(&query)?)?))?;
        let (reply, receive) = oneshot::channel();
        self.enqueue(
            route,
            Work::Query(
                self.token()?,
                query,
                Completion {
                    reply,
                    _credit: credit,
                },
            ),
            false,
        )?;
        Ok(Pending(receive))
    }
    /// Starts an election; success is not a leadership or durability receipt.
    pub fn campaign(&self, group: u64) -> Result<Pending<()>> {
        let route = self.route(group)?;
        let credit = self.shared.control.reserve(0)?;
        let (reply, receive) = oneshot::channel();
        self.enqueue(
            route,
            Work::Campaign(Completion {
                reply,
                _credit: credit,
            }),
            false,
        )?;
        Ok(Pending(receive))
    }
    /// Authenticated transport enqueues a bounded decoded packet. Only the
    /// worker validates entry payloads; Ok here is not peer persistence.
    pub fn step(&self, packet: Packet) -> Result<()> {
        let route = self.route(packet.group)?;
        ensure!(
            packet.cluster == self.cluster
                && packet.layout == self.layout
                && packet.policy == self.policy,
            "foreign peer scope or policy"
        );
        ensure!(
            packet.message.to == self.id
                && packet.message.from != self.id
                && self.voters[&packet.group].contains(&packet.message.from),
            "foreign peer route"
        );
        ensure!(
            packet.message.snapshot.is_none(),
            "snapshot transfer requires separate durable installation"
        );
        let bytes = packet.message.compute_size() as usize;
        ensure!(bytes <= 16 << 20, "oversized peer message");
        let credit = self.shared.incoming.reserve(bytes)?;
        route
            .peers
            .try_send(PeerWork { packet, credit })
            .map_err(|_| anyhow::anyhow!("runtime peer queue unavailable"))?;
        route.wake.unpark();
        Ok(())
    }
    pub fn observation(&self, group: u64) -> Result<Observation> {
        let route = self.routes.get(&group).context("group not hosted")?;
        Ok(route
            .observation
            .lock()
            .map_err(|_| anyhow::anyhow!("observation poisoned"))?
            .clone())
    }
    pub fn peer_drops(&self) -> (u64, u64) {
        (
            self.shared.dropped_outgoing.load(Ordering::Relaxed),
            self.shared.rejected_incoming.load(Ordering::Relaxed),
        )
    }
}

pub struct Runtime {
    handle: Handle,
    workers: Vec<JoinHandle<Result<()>>>,
}
impl Runtime {
    /// Pass an already opened, configured node with any required signing keys.
    /// Queue credits do not establish physical disk bounds. Integrate node-wide
    /// disk admission, maintenance and snapshot transfer before exposing ingress.
    pub fn spawn(node: Node, options: Options) -> Result<(Self, async_mpsc::Receiver<Outbound>)> {
        options.validate()?;
        node.healthy()?;
        let shared = Arc::new(Shared {
            stopped: AtomicBool::new(false),
            token: AtomicU64::new(1),
            data: Credits::new(options.data_requests, options.data_bytes),
            control: Credits::new(options.control_requests, options.control_bytes),
            incoming: Credits::new(options.peer_requests, options.peer_bytes),
            outgoing: Credits::new(options.peer_requests, options.peer_bytes),
            dropped_outgoing: AtomicU64::new(0),
            rejected_incoming: AtomicU64::new(0),
        });
        let (outbox, receive) = async_mpsc::channel(options.peer_requests);
        let mut handle = Handle {
            routes: Arc::new(BTreeMap::new()),
            shared: shared.clone(),
            catalog: p::control_group(&node.layout),
            cluster: node.layout.cluster,
            layout: node.layout_hash,
            policy: node.policy,
            id: node.id,
            voters: Arc::new(node.layout.groups.clone()),
            max_record: node.limits.max_record_bytes as usize,
        };
        // Validate every initial view before starting any thread. A failure
        // must not leave an earlier group detached and still owning its log.
        let nodes = node
            .into_groups()
            .into_iter()
            .map(|node| {
                let group = *node.groups.keys().next().context("empty worker group")?;
                let status = node.status(group)?;
                Ok((node, group, status))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut routes = BTreeMap::new();
        let mut workers: Vec<JoinHandle<Result<()>>> = Vec::new();
        for (node, group, status) in nodes {
            let observation = Arc::new(Mutex::new(Observation {
                status,
                observed_at: Instant::now(),
                polling: false,
                stopped: false,
                failure: None,
            }));
            let (data, dr) = mpsc::sync_channel(options.data_requests);
            let (control, cr) = mpsc::sync_channel(options.control_requests);
            let (peers, pr) = mpsc::sync_channel(options.peer_requests);
            let (snapshot_feedback, snapshot_results) = mpsc::sync_channel(options.peer_requests);
            let worker = Worker {
                node,
                group,
                data: dr,
                control: cr,
                peers: pr,
                snapshot_feedback,
                snapshot_results,
                snapshot_attempt: 0,
                snapshot_attempts: BTreeMap::new(),
                shared: shared.clone(),
                outbox: outbox.clone(),
                observation: observation.clone(),
                options: options.clone(),
                writes: BTreeMap::new(),
                reads: BTreeMap::new(),
            };
            let join = match thread::Builder::new()
                .name(format!("wal-group-{group}"))
                .spawn(move || worker.run())
            {
                Ok(join) => join,
                Err(error) => {
                    shared.stopped.store(true, Ordering::Release);
                    for worker in &workers {
                        worker.thread().unpark();
                    }
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(error.into());
                }
            };
            routes.insert(
                group,
                Route {
                    data,
                    control,
                    peers,
                    wake: join.thread().clone(),
                    observation,
                },
            );
            workers.push(join);
        }
        handle.routes = Arc::new(routes);
        drop(outbox);
        Ok((Self { handle, workers }, receive))
    }
    pub fn handle(&self) -> Handle {
        self.handle.clone()
    }
    fn stop(&self) {
        self.handle.shared.stopped.store(true, Ordering::Release);
        for route in self.handle.routes.values() {
            route.wake.unpark();
        }
    }
    /// Await before reopening the original directories. Dropping/timing out
    /// this future cannot cancel an in-flight filesystem write or its join.
    pub async fn shutdown(mut self) -> Result<()> {
        self.stop();
        let workers = std::mem::take(&mut self.workers);
        tokio::task::spawn_blocking(move || {
            let mut failure = None;
            for worker in workers {
                match worker.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => failure = Some(e),
                    Err(_) => failure = Some(anyhow::anyhow!("replica worker panicked")),
                }
            }
            failure.map_or(Ok(()), Err)
        })
        .await?
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Worker {
    node: Node,
    group: u64,
    data: mpsc::Receiver<Work>,
    control: mpsc::Receiver<Work>,
    peers: mpsc::Receiver<PeerWork>,
    snapshot_feedback: mpsc::SyncSender<images::SnapshotResult>,
    snapshot_results: mpsc::Receiver<images::SnapshotResult>,
    snapshot_attempt: u64,
    snapshot_attempts: BTreeMap<u64, u64>,
    shared: Arc<Shared>,
    outbox: async_mpsc::Sender<Outbound>,
    observation: Arc<Mutex<Observation>>,
    options: Options,
    writes: BTreeMap<u64, Completion<Outcome>>,
    reads: BTreeMap<u64, Completion<query::Outcome>>,
}
impl Worker {
    fn track_write(&mut self, token: u64, reply: Completion<Outcome>, result: Result<()>) {
        match result {
            Ok(()) => {
                self.writes.insert(token, reply);
            }
            Err(e) => {
                reply.finish(Err(e));
            }
        }
    }
    fn work(&mut self, work: Work, after_poll: &mut Vec<AfterPoll>) {
        match work {
            Work::Append(token, batch, reply) => {
                let result = self.node.propose(self.group, token, batch);
                self.track_write(token, reply, result);
            }
            Work::Commit(token, command, reply) => {
                let result = command.execute(&mut self.node, token);
                self.track_write(token, reply, result);
            }
            Work::Query(token, query, reply) => match self.node.query(self.group, token, query) {
                Ok(()) => {
                    self.reads.insert(token, reply);
                }
                Err(e) => {
                    reply.finish(Err(e));
                }
            },
            Work::Campaign(reply) => match self.node.campaign(self.group) {
                Ok(()) => after_poll.push(Box::new(move || reply.finish(Ok(())))),
                Err(e) => {
                    reply.finish(Err(e));
                }
            },
            Work::Local(work) => after_poll.push(work(&mut self.node)),
        }
    }
    fn drive(&mut self) -> Result<()> {
        let mut next_tick = Instant::now() + self.options.tick_interval;
        while !self.shared.stopped.load(Ordering::Acquire) {
            let mut activity = false;
            let mut peer_credits = Vec::new();
            let mut after_poll = Vec::new();
            for _ in 0..self.options.batch_requests {
                let mut any = false;
                if let Ok(result) = self.snapshot_results.try_recv() {
                    self.snapshot_result(&result)?;
                    peer_credits.push(result.credit);
                    any = true;
                }
                if let Ok(work) = self.control.try_recv() {
                    self.work(work, &mut after_poll);
                    any = true;
                }
                if let Ok(peer) = self.peers.try_recv() {
                    if self.node.step(peer.packet).is_err() {
                        self.shared
                            .rejected_incoming
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    peer_credits.push(peer.credit);
                    any = true;
                }
                if let Ok(work) = self.data.try_recv() {
                    self.work(work, &mut after_poll);
                    any = true;
                }
                activity |= any;
                if !any {
                    break;
                }
            }
            if Instant::now() >= next_tick {
                self.node.tick()?;
                next_tick = Instant::now() + self.options.tick_interval;
                activity = true;
            }
            if activity {
                self.observation.lock().unwrap().polling = true;
                let events = self.node.poll()?;
                for done in events.completions {
                    let pending = self
                        .writes
                        .remove(&done.request)
                        .context("uncorrelated durable completion")?;
                    pending.finish(Ok(done.outcome));
                }
                for done in events.reads {
                    let pending = self
                        .reads
                        .remove(&done.request)
                        .context("uncorrelated quorum read")?;
                    pending.finish(Ok(done.outcome));
                }
                for packet in events.messages {
                    let destination = packet.message.to;
                    let snapshot = packet.message.snapshot.is_some();
                    let image = if snapshot {
                        match self
                            .node
                            .snapshot_image(packet.group, packet.message.get_snapshot())
                        {
                            Ok(image) => Some(image),
                            Err(_) => {
                                self.node.report_snapshot(self.group, destination, false)?;
                                self.shared.dropped_outgoing.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                        }
                    } else {
                        None
                    };
                    let bytes = packet.message.compute_size() as usize;
                    let credit = match self.shared.outgoing.reserve(bytes) {
                        Ok(credit) => credit,
                        Err(_) => {
                            if snapshot {
                                self.node.report_snapshot(self.group, destination, false)?;
                            }
                            self.shared.dropped_outgoing.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    };
                    let snapshot_feedback = if snapshot {
                        Some(self.snapshot_feedback(&packet)?)
                    } else {
                        None
                    };
                    // A rejected snapshot owner reports failure via its reserved
                    // feedback lane. Do not also report it inline.
                    if self
                        .outbox
                        .try_send(Outbound {
                            packet,
                            _credit: Some(credit),
                            image,
                            snapshot_feedback,
                        })
                        .is_err()
                    {
                        self.shared.dropped_outgoing.fetch_add(1, Ordering::Relaxed);
                    }
                }
                for completion in after_poll {
                    completion();
                }
                let status = self.node.status(self.group)?;
                let mut observation = self.observation.lock().unwrap();
                observation.status = status;
                observation.observed_at = Instant::now();
                observation.polling = false;
            }
            drop(peer_credits);
            if !activity {
                thread::park_timeout(next_tick.saturating_duration_since(Instant::now()));
            }
        }
        Ok(())
    }
    fn run(mut self) -> Result<()> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.drive()))
            .unwrap_or_else(|_| {
                Err(anyhow::anyhow!(
                    "replica worker panicked; outstanding outcomes unknown"
                ))
            });
        if result.is_err() {
            self.shared.stopped.store(true, Ordering::Release);
        }
        let mut observation = self.observation.lock().unwrap();
        observation.polling = false;
        observation.stopped = true;
        observation.failure = result.as_ref().err().map(ToString::to_string);
        drop(observation);
        // Dropping all queued and pending completions resolves receivers as
        // uncertain and releases credits. Never manufacture a successful ACK.
        result
    }
}

#[cfg(test)]
mod tests;
