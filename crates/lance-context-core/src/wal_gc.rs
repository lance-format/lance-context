//! Bounded process-wide execution of durable WAL retirement tickets.
//!
//! Merge ticket admission never waits for a reader or storage. Repeated submissions for
//! one table share a queue and at most one running collector. Saturation leaves
//! the already-persisted intent for lifecycle discovery; it cannot undo a merge.
//! Lifecycle discovery shares these bounds and yields after each listed intent.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use futures::{stream::BoxStream, TryStreamExt};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use tokio::sync::Notify;

use crate::metrics::observe_duration;

/// Both phases register readers and persist pre-drain retirement intents.
/// Retain is the rolling-upgrade phase: logical merges continue, but this
/// process cannot delete WAL payloads. Collect is an explicit operator barrier
/// AFTER all old readers and immediate deleters have drained from the fleet.
/// A local mode cannot prove that fleet-wide precondition by itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Mode {
    #[default]
    Retain,
    Collect,
}

impl Mode {
    fn parse(value: Option<&str>) -> lance::Result<Self> {
        match value {
            None | Some("retain") => Ok(Self::Retain),
            Some("collect") => Ok(Self::Collect),
            _ => Err(lance::Error::invalid_input(
                "ROCKETKEEP_WAL_LIFECYCLE must be retain or collect",
            )),
        }
    }

    pub(crate) fn configured() -> lance::Result<Self> {
        match std::env::var("ROCKETKEEP_WAL_LIFECYCLE") {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => Err(lance::Error::invalid_input(error.to_string())),
        }
    }
}

/// Whether this process may discover and collect retired WAL. Configure only
/// after the documented fleet upgrade barrier; never infer it from pending=0.
pub fn wal_collection_enabled() -> lance::Result<bool> {
    Ok(Mode::configured()? == Mode::Collect)
}

/// Admit durable WAL retirement work for a canonical main-branch table URI.
///
/// This opens only the object store, not a dataset, schema, index, or payload.
/// It works for Context and Datagen as well as Rollout and Generic tables,
/// including cold tables whose original process has exited. Every intent still
/// revalidates the shard manifest and registered readers before deleting files.
/// The service must supply durable table discovery and revisit failed attempts.
/// Admission shares the process-wide bounds and may wait for collector capacity;
/// do not call this from reads, pending probes, or while holding a merge claim.
pub async fn discover_retired_wal_at_uri(uri: &str) -> lance::Result<()> {
    discover_uri(Mode::configured()?, uri).await
}

async fn discover_uri(mode: Mode, uri: &str) -> lance::Result<()> {
    if mode == Mode::Retain {
        return Ok(());
    }
    let (store, root) = ObjectStore::from_uri(uri).await?;
    discover(mode, store, root).await
}

#[path = "wal_gc_root_discovery.rs"]
mod root_discovery;
pub use root_discovery::discover_retired_wal_under_root;

type TableKey = (String, Path);

struct Limits {
    workers: usize,
    tables: usize,
    tickets: usize,
}

struct Table {
    store: Arc<ObjectStore>,
    root: Path,
    pending: VecDeque<Path>,
    // Includes the running ticket, so cancellation can always put it back
    // without exceeding the admission bound.
    admitted: HashSet<Path>,
    running: bool,
    queued: bool,
    discovery: Option<Discovery>,
    // Remains true while a worker owns the listing, preventing duplicate scans.
    discovery_admitted: bool,
    next_discovery: bool,
}

struct Discovery {
    // ObjectStore listings are unordered. Retain the actual stream rather than
    // inventing a last-key cursor that could skip unseen durable intents.
    listing: BoxStream<'static, object_store::Result<object_store::ObjectMeta>>,
    retry: Option<Path>,
}

impl Table {
    fn new(store: Arc<ObjectStore>, root: Path) -> Self {
        Self {
            store,
            root,
            pending: VecDeque::new(),
            admitted: HashSet::new(),
            running: false,
            queued: false,
            discovery: None,
            discovery_admitted: false,
            next_discovery: false,
        }
    }
}

#[derive(Default)]
struct State {
    tables: HashMap<TableKey, Table>,
    ready: VecDeque<TableKey>,
    workers: usize,
    tickets: usize,
}

struct Collector {
    limits: Limits,
    state: Mutex<State>,
    capacity: Notify,
    #[cfg(test)]
    discovery_consumed: Notify,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Submission {
    accepted: usize,
    deferred: usize,
}

impl Collector {
    fn new(limits: Limits) -> Arc<Self> {
        assert!(limits.workers > 0 && limits.tables > 0 && limits.tickets > 0);
        Arc::new(Self {
            limits,
            state: Mutex::new(State::default()),
            capacity: Notify::new(),
            #[cfg(test)]
            discovery_consumed: Notify::new(),
        })
    }

    fn submit(
        self: &Arc<Self>,
        store: Arc<ObjectStore>,
        root: Path,
        tickets: Vec<Path>,
    ) -> Submission {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return Submission {
                accepted: 0,
                deferred: tickets.len(),
            };
        };
        // store_prefix includes account/container identity, unlike root alone.
        let key = (store.store_prefix.clone(), root.clone());
        let mut result = Submission::default();
        let spawn = {
            let mut state = self.state.lock().unwrap();
            for ticket in tickets {
                if state
                    .tables
                    .get(&key)
                    .is_some_and(|table| table.admitted.contains(&ticket))
                {
                    result.accepted += 1;
                    continue;
                }
                if state.tickets == self.limits.tickets
                    || (!state.tables.contains_key(&key)
                        && state.tables.len() == self.limits.tables)
                {
                    result.deferred += 1;
                    continue;
                }
                let table = state
                    .tables
                    .entry(key.clone())
                    .or_insert_with(|| Table::new(store.clone(), root.clone()));
                table.admitted.insert(ticket.clone());
                table.pending.push_back(ticket);
                if !table.running && !table.queued {
                    table.queued = true;
                    state.ready.push_back(key.clone());
                }
                state.tickets += 1;
                result.accepted += 1;
            }
            let spawn = (self.limits.workers - state.workers).min(state.ready.len());
            state.workers += spawn;
            spawn
        };
        self.spawn_workers(&runtime, spawn);
        result
    }

    /// Only the bounded background lifecycle traversal may wait here. Merge
    /// completion and metadata pending probes use nonblocking admission.
    async fn discover(self: &Arc<Self>, store: Arc<ObjectStore>, root: Path) -> lance::Result<()> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|error| lance::Error::io(error.to_string()))?;
        let key = (store.store_prefix.clone(), root.clone());
        loop {
            let changed = self.capacity.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let (admitted, spawn) = {
                let mut state = self.state.lock().unwrap();
                let admitted = if state.tables.get(&key).is_some_and(|t| t.discovery_admitted) {
                    true
                } else if state.tickets < self.limits.tickets
                    && (state.tables.contains_key(&key) || state.tables.len() < self.limits.tables)
                {
                    let table = state
                        .tables
                        .entry(key.clone())
                        .or_insert_with(|| Table::new(store.clone(), root.clone()));
                    table.discovery = Some(Discovery {
                        listing: store
                            .inner
                            .list(Some(&root.clone().join("_wal_retirements_v1"))),
                        retry: None,
                    });
                    table.discovery_admitted = true;
                    if !table.running && !table.queued {
                        table.queued = true;
                        state.ready.push_back(key.clone());
                    }
                    // A listing and its one in-flight intent occupy one ticket
                    // for their whole lifetime; no unbounded discovered queue.
                    state.tickets += 1;
                    true
                } else {
                    false
                };
                let spawn = (self.limits.workers - state.workers).min(state.ready.len());
                state.workers += spawn;
                (admitted, spawn)
            };
            self.spawn_workers(&runtime, spawn);
            if admitted {
                return Ok(());
            }
            changed.await;
        }
    }

    fn spawn_workers(self: &Arc<Self>, runtime: &tokio::runtime::Handle, spawn: usize) {
        for _ in 0..spawn {
            // Construct before spawn: cancellation before the first poll must
            // also return the worker reservation. No semaphore waiters spawn.
            let reservation = Worker {
                collector: self.clone(),
                registered: true,
            };
            runtime.spawn(async move {
                let mut reservation = reservation;
                while let Some(mut item) = reservation.take() {
                    let started = std::time::Instant::now();
                    item.run().await;
                    observe_duration!(
                        crate::metrics::ROLLOUT_WAL_MERGE_DURATION,
                        started.elapsed(),
                        "phase" => "delete",
                    );
                    // Failures retain the durable object but do not spin in an
                    // immediate retry loop. Periodic discovery must revisit it.
                    // Drop puts any other tickets for this table at the back
                    // of ready, giving another table a turn after every ticket.
                }
            });
        }
    }
}

struct Worker {
    collector: Arc<Collector>,
    registered: bool,
}

impl Worker {
    fn take(&mut self) -> Option<Work> {
        let mut state = self.collector.state.lock().unwrap();
        let Some(key) = state.ready.pop_front() else {
            // Return capacity while still holding the admission mutex. A new
            // submission cannot race an empty observation and strand work
            // behind a worker that is already on its way out.
            state.workers -= 1;
            self.registered = false;
            return None;
        };
        let table = state.tables.get_mut(&key).unwrap();
        debug_assert!(table.queued && !table.running);
        table.queued = false;
        table.running = true;
        let kind =
            if table.discovery.is_some() && (table.next_discovery || table.pending.is_empty()) {
                table.next_discovery = false;
                WorkKind::Discovery(table.discovery.take())
            } else {
                table.next_discovery = true;
                WorkKind::Ticket(table.pending.pop_front().unwrap())
            };
        Some(Work {
            collector: self.collector.clone(),
            key,
            store: table.store.clone(),
            root: table.root.clone(),
            kind,
            finished: false,
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if self.registered {
            self.collector.state.lock().unwrap().workers -= 1;
            self.collector.capacity.notify_waiters();
        }
    }
}

struct Work {
    collector: Arc<Collector>,
    key: TableKey,
    store: Arc<ObjectStore>,
    root: Path,
    kind: WorkKind,
    finished: bool,
}

enum WorkKind {
    Ticket(Path),
    Discovery(Option<Discovery>),
}

impl Work {
    async fn run(&mut self) {
        let key = match &mut self.kind {
            WorkKind::Ticket(key) => key.clone(),
            WorkKind::Discovery(slot) => {
                let discovery = slot.as_mut().unwrap();
                if discovery.retry.is_none() {
                    match tokio::time::timeout(Duration::from_secs(5), discovery.listing.try_next())
                        .await
                    {
                        Ok(Ok(Some(meta))) => {
                            discovery.retry = Some(meta.location);
                            #[cfg(test)]
                            self.collector.discovery_consumed.notify_one();
                        }
                        Ok(Ok(None)) => {
                            *slot = None;
                            return;
                        }
                        error => {
                            tracing::warn!(
                                ?error,
                                "WAL retirement discovery deferred; durable intents retained"
                            );
                            *slot = None;
                            return;
                        }
                    }
                }
                discovery.retry.clone().unwrap()
            }
        };
        if let Err(error) =
            crate::wal_retirement::recover_one(self.store.clone(), &self.root, &key).await
        {
            tracing::warn!(%error, "WAL retirement deferred; durable intent retained for recovery");
        }
        // A bad/live intent must not hide later entries. Retain the durable
        // object for the next traversal, but advance this actual listing.
        if let WorkKind::Discovery(Some(discovery)) = &mut self.kind {
            discovery.retry = None;
        }
        self.finished = true;
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        let mut state = self.collector.state.lock().unwrap();
        let table = state.tables.get_mut(&self.key).unwrap();
        table.running = false;
        let released = match &mut self.kind {
            WorkKind::Ticket(ticket) => {
                if self.finished {
                    table.admitted.remove(ticket);
                } else {
                    table.pending.push_front(ticket.clone());
                }
                self.finished
            }
            WorkKind::Discovery(discovery) => {
                // Cancellation preserves both the listing and any already
                // consumed key. A replacement runtime resumes that exact key.
                table.discovery = discovery.take();
                table.discovery_admitted = table.discovery.is_some();
                !table.discovery_admitted
            }
        };
        if table.pending.is_empty() && !table.discovery_admitted {
            debug_assert!(table.admitted.is_empty());
            state.tables.remove(&self.key);
        } else {
            table.queued = true;
            state.ready.push_back(self.key.clone());
        }
        if released {
            state.tickets -= 1;
        }
        drop(state);
        self.collector.capacity.notify_waiters();
    }
}

fn collector() -> &'static Arc<Collector> {
    static COLLECTOR: OnceLock<Arc<Collector>> = OnceLock::new();
    COLLECTOR.get_or_init(|| {
        Collector::new(Limits {
            workers: 4,
            tables: 256,
            tickets: 1024,
        })
    })
}

pub(crate) async fn discover(mode: Mode, store: Arc<ObjectStore>, root: Path) -> lance::Result<()> {
    if mode == Mode::Retain {
        return Ok(());
    }
    collector().discover(store, root).await
}

pub(crate) fn schedule(mode: Mode, store: Arc<ObjectStore>, root: Path, tickets: Vec<Path>) {
    if mode == Mode::Retain {
        return;
    }
    let submitted = collector().submit(store, root, tickets);
    if submitted.deferred > 0 {
        tracing::warn!(
            deferred = submitted.deferred,
            "WAL retirement queue full or runtime unavailable; durable intents retained"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance::dataset::mem_wal::ShardManifestStore;
    use object_store::ObjectStoreExt;
    use uuid::Uuid;

    #[test]
    fn collection_requires_explicit_activation() {
        assert_eq!(Mode::parse(None).unwrap(), Mode::Retain);
        assert_eq!(Mode::parse(Some("retain")).unwrap(), Mode::Retain);
        assert_eq!(Mode::parse(Some("collect")).unwrap(), Mode::Collect);
        for value in ["", "true", "COLLECT", " collect", "legacy"] {
            assert!(Mode::parse(Some(value)).is_err());
        }
    }

    fn ticket(root: &Path) -> Path {
        root.clone()
            .join("_wal_retirements_v1")
            .join(Uuid::new_v4().simple().to_string())
    }

    async fn idle(collector: &Collector) {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                {
                    let state = collector.state.lock().unwrap();
                    if state.workers == 0 {
                        assert_eq!(state.tickets, 0);
                        assert!(state.tables.is_empty() && state.ready.is_empty());
                        return;
                    }
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn submissions_coalesce_and_both_resource_bounds_hold() {
        let dir = tempfile::tempdir().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let collector = Collector::new(Limits {
            workers: 2,
            tables: 2,
            tickets: 3,
        });
        let a = ticket(&root);
        let b = ticket(&root);
        assert_eq!(
            collector.submit(store.clone(), root.clone(), vec![a.clone(), b]),
            Submission {
                accepted: 2,
                deferred: 0
            }
        );
        assert_eq!(
            collector.submit(
                store.clone(),
                root.clone(),
                vec![a, ticket(&root), ticket(&root)]
            ),
            Submission {
                accepted: 2,
                deferred: 1
            }
        );
        // The current-thread runtime has not polled spawned work yet.
        {
            let state = collector.state.lock().unwrap();
            assert_eq!(state.tables.len(), 1);
            assert_eq!(state.ready.len(), 1);
            assert_eq!(state.tickets, 3);
            assert!(state.workers <= 2);
        }
        assert_eq!(
            collector
                .submit(
                    store.clone(),
                    root.clone().join("other"),
                    vec![ticket(&root)]
                )
                .deferred,
            1
        );
        idle(&collector).await;

        let collector = Collector::new(Limits {
            workers: 1,
            tables: 2,
            tickets: 10,
        });
        for suffix in ["one", "two"] {
            let path = root.clone().join(suffix);
            assert_eq!(
                collector
                    .submit(store.clone(), path.clone(), vec![ticket(&path)])
                    .accepted,
                1
            );
        }
        let third = root.clone().join("three");
        assert_eq!(
            collector
                .submit(store, third.clone(), vec![ticket(&third)])
                .deferred,
            1
        );
        idle(&collector).await;
    }

    #[test]
    fn runtime_shutdown_before_first_poll_does_not_leak_worker_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (store, root) = runtime
            .block_on(ObjectStore::from_uri(dir.path().to_str().unwrap()))
            .unwrap();
        let collector = Collector::new(Limits {
            workers: 1,
            tables: 2,
            tickets: 4,
        });
        let key = ticket(&root);
        {
            let _entered = runtime.enter();
            collector.submit(store.clone(), root.clone(), vec![key.clone()]);
        }
        drop(runtime);
        assert_eq!(collector.state.lock().unwrap().workers, 0);
        assert_eq!(collector.state.lock().unwrap().tickets, 1);
        let restarted = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        restarted.block_on(async {
            collector.submit(store, root, vec![key]);
            idle(&collector).await;
        });
    }

    #[test]
    fn cancelling_an_active_collector_retains_and_recovers_the_intent() {
        cancellation_recovery(false);
    }

    #[test]
    fn cancelling_discovery_preserves_the_consumed_key_across_runtime_replacement() {
        cancellation_recovery(true);
    }

    fn cancellation_recovery(discovery: bool) {
        let dir = tempfile::tempdir().unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let collector = Collector::new(Limits {
            workers: 1,
            tables: 2,
            tickets: 4,
        });
        let (store, root, key, payload, reader) = runtime.block_on(async {
            let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
                .await
                .unwrap();
            let shard = Uuid::new_v4();
            // Metadata-only fixture for cancellation; real base publication is
            // covered by the paused rollout reader / grouped merge tests.
            let manifests = ShardManifestStore::new(store.clone(), &root, shard, 16);
            manifests
                .initialize_shard(0, Default::default())
                .await
                .unwrap();
            manifests
                .commit_update(0, |current| {
                    let mut next = current.clone();
                    next.version += 1;
                    next.current_generation = 2;
                    next
                })
                .await
                .unwrap();
            let keys = crate::wal_retirement::record(
                store.clone(),
                &root,
                shard,
                vec![crate::rollout_append::Generation {
                    number: 1,
                    path: "abcdef12_gen_1".into(),
                }],
            )
            .await
            .unwrap();
            let payload = root
                .clone()
                .join("_mem_wal")
                .join(shard.to_string())
                .join("abcdef12_gen_1")
                .join("data")
                .join("payload");
            store.put(&payload, b"retained").await.unwrap();
            let reader = root
                .clone()
                .join("_wal_readers_v1")
                .join(Uuid::new_v4().simple().to_string());
            store
                .put(&reader, Uuid::new_v4().simple().to_string().as_bytes())
                .await
                .unwrap();
            if discovery {
                collector
                    .discover(store.clone(), root.clone())
                    .await
                    .unwrap();
            } else {
                collector.submit(store.clone(), root.clone(), keys.clone());
            }
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    if collector
                        .state
                        .lock()
                        .unwrap()
                        .tables
                        .values()
                        .any(|t| t.running)
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            if discovery {
                tokio::time::timeout(
                    Duration::from_secs(5),
                    collector.discovery_consumed.notified(),
                )
                .await
                .unwrap();
            }
            (store, root, keys[0].clone(), payload, reader)
        });
        drop(runtime);
        {
            let state = collector.state.lock().unwrap();
            assert_eq!(state.workers, 0);
            assert_eq!(state.tickets, 1);
            assert_eq!(state.ready.len(), 1);
            assert!(!state.tables.values().next().unwrap().running);
        }
        let restarted = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        restarted.block_on(async {
            assert!(store.inner.head(&key).await.is_ok());
            assert!(store.inner.head(&payload).await.is_ok());
            store.inner.delete(&reader).await.unwrap();
            if discovery {
                // No explicit ticket resubmission: resume the retained stream.
                collector.discover(store.clone(), root).await.unwrap();
            } else {
                collector.submit(store.clone(), root, vec![key.clone()]);
            }
            idle(&collector).await;
            assert!(store.inner.head(&key).await.is_err());
            assert!(store.inner.head(&payload).await.is_err());
        });
    }
    #[tokio::test]
    async fn cold_reopen_discovers_more_intents_than_capacity_and_skips_poison() {
        let dir = tempfile::tempdir().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let shard = Uuid::new_v4();
        let manifests = ShardManifestStore::new(store.clone(), &root, shard, 16);
        manifests
            .initialize_shard(0, Default::default())
            .await
            .unwrap();
        manifests
            .commit_update(0, |current| {
                let mut next = current.clone();
                next.version += 1;
                next.current_generation = 10;
                next
            })
            .await
            .unwrap();
        let mut tickets = Vec::new();
        let mut payloads = Vec::new();
        for number in 1..=8 {
            let path = format!("abcdef12_gen_{number}");
            tickets.extend(
                crate::wal_retirement::record(
                    store.clone(),
                    &root,
                    shard,
                    vec![crate::rollout_append::Generation {
                        number,
                        path: path.clone(),
                    }],
                )
                .await
                .unwrap(),
            );
            let payload = root
                .clone()
                .join("_mem_wal")
                .join(shard.to_string())
                .join(path)
                .join("data")
                .join("payload");
            store.put(&payload, b"retired").await.unwrap();
            payloads.push(payload);
        }
        let poison = root
            .clone()
            .join("_wal_retirements_v1")
            .join("00000000000000000000000000000000");
        store.put(&poison, b"invalid intent").await.unwrap();
        drop(manifests);
        drop(store);
        let (reopened, reopened_root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let collector = Collector::new(Limits {
            workers: 1,
            tables: 1,
            tickets: 1,
        });
        // No remembered ticket is given to the new collector. Zero live WAL
        // generations do not make durable retirement work disappear.
        collector
            .discover(reopened.clone(), reopened_root.clone())
            .await
            .unwrap();
        collector
            .discover(reopened.clone(), reopened_root)
            .await
            .unwrap();
        {
            let state = collector.state.lock().unwrap();
            assert_eq!(state.tickets, 1);
            assert_eq!(state.tables.len(), 1);
        }
        idle(&collector).await;
        for key in tickets.iter().chain(payloads.iter()) {
            assert!(matches!(
                reopened.inner.head(key).await,
                Err(object_store::Error::NotFound { .. })
            ));
        }
        assert!(reopened.inner.head(&poison).await.is_ok());
    }
    #[tokio::test]
    async fn saturated_discovery_waits_without_spawning_and_resumes_after_capacity_release() {
        let dir = tempfile::tempdir().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let collector = Collector::new(Limits {
            workers: 1,
            tables: 1,
            tickets: 1,
        });
        // Reserve a discovery before its spawned worker gets its first poll.
        collector
            .discover(store.clone(), root.clone())
            .await
            .unwrap();
        let next = collector.discover(store, root.join("other"));
        tokio::pin!(next);
        assert!(futures::poll!(&mut next).is_pending());
        {
            let state = collector.state.lock().unwrap();
            assert_eq!(state.workers, 1);
            assert_eq!(state.tables.len(), 1);
            assert_eq!(state.tickets, 1);
        }
        tokio::time::timeout(Duration::from_secs(5), &mut next)
            .await
            .unwrap()
            .unwrap();
        idle(&collector).await;
    }
}

#[cfg(test)]
#[path = "wal_gc_cold_store_tests.rs"]
mod cold_store_tests;
