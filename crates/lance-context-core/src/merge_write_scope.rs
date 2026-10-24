//! Cancel merge preparation without abandoning a manifest commit already sent
//! to storage. The executor must drain this scope before acknowledging terminal.
use futures::stream::BoxStream;
use lance::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_table::{
    format::{IndexMetadata, Manifest, Transaction},
    io::commit::{
        CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
    },
};
use object_store::path::Path;
use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};
use tokio::sync::{oneshot, Notify};

tokio::task_local! { static CURRENT: Arc<MergeWriteScope>; }
tokio::task_local! { static EXPECTED_BASE_VERSION: u64; }
tokio::task_local! { static INITIAL_DATASET_CREATE: (); }

/// Pin the next publication without replacing a dataset's captured owner guard.
pub(crate) async fn at_base_version<F: Future>(version: u64, work: F) -> F::Output {
    EXPECTED_BASE_VERSION.scope(version, work).await
}

/// A racing create may only publish version one. This is scoped to the create
/// future so the returned dataset remains usable for subsequent normal writes.
pub(crate) async fn at_initial_version<F: Future>(work: F) -> F::Output {
    INITIAL_DATASET_CREATE.scope((), work).await
}

#[derive(Debug, Default)]
struct Progress {
    closed: bool,
    active: usize,
    uncertain: bool,
}

/// Completed work for one execution. Decoded bytes are Arrow buffer sizes,
/// not network bytes. File encoding is not evidence of a committed merge.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MergeWork {
    /// Parallel staging can occupy several phases at once. Entering/leaving
    /// a phase is diagnostic only and never advances completed_steps.
    pub active_stages: std::collections::BTreeMap<String, u32>,
    pub wal_batches_read: u64,
    pub wal_rows_read: u64,
    pub wal_decoded_bytes: u64,
    pub encoded_bytes: u64,
    pub encoded_rows: u64,
    pub encoded_files: u64,
    /// Existing data/deletion references confirmed by validation HEADs.
    /// This is metadata progress, never evidence of rows merged or reclaimed.
    #[serde(default)]
    pub references_validated: u64,
    pub base_commits: u64,
    pub generations_reclaimed: u64,
    pub last_completed_stage: Option<String>,
}

pub(crate) struct WorkStage {
    scope: Option<Arc<MergeWriteScope>>,
    name: &'static str,
}

impl Drop for WorkStage {
    fn drop(&mut self) {
        if let Some(scope) = &self.scope {
            let mut work = scope.work.lock().unwrap();
            if let Some(count) = work.active_stages.get_mut(self.name) {
                *count -= 1;
                if *count == 0 {
                    work.active_stages.remove(self.name);
                }
            }
        }
    }
}

pub(crate) fn work_stage(name: &'static str) -> WorkStage {
    let scope = CURRENT.try_with(Arc::clone).ok();
    if let Some(scope) = &scope {
        *scope
            .work
            .lock()
            .unwrap()
            .active_stages
            .entry(name.into())
            .or_default() += 1;
    }
    WorkStage { scope, name }
}

pub trait CommitAuthorizer: std::fmt::Debug + Send + Sync {
    fn authorize<'a>(
        &'a self,
        resource: &'a str,
        version: u64,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

#[derive(Debug, Default)]
pub struct MergeWriteScope {
    completed_steps: Arc<AtomicU64>,
    work: Mutex<MergeWork>,
    observe_file_io: bool,
    /// Legacy diagnostics must not introduce detached storage tasks or alter
    /// cancellation/slot lifetimes before durable ownership is available.
    observation_only: bool,
    local_io: Mutex<Vec<LocalPreparationIo>>,
    // Lance keys its store cache by wrapper allocation. Reuse within this
    // execution, while keeping different executions' progress isolated.
    preparation_wrapper: OnceLock<Arc<dyn lance_io::object_store::WrappingObjectStore>>,
    progress: Mutex<Progress>,
    changed: Notify,
    authorizer: Option<Arc<dyn CommitAuthorizer>>,
    pin_opened_handles: bool,
    leaves: Mutex<Vec<tokio::task::AbortHandle>>,
    /// Test-only storage shim applied **beneath** the progress wrapper, so a
    /// delayed or frozen store is still observed as real IO by the watchdog.
    io_shim: Option<Arc<dyn lance_io::object_store::WrappingObjectStore>>,
}

#[derive(Debug)]
struct LocalPreparationIo {
    // Retain the identity so its address cannot be reused while registered.
    store: Arc<dyn object_store::ObjectStore>,
    tracker: lance_io::utils::tracking_store::IOTracker,
    baseline: u64,
}

/// Local write observations, not a durable fence or proof of remote quiescence.
#[derive(Clone, Debug, serde::Serialize)]
pub struct MergeWriteStatus {
    pub active_commits: usize,
    pub admission_closed: bool,
    pub uncertain_commit: bool,
}

impl MergeWriteScope {
    pub fn write_status(&self) -> MergeWriteStatus {
        let progress = self.progress.lock().unwrap();
        MergeWriteStatus {
            active_commits: progress.active,
            admission_closed: progress.closed,
            uncertain_commit: progress.uncertain,
        }
    }
    pub fn work(&self) -> MergeWork {
        self.work.lock().unwrap().clone()
    }
    pub fn completed_steps(&self) -> u64 {
        self.local_io.lock().unwrap().iter().fold(
            self.completed_steps.load(Ordering::Relaxed),
            |steps, io| {
                let stats = io.tracker.stats();
                steps.saturating_add(
                    stats
                        .read_bytes
                        .saturating_add(stats.written_bytes)
                        .saturating_sub(io.baseline),
                )
            },
        )
    }
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn for_legacy_observation() -> Arc<Self> {
        Arc::new(Self {
            observation_only: true,
            ..Self::default()
        })
    }

    /// Observe file IO on ephemeral legacy maintenance handles without changing
    /// their commit authority or detaching writes. This is diagnostics only:
    /// absence of progress does not authorize cancelling an unfenced writer.
    pub fn for_legacy_file_observation() -> Arc<Self> {
        Arc::new(Self {
            observation_only: true,
            observe_file_io: true,
            ..Self::default()
        })
    }

    pub fn with_authorizer(authorizer: Arc<dyn CommitAuthorizer>) -> Arc<Self> {
        Arc::new(Self {
            authorizer: Some(authorizer),
            ..Self::default()
        })
    }

    /// Bind newly opened dataset handles to this execution, including commits
    /// from child tasks. Use only for ephemeral maintenance handles: worker
    /// caches must keep resolving the current execution at each call instead.
    pub fn with_pinned_authorizer(authorizer: Arc<dyn CommitAuthorizer>) -> Arc<Self> {
        Arc::new(Self {
            authorizer: Some(authorizer),
            pin_opened_handles: true,
            ..Self::default()
        })
    }

    /// Preparation reads/writes immutable files. Observe successful file IO so
    /// long scans do not appear idle when Lance omits training callbacks.
    pub fn with_preparation_authorizer(authorizer: Arc<dyn CommitAuthorizer>) -> Arc<Self> {
        Arc::new(Self {
            authorizer: Some(authorizer),
            pin_opened_handles: true,
            observe_file_io: true,
            ..Self::default()
        })
    }

    /// Preparation scope whose object stores are wrapped by `shim` under the
    /// progress wrapper. For tests that delay or freeze storage beneath a
    /// real build to prove the watchdog judges actual IO.
    pub fn with_preparation_authorizer_and_io_shim(
        authorizer: Arc<dyn CommitAuthorizer>,
        shim: Arc<dyn lance_io::object_store::WrappingObjectStore>,
    ) -> Arc<Self> {
        Arc::new(Self {
            authorizer: Some(authorizer),
            pin_opened_handles: true,
            observe_file_io: true,
            io_shim: Some(shim),
            ..Self::default()
        })
    }

    /// Only after dropping the merge future and after a durable storage
    /// barrier fenced every admitted version. Aborting before that evidence
    /// would abandon an uncertain storage write.
    pub async fn abort_fenced_leaves(&self) {
        self.progress.lock().unwrap().closed = true;
        for leaf in std::mem::take(&mut *self.leaves.lock().unwrap()) {
            leaf.abort();
        }
        self.drain().await;
    }

    pub async fn run<F: Future>(self: &Arc<Self>, future: F) -> F::Output {
        CURRENT.scope(self.clone(), future).await
    }

    pub fn has_uncertain_commit(&self) -> bool {
        self.progress.lock().unwrap().uncertain
    }

    fn mark_uncertain(&self) {
        self.progress.lock().unwrap().uncertain = true;
    }

    /// Call only after dropping/joining the merge future. Prevent new commits
    /// and join all manifest writes that started before cancellation.
    pub async fn drain(&self) {
        self.progress.lock().unwrap().closed = true;
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.progress.lock().unwrap().active == 0 {
                return;
            }
            changed.await;
        }
    }
}

/// Count completed work, never timer ticks or admission attempts. This is a
/// stall diagnostic, not evidence that WAL has been durably reclaimed.
pub fn checkpoint() {
    let _ = CURRENT.try_with(|scope| scope.completed_steps.fetch_add(1, Ordering::Relaxed));
}

pub(crate) fn reference_validated() {
    let _ = CURRENT.try_with(|scope| {
        let mut work = scope.work.lock().unwrap();
        work.references_validated = work.references_validated.saturating_add(1);
        work.last_completed_stage = Some("reference_validation".into());
    });
    checkpoint();
}

pub(crate) fn wal_batch_read(rows: usize, bytes: usize) {
    let _ = CURRENT.try_with(|scope| {
        let mut work = scope.work.lock().unwrap();
        work.wal_batches_read += 1;
        work.wal_rows_read = work.wal_rows_read.saturating_add(rows as u64);
        work.wal_decoded_bytes = work.wal_decoded_bytes.saturating_add(bytes as u64);
        work.last_completed_stage = Some("wal_read".into());
    });
    checkpoint();
}

pub(crate) fn base_committed() {
    // Legacy observation counts at the commit handler, including fallback
    // paths which also invoke the owned executor's logical checkpoints.
    if CURRENT
        .try_with(|scope| scope.observation_only)
        .unwrap_or(false)
    {
        return;
    }
    record_base_commit();
}

fn record_base_commit() {
    let _ = CURRENT.try_with(|scope| {
        let mut work = scope.work.lock().unwrap();
        work.base_commits += 1;
        work.last_completed_stage = Some("base_commit".into());
    });
    checkpoint();
}

pub(crate) fn generations_reclaimed(count: usize) {
    if CURRENT
        .try_with(|scope| scope.observation_only)
        .unwrap_or(false)
    {
        return;
    }
    record_reclaimed(count);
}

fn record_reclaimed(count: usize) {
    if count == 0 {
        return;
    }
    let _ = CURRENT.try_with(|scope| {
        let mut work = scope.work.lock().unwrap();
        work.generations_reclaimed = work.generations_reclaimed.saturating_add(count as u64);
        work.last_completed_stage = Some("wal_drain".into());
    });
    checkpoint();
}

/// Capture progress explicitly for Lance callbacks which can run on another task.
pub(crate) fn write_progress() -> lance::dataset::write::WriteProgressFn {
    let scope = CURRENT.try_with(Arc::clone).ok();
    let previous = std::sync::Mutex::new((0u64, 0u64, 0u32));
    lance::dataset::write::WriteProgressFn::new(move |stats| {
        let current = (stats.bytes_written, stats.rows_written, stats.files_written);
        let mut old = previous.lock().unwrap();
        if current.0 > old.0 || current.1 > old.1 || current.2 > old.2 {
            if let Some(scope) = &scope {
                let mut work = scope.work.lock().unwrap();
                work.encoded_bytes = work
                    .encoded_bytes
                    .saturating_add(current.0.saturating_sub(old.0));
                work.encoded_rows = work
                    .encoded_rows
                    .saturating_add(current.1.saturating_sub(old.1));
                work.encoded_files = work
                    .encoded_files
                    .saturating_add(current.2.saturating_sub(old.2) as u64);
                work.last_completed_stage = Some("file_encoding".into());
                scope.completed_steps.fetch_add(1, Ordering::Relaxed);
            }
            *old = (
                old.0.max(current.0),
                old.1.max(current.1),
                old.2.max(current.2),
            );
        }
    })
}

/// Capture only the preparation scope. Cached handles from ordinary writers
/// must never acquire another task's diagnostic counter.
pub(crate) fn preparation_store_wrapper(
) -> Option<Arc<dyn lance_io::object_store::WrappingObjectStore>> {
    CURRENT
        .try_with(|scope| {
            scope.observe_file_io.then(|| {
                scope
                    .preparation_wrapper
                    .get_or_init(|| {
                        let progress: Arc<dyn lance_io::object_store::WrappingObjectStore> =
                            Arc::new(crate::preparation_io::ProgressWrapper(
                                scope.completed_steps.clone(),
                            ));
                        match &scope.io_shim {
                            // Progress outermost: it counts the shimmed store's
                            // completed operations, exactly as it would real storage.
                            Some(shim) => Arc::new(Layered {
                                outer: progress,
                                inner: shim.clone(),
                            })
                                as Arc<dyn lance_io::object_store::WrappingObjectStore>,
                            None => progress,
                        }
                    })
                    .clone()
            })
        })
        .ok()
        .flatten()
}

#[derive(Debug)]
struct Layered {
    outer: Arc<dyn lance_io::object_store::WrappingObjectStore>,
    inner: Arc<dyn lance_io::object_store::WrappingObjectStore>,
}

impl lance_io::object_store::WrappingObjectStore for Layered {
    fn wrap_paginated(
        &self,
        name: &str,
        original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        self.inner
            .wrap_paginated(name, original)
            .and_then(|inner| self.outer.wrap_paginated(name, inner))
    }
    fn wrap(
        &self,
        name: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        self.outer.wrap(name, self.inner.wrap(name, original))
    }
}

pub(crate) fn observes_preparation_io() -> bool {
    CURRENT
        .try_with(|scope| scope.observe_file_io)
        .unwrap_or(false)
}

/// Lance's direct local/uring readers and writers bypass ObjectStore wrappers.
/// Their tracker records completed byte IO. The unique preparation wrapper in
/// the store params isolates this handle's tracker from other cached stores.
pub(crate) fn register_local_preparation_io(store: &ObjectStore) {
    if !store.is_local() {
        return;
    }
    let _ = CURRENT.try_with(|scope| {
        if scope.observe_file_io {
            let mut registered = scope.local_io.lock().unwrap();
            if registered
                .iter()
                .any(|io| Arc::ptr_eq(&io.store, &store.inner))
            {
                return;
            }
            let tracker = store.io_tracker().clone();
            let stats = tracker.stats();
            registered.push(LocalPreparationIo {
                store: store.inner.clone(),
                tracker,
                baseline: stats.read_bytes.saturating_add(stats.written_bytes),
            });
        }
    });
}

pub(crate) async fn authorize(resource: &str, version: u64) -> Result<()> {
    if let Ok(scope) = CURRENT.try_with(Arc::clone) {
        if let Some(authorizer) = &scope.authorizer {
            authorizer.authorize(resource, version).await?;
        }
    }
    Ok(())
}

struct Active {
    scope: Arc<MergeWriteScope>,
    completed: bool,
}
impl Drop for Active {
    fn drop(&mut self) {
        let mut progress = self.scope.progress.lock().unwrap();
        progress.active -= 1;
        progress.uncertain |= !self.completed;
        drop(progress);
        self.scope.changed.notify_waiters();
    }
}

#[cfg(test)]
pub(crate) async fn shield<F, T, E>(write: F) -> std::result::Result<T, E>
where
    F: Future<Output = std::result::Result<T, E>> + Send + 'static,
    T: Send + 'static,
    E: From<Error> + Send + 'static,
{
    shield_observed(write, |result| result.is_ok()).await
}

async fn shield_observed<F, T, E, P>(write: F, progressed: P) -> std::result::Result<T, E>
where
    F: Future<Output = std::result::Result<T, E>> + Send + 'static,
    T: Send + 'static,
    E: From<Error> + Send + 'static,
    P: FnOnce(&std::result::Result<T, E>) -> bool + Send + 'static,
{
    let Ok(scope) = CURRENT.try_with(Arc::clone) else {
        return write.await;
    };
    {
        let mut progress = scope.progress.lock().unwrap();
        if progress.closed {
            return Err(Error::io("merge cancelled before manifest commit").into());
        }
        progress.active += 1;
    }
    let handles = scope.clone();
    let active = Active {
        scope,
        completed: false,
    };
    if active.scope.observation_only {
        let mut active = active;
        let result = write.await;
        if progressed(&result) {
            active.scope.completed_steps.fetch_add(1, Ordering::Relaxed);
        }
        if result.is_err() {
            active.scope.mark_uncertain();
        }
        active.completed = true;
        return result;
    }
    let (send, receive) = oneshot::channel();
    let leaf = tokio::spawn(async move {
        let mut active = active;
        let result = write.await;
        if progressed(&result) {
            active.scope.completed_steps.fetch_add(1, Ordering::Relaxed);
        }
        if result.is_err() {
            active.scope.mark_uncertain();
        }
        active.completed = true;
        drop(active);
        let _ = send.send(result);
    });
    handles.leaves.lock().unwrap().push(leaf.abort_handle());
    receive
        .await
        .map_err(|_| E::from(Error::io("manifest commit executor panicked")))?
}

/// A relative shard drain with admission before *each* immutable version write.
/// Putting one guard around commit_update's retry loop would let a revoked
/// execution obtain new versions after recovery had already fenced it.
pub(crate) async fn drain_generations(
    store: lance::dataset::mem_wal::ShardManifestStore,
    epoch: u64,
    generations: std::collections::HashSet<u64>,
) -> Result<lance_index::mem_wal::ShardManifest> {
    let store = Arc::new(store);
    let resource = format!("shard:{}", store.shard_id());
    for _ in 0..10 {
        let mut next = store
            .refresh_latest()
            .await?
            .ok_or_else(|| Error::io("Shard manifest not found"))?;
        if next.writer_epoch != epoch {
            return Err(Error::io("merge writer epoch changed before drain"));
        }
        next.version = next
            .version
            .checked_add(1)
            .ok_or_else(|| Error::io("manifest version overflow"))?;
        let before = next.sstables.len();
        next.sstables
            .retain(|g| !generations.contains(&g.generation));
        let removed = before - next.sstables.len();
        authorize(&resource, next.version).await?;
        let writer = store.clone();
        let scope = CURRENT.try_with(Arc::clone).ok();
        let (next, written) = shield_observed(
            async move {
                let written = commit_exact_shard(&writer, &next).await;
                if written
                    .as_ref()
                    .is_err_and(|e| !matches!(e, Error::RetryableCommitConflict { .. }))
                {
                    if let Some(scope) = scope {
                        scope.mark_uncertain();
                    }
                }
                Ok::<_, Error>((next, written))
            },
            |result| result.as_ref().is_ok_and(|(_, written)| written.is_ok()),
        )
        .await?;
        match written {
            Ok(_) => {
                if CURRENT
                    .try_with(|scope| scope.observation_only)
                    .unwrap_or(false)
                {
                    record_reclaimed(removed);
                }
                return Ok(next);
            }
            Err(Error::RetryableCommitConflict { .. }) => continue,
            Err(error) => return Err(error),
        }
    }
    Err(Error::io("shard drain exceeded conditional-write retries"))
}

/// Adapt Lance's retrying update API to one previously authorized version.
/// Returning the SAME manifest on a retry makes Lance's successor check reject
/// a moved predecessor before any new PUT; the caller must authorize again.
pub(crate) async fn commit_exact_shard(
    store: &lance::dataset::mem_wal::ShardManifestStore,
    next: &lance_index::mem_wal::ShardManifest,
) -> Result<lance_index::mem_wal::ShardManifest> {
    let moved = std::sync::atomic::AtomicBool::new(false);
    let result = store
        .commit_update(next.writer_epoch, |current| {
            if current.version.checked_add(1) != Some(next.version) {
                moved.store(true, Ordering::Relaxed);
            }
            next.clone()
        })
        .await;
    if result.is_err() && moved.load(Ordering::Relaxed) {
        return Err(Error::retryable_commit_conflict_source(
            next.version,
            "shard manifest predecessor changed; reauthorize".into(),
        ));
    }
    result
}

/// The owned protocol is supported only with immutable conditional manifest
/// creation, not Lance's fallback UnsafeCommitHandler or external catalogues.
pub fn supports_version_fencing(uri: &str) -> bool {
    let Some((scheme, _)) = uri.split_once(':') else {
        return true;
    };
    let is_scheme = scheme
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphabetic)
        && scheme
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'-' | b'.'));
    if !is_scheme || (cfg!(windows) && scheme.len() == 1) {
        return true;
    }
    matches!(
        scheme.to_ascii_lowercase().as_str(),
        "file"
            | "file-object-store"
            | "s3"
            | "gs"
            | "az"
            | "abfss"
            | "memory"
            | "oss"
            | "cos"
            | "tos"
            | "shared-memory"
    )
}

/// After durable commit admission has closed, occupy versions beyond its
/// watermarks using metadata-only conditional writes. A late old PUT can then
/// only conflict with an occupied version; it cannot change visible table/WAL
/// state. This neither opens a shard writer nor changes its epoch or WAL list.
pub async fn fence_manifest_versions(
    uri: &str,
    storage_options: Option<std::collections::HashMap<String, String>>,
    watermarks: &std::collections::BTreeMap<String, u64>,
    recovery_id: &str,
) -> Result<()> {
    if !supports_version_fencing(uri) {
        return Err(Error::invalid_input(
            "storage does not support merge version fencing",
        ));
    }
    if watermarks.is_empty() {
        return Ok(());
    }
    let mut builder = lance::dataset::builder::DatasetBuilder::from_uri(uri);
    if let Some(options) = storage_options {
        builder = builder.with_storage_options(options);
    }
    let mut dataset = builder.load().await?;
    if let Some(high) = watermarks.get("base") {
        for _ in 0..64 {
            dataset.checkout_latest().await?;
            if dataset.version().version > *high {
                break;
            }
            let marker = format!("{recovery_id}:{}", dataset.version().version);
            dataset
                .update_metadata([("lance-context.merge-recovery", marker.as_str())])
                .await?;
        }
        dataset.checkout_latest().await?;
        if dataset.version().version <= *high {
            return Err(Error::io(
                "base manifest barrier did not cross admitted versions",
            ));
        }
    }
    for (resource, high) in watermarks {
        if resource == "base" {
            continue;
        }
        let id = resource
            .strip_prefix("shard:")
            .and_then(|s| uuid::Uuid::parse_str(s).ok())
            .ok_or_else(|| Error::invalid_input("invalid shard manifest watermark"))?;
        let store = lance::dataset::mem_wal::ShardManifestStore::new(
            dataset.object_store(None).await?,
            &dataset.branch_location().path,
            id,
            16,
        );
        let mut fenced = false;
        for _ in 0..64 {
            let mut next = store
                .refresh_latest()
                .await?
                .ok_or_else(|| Error::io("Shard manifest missing during recovery"))?;
            if next.version > *high {
                fenced = true;
                break;
            }
            next.version = next
                .version
                .checked_add(1)
                .ok_or_else(|| Error::io("manifest version overflow"))?;
            match commit_exact_shard(&store, &next).await {
                Ok(_) => {}
                Err(Error::RetryableCommitConflict { .. }) => {}
                Err(error) => return Err(error),
            }
        }
        if !fenced {
            return Err(Error::io(
                "shard manifest barrier did not cross admitted versions",
            ));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct GuardedCommit {
    inner: Arc<dyn CommitHandler>,
    scope: Option<Arc<MergeWriteScope>>,
}
impl GuardedCommit {
    pub(crate) fn new(inner: Arc<dyn CommitHandler>) -> Self {
        Self {
            inner,
            scope: CURRENT
                .try_with(Arc::clone)
                .ok()
                .filter(|scope| scope.pin_opened_handles),
        }
    }
}

#[async_trait::async_trait]
#[allow(clippy::too_many_arguments)]
impl CommitHandler for GuardedCommit {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        let work = async {
            if INITIAL_DATASET_CREATE
                .try_with(|()| manifest.version != 1)
                .unwrap_or(false)
            {
                // Lance can rebase a raced Create into an overwrite of the
                // winner. Reject before the inner commit, without inviting
                // another conflict retry or retaining a restriction on the
                // returned dataset handle.
                return Err(CommitError::OtherError(Error::invalid_input(
                    "dataset creation cannot publish a successor manifest",
                )));
            }
            if EXPECTED_BASE_VERSION
                .try_with(|version| manifest.version != *version)
                .unwrap_or(false)
            {
                return Err(CommitError::CommitConflict);
            }
            authorize("base", manifest.version).await?;
            let inner = self.inner.clone();
            let mut owned_manifest = manifest.clone();
            let path = base_path.clone();
            let store = object_store.clone();
            let scope = CURRENT.try_with(Arc::clone).ok();
            let (location, committed) = shield_observed(
                async move {
                    let location = inner
                        .commit(
                            &mut owned_manifest,
                            indices,
                            &path,
                            &store,
                            manifest_writer,
                            naming_scheme,
                            transaction,
                        )
                        .await;
                    // Only a definite conflict proves this conditional commit did
                    // not happen. A transport/storage error can arrive after the
                    // remote service accepted it, even though the Rust future ended.
                    if matches!(&location, Err(CommitError::OtherError(_))) {
                        if let Some(scope) = scope {
                            scope.mark_uncertain();
                        }
                    }
                    Ok::<_, CommitError>((location, owned_manifest))
                },
                |result| result.as_ref().is_ok_and(|(location, _)| location.is_ok()),
            )
            .await?;
            *manifest = committed;
            if location.is_ok()
                && CURRENT
                    .try_with(|scope| scope.observation_only)
                    .unwrap_or(false)
            {
                record_base_commit();
            }
            location
        };
        match &self.scope {
            Some(scope) => scope.run(work).await,
            None => work.await,
        }
    }

    async fn resolve_latest_location(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> Result<ManifestLocation> {
        self.inner
            .resolve_latest_location(base_path, object_store)
            .await
    }
    async fn resolve_version_location(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &dyn object_store::ObjectStore,
    ) -> Result<ManifestLocation> {
        self.inner
            .resolve_version_location(base_path, version, object_store)
            .await
    }
    async fn version_exists(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &dyn object_store::ObjectStore,
        naming: ManifestNamingScheme,
    ) -> Result<bool> {
        self.inner
            .version_exists(base_path, version, object_store, naming)
            .await
    }
    fn list_detached_manifest_locations<'a>(
        &self,
        base_path: &Path,
        object_store: &'a ObjectStore,
    ) -> BoxStream<'a, Result<ManifestLocation>> {
        self.inner
            .list_detached_manifest_locations(base_path, object_store)
    }
    fn list_manifest_locations<'a>(
        &self,
        base_path: &Path,
        object_store: &'a ObjectStore,
        sorted: bool,
    ) -> BoxStream<'a, Result<ManifestLocation>> {
        self.inner
            .list_manifest_locations(base_path, object_store, sorted)
    }
    fn list_manifest_locations_since<'a>(
        &self,
        base_path: &Path,
        object_store: &'a ObjectStore,
        since: u64,
    ) -> BoxStream<'a, Result<ManifestLocation>> {
        self.inner
            .list_manifest_locations_since(base_path, object_store, since)
    }
    async fn delete(&self, base_path: &Path) -> Result<()> {
        self.inner.delete(base_path).await
    }
}

#[cfg(test)]
#[path = "merge_write_scope/create_tests.rs"]
mod create_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn preparation_store_cache_reuses_clients_without_cross_execution_progress() {
        use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry};
        let registry = ObjectStoreRegistry::default();
        let first = Arc::new(MergeWriteScope {
            observe_file_io: true,
            ..Default::default()
        });
        let second = Arc::new(MergeWriteScope {
            observe_file_io: true,
            ..Default::default()
        });
        let open = || async {
            registry
                .get_store(
                    "memory://preparation-cache/table".parse().unwrap(),
                    &ObjectStoreParams {
                        object_store_wrapper: preparation_store_wrapper(),
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
        };
        let store = first.run(open()).await;
        for _ in 0..24 {
            let reopened = first.run(open()).await;
            assert!(
                Arc::ptr_eq(&store, &reopened),
                "each open rebuilt the client"
            );
        }
        let other = second.run(open()).await;
        let ordinary = open().await;
        assert!(!Arc::ptr_eq(&store, &other));
        assert!(!Arc::ptr_eq(&store, &ordinary));
        let path = Path::from("data/batch");
        store
            .inner
            .put_opts(
                &path,
                bytes::Bytes::from_static(b"payload").into(),
                Default::default(),
            )
            .await
            .unwrap();
        assert_eq!(second.completed_steps(), 0);
        // A cached handle keeps its original attribution even while another
        // execution is current. Repeated real reads must not keep it alive.
        for _ in 0..24 {
            let before = first.completed_steps();
            let body = second
                .run(async {
                    store
                        .inner
                        .get_opts(&path, Default::default())
                        .await
                        .unwrap()
                        .bytes()
                        .await
                        .unwrap()
                })
                .await;
            assert_eq!(body.as_ref(), b"payload");
            assert!(first.completed_steps() > before);
            assert_eq!(second.completed_steps(), 0);
        }
    }

    #[tokio::test]
    async fn preparation_store_cache_registers_shared_local_tracker_once() {
        use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry};
        let dir = tempfile::tempdir().unwrap();
        let scope = Arc::new(MergeWriteScope {
            observe_file_io: true,
            ..Default::default()
        });
        scope
            .run(async {
                let registry = ObjectStoreRegistry::default();
                let store = registry
                    .get_store(
                        format!("file://{}", dir.path().display()).parse().unwrap(),
                        &ObjectStoreParams {
                            object_store_wrapper: preparation_store_wrapper(),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
                assert!(store.is_local());
                register_local_preparation_io(&store);
                store
                    .io_tracker()
                    .record_read("read", Path::from("data/file"), 7, None);
                // Reopening must retain the initial baseline, not duplicate it or
                // erase completed work by taking a new baseline on every open.
                for _ in 0..24 {
                    register_local_preparation_io(&store);
                }
                assert_eq!(scope.local_io.lock().unwrap().len(), 1);
                assert_eq!(scope.completed_steps(), 7);
                store
                    .io_tracker()
                    .record_read("read", Path::from("data/file"), 3, None);
                assert_eq!(scope.completed_steps(), 10);
            })
            .await;
    }

    #[tokio::test]
    async fn legacy_observation_retains_uncertainty_without_detaching_storage() {
        struct LocalWrite(Arc<AtomicBool>);
        impl Drop for LocalWrite {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        for scope in [
            MergeWriteScope::for_legacy_observation(),
            MergeWriteScope::for_legacy_file_observation(),
        ] {
            let writing = Arc::new(AtomicBool::new(false));
            let flag = writing.clone();
            let (entered, began) = oneshot::channel();
            let run_scope = scope.clone();
            let caller = tokio::spawn(async move {
                run_scope
                    .run(shield(async move {
                        flag.store(true, Ordering::SeqCst);
                        let _local_write = LocalWrite(flag);
                        entered.send(()).unwrap();
                        std::future::pending::<Result<()>>().await
                    }))
                    .await
            });
            began.await.unwrap();
            assert_eq!(scope.write_status().active_commits, 1);
            caller.abort();
            caller.await.unwrap_err();
            assert!(
                !writing.load(Ordering::SeqCst),
                "diagnostics must not detach storage work"
            );
            assert_eq!(scope.write_status().active_commits, 0);
            assert!(
                scope.write_status().uncertain_commit,
                "remote completion is still unknown"
            );
            assert_eq!(scope.completed_steps(), 0);
        }
    }

    #[tokio::test]
    async fn phase_changes_and_observation_do_not_count_as_completed_work() {
        let scope = MergeWriteScope::new();
        scope
            .run(async {
                let first = work_stage("memory_wait");
                let second = work_stage("memory_wait");
                assert_eq!(scope.work().active_stages["memory_wait"], 2);
                assert_eq!(scope.completed_steps(), 0);
                drop(first);
                assert_eq!(scope.work().active_stages["memory_wait"], 1);
                drop(second);
                assert!(scope.work().active_stages.is_empty());
                assert_eq!(scope.completed_steps(), 0);
                wal_batch_read(3, 1024);
                assert_eq!(scope.completed_steps(), 1);
                assert_eq!(scope.work().wal_rows_read, 3);
            })
            .await;
    }

    #[test]
    fn recovery_rejects_unsafe_and_external_commit_handlers() {
        for uri in [
            "custom:table",
            "custom://table",
            "https://host/table",
            "s3+ddb://bucket/table",
        ] {
            assert!(!supports_version_fencing(uri), "{uri}");
        }
        for uri in [
            "/tmp/table:with-colon",
            "relative/table",
            "file:/tmp/table",
            "S3://bucket/table",
            "az://container/table",
        ] {
            assert!(supports_version_fencing(uri), "{uri}");
        }
    }

    #[tokio::test]
    async fn conditional_conflicts_and_errors_do_not_count_as_progress() {
        let scope = MergeWriteScope::new();
        for committed in [false, false, true] {
            scope
                .run(shield_observed(
                    async move { Ok::<_, Error>(committed) },
                    |result| matches!(result, Ok(true)),
                ))
                .await
                .unwrap();
        }
        assert_eq!(scope.completed_steps(), 1);
    }

    #[tokio::test]
    async fn confirmed_fence_can_abort_a_permanently_stalled_leaf() {
        struct Writing(Arc<AtomicBool>);
        impl Drop for Writing {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let scope = MergeWriteScope::new();
        let writing = Arc::new(AtomicBool::new(false));
        let flag = writing.clone();
        let (entered, began) = oneshot::channel();
        let worker_scope = scope.clone();
        let caller = tokio::spawn(async move {
            worker_scope
                .run(shield(async move {
                    flag.store(true, Ordering::SeqCst);
                    let _writing = Writing(flag);
                    entered.send(()).unwrap();
                    std::future::pending::<Result<()>>().await
                }))
                .await
        });
        began.await.unwrap();
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        assert!(writing.load(Ordering::SeqCst));
        // Real base/shard barriers are exercised by the rollout fault tests.
        // Once that proof exists, a stalled local I/O future must release too.
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            scope.abort_fenced_leaves(),
        )
        .await
        .unwrap();
        assert!(!writing.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn failed_or_panicked_leaf_commit_needs_reconciliation() {
        let scope = MergeWriteScope::new();
        let result = scope
            .run(shield(async {
                Err::<(), _>(Error::io("lost storage response"))
            }))
            .await;
        assert!(result.is_err());
        scope.drain().await;
        assert!(
            scope.has_uncertain_commit(),
            "an error is not proof that remote storage did not commit"
        );
        let panicked = MergeWriteScope::new();
        let result = panicked
            .run(shield(async {
                panic!("commit panic");
                #[allow(unreachable_code)]
                Ok::<(), Error>(())
            }))
            .await;
        assert!(result.is_err());
        panicked.drain().await;
        assert!(panicked.has_uncertain_commit());
    }

    #[tokio::test]
    async fn dropping_merge_waits_for_the_surviving_manifest_write() {
        let scope = MergeWriteScope::new();
        let (entered, began) = oneshot::channel();
        let (release, blocked) = oneshot::channel();
        let committed = Arc::new(AtomicBool::new(false));
        let flag = committed.clone();
        let task_scope = scope.clone();
        let merge = tokio::spawn(async move {
            task_scope
                .run(shield(async move {
                    entered.send(()).unwrap();
                    blocked.await.unwrap();
                    flag.store(true, Ordering::SeqCst);
                    Ok::<_, Error>(())
                }))
                .await
        });
        began.await.unwrap();
        merge.abort();
        assert!(merge.await.unwrap_err().is_cancelled());
        assert_eq!(scope.write_status().active_commits, 1);
        assert!(!scope.write_status().uncertain_commit);
        let draining_scope = scope.clone();
        let drain = tokio::spawn(async move { draining_scope.drain().await });
        tokio::task::yield_now().await;
        assert!(
            !drain.is_finished(),
            "cannot hand off while storage write survives"
        );
        release.send(()).unwrap();
        drain.await.unwrap();
        assert_eq!(scope.write_status().active_commits, 0);
        assert!(scope.write_status().admission_closed);
        assert!(committed.load(Ordering::SeqCst));
        assert!(!scope.has_uncertain_commit());
        assert!(
            scope
                .run(shield(async { Ok::<_, Error>(()) }))
                .await
                .is_err(),
            "cancelled scope cannot issue another commit"
        );
    }
}
