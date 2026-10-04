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
        Arc, Mutex,
    },
};
use tokio::sync::{oneshot, Notify};

tokio::task_local! { static CURRENT: Arc<MergeWriteScope>; }
tokio::task_local! { static EXPECTED_BASE_VERSION: u64; }

/// Pin the next publication without replacing a dataset's captured owner guard.
pub(crate) async fn at_base_version<F: Future>(version: u64, work: F) -> F::Output {
    EXPECTED_BASE_VERSION.scope(version, work).await
}

#[derive(Debug, Default)]
struct Progress {
    closed: bool,
    active: usize,
    uncertain: bool,
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
    completed_steps: AtomicU64,
    progress: Mutex<Progress>,
    changed: Notify,
    authorizer: Option<Arc<dyn CommitAuthorizer>>,
    pin_opened_handles: bool,
    leaves: Mutex<Vec<tokio::task::AbortHandle>>,
}

impl MergeWriteScope {
    pub fn completed_steps(&self) -> u64 {
        self.completed_steps.load(Ordering::Relaxed)
    }
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
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

/// Capture progress explicitly for Lance callbacks which can run on another task.
pub(crate) fn write_progress() -> lance::dataset::write::WriteProgressFn {
    let scope = CURRENT.try_with(Arc::clone).ok();
    let previous = std::sync::Mutex::new((0u64, 0u64, 0u32));
    lance::dataset::write::WriteProgressFn::new(move |stats| {
        let current = (stats.bytes_written, stats.rows_written, stats.files_written);
        let mut old = previous.lock().unwrap();
        if current.0 > old.0 || current.1 > old.1 || current.2 > old.2 {
            *old = current;
            if let Some(scope) = &scope {
                scope.completed_steps.fetch_add(1, Ordering::Relaxed);
            }
        }
    })
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
            .read_latest()
            .await?
            .ok_or_else(|| Error::io("Shard manifest not found"))?;
        if next.writer_epoch != epoch {
            return Err(Error::io("merge writer epoch changed before drain"));
        }
        next.version = next
            .version
            .checked_add(1)
            .ok_or_else(|| Error::io("manifest version overflow"))?;
        next.flushed_generations
            .retain(|g| !generations.contains(&g.generation));
        authorize(&resource, next.version).await?;
        let writer = store.clone();
        let scope = CURRENT.try_with(Arc::clone).ok();
        let (next, written) = shield_observed(
            async move {
                let written = writer.write(&next).await;
                if written
                    .as_ref()
                    .is_err_and(|e| !e.to_string().contains("already exists"))
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
            Ok(_) => return Ok(next),
            Err(error) if error.to_string().contains("already exists") => continue,
            Err(error) => return Err(error),
        }
    }
    Err(Error::io("shard drain exceeded conditional-write retries"))
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
                .read_latest()
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
            match store.write(&next).await {
                Ok(_) => {}
                Err(error) if error.to_string().contains("already exists") => {}
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
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

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
        let draining_scope = scope.clone();
        let drain = tokio::spawn(async move { draining_scope.drain().await });
        tokio::task::yield_now().await;
        assert!(
            !drain.is_finished(),
            "cannot hand off while storage write survives"
        );
        release.send(()).unwrap();
        drain.await.unwrap();
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
