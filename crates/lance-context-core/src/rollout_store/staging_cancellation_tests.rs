//! Cancel the caller while a real Lance immutable data write is blocked.
use super::*;
use crate::merge_write_scope::{CommitAuthorizer, MergeWriteScope};
use async_trait::async_trait;
use futures::stream::BoxStream;
use lance_io::object_store::WrappingObjectStore;
use object_store::{path::Path, *};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::Notify;

#[derive(Debug, Default)]
struct WriteGate {
    fail: AtomicBool,
    pause: AtomicBool,
    entered: Notify,
    release: Notify,
    completed: AtomicUsize,
}
#[derive(Debug)]
struct Wrapper(Arc<WriteGate>);
impl WrappingObjectStore for Wrapper {
    fn wrap_paginated(
        &self,
        _: &str,
        store: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        Some(store)
    }
    fn wrap(&self, _: &str, store: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(WriteStore {
            inner: store,
            gate: self.0.clone(),
        })
    }
}
#[derive(Debug)]
struct WriteStore {
    inner: Arc<dyn ObjectStore>,
    gate: Arc<WriteGate>,
}
impl std::fmt::Display for WriteStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WriteFault({})", self.inner)
    }
}
#[async_trait]
impl ObjectStore for WriteStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        if base_data(path) {
            self.gate.wait().await?;
        }
        self.inner.put_opts(path, payload, opts).await
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        let inner = self.inner.put_multipart_opts(path, opts).await?;
        if base_data(path) {
            Ok(Box::new(WriteUpload {
                inner,
                gate: self.gate.clone(),
            }))
        } else {
            Ok(inner)
        }
    }
    async fn get_opts(&self, path: &Path, opts: GetOptions) -> Result<GetResult> {
        self.inner.get_opts(path, opts).await
    }
    fn delete_stream(
        &self,
        paths: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(paths)
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }
    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> Result<()> {
        self.inner.copy_opts(from, to, opts).await
    }
    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> Result<()> {
        self.inner.rename_opts(from, to, opts).await
    }
}
#[derive(Debug)]
struct NoCommit;
impl CommitAuthorizer for NoCommit {
    fn authorize<'a>(
        &'a self,
        _: &'a str,
        _: u64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = lance::Result<()>> + Send + 'a>> {
        Box::pin(async { Err(lance::Error::io("validation must never commit")) })
    }
}

fn base_data(path: &Path) -> bool {
    path.as_ref().contains("/data/") && !path.as_ref().contains("/_mem_wal/")
}
impl WriteGate {
    async fn wait(&self) -> Result<()> {
        if self.pause.swap(false, Ordering::SeqCst) {
            self.entered.notify_one();
            self.release.notified().await;
            self.completed.fetch_add(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                return Err(Error::Generic {
                    store: "staging-cancel-test",
                    source: std::io::Error::other("injected encoded data failure").into(),
                });
            }
        }
        Ok(())
    }
}
#[derive(Debug)]
struct WriteUpload {
    inner: Box<dyn MultipartUpload>,
    gate: Arc<WriteGate>,
}
#[async_trait]
impl MultipartUpload for WriteUpload {
    fn put_part(&mut self, payload: PutPayload) -> UploadPart {
        let part = self.inner.put_part(payload);
        let gate = self.gate.clone();
        Box::pin(async move {
            gate.wait().await?;
            part.await
        })
    }
    async fn complete(&mut self) -> Result<PutResult> {
        self.inner.complete().await
    }
    async fn abort(&mut self) -> Result<()> {
        self.inner.abort().await
    }
}

#[tokio::test]
async fn staged_wal_cancellation_retains_real_encoder_budget_until_scope_drain() {
    use crate::{merge_write_scope::run_owned_preparation, MergeMemoryBudget};
    use futures::future::BoxFuture;
    use std::time::Duration;
    for fail in [false, true] {
        let dir = tempfile::TempDir::new().unwrap();
        let uri = format!("file-object-store://{}", dir.path().display());
        let budget = MergeMemoryBudget::new(4 * 1024 * 1024);
        let options = RolloutStoreOptions {
            shard_id: Some("cancelled-stage".into()),
            merge_after_generations: Some(0),
            merge_max_bytes: Some(1024 * 1024),
            merge_budget: Some(budget.clone()),
            ..Default::default()
        };
        let mut writer = RolloutStore::open_with_options(&uri, options.clone())
            .await
            .unwrap();
        writer
            .add(&[super::tests::artifact_record(
                "pending",
                &vec![7; 2 * 1024 * 1024],
            )])
            .await
            .unwrap();
        writer.flush().await.unwrap();
        writer.close().await.unwrap();
        let before = writer.version();
        let gate = Arc::new(WriteGate::default());
        gate.pause.store(true, Ordering::SeqCst);
        gate.fail.store(fail, Ordering::SeqCst);
        let scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
            Arc::new(NoCommit),
            Arc::new(Wrapper(gate.clone())),
        );
        let runner_scope = scope.clone();
        let runner_uri = uri.clone();
        let reserve_budget = budget.clone();
        let caller = tokio::spawn(async move {
            let reservation = reserve_budget.reserve(1024 * 1024).await;
            runner_scope
                .run(run_owned_preparation(reservation.retain(), async move {
                    let mut store =
                        RolloutStore::open_existing_with_options(&runner_uri, options).await?;
                    let (manifests, manifest, mut prepared) =
                        store.prepare_merge_reserved(reservation).await?.unwrap();
                    store.stage_prepared_merge(&mut prepared).await?;
                    Ok::<_, lance::Error>(Box::pin(async move {
                        store
                            .commit_prepared_merge(&manifests, &manifest, prepared)
                            .await
                    })
                        as BoxFuture<'static, lance::Result<usize>>)
                }))
                .await
        });
        tokio::time::timeout(Duration::from_secs(20), gate.entered.notified())
            .await
            .unwrap();
        caller.abort();
        assert!(matches!(caller.await, Err(error) if error.is_cancelled()));
        assert!(budget.reserved() >= 2 * 1024 * 1024);
        assert!(budget.try_reserve(budget.limit()).is_none());
        assert!(
            tokio::time::timeout(Duration::from_millis(30), scope.drain())
                .await
                .is_err()
        );
        // A version fence must not abort the payload child and release memory.
        assert!(
            tokio::time::timeout(Duration::from_millis(30), scope.abort_fenced_leaves())
                .await
                .is_err()
        );
        assert!(budget.reserved() >= 2 * 1024 * 1024);
        gate.release.notify_one();
        tokio::time::timeout(Duration::from_secs(20), scope.drain())
            .await
            .unwrap();
        assert_eq!(budget.reserved(), 0);
        assert_eq!(gate.completed.load(Ordering::SeqCst), 1);
        assert_eq!(writer.version(), before);
        assert_eq!(writer.pending_wal_generations().await.unwrap(), 1);
        assert_eq!(
            lance::Dataset::open(&uri)
                .await
                .unwrap()
                .count_rows(None)
                .await
                .unwrap(),
            0
        );
        assert!(
            !scope.has_uncertain_commit(),
            "immutable output is not an uncertain base commit"
        );
    }
}

#[tokio::test]
async fn staged_wal_cancellation_joins_actual_lance_cpu_child_before_releasing_budget() {
    use crate::{merge_write_scope::run_owned_preparation, MergeMemoryBudget};
    use std::{
        sync::{Condvar, Mutex},
        time::Duration,
    };
    let budget = MergeMemoryBudget::new(4 * 1024 * 1024);
    let scope = MergeWriteScope::with_authorizer(Arc::new(NoCommit));
    struct ReleaseOnDrop(Arc<(Mutex<bool>, Condvar)>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            *self.0 .0.lock().unwrap() = true;
            self.0 .1.notify_all();
        }
    }
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let _release_on_drop = ReleaseOnDrop(gate.clone());
    let entered = Arc::new(Notify::new());
    let finished = Arc::new(AtomicBool::new(false));
    let (runner_scope, runner_budget) = (scope.clone(), budget.clone());
    let (runner_gate, runner_entered, runner_finished) =
        (gate.clone(), entered.clone(), finished.clone());
    let caller = tokio::spawn(async move {
        let reservation = runner_budget.reserve(2 * 1024 * 1024).await;
        runner_scope
            .run(run_owned_preparation(reservation.retain(), async move {
                let payload = vec![3u8; 2 * 1024 * 1024];
                // Exactly the primitive used by Lance's page encoders: dropping
                // its JoinHandle detaches this blocking child. The owned scope
                // must keep its parent await/reservation alive instead.
                let size = lance_core::utils::tokio::spawn_cpu(move || {
                    runner_entered.notify_one();
                    let (lock, cond) = &*runner_gate;
                    let mut released = lock.lock().unwrap();
                    while !*released {
                        released = cond.wait(released).unwrap();
                    }
                    runner_finished.store(true, Ordering::SeqCst);
                    Ok::<_, lance::Error>(payload.len())
                })
                .await?;
                assert_eq!(size, 2 * 1024 * 1024);
                drop(reservation);
                Ok::<_, lance::Error>(())
            }))
            .await
    });
    tokio::time::timeout(Duration::from_secs(20), entered.notified())
        .await
        .unwrap();
    caller.abort();
    assert!(matches!(caller.await, Err(error) if error.is_cancelled()));
    assert_eq!(budget.reserved(), 2 * 1024 * 1024);
    assert!(!finished.load(Ordering::SeqCst));
    assert!(
        tokio::time::timeout(Duration::from_millis(30), scope.drain())
            .await
            .is_err()
    );
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    tokio::time::timeout(Duration::from_secs(20), scope.drain())
        .await
        .unwrap();
    assert!(finished.load(Ordering::SeqCst));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn staged_wal_preparation_rejects_unowned_or_closed_scope_before_polling_payload() {
    use crate::merge_write_scope::run_owned_preparation;
    let polled = Arc::new(AtomicBool::new(false));
    for scope in [
        MergeWriteScope::new(),
        MergeWriteScope::for_legacy_file_observation(),
        MergeWriteScope::with_authorizer(Arc::new(NoCommit)),
    ] {
        scope.drain().await;
        let observed = polled.clone();
        let reservation = crate::MergeMemoryBudget::new(1).reserve(1).await;
        let result = scope
            .run(run_owned_preparation(reservation.retain(), async move {
                observed.store(true, Ordering::SeqCst);
            }))
            .await;
        assert!(result.is_err());
        assert!(!polled.load(Ordering::SeqCst));
    }
}

/// The writer can discard a CPU handle on an unrelated encoding/IO error. This
/// is different from cancelling a caller whose writer still awaits the handle.
#[tokio::test]
async fn staged_wal_internal_error_or_panic_joins_discarded_cpu_and_retains_budget() {
    use crate::{merge_write_scope::run_owned_preparation, MergeMemoryBudget};
    use std::{
        sync::{Condvar, Mutex},
        time::Duration,
    };
    for panic in [false, true] {
        let budget = MergeMemoryBudget::new(4 * 1024 * 1024);
        let reservation = budget.reserve(1024 * 1024).await;
        let hold = reservation.retain();
        let scope = MergeWriteScope::with_authorizer(Arc::new(NoCommit));
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        struct Release(Arc<(Mutex<bool>, Condvar)>);
        impl Drop for Release {
            fn drop(&mut self) {
                *self.0 .0.lock().unwrap() = true;
                self.0 .1.notify_all();
            }
        }
        let release = Release(gate.clone());
        let entered = Arc::new(Notify::new());
        let failed = Arc::new(Notify::new());
        let (child_entered, child_failed, runner_scope) =
            (entered.clone(), failed.clone(), scope.clone());
        let caller = tokio::spawn(async move {
            runner_scope
                .run(run_owned_preparation(hold, async move {
                    let mut reservation = reservation;
                    assert!(reservation.try_grow_to(3 * 1024 * 1024));
                    let payload = vec![3u8; 3 * 1024 * 1024];
                    // No await: exactly the detached-handle path inside a failed writer.
                    drop(lance_core::utils::tokio::spawn_cpu(move || {
                        child_entered.notify_one();
                        let mut ready = gate.0.lock().unwrap();
                        while !*ready {
                            ready = gate.1.wait(ready).unwrap();
                        }
                        Ok::<_, lance::Error>(payload.len())
                    }));
                    drop(reservation);
                    child_failed.notify_one();
                    if panic {
                        panic!("injected writer panic after CPU submission");
                    }
                    Err::<(), _>(lance::Error::io(
                        "injected writer error after CPU submission",
                    ))
                }))
                .await
        });
        tokio::time::timeout(Duration::from_secs(20), entered.notified())
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(20), failed.notified())
            .await
            .unwrap();
        assert_eq!(budget.reserved(), 3 * 1024 * 1024);
        assert!(budget.try_reserve(2 * 1024 * 1024).is_none());
        assert!(!caller.is_finished());
        assert!(
            tokio::time::timeout(Duration::from_millis(30), scope.drain())
                .await
                .is_err()
        );
        drop(release);
        let result = tokio::time::timeout(Duration::from_secs(20), caller)
            .await
            .unwrap()
            .unwrap();
        if panic {
            assert!(result.unwrap_err().to_string().contains("panicked"));
        } else {
            assert!(result
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("injected writer error"));
        }
        scope.drain().await;
        assert_eq!(budget.reserved(), 0);
        drop(budget.try_reserve(4 * 1024 * 1024).unwrap());
        assert_eq!(budget.reserved(), 0);
    }
}

#[tokio::test]
async fn staged_wal_cpu_scopes_join_nested_work_and_isolate_concurrent_operations() {
    use lance_core::utils::tokio::{join_cpu_tasks, spawn_cpu};
    use std::{
        sync::{Condvar, Mutex},
        time::Duration,
    };
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    struct Release(Arc<(Mutex<bool>, Condvar)>);
    impl Drop for Release {
        fn drop(&mut self) {
            *self.0 .0.lock().unwrap() = true;
            self.0 .1.notify_all();
        }
    }
    let release = Release(gate.clone());
    let entered = Arc::new(Notify::new());
    let child_entered = entered.clone();
    let outer = tokio::spawn(join_cpu_tasks(async move {
        // Parent returns without awaiting the child. Even a single CPU thread
        // can run this test: only the last nested closure is held by the gate.
        spawn_cpu(move || {
            drop(spawn_cpu(move || {
                child_entered.notify_one();
                let mut ready = gate.0.lock().unwrap();
                while !*ready {
                    ready = gate.1.wait(ready).unwrap();
                }
                Ok::<_, lance::Error>(())
            }));
            Ok::<_, lance::Error>(())
        })
        .await
        .unwrap();
    }));
    tokio::time::timeout(Duration::from_secs(20), entered.notified())
        .await
        .unwrap();
    assert!(
        !outer.is_finished(),
        "the parent has exited but its nested child is live"
    );
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), join_cpu_tasks(async { 42 }))
            .await
            .unwrap(),
        42
    );
    drop(release);
    tokio::time::timeout(Duration::from_secs(20), outer)
        .await
        .unwrap()
        .unwrap();
    // A panic from an awaited CPU handle must still carry the original payload.
    let panic = tokio::spawn(join_cpu_tasks(async {
        spawn_cpu(|| -> std::result::Result<(), std::io::Error> { panic!("original CPU payload") })
            .await
    }))
    .await
    .unwrap_err()
    .into_panic();
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"original CPU payload"));
}
