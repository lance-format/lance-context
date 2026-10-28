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
    publication_pause: AtomicBool,
    publication_entered: Notify,
    publication_release: Notify,
    fail: AtomicBool,
    pause: AtomicBool,
    entered: Notify,
    release: Notify,
    completed: AtomicUsize,
    read_enabled: AtomicBool,
    read_fail_first: AtomicBool,
    read_fail_path: std::sync::Mutex<Option<String>>,
    read_calls: AtomicUsize,
    read_active: AtomicUsize,
    read_entered: Notify,
    read_release: Notify,
    read_failed: Notify,
    read_verified: AtomicBool,
    abort_pause: AtomicBool,
    abort_entered: Notify,
    abort_release: Notify,
    abort_completed: AtomicUsize,
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
        self.gate.wait_publication(path).await;
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
        if self.gate.read_enabled.load(Ordering::SeqCst) && base_data(path) {
            let call = self.gate.read_calls.fetch_add(1, Ordering::SeqCst);
            let fail = self.gate.read_fail_first.load(Ordering::SeqCst);
            let inject_failure = {
                let mut failed_path = self.gate.read_fail_path.lock().unwrap();
                if fail && call == 0 {
                    *failed_path = Some(path.to_string());
                }
                fail && failed_path.as_deref() == Some(path.as_ref())
            };
            if inject_failure {
                // Put the error first in ordered delivery, but wait until a
                // later real fragment-open child is outstanding before failing.
                // Keep failing this file through CloudObjectReader retries.
                if call == 0 {
                    self.gate.read_entered.notified().await;
                }
                self.gate.read_failed.notify_one();
                return Err(Error::Generic {
                    store: "staging-scan-test",
                    source: std::io::Error::other("injected fragment read failure").into(),
                });
            }
            if call == usize::from(fail) {
                struct ActiveRead(Arc<WriteGate>);
                impl Drop for ActiveRead {
                    fn drop(&mut self) {
                        self.0.read_active.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                self.gate.read_active.fetch_add(1, Ordering::SeqCst);
                let _active = ActiveRead(self.gate.clone());
                self.gate.read_entered.notify_waiters();
                // notify_one also retains a permit for the first error task if
                // it has not started polling the notification yet.
                self.gate.read_entered.notify_one();
                self.gate.read_release.notified().await;
            }
        }
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
        self.gate.wait_publication(to).await;
        self.inner.copy_opts(from, to, opts).await
    }
    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> Result<()> {
        self.gate.wait_publication(to).await;
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
    async fn wait_publication(&self, path: &Path) {
        if path.as_ref().contains("/_versions/")
            && path.as_ref().contains(".manifest")
            && self.publication_pause.swap(false, Ordering::SeqCst)
        {
            self.publication_entered.notify_one();
            self.publication_release.notified().await;
        }
    }
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
        if self.gate.abort_pause.swap(false, Ordering::SeqCst) {
            self.gate.abort_entered.notify_one();
            self.gate.abort_release.notified().await;
        }
        let result = self.inner.abort().await;
        self.gate.abort_completed.fetch_add(1, Ordering::SeqCst);
        result
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
    for (panic, blocking_io) in [(false, false), (true, false), (false, true), (true, true)] {
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
                    let child = move || {
                        child_entered.notify_one();
                        let mut ready = gate.0.lock().unwrap();
                        while !*ready {
                            ready = gate.1.wait(ready).unwrap();
                        }
                        Ok::<_, lance::Error>(payload.len())
                    };
                    if blocking_io {
                        drop(lance_core::utils::tokio::spawn_blocking_scoped(child));
                    } else {
                        drop(lance_core::utils::tokio::spawn_cpu(child));
                    }
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn staged_wal_real_scanner_error_and_cancellation_join_fragment_children() {
    use crate::{merge_write_scope::run_owned_preparation, MergeMemoryBudget};
    use std::time::Duration;
    for fail in [false, true] {
        let dir = tempfile::TempDir::new().unwrap();
        let uri = format!("file-object-store://{}", dir.path().display());
        let payload = vec![37u8; 1024 * 1024];
        let batch = RecordBatch::try_from_iter([(
            "payload",
            Arc::new(LargeBinaryArray::from_vec(vec![payload.as_slice(); 4])) as ArrayRef,
        )])
        .unwrap();
        let schema = batch.schema();
        let dataset = lance::Dataset::write(
            arrow_array::RecordBatchIterator::new(vec![Ok(batch)], schema),
            &uri,
            Some(lance::dataset::WriteParams {
                max_rows_per_file: 1,
                max_rows_per_group: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 4);
        drop(dataset);
        let gate = Arc::new(WriteGate::default());
        gate.read_enabled.store(true, Ordering::SeqCst);
        gate.read_fail_first.store(fail, Ordering::SeqCst);
        let scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
            Arc::new(NoCommit),
            Arc::new(Wrapper(gate.clone())),
        );
        let budget = MergeMemoryBudget::new(8 * 1024 * 1024);
        let reservation = budget.reserve(budget.limit()).await;
        let hold = reservation.retain();
        let (runner_scope, runner_gate) = (scope.clone(), gate.clone());
        let mut caller = tokio::spawn(async move {
            runner_scope
                .run(run_owned_preparation(hold, async move {
                    let dataset = StorageBase::load_with_options(&uri, None, None).await?;
                    let mut scanner = dataset.scan();
                    scanner
                        .fragment_readahead(4)
                        .batch_readahead(4)
                        .batch_size(1);
                    let batches: Vec<_> = scanner.try_into_stream().await?.try_collect().await?;
                    let mut rows = 0;
                    for batch in batches {
                        let array = batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<LargeBinaryArray>()
                            .unwrap();
                        for row in 0..array.len() {
                            assert_eq!(array.value(row), payload);
                            rows += 1;
                        }
                    }
                    assert_eq!(rows, 4);
                    runner_gate.read_verified.store(true, Ordering::SeqCst);
                    drop(reservation);
                    Ok::<_, lance::Error>(rows)
                }))
                .await
        });
        // Observe actual IO entry without stealing the error task's Notify permit.
        tokio::time::timeout(Duration::from_secs(20), async {
            while gate.read_calls.load(Ordering::SeqCst) < if fail { 2 } else { 1 } {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        if fail {
            tokio::time::timeout(Duration::from_secs(20), gate.read_failed.notified())
                .await
                .unwrap();
            let early = tokio::time::timeout(Duration::from_millis(50), &mut caller).await;
            match early {
                Ok(result) => {
                    // Abort-on-drop scanner implementations may have already
                    // ended the other child. Completion must then mean no live IO.
                    assert_eq!(gate.read_active.load(Ordering::SeqCst), 0);
                    assert!(result
                        .unwrap()
                        .unwrap()
                        .unwrap_err()
                        .to_string()
                        .contains("injected fragment read failure"));
                }
                Err(_) => {
                    assert_eq!(budget.reserved(), budget.limit());
                    gate.read_release.notify_one();
                    let result = tokio::time::timeout(Duration::from_secs(20), caller)
                        .await
                        .unwrap()
                        .unwrap();
                    assert!(result
                        .unwrap()
                        .unwrap_err()
                        .to_string()
                        .contains("injected fragment read failure"));
                }
            }
        } else {
            caller.abort();
            assert!(caller.await.unwrap_err().is_cancelled());
            assert_eq!(budget.reserved(), budget.limit());
            assert!(
                tokio::time::timeout(Duration::from_millis(30), scope.drain())
                    .await
                    .is_err()
            );
            gate.read_release.notify_one();
        }
        tokio::time::timeout(Duration::from_secs(20), scope.drain())
            .await
            .unwrap();
        assert_eq!(gate.read_active.load(Ordering::SeqCst), 0);
        assert_eq!(budget.reserved(), 0);
        assert_eq!(gate.read_verified.load(Ordering::SeqCst), !fail);
    }
}

#[tokio::test]
async fn staged_wal_scoped_async_children_propagate_cpu_and_keep_request_groups_separate() {
    use lance_core::utils::tokio::{join_cpu_tasks, retain_task_scope, spawn_scoped};
    use std::time::Duration;
    let (release, waiting) = tokio::sync::oneshot::channel();
    let (token_send, token_receive) = tokio::sync::oneshot::channel();
    let first = tokio::spawn(join_cpu_tasks(async move {
        // A shared batching service keeps only this request's token; it need
        // not run inside or inherit the requester's async task local.
        token_send.send(retain_task_scope().unwrap()).unwrap();
        drop(spawn_scoped(async move {
            let _ = waiting.await;
        }));
    }));
    let token = token_receive.await.unwrap();
    let independent = join_cpu_tasks(async {
        spawn_scoped(async {
            lance_core::utils::tokio::spawn_cpu(|| Ok::<_, std::io::Error>(42))
                .await
                .unwrap()
        })
        .await
        .unwrap()
    });
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), independent)
            .await
            .unwrap(),
        42
    );
    release.send(()).unwrap();
    assert!(
        !first.is_finished(),
        "the shared-service request still holds its own group"
    );
    drop(token);
    tokio::time::timeout(Duration::from_secs(2), first)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn staged_wal_unpolled_async_child_drops_payload_before_releasing_group() {
    use lance_core::utils::tokio::{join_cpu_tasks, scope_task};
    use std::{
        sync::{Condvar, Mutex},
        time::Duration,
    };
    struct Payload {
        gate: Arc<(Mutex<bool>, Condvar)>,
        entered: Arc<Notify>,
    }
    impl std::future::Future for Payload {
        type Output = ();
        fn poll(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<()> {
            panic!("queued child must never be polled in this test");
        }
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            self.entered.notify_one();
            let mut ready = self.gate.0.lock().unwrap();
            while !*ready {
                ready = self.gate.1.wait(ready).unwrap();
            }
        }
    }
    struct Release(Arc<(Mutex<bool>, Condvar)>);
    impl Drop for Release {
        fn drop(&mut self) {
            *self.0 .0.lock().unwrap() = true;
            self.0 .1.notify_all();
        }
    }
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let release = Release(gate.clone());
    let entered = Arc::new(Notify::new());
    let payload = Payload {
        gate,
        entered: entered.clone(),
    };
    let (send, receive) = tokio::sync::oneshot::channel();
    let mut group = tokio::spawn(join_cpu_tasks(async move {
        assert!(send.send(scope_task(payload)).is_ok());
    }));
    let queued_child = receive.await.unwrap();
    let cleanup = tokio::task::spawn_blocking(move || drop(queued_child));
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(30), &mut group)
            .await
            .is_err(),
        "group completion must wait for actual payload destruction"
    );
    drop(release);
    cleanup.await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), group)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn staged_wal_discarded_scheduler_request_keeps_budget_until_physical_io_ends() {
    use crate::{merge_write_scope::run_owned_preparation, MergeMemoryBudget};
    use futures::future::BoxFuture;
    use lance_io::{
        scheduler::{ScanScheduler, SchedulerConfig},
        traits::Reader,
    };
    use std::{ops::Range, time::Duration};
    #[derive(Debug)]
    struct GatedReader {
        gate: Arc<WriteGate>,
        path: Path,
        fail: bool,
    }
    impl lance_core::deepsize::DeepSizeOf for GatedReader {
        fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
            0
        }
    }
    impl Reader for GatedReader {
        fn path(&self) -> &Path {
            &self.path
        }
        fn block_size(&self) -> usize {
            1
        }
        fn io_parallelism(&self) -> usize {
            2
        }
        fn size(&self) -> BoxFuture<'_, Result<usize>> {
            Box::pin(async { Ok(1024 * 1024) })
        }
        fn get_all(&self) -> BoxFuture<'_, Result<bytes::Bytes>> {
            self.get_range(0..1024 * 1024)
        }
        fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, Result<bytes::Bytes>> {
            let gate = self.gate.clone();
            let fail = self.fail;
            Box::pin(async move {
                struct Active(Arc<WriteGate>);
                impl Drop for Active {
                    fn drop(&mut self) {
                        self.0.read_active.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                gate.read_calls.fetch_add(1, Ordering::SeqCst);
                gate.read_active.fetch_add(1, Ordering::SeqCst);
                let _active = Active(gate.clone());
                // Keep a real buffer resident while the physical read is pending.
                let bytes = bytes::Bytes::from(vec![7; range.len()]);
                gate.read_entered.notify_one();
                gate.read_release.notified().await;
                if fail {
                    return Err(Error::Generic {
                        store: "scoped-io-test",
                        source: std::io::Error::other("injected physical read error").into(),
                    });
                }
                Ok(bytes)
            })
        }
    }
    for lite in [false, true] {
        for physical_error in [false, true] {
            let gate = Arc::new(WriteGate::default());
            let scope = MergeWriteScope::with_authorizer(Arc::new(NoCommit));
            let budget = MergeMemoryBudget::new(2 * 1024 * 1024);
            let reservation = budget.reserve(budget.limit()).await;
            let hold = reservation.retain();
            let (runner_scope, runner_gate) = (scope.clone(), gate.clone());
            let mut caller = tokio::spawn(async move {
                runner_scope
                    .run(run_owned_preparation(hold, async move {
                        let scheduler = ScanScheduler::new(
                            Arc::new(lance_io::object_store::ObjectStore::memory()),
                            SchedulerConfig {
                                io_buffer_size_bytes: 4 * 1024 * 1024,
                                use_lite_scheduler: Some(lite),
                            },
                        );
                        let file = scheduler.open_reader(Arc::new(GatedReader {
                            gate: runner_gate.clone(),
                            path: Path::from("payload"),
                            fail: physical_error,
                        }));
                        {
                            let request =
                                file.submit_request(std::iter::once(0..1024 * 1024).collect(), 0);
                            tokio::pin!(request);
                            tokio::select! {
                                _ = &mut request => panic!("physical read must remain blocked"),
                                _ = runner_gate.read_entered.notified() => {}
                            }
                            // Discard the receiving future after the physical read
                            // has started, exactly like a sibling scan/encoding error.
                        }
                        drop((file, scheduler, reservation));
                        Err::<(), _>(lance::Error::io("outer preparation failed"))
                    }))
                    .await
            });
            tokio::time::timeout(Duration::from_secs(20), async {
                while gate.read_calls.load(Ordering::SeqCst) == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            if lite {
                // Lite reads run in the receiving future: discarding it must
                // destroy the physical read and its token without a phantom wait.
                let result = tokio::time::timeout(Duration::from_secs(2), &mut caller)
                    .await
                    .unwrap()
                    .unwrap();
                assert!(result
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("outer preparation failed"));
                assert_eq!(gate.read_active.load(Ordering::SeqCst), 0);
            } else {
                assert!(tokio::time::timeout(Duration::from_millis(30), &mut caller)
                    .await
                    .is_err());
                assert_eq!(budget.reserved(), budget.limit());
                assert_eq!(gate.read_active.load(Ordering::SeqCst), 1);
                gate.read_release.notify_one();
                let result = tokio::time::timeout(Duration::from_secs(20), caller)
                    .await
                    .unwrap()
                    .unwrap();
                assert!(result
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("outer preparation failed"));
            }
            scope.drain().await;
            assert_eq!(gate.read_active.load(Ordering::SeqCst), 0);
            assert_eq!(budget.reserved(), 0);
        }
    }
}

#[tokio::test]
async fn staged_wal_real_multipart_writer_error_joins_abort_cleanup_before_budget_release() {
    use crate::{merge_write_scope::run_owned_preparation, MergeMemoryBudget};
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;
    let part_size = std::env::var("LANCE_INITIAL_UPLOAD_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5 * 1024 * 1024)
        .max(5 * 1024 * 1024);
    assert!(
        part_size <= 16 * 1024 * 1024,
        "this bounded test requires LANCE_INITIAL_UPLOAD_SIZE <= 16 MiB"
    );
    let gate = Arc::new(WriteGate::default());
    gate.pause.store(true, Ordering::SeqCst);
    gate.abort_pause.store(true, Ordering::SeqCst);
    let scope = MergeWriteScope::with_authorizer(Arc::new(NoCommit));
    let budget = MergeMemoryBudget::new(4 * part_size);
    let reservation = budget.reserve(budget.limit()).await;
    let hold = reservation.retain();
    let (runner_scope, runner_gate) = (scope.clone(), gate.clone());
    let mut caller = tokio::spawn(async move {
        runner_scope
            .run(run_owned_preparation(hold, async move {
                let mut store = lance_io::object_store::ObjectStore::memory();
                store.inner = Arc::new(WriteStore {
                    inner: store.inner.clone(),
                    gate: runner_gate.clone(),
                });
                let mut writer = lance_io::object_writer::ObjectWriter::new(
                    &store,
                    &Path::from("staging/data/payload.lance"),
                )
                .await?;
                let payload = vec![19u8; 2 * part_size + 1];
                {
                    let write = writer.write_all(&payload);
                    tokio::pin!(write);
                    tokio::select! {
                        result = &mut write => result.map_err(|e| lance::Error::io(e.to_string()))?,
                        _ = runner_gate.entered.notified() => {}
                    }
                }
                // write_all can finish buffering before the upload child is polled.
                while runner_gate.pause.load(Ordering::SeqCst) {
                    tokio::task::yield_now().await;
                }
                // Model a sibling preparation failure after real multipart work
                // started. ObjectWriter's actual Drop aborts its JoinSet and spawns
                // multipart cleanup; that cleanup is deliberately held below.
                drop((writer, payload, reservation));
                Err::<(), _>(lance::Error::io(
                    "injected preparation failure after multipart start",
                ))
            }))
            .await
    });
    tokio::time::timeout(Duration::from_secs(20), gate.abort_entered.notified())
        .await
        .unwrap();
    assert!(tokio::time::timeout(Duration::from_millis(30), &mut caller)
        .await
        .is_err());
    assert_eq!(budget.reserved(), budget.limit());
    assert_eq!(gate.abort_completed.load(Ordering::SeqCst), 0);
    gate.abort_release.notify_one();
    let result = tokio::time::timeout(Duration::from_secs(20), caller)
        .await
        .unwrap()
        .unwrap();
    assert!(result
        .unwrap()
        .unwrap_err()
        .to_string()
        .contains("injected preparation failure"));
    scope.drain().await;
    assert_eq!(gate.abort_completed.load(Ordering::SeqCst), 1);
    assert_eq!(budget.reserved(), 0);
    assert!(!scope.has_uncertain_commit());
}

mod worker_prestage {
    use super::*;
    use std::{future::Future, pin::Pin, time::Duration};
    use tokio::sync::RwLock;

    #[derive(Debug)]
    struct Allow;
    impl CommitAuthorizer for Allow {
        fn authorize<'a>(
            &'a self,
            _: &'a str,
            _: u64,
        ) -> Pin<Box<dyn Future<Output = lance::Result<()>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }
    }

    async fn fixture() -> (
        tempfile::TempDir,
        String,
        RolloutStoreOptions,
        Arc<crate::MergeMemoryBudget>,
    ) {
        let dir = tempfile::TempDir::new().unwrap();
        let uri = format!("file-object-store://{}", dir.path().display());
        let budget = crate::MergeMemoryBudget::new(8 * 1024 * 1024);
        let options = RolloutStoreOptions {
            shard_id: Some("ordinary-worker".into()),
            merge_after_generations: Some(0),
            merge_max_bytes: Some(1024 * 1024),
            merge_budget: Some(budget.clone()),
            ..Default::default()
        };
        let mut writer = RolloutStore::open_with_options(&uri, options.clone())
            .await
            .unwrap();
        writer
            .add(&[super::super::tests::artifact_record("seed", &[1; 16])])
            .await
            .unwrap();
        writer.cleanup_own_shard().await.unwrap();
        writer
            .add(&[super::super::tests::artifact_record(
                "pending",
                &vec![7; 2 * 1024 * 1024],
            )])
            .await
            .unwrap();
        writer.flush().await.unwrap();
        writer.close().await.unwrap();
        (dir, uri, options, budget)
    }

    async fn read(store: &Arc<RwLock<RolloutStore>>) {
        assert_eq!(
            store
                .read()
                .await
                .get_by_id("seed")
                .await
                .unwrap()
                .unwrap()
                .id,
            "seed"
        );
    }

    // Both arms block actual encoded-data IO. The publication gate is a real
    // manifest write, independent of the authorizer (which always succeeds).
    #[tokio::test]
    async fn worker_prestage_encoding_allows_reader_but_publication_still_blocks() {
        for stage in [false, true] {
            let (_dir, uri, options, budget) = fixture().await;
            let (other_dir, other_uri, other_options, _) = fixture().await;
            let other = Arc::new(RwLock::new(
                RolloutStore::open_existing_with_options(&other_uri, other_options)
                    .await
                    .unwrap(),
            ));
            let gate = Arc::new(WriteGate::default());
            let scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
                Arc::new(Allow),
                Arc::new(Wrapper(gate.clone())),
            );
            let store = scope
                .run(async {
                    Arc::new(RwLock::new(
                        RolloutStore::open_existing_with_options(&uri, options)
                            .await
                            .unwrap(),
                    ))
                })
                .await;
            gate.pause.store(true, Ordering::SeqCst);
            gate.publication_pause.store(true, Ordering::SeqCst);
            let (task_scope, task_store) = (scope.clone(), store.clone());
            let merge = tokio::spawn(async move {
                task_scope
                    .run(async {
                        let input = {
                            let guard = task_store.read().await;
                            let input = guard.prepare_recovered_cleanup_merge().await.unwrap();
                            if stage {
                                guard.stage_owned_prepared_merge(input).await.unwrap()
                            } else {
                                input
                            }
                        }
                        .unwrap();
                        task_store
                            .write()
                            .await
                            .commit_prepared_merge(&input.0, &input.1, input.2)
                            .await
                            .unwrap()
                    })
                    .await
            });
            tokio::time::timeout(Duration::from_secs(20), gate.entered.notified())
                .await
                .unwrap();
            let mut reader = Box::pin(read(&store));
            if stage {
                tokio::time::timeout(Duration::from_secs(5), &mut reader)
                    .await
                    .expect("reader must progress during owned encoding");
            } else {
                assert!(
                    futures::poll!(&mut reader).is_pending(),
                    "old encoding holds the exclusive guard"
                );
            }
            tokio::time::timeout(Duration::from_secs(5), read(&other))
                .await
                .unwrap();
            gate.release.notify_one();
            tokio::time::timeout(Duration::from_secs(20), gate.publication_entered.notified())
                .await
                .unwrap();
            let mut publication_reader = Box::pin(read(&store));
            assert!(
                futures::poll!(&mut publication_reader).is_pending(),
                "publication still holds the exclusive guard"
            );
            tokio::time::timeout(Duration::from_secs(5), read(&other))
                .await
                .unwrap();
            gate.publication_release.notify_one();
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(20), merge)
                    .await
                    .unwrap()
                    .unwrap(),
                1
            );
            if !stage {
                tokio::time::timeout(Duration::from_secs(5), &mut reader)
                    .await
                    .unwrap();
            }
            tokio::time::timeout(Duration::from_secs(5), &mut publication_reader)
                .await
                .unwrap();
            scope.drain().await;
            assert_eq!(budget.reserved(), 0);
            assert_eq!(scope.work().base_commits, 1);
            assert_eq!(scope.work().generations_reclaimed, 1);
            assert_eq!(
                store.read().await.pending_wal_generations().await.unwrap(),
                0
            );
            assert_eq!(
                lance::Dataset::open(&uri)
                    .await
                    .unwrap()
                    .count_rows(None)
                    .await
                    .unwrap(),
                2
            );
            drop(other_dir);
        }
    }

    #[tokio::test]
    async fn worker_prestage_cancel_retains_encoding_until_join_then_recovers_once() {
        let (_dir, uri, options, budget) = fixture().await;
        let gate = Arc::new(WriteGate::default());
        let scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
            Arc::new(Allow),
            Arc::new(Wrapper(gate.clone())),
        );
        let store = scope
            .run(async {
                Arc::new(RwLock::new(
                    RolloutStore::open_existing_with_options(&uri, options.clone())
                        .await
                        .unwrap(),
                ))
            })
            .await;
        let before = store.read().await.version();
        gate.pause.store(true, Ordering::SeqCst);
        let (task_scope, task_store) = (scope.clone(), store.clone());
        let caller = tokio::spawn(async move {
            task_scope
                .run(async {
                    let guard = task_store.read().await;
                    let input = guard.prepare_recovered_cleanup_merge().await.unwrap();
                    guard.stage_owned_prepared_merge(input).await.unwrap()
                })
                .await
        });
        tokio::time::timeout(Duration::from_secs(20), gate.entered.notified())
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), read(&store))
            .await
            .unwrap();
        caller.abort();
        assert!(matches!(caller.await, Err(error) if error.is_cancelled())); // actual caller joined
        assert!(budget.reserved() >= 2 * 1024 * 1024);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), scope.drain())
                .await
                .is_err()
        );
        assert!(budget.try_reserve(budget.limit()).is_none());
        gate.release.notify_one();
        tokio::time::timeout(Duration::from_secs(20), scope.drain())
            .await
            .unwrap();
        assert_eq!(budget.reserved(), 0);
        assert_eq!(scope.work().base_commits, 0);
        assert_eq!(scope.work().generations_reclaimed, 0);
        assert_eq!(store.read().await.version(), before);
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            1
        );
        assert_eq!(
            lance::Dataset::open(&uri)
                .await
                .unwrap()
                .count_rows(None)
                .await
                .unwrap(),
            1
        );
        let recovery = MergeWriteScope::with_pinned_authorizer(Arc::new(Allow));
        recovery
            .run(async {
                let mut fresh = RolloutStore::open_existing_with_options(&uri, options)
                    .await
                    .unwrap();
                let input = fresh.prepare_recovered_cleanup_merge().await.unwrap();
                let (ms, manifest, prepared) = fresh
                    .stage_owned_prepared_merge(input)
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    fresh
                        .commit_prepared_merge(&ms, &manifest, prepared)
                        .await
                        .unwrap(),
                    1
                );
                assert_eq!(fresh.cleanup_own_shard().await.unwrap(), 0);
                assert_eq!(
                    fresh.get_blob("pending").await.unwrap().unwrap(),
                    vec![7; 2 * 1024 * 1024]
                );
                assert_eq!(fresh.pending_wal_generations().await.unwrap(), 0);
            })
            .await;
        recovery.drain().await;
        assert_eq!(recovery.work().base_commits, 1);
        assert_eq!(recovery.work().generations_reclaimed, 1);
        assert_eq!(budget.reserved(), 0);
        assert_eq!(
            lance::Dataset::open(&uri)
                .await
                .unwrap()
                .count_rows(None)
                .await
                .unwrap(),
            2
        );
    }

    #[tokio::test]
    async fn worker_prestage_does_not_transfer_preparation_authority() {
        let (_dir, uri, options, budget) = fixture().await;
        let store = RolloutStore::open_existing_with_options(&uri, options)
            .await
            .unwrap();
        let first = MergeWriteScope::with_authorizer(Arc::new(Allow));
        let second = MergeWriteScope::with_authorizer(Arc::new(Allow));
        let input = first
            .run(store.prepare_recovered_cleanup_merge())
            .await
            .unwrap();
        let err = second
            .run(store.stage_owned_prepared_merge(input))
            .await
            .err()
            .unwrap();
        assert!(err.to_string().contains("owner changed or closed"));
        assert_eq!(budget.reserved(), 0);
        let input = first
            .run(store.prepare_recovered_cleanup_merge())
            .await
            .unwrap();
        first.drain().await;
        assert!(first
            .run(store.stage_owned_prepared_merge(input))
            .await
            .is_err());
        let input = store.prepare_recovered_cleanup_merge().await.unwrap();
        let result = second
            .run(store.stage_owned_prepared_merge(input))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            second.work().encoded_files,
            0,
            "unowned input must retain legacy behavior"
        );
        drop(result);
        second.drain().await;
        assert_eq!(budget.reserved(), 0);
        assert_eq!(store.pending_wal_generations().await.unwrap(), 1);
    }
}
