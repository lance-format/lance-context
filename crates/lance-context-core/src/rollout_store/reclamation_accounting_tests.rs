//! Reclamation counts follow successful shard writes, including cancelled callers.
use super::*;
use crate::merge_write_scope::{drain_generations, CommitAuthorizer, MergeWriteScope};
use async_trait::async_trait;
use futures::stream::BoxStream;
use lance::dataset::mem_wal::ShardManifestStore;
use object_store::{path::Path, *};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Notify;

#[derive(Debug)]
struct Allow;
impl CommitAuthorizer for Allow {
    fn authorize<'a>(
        &'a self,
        _: &'a str,
        _: u64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = lance::Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

fn scopes() -> [Arc<MergeWriteScope>; 3] {
    [
        MergeWriteScope::with_authorizer(Arc::new(Allow)),
        MergeWriteScope::for_legacy_observation(),
        MergeWriteScope::for_legacy_file_observation(),
    ]
}

#[derive(Debug)]
struct PeerDrainsFirst {
    peer: ShardManifestStore,
    once: AtomicBool,
    generation: u64,
    epoch: u64,
}
impl CommitAuthorizer for PeerDrainsFirst {
    fn authorize<'a>(
        &'a self,
        _: &'a str,
        _: u64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = lance::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if self.once.swap(false, Ordering::SeqCst) {
                self.peer
                    .commit_update(self.epoch, |current| {
                        let mut next = current.clone();
                        next.version += 1;
                        next.sstables.retain(|g| g.generation != self.generation);
                        next
                    })
                    .await?;
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn retry_counts_only_generations_removed_by_its_successful_write() {
    for count in [1, 2] {
        let dir = tempfile::tempdir().unwrap();
        let mut store = RolloutStore::open(dir.path().to_str().unwrap())
            .await
            .unwrap();
        for id in 0..count {
            store
                .add(&[tests::assistant_record(&id.to_string())])
                .await
                .unwrap();
            store.flush().await.unwrap();
        }
        let object_store = store.base.dataset.object_store(None).await.unwrap();
        let path = &store.base.dataset.branch_location().path;
        let shard = store.base.write_shard;
        let writer = ShardManifestStore::new(object_store.clone(), path, shard, 16);
        let before = writer.refresh_latest().await.unwrap().unwrap();
        assert_eq!(before.sstables.len(), count);
        let peer = ShardManifestStore::new(object_store, path, shard, 16);
        let scope = MergeWriteScope::with_authorizer(Arc::new(PeerDrainsFirst {
            peer,
            once: AtomicBool::new(true),
            generation: before.sstables[0].generation,
            epoch: before.writer_epoch,
        }));
        let after = scope
            .run(drain_generations(
                writer,
                before.writer_epoch,
                before.sstables.iter().map(|g| g.generation).collect(),
            ))
            .await
            .unwrap();
        assert!(after.sstables.is_empty());
        assert_eq!(scope.work().generations_reclaimed, (count - 1) as u64);
        scope.drain().await;
        store.close().await.unwrap();
    }
}

#[tokio::test]
async fn general_merge_counts_reclaimed_once_in_owned_and_legacy_scopes() {
    for scope in scopes() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = RolloutStore::open(dir.path().to_str().unwrap())
            .await
            .unwrap();
        for generation in 0..3 {
            let mut row = tests::assistant_record("same-key");
            row.content = Some(generation.to_string());
            store.add(&[row]).await.unwrap();
            store.flush().await.unwrap();
        }
        assert_eq!(scope.work().generations_reclaimed, 0);
        scope.run(store.cleanup_own_shard()).await.unwrap();
        assert_eq!(scope.work().generations_reclaimed, 3);
        scope.run(store.cleanup_own_shard()).await.unwrap();
        assert_eq!(
            scope.work().generations_reclaimed,
            3,
            "empty replay double-counted"
        );
        scope.drain().await;
        let rows = store.list(None, None).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].content.as_deref(), Some("2"));
        store.close().await.unwrap();
    }
}

#[derive(Debug, Default)]
struct DrainGate {
    pause: AtomicBool,
    fail_response: AtomicBool,
    entered: Notify,
    release: Notify,
}

#[derive(Debug)]
struct DelayedRename {
    inner: Arc<dyn ObjectStore>,
    gate: Arc<DrainGate>,
}
impl std::fmt::Display for DelayedRename {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DelayedDrain({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for DelayedRename {
    async fn put_opts(&self, p: &Path, b: PutPayload, o: PutOptions) -> Result<PutResult> {
        self.inner.put_opts(p, b, o).await
    }
    async fn put_multipart_opts(
        &self,
        p: &Path,
        o: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(p, o).await
    }
    async fn get_opts(&self, p: &Path, o: GetOptions) -> Result<GetResult> {
        self.inner.get_opts(p, o).await
    }
    fn delete_stream(
        &self,
        p: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(p)
    }
    fn list(&self, p: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(p)
    }
    async fn list_with_delimiter(&self, p: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(p).await
    }
    async fn copy_opts(&self, a: &Path, b: &Path, o: CopyOptions) -> Result<()> {
        self.inner.copy_opts(a, b, o).await
    }
    async fn rename_opts(&self, a: &Path, b: &Path, o: RenameOptions) -> Result<()> {
        self.inner.rename_opts(a, b, o).await?;
        if b.as_ref().contains("/manifest/")
            && self.gate.fail_response.swap(false, Ordering::SeqCst)
        {
            return Err(Error::Generic {
                store: "drain-accounting-test",
                source: std::io::Error::other("lost committed manifest response").into(),
            });
        }
        if b.as_ref().contains("/manifest/") && self.gate.pause.swap(false, Ordering::SeqCst) {
            self.gate.entered.notify_one();
            self.gate.release.notified().await;
        }
        Ok(())
    }
}

#[tokio::test]
async fn uncertain_drain_does_not_claim_confirmed_reclamation() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RolloutStore::open(dir.path().to_str().unwrap())
        .await
        .unwrap();
    store
        .add(&[tests::assistant_record("fixture")])
        .await
        .unwrap();
    store.flush().await.unwrap();
    let original = store.base.dataset.object_store(None).await.unwrap();
    let path = &store.base.dataset.branch_location().path;
    let shard = store.base.write_shard;
    let reader = ShardManifestStore::new(original.clone(), path, shard, 16);
    let before = reader.refresh_latest().await.unwrap().unwrap();
    let gate = Arc::new(DrainGate::default());
    let mut wrapped = (*original).clone();
    wrapped.inner = Arc::new(DelayedRename {
        inner: wrapped.inner,
        gate: gate.clone(),
    });
    let writer = ShardManifestStore::new(Arc::new(wrapped), path, shard, 16);
    let scope = MergeWriteScope::with_authorizer(Arc::new(Allow));
    gate.fail_response.store(true, Ordering::SeqCst);
    assert!(scope
        .run(drain_generations(
            writer,
            before.writer_epoch,
            before.sstables.iter().map(|g| g.generation).collect()
        ))
        .await
        .is_err());
    scope.drain().await;
    assert!(scope.has_uncertain_commit());
    assert_eq!(scope.work().generations_reclaimed, 0);
    // Independent readback sees the write, but an ambiguous result cannot claim
    // confirmed progress or authorize an unfenced handoff.
    assert!(reader
        .refresh_latest()
        .await
        .unwrap()
        .unwrap()
        .sstables
        .is_empty());
    store.close().await.unwrap();
}

#[tokio::test]
async fn cancelled_drain_counts_actual_removals_in_its_original_scope() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RolloutStore::open(dir.path().to_str().unwrap())
        .await
        .unwrap();
    store
        .add(&[tests::assistant_record("fixture")])
        .await
        .unwrap();
    store.flush().await.unwrap();
    let original = store.base.dataset.object_store(None).await.unwrap();
    let path = &store.base.dataset.branch_location().path;
    let shard = store.base.write_shard;
    let reader = ShardManifestStore::new(original.clone(), path, shard, 16);
    let before = reader.refresh_latest().await.unwrap().unwrap();
    let generations: std::collections::HashSet<_> =
        before.sstables.iter().map(|s| s.generation).collect();
    assert_eq!(generations.len(), 1);
    let gate = Arc::new(DrainGate::default());
    let mut wrapped = (*original).clone();
    wrapped.inner = Arc::new(DelayedRename {
        inner: wrapped.inner,
        gate: gate.clone(),
    });
    let writer = ShardManifestStore::new(Arc::new(wrapped), path, shard, 16);
    let scope = MergeWriteScope::with_authorizer(Arc::new(Allow));
    let unrelated = MergeWriteScope::with_authorizer(Arc::new(Allow));
    gate.pause.store(true, Ordering::SeqCst);
    let run_scope = scope.clone();
    let ids = generations.clone();
    let caller = tokio::spawn(async move {
        run_scope
            .run(drain_generations(writer, before.writer_epoch, ids))
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), gate.entered.notified())
        .await
        .unwrap();
    caller.abort();
    assert!(caller.await.unwrap_err().is_cancelled());
    assert_eq!(
        scope.work().generations_reclaimed,
        0,
        "response still unconfirmed"
    );
    assert_eq!(scope.write_status().active_commits, 1);
    unrelated
        .run(async {
            gate.release.notify_one();
            tokio::time::timeout(std::time::Duration::from_secs(5), scope.drain())
                .await
                .unwrap();
        })
        .await;
    assert_eq!(scope.work().generations_reclaimed, 1);
    assert_eq!(unrelated.work().generations_reclaimed, 0);
    assert!(!scope.has_uncertain_commit());
    assert!(reader
        .refresh_latest()
        .await
        .unwrap()
        .unwrap()
        .sstables
        .is_empty());
    // A replay of already drained IDs is not new work, even though its manifest write succeeds.
    let replay = ShardManifestStore::new(
        store.base.dataset.object_store(None).await.unwrap(),
        path,
        shard,
        16,
    );
    unrelated
        .run(drain_generations(replay, before.writer_epoch, generations))
        .await
        .unwrap();
    assert_eq!(unrelated.work().generations_reclaimed, 0);
    unrelated.drain().await;
    store.close().await.unwrap();
}
