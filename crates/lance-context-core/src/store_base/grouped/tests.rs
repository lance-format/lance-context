use super::*;
use arrow_array::{BinaryArray, StringArray};
use arrow_schema::{DataType, Field};

const MIB: usize = 1024 * 1024;

#[derive(Debug)]
struct Allow;
impl crate::merge_write_scope::CommitAuthorizer for Allow {
    fn authorize<'a>(
        &'a self,
        _resource: &'a str,
        _version: u64,
    ) -> futures::future::BoxFuture<'a, LanceResult<()>> {
        Box::pin(async { Ok(()) })
    }
}

async fn commit(
    store: &mut StorageBase,
    shards: Vec<PreparedShardMerge>,
    cap: usize,
) -> LanceResult<usize> {
    let scope = crate::merge_write_scope::MergeWriteScope::with_authorizer(Arc::new(Allow));
    let result = scope.run(store.commit_prepared_group(shards, cap)).await;
    scope.drain().await;
    result
}

fn batch(rows: &[(&str, &[u8])]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("payload", DataType::Binary, false),
        ])),
        vec![
            Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.0))),
            Arc::new(BinaryArray::from_iter_values(rows.iter().map(|r| r.1))),
        ],
    )
    .unwrap()
}

async fn open(uri: &str, shard: &str, budget: Arc<MergeMemoryBudget>) -> StorageBase {
    StorageBase::open(
        uri,
        StorageBaseOptions {
            storage_options: None,
            shard_id: Some(shard.into()),
            merge_after_generations: Some(0),
            merge_max_generations: Some(64),
            merge_max_bytes: Some(MIB),
            key_index_type: crate::KeyIndexType::Btree,
            pending_generations_warn: Some(0),
            pending_generations_max: Some(0),
            merge_budget: Some(budget),
            session: Some(StorageBase::build_session(MIB, MIB)),
            schema: batch(&[]).schema(),
            key_column: "id".into(),
            latest_schema: None,
            seal_on_put: false,
        },
        true,
    )
    .await
    .unwrap()
}

async fn put(store: &StorageBase, rows: &[(&str, &[u8])]) {
    store.put(vec![batch(rows)]).await.unwrap();
    store.flush().await.unwrap();
}

async fn prepare(store: &StorageBase, budget: &Arc<MergeMemoryBudget>) -> PreparedShardMerge {
    store
        .prepare_grouped_merge_reserved(budget.reserve(MIB).await)
        .await
        .unwrap()
        .unwrap()
}

async fn values(uri: &str) -> HashMap<String, Vec<u8>> {
    let dataset = Dataset::open(uri).await.unwrap();
    let rows = dataset.scan().try_into_batch().await.unwrap();
    let keys = rows
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let payloads = rows
        .column(1)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    let mut result = HashMap::new();
    for i in 0..rows.num_rows() {
        assert!(result
            .insert(keys.value(i).into(), payloads.value(i).to_vec())
            .is_none());
    }
    result
}

#[tokio::test]
async fn ordered_group_updates_repeated_keys_and_publishes_all_watermarks_once() {
    for reversed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
        let budget = MergeMemoryBudget::new(16 * MIB);
        let mut seed = open(&uri, "seed", budget.clone()).await;
        let untouched = vec![7; 64 * 1024];
        put(&seed, &[("shared", b"base"), ("untouched", &untouched)]).await;
        seed.cleanup_own_shard().await.unwrap();
        let mut first = open(&uri, "first", budget.clone()).await;
        let mut second = open(&uri, "second", budget.clone()).await;
        put(&first, &[("shared", b"first-old")]).await;
        put(&first, &[("shared", b"first-new"), ("only-first", b"one")]).await;
        put(&second, &[("shared", b"second"), ("only-second", b"two")]).await;
        let a = prepare(&first, &budget).await;
        let b = prepare(&second, &budget).await;
        let marks: Vec<_> = [&a, &b]
            .iter()
            .map(|s| {
                (
                    s.manifest.shard_id,
                    *s.prepared.merged_generations.iter().max().unwrap(),
                )
            })
            .collect();
        let before = Dataset::open(&uri).await.unwrap().version().version;
        let group = if reversed { vec![b, a] } else { vec![a, b] };
        assert_eq!(commit(&mut first, group, MIB).await.unwrap(), 3);
        let after = Dataset::open(&uri).await.unwrap();
        assert_eq!(after.version().version, before + 1);
        let coverage = crate::rollout_append::watermarks(&after).await.unwrap();
        for (shard, high) in marks {
            assert_eq!(coverage.get(&shard), Some(&high));
        }
        assert_eq!(
            values(&uri).await,
            HashMap::from([
                (
                    "shared".into(),
                    if reversed {
                        b"first-new".to_vec()
                    } else {
                        b"second".to_vec()
                    }
                ),
                ("untouched".into(), untouched),
                ("only-first".into(), b"one".to_vec()),
                ("only-second".into(), b"two".to_vec()),
            ])
        );
        assert_eq!(first.pending_wal_generations().await.unwrap(), 0);
        assert_eq!(budget.reserved(), 0);
        first.close().await.unwrap();
        second.close().await.unwrap();
        seed.close().await.unwrap();
    }
}

#[tokio::test]
async fn concurrent_flush_is_retained_and_live_writers_keep_their_epochs() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(8 * MIB);
    let mut first = open(&uri, "first", budget.clone()).await;
    let mut second = open(&uri, "second", budget.clone()).await;
    put(&first, &[("one", b"before")]).await;
    put(&second, &[("two", b"before")]).await;
    let a = prepare(&first, &budget).await;
    let b = prepare(&second, &budget).await;
    let epoch = a.manifest.writer_epoch;
    put(&first, &[("one", b"after")]).await;
    assert_eq!(commit(&mut first, vec![a, b], MIB).await.unwrap(), 2);
    let current = first.own_shard_watermark().await.unwrap().unwrap();
    assert_eq!(current.writer_epoch, epoch);
    assert_eq!(current.pending_generations, 1);
    assert_eq!(values(&uri).await["one"], b"before");
    first.cleanup_own_shard().await.unwrap();
    assert_eq!(values(&uri).await["one"], b"after");
    put(&second, &[("still-live", b"yes")]).await;
    second.cleanup_own_shard().await.unwrap();
    assert_eq!(values(&uri).await["still-live"], b"yes");
    first.close().await.unwrap();
    second.close().await.unwrap();
}

#[tokio::test]
async fn group_rejects_another_table_before_publication_and_releases_memory() {
    let dir = tempfile::tempdir().unwrap();
    let budget = MergeMemoryBudget::new(8 * MIB);
    let left = dir.path().join("left.lance").to_str().unwrap().to_owned();
    let right = dir.path().join("right.lance").to_str().unwrap().to_owned();
    let mut a = open(&left, "first", budget.clone()).await;
    let mut b = open(&right, "second", budget.clone()).await;
    put(&a, &[("a", b"left")]).await;
    put(&b, &[("b", b"right")]).await;
    let group = vec![prepare(&a, &budget).await, prepare(&b, &budget).await];
    let before = Dataset::open(&left).await.unwrap().version().version;
    assert!(commit(&mut a, group, MIB)
        .await
        .unwrap_err()
        .to_string()
        .contains("another table"));
    assert_eq!(
        Dataset::open(&left).await.unwrap().version().version,
        before
    );
    assert_eq!(a.pending_wal_generations().await.unwrap(), 1);
    assert_eq!(b.pending_wal_generations().await.unwrap(), 1);
    assert_eq!(budget.reserved(), 0);
    a.close().await.unwrap();
    b.close().await.unwrap();
}

#[tokio::test]
async fn insufficient_scratch_budget_commits_serially_without_waiting() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(3 * MIB);
    let mut first = open(&uri, "first", budget.clone()).await;
    let mut second = open(&uri, "second", budget.clone()).await;
    put(&first, &[("same", b"first")]).await;
    put(&second, &[("same", b"second")]).await;
    let group = vec![
        prepare(&first, &budget).await,
        prepare(&second, &budget).await,
    ];
    let held = budget.try_reserve(MIB).unwrap();
    let before = Dataset::open(&uri).await.unwrap().version().version;
    let reclaimed = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        commit(&mut first, group, MIB),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(reclaimed, 2);
    assert_eq!(budget.reserved(), MIB);
    assert_eq!(
        Dataset::open(&uri).await.unwrap().version().version,
        before + 2
    );
    assert_eq!(values(&uri).await["same"], b"second");
    assert_eq!(first.pending_wal_generations().await.unwrap(), 0);
    drop(held);
    first.close().await.unwrap();
    second.close().await.unwrap();
}

#[derive(Debug)]
struct RejectDrain(String);
impl crate::merge_write_scope::CommitAuthorizer for RejectDrain {
    fn authorize<'a>(
        &'a self,
        resource: &'a str,
        _version: u64,
    ) -> futures::future::BoxFuture<'a, LanceResult<()>> {
        Box::pin(async move {
            if resource == self.0 {
                Err(LanceError::io("injected drain rejection"))
            } else {
                Ok(())
            }
        })
    }
}

#[tokio::test]
async fn interrupted_drain_reconciles_all_watermarks_without_replaying_over_newer_data() {
    // Fail before either drain, then after one successful drain. Both cases
    // must preserve the atomic publication and reconcile only manifest state.
    for reject_second in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
        let budget = MergeMemoryBudget::new(8 * MIB);
        let mut first = open(&uri, "first", budget.clone()).await;
        let mut second = open(&uri, "second", budget.clone()).await;
        put(&first, &[("shared", b"first"), ("only-first", b"one")]).await;
        put(&second, &[("shared", b"second"), ("only-second", b"two")]).await;
        let a = prepare(&first, &budget).await;
        let b = prepare(&second, &budget).await;
        let rejected = if reject_second {
            b.manifest.shard_id
        } else {
            a.manifest.shard_id
        };
        let before = Dataset::open(&uri).await.unwrap().version().version;
        let scope = crate::merge_write_scope::MergeWriteScope::with_authorizer(Arc::new(
            RejectDrain(format!("shard:{rejected}")),
        ));
        let error = scope
            .run(first.commit_prepared_group(vec![a, b], MIB))
            .await
            .unwrap_err();
        scope.drain().await;
        assert!(error.to_string().contains("injected drain rejection"));
        assert_eq!(
            Dataset::open(&uri).await.unwrap().version().version,
            before + 1
        );
        assert_eq!(
            first
                .own_shard_watermark()
                .await
                .unwrap()
                .unwrap()
                .pending_generations,
            if reject_second { 0 } else { 1 }
        );
        assert_eq!(
            second
                .own_shard_watermark()
                .await
                .unwrap()
                .unwrap()
                .pending_generations,
            1
        );
        assert_eq!(values(&uri).await["shared"], b"second");
        assert_eq!(budget.reserved(), 0);

        // A different shard publishes a newer value before recovery. Replaying
        // either old preparation would regress this value despite deduplication.
        let mut later = open(&uri, "later", budget.clone()).await;
        put(&later, &[("shared", b"newest")]).await;
        later.cleanup_own_shard().await.unwrap();
        let published = Dataset::open(&uri).await.unwrap().version().version;
        let recovery = crate::merge_write_scope::MergeWriteScope::with_authorizer(Arc::new(Allow));
        for store in [&first, &second] {
            let result = recovery
                .run(store.prepare_grouped_merge_reserved(budget.reserve(MIB).await))
                .await
                .unwrap();
            assert!(
                result.is_none(),
                "covered generations must not be prepared again"
            );
        }
        recovery.drain().await;
        assert_eq!(
            Dataset::open(&uri).await.unwrap().version().version,
            published
        );
        assert_eq!(
            values(&uri).await,
            HashMap::from([
                ("shared".into(), b"newest".to_vec()),
                ("only-first".into(), b"one".to_vec()),
                ("only-second".into(), b"two".to_vec()),
            ])
        );
        assert_eq!(first.pending_wal_generations().await.unwrap(), 0);
        assert_eq!(budget.reserved(), 0);
        first.close().await.unwrap();
        second.close().await.unwrap();
        later.close().await.unwrap();
    }
}

#[tokio::test]
async fn aggregate_byte_cap_falls_back_without_rereading_or_losing_order() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(8 * MIB);
    let mut first = open(&uri, "first", budget.clone()).await;
    let mut second = open(&uri, "second", budget.clone()).await;
    put(&first, &[("shared", b"first")]).await;
    put(&second, &[("shared", b"second")]).await;
    let a = prepare(&first, &budget).await;
    let b = prepare(&second, &budget).await;
    let cap = a.decoded_bytes().max(b.decoded_bytes());
    assert!(a.decoded_bytes() + b.decoded_bytes() > cap);
    let before = Dataset::open(&uri).await.unwrap().version().version;
    let scope = crate::merge_write_scope::MergeWriteScope::with_authorizer(Arc::new(Allow));
    assert_eq!(
        scope
            .run(first.commit_prepared_group(vec![a, b], cap))
            .await
            .unwrap(),
        2
    );
    scope.drain().await;
    assert_eq!(
        scope.work().wal_batches_read,
        0,
        "fallback must reuse preparations"
    );
    assert_eq!(
        Dataset::open(&uri).await.unwrap().version().version,
        before + 2
    );
    assert_eq!(values(&uri).await["shared"], b"second");
    assert_eq!(first.pending_wal_generations().await.unwrap(), 0);
    assert_eq!(budget.reserved(), 0);
    first.close().await.unwrap();
    second.close().await.unwrap();
}

#[tokio::test]
async fn changed_writer_epoch_rejects_group_before_any_publication() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(8 * MIB);
    let mut first = open(&uri, "first", budget.clone()).await;
    let second = open(&uri, "second", budget.clone()).await;
    put(&first, &[("one", b"first")]).await;
    put(&second, &[("two", b"second")]).await;
    let a = prepare(&first, &budget).await;
    let b = prepare(&second, &budget).await;
    let (epoch, _) = b.manifest_store.claim_epoch(0).await.unwrap();
    assert!(epoch > b.manifest.writer_epoch);
    let before = Dataset::open(&uri).await.unwrap().version().version;
    assert!(commit(&mut first, vec![a, b], MIB)
        .await
        .unwrap_err()
        .to_string()
        .contains("epoch or membership changed"));
    assert_eq!(Dataset::open(&uri).await.unwrap().version().version, before);
    assert_eq!(first.pending_wal_generations().await.unwrap(), 2);
    assert_eq!(budget.reserved(), 0);
    first.close().await.unwrap();
    // The test deliberately fenced the second writer. Dropping its empty
    // memtable cannot produce any publication under the stale epoch.
    drop(second);
}

#[derive(Debug)]
struct GateFirstBase {
    once: std::sync::atomic::AtomicBool,
    entered: tokio::sync::Notify,
    release: tokio::sync::Semaphore,
}
impl crate::merge_write_scope::CommitAuthorizer for GateFirstBase {
    fn authorize<'a>(
        &'a self,
        resource: &'a str,
        _version: u64,
    ) -> futures::future::BoxFuture<'a, LanceResult<()>> {
        Box::pin(async move {
            if resource == "base" && self.once.swap(false, std::sync::atomic::Ordering::SeqCst) {
                self.entered.notify_one();
                self.release.acquire().await.unwrap().forget();
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn conflict_on_nonfirst_shard_coverage_rejects_old_group_without_replay() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(8 * MIB);
    let mut first = open(&uri, "first", budget.clone()).await;
    let mut second = open(&uri, "second", budget.clone()).await;
    put(&first, &[("only-first", b"one")]).await;
    put(&second, &[("shared", b"old")]).await;
    let a = prepare(&first, &budget).await;
    let b = prepare(&second, &budget).await;
    let gate = Arc::new(GateFirstBase {
        once: std::sync::atomic::AtomicBool::new(true),
        entered: tokio::sync::Notify::new(),
        release: tokio::sync::Semaphore::new(0),
    });
    let scope = crate::merge_write_scope::MergeWriteScope::with_authorizer(gate.clone());
    {
        let publishing = scope.run(first.commit_prepared_group(vec![a, b], MIB));
        tokio::pin!(publishing);
        tokio::select! {
            result = &mut publishing => panic!("publication escaped gate: {result:?}"),
            _ = gate.entered.notified() => (),
            _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => panic!("base gate not reached"),
        }
        put(&second, &[("shared", b"newest")]).await;
        second.cleanup_own_shard().await.unwrap();
        let newer = Dataset::open(&uri).await.unwrap().version().version;
        gate.release.add_permits(1);
        let error = publishing.await.unwrap_err();
        assert!(
            error.to_string().contains("shard coverage changed"),
            "{error}"
        );
        assert_eq!(Dataset::open(&uri).await.unwrap().version().version, newer);
    }
    scope.drain().await;
    assert_eq!(values(&uri).await["shared"], b"newest");
    assert_eq!(
        first
            .own_shard_watermark()
            .await
            .unwrap()
            .unwrap()
            .pending_generations,
        1
    );
    assert_eq!(budget.reserved(), 0);
    first.cleanup_own_shard().await.unwrap();
    assert_eq!(values(&uri).await["shared"], b"newest");
    assert_eq!(values(&uri).await["only-first"], b"one");
    first.close().await.unwrap();
    second.close().await.unwrap();
}

/// Publish the real manifest, then lose the response. The caller must never
/// interpret an IO error as permission to replay the prepared group.
#[derive(Debug, Default)]
struct LostCommitReply(std::sync::atomic::AtomicUsize);
#[async_trait::async_trait]
impl lance_table::io::commit::CommitHandler for LostCommitReply {
    async fn commit(
        &self,
        manifest: &mut lance_table::format::Manifest,
        indices: Option<Vec<lance_table::format::IndexMetadata>>,
        path: &object_store::path::Path,
        store: &lance_io::object_store::ObjectStore,
        writer: lance_table::io::commit::ManifestWriter,
        naming: lance_table::io::commit::ManifestNamingScheme,
        transaction: Option<lance_table::format::Transaction>,
    ) -> Result<lance_table::io::commit::ManifestLocation, lance_table::io::commit::CommitError>
    {
        use lance_table::io::commit::{CommitError, RenameCommitHandler};
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        RenameCommitHandler
            .commit(manifest, indices, path, store, writer, naming, transaction)
            .await?;
        Err(CommitError::OtherError(LanceError::io(
            "injected lost commit response",
        )))
    }
}

#[tokio::test]
async fn lost_commit_response_does_not_retry_or_drain_and_recovers_all_shards() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(8 * MIB);
    let mut first = open(&uri, "first", budget.clone()).await;
    let mut second = open(&uri, "second", budget.clone()).await;
    put(&first, &[("shared", b"first")]).await;
    put(&second, &[("shared", b"second")]).await;
    let a = prepare(&first, &budget).await;
    let b = prepare(&second, &budget).await;
    let before = Dataset::open(&uri).await.unwrap().version().version;
    let injected = Arc::new(LostCommitReply::default());
    let scope = crate::merge_write_scope::MergeWriteScope::with_authorizer(Arc::new(Allow));
    first.dataset = lance::dataset::builder::DatasetBuilder::from_uri(&uri)
        .with_commit_handler(Arc::new(crate::merge_write_scope::GuardedCommit::new(
            injected.clone(),
        )))
        .load()
        .await
        .unwrap();
    let error = scope
        .run(first.commit_prepared_group(vec![a, b], MIB))
        .await
        .unwrap_err();
    scope.drain().await;
    assert!(
        error.to_string().contains("injected lost commit response"),
        "{error}"
    );
    assert_eq!(injected.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(scope.has_uncertain_commit());
    assert_eq!(
        Dataset::open(&uri).await.unwrap().version().version,
        before + 1
    );
    assert_eq!(values(&uri).await["shared"], b"second");
    assert_eq!(
        first
            .own_shard_watermark()
            .await
            .unwrap()
            .unwrap()
            .pending_generations,
        1
    );
    assert_eq!(
        second
            .own_shard_watermark()
            .await
            .unwrap()
            .unwrap()
            .pending_generations,
        1
    );
    assert_eq!(budget.reserved(), 0);

    let mut later = open(&uri, "later", budget.clone()).await;
    put(&later, &[("shared", b"newest")]).await;
    later.cleanup_own_shard().await.unwrap();
    let published = Dataset::open(&uri).await.unwrap().version().version;
    let recovery = crate::merge_write_scope::MergeWriteScope::with_authorizer(Arc::new(Allow));
    for store in [&first, &second] {
        assert!(recovery
            .run(store.prepare_grouped_merge_reserved(budget.reserve(MIB).await))
            .await
            .unwrap()
            .is_none());
    }
    recovery.drain().await;
    assert_eq!(
        Dataset::open(&uri).await.unwrap().version().version,
        published
    );
    assert_eq!(values(&uri).await["shared"], b"newest");
    assert_eq!(first.pending_wal_generations().await.unwrap(), 0);
    assert_eq!(injected.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(budget.reserved(), 0);
    first.close().await.unwrap();
    second.close().await.unwrap();
    later.close().await.unwrap();
}

#[tokio::test]
async fn narrow_rows_require_hash_workspace_not_just_arrow_output_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(8 * MIB);
    let mut first = open(&uri, "first", budget.clone()).await;
    let mut second = open(&uri, "second", budget.clone()).await;
    let first_ids: Vec<_> = (0..1024).map(|i| format!("{i:04}")).collect();
    let second_ids: Vec<_> = (512..1536).map(|i| format!("{i:04}")).collect();
    let first_rows: Vec<_> = first_ids
        .iter()
        .map(|id| (id.as_str(), b"a".as_slice()))
        .collect();
    let second_rows: Vec<_> = second_ids
        .iter()
        .map(|id| (id.as_str(), b"b".as_slice()))
        .collect();
    put(&first, &first_rows).await;
    put(&second, &second_rows).await;
    let a = prepare(&first, &budget).await;
    let b = prepare(&second, &budget).await;
    let arrow_bytes = a.decoded_bytes() + b.decoded_bytes();
    let group = vec![a, b];
    assert!(reduction_workspace(&group, "id").unwrap() > arrow_bytes);
    // Enough room for all additional Arrow buffers, but not the reducer's
    // owned keys, hash buckets and selection indices. Do not over-admit it.
    let held = budget
        .try_reserve(budget.limit() - budget.reserved() - arrow_bytes)
        .unwrap();
    let before = Dataset::open(&uri).await.unwrap().version().version;
    assert_eq!(
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            commit(&mut first, group, MIB)
        )
        .await
        .unwrap()
        .unwrap(),
        2
    );
    assert_eq!(
        Dataset::open(&uri).await.unwrap().version().version,
        before + 2
    );
    let rows = values(&uri).await;
    assert_eq!(rows.len(), 1536);
    assert_eq!(rows["0511"], b"a");
    assert_eq!(rows["0512"], b"b");
    assert_eq!(rows["1535"], b"b");
    assert_eq!(first.pending_wal_generations().await.unwrap(), 0);
    assert_eq!(budget.reserved(), held.bytes());
    drop(held);
    first.close().await.unwrap();
    second.close().await.unwrap();
}

#[tokio::test]
async fn changed_base_schema_rejects_prepared_group_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(8 * MIB);
    let mut first = open(&uri, "first", budget.clone()).await;
    let mut second = open(&uri, "second", budget.clone()).await;
    put(&first, &[("one", b"first")]).await;
    put(&second, &[("two", b"second")]).await;
    let group = vec![
        prepare(&first, &budget).await,
        prepare(&second, &budget).await,
    ];
    let mut changed = Dataset::open(&uri).await.unwrap();
    changed
        .update_schema_metadata([("grouped-test", "changed")])
        .await
        .unwrap();
    let before = changed.version().version;
    assert!(commit(&mut first, group, MIB)
        .await
        .unwrap_err()
        .to_string()
        .contains("base schema changed"));
    assert_eq!(Dataset::open(&uri).await.unwrap().version().version, before);
    assert_eq!(first.pending_wal_generations().await.unwrap(), 2);
    assert_eq!(budget.reserved(), 0);
    first.close().await.unwrap();
    second.close().await.unwrap();
}

#[tokio::test]
async fn superseded_generation_membership_rejects_group_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(8 * MIB);
    let mut first = open(&uri, "first", budget.clone()).await;
    let mut second = open(&uri, "second", budget.clone()).await;
    put(&first, &[("one", b"first")]).await;
    put(&second, &[("two", b"second")]).await;
    let group = vec![
        prepare(&first, &budget).await,
        prepare(&second, &budget).await,
    ];
    second.cleanup_own_shard().await.unwrap();
    let before = Dataset::open(&uri).await.unwrap().version().version;
    assert!(commit(&mut first, group, MIB)
        .await
        .unwrap_err()
        .to_string()
        .contains("epoch or membership changed"));
    assert_eq!(Dataset::open(&uri).await.unwrap().version().version, before);
    assert_eq!(
        values(&uri).await,
        HashMap::from([("two".into(), b"second".to_vec())])
    );
    assert_eq!(first.pending_wal_generations().await.unwrap(), 1);
    assert_eq!(budget.reserved(), 0);
    first.close().await.unwrap();
    second.close().await.unwrap();
}

#[tokio::test]
async fn maximum_width_group_publishes_one_version_and_all_32_watermarks() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("base.lance").to_str().unwrap().to_owned();
    let budget = MergeMemoryBudget::new(64 * MIB);
    let mut writers = Vec::new();
    let mut group = Vec::new();
    for i in 0..32 {
        let writer = open(&uri, &format!("writer-{i}"), budget.clone()).await;
        let value = i.to_string();
        put(
            &writer,
            &[
                ("shared", value.as_bytes()),
                (&format!("unique-{i}"), value.as_bytes()),
            ],
        )
        .await;
        group.push(prepare(&writer, &budget).await);
        writers.push(writer);
    }
    let marks: Vec<_> = group
        .iter()
        .map(|s| {
            (
                s.manifest.shard_id,
                *s.prepared.merged_generations.iter().max().unwrap(),
            )
        })
        .collect();
    let before = Dataset::open(&uri).await.unwrap().version().version;
    assert_eq!(commit(&mut writers[0], group, MIB).await.unwrap(), 32);
    let base = Dataset::open(&uri).await.unwrap();
    assert_eq!(base.version().version, before + 1);
    let coverage = crate::rollout_append::watermarks(&base).await.unwrap();
    for (shard, high) in marks {
        assert_eq!(coverage.get(&shard), Some(&high));
    }
    let rows = values(&uri).await;
    assert_eq!(rows.len(), 33);
    assert_eq!(rows["shared"], b"31");
    for i in 0..32 {
        assert_eq!(rows[&format!("unique-{i}")], i.to_string().as_bytes());
    }
    assert_eq!(writers[0].pending_wal_generations().await.unwrap(), 0);
    assert_eq!(budget.reserved(), 0);
    for mut writer in writers {
        writer.close().await.unwrap();
    }
}
