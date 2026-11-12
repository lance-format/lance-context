//! Exercise failed flush recovery through the service's pinned Lance dependency.
use super::*;
use arrow_array::StringArray;
use arrow_schema::{DataType, Field};
use futures::stream::BoxStream;
use object_store::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

#[derive(Debug)]
struct FaultStore {
    inner: Arc<dyn object_store::ObjectStore>,
    mode: Arc<AtomicU8>,
    gate: Option<Arc<WalGate>>,
}
#[derive(Debug)]
struct Wrapper(Arc<AtomicU8>);
impl lance_io::object_store::WrappingObjectStore for Wrapper {
    fn wrap(
        &self,
        _: &str,
        inner: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(FaultStore {
            inner,
            mode: self.0.clone(),
            gate: None,
        })
    }
    fn wrap_paginated(
        &self,
        _: &str,
        _: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
}
#[derive(Debug, Default)]
struct WalGate {
    entered: tokio::sync::Notify,
    release: tokio::sync::Notify,
    used: std::sync::atomic::AtomicBool,
}
#[derive(Debug)]
struct GateWrapper(Arc<WalGate>);
impl lance_io::object_store::WrappingObjectStore for GateWrapper {
    fn wrap(
        &self,
        _: &str,
        inner: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(FaultStore {
            inner,
            mode: Arc::new(AtomicU8::new(0)),
            gate: Some(self.0.clone()),
        })
    }
    fn wrap_paginated(
        &self,
        _: &str,
        _: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
}
impl std::fmt::Display for FaultStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "flush fault")
    }
}
fn fault() -> object_store::Error {
    object_store::Error::Generic {
        store: "flush retry test",
        source: "injected publication failure".into(),
    }
}
#[async_trait::async_trait]
impl object_store::ObjectStore for FaultStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: object_store::PutPayload,
        options: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        if path.as_ref().contains("/wal/") {
            let mode = self.mode.load(Ordering::SeqCst);
            if mode == 4
                && self
                    .mode
                    .compare_exchange(mode, 0, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                return Err(fault());
            }
            if let Some(gate) = &self.gate {
                if !gate.used.swap(true, Ordering::SeqCst) {
                    gate.entered.notify_one();
                    gate.release.notified().await;
                }
            }
        }
        if self.mode.load(Ordering::SeqCst) == 3
            && path.as_ref().contains("/_mem_wal/")
            && path.as_ref().contains("/_versions/")
        {
            return Err(fault());
        }
        self.inner.put_opts(path, payload, options).await
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }

    async fn get_opts(
        &self,
        path: &Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        if self.mode.load(Ordering::SeqCst) == 6
            && path.as_ref().contains("/_mem_wal/")
            && path.as_ref().contains("/manifest/")
        {
            return Err(fault());
        }
        self.inner.get_opts(path, options).await
    }
    fn delete_stream(
        &self,
        paths: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(paths)
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        let mode = self.mode.load(Ordering::SeqCst);
        if mode == 5
            && to.as_ref().contains("/wal/")
            && self
                .mode
                .compare_exchange(5, 0, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            // Local atomic create writes a temporary object, then publishes it
            // with rename-if-absent. Lose the ACK of the final WAL publication,
            // not the earlier temporary-file PUT.
            self.inner.copy_opts(from, to, options).await?;
            return Err(fault());
        }
        if matches!(mode, 1 | 2)
            && to.as_ref().contains("/manifest/")
            && to.as_ref().ends_with(".binpb")
        {
            if mode == 2 {
                self.inner.copy_opts(from, to, options).await?;
            }
            return Err(fault());
        }
        self.inner.copy_opts(from, to, options).await
    }
}

#[test]
#[cfg(feature = "metrics")]
fn failed_flush_recovers_without_new_rows_and_reports_retry_metrics() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let gauge = |name: &str| -> f64 {
        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .find_map(|(key, _, _, value)| {
                if key.key().name() == name {
                    if let DebugValue::Gauge(value) = value {
                        return Some(value.into_inner());
                    }
                }
                None
            })
            .expect("metric must be emitted, including zero")
    };
    metrics::with_local_recorder(&recorder, || {
        runtime.block_on(async {
            for failure_mode in [1, 2, 3] {
                let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(StringArray::from(vec!["preserve-this-row"]))],
                )
                .unwrap();
                let dir = tempfile::TempDir::new().unwrap();
                let uri = format!("file://{}", dir.path().display());
                let schema = batch.schema();
                let mut base = StorageBase::open(
                    &uri,
                    StorageBaseOptions {
                        storage_options: None,
                        shard_id: Some("recovery-fault-test".into()),
                        merge_after_generations: None,
                        merge_max_generations: None,
                        merge_max_bytes: None,
                        key_index_type: Default::default(),
                        pending_generations_warn: None,
                        pending_generations_max: None,
                        merge_budget: None,
                        session: None,
                        schema: schema.clone(),
                        key_column: "id".into(),
                        latest_schema: None,
                        seal_on_put: false,
                    },
                    true,
                )
                .await
                .unwrap();

                let mode = Arc::new(AtomicU8::new(0));
                let params = ObjectStoreParams {
                    object_store_wrapper: Some(Arc::new(Wrapper(mode.clone()))),
                    ..Default::default()
                };
                base.dataset = DatasetBuilder::from_uri(&uri)
                    .with_store_params(params.clone())
                    .load()
                    .await
                    .unwrap();
                let config = ShardWriterConfig {
                    shard_id: base.write_shard,
                    observer: Some(Arc::new(crate::metrics::MemtableFlushMetrics::new(
                        &uri,
                        base.write_shard,
                    ))),
                    store_params: Some(params),
                    ..Default::default()
                };
                let writer = Arc::new(
                    ShardWriter::open(
                        base.dataset.object_store(None).await.unwrap(),
                        base.dataset.branch_location().path.clone(),
                        uri,
                        config,
                        schema,
                        vec![],
                    )
                    .await
                    .unwrap(),
                );
                writer.put(vec![batch.clone()]).await.unwrap();
                *base.write_writer.lock().await = Some(writer.clone());
                mode.store(failure_mode, Ordering::SeqCst);
                assert!(
                    base.flush().await.is_err(),
                    "injected failure mode {failure_mode} must be surfaced"
                );
                assert!(writer.memory().frozen_bytes() > 0);
                assert_eq!(gauge("rollout_memtable_flush_retry_generations"), 1.0);
                assert!(gauge("rollout_memtable_flush_retry_bytes") > 0.0);
                let health = crate::flush_health::snapshot();
                let failed = health
                    .active
                    .iter()
                    .find(|row| row.shard == base.write_shard.to_string())
                    .unwrap();
                assert_eq!(failed.retry_generations, 1);
                assert!(failed.failure_age_seconds.is_some());
                assert!(!failed.terminal);
                assert!(matches!(failed.checkpoint, "wal_durable" | "prepared"));
                // A second failure must not inflate the gauge or discard the batch.
                if failure_mode != 2 {
                    assert!(base.flush().await.is_err());
                    assert_eq!(gauge("rollout_memtable_flush_retry_generations"), 1.0);
                }
                let boundary = writer.wal_stats().next_wal_entry_position - 1;
                if failure_mode == 3 {
                    // A newer active WAL append must not move this failed
                    // generation's original replay boundary when it retries.
                    writer.put(vec![batch]).await.unwrap();
                    assert!(writer.wal_stats().next_wal_entry_position - 1 > boundary);
                }
                mode.store(0, Ordering::SeqCst);
                if failure_mode == 2 {
                    // The ACK was lost after publication. A real merge can
                    // already consume the generation before the writer retries.
                    let (store, manifest, prepared) =
                        base.prepare_merge_if_ready(1).await.unwrap().unwrap();
                    assert_eq!(
                        base.commit_prepared_merge(&store, &manifest, prepared)
                            .await
                            .unwrap(),
                        1
                    );
                    assert!(store
                        .refresh_latest()
                        .await
                        .unwrap()
                        .unwrap()
                        .sstables
                        .is_empty());
                }
                if failure_mode == 3 {
                    // No new flush request: the periodic retry must recover.
                    tokio::time::timeout(Duration::from_secs(40), async {
                        while writer.memory().frozen_bytes() != 0 {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                    })
                    .await
                    .expect("idle retry must recover retained data");
                } else {
                    tokio::time::timeout(Duration::from_secs(10), base.flush())
                        .await
                        .unwrap()
                        .unwrap();
                }
                assert_eq!(writer.memory().frozen_bytes(), 0);
                assert_eq!(gauge("rollout_memtable_flush_retry_generations"), 0.0);
                assert_eq!(gauge("rollout_memtable_flush_retry_bytes"), 0.0);
                assert!(!crate::flush_health::snapshot()
                    .active
                    .iter()
                    .any(|row| row.shard == base.write_shard.to_string()));
                let manifest = writer.manifest().await.unwrap().unwrap();
                assert_eq!(manifest.replay_after_wal_entry_position, boundary);
                let rows = if failure_mode == 2 {
                    assert!(
                        manifest.sstables.is_empty(),
                        "retry must not republish a merged generation"
                    );
                    base.dataset.scan().try_into_batch().await.unwrap()
                } else {
                    assert_eq!(manifest.sstables.len(), 1);
                    assert_eq!(manifest.sstables[0].physical_rows, Some(1));
                    let generation_uri = format!(
                        "{}/_mem_wal/{}/{}",
                        base.uri(),
                        base.write_shard,
                        manifest.sstables[0].path
                    );
                    DatasetBuilder::from_uri(generation_uri)
                        .load()
                        .await
                        .unwrap()
                        .scan()
                        .try_into_batch()
                        .await
                        .unwrap()
                };
                assert_eq!(rows.num_rows(), 1);
                assert_eq!(
                    rows.column_by_name("id")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .value(0),
                    "preserve-this-row"
                );
                drop(writer);
                base.close().await.unwrap();
            }
        })
    });
    let counters = snapshotter.snapshot().into_vec();
    assert!(counters.iter().any(|(key, _, _, value)| key.key().name()
        == "rollout_memtable_flush_retries_total"
        && matches!(value, DebugValue::Counter(n) if *n >= 3)));
}

#[tokio::test]
async fn coalesced_wal_flush_preserves_each_generation_boundary() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
    let dir = tempfile::TempDir::new().unwrap();
    let uri = format!("file://{}", dir.path().display());
    let gate = Arc::new(WalGate::default());
    let params = ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(GateWrapper(gate.clone()))),
        ..Default::default()
    };
    let (store, path) = lance_io::object_store::ObjectStore::from_uri_and_params(
        Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
        &uri,
        &params,
    )
    .await
    .unwrap();
    let shard_id = uuid::Uuid::new_v4();
    let writer = ShardWriter::open(
        store,
        path,
        uri.clone(),
        ShardWriterConfig {
            shard_id,
            store_params: Some(params),
            durable_write: false,
            max_wal_buffer_size: 1024,
            max_wal_flush_interval: None,
            ..Default::default()
        },
        schema.clone(),
        vec![],
    )
    .await
    .unwrap();
    let batch = |value: &str| {
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![value]))],
        )
        .unwrap()
    };
    // The small first generation is flushed only by its seal. Hold that WAL PUT
    // while the second generation queues both a size-trigger and its own seal.
    writer.put(vec![batch("first")]).await.unwrap();
    let first = writer.force_seal_active().await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), gate.entered.notified())
        .await
        .unwrap();
    let second_value = "x".repeat(4096);
    writer.put(vec![batch(&second_value)]).await.unwrap();
    let second = writer.force_seal_active().await.unwrap();
    gate.release.notify_one();
    tokio::time::timeout(Duration::from_secs(15), first.wait())
        .await
        .unwrap()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(15), second.wait())
        .await
        .unwrap()
        .unwrap();
    let manifest = writer.manifest().await.unwrap().unwrap();
    assert_eq!(manifest.replay_after_wal_entry_position, 2);
    assert_eq!(manifest.sstables.len(), 2);
    for (generation, expected) in manifest
        .sstables
        .iter()
        .zip(["first", second_value.as_str()])
    {
        let rows =
            DatasetBuilder::from_uri(format!("{}/_mem_wal/{}/{}", uri, shard_id, generation.path))
                .load()
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap();
        assert_eq!(rows.num_rows(), 1);
        assert_eq!(
            rows.column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            expected
        );
    }
    writer.close().await.unwrap();
}

/// Exercise the service reopen path, not just WalAppender retry: previously
/// acknowledged rows must survive and the predecessor's tasks must release
/// their observer, including when a WAL PUT landed but its ACK was lost.
#[tokio::test]
#[cfg(feature = "metrics")]
async fn fenced_writer_reopen_replays_wal_and_releases_old_tasks() {
    for failure_mode in [4, 5] {
        for recover_on_flush in [false, true] {
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
            let batch = |value: &str| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(StringArray::from(vec![value]))],
                )
                .unwrap()
            };
            let dir = tempfile::TempDir::new().unwrap();
            let uri = format!("file://{}", dir.path().display());
            let mut base = StorageBase::open(
                &uri,
                StorageBaseOptions {
                    storage_options: None,
                    shard_id: Some("fenced-lifetime-test".into()),
                    merge_after_generations: None,
                    merge_max_generations: None,
                    merge_max_bytes: None,
                    key_index_type: Default::default(),
                    pending_generations_warn: None,
                    pending_generations_max: None,
                    merge_budget: None,
                    session: None,
                    schema: schema.clone(),
                    key_column: "id".into(),
                    latest_schema: None,
                    seal_on_put: false,
                },
                true,
            )
            .await
            .unwrap();
            let mode = Arc::new(AtomicU8::new(0));
            let params = ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(Wrapper(mode.clone()))),
                ..Default::default()
            };
            base.dataset = DatasetBuilder::from_uri(&uri)
                .with_store_params(params.clone())
                .load()
                .await
                .unwrap();
            let observer = Arc::new(crate::metrics::MemtableFlushMetrics::new(
                &uri,
                base.write_shard,
            ));
            let old_observer = Arc::downgrade(&observer);
            let writer = Arc::new(
                ShardWriter::open(
                    base.dataset.object_store(None).await.unwrap(),
                    base.dataset.branch_location().path.clone(),
                    uri.clone(),
                    ShardWriterConfig {
                        shard_id: base.write_shard,
                        observer: Some(observer),
                        store_params: Some(params),
                        max_wal_persist_retries: 0,
                        max_wal_flush_interval: Some(Duration::from_millis(10)),
                        ..Default::default()
                    },
                    schema.clone(),
                    vec![],
                )
                .await
                .unwrap(),
            );
            *base.write_writer.lock().await = Some(writer.clone());
            base.put(vec![batch("acknowledged-before-failure")])
                .await
                .unwrap();
            let old_epoch = writer.manifest().await.unwrap().unwrap().writer_epoch;
            mode.store(failure_mode, Ordering::SeqCst);
            if recover_on_flush {
                // No subsequent put: a periodic flush must reopen/replay too.
                let error = writer
                    .put(vec![batch("ambiguous-write")])
                    .await
                    .unwrap_err();
                assert_eq!(
                    error.fence_reason(),
                    Some(lance_core::error::FenceReason::PersistenceFailure)
                );
                // Repeated reopen failures must not erase the resident slot:
                // the periodic flusher needs to find it again without a put.
                mode.store(6, Ordering::SeqCst);
                for _ in 0..2 {
                    assert!(
                        tokio::time::timeout(Duration::from_secs(10), base.flush_inner())
                            .await
                            .unwrap()
                            .is_err()
                    );
                    assert!(Arc::ptr_eq(
                        base.write_writer.lock().await.as_ref().unwrap(),
                        &writer
                    ));
                }
                mode.store(0, Ordering::SeqCst);
                assert_eq!(
                    tokio::time::timeout(Duration::from_secs(10), base.flush_inner())
                        .await
                        .unwrap()
                        .unwrap(),
                    FlushOutcome::RecoveredFence,
                );
            } else {
                tokio::time::timeout(
                    Duration::from_secs(10),
                    base.put(vec![batch("ambiguous-write")]),
                )
                .await
                .unwrap()
                .unwrap();
                base.flush().await.unwrap();
            }
            let successor = base.resident_writer(None).await.unwrap();
            assert!(!Arc::ptr_eq(&writer, &successor));
            let manifest = successor.manifest().await.unwrap().unwrap();
            assert_eq!(manifest.writer_epoch, old_epoch + 1);
            assert!(manifest.replay_after_wal_entry_position > 0);

            // Validate actual committed content before asserting task release.
            let mut rows = std::collections::BTreeSet::new();
            for generation in &manifest.sstables {
                let ds = DatasetBuilder::from_uri(format!(
                    "{}/_mem_wal/{}/{}",
                    uri, base.write_shard, generation.path
                ))
                .load()
                .await
                .unwrap();
                let records = ds.scan().try_into_batch().await.unwrap();
                let ids = records
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                rows.extend(ids.iter().map(|value| value.unwrap().to_owned()));
            }
            assert!(rows.contains("acknowledged-before-failure"));
            assert_eq!(
                rows.contains("ambiguous-write"),
                !recover_on_flush || failure_mode == 5
            );
            assert_eq!(
                rows.len(),
                if !recover_on_flush || failure_mode == 5 {
                    2
                } else {
                    1
                }
            );
            drop(writer);
            tokio::time::timeout(Duration::from_secs(5), async {
                while old_observer.upgrade().is_some() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("old dispatcher must release its observer and retained state");
            drop(successor);
            base.close().await.unwrap();
        }
    }
}
