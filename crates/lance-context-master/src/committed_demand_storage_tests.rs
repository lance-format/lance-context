//! Acknowledgment loss after real immutable base/shard publication.
use crate::{config::MasterConfig, state::MasterState};
use async_trait::async_trait;
use clap::Parser;
use futures::stream::BoxStream;
use lance_context_core::{
    merge_write_scope::{CommitAuthorizer, MergeWriteScope},
    RolloutStore, RolloutStoreOptions,
};
use lance_context_merge::demand::{DemandEvent, EventSource, SCHEMA_VERSION};
use lance_io::object_store::WrappingObjectStore;
use object_store::{path::Path, *};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::{Notify, Semaphore};

#[derive(Clone, Copy, Debug)]
enum Boundary {
    Base,
    Shard,
}

#[derive(Clone, Copy, Debug)]
enum Outcome {
    Success,
    Error,
    Cancel,
}

#[derive(Debug)]
struct Gate {
    boundary: Boundary,
    outcome: Outcome,
    armed: AtomicBool,
    entered: Notify,
    release: Semaphore,
}

impl Gate {
    async fn after_publication(&self, path: &Path) -> Result<()> {
        let matches = match self.boundary {
            Boundary::Base => {
                path.as_ref().contains("/_versions/")
                    && !path.as_ref().contains("/_mem_wal/")
                    && path
                        .filename()
                        .and_then(|n| n.strip_suffix(".manifest"))
                        .is_some_and(|n| n.parse::<u64>().is_ok())
            }
            Boundary::Shard => {
                path.as_ref().contains("/_mem_wal/")
                    && path.as_ref().contains("/manifest/")
                    && path.filename().is_some_and(|n| n.ends_with(".binpb"))
            }
        };
        if matches && self.armed.swap(false, Ordering::SeqCst) {
            self.entered.notify_one();
            self.release.acquire().await.unwrap().forget();
            if matches!(self.outcome, Outcome::Error) {
                return Err(Error::Generic {
                    store: "committed-demand-test",
                    source: std::io::Error::other("publication succeeded but acknowledgment lost")
                        .into(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Wrapper(Arc<Gate>);
impl WrappingObjectStore for Wrapper {
    fn wrap_paginated(
        &self,
        _: &str,
        store: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        Some(store)
    }
    fn wrap(&self, _: &str, inner: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(FaultStore {
            inner,
            gate: self.0.clone(),
        })
    }
}

#[derive(Debug)]
struct FaultStore {
    inner: Arc<dyn ObjectStore>,
    gate: Arc<Gate>,
}
impl std::fmt::Display for FaultStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CommittedDemandFault({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for FaultStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        let result = self.inner.put_opts(path, payload, opts).await?;
        self.gate.after_publication(path).await?;
        Ok(result)
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, opts).await
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
        self.inner.copy_opts(from, to, opts).await?;
        self.gate.after_publication(to).await
    }
    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> Result<()> {
        self.inner.rename_opts(from, to, opts).await?;
        self.gate.after_publication(to).await
    }
}

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

async fn demand(client: &mut etcd_client::Client, key: &str) -> DemandEvent {
    serde_json::from_slice(client.get(key, None).await.unwrap().kvs()[0].value()).unwrap()
}

#[tokio::test]
#[ignore = "requires isolated loopback ETCD_TEST_ENDPOINTS and ETCD_TEST_EXCLUSIVE=1"]
async fn committed_demand_real_publication_ack_loss_and_cancellation() {
    assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").unwrap(), "1");
    let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
    assert!(endpoint.starts_with("http://127.0.0.1:") && !endpoint.contains(','));
    for grouped in [false, true] {
        for boundary in [Boundary::Base, Boundary::Shard] {
            for outcome in [Outcome::Success, Outcome::Error, Outcome::Cancel] {
                let dir = tempfile::tempdir().unwrap();
                let mut cfg = MasterConfig::parse_from(["test"]);
                // Force real local storage through the injectable object-store path.
                cfg.data_dir = format!("file-object-store://{}", dir.path().display());
                cfg.etcd.etcd_endpoints = vec![endpoint.clone()];
                cfg.etcd.etcd_prefix =
                    format!("/committed-storage/{}", lance_context_core::generate_id());
                cfg.worker_endpoints.clear();
                cfg.stats_scan_interval_secs = 0;
                cfg.merge_rollout.owned_targets = vec!["hot".into()];
                cfg.local_merge.master_local_merge_targets = vec!["hot".into()];
                cfg.local_merge.master_local_merge_committed_demand_targets = vec!["hot".into()];
                cfg.catchup.shards = vec!["writer".into()];
                if grouped {
                    cfg.catchup.shards.push("other".into());
                    cfg.catchup.grouped_targets = vec!["hot".into()];
                    cfg.catchup.grouped_shards = 2;
                }
                let state = MasterState::new(cfg.clone()).await.unwrap();
                let uri = state.rollout_uri("hot");
                let writer = RolloutStore::open_with_options(
                    &uri,
                    RolloutStoreOptions {
                        shard_id: Some("writer".into()),
                        merge_after_generations: Some(0),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
                let request = serde_json::from_value(
                    serde_json::json!({"id":"one", "rollout_id":"r", "content":"retained"}),
                )
                .unwrap();
                writer
                    .add(&[lance_context_core::rollout_record_from_add_request(
                        &request,
                    )])
                    .await
                    .unwrap();
                writer.flush().await.unwrap();
                let mark = writer.own_shard_watermark().await.unwrap().unwrap();
                let mut marks = vec![mark];
                let mut _other_writer = None;
                if grouped {
                    let other = RolloutStore::open_with_options(
                        &uri,
                        RolloutStoreOptions {
                            shard_id: Some("other".into()),
                            merge_after_generations: Some(0),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
                    let request = serde_json::from_value(
                    serde_json::json!({"id":"two", "rollout_id":"r", "content":"also retained"}),
                ).unwrap();
                    other
                        .add(&[lance_context_core::rollout_record_from_add_request(
                            &request,
                        )])
                        .await
                        .unwrap();
                    other.flush().await.unwrap();
                    marks.push(other.own_shard_watermark().await.unwrap().unwrap());
                    _other_writer = Some(other);
                }
                let expected_rows = marks.len();
                let coordinator = state.task_store.merge_coordinator();
                for mark in &marks {
                    coordinator
                        .publish_demand_event(
                            "hot",
                            &DemandEvent {
                                v: SCHEMA_VERSION,
                                shard: mark.shard_id.to_string(),
                                sealed_through: mark.sealed_through,
                                sealed_bytes_through: 0,
                                merged_through: None,
                                flushed_at_ms: 1234,
                                writer_epoch: mark.writer_epoch,
                                source: EventSource::Writer,
                                merged_epoch: None,
                                sealed_times: Default::default(),
                            },
                        )
                        .await
                        .unwrap();
                }
                let keys: Vec<_> = marks
                    .iter()
                    .map(|mark| {
                        format!(
                            "{}/demand-events/686f74/{}",
                            cfg.etcd.etcd_prefix, mark.shard_id
                        )
                    })
                    .collect();
                let mut client = etcd_client::Client::connect([endpoint.as_str()], None)
                    .await
                    .unwrap();
                let gate = Arc::new(Gate {
                    boundary,
                    outcome,
                    armed: AtomicBool::new(true),
                    entered: Notify::new(),
                    release: Semaphore::new(0),
                });
                let scope = MergeWriteScope::with_preparation_authorizer_and_io_shim(
                    Arc::new(Allow),
                    Arc::new(Wrapper(gate.clone())),
                );
                let runner = {
                    let state = state.clone();
                    let scope = scope.clone();
                    tokio::spawn(
                        async move { scope.run(crate::local_merge::run(&state, "hot")).await },
                    )
                };
                tokio::time::timeout(Duration::from_secs(30), gate.entered.notified())
                    .await
                    .unwrap();
                // The real publication has happened, but its caller has no acknowledgment.
                assert!(!runner.is_finished());
                assert_eq!(
                    lance::Dataset::open(&uri)
                        .await
                        .unwrap()
                        .count_rows(None)
                        .await
                        .unwrap(),
                    expected_rows
                );
                let pending = writer.pending_wal_generations().await.unwrap();
                if matches!(boundary, Boundary::Base) {
                    assert_eq!(pending, expected_rows);
                } else {
                    // The selected shard publication landed; grouped siblings may
                    // also finish draining while its acknowledgment is withheld.
                    assert!(pending < expected_rows);
                }
                for key in &keys {
                    assert_eq!(demand(&mut client, key).await.merged_through, None);
                }
                if matches!(outcome, Outcome::Cancel) {
                    runner.abort();
                    assert!(runner.await.unwrap_err().is_cancelled());
                    gate.release.add_permits(1);
                } else {
                    gate.release.add_permits(1);
                    let result = tokio::time::timeout(Duration::from_secs(30), runner)
                        .await
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        result.is_ok(),
                        matches!(outcome, Outcome::Success),
                        "{boundary:?} {outcome:?}: {result:?}"
                    );
                }
                tokio::time::timeout(Duration::from_secs(30), scope.drain())
                    .await
                    .unwrap();
                if matches!(outcome, Outcome::Success) {
                    tokio::time::timeout(Duration::from_secs(5), async {
                        for (key, mark) in keys.iter().zip(&marks) {
                            while demand(&mut client, key).await.merged_through
                                != Some(mark.sealed_through)
                            {
                                tokio::time::sleep(Duration::from_millis(10)).await;
                            }
                        }
                    })
                    .await
                    .unwrap();
                } else {
                    for key in &keys {
                        assert_eq!(
                            demand(&mut client, key).await.merged_through,
                            None,
                            "uncertain/canceled group must not publish even a drained sibling"
                        );
                    }
                }
                // Recover from durable storage with the optimization OFF; then exercise
                // the unchanged storage-reading publisher, rather than inventing progress.
                cfg.local_merge
                    .master_local_merge_committed_demand_targets
                    .clear();
                let recovery = MasterState::new(cfg).await.unwrap();
                let recovery_scope = MergeWriteScope::with_pinned_authorizer(Arc::new(Allow));
                recovery_scope
                    .run(crate::local_merge::run(&recovery, "hot"))
                    .await
                    .unwrap();
                recovery_scope.drain().await;
                assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
                assert_eq!(
                    lance::Dataset::open(&uri)
                        .await
                        .unwrap()
                        .count_rows(None)
                        .await
                        .unwrap(),
                    expected_rows
                );
                crate::demand_publish::spawn_merged_demand(coordinator, "hot".into(), uri);
                tokio::time::timeout(Duration::from_secs(5), async {
                    for (key, mark) in keys.iter().zip(&marks) {
                        while demand(&mut client, key).await.merged_through
                            != Some(mark.sealed_through)
                        {
                            tokio::time::sleep(Duration::from_millis(10)).await;
                        }
                    }
                })
                .await
                .unwrap();
                eprintln!("verified actual grouped={grouped} {boundary:?} publication with {outcome:?}; durable data retained and demand recovered");
            }
        }
    }
}

#[tokio::test]
#[ignore = "requires isolated loopback ETCD_TEST_ENDPOINTS and ETCD_TEST_EXCLUSIVE=1"]
async fn committed_demand_counts_drained_generations_with_no_rows() {
    use crate::resident_recovery::tests::{
        generic_fixture_with_auto, generic_write, generic_writer,
    };
    use arrow_array::{RecordBatch, RecordBatchIterator};
    use futures::TryStreamExt;
    use lance::dataset::mem_wal::ShardManifestStore;
    assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").as_deref(), Ok("1"));
    let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
    assert!(endpoint.starts_with("http://127.0.0.1:") && !endpoint.contains(','));
    let (_dir, initial) = generic_fixture_with_auto(true).await;
    let mut cfg = initial.config.clone();
    cfg.local_merge.master_local_merge_committed_demand_targets = vec!["generic:a".into()];
    let state = MasterState::new(cfg).await.unwrap();
    let mut writer = generic_writer(&state, "a").await;
    generic_write(
        &writer,
        "retained",
        "real data preceding an empty generation",
    )
    .await;
    let first = writer.own_shard_watermark().await.unwrap().unwrap();
    writer.close().await.unwrap();
    let uri = state.generic_uri("a");
    let base = lance::Dataset::open(&uri).await.unwrap();
    let manifests = ShardManifestStore::new(
        base.object_store(None).await.unwrap(),
        &base.branch_location().path,
        first.shard_id,
        16,
    );
    let before = manifests.refresh_latest().await.unwrap().unwrap();
    assert_eq!(before.sstables.len(), 1);
    // The current ingestion writer rejects empty batches. This is explicitly
    // a storage-format compatibility fixture: publish a complete, valid empty
    // Lance dataset and a truthful successor manifest with the public APIs.
    // No existing payload is edited, and the original writer is fully closed.
    let generation = before.current_generation;
    assert!(generation > before.sstables[0].generation);
    let mut empty = before.sstables[0].clone();
    empty.generation = generation;
    empty.path = format!("empty-format-fixture-{generation}.lance");
    empty.in_memory_bytes = Some(0);
    empty.physical_rows = Some(0);
    empty.primary_key_bytes = Some(0);
    let empty_uri = format!("{uri}/_mem_wal/{}/{}", first.shard_id, empty.path);
    let schema = Arc::new(arrow_schema::Schema::from(base.schema()));
    let batches = RecordBatchIterator::new(
        vec![Ok::<_, arrow_schema::ArrowError>(RecordBatch::new_empty(
            schema.clone(),
        ))],
        schema,
    );
    let empty_dataset = lance::Dataset::write(batches, &empty_uri, None)
        .await
        .unwrap();
    let rows = empty_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    manifests
        .commit_update(before.writer_epoch, |current| {
            let mut next = current.clone();
            next.version = current.next_version();
            next.current_generation = generation + 1;
            next.sstables.push(empty.clone());
            next
        })
        .await
        .unwrap();
    assert_eq!(writer.pending_wal_generations().await.unwrap(), 2);
    let coordinator = state.task_store.merge_coordinator();
    coordinator
        .publish_demand_event(
            "generic:a",
            &DemandEvent {
                v: SCHEMA_VERSION,
                shard: first.shard_id.to_string(),
                sealed_through: generation,
                sealed_bytes_through: 0,
                merged_through: None,
                merged_epoch: None,
                flushed_at_ms: chrono::Utc::now().timestamp_millis(),
                writer_epoch: before.writer_epoch,
                source: EventSource::Writer,
                sealed_times: Default::default(),
            },
        )
        .await
        .unwrap();
    let key = format!(
        "{}/demand-events/67656e657269633a61/{}",
        state.config.etcd.prefix(),
        first.shard_id
    );
    let mut client = state.task_store.etcd_client().clone();
    let scope = MergeWriteScope::with_pinned_authorizer(Arc::new(Allow));
    let result = scope
        .run(crate::local_merge::run(&state, "generic:a"))
        .await
        .unwrap();
    scope.drain().await;
    assert_eq!(result, "local WAL merge reclaimed 2 generations");
    assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
    assert_eq!(
        lance::Dataset::open(&uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        1
    );
    // No maintenance release/fallback publisher ran in this scope. The empty
    // generation's progress must come from its successful native commit.
    tokio::time::timeout(Duration::from_secs(5), async {
        while demand(&mut client, &key).await.merged_through != Some(generation) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    eprintln!("verified actual empty-data generation drain: two generations reclaimed, one real row retained, committed demand reaches empty generation");
}
