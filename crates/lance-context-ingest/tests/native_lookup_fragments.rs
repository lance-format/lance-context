//! Opt-in single-query fragment concurrency and cancellation experiment.
#![cfg(feature = "materializer")]
use anyhow::{ensure, Context, Result};
use arrow_array::{BinaryArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::future::BoxFuture;
use lance::{
    dataset::{builder::DatasetBuilder, WriteParams},
    Dataset,
};
use lance_context_core::MergeMemoryBudget;
use lance_context_ingest::turn_ingestion::TurnIngestionSpec;
use lance_core::memory::{NativeMemoryKind, NativeMemoryPool, NativeMemoryReservation};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Debug, Default)]
struct Accounting {
    total: Arc<AtomicUsize>,
    total_peak: AtomicUsize,
}
#[derive(Debug)]
struct Charge {
    total: Arc<AtomicUsize>,
    bytes: usize,
}
impl NativeMemoryReservation for Charge {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn set_backed_bytes(&mut self, n: usize) {
        assert!(n <= self.bytes);
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.total.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
impl NativeMemoryPool for Accounting {
    fn reserve(
        &self,
        kind: NativeMemoryKind,
        bytes: usize,
    ) -> BoxFuture<'_, lance_core::Result<Box<dyn NativeMemoryReservation>>> {
        Box::pin(async move { self.try_reserve(kind, bytes) })
    }
    fn try_reserve(
        &self,
        _: NativeMemoryKind,
        bytes: usize,
    ) -> lance_core::Result<Box<dyn NativeMemoryReservation>> {
        let mut current = self.total.load(Ordering::SeqCst);
        loop {
            let next = current
                .checked_add(bytes)
                .filter(|n| *n <= 256 << 20)
                .ok_or_else(|| {
                    lance_core::Error::backpressure("experiment native budget exhausted")
                })?;
            match self.total.compare_exchange_weak(
                current,
                next,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => {
                    self.total_peak.fetch_max(next, Ordering::SeqCst);
                    break;
                }
                Err(observed) => current = observed,
            }
        }
        Ok(Box::new(Charge {
            total: self.total.clone(),
            bytes,
        }))
    }
}
fn hash(mut n: u64) -> String {
    n = n.wrapping_add(0x9e3779b97f4a7c15);
    n = (n ^ (n >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    n = (n ^ (n >> 27)).wrapping_mul(0x94d049bb133111eb);
    format!("{:016x}", n ^ (n >> 31))
}
fn session(n: u64) -> String {
    hash(n % 4096)
}
use datafusion_physical_plan::{displayable, ExecutionPlan};
use futures::{stream::BoxStream, StreamExt, TryStreamExt};
use lance::io::exec::filtered_read::{FilteredReadExec, FilteredReadThreadingMode};
use lance_datafusion::exec::{
    collect_execution_metrics, execute_plan, get_session_context, ExecutionSummaryCounts,
    LanceExecutionOptions,
};
use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
use object_store::{
    path::Path, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{atomic::AtomicBool, Mutex},
    time::Duration,
};

#[derive(Debug, Default)]
struct Reads {
    files: Mutex<BTreeMap<String, usize>>,
    peak_files: AtomicUsize,
    active: AtomicUsize,
    peak: AtomicUsize,
    slow_path: String,
    slowed: AtomicBool,
    delay: Duration,
    release: Option<Arc<tokio::sync::Notify>>,
}
struct ReadGuard<'a>(&'a Reads, String);
impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
        let mut files = self.0.files.lock().unwrap();
        let count = files.get_mut(&self.1).unwrap();
        *count -= 1;
        if *count == 0 {
            files.remove(&self.1);
        }
    }
}
impl Reads {
    async fn enter(&self, path: &Path) -> Option<ReadGuard<'_>> {
        if !path.as_ref().contains("/data/") {
            return None;
        }
        let name = path.to_string();
        let guard = ReadGuard(self, name.clone());
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        {
            let mut files = self.files.lock().unwrap();
            *files.entry(name.clone()).or_default() += 1;
            self.peak_files.fetch_max(files.len(), Ordering::SeqCst);
        }
        if !self.delay.is_zero()
            && name.ends_with(&self.slow_path)
            && !self.slowed.swap(true, Ordering::SeqCst)
        {
            if let Some(release) = &self.release {
                release.notified().await;
            } else {
                tokio::time::sleep(self.delay).await;
            }
        }
        Some(guard)
    }
}
#[derive(Debug)]
struct Wrapper(Arc<Reads>);
impl WrappingObjectStore for Wrapper {
    fn wrap(&self, _: &str, inner: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(Store {
            inner,
            reads: self.0.clone(),
        })
    }
    fn wrap_paginated(
        &self,
        _: &str,
        original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        Some(original)
    }
}
#[derive(Debug)]
struct Store {
    inner: Arc<dyn ObjectStore>,
    reads: Arc<Reads>,
}
impl std::fmt::Display for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("experiment read concurrency")
    }
}
#[async_trait::async_trait]
impl ObjectStore for Store {
    async fn put_opts(
        &self,
        p: &Path,
        b: PutPayload,
        o: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(p, b, o).await
    }
    async fn put_multipart_opts(
        &self,
        p: &Path,
        o: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(p, o).await
    }
    async fn get_opts(&self, p: &Path, o: GetOptions) -> object_store::Result<GetResult> {
        let _guard = self.reads.enter(p).await;
        let result = self.inner.get_opts(p, o).await?;
        let meta = result.meta.clone();
        let range = result.range.clone();
        let attributes = result.attributes.clone();
        let extensions = result.extensions.clone();
        // Count request lifetime through body completion, not retained buffers.
        let bytes = result.bytes().await?;
        Ok(GetResult {
            meta,
            range,
            attributes,
            extensions,
            payload: GetResultPayload::Stream(
                futures::stream::once(async move { Ok(bytes) }).boxed(),
            ),
        })
    }
    fn list(&self, p: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(p)
    }
    async fn list_with_delimiter(&self, p: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(p).await
    }
    fn delete_stream(
        &self,
        p: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(p)
    }
    async fn copy_opts(&self, a: &Path, b: &Path, o: CopyOptions) -> object_store::Result<()> {
        self.inner.copy_opts(a, b, o).await
    }
}

// Native partitions consume disjoint tasks from one shared stream; not N copies
// of the table scan. Keep the existing selection, index input, schema and pool.
fn partitions(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(read) = plan.downcast_ref::<FilteredReadExec>() {
        ensure!(
            read.row_stream_input().is_none(),
            "experiment expects table scan"
        );
        return Ok(Arc::new(FilteredReadExec::try_new(
            read.dataset().clone(),
            read.options()
                .clone()
                .with_threading_mode(FilteredReadThreadingMode::MultiplePartitions(8)),
            read.index_input().cloned(),
        )?));
    }
    let children = plan
        .children()
        .iter()
        .map(|p| partitions((*p).clone()))
        .collect::<Result<Vec<_>>>()?;
    if children.is_empty() {
        Ok(plan)
    } else {
        Ok(plan.with_new_children(children)?)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "controlled native single-query experiment; local I/O is not production throughput"]
async fn single_query_fragment_parallelism() -> Result<()> {
    const ROWS: u64 = 4_194_304;
    const FILE_ROWS: u64 = 16_384;
    const KEYS: u64 = 9600;
    let root = tempfile::tempdir()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("session_id", DataType::Utf8, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::Binary, false),
    ]));
    let spec = TurnIngestionSpec::new(
        schema.clone(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let batch_schema = schema.clone();
    let batches = (0..ROWS).step_by(FILE_ROWS as usize).map(move |start| {
        RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from_iter_values(
                    (start..start + FILE_ROWS).map(session),
                )),
                Arc::new(StringArray::from_iter_values(
                    (start..start + FILE_ROWS).map(hash),
                )),
                Arc::new(BinaryArray::from_iter_values(
                    (start..start + FILE_ROWS).map(|_| [0xffu8; 128]),
                )),
            ],
        )
    });
    let uri = format!(
        "file-object-store://{}",
        root.path().join("native.lance").display()
    );
    let dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema.clone()),
        &uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: FILE_ROWS as usize,
            ..Default::default()
        }),
    )
    .await?;
    assert_eq!(dataset.manifest().fragments.len(), 256);
    let slow_path = dataset.manifest().fragments[0].files[0].path.clone();
    eprintln!(
        "fixture first_fragment={} file_version={:?}",
        slow_path,
        dataset.manifest().fragments[0].files[0]
    );
    for delay_ms in [0u64, 2000] {
        for mode in ["ordered", "unordered_only", "native_partitions"] {
            let hits = KEYS * 4 / 5;
            let numbers = (0..KEYS)
                .map(|i| {
                    if i < hits {
                        (i * 104729) % ROWS
                    } else {
                        ROWS + i
                    }
                })
                .collect::<Vec<_>>();
            let keys = numbers
                .iter()
                .map(|n| spec.business_key(&[&session(*n), &hash(*n)], 4096))
                .collect::<Result<Vec<_>>>()?;
            let memory = Arc::new(Accounting::default());
            let reads = Arc::new(Reads {
                slow_path: slow_path.clone(),
                delay: Duration::from_millis(delay_ms),
                ..Default::default()
            });
            let base = Arc::new(
                DatasetBuilder::from_uri(&uri)
                    .with_store_params(ObjectStoreParams {
                        object_store_wrapper: Some(Arc::new(Wrapper(reads.clone()))),
                        ..Default::default()
                    })
                    .with_session(Arc::new(
                        lance::session::Session::default().with_native_memory(memory.clone()),
                    ))
                    .load()
                    .await?,
            );
            let values = [
                numbers.iter().map(|n| session(*n)).collect::<BTreeSet<_>>(),
                numbers.iter().map(|n| hash(*n)).collect::<BTreeSet<_>>(),
            ];
            let predicate = ["session_id", "id"]
                .into_iter()
                .zip(values)
                .map(|(name, values)| {
                    format!(
                        "`{name}` IN ({})",
                        values
                            .into_iter()
                            .map(|v| format!("'{v}'"))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(" AND ");
            let mut scan = base.scan();
            scan.project(&["session_id", "id"])?;
            scan.filter(&predicate)?;
            scan.batch_size_bytes(1 << 20)
                .io_buffer_size(8 << 20)
                .batch_size(2048)
                .batch_readahead(8)
                .fragment_readahead(8);
            if mode != "ordered" {
                scan.scan_in_order(false);
            }
            let store = base.object_store(None).await?;
            let before = store.io_stats_snapshot();
            let started = Instant::now();
            let plan = scan.create_plan().await?;
            let plan = if mode == "native_partitions" {
                partitions(plan)?
            } else {
                plan
            };
            // Store the full predicate-bearing plan, but keep console output bounded.
            std::fs::write(
                format!("/tmp/lance-wal-fragment-plan-{mode}-{delay_ms}.txt"),
                displayable(plan.as_ref()).indent(true).to_string(),
            )?;
            eprintln!(
                "plan mode={mode} delay_ms={delay_ms} root={} partitions={}",
                plan.name(),
                plan.properties().partitioning.partition_count()
            );
            let options = LanceExecutionOptions {
                runtime_env: base.session().runtime_env(),
                batch_size: Some(2048),
                ..Default::default()
            };
            let ctx = get_session_context(&options).task_ctx();
            let mut nodes = vec![plan.clone()];
            while let Some(node) = nodes.pop() {
                if let Some(read) = node.downcast_ref::<FilteredReadExec>() {
                    read.get_or_create_plan(ctx.clone()).await?;
                }
                nodes.extend(node.children().into_iter().cloned());
            }
            ensure!(
                !reads.slowed.load(Ordering::SeqCst) && reads.peak.load(Ordering::SeqCst) == 0,
                "fixture delay unexpectedly hit planning metadata"
            );
            let mut stream = execute_plan(plan.clone(), options)?;
            let mut first_ms = None;
            let mut found = BTreeSet::new();
            while let Some(batch) = stream.try_next().await? {
                first_ms.get_or_insert(started.elapsed().as_millis());
                let a = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let b = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                for i in 0..batch.num_rows() {
                    found.insert(spec.business_key(&[a.value(i), b.value(i)], 4096)?);
                }
            }
            drop(stream);
            let elapsed_ms = started.elapsed().as_millis();
            for (i, key) in keys.iter().enumerate() {
                ensure!(
                    found.contains(key) == ((i as u64) < hits),
                    "incorrect membership"
                );
            }
            let after = store.io_stats_snapshot();
            let mut metrics = ExecutionSummaryCounts::default();
            collect_execution_metrics(plan.as_ref(), &mut metrics);
            ensure!(
                metrics.all_counts.get("fragments_scanned") == Some(&256)
                    && metrics.all_counts.get("rows_scanned") == Some(&(ROWS as usize)),
                "fragment partitioning must cover the table exactly once"
            );
            eprintln!("fragment_parallel mode={mode} delay_ms={delay_ms} elapsed_ms={elapsed_ms} first_ms={first_ms:?} read_ops={} read_bytes={} native_peak={} io_peak={} io_files_peak={} slow_injected={} metrics={metrics:?}", after.read_iops-before.read_iops, after.read_bytes-before.read_bytes, memory.total_peak.load(Ordering::SeqCst), reads.peak.load(Ordering::SeqCst), reads.peak_files.load(Ordering::SeqCst), reads.slowed.load(Ordering::SeqCst));
            ensure!(
                reads.peak.load(Ordering::SeqCst) > 0,
                "I/O instrumentation bypassed"
            );
            drop(plan);
            drop(base);
            drop(store);
            tokio::time::timeout(Duration::from_secs(5), async {
                while memory.total.load(Ordering::SeqCst) > 0 {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .context("native reservations leaked")?;
        }
    }
    for all_hits in [false, true] {
        // All-hit keys are in the second fragment. Gate the first file read:
        // successful lookup must return before it completes. Native scheduler
        // drop cancels queued reads; already issued I/O completes independently.
        let count = if all_hits { 512 } else { KEYS };
        let hits = if all_hits { count } else { count * 4 / 5 };
        let keys = (0..count)
            .map(|i| {
                let n = if all_hits {
                    FILE_ROWS + i
                } else if i < hits {
                    (i * 104729) % ROWS
                } else {
                    ROWS + i
                };
                spec.business_key(&[&session(n), &hash(n)], 4096)
            })
            .collect::<Result<Vec<_>>>()?;
        let memory = Arc::new(Accounting::default());
        let reads = Arc::new(Reads {
            slow_path: slow_path.clone(),
            delay: Duration::from_secs(if all_hits { 60 } else { 2 }),
            release: all_hits.then(|| Arc::new(tokio::sync::Notify::new())),
            ..Default::default()
        });
        let base = Arc::new(
            DatasetBuilder::from_uri(&uri)
                .with_store_params(ObjectStoreParams {
                    object_store_wrapper: Some(Arc::new(Wrapper(reads.clone()))),
                    ..Default::default()
                })
                .with_session(Arc::new(
                    lance::session::Session::default().with_native_memory(memory.clone()),
                ))
                .load()
                .await?,
        );
        let refs = keys.iter().map(String::as_str).collect::<Vec<_>>();
        let started = Instant::now();
        let lookup = tokio::time::timeout(
            Duration::from_secs(10),
            spec.lookup_existing(base.clone(), &refs, &MergeMemoryBudget::new(32 << 20)),
        )
        .await
        .context("product lookup waited for irrelevant slow fragment")??
        .context("workspace refused")?;
        for (i, key) in refs.iter().enumerate() {
            ensure!(
                lookup.get(key) == Some((i as u64) < hits),
                "incorrect product membership"
            );
        }
        ensure!(
            reads.slowed.load(Ordering::SeqCst),
            "slow fragment was never scheduled"
        );
        ensure!(
            lookup.observation().scan_partitions == 8
                && lookup.observation().fragment_streams_peak == Some(8),
            "product must report the actual bounded native producer path"
        );
        eprintln!("product_fragment_parallel all_hits={all_hits} elapsed_ms={} native_peak={} io_peak={} io_files_peak={} observation={:?}", started.elapsed().as_millis(), memory.total_peak.load(Ordering::SeqCst), reads.peak.load(Ordering::SeqCst), reads.peak_files.load(Ordering::SeqCst), lookup.observation());
        drop(lookup);
        drop(base);
        if let Some(release) = &reads.release {
            eprintln!(
                "early_return_residual native_held={} active_reads={}",
                memory.total.load(Ordering::SeqCst),
                reads.active.load(Ordering::SeqCst)
            );
            ensure!(
                memory.total.load(Ordering::SeqCst) <= 256 << 20
                    && reads.active.load(Ordering::SeqCst) <= 8,
                "early-return work exceeded admission/concurrency"
            );
            release.notify_one();
        }
        tokio::time::timeout(Duration::from_secs(5), async {
            while memory.total.load(Ordering::SeqCst) > 0 || reads.active.load(Ordering::SeqCst) > 0
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .context("cancelled product scan did not release reads/reservations")?;
    }
    Ok(())
}
