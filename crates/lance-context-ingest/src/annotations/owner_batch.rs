//! Owner-side streaming batch pipeline for full-body first-create ingestion.
//!
//! Three stages overlap. None of them moves the ordered decision point:
//!
//! 1. **Collect** (`prefetch::prepare`, run by `pipeline::Ahead`): committed
//!    records are decoded once, their keys kept and bodies released at once.
//!    Bodies stay in the WAL. The key buffer is bounded by the key workspace
//!    (`KeyBound`) and the configured max wait (`coalesce` deadline).
//! 2. **Look up** (`pipeline::Query`): ONE historical lookup for the whole
//!    buffered id set, with the predicate on the indexed key columns so an
//!    available ZoneMap/Bloom index prunes fragments. While it runs, the
//!    successor `Ahead` keeps collecting the next batch.
//! 3. **Write** (`WriteBatch` + the publisher): ordered apply records only
//!    the surviving `(record, row)` ordinals of each committed request. The
//!    list spills to a caller directory (e.g. a PVC) when large. Publication
//!    external-sorts survivor keys by `(session_id, id)` across the whole
//!    frozen batch, refetches bodies from the WAL in key order, and writes
//!    ONE globally ordered fragment per published batch.
//!
//! Exactness is unchanged: ordered apply under the staging mutation gate
//! still decides first create per shard, and keys staged by earlier,
//! still unpublished batches are marked present when the lookup captures its
//! base. Applied progress and receipts move only at that ordered decision.
//! Source WAL records are retired only after the publication that contains
//! their survivors is selected, so recovery replays and rebuilds the lists.
use super::*;
use datafusion_common::arrow::compute::interleave_record_batch;
use datafusion_common::DataFusionError;
use datafusion_physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

/// Opt-in owner batch pipeline configuration.
#[derive(Clone, Debug)]
pub struct Options {
    /// Directory for write-batch spill files, normally on a caller PVC. A
    /// per-owner subdirectory is created and cleared at open. `None` keeps
    /// survivor lists in RAM (still keys-only, bodies remain in the WAL).
    /// Defaults to `$TMPDIR/lance-context-owner-batch`.
    pub spill_dir: Option<PathBuf>,
    /// Spill the in-RAM survivor list once it exceeds this many bytes.
    pub spill_bytes: usize,
    /// Publish once the write batch holds this many surviving rows, in
    /// addition to the publisher's existing source byte/record/wait triggers.
    pub target_rows: u64,
    /// Bound on decoded WAL payload bytes refetched together during
    /// publication. Output order is global regardless of this bound.
    pub refetch_bytes: u64,
    /// Retained for configuration compatibility. Publication now always
    /// orders the whole frozen batch by `(session_id, id)`.
    pub cluster_rows: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            // Default spill target is the process temp directory; hosts
            // should point this at a PVC via `with_owner_batch`.
            spill_dir: Some(std::env::temp_dir().join("lance-context-owner-batch")),
            spill_bytes: 64 << 20,
            target_rows: 1 << 20,
            refetch_bytes: 256 << 20,
            cluster_rows: 4096,
        }
    }
}

/// Surviving source rows of one committed request. Only ordinals are kept.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Survivors {
    pub index: u64,
    pub meta: BatchMeta,
    pub rows: Vec<u32>,
}

impl Survivors {
    fn ram_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.rows.capacity() * std::mem::size_of::<u32>()
    }
}

/// A spill file owned by exactly one active or frozen segment.
struct SpillFile {
    path: PathBuf,
    bytes: u64,
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Default)]
struct Segment {
    spilled: Vec<SpillFile>,
    ram: Vec<Survivors>,
    ram_bytes: usize,
    rows: u64,
    metas: BTreeMap<u64, BatchMeta>,
}

/// One frozen write batch, bound to exactly one frozen checkpoint.
pub(crate) struct Frozen {
    segment: Segment,
}

impl Frozen {
    pub fn rows(&self) -> u64 {
        self.segment.rows
    }
}

pub(crate) struct WriteBatch {
    pub(super) options: Options,
    dir: Option<PathBuf>,
    active: Segment,
    frozen: Option<Arc<Frozen>>,
}

impl WriteBatch {
    /// Create the owner's spill directory, discarding files of a previous
    /// incarnation. Those lists are rebuilt by WAL replay after recovery.
    pub fn open(options: Options, owner: &str) -> Result<Self> {
        ensure!(
            options.spill_bytes > 0 && options.refetch_bytes > 0 && options.target_rows > 0,
            "invalid owner batch options"
        );
        let dir = match &options.spill_dir {
            Some(root) => {
                let dir = root.join(owner);
                if dir.exists() {
                    std::fs::remove_dir_all(&dir)
                        .with_context(|| format!("clear write batch spill {}", dir.display()))?;
                }
                std::fs::create_dir_all(&dir)
                    .with_context(|| format!("create write batch spill {}", dir.display()))?;
                Some(dir)
            }
            None => None,
        };
        Ok(Self {
            options,
            dir,
            active: Segment::default(),
            frozen: None,
        })
    }

    /// Record one committed decision. Called in the synchronous apply handoff,
    /// so a cancelled step can never record a partial list. A spill failure
    /// stops the owner before progress or receipts advance; fenced recovery
    /// replays the WAL from the last publication and rebuilds the list.
    pub(super) fn record(&mut self, survivors: Survivors) -> Result<()> {
        self.active.rows += survivors.rows.len() as u64;
        if !survivors.rows.is_empty() {
            self.active
                .metas
                .insert(survivors.index, survivors.meta.clone());
        }
        self.active.ram_bytes += survivors.ram_bytes();
        self.active.ram.push(survivors);
        if self.active.ram_bytes >= self.options.spill_bytes {
            if let Some(dir) = &self.dir {
                let file = spill(dir, &self.active.ram)?;
                self.active.ram = Vec::new();
                self.active.ram_bytes = 0;
                self.active.spilled.push(file);
            }
        }
        Ok(())
    }

    pub fn rows(&self) -> u64 {
        self.active.rows + self.frozen.as_ref().map_or(0, |f| f.segment.rows)
    }

    /// Rows not yet frozen into a publication.
    pub fn active_rows(&self) -> u64 {
        self.active.rows
    }

    pub fn freeze(&mut self) -> Arc<Frozen> {
        assert!(self.frozen.is_none());
        let frozen = Arc::new(Frozen {
            segment: std::mem::take(&mut self.active),
        });
        self.frozen = Some(frozen.clone());
        frozen
    }

    pub fn is_current(&self, frozen: &Arc<Frozen>) -> bool {
        self.frozen.as_ref().is_some_and(|f| Arc::ptr_eq(f, frozen))
    }

    pub fn retire(&mut self, frozen: &Arc<Frozen>) {
        assert!(self.is_current(frozen));
        self.frozen = None;
    }

    /// Bytes held in RAM and on the spill directory, active plus frozen.
    pub fn bytes(&self) -> (u64, u64) {
        self.active
            .spilled
            .iter()
            .chain(self.frozen.iter().flat_map(|f| f.segment.spilled.iter()))
            .fold(
                (
                    (self.active.ram_bytes
                        + self.frozen.as_ref().map_or(0, |f| f.segment.ram_bytes))
                        as u64,
                    0u64,
                ),
                |(ram, disk), file| (ram, disk + file.bytes),
            )
    }

    /// Committed metadata of a deferred request, so a staged point read can
    /// refetch its body from the WAL. Metadata only; bounded by the source
    /// window of the active plus frozen generations.
    pub fn deferred(&self, index: u64) -> Option<BatchMeta> {
        self.active
            .metas
            .get(&index)
            .or_else(|| {
                self.frozen
                    .as_ref()
                    .and_then(|f| f.segment.metas.get(&index))
            })
            .cloned()
    }
}

fn spill<T: Serialize>(dir: &Path, survivors: &[T]) -> Result<SpillFile> {
    let path = dir.join(format!("{}.jsonl", Uuid::new_v4()));
    let file = std::fs::File::create(&path)
        .with_context(|| format!("create write batch spill {}", path.display()))?;
    // Own the path before writing so an error still removes a partial file.
    let mut owned = SpillFile { path, bytes: 0 };
    let mut writer = BufWriter::new(file);
    for entry in survivors {
        serde_json::to_writer(&mut writer, entry)?;
        writer.write_all(b"\n")?;
    }
    let file = writer.into_inner().map_err(|e| e.into_error())?;
    file.sync_data()?;
    owned.bytes = file.metadata()?.len();
    Ok(owned)
}

fn read_spill(file: &SpillFile) -> Result<Vec<Survivors>> {
    let reader = std::io::BufReader::new(
        std::fs::File::open(&file.path)
            .with_context(|| format!("open write batch spill {}", file.path.display()))?,
    );
    reader
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

/// Rolling stage metrics for the owner batch pipeline.
#[derive(Clone, Debug, Default, Serialize)]
pub struct OwnerBatchObservation {
    pub enabled: bool,
    /// Historical lookups issued by the pipeline and their keys.
    pub lookups: u64,
    pub lookup_keys: u64,
    pub last_keys_per_lookup: u64,
    pub max_keys_per_lookup: u64,
    pub keys_per_lookup: [u64; super::lookup_observation::KEYS_PER_WINDOW_BUCKETS],
    /// Lookups completed within the last 60 seconds at sampling time.
    pub lookups_per_minute: u64,
    /// Lookups that used at least one scalar index on a key column.
    pub indexed_lookups: u64,
    /// Fragments written by owner batch publication and their rows.
    pub fragments: u64,
    pub fragment_rows: u64,
    pub last_rows_per_fragment: u64,
    pub max_rows_per_fragment: u64,
    /// Refetched chunks that were clustered by (session_id, id).
    pub clustered_chunks: u64,
    pub refetched_records: u64,
    pub refetched_bytes: u64,
    /// Current write batch, active plus frozen.
    pub write_batch_rows: u64,
    pub write_batch_ram_bytes: u64,
    pub write_batch_pvc_bytes: u64,
    pub max_write_batch_ram_bytes: u64,
    pub max_write_batch_pvc_bytes: u64,
    /// Cumulative wall time per stage (microseconds). Stages overlap, so
    /// these must not be summed into elapsed time.
    pub collect_us: u64,
    pub lookup_us: u64,
    pub decide_us: u64,
    pub write_us: u64,
}

#[derive(Default)]
pub(crate) struct Metrics {
    pub observed: OwnerBatchObservation,
    recent_lookups: std::collections::VecDeque<Instant>,
}

impl Metrics {
    pub fn lookup(&mut self, keys: usize, elapsed_us: u64, indexed: bool) {
        let o = &mut self.observed;
        let keys = keys as u64;
        o.lookups += 1;
        o.lookup_keys = o.lookup_keys.saturating_add(keys);
        o.last_keys_per_lookup = keys;
        o.max_keys_per_lookup = o.max_keys_per_lookup.max(keys);
        o.keys_per_lookup[super::lookup_observation::keys_bucket(keys)] += 1;
        o.lookup_us = o.lookup_us.saturating_add(elapsed_us);
        o.indexed_lookups += u64::from(indexed);
        let now = Instant::now();
        self.recent_lookups.push_back(now);
        self.trim(now);
    }

    fn trim(&mut self, now: Instant) {
        while self
            .recent_lookups
            .front()
            .is_some_and(|t| now.duration_since(*t) > Duration::from_secs(60))
        {
            self.recent_lookups.pop_front();
        }
    }

    pub fn fragment(&mut self, rows: u64, elapsed_us: u64) {
        let o = &mut self.observed;
        o.fragments += 1;
        o.fragment_rows = o.fragment_rows.saturating_add(rows);
        o.last_rows_per_fragment = rows;
        o.max_rows_per_fragment = o.max_rows_per_fragment.max(rows);
        o.write_us = o.write_us.saturating_add(elapsed_us);
    }

    pub fn write_batch(&mut self, rows: u64, (ram, pvc): (u64, u64)) {
        let o = &mut self.observed;
        o.write_batch_rows = rows;
        o.write_batch_ram_bytes = ram;
        o.write_batch_pvc_bytes = pvc;
        o.max_write_batch_ram_bytes = o.max_write_batch_ram_bytes.max(ram);
        o.max_write_batch_pvc_bytes = o.max_write_batch_pvc_bytes.max(pvc);
    }

    pub fn snapshot(&mut self) -> OwnerBatchObservation {
        self.trim(Instant::now());
        self.observed.lookups_per_minute = self.recent_lookups.len() as u64;
        self.observed.clone()
    }
}

impl Service {
    pub(super) fn ensure_owner_batch_unstarted(&self) -> Result<()> {
        ensure!(
            self.is_single_body(),
            "owner batch pipeline requires full-body ingestion"
        );
        ensure!(
            self.shared.progress.borrow().through
                == self.shared.results.lock().unwrap().base.through
                && self.shared.frames.lock().unwrap().retained() == (0, 0)
                && self
                    .shared
                    .write_batch
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_none_or(|b| b.rows() == 0),
            "configure the owner batch pipeline before applying"
        );
        Ok(())
    }

    /// Enable (or reconfigure) the pipeline before the owner applies.
    pub(super) fn enable_owner_batch(&self, options: Options) -> Result<()> {
        self.ensure_owner_batch_unstarted()?;
        let owner = format!(
            "{}-{}-{}",
            self.ingress.scope.table,
            self.ingress.scope.incarnation,
            self.bucket()
        );
        // Drop the previous batch (and its spill directory) first.
        *self.shared.write_batch.lock().unwrap() = None;
        let batch = WriteBatch::open(options, &owner)?;
        *self.shared.write_batch.lock().unwrap() = Some(batch);
        self.shared.batch_metrics.lock().unwrap().observed.enabled = true;
        Ok(())
    }

    pub(super) fn owner_batch(&self) -> bool {
        self.shared.write_batch.lock().unwrap().is_some()
    }

    pub fn owner_batch_observation(&self) -> OwnerBatchObservation {
        let mut metrics = self.shared.batch_metrics.lock().unwrap();
        if let Some(batch) = self.shared.write_batch.lock().unwrap().as_ref() {
            metrics.write_batch(batch.rows(), batch.bytes());
        }
        metrics.snapshot()
    }

    pub(super) fn record_stage(&self, stage: Stage, started: Instant) {
        let us = started.elapsed().as_micros().try_into().unwrap_or(u64::MAX);
        let mut metrics = self.shared.batch_metrics.lock().unwrap();
        let o = &mut metrics.observed;
        let total = match stage {
            Stage::Collect => &mut o.collect_us,
            Stage::Decide => &mut o.decide_us,
        };
        *total = total.saturating_add(us);
    }

    /// Whether publication should run now because the write batch is large.
    pub(super) fn write_batch_full(&self) -> bool {
        self.shared
            .write_batch
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|b| b.active_rows() >= b.options.target_rows)
    }

    /// Fetch one committed body record, returning `None` on admission pressure.
    pub(super) async fn try_refetch_bodies(
        &self,
        index: u64,
        meta: BatchMeta,
    ) -> Result<Option<(Vec<RecordBatch>, MergeReservation)>> {
        let Some(pending) = prefetch::fetch(self, index, meta).await? else {
            return Ok(None);
        };
        let mixed::Decoded::Creates { bodies, .. } = pending.decoded else {
            bail!("immutable body cannot contain patches");
        };
        Ok(Some((bodies, pending._reservation)))
    }

    /// Stream exactly the frozen survivors in the stored full-body schema,
    /// ordered by `(session_id, id)` across the WHOLE frozen batch.
    ///
    /// Pass 1 refetches one committed record at a time, extracts the keys of
    /// its survivors and releases the body at once. Keys and ordinals form
    /// sorted runs bounded by `spill_bytes`, spilled to the write batch's spill
    /// directory. Pass 2 k-way merges the runs and refetches bodies in key
    /// order, in windows of at most `refetch_bytes` of distinct records. Under
    /// apply-memory pressure, emit the prefix that fits as a smaller Arrow
    /// batch and continue with its suffix in the same sorted fragment.
    pub(super) fn survivor_stream(
        &self,
        frozen: Arc<Frozen>,
        expected: BTreeMap<u64, u32>,
    ) -> Result<SendableRecordBatchStream> {
        let spec = self
            .ingress
            .turn_spec
            .clone()
            .context("missing body spec")?;
        let schema = single_body::storage_schema(&spec)?;
        let (options, dir) = {
            let guard = self.shared.write_batch.lock().unwrap();
            let batch = guard.as_ref().context("owner batch pipeline disabled")?;
            (batch.options.clone(), batch.dir.clone())
        };
        let service = self.clone();
        let value_bytes = self.limits.value_bytes;
        let stream_schema = schema.clone();
        let stream = async_stream::try_stream! {
            let schema = stream_schema;
            // Pass 1: sorted key runs.
            let mut runs = KeyRuns::new(dir.clone(), options.spill_bytes);
            let mut seen = BTreeMap::<u64, u32>::new();
            let mut files = frozen.segment.spilled.iter();
            let mut ram = Some(&frozen.segment.ram);
            loop {
                // Spill files hold older records than the RAM tail.
                let entries = if let Some(file) = files.next() {
                    read_spill(file).map_err(external)?
                } else if let Some(ram) = ram.take() {
                    ram.clone()
                } else {
                    break;
                };
                for survivors in entries {
                    if survivors.rows.is_empty() {
                        continue;
                    }
                    // Holds nothing while waiting, so waiting cannot deadlock.
                    let (bodies, reservation) =
                        refetch_waiting(&service, survivors.index, &survivors.meta).await.map_err(external)?;
                    {
                        let mut metrics = service.shared.batch_metrics.lock().unwrap();
                        metrics.observed.refetched_records += 1;
                        metrics.observed.refetched_bytes += survivors.meta.body_bytes;
                    }
                    let body = spec.body_schema().clone();
                    let mut locate = Locate::new(&bodies);
                    for &row in &survivors.rows {
                        let (b, local) = locate.at(row).map_err(external)?;
                        let original = RecordBatch::try_new(body.clone(), bodies[b].columns().to_vec())
                            .map_err(|e| external(e.into()))?;
                        let id = spec.row_key(&original, local, value_bytes).map_err(external)?;
                        let routing = spec.routing_key(&id).map_err(external)?;
                        runs.push(SortKey { routing, id, index: survivors.index, row }).map_err(external)?;
                    }
                    *seen.entry(survivors.index).or_default() += survivors.rows.len() as u32;
                    drop(bodies);
                    drop(reservation);
                }
            }
            if seen != expected {
                Err::<(), _>(external(anyhow::anyhow!(
                    "owner write batch differs from frozen first-create rows"
                )))?;
            }
            // Pass 2: merge runs, refetch bodies in key order.
            let mut merge = runs.finish().map_err(external)?;
            let mut pending: Option<SortKey> = None;
            loop {
                let mut window = Vec::<(u64, u32)>::new();
                let mut records = BTreeMap::<u64, BatchMeta>::new();
                let mut bytes = 0u64;
                loop {
                    let next = match pending.take() {
                        Some(key) => Some(key),
                        None => merge.next().map_err(external)?,
                    };
                    let Some(key) = next else { break };
                    if !records.contains_key(&key.index) {
                        let meta = frozen
                            .segment
                            .metas
                            .get(&key.index)
                            .cloned()
                            .context("survivor metadata missing")
                            .map_err(external)?;
                        if !records.is_empty() && bytes + meta.body_bytes > options.refetch_bytes {
                            pending = Some(key);
                            break;
                        }
                        bytes += meta.body_bytes;
                        records.insert(key.index, meta);
                    }
                    window.push((key.index, key.row));
                }
                if window.is_empty() {
                    break;
                }
                let mut start = 0;
                while start < window.len() {
                    let (fetched, admitted) = fetch_window_prefix(&service, &records, &window[start..])
                        .await
                        .map_err(external)?;
                    let end = start + admitted;
                    let batch = gather(&schema, &fetched, &window[start..end]).map_err(external)?;
                    service.shared.batch_metrics.lock().unwrap().observed.clustered_chunks += 1;
                    drop(fetched);
                    start = end;
                    yield batch;
                }
            }
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

/// One survivor's global sort key and source ordinal.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct SortKey {
    routing: String,
    id: String,
    index: u64,
    row: u32,
}
impl SortKey {
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.routing.len() + self.id.len()
    }
}

/// Sorted key runs: in RAM up to `limit` bytes, then spilled to `dir`.
struct KeyRuns {
    dir: Option<PathBuf>,
    limit: usize,
    current: Vec<SortKey>,
    current_bytes: usize,
    files: Vec<SpillFile>,
}
impl KeyRuns {
    fn new(dir: Option<PathBuf>, limit: usize) -> Self {
        Self {
            dir,
            limit,
            current: Vec::new(),
            current_bytes: 0,
            files: Vec::new(),
        }
    }
    fn push(&mut self, key: SortKey) -> Result<()> {
        self.current_bytes += key.bytes();
        self.current.push(key);
        if self.current_bytes >= self.limit {
            if let Some(dir) = &self.dir {
                let mut run = std::mem::take(&mut self.current);
                self.current_bytes = 0;
                run.sort_unstable();
                self.files.push(spill(dir, &run)?);
            }
        }
        Ok(())
    }
    fn finish(mut self) -> Result<KeyMerge> {
        self.current.sort_unstable();
        let mut sources = Vec::with_capacity(self.files.len() + 1);
        for file in &self.files {
            let reader = std::io::BufReader::new(
                std::fs::File::open(&file.path)
                    .with_context(|| format!("open key run {}", file.path.display()))?,
            );
            sources.push(RunSource::File(reader.lines()));
        }
        sources.push(RunSource::Ram(
            std::mem::take(&mut self.current).into_iter(),
        ));
        let mut merge = KeyMerge {
            _files: self.files,
            sources,
            heap: std::collections::BinaryHeap::new(),
        };
        for at in 0..merge.sources.len() {
            merge.refill(at)?;
        }
        Ok(merge)
    }
}

enum RunSource {
    File(std::io::Lines<std::io::BufReader<std::fs::File>>),
    Ram(std::vec::IntoIter<SortKey>),
}

/// K-way merge of sorted runs; run files are removed when it drops.
struct KeyMerge {
    _files: Vec<SpillFile>,
    sources: Vec<RunSource>,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(SortKey, usize)>>,
}
impl KeyMerge {
    fn refill(&mut self, at: usize) -> Result<()> {
        let next = match &mut self.sources[at] {
            RunSource::File(lines) => match lines.next() {
                Some(line) => Some(serde_json::from_str(&line?)?),
                None => None,
            },
            RunSource::Ram(keys) => keys.next(),
        };
        if let Some(key) = next {
            self.heap.push(std::cmp::Reverse((key, at)));
        }
        Ok(())
    }
    fn next(&mut self) -> Result<Option<SortKey>> {
        let Some(std::cmp::Reverse((key, at))) = self.heap.pop() else {
            return Ok(None);
        };
        self.refill(at)?;
        Ok(Some(key))
    }
}

/// Map a record-level row ordinal to (batch, local row).
struct Locate {
    ends: Vec<u32>,
}
impl Locate {
    fn new(bodies: &[RecordBatch]) -> Self {
        let mut end = 0u32;
        Self {
            ends: bodies
                .iter()
                .map(|b| {
                    end += b.num_rows() as u32;
                    end
                })
                .collect(),
        }
    }
    fn at(&mut self, row: u32) -> Result<(usize, usize)> {
        let b = self.ends.partition_point(|end| *end <= row);
        ensure!(
            b < self.ends.len(),
            "write batch survivor exceeds its source rows"
        );
        let start = if b == 0 { 0 } else { self.ends[b - 1] };
        Ok((b, (row - start) as usize))
    }
}

async fn refetch_waiting(
    service: &Service,
    index: u64,
    meta: &BatchMeta,
) -> Result<(Vec<RecordBatch>, MergeReservation)> {
    loop {
        if let Some(fetched) = service.try_refetch_bodies(index, meta.clone()).await? {
            return Ok(fetched);
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Fetch a contiguous prefix of the sorted rows, keeping the decoded bodies
/// for that prefix until the caller has formed its output batch. Only the
/// first record can wait for admission, when the prefix holds no reservation.
async fn fetch_window_prefix(
    service: &Service,
    records: &BTreeMap<u64, BatchMeta>,
    sorted: &[(u64, u32)],
) -> Result<(BTreeMap<u64, (Vec<RecordBatch>, MergeReservation)>, usize)> {
    let mut held = BTreeMap::new();
    let mut admitted = 0;
    let mut fetched_bytes = 0;
    for &(index, _) in sorted {
        if let std::collections::btree_map::Entry::Vacant(entry) = held.entry(index) {
            let meta = records
                .get(&index)
                .context("sorted source record missing")?;
            let fetched = if admitted == 0 {
                refetch_waiting(service, index, meta).await?
            } else {
                let Some(fetched) = service.try_refetch_bodies(index, meta.clone()).await? else {
                    break;
                };
                fetched
            };
            fetched_bytes += meta.body_bytes;
            entry.insert(fetched);
        }
        admitted += 1;
    }
    let mut metrics = service.shared.batch_metrics.lock().unwrap();
    metrics.observed.refetched_records += held.len() as u64;
    metrics.observed.refetched_bytes += fetched_bytes;
    Ok((held, admitted))
}

/// Copy the window's rows, in key order, into one storage-schema batch.
fn gather(
    schema: &SchemaRef,
    fetched: &BTreeMap<u64, (Vec<RecordBatch>, MergeReservation)>,
    window: &[(u64, u32)],
) -> Result<RecordBatch> {
    let mut batches = Vec::<RecordBatch>::new();
    let mut base = BTreeMap::<u64, (usize, Locate)>::new();
    for (&index, (bodies, _)) in fetched {
        base.insert(index, (batches.len(), Locate::new(bodies)));
        for batch in bodies {
            batches.push(RecordBatch::try_new(
                schema.clone(),
                batch.columns().to_vec(),
            )?);
        }
    }
    let order = window
        .iter()
        .map(|(index, row)| {
            let (first, locate) = base.get_mut(index).context("window record missing")?;
            let (b, local) = locate.at(*row)?;
            Ok((*first + b, local))
        })
        .collect::<Result<Vec<_>>>()?;
    let refs = batches.iter().collect::<Vec<_>>();
    Ok(interleave_record_batch(&refs, &order)?)
}

pub(super) enum Stage {
    Collect,
    Decide,
}

fn external(error: anyhow::Error) -> DataFusionError {
    DataFusionError::External(error.into())
}
