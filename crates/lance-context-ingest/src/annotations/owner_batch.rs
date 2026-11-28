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
//!    streams it back by refetching bodies from the WAL for survivors only,
//!    clusters each bounded chunk by `(session_id, id)`, and writes ONE
//!    fragment per published batch.
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
    /// publication. Each such chunk is clustered independently.
    pub refetch_bytes: u64,
    /// Cluster a refetched chunk by `(session_id, id)` once it holds at
    /// least this many rows; smaller chunks keep WAL order.
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
    pub fn record(&mut self, survivors: Survivors) -> Result<()> {
        self.active.rows += survivors.rows.len() as u64;
        if !survivors.rows.is_empty() {
            self.active.metas.insert(survivors.index, survivors.meta.clone());
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
            .or_else(|| self.frozen.as_ref().and_then(|f| f.segment.metas.get(&index)))
            .cloned()
    }
}

fn spill(dir: &Path, survivors: &[Survivors]) -> Result<SpillFile> {
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

    /// Refetch and decode one committed WAL request for publication or a
    /// staged point read. Waits for apply memory rather than refusing: the
    /// caller holds no ordered gate, and the publisher must make progress.
    pub(super) async fn refetch_bodies(
        &self,
        index: u64,
        meta: BatchMeta,
    ) -> Result<(Vec<RecordBatch>, MergeReservation)> {
        loop {
            if let Some(pending) = prefetch::fetch(self, index, meta.clone()).await? {
                let mixed::Decoded::Creates { bodies, .. } = pending.decoded else {
                    bail!("immutable body cannot contain patches")
                };
                return Ok((bodies, pending._reservation));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Stream exactly the frozen survivors in the stored full-body schema.
    /// Bodies are refetched from the WAL in bounded chunks; each chunk at or
    /// above `cluster_rows` is regrouped by `(session_id, id)` before it is
    /// handed to the single-fragment writer.
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
        let options = self
            .shared
            .write_batch
            .lock()
            .unwrap()
            .as_ref()
            .context("owner batch pipeline disabled")?
            .options
            .clone();
        let service = self.clone();
        let value_bytes = self.limits.value_bytes;
        let stream_schema = schema.clone();
        let stream = async_stream::try_stream! {
            let schema = stream_schema;
            let mut seen = BTreeMap::<u64, u32>::new();
            let mut chunk = Vec::<(Survivors, Vec<RecordBatch>, MergeReservation)>::new();
            let mut chunk_bytes = 0u64;
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
                    // Never wait for apply memory while this chunk holds
                    // some: flush it first so its reservations are released.
                    let (bodies, reservation) = loop {
                        let fetched = prefetch::fetch(&service, survivors.index, survivors.meta.clone())
                            .await
                            .map_err(external)?;
                        if let Some(pending) = fetched {
                            let mixed::Decoded::Creates { bodies, .. } = pending.decoded else {
                                Err::<(), _>(external(anyhow::anyhow!("immutable body cannot contain patches")))?;
                                unreachable!()
                            };
                            break (bodies, pending._reservation);
                        }
                        if chunk.is_empty() {
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        } else {
                            let batch = cluster(&service, &spec, &schema, std::mem::take(&mut chunk), options.cluster_rows, value_bytes).map_err(external)?;
                            chunk_bytes = 0;
                            if let Some(batch) = batch { yield batch; }
                        }
                    };
                    {
                        let mut metrics = service.shared.batch_metrics.lock().unwrap();
                        metrics.observed.refetched_records += 1;
                        metrics.observed.refetched_bytes += survivors.meta.body_bytes;
                    }
                    *seen.entry(survivors.index).or_default() += survivors.rows.len() as u32;
                    chunk_bytes += survivors.meta.body_bytes;
                    chunk.push((survivors, bodies, reservation));
                    if chunk_bytes >= options.refetch_bytes {
                        let batch = cluster(&service, &spec, &schema, std::mem::take(&mut chunk), options.cluster_rows, value_bytes).map_err(external)?;
                        chunk_bytes = 0;
                        if let Some(batch) = batch { yield batch; }
                    }
                }
            }
            let batch = cluster(&service, &spec, &schema, chunk, options.cluster_rows, value_bytes).map_err(external)?;
            if let Some(batch) = batch { yield batch; }
            if seen != expected {
                Err::<(), _>(external(anyhow::anyhow!(
                    "owner write batch differs from frozen first-create rows"
                )))?;
            }
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

pub(super) enum Stage {
    Collect,
    Decide,
}

fn external(error: anyhow::Error) -> DataFusionError {
    DataFusionError::External(error.into())
}

/// Select survivors from one refetched chunk and, when large enough, group
/// them by `(session_id, id)`. This is a per-chunk sort, not a global one.
fn cluster(
    service: &Service,
    spec: &crate::turn_ingestion::TurnIngestionSpec,
    schema: &SchemaRef,
    chunk: Vec<(Survivors, Vec<RecordBatch>, MergeReservation)>,
    cluster_rows: usize,
    value_bytes: usize,
) -> Result<Option<RecordBatch>> {
    let mut selected = Vec::<RecordBatch>::new();
    for (survivors, bodies, _reservation) in &chunk {
        let mut remaining = survivors.rows.iter().copied().peekable();
        let mut offset = 0u32;
        for batch in bodies {
            let end = offset + batch.num_rows() as u32;
            let mut run: Option<(u32, u32)> = None;
            while let Some(row) = remaining.next_if(|row| *row < end) {
                match &mut run {
                    Some((_, last)) if row == *last + 1 => *last = row,
                    _ => {
                        if let Some((first, last)) = run.take() {
                            selected.push(
                                batch.slice((first - offset) as usize, (last - first + 1) as usize),
                            );
                        }
                        run = Some((row, row));
                    }
                }
            }
            if let Some((first, last)) = run {
                selected.push(batch.slice((first - offset) as usize, (last - first + 1) as usize));
            }
            offset = end;
        }
        ensure!(
            remaining.next().is_none(),
            "write batch survivor exceeds its source rows"
        );
    }
    let rows = selected.iter().map(RecordBatch::num_rows).sum::<usize>();
    if rows == 0 {
        return Ok(None);
    }
    let selected = selected
        .into_iter()
        .map(|batch| Ok(RecordBatch::try_new(schema.clone(), batch.columns().to_vec())?))
        .collect::<Result<Vec<_>>>()?;
    let mut order = Vec::with_capacity(rows);
    for (b, batch) in selected.iter().enumerate() {
        for row in 0..batch.num_rows() {
            order.push((b, row));
        }
    }
    let clustered = rows >= cluster_rows;
    if clustered {
        let body = spec.body_schema().clone();
        let mut keyed = order
            .into_iter()
            .map(|(b, row)| {
                let original = RecordBatch::try_new(body.clone(), selected[b].columns().to_vec())?;
                let id = spec.row_key(&original, row, value_bytes)?;
                Ok(((spec.routing_key(&id)?, id), (b, row)))
            })
            .collect::<Result<Vec<_>>>()?;
        keyed.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        order = keyed.into_iter().map(|(_, at)| at).collect();
        service
            .shared
            .batch_metrics
            .lock()
            .unwrap()
            .observed
            .clustered_chunks += 1;
    }
    let refs = selected.iter().collect::<Vec<_>>();
    // Copies only the surviving rows of this chunk; the refetched bodies and
    // their reservations are released when `chunk` drops below.
    let batch = interleave_record_batch(&refs, &order)?;
    drop(chunk);
    Ok(Some(batch))
}
