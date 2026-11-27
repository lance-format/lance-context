//! Reusable commit-ack streaming producer for generic turn tables.
//!
//! Callers hand in turns (Arrow batches of the table's `TurnIngestionSpec`
//! body schema). Rows are routed by session bucket into ONE shared turn buffer
//! that spans every shard and lane. The buffer holds a fixed RAM budget charged
//! to the caller's memory pool, spills to a caller-supplied directory (a PVC)
//! when RAM is full, and backpressures `send` when the spill budget is full too.
//!
//! Each lane owns a fixed memory reservation for one in-flight record. A lane
//! pulls turns for its shard until the global `max_rows`/`max_record_bytes`
//! cap is reached, commits them (`x-lance-context-ack: commit`), and pulls again
//! as soon as the commit ACK arrives. Applied outcomes are joined only at
//! `fence()`, which waits `/applied` on every lane touched since the previous
//! fence and returns per-shard Applied cuts.
//!
//! A turn batch larger than the cap goes through the oversize split path: it is
//! cut into cap-sized slices before admission, so it is never rejected for its
//! size and can never wait for buffer space larger than the budget. A single
//! row that still encodes above `max_record_bytes` cannot be represented as one
//! WAL record; it fails the lane with [`RowExceedsRecord`].
//!
//! Neither a commit ACK nor a fence result is a source checkpoint; callers
//! decide what an Applied cut means for their own cursor.
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use anyhow::{ensure, Context, Result};
use arrow_array::{RecordBatch, StringArray, UInt32Array};
use bytes::Bytes;
use lance_context_core::annotation_staging::Limits;
use lance_context_wal::{
    placement::Layout,
    replication::runtime::client::payload_memory::{Pool, Reservation},
};

use crate::{
    annotations::{MutationResult, Shard},
    http::{
        annotations::{CommitReply, MutationReply},
        journal::DurableMutationClient,
    },
    turn_ingestion::TurnIngestionSpec,
    TableBinding,
};

mod buffer;
pub mod key_cache;
mod lane;
mod metrics;

use buffer::Buffer;
pub use key_cache::AppliedKeyCache;
pub use metrics::{Metrics, MetricsSnapshot};

/// One canonical producer lane of a physical shard.
pub struct Lane {
    pub client: Arc<DurableMutationClient>,
    /// Client-local lane key passed to `commit`/`wait_applied`.
    pub key: usize,
}

/// Lanes and binding for one provisioned physical shard.
pub struct ShardLanes {
    pub binding: TableBinding,
    pub lanes: Vec<Lane>,
}

/// Classifies an error as retryable network ambiguity. Non-transient errors
/// fail the producer; the canonical journal keeps the exact staged intent.
pub type Transient = Arc<dyn Fn(&anyhow::Error) -> bool + Send + Sync>;

pub struct Config {
    pub spec: TurnIngestionSpec,
    pub limits: Limits,
    pub layout: Arc<Layout>,
    pub shards: BTreeMap<u32, ShardLanes>,
    /// Caller pool charged for buffered RAM turns and every lane's fixed slot.
    pub memory: Arc<dyn Pool>,
    /// RAM budget of the shared turn buffer (excludes lane slots).
    pub buffer_ram_bytes: usize,
    /// Spill directory on a PVC; created if absent, owned by this producer.
    pub spill_dir: PathBuf,
    /// Spill byte budget; 0 disables spill (RAM full backpressures directly).
    pub spill_bytes: u64,
    /// Concurrent `/applied` waits at a fence.
    pub applied_parallelism: usize,
    pub transient: Transient,
    /// Optional applied-key cache consulted while splitting turns.
    pub key_cache: Option<Arc<AppliedKeyCache>>,
}

/// Applied outcome of one lane's latest record at a fence. Records on one lane
/// commit in sequence, so its latest Applied index covers all earlier ones; an
/// earlier rejection of the same producer surfaces as HTTP 409.
#[derive(Clone, Debug)]
pub struct LaneApplied {
    pub lane: usize,
    pub records: u64,
    /// Submitted rows, including duplicate keys; not inserted rows.
    pub rows: usize,
    pub reply: MutationReply,
}

/// Per-shard Applied cut joined at a fence.
#[derive(Clone, Debug)]
pub struct AppliedCut {
    pub shard: Shard,
    pub lanes: Vec<LaneApplied>,
}

impl AppliedCut {
    pub fn rejected(&self) -> Option<&LaneApplied> {
        self.lanes
            .iter()
            .find(|lane| lane.reply.completion.result != MutationResult::Applied)
    }
}

/// All shards touched since the previous fence. Untouched shards are absent.
#[derive(Clone, Debug, Default)]
pub struct Fence {
    pub shards: BTreeMap<u32, AppliedCut>,
}

impl Fence {
    pub fn rejected(&self) -> Option<(&AppliedCut, &LaneApplied)> {
        self.shards
            .values()
            .find_map(|cut| cut.rejected().map(|lane| (cut, lane)))
    }
}

/// A single row whose encoded record exceeds `max_record_bytes`.
#[derive(Debug)]
pub struct RowExceedsRecord {
    pub shard: u32,
    pub encoded_bytes: usize,
    pub max_record_bytes: usize,
}

impl std::fmt::Display for RowExceedsRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "single turn on shard {} encodes to {} bytes, above max_record_bytes {}",
            self.shard, self.encoded_bytes, self.max_record_bytes
        )
    }
}
impl std::error::Error for RowExceedsRecord {}

/// Latest commit on one lane whose applied outcome is not yet joined.
struct Pending {
    latest: CommitReply,
    records: u64,
    rows: usize,
    // Keys committed since the last fence; resolved to Applied/Rejected there.
    keys: Vec<key_cache::Key>,
}

struct LaneSlot {
    shard: Shard,
    index: usize,
    client: Arc<DurableMutationClient>,
    key: usize,
    pending: Mutex<Option<Pending>>,
}

struct Shared {
    spec: TurnIngestionSpec,
    limits: Limits,
    layout: Arc<Layout>,
    tenant: String,
    table: String,
    max_rows: usize,
    max_bytes: usize,
    buffer: Buffer,
    metrics: Arc<Metrics>,
    transient: Transient,
    stop: AtomicBool,
    key_cache: Option<Arc<AppliedKeyCache>>,
    key_columns: Vec<String>,
}

pub struct StreamingProducer {
    shared: Arc<Shared>,
    lanes: Vec<Arc<LaneSlot>>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    // Serializes send/fence so per-shard buffer order is the caller's order.
    sender: tokio::sync::Mutex<()>,
    applied_parallelism: usize,
}

impl StreamingProducer {
    /// Reserve every lane's fixed slot, then start one task per lane. Resume
    /// recovered journal intents on the clients before opening.
    pub fn open(config: Config) -> Result<Self> {
        ensure!(!config.shards.is_empty(), "streaming producer has no shards");
        ensure!(config.applied_parallelism > 0, "zero applied parallelism");
        let first = &config.shards.values().next().unwrap().binding;
        let (tenant, table) = (first.tenant.clone(), first.qualified_table.clone());
        let mut max_rows = usize::MAX;
        let mut max_bytes = usize::MAX;
        for (bucket, shard) in &config.shards {
            let binding = &shard.binding;
            ensure!(
                binding.tenant == tenant && binding.qualified_table == table,
                "streaming shards span table families"
            );
            ensure!(!shard.lanes.is_empty(), "shard {bucket} has no lanes");
            ensure!(
                binding.max_rows > 0 && binding.max_record_bytes > 0,
                "empty request limits"
            );
            config.layout.group_for_bucket(*bucket)?;
            max_rows = max_rows.min(binding.max_rows as usize);
            max_bytes = max_bytes.min(binding.max_record_bytes as usize);
        }
        ensure!(
            config.buffer_ram_bytes >= max_bytes,
            "buffer RAM budget below one record"
        );
        let metrics = Arc::new(Metrics::default());
        let buffer = Buffer::new(
            config.memory.clone(),
            config.buffer_ram_bytes,
            config.spill_dir,
            config.spill_bytes,
            metrics.clone(),
        )?;
        for bucket in config.shards.keys() {
            buffer.register(*bucket);
        }
        let key_columns = config.spec.primary_key().to_vec();
        let shared = Arc::new(Shared {
            spec: config.spec,
            limits: config.limits,
            layout: config.layout,
            tenant,
            table,
            max_rows,
            max_bytes,
            buffer,
            metrics,
            transient: config.transient,
            stop: AtomicBool::new(false),
            key_cache: config.key_cache,
            key_columns,
        });
        let mut lanes = Vec::new();
        let mut tasks = Vec::new();
        for (bucket, shard) in config.shards {
            for (index, lane) in shard.lanes.into_iter().enumerate() {
                // One in-flight record: gathered rows plus their encoded body.
                let slot_bytes = max_bytes.saturating_mul(2);
                let mut reservation = config
                    .memory
                    .try_reserve(slot_bytes)?
                    .context("lane slot memory unavailable")?;
                ensure!(reservation.bytes() >= slot_bytes, "undersized lane slot");
                reservation.set_backed_bytes(slot_bytes);
                let slot = Arc::new(LaneSlot {
                    shard: Shard { bucket },
                    index,
                    client: lane.client,
                    key: lane.key,
                    pending: Mutex::new(None),
                });
                tasks.push(tokio::spawn(lane::run(
                    shared.clone(),
                    slot.clone(),
                    reservation,
                )));
                lanes.push(slot);
            }
        }
        Ok(Self {
            shared,
            lanes,
            tasks,
            sender: tokio::sync::Mutex::new(()),
            applied_parallelism: config.applied_parallelism,
        })
    }

    pub fn metrics(&self) -> MetricsSnapshot {
        self.shared.metrics.snapshot()
    }

    /// Route one turn batch into the shared buffer. Returns once every row is
    /// buffered (RAM or spill); waits while both budgets are full.
    pub async fn send(&self, batch: RecordBatch) -> Result<()> {
        let _order = self.sender.lock().await;
        ensure!(
            batch.schema() == *self.shared.spec.body_schema(),
            "foreign turn body schema"
        );
        let batch = self.drop_known(batch).await?;
        if batch.num_rows() == 0 {
            return Ok(());
        }
        for (bucket, rows) in self.route(&batch)? {
            let part = arrow_select::take::take_record_batch(&batch, &rows)?;
            self.admit(bucket, part).await?;
        }
        Ok(())
    }

    /// Drop rows whose key the applied-key cache already holds, before they
    /// are routed or buffered. Within-batch duplicates are left to the owner.
    async fn drop_known(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let Some(cache) = self.shared.key_cache.clone() else {
            return Ok(batch);
        };
        let keys = key_cache::batch_keys(&batch, &self.shared.key_columns)?;
        let found = tokio::task::spawn_blocking(move || cache.contains_many(&keys)).await??;
        let hits = found.iter().filter(|hit| **hit).count();
        if hits == 0 {
            return Ok(batch);
        }
        self.shared.metrics.dropped_known(hits);
        let keep = arrow_array::BooleanArray::from(found.iter().map(|hit| !hit).collect::<Vec<_>>());
        Ok(arrow_select::filter::filter_record_batch(&batch, &keep)?)
    }

    fn route(&self, batch: &RecordBatch) -> Result<BTreeMap<u32, UInt32Array>> {
        let shared = &self.shared;
        let sessions = batch
            .column_by_name(shared.spec.routing_column())
            .context("routing column absent")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("invalid routing column")?;
        let mut groups = BTreeMap::<u32, Vec<u32>>::new();
        for row in 0..batch.num_rows() {
            let bucket = shared
                .layout
                .session_bucket(&shared.tenant, &shared.table, sessions.value(row))?;
            ensure!(
                shared.buffer.has_shard(bucket),
                "turn routed to unconfigured shard {bucket}"
            );
            groups.entry(bucket).or_default().push(row.try_into()?);
        }
        Ok(groups
            .into_iter()
            .map(|(bucket, rows)| (bucket, UInt32Array::from(rows)))
            .collect())
    }

    /// Admit one shard's rows, cutting anything above the record cap into
    /// cap-sized slices first (the oversize split path).
    async fn admit(&self, bucket: u32, part: RecordBatch) -> Result<()> {
        let shared = &self.shared;
        let rows = part.num_rows();
        if rows == 0 {
            return Ok(());
        }
        let bytes = part.get_array_memory_size();
        if rows <= shared.max_rows && bytes <= shared.max_bytes {
            return shared.buffer.push(bucket, part).await;
        }
        shared.metrics.oversize_turn();
        // Proportional row estimate, then halve until each slice fits.
        let mut offset = 0;
        while offset < rows {
            let mut count = (rows - offset).min(shared.max_rows);
            let slice = loop {
                let slice = part.slice(offset, count);
                if count == 1 || slice_bytes(&slice)? <= shared.max_bytes {
                    break slice;
                }
                count /= 2;
            };
            // Compact so the slice does not retain the whole parent buffer.
            let compact = arrow_select::concat::concat_batches(&slice.schema(), [&slice])?;
            shared.metrics.oversize_slice();
            shared.buffer.push(bucket, compact).await?;
            offset += count;
        }
        Ok(())
    }

    /// Wait until every buffered turn is committed, then wait `/applied` on
    /// every lane touched since the previous fence. Returns the per-shard
    /// Applied cuts; a Rejected outcome is returned, not raised, so the caller
    /// decides. Pending state is cleared only for lanes that resolved.
    pub async fn fence(&self) -> Result<Fence> {
        let _order = self.sender.lock().await;
        self.shared.buffer.drained().await?;
        let touched: Vec<_> = self
            .lanes
            .iter()
            .filter_map(|slot| {
                let pending = slot.pending.lock().unwrap();
                pending
                    .as_ref()
                    .map(|p| (slot.clone(), p.latest.clone(), p.records, p.rows))
                // keys stay in the slot until the outcome is known
            })
            .collect();
        use futures::StreamExt;
        let results: Vec<Result<(Arc<LaneSlot>, LaneApplied)>> =
            futures::stream::iter(touched.into_iter().map(|(slot, latest, records, rows)| {
                let shared = self.shared.clone();
                async move {
                    let started = std::time::Instant::now();
                    let reply = lane::retry(&shared, || {
                        slot.client.wait_applied(slot.key, &latest)
                    })
                    .await;
                    shared.metrics.applied_wait(started.elapsed(), reply.is_ok());
                    let reply = reply?;
                    Ok((
                        slot.clone(),
                        LaneApplied {
                            lane: slot.index,
                            records,
                            rows,
                            reply,
                        },
                    ))
                }
            }))
            .buffer_unordered(self.applied_parallelism)
            .collect()
            .await;
        let mut fence = Fence::default();
        for result in results {
            let (slot, applied) = result?;
            // Nothing committed after the snapshot: sender lock held.
            let keys = slot.pending.lock().unwrap().take().map(|p| p.keys).unwrap_or_default();
            if let Some(cache) = self.shared.key_cache.clone() {
                let applied_ok = applied.reply.completion.result == MutationResult::Applied;
                tokio::task::spawn_blocking(move || {
                    if applied_ok {
                        cache.insert(&keys, key_cache::Reason::Applied)
                    } else {
                        // The latest outcome covers the lane's records since
                        // the previous fence: drop their commit-time inserts.
                        cache.remove(&keys)
                    }
                })
                .await??;
            }
            fence
                .shards
                .entry(slot.shard.bucket)
                .or_insert_with(|| AppliedCut {
                    shard: slot.shard,
                    lanes: Vec::new(),
                })
                .lanes
                .push(applied);
        }
        for cut in fence.shards.values_mut() {
            cut.lanes.sort_by_key(|lane| lane.lane);
        }
        Ok(fence)
    }

    /// Stop lanes after their current record and return the canonical clients
    /// for shutdown. Buffered, uncommitted turns are dropped (spill files too):
    /// call `fence` first for a clean stop.
    pub async fn close(mut self) -> Result<Vec<Arc<DurableMutationClient>>> {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.buffer.close();
        let mut first_error = None;
        for task in self.tasks.drain(..) {
            if let Err(error) = task.await {
                first_error.get_or_insert_with(|| anyhow::anyhow!("lane task: {error}"));
            }
        }
        self.shared.buffer.clear_spill();
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(self.lanes.iter().map(|slot| slot.client.clone()).collect())
    }
}

impl Drop for StreamingProducer {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.buffer.close();
    }
}

fn slice_bytes(batch: &RecordBatch) -> Result<usize> {
    let mut bytes = 0usize;
    for column in batch.columns() {
        bytes = bytes.saturating_add(column.to_data().get_slice_memory_size()?);
    }
    Ok(bytes)
}

fn encode(shared: &Shared, shard: Shard, body: &RecordBatch) -> Result<Bytes> {
    let batch = crate::annotations::single_body::batch(&shared.spec, &shared.limits, shard, body)?;
    let mut out = Vec::new();
    let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut out, batch.schema().as_ref())?;
    writer.write(&batch)?;
    writer.finish()?;
    drop(writer);
    Ok(Bytes::from(out))
}

const BACKOFF_MIN: Duration = Duration::from_millis(100);
const BACKOFF_MAX: Duration = Duration::from_secs(5);
