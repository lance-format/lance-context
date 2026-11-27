//! One turn buffer shared by every shard and lane. RAM entries hold a charge
//! from the caller pool; when the RAM budget (or the pool) is full, entries
//! spill to Arrow IPC files in the spill directory; when the spill budget is
//! full too, `push` waits. Per-shard FIFO order is preserved.
use std::{
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{bail, ensure, Context, Result};
use arrow_array::RecordBatch;
use lance_context_wal::replication::runtime::client::payload_memory::{Pool, Reservation};
use tokio::sync::Notify;

use super::metrics::Metrics;

enum Entry {
    Ram {
        batch: RecordBatch,
        bytes: usize,
        _charge: Box<dyn Reservation>,
    },
    Spilled {
        path: PathBuf,
        rows: usize,
        bytes: usize,
        file_bytes: u64,
    },
}

impl Entry {
    fn rows(&self) -> usize {
        match self {
            Entry::Ram { batch, .. } => batch.num_rows(),
            Entry::Spilled { rows, .. } => *rows,
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Entry::Ram { bytes, .. } | Entry::Spilled { bytes, .. } => *bytes,
        }
    }
}

#[derive(Default)]
struct State {
    queues: BTreeMap<u32, VecDeque<Entry>>,
    ram_bytes: usize,
    spill_bytes: u64,
    turns: usize,
    // Records pulled by a lane and not yet committed (or failed).
    in_flight: usize,
    next_file: u64,
    failed: Option<String>,
    closed: bool,
}

pub(super) struct Buffer {
    memory: Arc<dyn Pool>,
    ram_budget: usize,
    dir: PathBuf,
    spill_budget: u64,
    metrics: Arc<Metrics>,
    state: Mutex<State>,
    // Wakes lanes (new turns) and senders/fences (space freed, drained).
    pushed: Notify,
    freed: Notify,
}

/// Turns pulled for one record; `done` must be called after commit.
pub(super) struct Pulled {
    pub batches: Vec<RecordBatch>,
    pub rows: usize,
}

impl Buffer {
    pub fn new(
        memory: Arc<dyn Pool>,
        ram_budget: usize,
        dir: PathBuf,
        spill_budget: u64,
        metrics: Arc<Metrics>,
    ) -> Result<Self> {
        if spill_budget > 0 {
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("create spill dir {}", dir.display()))?;
            // Spill is a cache of uncommitted input; a restart replays source.
            for entry in std::fs::read_dir(&dir)? {
                let path = entry?.path();
                if path.extension().is_some_and(|ext| ext == "spill") {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        Ok(Self {
            memory,
            ram_budget,
            dir,
            spill_budget,
            metrics,
            state: Mutex::default(),
            pushed: Notify::new(),
            freed: Notify::new(),
        })
    }

    pub fn register(&self, bucket: u32) {
        self.state.lock().unwrap().queues.entry(bucket).or_default();
    }

    pub fn has_shard(&self, bucket: u32) -> bool {
        self.state.lock().unwrap().queues.contains_key(&bucket)
    }

    pub async fn push(&self, bucket: u32, batch: RecordBatch) -> Result<()> {
        let bytes = batch.get_array_memory_size();
        let rows = batch.num_rows();
        loop {
            let freed = self.freed.notified();
            tokio::pin!(freed);
            freed.as_mut().enable();
            let spill = {
                let mut state = self.state.lock().unwrap();
                check(&state)?;
                if state.ram_bytes + bytes <= self.ram_budget {
                    if let Some(mut charge) = self.memory.try_reserve(bytes)? {
                        charge.set_backed_bytes(bytes);
                        state.ram_bytes += bytes;
                        state.turns += rows;
                        state.queues.get_mut(&bucket).context("unregistered shard")?.push_back(
                            Entry::Ram { batch, bytes, _charge: charge },
                        );
                        self.metrics.buffered(state.ram_bytes, state.spill_bytes, state.turns);
                        drop(state);
                        self.pushed.notify_waiters();
                        return Ok(());
                    }
                }
                // Encoded IPC is close to the in-memory size; reserve that.
                if self.spill_budget > 0 && state.spill_bytes + bytes as u64 <= self.spill_budget {
                    state.spill_bytes += bytes as u64;
                    state.next_file += 1;
                    Some(self.dir.join(format!("{:016}.spill", state.next_file)))
                } else {
                    None
                }
            };
            if let Some(path) = spill {
                let file_bytes = match write_spill(path.clone(), batch.clone()).await {
                    Ok(written) => written,
                    Err(error) => {
                        self.state.lock().unwrap().spill_bytes -= bytes as u64;
                        let _ = std::fs::remove_file(&path);
                        self.metrics.spill_error();
                        return Err(error);
                    }
                };
                let mut state = self.state.lock().unwrap();
                // Settle the reservation to the actual file size.
                state.spill_bytes = state.spill_bytes - bytes as u64 + file_bytes;
                state.turns += rows;
                state.queues.get_mut(&bucket).context("unregistered shard")?.push_back(
                    Entry::Spilled { path, rows, bytes, file_bytes },
                );
                self.metrics.spilled(file_bytes);
                self.metrics.buffered(state.ram_bytes, state.spill_bytes, state.turns);
                drop(state);
                self.pushed.notify_waiters();
                return Ok(());
            }
            let _wait = self.metrics.backpressure();
            freed.await;
        }
    }

    /// Wait for turns of `bucket` and take entries until the next one would
    /// exceed `max_rows`/`max_bytes`. The first entry always fits: admission
    /// cuts oversized turns to the cap. Returns None when closed.
    pub async fn pull(
        &self,
        bucket: u32,
        max_rows: usize,
        max_bytes: usize,
    ) -> Result<Option<Pulled>> {
        loop {
            let pushed = self.pushed.notified();
            tokio::pin!(pushed);
            pushed.as_mut().enable();
            let taken = {
                let mut state = self.state.lock().unwrap();
                check(&state)?;
                if state.closed {
                    return Ok(None);
                }
                let queue = state.queues.get_mut(&bucket).context("unregistered shard")?;
                let mut taken = Vec::new();
                let (mut rows, mut bytes) = (0usize, 0usize);
                while let Some(next) = queue.front() {
                    if !taken.is_empty()
                        && (rows + next.rows() > max_rows || bytes + next.bytes() > max_bytes)
                    {
                        break;
                    }
                    rows += next.rows();
                    bytes += next.bytes();
                    taken.push(queue.pop_front().unwrap());
                }
                if !taken.is_empty() {
                    state.in_flight += 1;
                    state.turns -= rows;
                }
                taken
            };
            if taken.is_empty() {
                pushed.await;
                continue;
            }
            let mut batches = Vec::with_capacity(taken.len());
            let mut rows = 0;
            let mut error = None;
            for entry in taken {
                rows += entry.rows();
                match entry {
                    Entry::Ram { batch, bytes, _charge } => {
                        // The lane slot now covers these bytes.
                        self.state.lock().unwrap().ram_bytes -= bytes;
                        drop(_charge);
                        batches.push(batch);
                    }
                    Entry::Spilled { path, file_bytes, .. } => {
                        let read = if error.is_none() { Some(read_spill(path.clone()).await) } else { None };
                        let _ = std::fs::remove_file(&path);
                        self.state.lock().unwrap().spill_bytes -= file_bytes;
                        match read {
                            Some(Ok(mut read)) => batches.append(&mut read),
                            Some(Err(e)) => error = Some(e),
                            None => {}
                        }
                    }
                }
            }
            {
                let state = self.state.lock().unwrap();
                self.metrics.buffered(state.ram_bytes, state.spill_bytes, state.turns);
            }
            self.freed.notify_waiters();
            if let Some(error) = error {
                self.fail(&error);
                return Err(error);
            }
            return Ok(Some(Pulled { batches, rows }));
        }
    }

    /// The pulled record is committed (or the lane failed via `fail`).
    pub fn done(&self) {
        self.state.lock().unwrap().in_flight -= 1;
        self.freed.notify_waiters();
    }

    pub fn fail(&self, error: &anyhow::Error) {
        let mut state = self.state.lock().unwrap();
        state.failed.get_or_insert_with(|| format!("{error:#}"));
        drop(state);
        self.pushed.notify_waiters();
        self.freed.notify_waiters();
    }

    /// Resolves when no turn is buffered and no pulled record is uncommitted.
    pub async fn drained(&self) -> Result<()> {
        loop {
            let freed = self.freed.notified();
            tokio::pin!(freed);
            freed.as_mut().enable();
            {
                let state = self.state.lock().unwrap();
                check(&state)?;
                ensure!(!state.closed, "streaming producer closed");
                if state.in_flight == 0 && state.queues.values().all(VecDeque::is_empty) {
                    return Ok(());
                }
            }
            freed.await;
        }
    }

    pub fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.pushed.notify_waiters();
        self.freed.notify_waiters();
    }

    pub fn clear_spill(&self) {
        let mut state = self.state.lock().unwrap();
        for queue in state.queues.values_mut() {
            for entry in queue.drain(..) {
                if let Entry::Spilled { path, .. } = entry {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        state.ram_bytes = 0;
        state.spill_bytes = 0;
        state.turns = 0;
    }
}

fn check(state: &State) -> Result<()> {
    if let Some(error) = &state.failed {
        bail!("streaming producer failed: {error}");
    }
    Ok(())
}

async fn write_spill(path: PathBuf, batch: RecordBatch) -> Result<u64> {
    tokio::task::spawn_blocking(move || -> Result<u64> {
        let file = std::fs::File::create(&path)
            .with_context(|| format!("create spill {}", path.display()))?;
        let mut writer = arrow_ipc::writer::FileWriter::try_new(
            std::io::BufWriter::new(file),
            batch.schema().as_ref(),
        )?;
        writer.write(&batch)?;
        writer.finish()?;
        // A cache of uncommitted input: no fsync, a restart replays source.
        let file = writer.into_inner()?.into_inner().map_err(|e| e.into_error())?;
        Ok(file.metadata()?.len())
    })
    .await?
}

async fn read_spill(path: PathBuf) -> Result<Vec<RecordBatch>> {
    tokio::task::spawn_blocking(move || -> Result<Vec<RecordBatch>> {
        let file = std::fs::File::open(&path)
            .with_context(|| format!("open spill {}", path.display()))?;
        let reader = arrow_ipc::reader::FileReader::try_new(std::io::BufReader::new(file), None)?;
        Ok(reader.collect::<std::result::Result<Vec<_>, _>>()?)
    })
    .await?
}
