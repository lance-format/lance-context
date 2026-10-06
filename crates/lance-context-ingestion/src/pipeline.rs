use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::{stream, StreamExt};
use tokio::sync::{mpsc, oneshot, watch, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio::time::{timeout_at, Instant};

use crate::journal::digest;
use crate::{Binding, Entry, Error, Journal, Position, Result, Transition, Writer};

#[derive(Clone, Debug)]
pub struct Request {
    /// Contiguous, stable sequence within this virtual partition, starting at 1.
    /// A source fan-out must wait for every partition ACK before advancing its
    /// source checkpoint. Retries preserve sequence, receipt, session and bytes.
    pub sequence: u64,
    pub session: String,
    pub receipt: String,
    pub payload: Vec<u8>,
}

/// Adapter owns session state and its cache budget. Alignment can run ahead of
/// WAL durability, so an error discards this adapter and its speculative suffix.
/// Checkpoints restore exact committed deltas, never freshly recomputed IDs.
#[async_trait]
pub trait Aligner: Send + 'static {
    async fn restore(&mut self, binding: &Binding) -> Result<Position>;
    async fn replay(&mut self, entry: &Entry) -> Result<()>;
    /// Prefetched history is an immutable checkpoint, potentially behind this
    /// adapter's speculative state. Reconcile its revision before using it;
    /// never replace newer same-session state with an older prefetch result.
    async fn align(&mut self, request: &Request, history: &[u8]) -> Result<Transition>;
}

/// Concurrent, read-only history loading. The adapter must respect `max_bytes`
/// while fetching/decoding, not only after allocation. Loaded state carries its
/// revision in the adapter's encoding so alignment can reconcile stale loads.
#[async_trait]
pub trait HistoryLoader: Send + Sync + 'static {
    async fn load(&self, request: &Request, max_bytes: usize) -> Result<Vec<u8>>;
}

struct NoHistory;

#[async_trait]
impl HistoryLoader for NoHistory {
    async fn load(&self, _request: &Request, _max_bytes: usize) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

#[derive(Clone, Debug)]
pub struct BatchPolicy {
    pub max_entries: usize,
    /// Serialized Entry bytes, excluding the bounded segment envelope.
    pub max_bytes: usize,
    pub max_delay: Duration,
}

impl BatchPolicy {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.max_entries == 0 || self.max_bytes == 0 || self.max_delay.is_zero() {
            return Err(Error::Invalid(
                "batch count, bytes and delay must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub queue_entries: usize,
    pub load_concurrency: usize,
    /// Bounds reserved input/output/serialization bytes across both queues and
    /// active batches. Adapter state, runtime and object-store buffers are extra.
    pub memory_bytes: u32,
    pub max_input_bytes: usize,
    pub max_transition_bytes: usize,
    pub max_history_bytes: usize,
    pub wal: BatchPolicy,
}

struct Input {
    request: Request,
    ack: oneshot::Sender<Result<Position>>,
    permit: OwnedSemaphorePermit,
}

struct Prepared {
    input: Input,
    history: Vec<u8>,
}

struct Aligned {
    entry: Entry,
    encoded_bytes: usize,
    ack: oneshot::Sender<Result<Position>>,
    _permit: OwnedSemaphorePermit,
}

/// Dropping an ACK waiter does not cancel already admitted durable work.
pub struct Ack(oneshot::Receiver<Result<Position>>);

impl Ack {
    pub async fn wait(self) -> Result<Position> {
        self.0.await.map_err(|_| Error::Stopped)?
    }
}

/// One stable virtual partition, with independent alignment and WAL tasks.
/// Run different partitions on independent workers. A session must always route
/// to the same partition; changing worker count must not change that mapping.
pub struct Partition {
    input: Option<mpsc::Sender<Input>>,
    loading: Option<JoinHandle<Result<()>>>,
    alignment: Option<JoinHandle<Result<()>>>,
    wal: Option<JoinHandle<Result<()>>>,
    budget: Arc<Semaphore>,
    config: PipelineConfig,
    durable: watch::Receiver<Position>,
}

impl Partition {
    pub async fn start<A: Aligner>(
        writer: Writer,
        aligner: A,
        config: PipelineConfig,
    ) -> Result<Self> {
        Self::start_with_loader(writer, aligner, NoHistory, config).await
    }

    pub async fn start_with_loader<A: Aligner, L: HistoryLoader>(
        writer: Writer,
        mut aligner: A,
        loader: L,
        config: PipelineConfig,
    ) -> Result<Self> {
        config.wal.validate()?;
        let reserve = reservation(
            config.max_input_bytes,
            config.max_transition_bytes,
            config.max_history_bytes,
        )?;
        if config.queue_entries == 0
            || config.load_concurrency == 0
            || reserve > config.memory_bytes
        {
            return Err(Error::Invalid(
                "queue empty or maximum request exceeds memory budget".into(),
            ));
        }
        let journal = writer.journal().clone();
        let mut checkpoint = aligner.restore(journal.binding()).await?;
        while &checkpoint != writer.position() {
            for position in journal.pending(&checkpoint, writer.position()).await? {
                for entry in journal.entries(&position).await? {
                    aligner.replay(&entry).await?;
                }
                checkpoint = position;
            }
        }
        let (input_tx, input_rx) = mpsc::channel(config.queue_entries);
        let (loaded_tx, loaded_rx) = mpsc::channel(config.queue_entries);
        let (wal_tx, wal_rx) = mpsc::channel(config.queue_entries);
        let (durable_tx, durable_rx) = watch::channel(writer.position().clone());
        let loading = tokio::spawn(load_loop(loader, input_rx, loaded_tx, config.clone()));
        let alignment = tokio::spawn(align_loop(
            aligner,
            journal,
            loaded_rx,
            wal_tx,
            durable_rx.clone(),
            config.clone(),
        ));
        let wal = tokio::spawn(wal_loop(writer, wal_rx, durable_tx, config.wal.clone()));
        Ok(Self {
            input: Some(input_tx),
            loading: Some(loading),
            alignment: Some(alignment),
            wal: Some(wal),
            budget: Arc::new(Semaphore::new(config.memory_bytes as usize)),
            config,
            durable: durable_rx,
        })
    }

    /// Admission is bounded; the returned ACK resolves only after a conditional
    /// durable head publication. Caller must serialize admission order within a
    /// partition. Cancellation before admission leaves the receipt uncommitted.
    pub async fn enqueue(&self, request: Request) -> Result<Ack> {
        let input_bytes = request
            .payload
            .capacity()
            .checked_add(request.session.capacity())
            .and_then(|n| n.checked_add(request.receipt.capacity()))
            .ok_or_else(|| Error::Invalid("input size overflow".into()))?;
        if input_bytes > self.config.max_input_bytes
            || request.session.is_empty()
            || request.receipt.is_empty()
        {
            return Err(Error::Invalid(
                "input exceeds limit or lacks session/receipt".into(),
            ));
        }
        let permit = self
            .budget
            .clone()
            .acquire_many_owned(reservation(
                input_bytes,
                self.config.max_transition_bytes,
                self.config.max_history_bytes,
            )?)
            .await
            .map_err(|_| Error::Stopped)?;
        let (ack, receiver) = oneshot::channel();
        self.input
            .as_ref()
            .ok_or(Error::Stopped)?
            .send(Input {
                request,
                ack,
                permit,
            })
            .await
            .map_err(|_| Error::Stopped)?;
        Ok(Ack(receiver))
    }

    pub fn durable_position(&self) -> Position {
        self.durable.borrow().clone()
    }

    /// Stop admission, drain alignment and flush even a partially filled WAL.
    /// Dropping Partition instead aborts tasks; recovery reconciles any uncertain
    /// head write. Neither path advances checkpoint or merge consumer cursors.
    pub async fn shutdown(mut self) -> Result<()> {
        self.input.take();
        let loading = self
            .loading
            .as_mut()
            .unwrap()
            .await
            .map_err(|error| Error::Stage(error.to_string()))?;
        self.loading.take();
        let alignment = self
            .alignment
            .as_mut()
            .unwrap()
            .await
            .map_err(|error| Error::Stage(error.to_string()))?;
        self.alignment.take();
        let wal = self
            .wal
            .as_mut()
            .unwrap()
            .await
            .map_err(|error| Error::Stage(error.to_string()))?;
        self.wal.take();
        loading?;
        alignment?;
        wal
    }
}

impl Drop for Partition {
    fn drop(&mut self) {
        if let Some(task) = &self.loading {
            task.abort();
        }
        if let Some(task) = &self.alignment {
            task.abort();
        }
        if let Some(task) = &self.wal {
            task.abort();
        }
    }
}

fn reservation(input: usize, output: usize, history: usize) -> Result<u32> {
    // Include retained buffer capacities, escaped strings (up to six bytes per
    // byte), geometric serializer capacity growth and fixed envelope space.
    input
        .checked_add(output)
        .and_then(|n| n.checked_add(history))
        .and_then(|n| n.checked_mul(16))
        .and_then(|n| n.checked_add(4096))
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Error::Invalid("request reservation overflow".into()))
}

async fn load_loop<L: HistoryLoader>(
    loader: L,
    input: mpsc::Receiver<Input>,
    output: mpsc::Sender<Prepared>,
    config: PipelineConfig,
) -> Result<()> {
    let loader = Arc::new(loader);
    let mut prepared = stream::unfold(input, |mut receiver| async move {
        receiver.recv().await.map(|item| (item, receiver))
    })
    .map(|input| {
        let loader = loader.clone();
        async move {
            let history = loader
                .load(&input.request, config.max_history_bytes)
                .await?;
            if history.capacity() > config.max_history_bytes {
                return Err(Error::Invalid(
                    "history loader exceeded reserved bytes".into(),
                ));
            }
            Ok(Prepared { input, history })
        }
    })
    .buffered(config.load_concurrency)
    .boxed();
    while let Some(item) = prepared.next().await {
        output.send(item?).await.map_err(|_| Error::Stopped)?;
    }
    Ok(())
}

async fn align_loop<A: Aligner>(
    mut aligner: A,
    journal: Journal,
    mut input: mpsc::Receiver<Prepared>,
    wal: mpsc::Sender<Aligned>,
    mut durable: watch::Receiver<Position>,
    config: PipelineConfig,
) -> Result<()> {
    let mut next = durable
        .borrow()
        .sequence
        .checked_add(1)
        .ok_or_else(|| Error::Invalid("sequence overflow".into()))?;
    while let Some(input) = input.recv().await {
        let Prepared { input, history } = input;
        let Input {
            request,
            ack,
            permit,
        } = input;
        let input_digest = digest(&request.payload);
        if request.sequence < next && request.sequence > 0 {
            // A repeated in-flight receipt waits for its original commit without
            // applying alignment twice. Closed writer watch means uncertain I/O.
            while durable.borrow().sequence < request.sequence {
                durable.changed().await.map_err(|_| Error::Stopped)?;
            }
            let through = durable.borrow().clone();
            let original = journal.receipt(request.sequence, &through).await?;
            let response = if original.session == request.session
                && original.receipt == request.receipt
                && original.input_digest == input_digest
            {
                Ok(through)
            } else {
                Err(Error::Invalid("retry receipt identity changed".into()))
            };
            let _ = ack.send(response);
            continue;
        }
        if request.sequence != next {
            let _ = ack.send(Err(Error::Invalid(format!(
                "expected partition sequence {next}"
            ))));
            continue;
        }
        let transition = aligner.align(&request, &history).await?;
        if transition
            .delta
            .capacity()
            .saturating_add(transition.records.capacity())
            > config.max_transition_bytes
        {
            return Err(Error::Invalid(
                "aligner exceeded reserved output bytes".into(),
            ));
        }
        let entry = Entry {
            sequence: next,
            session: request.session,
            receipt: request.receipt,
            input_digest,
            transition,
        };
        let encoded_bytes = serde_json::to_vec(&entry)?.len();
        if encoded_bytes > config.wal.max_bytes {
            return Err(Error::Invalid(
                "single transition exceeds WAL batch limit".into(),
            ));
        }
        wal.send(Aligned {
            entry,
            encoded_bytes,
            ack,
            _permit: permit,
        })
        .await
        .map_err(|_| Error::Stopped)?;
        next = next
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("sequence overflow".into()))?;
    }
    Ok(())
}

async fn wal_loop(
    mut writer: Writer,
    mut input: mpsc::Receiver<Aligned>,
    durable: watch::Sender<Position>,
    policy: BatchPolicy,
) -> Result<()> {
    let mut carry: Option<Aligned> = None;
    loop {
        let first = match carry.take() {
            Some(first) => first,
            None => match input.recv().await {
                Some(first) => first,
                None => return Ok(()),
            },
        };
        let deadline = Instant::now() + policy.max_delay;
        let mut bytes = first.encoded_bytes;
        let mut batch = vec![first];
        while batch.len() < policy.max_entries && bytes < policy.max_bytes {
            match timeout_at(deadline, input.recv()).await {
                Ok(Some(next)) if next.encoded_bytes <= policy.max_bytes - bytes => {
                    bytes += next.encoded_bytes;
                    batch.push(next);
                }
                Ok(Some(next)) => {
                    carry = Some(next);
                    break;
                }
                Ok(None) | Err(_) => break,
            }
        }
        let entries = batch
            .iter_mut()
            .map(|item| Entry {
                sequence: item.entry.sequence,
                session: std::mem::take(&mut item.entry.session),
                receipt: std::mem::take(&mut item.entry.receipt),
                input_digest: std::mem::take(&mut item.entry.input_digest),
                transition: Transition {
                    delta: std::mem::take(&mut item.entry.transition.delta),
                    records: std::mem::take(&mut item.entry.transition.records),
                },
            })
            .collect();
        let position = writer.append(entries).await?;
        durable.send_replace(position.clone());
        for item in batch {
            let _ = item.ack.send(Ok(position.clone()));
        }
    }
}
