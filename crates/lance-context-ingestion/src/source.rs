use std::collections::HashMap;

use tokio::sync::watch;

use crate::journal::digest;
use crate::{
    Ack, Aligner, Consumer, Error, HistoryLoader, Partition, PipelineConfig, Position,
    ReceiptIndex, Request, Result, Writer,
};

/// Dedicated cursor for this admission layer's immutable receipt index.
/// Include it in Writer::with_backlog when limiting unindexed source receipts.
pub const SOURCE_RECEIPT_CONSUMER: &str = "source-receipts";

#[derive(Clone, Debug)]
pub struct SourceRequest {
    pub session: String,
    pub receipt: String,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct SourceConfig {
    pub max_batch_requests: usize,
    /// Bounds retained input buffer capacities for one serialized admission call.
    /// Pipeline reservations, in-flight receipt metadata, and caller/server
    /// queues have separate budgets. This is not a whole-process RSS limit.
    pub max_batch_bytes: usize,
    pub receipt_read_concurrency: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity {
    session: String,
    digest: String,
}

struct Pending {
    identity: Identity,
    sequence: u64,
}

/// Exact input sequence and a durable prefix that contains it. Source fan-out
/// must receive every partition's ACK before advancing its own checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceCommit {
    pub sequence: u64,
    pub through: Position,
}

enum Wait {
    New(Ack),
    Pending(watch::Receiver<Position>),
    Committed(Position),
}

/// Dropping a waiter never cancels admitted work. In-flight retries wait on the
/// original durable sequence without entering the alignment dispatch queue.
pub struct SourceAck {
    sequence: u64,
    wait: Wait,
}

impl SourceAck {
    pub async fn wait(self) -> Result<SourceCommit> {
        let through = match self.wait {
            Wait::New(ack) => ack.wait().await?,
            Wait::Committed(position) => position,
            Wait::Pending(mut durable) => loop {
                let position = durable.borrow().clone();
                if position.sequence >= self.sequence {
                    break position;
                }
                durable.changed().await.map_err(|_| Error::Stopped)?;
            },
        };
        Ok(SourceCommit {
            sequence: self.sequence,
            through,
        })
    }
}

/// One partition's exclusive source admission owner. Stable receipt identities
/// survive restart; sequence assignment for an uncommitted suffix may change.
/// Serializing admission does not wait for WAL ACKs or index/checkpoint work.
/// The caller supplies bounded HTTP/source queues and owns partition routing.
/// Cancellation during admission poisons this handle: drain/drop and recover
/// before retrying, because an unknown prefix may already have been enqueued.
pub struct SourcePartition {
    partition: Partition,
    index: ReceiptIndex,
    config: SourceConfig,
    max_input_bytes: usize,
    last_assigned: u64,
    pending: HashMap<String, Pending>,
    poisoned: bool,
    journal: crate::Journal,
}

impl SourcePartition {
    pub async fn start<A: Aligner, L: HistoryLoader>(
        writer: Writer,
        aligners: Vec<A>,
        loader: L,
        pipeline: PipelineConfig,
        config: SourceConfig,
    ) -> Result<Self> {
        if config.max_batch_requests == 0
            || config.max_batch_bytes == 0
            || config.receipt_read_concurrency == 0
        {
            return Err(Error::Invalid("zero source admission budget".into()));
        }
        let journal = writer.journal().clone();
        let last_assigned = writer.position().sequence;
        let max_input_bytes = pipeline.max_input_bytes;
        let partition = Partition::start_with_aligners(writer, aligners, loader, pipeline).await?;
        Ok(Self {
            partition,
            index: ReceiptIndex::new(journal.clone()),
            config,
            max_input_bytes,
            last_assigned,
            pending: HashMap::new(),
            poisoned: false,
            journal,
        })
    }

    pub fn receipt_index(&self) -> &ReceiptIndex {
        &self.index
    }

    /// Scheduler must run only one consumer for this partition/cursor. It may
    /// batch independently of admission and WAL; lookup reconciles any lag.
    pub async fn open_receipt_consumer(&self) -> Result<Consumer> {
        Consumer::open(self.journal.clone(), SOURCE_RECEIPT_CONSUMER).await
    }

    pub fn durable_position(&self) -> Position {
        self.partition.durable_position()
    }

    pub async fn enqueue_many(&mut self, requests: Vec<SourceRequest>) -> Result<Vec<SourceAck>> {
        if self.poisoned {
            return Err(Error::Fenced);
        }
        if self.partition.subscribe_durable().has_changed().is_err() {
            return Err(Error::Stopped);
        }
        if requests.is_empty() || requests.len() > self.config.max_batch_requests {
            return Err(Error::Invalid("empty or oversized source batch".into()));
        }
        let mut bytes = requests
            .capacity()
            .checked_mul(std::mem::size_of::<SourceRequest>())
            .ok_or_else(|| Error::Invalid("source batch size overflow".into()))?;
        let mut identities = HashMap::new();
        for request in &requests {
            let size = request
                .payload
                .capacity()
                .checked_add(request.session.capacity())
                .and_then(|n| n.checked_add(request.receipt.capacity()))
                .ok_or_else(|| Error::Invalid("source request size overflow".into()))?;
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| Error::Invalid("source batch size overflow".into()))?;
            if request.receipt.is_empty()
                || request.session.is_empty()
                || size > self.max_input_bytes
                || bytes > self.config.max_batch_bytes
            {
                return Err(Error::Invalid(
                    "source request exceeds input budget or lacks identity".into(),
                ));
            }
            let identity = Identity {
                session: request.session.clone(),
                digest: digest(&request.payload),
            };
            if let Some(previous) = identities.insert(request.receipt.clone(), identity.clone()) {
                if previous != identity {
                    return Err(Error::Invalid("changed source retry in batch".into()));
                }
            }
        }
        // Committed mappings can be dropped: lookup below reconciles the index
        // and WAL tail. Keep all speculative ones even if their ACK was dropped.
        let durable = self.partition.durable_position().sequence;
        self.pending.retain(|_, pending| pending.sequence > durable);
        let mut unknown = Vec::new();
        for (receipt, identity) in &identities {
            if let Some(pending) = self.pending.get(receipt) {
                if &pending.identity != identity {
                    return Err(Error::Invalid("changed in-flight source retry".into()));
                }
            } else {
                unknown.push(receipt.as_str());
            }
        }
        let lookup = if unknown.is_empty() {
            None
        } else {
            Some(
                self.index
                    .find_many(
                        SOURCE_RECEIPT_CONSUMER,
                        &unknown,
                        self.config.receipt_read_concurrency,
                    )
                    .await?,
            )
        };
        if let Some(found) = &lookup {
            if found.through.sequence > self.last_assigned {
                self.poisoned = true;
                return Err(Error::Fenced);
            }
            for (receipt, original) in &found.receipts {
                let identity = &identities[receipt];
                if identity.session != original.session || identity.digest != original.input_digest
                {
                    return Err(Error::Invalid("changed committed source retry".into()));
                }
            }
        }
        let new_count = identities
            .keys()
            .filter(|receipt| {
                !self.pending.contains_key(*receipt)
                    && !lookup
                        .as_ref()
                        .is_some_and(|found| found.receipts.contains_key(*receipt))
            })
            .count();
        self.last_assigned
            .checked_add(new_count as u64)
            .ok_or_else(|| Error::Invalid("source sequence overflow".into()))?;

        self.poisoned = true;
        let mut acks = Vec::with_capacity(requests.len());
        for request in requests {
            if let Some(pending) = self.pending.get(&request.receipt) {
                acks.push(SourceAck {
                    sequence: pending.sequence,
                    wait: Wait::Pending(self.partition.subscribe_durable()),
                });
            } else if let Some((original, through)) = lookup.as_ref().and_then(|found| {
                found
                    .receipts
                    .get(&request.receipt)
                    .map(|receipt| (receipt, &found.through))
            }) {
                acks.push(SourceAck {
                    sequence: original.sequence,
                    wait: Wait::Committed(through.clone()),
                });
            } else {
                self.last_assigned += 1;
                let sequence = self.last_assigned;
                self.pending.insert(
                    request.receipt.clone(),
                    Pending {
                        sequence,
                        identity: identities[&request.receipt].clone(),
                    },
                );
                let ack = self
                    .partition
                    .enqueue(Request {
                        sequence,
                        session: request.session,
                        receipt: request.receipt,
                        payload: request.payload,
                    })
                    .await?;
                acks.push(SourceAck {
                    sequence,
                    wait: Wait::New(ack),
                });
            }
        }
        self.poisoned = false;
        Ok(acks)
    }

    /// Drain an admitted prefix, including after a cancelled admission call.
    /// This handle cannot be reused; recover from the resulting durable head.
    pub async fn shutdown(self) -> Result<()> {
        self.partition.shutdown().await
    }
}
