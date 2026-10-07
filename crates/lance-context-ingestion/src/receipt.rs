use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};
use object_store::PutMode;
use serde::{Deserialize, Serialize};

use crate::journal::digest;
use crate::{Binding, Entry, Error, Journal, Position, Result, Sink};

/// Exact source identity attached to one committed alignment input. This is
/// separate from an output turn ID: a source may retry a call with many turns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceReceipt {
    pub binding: Binding,
    pub receipt: String,
    pub sequence: u64,
    pub session: String,
    pub input_digest: String,
}

impl SourceReceipt {
    fn from_entry(binding: &Binding, entry: &Entry) -> Result<Self> {
        if entry.sequence == 0
            || entry.receipt.is_empty()
            || entry.session.is_empty()
            || entry.input_digest.is_empty()
        {
            return Err(Error::Invalid("empty source receipt identity".into()));
        }
        Ok(Self {
            binding: binding.clone(),
            receipt: entry.receipt.clone(),
            sequence: entry.sequence,
            session: entry.session.clone(),
            input_digest: entry.input_digest.clone(),
        })
    }
}

/// Lookup reconciles index lag through this exact committed WAL position.
/// A miss does not cover later commits or another request admitted in memory.
#[derive(Debug)]
pub struct ReceiptLookup {
    pub receipt: Option<SourceReceipt>,
    pub through: Position,
}

/// Only found receipts are present; all requested identities share one captured
/// committed position. Memory is proportional to the caller-bounded request set.
#[derive(Debug)]
pub struct ReceiptBatchLookup {
    pub receipts: HashMap<String, SourceReceipt>,
    pub through: Position,
}

/// Immutable source-receipt lookup with an independently checkpointed index.
/// Use a dedicated Consumer name and the same index for both `find` and `sink`.
/// This is not an admission lock or sequence allocator: the partition's single
/// admission owner must also reconcile its in-flight requests before assigning
/// a new sequence. A lost HTTP response must retry the same receipt and bytes.
#[derive(Clone)]
pub struct ReceiptIndex {
    journal: Journal,
}

impl ReceiptIndex {
    pub fn new(journal: Journal) -> Self {
        Self { journal }
    }

    fn path(&self, receipt: &str) -> object_store::path::Path {
        self.journal
            .path(&format!("receipts/{}.json", digest(receipt.as_bytes())))
    }

    async fn read(&self, receipt: &str) -> Result<Option<SourceReceipt>> {
        let value: SourceReceipt = match self.journal.read(&self.path(receipt)).await {
            Ok((value, _)) => value,
            Err(Error::Storage(object_store::Error::NotFound { .. })) => return Ok(None),
            Err(error) => return Err(error),
        };
        if value.binding != *self.journal.binding()
            || value.receipt != receipt
            || value.sequence == 0
            || value.session.is_empty()
            || value.input_digest.is_empty()
        {
            return Err(Error::Invalid(
                "source receipt index binding mismatch".into(),
            ));
        }
        Ok(Some(value))
    }

    /// Read the index cursor before its entry, then capture the committed head.
    /// Replay only the unindexed suffix in bounded pages, including when an
    /// index write succeeded but its cursor write failed. A duplicate receipt
    /// at a different sequence fails closed. No objects are written here.
    pub async fn find(&self, consumer: &str, receipt: &str) -> Result<ReceiptLookup> {
        let mut found = self.find_many(consumer, &[receipt], 1).await?;
        Ok(ReceiptLookup {
            receipt: found.receipts.remove(receipt),
            through: found.through,
        })
    }

    /// Resolve one source batch with bounded concurrent index reads and one
    /// shared WAL suffix scan, rather than replaying the tail for every call.
    /// `receipts` must be nonempty and unique. The admission owner bounds its
    /// total count/bytes and retains any newer speculative receipt mappings.
    pub async fn find_many(
        &self,
        consumer: &str,
        receipts: &[&str],
        concurrency: usize,
    ) -> Result<ReceiptBatchLookup> {
        let wanted = receipts.iter().copied().collect::<HashSet<_>>();
        if concurrency == 0
            || receipts.is_empty()
            || wanted.contains("")
            || wanted.len() != receipts.len()
        {
            return Err(Error::Invalid(
                "empty or duplicate receipt lookup identity/budget".into(),
            ));
        }
        let mut after = self.journal.consumer_position(consumer).await?;
        let indexed = stream::iter(receipts.iter().copied())
            .map(|receipt| self.read(receipt))
            .buffer_unordered(concurrency)
            .try_collect::<Vec<_>>()
            .await?;
        let mut found = indexed
            .into_iter()
            .flatten()
            .map(|value| (value.receipt.clone(), value))
            .collect::<HashMap<_, _>>();
        let through = self.journal.position().await?;
        if found
            .values()
            .any(|value| value.sequence > through.sequence)
        {
            return Err(Error::Invalid("receipt ahead of committed WAL".into()));
        }
        loop {
            let page = self.journal.pending(&after, &through).await?;
            if page.is_empty() {
                break;
            }
            for position in page {
                for entry in self.journal.entries(&position).await? {
                    if wanted.contains(entry.receipt.as_str()) {
                        let actual = SourceReceipt::from_entry(self.journal.binding(), &entry)?;
                        if found
                            .get(&entry.receipt)
                            .is_some_and(|value| value != &actual)
                        {
                            return Err(Error::Invalid("source receipt reused or changed".into()));
                        }
                        found.insert(entry.receipt, actual);
                    }
                }
                after = position;
            }
        }
        Ok(ReceiptBatchLookup {
            receipts: found,
            through,
        })
    }

    pub fn sink(&self, concurrency: usize) -> Result<ReceiptSink> {
        if concurrency == 0 {
            return Err(Error::Invalid("zero receipt index concurrency".into()));
        }
        Ok(ReceiptSink {
            index: self.clone(),
            concurrency,
        })
    }

    async fn insert(&self, receipt: SourceReceipt) -> Result<()> {
        match self
            .journal
            .put(&self.path(&receipt.receipt), &receipt, PutMode::Create)
            .await
        {
            Ok(_) => Ok(()),
            Err(Error::Storage(object_store::Error::AlreadyExists { .. })) => {
                if self.read(&receipt.receipt).await?.as_ref() == Some(&receipt) {
                    Ok(())
                } else {
                    Err(Error::Invalid("source receipt reused or changed".into()))
                }
            }
            Err(error) => Err(error),
        }
    }
}

pub struct ReceiptSink {
    index: ReceiptIndex,
    concurrency: usize,
}

#[async_trait]
impl Sink for ReceiptSink {
    async fn apply(&mut self, binding: &Binding, entries: &[Entry]) -> Result<()> {
        if binding != self.index.journal.binding() {
            return Err(Error::Invalid("receipt sink binding mismatch".into()));
        }
        let mut unique = HashMap::new();
        for entry in entries {
            let receipt = SourceReceipt::from_entry(binding, entry)?;
            if let Some(previous) = unique.insert(entry.receipt.as_str(), receipt.clone()) {
                if previous != receipt {
                    return Err(Error::Invalid("source receipt reused or changed".into()));
                }
            }
        }
        stream::iter(unique.into_values())
            .map(|receipt| self.index.insert(receipt))
            .buffer_unordered(self.concurrency)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(())
    }
}
