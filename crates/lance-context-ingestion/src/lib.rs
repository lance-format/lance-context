//! Session-ordered alignment and durable WAL publication run in separate tasks.
//! Checkpoint and table consumers have independent, durable progress cursors.
//! A successful submission means the WAL is recoverable, not that it is indexed.

mod checkpoint;
mod consumer;
mod journal;
#[cfg(feature = "lance")]
pub mod lance_sink;
#[cfg(feature = "lance")]
pub mod local_lance;
mod pipeline;
mod receipt;
mod source;

pub use checkpoint::{CheckpointSink, RecoveredSession, Reducer, SessionCheckpoints, SessionState};
pub use consumer::{Consumer, Sink};
pub use journal::{BacklogPolicy, Binding, Entry, Journal, Position, Transition, Writer};
pub use pipeline::{
    Ack, Aligner, BatchFlush, BatchPolicy, HistoryLoader, Partition, PipelineConfig, Request,
};
pub use receipt::{ReceiptBatchLookup, ReceiptIndex, ReceiptLookup, ReceiptSink, SourceReceipt};
pub use source::{
    SourceAck, SourceCommit, SourceConfig, SourcePartition, SourceRequest, SOURCE_RECEIPT_CONSUMER,
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("object storage: {0}")]
    Storage(#[from] object_store::Error),
    #[error("WAL encoding: {0}")]
    Encoding(#[from] serde_json::Error),
    #[error("invalid ingestion state: {0}")]
    Invalid(String),
    #[error("writer fenced or commit outcome uncertain; reopen and recover before retrying")]
    Fenced,
    #[error("pipeline stopped; retry the same receipt after recovery")]
    Stopped,
    #[error("stage failed: {0}")]
    Stage(String),
}
