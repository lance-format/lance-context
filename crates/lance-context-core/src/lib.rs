//! Core types for the lance-context storage layer.

mod context;
mod record;
pub mod serde;
mod store;

pub use context::{Context, ContextEntry, Snapshot};
pub use record::{ContextRecord, SearchResult, StateMetadata};
pub use store::{CompactionConfig, CompactionStats, ContextStore, ContextStoreOptions};

// Re-export CompactionMetrics from lance for Python bindings
pub use lance::dataset::optimize::CompactionMetrics;
