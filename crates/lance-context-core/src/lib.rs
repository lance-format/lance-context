//! Core types for the lance-context storage layer.
#![recursion_limit = "256"]

mod context;
mod record;
pub mod serde;
mod store;

pub use context::{Context, ContextEntry, Snapshot};
pub use record::{ContextRecord, MetadataFilter, RecordFilters, SearchResult, StateMetadata};
pub use store::{
    CompactionConfig, CompactionStats, ContextStore, ContextStoreOptions, IdIndexType,
};

// Re-export CompactionMetrics from lance for Python bindings
pub use lance::dataset::optimize::CompactionMetrics;
