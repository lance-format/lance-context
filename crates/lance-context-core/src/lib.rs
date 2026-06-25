//! Core types for the lance-context storage layer.
#![recursion_limit = "256"]

mod api_impl;
mod context;
mod export;
mod namespace;
mod record;
pub mod serde;
mod store;

pub use context::{Context, ContextEntry, Snapshot};
pub use export::{
    ExportConfig, ExportCounts, ExportManifest, ExportTask, GroupBy, Message, PreferenceExample,
    PreferenceForm, Provenance, RankedCandidate, RolloutExample, RolloutResponse, SftExample,
    EXPORT_SCHEMA_VERSION,
};
pub use namespace::{ContextNamespace, PartitionInfo, PartitionSelector, PartitionSpec};
pub use record::{
    ContextRecord, LifecycleQueryOptions, MetadataFilter, RecordFilters, RecordPatch, Relationship,
    RetrieveResult, SearchResult, StateMetadata, UpdateResult, UpsertResult, LIFECYCLE_ACTIVE,
    LIFECYCLE_CONTRADICTED,
};
pub use store::{
    CompactionConfig, CompactionStats, ContextStore, ContextStoreOptions, DistanceMetric,
    IdIndexType,
};

// Re-export CompactionMetrics from lance for Python bindings
pub use lance::dataset::optimize::CompactionMetrics;
