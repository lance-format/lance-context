//! Core types for the lance-context storage layer.
#![recursion_limit = "256"]

mod api_impl;
mod context;
mod eval;
mod export;
mod namespace;
mod record;
mod rollout;
mod rollout_store;
pub mod serde;
mod store;

pub use context::{Context, ContextEntry, Snapshot};
pub use eval::{
    AbReport, EvalConfig, EvalQuery, EvalQuerySet, EvalReport, MetricScores, QueryEval,
    RelevanceLabel, RetrievalMode,
};
pub use export::{
    Distribution, ExcludedCounts, ExportConfig, ExportCounts, ExportManifest, ExportStats,
    ExportTask, GroupBy, Message, PreferenceExample, PreferenceForm, Provenance, RankedCandidate,
    RolloutExample, RolloutResponse, SftExample, SplitConfig, SplitManifest, TokenStats,
    EXPORT_SCHEMA_VERSION,
};
pub use namespace::{ContextNamespace, PartitionInfo, PartitionSelector, PartitionSpec};
pub use record::{
    ContextRecord, LifecycleQueryOptions, MetadataFilter, RecordFilters, RecordPatch, Relationship,
    RetrieveResult, SearchResult, StateMetadata, UpdateResult, UpsertResult, LIFECYCLE_ACTIVE,
    LIFECYCLE_CONTRADICTED,
};
pub use rollout::{RolloutRecord, ROLE_ARTIFACT, ROLE_ASSISTANT, ROLE_GRADE, ROLE_TOOL};
pub use rollout_store::{rollout_schema, RolloutStore, RolloutStoreOptions};
pub use store::{
    CompactionConfig, CompactionStats, ContextStore, ContextStoreOptions, DistanceMetric,
    IdIndexType, ReadProjection,
};

// Re-export CompactionMetrics from lance for Python bindings
pub use lance::dataset::optimize::CompactionMetrics;
