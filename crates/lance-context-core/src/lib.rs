//! Core types for the lance-context storage layer.
#![recursion_limit = "256"]

mod api_impl;
mod context;
mod datagen;
mod datagen_store;
mod eval;
mod export;
mod id;
mod namespace;
mod record;
mod registry;
mod rollout;
mod rollout_store;
pub mod serde;
mod storage;
mod store;

pub use api_impl::rollout_record_to_dto;
pub use context::{Context, ContextEntry, Snapshot};
pub use datagen::{
    datagen_event_id, datagen_trajectory, fold_datagen_events, DatagenBlobValue, DatagenEvent,
    DatagenEventType, DatagenFailure, DatagenFieldState, DatagenItemStatus, DatagenStepCursor,
    DatagenTerminal, DatagenTrajectoryPoint, DatagenValue, FoldedDatagenItem,
    DATAGEN_SCHEMA_VERSION,
};
pub use datagen_store::{datagen_log_schema, DatagenStore, DatagenStoreOptions};
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
pub use id::{generate_id, new_uuid};
pub use namespace::{ContextNamespace, PartitionInfo, PartitionSelector, PartitionSpec};
pub use record::{
    ContextRecord, LifecycleQueryOptions, MetadataFilter, RecordFilters, RecordPatch, Relationship,
    RetrieveResult, SearchResult, StateMetadata, UpdateResult, UpsertResult, LIFECYCLE_ACTIVE,
    LIFECYCLE_CONTRADICTED,
};
pub use registry::{RegistryEntry, RolloutRegistry};
pub use rollout::{RolloutRecord, ROLE_ARTIFACT, ROLE_ASSISTANT, ROLE_GRADE, ROLE_TOOL};
pub use rollout_store::{
    rollout_schema, ListSource, RolloutFilters, RolloutObservation, RolloutPage, RolloutStore,
    RolloutStoreOptions, SqlQueryResult, SQL_MAX_RESULT_ROWS, SQL_MAX_SCAN_ROWS, SQL_TABLE_NAME,
};
pub use storage::{create_local_dir_if_needed, join_uri, validate_store_name, MAX_STORE_NAME_LEN};
pub use store::{
    CompactionConfig, CompactionStats, ContextStore, ContextStoreOptions, DistanceMetric,
    IdIndexType, ReadProjection,
};

// Re-export CompactionMetrics from lance for Python bindings
pub use lance::dataset::optimize::CompactionMetrics;

// Re-export the Lance error type so downstream crates (e.g. the server) can
// match on its typed variants instead of string-matching `Display` output.
pub use lance::Error as LanceError;

// Re-export the Lance session type so the server can build one shared,
// capacity-bounded cache session across all resident rollout stores.
pub use lance::session::Session;
