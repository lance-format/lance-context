//! Core types for the lance-context storage layer.
#![recursion_limit = "256"]

mod api_impl;
mod context;
mod datagen;
mod datagen_store;
pub mod etcd;
mod eval;
mod export;
pub mod generic_codec;
mod generic_store;
mod id;
mod key_index;
pub mod merge_budget;
pub mod merge_write_scope;
pub mod metrics;
mod namespace;
mod record;
mod registry;
mod registry_etcd;
mod rollout;
pub mod rollout_append;
mod rollout_store;
pub mod serde;
mod storage;
mod store;
mod store_base;
pub use store_base::{
    is_pending_generations_exceeded, DEFAULT_PENDING_GENERATIONS_MAX, PENDING_GENERATIONS_EXCEEDED,
};

// Request/DTO conversions, exported so the server does not keep its own copies.
// These were duplicated verbatim between here and `routes/`; see #214.
pub use api_impl::{
    datagen_event_to_dto, datagen_events_from_dtos, dto_to_relationship, folded_item_to_dto,
    patch_from_dto, record_from_add_request, record_to_dto, relationship_to_dto,
    rollout_record_from_add_request, rollout_record_to_dto,
};
pub use context::{Context, ContextEntry, Snapshot};
pub use datagen::{
    datagen_event_id, datagen_failures, datagen_trajectory, fold_datagen_events,
    open_stream_events, DatagenBlobValue, DatagenErrorInfo, DatagenEvent, DatagenEventType,
    DatagenFailure, DatagenFieldChange, DatagenFieldState, DatagenItemId, DatagenItemLookup,
    DatagenItemNode, DatagenItemStatus, DatagenItemTree, DatagenNewStream, DatagenOpenStream,
    DatagenRootItemStatuses, DatagenStepCursor, DatagenStepId, DatagenStepKind,
    DatagenStreamPosition, DatagenStreamWriter, DatagenTerminal, DatagenTrajectory, DatagenValue,
    DatagenWriteContext, FieldOp, FoldedDatagenItem, DATAGEN_SCHEMA_VERSION,
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
pub use generic_codec::{batch_to_rows, ids_from_batch, rows_to_batch, Row};
pub use generic_store::{GenericStore, GenericStoreOptions};
pub use id::{generate_id, new_uuid};
pub use key_index::KeyIndexType;
pub use merge_budget::{MergeMemoryBudget, MergeReservation};
pub use namespace::{ContextNamespace, PartitionInfo, PartitionSelector, PartitionSpec};
pub use record::{
    ContextRecord, LifecycleQueryOptions, MetadataFilter, RecordFilters, RecordPatch, Relationship,
    RetrieveResult, SearchResult, StateMetadata, UpdateResult, UpsertResult, LIFECYCLE_ACTIVE,
    LIFECYCLE_CONTRADICTED,
};
pub use registry::{
    backfill_registry, diff_registries, LanceRegistry, MirroredRegistry, RegistryDiff,
    RegistryEntry, RegistryMismatch, RolloutRegistry, StoreRegistry,
};
pub use registry_etcd::EtcdRegistry;
pub use rollout::{RolloutRecord, ROLE_ARTIFACT, ROLE_ASSISTANT, ROLE_GRADE, ROLE_TOOL};
pub use rollout_store::{
    rollout_schema, ListSource, PreparedMerge, RolloutFilters, RolloutObservation, RolloutPage,
    RolloutStore, RolloutStoreOptions, SqlQueryResult, SQL_MAX_RESULT_ROWS, SQL_MAX_SCAN_ROWS,
    SQL_TABLE_NAME,
};
// Schema declaration lives in the API crate: it is part of the wire contract.
pub use lance_context_api::{
    trace_schema, ColumnSpec, ColumnType, SchemaSpec, TraceRecord, TraceStore, ID_COLUMN,
};
pub use storage::{
    create_local_dir_if_needed, join_uri, remove_dataset, validate_store_name, MAX_STORE_NAME_LEN,
};
pub use store::{
    CompactionConfig, CompactionStats, ContextStore, ContextStoreOptions, DistanceMetric,
    DroppedFragment, IdIndexType, ReadProjection, RepairReport,
};

// Re-export CompactionMetrics from lance for Python bindings
pub use lance::dataset::optimize::CompactionMetrics;

// Re-export the Lance error type so downstream crates (e.g. the server) can
// match on its typed variants instead of string-matching `Display` output.
pub use lance::Error as LanceError;

// Re-export the Lance session type so the server can build one shared,
// capacity-bounded cache session across all resident rollout stores.
pub use lance::session::Session;
