#![recursion_limit = "256"]

// Explicit re-exports from core (no glob to avoid recursion depth overflow)
pub use lance_context_core::serde;
pub use lance_context_core::{
    datagen_event_id, datagen_log_schema, datagen_trajectory, fold_datagen_events,
    CompactionConfig, CompactionMetrics, CompactionStats, Context, ContextEntry, ContextNamespace,
    ContextRecord, ContextStoreOptions, DatagenBlobValue, DatagenEvent, DatagenEventType,
    DatagenFailure, DatagenFieldState, DatagenItemStatus, DatagenStepCursor, DatagenStore,
    DatagenStoreOptions, DatagenTerminal, DatagenTrajectory, DatagenValue, FoldedDatagenItem,
    IdIndexType, LifecycleQueryOptions, MetadataFilter, PartitionInfo, PartitionSelector,
    PartitionSpec, RecordFilters, Relationship, RetrieveResult, RolloutFilters, RolloutRecord,
    SearchResult, Snapshot, StateMetadata, DATAGEN_SCHEMA_VERSION, LIFECYCLE_ACTIVE,
    LIFECYCLE_CONTRADICTED,
};

pub use lance_context_api::{
    AddRecordRequest, AddRecordsResponse, AddRolloutRequest, AddRolloutsResponse, CompactRequest,
    CompactResponse, CompactStatsResponse, ContextError, ContextResult, ContextStoreApi,
    CreateRolloutStoreRequest, DeleteRecordResponse, RecordDto, RelationshipDto, RetrieveRequest,
    RetrieveResponse, RetrieveResultDto, RolloutRecordDto, RolloutStoreApi, SearchResultDto,
    UpsertRecordRequest, UpsertRecordResponse,
};

#[cfg(feature = "remote")]
pub use lance_context_client::{ClientError, RemoteContextStore, RemoteRolloutStore};

mod unified;
pub use unified::ContextStore;

mod unified_rollout;
pub use unified_rollout::RolloutStore;
