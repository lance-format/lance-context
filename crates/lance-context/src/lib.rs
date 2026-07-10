#![recursion_limit = "256"]

// Explicit re-exports from core (no glob to avoid recursion depth overflow)
pub use lance_context_core::serde;
pub use lance_context_core::{
    CompactionConfig, CompactionMetrics, CompactionStats, Context, ContextEntry, ContextNamespace,
    ContextRecord, ContextStoreOptions, IdIndexType, LifecycleQueryOptions, MetadataFilter,
    PartitionInfo, PartitionSelector, PartitionSpec, RecordFilters, Relationship, RetrieveResult,
    RolloutRecord, SearchResult, Snapshot, StateMetadata, LIFECYCLE_ACTIVE, LIFECYCLE_CONTRADICTED,
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
