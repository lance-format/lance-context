#![recursion_limit = "256"]

// Explicit re-exports from core (no glob to avoid recursion depth overflow)
pub use lance_context_core::serde;
pub use lance_context_core::{
    CompactionConfig, CompactionMetrics, CompactionStats, Context, ContextEntry, ContextRecord,
    ContextStoreOptions, IdIndexType, LifecycleQueryOptions, MetadataFilter, RecordFilters,
    Relationship, RetrieveResult, SearchResult, Snapshot, StateMetadata, LIFECYCLE_ACTIVE,
    LIFECYCLE_CONTRADICTED,
};

pub use lance_context_api::{
    AddRecordRequest, AddRecordsResponse, CompactRequest, CompactResponse, CompactStatsResponse,
    ContextError, ContextResult, ContextStoreApi, DeleteRecordResponse, RecordDto, RelationshipDto,
    RetrieveRequest, RetrieveResponse, RetrieveResultDto, SearchResultDto,
};

#[cfg(feature = "remote")]
pub use lance_context_client::{ClientError, RemoteContextStore};

mod unified;
pub use unified::ContextStore;
