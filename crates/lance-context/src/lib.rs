#![recursion_limit = "256"]

// Explicit re-exports from core (no glob to avoid recursion depth overflow)
pub use lance_context_core::serde;
pub use lance_context_core::{
    datagen_event_id, datagen_event_to_dto, datagen_log_schema, datagen_trajectory,
    fold_datagen_events, folded_item_to_dto, open_stream_events, CompactionConfig,
    CompactionMetrics, CompactionStats, Context, ContextEntry, ContextNamespace, ContextRecord,
    ContextStoreOptions, DatagenBlobValue, DatagenErrorInfo, DatagenEvent, DatagenEventType,
    DatagenFailure, DatagenFieldChange, DatagenFieldState, DatagenItemId, DatagenItemNode,
    DatagenItemStatus, DatagenItemTree, DatagenNewStream, DatagenOpenStream, DatagenStepCursor,
    DatagenStepId, DatagenStepKind, DatagenStoreOptions, DatagenStreamPosition,
    DatagenStreamWriter, DatagenTerminal, DatagenTrajectory, DatagenValue, DatagenWriteContext,
    FieldOp, FoldedDatagenItem, IdIndexType, LifecycleQueryOptions, MetadataFilter, PartitionInfo,
    PartitionSelector, PartitionSpec, RecordFilters, Relationship, RetrieveResult, RolloutFilters,
    RolloutRecord, SearchResult, Snapshot, StateMetadata, DATAGEN_SCHEMA_VERSION, LIFECYCLE_ACTIVE,
    LIFECYCLE_CONTRADICTED,
};

pub use lance_context_api::{
    AddDatagenEventsRequest, AddDatagenEventsResponse, AddRecordRequest, AddRecordsResponse,
    AddRolloutRequest, AddRolloutsResponse, AddRowsRequest, AddRowsResponse, ColumnSpec,
    ColumnType, CompactRequest, CompactResponse, CompactStatsResponse, ContextError, ContextResult,
    ContextStoreApi, CreateDatagenStoreRequest, CreateGenericStoreRequest,
    CreateRolloutStoreRequest, DatagenEventDto, DatagenFailureBucketDto, DatagenFailureDto,
    DatagenFieldStateDto, DatagenRootItemStatusesResponse, DatagenRunOverviewDto,
    DatagenStepCursorDto, DatagenStoreApi, DatagenStreamPositionDto, DatagenValueDto,
    DeleteRecordResponse, FoldedDatagenItemDto, GenericStoreApi, GenericStoreInfo, RecordDto,
    RelationshipDto, RetrieveRequest, RetrieveResponse, RetrieveResultDto, RolloutRecordDto,
    RolloutStoreApi, SchemaSpec, SearchResultDto, UpsertRecordRequest, UpsertRecordResponse,
    ID_COLUMN,
};

#[cfg(feature = "remote")]
pub use lance_context_client::{
    ClientError, RemoteContextStore, RemoteDatagenStore, RemoteGenericStore, RemoteRolloutStore,
};

mod unified;
pub use unified::ContextStore;

mod unified_rollout;
pub use unified_rollout::RolloutStore;

mod unified_datagen;
pub use unified_datagen::DatagenStore;

mod unified_generic;
pub use unified_generic::GenericStore;
