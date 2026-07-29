pub mod schema_spec;
pub use schema_spec::{ColumnSpec, ColumnType, SchemaSpec, ID_COLUMN};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::future::Future;

// ---------------------------------------------------------------------------
// Unified error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    AlreadyExists(String),
    #[error("{0}")]
    InvalidRequest(String),
    #[error("{0}")]
    Internal(String),
    #[error("Compaction already in progress")]
    CompactionInProgress,
}

pub type ContextResult<T> = Result<T, ContextError>;

// ---------------------------------------------------------------------------
// Unified trait
// ---------------------------------------------------------------------------

pub trait ContextStoreApi {
    fn add(
        &mut self,
        records: &[AddRecordRequest],
    ) -> impl Future<Output = ContextResult<AddRecordsResponse>> + Send;

    fn upsert(
        &mut self,
        request: &UpsertRecordRequest,
    ) -> impl Future<Output = ContextResult<UpsertRecordResponse>> + Send;

    fn upsert_many(
        &mut self,
        request: &UpsertRecordsRequest,
    ) -> impl Future<Output = ContextResult<UpsertRecordsResponse>> + Send;

    fn update(
        &mut self,
        request: &UpdateRecordRequest,
    ) -> impl Future<Output = ContextResult<UpdateRecordResponse>> + Send;

    fn get(&self, id: &str) -> impl Future<Output = ContextResult<Option<RecordDto>>> + Send;

    fn get_by_external_id(
        &self,
        external_id: &str,
    ) -> impl Future<Output = ContextResult<Option<RecordDto>>> + Send;

    fn delete_by_id(
        &mut self,
        id: &str,
    ) -> impl Future<Output = ContextResult<DeleteRecordResponse>> + Send;

    fn delete_by_external_id(
        &mut self,
        external_id: &str,
    ) -> impl Future<Output = ContextResult<DeleteRecordResponse>> + Send;

    fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<Value>,
        include_expired: bool,
        include_retired: bool,
    ) -> impl Future<Output = ContextResult<Vec<RecordDto>>> + Send;

    fn related(
        &self,
        target_id: &str,
        relation: Option<&str>,
        limit: Option<usize>,
        include_expired: bool,
        include_retired: bool,
    ) -> impl Future<Output = ContextResult<Vec<RecordDto>>> + Send;

    fn search(
        &self,
        request: &SearchRequest,
    ) -> impl Future<Output = ContextResult<Vec<SearchResultDto>>> + Send;

    fn retrieve(
        &self,
        request: &RetrieveRequest,
    ) -> impl Future<Output = ContextResult<Vec<RetrieveResultDto>>> + Send;

    fn version(&self) -> u64;

    fn checkout(&mut self, version: u64) -> impl Future<Output = ContextResult<()>> + Send;

    fn compact(
        &mut self,
        options: Option<CompactRequest>,
    ) -> impl Future<Output = ContextResult<CompactResponse>> + Send;

    fn compaction_stats(&self) -> impl Future<Output = ContextResult<CompactStatsResponse>> + Send;
}

// ---------------------------------------------------------------------------
// Rollout trait
// ---------------------------------------------------------------------------

/// Remote-capable surface of a rollout store — the subset of `RolloutStore`
/// operations that cross a process boundary. A rollout store is far smaller than
/// a context store: append, read, and version, with no upsert/search/compaction.
///
/// Artifact bytes (`binary_payload`) travel inline as base64 in JSON or, for
/// large payloads on the batch endpoint, as raw multipart parts. By the time a
/// call reaches this trait the bytes are already materialized in memory, so the
/// signatures are transport-agnostic.
pub trait RolloutStoreApi {
    fn add(
        &mut self,
        records: &[AddRolloutRequest],
    ) -> impl Future<Output = ContextResult<AddRolloutsResponse>> + Send;

    fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> impl Future<Output = ContextResult<Vec<RolloutRecordDto>>> + Send;

    fn list_filtered(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        filters: Option<Value>,
    ) -> impl Future<Output = ContextResult<Vec<RolloutRecordDto>>> + Send;

    fn get_trajectory(
        &self,
        rollout_id: &str,
    ) -> impl Future<Output = ContextResult<Vec<RolloutRecordDto>>> + Send
    where
        Self: Sync,
    {
        async move {
            if rollout_id.is_empty() {
                return Err(ContextError::InvalidRequest(
                    "rollout_id must not be empty".to_string(),
                ));
            }
            let mut records = self
                .list_filtered(
                    None,
                    None,
                    Some(serde_json::json!({"rollout_id": rollout_id})),
                )
                .await?;
            records.sort_by(|left, right| {
                left.sequence_order
                    .cmp(&right.sequence_order)
                    .then_with(|| left.id.cmp(&right.id))
            });
            Ok(records)
        }
    }

    fn get(&self, id: &str)
        -> impl Future<Output = ContextResult<Option<RolloutRecordDto>>> + Send;

    /// Materialize a single artifact row's offloaded `binary_payload` bytes.
    /// Returns `None` when the row or its payload is absent.
    fn get_blob(&self, id: &str) -> impl Future<Output = ContextResult<Option<Vec<u8>>>> + Send;

    fn version(&self) -> u64;

    fn checkout(&mut self, version: u64) -> impl Future<Output = ContextResult<()>> + Send;
}

// ---------------------------------------------------------------------------
// Generic stores (user-defined schemas)
// ---------------------------------------------------------------------------

/// Create a store whose columns the caller declares.
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateGenericStoreRequest {
    /// Portable dataset name: 1-128 ASCII characters matching
    /// `[A-Za-z0-9_][A-Za-z0-9._-]*`; `_registry` and `_stats` are reserved.
    pub name: String,
    /// The store's columns. Must declare an `id` column (non-nullable string);
    /// see [`SchemaSpec`].
    pub schema: SchemaSpec,
    #[serde(default)]
    pub storage_options: Option<std::collections::HashMap<String, String>>,
    /// Whether an append seals before returning, making its rows immediately
    /// readable. Defaults to `false` — durable but not visible until a flush,
    /// which is what keeps concurrent appends from serializing.
    #[serde(default)]
    pub seal_on_add: bool,
}

/// A generic store's identity and, for single-store lookups, its schema.
#[derive(Debug, Serialize, Deserialize)]
pub struct GenericStoreInfo {
    pub name: String,
    pub uri: String,
    /// Dataset version. `None` in list responses, which are served without
    /// opening each dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u64>,
    /// The store's schema. `None` in list responses for the same reason as
    /// `version`: the schema is stored *in the dataset*, so reporting it means
    /// opening every store.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<SchemaSpec>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListGenericStoresResponse {
    pub stores: Vec<GenericStoreInfo>,
}

/// Rows to append, as JSON objects keyed by column name.
///
/// There is deliberately no per-column DTO: the schema is declared at runtime,
/// so a static struct could not describe it. Values are validated against the
/// store's schema on the way in — an undeclared key is an error rather than
/// being dropped, and an omitted nullable column is written as null.
#[derive(Debug, Serialize, Deserialize)]
pub struct AddRowsRequest {
    pub rows: Vec<Map<String, Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddRowsResponse {
    /// Base dataset version after the append. MemWAL appends do not advance it,
    /// so this identifies the table, not the rows just written.
    pub version: u64,
    /// Number of rows accepted.
    pub count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListRowsResponse {
    pub rows: Vec<Map<String, Value>>,
}

/// Remote-capable surface of a store over a user-declared schema.
///
/// Rows are `serde_json` maps in both directions, so this trait needs no DTO
/// conversion layer — the wire form *is* the row form.
pub trait GenericStoreApi {
    /// The schema this store was created with.
    fn spec(&self) -> &SchemaSpec;

    /// Append rows. Visibility on return depends on the store's `seal_on_add`.
    fn add(
        &self,
        rows: &[Map<String, Value>],
    ) -> impl Future<Output = ContextResult<AddRowsResponse>> + Send;

    /// Read rows, newest-write-wins by `id`. Blob columns are projected out.
    fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> impl Future<Output = ContextResult<Vec<Map<String, Value>>>> + Send;

    /// [`Self::list`], filtered by a SQL predicate over the store's columns.
    fn list_filtered(
        &self,
        filter: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> impl Future<Output = ContextResult<Vec<Map<String, Value>>>> + Send;

    /// Fetch one row by `id`. `columns` selects what to read; `None` reads
    /// everything except blob columns. Pass an explicit list to fetch a blob —
    /// that is the intended way to read a large payload.
    fn get(
        &self,
        id: &str,
        columns: Option<&[String]>,
    ) -> impl Future<Output = ContextResult<Option<Map<String, Value>>>> + Send;

    /// Seal the active memtable so previously added rows become readable.
    fn flush(&self) -> impl Future<Output = ContextResult<()>> + Send;

    fn version(&self) -> u64;
}

// ---------------------------------------------------------------------------
// Datagen trait
// ---------------------------------------------------------------------------

/// Remote-capable surface of a datagen checkpoint store — the append-only
/// delta-log of item lifecycle / field events plus the folded read lenses.
///
/// A datagen store has no upsert/search/compaction: writers only append events
/// (`append` for lifecycle rows, `append_checkpoint` for an atomic step
/// boundary), and readers fold an item's events into latest state
/// (`fold_item`), classify root items in bulk (`root_item_statuses`), list an
/// item's failure pointers (`item_failures`), or materialize one field's
/// offloaded blob bytes (`get_blob`).
///
/// Field blob bytes (`DatagenValueDto::bytes`) travel inline as base64 in JSON
/// or, for large payloads on the append endpoints, as raw multipart parts. By
/// the time a call reaches this trait the bytes are already materialized in
/// memory, so the signatures are transport-agnostic.
pub trait DatagenStoreApi {
    fn append(
        &mut self,
        events: &[DatagenEventDto],
    ) -> impl Future<Output = ContextResult<AddDatagenEventsResponse>> + Send;

    fn append_checkpoint(
        &mut self,
        events: &[DatagenEventDto],
    ) -> impl Future<Output = ContextResult<AddDatagenEventsResponse>> + Send;

    fn fold_item(
        &self,
        item_id: &str,
    ) -> impl Future<Output = ContextResult<Option<FoldedDatagenItemDto>>> + Send;

    /// Like [`DatagenStoreApi::fold_item`], but `load_blobs` selects the blob projection:
    /// `false` (the `fold_item` default) leaves blob fields lazy — bytes absent, resolved later
    /// through `get_blob` — while `true` materializes them inline, at the cost of reading the
    /// payload column.
    fn fold_item_with_blobs(
        &self,
        item_id: &str,
        load_blobs: bool,
    ) -> impl Future<Output = ContextResult<Option<FoldedDatagenItemDto>>> + Send;

    fn root_item_statuses(
        &self,
        root_item_ids: &[String],
    ) -> impl Future<Output = ContextResult<DatagenRootItemStatusesResponse>> + Send;

    fn item_failures(
        &self,
        item_id: &str,
    ) -> impl Future<Output = ContextResult<Vec<DatagenFailureDto>>> + Send;

    /// Raw-dump every event for a root and its projected descendants, oldest
    /// first. The transport-thin read the inspection tree is folded from
    /// client-side, so the same `DatagenItemTree` assembly runs for embedded and
    /// remote without duplicating fold logic on the server.
    fn events_for_root(
        &self,
        root_item_id: &str,
    ) -> impl Future<Output = ContextResult<Vec<DatagenEventDto>>> + Send;

    /// Aggregate the whole log into a run overview: per-status root item counts,
    /// failure counts by error type, and completed-step counts.
    fn overview(&self) -> impl Future<Output = ContextResult<DatagenRunOverviewDto>> + Send;

    /// Materialize one FIELD_* event's offloaded blob bytes by event id.
    /// Returns `None` when the event or its payload is absent.
    fn get_blob(
        &self,
        event_id: &str,
    ) -> impl Future<Output = ContextResult<Option<Vec<u8>>>> + Send;

    fn version(&self) -> u64;
}

// ---------------------------------------------------------------------------
// Context lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateContextRequest {
    /// Portable dataset name: 1-128 ASCII characters matching
    /// `[A-Za-z0-9_][A-Za-z0-9._-]*`; `_registry` and `_stats` are reserved.
    pub name: String,
    #[serde(default)]
    pub storage_options: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    pub id_index_type: Option<String>,
    #[serde(default)]
    pub blob_columns: Option<Vec<String>>,
    #[serde(default)]
    pub embedding_dim: Option<i32>,
    #[serde(default)]
    pub distance_metric: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContextInfo {
    pub name: String,
    pub uri: String,
    pub version: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListContextsResponse {
    pub contexts: Vec<ContextInfo>,
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMetadataDto {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_plan_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_used: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelationshipDto {
    pub target_id: String,
    pub relation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AddRecordRequest {
    #[serde(default = "default_role")]
    pub role: String,
    #[serde(default = "default_content_type")]
    pub content_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_payload: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_base64_opt",
        deserialize_with = "deserialize_base64_opt"
    )]
    pub binary_payload: Option<Vec<u8>>,
    /// Typed reference to a payload object stored outside the dataset
    /// (e.g. `gs://bucket/prefix/<id>`). Distinct from inline `binary_payload`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_size: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_checksum: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_metadata: Option<StateMetadataDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relationships: Vec<RelationshipDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddRecordsRequest {
    pub records: Vec<AddRecordRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddRecordsResponse {
    pub version: u64,
    pub ids: Vec<String>,
    pub count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpsertRecordRequest {
    pub record: AddRecordRequest,
    #[serde(default = "default_upsert_key")]
    pub key: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpsertRecordResponse {
    pub version: u64,
    pub inserted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaced_id: Option<String>,
    pub record: RecordDto,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpsertRecordsRequest {
    pub records: Vec<AddRecordRequest>,
    #[serde(default = "default_upsert_key")]
    pub key: String,
}

/// Per-record outcome of a batch upsert, in input order.
#[derive(Debug, Serialize, Deserialize)]
pub struct UpsertResultDto {
    pub inserted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaced_id: Option<String>,
    pub record: RecordDto,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpsertRecordsResponse {
    pub version: u64,
    pub results: Vec<UpsertResultDto>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecordPatchDto {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_metadata: Option<StateMetadataDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relationships: Option<Vec<RelationshipDto>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retired_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retired_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_size: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_checksum: Option<String>,
}

impl RecordPatchDto {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bot_id.is_none()
            && self.session_id.is_none()
            && self.tenant.is_none()
            && self.source.is_none()
            && self.state_metadata.is_none()
            && self.metadata.is_none()
            && self.relationships.is_none()
            && self.expires_at.is_none()
            && self.retention_policy.is_none()
            && self.lifecycle_status.is_none()
            && self.retired_at.is_none()
            && self.retired_reason.is_none()
            && self.embedding.is_none()
            && self.payload_uri.is_none()
            && self.payload_size.is_none()
            && self.payload_checksum.is_none()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateRecordRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
    #[serde(default)]
    pub patch: RecordPatchDto,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateRecordResponse {
    pub version: u64,
    pub updated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaced_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record: Option<RecordDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordDto {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub created_at: DateTime<Utc>,
    pub role: String,
    pub content_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_payload: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_base64_opt",
        deserialize_with = "deserialize_base64_opt"
    )]
    pub binary_payload: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_size: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_checksum: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_metadata: Option<StateMetadataDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relationships: Vec<RelationshipDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention_policy: Option<String>,
    pub lifecycle_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retired_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retired_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListRecordsResponse {
    pub records: Vec<RecordDto>,
}

// ---------------------------------------------------------------------------
// Single record lookup
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct GetRecordResponse {
    pub record: Option<RecordDto>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteRecordResponse {
    pub deleted: bool,
    pub version: u64,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<Value>,
    #[serde(default)]
    pub include_expired: bool,
    #[serde(default)]
    pub include_retired: bool,
    #[serde(default)]
    pub include_relationships: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResultDto {
    pub record: RecordDto,
    pub distance: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultDto>,
}

// ---------------------------------------------------------------------------
// Hybrid retrieval
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct RetrieveRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<Value>,
    #[serde(default)]
    pub include_expired: bool,
    #[serde(default)]
    pub include_retired: bool,
    #[serde(default)]
    pub include_relationships: bool,
    #[serde(default = "default_retrieve_fusion")]
    pub fusion: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RetrieveResultDto {
    pub record: RecordDto,
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_distance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matched_channels: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RetrieveResponse {
    pub results: Vec<RetrieveResultDto>,
}

// ---------------------------------------------------------------------------
// Versioning
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct VersionResponse {
    pub version: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckoutRequest {
    pub version: u64,
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CompactRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_rows_per_fragment: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub materialize_deletions: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompactResponse {
    pub fragments_removed: usize,
    pub fragments_added: usize,
    pub files_removed: usize,
    pub files_added: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompactStatsResponse {
    pub total_fragments: usize,
    pub is_compacting: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_compaction: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub total_compactions: u64,
}

// ---------------------------------------------------------------------------
// Rollout lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRolloutStoreRequest {
    /// Portable dataset name: 1-128 ASCII characters matching
    /// `[A-Za-z0-9_][A-Za-z0-9._-]*`; `_registry` and `_stats` are reserved.
    pub name: String,
    #[serde(default)]
    pub storage_options: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RolloutStoreInfo {
    pub name: String,
    pub uri: String,
    /// Dataset version. `None` in list responses, which are served from the
    /// durable registry without opening each dataset (a store may not be
    /// resident in memory). Single-store lookups (`get`/`create`) always
    /// populate it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListRolloutStoresResponse {
    pub stores: Vec<RolloutStoreInfo>,
}

// ---------------------------------------------------------------------------
// Rollout records
// ---------------------------------------------------------------------------

/// One rollout row to append. Unlike [`AddRecordRequest`], `id` is
/// client-supplied: rollout rows carry externally-meaningful identity (a
/// trajectory row from a harness). Because those ids are opaque and may repeat,
/// the multipart upload path matches raw binary parts to records by their
/// zero-based index in the `records` array, not by `id`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AddRolloutRequest {
    pub id: String,
    pub rollout_id: String,
    /// Defaults to `rollout_id` server-side when omitted (non-grouped rollouts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    #[serde(default)]
    pub sequence_order: i32,
    #[serde(default = "default_rollout_role")]
    pub role: String,
    /// Defaults to the server's current time when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default = "default_content_type")]
    pub content_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_input_string: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_output_string: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_metadata: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<Vec<i32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<Vec<i32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_input_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_output_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_logprobs: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_logprobs: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_logprobs: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_mask: Option<Vec<i8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advantage: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_reward: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grader_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_in_training: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_version: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relationships: Vec<RelationshipDto>,
    /// Inline artifact bytes. On the batch endpoint, large payloads may instead
    /// travel as a raw multipart part named for this record's zero-based index
    /// in the `records` array, avoiding base64 inflation.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_base64_opt",
        deserialize_with = "deserialize_base64_opt"
    )]
    pub binary_payload: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_size: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_checksum: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddRolloutsRequest {
    pub records: Vec<AddRolloutRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddRolloutsResponse {
    pub version: u64,
    pub ids: Vec<String>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutRecordDto {
    pub id: String,
    pub rollout_id: String,
    pub problem_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    pub sequence_order: i32,
    pub role: String,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    pub content_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_input_string: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_output_string: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_metadata: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<Vec<i32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<Vec<i32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_input_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_output_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_logprobs: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_logprobs: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_logprobs: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_mask: Option<Vec<i8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advantage: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_reward: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grader_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_in_training: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_version: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relationships: Vec<RelationshipDto>,
    /// Present only when materialized on demand (`get_blob`); a plain list/get
    /// scan reads the offloaded column back as `None`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_base64_opt",
        deserialize_with = "deserialize_base64_opt"
    )]
    pub binary_payload: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_size: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_checksum: Option<String>,
    /// User-defined semantic artifact category (e.g. `"excel_grade_screenshot"`),
    /// orthogonal to `content_type`. A first-class, filterable column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListRolloutsResponse {
    pub records: Vec<RolloutRecordDto>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetRolloutResponse {
    pub record: Option<RolloutRecordDto>,
}

// ---------------------------------------------------------------------------
// Datagen lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateDatagenStoreRequest {
    /// Portable dataset name: 1-128 ASCII characters matching
    /// `[A-Za-z0-9_][A-Za-z0-9._-]*`; `_registry` and `_stats` are reserved.
    pub name: String,
    #[serde(default)]
    pub storage_options: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatagenStoreInfo {
    pub name: String,
    pub uri: String,
    /// Dataset version. `None` in list responses, which are served from the
    /// durable registry without opening each dataset. Single-store lookups
    /// (`get`/`create`) always populate it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListDatagenStoresResponse {
    pub stores: Vec<DatagenStoreInfo>,
}

// ---------------------------------------------------------------------------
// Datagen values
// ---------------------------------------------------------------------------

/// The wire form of a core `DatagenValue`. `kind` tags the payload:
/// `"int"`/`"float"`/`"bool"`/`"str"`/`"json"` carry a JSON scalar in `value`;
/// `"blob"` carries raw bytes in `bytes` (inline base64 in JSON, or a raw
/// multipart part on the append endpoints) plus `size`/`checksum`. Mirrors the
/// `DatagenValue.to_wire`/`from_wire` dict shape in the Python binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatagenValueDto {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_base64_opt",
        deserialize_with = "deserialize_base64_opt"
    )]
    pub bytes: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

// ---------------------------------------------------------------------------
// Datagen events (append)
// ---------------------------------------------------------------------------

/// One append-only datagen log row. Field names and semantics mirror the core
/// `DatagenEvent` and the Python `datagen_events` wire dict; enum-valued columns
/// (`event_type`, `step_kind`, `status`) travel as their canonical strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatagenEventDto {
    pub event_id: String,
    pub item_id: String,
    pub root_item_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_item_id: Option<String>,
    pub item_seq: i64,
    pub checkpoint_id: String,
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_index: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enclosing_step: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector_step: Option<String>,
    #[serde(default)]
    pub attempt: i32,
    pub run_id: String,
    pub writer_epoch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec_version: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<DatagenValueDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_tags: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_dump: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub traceback: Option<String>,
    /// Defaults to the server's current time when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_ts: Option<DateTime<Utc>>,
    pub schema_version: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddDatagenEventsRequest {
    pub events: Vec<DatagenEventDto>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddDatagenEventsResponse {
    pub version: u64,
    pub count: usize,
}

// ---------------------------------------------------------------------------
// Datagen folded read lenses
// ---------------------------------------------------------------------------

/// A folded field: `mode = "set"` carries a single `value`; `mode = "append"`
/// carries an ordered `values` list. Mirrors the Python `FieldState` wire dict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatagenFieldStateDto {
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<DatagenValueDto>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<DatagenValueDto>,
}

/// One completed step position an item passed through (a single STEP_COMPLETED).
/// Mirrors the Python `StepCursor` wire dict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatagenStepCursorDto {
    pub step_name: String,
    pub step_kind: String,
    pub step_index: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enclosing_step: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector_step: Option<String>,
    pub item_seq: i64,
}

/// A position within one stream's step tree, without the `item_seq` a cursor carries.
/// Mirrors the Python `StepPosition` wire dict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatagenStreamPositionDto {
    pub step_name: String,
    pub step_kind: String,
    pub step_index: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enclosing_step: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector_step: Option<String>,
}

/// An item reconstructed by folding its events into latest state.
/// Mirrors the Python `FoldedItem` wire dict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldedDatagenItemDto {
    pub item_id: String,
    pub root_item_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_item_id: Option<String>,
    pub status: String,
    pub last_item_seq: i64,
    pub last_attempt: i32,
    pub fields: std::collections::BTreeMap<String, DatagenFieldStateDto>,
    pub trajectory: Vec<DatagenStepCursorDto>,
    /// Positions with a STEP_STARTED — gates driver-frame (re-)opening on resume. `started`
    /// minus `completed` = frames that were open when the process died.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub started: Vec<DatagenStreamPositionDto>,
    /// Positions with a STEP_COMPLETED — gates STEP_COMPLETED re-emission on resume.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completed: Vec<DatagenStreamPositionDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_tags: Option<Value>,
    /// `field_name -> event_id` for the folded blob fields, so a caller can resolve a blob by field
    /// name (via `load_blob`) without recomputing an `event_id`.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub blob_event_ids: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetFoldedDatagenItemResponse {
    pub item: Option<FoldedDatagenItemDto>,
}

/// Bulk startup classification of root items. A missing id means "never started".
#[derive(Debug, Serialize, Deserialize)]
pub struct DatagenRootItemStatusesResponse {
    pub statuses: std::collections::HashMap<String, String>,
}

/// One failure record for an item (the failure lens). Mirrors the Python
/// `Failure` wire dict, with the step position flattened under `at`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatagenFailureDto {
    pub at: DatagenStepCursorDto,
    pub run_id: String,
    pub attempt: i32,
    pub error_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_dump: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub traceback: Option<String>,
}

/// Whole-run aggregation over a datagen log. `items` counts root items only
/// (`running + completed + filtered`); `failures` counts FAILED events, which are
/// non-terminal, so a failed-then-retried item still counts as `running`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatagenRunOverviewDto {
    pub items: usize,
    pub running: usize,
    pub completed: usize,
    pub filtered: usize,
    pub failures: usize,
    /// FAILED-event count per `error_type`.
    #[serde(default)]
    pub failures_by_error_type: std::collections::BTreeMap<String, usize>,
    /// Failure roll-up grouped by the `run_id` that emitted the FAILED event.
    #[serde(default)]
    pub failures_by_run: std::collections::BTreeMap<String, DatagenFailureBucketDto>,
    /// STEP_COMPLETED count per step name, across every item.
    #[serde(default)]
    pub completed_steps: std::collections::BTreeMap<String, usize>,
}

/// One `run_id`'s slice of an overview's failure roll-up, with a capped sample of failing root
/// item ids to drill into.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatagenFailureBucketDto {
    pub failures: usize,
    #[serde(default)]
    pub failures_by_error_type: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    pub sample_root_item_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListDatagenFailuresResponse {
    pub failures: Vec<DatagenFailureDto>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListDatagenEventsResponse {
    pub events: Vec<DatagenEventDto>,
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_content_type() -> String {
    "text/plain".to_string()
}

fn default_role() -> String {
    "user".to_string()
}

fn default_rollout_role() -> String {
    "assistant".to_string()
}

fn default_upsert_key() -> String {
    "external_id".to_string()
}

fn default_search_limit() -> usize {
    10
}

fn default_retrieve_fusion() -> String {
    "rrf".to_string()
}

fn serialize_base64_opt<S>(data: &Option<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match data {
        Some(bytes) => serializer.serialize_some(&BASE64.encode(bytes)),
        None => serializer.serialize_none(),
    }
}

fn deserialize_base64_opt<'de, D>(deserializer: D) -> Result<Option<Vec<u8>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    match opt {
        Some(s) => BASE64
            .decode(&s)
            .map(Some)
            .map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Master control-plane DTOs
// ---------------------------------------------------------------------------

/// One row of the control-plane experiment listing, backed by the master's
/// periodically-refreshed stats table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentSummary {
    /// Logical experiment / rollout-store name (unique key).
    pub name: String,
    /// Physical dataset URI.
    pub uri: String,
    /// Logical row count across the base table and all flushed MemWAL shards.
    pub row_count: i64,
    /// Base-table fragment count as of the last scan.
    pub fragment_count: i64,
    /// Base-table manifest timestamp, Unix milliseconds.
    pub last_updated: i64,
    /// Flushed MemWAL generations pending merge as of the last scan.
    pub pending_wal_generations: i64,
    /// Timestamp of the last successful compaction, Unix ms, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_compaction: Option<i64>,
    /// Total compactions the master has driven for this experiment.
    pub total_compactions: i64,
    /// When the master last scanned this experiment, Unix milliseconds.
    pub scanned_at: i64,
}

/// Detailed view of a single experiment. Currently the same shape as
/// [`ExperimentSummary`]; a distinct type leaves room for detail-only fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentDetail {
    #[serde(flatten)]
    pub summary: ExperimentSummary,
}

/// Paginated experiment listing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentListResponse {
    pub experiments: Vec<ExperimentSummary>,
    /// Total number of experiments matching the (optional) search filter,
    /// ignoring pagination.
    pub total: i64,
}

/// Paginated rollout records for one experiment in the master data browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRecordsResponse {
    pub records: Vec<RolloutRecordDto>,
    /// Whether another page exists after this response.
    pub has_more: bool,
    pub limit: usize,
    pub offset: usize,
    /// The data source that was scanned: `"fragments"` (base table only),
    /// `"wal"` (pending MemWAL generations only), or `"all"` (the union).
    pub source: String,
}

/// Body for a read-only SQL query against one experiment's rollout records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqlQueryRequest {
    /// A single read-only `SELECT` statement. The records are exposed as a
    /// table named `records`.
    pub sql: String,
}

/// Result of a read-only SQL query: output columns plus JSON-encoded rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqlQueryResponse {
    /// Output column names, in select order.
    pub columns: Vec<String>,
    /// Rows as JSON values (one inner vec per row, aligned to `columns`).
    pub rows: Vec<Vec<serde_json::Value>>,
    /// Number of rows returned in this response.
    pub row_count: usize,
    /// True when the result was capped at the server's row limit.
    pub truncated: bool,
}

/// State of a manual or automatic compaction job for one experiment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum CompactJobStatus {
    /// Queued but not yet started.
    Queued,
    /// Currently running.
    Running,
    /// Finished successfully.
    Done {
        fragments_removed: usize,
        fragments_added: usize,
    },
    /// Failed with an error message.
    Failed { error: String },
    /// No compaction has been requested for this experiment.
    None,
}

// ---------------------------------------------------------------------------
// Unified task scheduler (master control-plane)
// ---------------------------------------------------------------------------

/// The kind of work a scheduled task performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// Compact an experiment's base table (rewrites fragments; runs on the
    /// master, serialized per experiment because two `Rewrite`s conflict).
    Compact,
    /// Fold flushed MemWAL generations back into the base table. The master
    /// cannot do this directly without fencing the live shard writer, so this
    /// task fans out to the configured worker endpoints and each worker merges
    /// its own shard.
    MergeWal,
    /// Build a ZoneMap scalar index on the experiment base table's `id` column.
    /// Runs on the master; serialized per experiment against `Compact` because
    /// both mutate the shared base table.
    IndexId,
}

/// Lifecycle state of a scheduled task, generalized from [`CompactJobStatus`]
/// so it applies uniformly to every [`TaskKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    /// Accepted and waiting for a scheduler slot.
    Queued,
    /// Currently executing.
    Running,
    /// Finished successfully.
    Done,
    /// Finished with an error (see [`TaskRecord::error`]).
    Failed,
}

/// One unit of scheduled work plus its lifecycle, as surfaced to the queue UI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskRecord {
    /// Time-ordered unique id (UUIDv7).
    pub id: String,
    pub kind: TaskKind,
    /// Experiment / rollout-store name this task acts on.
    pub target: String,
    pub state: TaskState,
    /// Error message when `state == Failed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Human-readable outcome summary when `state == Done`
    /// (e.g. `"removed 3 / added 1 fragments"` or `"merged 4 gens on 2/3 workers"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// When the task was enqueued, Unix milliseconds.
    pub enqueued_at: i64,
    /// When execution began, Unix milliseconds, if started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<i64>,
    /// When the task reached a terminal state, Unix milliseconds, if finished.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<i64>,
    /// Task ids that must reach `Done` before this task may run. The scheduler
    /// defers a task while any dependency is still `Queued`/`Running`, and marks
    /// it `Failed` if any dependency `Failed`. Empty for standalone tasks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
}

/// Request body for `POST /api/v1/tasks`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnqueueTaskRequest {
    pub kind: TaskKind,
    pub target: String,
    /// Optional task ids this task must wait for (see [`TaskRecord::depends_on`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
}

/// Response for `GET /api/v1/tasks`. Paginated newest-first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskListResponse {
    pub tasks: Vec<TaskRecord>,
    /// Total number of tasks (queue + retained history) before paging.
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_request_legacy_payload_defaults_filters_and_lifecycle() {
        // Clients written against the pre-#89 shape send only query/limit.
        let req: SearchRequest =
            serde_json::from_str(r#"{"query": [0.1, 0.2], "limit": 5}"#).unwrap();
        assert_eq!(req.query, vec![0.1, 0.2]);
        assert_eq!(req.limit, 5);
        assert!(req.filters.is_none());
        assert!(!req.include_expired);
        assert!(!req.include_retired);
        assert!(!req.include_relationships);
    }

    #[test]
    fn search_request_defaults_limit_when_omitted() {
        let req: SearchRequest = serde_json::from_str(r#"{"query": [1.0]}"#).unwrap();
        assert_eq!(req.limit, default_search_limit());
    }

    #[test]
    fn search_request_parses_filters_and_lifecycle() {
        let req: SearchRequest = serde_json::from_str(
            r#"{"query": [1.0], "filters": {"tenant": "acme"}, "include_expired": true, "include_retired": true}"#,
        )
        .unwrap();
        assert_eq!(req.filters, Some(serde_json::json!({"tenant": "acme"})));
        assert!(req.include_expired);
        assert!(req.include_retired);
    }

    #[test]
    fn add_request_omits_payload_reference_when_absent() {
        // Records without an external reference must not emit the new keys, so
        // older servers/clients keep round-tripping unchanged.
        let req = AddRecordRequest {
            role: "user".to_string(),
            content_type: "text/plain".to_string(),
            text_payload: Some("hi".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("payload_uri"));
        assert!(!json.contains("payload_size"));
        assert!(!json.contains("payload_checksum"));
    }

    #[test]
    fn add_request_roundtrips_payload_reference() {
        let req = AddRecordRequest {
            role: "user".to_string(),
            content_type: "image/png".to_string(),
            payload_uri: Some("gs://bucket/prefix/obj.png".to_string()),
            payload_size: Some(2048),
            payload_checksum: Some("sha256:abc".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: AddRecordRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.payload_uri.as_deref(),
            Some("gs://bucket/prefix/obj.png")
        );
        assert_eq!(back.payload_size, Some(2048));
        assert_eq!(back.payload_checksum.as_deref(), Some("sha256:abc"));
    }

    #[test]
    fn record_dto_decodes_payload_reference_and_legacy_shape() {
        // New shape with a reference.
        let dto: RecordDto = serde_json::from_str(
            r#"{"id":"r1","run_id":"run","created_at":"2026-06-27T00:00:00Z","role":"user","content_type":"image/png","lifecycle_status":"active","payload_uri":"s3://b/obj","payload_size":10}"#,
        )
        .unwrap();
        assert_eq!(dto.payload_uri.as_deref(), Some("s3://b/obj"));
        assert_eq!(dto.payload_size, Some(10));
        assert_eq!(dto.payload_checksum, None);

        // Legacy shape lacking the reference fields still decodes.
        let legacy: RecordDto = serde_json::from_str(
            r#"{"id":"r1","run_id":"run","created_at":"2026-06-27T00:00:00Z","role":"user","content_type":"text/plain","lifecycle_status":"active"}"#,
        )
        .unwrap();
        assert_eq!(legacy.payload_uri, None);
    }

    #[test]
    fn task_record_roundtrips_and_omits_empty_optionals() {
        let queued = TaskRecord {
            id: "0190-abc".to_string(),
            kind: TaskKind::MergeWal,
            target: "exp-1".to_string(),
            state: TaskState::Queued,
            error: None,
            detail: None,
            enqueued_at: 1_700_000_000_000,
            started_at: None,
            finished_at: None,
            depends_on: Vec::new(),
        };
        let json = serde_json::to_string(&queued).unwrap();
        // snake_case tags for the enums.
        assert!(json.contains(r#""kind":"merge_wal""#));
        assert!(json.contains(r#""state":"queued""#));
        // Absent optionals are not serialized.
        assert!(!json.contains("error"));
        assert!(!json.contains("started_at"));
        assert!(!json.contains("finished_at"));
        assert!(!json.contains("depends_on"));
        let back: TaskRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, queued);
    }

    #[test]
    fn task_record_done_and_failed_carry_details() {
        let done = TaskRecord {
            id: "id".to_string(),
            kind: TaskKind::Compact,
            target: "exp".to_string(),
            state: TaskState::Done,
            error: None,
            detail: Some("removed 3 / added 1 fragments".to_string()),
            enqueued_at: 1,
            started_at: Some(2),
            finished_at: Some(3),
            depends_on: Vec::new(),
        };
        let back: TaskRecord =
            serde_json::from_str(&serde_json::to_string(&done).unwrap()).unwrap();
        assert_eq!(back, done);
    }

    #[test]
    fn enqueue_task_request_parses_snake_case_kind() {
        let req: EnqueueTaskRequest =
            serde_json::from_str(r#"{"kind":"compact","target":"exp-7"}"#).unwrap();
        assert_eq!(req.kind, TaskKind::Compact);
        assert_eq!(req.target, "exp-7");
    }

    #[test]
    fn task_kind_index_id_roundtrips_snake_case() {
        assert_eq!(
            serde_json::to_string(&TaskKind::IndexId).unwrap(),
            r#""index_id""#
        );
        let back: TaskKind = serde_json::from_str(r#""index_id""#).unwrap();
        assert_eq!(back, TaskKind::IndexId);
        let req: EnqueueTaskRequest =
            serde_json::from_str(r#"{"kind":"index_id","target":"exp-9"}"#).unwrap();
        assert_eq!(req.kind, TaskKind::IndexId);
    }
}
