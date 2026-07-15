use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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

    fn get(&self, id: &str)
        -> impl Future<Output = ContextResult<Option<RolloutRecordDto>>> + Send;

    /// Materialize a single artifact row's offloaded `binary_payload` bytes.
    /// Returns `None` when the row or its payload is absent.
    fn get_blob(&self, id: &str) -> impl Future<Output = ContextResult<Option<Vec<u8>>>> + Send;

    fn version(&self) -> u64;

    fn checkout(&mut self, version: u64) -> impl Future<Output = ContextResult<()>> + Send;
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
}
