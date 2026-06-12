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

    fn get(&self, id: &str) -> impl Future<Output = ContextResult<Option<RecordDto>>> + Send;

    fn list(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> impl Future<Output = ContextResult<Vec<RecordDto>>> + Send;

    fn search(
        &self,
        query: &[f32],
        limit: Option<usize>,
        include_relationships: bool,
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
// Context lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateContextRequest {
    pub name: String,
    #[serde(default)]
    pub storage_options: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    pub id_index_type: Option<String>,
    #[serde(default)]
    pub blob_columns: Option<Vec<String>>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
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

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
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
