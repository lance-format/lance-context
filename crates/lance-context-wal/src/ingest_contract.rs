//! Immutable ingestion metadata shared by a table's ingress and publisher.
//!
//! These types validate the representation, not persistence or enforcement.
//! Callers must compare the complete contract when binding a generation and
//! validate actual Arrow input before ACK. This is not authentication, producer
//! deduplication, cross-producer ID uniqueness or a table migration protocol.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(feature = "arrow-contract")]
mod arrow;
#[cfg(feature = "arrow-contract")]
pub use arrow::arrow_schema_fingerprint;

/// Protocol ceiling. Deployments may set a smaller maximum in each contract;
/// ingress and materializers must both have capacity for that maximum.
pub const MAX_RECORD_BYTES: u32 = 255 * 1024 * 1024;
pub const MAX_ROWS: u32 = 1_048_576;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WritePolicy {
    /// Caller owns unique immutable row IDs; producer dedup is a separate layer.
    ImmutableUniqueIds,
    /// Ordered patches to existing turns' narrow annotations. This is a new
    /// binding, not permission to reinterpret an immutable append stream.
    /// Enable only after all voters and application workers support it.
    TurnAnnotations,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestContract {
    /// Empty is permitted for a deployment without tenant routing. Otherwise
    /// callers supply a canonical tenant key, not an untrusted upload field.
    pub tenant: String,
    pub qualified_table: String,
    /// Fingerprint of a canonical Arrow schema document, including metadata.
    pub schema: [u8; 32],
    pub policy: WritePolicy,
    pub max_record_bytes: u32,
    pub max_rows: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationError {
    TenantTooLong,
    InvalidTable,
    MissingSchema,
    InvalidRecordLimit,
    InvalidRowLimit,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TenantTooLong => "ingestion tenant exceeds 1024 UTF-8 bytes",
            Self::InvalidTable => "ingestion table must contain 1..=1024 UTF-8 bytes",
            Self::MissingSchema => "ingestion schema fingerprint must be nonzero",
            Self::InvalidRecordLimit => "ingestion record limit exceeds protocol bounds",
            Self::InvalidRowLimit => "ingestion row limit exceeds protocol bounds",
        })
    }
}
impl std::error::Error for ValidationError {}

impl IngestContract {
    /// Validate after decoding and before persisting or comparing a contract.
    /// Serde decoding itself does not establish validity. Byte limits do not
    /// reserve memory, and a schema fingerprint cannot validate an Arrow body.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.tenant.len() > 1024 {
            return Err(ValidationError::TenantTooLong);
        }
        if self.qualified_table.is_empty() || self.qualified_table.len() > 1024 {
            return Err(ValidationError::InvalidTable);
        }
        if self.schema == [0; 32] {
            return Err(ValidationError::MissingSchema);
        }
        if !(1..=MAX_RECORD_BYTES).contains(&self.max_record_bytes) {
            return Err(ValidationError::InvalidRecordLimit);
        }
        if !(1..=MAX_ROWS).contains(&self.max_rows) {
            return Err(ValidationError::InvalidRowLimit);
        }
        Ok(())
    }
}

/// Hash a caller-canonicalized schema document. This hashes bytes as provided:
/// object-key order/metadata normalization belong to the caller. Replicas can
/// retain this identity without Arrow dependencies or decoding payloads.
pub fn schema_fingerprint(canonical_document: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"lance-context/ingest-arrow-schema/v1");
    digest.update((canonical_document.len() as u64).to_le_bytes());
    digest.update(canonical_document);
    digest.finalize().into()
}

#[cfg(test)]
mod tests;
