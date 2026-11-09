//! Opaque WAL records with shared payload ownership and bounded validation.
//!
//! Serialize `BatchMeta` separately from bulk payloads on the network. Cloning
//! `Batch` shares the input `Bytes`; construction and validation each scan the
//! payload for its SHA-256 digest. No Arrow decoding is needed here.
//! A valid envelope establishes neither routing authorization nor a durable ACK.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Keep SHA-256 and its persisted bytes unchanged. Ring also has optimized
// implementations for CPUs that do not expose dedicated SHA instructions.
fn payload_digest(body: &[u8]) -> [u8; 32] {
    let digest = ring::digest::digest(&ring::digest::SHA256, body);
    let mut bytes = [0; 32];
    bytes.copy_from_slice(digest.as_ref());
    bytes
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchMeta {
    #[serde(default)]
    pub kind: RecordKind,
    /// Stable qualified table identity, distinct from an RPC connection/session.
    pub stream: Uuid,
    pub bucket: u32,
    pub producer: Uuid,
    pub sequence: u64,
    pub digest: [u8; 32],
    pub body_bytes: u64,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Batch {
    pub meta: BatchMeta,
    pub body: Bytes,
}

/// Local evidence that these exact immutable bytes match their metadata.
/// This grants no routing, producer, disk admission, or durability authority.
#[cfg(feature = "peer-transport")]
pub(crate) struct ValidatedBatch(Batch);

#[cfg(feature = "peer-transport")]
impl ValidatedBatch {
    pub(crate) fn new(batch: Batch, limit: usize) -> Result<Self, ValidationError> {
        batch.validate(limit)?;
        Ok(Self(batch))
    }

    pub(crate) fn into_inner(self) -> Batch {
        self.0
    }
}

/// Record domains used by the ingestion experiments. Control payloads have
/// their own command validation and smaller limits at the replication layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordKind {
    #[default]
    Data,
    /// Legacy single-group publication commands.
    Publication,
    /// Table-level publication plans and outcomes across source groups.
    MultiPublication,
    /// Per-group source pins, release and retention commands.
    Source,
    /// Generation-fenced producer allocation, closure and revocation.
    Producer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    EmptyBody,
    ZeroSequence,
    BodyTooLarge { bytes: usize, limit: usize },
    LengthMismatch { declared: u64, actual: usize },
    DigestMismatch,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBody => f.write_str("empty WAL record body"),
            Self::ZeroSequence => f.write_str("WAL record sequence must be positive"),
            Self::BodyTooLarge { bytes, limit } => {
                write!(
                    f,
                    "WAL record body has {bytes} bytes, exceeding limit {limit}"
                )
            }
            Self::LengthMismatch { declared, actual } => {
                write!(f, "WAL record declares {declared} bytes, received {actual}")
            }
            Self::DigestMismatch => f.write_str("WAL record payload digest mismatch"),
        }
    }
}
impl std::error::Error for ValidationError {}

impl BatchMeta {
    /// Validate the declared frame length before reserving/reading its body.
    /// The caller supplies the applicable data/control limit. This does not
    /// validate table ownership, placement, producer fencing or command contents.
    pub fn check_body_length(&self, actual: usize, limit: usize) -> Result<(), ValidationError> {
        if actual > limit {
            return Err(ValidationError::BodyTooLarge {
                bytes: actual,
                limit,
            });
        }
        if actual == 0 {
            return Err(ValidationError::EmptyBody);
        }
        if self.body_bytes != actual as u64 {
            return Err(ValidationError::LengthMismatch {
                declared: self.body_bytes,
                actual,
            });
        }
        if self.sequence == 0 {
            return Err(ValidationError::ZeroSequence);
        }
        Ok(())
    }

    /// Validate separately framed payload bytes without constructing or copying
    /// an envelope. Length/admission checks run before the digest scan.
    pub fn validate_payload(&self, body: &[u8], limit: usize) -> Result<(), ValidationError> {
        self.check_body_length(body.len(), limit)?;
        if self.digest != payload_digest(body) {
            return Err(ValidationError::DigestMismatch);
        }
        Ok(())
    }
}

impl Batch {
    /// Retain the caller's allocation and compute its digest once. Routing and
    /// sequence admission still belong to the caller/replicated state machine.
    pub fn new(stream: Uuid, bucket: u32, producer: Uuid, sequence: u64, body: Bytes) -> Self {
        let digest = payload_digest(&body);
        let body_bytes = body.len() as u64;
        Self {
            meta: BatchMeta {
                kind: RecordKind::Data,
                stream,
                bucket,
                producer,
                sequence,
                digest,
                body_bytes,
            },
            body,
        }
    }

    /// Check bounds and the complete opaque payload before admitting an append.
    pub fn validate(&self, limit: usize) -> Result<(), ValidationError> {
        self.meta.validate_payload(&self.body, limit)
    }
}

impl std::fmt::Debug for Batch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batch")
            .field("meta", &self.meta)
            .field("bytes", &self.body.len())
            .finish()
    }
}

#[cfg(test)]
mod tests;
