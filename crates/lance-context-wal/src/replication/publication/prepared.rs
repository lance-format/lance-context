//! One immutable materialization result per pending plan. Recording descriptors
//! does not authorize a Lance commit or certify that remote source pins exist.
use super::*;
use crate::ingest_contract::WritePolicy;

#[cfg(test)]
mod tests;

pub const MAX_DESCRIPTOR_BYTES: usize = 1024 * 1024;
pub(super) const MAX_PREPARE_BYTES: usize = MAX_DESCRIPTOR_BYTES + 1024;

/// Versioned materializer metadata. Replicas validate length and digest, without
/// decoding Lance fragments or Arrow payloads. Only materializers interpret it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Descriptors {
    pub format: u32,
    pub bytes: Bytes,
    pub digest: [u8; 32],
}
impl Descriptors {
    /// JSON representation of Lance 13 `Vec<Fragment>`, interpreted by the
    /// materializer. This constructor does not validate the fragment semantics.
    pub fn lance_fragments(bytes: Bytes) -> Result<Self> {
        Self::new(1, bytes)
    }

    /// JSON representation of Lance 13 `RewriteResult`, interpreted by the
    /// compactor. Replicas do not validate the referenced fragment semantics.
    pub fn lance_compaction(bytes: Bytes) -> Result<Self> {
        Self::new(2, bytes)
    }

    /// Native scalar-index metadata, interpreted and validated by the
    /// materializer. Enable only after every catalog voter supports format 3.
    pub fn lance_index(bytes: Bytes) -> Result<Self> {
        Self::new(3, bytes)
    }

    /// Native narrow-column annotation update metadata. The application must
    /// validate the operation and its source cut before publishing. Replicas
    /// enforce the table's annotation policy, length and digest only.
    /// Enable only after all voters and publishers support format 4.
    pub fn lance_annotations(bytes: Bytes) -> Result<Self> {
        Self::new(4, bytes)
    }

    fn new(format: u32, bytes: Bytes) -> Result<Self> {
        let descriptors = Self {
            format,
            digest: Sha256::digest(&bytes).into(),
            bytes,
        };
        descriptors.validate(MAX_DESCRIPTOR_BYTES as u64)?;
        Ok(descriptors)
    }

    pub(super) fn matches_plan(&self, plan: &Plan, policy: &WritePolicy) -> bool {
        if plan.is_maintenance() {
            matches!(self.format, 2 | 3)
        } else {
            match policy {
                WritePolicy::ImmutableUniqueIds => self.format == 1,
                WritePolicy::TurnAnnotations => self.format == 4,
            }
        }
    }

    pub fn validate(&self, maximum: u64) -> Result<()> {
        ensure!(
            (1..=4).contains(&self.format)
                && !self.bytes.is_empty()
                && self.bytes.len() <= MAX_DESCRIPTOR_BYTES
                && self.bytes.len() as u64 <= maximum
                && self.digest == <[u8; 32]>::from(Sha256::digest(&self.bytes)),
            "invalid prepared descriptor format, length or digest"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareRequest {
    pub authority: Authority,
    pub transaction: Uuid,
    pub plan_index: u64,
    pub descriptors: Descriptors,
}
impl PrepareRequest {
    pub(in crate::replication) fn batch(self, layout: &Layout, limits: &Limits) -> Result<Batch> {
        self.validate(layout, limits)?;
        Command::Prepare(Box::new(self)).batch(layout)
    }

    pub(super) fn validate(&self, layout: &Layout, limits: &Limits) -> Result<()> {
        self.authority.validate(layout)?;
        ensure!(
            !self.transaction.is_nil() && self.plan_index > 0 && self.plan_index < u64::MAX,
            "invalid prepared plan identity"
        );
        self.descriptors.validate(limits.max_descriptor_bytes)
    }
}

/// Shared by captured views and retained across publisher takeover. Its index
/// is the original durable preparation entry, not the latest retry entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prepared {
    pub index: u64,
    pub transaction: Uuid,
    pub plan_index: u64,
    pub descriptors: Descriptors,
}

impl Catalog {
    pub(super) fn prepare(&mut self, request: PrepareRequest, index: u64) -> Reply {
        let Some(table) = self.tables.get_mut(&request.authority.table) else {
            return Reply::Rejected(Rejection::Authority);
        };
        if table.authority != request.authority {
            return Reply::Rejected(Rejection::Authority);
        }
        let Some(pending) = &table.progress.pending else {
            return Reply::Rejected(Rejection::PlanIdentity);
        };
        if pending.index != request.plan_index || pending.plan.transaction != request.transaction {
            return Reply::Rejected(Rejection::PlanIdentity);
        }
        if !request
            .descriptors
            .matches_plan(&pending.plan, &table.ingest.policy)
        {
            return Reply::Rejected(Rejection::PreparedConflict);
        }
        if let Some(prepared) = &table.progress.prepared {
            return if prepared.descriptors == request.descriptors {
                Reply::Prepared {
                    index: prepared.index,
                }
            } else {
                Reply::Rejected(Rejection::PreparedConflict)
            };
        }
        table.progress.prepared = Some(Arc::new(Prepared {
            index,
            transaction: request.transaction,
            plan_index: request.plan_index,
            descriptors: request.descriptors,
        }));
        Reply::Prepared { index }
    }
}
