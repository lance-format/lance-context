//! Replicated table bindings and publisher epochs. These commands run through
//! the same durable Raft path as appends; a local view is not an active lease.
//! Immutable plans and prepared descriptors survive takeover. Source pins live
//! in the source module; source closure and retirement remain separate work.
//! No Lance manifest is changed here.
use super::*;
use crate::{ingest_contract::IngestContract, record::RecordKind};
use bincode::Options;
use bytes::Bytes;
mod advance;
mod manifest;
mod outcome;
mod plan;
mod prepared;
mod table_codec;
pub use advance::{AdvancePublication, Advanced};
pub use manifest::{
    BootstrapManifest, HistoryBoundary, ManifestIndex, ManifestObject, ManifestRecord,
    ReserveManifest, RetireManifestHistory, MAX_MANIFEST_VERSIONS,
};
pub use outcome::{ConfirmPublication, Published};
pub(super) use plan::MAX_PLAN_BYTES;
pub use plan::{PendingPlan, Plan, PlanRequest, Progress, SourceRange, MAX_INPUTS};
pub use prepared::{Descriptors, PrepareRequest, Prepared, MAX_DESCRIPTOR_BYTES};
#[cfg(test)]
mod tests;

#[cfg(feature = "lance-integration")]
pub mod lance;

pub(super) const MAX_ACQUIRE_BYTES: usize = 8192;
pub(super) const MAX_COMMAND_BYTES: usize = prepared::MAX_PREPARE_BYTES;
pub(super) const MAX_BASE_TABLE_BYTES: usize = MAX_ACQUIRE_BYTES + 2048;
pub(super) const MAX_TABLE_BYTES: usize = MAX_BASE_TABLE_BYTES
    + plan::MAX_PLAN_BYTES
    + prepared::MAX_PREPARE_BYTES
    + manifest::MAX_MANIFEST_INDEX_BYTES
    + outcome::MAX_OUTCOME_BYTES
    + advance::MAX_ADVANCED_BYTES;
pub(super) const fn advance_metadata_bytes() -> u64 {
    advance::MAX_ADVANCED_BYTES as u64
}
fn codec(limit: usize) -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .with_limit(limit as u64)
        .reject_trailing_bytes()
}

/// The authorized catalog supplies one stable storage namespace and a canonical
/// dataset key within it. Raw aliases or client-provided URIs are not identities.
/// This representation does not discover aliases or authenticate the caller.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Destination {
    pub namespace: Uuid,
    pub dataset: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Acquire {
    pub table: Uuid,
    pub destination: Destination,
    pub ingest: IngestContract,
    pub owner: Uuid,
    /// Stable across an ambiguous retry; use a fresh UUID for a new takeover.
    pub request: Uuid,
    pub expected_epoch: u64,
}
impl Acquire {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.table.is_nil() && !self.owner.is_nil() && !self.request.is_nil(),
            "nil publisher identity"
        );
        validate_binding(&self.destination, &self.ingest)?;
        ensure!(self.expected_epoch < u64::MAX, "publisher epoch exhausted");
        Ok(())
    }
    pub(super) fn batch(self, layout: &Layout) -> Result<Batch> {
        self.validate()?;
        Command::Acquire(Box::new(self)).batch(layout)
    }
}

#[derive(Serialize, Deserialize)]
enum Command {
    Acquire(Box<Acquire>),
    Plan(Box<PlanRequest>),
    Prepare(Box<PrepareRequest>),
    BootstrapManifest(Box<BootstrapManifest>),
    ReserveManifest(Box<ReserveManifest>),
    ConfirmPublication(Box<ConfirmPublication>),
    AdvancePublication(Box<AdvancePublication>),
    RetireManifestHistory(Box<RetireManifestHistory>),
}
impl Command {
    fn identity(&self) -> (Uuid, Uuid, u64) {
        match self {
            Self::Acquire(a) => (a.table, a.owner, a.expected_epoch + 1),
            Self::Plan(p) => (p.authority.table, p.authority.owner, p.authority.epoch),
            Self::Prepare(p) => (p.authority.table, p.authority.owner, p.authority.epoch),
            Self::BootstrapManifest(p) => (p.authority.table, p.authority.owner, p.authority.epoch),
            Self::ReserveManifest(p) => (p.authority.table, p.authority.owner, p.authority.epoch),
            Self::ConfirmPublication(p) => {
                (p.authority.table, p.authority.owner, p.authority.epoch)
            }
            Self::AdvancePublication(p) => {
                (p.authority.table, p.authority.owner, p.authority.epoch)
            }
            Self::RetireManifestHistory(p) => {
                (p.authority.table, p.authority.owner, p.authority.epoch)
            }
        }
    }
    fn byte_limit(&self) -> usize {
        match self {
            Self::Acquire(_) => MAX_ACQUIRE_BYTES,
            Self::Plan(_) => plan::MAX_PLAN_BYTES,
            Self::Prepare(_) => prepared::MAX_PREPARE_BYTES,
            Self::BootstrapManifest(_)
            | Self::ReserveManifest(_)
            | Self::RetireManifestHistory(_) => manifest::MAX_MANIFEST_COMMAND_BYTES,
            Self::ConfirmPublication(_) => outcome::MAX_OUTCOME_BYTES,
            Self::AdvancePublication(_) => advance::MAX_ADVANCE_BYTES,
        }
    }
    fn batch(self, layout: &Layout) -> Result<Batch> {
        let (table, owner, sequence) = self.identity();
        let limit = self.byte_limit();
        let envelope = Envelope {
            version: 5,
            cluster: layout.cluster,
            layout: layout.fingerprint()?,
            group: control_group(layout),
            command: self,
        };
        let bytes = codec(limit).serialize(&envelope)?;
        ensure!(bytes.len() <= limit, "publisher command too large");
        let mut batch = Batch::new(table, 0, owner, sequence, Bytes::from(bytes));
        batch.meta.kind = RecordKind::MultiPublication;
        Ok(batch)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Authority {
    pub cluster: Uuid,
    pub layout: [u8; 32],
    pub group: u64,
    pub table: Uuid,
    pub owner: Uuid,
    pub epoch: u64,
    pub grant_index: u64,
}

impl Authority {
    fn validate(&self, layout: &Layout) -> Result<()> {
        let a = self;
        ensure!(
            a.cluster == layout.cluster
                && a.layout == layout.fingerprint()?
                && a.group == control_group(layout)
                && !a.owner.is_nil()
                && a.epoch > 0
                && a.epoch <= a.grant_index
                && a.grant_index < u64::MAX,
            "invalid publication authority scope"
        );
        ensure!(!a.table.is_nil(), "nil authority table");
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Table {
    pub manifests: Option<Arc<ManifestIndex>>,
    pub destination: Destination,
    pub ingest: IngestContract,
    pub authority: Authority,
    pub request: Uuid,
    pub progress: Progress,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rejection {
    BindingConflict,
    DestinationBound,
    Epoch { current: u64 },
    RequestReused,
    Capacity,
    Authority,
    PendingPlan,
    SourcePosition,
    PlanIdentity,
    PreparedConflict,
    ManifestIdentity,
    ManifestPredecessor,
    PublicationIdentity,
    SourceClosure,
    ManifestHistoryBoundary,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reply {
    Acquired(Authority),
    Planned { index: u64 },
    Rejected(Rejection),
    Prepared { index: u64 },
    ManifestReserved(Arc<ManifestRecord>),
    ManifestTaken(Arc<ManifestRecord>),
    Published(Arc<Published>),
    Advanced(Arc<Advanced>),
    ManifestHistory(HistoryBoundary),
    ManifestRetired(HistoryBoundary),
}

#[derive(Serialize, Deserialize)]
pub(super) struct Envelope {
    version: u32,
    cluster: Uuid,
    layout: [u8; 32],
    group: u64,
    command: Command,
}
pub(super) fn control_group(layout: &Layout) -> u64 {
    *layout
        .groups
        .first_key_value()
        .expect("validated nonempty layout")
        .0
}
fn validate_binding(destination: &Destination, ingest: &IngestContract) -> Result<()> {
    ensure!(
        !destination.namespace.is_nil()
            && !destination.dataset.is_empty()
            && destination.dataset.len() <= 4096,
        "invalid canonical destination"
    );
    ingest.validate()?;
    Ok(())
}
pub(super) fn decode(bytes: &[u8]) -> Result<Envelope> {
    ensure!(
        bytes.len() <= MAX_COMMAND_BYTES,
        "publisher command too large"
    );
    let envelope: Envelope = codec(MAX_COMMAND_BYTES).deserialize(bytes)?;
    ensure!(envelope.version == 5, "unsupported publisher command");
    ensure!(
        bytes.len() <= envelope.command.byte_limit(),
        "publisher command type byte limit"
    );
    if let Command::Acquire(a) = &envelope.command {
        a.validate()?;
    }
    ensure!(
        codec(MAX_COMMAND_BYTES).serialize(&envelope)? == bytes,
        "noncanonical publisher command"
    );
    Ok(envelope)
}
pub(super) fn validate_record(
    layout: &Layout,
    group: u64,
    limits: &Limits,
    meta: &BatchMeta,
    body: &[u8],
) -> Result<()> {
    let e = decode(body)?;
    let (table, owner, sequence) = e.command.identity();
    ensure!(
        group == control_group(layout)
            && e.group == group
            && e.cluster == layout.cluster
            && e.layout == layout.fingerprint()?
            && meta.kind == RecordKind::MultiPublication
            && meta.stream == table
            && meta.bucket == 0
            && meta.producer == owner
            && meta.sequence == sequence,
        "publisher scope/envelope or ingestion limit mismatch"
    );
    match e.command {
        Command::Acquire(a) => ensure!(
            u64::from(a.ingest.max_record_bytes) <= limits.max_record_bytes
                && u64::from(a.ingest.max_record_bytes) <= limits.max_plan_source_bytes,
            "unserviceable ingestion limit"
        ),
        Command::Plan(p) => p.validate(layout, limits)?,
        Command::Prepare(p) => p.validate(layout, limits)?,
        Command::BootstrapManifest(p) => p.validate(layout)?,
        Command::ReserveManifest(p) => p.validate(layout)?,
        Command::RetireManifestHistory(p) => p.validate(layout)?,
        Command::ConfirmPublication(p) => p.validate(layout)?,
        Command::AdvancePublication(p) => p.validate(layout, limits)?,
    }
    Ok(())
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Catalog {
    pub tables: BTreeMap<Uuid, Table>,
}
impl Catalog {
    pub fn apply(&mut self, e: Envelope, index: u64, limits: &Limits) -> Reply {
        let a = match e.command {
            Command::Plan(p) => return self.plan(*p, index, limits),
            Command::Prepare(request) => return self.prepare(*request, index),
            Command::BootstrapManifest(request) => {
                return self.bootstrap_manifest(*request, index, limits)
            }
            Command::ReserveManifest(request) => {
                return self.reserve_manifest(*request, index, limits)
            }
            Command::ConfirmPublication(request) => {
                return self.confirm_publication(*request, index)
            }
            Command::AdvancePublication(request) => {
                return self.advance_publication(*request, index)
            }
            Command::RetireManifestHistory(request) => {
                return self.retire_manifest_history(*request)
            }
            Command::Acquire(a) => *a,
        };
        let old = self.tables.get(&a.table);
        if let Some(old) = old {
            if old.destination != a.destination || old.ingest != a.ingest {
                return Reply::Rejected(Rejection::BindingConflict);
            }
            if old.request == a.request {
                return if old.authority.owner == a.owner
                    && old.authority.epoch == a.expected_epoch + 1
                {
                    Reply::Acquired(old.authority.clone())
                } else {
                    Reply::Rejected(Rejection::RequestReused)
                };
            }
        }
        let current = old.map_or(0, |old| old.authority.epoch);
        if a.expected_epoch != current {
            return Reply::Rejected(Rejection::Epoch { current });
        }
        if old.is_none() {
            if self.tables.values().any(|t| t.destination == a.destination) {
                return Reply::Rejected(Rejection::DestinationBound);
            }
            if self.tables.len() as u64 >= limits.max_tables {
                return Reply::Rejected(Rejection::Capacity);
            }
        }
        let authority = Authority {
            cluster: e.cluster,
            layout: e.layout,
            group: e.group,
            table: a.table,
            owner: a.owner,
            epoch: current + 1,
            grant_index: index,
        };
        let progress = old.map(|table| table.progress.clone()).unwrap_or_default();
        let manifests = old.and_then(|table| table.manifests.clone());
        self.tables.insert(
            a.table,
            Table {
                manifests,
                destination: a.destination,
                ingest: a.ingest,
                authority: authority.clone(),
                request: a.request,
                progress,
            },
        );
        Reply::Acquired(authority)
    }
    pub fn validate(&self, layout: &Layout, group: u64, cut: u64, limits: &Limits) -> Result<()> {
        ensure!(
            self.manifest_records() as u64 <= limits.max_manifest_records,
            "manifest record capacity"
        );
        ensure!(
            self.tables.len() as u64 <= limits.max_tables,
            "publisher catalog exceeds limit"
        );
        ensure!(
            self.tables
                .values()
                .filter(|table| table.progress.pending.is_some())
                .count() as u64
                <= limits.max_active_plans,
            "active publication plan limit"
        );
        ensure!(
            group == control_group(layout) || self.tables.is_empty(),
            "catalog in source group"
        );
        let fingerprint = layout.fingerprint()?;
        let mut destinations = std::collections::BTreeSet::new();
        for (id, table) in &self.tables {
            validate_binding(&table.destination, &table.ingest)?;
            table.progress.validate(layout, table, cut, limits)?;
            if let Some(manifests) = &table.manifests {
                manifests.validate(table, cut)?;
            }
            let a = &table.authority;
            ensure!(
                !id.is_nil()
                    && *id == a.table
                    && !a.owner.is_nil()
                    && !table.request.is_nil()
                    && a.cluster == layout.cluster
                    && a.layout == fingerprint
                    && a.group == group
                    && a.epoch > 0
                    && a.epoch <= a.grant_index
                    && a.grant_index <= cut
                    && u64::from(table.ingest.max_record_bytes) <= limits.max_record_bytes
                    && u64::from(table.ingest.max_record_bytes) <= limits.max_plan_source_bytes
                    && destinations.insert(&table.destination),
                "invalid publisher checkpoint state"
            );
        }
        Ok(())
    }
}
pub(super) fn encode_table(table: &Table) -> Result<Vec<u8>> {
    table_codec::encode(table)
}
pub(super) fn decode_table(bytes: &[u8]) -> Result<Table> {
    table_codec::decode(bytes)
}
