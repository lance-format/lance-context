//! Signed catalog observations and durable, complete source-prefix pins.
//! Attestations are issued only after safe ReadIndex completion. They are not
//! publisher leases or permission to commit/delete external Lance objects.
use super::*;
use crate::{ingest_contract::IngestContract, record::RecordKind};
use bincode::Options;
use bytes::Bytes;
use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};
mod closure;
mod proof;
pub(super) use closure::{decode_closed, encode_closed, MAX_CLOSED_BYTES};
pub use closure::{Closed, Closure, ClosurePermit};
pub use proof::{ClosedProof, SourceSigner};

pub(super) struct QuerySigners<'a> {
    pub catalog: Option<&'a CatalogSigner>,
    pub source: Option<&'a SourceSigner>,
}

#[cfg(test)]
mod tests;

// A source pin includes the selected plan range plus its signed envelope and
// ingestion contract. Keep its bound coupled to the catalog plan bound.
pub(super) const MAX_BYTES: usize = publication::MAX_PLAN_BYTES + 8 * 1024;
pub(super) const MAX_PIN_BYTES: usize = MAX_BYTES + 8;
const DOMAIN: &str = "lance-context/catalog-source-pin/v1";

fn codec(limit: usize) -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .with_limit(limit as u64)
        .reject_trailing_bytes()
}

/// Private key holder for trusted catalog replicas. It has no public signing
/// operation and is neither serializable nor printable. Provision the same key
/// to catalog replicas; persist only its public key in `Limits`.
pub struct CatalogSigner(Ed25519KeyPair);
impl CatalogSigner {
    pub fn from_pkcs8(bytes: &[u8]) -> Result<Self> {
        Ok(Self(Ed25519KeyPair::from_pkcs8(bytes).map_err(|_| {
            anyhow::anyhow!("invalid catalog signing key")
        })?))
    }
    pub fn public_key(&self) -> [u8; 32] {
        self.0
            .public_key()
            .as_ref()
            .try_into()
            .expect("Ed25519 key")
    }
}

/// Stable across refreshed observations and publisher takeover. The catalog
/// index identifies a plan; it is never compared with another group's index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub table: Uuid,
    pub transaction: Uuid,
    pub plan_index: u64,
    pub read_version: u64,
    pub group: u64,
    pub ingest: IngestContract,
    pub range: publication::SourceRange,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permit {
    policy: [u8; 32],
    cut: query::Cut,
    selection: Selection,
    // Two fixed arrays preserve a fixed-size serde representation.
    signature: ([u8; 32], [u8; 32]),
}
impl Permit {
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    pub fn catalog_cut(&self) -> &query::Cut {
        &self.cut
    }
    fn signed_bytes(&self) -> Result<Vec<u8>> {
        Ok(codec(MAX_BYTES).serialize(&(DOMAIN, self.policy, &self.cut, &self.selection))?)
    }
    pub(super) fn issue(
        signer: &CatalogSigner,
        cut: query::Cut,
        layout: &Layout,
        limits: &Limits,
        table: &publication::Table,
        group: u64,
    ) -> Result<Self> {
        ensure!(
            limits.catalog_key == Some(signer.public_key()),
            "catalog signing policy mismatch"
        );
        let pending = table.progress.pending.as_ref().context("no pending plan")?;
        let range = pending
            .plan
            .sources
            .get(&group)
            .context("source not selected")?
            .clone();
        let mut permit = Self {
            policy: Sha256::digest(limits.identity()?).into(),
            cut,
            selection: Selection {
                table: table.authority.table,
                transaction: pending.plan.transaction,
                plan_index: pending.index,
                read_version: pending.plan.read_version,
                group,
                ingest: table.ingest.clone(),
                range,
            },
            signature: ([0; 32], [0; 32]),
        };
        let signature = signer.0.sign(&permit.signed_bytes()?);
        permit.signature = (
            signature.as_ref()[..32].try_into()?,
            signature.as_ref()[32..].try_into()?,
        );
        permit.verify(layout, limits, group)?;
        Ok(permit)
    }

    fn verify(&self, layout: &Layout, limits: &Limits, group: u64) -> Result<()> {
        let key = limits
            .catalog_key
            .context("source pins require a catalog verification key")?;
        ensure!(
            self.policy == <[u8; 32]>::from(Sha256::digest(limits.identity()?)),
            "foreign pin policy"
        );
        let mut signature = [0; 64];
        signature[..32].copy_from_slice(&self.signature.0);
        signature[32..].copy_from_slice(&self.signature.1);
        UnparsedPublicKey::new(&ED25519, key)
            .verify(&self.signed_bytes()?, &signature)
            .map_err(|_| anyhow::anyhow!("invalid catalog pin attestation"))?;
        let s = &self.selection;
        ensure!(
            self.cut.cluster == layout.cluster
                && self.cut.layout == layout.fingerprint()?
                && self.cut.group == publication::control_group(layout)
                && self.cut.term > 0
                && self.cut.quorum_index > 0
                && self.cut.quorum_index <= self.cut.applied
                && self.cut.applied < u64::MAX
                && s.plan_index > 0
                && s.plan_index <= self.cut.applied
                && !s.table.is_nil()
                && !s.transaction.is_nil()
                && s.group == group
                && s.read_version > 0
                && s.read_version < u64::MAX,
            "invalid catalog pin scope or cut"
        );
        s.ingest.validate()?;
        ensure!(
            u64::from(s.ingest.max_record_bytes) <= limits.max_record_bytes
                && u64::from(s.ingest.max_record_bytes) <= limits.max_plan_source_bytes,
            "unserviceable pin contract"
        );
        s.range.validate(layout, s.table, group, limits)?;
        ensure!(
            s.range
                .records
                .iter()
                .all(|(_, m)| m.body_bytes <= u64::from(s.ingest.max_record_bytes))
                && (group != self.cut.group || s.range.through < s.plan_index),
            "pin source exceeds its contract or catalog plan"
        );
        Ok(())
    }

    pub(super) fn batch(self, layout: &Layout, limits: &Limits) -> Result<Batch> {
        self.verify(layout, limits, self.selection.group)?;
        let (table, transaction, plan_index) = (
            self.selection.table,
            self.selection.transaction,
            self.selection.plan_index,
        );
        let bytes = codec(MAX_BYTES).serialize(&Command::Pin(self))?;
        ensure!(bytes.len() <= MAX_BYTES, "oversized source pin command");
        let mut batch = Batch::new(table, 0, transaction, plan_index, Bytes::from(bytes));
        batch.meta.kind = RecordKind::Source;
        Ok(batch)
    }
}

#[derive(Serialize, Deserialize)]
pub(super) enum Command {
    Pin(Permit),
    Close(ClosurePermit),
}
pub(super) fn decode(bytes: &[u8]) -> Result<Command> {
    ensure!(bytes.len() <= MAX_BYTES, "oversized source pin command");
    let permit = codec(MAX_BYTES).deserialize(bytes)?;
    ensure!(
        codec(MAX_BYTES).serialize(&permit)? == bytes,
        "noncanonical source pin"
    );
    Ok(permit)
}
pub(super) fn validate_record(
    layout: &Layout,
    group: u64,
    limits: &Limits,
    meta: &BatchMeta,
    body: &[u8],
) -> Result<()> {
    let command = decode(body)?;
    let (table, transaction, plan_index) = match command {
        Command::Pin(permit) => {
            permit.verify(layout, limits, group)?;
            let s = permit.selection();
            (s.table, s.transaction, s.plan_index)
        }
        Command::Close(permit) => {
            permit.verify(layout, limits, group)?;
            let c = permit.closure();
            (c.table, c.transaction, c.plan_index)
        }
    };
    ensure!(
        meta.kind == RecordKind::Source
            && meta.stream == table
            && meta.producer == transaction
            && meta.sequence == plan_index
            && meta.bucket == 0,
        "source pin envelope mismatch"
    );
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pin {
    pub index: u64,
    pub permit: Permit,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rejection {
    Pending,
    Prefix,
    Capacity,
    Closed,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reply {
    Pinned { index: u64 },
    Rejected(Rejection),
    Closed { index: u64 },
}

fn complete_prefix(state: &State, selection: &Selection, index: u64) -> bool {
    let range = &selection.range;
    let previous = state
        .closed
        .get(&selection.table)
        .map(|c| c.permit.closure());
    range.after == previous.map_or(0, |c| c.through)
        && previous.is_none_or(|c| {
            selection.plan_index > c.plan_index && selection.read_version >= c.version
        })
        && range.through < index
        && state
            .accepted
            .range((
                std::ops::Bound::Excluded(range.after),
                std::ops::Bound::Included(range.through),
            ))
            .filter(|(_, meta)| meta.stream == selection.table)
            .map(|(i, meta)| (*i, meta))
            .eq(range.records.iter().map(|(i, meta)| (*i, meta)))
}
pub(super) fn apply(
    state: &mut State,
    command: Command,
    index: u64,
    limits: &Limits,
) -> Result<Reply> {
    match command {
        Command::Pin(permit) => Ok(pin(state, permit, index, limits)),
        Command::Close(permit) => closure::apply(state, permit, index, limits),
    }
}
fn pin(state: &mut State, permit: Permit, index: u64, limits: &Limits) -> Reply {
    let selection = permit.selection();
    if state
        .closed
        .get(&selection.table)
        .is_some_and(|closed| selection.plan_index <= closed.permit.closure().plan_index)
    {
        return Reply::Rejected(Rejection::Closed);
    }
    if let Some(old) = state.pins.get(&selection.table) {
        return if old.permit.selection() == selection {
            Reply::Pinned { index: old.index }
        } else {
            Reply::Rejected(Rejection::Pending)
        };
    }
    if !complete_prefix(state, selection, index) {
        return Reply::Rejected(Rejection::Prefix);
    }
    if state.pins.len() as u64 >= limits.max_tables.min(limits.max_active_plans) {
        return Reply::Rejected(Rejection::Capacity);
    }
    state
        .pins
        .insert(selection.table, Arc::new(Pin { index, permit }));
    Reply::Pinned { index }
}
pub(super) fn validate_state(
    state: &State,
    layout: &Layout,
    group: u64,
    cut: u64,
    limits: &Limits,
) -> Result<()> {
    closure::validate_state(state, layout, group, cut, limits)?;
    ensure!(
        state.pins.len() as u64 <= limits.max_tables.min(limits.max_active_plans),
        "source pin capacity"
    );
    for (table, pin) in &state.pins {
        pin.permit.verify(layout, limits, group)?;
        ensure!(
            *table == pin.permit.selection.table
                && pin.index > 0
                && pin.index <= cut
                && complete_prefix(state, pin.permit.selection(), pin.index),
            "invalid checkpoint source pin or prefix"
        );
    }
    Ok(())
}
pub(super) fn encode_pin(pin: &Pin) -> Result<Vec<u8>> {
    Ok(codec(MAX_PIN_BYTES).serialize(pin)?)
}
pub(super) fn decode_pin(bytes: &[u8]) -> Result<Pin> {
    ensure!(bytes.len() <= MAX_PIN_BYTES, "oversized source pin frame");
    let pin = codec(MAX_PIN_BYTES).deserialize(bytes)?;
    ensure!(encode_pin(&pin)? == bytes, "noncanonical source pin frame");
    Ok(pin)
}
