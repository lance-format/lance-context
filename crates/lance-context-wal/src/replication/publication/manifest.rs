//! A bounded, durable index of immutable manifests. Only a catalog commit may
//! select a staged path. No canonical-path repair, fallback import or GC occurs.
use super::*;

mod retention;
pub use retention::{HistoryBoundary, RetireManifestHistory};

#[cfg(test)]
pub(super) mod tests;

pub const MAX_MANIFEST_VERSIONS: usize = 128;
pub(super) const MAX_MANIFEST_BYTES: usize = 2048;
pub(super) const MAX_MANIFEST_COMMAND_BYTES: usize = 4096;
pub(super) const MAX_MANIFEST_INDEX_BYTES: usize = MAX_MANIFEST_VERSIONS * MAX_MANIFEST_BYTES + 256;

/// A path relative to the registered dataset, never a credential-bearing URI.
/// The materializer verifies the actual immutable object before registration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestObject {
    pub path: String,
    pub bytes: u64,
    pub v2_names: bool,
}
impl ManifestObject {
    pub fn validate(&self, version: u64) -> Result<()> {
        ensure!(
            version > 0 && version < (1u64 << 63),
            "invalid attached manifest version"
        );
        let name = self
            .path
            .strip_prefix("_versions/")
            .context("manifest path scope")?;
        ensure!(
            !name.is_empty()
                && self.path.len() <= 1024
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                && !name.contains("..")
                && self.bytes > 0
                && self.bytes <= 64 * 1024 * 1024,
            "invalid manifest path or size"
        );
        let number = if self.v2_names {
            format!("{:020}", u64::MAX - version)
        } else {
            version.to_string()
        };
        let canonical = format!("{number}.manifest");
        // Lance's conditioned handler appends a UUID to the canonical basename.
        ensure!(
            name == canonical
                || name
                    .strip_prefix(&(canonical + "-"))
                    .is_some_and(|id| { Uuid::parse_str(id).is_ok_and(|id| !id.is_nil()) }),
            "manifest path does not identify its fixed version"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestRecord {
    pub version: u64,
    /// Logical identity minted by the durable catalog, not an object-store ETag.
    pub identity: Uuid,
    pub object: ManifestObject,
    pub transaction: Uuid,
    pub index: u64,
    pub plan_index: u64,
    pub prepared_index: u64,
    pub descriptor_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestIndex {
    pub incarnation: Uuid,
    pub bootstrap_request: Uuid,
    pub initial_version: u64,
    pub head: u64,
    /// Captured views and updates share the immutable records.
    pub versions: BTreeMap<u64, Arc<ManifestRecord>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapManifest {
    pub authority: Authority,
    pub incarnation: Uuid,
    pub request: Uuid,
    pub version: u64,
    pub object: ManifestObject,
    pub transaction: Uuid,
}
impl BootstrapManifest {
    pub(in crate::replication) fn batch(self, layout: &Layout) -> Result<Batch> {
        self.validate(layout)?;
        Command::BootstrapManifest(Box::new(self)).batch(layout)
    }
    pub(super) fn validate(&self, layout: &Layout) -> Result<()> {
        self.authority.validate(layout)?;
        ensure!(
            !self.incarnation.is_nil() && !self.request.is_nil() && !self.transaction.is_nil(),
            "invalid manifest bootstrap identity"
        );
        self.object.validate(self.version)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReserveManifest {
    pub authority: Authority,
    pub incarnation: Uuid,
    pub transaction: Uuid,
    pub plan_index: u64,
    pub prepared_index: u64,
    pub descriptor_digest: [u8; 32],
    pub predecessor: Uuid,
    pub target: u64,
    pub object: ManifestObject,
}
impl ReserveManifest {
    pub(in crate::replication) fn batch(self, layout: &Layout) -> Result<Batch> {
        self.validate(layout)?;
        Command::ReserveManifest(Box::new(self)).batch(layout)
    }
    pub(super) fn validate(&self, layout: &Layout) -> Result<()> {
        self.authority.validate(layout)?;
        ensure!(
            !self.incarnation.is_nil()
                && !self.transaction.is_nil()
                && !self.predecessor.is_nil()
                && self.plan_index > 0
                && self.prepared_index > self.plan_index
                && self.prepared_index < u64::MAX
                && self.descriptor_digest != [0; 32],
            "invalid manifest reservation identity"
        );
        self.object.validate(self.target)?;
        ensure!(
            self.object.path.contains(".manifest-"),
            "reservation needs a unique staging path"
        );
        Ok(())
    }
}

pub(super) fn identity(authority: &Authority, incarnation: Uuid, version: u64, index: u64) -> Uuid {
    identity_for_binding(
        authority.cluster,
        authority.layout,
        authority.table,
        incarnation,
        version,
        index,
    )
}

pub(super) fn identity_for_binding(
    cluster: Uuid,
    layout: [u8; 32],
    table: Uuid,
    incarnation: Uuid,
    version: u64,
    index: u64,
) -> Uuid {
    let mut hash = Sha256::new();
    hash.update(b"lance-context/manifest-identity/v1");
    hash.update(cluster.as_bytes());
    hash.update(layout);
    hash.update(table.as_bytes());
    hash.update(incarnation.as_bytes());
    hash.update(version.to_le_bytes());
    hash.update(index.to_le_bytes());
    Uuid::from_bytes(hash.finalize()[..16].try_into().expect("digest size"))
}

impl ManifestIndex {
    pub(super) fn validate(&self, table: &Table, cut: u64) -> Result<()> {
        ensure!(
            !self.incarnation.is_nil()
                && !self.bootstrap_request.is_nil()
                && self.initial_version > 0
                && self.head < u64::MAX
                && self.head >= self.initial_version
                && self.versions.len() <= MAX_MANIFEST_VERSIONS
                && self.versions.first_key_value().map(|(v, _)| *v) == Some(self.initial_version)
                && self.versions.last_key_value().map(|(v, _)| *v) == Some(self.head),
            "invalid manifest index bounds"
        );
        let mut previous_index = 0;
        let mut previous_version = self.initial_version;
        for (offset, (version, record)) in self.versions.iter().enumerate() {
            record.object.validate(*version)?;
            ensure!(
                (offset == 0 || *version > previous_version)
                    && (offset <= 1 || *version == previous_version + 1)
                    && record.version == *version
                    && !record.transaction.is_nil()
                    && record.index > previous_index
                    && record.index <= cut
                    && record.index != table.authority.grant_index
                    && record.identity
                        == identity(&table.authority, self.incarnation, *version, record.index),
                "invalid manifest record identity or order"
            );
            if offset == 0 {
                ensure!(
                    record.plan_index == 0
                        && record.prepared_index == 0
                        && record.descriptor_digest == [0; 32],
                    "bootstrap carries publication metadata"
                );
            } else {
                ensure!(
                    record.plan_index > 0
                        && record.prepared_index > record.plan_index
                        && record.index > record.prepared_index
                        && record.descriptor_digest != [0; 32]
                        && record.object.path.contains(".manifest-"),
                    "invalid prepared manifest record"
                );
            }
            previous_index = record.index;
            previous_version = *version;
        }
        if let Some(pending) = &table.progress.pending {
            ensure!(
                pending.plan.read_version >= self.initial_version
                    && self.versions.contains_key(&pending.plan.read_version)
                    && self.head >= pending.plan.read_version
                    && self.head <= pending.plan.target_version()?,
                "managed plan does not name the current predecessor or target"
            );
            if let Some(record) = self.versions.get(&pending.plan.target_version()?) {
                let prepared = table
                    .progress
                    .prepared
                    .as_ref()
                    .context("manifest without descriptors")?;
                ensure!(
                    record.transaction == pending.plan.transaction
                        && record.plan_index == pending.index
                        && record.prepared_index == prepared.index
                        && record.descriptor_digest == prepared.descriptors.digest,
                    "manifest disagrees with immutable prepared plan"
                );
            }
        }
        Ok(())
    }
}

impl Catalog {
    pub(super) fn manifest_records(&self) -> usize {
        self.tables
            .values()
            .filter_map(|t| t.manifests.as_ref())
            .map(|m| m.versions.len())
            .sum()
    }

    pub(super) fn bootstrap_manifest(
        &mut self,
        request: BootstrapManifest,
        index: u64,
        limits: &Limits,
    ) -> Reply {
        let Some(table) = self.tables.get(&request.authority.table) else {
            return Reply::Rejected(Rejection::Authority);
        };
        if table.authority != request.authority {
            return Reply::Rejected(Rejection::Authority);
        }
        if let Some(manifests) = &table.manifests {
            let first = &manifests.versions[&manifests.initial_version];
            return if manifests.incarnation == request.incarnation
                && manifests.bootstrap_request == request.request
                && first.version == request.version
                && first.object == request.object
                && first.transaction == request.transaction
            {
                Reply::ManifestReserved(first.clone())
            } else {
                Reply::Rejected(Rejection::ManifestIdentity)
            };
        }
        if table.progress.pending.is_some() {
            return Reply::Rejected(Rejection::PendingPlan);
        }
        if self.manifest_records() as u64 >= limits.max_manifest_records {
            return Reply::Rejected(Rejection::Capacity);
        }
        let record = Arc::new(ManifestRecord {
            version: request.version,
            identity: identity(
                &request.authority,
                request.incarnation,
                request.version,
                index,
            ),
            object: request.object,
            transaction: request.transaction,
            index,
            plan_index: 0,
            prepared_index: 0,
            descriptor_digest: [0; 32],
        });
        self.tables
            .get_mut(&request.authority.table)
            .expect("known table")
            .manifests = Some(Arc::new(ManifestIndex {
            incarnation: request.incarnation,
            bootstrap_request: request.request,
            initial_version: request.version,
            head: request.version,
            versions: BTreeMap::from([(request.version, record.clone())]),
        }));
        Reply::ManifestReserved(record)
    }

    pub(super) fn reserve_manifest(
        &mut self,
        request: ReserveManifest,
        index: u64,
        limits: &Limits,
    ) -> Reply {
        let Some(table) = self.tables.get(&request.authority.table) else {
            return Reply::Rejected(Rejection::Authority);
        };
        let Some(manifests) = &table.manifests else {
            return Reply::Rejected(Rejection::ManifestIdentity);
        };
        if manifests.incarnation != request.incarnation {
            return Reply::Rejected(Rejection::ManifestIdentity);
        }
        let boundary = manifests.history_boundary();
        if boundary.is_retired(request.target) {
            // Absence below the retained suffix is not evidence that this
            // caller's immutable object was never selected. It cannot be deleted
            // on the strength of this reply, including after publisher takeover.
            return Reply::ManifestRetired(boundary);
        }
        // Resolve an already chosen path before a new authority/predecessor
        // check. A lost successful reply must never let the caller delete the
        // selected object as a newly refused staging upload after takeover.
        if let Some(record) = manifests.versions.get(&request.target) {
            return if record.transaction == request.transaction
                && record.plan_index == request.plan_index
                && record.prepared_index == request.prepared_index
                && record.descriptor_digest == request.descriptor_digest
                && record.object == request.object
            {
                Reply::ManifestReserved(record.clone())
            } else {
                Reply::ManifestTaken(record.clone())
            };
        }
        if table.authority != request.authority {
            return Reply::Rejected(Rejection::Authority);
        }
        let Some(pending) = &table.progress.pending else {
            return Reply::Rejected(Rejection::PlanIdentity);
        };
        let Some(prepared) = &table.progress.prepared else {
            return Reply::Rejected(Rejection::PlanIdentity);
        };
        if pending.index != request.plan_index
            || pending.plan.transaction != request.transaction
            || prepared.index != request.prepared_index
            || prepared.descriptors.digest != request.descriptor_digest
            || pending.plan.target_version().ok() != Some(request.target)
        {
            return Reply::Rejected(Rejection::PlanIdentity);
        }
        if manifests.head != pending.plan.read_version
            || manifests.versions[&manifests.head].identity != request.predecessor
        {
            return Reply::Rejected(Rejection::ManifestPredecessor);
        }
        if manifests.versions.len() >= MAX_MANIFEST_VERSIONS
            || self.manifest_records() as u64 >= limits.max_manifest_records
        {
            return Reply::Rejected(Rejection::Capacity);
        }
        let record = Arc::new(ManifestRecord {
            version: request.target,
            identity: identity(
                &request.authority,
                request.incarnation,
                request.target,
                index,
            ),
            object: request.object,
            transaction: request.transaction,
            index,
            plan_index: request.plan_index,
            prepared_index: request.prepared_index,
            descriptor_digest: request.descriptor_digest,
        });
        let table = self
            .tables
            .get_mut(&request.authority.table)
            .expect("known table");
        let manifests = Arc::make_mut(table.manifests.as_mut().expect("managed table"));
        manifests.head = request.target;
        manifests.versions.insert(request.target, record.clone());
        Reply::ManifestReserved(record)
    }
}
