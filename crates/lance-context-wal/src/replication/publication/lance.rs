//! Lance manifest reads and fixed-target publication through the durable catalog.
//!
//! The caller binds this handler to the authorized storage namespace and dataset
//! path. The catalog implementation must authenticate/correlate RPCs, bound its
//! queues and deadlines, and return committed outcomes and quorum read results.
//! This adapter provides neither that network runtime nor table authorization.
//! Use it for every reader/writer of a managed table; raw URI readers cannot
//! discover catalog-selected staging paths. Bootstrap/migration is separate.
use super::{manifest, Authority, ManifestObject, ManifestRecord, ReserveManifest};
use crate::{
    placement::Layout,
    replication::{query, Applied, Outcome},
};
use anyhow::{ensure, Result};
use futures::{stream, stream::BoxStream, StreamExt};
use lance_io::object_store::ObjectStore;
use lance_table::{
    format::{IndexMetadata, Manifest, Transaction},
    io::commit::{
        CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
    },
};
use object_store::path::Path;
use std::{fmt::Debug, sync::Arc};
use uuid::Uuid;

/// A trusted, bounded transport adapter. A local replica read is not sufficient.
#[async_trait::async_trait]
pub trait ManifestCatalog: Debug + Send + Sync {
    async fn read_manifest(&self, table: Uuid, version: Option<u64>) -> Result<query::Outcome>;
    async fn reserve_manifest(&self, request: ReserveManifest) -> Result<Outcome>;
}

/// Immutable publication identity; retries cannot move to a later version.
#[derive(Clone, Debug)]
pub struct ManifestCommit {
    pub authority: Authority,
    pub incarnation: Uuid,
    pub transaction: Uuid,
    pub plan_index: u64,
    pub prepared_index: u64,
    pub descriptor_digest: [u8; 32],
    pub predecessor: Uuid,
    pub target: u64,
}
impl From<&ReserveManifest> for ManifestCommit {
    fn from(request: &ReserveManifest) -> Self {
        Self {
            authority: request.authority.clone(),
            incarnation: request.incarnation,
            transaction: request.transaction,
            plan_index: request.plan_index,
            prepared_index: request.prepared_index,
            descriptor_digest: request.descriptor_digest,
            predecessor: request.predecessor,
            target: request.target,
        }
    }
}
impl ManifestCommit {
    fn request(&self, object: ManifestObject) -> ReserveManifest {
        ReserveManifest {
            authority: self.authority.clone(),
            incarnation: self.incarnation,
            transaction: self.transaction,
            plan_index: self.plan_index,
            prepared_index: self.prepared_index,
            descriptor_digest: self.descriptor_digest,
            predecessor: self.predecessor,
            target: self.target,
            object,
        }
    }
}

/// Reusable Lance adapter. Construction does not establish current ownership;
/// the catalog rechecks the authority and exact predecessor when reserving.
/// Staged objects survive every error/unknown outcome. Collection and deletion
/// require a separate coordinated protocol and are explicitly unsupported here.
#[derive(Clone, Debug)]
pub struct ManagedManifest {
    catalog: Arc<dyn ManifestCatalog>,
    base: Path,
    binding: ManifestBinding,
    scope: Option<ManifestCommit>,
}

#[derive(Clone, Debug)]
struct ManifestBinding {
    cluster: Uuid,
    layout: [u8; 32],
    group: u64,
    table: Uuid,
    incarnation: Uuid,
}
impl ManagedManifest {
    /// Open catalog-selected manifests without acquiring publisher ownership or
    /// fabricating a pending plan. Resolve table/incarnation and the backing
    /// object store from the authorized catalog before constructing the reader.
    pub fn reader(
        catalog: Arc<dyn ManifestCatalog>,
        base: Path,
        layout: &Layout,
        table: Uuid,
        incarnation: Uuid,
    ) -> Result<Self> {
        ensure!(
            !table.is_nil() && !incarnation.is_nil(),
            "invalid managed reader identity"
        );
        Ok(Self {
            catalog,
            base,
            binding: ManifestBinding {
                cluster: layout.cluster,
                layout: layout.fingerprint()?,
                group: super::control_group(layout),
                table,
                incarnation,
            },
            scope: None,
        })
    }
    pub fn new(
        catalog: Arc<dyn ManifestCatalog>,
        base: Path,
        layout: &Layout,
        scope: ManifestCommit,
    ) -> Result<Self> {
        scope.authority.validate(layout)?;
        ensure!(
            !scope.incarnation.is_nil()
                && !scope.transaction.is_nil()
                && !scope.predecessor.is_nil()
                && scope.plan_index > 0
                && scope.prepared_index > scope.plan_index
                && scope.prepared_index < u64::MAX
                && scope.descriptor_digest != [0; 32]
                && (2..(1u64 << 63)).contains(&scope.target),
            "invalid managed manifest commit identity"
        );
        let mut handler = Self::reader(
            catalog,
            base,
            layout,
            scope.authority.table,
            scope.incarnation,
        )?;
        handler.scope = Some(scope);
        Ok(handler)
    }
    fn check_base(&self, base: &Path) -> ::lance::Result<()> {
        if base != &self.base {
            return Err(::lance::Error::invalid_input(
                "foreign managed dataset path",
            ));
        }
        Ok(())
    }
    fn validate_record(&self, record: &ManifestRecord) -> Result<()> {
        record.object.validate(record.version)?;
        ensure!(
            record.index > 0
                && !record.transaction.is_nil()
                && record.identity
                    == manifest::identity_for_binding(
                        self.binding.cluster,
                        self.binding.layout,
                        self.binding.table,
                        self.binding.incarnation,
                        record.version,
                        record.index
                    ),
            "foreign or invalid managed manifest record"
        );
        Ok(())
    }
    fn location(&self, record: &ManifestRecord) -> ::lance::Result<ManifestLocation> {
        self.validate_record(record).map_err(error)?;
        Ok(ManifestLocation {
            version: record.version,
            path: Path::parse(format!("{}/{}", self.base, record.object.path)).map_err(error)?,
            size: Some(record.object.bytes),
            naming_scheme: if record.object.v2_names {
                ManifestNamingScheme::V2
            } else {
                ManifestNamingScheme::V1
            },
            e_tag: None,
            identity: Some(record.identity.to_string()),
        })
    }
    async fn read(&self, version: Option<u64>) -> ::lance::Result<Option<Arc<ManifestRecord>>> {
        if version.is_some_and(|v| v == 0 || v >= 1u64 << 63) {
            return Err(::lance::Error::invalid_input(
                "invalid attached manifest version",
            ));
        }
        let query::Outcome::Observed { cut, value } = self
            .catalog
            .read_manifest(self.binding.table, version)
            .await
            .map_err(error)?
        else {
            return Err(error("manifest quorum read outcome unknown"));
        };
        let a = &self.binding;
        if cut.cluster != a.cluster
            || cut.layout != a.layout
            || cut.group != a.group
            || cut.term == 0
            || cut.quorum_index == 0
            || cut.applied < cut.quorum_index
        {
            return Err(error("invalid manifest quorum read scope"));
        }
        match value {
            query::Value::Manifest(record) => {
                if let Some(record) = &record {
                    self.validate_record(record).map_err(error)?;
                    if record.index > cut.applied || version.is_some_and(|v| v != record.version) {
                        return Err(error("manifest record differs from requested cut/version"));
                    }
                }
                Ok(record)
            }
            query::Value::ManifestRetired(_) => Err(error("managed manifest version was retired")),
            _ => Err(error("unexpected manifest query response")),
        }
    }
}
fn error(value: impl std::fmt::Display) -> ::lance::Error {
    ::lance::Error::io(value.to_string())
}
fn no_listing<'a>() -> BoxStream<'a, ::lance::Result<ManifestLocation>> {
    stream::once(async {
        Err(::lance::Error::not_supported(
            "managed manifest listing requires a catalog enumeration protocol",
        ))
    })
    .boxed()
}

#[async_trait::async_trait]
#[allow(clippy::too_many_arguments)]
impl CommitHandler for ManagedManifest {
    async fn resolve_latest_location(
        &self,
        base: &Path,
        _: &ObjectStore,
    ) -> ::lance::Result<ManifestLocation> {
        self.check_base(base)?;
        let record = self
            .read(None)
            .await?
            .ok_or_else(|| ::lance::Error::not_found("managed head"))?;
        self.location(&record)
    }
    async fn resolve_version_location(
        &self,
        base: &Path,
        version: u64,
        _: &dyn object_store::ObjectStore,
    ) -> ::lance::Result<ManifestLocation> {
        self.check_base(base)?;
        let record = self
            .read(Some(version))
            .await?
            .ok_or_else(|| ::lance::Error::not_found(format!("managed version {version}")))?;
        self.location(&record)
    }
    async fn version_exists(
        &self,
        base: &Path,
        version: u64,
        _: &dyn object_store::ObjectStore,
        _: ManifestNamingScheme,
    ) -> ::lance::Result<bool> {
        self.check_base(base)?;
        Ok(self.read(Some(version)).await?.is_some())
    }
    fn list_detached_manifest_locations<'a>(
        &self,
        _: &Path,
        _: &'a ObjectStore,
    ) -> BoxStream<'a, ::lance::Result<ManifestLocation>> {
        no_listing()
    }
    fn list_manifest_locations<'a>(
        &self,
        _: &Path,
        _: &'a ObjectStore,
        _: bool,
    ) -> BoxStream<'a, ::lance::Result<ManifestLocation>> {
        no_listing()
    }
    fn list_manifest_locations_since<'a>(
        &self,
        base: &Path,
        _: &'a ObjectStore,
        since: u64,
    ) -> BoxStream<'a, ::lance::Result<ManifestLocation>> {
        // Lance uses this during conflict resolution. Bound the history walk,
        // and resolve every position from the catalog; a missing/retired member
        // is an error, never permission to skip an intervening transaction.
        let handler = self.clone();
        let base = base.clone();
        stream::try_unfold(
            (handler, base, None, false),
            move |(h, base, next, started)| async move {
                h.check_base(&base)?;
                let version = if started {
                    next
                } else {
                    let head = h
                        .read(None)
                        .await?
                        .ok_or_else(|| error("managed head missing"))?;
                    if head.version.saturating_sub(since) > super::MAX_MANIFEST_VERSIONS as u64 {
                        return Err(error(
                            "manifest history request exceeds retained history bound",
                        ));
                    }
                    (head.version > since).then_some(head.version)
                };
                let Some(version) = version else {
                    return Ok(None);
                };
                let record = h
                    .read(Some(version))
                    .await?
                    .ok_or_else(|| error("gap in managed manifest history"))?;
                let location = h.location(&record)?;
                let next = (version - 1 > since).then_some(version - 1);
                Ok(Some((location, (h, base, next, true))))
            },
        )
        .boxed()
    }
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base: &Path,
        store: &ObjectStore,
        writer: ManifestWriter,
        naming: ManifestNamingScheme,
        transaction: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        self.check_base(base)?;
        let scope = self
            .scope
            .as_ref()
            .ok_or_else(|| ::lance::Error::not_supported("read-only managed manifest handler"))?;
        if manifest.version != scope.target
            || transaction
                .as_ref()
                .is_none_or(|tx| tx.as_pb().uuid != scope.transaction.to_string())
        {
            return Err(CommitError::CommitConflict);
        }
        let path = Path::parse(format!(
            "{}-{}",
            naming.manifest_path(base, manifest.version),
            Uuid::new_v4()
        ))
        .map_err(error)?;
        let written = writer(store, manifest, indices, &path, transaction).await?;
        let object = ManifestObject {
            path: format!("_versions/{}", path.filename().expect("staging filename")),
            bytes: written.size as u64,
            v2_names: naming == ManifestNamingScheme::V2,
        };
        object.validate(scope.target).map_err(error)?;
        let outcome = self
            .catalog
            .reserve_manifest(scope.request(object))
            .await
            .map_err(error)?;
        let Outcome::Committed(Applied::Publisher(super::Reply::ManifestReserved(record))) =
            outcome
        else {
            return Err(
                error("manifest reservation unavailable; reconcile original publication").into(),
            );
        };
        self.validate_record(&record).map_err(error)?;
        if record.version != scope.target
            || record.transaction != scope.transaction
            || record.plan_index != scope.plan_index
            || record.prepared_index != scope.prepared_index
            || record.descriptor_digest != scope.descriptor_digest
            || record.index <= record.prepared_index
        {
            return Err(error("manifest reservation returned a different publication").into());
        }
        // An exact retry may select an earlier attempt's immutable object.
        Ok(self.location(&record)?)
    }
    async fn forget_version(&self, _: &Path, _: u64, _: &str) -> ::lance::Result<()> {
        Err(::lance::Error::not_supported(
            "managed manifest collection requires coordination",
        ))
    }
    async fn delete(&self, _: &Path) -> ::lance::Result<()> {
        Err(::lance::Error::not_supported(
            "managed dataset deletion requires coordination",
        ))
    }
}

#[cfg(test)]
mod tests;
