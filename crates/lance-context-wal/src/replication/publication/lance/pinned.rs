//! Read one already selected manifest without a catalog RPC for each open.
use super::*;

/// A read-only Lance handler for an exact, previously selected snapshot.
///
/// Obtain `record` from an authenticated, quorum-observed selection and bind it
/// to the authorized table/incarnation. Construction validates that binding and
/// path; it does not authenticate the record or prove current selection. The
/// caller must retain the selected files while this snapshot is in use.
///
/// Pass this handler to `DatasetBuilder::with_commit_handler`. Raw URI/version
/// opens cannot discover the catalog's unique staging path. This handler never
/// falls back to a raw path, lists other versions, or grants write authority.
#[derive(Clone, Debug)]
pub struct PinnedManifest {
    base: Path,
    location: ManifestLocation,
}

impl PinnedManifest {
    pub fn new(
        base: Path,
        cluster: Uuid,
        layout: [u8; 32],
        table: Uuid,
        incarnation: Uuid,
        record: &ManifestRecord,
    ) -> Result<Self> {
        record.object.validate(record.version)?;
        ensure!(
            !cluster.is_nil()
                && layout != [0; 32]
                && !table.is_nil()
                && !incarnation.is_nil()
                && record.index > 0
                && !record.transaction.is_nil()
                && record.identity
                    == manifest::identity_for_binding(
                        cluster,
                        layout,
                        table,
                        incarnation,
                        record.version,
                        record.index,
                    ),
            "foreign or invalid pinned manifest record"
        );
        let location = ManifestLocation {
            version: record.version,
            path: Path::parse(format!("{}/{}", base, record.object.path))?,
            size: Some(record.object.bytes),
            naming_scheme: if record.object.v2_names {
                ManifestNamingScheme::V2
            } else {
                ManifestNamingScheme::V1
            },
            e_tag: None,
            identity: Some(record.identity.to_string()),
        };
        Ok(Self { base, location })
    }

    fn check_base(&self, base: &Path) -> ::lance::Result<()> {
        if base != &self.base {
            return Err(error("foreign pinned dataset path"));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl CommitHandler for PinnedManifest {
    async fn resolve_latest_location(
        &self,
        base: &Path,
        _: &ObjectStore,
    ) -> ::lance::Result<ManifestLocation> {
        self.check_base(base)?;
        Ok(self.location.clone())
    }

    async fn resolve_version_location(
        &self,
        base: &Path,
        version: u64,
        _: &dyn object_store::ObjectStore,
    ) -> ::lance::Result<ManifestLocation> {
        self.check_base(base)?;
        if version != self.location.version {
            return Err(error("version differs from pinned manifest"));
        }
        Ok(self.location.clone())
    }

    async fn version_exists(
        &self,
        base: &Path,
        version: u64,
        _: &dyn object_store::ObjectStore,
        _: ManifestNamingScheme,
    ) -> ::lance::Result<bool> {
        self.check_base(base)?;
        if version != self.location.version {
            // Unknown is not proof of absence and must not authorize cleanup.
            return Err(error("version outside pinned manifest"));
        }
        Ok(true)
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
        _: &Path,
        _: &'a ObjectStore,
        _: u64,
    ) -> BoxStream<'a, ::lance::Result<ManifestLocation>> {
        no_listing()
    }

    async fn commit(
        &self,
        _: &mut Manifest,
        _: Option<Vec<IndexMetadata>>,
        _: &Path,
        _: &ObjectStore,
        _: ManifestWriter,
        _: ManifestNamingScheme,
        _: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        Err(::lance::Error::not_supported("read-only pinned manifest handler").into())
    }

    async fn forget_version(&self, _: &Path, _: u64, _: &str) -> ::lance::Result<()> {
        Err(error(
            "pinned snapshot grants no manifest retirement authority",
        ))
    }

    async fn delete(&self, _: &Path) -> ::lance::Result<()> {
        Err(error(
            "pinned snapshot grants no dataset deletion authority",
        ))
    }
}

#[cfg(test)]
mod tests;
