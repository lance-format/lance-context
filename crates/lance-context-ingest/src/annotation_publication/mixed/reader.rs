//! Reuse Lance's bounded manifest cache when validating detached body snapshots.
use crate::materializer::Storage;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, StorageOptionsAccessor};
use lance_table::{
    format::{is_detached_version, IndexMetadata, Manifest, Transaction},
    io::commit::{
        commit_handler_from_url, CommitError, CommitHandler, ManifestLocation,
        ManifestNamingScheme, ManifestWriter,
    },
};
use object_store::{path::Path, ObjectStore as RawStore, ObjectStoreExt};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct BodyManifestReader(Arc<dyn CommitHandler>);

impl BodyManifestReader {
    pub(super) async fn new(storage: &Storage) -> lance::Result<Self> {
        // Match DatasetBuilder::with_storage_options: retain the configured
        // backend's resolver, including any external manifest authority.
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                storage.options.clone(),
            ))),
            ..Default::default()
        };
        Ok(Self(
            commit_handler_from_url(&storage.uri, &Some(params)).await?,
        ))
    }
}

#[async_trait::async_trait]
impl CommitHandler for BodyManifestReader {
    async fn resolve_version_location(
        &self,
        base: &Path,
        version: u64,
        store: &dyn RawStore,
    ) -> lance::Result<ManifestLocation> {
        let mut location = self
            .0
            .resolve_version_location(base, version, store)
            .await?;
        if is_detached_version(version) && location.size.is_none() {
            // The default detached resolver returns only a deterministic path.
            // Without a size, Lance bypasses its manifest cache and the reader
            // discovers the size again. Resolve metadata once instead, retaining
            // the existing session cache budget and its version/ETag key.
            // For this unresolved path, HEAD even on a cache hit: missing or
            // unavailable storage must fail. Preserve locations already resolved
            // with metadata by the configured backend without another HEAD.
            let metadata = store.head(&location.path).await?;
            location.size = Some(metadata.size);
            location.e_tag = metadata.e_tag;
        }
        Ok(location)
    }

    async fn commit(
        &self,
        _manifest: &mut Manifest,
        _indices: Option<Vec<IndexMetadata>>,
        _base: &Path,
        _store: &ObjectStore,
        _writer: ManifestWriter,
        _scheme: ManifestNamingScheme,
        _transaction: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        Err(CommitError::OtherError(lance::Error::not_supported(
            "body snapshot validation cannot commit",
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Resolved(ManifestLocation);

    #[async_trait::async_trait]
    impl CommitHandler for Resolved {
        async fn resolve_version_location(
            &self,
            _base: &Path,
            version: u64,
            _store: &dyn RawStore,
        ) -> lance::Result<ManifestLocation> {
            assert_eq!(version, self.0.version);
            Ok(self.0.clone())
        }

        async fn commit(
            &self,
            _manifest: &mut Manifest,
            _indices: Option<Vec<IndexMetadata>>,
            _base: &Path,
            _store: &ObjectStore,
            _writer: ManifestWriter,
            _scheme: ManifestNamingScheme,
            _transaction: Option<Transaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            unreachable!("read-only resolver fixture")
        }
    }

    #[tokio::test]
    async fn body_reader_preserves_backend_resolved_location_without_an_extra_head(
    ) -> lance::Result<()> {
        let location = ManifestLocation {
            version: (1 << 63) + 1,
            path: Path::from("catalog-selected/manifest"),
            size: Some(4096),
            e_tag: Some("selected-etag".into()),
            identity: Some("selected-identity".into()),
            naming_scheme: ManifestNamingScheme::V2,
        };
        // This store is empty: replacing the backend's location or issuing a
        // redundant HEAD would fail instead of preserving its resolved receipt.
        let store = object_store::memory::InMemory::new();
        let resolved = BodyManifestReader(Arc::new(Resolved(location.clone())))
            .resolve_version_location(&Path::from("body.lance"), location.version, &store)
            .await?;
        assert_eq!(resolved.path, location.path);
        assert_eq!(resolved.size, location.size);
        assert_eq!(resolved.e_tag, location.e_tag);
        assert_eq!(resolved.identity, location.identity);
        assert_eq!(resolved.naming_scheme, location.naming_scheme);
        Ok(())
    }
}
