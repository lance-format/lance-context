//! Reuse Lance's bounded manifest cache when validating detached body snapshots.
use lance_io::object_store::ObjectStore;
use lance_table::{
    format::{is_detached_version, IndexMetadata, Manifest, Transaction},
    io::commit::{
        CommitError, CommitHandler, ConditionalPutCommitHandler, ManifestLocation,
        ManifestNamingScheme, ManifestWriter,
    },
};
use object_store::{path::Path, ObjectStore as RawStore, ObjectStoreExt};

#[derive(Debug)]
pub(super) struct BodyManifestReader;

#[async_trait::async_trait]
impl CommitHandler for BodyManifestReader {
    async fn resolve_version_location(
        &self,
        base: &Path,
        version: u64,
        store: &dyn RawStore,
    ) -> lance::Result<ManifestLocation> {
        let mut location = ConditionalPutCommitHandler
            .resolve_version_location(base, version, store)
            .await?;
        if is_detached_version(version) {
            // The default detached resolver returns only a deterministic path.
            // Without a size, Lance bypasses its manifest cache and the reader
            // discovers the size again. Resolve metadata once instead, retaining
            // the existing session cache budget and its version/ETag key.
            // Always HEAD, even on a cache hit: missing/unavailable metadata
            // must still fail validation rather than returning a stale view.
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
