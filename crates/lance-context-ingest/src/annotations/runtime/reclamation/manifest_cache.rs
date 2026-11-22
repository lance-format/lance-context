//! Reuse immutable protection metadata within the existing session cache budget.
use super::*;
use lance::cache::{CacheKey, Context, DeepSizeOf, LanceCache};
use lance_context_core::annotation_staging::BodySnapshot;
use lance_io::object_store::ObjectStore;
use lance_table::format::Manifest;
use std::borrow::Cow;

struct ManifestKey<'a>(&'a p::ManifestRecord);

pub(crate) struct BodyReference(pub BodySnapshot);

impl DeepSizeOf for BodyReference {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.table_instance.deep_size_of_children(context)
            + self.0.uri.deep_size_of_children(context)
            + self.0.transaction.deep_size_of_children(context)
    }
}

struct BodyReferenceKey<'a>(&'a p::ManifestRecord);

impl CacheKey for BodyReferenceKey<'_> {
    type ValueType = BodyReference;

    fn key(&self) -> Cow<'_, str> {
        ManifestKey(self.0).key().into_owned().into()
    }

    fn type_name() -> &'static str {
        "lance-context.retained-body-reference"
    }
}

impl CacheKey for ManifestKey<'_> {
    type ValueType = Manifest;

    fn key(&self) -> Cow<'_, str> {
        // The UUID is the catalog-selected immutable identity, not just a
        // version number. The validated relative path contains no colons.
        let record = self.0;
        format!(
            "{}:{}:{}:{}",
            record.identity, record.version, record.object.bytes, record.object.path
        )
        .into()
    }

    fn type_name() -> &'static str {
        "lance-context.retained-manifest"
    }
}

pub(super) async fn read(
    cache: &LanceCache,
    store: &ObjectStore,
    path: &Path,
    record: &p::ManifestRecord,
) -> Result<Arc<Manifest>> {
    // The collector's namespace shares the session's weighted limit. Disabled
    // caches and evictions simply read the selected immutable object again.
    Ok(cache
        .get_or_insert_with_key(ManifestKey(record), || async {
            let manifest =
                lance_table::io::manifest::read_manifest(store, path, Some(record.object.bytes))
                    .await?;
            if manifest.version != record.version {
                return Err(lance::Error::invalid_input(
                    "retained manifest version mismatch",
                ));
            }
            Ok(manifest)
        })
        .await?)
}

/// Cache only the immutable reference, never a Dataset, reader pin or liveness
/// decision. The caller still captures current roots and protects each page.
pub(crate) async fn read_body(
    cache: &LanceCache,
    ingress: &Ingress,
    storage: &Storage,
    record: &p::ManifestRecord,
) -> Result<Arc<BodyReference>> {
    record.object.validate(record.version)?;
    let bounded = cache.capacity_bytes().is_some_and(|bytes| bytes > 0);
    let key = BodyReferenceKey(record);
    if bounded {
        if let Some(reference) = cache.get_with_key(&key).await {
            return Ok(reference);
        }
    }
    // ManagedManifest resolves this exact retained version on a miss. Reusing
    // its decoded binding avoids repeating catalog reads for every GC page.
    let narrow = checkpoint::load(ingress, storage, record.version).await?;
    let name = record.object.path.strip_prefix("_versions/").unwrap();
    let path = storage.base.clone().join("_versions").join(name);
    ensure!(
        narrow.manifest_location().path == path
            && narrow.manifest_location().identity.as_deref()
                == Some(record.identity.to_string().as_str()),
        "retained narrow manifest identity changed"
    );
    let reference = Arc::new(BodyReference(
        BodySnapshot::read(&narrow)
            .await?
            .context("protected narrow snapshot has no body reference")?,
    ));
    if bounded {
        cache.insert_with_key(&key, reference.clone()).await;
    }
    Ok(reference)
}
