//! Follow only body references selected by protected narrow snapshots.
use super::*;
use lance_context_core::annotation_staging::BodySnapshot;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, StorageOptionsAccessor};
use lance_table::io::commit::{CommitHandler, ConditionalPutCommitHandler};
use object_store::ObjectStoreExt;

/// One fixed-size memo for the current closed candidate page and destination.
/// Adjacent narrow roots often select the same immutable body snapshot.
#[derive(Default)]
pub(super) struct Protection(Option<(u64, [u8; 36], bool)>);

#[cfg(test)]
mod tests;

pub(super) async fn store(storage: &Storage) -> Result<Arc<ObjectStore>> {
    let (store, base) = ObjectStore::from_uri_and_params(
        storage.session.store_registry(),
        &storage.uri,
        &ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                storage.options.clone(),
            ))),
            ..Default::default()
        },
    )
    .await?;
    ensure!(
        base == storage.base,
        "body collection storage binding changed"
    );
    Ok(store)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn protect_body(
    candidates: &mut BTreeMap<Path, u64>,
    narrow: &Dataset,
    storage: &Storage,
    store: &ObjectStore,
    incarnation: Uuid,
    budget: &Arc<MergeMemoryBudget>,
    already_reserved: usize,
    retain_data: bool,
    protected: &mut Protection,
) -> Result<()> {
    let reference = BodySnapshot::read(narrow)
        .await?
        .context("protected narrow snapshot has no body reference")?;
    ensure!(
        reference.uri == storage.uri && reference.table_instance == incarnation.to_string(),
        "protected body reference differs from authorized destination"
    );
    let transaction: [u8; 36] = reference.transaction.as_bytes().try_into()?;
    if protected.0.is_some_and(|(version, identity, data)| {
        version == reference.version && identity == transaction && (data || !retain_data)
    }) {
        // The candidate page cannot gain paths during protection. Reopening
        // the same body cannot remove any additional paths from this page.
        return Ok(());
    }
    // Bound a single body manifest before decoding it. Retained roots are opened
    // one at a time; they do not form another unbounded manifest/root cache.
    let location = ConditionalPutCommitHandler
        .resolve_version_location(&storage.base, reference.version, store.inner.as_ref())
        .await?;
    let bytes = match location.size {
        Some(size) => size,
        None => store.inner.head(&location.path).await?.size,
    };
    let charge = usize::try_from(bytes)?
        .checked_mul(4)
        .context("body manifest workspace overflow")?;
    ensure!(
        already_reserved
            .checked_add(charge)
            .is_some_and(|total| total <= budget.limit()),
        "body retained manifest exceeds collection workspace"
    );
    let _manifest = budget.reserve(charge).await;
    let body = mixed::load_body(storage, &reference, &incarnation.to_string()).await?;
    ensure!(
        body.manifest_location().path == location.path,
        "body manifest location changed"
    );
    protect(
        candidates,
        body.manifest(),
        &location.path,
        &storage.base,
        retain_data,
    )?;
    protected.0 = Some((reference.version, transaction, retain_data));
    Ok(())
}
