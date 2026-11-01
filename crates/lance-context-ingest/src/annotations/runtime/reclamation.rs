//! Bounded file collection for an exclusively owned annotation table.
//! Called only between publications in the same serial publisher loop. A pending
//! listing response can span publications; candidate protection/sealing cannot
//! overlap native preparation. A successor fences the final catalog operation
//! and opens its recovery base after acquisition.
use super::*;
use futures::{stream, stream::BoxStream, StreamExt, TryStreamExt};
use object_store::{path::Path, ObjectMeta};
use std::{future::Future, pin::Pin, sync::Mutex, task::Poll};

mod deletion;
mod indices;
#[cfg(test)]
mod listing_tests;
mod manifest_cache;
mod paired;

#[derive(Debug)]
struct ListingFailure;

impl std::fmt::Display for ListingFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("annotation collection LIST failed")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub max_objects: usize,
    pub max_bytes: usize,
    /// Concurrent SDK delete calls. S3/Azure calls contain at most 256 paths;
    /// other backends retain one path per call.
    pub delete_parallelism: usize,
    /// Minimum delay after a completed collection pass. Slow object storage I/O
    /// does not consume this cooldown. Data listing yields after at most this
    /// interval (capped at one second), or when pending under write backpressure,
    /// retaining the same listing stream.
    pub interval: Duration,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_objects: 8192,
            max_bytes: 2 << 20,
            delete_parallelism: 16,
            interval: Duration::from_secs(1),
        }
    }
}

impl Options {
    /// Check deployment limits before acquiring ownership or opening storage.
    pub fn validate(&self, publication_workspace: usize) -> Result<()> {
        ensure!(
            (1..=65_536).contains(&self.max_objects)
                && (16 << 10..=16 << 20).contains(&self.max_bytes)
                && (1..=64).contains(&self.delete_parallelism)
                && (Duration::from_millis(1)..=Duration::from_secs(3600)).contains(&self.interval)
                && self.max_bytes * 3 + (64 << 10) <= publication_workspace,
            "invalid annotation collection limits or insufficient publication workspace"
        );
        Ok(())
    }
}

/// Process-local work counters. Delete counts include successful removal of an
/// already absent immutable path; they are not an object-store inventory.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Observation {
    pub passes: u64,
    pub examined: u64,
    pub deleted: u64,
    pub deleted_bytes: u64,
    /// Failed DELETE attempts, including unknown replies. Not failed mutations.
    #[serde(default)]
    pub delete_failures: u64,
    /// Failed read-only LIST attempts; no candidates from these attempts are deleted.
    #[serde(default)]
    pub list_failures: u64,
    /// Collection is waiting for a fresh LIST after a failure. Publication can
    /// continue: the failed attempt did not seal a page or issue any DELETE.
    #[serde(default)]
    pub listing_blocked: bool,
    /// Unresolved immutable paths in the one sealed page blocking publication.
    #[serde(default)]
    pub pending_objects: usize,
    #[serde(default)]
    pub pending_bytes: u64,
}

struct SealedPage {
    store: Arc<dyn object_store::ObjectStore>,
    paths: Vec<(Path, u64)>,
    batch_size: usize,
    failed: bool,
    _workspace: MergeReservation,
}

type DeleteOutcomes = Vec<(Vec<(Path, u64)>, Vec<object_store::Result<u64>>)>;
type DeleteJob = Pin<Box<dyn Future<Output = Result<DeleteOutcomes>> + Send>>;

pub(super) struct Collector {
    options: Options,
    metadata: lance::cache::LanceCache,
    next_run: tokio::time::Instant,
    listing: Option<BoxStream<'static, object_store::Result<ObjectMeta>>>,
    deferred: Option<(Path, u64)>,
    sealed: Option<SealedPage>,
    deleting: Mutex<Option<DeleteJob>>,
    listing_blocked: bool,
    // At most one retained SDK stream/deferred object per destination, but only
    // one candidate/sealed page in total. Alternate resolved pages so a large
    // table listing cannot starve collection of the other destination.
    other_listing: Option<BoxStream<'static, object_store::Result<ObjectMeta>>>,
    other_deferred: Option<(Path, u64)>,
    body: bool,
}
impl Collector {
    pub(super) fn new(
        options: Options,
        budget: &Arc<MergeMemoryBudget>,
        storage: &Storage,
    ) -> Result<Self> {
        options.validate(budget.limit())?;
        Ok(Self {
            options,
            // This collector has one fixed table/storage binding. A private
            // namespace prevents cross-client reuse even for equal URI strings
            // resolved through different endpoints in a shared Session.
            metadata: storage
                .session
                .file_metadata_cache()
                .with_key_prefix("lance-context.annotation-collection")
                .with_key_prefix(&Uuid::new_v4().to_string()),
            next_run: tokio::time::Instant::now(),
            listing: None,
            deferred: None,
            sealed: None,
            deleting: Mutex::new(None),
            listing_blocked: false,
            other_listing: None,
            other_deferred: None,
            body: false,
        })
    }

    pub(super) async fn step(
        &mut self,
        service: &Service,
        catalog: &materializer::Publisher,
        storage: &Storage,
        budget: &Arc<MergeMemoryBudget>,
    ) -> Result<bool> {
        // A sealed page cannot gain paths. Its protection/fence completed before
        // any native preparation resumes; future publications use protected live
        // inputs and fresh immutable outputs. Poll the same DELETE future while
        // the serial publisher drains its bounded source window. No new listing
        // or protection pass starts while this one charged page is outstanding.
        if self.deleting.get_mut().unwrap().is_some() {
            self.poll_pending_delete(service).await?;
            return Ok(self.may_publish());
        }
        let result = if tokio::time::Instant::now() < self.next_run {
            return Ok(self.sealed.is_none());
        } else if self.sealed.is_some() {
            // Reissue only unresolved immutable DELETE requests from the original,
            // successfully fenced page. Never relist or add paths here.
            service.check_owner(catalog.authority()).await?;
            self.delete_page(service).await?;
            Ok(())
        } else {
            self.collect_page(service, catalog, storage, budget).await
        };
        if self.deleting.get_mut().unwrap().is_some() {
            return Ok(self.may_publish());
        }
        // LIST/protection/DELETE can take longer than the configured interval.
        // Starting the clock before that I/O makes every following publisher
        // iteration eligible for another pass, starving materialization.
        self.next_run = tokio::time::Instant::now() + self.options.interval;
        if let Err(error) = &result {
            if error.is::<ListingFailure>() {
                // The failed stream may have advanced its SDK continuation or
                // yielded a partial candidate page. Discard both and enumerate
                // this destination anew. No protection fence or DELETE has run.
                self.listing = None;
                self.deferred = None;
                self.listing_blocked = true;
                let mut observed = service.shared.reclamation.lock().unwrap();
                observed.list_failures += 1;
                observed.listing_blocked = true;
                eprintln!("{error:#}");
                // There is no unresolved deletion page. Let the bounded source
                // window drain while collection waits for its next fresh LIST.
                // That attempt captures current protection roots before sealing.
                return Ok(self.sealed.is_none());
            }
        }
        if result.is_ok()
            && self.sealed.is_none()
            && !self.listing_blocked
            && service.shared.body.is_some()
        {
            std::mem::swap(&mut self.listing, &mut self.other_listing);
            std::mem::swap(&mut self.deferred, &mut self.other_deferred);
            self.body = !self.body;
        }
        result.map(|()| self.sealed.is_none())
    }

    fn may_publish(&self) -> bool {
        // A slow first response is not a failed deletion. Once the page has an
        // actual failed outcome, retain the existing publication backpressure
        // through its retry attempts; otherwise a persistent outage could keep
        // producing obsolete files every time another request becomes pending.
        self.sealed.as_ref().is_none_or(|page| !page.failed)
    }

    pub(super) fn has_pending_delete(&self) -> bool {
        self.deleting.lock().unwrap().is_some()
    }

    // Also called while the publisher awaits its own workspace or native I/O.
    // Completing this same request releases the sealed page's reservation;
    // publication must not deadlock waiting for memory retained by unpolled GC.
    pub(super) async fn poll_pending_delete(&mut self, service: &Service) -> Result<()> {
        if self.delete_page(service).await? {
            self.next_run = tokio::time::Instant::now() + self.options.interval;
            if self.sealed.is_none() && !self.listing_blocked && service.shared.body.is_some() {
                std::mem::swap(&mut self.listing, &mut self.other_listing);
                std::mem::swap(&mut self.deferred, &mut self.other_deferred);
                self.body = !self.body;
            }
        }
        Ok(())
    }

    async fn collect_page(
        &mut self,
        service: &Service,
        catalog: &materializer::Publisher,
        storage: &Storage,
        budget: &Arc<MergeMemoryBudget>,
    ) -> Result<()> {
        let phases = service.shared.phases.clone();
        let state = catalog.state().await?;
        // A retained plan must first finish exact recovery, including its old
        // base. Do not classify uncommitted files while such a plan exists.
        if state.progress.pending.is_some() {
            return Ok(());
        }
        catalog.retire_history(&state).await?;
        let state = catalog.state().await?;
        ensure!(
            state.progress.pending.is_none(),
            "publication overlapped collection"
        );
        let manifests = state
            .manifests
            .as_ref()
            .context("annotation manifests unavailable")?;
        let boundary = manifests.history_boundary();
        // A successor alone is not retirement. Once a gap exists, bootstrap
        // remains an adoption receipt, not a permanently readable snapshot.
        // Exclusive annotation reads can only capture the current local base;
        // already captured old bases remain protected independently below.
        let bootstrap_receipt_only = boundary.retained_from > manifests.initial_version + 1;
        let roots = service.shared.staging.protected_bases().await?;
        let base = roots
            .datasets()
            .next()
            .context("annotation local base missing")?;
        let body_storage = service.shared.body.as_ref().filter(|_| self.body);
        let destination = body_storage.unwrap_or(storage);
        let object_store = if let Some(body) = body_storage {
            paired::store(body).await?
        } else {
            base.object_store(None).await?
        };
        let store = object_store.inner.clone();
        // Inspect at most the first reference. Do not enumerate arbitrary tag
        // or branch histories into memory. External writers are prohibited by
        // the exclusive-ownership contract, including during this check.
        for prefix in [
            lance::dataset::refs::base_tags_path(&destination.base),
            lance::dataset::refs::base_branches_contents_path(&destination.base),
        ] {
            ensure!(
                phases
                    .measure(Phase::CollectList, store.list(Some(&prefix)).try_next())
                    .await
                    .context(ListingFailure)?
                    .is_none(),
                "annotation data collection requires a table without tags or branches"
            );
        }
        let page_charge = self.options.max_bytes * 3;
        let workspace = budget.reserve(page_charge).await;
        let mut candidates = phases
            .measure(Phase::CollectList, async {
                if self.listing.is_none() {
                    self.listing = Some(store.list(Some(&destination.base)));
                }
                let mut candidates = BTreeMap::new();
                let mut progress = service.shared.progress.subscribe();
                let mut bytes = 0;
                let mut examined = 0;
                let until =
                    tokio::time::Instant::now() + self.options.interval.min(Duration::from_secs(1));
                for _ in 0..self.options.max_objects {
                    // Check even when the stream is immediately ready. Only stop polling
                    // this borrowed next-item future: the stream retains its in-flight
                    // request and continuation token across publications.
                    if tokio::time::Instant::now() >= until {
                        break;
                    }
                    let (path, size) = if let Some(deferred) = self.deferred.take() {
                        deferred
                    } else {
                        let Some(next) =
                            next_listing(self.listing.as_mut().unwrap(), &mut progress, until)
                                .await
                        else {
                            break;
                        };
                        let next = next.context(ListingFailure)?;
                        self.listing_blocked = false;
                        service.shared.reclamation.lock().unwrap().listing_blocked = false;
                        let Some(meta) = next else {
                            self.listing = None;
                            break;
                        };
                        examined += 1;
                        (meta.location, meta.size)
                    };
                    // This owner builds ID indexes only in the narrow table.
                    // Body index objects keep their existing retention behavior.
                    if !candidate(&path, &destination.base, service.ingress.scope.incarnation)
                        && !(body_storage.is_none()
                            && indices::candidate(&path, &storage.base).is_some())
                    {
                        continue;
                    }
                    // Charge string storage and map nodes conservatively. A deferred
                    // single item prevents dropping the first object of the next page.
                    let charge = 2 * path.as_ref().len() + 256;
                    if bytes + charge > self.options.max_bytes {
                        self.deferred = Some((path, size));
                        break;
                    }
                    bytes += charge;
                    candidates.insert(path, size);
                }
                {
                    let mut observed = service.shared.reclamation.lock().unwrap();
                    observed.passes += 1;
                    observed.examined += examined;
                }
                Ok::<_, anyhow::Error>(candidates)
            })
            .await?;
        if candidates.is_empty() {
            // Nothing to protect, seal or delete in this slice. The next pass
            // captures current roots before classifying further listing items.
            return Ok(());
        }
        // This finite page is now closed. No more listing or candidate additions
        // occur after protection/sealing, even if a DELETE outlives this owner.
        phases
            .measure(Phase::CollectProtect, async {
                for base in roots.datasets() {
                    if let Some(body) = body_storage {
                        paired::protect_body(
                            &mut candidates,
                            base,
                            body,
                            &object_store,
                            service.ingress.scope.incarnation,
                            budget,
                            page_charge,
                            true,
                        )
                        .await?;
                    } else {
                        protect(
                            &mut candidates,
                            base.manifest(),
                            &base.manifest_location().path,
                            &storage.base,
                            true,
                        )?;
                        indices::protect(
                            &mut candidates,
                            base.manifest(),
                            base.manifest_location(),
                            &storage.base,
                            &object_store,
                            &self.metadata,
                            budget,
                            page_charge,
                        )
                        .await?;
                    }
                }
                for record in manifests.versions.values() {
                    // A fully protected page cannot delete anything. Further immutable
                    // manifests cannot change that; the next page captures fresh roots.
                    if candidates.is_empty() {
                        return Ok(());
                    }
                    // Open one retained manifest at a time, including bootstrap and the
                    // durable advancement receipt. Session metadata has its own bound.
                    let charge = usize::try_from(record.object.bytes)?
                        .checked_mul(4)
                        .context("annotation manifest workspace overflow")?;
                    ensure!(
                        page_charge + charge <= budget.limit(),
                        "annotation retained manifest exceeds collection workspace"
                    );
                    let _manifest = budget.reserve(charge).await;
                    // The captured quorum state already names the selected immutable
                    // object. Read its metadata directly; reopening a Dataset here would
                    // issue redundant catalog queries for every retained version/page.
                    record.object.validate(record.version)?;
                    // join takes one path component: passing "_versions/name" would
                    // encode the slash as %2F and address a different blob object.
                    let name = record.object.path.strip_prefix("_versions/").unwrap();
                    let path = storage.base.clone().join("_versions").join(name);
                    let retain_data =
                        record.version != manifests.initial_version || !bootstrap_receipt_only;
                    if let Some(body) = body_storage {
                        // The same finite durable narrow history selects the body
                        // roots. Never infer liveness from the body table's latest
                        // mainline or from detached-version numeric ordering.
                        let narrow =
                            checkpoint::load(&service.ingress, storage, record.version).await?;
                        ensure!(
                            narrow.manifest_location().path == path,
                            "retained narrow manifest identity changed"
                        );
                        paired::protect_body(
                            &mut candidates,
                            &narrow,
                            body,
                            &object_store,
                            service.ingress.scope.incarnation,
                            budget,
                            page_charge + charge,
                            retain_data,
                        )
                        .await?;
                        continue;
                    }
                    let manifest =
                        manifest_cache::read(&self.metadata, &object_store, &path, record).await?;
                    ensure!(
                        manifest.version == record.version,
                        "retained manifest version mismatch"
                    );
                    protect(
                        &mut candidates,
                        &manifest,
                        &path,
                        &storage.base,
                        retain_data,
                    )?;
                    if retain_data {
                        let location = lance_table::io::commit::ManifestLocation {
                            version: record.version,
                            path: path.clone(),
                            size: Some(record.object.bytes),
                            naming_scheme: if record.object.v2_names {
                                lance_table::io::commit::ManifestNamingScheme::V2
                            } else {
                                lance_table::io::commit::ManifestNamingScheme::V1
                            },
                            e_tag: None,
                            identity: Some(record.identity.to_string()),
                        };
                        indices::protect(
                            &mut candidates,
                            &manifest,
                            &location,
                            &storage.base,
                            &object_store,
                            &self.metadata,
                            budget,
                            page_charge + charge,
                        )
                        .await?;
                    }
                }
                Ok::<_, anyhow::Error>(())
            })
            .await?;
        if candidates.is_empty() {
            return Ok(());
        }
        let reply = materializer::commit(
            &service.ingress.client,
            Command::RetireHistory(p::RetireManifestHistory {
                authority: catalog.authority().clone(),
                incarnation: service.ingress.scope.incarnation,
                retain_from: boundary.retained_from,
            }),
        )
        .await?;
        ensure!(
            matches!(reply, Applied::Publisher(p::Reply::ManifestHistory(ref actual)) if *actual == boundary),
            "annotation collection fence rejected: {reply:?}"
        );
        // An unknown fence result returns above without deleting anything. A
        // subsequent owner enumerates again; it never guesses the old outcome.
        // Each cloud chunk fits a single native batch (Azure: 256, S3: 1000).
        // Keep the same sealed page and configured call concurrency. Passing
        // one path to ObjectStoreExt::delete would defeat SDK bulk deletion.
        let batch_size = if matches!(object_store.scheme(), "az" | "s3") {
            256
        } else {
            1
        };
        self.sealed = Some(SealedPage {
            store,
            paths: candidates.into_iter().collect(),
            batch_size,
            failed: false,
            _workspace: workspace,
        });
        self.delete_page(service).await?;
        Ok(())
    }

    /// Returns true only after the original request's complete outcomes arrive.
    /// A pending request stays in this collector; dropping the owner cancels its
    /// local future, while the existing sealed-page protocol covers remote I/O
    /// that outlives that owner. This does not spawn a detached worker or resend.
    async fn delete_page(&mut self, service: &Service) -> Result<bool> {
        if self.deleting.get_mut().unwrap().is_none() {
            let page = self.sealed.as_mut().unwrap();
            let paths = std::mem::take(&mut page.paths);
            let store = page.store.clone();
            let batch_size = page.batch_size;
            let parallelism = self.options.delete_parallelism;
            let phases = service.shared.phases.clone();
            *self.deleting.get_mut().unwrap() = Some(Box::pin(async move {
                phases
                    .measure(Phase::CollectDelete, async move {
                        let outcomes = stream::iter(paths)
                            .chunks(batch_size)
                            .map(move |batch| {
                                let store = store.clone();
                                async move {
                                    let results = deletion::delete_batch(store, &batch).await;
                                    (batch, results)
                                }
                            })
                            .buffer_unordered(parallelism)
                            .collect::<Vec<_>>()
                            .await;
                        if outcomes.iter().any(|(_, results)| {
                            results
                                .iter()
                                .any(|result| result.as_ref().err().is_some_and(deletion::invalid))
                        }) {
                            bail!("invalid sealed annotation deletion response");
                        }
                        Ok(outcomes)
                    })
                    .await
            }));
        }
        let outcomes =
            match futures::poll!(self.deleting.get_mut().unwrap().as_mut().unwrap().as_mut()) {
                Poll::Pending => return Ok(false),
                Poll::Ready(outcomes) => outcomes,
            };
        // Retire a completed future before propagating its error. The first
        // poll can already fail; it must not appear pending to step().
        *self.deleting.get_mut().unwrap() = None;
        let outcomes = outcomes?;
        let page = self.sealed.as_mut().unwrap();
        // Locally detected response corruption remains a protocol failure.
        // It must not be treated like an unresolved storage DELETE.
        if outcomes.iter().any(|(_, results)| {
            results
                .iter()
                .any(|result| result.as_ref().err().is_some_and(deletion::invalid))
        }) {
            bail!("invalid sealed annotation deletion response");
        }
        let mut first_error = None;
        let mut observed = service.shared.reclamation.lock().unwrap();
        for (batch, results) in outcomes {
            if results.len() != batch.len() {
                // A whole-request failure leaves every path unresolved. Some
                // may already be absent; repeating these immutable DELETE requests is safe.
                observed.delete_failures += batch.len() as u64;
                page.paths.extend(batch);
                first_error = first_error.or_else(|| results.into_iter().find_map(Result::err));
                continue;
            }
            for (path, outcome) in batch.into_iter().zip(results) {
                match outcome {
                    Ok(bytes) => {
                        observed.deleted += 1;
                        observed.deleted_bytes += bytes;
                    }
                    Err(error) => {
                        observed.delete_failures += 1;
                        page.paths.push(path);
                        first_error.get_or_insert(error);
                    }
                }
            }
        }
        observed.pending_objects = page.paths.len();
        observed.pending_bytes = page.paths.iter().map(|(_, bytes)| bytes).sum();
        if let Some(error) = first_error {
            page.failed = true;
            eprintln!("sealed annotation deletion remains pending: {error}");
        }
        if page.paths.is_empty() {
            self.sealed = None;
        }
        Ok(true)
    }
}

async fn next_listing(
    listing: &mut BoxStream<'static, object_store::Result<ObjectMeta>>,
    progress: &mut watch::Receiver<Progress>,
    until: tokio::time::Instant,
) -> Option<object_store::Result<Option<ObjectMeta>>> {
    tokio::select! {
        // Consume ready data/EOF/errors even under pressure so collection can
        // progress. Yield only a pending response, retaining the listing stream
        // while the serial publisher frees source/staging capacity.
        biased;
        next = tokio::time::timeout_at(until, listing.try_next()) => next.ok(),
        _ = progress.wait_for(|p| p.backpressure.is_some()) => None,
    }
}

fn candidate(path: &Path, base: &Path, incarnation: Uuid) -> bool {
    if path.as_ref().len() > 4096 {
        return false;
    }
    let Some(relative) = path
        .as_ref()
        .strip_prefix(base.as_ref())
        .and_then(|p| p.strip_prefix('/'))
    else {
        return false;
    };
    let Some((directory, name)) = relative.split_once('/') else {
        return false;
    };
    if directory == "_annotation_preparations" {
        // The serial collector only runs after pending preparation/confirmation
        // recovery has finished. No reader depends on this sidecar: selected
        // Lance versions carry their own native transaction file, protected
        // below with the manifest. Keep the closed-page quorum fence for these
        // objects as well, including requests outliving an old owner.
        let mut parts = name.split('/');
        let (Some(instance), Some(transaction), Some(file), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return false;
        };
        return Uuid::parse_str(instance).is_ok_and(|id| id == incarnation)
            && Uuid::parse_str(transaction).is_ok_and(|id| !id.is_nil())
            && file.strip_suffix(".pb").is_some_and(|digest| {
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            });
    }
    if name.contains('/') {
        return false;
    }
    match directory {
        "data" => name.ends_with(".lance"),
        "_transactions" => name.ends_with(".txn"),
        "_versions" => {
            name.ends_with(".manifest")
                || name
                    .split_once(".manifest-")
                    .is_some_and(|(_, suffix)| Uuid::parse_str(suffix).is_ok())
        }
        _ => false,
    }
}

fn protect(
    candidates: &mut BTreeMap<Path, u64>,
    manifest: &lance_table::format::Manifest,
    manifest_path: &Path,
    base: &Path,
    retain_data: bool,
) -> Result<()> {
    ensure!(
        manifest.base_paths.is_empty() && manifest.branch.is_none() && manifest.tag.is_none(),
        "annotation data collection does not support shared or branched bases"
    );
    // Local readers carry their exact immutable manifest location; a retired
    // catalog version can still be required by one of these admitted views.
    candidates.remove(manifest_path);
    if let Some(transaction) = &manifest.transaction_file {
        ensure!(
            !transaction.is_empty() && !transaction.contains('/'),
            "invalid annotation transaction path"
        );
        candidates.remove(
            &base
                .clone()
                .join("_transactions")
                .join(transaction.as_str()),
        );
    }
    // Keep the bootstrap manifest and native transaction for exact adoption
    // receipts. Current/retained snapshots and live readers protect any of its
    // data they still use; obsolete initial files need not live forever.
    if !retain_data {
        return Ok(());
    }
    let prefix = base.clone().join("data");
    for file in manifest
        .fragments
        .iter()
        .flat_map(|f| f.referenced_lance_files())
    {
        ensure!(
            file.base_id.is_none(),
            "annotation data file has an external base"
        );
        candidates.remove(&prefix.clone().join(file.path.as_str()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_candidates_are_scoped_to_this_table_incarnation() {
        let base = Path::from("owned/table");
        let incarnation = Uuid::new_v4();
        let transaction = Uuid::new_v4();
        let digest = "a".repeat(64);
        let path = |name: &str| Path::from(format!("owned/table/_annotation_preparations/{name}"));
        assert!(candidate(
            &path(&format!("{incarnation}/{transaction}/{digest}.pb")),
            &base,
            incarnation
        ));
        for name in [
            format!("{}/{transaction}/{digest}.pb", Uuid::new_v4()),
            format!("{incarnation}/{}/{digest}.pb", Uuid::nil()),
            format!("{incarnation}/{transaction}/short.pb"),
            format!("{incarnation}/{transaction}/{}.pb", "z".repeat(64)),
            format!("{incarnation}/{transaction}/{digest}.pb/child"),
            format!("{incarnation}/{transaction}/{digest}.json"),
        ] {
            assert!(!candidate(&path(&name), &base, incarnation), "{name}");
        }
        assert!(!candidate(
            &Path::from(format!(
                "owned/table-other/_annotation_preparations/{incarnation}/{transaction}/{digest}.pb"
            )),
            &base,
            incarnation
        ));
    }
}
