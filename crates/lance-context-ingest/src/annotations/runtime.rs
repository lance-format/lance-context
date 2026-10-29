//! One owned applier and continuously draining publisher for an annotation shard.
use super::*;
use crate::{annotation_publication as publication, materializer};
use lance::Dataset;
use lance_context_core::annotation_staging::MAX_RESULTS_BYTES;
use lance_context_wal::replication::publication as p;
use materializer::observation::Phase;
use std::collections::BTreeMap;

mod compaction;
pub mod reclamation;

pub struct Configuration {
    pub limits: Limits,
    pub upload_budget: Arc<MergeMemoryBudget>,
    pub apply_budget: Arc<MergeMemoryBudget>,
    pub publication_budget: Arc<MergeMemoryBudget>,
    /// Source thresholds and history bound use the existing publisher options.
    /// Compaction preserves the last annotation checkpoint; session indexing
    /// and remote preparation are not enabled here.
    pub publication: materializer::Options,
}

pub struct Runtime {
    driver: Driver,
    publisher: Publisher,
    #[cfg(feature = "http")]
    read_cut: u64,
}

struct Publisher {
    service: Service,
    catalog: materializer::Publisher,
    storage: Storage,
    options: materializer::Options,
    budget: Arc<MergeMemoryBudget>,
    batch_started: Option<tokio::time::Instant>,
    compaction_started: Option<tokio::time::Instant>,
    reclamation: Option<reclamation::Collector>,
}

impl Runtime {
    /// Enable data/manifest/transaction collection for an exclusively owned, unbranched
    /// table. Readers must check ownership after materializing their response,
    /// as the annotation HTTP Owner does. Independently served Services, raw
    /// Dataset readers, shared files and URI reuse are outside this contract.
    /// Only this runtime may prepare/publish files. The ordinary constructor
    /// keeps physical collection disabled.
    pub fn with_exclusive_reclamation(mut self, options: reclamation::Options) -> Result<Self> {
        self.publisher.reclamation = Some(reclamation::Collector::new(
            options,
            &self.publisher.budget,
            &self.publisher.storage,
        )?);
        Ok(self)
    }
    #[cfg(feature = "http")]
    pub(crate) fn http_binding(&self) -> (Service, p::Authority, u64) {
        (
            self.publisher.service.clone(),
            self.publisher.catalog.authority().clone(),
            self.read_cut,
        )
    }
    /// The caller supplies the existing durable scheduler's acquisition identity.
    /// Reopening resolves a retained preparation before exposing the new service.
    pub async fn connect(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        storage: Storage,
        configuration: Configuration,
        acquisition: materializer::Acquisition,
    ) -> Result<(Service, Self)> {
        let c = configuration;
        ensure!(
            c.publication.session_index.is_none() && !c.publication.pipeline_preparation,
            "annotation runtime does not support session indexing or remote preparation"
        );
        let workspace = c
            .limits
            .retained_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(2 * MAX_RESULTS_BYTES))
            .context("annotation publication workspace overflow")?;
        ensure!(
            workspace <= c.publication_budget.limit()
                && !Arc::ptr_eq(&c.publication_budget, &c.upload_budget)
                && !Arc::ptr_eq(&c.publication_budget, &c.apply_budget),
            "annotation publication requires independent capacity for a full frozen generation"
        );
        let ingress = Service::validated_ingress(
            client,
            layout,
            binding,
            &c.limits,
            c.upload_budget.clone(),
            &c.apply_budget,
        )
        .await?;
        let connect = || {
            Service::open(
                ingress.clone(),
                &storage,
                c.limits.clone(),
                c.apply_budget.clone(),
            )
        };
        let catalog = materializer::Publisher::acquire(
            ingress.clone(),
            storage.clone(),
            c.publication.clone(),
            c.publication_budget.clone(),
            acquisition,
        )
        .await?;
        // Fence the predecessor before selecting a recovery base. Opening first
        // could capture a snapshot that it supersedes before this acquisition.
        let (mut service, mut driver) = connect().await?;
        let mut publisher = Publisher {
            service: service.clone(),
            catalog,
            storage: storage.clone(),
            options: c.publication.clone(),
            budget: c.publication_budget.clone(),
            batch_started: None,
            compaction_started: None,
            reclamation: None,
        };
        let mut view = publisher.catalog.state().await?;
        if view.progress.prepared.is_some() {
            // No applier has run. Reuse the durable files/transaction rather
            // than replaying and producing different files for this plan.
            publisher.resolve_prepared(&view).await?;
            publisher.catalog.finish_pending().await?;
            (service, driver) = connect().await?;
            publisher.service = service.clone();
            view = publisher.catalog.state().await?;
        }
        driver = driver.with_publication_limits(
            c.publication.max_records,
            c.publication.max_source_bytes as u64,
        )?;
        if let Some(pending) = view
            .progress
            .pending
            .as_ref()
            .filter(|p| !p.plan.is_maintenance())
        {
            let selection = publisher.selection(pending);
            let (_, through) = selection.scope()?;
            let range = &pending.plan.sources[&service.ingress.group(0)?];
            ensure!(
                range.after == service.source_observation().after
                    && range.records.len() <= c.publication.max_records
                    && pending.plan.source_bytes()? <= c.publication.max_source_bytes as u64,
                "retained annotation plan exceeds this applier's source window"
            );
            *service.shared.replay_cut.lock().unwrap() = Some(through);
        }
        publisher
            .catalog
            .observe_with(service.shared.phases.clone());
        // Ownership was fenced before opening this view. An owned point read
        // must first replay every predecessor write, including ACKed updates
        // absent from Lance. Later successful writes already await local apply;
        // reads need not wait for later, still-unacknowledged WAL proposals.
        #[cfg(feature = "http")]
        let read_cut = service.barrier().await?;
        Ok((
            service,
            Self {
                driver,
                publisher,
                #[cfg(feature = "http")]
                read_cut,
            },
        ))
    }

    /// No detached tasks. Publication/protocol failures stop this owner and its
    /// read barriers. Restart reconciles durable plan/preparation/selection state.
    /// Sealed DELETE failures retain one bounded page and pause publication while
    /// keeping applied values readable. Only immutable DELETE requests are repeated;
    /// this loop never invents a request or blindly retries failed mutations.
    pub async fn run(mut self, shutdown: impl Future<Output = ()>) -> Result<()> {
        tokio::pin!(shutdown);
        tokio::select! {
            _ = &mut shutdown => Ok(()),
            result = self.driver.run(std::future::pending()) => result,
            result = self.publisher.run() => result,
        }
    }
}

impl Publisher {
    fn selection(&self, pending: &p::PendingPlan) -> publication::Selection {
        publication::Selection {
            table: self.service.ingress.scope.table,
            incarnation: self.service.ingress.scope.incarnation,
            pending: pending.clone(),
        }
    }

    async fn run(&mut self) -> Result<()> {
        let phases = self.service.shared.phases.clone();
        loop {
            if let Some(collector) = &mut self.reclamation {
                let cleared = phases
                    .measure(
                        Phase::Collect,
                        collector.step(&self.service, &self.catalog, &self.storage, &self.budget),
                    )
                    .await?;
                if !cleared {
                    // Keep this owner and its applied read view alive, but do
                    // not create more files behind an unresolved deletion page.
                    // The independent applier stops at its existing source and
                    // staging bounds; it never ACKs an unapplied update.
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    continue;
                }
            }
            self.publish_next().await?;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn publish_next(&mut self) -> Result<()> {
        let phases = self.service.shared.phases.clone();
        let mut view = self.catalog.state().await?;
        if phases
            .measure(Phase::Compact, self.compact_next(&view))
            .await?
        {
            return Ok(());
        }
        let frozen = if let Some(pending) = view.progress.pending.as_ref() {
            ensure!(
                view.progress.prepared.is_none(),
                "preparation must be resolved before application starts"
            );
            let selection = self.selection(pending);
            let (_, through) = selection.scope()?;
            self.service.wait_through(through).await?;
            let frozen = self
                .service
                .freeze(pending.plan.transaction)
                .await?
                .context("retained plan has no application checkpoint")?;
            ensure!(
                frozen.sources() == &pending.plan.sources[&self.service.ingress.group(0)?],
                "replayed annotation prefix differs from its retained plan"
            );
            frozen
        } else {
            let source = self.service.source_observation();
            if source.records == 0 {
                self.batch_started = None;
                return Ok(());
            }
            let started = self
                .batch_started
                .get_or_insert_with(tokio::time::Instant::now);
            if source.bytes < self.options.min_source_bytes as u64
                && source.records < self.options.max_records
                && !self.service.shared.progress.borrow().backpressured
                && started.elapsed() < self.options.max_batch_wait
            {
                return Ok(());
            }
            self.catalog.retire_history(&view).await?;
            let frozen = self
                .service
                .freeze(Uuid::new_v4())
                .await?
                .context("nonempty annotation source window has no checkpoint")?;
            let group = self.service.ingress.group(0)?;
            let plan = p::Plan {
                transaction: Uuid::parse_str(frozen.identity())?,
                // A handoff can start above version one. The progress value is
                // only a lower bound until the first managed publication.
                read_version: view
                    .manifests
                    .as_ref()
                    .context("annotation manifests unavailable")?
                    .head,
                sources: BTreeMap::from([(group, frozen.sources().clone())]),
                next_group: group,
            };
            let reply = materializer::commit(
                &self.service.ingress.client,
                Command::Plan(p::PlanRequest {
                    authority: self.catalog.authority().clone(),
                    plan,
                }),
            )
            .await?;
            ensure!(
                matches!(reply, Applied::Publisher(p::Reply::Planned { .. })),
                "annotation plan rejected: {reply:?}"
            );
            view = self.catalog.state().await?;
            frozen
        };
        self.batch_started = None;
        self.compaction_started = None;
        let pending = view
            .progress
            .pending
            .as_ref()
            .context("annotation plan unavailable")?;
        self.catalog.pin(pending).await?;
        let selection = self.selection(pending);
        let prepared = phases
            .measure(Phase::Prepare, frozen.prepare(&self.budget, 8192))
            .await?;
        // Preserve these exact bytes until Prepare is resolved. On any unknown
        // result this runtime stops; a fenced restart reads the retained result.
        let descriptors = phases
            .measure(
                Phase::BlobWrite,
                publication::persist_descriptors(
                    prepared.annotations(),
                    &selection,
                    &self.storage,
                    self.service.ingress.client.max_descriptor_bytes(),
                ),
            )
            .await?;
        let reply = materializer::commit(
            &self.service.ingress.client,
            Command::Prepare(p::PrepareRequest {
                authority: self.catalog.authority().clone(),
                transaction: pending.plan.transaction,
                plan_index: pending.index,
                descriptors,
            }),
        )
        .await?;
        ensure!(
            matches!(reply, Applied::Publisher(p::Reply::Prepared { .. })),
            "annotation preparation rejected: {reply:?}"
        );
        let ready = self.catalog.state().await?;
        let selected = self.resolve_prepared(&ready).await?;
        self.service.install_published(&prepared, selected).await?;
        // The confirmed selected base now owns recovery and read visibility.
        // Release this publisher's dirty-value/workspace permits before waiting
        // on source retirement RPCs; held readers retain their own references.
        drop(prepared);
        drop(frozen);
        self.catalog.finish_pending().await?;
        Ok(())
    }

    /// Reuse the exact durable descriptors or verify the already selected native
    /// transaction. Upload completion alone is never evidence of publication.
    /// An exclusive borrow keeps the future Send: the collector's retained LIST
    /// stream is Send but need not be Sync.
    async fn resolve_prepared(&mut self, view: &p::Table) -> Result<Arc<Dataset>> {
        self.service
            .shared
            .phases
            .clone()
            .measure(Phase::Commit, self.resolve_prepared_inner(view))
            .await
    }

    async fn resolve_prepared_inner(&mut self, view: &p::Table) -> Result<Arc<Dataset>> {
        let pending = view
            .progress
            .pending
            .as_ref()
            .context("annotation plan unavailable")?;
        let prepared = view
            .progress
            .prepared
            .as_ref()
            .context("annotation preparation unavailable")?;
        let selection = self.selection(pending);
        if !pending.plan.is_maintenance() {
            selection.scope()?;
        }
        let base = checkpoint::load(
            &self.service.ingress,
            &self.storage,
            pending.plan.read_version,
        )
        .await?;
        let target = pending.plan.target_version()?;
        let manifests = view
            .manifests
            .as_ref()
            .context("annotation manifests unavailable")?;
        if !manifests.versions.contains_key(&target) {
            let predecessor = manifests
                .versions
                .get(&pending.plan.read_version)
                .context("annotation predecessor unavailable")?;
            if pending.plan.is_maintenance() {
                materializer::stage::publish(
                    &self.catalog,
                    pending,
                    prepared,
                    predecessor,
                    (*base).clone(),
                )
                .await?;
            } else {
                publication::publish(
                    Arc::new(self.service.ingress.client.manifest_catalog()),
                    &self.service.ingress.layout,
                    &self.storage,
                    p::lance::ManifestCommit {
                        authority: self.catalog.authority().clone(),
                        incarnation: self.service.ingress.scope.incarnation,
                        transaction: pending.plan.transaction,
                        plan_index: pending.index,
                        prepared_index: prepared.index,
                        descriptor_digest: prepared.descriptors.digest,
                        predecessor: predecessor.identity,
                        target,
                    },
                    &selection,
                    &prepared.descriptors,
                    base.clone(),
                )
                .await?;
            }
        }
        let current = self.catalog.state().await?;
        let record = current
            .manifests
            .as_ref()
            .context("annotation manifests unavailable")?
            .versions
            .get(&target)
            .context("annotation target not selected")?;
        ensure!(
            record.prepared_index == prepared.index,
            "annotation preparation index differs"
        );
        let selected = checkpoint::load(&self.service.ingress, &self.storage, target).await?;
        if pending.plan.is_maintenance() {
            checkpoint::validate_compacted(&selected, &base, pending, prepared, record).await?;
        } else {
            publication::selected_cut(
                &selected,
                &base,
                &prepared.descriptors,
                &selection,
                record,
                &self.storage,
            )
            .await?;
        }
        let reply = materializer::commit(
            &self.service.ingress.client,
            Command::Confirm(p::ConfirmPublication {
                authority: self.catalog.authority().clone(),
                incarnation: self.service.ingress.scope.incarnation,
                transaction: pending.plan.transaction,
                plan_index: pending.index,
                prepared_index: prepared.index,
                descriptor_digest: prepared.descriptors.digest,
                target,
                manifest_identity: record.identity,
            }),
        )
        .await?;
        ensure!(
            matches!(reply, Applied::Publisher(p::Reply::Published(_))),
            "annotation confirmation rejected: {reply:?}"
        );
        Ok(selected)
    }
}
