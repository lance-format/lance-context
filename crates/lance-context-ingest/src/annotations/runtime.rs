//! One owned applier and continuously draining publisher for an annotation shard.
use super::*;
use crate::{annotation_publication as publication, materializer};
use lance::Dataset;
use lance_context_core::annotation_staging::MAX_RESULTS_BYTES;
use lance_context_wal::replication::publication as p;
use std::collections::BTreeMap;

pub struct Configuration {
    pub limits: Limits,
    pub upload_budget: Arc<MergeMemoryBudget>,
    pub apply_budget: Arc<MergeMemoryBudget>,
    pub publication_budget: Arc<MergeMemoryBudget>,
    /// Source thresholds and history bound use the existing publisher options.
    /// Compaction, session indexing and remote preparation are not enabled here.
    pub publication: materializer::Options,
}

pub struct Runtime {
    driver: Driver,
    publisher: Publisher,
}

struct Publisher {
    service: Service,
    catalog: materializer::Publisher,
    storage: Storage,
    options: materializer::Options,
    budget: Arc<MergeMemoryBudget>,
    batch_started: Option<tokio::time::Instant>,
}

impl Runtime {
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
            c.publication.compaction.is_none()
                && c.publication.session_index.is_none()
                && !c.publication.pipeline_preparation,
            "annotation runtime does not support ingestion maintenance options"
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
        let connect = || {
            Service::connect(
                client.clone(),
                layout.clone(),
                binding.clone(),
                &storage,
                c.limits.clone(),
                c.upload_budget.clone(),
                c.apply_budget.clone(),
            )
        };
        let (mut service, mut driver) = connect().await?;
        let catalog = materializer::Publisher::acquire(
            service.ingress.clone(),
            storage.clone(),
            c.publication.clone(),
            c.publication_budget.clone(),
            acquisition,
        )
        .await?;
        let mut publisher = Publisher {
            service: service.clone(),
            catalog,
            storage: storage.clone(),
            options: c.publication.clone(),
            budget: c.publication_budget.clone(),
            batch_started: None,
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
        if let Some(pending) = view.progress.pending.as_ref() {
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
        Ok((service, Self { driver, publisher }))
    }

    /// No detached tasks. A storage/protocol failure stops this owner and its
    /// read barriers. Restart reconciles durable plan/preparation/selection state.
    /// This loop does not invent a new request or blindly retry failed mutations.
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
        loop {
            self.publish_next().await?;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn publish_next(&mut self) -> Result<()> {
        let mut view = self.catalog.state().await?;
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
                read_version: view.progress.minimum_read_version,
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
        let pending = view
            .progress
            .pending
            .as_ref()
            .context("annotation plan unavailable")?;
        self.catalog.pin(pending).await?;
        let selection = self.selection(pending);
        let prepared = frozen.prepare(&self.budget, 8192).await?;
        // Preserve these exact bytes until Prepare is resolved. On any unknown
        // result this runtime stops; a fenced restart reads the retained result.
        let descriptors = publication::descriptors(prepared.annotations(), &selection)?;
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
        self.catalog.finish_pending().await?;
        Ok(())
    }

    /// Reuse the exact durable descriptors or verify the already selected native
    /// transaction. Upload completion alone is never evidence of publication.
    async fn resolve_prepared(&self, view: &p::Table) -> Result<Arc<Dataset>> {
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
        selection.scope()?;
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
        publication::selected_cut(&selected, &base, &prepared.descriptors, &selection, record)
            .await?;
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
