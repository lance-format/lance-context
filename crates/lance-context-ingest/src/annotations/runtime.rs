//! One owned applier and continuously draining publisher for an annotation shard.
use super::*;
use crate::{annotation_publication as publication, materializer};
use lance::Dataset;
use lance_context_core::annotation_staging::MAX_RESULTS_BYTES;
use lance_context_wal::replication::publication as p;
use materializer::observation::Phase;
use std::collections::BTreeMap;

mod body_compaction;
mod compaction;
pub mod reclamation;

/// Native body preparation failed before managed publication. An owning task
/// must close its admission and recover through a newly fenced runtime using
/// the retained WAL and publication plan. This does not retry the client intent
/// or assert that the underlying storage failure is transient.
#[derive(Debug)]
pub struct PublicationWriteFailed(lance::Error);

impl std::fmt::Display for PublicationWriteFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "uncommitted body upload failed; fenced recovery required: {}",
            self.0
        )
    }
}

impl std::error::Error for PublicationWriteFailed {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

pub(super) fn publication_write_error(error: lance::Error) -> anyhow::Error {
    if matches!(error, lance::Error::IO { .. }) {
        anyhow::Error::new(PublicationWriteFailed(error))
    } else {
        error.into()
    }
}

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
    compaction_checked_version: Option<u64>,
    maintenance_deferred_for_pressure: bool,
    index_preparation: Option<materializer::annotation_index::Preparation>,
    compaction_preparation: Option<materializer::annotation_compaction::Preparation>,
    compaction_retry_serial: bool,
    reclamation: Option<reclamation::Collector>,
}

impl Runtime {
    /// Override parallelism within each immutable-body historical query,
    /// including speculative windows. Configure before running the owner.
    /// The original shared Session/native pool still admits actual allocations;
    /// this adds no separate scheduler or budget. Defaults remain unchanged.
    /// Measure fixed-snapshot query cost and live backlog before choosing values.
    pub fn with_body_lookup_parallelism(
        mut self,
        parallelism: crate::turn_ingestion::TurnLookupParallelism,
    ) -> Result<Self> {
        ensure!(
            self.driver.service.is_single_body(),
            "body lookup parallelism requires full-body ingestion"
        );
        parallelism.validate(&self.publisher.storage.session)?;
        self.driver.service.body_lookup_parallelism = Some(parallelism);
        Ok(self)
    }

    /// Opt in to bounded coalescing of underfilled full-body lookup windows.
    /// This gives producer lanes released by Applied time to submit their next
    /// record. It can add up to `wait` of intentional delay per short window;
    /// request processing and network timeouts retain their existing bounds.
    /// The default is zero. Configure before running this owner and measure
    /// source ACK latency/backlog as well as keys per scan before enabling.
    pub fn with_body_lookup_batch_wait(mut self, wait: Duration) -> Result<Self> {
        ensure!(
            self.driver.service.is_single_body() && wait <= Duration::from_secs(1),
            "body lookup batch wait requires full-body ingestion and at most one second"
        );
        self.driver.body_batch_wait = wait;
        Ok(self)
    }

    /// Obtain a process-local, opt-in diagnostic handle bound to this owner.
    /// Arming it captures one future historical lookup's keys, never bodies.
    pub fn body_lookup_capture(&self) -> Result<super::lookup_capture::Handle> {
        ensure!(
            self.publisher.service.is_single_body(),
            "not a single body runtime"
        );
        Ok(super::lookup_capture::Handle::new(
            self.publisher.service.shared.lookup_capture.clone(),
            self.publisher.catalog.authority().clone(),
        ))
    }

    #[cfg(test)]
    pub(crate) async fn apply_once_for_test(&mut self) -> Result<()> {
        self.driver.step_for_test().await
    }

    #[cfg(test)]
    pub(crate) async fn publish_once_for_test(&mut self) -> Result<()> {
        let mut reclamation = self.publisher.reclamation.take();
        let result = self
            .publisher
            .publish_with_pending_delete(reclamation.as_mut())
            .await;
        self.publisher.reclamation = reclamation;
        result
    }

    #[cfg(test)]
    pub(crate) fn index_preparation_finished_for_test(&self) -> Option<bool> {
        self.publisher
            .index_preparation
            .as_ref()
            .map(materializer::annotation_index::Preparation::is_finished)
    }

    #[cfg(all(test, feature = "http"))]
    pub(crate) fn compaction_preparation_finished_for_test(&self) -> Option<bool> {
        self.publisher
            .compaction_preparation
            .as_ref()
            .map(materializer::annotation_compaction::Preparation::is_finished)
    }

    #[cfg(all(test, feature = "http"))]
    pub(crate) async fn collect_once_for_test(&mut self) -> Result<bool> {
        let publisher = &mut self.publisher;
        publisher
            .reclamation
            .as_mut()
            .context("test owner has no collector")?
            .step(
                &publisher.service,
                &publisher.catalog,
                &publisher.storage,
                &publisher.budget,
                publisher.index_preparation.as_ref().map(|job| &job.source),
                publisher.compaction_preparation.as_ref(),
            )
            .await
    }

    /// Enable data/manifest/transaction collection for an exclusively owned, unbranched
    /// table (both destinations for mixed ingestion). Readers must check ownership after materializing their response,
    /// as the annotation HTTP Owner does. Independently served Services, raw
    /// Dataset readers, shared files and URI reuse are outside this contract.
    /// Only this runtime may prepare/publish files. The ordinary constructor
    /// keeps physical collection disabled.
    pub fn with_exclusive_reclamation(mut self, options: reclamation::Options) -> Result<Self> {
        ensure!(
            self.publisher.service.ingress.turn_spec.is_none(),
            "generic turn readers are not yet integrated with exclusive physical reclamation"
        );
        if let Some(body) = &self.publisher.service.shared.body {
            let narrow = self.publisher.storage.base.as_ref();
            let body = body.base.as_ref();
            ensure!(
                narrow != body
                    && !narrow.starts_with(&format!("{body}/"))
                    && !body.starts_with(&format!("{narrow}/")),
                "paired collection requires disjoint destination paths"
            );
        }
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
        Self::connect_mode(
            client,
            layout,
            binding,
            storage,
            None,
            configuration,
            acquisition,
            None,
            None,
        )
        .await
    }

    /// One owner and one fenced selection for both native tables. The narrow
    /// bootstrap must already contain the authorized initial body identity.
    pub async fn connect_mixed(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        storage: Storage,
        body: Storage,
        configuration: Configuration,
        acquisition: materializer::Acquisition,
    ) -> Result<(Service, Self)> {
        Self::connect_mode(
            client,
            layout,
            binding,
            storage,
            Some(body),
            configuration,
            acquisition,
            None,
            None,
        )
        .await
    }

    /// Own one explicitly bound bucket and its independent body/narrow tables.
    /// A physical binding cannot be reused for a different bucket after restart.
    #[allow(clippy::too_many_arguments)] // The shard augments the existing mixed connection.
    pub async fn connect_mixed_shard(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        storage: Storage,
        body: Storage,
        configuration: Configuration,
        acquisition: materializer::Acquisition,
        shard: Shard,
    ) -> Result<(Service, Self)> {
        Self::connect_mode(
            client,
            layout,
            binding,
            storage,
            Some(body),
            configuration,
            acquisition,
            Some(shard),
            None,
        )
        .await
    }

    /// Own a fresh generic body/narrow pair. The exact schema and composite
    /// identity must match its durable bootstrap; existing rollout bindings are
    /// never reinterpreted. Equal session keys route to one physical bucket.
    #[allow(clippy::too_many_arguments)]
    pub async fn connect_turn_shard(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        storage: Storage,
        body: Storage,
        configuration: Configuration,
        acquisition: materializer::Acquisition,
        shard: Shard,
        spec: &crate::turn_ingestion::TurnIngestionSpec,
    ) -> Result<(Service, Self)> {
        Self::connect_mode(
            client,
            layout,
            binding,
            storage,
            Some(body),
            configuration,
            acquisition,
            Some(shard),
            Some(spec),
        )
        .await
    }

    /// Own one full-schema immutable table. All producer lanes share its
    /// ordered first-create state and the original publication/recovery path.
    #[allow(clippy::too_many_arguments)]
    pub async fn connect_body_shard(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        storage: Storage,
        configuration: Configuration,
        acquisition: materializer::Acquisition,
        shard: Shard,
        spec: &crate::turn_ingestion::TurnIngestionSpec,
    ) -> Result<(Service, Self)> {
        ensure!(
            configuration.publication.turn_id_index.is_none()
                && configuration.publication.status_index.is_none(),
            "single body indexing is supplied by its consumer"
        );
        Self::connect_mode(
            client,
            layout,
            binding,
            storage,
            None,
            configuration,
            acquisition,
            Some(shard),
            Some(spec),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)] // Explicit routing augments the existing owned connection.
    async fn connect_mode(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        storage: Storage,
        body: Option<Storage>,
        configuration: Configuration,
        acquisition: materializer::Acquisition,
        shard: Option<Shard>,
        spec: Option<&crate::turn_ingestion::TurnIngestionSpec>,
    ) -> Result<(Service, Self)> {
        let c = configuration;
        ensure!(
            c.publication.session_index.is_none() && !c.publication.pipeline_preparation,
            "annotation runtime does not support session indexing or remote preparation"
        );
        ensure!(
            (c.publication.turn_id_index.is_none() && c.publication.status_index.is_none())
                || c.limits.retained_entries <= materializer::turn_id_index::MAX_BUILD_ROWS,
            "annotation indexing requires at most 1048576 retained annotation entries"
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
        let ingress = Service::validated_ingress_mode(
            client,
            layout,
            binding,
            &c.limits,
            c.upload_budget.clone(),
            &c.apply_budget,
            body.is_some(),
            shard,
            spec,
        )
        .await?;
        let connect = || {
            Service::open_mode(
                ingress.clone(),
                &storage,
                body.clone(),
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
            compaction_checked_version: None,
            maintenance_deferred_for_pressure: false,
            index_preparation: None,
            compaction_preparation: None,
            compaction_retry_serial: false,
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
        if c.publication.turn_id_index.is_some() || c.publication.status_index.is_some() {
            // Before the first advance, minimum_read_version is still the
            // protocol floor (1). Bootstrap can select a later native version.
            let version = view
                .manifests
                .as_ref()
                .context("annotation manifests unavailable")?
                .head;
            let base = checkpoint::load(&service.ingress, &storage, version).await?;
            materializer::TurnIdIndex::validate_base(&base)?;
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
            let range = &pending.plan.sources[&service.group()?];
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

    /// Publication/protocol failures stop this owner and its read barriers.
    /// A structured uncertain Pin or Confirm holds its frozen batch while
    /// resolving that exact selected operation; bounded apply and point reads
    /// stay live under their existing budgets.
    /// An I/O failure writing an uncommitted body exits the runtime with
    /// `PublicationWriteFailed`. The owning task closes its
    /// admission and performs fenced recovery from the retained plan and WAL.
    /// The runtime does not retry the upload or replay a client intent.
    /// Other failures still require restart to reconcile preparation/selection.
    /// An in-flight sealed DELETE retains one bounded page while publication
    /// continues. Failed outcomes pause new publication until that sealed page
    /// clears; applied values remain readable. Only immutable DELETE requests repeat;
    /// an uncertain deletion fence first repeats its exact monotonic catalog
    /// retirement, with no DELETE or new publication until confirmed. This loop
    /// never invents a request or blindly retries failed business mutations.
    /// One admitted native annotation index writer can outlive cancellation, retaining
    /// its workspace until completion. It cannot publish a manifest itself.
    pub async fn run(mut self, shutdown: impl Future<Output = ()>) -> Result<()> {
        use futures::FutureExt;
        // These are task-lifetime allocations. Erase the two large loop types
        // so embedding hosts do not inherit their recursive future layouts.
        let driver = self.driver.run(std::future::pending()).boxed();
        let publisher = self.publisher.run().boxed();
        tokio::pin!(shutdown);
        tokio::select! {
            _ = &mut shutdown => Ok(()),
            result = driver => result.context("annotation apply driver stopped"),
            result = publisher => result.context("annotation publisher stopped"),
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
        let mut reclamation = self.reclamation.take();
        loop {
            if let Some(collector) = &mut reclamation {
                let cleared = phases
                    .measure(
                        Phase::Collect,
                        collector.step(
                            &self.service,
                            &self.catalog,
                            &self.storage,
                            &self.budget,
                            self.index_preparation.as_ref().map(|job| &job.source),
                            self.compaction_preparation.as_ref(),
                        ),
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
            let result = self.publish_with_pending_delete(reclamation.as_mut()).await;
            if let Err(error) = result {
                if error.is::<materializer::retirement::HistoryUnknown>() {
                    // Retirement runs before freezing or preparing a new
                    // publication. Keep the bounded apply/read path alive;
                    // the next iteration captures fresh catalog authority.
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    continue;
                }
                return Err(error);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn publish_with_pending_delete(
        &mut self,
        collector: Option<&mut reclamation::Collector>,
    ) -> Result<()> {
        let Some(collector) = collector.filter(|c| c.has_pending_delete()) else {
            return self.publish_next().await;
        };
        // Only an already fenced immutable DELETE overlaps publication.
        // Drive its original future while publication waits for the page's
        // workspace, including when a test steps the publisher manually.
        let service = self.service.clone();
        let publication = self.publish_next();
        tokio::pin!(publication);
        loop {
            tokio::select! {
                result = &mut publication => return result,
                _ = tokio::time::sleep(Duration::from_millis(5)), if collector.has_pending_delete() => {
                    collector.poll_pending_delete(&service).await?;
                }
            }
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
                frozen.sources() == &pending.plan.sources[&self.service.group()?],
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
                && self.service.shared.progress.borrow().backpressure.is_none()
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
            let group = self.service.group()?;
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
            self.catalog.record_plan(plan).await?;
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
            .measure(Phase::Prepare, async {
                if self.service.is_single_body() {
                    frozen.prepare_full_body(&self.budget).await
                } else if self.service.shared.body.is_some() {
                    frozen.prepare_mixed(&self.budget, 8192).await
                } else {
                    frozen.prepare(&self.budget, 8192).await
                }
            })
            .await?;
        // Preserve these exact bytes until Prepare is confirmed, including a
        // quorum observation after an unknown reply. An unconfirmed result
        // still stops this runtime; a fenced restart reads the retained result.
        let descriptors = phases
            .measure(Phase::BlobWrite, async {
                let maximum = self.service.ingress.client.max_descriptor_bytes();
                if let Some(body) = &self.service.shared.body {
                    publication::mixed::persist_descriptors(
                        prepared.annotations(),
                        &selection,
                        &self.storage,
                        body,
                        maximum,
                    )
                    .await
                } else {
                    publication::persist_descriptors(
                        prepared.annotations(),
                        &selection,
                        &self.storage,
                        maximum,
                    )
                    .await
                }
            })
            .await?;
        self.catalog
            .record_preparation(p::PrepareRequest {
                authority: self.catalog.authority().clone(),
                transaction: pending.plan.transaction,
                plan_index: pending.index,
                descriptors,
            })
            .await?;
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
        let phases = self.service.shared.phases.clone();
        let base = phases
            .measure(
                Phase::AnnotationCommitBase,
                checkpoint::load(
                    &self.service.ingress,
                    &self.storage,
                    pending.plan.read_version,
                ),
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
            phases
                .measure(Phase::AnnotationCommitPublish, async {
                    if pending.plan.is_maintenance() && matches!(prepared.descriptors.format, 2 | 3)
                    {
                        materializer::stage::publish(
                            &self.catalog,
                            pending,
                            prepared,
                            predecessor,
                            (*base).clone(),
                        )
                        .await?;
                    } else {
                        let commit = p::lance::ManifestCommit {
                            authority: self.catalog.authority().clone(),
                            incarnation: self.service.ingress.scope.incarnation,
                            transaction: pending.plan.transaction,
                            plan_index: pending.index,
                            prepared_index: prepared.index,
                            descriptor_digest: prepared.descriptors.digest,
                            predecessor: predecessor.identity,
                            target,
                        };
                        let catalog = Arc::new(self.service.ingress.client.manifest_catalog());
                        if let Some(body) = &self.service.shared.body {
                            publication::mixed::publish(
                                catalog,
                                &self.service.ingress.layout,
                                &self.storage,
                                body,
                                commit,
                                &selection,
                                &prepared.descriptors,
                                base.clone(),
                            )
                            .await?;
                        } else {
                            publication::publish(
                                catalog,
                                &self.service.ingress.layout,
                                &self.storage,
                                commit,
                                &selection,
                                &prepared.descriptors,
                                base.clone(),
                            )
                            .await?;
                        }
                    }
                    Ok::<_, anyhow::Error>(())
                })
                .await?;
        }
        let (selected, manifest_identity) = phases
            .measure(Phase::AnnotationCommitSelect, async {
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
                let selected =
                    checkpoint::load(&self.service.ingress, &self.storage, target).await?;
                if pending.plan.is_maintenance() && matches!(prepared.descriptors.format, 2 | 3) {
                    checkpoint::validate_compacted(&selected, &base, pending, prepared, record)
                        .await?;
                } else if let Some(body) = &self.service.shared.body {
                    publication::mixed::selected_cut(
                        &selected,
                        &base,
                        &prepared.descriptors,
                        &selection,
                        record,
                        &self.storage,
                        body,
                    )
                    .await?;
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
                Ok::<_, anyhow::Error>((selected, record.identity))
            })
            .await?;
        phases
            .measure(
                Phase::AnnotationCommitConfirm,
                self.catalog.confirm_publication(p::ConfirmPublication {
                    authority: self.catalog.authority().clone(),
                    incarnation: self.service.ingress.scope.incarnation,
                    transaction: pending.plan.transaction,
                    plan_index: pending.index,
                    prepared_index: prepared.index,
                    descriptor_digest: prepared.descriptors.digest,
                    target,
                    manifest_identity,
                }),
            )
            .await?;
        Ok(selected)
    }
}
