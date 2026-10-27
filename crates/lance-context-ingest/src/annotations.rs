//! Single-shard annotation apply and read barrier over the RF3 WAL.
//!
//! Native checkpoints restore values and bounded request outcomes together.
//! `runtime` composes the applier with continuous publication and recovery. The
//! application still supplies scheduling ownership and ingress routing.
use super::*;
use crate::{materializer::Storage, reader};
use lance::dataset::builder::DatasetBuilder;
use lance_context_api::annotation::TurnAnnotation;
use lance_context_core::annotation_staging::{
    AnnotationStaging, ApplyOutcome, Limits, Mutation, Rejection, Scope as StagingScope,
};
use lance_context_wal::replication::publication::lance::ManagedManifest;
use std::{future::Future, sync::Mutex};
use tokio::sync::watch;

pub(crate) mod checkpoint;
pub mod runtime;
mod window;
pub mod wire;
pub use checkpoint::{FrozenCheckpoint, PreparedCheckpoint};
pub use window::SourceObservation;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationResult {
    Applied,
    Rejected(Rejection),
}

/// A bounded latest outcome, independent of dirty turn values. Until these
/// outcomes are included in publication recovery state, retain the source WAL.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Completed {
    pub index: u64,
    pub meta: BatchMeta,
    pub result: MutationResult,
}

#[derive(Clone, Debug)]
struct Progress {
    through: u64,
    stopped: bool,
    backpressured: bool,
}

struct Shared {
    #[cfg(test)]
    apply_attempts: std::sync::atomic::AtomicU64,
    staging: AnnotationStaging,
    results: Mutex<checkpoint::Results>,
    window: Mutex<window::Window>,
    source_generations: Arc<tokio::sync::Semaphore>,
    replay_cut: Mutex<Option<u64>>,
    ordered: tokio::sync::Mutex<()>,
    progress: watch::Sender<Progress>,
    reclamation: Mutex<runtime::reclamation::Observation>,
}

/// Multiple independently connected ingresses may keep their own bounded view.
/// Every point read obtains a quorum barrier, then waits for its background
/// applier. Point reads never discover, fetch or decode WAL records themselves.
#[derive(Clone)]
pub struct Service {
    ingress: Ingress,
    limits: Limits,
    shared: Arc<Shared>,
    apply_budget: Arc<MergeMemoryBudget>,
}

struct Pending {
    index: u64,
    meta: BatchMeta,
    mutations: Vec<wire::Mutation>,
    _reservation: MergeReservation,
}

pub struct Driver {
    service: Service,
    pending: Option<Pending>,
    staging_pressure: Option<(u64, usize, usize)>,
}

/// Connect an authorized ingress without opening Lance or creating an applier.
/// Such ingresses forward annotation operations to the deployment's shard owner.
pub async fn connect_ingress(
    client: Arc<Client>,
    layout: Arc<Layout>,
    binding: TableBinding,
    limits: &Limits,
    budget: Arc<MergeMemoryBudget>,
) -> Result<Ingress> {
    ensure!(
        layout.buckets.len() == 1,
        "annotation ingress requires one shard"
    );
    let contract = Service::contract(&binding, limits)?;
    Ingress::connect_with_schema(
        client,
        layout,
        binding,
        budget,
        contract,
        wire::schema(limits),
    )
    .await
}

impl Service {
    #[cfg(test)]
    pub(crate) fn apply_attempts(&self) -> u64 {
        self.shared
            .apply_attempts
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn reclamation_observation(&self) -> runtime::reclamation::Observation {
        self.shared.reclamation.lock().unwrap().clone()
    }
    #[cfg(feature = "http")]
    pub(crate) fn ingress(&self) -> &Ingress {
        &self.ingress
    }

    pub(crate) async fn check_owner(
        &self,
        authority: &lance_context_wal::replication::publication::Authority,
    ) -> Result<()> {
        ensure!(
            !self.shared.progress.borrow().stopped,
            "annotation runtime stopped"
        );
        let query::Value::PublisherBinding(Some(view)) = crate::materializer::query(
            &self.ingress,
            self.ingress.client.catalog_group(),
            query::Query::PublisherBinding {
                table: self.ingress.scope.table,
            },
        )
        .await?
        else {
            bail!("annotation owner unavailable");
        };
        ensure!(
            view.authority == *authority
                && view.destination == self.ingress.scope.destination
                && view.ingest == self.ingress.scope.contract
                && view.incarnation == Some(self.ingress.scope.incarnation),
            "annotation owner changed"
        );
        Ok(())
    }
    /// Initially support exactly one virtual shard. The table is already
    /// provisioned with narrow annotation columns and the explicit policy.
    pub async fn connect(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        storage: &Storage,
        limits: Limits,
        budget: Arc<MergeMemoryBudget>,
        apply_budget: Arc<MergeMemoryBudget>,
    ) -> Result<(Self, Driver)> {
        let ingress =
            Self::validated_ingress(client, layout, binding, &limits, budget, &apply_budget)
                .await?;
        Self::open(ingress, storage, limits, apply_budget).await
    }

    /// Validate configuration before acquiring ownership, without opening a
    /// Lance snapshot that a still-active predecessor could supersede.
    pub(super) async fn validated_ingress(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        limits: &Limits,
        budget: Arc<MergeMemoryBudget>,
        apply_budget: &Arc<MergeMemoryBudget>,
    ) -> Result<Ingress> {
        ensure!(
            layout.buckets.len() == 1,
            "annotation service requires one shard"
        );
        let schema = wire::schema(limits);
        let contract = Self::contract(&binding, limits)?;
        ensure!(
            decode_workspace(&contract, limits, contract.max_record_bytes as usize)?
                <= budget.limit(),
            "annotation decode exceeds shared workspace"
        );
        ensure!(!Arc::ptr_eq(&budget, apply_budget)
            && decode_workspace(&contract, limits, contract.max_record_bytes as usize)? <= apply_budget.limit(),
            "annotation apply needs independent reserved workspace so waiting uploads cannot block it");
        Ingress::connect_with_schema(client, layout, binding, budget, contract, schema).await
    }

    /// Owned runtimes call this only after their publisher epoch is acquired.
    pub(super) async fn open(
        ingress: Ingress,
        storage: &Storage,
        limits: Limits,
        apply_budget: Arc<MergeMemoryBudget>,
    ) -> Result<(Self, Driver)> {
        let (base, snapshot) = checkpoint::open(&ingress, storage).await?;
        let through = snapshot.through;
        let staging = AnnotationStaging::new(
            base,
            StagingScope {
                table_instance: ingress.scope.incarnation.to_string(),
                source_group: ingress.group(0)?,
                shard: 0,
            },
            through,
            limits.clone(),
        )?;
        let (progress, _) = watch::channel(Progress {
            through,
            stopped: false,
            backpressured: false,
        });
        let outcomes = vec![None; ingress.client.producer_slots() as usize];
        let window = window::Window::new(through, ingress.client.max_plan_source_bytes());
        let service = Self {
            ingress,
            limits,
            apply_budget,
            shared: Arc::new(Shared {
                staging,
                results: Mutex::new(checkpoint::Results {
                    base: snapshot,
                    latest: outcomes,
                }),
                window: Mutex::new(window),
                source_generations: Arc::new(tokio::sync::Semaphore::new(2)),
                replay_cut: Mutex::new(None),
                ordered: tokio::sync::Mutex::new(()),
                #[cfg(test)]
                apply_attempts: std::sync::atomic::AtomicU64::new(0),
                progress,
                reclamation: Mutex::new(Default::default()),
            }),
        };
        service.barrier().await?;
        let driver = Driver {
            service: service.clone(),
            pending: None,
            staging_pressure: None,
        };
        Ok((service, driver))
    }

    pub fn contract(binding: &TableBinding, limits: &Limits) -> Result<IngestContract> {
        limits.validate()?;
        Ok(IngestContract {
            policy: WritePolicy::TurnAnnotations,
            schema: arrow_schema_fingerprint(&wire::schema(limits))?,
            ..binding.contract()?
        })
    }

    pub async fn open_producer(&self, open: ProducerOpen) -> Result<Producer> {
        ensure!(
            open.bucket == 0,
            "annotation producer belongs to another shard"
        );
        self.ingress.open_producer(open).await
    }

    /// Reserve before buffering an upload; preparation retains this allocation
    /// through its exact-byte retries. Narrow patch allocations are budgeted too.
    pub async fn reserve_upload(&self, bytes: usize) -> Result<Upload> {
        let contract = &self.ingress.scope.contract;
        ensure!(
            bytes > 0 && bytes <= contract.max_record_bytes as usize,
            "annotation upload too large"
        );
        Ok(Upload {
            scope: self.ingress.scope.clone(),
            bytes,
            reservation: self
                .ingress
                .budget
                .reserve(decode_workspace(contract, &self.limits, bytes)?)
                .await,
        })
    }

    pub async fn prepare(
        &self,
        upload: Upload,
        body: Bytes,
        producer: &Producer,
        sequence: u64,
    ) -> Result<PreparedAppend> {
        ensure!(
            *upload.scope == *self.ingress.scope
                && *producer.scope == *self.ingress.scope
                && producer.bucket() == 0
                && sequence > 0
                && sequence < u64::MAX
                && upload.bytes == body.len(),
            "invalid annotation request identity or upload"
        );
        let schema = self.ingress.schema.clone();
        let producer = producer.id();
        let value_bytes = self.limits.value_bytes;
        tokio::task::spawn_blocking(move || {
            let mutations = wire::decode(
                body.clone(),
                schema,
                upload.scope.contract.max_rows,
                value_bytes,
            )?;
            let rows = mutations.len() as u32;
            drop(mutations);
            // Validation owns decoded patches only inside this blocking task.
            // Keep the existing encoded-body allowance through WAL submission
            // and applied visibility, but release the discarded decode workspace.
            let mut reservation = upload.reservation;
            reservation.shrink_to(body.len() * 3);
            Ok(PreparedAppend {
                batch: Batch::new(upload.scope.table, 0, producer, sequence, body),
                rows,
                scope: upload.scope,
                _reservation: reservation,
            })
        })
        .await?
    }

    /// Success is quorum acceptance AND an applied outcome. An expired deadline
    /// is ambiguous: retain the original request identity and bytes, not a new
    /// sequence. A producer has at most one unresolved request; only its latest
    /// outcome is retained. Older identities are rejected, never re-executed.
    pub async fn submit(&self, prepared: &PreparedAppend, deadline: Duration) -> Result<Completed> {
        tokio::time::timeout(deadline, async {
            let accepted = self.ingress.append(prepared).await?;
            self.wait_through(accepted.position.index).await?;
            let slot = slot(&accepted.meta, self.ingress.client.producer_slots())?;
            self.shared.results.lock().unwrap().resolve(&accepted, slot)
        })
        .await
        .context("annotation apply deadline expired; reconcile the original request")?
    }

    pub async fn get(&self, turn: &str, deadline: Duration) -> Result<Option<TurnAnnotation>> {
        tokio::time::timeout(deadline, async {
            let through = self.barrier().await?;
            self.wait_through(through).await?;
            Ok(self.shared.staging.capture(turn).await?.resolve().await?)
        })
        .await
        .context("annotation read barrier deadline expired")?
    }

    /// The HTTP owner fences both sides of this read. Its acquisition cut
    /// covers all predecessor writes; this owner's ACKs already await apply.
    /// A later unacknowledged proposal must not block reading applied values.
    #[cfg(feature = "http")]
    pub(crate) async fn get_owned(
        &self,
        turn: &str,
        recovery_cut: u64,
        deadline: Duration,
    ) -> Result<Option<TurnAnnotation>> {
        tokio::time::timeout(deadline, async {
            self.wait_through(recovery_cut).await?;
            Ok(self.shared.staging.capture(turn).await?.resolve().await?)
        })
        .await
        .context("annotation owned read deadline expired")?
    }

    async fn barrier(&self) -> Result<u64> {
        let (cut, value) = reader::observe(
            &self.ingress,
            self.ingress.group(0)?,
            query::Query::SourceClosure {
                table: self.ingress.scope.table,
            },
        )
        .await?;
        let query::Value::SourceClosure(closed) = value else {
            bail!("source closure unavailable");
        };
        if let Some(closed) = closed {
            let closure = closed.permit.closure();
            ensure!(
                closure.table == self.ingress.scope.table
                    && closure.incarnation == self.ingress.scope.incarnation
                    && closure.group == self.ingress.group(0)?
                    && closure.through <= self.shared.results.lock().unwrap().base.through,
                "annotation view is behind source closure; reconnect to its durable checkpoint"
            );
        }
        Ok(cut.quorum_index)
    }

    async fn wait_through(&self, through: u64) -> Result<()> {
        let mut progress = self.shared.progress.subscribe();
        loop {
            let observed = progress.borrow_and_update().clone();
            ensure!(!observed.stopped, "annotation applier stopped");
            if observed.through >= through {
                return Ok(());
            }
            progress
                .changed()
                .await
                .context("annotation applier closed")?;
        }
    }

    pub async fn observation(&self) -> (lance_context_core::annotation_staging::Observation, bool) {
        let state = self.shared.staging.observation().await;
        (state, self.shared.progress.borrow().backpressured)
    }
}

impl Driver {
    /// No detached tasks: the caller owns and joins this driver. A storage or
    /// protocol failure stops it and makes readers fail instead of serving stale
    /// values. Pressure alone retains the decoded batch for the next attempt.
    pub async fn run(mut self, shutdown: impl Future<Output = ()>) -> Result<()> {
        tokio::pin!(shutdown);
        let result = loop {
            let before = self.service.shared.progress.borrow().through;
            tokio::select! {
                _ = &mut shutdown => break Ok(()),
                outcome = self.step() => if let Err(error) = outcome { break Err(error); }
            }
            if self.service.shared.progress.borrow().through == before {
                tokio::select! {
                    _ = &mut shutdown => break Ok(()),
                    _ = tokio::time::sleep(Duration::from_millis(5)) => {}
                }
            }
        };
        self.service
            .shared
            .progress
            .send_modify(|p| p.stopped = true);
        result
    }

    async fn step(&mut self) -> Result<()> {
        let service = &self.service;
        let ingress = &service.ingress;
        let after = service.shared.progress.borrow().through;
        let replay_cut = *service.shared.replay_cut.lock().unwrap();
        if replay_cut.is_some_and(|cut| after >= cut) {
            return Ok(());
        }
        if self.pending.is_none() {
            service.barrier().await?;
            let (_, value) = reader::observe(
                ingress,
                ingress.group(0)?,
                query::Query::AcceptedBucket {
                    table: ingress.scope.table,
                    bucket: 0,
                    after,
                    limit: 1,
                },
            )
            .await?;
            let query::Value::Accepted(mut page) = value else {
                bail!("annotation source page unavailable");
            };
            if let Some(cut) = replay_cut {
                page.through = page.through.min(cut);
                page.entries.retain(|(index, _)| *index <= cut);
            }
            ensure!(page.through >= after, "annotation source cursor regressed");
            if let Some((index, meta)) = page.entries.into_iter().next() {
                ensure!(
                    index > after
                        && index <= page.through
                        && meta.stream == ingress.scope.table
                        && meta.bucket == 0
                        && meta.kind == RecordKind::Data,
                    "foreign annotation source"
                );
                slot(&meta, ingress.client.producer_slots())?;
                let bytes = usize::try_from(meta.body_bytes)?;
                ensure!(
                    bytes <= ingress.scope.contract.max_record_bytes as usize,
                    "oversized annotation WAL record"
                );
                let reservation = service
                    .apply_budget
                    .reserve(decode_workspace(
                        &ingress.scope.contract,
                        &service.limits,
                        bytes,
                    )?)
                    .await;
                let group = ingress.group(0)?;
                let response = ingress
                    .client
                    .request_routed(Request::Fetch { group, index })
                    .await?;
                ensure!(
                    matches!(response.reply, Reply::Record(Some(ref observed)) if observed == &meta),
                    "committed annotation payload unavailable"
                );
                meta.check_body_length(
                    response.body.len(),
                    ingress.scope.contract.max_record_bytes as usize,
                )?;
                let schema = ingress.schema.clone();
                let max_rows = ingress.scope.contract.max_rows;
                let value_bytes = service.limits.value_bytes;
                self.pending = Some(
                    tokio::task::spawn_blocking(move || -> Result<Pending> {
                        let mutations = wire::decode(response.body, schema, max_rows, value_bytes)?;
                        Ok(Pending {
                            index,
                            meta,
                            mutations,
                            _reservation: reservation,
                        })
                    })
                    .await??,
                );
            } else {
                if page.through > after {
                    // Control progress participates in the same freeze gate;
                    // otherwise values could freeze beyond the captured result cut.
                    let _ordered = service.shared.ordered.lock().await;
                    service
                        .shared
                        .staging
                        .advance_processed(after, page.through)
                        .await?;
                    service
                        .shared
                        .progress
                        .send_modify(|p| p.through = page.through);
                }
                return Ok(());
            }
        }
        let pending = self.pending.as_ref().unwrap();
        let _ordered = service.shared.ordered.lock().await;
        if !service.shared.window.lock().unwrap().admits(&pending.meta) {
            service
                .shared
                .progress
                .send_modify(|p| p.backpressured = true);
            return Ok(());
        }
        let observed = service.shared.staging.observation().await;
        let capacity = (
            observed.materialized,
            observed.retained_bytes,
            observed.retained_entries,
        );
        // This driver is the only applier. An unchanged base and retained
        // capacity cannot admit a batch that was already backpressured. Keep
        // the decoded request, but do not rebuild/discard its values every
        // poll while the publisher needs CPU to release the frozen generation.
        // Reader/old-handle release also changes these counters, independently
        // of publication, so it can unblock the same pending batch.
        if self.staging_pressure == Some(capacity) {
            return Ok(());
        }
        let mutations: Vec<_> = pending
            .mutations
            .iter()
            .map(|m| Mutation {
                turn: &m.turn,
                patch: &m.patch,
            })
            .collect();
        #[cfg(test)]
        service
            .shared
            .apply_attempts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let result = match service
            .shared
            .staging
            .apply_committed(after, pending.index, &mutations)
            .await?
        {
            ApplyOutcome::Applied => MutationResult::Applied,
            ApplyOutcome::Rejected(rejected) => MutationResult::Rejected(rejected),
            ApplyOutcome::Backpressure => {
                self.staging_pressure = Some(capacity);
                service
                    .shared
                    .progress
                    .send_modify(|p| p.backpressured = true);
                return Ok(());
            }
        };
        // No await between applying the batch, retaining its outcome, and
        // publishing progress to waiters. Cancellation cannot split this handoff.
        let pending = self.pending.take().unwrap();
        self.staging_pressure = None;
        let slot = slot(&pending.meta, ingress.client.producer_slots())?;
        service
            .shared
            .window
            .lock()
            .unwrap()
            .record(pending.index, pending.meta.clone());
        service.shared.results.lock().unwrap().latest[slot] = Some(Completed {
            index: pending.index,
            meta: pending.meta,
            result,
        });
        service.shared.progress.send_replace(Progress {
            through: pending.index,
            stopped: false,
            backpressured: false,
        });
        Ok(())
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        self.service
            .shared
            .progress
            .send_modify(|p| p.stopped = true);
    }
}

fn slot(meta: &BatchMeta, capacity: u32) -> Result<usize> {
    let bits = meta.producer.as_u128();
    let slot = (bits >> 64) as u32;
    let generation = bits as u64;
    ensure!(
        slot < capacity
            && generation > 0
            && producer::producer_id(slot, generation) == meta.producer,
        "annotation requires a bounded producer identity"
    );
    Ok(slot as usize)
}

fn decode_workspace(contract: &IngestContract, limits: &Limits, bytes: usize) -> Result<usize> {
    bytes
        .checked_mul(3)
        .and_then(|n| {
            n.checked_add(
                (contract.max_rows as usize).checked_mul(limits.value_bytes.checked_add(128)?)?,
            )
        })
        .context("annotation workspace overflow")
}
