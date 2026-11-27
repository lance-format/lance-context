//! Single-shard annotation apply and read barrier over the RF3 WAL.
//!
//! Native checkpoints restore values and bounded request outcomes together.
//! `runtime` composes the applier with continuous publication and recovery. The
//! application still supplies scheduling ownership and ingress routing.
use super::*;
use crate::{
    materializer::{observation::Phase, Storage},
    reader,
};
use lance::dataset::builder::DatasetBuilder;
use lance_context_api::annotation::TurnAnnotation;
use lance_context_core::annotation_staging::{
    AnnotationStaging, ApplyOutcome, Limits, Mutation, PatchLookup, RegistrationLookup, Rejection,
    Scope as StagingScope,
};
use lance_context_wal::replication::publication::lance::ManagedManifest;
use lance_context_wal::replication::source_read;
use std::{collections::VecDeque, future::Future, sync::Mutex};
use tokio::sync::watch;

mod accepted;
pub(crate) mod checkpoint;
mod local_payload;
pub mod lookup_capture;
mod lookup_observation;
#[cfg(feature = "service")]
pub(crate) mod metadata_upload;
pub mod mixed;
mod pipeline;
mod prefetch;
#[cfg(all(feature = "service", feature = "lance-io-metrics"))]
pub(crate) mod read_io;
pub mod runtime;
mod shard;
pub mod single_body;
#[cfg(feature = "http")]
mod status_filter;
mod window;
pub mod wire;
pub use checkpoint::{FrozenCheckpoint, PreparedCheckpoint};
pub use lookup_observation::{
    BodyLookupObservation, BodyPrefetchObservation, LookupMetricObservation, PrefetchStop,
};
pub use shard::Shard;
pub use window::SourceObservation;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationResult {
    Applied,
    Rejected(Rejection),
}

/// Retained from validated input for local timing only; never encoded in the WAL.
#[derive(Clone, Copy)]
pub(crate) enum WriteKind {
    Create,
    Patch,
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
    backpressure: Option<ApplyBackpressure>,
}

/// The admission gate that most recently prevented ordered WAL application.
/// This is a sampled state, not a refusal count or a measure of blocked time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplyBackpressure {
    ApplyMemory,
    SourceWindow,
    Staging,
}

struct Shared {
    narrow: Storage,
    body: Option<Storage>,
    local_payloads: local_payload::Registry,
    frames: Mutex<mixed::frames::Window>,
    #[cfg(test)]
    apply_attempts: std::sync::atomic::AtomicU64,
    staging: AnnotationStaging,
    results: Mutex<checkpoint::Results>,
    window: Mutex<window::Window>,
    source_generations: Arc<tokio::sync::Semaphore>,
    replay_cut: Mutex<Option<u64>>,
    ordered: tokio::sync::Mutex<()>,
    /// Test-only cap on the full-body key window, in workspace bytes (0: none).
    #[cfg(test)]
    key_window_for_test: std::sync::atomic::AtomicUsize,
    progress: watch::Sender<Progress>,
    reclamation: Mutex<runtime::reclamation::Observation>,
    body_lookups: Mutex<BodyLookupObservation>,
    lookup_capture: Arc<lookup_capture::State>,
    phases: Arc<crate::materializer::observation::Counters>,
}

/// Multiple independently connected ingresses may keep their own bounded view.
/// Every point read obtains a quorum barrier, then waits for its background
/// applier. Point reads never discover, fetch or decode WAL records themselves.
#[derive(Clone)]
pub struct Service {
    ingress: Ingress,
    shard: Option<Shard>,
    limits: Limits,
    shared: Arc<Shared>,
    apply_budget: Arc<MergeMemoryBudget>,
    lookup_budget: Option<Arc<MergeMemoryBudget>>,
    body_lookup_parallelism: Option<crate::turn_ingestion::TurnLookupParallelism>,
}

struct Pending {
    index: u64,
    meta: BatchMeta,
    decoded: mixed::Decoded,
    // Input address and complete allocation size, when known at reception.
    input_allocation: Option<(usize, usize)>,
    _reservation: MergeReservation,
    /// Decoded bodies were released after key extraction to widen a lookup
    /// window. The record is durable in the WAL; it is re-fetched and decoded
    /// again before it becomes the ordered head. Only keys stay resident.
    keys_only: bool,
}

impl Pending {
    /// Keep only keys. The decoded bodies and the input frame are released,
    /// but `Vec<Created>` (owned turn/routing strings and default annotation
    /// structs) stays resident. Those bytes are charged to the window's key
    /// workspace, not the frame budget: `lookup_workspace` charges at least
    /// 8x each key plus 512 bytes per key column, which covers the retained
    /// row (checked by `prefetch::prepare` against `retained_key_bytes`),
    /// and `KeyBound` caps a window's workspace at half the reserved lookup
    /// budget. Peak transient decode is still admitted by `fetch` one record
    /// at a time; the shared query takes its own lookup reservation.
    fn drop_body(&mut self) {
        if let mixed::Decoded::Creates { bodies, .. } = &mut self.decoded {
            *bodies = Vec::new();
        }
        self.input_allocation = None;
        self._reservation.shrink_to(0);
        self.keys_only = true;
    }

    /// Heap and inline bytes still held by a keys-only record.
    pub(super) fn retained_key_bytes(&self) -> usize {
        match &self.decoded {
            mixed::Decoded::Creates { rows, .. } => rows.iter().fold(
                rows.capacity()
                    .saturating_mul(std::mem::size_of::<mixed::Created>()),
                |total, row| {
                    total
                        .saturating_add(row.turn.capacity())
                        .saturating_add(row.rollout_id.capacity())
                },
            ),
            mixed::Decoded::Patches(_) => 0,
        }
    }
}

pub struct Driver {
    service: Service,
    pending: Option<Pending>,
    lookahead: VecDeque<Pending>,
    lookup: prefetch::Lookup,
    source_page: Option<lance_context_wal::replication::Page>,
    staging_pressure: Option<(u64, usize, usize)>,
    body_query: Option<pipeline::Query>,
    ahead: Option<pipeline::Ahead>,
    prefetched_lookup: bool,
    prefetched_reused: bool,
    body_batch_wait: Duration,
    body_batch_deadline: Option<tokio::time::Instant>,
    pressure_episode: Option<(ApplyBackpressure, crate::materializer::observation::Guard)>,
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
    pub fn bucket(&self) -> u32 {
        self.shard.map_or(0, |shard| shard.bucket)
    }

    fn group(&self) -> Result<u64> {
        self.ingress.group(self.bucket())
    }

    pub(crate) fn validate_turn(&self, turn: &str) -> Result<()> {
        let routing = self
            .ingress
            .turn_spec
            .as_ref()
            .map(|spec| spec.routing_key(turn))
            .transpose()?;
        if let Some(shard) = self.shard {
            shard.validate_turn(
                &self.ingress.layout,
                &self.ingress.scope.contract,
                routing.as_deref().unwrap_or(turn),
            )?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) async fn capture_read_for_test(
        &self,
        turn: &str,
    ) -> Result<lance_context_core::annotation_staging::PointRead> {
        // Pause a real admitted point read immediately before its storage I/O.
        let through = self.barrier().await?;
        self.wait_through(through).await?;
        Ok(self.shared.staging.capture(turn).await?)
    }

    #[cfg(test)]
    pub(crate) async fn capture_status_for_test(
        &self,
        status: Option<String>,
    ) -> Result<lance_context_core::annotation_staging::StatusRead> {
        let through = self.barrier().await?;
        self.wait_through(through).await?;
        Ok(self
            .shared
            .staging
            .capture_status(status, 16, 64 << 10)
            .await?)
    }

    /// Hold the ordered gate as a publisher install would.
    #[cfg(test)]
    pub(crate) async fn ordered_lock_for_test(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.shared.ordered.lock().await
    }

    #[cfg(test)]
    pub(crate) fn apply_attempts(&self) -> u64 {
        self.shared
            .apply_attempts
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn reclamation_observation(&self) -> runtime::reclamation::Observation {
        self.shared.reclamation.lock().unwrap().clone()
    }
    pub fn phase_observation(
        &self,
    ) -> [crate::materializer::observation::Observation;
           crate::materializer::observation::PHASE_COUNT] {
        self.shared.phases.observation()
    }
    #[cfg(feature = "http")]
    pub(crate) fn phase_counters(&self) -> &Arc<crate::materializer::observation::Counters> {
        &self.shared.phases
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) fn ingress(&self) -> &Ingress {
        &self.ingress
    }

    pub(crate) async fn check_owner(
        &self,
        authority: &lance_context_wal::replication::publication::Authority,
    ) -> Result<()> {
        self.shared
            .phases
            .measure(
                Phase::AnnotationOwnerFence,
                self.check_owner_observed(authority),
            )
            .await
            .inspect_err(|error| {
                // HTTP intentionally keeps its stable public error contract;
                // retain the underlying cause in the service log for diagnosis.
                eprintln!("annotation owner fence failed: {error:#}");
            })
    }

    async fn check_owner_observed(
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
        Self::validated_ingress_mode(
            client,
            layout,
            binding,
            limits,
            budget,
            apply_budget,
            false,
            None,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)] // Explicit routing augments the existing owned connection.
    async fn validated_ingress_mode(
        client: Arc<Client>,
        layout: Arc<Layout>,
        binding: TableBinding,
        limits: &Limits,
        budget: Arc<MergeMemoryBudget>,
        apply_budget: &Arc<MergeMemoryBudget>,
        mixed: bool,
        shard: Option<Shard>,
        spec: Option<&crate::turn_ingestion::TurnIngestionSpec>,
    ) -> Result<Ingress> {
        ensure!(
            shard.is_some() || layout.buckets.len() == 1,
            "annotation service requires an explicit shard"
        );
        if let Some(shard) = shard {
            layout.group_for_bucket(shard.bucket)?;
        }
        ensure!(
            spec.is_none() || shard.is_some(),
            "generic turns require an explicit shard"
        );
        let schema = if let Some(spec) = spec {
            if mixed {
                shard.unwrap().turn_schema(spec, limits)?
            } else {
                single_body::wire_schema(spec, limits, shard.unwrap())?
            }
        } else if let Some(shard) = shard {
            shard.wire_schema(limits, mixed)?
        } else if mixed {
            mixed::wire::schema(limits)
        } else {
            wire::schema(limits)
        };
        let contract = if let Some(spec) = spec {
            if mixed {
                shard.unwrap().turn_contract(&binding, spec, limits)?
            } else {
                single_body::contract(&binding, spec, limits, shard.unwrap())?
            }
        } else if let Some(shard) = shard {
            shard.contract(&binding, limits, mixed)?
        } else if mixed {
            Self::mixed_contract(&binding, limits)?
        } else {
            Self::contract(&binding, limits)?
        };
        ensure!(
            decode_workspace(&contract, limits, contract.max_record_bytes as usize)?
                <= budget.limit(),
            "annotation decode exceeds shared workspace"
        );
        ensure!(!Arc::ptr_eq(&budget, apply_budget)
            && decode_workspace(&contract, limits, contract.max_record_bytes as usize)? <= apply_budget.limit(),
            "annotation apply needs independent reserved workspace so waiting uploads cannot block it");
        if let Some(spec) = spec.filter(|_| !mixed) {
            let (frames, lookup) = single_body::budget_parts(apply_budget.limit())?;
            ensure!(decode_workspace(&contract, limits, contract.max_record_bytes as usize)? <= frames,
                "single body record exceeds frame workspace after reserving historical lookup capacity");
            ensure!(
                spec.lookup_workspace(std::iter::repeat_n(
                    limits.value_bytes,
                    contract.max_rows as usize
                ))? <= lookup,
                "single body maximum key batch exceeds reserved historical lookup capacity"
            );
        }
        Ingress::connect_with_schema(client, layout, binding, budget, contract, schema).await
    }

    /// Owned runtimes call this only after their publisher epoch is acquired.
    pub(super) async fn open(
        ingress: Ingress,
        storage: &Storage,
        limits: Limits,
        apply_budget: Arc<MergeMemoryBudget>,
    ) -> Result<(Self, Driver)> {
        Self::open_mode(ingress, storage, None, limits, apply_budget).await
    }

    async fn open_mode(
        ingress: Ingress,
        storage: &Storage,
        body: Option<Storage>,
        limits: Limits,
        apply_budget: Arc<MergeMemoryBudget>,
    ) -> Result<(Self, Driver)> {
        let shard = Shard::from_ingress(&ingress)?;
        let bucket = shard.map_or(0, |shard| shard.bucket);
        let (base, snapshot) = checkpoint::open(&ingress, storage, body.as_ref(), bucket).await?;
        let narrow_schema: arrow_schema::Schema = base.schema().into();
        let stored_spec = crate::turn_ingestion::TurnIngestionSpec::from_schema(&narrow_schema)?;
        let immutable = single_body::is_schema(&ingress.schema);
        // Read-ahead must not occupy the capacity needed to apply its own head.
        // These two budgets partition, rather than increase, configured apply memory.
        let (apply_budget, lookup_budget) = if immutable {
            let (frames, lookup) = single_body::budget_parts(apply_budget.limit())?;
            (
                MergeMemoryBudget::new(frames),
                Some(MergeMemoryBudget::new(lookup)),
            )
        } else {
            (apply_budget, None)
        };
        ensure!(
            stored_spec.as_ref().map(|s| s.fingerprint()).transpose()?
                == ingress
                    .turn_spec
                    .as_ref()
                    .map(|s| s.fingerprint())
                    .transpose()?,
            "selected narrow table has a foreign ingestion specification"
        );
        ensure!(
            body.is_some()
                == (ingress.scope.contract.policy == WritePolicy::TurnIngestion && !immutable),
            "body storage differs from the ingestion contract"
        );
        if let Some(storage) = &body {
            mixed::open_body(storage, &base, &ingress.scope.incarnation.to_string()).await?;
        }
        let through = snapshot.through;
        let constructor = if immutable {
            AnnotationStaging::new_immutable_body
        } else {
            AnnotationStaging::new
        };
        if immutable {
            ensure!(
                narrow_schema == *single_body::storage_schema(ingress.turn_spec.as_ref().unwrap())?,
                "foreign full body schema"
            );
        }
        let staging = constructor(
            base,
            StagingScope {
                table_instance: ingress.scope.incarnation.to_string(),
                source_group: ingress.group(bucket)?,
                shard: bucket,
            },
            through,
            limits.clone(),
        )?;
        let (progress, _) = watch::channel(Progress {
            through,
            stopped: false,
            backpressure: None,
        });
        let outcomes = vec![None; ingress.client.producer_slots() as usize];
        let window = window::Window::new(through, ingress.client.max_plan_source_bytes());
        let service = Self {
            ingress,
            shard,
            limits,
            apply_budget,
            lookup_budget,
            body_lookup_parallelism: None,
            shared: Arc::new(Shared {
                narrow: storage.clone(),
                body,
                local_payloads: local_payload::Registry::default(),
                frames: Mutex::new(mixed::frames::Window::default()),
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
                key_window_for_test: std::sync::atomic::AtomicUsize::new(0),
                #[cfg(test)]
                apply_attempts: std::sync::atomic::AtomicU64::new(0),
                progress,
                reclamation: Mutex::new(Default::default()),
                body_lookups: Mutex::new(Default::default()),
                lookup_capture: Arc::default(),
                phases: Arc::default(),
            }),
        };
        service.barrier().await?;
        let driver = Driver {
            service: service.clone(),
            pending: None,
            lookahead: VecDeque::new(),
            lookup: prefetch::Lookup::default(),
            source_page: None,
            staging_pressure: None,
            body_query: None,
            ahead: None,
            prefetched_lookup: false,
            prefetched_reused: false,
            body_batch_wait: Duration::ZERO,
            body_batch_deadline: None,
            pressure_episode: None,
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
            open.bucket == self.bucket(),
            "annotation producer belongs to another shard"
        );
        self.ingress.open_producer(open).await
    }

    /// Reserve before buffering an upload; preparation retains this allocation
    /// through its exact-byte retries. Narrow patch allocations are budgeted too.
    pub async fn reserve_upload(&self, bytes: usize) -> Result<Upload> {
        self.shared
            .phases
            .measure(
                Phase::AnnotationUploadAdmission,
                self.reserve_upload_observed(bytes),
            )
            .await
    }

    async fn reserve_upload_observed(&self, bytes: usize) -> Result<Upload> {
        let workspace = self.upload_workspace(bytes)?;
        Ok(Upload {
            scope: self.ingress.scope.clone(),
            bytes,
            reservation: self.ingress.budget.reserve(workspace).await,
        })
    }

    /// HTTP/2 streams must not hold shared receive credit while waiting for
    /// memory needed by another, already admitted upload on that connection.
    #[cfg(feature = "http")]
    pub(crate) async fn try_reserve_upload(&self, bytes: usize) -> Result<Option<Upload>> {
        self.shared
            .phases
            .measure(Phase::AnnotationUploadAdmission, async {
                let workspace = self.upload_workspace(bytes)?;
                Ok(self
                    .ingress
                    .budget
                    .try_reserve(workspace)
                    .map(|reservation| Upload {
                        scope: self.ingress.scope.clone(),
                        bytes,
                        reservation,
                    }))
            })
            .await
    }

    fn upload_workspace(&self, bytes: usize) -> Result<usize> {
        let contract = &self.ingress.scope.contract;
        ensure!(
            bytes > 0 && bytes <= contract.max_record_bytes as usize,
            "annotation upload too large"
        );
        decode_workspace(contract, &self.limits, bytes)
    }

    /// Prepare shared input without assuming its visible length is the backing
    /// allocation size. The caller must keep the backing within its upload
    /// allowance; use `prepare_owned` to release validation credit precisely.
    /// With a shared payload pool, unknown backing is copied once into an
    /// admitted allocation. `prepare_owned` transfers known backing without copying.
    pub async fn prepare(
        &self,
        upload: Upload,
        body: Bytes,
        producer: &Producer,
        sequence: u64,
    ) -> Result<PreparedAppend> {
        let pool = self.ingress.client.payload_memory_pool();
        let reservation = crate::payload_memory::reserve(pool.as_deref(), body.len())?;
        let (body, capacity) = if reservation.is_some() {
            // A borrowed Bytes can retain a much larger parent. Only copy when
            // shared admission requires ownership of the complete allocation.
            let owned = body.to_vec();
            drop(body);
            let capacity = owned.capacity();
            (
                crate::payload_memory::retain(owned.into(), capacity, reservation)?,
                Some(capacity),
            )
        } else {
            (body, None)
        };
        self.prepare_received(upload, body, capacity, producer, sequence)
            .await
    }

    /// Transfer a complete allocation without copying its payload. Capacity,
    /// including unused space, remains charged after validation.
    pub async fn prepare_owned(
        &self,
        upload: Upload,
        body: Vec<u8>,
        producer: &Producer,
        sequence: u64,
    ) -> Result<PreparedAppend> {
        let capacity = body.capacity();
        let pool = self.ingress.client.payload_memory_pool();
        let reservation = crate::payload_memory::reserve(pool.as_deref(), capacity)?;
        let body = crate::payload_memory::retain(body.into(), capacity, reservation)?;
        self.prepare_received(upload, body, Some(capacity), producer, sequence)
            .await
    }

    /// Only receive paths that own the complete allocation may supply capacity.
    /// A caller's Bytes length (or an Arrow buffer capacity) is not that proof.
    pub(crate) async fn prepare_received(
        &self,
        upload: Upload,
        body: Bytes,
        payload_capacity: Option<usize>,
        producer: &Producer,
        sequence: u64,
    ) -> Result<PreparedAppend> {
        self.shared
            .phases
            .measure(
                Phase::AnnotationUploadPrepare,
                self.prepare_observed(upload, body, payload_capacity, producer, sequence),
            )
            .await
    }

    async fn prepare_observed(
        &self,
        mut upload: Upload,
        body: Bytes,
        payload_capacity: Option<usize>,
        producer: &Producer,
        sequence: u64,
    ) -> Result<PreparedAppend> {
        ensure!(
            *upload.scope == *self.ingress.scope
                && *producer.scope == *self.ingress.scope
                && producer.bucket() == self.bucket()
                && sequence > 0
                && sequence < u64::MAX
                && upload.bytes == body.len(),
            "invalid annotation request identity or upload"
        );
        ensure!(
            payload_capacity.is_none_or(|capacity| capacity >= body.len()),
            "invalid received allocation size"
        );
        if let Some(capacity) = payload_capacity {
            let workspace = upload
                .reservation
                .bytes()
                .checked_add(capacity - body.len())
                .context("received allocation workspace overflow")?;
            ensure!(
                (capacity == body.len() || workspace <= self.ingress.budget.limit())
                    && upload.reservation.try_grow_to(workspace),
                "received allocation exceeds available upload workspace"
            );
        }
        let schema = self.ingress.schema.clone();
        let producer = producer.id();
        let value_bytes = self.limits.value_bytes;
        let mixed = self.ingress.scope.contract.policy == WritePolicy::TurnIngestion;
        let shard = self.shard;
        let layout = self.ingress.layout.clone();
        let bucket = self.bucket();
        let spec = self.ingress.turn_spec.clone();
        tokio::task::spawn_blocking(move || {
            let decoded = if single_body::is_schema(&schema) {
                single_body::decode(
                    body.clone(),
                    schema,
                    upload.scope.contract.max_rows,
                    value_bytes,
                )?
            } else if mixed {
                mixed::wire::decode(
                    body.clone(),
                    schema,
                    upload.scope.contract.max_rows,
                    value_bytes,
                )?
            } else {
                mixed::Decoded::Patches(wire::decode(
                    body.clone(),
                    schema,
                    upload.scope.contract.max_rows,
                    value_bytes,
                )?)
            };
            decoded.validate_shard(shard, &layout, &upload.scope.contract, spec.as_deref())?;
            let rows = decoded.rows() as u32;
            let annotation_kind = Some(match &decoded {
                mixed::Decoded::Creates { .. } => WriteKind::Create,
                mixed::Decoded::Patches(_) => WriteKind::Patch,
            });
            drop(decoded);
            // Validation owns decoded patches only inside this blocking task.
            // The prepared batch and local apply registration share these Bytes.
            // WAL transport and apply decoding have their own reservations, so
            // only a known complete payload allocation may release decode credit.
            // Unknown caller-owned Bytes can retain a larger parent allocation.
            let mut reservation = upload.reservation;
            if let Some(capacity) = payload_capacity {
                reservation.shrink_to(capacity);
            }
            Ok(PreparedAppend {
                batch: Batch::new(upload.scope.table, bucket, producer, sequence, body),
                rows,
                annotation_kind,
                payload_capacity,
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
        ensure!(
            *prepared.scope == *self.ingress.scope && prepared.meta().bucket == self.bucket(),
            "prepared annotation belongs to another shard"
        );
        let (append_phase, apply_phase) = match prepared
            .annotation_kind
            .context("prepared request is not an annotation write")?
        {
            WriteKind::Create => (
                Phase::AnnotationCreateAppend,
                Phase::AnnotationCreateApplyWait,
            ),
            WriteKind::Patch => (
                Phase::AnnotationPatchAppend,
                Phase::AnnotationPatchApplyWait,
            ),
        };
        let producer_slot = slot(prepared.meta(), self.ingress.client.producer_slots())?;
        // Share the upload only while its caller awaits apply. This registration
        // proves neither acceptance nor ordering; the driver still checks the
        // committed source metadata and closure, then reserves apply workspace.
        let _local = self.shared.local_payloads.register(
            producer_slot,
            &prepared.batch,
            prepared.payload_capacity,
        );
        tokio::time::timeout(deadline, async {
            let accepted = self
                .shared
                .phases
                .measure(
                    Phase::AnnotationAppend,
                    self.shared
                        .phases
                        .measure(append_phase, self.append_observed(prepared)),
                )
                .await?;
            self.shared
                .phases
                .measure(
                    Phase::AnnotationApplyWait,
                    self.shared
                        .phases
                        .measure(apply_phase, self.wait_through(accepted.position.index)),
                )
                .await?;
            let slot = slot(&accepted.meta, self.ingress.client.producer_slots())?;
            self.shared.results.lock().unwrap().resolve(&accepted, slot)
        })
        .await
        .context("annotation apply deadline expired; reconcile the original request")?
    }

    pub async fn get(&self, turn: &str, deadline: Duration) -> Result<Option<TurnAnnotation>> {
        self.validate_turn(turn)?;
        tokio::time::timeout(deadline, async {
            let through = self.barrier().await?;
            self.wait_through(through).await?;
            Ok(self.shared.staging.capture(turn).await?.resolve().await?)
        })
        .await
        .context("annotation read barrier deadline expired")?
    }

    /// The HTTP owner fences the result after this read. Its acquisition cut
    /// covers all predecessor writes; this owner's ACKs already await apply.
    /// A later unacknowledged proposal must not block reading applied values.
    #[cfg(feature = "http")]
    pub(crate) async fn get_owned(
        &self,
        turn: &str,
        recovery_cut: u64,
        deadline: Duration,
    ) -> Result<Option<TurnAnnotation>> {
        self.shared
            .phases
            .measure(
                Phase::AnnotationPointRead,
                self.get_owned_observed(turn, recovery_cut, deadline),
            )
            .await
    }

    #[cfg(feature = "http")]
    async fn get_owned_observed(
        &self,
        turn: &str,
        recovery_cut: u64,
        deadline: Duration,
    ) -> Result<Option<TurnAnnotation>> {
        self.validate_turn(turn)?;
        tokio::time::timeout(deadline, async {
            self.wait_through(recovery_cut).await?;
            // HTTP admission already bounds these requests. A transient lack
            // of staging readers must wait within the same request deadline,
            // without holding an old read view or blocking ordered apply.
            let view = self
                .shared
                .phases
                .measure(
                    Phase::AnnotationPointCapture,
                    self.shared.staging.capture_when_available(turn),
                )
                .await?;
            Ok(self
                .shared
                .phases
                .measure(Phase::AnnotationPointResolve, view.resolve())
                .await?)
        })
        .await
        .context("annotation owned read deadline expired")?
    }

    async fn barrier(&self) -> Result<u64> {
        let (cut, value) = reader::observe(
            &self.ingress,
            self.group()?,
            query::Query::SourceClosure {
                table: self.ingress.scope.table,
            },
        )
        .await?;
        let query::Value::SourceClosure(closed) = value else {
            bail!("source closure unavailable");
        };
        self.check_source_closure(closed.as_deref())?;
        Ok(cut.quorum_index)
    }

    fn check_source_closure(
        &self,
        closed: Option<&lance_context_wal::replication::source::Closed>,
    ) -> Result<()> {
        if let Some(closed) = closed {
            let closure = closed.permit.closure();
            ensure!(
                closure.table == self.ingress.scope.table
                    && closure.incarnation == self.ingress.scope.incarnation
                    && closure.group == self.group()?
                    && closure.through <= self.shared.results.lock().unwrap().base.through,
                "annotation view is behind source closure; reconnect to its durable checkpoint"
            );
        }
        Ok(())
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
        (state, self.apply_backpressure().is_some())
    }

    /// Opt-in diagnostics of the installed base's loaded fragment metadata.
    /// Does not list objects, read bodies or load native index coverage.
    pub async fn observation_with_manifest(
        &self,
    ) -> (lance_context_core::annotation_staging::Observation, bool) {
        let state = self.shared.staging.observation_with_manifest().await;
        (state, self.apply_backpressure().is_some())
    }

    /// Explicit native index metadata inspection on a protected installed base.
    /// May read metadata, but never scans body rows or creates indices. The
    /// returned version/cut can lag catalog selection; coverage alone does not
    /// establish that a particular query used an index effectively.
    pub async fn inspect_installed_indices(
        &self,
    ) -> Result<lance_context_core::annotation_staging::IndexInspection> {
        Ok(self.shared.staging.inspect_installed_indices().await?)
    }

    /// Returns the last gate that blocked apply, cleared by successful progress.
    /// Capacity release alone does not clear it until the applier runs again.
    pub fn apply_backpressure(&self) -> Option<ApplyBackpressure> {
        self.shared.progress.borrow().backpressure
    }
}

impl Driver {
    fn record_pressure(
        service: &Service,
        episode: &mut Option<(ApplyBackpressure, crate::materializer::observation::Guard)>,
        reason: ApplyBackpressure,
    ) {
        if episode
            .as_ref()
            .is_none_or(|(current, _)| *current != reason)
        {
            if let Some((_, timing)) = episode.take() {
                timing.complete();
            }
            let phase = match reason {
                ApplyBackpressure::ApplyMemory => Phase::AnnotationApplyMemoryPressure,
                ApplyBackpressure::SourceWindow => Phase::AnnotationSourceWindowPressure,
                ApplyBackpressure::Staging => Phase::AnnotationStagingPressure,
            };
            *episode = Some((reason, service.shared.phases.enter(phase)));
        }
        service
            .shared
            .progress
            .send_modify(|p| p.backpressure = Some(reason));
    }

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
                // Idle backoff still drives the retained successor lookup.
                tokio::select! {
                    _ = &mut shutdown => break Ok(()),
                    _ = pipeline::drive(
                        tokio::time::sleep(Duration::from_millis(5)),
                        &mut self.ahead,
                    ) => {}
                }
            }
        };
        self.service
            .shared
            .progress
            .send_modify(|p| p.stopped = true);
        result
    }

    #[cfg(test)]
    pub(crate) async fn step_for_test(&mut self) -> Result<()> {
        self.step().await
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
            if let Some(stub) = self.lookahead.front().filter(|p| p.keys_only) {
                // Rehydrate before popping: cancellation or admission pressure
                // leaves the keys-only record (and its lookup coverage) intact.
                let (index, meta) = (stub.index, stub.meta.clone());
                let mut full = prefetch::fetch(service, index, meta.clone()).await?;
                if full.is_none() {
                    // The ordered head has priority over speculation. A later
                    // speculative window (or resident bodies of this window's
                    // tail) can hold the frame capacity this head needs, and
                    // they cannot be published before the head applies:
                    // waiting would hold-and-wait. Release reconstructible
                    // state (the successor's bodies and query reservations;
                    // tail bodies, keeping their keys, validated metadata and
                    // this window's lookup) and retry once. Neither apply nor
                    // publication progress advances here.
                    self.ahead = None;
                    for record in self.lookahead.iter_mut().skip(1) {
                        if !record.keys_only {
                            record.drop_body();
                        }
                    }
                    full = prefetch::fetch(service, index, meta).await?;
                }
                let Some(full) = full else {
                    // Only applied-but-unpublished frames remain; the
                    // publisher can drain them.
                    Self::record_pressure(
                        service,
                        &mut self.pressure_episode,
                        ApplyBackpressure::ApplyMemory,
                    );
                    return Ok(());
                };
                {
                    let mut observed = service.shared.body_lookups.lock().unwrap();
                    observed.prefetch.body_refetches =
                        observed.prefetch.body_refetches.saturating_add(1);
                }
                self.lookahead[0] = full;
            }
            self.pending = self.lookahead.pop_front();
            if self.pending.is_none() {
                self.lookup = prefetch::Lookup::default();
                self.body_query = None;
                self.prefetched_lookup = false;
                self.prefetched_reused = false;
                if let Some(ahead) = self.ahead.as_mut() {
                    ensure!(
                        ahead.after == after,
                        "speculative window lost its ordered prefix"
                    );
                    if let Some(window) = ahead.ready(service, self.body_batch_wait).await? {
                        self.pending = Some(window.head);
                        self.lookahead = window.tail;
                        self.source_page = Some(window.page);
                        self.prefetched_lookup =
                            window.query.as_ref().is_some_and(|query| query.speculative);
                        self.body_query = window.query;
                    }
                    self.ahead = None;
                }
            }
        }
        if self.pending.is_none() {
            if self
                .source_page
                .as_ref()
                .is_some_and(|page| page.entries.is_empty() && page.through == after)
            {
                self.source_page = None;
            }
            if self.source_page.is_none() {
                let Some(page) = prefetch::source_page(service, after, replay_cut).await? else {
                    return Ok(());
                };
                // Retain bounded committed metadata. Payload read-ahead has
                // separate admission; a page is not application progress.
                self.source_page = Some(page);
            }
            let page = self.source_page.as_ref().unwrap();
            if let Some((index, meta)) = page.entries.first().cloned() {
                ensure!(
                    index > after
                        && index <= page.through
                        && meta.stream == ingress.scope.table
                        && meta.bucket == service.bucket()
                        && meta.kind == RecordKind::Data,
                    "foreign annotation source"
                );
                let Some(pending) = prefetch::fetch(service, index, meta).await? else {
                    Self::record_pressure(
                        service,
                        &mut self.pressure_episode,
                        ApplyBackpressure::ApplyMemory,
                    );
                    return Ok(());
                };
                self.pending = Some(pending);
                // Keep the entry until fetch/decode succeeds. Cancellation or
                // admission pressure must not skip the next committed record.
                self.source_page.as_mut().unwrap().entries.remove(0);
                if service.is_single_body() {
                    self.body_batch_deadline =
                        Some(tokio::time::Instant::now() + self.body_batch_wait);
                } else {
                    self.lookup = prefetch::prepare(
                        service,
                        self.pending.as_ref().unwrap(),
                        self.source_page.as_mut().unwrap(),
                        &mut self.lookahead,
                        replay_cut,
                    )
                    .await?;
                }
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
                self.source_page = None;
                return Ok(());
            }
        }
        let pending = self.pending.as_ref().unwrap();
        if let Some(deadline) = self.body_batch_deadline {
            self.lookup = prefetch::coalesce(
                service,
                pending,
                self.source_page.as_mut().unwrap(),
                &mut self.lookahead,
                replay_cut,
                deadline,
            )
            .await?;
            self.body_batch_deadline = None;
        }
        let missing_body_lookup = service.is_single_body()
            && match &pending.decoded {
                mixed::Decoded::Creates { rows, .. } => {
                    self.lookup.bodies.as_ref().is_none_or(|lookup| {
                        rows.iter().any(|row| lookup.keys.get(&row.turn).is_none())
                    })
                }
                _ => true,
            };
        if missing_body_lookup {
            self.lookup.bodies = None;
            if self.body_query.is_none() {
                self.body_query = pipeline::Query::new(service, pending, &self.lookahead, false)?;
            }
            if self.body_query.is_some()
                && self.ahead.is_none()
                && !prefetch::exceeds_key_window(service, pending)?
                && self.staging_pressure.is_none()
                && service.apply_backpressure().is_none()
                && service.shared.narrow.session.native_memory().is_some()
            {
                if let Some(page) = &self.source_page {
                    let through = self.lookahead.back().unwrap_or(pending).index;
                    self.ahead = Some(pipeline::Ahead::new(
                        service,
                        through,
                        page.clone(),
                        replay_cut,
                        self.body_batch_wait,
                    ));
                }
            }
            if let Some(query) = self.body_query.as_mut() {
                self.lookup.bodies = pipeline::resolve(query, &mut self.ahead).await?;
                self.body_query = None;
            }
            if self.lookup.bodies.is_none() {
                // Speculation must never prevent the ordered head or publisher
                // from obtaining capacity. The original source page is intact.
                self.ahead = None;
                self.prefetched_lookup = false;
            }
            if self.lookup.bodies.is_none() && !self.lookahead.is_empty() {
                {
                    let mut observed = service.shared.body_lookups.lock().unwrap();
                    observed.lookahead_admission_fallbacks =
                        observed.lookahead_admission_fallbacks.saturating_add(1);
                }
                // A head-only query must also release future decoded inputs.
                // Otherwise they can keep native admission full when no
                // applied prefix exists for the publisher to drain. Restore
                // their already validated metadata in order before dropping
                // bodies/reservations; neither the page cut nor ACK advances.
                let page = self
                    .source_page
                    .as_mut()
                    .context("lookahead lost its committed source page")?;
                while let Some(record) = self.lookahead.pop_back() {
                    page.entries.insert(0, (record.index, record.meta));
                }
                self.body_query = pipeline::Query::new(service, pending, &self.lookahead, false)?;
                if let Some(query) = self.body_query.as_mut() {
                    self.lookup.bodies = pipeline::resolve(query, &mut self.ahead).await?;
                    self.body_query = None;
                }
            }
            if self.lookup.bodies.is_none() {
                Self::record_pressure(
                    service,
                    &mut self.pressure_episode,
                    ApplyBackpressure::ApplyMemory,
                );
                return Ok(());
            }
        }
        pipeline::tick(&mut self.ahead).await;
        let ordered_wait = service.shared.phases.enter(Phase::AnnotationOrderedWait);
        // The successor keeps scanning while this window waits for the
        // ordered gate (e.g. behind publisher install).
        let _ordered = pipeline::drive(service.shared.ordered.lock(), &mut self.ahead).await;
        ordered_wait.complete();
        if !service.shared.window.lock().unwrap().admits(&pending.meta) {
            self.ahead = None;
            Self::record_pressure(
                service,
                &mut self.pressure_episode,
                ApplyBackpressure::SourceWindow,
            );
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
        #[cfg(test)]
        service
            .shared
            .apply_attempts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut full_body_inserted_rows = None;
        let applied = if service.is_single_body() {
            match service
                .shared
                .phases
                .measure(
                    Phase::AnnotationFullBodyApply,
                    // Keep the successor's key-only scan progressing while
                    // this window applies. It was captured against an older
                    // base; exactness still comes from apply-time staged and
                    // covered-publication validation (stale => fresh lookup).
                    pipeline::drive(
                        service.apply_bodies(after, pending, self.lookup.bodies.as_ref().unwrap()),
                        &mut self.ahead,
                    ),
                )
                .await?
            {
                Some((applied, inserted)) => {
                    full_body_inserted_rows = Some(inserted);
                    applied
                }
                None => {
                    if self.prefetched_lookup {
                        let mut observed = service.shared.body_lookups.lock().unwrap();
                        observed.prefetched_lookups_stale =
                            observed.prefetched_lookups_stale.saturating_add(1);
                    }
                    self.prefetched_lookup = false;
                    self.ahead = None;
                    self.lookup.bodies = None;
                    return Ok(());
                }
            }
        } else {
            match &pending.decoded {
                mixed::Decoded::Patches(rows) => {
                    let mutations = rows
                        .iter()
                        .map(|m| Mutation {
                            turn: &m.turn,
                            patch: &m.patch,
                        })
                        .collect::<Vec<_>>();
                    service
                        .shared
                        .phases
                        .measure(Phase::AnnotationPatch, async {
                            match &self.lookup.patches {
                                Some(lookup) => {
                                    service
                                        .shared
                                        .staging
                                        .apply_committed_with_lookup(
                                            after,
                                            pending.index,
                                            &mutations,
                                            lookup,
                                        )
                                        .await
                                }
                                _ => {
                                    service
                                        .shared
                                        .staging
                                        .apply_committed(after, pending.index, &mutations)
                                        .await
                                }
                            }
                        })
                        .await?
                }
                mixed::Decoded::Creates { rows, .. } => {
                    let registrations = rows
                        .iter()
                        .map(|r| lance_context_core::annotation_staging::Registration {
                            turn: &r.turn,
                            rollout_id: &r.rollout_id,
                            annotation: &r.annotation,
                        })
                        .collect::<Vec<_>>();
                    service
                        .shared
                        .phases
                        .measure(Phase::AnnotationCreate, async {
                            match &self.lookup.registrations {
                                Some(lookup) => {
                                    service
                                        .shared
                                        .staging
                                        .register_committed_with_lookup(
                                            after,
                                            pending.index,
                                            &registrations,
                                            lookup,
                                        )
                                        .await
                                }
                                _ => {
                                    service
                                        .shared
                                        .staging
                                        .register_committed(after, pending.index, &registrations)
                                        .await
                                }
                            }
                        })
                        .await?
                }
            }
        };
        let result = match applied {
            ApplyOutcome::Applied => MutationResult::Applied,
            ApplyOutcome::Rejected(rejected) => MutationResult::Rejected(rejected),
            ApplyOutcome::Backpressure => {
                self.ahead = None;
                self.staging_pressure = Some(capacity);
                Self::record_pressure(
                    service,
                    &mut self.pressure_episode,
                    ApplyBackpressure::Staging,
                );
                return Ok(());
            }
        };
        // No await between applying the batch, retaining its outcome, and
        // publishing progress to waiters. Cancellation cannot split this handoff.
        service
            .shared
            .phases
            .blocking(Phase::AnnotationApplyHandoff, || {
                if self.prefetched_lookup && !self.prefetched_reused {
                    self.prefetched_reused = true;
                    let mut observed = service.shared.body_lookups.lock().unwrap();
                    observed.prefetched_lookups_reused =
                        observed.prefetched_lookups_reused.saturating_add(1);
                }
                let pending = self.pending.take().unwrap();
                if let mixed::Decoded::Creates { bodies, rows } = pending.decoded {
                    let input_rows = rows.len() as u64;
                    // Release decoded narrow rows before returning transient decode
                    // credit. The retained body frame keeps every shared IPC allocation.
                    drop(rows);
                    if result == MutationResult::Applied {
                        let retained = service.shared.frames.lock().unwrap().record_with_spec(
                            pending.index,
                            bodies,
                            pending._reservation,
                            pending.meta.body_bytes as usize,
                            pending.input_allocation,
                            ingress.turn_spec.clone(),
                        );
                        if full_body_inserted_rows == Some(0) {
                            let mut observed = service.shared.body_lookups.lock().unwrap();
                            observed.all_duplicate_batches =
                                observed.all_duplicate_batches.saturating_add(1);
                            observed.all_duplicate_rows =
                                observed.all_duplicate_rows.saturating_add(input_rows);
                            observed.all_duplicate_reservation_bytes = observed
                                .all_duplicate_reservation_bytes
                                .saturating_add(retained.reservation_bytes as u64);
                            if let Some(bytes) = retained.arrow_backing_bytes {
                                observed.all_duplicate_known_arrow_batches =
                                    observed.all_duplicate_known_arrow_batches.saturating_add(1);
                                observed.all_duplicate_arrow_backing_bytes = observed
                                    .all_duplicate_arrow_backing_bytes
                                    .saturating_add(bytes as u64);
                            }
                        }
                    }
                }
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
                    backpressure: None,
                });
                if let Some((_, timing)) = self.pressure_episode.take() {
                    timing.complete();
                }
                Ok(())
            })
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
