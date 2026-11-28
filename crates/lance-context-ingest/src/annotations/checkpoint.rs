//! Bounded application results bound to the same native transaction as values.
use super::*;
use crate::{annotation_publication, materializer};
use lance::Dataset;
use lance_context_core::annotation_staging::{
    FrozenAnnotations, PreparedAnnotations, RejectionReason, MAX_RESULTS_BYTES, RESULTS_PROPERTY,
};
use lance_context_wal::replication::publication as p;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Encoded {
    version: u8,
    slots: u32,
    // Version 1: sorted slot, latest rejected index, rejected row, rejection
    // code. No producer identity and no earliest prefix rejection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rejected: Vec<(u32, u64, u32, u8)>,
    // Version 2: base64 of a slot presence bitmap followed by one packed
    // record per present slot (see `pack`). Carries both the latest exact
    // rejection and the earliest unresolved rejection of the slot's producer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    outcomes: Option<String>,
}

/// Persisted outcome evidence for one producer slot.
#[derive(Clone, Default)]
pub(crate) struct SlotOutcome {
    /// Producer generation that owns `sticky` (and produced `latest`).
    generation: u64,
    /// Latest exact rejection, as before version 2.
    latest: Option<(u64, Rejection)>,
    /// Earliest unresolved rejection of this producer generation. Never
    /// overwritten by a later result; kept apart from `latest`.
    sticky: Option<(u64, Rejection)>,
}

pub(crate) struct Snapshot {
    pub through: u64,
    slots: u32,
    /// Sorted by slot; only slots with some rejection evidence.
    outcomes: Vec<(u32, SlotOutcome)>,
}

fn reason_code(reason: &RejectionReason) -> u8 {
    match reason {
        RejectionReason::InvalidPatch => 0,
        RejectionReason::InvalidTurn => 1,
        RejectionReason::MissingTurn => 2,
        RejectionReason::ValueTooLarge => 3,
        RejectionReason::BatchTooLarge => 4,
    }
}

fn code_reason(code: u8) -> Result<RejectionReason> {
    Ok(match code {
        0 => RejectionReason::InvalidPatch,
        1 => RejectionReason::InvalidTurn,
        2 => RejectionReason::MissingTurn,
        3 => RejectionReason::ValueTooLarge,
        4 => RejectionReason::BatchTooLarge,
        _ => bail!("invalid annotation rejection code"),
    })
}

// Record layout: header u8 (bits 0-2 sticky code, 3-5 latest code, 6 has
// latest, 7 has sticky), generation u64, then [index u64, row u16] for sticky
// and for latest when present. At most 29 bytes; with the 2 KiB bitmap all
// 16,384 slots encode to under 637 KB of base64, within MAX_RESULTS_BYTES.
const RECORD_MAX: usize = 29;

impl Snapshot {
    fn empty(slots: u32) -> Self {
        Self {
            through: 0,
            slots,
            outcomes: Vec::new(),
        }
    }

    pub(crate) fn decode(value: &str, through: u64) -> Result<Self> {
        ensure!(
            value.len() <= MAX_RESULTS_BYTES && through > 0 && through < u64::MAX,
            "invalid annotation result snapshot size/cut"
        );
        let encoded: Encoded = serde_json::from_str(value)?;
        ensure!(
            (1..=producer::MAX_SLOTS).contains(&encoded.slots),
            "invalid annotation result snapshot"
        );
        let index_ok = |index: u64| index > 0 && index <= through;
        let mut outcomes = Vec::new();
        match encoded.version {
            1 => {
                ensure!(
                    encoded.outcomes.is_none()
                        && encoded.rejected.len() <= encoded.slots as usize
                        && encoded.rejected.windows(2).all(|w| w[0].0 < w[1].0)
                        && encoded.rejected.iter().all(|&(slot, index, row, _)| {
                            slot < encoded.slots && index_ok(index) && row < 65_536
                        }),
                    "invalid annotation result snapshot"
                );
                for &(slot, index, row, code) in &encoded.rejected {
                    outcomes.push((
                        slot,
                        SlotOutcome {
                            generation: 0,
                            latest: Some((
                                index,
                                Rejection {
                                    row: row as usize,
                                    reason: code_reason(code)?,
                                },
                            )),
                            sticky: None,
                        },
                    ));
                }
            }
            2 => {
                use base64::{engine::general_purpose::STANDARD, Engine as _};
                ensure!(encoded.rejected.is_empty(), "invalid annotation result snapshot");
                let bytes = STANDARD.decode(
                    encoded
                        .outcomes
                        .as_deref()
                        .context("annotation result snapshot has no outcomes")?,
                )?;
                let bitmap = (encoded.slots as usize).div_ceil(8);
                ensure!(bytes.len() >= bitmap, "truncated annotation result snapshot");
                let (bits, mut rest) = bytes.split_at(bitmap);
                fn take<'a>(rest: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
                    ensure!(rest.len() >= n, "truncated annotation result snapshot");
                    let (head, tail) = rest.split_at(n);
                    *rest = tail;
                    Ok(head)
                }
                for slot in 0..encoded.slots {
                    if bits[slot as usize / 8] & (1 << (slot % 8)) == 0 {
                        continue;
                    }
                    let header = take(&mut rest, 1)?[0];
                    let generation = u64::from_le_bytes(take(&mut rest, 8)?.try_into()?);
                    let mut entry = |code: u8| -> Result<(u64, Rejection)> {
                        let index = u64::from_le_bytes(take(&mut rest, 8)?.try_into()?);
                        let row = u16::from_le_bytes(take(&mut rest, 2)?.try_into()?);
                        ensure!(index_ok(index), "invalid annotation result index");
                        Ok((
                            index,
                            Rejection {
                                row: row as usize,
                                reason: code_reason(code)?,
                            },
                        ))
                    };
                    let sticky = (header & 0x80 != 0)
                        .then(|| entry(header & 7))
                        .transpose()?;
                    let latest = (header & 0x40 != 0)
                        .then(|| entry((header >> 3) & 7))
                        .transpose()?;
                    ensure!(
                        (sticky.is_some() || latest.is_some())
                            && (sticky.is_none() || generation > 0),
                        "invalid annotation slot outcome"
                    );
                    outcomes.push((
                        slot,
                        SlotOutcome {
                            generation,
                            latest,
                            sticky,
                        },
                    ));
                }
                ensure!(
                    rest.is_empty(),
                    "trailing annotation result snapshot bytes"
                );
                // Bitmap padding bits beyond `slots` must be clear.
                for slot in encoded.slots..(bitmap as u32 * 8) {
                    ensure!(
                        bits[slot as usize / 8] & (1 << (slot % 8)) == 0,
                        "invalid annotation result bitmap"
                    );
                }
            }
            _ => bail!("unsupported annotation result snapshot version"),
        }
        Ok(Self {
            through,
            slots: encoded.slots,
            outcomes,
        })
    }

    pub(crate) fn encode(&self) -> Result<String> {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let bitmap = (self.slots as usize).div_ceil(8);
        let mut bytes = vec![0u8; bitmap];
        bytes.reserve(self.outcomes.len() * RECORD_MAX);
        for (slot, outcome) in &self.outcomes {
            ensure!(*slot < self.slots, "result slot outside capacity");
            bytes[*slot as usize / 8] |= 1 << (slot % 8);
            let code = |e: &Option<(u64, Rejection)>| e.as_ref().map_or(0, |(_, r)| reason_code(&r.reason));
            let header = code(&outcome.sticky)
                | (code(&outcome.latest) << 3)
                | ((outcome.latest.is_some() as u8) << 6)
                | ((outcome.sticky.is_some() as u8) << 7);
            bytes.push(header);
            bytes.extend_from_slice(&outcome.generation.to_le_bytes());
            for (index, rejection) in outcome.sticky.iter().chain(outcome.latest.iter()) {
                bytes.extend_from_slice(&index.to_le_bytes());
                bytes.extend_from_slice(&u16::try_from(rejection.row)?.to_le_bytes());
            }
        }
        Ok(serde_json::to_string(&Encoded {
            version: 2,
            slots: self.slots,
            rejected: Vec::new(),
            outcomes: Some(STANDARD.encode(bytes)),
        })?)
    }

    fn outcome(&self, slot: usize) -> Option<&SlotOutcome> {
        let row = self
            .outcomes
            .binary_search_by_key(&(slot as u32), |e| e.0)
            .ok()?;
        Some(&self.outcomes[row].1)
    }

    fn rejected(&self, slot: usize) -> Option<(u64, Rejection)> {
        self.outcome(slot)?.latest.clone()
    }

    /// Earliest unresolved rejection persisted for `slot`, with its producer.
    pub(crate) fn sticky(&self, slot: usize) -> Option<(Uuid, u64, Rejection)> {
        let outcome = self.outcome(slot)?;
        let (index, rejection) = outcome.sticky.clone()?;
        Some((
            producer::producer_id(slot as u32, outcome.generation),
            index,
            rejection,
        ))
    }

    pub(crate) fn slots(&self) -> u32 {
        self.slots
    }
}

pub(super) struct Results {
    pub base: Arc<Snapshot>,
    pub latest: Vec<Option<Completed>>,
    /// Earliest rejection per slot for the slot's current producer identity.
    /// One entry per bounded producer slot. A commit-ack producer can stream
    /// later requests before learning an outcome, so a later Applied must not
    /// overwrite (in memory or in the published result snapshot) the
    /// rejection that the producer has not yet observed.
    pub sticky: Vec<Option<(Uuid, u64, Rejection)>>,
}

/// An earlier request from the same producer was rejected. The caller must
/// treat its source prefix as rejected; the exact earlier outcome is returned
/// by waiting on that earlier receipt.
#[derive(Debug)]
pub struct PriorRejected {
    pub index: u64,
    pub rejection: Rejection,
}
impl std::fmt::Display for PriorRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "earlier request at WAL index {} from this producer was rejected: {:?}",
            self.index, self.rejection
        )
    }
}
impl std::error::Error for PriorRejected {}

/// The receipt's producer generation is older than the generation that now
/// owns its slot. The slot's state describes another producer, so the
/// receipt's outcome can no longer be answered; it is never reported Applied.
#[derive(Debug)]
pub struct ReceiptSuperseded {
    pub receipt_generation: u64,
    pub slot_generation: u64,
}
impl std::fmt::Display for ReceiptSuperseded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "receipt producer generation {} superseded by slot generation {}",
            self.receipt_generation, self.slot_generation
        )
    }
}
impl std::error::Error for ReceiptSuperseded {}

impl Results {
    pub fn record_sticky(
        &mut self,
        slot: usize,
        index: u64,
        meta: &BatchMeta,
        result: &MutationResult,
    ) {
        let entry = &mut self.sticky[slot];
        if entry.as_ref().is_some_and(|(producer, _, _)| *producer != meta.producer) {
            *entry = None;
        }
        if let MutationResult::Rejected(rejection) = result {
            if entry.is_none() {
                *entry = Some((meta.producer, index, rejection.clone()));
            }
        }
    }

    /// Commit-ack resolution after `wait_through(index)`. The exact latest
    /// outcome and the same-producer earliest rejection are checked before
    /// falling back to the published snapshot, as `resolve` does.
    pub fn resolve_streamed(&self, accepted: &AcceptedAppend, slot: usize) -> Result<Completed> {
        self.check_generation(accepted, slot)?;
        if let Some((producer, index, rejection)) = &self.sticky[slot] {
            if *producer == accepted.meta.producer {
                if *index == accepted.position.index {
                    ensure!(
                        rejection.row < accepted.rows as usize,
                        "rejection row is outside the original request"
                    );
                    return Ok(Completed {
                        index: *index,
                        meta: accepted.meta.clone(),
                        result: MutationResult::Rejected(rejection.clone()),
                    });
                }
                if *index < accepted.position.index {
                    return Err(anyhow::Error::new(PriorRejected {
                        index: *index,
                        rejection: rejection.clone(),
                    }));
                }
            }
        }
        if let Some(result) = &self.latest[slot] {
            if result.meta.producer == accepted.meta.producer
                && result.index > accepted.position.index
                && result.index > self.base.through
            {
                // Applied in order after this producer's request with no
                // sticky rejection at or before it: this request applied.
                return Ok(Completed {
                    index: accepted.position.index,
                    meta: accepted.meta.clone(),
                    result: MutationResult::Applied,
                });
            }
        }
        self.resolve(accepted, slot)
    }


    /// Newest producer generation observed for `slot` (live latest, sticky,
    /// or the published snapshot).
    fn slot_generation(&self, slot: usize) -> u64 {
        let live = self.latest[slot]
            .as_ref()
            .map_or(0, |r| r.meta.producer.as_u128() as u64);
        let sticky = self.sticky[slot]
            .as_ref()
            .map_or(0, |(p, _, _)| p.as_u128() as u64);
        let base = self.base.outcome(slot).map_or(0, |o| o.generation);
        live.max(sticky).max(base)
    }

    /// Refuse a receipt whose producer generation is older than the slot's
    /// current generation: its state now belongs to a reused slot.
    pub fn check_generation(&self, accepted: &AcceptedAppend, slot: usize) -> Result<()> {
        let receipt_generation = accepted.meta.producer.as_u128() as u64;
        let slot_generation = self.slot_generation(slot);
        if receipt_generation < slot_generation {
            return Err(anyhow::Error::new(ReceiptSuperseded {
                receipt_generation,
                slot_generation,
            }));
        }
        Ok(())
    }

    pub fn resolve(&self, accepted: &AcceptedAppend, slot: usize) -> Result<Completed> {
        self.check_generation(accepted, slot)?;
        if let Some(result) = &self.latest[slot] {
            if result.index == accepted.position.index && result.meta == accepted.meta {
                return Ok(result.clone());
            }
        }
        // accepted came from a successful exact WAL append/retry, never a bare
        // caller position. The proven cut covers every application's outcome.
        ensure!(
            accepted.position.index <= self.base.through,
            "annotation request outcome expired; request must not be re-executed"
        );
        // The latest exact rejection and the earliest sticky rejection are
        // separate persisted fields; either may be this exact request.
        let exact = self
            .base
            .rejected(slot)
            .filter(|(index, _)| *index == accepted.position.index)
            .or_else(|| {
                self.base
                    .sticky(slot)
                    .filter(|(producer, index, _)| {
                        *producer == accepted.meta.producer && *index == accepted.position.index
                    })
                    .map(|(_, index, rejection)| (index, rejection))
            });
        let result = match exact {
            Some((_, rejected)) => {
                ensure!(
                    rejected.row < accepted.rows as usize,
                    "rejection row is outside the original request"
                );
                MutationResult::Rejected(rejected)
            }
            None => MutationResult::Applied,
        };
        Ok(Completed {
            index: accepted.position.index,
            meta: accepted.meta.clone(),
            result,
        })
    }

    fn snapshot(&self, through: u64) -> Result<Snapshot> {
        ensure!(through >= self.base.through, "result cut regressed");
        let mut outcomes = Vec::new();
        for (slot, latest) in self.latest.iter().enumerate() {
            // Latest exact outcome: a live result past the base, else the base.
            let (latest_rejection, latest_generation) =
                match latest.as_ref().filter(|r| r.index > self.base.through) {
                    Some(result) => {
                        ensure!(result.index <= through, "result cut precedes application");
                        let rejection = match &result.result {
                            MutationResult::Applied => None,
                            MutationResult::Rejected(r) => Some((result.index, r.clone())),
                        };
                        (rejection, Some(result.meta.producer.as_u128() as u64))
                    }
                    None => (
                        self.base.rejected(slot),
                        self.base.outcome(slot).map(|o| o.generation),
                    ),
                };
            // Earliest unresolved rejection, kept separately so it never
            // replaces the latest exact outcome. `record_sticky` clears it when
            // another producer generation takes the slot.
            let sticky = self.sticky[slot]
                .as_ref()
                .filter(|(_, index, _)| *index <= through);
            if latest_rejection.is_none() && sticky.is_none() {
                continue;
            }
            let generation = match sticky {
                Some((producer, _, _)) => producer.as_u128() as u64,
                None => latest_generation.unwrap_or(0),
            };
            outcomes.push((
                slot as u32,
                SlotOutcome {
                    generation,
                    latest: latest_rejection,
                    sticky: sticky.map(|(_, index, r)| (*index, r.clone())),
                },
            ));
        }
        Ok(Snapshot {
            through,
            slots: self.latest.len() as u32,
            outcomes,
        })
    }
}

pub struct FrozenCheckpoint {
    pub(super) origin: Arc<Shared>,
    pub(super) annotations: FrozenAnnotations,
    results: Arc<Snapshot>,
    sources: lance_context_wal::replication::publication::SourceRange,
    pub(super) bodies: mixed::frames::Frozen,
    /// Keys-only survivors of an owner batch, bound to this exact cut.
    pub(super) write_batch: Option<Arc<owner_batch::Frozen>>,
    /// The owning service, retained only to refetch owner batch bodies.
    owner: Option<Service>,
    pub(super) body_preparation:
        tokio::sync::Mutex<Option<lance_context_core::annotation_staging::BodySnapshot>>,
    _source_generation: tokio::sync::OwnedSemaphorePermit,
}
pub struct PreparedCheckpoint {
    origin: Arc<Shared>,
    annotations: PreparedAnnotations,
    results: Arc<Snapshot>,
    bodies: mixed::frames::Frozen,
    write_batch: Option<Arc<owner_batch::Frozen>>,
}
impl FrozenCheckpoint {
    pub async fn prepare_full_body(
        &self,
        workspace: &Arc<MergeMemoryBudget>,
    ) -> Result<PreparedCheckpoint> {
        if let Some(batch) = &self.write_batch {
            return self.prepare_owner_batch(workspace, batch.clone()).await;
        }
        let mut selection = std::collections::BTreeMap::<u64, Vec<(u32, &str)>>::new();
        for (key, source) in self.annotations.new_turns() {
            selection
                .entry(source.record_index)
                .or_default()
                .push((source.row, key));
        }
        let mut batches = Vec::new();
        for (index, mut rows) in selection {
            rows.sort_unstable_by_key(|row| row.0);
            batches.extend(
                self.bodies
                    .get(&index)
                    .context("missing frozen body frame")?
                    .select(&rows)?,
            );
        }
        Ok(PreparedCheckpoint {
            origin: self.origin.clone(),
            annotations: self
                .annotations
                .prepare_immutable_body(workspace, batches, self.bodies.clone())
                .await
                .map_err(runtime::publication_write_error)?,
            results: self.results.clone(),
            bodies: self.bodies.clone(),
            write_batch: None,
        })
    }

    /// Write stage: stream the frozen survivors back from the WAL, cluster
    /// each chunk by (session_id, id) and write them as ONE fragment.
    async fn prepare_owner_batch(
        &self,
        workspace: &Arc<MergeMemoryBudget>,
        batch: Arc<owner_batch::Frozen>,
    ) -> Result<PreparedCheckpoint> {
        let started = std::time::Instant::now();
        let mut expected = std::collections::BTreeMap::<u64, u32>::new();
        for (_, source) in self.annotations.new_turns() {
            *expected.entry(source.record_index).or_default() += 1;
        }
        let rows = expected.values().map(|n| *n as usize).sum::<usize>();
        ensure!(
            rows as u64 == batch.rows(),
            "owner write batch differs from frozen first-create rows"
        );
        let service = self.owner.as_ref().context("owner batch lost its service")?;
        let stream = service.survivor_stream(batch.clone(), expected)?;
        let annotations = self
            .annotations
            .prepare_immutable_body_stream(workspace, rows, stream, batch.clone())
            .await
            .map_err(runtime::publication_write_error)?;
        self.origin
            .batch_metrics
            .lock()
            .unwrap()
            .fragment(rows as u64, started.elapsed().as_micros().try_into().unwrap_or(u64::MAX));
        Ok(PreparedCheckpoint {
            origin: self.origin.clone(),
            annotations,
            results: self.results.clone(),
            bodies: self.bodies.clone(),
            write_batch: Some(batch),
        })
    }

    /// Complete source metadata captured under the same gate as values/results.
    /// The publisher still pins this selection before recording preparation.
    pub fn sources(&self) -> &lance_context_wal::replication::publication::SourceRange {
        &self.sources
    }
    pub fn through(&self) -> u64 {
        self.annotations.through()
    }
    pub fn identity(&self) -> &str {
        self.annotations.identity()
    }
    pub async fn prepare(
        &self,
        workspace: &Arc<MergeMemoryBudget>,
        batch_rows: usize,
    ) -> Result<PreparedCheckpoint> {
        ensure!(
            self.origin.body.is_none(),
            "mixed checkpoint requires an explicit body preparation"
        );
        Ok(PreparedCheckpoint {
            origin: self.origin.clone(),
            annotations: self.annotations.prepare(workspace, batch_rows).await?,
            results: self.results.clone(),
            bodies: self.bodies.clone(),
            write_batch: None,
        })
    }

    pub(super) async fn prepare_with_body(
        &self,
        workspace: &Arc<MergeMemoryBudget>,
        batch_rows: usize,
        body: lance_context_core::annotation_staging::BodySnapshot,
    ) -> Result<PreparedCheckpoint> {
        ensure!(
            self.origin.body.as_ref().is_some_and(|s| s.uri == body.uri),
            "body preparation differs from the authorized storage"
        );
        Ok(PreparedCheckpoint {
            origin: self.origin.clone(),
            annotations: self
                .annotations
                .prepare_with_body(workspace, batch_rows, body)
                .await?,
            results: self.results.clone(),
            bodies: self.bodies.clone(),
            write_batch: None,
        })
    }
}
impl PreparedCheckpoint {
    pub fn annotations(&self) -> &PreparedAnnotations {
        &self.annotations
    }
}

impl Service {
    /// Capture results and values before another committed batch can apply.
    /// The caller must select/pin exactly this complete source prefix. This
    /// method does not authorize publication, confirmation or source closure.
    pub async fn freeze(&self, transaction: Uuid) -> Result<Option<FrozenCheckpoint>> {
        let ordered_wait = self.shared.phases.enter(Phase::AnnotationOrderedWait);
        let _ordered = self.shared.ordered.lock().await;
        ordered_wait.complete();
        self.shared
            .phases
            .measure(
                Phase::AnnotationFreezeHold,
                self.freeze_ordered(transaction),
            )
            .await
    }

    async fn freeze_ordered(&self, transaction: Uuid) -> Result<Option<FrozenCheckpoint>> {
        let through = self.shared.progress.borrow().through;
        ensure!(
            self.shared
                .replay_cut
                .lock()
                .unwrap()
                .is_none_or(|cut| cut == through),
            "pending publication must freeze at its original cut"
        );
        let snapshot = {
            let results = self.shared.results.lock().unwrap();
            if !results
                .latest
                .iter()
                .flatten()
                .any(|r| r.index > results.base.through)
            {
                return Ok(None);
            }
            results.snapshot(through)?
        };
        let encoded = snapshot.encode()?;
        Snapshot::decode(&encoded, through)?;
        // Old public handles may outlive installation. Bound their retained
        // source vectors too, independently of the value/result allocations.
        let source_generation = self
            .shared
            .source_generations
            .clone()
            .try_acquire_owned()
            .context("annotation source metadata retained by older checkpoints")?;
        let annotations = self
            .shared
            .staging
            .freeze_with_results(transaction, Some(encoded))
            .await?
            .context("application results have no frozen generation")?;
        ensure!(
            annotations.through() == through,
            "value/result freeze cut differs"
        );
        // Take each guard in its own statement: owner_batch() locks the same
        // non-reentrant write_batch mutex, so it must not run while a guard
        // from a struct field initializer is still alive.
        let sources = self.shared.window.lock().unwrap().freeze(through);
        let bodies = self.shared.frames.lock().unwrap().freeze();
        let write_batch = self
            .shared
            .write_batch
            .lock()
            .unwrap()
            .as_mut()
            .map(owner_batch::WriteBatch::freeze);
        let owner = self.owner_batch().then(|| self.clone());
        let frozen = FrozenCheckpoint {
            origin: self.shared.clone(),
            annotations,
            results: Arc::new(snapshot),
            sources,
            bodies,
            write_batch,
            owner,
            body_preparation: tokio::sync::Mutex::new(None),
            _source_generation: source_generation,
        };
        *self.shared.replay_cut.lock().unwrap() = None;
        Ok(Some(frozen))
    }

    /// Install only after the caller resolves the original managed publication.
    /// Preserve later outcomes and values; retire only this frozen generation.
    pub async fn install_published(
        &self,
        prepared: &PreparedCheckpoint,
        selected: Arc<Dataset>,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.shared, &prepared.origin),
            "foreign application checkpoint"
        );
        let ordered_wait = self.shared.phases.enter(Phase::AnnotationOrderedWait);
        let _ordered = self.shared.ordered.lock().await;
        ordered_wait.complete();
        self.shared
            .phases
            .measure(
                Phase::AnnotationInstallHold,
                self.install_published_ordered(prepared, selected),
            )
            .await
    }

    async fn install_published_ordered(
        &self,
        prepared: &PreparedCheckpoint,
        selected: Arc<Dataset>,
    ) -> Result<()> {
        ensure!(
            self.shared
                .frames
                .lock()
                .unwrap()
                .is_current(&prepared.bodies),
            "body source generation differs from prepared checkpoint"
        );
        ensure!(
            match (&prepared.write_batch, self.shared.write_batch.lock().unwrap().as_ref()) {
                (Some(frozen), Some(batch)) => batch.is_current(frozen),
                (None, None) => true,
                _ => false,
            },
            "owner write batch differs from prepared checkpoint"
        );
        self.shared
            .staging
            .install_published(&prepared.annotations, selected)
            .await?;
        self.shared.frames.lock().unwrap().retire(&prepared.bodies);
        if let Some(frozen) = &prepared.write_batch {
            if let Some(batch) = self.shared.write_batch.lock().unwrap().as_mut() {
                batch.retire(frozen);
            }
        }
        self.shared.results.lock().unwrap().base = prepared.results.clone();
        Ok(())
    }
}

pub(super) async fn open(
    ingress: &Ingress,
    storage: &Storage,
    body: Option<&Storage>,
    bucket: u32,
) -> Result<(Arc<Dataset>, Arc<Snapshot>)> {
    let query::Value::Publisher(Some(view)) = materializer::query(
        ingress,
        ingress.client.catalog_group(),
        query::Query::Publisher {
            table: ingress.scope.table,
        },
    )
    .await?
    else {
        bail!("annotation catalog unavailable");
    };
    let manifests = view
        .manifests
        .as_ref()
        .context("annotation manifests unavailable")?;
    ensure!(
        view.destination == ingress.scope.destination
            && view.ingest == ingress.scope.contract
            && manifests.incarnation == ingress.scope.incarnation,
        "foreign annotation catalog binding"
    );
    let record = &manifests.versions[&manifests.head];
    let base = load(ingress, storage, record.version).await?;
    let native = base
        .read_transaction()
        .await?
        .context("annotation base has no native transaction")?;
    ensure!(
        native.uuid == record.transaction.to_string(),
        "native annotation transaction differs from catalog"
    );
    let properties = native.transaction_properties.as_ref();
    let cut = properties.and_then(|p| p.get("lance-context.annotation-cut"));
    let covered = view
        .progress
        .checkpoint
        .get(&ingress.group(bucket)?)
        .copied()
        .unwrap_or(0);
    let Some(cut) = cut else {
        ensure!(
            covered == 0 && record.plan_index == 0,
            "materialized annotations have no application cut"
        );
        return Ok((
            base,
            Arc::new(Snapshot::empty(ingress.client.producer_slots())),
        ));
    };
    let (scope, through): (StagingScope, u64) = serde_json::from_str(cut)?;
    ensure!(
        scope
            == (StagingScope {
                table_instance: ingress.scope.incarnation.to_string(),
                source_group: ingress.group(bucket)?,
                shard: bucket
            }),
        "foreign annotation application cut"
    );
    if through == 0 {
        ensure!(
            ingress.scope.contract.policy == WritePolicy::TurnIngestion
                && covered == 0
                && record.plan_index == 0
                && properties.is_some_and(|p| !p.contains_key(RESULTS_PROPERTY)),
            "zero-cut body binding is only valid for a mixed bootstrap"
        );
        ensure!(
            lance_context_core::annotation_staging::BodySnapshot::read(&base)
                .await?
                .is_some(),
            "mixed bootstrap has no body reference"
        );
        return Ok((
            base,
            Arc::new(Snapshot::empty(ingress.client.producer_slots())),
        ));
    }
    let results = properties
        .and_then(|p| p.get(RESULTS_PROPERTY))
        .context("annotation publication has no request results")?;
    let snapshot = Snapshot::decode(results, through)?;
    ensure!(
        snapshot.slots() == ingress.client.producer_slots(),
        "annotation producer capacity changed"
    );
    if let Some(pending) = view
        .progress
        .pending
        .as_ref()
        .filter(|p| p.plan.target_version().ok() == Some(base.version_id()))
    {
        let prepared = view
            .progress
            .prepared
            .as_ref()
            .context("selected publication has no preparation")?;
        let original = load(ingress, storage, pending.plan.read_version).await?;
        if pending.plan.is_maintenance() && matches!(prepared.descriptors.format, 2 | 3) {
            validate_compacted(&base, &original, pending, prepared, record).await?;
            ensure!(
                covered == through,
                "compaction advanced annotation application cut"
            );
        } else {
            let selection = annotation_publication::Selection {
                table: ingress.scope.table,
                incarnation: ingress.scope.incarnation,
                pending: (**pending).clone(),
            };
            let selected = if let Some(body) = body {
                annotation_publication::mixed::selected_cut(
                    &base,
                    &original,
                    &prepared.descriptors,
                    &selection,
                    record,
                    storage,
                    body,
                )
                .await?
            } else {
                annotation_publication::selected_cut(
                    &base,
                    &original,
                    &prepared.descriptors,
                    &selection,
                    record,
                    storage,
                )
                .await?
            };
            ensure!(selected == through, "selected request result cut differs");
            if pending.plan.is_maintenance() {
                ensure!(
                    covered == through,
                    "body maintenance advanced the application cut"
                );
            }
        }
    } else {
        ensure!(
            covered == through && view.progress.minimum_read_version == base.version_id(),
            "annotation application cut is not covered by completed publication"
        );
    }
    Ok((base, Arc::new(snapshot)))
}

pub(super) async fn validate_compacted(
    selected: &Dataset,
    base: &Dataset,
    pending: &p::PendingPlan,
    prepared: &p::Prepared,
    record: &p::ManifestRecord,
) -> Result<()> {
    let transaction = materializer::stage::annotation_compaction_transaction(
        base,
        pending,
        &prepared.descriptors,
    )
    .await?;
    ensure!(
        selected.uri() == base.uri()
            && selected.version_id() == pending.plan.target_version()?
            && record.version == selected.version_id()
            && record.transaction == pending.plan.transaction
            && record.plan_index == pending.index
            && record.prepared_index == prepared.index
            && record.descriptor_digest == prepared.descriptors.digest
            && selected.read_transaction().await?.as_ref() == Some(&transaction),
        "selected annotation compaction differs from its retained preparation"
    );
    Ok(())
}

pub(crate) async fn load(
    ingress: &Ingress,
    storage: &Storage,
    version: u64,
) -> Result<Arc<Dataset>> {
    load_with_read_params(ingress, storage, version, Default::default()).await
}

pub(crate) async fn load_with_read_params(
    ingress: &Ingress,
    storage: &Storage,
    version: u64,
    read_params: lance::dataset::ReadParams,
) -> Result<Arc<Dataset>> {
    let handler = ManagedManifest::reader(
        Arc::new(ingress.client.manifest_catalog()),
        storage.base.clone(),
        &ingress.layout,
        ingress.scope.table,
        ingress.scope.incarnation,
    )?;
    Ok(Arc::new(
        DatasetBuilder::from_uri(&storage.uri)
            .with_read_params(storage.read_params(read_params))
            .with_version(version)
            .with_session(storage.session.clone())
            .with_commit_handler(Arc::new(handler))
            .load()
            .await?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maximum_producer_outcomes_fit_result_metadata() -> Result<()> {
        let rejection = Rejection {
            row: 65_535,
            reason: RejectionReason::BatchTooLarge,
        };
        let snapshot = Snapshot {
            through: u64::MAX - 1,
            slots: producer::MAX_SLOTS,
            outcomes: (0..producer::MAX_SLOTS)
                .map(|slot| {
                    (
                        slot,
                        SlotOutcome {
                            generation: u64::MAX,
                            latest: Some((u64::MAX - 1, rejection.clone())),
                            sticky: Some((u64::MAX - 2, rejection.clone())),
                        },
                    )
                })
                .collect(),
        };
        let encoded = snapshot.encode()?;
        assert!(encoded.len() <= MAX_RESULTS_BYTES);
        let decoded = Snapshot::decode(&encoded, u64::MAX - 1)?;
        assert_eq!(decoded.outcomes.len(), producer::MAX_SLOTS as usize);
        Ok(())
    }
}
