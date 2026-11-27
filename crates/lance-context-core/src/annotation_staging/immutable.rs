//! First-create state for an immutable full-body table. Reuses the bounded
//! active/frozen generations, publication identity, and protected base registry.
use super::*;

/// A protected base for one batched historical existence query. The caller
/// queries only key columns and supplies each answer to ordered application.
/// Holding this handle consumes one existing reader permit and pins local GC.
pub struct ImmutableBase {
    base: Arc<views::Base>,
    covered: Option<(Arc<()>, u64)>,
    _reader: OwnedSemaphorePermit,
}

impl ImmutableBase {
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.base.dataset
    }
}

/// One original input row and its existence in the exact captured base. These
/// are trusted query results, not client-supplied deduplication decisions.
pub struct ImmutableRegistration<'a> {
    pub key: &'a str,
    pub exists_in_base: bool,
}

/// The service must hold its ordered frame gate until it retains the indicated
/// Arrow frame. A materialized result directly names the full body dataset.
pub struct ImmutableBodyRead {
    source: body::CapturedBody,
    _reader: OwnedSemaphorePermit,
}

impl ImmutableBodyRead {
    pub fn source(&self) -> TurnBodySource<'_> {
        match &self.source {
            body::CapturedBody::Staged(source) => TurnBodySource::Unmaterialized(*source),
            body::CapturedBody::Base(base) => TurnBodySource::Materialized(base),
        }
    }
}

impl AnnotationStaging {
    /// Full-body first-create mode. No annotation columns or auxiliary dataset
    /// are created. The caller binds and validates the business-key schema.
    pub fn new_immutable_body(
        base: Arc<Dataset>,
        scope: Scope,
        materialized: u64,
        limits: Limits,
    ) -> Result<Self> {
        Self::new_inner(base, scope, materialized, limits, true)
    }

    pub async fn capture_immutable_base(&self) -> Result<ImmutableBase> {
        self.require_immutable()?;
        let reader = reserve(&self.readers, 1)?;
        Ok(ImmutableBase {
            base: self.state.read().await.base.clone(),
            covered: None,
            _reader: reader,
        })
    }

    /// Capture without waiting for reader capacity. `None` means existing
    /// readers hold every permit; callers must retain their unapplied request.
    pub async fn try_capture_immutable_base(&self) -> Result<Option<ImmutableBase>> {
        self.try_capture_immutable_lookup(None).await
    }

    /// Mark keys already present in active/frozen while capturing the same
    /// protected base. The caller owns and budgets this bounded key map.
    /// False still requires a historical lookup; it does not mean absent.
    /// Results for exactly these keys may survive verified publications up to
    /// the captured applied cut. The caller must preserve each marked hit when
    /// supplying registrations; this does not certify any additional key.
    pub async fn try_capture_immutable_base_with_staged(
        &self,
        keys: &mut BTreeMap<String, bool>,
    ) -> Result<Option<ImmutableBase>> {
        self.try_capture_immutable_lookup(Some(keys)).await
    }

    async fn try_capture_immutable_lookup(
        &self,
        keys: Option<&mut BTreeMap<String, bool>>,
    ) -> Result<Option<ImmutableBase>> {
        self.require_immutable()?;
        let reader = match self.readers.clone().try_acquire_owned() {
            Ok(reader) => reader,
            Err(tokio::sync::TryAcquireError::NoPermits) => return Ok(None),
            Err(tokio::sync::TryAcquireError::Closed) => {
                return Err(invalid("annotation staging reader admission closed"));
            }
        };
        let state = self.state.read().await;
        let covered = keys
            .as_ref()
            .map(|_| (state.immutable_publications.clone(), state.applied));
        if let Some(keys) = keys {
            for (key, present) in keys {
                *present = state.active.contains_key(key)
                    || state
                        .frozen
                        .as_ref()
                        .is_some_and(|f| f.values.contains_key(key));
            }
        }
        Ok(Some(ImmutableBase {
            base: state.base.clone(),
            covered,
            _reader: reader,
        }))
    }

    /// Apply one already quorum-committed request. Active/frozen keys are
    /// rechecked under the mutation gate, so concurrent lanes and replay keep
    /// the first source row. A staged-covered lookup also includes every key
    /// published through its captured applied cut. It survives only verified
    /// publications on this instance's lineage up to that cut. `None` requires
    /// a fresh lookup for the SAME request, without advancing its WAL position.
    /// Backpressure also leaves the entire request unapplied.
    pub async fn register_immutable_committed(
        &self,
        after: u64,
        through: u64,
        rows: &[ImmutableRegistration<'_>],
        lookup: &ImmutableBase,
    ) -> Result<Option<ApplyOutcome>> {
        self.require_immutable()?;
        let _mutation = self.mutation.lock().await;
        let mut state = self.state.write().await;
        check_order(&state, after, through)?;
        let covered_publication = lookup.covered.as_ref().is_some_and(|(lineage, through)| {
            Arc::ptr_eq(lineage, &state.immutable_publications) && state.materialized <= *through
        });
        let changed_base = !Arc::ptr_eq(&state.base, &lookup.base);
        if changed_base && !covered_publication {
            state.immutable_lookup_invalidated =
                state.immutable_lookup_invalidated.saturating_add(1);
            return Ok(None);
        }
        let pending = self.immutable_pending(&state, through, rows)?;
        let outcome = match pending {
            PendingBatch::Values(values) => {
                let inserted = values.len() as u64;
                let observed = &mut state.immutable_apply;
                observed.batches = observed.batches.saturating_add(1);
                observed.rows = observed.rows.saturating_add(rows.len() as u64);
                observed.inserted_rows = observed.inserted_rows.saturating_add(inserted);
                observed.duplicate_rows = observed
                    .duplicate_rows
                    .saturating_add(rows.len() as u64 - inserted);
                state.active.extend(values);
                state.applied = through;
                ApplyOutcome::Applied
            }
            PendingBatch::Rejected(rejection) => {
                state.applied = through;
                ApplyOutcome::Rejected(rejection)
            }
            PendingBatch::Backpressure => ApplyOutcome::Backpressure,
        };
        if changed_base && !matches!(outcome, ApplyOutcome::Backpressure) {
            state.immutable_lookup_reused = state.immutable_lookup_reused.saturating_add(1);
        }
        Ok(Some(outcome))
    }

    fn immutable_pending(
        &self,
        state: &State,
        through: u64,
        rows: &[ImmutableRegistration<'_>],
    ) -> Result<PendingBatch> {
        if rows.is_empty() || rows.len() > 65_536 {
            return Ok(reject(0, RejectionReason::BatchTooLarge));
        }
        for (row, value) in rows.iter().enumerate() {
            if value.key.is_empty() || value.key.len() > self.limits.value_bytes {
                return Ok(reject(row, RejectionReason::InvalidTurn));
            }
        }
        let mut pending = Values::new();
        let mut batch_bytes = 0usize;
        for (row, value) in rows.iter().enumerate() {
            if value.exists_in_base
                || pending.contains_key(value.key)
                || state.active.contains_key(value.key)
                || state
                    .frozen
                    .as_ref()
                    .is_some_and(|f| f.values.contains_key(value.key))
            {
                continue;
            }
            let source = PendingTurn {
                rollout_id: String::new(),
                through,
                row: row as u32,
            };
            // Only key/origin bookkeeping is owned here; the service's frame
            // window holds the body arrays and their existing apply reservation.
            let annotation = TurnAnnotation::default();
            let bytes =
                registration::retained_charge(charge(value.key, &annotation)?, Some(&source))?;
            batch_bytes = batch_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("body key charge overflow"))?;
            if batch_bytes > self.limits.retained_bytes / 2
                || pending.len() == self.limits.retained_entries
            {
                return Ok(reject(row, RejectionReason::BatchTooLarge));
            }
            let Ok(allocation) = self.bytes.clone().try_acquire_many_owned(bytes as u32) else {
                return Ok(PendingBatch::Backpressure);
            };
            let Ok(entry) = self.entries.clone().try_acquire_owned() else {
                return Ok(PendingBatch::Backpressure);
            };
            pending.insert(
                value.key.to_owned(),
                Arc::new(Stored {
                    value: annotation,
                    pending_turn: Some(source),
                    _bytes: allocation,
                    _entry: Arc::new(entry),
                }),
            );
        }
        Ok(PendingBatch::Values(pending))
    }

    pub async fn capture_immutable_body(&self, key: &str) -> Result<ImmutableBodyRead> {
        self.require_immutable()?;
        if key.is_empty() || key.len() > self.limits.value_bytes {
            return Err(invalid("invalid body key"));
        }
        let reader = reserve(&self.readers, 1)?;
        let state = self.state.read().await;
        let staged = state
            .active
            .get(key)
            .or_else(|| state.frozen.as_ref().and_then(|f| f.values.get(key)));
        let source = match staged.and_then(|value| value.pending_turn.as_ref()) {
            Some(pending) => body::CapturedBody::Staged(pending.source()),
            None => body::CapturedBody::Base(state.base.clone()),
        };
        Ok(ImmutableBodyRead {
            source,
            _reader: reader,
        })
    }

    fn require_immutable(&self) -> Result<()> {
        if !self.immutable_body {
            return Err(invalid("not an immutable body staging instance"));
        }
        Ok(())
    }
}
