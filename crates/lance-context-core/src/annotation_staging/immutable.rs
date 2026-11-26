//! First-create state for an immutable full-body table. Reuses the bounded
//! active/frozen generations, publication identity, and protected base registry.
use super::*;

/// A protected base for one batched historical existence query. The caller
/// queries only key columns and supplies each answer to ordered application.
/// Holding this handle consumes one existing reader permit and pins local GC.
pub struct ImmutableBase {
    base: Arc<views::Base>,
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
            _reader: reader,
        })
    }

    /// Capture without waiting for reader capacity. `None` means existing
    /// readers hold every permit; callers must retain their unapplied request.
    pub async fn try_capture_immutable_base(&self) -> Result<Option<ImmutableBase>> {
        self.require_immutable()?;
        let reader = match self.readers.clone().try_acquire_owned() {
            Ok(reader) => reader,
            Err(tokio::sync::TryAcquireError::NoPermits) => return Ok(None),
            Err(tokio::sync::TryAcquireError::Closed) => {
                return Err(invalid("annotation staging reader admission closed"));
            }
        };
        Ok(Some(ImmutableBase {
            base: self.state.read().await.base.clone(),
            _reader: reader,
        }))
    }

    /// Apply one already quorum-committed request. Active/frozen keys are
    /// rechecked under the mutation gate, so concurrent lanes and replay keep
    /// the first source row. `None` means the queried base changed: refresh the
    /// lookup and retry the SAME request, without advancing its WAL position.
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
        if !Arc::ptr_eq(&state.base, &lookup.base) {
            return Ok(None);
        }
        let pending = self.immutable_pending(&state, through, rows)?;
        let outcome = match pending {
            PendingBatch::Values(values) => {
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
