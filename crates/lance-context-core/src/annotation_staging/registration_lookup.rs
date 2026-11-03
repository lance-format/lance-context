//! Bounded existence lookups shared by separately applied registration requests.
use super::*;
use crate::{MergeMemoryBudget, MergeReservation};

/// Temporary ID-only results for one staging instance's exact base generation.
/// This pins that base in the existing local GC registry, holds one read-view
/// permit, and charges owned keys to the caller's workspace until dropped.
/// It carries no application progress, request outcome, or write authority.
pub struct RegistrationLookup {
    base: Arc<views::Base>,
    // Sorted, unique keys. Only explicitly queried misses may skip a lookup.
    keys: Vec<(String, bool)>,
    _workspace: MergeReservation,
    _reader: OwnedSemaphorePermit,
}

impl RegistrationLookup {
    pub(super) fn matches(&self, base: &Arc<views::Base>) -> bool {
        // Different staging instances have distinct wrappers even when they
        // share a Dataset. Versions alone do not identify a table or a view.
        Arc::ptr_eq(&self.base, base)
    }

    pub(super) fn get(&self, turn: &str) -> Option<bool> {
        self.keys
            .binary_search_by(|(key, _)| key.as_str().cmp(turn))
            .ok()
            .map(|index| self.keys[index].1)
    }
}

impl AnnotationStaging {
    /// Query up to one existing cold-lookup window for several independent
    /// registration requests. This does not apply or validate those requests.
    /// Invalid/oversized key windows and unavailable workspace/read admission
    /// return `None`; callers should continue their ordinary ordered apply.
    /// Admission never waits for more memory while holding pending requests.
    ///
    /// Use the caller's existing bounded apply workspace. Owned-key accounting
    /// does not include Lance's internal scan workspace or establish an RSS cap.
    pub async fn prepare_registration_lookup(
        &self,
        turns: &[&str],
        workspace: &Arc<MergeMemoryBudget>,
    ) -> Result<Option<RegistrationLookup>> {
        if turns.is_empty()
            || turns.len() > COLD_LOOKUP_KEYS
            || turns
                .iter()
                .any(|turn| turn.is_empty() || turn.len() > self.limits.value_bytes)
        {
            return Ok(None);
        }
        // Covers key ownership and temporary key/vector/tree bookkeeping.
        let Some(bytes) = turns.iter().try_fold(0usize, |n, turn| {
            n.checked_add(512)?.checked_add(turn.len())
        }) else {
            return Ok(None);
        };
        // The general merge budget can admit an oversized merge alone. This
        // speculative helper must instead fall back when its window is too big.
        if bytes > workspace.limit() {
            return Ok(None);
        }
        let Some(reservation) = workspace.try_reserve(bytes) else {
            return Ok(None);
        };
        let Ok(reader) = self.readers.clone().try_acquire_owned() else {
            return Ok(None);
        };
        let (base, mut keys) = {
            let state = self.state.read().await;
            registration::validate_target(&state.base)?;
            let keys = turns
                .iter()
                .filter(|turn| {
                    !state.active.contains_key(**turn)
                        && !state
                            .frozen
                            .as_ref()
                            .is_some_and(|f| f.values.contains_key(**turn))
                })
                .map(|turn| (*turn).to_owned())
                .collect::<Vec<_>>();
            (state.base.clone(), keys)
        };
        keys.sort_unstable();
        keys.dedup();
        let mut cold = Vec::with_capacity(keys.len());
        for key in keys {
            if self.published_values.get(&key).await.is_none() {
                cold.push(key);
            }
        }
        if cold.is_empty() {
            return Ok(None);
        }
        let requested = cold.iter().map(String::as_str).collect();
        let rows = read_existing(&base, &requested, &["id"]).await?;
        let found = cold
            .iter()
            .map(|key| rows.contains_key(key.as_str()))
            .collect::<Vec<_>>();
        drop(rows);
        drop(requested);
        Ok(Some(RegistrationLookup {
            base,
            keys: cold.into_iter().zip(found).collect(),
            _workspace: reservation,
            _reader: reader,
        }))
    }

    /// Apply one original committed request using a previously captured lookup.
    /// Active/frozen values take precedence. Keys absent from the lookup and
    /// every lookup whose base has changed use the ordinary native query path.
    /// Rejection, backpressure, source positions and result retention have the
    /// same contract as `register_committed`; this does not combine requests.
    pub async fn register_committed_with_lookup(
        &self,
        after: u64,
        through: u64,
        rows: &[Registration<'_>],
        lookup: &RegistrationLookup,
    ) -> Result<ApplyOutcome> {
        self.apply_inner(
            after,
            through,
            Batch::Registrations(rows),
            true,
            Some(lookup),
            None,
        )
        .await
    }
}
