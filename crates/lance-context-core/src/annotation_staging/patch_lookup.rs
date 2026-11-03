//! Bounded base values shared by separately applied annotation requests.
use super::*;
use crate::{MergeMemoryBudget, MergeReservation};

/// Temporary narrow annotations from one staging instance's exact base.
/// The existing GC registry, read-view permit and caller's workspace protect
/// these values until dropped. This carries no progress or write authority.
pub struct PatchLookup {
    base: Arc<views::Base>,
    // Sorted unique keys; None records an explicitly queried missing turn.
    keys: Vec<(String, Option<TurnAnnotation>)>,
    _workspace: MergeReservation,
    _reader: OwnedSemaphorePermit,
}

impl PatchLookup {
    pub(super) fn matches(&self, base: &Arc<views::Base>) -> bool {
        Arc::ptr_eq(&self.base, base)
    }

    pub(super) fn get(&self, turn: &str) -> Option<&Option<TurnAnnotation>> {
        self.keys
            .binary_search_by(|(key, _)| key.as_str().cmp(turn))
            .ok()
            .map(|index| &self.keys[index].1)
    }
}

impl AnnotationStaging {
    /// Read a bounded set of immutable base annotations for independent patch
    /// requests. Never projects the body, applies a patch or advances a cursor.
    /// Active/frozen/cached values are omitted and rechecked during application.
    ///
    /// Reserve the maximum decoded value size per key before reading. Invalid
    /// keys or unavailable workspace/read admission return `None` immediately;
    /// callers then use ordinary ordered apply. Lance scan buffers retain their
    /// existing separate accounting; this is not a process-wide RSS bound.
    pub async fn prepare_patch_lookup(
        &self,
        turns: &[&str],
        workspace: &Arc<MergeMemoryBudget>,
    ) -> Result<Option<PatchLookup>> {
        if turns.is_empty()
            || turns.len() > COLD_LOOKUP_KEYS
            || turns
                .iter()
                .any(|turn| turn.is_empty() || turn.len() > self.limits.value_bytes)
        {
            return Ok(None);
        }
        // Decoded annotations, owned keys and temporary collection bookkeeping.
        let Some(bytes) = turns.iter().try_fold(0usize, |n, turn| {
            n.checked_add(self.limits.value_bytes)?
                .checked_add(512)?
                .checked_add(turn.len())
        }) else {
            return Ok(None);
        };
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
        let rows = read_bases(&base, &requested).await?;
        let mut values = Vec::with_capacity(cold.len());
        for key in &cold {
            let value = match rows.get(key.as_str()) {
                Some(batch) => {
                    let Ok(value) = decode_base_row(batch, 0, key, self.limits.value_bytes) else {
                        // A bad later value must not fail an earlier request.
                        // Let ordinary apply diagnose it in request order.
                        return Ok(None);
                    };
                    Some(value)
                }
                None => None,
            };
            values.push(value);
        }
        // Do not retain scanner buffers through small RecordBatch slices.
        drop(rows);
        drop(requested);
        Ok(Some(PatchLookup {
            base,
            keys: cold.into_iter().zip(values).collect(),
            _workspace: reservation,
            _reader: reader,
        }))
    }

    /// Apply one original committed request with a captured base lookup.
    /// Current active/frozen/cache state wins; a different base or unqueried key
    /// uses the ordinary query path. Ordering, rejection, backpressure and
    /// durable result retention have the same contract as `apply_committed`.
    pub async fn apply_committed_with_lookup(
        &self,
        after: u64,
        through: u64,
        mutations: &[Mutation<'_>],
        lookup: &PatchLookup,
    ) -> Result<ApplyOutcome> {
        self.apply_inner(
            after,
            through,
            Batch::Patches(mutations),
            true,
            None,
            Some(lookup),
        )
        .await
    }
}
