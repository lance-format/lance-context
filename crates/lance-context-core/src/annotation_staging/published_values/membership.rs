//! Disposable positive membership in the existing bounded Session cache.
use super::*;
use lance_core::memory::{self, NativeMemoryKind, NativeReservation};
use std::sync::atomic::{AtomicU64, Ordering};

/// Process-local cache activity; neither application nor publication progress.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct MembershipObservation {
    pub hits: u64,
    /// Unknown keys, including disabled/evicted/stale entries; never absence.
    pub misses: u64,
    /// Insertions submitted to the cache, which may immediately evict them.
    pub warms: u64,
    /// Optional warming skipped because backing memory could not be admitted.
    pub refusals: u64,
}

#[derive(Default)]
pub(super) struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    warms: AtomicU64,
    refusals: AtomicU64,
}

struct Member {
    // Native cache keys are hashed. Keep exact business identity as well.
    key: Box<str>,
    lineage: Arc<()>,
    through: u64,
    // Borrowing an evicted entry retains admission, not a Dataset/reader pin.
    _reservation: NativeReservation,
}

impl DeepSizeOf for Member {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        // Conservative Arc/cache-node/reservation overhead, matching admission.
        512 + self.key.len() - std::mem::size_of::<Self>()
    }
}

struct MemberKey<'a>(&'a str);

impl CacheKey for MemberKey<'_> {
    type ValueType = Member;

    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.0)
    }

    fn type_name() -> &'static str {
        "lance-context.immutable-membership"
    }
    // No codec: membership is disposable resident data, never a disk authority.
}

impl PublishedValues {
    async fn resident_member(&self, key: &str, lineage: &Arc<()>, through: u64) -> bool {
        let Some(cache) = &self.cache else {
            return false;
        };
        cache
            .get_resident_with_key(&MemberKey(key))
            .await
            .is_some_and(|entry| {
                entry.key.as_ref() == key
                    && Arc::ptr_eq(&entry.lineage, lineage)
                    && entry.through <= through
            })
    }

    pub async fn contains_immutable(&self, key: &str, lineage: &Arc<()>, through: u64) -> bool {
        let present = self.resident_member(key, lineage, through).await;
        let counter = if present {
            &self.membership.hits
        } else {
            &self.membership.misses
        };
        counter.fetch_add(1, Ordering::Relaxed);
        present
    }

    /// The caller has proved this key exists in the installed or selected base.
    /// Failure only skips caching; it cannot reject a committed request.
    pub async fn remember_immutable(&self, key: &str, lineage: &Arc<()>, through: u64) {
        let Some(cache) = &self.cache else { return };
        if self.resident_member(key, lineage, through).await {
            return;
        }
        let Some(bytes) = key.len().checked_add(512) else {
            self.membership.refusals.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let Ok(mut reservation) = memory::try_reserve(&self.memory, NativeMemoryKind::Index, bytes)
        else {
            self.membership.refusals.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let key_copy = Box::<str>::from(key);
        if let Some(reservation) = &mut reservation {
            reservation.set_backed_bytes(bytes);
        }
        cache
            .insert_with_key(
                &MemberKey(key),
                Arc::new(Member {
                    key: key_copy,
                    lineage: lineage.clone(),
                    through,
                    _reservation: reservation,
                }),
            )
            .await;
        self.membership.warms.fetch_add(1, Ordering::Relaxed);
    }

    pub fn membership_observation(&self) -> MembershipObservation {
        MembershipObservation {
            hits: self.membership.hits.load(Ordering::Relaxed),
            misses: self.membership.misses.load(Ordering::Relaxed),
            warms: self.membership.warms.load(Ordering::Relaxed),
            refusals: self.membership.refusals.load(Ordering::Relaxed),
        }
    }
}
