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
    pub miss_disabled: u64,
    /// No resident marker; does not distinguish new keys from eviction.
    pub miss_absent: u64,
    pub miss_wrong_key: u64,
    pub miss_wrong_lineage: u64,
    pub miss_future_cut: u64,
    /// Insertions submitted to the cache, which may immediately evict them.
    pub warms: u64,
    /// Optional warming skipped because backing memory could not be admitted.
    pub refusals: u64,
    /// Entire shared Session cache, not this membership namespace.
    pub session_cache_capacity_bytes: Option<usize>,
    /// Backend's cheap approximate entry count across the shared cache.
    /// Zero can also mean unsupported; this is not resident membership size.
    pub session_cache_reported_entries: Option<usize>,
}

#[derive(Default)]
pub(super) struct Counters {
    hits: AtomicU64,
    miss_disabled: AtomicU64,
    miss_absent: AtomicU64,
    miss_wrong_key: AtomicU64,
    miss_wrong_lineage: AtomicU64,
    miss_future_cut: AtomicU64,
    warms: AtomicU64,
    refusals: AtomicU64,
}

enum Probe {
    Hit,
    Disabled,
    Absent,
    WrongKey,
    WrongLineage,
    FutureCut,
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
    async fn resident_member(&self, key: &str, lineage: &Arc<()>, through: u64) -> Probe {
        let Some(cache) = &self.cache else {
            return Probe::Disabled;
        };
        let Some(entry) = cache.get_resident_with_key(&MemberKey(key)).await else {
            return Probe::Absent;
        };
        if entry.key.as_ref() != key {
            Probe::WrongKey
        } else if !Arc::ptr_eq(&entry.lineage, lineage) {
            Probe::WrongLineage
        } else if entry.through > through {
            Probe::FutureCut
        } else {
            Probe::Hit
        }
    }

    pub async fn contains_immutable(&self, key: &str, lineage: &Arc<()>, through: u64) -> bool {
        let probe = self.resident_member(key, lineage, through).await;
        let counter = match probe {
            Probe::Hit => &self.membership.hits,
            Probe::Disabled => &self.membership.miss_disabled,
            Probe::Absent => &self.membership.miss_absent,
            Probe::WrongKey => &self.membership.miss_wrong_key,
            Probe::WrongLineage => &self.membership.miss_wrong_lineage,
            Probe::FutureCut => &self.membership.miss_future_cut,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        matches!(probe, Probe::Hit)
    }

    /// The caller has proved this key exists in the installed or selected base.
    /// Failure only skips caching; it cannot reject a committed request.
    pub async fn remember_immutable(&self, key: &str, lineage: &Arc<()>, through: u64) {
        let Some(cache) = &self.cache else { return };
        if matches!(
            self.resident_member(key, lineage, through).await,
            Probe::Hit
        ) {
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
        let miss_disabled = self.membership.miss_disabled.load(Ordering::Relaxed);
        let miss_absent = self.membership.miss_absent.load(Ordering::Relaxed);
        let miss_wrong_key = self.membership.miss_wrong_key.load(Ordering::Relaxed);
        let miss_wrong_lineage = self.membership.miss_wrong_lineage.load(Ordering::Relaxed);
        let miss_future_cut = self.membership.miss_future_cut.load(Ordering::Relaxed);
        MembershipObservation {
            hits: self.membership.hits.load(Ordering::Relaxed),
            // Reuse the sampled reasons: one counter update per probe, with
            // no extra cache lookup or independently sampled total.
            misses: miss_disabled
                .saturating_add(miss_absent)
                .saturating_add(miss_wrong_key)
                .saturating_add(miss_wrong_lineage)
                .saturating_add(miss_future_cut),
            miss_disabled,
            miss_absent,
            miss_wrong_key,
            miss_wrong_lineage,
            miss_future_cut,
            warms: self.membership.warms.load(Ordering::Relaxed),
            refusals: self.membership.refusals.load(Ordering::Relaxed),
            session_cache_capacity_bytes: self.cache.as_ref().and_then(LanceCache::capacity_bytes),
            session_cache_reported_entries: self.cache.as_ref().map(LanceCache::approx_size),
        }
    }
}
