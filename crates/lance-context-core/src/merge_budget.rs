//! Process-wide byte budget for MemWAL merges.
//!
//! A merge reads up to `merge_max_bytes` of Arrow batches into memory before it
//! commits. That bounds one merge. Nothing bounded how many merges a process
//! ran at once: every merge entry point -- the worker's flush and cleanup
//! sweepers, the count trigger, the manual `/merge-wal` route, and the master's
//! fan-out -- reserved memory independently. With the master scheduling merges
//! for every store and fanning out to every worker, a worker could be asked to
//! hold a dozen 1 GiB merge reads at once, and 17 of 20 production workers
//! OOMKilled within ninety seconds of each other.
//!
//! [`MergeMemoryBudget`] is the missing bound. It is measured in **bytes**, not
//! merges: twenty tiny merges and one huge merge are priced by what they hold.
//! A merge reserves from it before reading and grows its reservation as it
//! reads; when the budget is exhausted the next merge **waits** for a release
//! rather than being rejected, so a master fanning out to a busy worker sees a
//! slower worker, not a failure. The reservation is RAII and travels inside
//! [`crate::store_base::PreparedMerge`], so it is released exactly when the
//! merged batches are dropped after commit.
//!
//! # Growth must never wait while holding a reservation
//!
//! Atomic initial reservations do not prevent a hold-and-wait cycle: six
//! 2-GiB readers can fill a 12-GiB budget, then all need one more batch to
//! finish their last generation. Growth therefore uses a nonblocking acquire.
//! On contention the caller must discard the incomplete generation and commit
//! an already complete prefix, or release the read and retry if no prefix fits.
//! An oversized generation can still proceed when it is the sole holder.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Semaphore;

/// Reservation granularity. One permit is one MiB; the semaphore's permit
/// count is bounded (`Semaphore::MAX_PERMITS`), and byte-granular permits would
/// overflow it for budgets past a few GiB.
const PERMIT_BYTES: usize = 1024 * 1024;

/// Byte budget shared by every merge in a process.
#[derive(Debug)]
pub struct MergeMemoryBudget {
    permits: Semaphore,
    limit: usize,
    /// Bytes currently reserved, for the gauge. Tracked separately because
    /// `Semaphore::available_permits` rounds to MiB.
    reserved: AtomicUsize,
}

/// RAII reservation held by an in-flight merge. Releases on drop.
#[derive(Debug)]
pub struct MergeReservation {
    budget: Arc<MergeMemoryBudget>,
    permits: u32,
    bytes: usize,
    lifetime: Arc<ReservationLifetime>,
}

/// Keeps the final reservation charged while detached encoder children finish.
/// It does not duplicate permits or allow concurrent mutation of a reservation.
#[derive(Debug)]
pub struct MergeReservationHold {
    _lifetime: Arc<ReservationLifetime>,
}

#[derive(Debug)]
struct ReservationLifetime {
    budget: Arc<MergeMemoryBudget>,
    finished: Mutex<Option<(u32, usize)>>,
}

impl Drop for ReservationLifetime {
    fn drop(&mut self) {
        if let Some((permits, bytes)) = self.finished.get_mut().unwrap().take() {
            self.budget.reserved.fetch_sub(bytes, Ordering::AcqRel);
            self.budget.permits.add_permits(permits as usize);
        }
    }
}

impl MergeMemoryBudget {
    /// A budget admitting up to `limit` bytes of concurrently buffered merge
    /// data. `limit` is rounded up to whole MiB.
    #[must_use]
    pub fn new(limit: usize) -> Arc<Self> {
        let permits = Self::permits_for(limit).max(1);
        Arc::new(Self {
            permits: Semaphore::new(permits as usize),
            limit: permits as usize * PERMIT_BYTES,
            reserved: AtomicUsize::new(0),
        })
    }

    /// The configured limit, in bytes (rounded up to whole MiB).
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes currently reserved across all in-flight merges.
    #[must_use]
    pub fn reserved(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }

    fn permits_for(bytes: usize) -> u32 {
        let permits = bytes.div_ceil(PERMIT_BYTES);
        u32::try_from(permits).unwrap_or(u32::MAX)
    }

    /// Reserve `bytes`, waiting until they fit. A request larger than the whole
    /// budget waits for the budget to be idle and then takes all of it: an
    /// oversized merge must still make progress, so it is admitted alone.
    pub async fn reserve(self: &Arc<Self>, bytes: usize) -> MergeReservation {
        let want = Self::permits_for(bytes);
        let total = self.permits_total();
        let permits = want.min(total);
        // `acquire_many` only fails if the semaphore is closed, which we never do.
        let permit = self
            .permits
            .acquire_many(permits)
            .await
            .expect("merge memory budget semaphore is never closed");
        permit.forget();
        self.reserved.fetch_add(bytes, Ordering::AcqRel);
        MergeReservation {
            budget: Arc::clone(self),
            permits,
            bytes,
            lifetime: Arc::new(ReservationLifetime {
                budget: self.clone(),
                finished: Mutex::new(None),
            }),
        }
    }

    /// Admit speculative work only if capacity is available now. A caller
    /// holding prepared batches must never wait for the next reservation.
    pub fn try_reserve(self: &Arc<Self>, bytes: usize) -> Option<MergeReservation> {
        let permits = Self::permits_for(bytes).min(self.permits_total());
        let permit = self.permits.try_acquire_many(permits).ok()?;
        permit.forget();
        self.reserved.fetch_add(bytes, Ordering::AcqRel);
        Some(MergeReservation {
            budget: self.clone(),
            permits,
            bytes,
            lifetime: Arc::new(ReservationLifetime {
                budget: self.clone(),
                finished: Mutex::new(None),
            }),
        })
    }

    fn permits_total(&self) -> u32 {
        Self::permits_for(self.limit)
    }
}

impl MergeReservation {
    /// Delay release on owner drop until this hold is also dropped. Growth and
    /// shrinkage still belong solely to the owner; the hold retains its final
    /// accounting, including oversized reservations.
    pub fn retain(&self) -> MergeReservationHold {
        MergeReservationHold {
            _lifetime: self.lifetime.clone(),
        }
    }

    pub(crate) fn belongs_to(&self, budget: &Arc<MergeMemoryBudget>) -> bool {
        Arc::ptr_eq(&self.budget, budget)
    }

    /// Bytes this reservation currently accounts for.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Release capacity after the covered allocations have been dropped.
    /// A larger value leaves the reservation unchanged; growth requires
    /// `try_grow_to` so shrinking can never wait while holding permits.
    pub fn shrink_to(&mut self, new_total: usize) {
        if new_total >= self.bytes {
            return;
        }
        let keep = MergeMemoryBudget::permits_for(new_total).min(self.budget.permits_total());
        let release = self.permits - keep;
        self.budget
            .reserved
            .fetch_sub(self.bytes - new_total, Ordering::AcqRel);
        self.bytes = new_total;
        self.permits = keep;
        self.budget.permits.add_permits(release as usize);
    }

    /// Try to cover `new_total` without waiting while holding permits.
    /// False leaves the reservation unchanged. The caller must release any
    /// unaccounted batches and either commit a complete prefix or retry later.
    #[must_use]
    pub fn try_grow_to(&mut self, new_total: usize) -> bool {
        if new_total <= self.bytes {
            return true;
        }
        let want_total = MergeMemoryBudget::permits_for(new_total).min(self.budget.permits_total());
        let extra = want_total.saturating_sub(self.permits);
        if extra > 0 {
            let Ok(permit) = self.budget.permits.try_acquire_many(extra) else {
                return false;
            };
            permit.forget();
            self.permits += extra;
        }
        self.budget
            .reserved
            .fetch_add(new_total - self.bytes, Ordering::AcqRel);
        self.bytes = new_total;
        true
    }
}

impl Drop for MergeReservation {
    fn drop(&mut self) {
        *self.lifetime.finished.lock().unwrap() = Some((self.permits, self.bytes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn retained_reservation_releases_final_charge_only_after_last_holder() {
        let mib = PERMIT_BYTES;
        let budget = MergeMemoryBudget::new(4 * mib);
        for (start, grown, shrunk) in [(mib, 3 * mib, mib + 1), (10 * mib, 12 * mib, 7 * mib)] {
            let mut reservation = budget.reserve(start).await;
            let first = reservation.retain();
            let second = reservation.retain();
            assert!(reservation.try_grow_to(grown));
            reservation.shrink_to(shrunk);
            let available = budget.permits.available_permits();
            drop(reservation);
            assert_eq!(budget.reserved(), shrunk);
            assert_eq!(budget.permits.available_permits(), available);
            drop(first);
            assert_eq!(budget.reserved(), shrunk);
            drop(second);
            assert_eq!(budget.reserved(), 0);
            assert_eq!(budget.permits.available_permits(), 4);
        }
        let reservation = budget.reserve(mib).await;
        drop(reservation.retain());
        assert_eq!(
            budget.reserved(),
            mib,
            "a returned committer still owns its reservation"
        );
        drop(reservation);
        assert_eq!(budget.reserved(), 0);
    }

    #[tokio::test]
    async fn reservations_release_on_drop() {
        let budget = MergeMemoryBudget::new(4 * PERMIT_BYTES);
        let a = budget.reserve(2 * PERMIT_BYTES).await;
        assert_eq!(budget.reserved(), 2 * PERMIT_BYTES);
        drop(a);
        assert_eq!(budget.reserved(), 0);
        assert_eq!(budget.permits.available_permits(), 4);
    }

    #[tokio::test]
    async fn releasing_workspace_admits_a_waiter_while_payload_remains_held() {
        let budget = MergeMemoryBudget::new(4 * PERMIT_BYTES);
        let mut held = budget.reserve(3 * PERMIT_BYTES).await;
        let next = budget.reserve(2 * PERMIT_BYTES);
        tokio::pin!(next);
        assert!(tokio::time::timeout(Duration::from_millis(20), &mut next)
            .await
            .is_err());

        // Retain a partial-MiB payload charge while freeing its decode workspace.
        held.shrink_to(PERMIT_BYTES + 1);
        let admitted = tokio::time::timeout(Duration::from_secs(1), next)
            .await
            .expect("the waiter must use released workspace before payload drop");
        assert_eq!(budget.reserved(), 3 * PERMIT_BYTES + 1);
        assert_eq!(budget.permits.available_permits(), 0);
        held.shrink_to(4 * PERMIT_BYTES);
        assert_eq!(held.bytes(), PERMIT_BYTES + 1);
        drop((held, admitted));
        assert_eq!(budget.reserved(), 0);
        assert_eq!(budget.permits.available_permits(), 4);
    }

    #[tokio::test]
    async fn oversized_reservation_can_shrink_then_grow_without_losing_permits() {
        let budget = MergeMemoryBudget::new(2 * PERMIT_BYTES);
        let mut held = budget.reserve(10 * PERMIT_BYTES).await;
        held.shrink_to(3 * PERMIT_BYTES);
        assert_eq!(budget.reserved(), 3 * PERMIT_BYTES);
        assert_eq!(budget.permits.available_permits(), 0);
        held.shrink_to(1);
        assert_eq!(budget.permits.available_permits(), 1);
        assert!(held.try_grow_to(2 * PERMIT_BYTES));
        held.shrink_to(0);
        assert_eq!(budget.reserved(), 0);
        assert_eq!(budget.permits.available_permits(), 2);
        drop(held);
        assert_eq!(budget.permits.available_permits(), 2);
    }

    /// The property that matters in production: with the budget full, the next
    /// merge waits instead of proceeding, and resumes when a holder releases.
    #[tokio::test]
    async fn exhausted_budget_makes_the_next_merge_wait() {
        let budget = MergeMemoryBudget::new(2 * PERMIT_BYTES);
        let held = budget.reserve(2 * PERMIT_BYTES).await;

        let waiter = {
            let budget = budget.clone();
            tokio::spawn(async move { budget.reserve(PERMIT_BYTES).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!waiter.is_finished(), "must wait while the budget is full");

        drop(held);
        let got = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("waiter must proceed once bytes are released")
            .unwrap();
        assert_eq!(got.bytes(), PERMIT_BYTES);
    }

    /// A single merge larger than the whole budget is admitted (alone) rather
    /// than waiting forever; an oversized generation must be folded whole.
    #[tokio::test]
    async fn oversized_request_takes_the_whole_budget_and_proceeds() {
        let budget = MergeMemoryBudget::new(2 * PERMIT_BYTES);
        let big = tokio::time::timeout(Duration::from_secs(2), budget.reserve(10 * PERMIT_BYTES))
            .await
            .expect("an oversized request must not wait forever on an idle budget");
        assert_eq!(budget.permits.available_permits(), 0);
        drop(big);
        assert_eq!(budget.permits.available_permits(), 2);
    }

    #[tokio::test]
    async fn full_budget_growth_fails_without_holding_and_waiting() {
        let budget = MergeMemoryBudget::new(12 * PERMIT_BYTES);
        let mut held = Vec::new();
        for _ in 0..6 {
            held.push(budget.reserve(2 * PERMIT_BYTES).await);
        }
        for reservation in &mut held {
            assert!(!reservation.try_grow_to(2 * PERMIT_BYTES + 1));
            assert_eq!(reservation.bytes(), 2 * PERMIT_BYTES);
        }
        assert_eq!(budget.reserved(), 12 * PERMIT_BYTES);
        drop(held.pop());
        assert!(held[0].try_grow_to(2 * PERMIT_BYTES + 1));
        drop(held);
        assert_eq!(budget.reserved(), 0);
        assert_eq!(budget.permits.available_permits(), 12);
    }

    #[tokio::test]
    async fn queued_initial_reservation_cannot_deadlock_a_growing_holder() {
        let budget = MergeMemoryBudget::new(3 * PERMIT_BYTES);
        let mut held = budget.reserve(2 * PERMIT_BYTES).await;
        let waiter_budget = budget.clone();
        let waiter = tokio::spawn(async move { waiter_budget.reserve(2 * PERMIT_BYTES).await });
        tokio::task::yield_now().await;
        // The fair semaphore may have assigned the free permit to the waiter.
        assert!(!held.try_grow_to(3 * PERMIT_BYTES));
        drop(held);
        let mut next = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap();
        assert!(next.try_grow_to(10 * PERMIT_BYTES));
        assert!(next.try_grow_to(PERMIT_BYTES));
        assert_eq!(next.bytes(), 10 * PERMIT_BYTES);
        drop(next);
        assert_eq!(budget.reserved(), 0);
        assert_eq!(budget.permits.available_permits(), 3);
    }
}
