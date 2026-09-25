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
//! # Why it cannot deadlock
//!
//! Every merge takes its full initial reservation --
//! `min(merge_max_bytes, budget)` -- **before** it reads anything, in one
//! atomic acquire. A merge therefore never holds part of what it needs while
//! waiting for the rest. The only case that grows a reservation mid-read is a
//! single generation larger than `merge_max_bytes`, which the merge folds whole
//! (a generation is indivisible). At that point it already holds at least as
//! much as any other merge could be waiting for, so growing cannot form a
//! cycle. If it needs more than the whole budget it is admitted anyway once it
//! is the sole holder, mirroring the "lone oversized request" rule of the blob
//! budget: an oversized generation must make progress or the shard wedges.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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
        }
    }

    fn permits_total(&self) -> u32 {
        Self::permits_for(self.limit)
    }
}

impl MergeReservation {
    /// Bytes this reservation currently accounts for.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Grow the reservation to cover `new_total` bytes, waiting for the
    /// additional permits if the budget is exhausted. A no-op when `new_total`
    /// does not exceed what is already held. The extra is capped at the budget:
    /// a reservation never needs more permits than exist, and a holder that
    /// already has them all is the sole merge in flight.
    pub async fn grow_to(&mut self, new_total: usize) {
        if new_total <= self.bytes {
            return;
        }
        let want_total = MergeMemoryBudget::permits_for(new_total).min(self.budget.permits_total());
        let extra = want_total.saturating_sub(self.permits);
        if extra > 0 {
            let permit = self
                .budget
                .permits
                .acquire_many(extra)
                .await
                .expect("merge memory budget semaphore is never closed");
            permit.forget();
            self.permits += extra;
        }
        self.budget
            .reserved
            .fetch_add(new_total - self.bytes, Ordering::AcqRel);
        self.bytes = new_total;
    }
}

impl Drop for MergeReservation {
    fn drop(&mut self) {
        self.budget.permits.add_permits(self.permits as usize);
        self.budget.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn reservations_release_on_drop() {
        let budget = MergeMemoryBudget::new(4 * PERMIT_BYTES);
        let a = budget.reserve(2 * PERMIT_BYTES).await;
        assert_eq!(budget.reserved(), 2 * PERMIT_BYTES);
        drop(a);
        assert_eq!(budget.reserved(), 0);
        assert_eq!(budget.permits.available_permits(), 4);
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
    async fn grow_to_waits_for_the_extra_and_is_idempotent_downward() {
        let budget = MergeMemoryBudget::new(3 * PERMIT_BYTES);
        let mut a = budget.reserve(PERMIT_BYTES).await;
        let b = budget.reserve(2 * PERMIT_BYTES).await;

        let grow = {
            // Growing `a` to 2 MiB needs one more permit; none free until `b` drops.
            let handle = tokio::spawn(async move {
                a.grow_to(2 * PERMIT_BYTES).await;
                a
            });
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert!(!handle.is_finished());
            handle
        };
        drop(b);
        let a = tokio::time::timeout(Duration::from_secs(5), grow)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(a.bytes(), 2 * PERMIT_BYTES);
        assert_eq!(budget.reserved(), 2 * PERMIT_BYTES);

        let mut a = a;
        a.grow_to(PERMIT_BYTES).await; // smaller: no-op
        assert_eq!(a.bytes(), 2 * PERMIT_BYTES);
    }
}
