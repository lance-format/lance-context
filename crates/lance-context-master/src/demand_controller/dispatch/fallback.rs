//! One bounded grace period per queued task, with explicit scheduler wakeups.
use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

#[derive(Default)]
pub(super) struct DeferredFallbacks {
    grace: Duration,
    deadlines: BTreeMap<String, Instant>,
    wakes: BTreeSet<(Instant, String)>,
}

impl DeferredFallbacks {
    pub(super) fn new(grace: Duration) -> Self {
        Self {
            grace,
            ..Self::default()
        }
    }

    pub(super) fn defer(&mut self, id: &str, now: Instant) -> bool {
        if self.grace.is_zero() {
            return false;
        }
        let deadline = *self.deadlines.entry(id.into()).or_insert_with(|| {
            let deadline = now + self.grace;
            self.wakes.insert((deadline, id.into()));
            deadline
        });
        if now >= deadline {
            self.wakes.remove(&(deadline, id.into()));
            false
        } else {
            // An earlier resident offer may have canceled this wake before
            // being rejected. Restore the same deadline, never a new one.
            self.wakes.insert((deadline, id.into()));
            true
        }
    }

    pub(super) fn next_deadline(&self) -> Option<Instant> {
        self.wakes.first().map(|(at, _)| *at)
    }

    pub(super) fn retain(&mut self, still_queued: impl Fn(&str) -> bool) {
        self.deadlines.retain(|id, _| still_queued(id));
        self.wakes.retain(|(_, id)| self.deadlines.contains_key(id));
    }

    /// Bound a scheduler step even when many deadlines expire together. Keep
    /// the remembered deadline until queue deletion: unrelated Watch traffic
    /// or an unsuccessful offer must never restart the grace period.
    pub(super) fn expire(&mut self, now: Instant) {
        for _ in 0..64 {
            if self.next_deadline().is_none_or(|at| at > now) {
                break;
            }
            self.wakes.pop_first();
        }
    }

    pub(super) fn offered(&mut self, id: &str) {
        if let Some(deadline) = self.deadlines.get(id) {
            self.wakes.remove(&(*deadline, id.into()));
        }
    }

    pub(super) fn forget(&mut self, id: &str) {
        if let Some(deadline) = self.deadlines.remove(id) {
            self.wakes.remove(&(deadline, id.into()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_busy_or_missing_resident_cannot_extend_the_fallback_deadline() {
        let start = Instant::now();
        let grace = Duration::from_millis(500);
        let mut pending = DeferredFallbacks::new(grace);
        assert!(pending.defer("merge", start));
        for ms in [6, 70, 150, 362, 499] {
            assert!(pending.defer("merge", start + Duration::from_millis(ms)));
            assert_eq!(pending.next_deadline(), Some(start + grace));
        }
        pending.expire(start + grace);
        assert_eq!(pending.next_deadline(), None);
        assert!(!pending.defer("merge", start + grace));
        assert!(!pending.defer("merge", start + Duration::from_secs(60)));
        pending.forget("merge");
        assert!(pending.deadlines.is_empty());
    }

    #[test]
    fn independent_deadlines_and_queue_deletion_do_not_leak_or_starve() {
        let start = Instant::now();
        let grace = Duration::from_millis(500);
        let mut pending = DeferredFallbacks::new(grace);
        for n in 0..130 {
            assert!(pending.defer(&n.to_string(), start));
        }
        assert!(pending.defer("later", start + grace));
        pending.expire(start + grace);
        assert_eq!(pending.wakes.len(), 67);
        assert_eq!(pending.next_deadline(), Some(start + grace));
        pending.expire(start + grace);
        pending.expire(start + grace);
        assert_eq!(pending.next_deadline(), Some(start + grace * 2));
        pending.offered("later");
        assert_eq!(pending.next_deadline(), None);
        assert!(pending.defer("later", start + grace + Duration::from_millis(1)));
        assert_eq!(pending.next_deadline(), Some(start + grace * 2));
        // A rejected offer does not get another grace period.
        assert!(!pending.defer("later", start + grace * 2));
        pending.forget("later");
        for n in 0..130 {
            pending.forget(&n.to_string());
        }
        assert!(pending.deadlines.is_empty());
        assert!(pending.wakes.is_empty());
        assert!(!DeferredFallbacks::default().defer("disabled", start));
    }
}
