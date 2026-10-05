//! Executor-local admission and joined shutdown, independent of table ownership.
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub executor_id: String,
    pub accepting: bool,
    /// Includes claim RPCs in flight, admitted tasks, and scanner passes.
    pub active_operations: usize,
}

impl Status {
    pub fn drained(&self) -> bool {
        !self.accepting && self.active_operations == 0
    }
}

pub struct Admission {
    state: watch::Sender<Status>,
}

impl Default for Admission {
    fn default() -> Self {
        let (state, _) = watch::channel(Status {
            executor_id: lance_context_core::generate_id(),
            accepting: true,
            active_operations: 0,
        });
        Self { state }
    }
}

impl Admission {
    pub fn status(&self) -> Status {
        self.state.borrow().clone()
    }

    /// The same lock closes admission and reserves an operation before its
    /// first ownership RPC. Drain cannot miss an accepted but delayed claim.
    pub fn try_admit(self: &Arc<Self>) -> Option<Operation> {
        let mut admitted = false;
        self.state.send_if_modified(|state| {
            if state.accepting {
                state.active_operations += 1;
                admitted = true;
            }
            admitted
        });
        admitted.then(|| Operation(self.clone()))
    }

    pub fn begin_drain(&self) -> Status {
        self.state.send_modify(|state| state.accepting = false);
        self.status()
    }

    /// No elapsed-time cancellation: admitted writers retain their existing
    /// progress watchdog, lease and fencing protocol until they return.
    pub async fn wait_drained(&self) {
        let mut updates = self.state.subscribe();
        updates
            .wait_for(Status::drained)
            .await
            .expect("admission sender remains alive");
    }
}

pub struct Operation(Arc<Admission>);

impl Drop for Operation {
    fn drop(&mut self) {
        self.0
            .state
            .send_modify(|state| state.active_operations -= 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn drain_waits_for_admitted_claim_and_rejects_new_work() {
        let admission = Arc::new(Admission::default());
        let claim_in_flight = admission.try_admit().unwrap();
        let running_writer = admission.try_admit().unwrap();
        let snapshot = admission.begin_drain();
        assert!(!snapshot.accepting);
        assert_eq!(snapshot.active_operations, 2);
        assert!(admission.try_admit().is_none());
        let waiter = tokio::spawn({
            let admission = admission.clone();
            async move { admission.wait_drained().await }
        });
        drop(claim_in_flight);
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        drop(running_writer);
        waiter.await.unwrap();
        assert!(admission.status().drained());
        assert_eq!(admission.begin_drain().executor_id, snapshot.executor_id);
        assert!(admission.try_admit().is_none());
    }

    #[tokio::test]
    async fn drained_notification_before_subscription_is_not_lost() {
        let admission = Arc::new(Admission::default());
        admission.begin_drain();
        admission.wait_drained().await;
    }
}
