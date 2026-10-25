//! Executor-local admission and joined shutdown, independent of table ownership.
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub executor_id: String,
    pub accepting: bool,
    /// Includes claim RPCs in flight, admitted tasks, and scanner passes.
    pub active_operations: usize,
    /// Local diagnostics only. Neither an empty list nor stale progress fences
    /// storage writes or authorizes cancellation of another executor.
    pub operations: BTreeMap<String, OperationStatus>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OperationStatus {
    pub kind: &'static str,
    pub phase: &'static str,
    pub task_id: Option<String>,
    pub target: Option<String>,
    pub started_at_ms: i64,
    /// None means this operation has no instrumented progress counter. It
    /// must not be interpreted as zero progress or permission to kill it.
    pub progress: Option<Progress>,
    /// Claim diagnostics are distinct from execution progress and never authorize fencing.
    pub claim_progress: Option<ClaimProgress>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ClaimProgress {
    pub stage: &'static str,
    pub completed_etcd_responses: u64,
    pub queue_candidates: u64,
    pub recovery_candidates: u64,
    pub last_completed_at_ms: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Progress {
    pub unit: &'static str,
    pub completed: u64,
    pub last_completed_at_ms: i64,
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
            operations: BTreeMap::new(),
        });
        Self { state }
    }
}

impl Admission {
    pub(crate) fn is_accepting(&self) -> bool {
        self.state.borrow().accepting
    }

    pub(crate) async fn wait_closed(&self) {
        let mut updates = self.state.subscribe();
        updates
            .wait_for(|status| !status.accepting)
            .await
            .expect("admission sender remains alive");
    }

    pub fn status(&self) -> Status {
        self.state.borrow().clone()
    }

    /// The same lock closes admission and reserves an operation before its
    /// first ownership RPC. Drain cannot miss an accepted but delayed claim.
    #[cfg(test)]
    pub fn try_admit(self: &Arc<Self>) -> Option<Operation> {
        self.try_admit_named("unspecified")
    }

    pub fn try_admit_named(self: &Arc<Self>, kind: &'static str) -> Option<Operation> {
        let mut admitted = false;
        let id = lance_context_core::generate_id();
        self.state.send_if_modified(|state| {
            if state.accepting {
                state.active_operations += 1;
                state.operations.insert(
                    id.clone(),
                    OperationStatus {
                        kind,
                        phase: "admitted",
                        task_id: None,
                        target: None,
                        started_at_ms: chrono::Utc::now().timestamp_millis(),
                        progress: None,
                        claim_progress: None,
                    },
                );
                admitted = true;
            }
            admitted
        });
        admitted.then(|| Operation {
            admission: self.clone(),
            id,
        })
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

pub struct Operation {
    admission: Arc<Admission>,
    id: String,
}

impl Operation {
    pub fn phase(&self, phase: &'static str) {
        self.update(|operation| operation.phase = phase);
    }

    pub fn target(&self, target: &str) {
        self.update(|operation| operation.target = Some(target.to_owned()));
    }

    pub fn task(&self, id: &str, target: &str) {
        self.update(|operation| {
            operation.task_id = Some(id.to_owned());
            operation.target = Some(target.to_owned());
            operation.phase = "running";
        });
    }

    /// Record completed work, never a heartbeat, poll, lease renewal, or retry.
    pub fn observed_table(&self) {
        self.update(|operation| {
            let progress = operation.progress.get_or_insert(Progress {
                unit: "tables_observed",
                completed: 0,
                last_completed_at_ms: 0,
            });
            progress.completed = progress.completed.saturating_add(1);
            progress.last_completed_at_ms = chrono::Utc::now().timestamp_millis();
        });
    }

    /// One sample in which this operation completed additional IO or logical
    /// work. The unit is samples, not bytes, polls, or another task's progress.
    pub(crate) fn observed_work(&self) {
        self.update(|operation| {
            let progress = operation.progress.get_or_insert(Progress {
                unit: "work_progress_samples",
                completed: 0,
                last_completed_at_ms: 0,
            });
            progress.completed = progress.completed.saturating_add(1);
            progress.last_completed_at_ms = chrono::Utc::now().timestamp_millis();
        });
    }

    pub(crate) fn claim_phase(&self, stage: &'static str) {
        self.update(|operation| {
            operation
                .claim_progress
                .get_or_insert_with(ClaimProgress::default)
                .stage = stage;
        });
    }

    /// Completed IO responses / inspected candidates, never a poll heartbeat.
    /// This does not increment the running merge's independent work counter.
    pub(crate) fn claim_completed(&self, responses: u64, queued: u64, recovery: u64) {
        self.update(|operation| {
            let p = operation
                .claim_progress
                .get_or_insert_with(ClaimProgress::default);
            p.completed_etcd_responses += responses;
            p.queue_candidates += queued;
            p.recovery_candidates += recovery;
            if responses + queued + recovery > 0 {
                p.last_completed_at_ms = Some(chrono::Utc::now().timestamp_millis());
            }
        });
    }

    fn update(&self, update: impl FnOnce(&mut OperationStatus)) {
        self.admission.state.send_modify(|state| {
            // The operation guard owns this entry until Drop. No separate
            // registry lock can disagree with the admission/drain count.
            update(
                state
                    .operations
                    .get_mut(&self.id)
                    .expect("admitted operation"),
            );
        });
    }
}

impl Drop for Operation {
    fn drop(&mut self) {
        self.admission.state.send_modify(|state| {
            state.operations.remove(&self.id);
            state.active_operations -= 1;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn claim_progress_is_local_and_does_not_become_merge_work() {
        let admission = Arc::new(Admission::default());
        let first = admission.try_admit_named("merge").unwrap();
        let second = admission.try_admit_named("merge").unwrap();
        first.claim_phase("admission_snapshot");
        second.claim_phase("queue_page");
        let second_id = second.id.clone();
        let advancing = tokio::spawn(async move {
            for _ in 0..100 {
                second.claim_completed(1, 64, 0);
                tokio::task::yield_now().await;
            }
            second
        });
        for _ in 0..100 {
            let status = admission.status();
            let stalled = status.operations[&first.id]
                .claim_progress
                .as_ref()
                .unwrap();
            assert_eq!(stalled.completed_etcd_responses, 0);
            assert!(stalled.last_completed_at_ms.is_none());
            assert!(status.operations[&second_id].progress.is_none());
            tokio::task::yield_now().await;
        }
        let second = advancing.await.unwrap();
        let status = admission.status();
        let stalled = &status.operations[&first.id];
        assert_eq!(
            stalled
                .claim_progress
                .as_ref()
                .unwrap()
                .completed_etcd_responses,
            0
        );
        assert!(stalled
            .claim_progress
            .as_ref()
            .unwrap()
            .last_completed_at_ms
            .is_none());
        assert_eq!(
            status.operations[&second.id]
                .claim_progress
                .as_ref()
                .unwrap()
                .queue_candidates,
            6400
        );
        second.task("owned-task", "hot");
        assert!(admission.status().operations[&second.id].progress.is_none());
        second.observed_work();
        assert_eq!(
            admission.status().operations[&second.id]
                .progress
                .as_ref()
                .unwrap()
                .completed,
            1
        );
    }

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

    #[tokio::test]
    async fn concurrent_progress_does_not_mask_a_different_stalled_operation() {
        let admission = Arc::new(Admission::default());
        let merge = admission.try_admit_named("merge").unwrap();
        merge.phase("claiming");
        merge.task("task-1", "hot-table");
        let scan = admission.try_admit_named("stats_scan").unwrap();
        scan.phase("observe_tables");
        let scan_id = scan.id.clone();
        let merge_id = merge.id.clone();
        admission.begin_drain();
        let (done, wait_done) = tokio::sync::oneshot::channel();
        let (release, wait_release) = tokio::sync::oneshot::channel();
        let scanner = tokio::spawn(async move {
            for _ in 0..100 {
                scan.observed_table();
                tokio::task::yield_now().await;
            }
            done.send(()).unwrap();
            wait_release.await.unwrap();
            drop(scan);
        });
        for _ in 0..100 {
            let status = admission.status();
            assert_eq!(status.active_operations, status.operations.len());
            let pending = &status.operations[&merge_id];
            assert_eq!(pending.task_id.as_deref(), Some("task-1"));
            assert_eq!(pending.target.as_deref(), Some("hot-table"));
            assert!(
                pending.progress.is_none(),
                "another operation cannot show merge progress"
            );
            assert!(!status.drained());
            tokio::task::yield_now().await;
        }
        wait_done.await.unwrap();
        let status = admission.status();
        assert_eq!(
            status.operations[&scan_id]
                .progress
                .as_ref()
                .unwrap()
                .completed,
            100
        );
        let before = serde_json::to_value(&status.operations[&scan_id].progress).unwrap();
        // Status reads and phase transitions are not progress events.
        merge.phase("running");
        assert_eq!(
            before,
            serde_json::to_value(&admission.status().operations[&scan_id].progress).unwrap()
        );
        assert!(admission.try_admit_named("new_work").is_none());
        drop(merge);
        assert_eq!(admission.status().active_operations, 1);
        assert!(!admission.status().drained());
        release.send(()).unwrap();
        scanner.await.unwrap();
        admission.wait_drained().await;
        assert!(admission.status().operations.is_empty());
    }
}
