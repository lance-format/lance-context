//! Process-local legacy merge evidence. These observations never authorize
//! takeover: other processes and pre-upgrade writers remain untracked.
use std::{
    collections::VecDeque,
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use axum::{
    extract::{Query, State},
    Json,
};
use lance_context_core::merge_write_scope::{MergeWork, MergeWriteScope, MergeWriteStatus};
use serde::{Deserialize, Serialize};

const MAX_TRACKED: usize = 256;
const MAX_RETURNED: usize = 128;
const MAX_RESPONSE: usize = 32;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    WaitingSlot,
    Running,
    ReturnedOk,
    ReturnedError,
    Interrupted,
}

struct Entry {
    id: String,
    target: String,
    origin: &'static str,
    started: Instant,
    phase: Phase,
    scope: Arc<MergeWriteScope>,
    started_work: Arc<AtomicBool>,
}

#[derive(Default)]
struct Entries {
    active: Vec<Entry>,
    returned: VecDeque<Entry>,
    untracked: u64,
    evicted: u64,
}

#[derive(Default)]
pub(crate) struct LegacyMerges(Mutex<Entries>);

#[derive(Serialize)]
pub(crate) struct Observation {
    id: String,
    target: String,
    origin: &'static str,
    phase: Phase,
    elapsed_ms: u64,
    completed_steps: u64,
    work: MergeWork,
    writes: MergeWriteStatus,
}

#[derive(Serialize)]
pub(crate) struct Snapshot {
    protocol: u32,
    /// Explicitly false even with an empty, untruncated result.
    takeover_authorized: bool,
    scope: &'static str,
    untracked_admissions: u64,
    evicted_history: u64,
    truncated: bool,
    executions: Vec<Observation>,
}

struct Tracked<'a> {
    registry: &'a LegacyMerges,
    id: String,
    phase: Phase,
}
pub(crate) struct Started(Arc<AtomicBool>);
impl Started {
    pub(crate) fn running(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
impl Drop for Tracked<'_> {
    fn drop(&mut self) {
        let mut entries = self.registry.0.lock().unwrap();
        if let Some(index) = entries.active.iter().position(|e| e.id == self.id) {
            let mut entry = entries.active.swap_remove(index);
            entry.phase = self.phase;
            if entries.returned.len() == MAX_RETURNED {
                entries.returned.pop_front();
                entries.evicted = entries.evicted.saturating_add(1);
            }
            entries.returned.push_back(entry);
        }
    }
}

impl LegacyMerges {
    /// Register before waiting for the slot. Cancellation retains the scope,
    /// including uncertainty when a storage future was cancelled.
    /// Overflow affects telemetry only; ingestion and merge admission keep
    /// their existing limits and are never failed by this registry.
    pub(crate) async fn run<T, E, F, Fut>(
        &self,
        target: &str,
        origin: &'static str,
        waiting_slot: bool,
        work: F,
    ) -> Result<T, E>
    where
        F: FnOnce(Started) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let scope = MergeWriteScope::for_legacy_observation();
        let started_work = Arc::new(AtomicBool::new(false));
        let mut tracked = Tracked {
            registry: self,
            id: uuid::Uuid::new_v4().to_string(),
            phase: Phase::Interrupted,
        };
        {
            let mut entries = self.0.lock().unwrap();
            if entries.active.len() < MAX_TRACKED {
                entries.active.push(Entry {
                    id: tracked.id.clone(),
                    target: target.into(),
                    origin,
                    started: Instant::now(),
                    phase: if waiting_slot {
                        Phase::WaitingSlot
                    } else {
                        Phase::Running
                    },
                    scope: scope.clone(),
                    started_work: started_work.clone(),
                });
            } else {
                entries.untracked = entries.untracked.saturating_add(1);
            }
        }
        let result = scope.run(work(Started(started_work))).await;
        tracked.phase = if result.is_ok() {
            Phase::ReturnedOk
        } else {
            Phase::ReturnedError
        };
        result
    }

    fn snapshot(&self, target: Option<&str>) -> Snapshot {
        let entries = self.0.lock().unwrap();
        let mut matching = entries
            .active
            .iter()
            .chain(entries.returned.iter().rev())
            .filter(|e| target.is_none_or(|t| e.target == t));
        let executions = matching
            .by_ref()
            .take(MAX_RESPONSE)
            .map(|e| Observation {
                id: e.id.clone(),
                target: e.target.clone(),
                origin: e.origin,
                phase: if e.phase == Phase::WaitingSlot && e.started_work.load(Ordering::Relaxed) {
                    Phase::Running
                } else {
                    e.phase
                },
                elapsed_ms: e.started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                completed_steps: e.scope.completed_steps(),
                work: e.scope.work(),
                writes: e.scope.write_status(),
            })
            .collect();
        Snapshot {
            protocol: 1,
            takeover_authorized: false,
            scope: "this_process_legacy_merge_only",
            untracked_admissions: entries.untracked,
            evicted_history: entries.evicted,
            truncated: matching.next().is_some(),
            executions,
        }
    }
}

#[derive(Default, Deserialize)]
pub(crate) struct Filter {
    target: Option<String>,
}

pub(crate) async fn status(
    State(state): State<Arc<crate::state::AppState>>,
    Query(filter): Query<Filter>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "instance": state.merge_executions.instance(),
        "snapshot": state.merge_executions.legacy.snapshot(filter.target.as_deref()),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_core::merge_write_scope::checkpoint;
    use tokio::sync::oneshot;

    #[tokio::test]
    async fn concurrent_progress_is_isolated_and_cancelled_work_is_retained() {
        let registry = Arc::new(LegacyMerges::default());
        let (began, entered) = oneshot::channel();
        let first = registry.clone();
        let frozen = tokio::spawn(async move {
            first
                .run("frozen", "http", true, |started| async move {
                    started.running();
                    checkpoint();
                    began.send(()).unwrap();
                    std::future::pending::<Result<(), ()>>().await
                })
                .await
        });
        entered.await.unwrap();
        let frozen_id = registry.snapshot(Some("frozen")).executions[0].id.clone();
        for _ in 0..20 {
            registry
                .run("healthy", "count_sweeper", false, |_| async {
                    for _ in 0..10 {
                        checkpoint();
                        tokio::task::yield_now().await;
                    }
                    Ok::<_, ()>(())
                })
                .await
                .unwrap();
            let snapshot = registry.snapshot(Some("frozen"));
            assert_eq!(snapshot.executions.len(), 1);
            assert_eq!(snapshot.executions[0].id, frozen_id);
            assert_eq!(snapshot.executions[0].completed_steps, 1);
            assert_eq!(snapshot.executions[0].phase, Phase::Running);
            assert!(!snapshot.takeover_authorized);
        }
        frozen.abort();
        assert!(frozen.await.unwrap_err().is_cancelled());
        let snapshot = registry.snapshot(Some("frozen"));
        assert_eq!(snapshot.executions[0].phase, Phase::Interrupted);
        assert_eq!(snapshot.executions[0].completed_steps, 1);
        assert!(!snapshot.takeover_authorized);
        assert!(registry.snapshot(Some("absent")).executions.is_empty());
        assert!(!registry.snapshot(Some("absent")).takeover_authorized);
    }

    #[tokio::test]
    async fn slot_wait_does_not_report_work_and_errors_are_distinct() {
        let registry = Arc::new(LegacyMerges::default());
        let (began, entered) = oneshot::channel();
        let worker_registry = registry.clone();
        let worker = tokio::spawn(async move {
            worker_registry
                .run("hot", "http", true, |_| async move {
                    began.send(()).unwrap();
                    std::future::pending::<Result<(), ()>>().await
                })
                .await
        });
        entered.await.unwrap();
        let snapshot = registry.snapshot(None);
        assert_eq!(snapshot.executions[0].phase, Phase::WaitingSlot);
        assert_eq!(snapshot.executions[0].completed_steps, 0);
        worker.abort();
        worker.await.unwrap_err();
        registry
            .run("error", "http", false, |_| async { Err::<(), _>(()) })
            .await
            .unwrap_err();
        assert_eq!(
            registry.snapshot(Some("error")).executions[0].phase,
            Phase::ReturnedError
        );
    }

    #[tokio::test]
    async fn history_and_response_are_bounded_and_loss_is_explicit() {
        let registry = LegacyMerges::default();
        for _ in 0..MAX_RETURNED + 5 {
            registry
                .run("hot", "cleanup_sweeper", false, |_| async {
                    Ok::<_, ()>(())
                })
                .await
                .unwrap();
        }
        let snapshot = registry.snapshot(None);
        assert_eq!(snapshot.executions.len(), MAX_RESPONSE);
        assert!(snapshot.truncated);
        assert_eq!(snapshot.evicted_history, 5);
        assert!(!snapshot.takeover_authorized);
        let mut tasks = Vec::new();
        for _ in 0..MAX_TRACKED + 1 {
            let mut future = Box::pin(registry.run("hot", "http", true, |_| {
                std::future::pending::<Result<(), ()>>()
            }));
            assert!(futures::poll!(&mut future).is_pending());
            tasks.push(future);
        }
        let snapshot = registry.snapshot(None);
        assert!(snapshot.truncated);
        assert_eq!(snapshot.untracked_admissions, 1);
        assert_eq!(snapshot.executions.len(), MAX_RESPONSE);
        assert!(snapshot
            .executions
            .iter()
            .all(|e| e.phase == Phase::WaitingSlot));
        drop(tasks);
    }
}
