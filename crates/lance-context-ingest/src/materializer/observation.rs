//! Bounded cumulative phase observations, independent of publication authority.
use serde::Serialize;
use std::{
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Phase {
    Claim,
    Acquire,
    State,
    Prepare,
    Commit,
    Retire,
    Fetch,
    DecodeValidate,
    BlobWrite,
    Admission,
    Collect,
    Compact,
    CollectList,
    CollectProtect,
    CollectDelete,
    AnnotationObserve,
    AnnotationFetch,
    AnnotationDecode,
    AnnotationCreate,
    AnnotationPatch,
    /// WAL append, including exact-identity observation after an unknown reply.
    AnnotationAppend,
    /// Waiting for the local ordered applier to reach an accepted WAL position.
    AnnotationApplyWait,
    /// Upload validation and workspace reservation before reading its body.
    AnnotationUploadAdmission,
    /// Reading the HTTP owner's exact request body after upload admission.
    AnnotationUploadRead,
    /// Request validation and Arrow preparation, including blocking-worker wait.
    AnnotationUploadPrepare,
    /// Owned point lookup and recovery-cut wait, excluding the authority fence.
    AnnotationPointRead,
    /// Quorum publisher-binding lookup and validation, including HTTP health checks.
    AnnotationOwnerFence,
    /// Loading the fixed predecessor during annotation commit resolution.
    AnnotationCommitBase,
    /// Publishing the prepared target, including native validation and catalog selection.
    AnnotationCommitPublish,
    /// Loading and validating the selected target before confirmation.
    AnnotationCommitSelect,
    /// Durably confirming the already selected target in the WAL catalog.
    AnnotationCommitConfirm,
    /// Preparing the immutable body branch, including its workspace and snapshot.
    AnnotationBodyPrepare,
    /// Writing native body fragments inside mixed publication preparation.
    AnnotationBodyWrite,
    /// Committing the private detached body branch before narrow publication.
    AnnotationBodyCommit,
    /// Preparing the frozen narrow annotations against that body snapshot.
    AnnotationNarrowPrepare,
    /// A shared existence lookup for a small prefix of separate create requests.
    AnnotationRegistrationLookup,
    /// Create submissions within AnnotationAppend; counts attempts, not unique rows.
    AnnotationCreateAppend,
    /// Patch submissions within AnnotationAppend; counts attempts, not unique rows.
    AnnotationPatchAppend,
    /// Create submissions waiting for local apply, within AnnotationApplyWait.
    AnnotationCreateApplyWait,
    /// Patch submissions waiting for local apply, within AnnotationApplyWait.
    AnnotationPatchApplyWait,
    /// Native fragment rewrite, including reads and writes, within BlobWrite.
    NativeCompaction,
    /// Native mixed-ingestion body rewrite, within BlobWrite.
    NativeBodyCompaction,
    /// Native session index build or segment merge, within BlobWrite.
    NativeSessionIndex,
    /// Native turn-ID index build or segment merge, within BlobWrite.
    NativeTurnIdIndex,
    /// Native status index build or segment merge, within BlobWrite.
    NativeStatusIndex,
    /// A shared narrow-value lookup for separately applied patch requests.
    AnnotationPatchLookup,
    /// Waiting for reader admission and capturing a compatible point-read view.
    AnnotationPointCapture,
    /// Resolving a captured point view from staging, cache, or narrow Lance data.
    AnnotationPointResolve,
    /// Maintenance selection, including candidate checks, completed-index rebasing,
    /// and recording its plan/preparation; excludes retained-plan completion.
    AnnotationMaintenanceSelect,
    /// Preparing the retained maintenance plan, including native rewrite work.
    AnnotationMaintenancePrepare,
    /// Resolving, installing and finishing the prepared maintenance publication.
    AnnotationMaintenancePublish,
    /// Validating a narrow rewrite and preparing its native index remaps.
    AnnotationCompactionRemap,
    /// Validating and persisting narrow rewrite metadata after index remapping.
    AnnotationCompactionMetadata,
    /// Waiting for the shared ordered gate in apply, freeze or installation.
    AnnotationOrderedWait,
    /// Freezing a checkpoint while holding the ordered gate.
    AnnotationFreezeHold,
    /// Installing a confirmed publication while holding the ordered gate.
    AnnotationInstallHold,
    /// Full-body registration, including core locks and optional cache warming.
    AnnotationFullBodyApply,
    /// Synchronous frame/result accounting after apply, through progress delivery.
    AnnotationApplyHandoff,
    /// Last-reported pressure episodes, through a reason change or progress.
    /// Includes retries; these are not exclusive resource-stall or CPU times.
    AnnotationApplyMemoryPressure,
    AnnotationSourceWindowPressure,
    AnnotationStagingPressure,
    /// A full-body point read waiting for its recovery cut to be applied.
    AnnotationPointRecoveryWait,
    /// Fetching and decoding the exact WAL record captured by a point read.
    AnnotationPointWalFetch,
    /// Looking up a point body in the captured installed Lance version.
    AnnotationPointLanceScan,
}
const NAMES: [&str; 64] = [
    "claim",
    "acquire",
    "state",
    "prepare",
    "commit",
    "retire",
    "fetch",
    "decode_validate",
    "blob_write",
    "admission",
    "collect",
    "compact",
    "collect_list",
    "collect_protect",
    "collect_delete",
    "annotation_observe",
    "annotation_fetch",
    "annotation_decode",
    "annotation_create",
    "annotation_patch",
    "annotation_append",
    "annotation_apply_wait",
    "annotation_upload_admission",
    "annotation_upload_read",
    "annotation_upload_prepare",
    "annotation_point_read",
    "annotation_owner_fence",
    "annotation_commit_base",
    "annotation_commit_publish",
    "annotation_commit_select",
    "annotation_commit_confirm",
    "annotation_body_prepare",
    "annotation_body_write",
    "annotation_body_commit",
    "annotation_narrow_prepare",
    "annotation_registration_lookup",
    "annotation_create_append",
    "annotation_patch_append",
    "annotation_create_apply_wait",
    "annotation_patch_apply_wait",
    "native_compaction",
    "native_body_compaction",
    "native_session_index",
    "native_turn_id_index",
    "native_status_index",
    "annotation_patch_lookup",
    "annotation_point_capture",
    "annotation_point_resolve",
    "annotation_maintenance_select",
    "annotation_maintenance_prepare",
    "annotation_maintenance_publish",
    "annotation_compaction_remap",
    "annotation_compaction_metadata",
    "annotation_ordered_wait",
    "annotation_freeze_hold",
    "annotation_install_hold",
    "annotation_full_body_apply",
    "annotation_apply_handoff",
    "annotation_apply_memory_pressure",
    "annotation_source_window_pressure",
    "annotation_staging_pressure",
    "annotation_point_recovery_wait",
    "annotation_point_wal_fetch",
    "annotation_point_lance_scan",
];
pub const PHASE_COUNT: usize = NAMES.len();

#[derive(Default)]
struct Counter {
    started: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    elapsed_micros: AtomicU64,
    max_elapsed_micros: AtomicU64,
}

pub struct Counters([Counter; PHASE_COUNT]);

impl Default for Counters {
    fn default() -> Self {
        Self(std::array::from_fn(|_| Counter::default()))
    }
}

/// Each field is sampled independently. Parent phases contain child phases;
/// concurrent phase times overlap and must not be summed into elapsed or CPU time.
#[derive(Clone, Debug, Serialize)]
pub struct Observation {
    pub phase: &'static str,
    pub started: u64,
    /// Finished calls, including the failed subset below; excludes cancellation.
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    /// Approximate in-flight count from independently sampled counters.
    /// Its elapsed time is not included in elapsed_micros until it finishes.
    pub active: u64,
    /// Includes failed/cancelled work, but not time still in flight.
    pub elapsed_micros: u64,
    /// Longest finished interval, including failed/cancelled work.
    pub max_elapsed_micros: u64,
}

// Serde implements fixed arrays only through length 32. Serialize the bounded
// observation array as a slice without allocating or changing its JSON shape.
pub fn serialize_observations<S: serde::Serializer>(
    observations: &[Observation; PHASE_COUNT],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    observations.as_slice().serialize(serializer)
}

impl Counters {
    pub fn observation(&self) -> [Observation; PHASE_COUNT] {
        std::array::from_fn(|i| {
            let c = &self.0[i];
            let started = c.started.load(Ordering::Relaxed);
            let completed = c.completed.load(Ordering::Relaxed);
            let cancelled = c.cancelled.load(Ordering::Relaxed);
            Observation {
                phase: NAMES[i],
                started,
                completed,
                failed: c.failed.load(Ordering::Relaxed),
                cancelled,
                active: started.saturating_sub(completed.saturating_add(cancelled)),
                elapsed_micros: c.elapsed_micros.load(Ordering::Relaxed),
                max_elapsed_micros: c.max_elapsed_micros.load(Ordering::Relaxed),
            }
        })
    }

    pub async fn measure<T, E>(
        self: &Arc<Self>,
        phase: Phase,
        future: impl Future<Output = Result<T, E>>,
    ) -> Result<T, E> {
        let mut guard = self.enter(phase);
        let result = future.await;
        guard.success = Some(result.is_ok());
        result
    }

    pub(crate) fn blocking<T, E>(
        self: &Arc<Self>,
        phase: Phase,
        work: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let mut guard = self.enter(phase);
        let result = work();
        guard.success = Some(result.is_ok());
        result
    }

    pub(crate) fn enter(self: &Arc<Self>, phase: Phase) -> Guard {
        self.0[phase as usize]
            .started
            .fetch_add(1, Ordering::Relaxed);
        Guard {
            counters: self.clone(),
            phase,
            start: Instant::now(),
            success: None,
        }
    }
}

pub(crate) struct Guard {
    counters: Arc<Counters>,
    phase: Phase,
    start: Instant,
    success: Option<bool>,
}
impl Guard {
    /// End a successful synchronous section or a resolved pressure episode.
    /// Dropping an unfinished guard continues to record cancellation.
    pub(crate) fn complete(mut self) {
        self.success = Some(true);
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        let c = &self.counters.0[self.phase as usize];
        let elapsed = self.start.elapsed().as_micros().min(u64::MAX as u128) as u64;
        c.elapsed_micros.fetch_add(elapsed, Ordering::Relaxed);
        c.max_elapsed_micros.fetch_max(elapsed, Ordering::Relaxed);
        match self.success {
            Some(success) => {
                if !success {
                    c.failed.fetch_add(1, Ordering::Relaxed);
                }
                c.completed.fetch_add(1, Ordering::Relaxed);
            }
            None => {
                c.cancelled.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancelled_observation_is_not_successful_work() {
        let counters = Arc::new(Counters::default());
        let owned = counters.clone();
        let (entered, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            owned
                .measure(Phase::Fetch, async {
                    entered.send(()).unwrap();
                    std::future::pending::<Result<(), ()>>().await
                })
                .await
        });
        ready.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let cancelled = &counters.observation()[Phase::Fetch as usize];
        assert_eq!(cancelled.started, 1);
        assert_eq!(cancelled.cancelled, 1);
        assert_eq!(cancelled.completed, 0);
        assert_eq!(cancelled.failed, 0);
        assert!(counters
            .measure(Phase::Fetch, async { Err::<(), _>(()) })
            .await
            .is_err());
        let failed = &counters.observation()[Phase::Fetch as usize];
        assert_eq!(failed.started, 2);
        assert_eq!(failed.completed, 1);
        assert_eq!(failed.failed, 1);
        assert_eq!(failed.cancelled, 1);
    }
}
