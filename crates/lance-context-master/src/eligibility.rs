//! Pure admission decisions for queued maintenance tasks.
//!
//! `claim_next` in `task_store.rs` scans the etcd queue and, for each
//! candidate, decides whether this poller may attempt the claim transaction.
//! Those decisions used to be inlined as a chain of `continue`s. They are
//! collected here as functions of plain inputs so they can be unit tested
//! without etcd and so every skip has a nameable reason.
//!
//! Nothing in this module reads or writes etcd. The final claim transaction
//! still re-checks every ownership predicate; this is the cheap pre-filter.

use lance_context_api::TaskKind;
use lance_context_merge::rollout::MergeRollout;

/// Why a queued task was not attempted by this poller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// The poller's kind set does not include this task kind.
    KindNotRunnable,
    /// A resident-only poller saw a task outside the owned, non-draining local scope.
    NotResidentEligible,
    /// A prepared compaction or index is waiting for its commit turn on
    /// this table; merge yields one admission turn.
    YieldToPreparedCommit,
    /// Another claim, preparation, catch-up owner or write owner already
    /// holds what this task needs.
    OwnershipBlocked,
    /// Only MergeWal reconciles a worker (non-maintenance) execution.
    WorkerExecutionNeedsMerge,
}

/// Static per-target scheduling configuration consulted by the filter.
pub struct Policy<'a> {
    pub rollout: &'a MergeRollout,
    pub compaction_prepare_targets: &'a [String],
    pub index_prepare_targets: &'a [String],
    pub maintenance_catchup_targets: &'a [String],
}

/// The seven ownership keys `claim_next` snapshots for one candidate.
/// Each is the raw etcd value or `None` when the key is absent.
pub struct Snapshot<'a> {
    pub merge_execution: Option<&'a [u8]>,
    pub target_lock: Option<&'a [u8]>,
    pub catchup_active: Option<&'a [u8]>,
    pub merge_claim: Option<&'a [u8]>,
    pub task_claim: Option<&'a [u8]>,
    pub compact_preparation: Option<&'a [u8]>,
    pub compact_commit_ready: Option<&'a [u8]>,
}

/// How the candidate would be claimed if admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimShape {
    /// Compact/IndexId preparing outside the table write lock.
    pub preparing: bool,
    /// Whether the claim takes the table write lock.
    pub write_claim: bool,
    /// Catch-up/maintenance sharing lets this maintenance task run while an
    /// external catch-up owner is active on the table.
    pub shared_owner: bool,
}

fn listed(targets: &[String], target: &str) -> bool {
    targets.iter().any(|t| t == "*" || t == target)
}

fn is_generic(target: &str) -> bool {
    target.starts_with("generic:")
}

pub(crate) fn requires_target_lock(kind: TaskKind) -> bool {
    matches!(
        kind,
        TaskKind::Compact | TaskKind::IndexId | TaskKind::Repair | TaskKind::MergeWal
    )
}

/// Filter applied before any etcd read beyond the queue page itself.
pub fn pre_dependency_filter(
    kind: TaskKind,
    target: &str,
    runnable: impl Fn(TaskKind) -> bool,
    resident_targets: Option<&[String]>,
    policy: &Policy<'_>,
) -> Result<(), Skip> {
    if !runnable(kind) {
        return Err(Skip::KindNotRunnable);
    }
    if let Some(targets) = resident_targets {
        let eligible = kind == TaskKind::MergeWal
            && policy.rollout.owned(target)
            && !policy.rollout.draining(target)
            && (targets.iter().any(|t| t == target)
                || (!is_generic(target) && listed(targets, target)));
        if !eligible {
            return Err(Skip::NotResidentEligible);
        }
    }
    Ok(())
}

/// Decide the claim shape from kind, target and policy alone.
pub fn claim_shape(
    kind: TaskKind,
    target: &str,
    policy: &Policy<'_>,
    catchup_active: bool,
) -> ClaimShape {
    let preparation_targets: &[String] = match kind {
        TaskKind::Compact => policy.compaction_prepare_targets,
        TaskKind::IndexId => policy.index_prepare_targets,
        _ => &[],
    };
    let maintenance = matches!(kind, TaskKind::Compact | TaskKind::IndexId);
    let preparing = maintenance
        && !is_generic(target)
        && !policy.rollout.draining(target)
        && listed(preparation_targets, target);
    let shared_owner = catchup_active
        && maintenance
        && policy.rollout.owned(target)
        && !policy.rollout.draining(target)
        && !is_generic(target)
        && policy
            .maintenance_catchup_targets
            .iter()
            .any(|t| t == target);
    ClaimShape {
        preparing,
        write_claim: requires_target_lock(kind) && !preparing,
        shared_owner,
    }
}

/// Filter applied after the seven-key ownership snapshot.
///
/// `expected_owner` is the `merge-execution:{id}` owner string when a merge
/// execution record exists; `execution_is_worker` says that record has no
/// maintenance kind. `expected_job` is the catch-up job a dedicated
/// executor claims under, if any.
pub fn admission_filter(
    kind: TaskKind,
    shape: ClaimShape,
    snapshot: &Snapshot<'_>,
    expected_owner: Option<&str>,
    execution_is_worker: bool,
    expected_job: Option<&[u8]>,
) -> Result<(), Skip> {
    let has_execution = snapshot.merge_execution.is_some();
    if kind == TaskKind::MergeWal
        && !has_execution
        && snapshot.compact_commit_ready.is_some()
        && snapshot.compact_commit_ready == snapshot.compact_preparation
    {
        return Err(Skip::YieldToPreparedCommit);
    }
    let expected_job = expected_job.or(if shape.shared_owner {
        snapshot.catchup_active
    } else {
        None
    });
    if snapshot.task_claim.is_some()
        || (shape.preparing
            && (snapshot.compact_preparation.is_some()
                || (snapshot.catchup_active.is_some() && !shape.shared_owner)))
        || (shape.write_claim
            && (snapshot.catchup_active != expected_job
                || snapshot.merge_claim.is_some()
                || snapshot.target_lock != expected_owner.map(str::as_bytes)))
    {
        return Err(Skip::OwnershipBlocked);
    }
    if !shape.preparing && kind != TaskKind::MergeWal && has_execution && execution_is_worker {
        return Err(Skip::WorkerExecutionNeedsMerge);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rollout(owned: &[&str], draining: &[&str]) -> MergeRollout {
        MergeRollout {
            owned_targets: owned.iter().map(|s| s.to_string()).collect(),
            drain_targets: draining.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    const EMPTY: Snapshot<'static> = Snapshot {
        merge_execution: None,
        target_lock: None,
        catchup_active: None,
        merge_claim: None,
        task_claim: None,
        compact_preparation: None,
        compact_commit_ready: None,
    };

    #[test]
    fn resident_filter_only_admits_owned_opted_in_non_draining_merges() {
        let rollout = rollout(
            &["hot", "draining", "excluded", "generic:gs"],
            &["draining"],
        );
        let policy = Policy {
            rollout: &rollout,
            compaction_prepare_targets: &[],
            index_prepare_targets: &[],
            maintenance_catchup_targets: &[],
        };
        let targets = strings(&["hot", "draining", "generic:gs"]);
        let run = |k| k == TaskKind::MergeWal;
        let check =
            |kind, target| pre_dependency_filter(kind, target, run, Some(&targets), &policy);
        assert_eq!(check(TaskKind::MergeWal, "hot"), Ok(()));
        assert_eq!(
            check(TaskKind::MergeWal, "legacy"),
            Err(Skip::NotResidentEligible)
        );
        assert_eq!(
            check(TaskKind::MergeWal, "draining"),
            Err(Skip::NotResidentEligible)
        );
        assert_eq!(
            check(TaskKind::MergeWal, "excluded"),
            Err(Skip::NotResidentEligible)
        );
        assert_eq!(check(TaskKind::MergeWal, "generic:gs"), Ok(()));
        assert_eq!(check(TaskKind::Compact, "hot"), Err(Skip::KindNotRunnable));
        let wildcard = strings(&["*"]);
        assert_eq!(
            pre_dependency_filter(
                TaskKind::MergeWal,
                "excluded",
                run,
                Some(&wildcard),
                &policy
            ),
            Ok(())
        );
        assert_eq!(
            pre_dependency_filter(TaskKind::MergeWal, "legacy", run, None, &policy),
            Ok(())
        );
    }

    #[test]
    fn claim_shape_prepares_only_opted_in_maintenance_on_live_rollouts() {
        let rollout = rollout(&["t", "d"], &["d"]);
        let compact = strings(&["t", "d"]);
        let index = strings(&["*"]);
        let policy = Policy {
            rollout: &rollout,
            compaction_prepare_targets: &compact,
            index_prepare_targets: &index,
            maintenance_catchup_targets: &[],
        };
        let shape = claim_shape(TaskKind::Compact, "t", &policy, false);
        assert!(shape.preparing && !shape.write_claim);
        let shape = claim_shape(TaskKind::Compact, "d", &policy, false);
        assert!(
            !shape.preparing && shape.write_claim,
            "draining never prepares"
        );
        let shape = claim_shape(TaskKind::IndexId, "generic:x", &policy, false);
        assert!(
            !shape.preparing && shape.write_claim,
            "generic never prepares"
        );
        let shape = claim_shape(TaskKind::IndexId, "anything", &policy, false);
        assert!(shape.preparing, "wildcard applies");
        let shape = claim_shape(TaskKind::MergeWal, "t", &policy, false);
        assert!(!shape.preparing && shape.write_claim);
        let shape = claim_shape(TaskKind::Repair, "t", &policy, false);
        assert!(!shape.preparing && shape.write_claim);
    }

    #[test]
    fn shared_owner_requires_catchup_active_and_exact_listing() {
        let rollout = rollout(&["t"], &[]);
        let shared = strings(&["t"]);
        let policy = Policy {
            rollout: &rollout,
            compaction_prepare_targets: &[],
            index_prepare_targets: &[],
            maintenance_catchup_targets: &shared,
        };
        assert!(claim_shape(TaskKind::Compact, "t", &policy, true).shared_owner);
        assert!(!claim_shape(TaskKind::Compact, "t", &policy, false).shared_owner);
        assert!(!claim_shape(TaskKind::MergeWal, "t", &policy, true).shared_owner);
        let wildcard = strings(&["*"]);
        let policy = Policy {
            maintenance_catchup_targets: &wildcard,
            ..policy
        };
        assert!(
            !claim_shape(TaskKind::Compact, "t", &policy, true).shared_owner,
            "catch-up sharing is exact-match only"
        );
    }

    #[test]
    fn merge_yields_to_matching_prepared_commit_only_without_execution() {
        let shape = ClaimShape {
            preparing: false,
            write_claim: true,
            shared_owner: false,
        };
        let ready = Snapshot {
            compact_preparation: Some(b"tok"),
            compact_commit_ready: Some(b"tok"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(TaskKind::MergeWal, shape, &ready, None, false, None),
            Err(Skip::YieldToPreparedCommit)
        );
        let stale = Snapshot {
            compact_preparation: Some(b"new"),
            compact_commit_ready: Some(b"old"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(TaskKind::MergeWal, shape, &stale, None, false, None),
            Ok(()),
            "a hint from a dead preparer is ignored"
        );
        let recovering = Snapshot {
            merge_execution: Some(b"{}"),
            target_lock: Some(b"merge-execution:x"),
            ..ready
        };
        assert_eq!(
            admission_filter(
                TaskKind::MergeWal,
                shape,
                &recovering,
                Some("merge-execution:x"),
                true,
                None
            ),
            Ok(()),
            "recovery is never blocked by a fairness hint"
        );
        let compact = ClaimShape {
            preparing: true,
            write_claim: false,
            shared_owner: false,
        };
        assert_eq!(
            admission_filter(TaskKind::Compact, compact, &EMPTY, None, false, None),
            Ok(()),
            "the hint only affects merge"
        );
    }

    #[test]
    fn ownership_predicates_block_as_before() {
        let write = ClaimShape {
            preparing: false,
            write_claim: true,
            shared_owner: false,
        };
        assert_eq!(
            admission_filter(TaskKind::MergeWal, write, &EMPTY, None, false, None),
            Ok(())
        );
        let claimed = Snapshot {
            task_claim: Some(b"x"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(TaskKind::MergeWal, write, &claimed, None, false, None),
            Err(Skip::OwnershipBlocked)
        );
        let locked = Snapshot {
            target_lock: Some(b"other"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(TaskKind::MergeWal, write, &locked, None, false, None),
            Err(Skip::OwnershipBlocked)
        );
        let merge_claimed = Snapshot {
            merge_claim: Some(b"m"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(TaskKind::MergeWal, write, &merge_claimed, None, false, None),
            Err(Skip::OwnershipBlocked)
        );
        let catchup = Snapshot {
            catchup_active: Some(b"job-1"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(TaskKind::MergeWal, write, &catchup, None, false, None),
            Err(Skip::OwnershipBlocked)
        );
        assert_eq!(
            admission_filter(
                TaskKind::MergeWal,
                write,
                &catchup,
                None,
                false,
                Some(b"job-1")
            ),
            Ok(()),
            "the dedicated executor claims under its own job"
        );
        let owned = Snapshot {
            merge_execution: Some(b"{}"),
            target_lock: Some(b"merge-execution:e1"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(
                TaskKind::MergeWal,
                write,
                &owned,
                Some("merge-execution:e1"),
                true,
                None
            ),
            Ok(()),
            "merge reconciles its own execution's lock"
        );
    }

    #[test]
    fn preparation_shares_only_with_catchup_when_configured() {
        let prepare = ClaimShape {
            preparing: true,
            write_claim: false,
            shared_owner: false,
        };
        let catchup = Snapshot {
            catchup_active: Some(b"job"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(TaskKind::Compact, prepare, &catchup, None, false, None),
            Err(Skip::OwnershipBlocked)
        );
        let shared = ClaimShape {
            shared_owner: true,
            ..prepare
        };
        assert_eq!(
            admission_filter(TaskKind::Compact, shared, &catchup, None, false, None),
            Ok(())
        );
        let other_prep = Snapshot {
            compact_preparation: Some(b"p"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(TaskKind::Compact, prepare, &other_prep, None, false, None),
            Err(Skip::OwnershipBlocked),
            "one preparation slot per table"
        );
        let write_lock_held = Snapshot {
            target_lock: Some(b"someone"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(
                TaskKind::Compact,
                prepare,
                &write_lock_held,
                None,
                false,
                None
            ),
            Ok(()),
            "preparation does not need the write lock"
        );
    }

    #[test]
    fn worker_execution_is_only_reconciled_by_merge() {
        let write = ClaimShape {
            preparing: false,
            write_claim: true,
            shared_owner: false,
        };
        let worker = Snapshot {
            merge_execution: Some(b"{}"),
            target_lock: Some(b"merge-execution:w"),
            ..EMPTY
        };
        assert_eq!(
            admission_filter(
                TaskKind::Compact,
                write,
                &worker,
                Some("merge-execution:w"),
                true,
                None
            ),
            Err(Skip::WorkerExecutionNeedsMerge)
        );
        assert_eq!(
            admission_filter(
                TaskKind::Compact,
                write,
                &worker,
                Some("merge-execution:w"),
                false,
                None
            ),
            Ok(()),
            "a local maintenance execution may be recovered by any table writer"
        );
        let prepare = ClaimShape {
            preparing: true,
            write_claim: false,
            shared_owner: false,
        };
        assert_eq!(
            admission_filter(
                TaskKind::IndexId,
                prepare,
                &worker,
                Some("merge-execution:w"),
                true,
                None
            ),
            Ok(()),
            "preparation runs beside a worker execution"
        );
    }
}
