//! Admission telemetry follows acknowledged store outcomes, never enqueue calls.
//! Task IDs and observation evidence stay in logs, not metric labels.
use lance_context_api::TaskKind;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Origin {
    Explicit,
    PeriodicStats,
    Scan,
    AggregateFrontier,
    WriterRequest,
    ExecutionRecovery,
    FailureRecovery,
    WalTail,
}

impl Origin {
    fn label(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::PeriodicStats => "periodic_stats",
            Self::Scan => "scan",
            Self::AggregateFrontier => "aggregate_frontier",
            Self::WriterRequest => "writer_request",
            Self::ExecutionRecovery => "execution_recovery",
            Self::FailureRecovery => "failure_recovery",
            Self::WalTail => "wal_tail",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Outcome {
    Created,
    Deduplicated,
    Superseded,
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Deduplicated => "deduplicated",
            Self::Superseded => "superseded",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Context {
    origin: Origin,
    revision: Option<i64>,
    stats: Option<(i64, i64, i64)>,
}

impl Context {
    pub(crate) fn new(origin: Origin) -> Self {
        Self {
            origin,
            revision: None,
            stats: None,
        }
    }

    pub(crate) fn revision(origin: Origin, revision: i64) -> Self {
        Self {
            revision: Some(revision),
            ..Self::new(origin)
        }
    }

    pub(crate) fn stats(origin: Origin, row: &crate::stats_store::StatRow) -> Self {
        Self {
            stats: Some((row.version, row.scanned_at, row.pending_wal_generations)),
            ..Self::new(origin)
        }
    }

    /// Call only after a definitive response. A transport error may follow a
    /// committed enqueue: it must not be counted as superseded or deduplicated.
    pub(crate) fn record(self, kind: TaskKind, target: &str, task: Option<&str>, outcome: Outcome) {
        let kind = match kind {
            TaskKind::MergeWal => "merge_wal",
            TaskKind::Compact => "compact",
            TaskKind::IndexId => "index_id",
            TaskKind::Repair => "repair",
        };
        let origin = self.origin.label();
        let outcome_label = outcome.label();
        metrics::counter!("master_task_admissions_total", "kind" => kind, "origin" => origin, "outcome" => outcome_label).increment(1);
        if matches!(outcome, Outcome::Created) {
            metrics::counter!("master_task_enqueued_total", "kind" => kind).increment(1);
        }
        tracing::info!(
            kind,
            origin,
            outcome = outcome_label,
            task_id = task,
            target,
            observation_revision = self.revision,
            observed_base_version = self.stats.map(|s| s.0),
            scanned_at_ms = self.stats.map(|s| s.1),
            pending_wal_generations = self.stats.map(|s| s.2),
            "task admission outcome"
        );
    }
}
