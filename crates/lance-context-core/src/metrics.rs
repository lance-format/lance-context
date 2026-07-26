//! Feature-gated metrics shim.
//!
//! Call sites use [`observe_duration!`] and [`count!`] unconditionally; when the
//! `metrics` feature is off these expand to nothing (the timing `Instant` is not
//! even taken), so a consumer embedding this crate pays zero cost and takes no
//! dependency.
//!
//! Metric names and label conventions live here so they cannot drift between the
//! emission site and the bucket configuration in `lance-context-metrics`.
//!
//! # Cardinality
//!
//! Every label combination times every histogram bucket is a separate exported
//! series — and in Datadog, a separately-billed custom metric. Two rules keep
//! that bounded:
//!
//! 1. **No unbounded labels.** Never a dataset URI, store name, experiment, or
//!    shard id. Those belong in a tracing span, which is queryable without
//!    multiplying series.
//! 2. **Failures are counted, not timed.** A `result="error"` label doubles a
//!    histogram's series count to describe the *latency distribution of a rare
//!    event*, which is almost never actionable. The actionable signal is the
//!    rate, so errors get a flat counter (1 series) and histograms measure the
//!    success path only.

/// Latency of one [`crate::RolloutStore::add`] — the durable WAL append only.
/// Unlabelled: failures are counted by [`ROLLOUT_ADD_ERRORS`] instead.
pub const ROLLOUT_ADD_DURATION: &str = "rollout_add_duration_seconds";

/// Failed [`crate::RolloutStore::add`] calls.
pub const ROLLOUT_ADD_ERRORS: &str = "rollout_add_errors_total";

/// Latency of one [`crate::RolloutStore::flush`] — sealing the memtable so
/// previously added rows become readable.
///
/// Label `outcome`:
/// - `sealed` — a memtable was actually sealed and drained (real work)
/// - `noop`   — no resident writer, returned immediately (the common case)
/// - `fenced` — the epoch was superseded by a merge; nothing to flush
///
/// `outcome` is kept despite the cardinality cost because the three paths differ
/// by orders of magnitude: without it the distribution is dominated by near-zero
/// `noop` samples and its high percentiles say nothing about real flush cost.
pub const ROLLOUT_FLUSH_DURATION: &str = "rollout_flush_duration_seconds";

/// Failed [`crate::RolloutStore::flush`] calls.
pub const ROLLOUT_FLUSH_ERRORS: &str = "rollout_flush_errors_total";

/// Per-phase latency of a WAL self-merge. Label `phase`:
/// `seal` | `read` | `append` | `claim_epoch` | `drain` | `delete`.
pub const ROLLOUT_WAL_MERGE_DURATION: &str = "rollout_wal_merge_duration_seconds";

/// Failed WAL self-merge phases, labelled by the `phase` that failed. A merge
/// aborts on the first failing phase, so this also identifies where it died.
pub const ROLLOUT_WAL_MERGE_ERRORS: &str = "rollout_wal_merge_errors_total";

/// Emit a histogram sample in seconds. No-op without the `metrics` feature.
#[cfg(feature = "metrics")]
macro_rules! observe_duration {
    ($name:expr, $elapsed:expr $(, $k:expr => $v:expr)* $(,)?) => {
        ::metrics::histogram!($name $(, $k => $v)*).record($elapsed.as_secs_f64())
    };
}

#[cfg(not(feature = "metrics"))]
macro_rules! observe_duration {
    ($name:expr, $elapsed:expr $(, $k:expr => $v:expr)* $(,)?) => {{
        let _ = &$elapsed;
    }};
}

/// Increment a counter by one. No-op without the `metrics` feature.
///
/// The disabled arm still expands to a unit *expression* rather than an empty
/// block, so a `match` whose arms are `observe_duration!`/`count!` keeps both
/// arms inhabited and does not collapse into a clippy `single_match` warning
/// when the feature is off.
#[cfg(feature = "metrics")]
macro_rules! count {
    ($name:expr $(, $k:expr => $v:expr)* $(,)?) => {
        ::metrics::counter!($name $(, $k => $v)*).increment(1)
    };
}

#[cfg(not(feature = "metrics"))]
macro_rules! count {
    ($name:expr $(, $k:expr => $v:expr)* $(,)?) => {{
        let _ = $name;
    }};
}

/// Start a timer, or evaluate to `()` when metrics are compiled out.
#[cfg(feature = "metrics")]
macro_rules! timer_start {
    () => {
        std::time::Instant::now()
    };
}

#[cfg(not(feature = "metrics"))]
macro_rules! timer_start {
    () => {
        ()
    };
}

/// Elapsed time since a [`timer_start!`], or a zero duration when compiled out.
#[cfg(feature = "metrics")]
macro_rules! timer_elapsed {
    ($t:expr) => {
        $t.elapsed()
    };
}

#[cfg(not(feature = "metrics"))]
macro_rules! timer_elapsed {
    ($t:expr) => {{
        let _ = &$t;
        std::time::Duration::ZERO
    }};
}

/// Time one WAL-merge phase: record its duration on success, or increment the
/// phase-labelled error counter on failure. Evaluates to the wrapped `Result`.
///
/// Keeps the success/failure split identical across all six phases, which is
/// what stops `result` creeping back onto the histogram.
macro_rules! observe_phase {
    ($phase:expr, $body:expr) => {{
        let __start = $crate::metrics::timer_start!();
        let __result = $body;
        match &__result {
            Ok(_) => $crate::metrics::observe_duration!(
                $crate::metrics::ROLLOUT_WAL_MERGE_DURATION,
                $crate::metrics::timer_elapsed!(__start),
                "phase" => $phase,
            ),
            Err(_) => $crate::metrics::count!(
                $crate::metrics::ROLLOUT_WAL_MERGE_ERRORS,
                "phase" => $phase,
            ),
        }
        __result
    }};
}

pub(crate) use {count, observe_duration, observe_phase, timer_elapsed, timer_start};
