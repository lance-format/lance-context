//! Feature-gated metrics shim.
//!
//! Call sites use [`observe_duration!`] and [`count!`] unconditionally; when the
//! `metrics` feature is off these expand to nothing (the timing `Instant` is not
//! even taken), so a consumer embedding this crate pays zero cost and takes no
//! dependency.
//!
//! Metric names and label conventions live here so they cannot drift between the
//! emission site and the bucket configuration in `lance-context-metrics`.

/// Latency of one [`crate::RolloutStore::add`] — the durable WAL append only.
/// Label: `result` = `ok` | `error`.
pub const ROLLOUT_ADD_DURATION: &str = "rollout_add_duration_seconds";

/// Latency of one [`crate::RolloutStore::flush`] — sealing the memtable so
/// previously added rows become readable.
///
/// Labels: `result` = `ok` | `error`, and `outcome`:
/// - `sealed` — a memtable was actually sealed and drained (real work)
/// - `noop`   — no resident writer, returned immediately (the common case)
/// - `fenced` — the epoch was superseded by a merge; nothing to flush
///
/// Without `outcome` this histogram is dominated by near-zero `noop` samples and
/// its high percentiles say nothing about real flush cost.
pub const ROLLOUT_FLUSH_DURATION: &str = "rollout_flush_duration_seconds";

/// Per-phase latency of a WAL self-merge. Label `phase`:
/// `seal` | `read` | `append` | `claim_epoch` | `drain` | `delete`,
/// plus `result` = `ok` | `error`.
pub const ROLLOUT_WAL_MERGE_DURATION: &str = "rollout_wal_merge_duration_seconds";

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

pub(crate) use {observe_duration, timer_elapsed, timer_start};
