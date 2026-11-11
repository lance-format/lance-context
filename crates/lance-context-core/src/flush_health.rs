//! Live, in-memory flush diagnostics. No storage/etcd reads and no table labels.
//! Retry attempts and unrelated writers never refresh a generation's progress.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

static NEXT: AtomicU64 = AtomicU64::new(1);
type Registry = BTreeMap<u64, Weak<Mutex<Health>>>;
fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(Mutex::default)
}

#[derive(Debug)]
struct Active {
    generation: u64,
    started: Instant,
    progress: Instant,
    checkpoint: u8,
    failed_at: Option<Instant>,
    terminal: bool,
    attempts: u64,
}

#[derive(Debug)]
struct Health {
    uri: String,
    shard: String,
    active: Option<Active>,
    generations: usize,
    bytes: usize,
}

#[derive(Debug)]
pub(crate) struct WriterHealth {
    id: u64,
    state: Arc<Mutex<Health>>,
}

impl WriterHealth {
    pub(crate) fn new(uri: String, shard: String) -> Self {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(Mutex::new(Health {
            uri,
            shard,
            active: None,
            generations: 0,
            bytes: 0,
        }));
        registry()
            .lock()
            .unwrap()
            .insert(id, Arc::downgrade(&state));
        Self { id, state }
    }

    pub(crate) fn stage(&self, generation: u64, stage: &str) {
        self.stage_at(generation, stage, Instant::now());
    }

    fn stage_at(&self, generation: u64, stage: &str, now: Instant) {
        let mut state = self.state.lock().unwrap();
        let active = state.active.get_or_insert(Active {
            generation,
            started: now,
            progress: now,
            checkpoint: 0,
            failed_at: None,
            terminal: false,
            attempts: 1,
        });
        // Flushes are FIFO; completion clears the previous generation.
        debug_assert_eq!(active.generation, generation);
        let checkpoint = match stage {
            "wal_durable" => 1,
            "prepared" => 2,
            _ => 0,
        };
        if checkpoint > active.checkpoint {
            active.checkpoint = checkpoint;
            active.progress = now;
        }
    }

    pub(crate) fn failed(&self, generation: u64, terminal: bool, error: &str) {
        self.stage(generation, "started");
        let mut state = self.state.lock().unwrap();
        let active = state.active.as_mut().unwrap();
        active.failed_at.get_or_insert_with(Instant::now);
        active.terminal |= terminal;
        tracing::warn!(uri = %state.uri, shard = %state.shard, generation,
            terminal, error, "memtable flush failed; data retained for recovery");
    }

    pub(crate) fn retry(&self) {
        if let Some(active) = self.state.lock().unwrap().active.as_mut() {
            active.attempts += 1;
        }
    }

    pub(crate) fn queue(&self, generations: usize, bytes: usize) {
        let mut state = self.state.lock().unwrap();
        state.generations = generations;
        state.bytes = bytes;
    }

    pub(crate) fn completed(&self) {
        self.state.lock().unwrap().active = None;
    }
}

impl Drop for WriterHealth {
    fn drop(&mut self) {
        let state = self.state.lock().unwrap();
        if state.generations > 0 {
            metrics::counter!("rollout_memtable_flush_unresolved_writer_drops_total").increment(1);
            tracing::warn!(uri = %state.uri, shard = %state.shard,
                generations = state.generations, bytes = state.bytes,
                "writer dropped with unresolved flushes; recovery is not verified");
        }
        drop(state);
        registry().lock().unwrap().remove(&self.id);
    }
}

#[derive(Serialize)]
pub struct FlushHealth {
    pub sampled_at_ms: u64,
    pub complete: bool,
    pub writers_observed: usize,
    pub writers_unknown: usize,
    pub active: Vec<FlushStatus>,
}

#[derive(Serialize)]
pub struct FlushStatus {
    pub writer_id: u64,
    pub uri: String,
    pub shard: String,
    pub generation: u64,
    pub checkpoint: &'static str,
    pub elapsed_seconds: f64,
    pub checkpoint_idle_seconds: f64,
    pub failure_age_seconds: Option<f64>,
    pub terminal: bool,
    pub attempts: u64,
    pub retry_generations: usize,
    pub retry_bytes: usize,
}

/// Snapshot existing writers only. Contended observations are explicitly unknown.
/// No payloads, network requests, I/O, or waiting on a writer's async locks.
pub fn snapshot() -> FlushHealth {
    let mut result = FlushHealth {
        sampled_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        complete: true,
        writers_observed: 0,
        writers_unknown: 0,
        active: Vec::new(),
    };
    let writers: Vec<_> = match registry().try_lock() {
        Ok(registry) => registry
            .iter()
            .map(|(id, state)| (*id, state.clone()))
            .collect(),
        Err(_) => {
            result.complete = false;
            return result;
        }
    };
    let now = Instant::now();
    for (id, weak) in writers {
        let Some(state) = weak.upgrade() else {
            continue;
        };
        let Ok(state) = state.try_lock() else {
            result.writers_unknown += 1;
            result.complete = false;
            continue;
        };
        result.writers_observed += 1;
        let Some(active) = &state.active else {
            continue;
        };
        result.active.push(FlushStatus {
            writer_id: id,
            uri: state.uri.clone(),
            shard: state.shard.clone(),
            generation: active.generation,
            checkpoint: match active.checkpoint {
                1 => "wal_durable",
                2 => "prepared",
                _ => "started",
            },
            elapsed_seconds: now.saturating_duration_since(active.started).as_secs_f64(),
            checkpoint_idle_seconds: now.saturating_duration_since(active.progress).as_secs_f64(),
            failure_age_seconds: active
                .failed_at
                .map(|t| now.saturating_duration_since(t).as_secs_f64()),
            terminal: active.terminal,
            attempts: active.attempts,
            retry_generations: state.generations,
            retry_bytes: state.bytes,
        });
    }
    result
}

/// Low-cardinality gauges computed at scrape time, so ages keep advancing even
/// when the operation is hung. Missing observations never become healthy zeros.
pub fn render() -> String {
    let snapshot = snapshot();
    render_snapshot(&snapshot)
}

fn render_snapshot(snapshot: &FlushHealth) -> String {
    let mut out = String::new();
    let mut gauge = |name: &str, help: &str, value: f64| {
        use std::fmt::Write;
        let _ = writeln!(
            out,
            "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}"
        );
    };
    gauge(
        "rollout_memtable_flush_health_complete",
        "All resident writer observations were readable.",
        u8::from(snapshot.complete) as f64,
    );
    gauge(
        "rollout_memtable_flush_health_sample_timestamp_seconds",
        "Time this process generated the health snapshot.",
        snapshot.sampled_at_ms as f64 / 1000.0,
    );
    if snapshot.complete {
        gauge(
            "rollout_memtable_flush_active_generations",
            "Generations with an unfinished flush attempt, including retained failures.",
            snapshot.active.len() as f64,
        );
        gauge(
            "rollout_memtable_flush_terminal_writers",
            "Writers retaining a terminal fenced generation; retry alone cannot recover it.",
            snapshot.active.iter().filter(|s| s.terminal).count() as f64,
        );
        gauge(
            "rollout_memtable_flush_oldest_failure_age_seconds",
            "Oldest first failure still unresolved; other writers cannot reset it.",
            snapshot
                .active
                .iter()
                .filter_map(|s| s.failure_age_seconds)
                .fold(0.0, f64::max),
        );
        gauge("rollout_memtable_flush_checkpoint_idle_max_seconds", "Longest interval without a new durable checkpoint; coarse evidence, not proof of a stalled I/O operation.", snapshot.active.iter().map(|s| s.checkpoint_idle_seconds).fold(0.0, f64::max));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn retry_and_unrelated_success_do_not_reset_failed_generation_progress() {
        let stalled = WriterHealth::new("stalled".into(), "s1".into());
        let healthy = WriterHealth::new("healthy".into(), "s2".into());
        let first = Instant::now() - Duration::from_secs(60);
        stalled.stage_at(7, "started", first);
        stalled.stage_at(7, "wal_durable", first);
        stalled.failed(7, true, "injected fence");
        stalled.queue(1, 100);
        stalled.retry();
        stalled.stage_at(7, "started", Instant::now());
        stalled.stage_at(7, "wal_durable", Instant::now());
        healthy.stage(3, "prepared");
        healthy.completed();
        let sample = snapshot();
        let entry = sample
            .active
            .iter()
            .find(|e| e.writer_id == stalled.id)
            .unwrap();
        assert_eq!(entry.generation, 7);
        assert!(entry.terminal);
        assert!(entry.checkpoint_idle_seconds >= 60.0);
        assert_eq!(entry.attempts, 2);
        assert_eq!(entry.retry_generations, 1);
        stalled.stage(7, "prepared");
        assert!(
            stalled
                .state
                .lock()
                .unwrap()
                .active
                .as_ref()
                .unwrap()
                .progress
                > first
        );
        stalled.queue(0, 0);
        stalled.completed();
        assert!(!snapshot().active.iter().any(|e| e.writer_id == stalled.id));
        let guard = registry().lock().unwrap();
        let sample = snapshot();
        assert!(!sample.complete);
        let text = render_snapshot(&sample);
        assert!(text.contains("rollout_memtable_flush_health_complete 0"));
        assert!(!text.contains("rollout_memtable_flush_terminal_writers"));
        drop(guard);
    }
}
