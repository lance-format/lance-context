//! Bounded diagnostics for store failures on the append HTTP route.
//! Error text may contain records or storage credentials: inspect types only.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use lance_context_core::LanceError;

use crate::error::AppError;

pub(super) fn open(name: &str, error: &AppError) {
    let category = match error {
        AppError::NotFound(_) => "not_found",
        AppError::AlreadyExists(_) => "already_exists",
        AppError::InvalidRequest(_) => "invalid_request",
        AppError::Internal(_) => "internal",
        AppError::CompactionInProgress => "compaction_in_progress",
        AppError::ClientDisconnected => "client_disconnected",
        AppError::Overloaded(_) => "overloaded",
    };
    report(name, "resolve_store", category);
}

fn lance_category(error: &LanceError) -> &'static str {
    if error.is_backpressure() {
        "backpressure"
    } else {
        super::list_read_error_category(error)
    }
}

pub(super) fn lance(name: &str, phase: &'static str, error: &LanceError) {
    report(name, phase, lance_category(error));
}

fn report(name: &str, phase: &'static str, category: &'static str) {
    static CLOCK: OnceLock<Instant> = OnceLock::new();
    static NEXT_LOG_MS: AtomicU64 = AtomicU64::new(0);
    let now = CLOCK.get_or_init(Instant::now).elapsed().as_millis();
    record(
        name,
        phase,
        category,
        &NEXT_LOG_MS,
        now.min(u64::MAX as u128) as u64,
    );
}

fn allow_log(next: &AtomicU64, now: u64) -> bool {
    let deadline = next.load(Ordering::Relaxed);
    now >= deadline
        && next
            .compare_exchange(
                deadline,
                now.saturating_add(1000),
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
}

fn record(name: &str, phase: &'static str, category: &'static str, next: &AtomicU64, now: u64) {
    // Both labels are closed sets; never put table names or messages in metrics.
    metrics::counter!("rollout_write_failures_total", "phase" => phase, "category" => category)
        .increment(1);
    if allow_log(next, now) {
        let target: String = name.chars().take(128).collect();
        tracing::warn!(
            target = ?target,
            phase,
            category,
            "rollout write store failure (sampled at most once per second)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_failures_share_a_bounded_log_budget() {
        let next = AtomicU64::new(0);
        let admitted = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..32)
                .map(|_| scope.spawn(|| allow_log(&next, 0)))
                .collect();
            handles
                .into_iter()
                .map(|h| usize::from(h.join().unwrap()))
                .sum::<usize>()
        });
        assert_eq!(admitted, 1);
        assert!(!allow_log(&next, 999));
        assert!(allow_log(&next, 1000));
    }

    #[test]
    fn typed_errors_are_counted_without_raw_text_or_table_metric_labels() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};
        let recorder = DebuggingRecorder::new();
        let snapshot = recorder.snapshotter();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let next = AtomicU64::new(1000);
        let error = LanceError::backpressure("private-error-payload");
        assert_eq!(lance_category(&error), "backpressure");
        assert_eq!(
            lance_category(&LanceError::io("private-storage-url")),
            "storage_io"
        );
        for _ in 0..3 {
            record("private-table", "append", lance_category(&error), &next, 0);
        }
        let values = snapshot.snapshot().into_vec();
        assert_eq!(values.len(), 1);
        let (key, _, _, value) = &values[0];
        assert_eq!(key.key().name(), "rollout_write_failures_total");
        let labels: std::collections::BTreeSet<_> =
            key.key().labels().map(|l| (l.key(), l.value())).collect();
        assert_eq!(
            labels,
            [("phase", "append"), ("category", "backpressure")].into()
        );
        assert_eq!(*value, DebugValue::Counter(3));
    }

    #[test]
    fn sampled_logs_bound_table_names_and_omit_error_messages() {
        use std::io::{self, Write};
        use std::sync::{Arc, Mutex};
        #[derive(Clone)]
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl Write for Sink {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let sink = Sink(Arc::default());
        let capture = sink.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(move || capture.clone())
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        let name = format!("{}private-name-suffix", "界".repeat(128));
        let error = LanceError::backpressure("private-error-payload");
        let next = AtomicU64::new(0);
        record(&name, "append", lance_category(&error), &next, 0);
        record(&name, "append", lance_category(&error), &next, 0);
        let output = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
        assert_eq!(output.lines().count(), 1);
        assert!(output.contains("backpressure") && output.contains("append"));
        assert!(output.contains(&"界".repeat(128)));
        assert!(!output.contains("private-name-suffix"));
        assert!(!output.contains("private-error-payload"));
    }
}
