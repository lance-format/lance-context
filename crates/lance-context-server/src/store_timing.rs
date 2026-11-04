//! Timings for store resolution and metadata reads, without changing admission.
//! `info_resolve` includes the cold-open child stages; these are not additive.
use std::future::Future;
use tokio::time::Instant;

#[derive(Clone, Copy)]
pub(crate) enum Stage {
    CacheLock,
    NameLock,
    RegistryLookup,
    DatasetOpen,
    RegistryRecheck,
    InfoResolve,
    InfoLock,
}

impl Stage {
    fn label(self) -> &'static str {
        match self {
            Self::CacheLock => "cache_lock",
            Self::NameLock => "name_lock",
            Self::RegistryLookup => "registry_lookup",
            Self::DatasetOpen => "dataset_open",
            Self::RegistryRecheck => "registry_recheck",
            Self::InfoResolve => "info_resolve",
            Self::InfoLock => "info_lock",
        }
    }
}

struct Timing<'a> {
    table: &'a str,
    stage: Stage,
    started: Instant,
    completed: bool,
}

impl Drop for Timing<'_> {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        // Completed means the await returned, including an error. It is not a
        // successful-read counter. Dropping a pending request is interrupted.
        let outcome = if self.completed {
            "completed"
        } else {
            "interrupted"
        };
        let stage = self.stage.label();
        metrics::histogram!("rollout_store_stage_duration_seconds", "stage" => stage, "outcome" => outcome)
            .record(elapsed.as_secs_f64());
        if elapsed >= std::time::Duration::from_secs(5) {
            tracing::warn!(
                target: "lance_context_server::store_timing",
                table = %self.table.chars().take(256).collect::<String>(),
                stage,
                outcome,
                elapsed_ms = elapsed.as_millis() as u64,
                "rollout store resolution diagnostic"
            );
        }
    }
}

pub(crate) async fn measure<T>(table: &str, stage: Stage, future: impl Future<Output = T>) -> T {
    let mut timing = Timing {
        table,
        stage,
        started: Instant::now(),
        completed: false,
    };
    let result = future.await;
    timing.completed = true;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use std::time::Duration;

    #[tokio::test(start_paused = true)]
    async fn measures_completed_errors_and_cancelled_waits_without_table_labels() {
        let recorder = DebuggingRecorder::new();
        let snapshots = recorder.snapshotter();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let result = measure("private-table", Stage::RegistryLookup, async {
            tokio::time::sleep(Duration::from_secs(6)).await;
            Err::<(), _>("private-error")
        })
        .await;
        assert_eq!(result, Err("private-error"));

        let mut pending = Box::pin(measure(
            "private-table",
            Stage::InfoLock,
            std::future::pending::<()>(),
        ));
        assert!(futures::poll!(&mut pending).is_pending());
        tokio::time::advance(Duration::from_secs(7)).await;
        drop(pending);

        let series = snapshots.snapshot().into_vec();
        assert_eq!(series.len(), 2);
        for (key, _, _, value) in series {
            assert_eq!(key.key().name(), "rollout_store_stage_duration_seconds");
            let labels: std::collections::BTreeMap<_, _> =
                key.key().labels().map(|l| (l.key(), l.value())).collect();
            assert_eq!(labels.len(), 2);
            let seconds = match (labels["stage"], labels["outcome"]) {
                ("registry_lookup", "completed") => 6.0,
                ("info_lock", "interrupted") => 7.0,
                other => panic!("unexpected labels: {other:?}"),
            };
            let DebugValue::Histogram(values) = value else {
                panic!("expected histogram")
            };
            assert_eq!(values.len(), 1);
            assert_eq!(values[0].into_inner(), seconds);
        }
    }
}
