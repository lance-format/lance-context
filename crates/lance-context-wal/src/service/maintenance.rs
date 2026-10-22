use super::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Mutex,
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub interval_ms: u64,
    pub checkpoint_entries: u64,
    pub checkpoint_interval_ms: u64,
    pub log_interval_ms: u64,
    pub scan_entries: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            interval_ms: 1000,
            checkpoint_entries: 8192,
            checkpoint_interval_ms: 60_000,
            log_interval_ms: 30_000,
            scan_entries: 512,
        }
    }
}
impl Options {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            (10..=60_000).contains(&self.interval_ms)
                && self.checkpoint_entries > 0
                && (self.interval_ms..=3_600_000).contains(&self.checkpoint_interval_ms)
                && (self.interval_ms..=3_600_000).contains(&self.log_interval_ms)
                && (1..=4096).contains(&self.scan_entries),
            "invalid maintenance scheduling"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Progress {
    pub checkpoints: u64,
    pub checkpoint_micros: u64,
    pub collected_files: u64,
    pub collected_bytes: u64,
    pub log_passes: u64,
    pub log_micros: u64,
    pub attempts_failed: u64,
    pub last_error: Option<String>,
}
pub(super) type Shared = Arc<Mutex<Progress>>;
fn error(progress: &Shared, error: anyhow::Error) {
    let mut progress = progress.lock().unwrap();
    progress.attempts_failed = progress.attempts_failed.saturating_add(1);
    progress.last_error = Some(error.to_string().chars().take(2048).collect());
}
fn micros(start: Instant) -> u64 {
    start.elapsed().as_micros().min(u64::MAX as u128) as u64
}

/// One scheduler per physical node, with at most one blocking job outstanding.
/// Cancellation stops between file operations; every submitted job is joined.
pub(super) async fn run(
    handle: runtime::Handle,
    groups: Vec<u64>,
    options: Options,
    progress: Shared,
    mut stop: tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    let mut checkpoints: BTreeMap<_, _> = groups.iter().map(|&g| (g, Instant::now())).collect();
    let mut last_log = Instant::now();
    let mut suggested = BTreeSet::new();
    let mut tick = tokio::time::interval(Duration::from_millis(options.interval_ms));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! { _ = stop.changed() => return Ok(()), _ = tick.tick() => {} }
        if *stop.borrow() {
            return Ok(());
        }
        // Preserve the scan cursor across bounded batches. Do not repeatedly
        // restart at the first directory entry and starve later orphan files.
        match handle.prepare_image_collection() {
            Ok(pending) => match pending.await {
                Ok(mut collection) => loop {
                    let count = options.scan_entries;
                    let (returned, result) = tokio::task::spawn_blocking(move || {
                        let result = collection.run_batch(count);
                        (collection, result)
                    })
                    .await?;
                    collection = returned;
                    match result {
                        Ok(batch) => {
                            let mut p = progress.lock().unwrap();
                            p.collected_files =
                                p.collected_files.saturating_add(batch.removed_files);
                            p.collected_bytes =
                                p.collected_bytes.saturating_add(batch.removed_bytes);
                            if batch.finished || *stop.borrow() {
                                break;
                            }
                        }
                        Err(e) => {
                            error(&progress, e);
                            break;
                        }
                    }
                    tokio::task::yield_now().await;
                },
                Err(e) => error(&progress, e),
            },
            Err(e) => error(&progress, e),
        }
        if *stop.borrow() {
            return Ok(());
        }
        for &group in &groups {
            let observed = handle.observation(group)?;
            ensure!(
                !observed.stopped && observed.failure.is_none(),
                "replica worker failed during maintenance"
            );
            let new = observed
                .status
                .applied
                .saturating_sub(observed.checkpoint_index);
            if new == 0
                || (!suggested.contains(&group)
                    && new < options.checkpoint_entries
                    && checkpoints[&group].elapsed()
                        < Duration::from_millis(options.checkpoint_interval_ms))
            {
                continue;
            }
            let start = Instant::now();
            let result: Result<()> = async {
                let capture = handle
                    .prepare_checkpoint(group, uuid::Uuid::new_v4())?
                    .await?;
                let built = tokio::task::spawn_blocking(move || capture.run()).await??;
                handle.install_checkpoint(built)?.await?;
                Ok(())
            }
            .await;
            checkpoints.insert(group, Instant::now());
            match result {
                Ok(()) => {
                    suggested.remove(&group);
                    let mut p = progress.lock().unwrap();
                    p.checkpoints = p.checkpoints.saturating_add(1);
                    p.checkpoint_micros = p.checkpoint_micros.saturating_add(micros(start));
                }
                Err(e) => error(&progress, e),
            }
            if *stop.borrow() {
                return Ok(());
            }
        }
        if last_log.elapsed() >= Duration::from_millis(options.log_interval_ms) {
            let start = Instant::now();
            let result: Result<Vec<u64>> = async {
                let task = handle.prepare_log_maintenance()?.await?;
                tokio::task::spawn_blocking(move || task.run()).await?
            }
            .await;
            last_log = Instant::now();
            match result {
                Ok(next) => {
                    suggested.extend(next.into_iter().filter(|group| groups.contains(group)));
                    let mut p = progress.lock().unwrap();
                    p.log_passes = p.log_passes.saturating_add(1);
                    p.log_micros = p.log_micros.saturating_add(micros(start));
                }
                Err(e) => error(&progress, e),
            }
        }
        if *stop.borrow() {
            return Ok(());
        }
        let result: Result<()> = async {
            let task = handle.prepare_disk_measurement()?.await?;
            tokio::task::spawn_blocking(move || task.run()).await??;
            Ok(())
        }
        .await;
        if let Err(e) = result {
            error(&progress, e);
        }
    }
}
