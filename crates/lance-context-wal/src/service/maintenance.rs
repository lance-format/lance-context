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
pub(crate) async fn run(
    handle: runtime::Handle,
    groups: Vec<u64>,
    options: Options,
    progress: Shared,
    mut stop: tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    let mut checkpoints: BTreeMap<_, _> = groups.iter().map(|&g| (g, Instant::now())).collect();
    let mut tick = tokio::time::interval(Duration::from_millis(options.interval_ms));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut background = Background {
        handle: &handle,
        groups: &groups,
        options: &options,
        progress: &progress,
        stop: &mut stop,
        tick,
        last_log: Instant::now(),
        suggested: BTreeSet::new(),
    };
    loop {
        tokio::select! { _ = background.stop.changed() => return Ok(()), _ = background.tick.tick() => {} }
        if *background.stop.borrow() {
            return Ok(());
        }
        // Preserve the scan cursor across bounded batches. Do not repeatedly
        // restart at the first directory entry and starve later orphan files.
        let Some(prepared) = background
            .admit(handle.prepare_image_collection_wait())
            .await
        else {
            return Ok(());
        };
        match prepared {
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
                            if batch.finished || *background.stop.borrow() {
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
        if *background.stop.borrow() {
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
                || (!background.suggested.contains(&group)
                    && new < options.checkpoint_entries
                    && checkpoints[&group].elapsed()
                        < Duration::from_millis(options.checkpoint_interval_ms))
            {
                continue;
            }
            let start = Instant::now();
            let Some(prepared) = background
                .admit(handle.prepare_checkpoint_wait(group, uuid::Uuid::new_v4()))
                .await
            else {
                return Ok(());
            };
            let result: Result<bool> = async {
                let Some(capture) = prepared?.await? else {
                    return Ok(false);
                };
                let built = tokio::task::spawn_blocking(move || capture.run()).await??;
                handle.install_checkpoint(built)?.await?;
                Ok(true)
            }
            .await;
            match result {
                Ok(built) => {
                    // A received snapshot may already have covered this cut.
                    // A failed attempt leaves this cut overdue. Retry on the
                    // next scheduler tick instead of extending log retention.
                    checkpoints.insert(group, Instant::now());
                    background.suggested.remove(&group);
                    if built {
                        let mut p = progress.lock().unwrap();
                        p.checkpoints = p.checkpoints.saturating_add(1);
                        p.checkpoint_micros = p.checkpoint_micros.saturating_add(micros(start));
                    }
                }
                Err(e) => error(&progress, e),
            }
            if *background.stop.borrow() {
                return Ok(());
            }
        }
        background.maintain().await;
    }
}

// Image admission can wait for a slow receiver. While queued, continue the
// independent engine purge and disk accounting on the same blocking-job lane.
// Once admitted, the image operation runs to completion before another local
// blocking job starts. Keep the admission future pinned across scheduler ticks
// so periodic work cannot cancel and lose our place in the fair semaphore queue.
struct Background<'a> {
    handle: &'a runtime::Handle,
    groups: &'a [u64],
    options: &'a Options,
    progress: &'a Shared,
    stop: &'a mut tokio::sync::watch::Receiver<bool>,
    tick: tokio::time::Interval,
    last_log: Instant,
    suggested: BTreeSet<u64>,
}
impl Background<'_> {
    async fn admit<T>(
        &mut self,
        admission: impl std::future::Future<Output = Result<T>>,
    ) -> Option<Result<T>> {
        tokio::pin!(admission);
        loop {
            if *self.stop.borrow() {
                return None;
            }
            tokio::select! {
                _ = self.stop.changed() => return None,
                result = &mut admission => return Some(result),
                _ = self.tick.tick() => self.maintain().await,
            }
        }
    }

    async fn maintain(&mut self) {
        if *self.stop.borrow() {
            return;
        }
        if self.last_log.elapsed() >= Duration::from_millis(self.options.log_interval_ms) {
            let start = Instant::now();
            let result: Result<Vec<u64>> = async {
                let task = self.handle.prepare_log_maintenance()?.await?;
                tokio::task::spawn_blocking(move || task.run()).await?
            }
            .await;
            match result {
                Ok(next) => {
                    self.last_log = Instant::now();
                    self.suggested
                        .extend(next.into_iter().filter(|group| self.groups.contains(group)));
                    let mut p = self.progress.lock().unwrap();
                    p.log_passes = p.log_passes.saturating_add(1);
                    p.log_micros = p.log_micros.saturating_add(micros(start));
                }
                Err(e) => error(self.progress, e),
            }
        }
        if *self.stop.borrow() {
            return;
        }
        let result: Result<()> = async {
            let task = self.handle.prepare_disk_measurement()?.await?;
            tokio::task::spawn_blocking(move || task.run()).await??;
            Ok(())
        }
        .await;
        if let Err(e) = result {
            error(self.progress, e);
        }
    }
}
