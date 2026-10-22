//! Bounded materializer scheduling over etcd leases. Leases distribute work;
//! durable WAL epochs and manifest selection fence stale workers. The input
//! assignments must come from the application's authorized table registry.
use super::*;
use crate::materializer::{
    Acquisition, Options as PublishOptions, Publisher, ScanProgress, Storage,
};
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Mutex,
    },
};
use tokio::{
    sync::watch,
    task::{JoinHandle, JoinSet},
};

mod lease;
pub use lease::Claims;

#[derive(Clone)]
pub struct Assignment {
    pub ingress: Ingress,
    pub storage: Storage,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub parallel_tables: usize,
    /// Release a healthy claim periodically so another worker can take a turn.
    pub publications_per_claim: usize,
    pub poll_interval: Duration,
    pub publication: PublishOptions,
    /// Runtime pool resolved from deployment configuration, never from uploads.
    #[cfg(feature = "service")]
    #[serde(skip)]
    pub preparers: Option<Arc<crate::preparation::Pool>>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            parallel_tables: 4,
            publications_per_claim: 16,
            poll_interval: Duration::from_millis(500),
            publication: PublishOptions::default(),
            #[cfg(feature = "service")]
            preparers: None,
        }
    }
}
#[derive(Default)]
struct Counters {
    phases: Arc<crate::materializer::observation::Counters>,
    active: AtomicUsize,
    claimed: AtomicU64,
    published: AtomicU64,
    errors: AtomicU64,
    last_failure: Mutex<Option<Failure>>,
}
/// A bounded diagnostic retained only on failure, never a payload or config dump.
#[derive(Debug, Clone, Serialize)]
pub struct Failure {
    pub table: Uuid,
    pub message: String,
}
impl Counters {
    fn failed(&self, table: Uuid, error: &anyhow::Error) {
        // Stop formatting at the limit, including inside backend error Display
        // implementations. Do not first allocate an unbounded error chain.
        struct Bounded(String);
        impl std::fmt::Write for Bounded {
            fn write_str(&mut self, value: &str) -> std::fmt::Result {
                let available = 2048 - self.0.len();
                let mut end = available.min(value.len());
                while !value.is_char_boundary(end) {
                    end -= 1;
                }
                self.0.push_str(&value[..end]);
                if end < value.len() {
                    Err(std::fmt::Error)
                } else {
                    Ok(())
                }
            }
        }
        let mut message = Bounded(String::new());
        let _ = std::fmt::write(&mut message, format_args!("{error:#}"));
        *self.last_failure.lock().unwrap_or_else(|p| p.into_inner()) = Some(Failure {
            table,
            message: message.0,
        });
        self.errors.fetch_add(1, Ordering::Relaxed);
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct Observation {
    pub phases: [crate::materializer::observation::Observation; 10],
    pub running: bool,
    pub active: usize,
    pub claims: u64,
    pub publications: u64,
    pub errors: u64,
    pub last_failure: Option<Failure>,
    pub draining: bool,
}
pub struct Worker {
    stop: watch::Sender<bool>,
    task: Option<JoinHandle<Result<()>>>,
    counters: Arc<Counters>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        // Dropping a JoinHandle detaches the supervisor. It still owns and joins
        // its in-flight publication tasks; cancellation must not abort file work.
        let _ = self.stop.send(true);
    }
}
impl Worker {
    /// Start a fixed, authorized assignment set. Separate workers can receive
    /// overlapping sets; one expiring scheduling claim coordinates each table.
    pub fn start(
        claims: Claims,
        assignments: Vec<Assignment>,
        options: Options,
        budget: Arc<MergeMemoryBudget>,
    ) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&assignments.len())
                && (1..=64).contains(&options.parallel_tables)
                && (1..=1024).contains(&options.publications_per_claim)
                && (Duration::from_millis(10)..=Duration::from_secs(60))
                    .contains(&options.poll_interval),
            "invalid worker bounds"
        );
        let mut tables = BTreeSet::new();
        for assignment in &assignments {
            ensure!(
                tables.insert(assignment.ingress.scope.table),
                "duplicate worker table"
            );
            options.publication.validate(&assignment.ingress, &budget)?;
        }
        let counters = Arc::new(Counters::default());
        let (stop, stopped) = watch::channel(false);
        let task = tokio::spawn(run(
            claims,
            assignments,
            options,
            budget,
            stopped,
            counters.clone(),
        ));
        Ok(Self {
            stop,
            task: Some(task),
            counters,
        })
    }
    pub fn observation(&self) -> Observation {
        Observation {
            phases: self.counters.phases.observation(),
            running: self.task.as_ref().is_some_and(|task| !task.is_finished()),
            active: self.counters.active.load(Ordering::Acquire),
            claims: self.counters.claimed.load(Ordering::Relaxed),
            publications: self.counters.published.load(Ordering::Relaxed),
            errors: self.counters.errors.load(Ordering::Relaxed),
            last_failure: self
                .counters
                .last_failure
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone(),
            draining: *self.stop.borrow(),
        }
    }
    /// Stop new scheduling, finish admitted publication work, stop renewal, and
    /// revoke only each owned lease. A cancelled waiter does not abort the owner.
    pub async fn shutdown(mut self) -> Result<()> {
        let _ = self.stop.send(true);
        self.task.take().context("worker already joined")?.await?
    }
}

async fn run(
    claims: Claims,
    assignments: Vec<Assignment>,
    options: Options,
    budget: Arc<MergeMemoryBudget>,
    mut stop: watch::Receiver<bool>,
    counters: Arc<Counters>,
) -> Result<()> {
    let mut jobs: JoinSet<(usize, Result<()>, Option<ScanProgress>)> = JoinSet::new();
    // One bounded map per fixed assignment, moved into its only active job.
    // Releasing an idle claim must not restart scanning at the durable cursor.
    let mut scans: Vec<Option<ScanProgress>> = (0..assignments.len()).map(|_| None).collect();
    let mut active = BTreeSet::new();
    let mut cursor = Uuid::new_v4().as_u128() as usize % assignments.len();
    let mut tick = tokio::time::interval(options.poll_interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut failure = None;
    loop {
        if *stop.borrow() {
            break;
        }
        tokio::select! {
            _ = stop.changed() => break,
            done = jobs.join_next(), if !jobs.is_empty() => {
                counters.active.fetch_sub(1, Ordering::Release);
                match done {
                    Some(Ok((index, result, progress))) => {
                        active.remove(&index);
                        scans[index] = progress;
                        if let Err(error) = result { counters.failed(assignments[index].ingress.scope.table, &error); }
                    }
                    Some(Err(error)) => {
                        failure = Some(error);
                        break;
                    }
                    None => unreachable!("nonempty join set ended"),
                }
            }
            _ = tick.tick() => {
                // At most one pass and at most the configured number of jobs.
                // Failed/busy claims wait for the next tick, avoiding a hot loop.
                for _ in 0..assignments.len() {
                    if active.len() == options.parallel_tables { break; }
                    let index = cursor;
                    cursor = (cursor + 1) % assignments.len();
                    if !active.insert(index) { continue; }
                    counters.active.fetch_add(1, Ordering::Release);
                    let (claims, assignment, options, budget, stop, counters) = (
                        claims.clone(), assignments[index].clone(), options.clone(),
                        budget.clone(), stop.clone(), counters.clone(),
                    );
                    let mut progress = scans[index].take();
                    jobs.spawn(async move {
                        let result = publish_claim(claims, assignment, options, budget, stop, counters, &mut progress).await;
                        (index, result, progress)
                    });
                }
            }
        }
    }
    // Do not abort a submitted blob write or lose its memory reservation while
    // it is completing. The deployer budgets termination grace for this drain.
    while let Some(done) = jobs.join_next().await {
        counters.active.fetch_sub(1, Ordering::Release);
        match done {
            Ok((_, Ok(()), _)) => {}
            Ok((index, Err(error), _)) => {
                counters.failed(assignments[index].ingress.scope.table, &error);
            }
            Err(error) => {
                failure = Some(error);
            }
        }
    }
    if let Some(error) = failure {
        return Err(error.into());
    }
    Ok(())
}
async fn publish_claim(
    claims: Claims,
    assignment: Assignment,
    options: Options,
    budget: Arc<MergeMemoryBudget>,
    stop: watch::Receiver<bool>,
    counters: Arc<Counters>,
    scan_progress: &mut Option<ScanProgress>,
) -> Result<()> {
    use crate::materializer::observation::Phase;
    let Some(claim) = counters
        .phases
        .measure(Phase::Claim, claims.take(&assignment.ingress))
        .await
        .context("acquiring scheduling claim")?
    else {
        return Ok(());
    };
    counters.claimed.fetch_add(1, Ordering::Relaxed);
    let result = async {
        if *stop.borrow() {
            return Ok(());
        }
        let Some(mut publisher) = counters
            .phases
            .measure(
                Phase::Acquire,
                claim.acquire(assignment, options.publication, budget),
            )
            .await
            .context("acquiring publisher epoch")?
        else {
            return Ok(());
        };
        if let Some(progress) = scan_progress.take() {
            publisher.resume_scanning(progress)?;
        }
        publisher.observe_with(counters.phases.clone());
        #[cfg(feature = "service")]
        if let Some(pool) = options.preparers {
            publisher.use_preparers(pool);
        }
        let result = async {
            for _ in 0..options.publications_per_claim {
                if *stop.borrow() || !claim.valid().await? {
                    break;
                }
                if publisher
                    .run_once()
                    .await
                    .context("publishing Lance data")?
                    .is_none()
                {
                    break;
                }
                counters.published.fetch_add(1, Ordering::Relaxed);
            }
            Ok::<_, anyhow::Error>(())
        }
        .await;
        // Even an error after an empty page does not invalidate that observed
        // prefix. Selected records are never included in these skip hints.
        *scan_progress = Some(publisher.take_scan_progress());
        result
    }
    .await;
    let release = claim.release().await;
    result.and(release)
}
