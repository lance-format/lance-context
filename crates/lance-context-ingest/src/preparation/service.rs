use super::*;
use crate::materializer::query;
use axum::{
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};
use tokio::sync::{watch, Notify, Semaphore};

type ResultValue = std::result::Result<p::Descriptors, String>;
type Completion = watch::Receiver<Option<ResultValue>>;
#[derive(Default)]
struct StateData {
    closed: bool,
    jobs: BTreeMap<Task, Completion>,
}
struct Inner {
    assignments: BTreeMap<Uuid, (Ingress, Storage)>,
    options: Options,
    budget: Arc<MergeMemoryBudget>,
    jobs: Arc<Semaphore>,
    requests: Arc<Semaphore>,
    active_limit: usize,
    retained: usize,
    state: Mutex<StateData>,
    idle: Notify,
    phases: Arc<Counters>,
    completed: AtomicU64,
    failed: AtomicU64,
}
#[derive(Clone)]
pub struct Service(Arc<Inner>);
#[derive(Serialize)]
pub struct Observation {
    pub active: usize,
    pub cached: usize,
    pub reserved_bytes: usize,
    pub completed: u64,
    pub failed: u64,
    pub draining: bool,
    pub phases: [crate::materializer::observation::Observation; 10],
}
impl Service {
    pub fn new(
        assignments: Vec<(Ingress, Storage)>,
        options: Options,
        budget: Arc<MergeMemoryBudget>,
        active: usize,
        retained: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=32).contains(&active)
                && (active..=128).contains(&retained)
                && (1..=4096).contains(&assignments.len()),
            "invalid preparer limits"
        );
        let mut tables = BTreeMap::new();
        for (ingress, storage) in assignments {
            options.validate(&ingress, &budget)?;
            ensure!(
                tables
                    .insert(ingress.scope.table, (ingress, storage))
                    .is_none(),
                "duplicate preparer table"
            );
        }
        Ok(Self(Arc::new(Inner {
            assignments: tables,
            options,
            budget,
            jobs: Arc::new(Semaphore::new(active)),
            requests: Arc::new(Semaphore::new(active * 2)),
            active_limit: active,
            retained,
            state: Mutex::new(StateData::default()),
            idle: Notify::new(),
            phases: Arc::default(),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
        })))
    }
    pub fn router(&self) -> Router {
        Router::new()
            .route(ROUTE, post(prepare))
            .route("/api/v1/health", get(health))
            .with_state(self.clone())
    }
    pub fn observation(&self) -> Observation {
        let state = self.0.state.lock().unwrap_or_else(|p| p.into_inner());
        Observation {
            active: self.0.active_limit - self.0.jobs.available_permits(),
            cached: state.jobs.len(),
            reserved_bytes: self.0.budget.reserved(),
            completed: self.0.completed.load(Ordering::Relaxed),
            failed: self.0.failed.load(Ordering::Relaxed),
            draining: state.closed,
            phases: self.0.phases.observation(),
        }
    }
    pub async fn shutdown(&self) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .closed = true;
        loop {
            let notified = self.0.idle.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.0.jobs.available_permits() == self.0.active_limit {
                break;
            }
            notified.await;
        }
    }
    async fn submit(&self, task: Task) -> Result<p::Descriptors> {
        let (ingress, _) = self
            .0
            .assignments
            .get(&task.table)
            .context("table outside preparer registry")?;
        task.validate(ingress)?;
        let mut receiver = {
            let mut state = self.0.state.lock().unwrap_or_else(|p| p.into_inner());
            ensure!(!state.closed, "preparer draining");
            if let Some(result) = state.jobs.get(&task) {
                result.clone()
            } else {
                let permit = self
                    .0
                    .jobs
                    .clone()
                    .try_acquire_owned()
                    .context("preparer capacity")?;
                if state.jobs.len() == self.0.retained {
                    let old = state
                        .jobs
                        .iter()
                        .find(|(_, result)| result.borrow().is_some())
                        .map(|(key, _)| key.clone());
                    let old = old.context("preparer result capacity")?;
                    state.jobs.remove(&old);
                }
                let (sender, receiver) = watch::channel(None);
                state.jobs.insert(task.clone(), receiver.clone());
                let owner = self.clone();
                tokio::spawn(async move {
                    let job_owner = owner.clone();
                    let job_task = task.clone();
                    let result =
                        tokio::spawn(async move { job_owner.materialize(&job_task).await })
                            .await
                            .context("preparation task panicked")
                            .and_then(|result| result);
                    let result = match result {
                        Ok(value) => {
                            owner.0.completed.fetch_add(1, Ordering::Relaxed);
                            Ok(value)
                        }
                        Err(error) => {
                            owner.0.failed.fetch_add(1, Ordering::Relaxed);
                            owner
                                .0
                                .state
                                .lock()
                                .unwrap_or_else(|p| p.into_inner())
                                .jobs
                                .remove(&task);
                            // Do not expose backend configuration/credentials over HTTP.
                            #[cfg(test)]
                            eprintln!("preparation test failure: {error:#}");
                            let _ = error;
                            Err("file preparation failed".to_owned())
                        }
                    };
                    sender.send_replace(Some(result));
                    drop(permit);
                    owner.0.idle.notify_waiters();
                });
                receiver
            }
        };
        loop {
            if let Some(result) = receiver.borrow().clone() {
                return result.map_err(anyhow::Error::msg);
            }
            receiver
                .changed()
                .await
                .context("preparation completion unavailable")?;
        }
    }
    async fn materialize(&self, task: &Task) -> Result<p::Descriptors> {
        let (ingress, storage) = &self.0.assignments[&task.table];
        let query::Value::Publisher(Some(view)) = query(
            ingress,
            ingress.client.catalog_group(),
            query::Query::Publisher { table: task.table },
        )
        .await?
        else {
            bail!("preparation binding missing");
        };
        ensure!(
            view.destination == ingress.scope.destination
                && view.ingest == ingress.scope.contract
                && view
                    .manifests
                    .as_ref()
                    .is_some_and(|m| m.incarnation == task.incarnation)
                && view.progress.prepared.is_none(),
            "preparation binding changed or already prepared"
        );
        let pending = view.progress.pending.context("preparation plan missing")?;
        let inputs = task.select(&pending)?;
        ensure!(
            pending.plan.source_bytes()? <= self.0.options.max_source_bytes as u64
                && pending
                    .plan
                    .sources
                    .values()
                    .map(|s| s.records.len())
                    .sum::<usize>()
                    <= self.0.options.max_records,
            "preparation plan exceeds configured bounds"
        );
        for &group in pending.plan.sources.keys() {
            let query::Value::SourcePin(Some(pin)) = query(
                ingress,
                group,
                query::Query::SourcePin { table: task.table },
            )
            .await?
            else {
                bail!("source pin missing");
            };
            let selection = pin.permit.selection();
            ensure!(
                selection.table == task.table
                    && selection.transaction == task.transaction
                    && selection.plan_index == task.plan_index
                    && selection.read_version == task.read_version
                    && selection.group == group
                    && selection.ingest == ingress.scope.contract
                    && selection.range == pending.plan.sources[&group],
                "source pin differs from preparation plan"
            );
        }
        let bytes = inputs.iter().try_fold(0usize, |sum, (_, _, meta)| {
            sum.checked_add(usize::try_from(meta.body_bytes)?)
                .context("partition source overflow")
        })?;
        let workspace = self
            .0
            .options
            .workspace(&ingress.scope.contract, bytes, inputs.len())?;
        ensure!(
            workspace <= self.0.budget.limit(),
            "preparer workspace exceeds budget"
        );
        let reservation = Arc::new(self.0.budget.reserve(workspace).await);
        let dataset =
            crate::materializer::read_dataset(ingress, storage, Some(task.read_version)).await?;
        let result = crate::materializer::stage::materialize_inputs(
            ingress.clone(),
            self.0.options.clone(),
            inputs,
            dataset,
            reservation,
            self.0.phases.clone(),
        )
        .await?;
        fragments(&result, ingress.client.max_descriptor_bytes())?;
        Ok(result)
    }
}
async fn prepare(State(service): State<Service>, request: Request) -> Response {
    let Ok(_request) = service.0.requests.clone().try_acquire_owned() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        axum::body::to_bytes(request.into_body(), 4096),
    )
    .await;
    let Ok(Ok(body)) = result else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let Ok(task) = serde_json::from_slice::<Task>(&body) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let Ok(identity) = task.identity() else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    match service.submit(task).await {
        Ok(result) => (
            [
                (axum::http::header::CONTENT_TYPE.as_str(), CONTENT_TYPE),
                (IDENTITY, &identity),
            ],
            result.bytes,
        )
            .into_response(),
        Err(_) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}
async fn health(State(service): State<Service>) -> Response {
    let observation = service.observation();
    if observation.draining {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    axum::Json(observation).into_response()
}
