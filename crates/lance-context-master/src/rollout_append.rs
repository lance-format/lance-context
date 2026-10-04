//! Workers stage files concurrently; the master publishes bounded groups under
//! the existing table claim, idle watchdog, and manifest-version fence.
use crate::state::MasterState;
use futures::{stream, FutureExt, StreamExt};
use lance_context_core::{
    merge_write_scope::checkpoint,
    rollout_append::{AppendCoordinator, AppendPlan, StageEvent, StagedAppend},
};
use serde::Deserialize;
use std::{collections::HashSet, sync::Arc, time::Duration};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, clap::Args)]
pub struct AppendConfig {
    /// Opt-in immutable rollout targets. '*' selects all owned rollouts.
    #[arg(long, env = "ROLLOUT_APPEND_TARGETS", value_delimiter = ',')]
    pub rollout_append_targets: Vec<String>,
    /// Concurrent staging RPCs per table. Uses the workers' shared merge slots/budget.
    #[arg(long, env = "ROLLOUT_APPEND_CONCURRENCY", default_value_t = 4)]
    pub rollout_append_concurrency: usize,
    #[arg(long, env = "ROLLOUT_APPEND_MAX_GENERATIONS", default_value_t = 64)]
    pub rollout_append_max_generations: usize,
    #[arg(long, env = "ROLLOUT_APPEND_MAX_BYTES", default_value_t = 67_108_864)]
    pub rollout_append_max_bytes: usize,
}
impl Default for AppendConfig {
    fn default() -> Self {
        Self {
            rollout_append_targets: Vec::new(),
            rollout_append_concurrency: 4,
            rollout_append_max_generations: 64,
            rollout_append_max_bytes: 67_108_864,
        }
    }
}
impl AppendConfig {
    pub fn enabled(&self, target: &str) -> bool {
        !target.starts_with("generic:")
            && self
                .rollout_append_targets
                .iter()
                .any(|t| t == "*" || t == target)
    }
    fn validate(&self) -> Result<()> {
        if !(1..=32).contains(&self.rollout_append_concurrency)
            || !(1..=256).contains(&self.rollout_append_max_generations)
            || self.rollout_append_max_bytes == 0
            || self.rollout_append_max_bytes > lance_context_core::rollout_append::MAX_STAGE_BYTES
        {
            return Err("invalid rollout append limits".into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct Capabilities {
    #[serde(default)]
    rollout_append_protocol: u32,
    shard_name: Option<String>,
    #[serde(default)]
    owned_targets: Vec<String>,
    #[serde(default)]
    drain_targets: Vec<String>,
}

async fn stage_remote(
    client: &reqwest::Client,
    endpoint: &str,
    target: &str,
    plan: AppendPlan,
) -> Result<StagedAppend> {
    let expected = plan.clone();
    let request = client
        .post(format!(
            "{}/api/v1/internal/rollout-append/{target}",
            endpoint.trim_end_matches('/')
        ))
        .json(&plan);
    let mut response = tokio::time::timeout(Duration::from_secs(10), request.send())
        .await
        .map_err(|_| "staging admission timed out")?
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    let mut pending = Vec::new();
    let mut sequence = 0;
    loop {
        // A silent socket is not evidence of worker progress. Completed read/
        // encoding steps are relayed to the existing maintenance idle watchdog.
        let chunk = tokio::time::timeout(Duration::from_secs(10), response.chunk())
            .await
            .map_err(|_| "staging progress stream disconnected")?
            .map_err(|e| e.to_string())?
            .ok_or("staging stream ended without a result")?;
        pending.extend_from_slice(&chunk);
        if pending.len() > 4 * 1024 * 1024 {
            return Err("staged metadata exceeds 4 MiB".into());
        }
        while let Some(end) = pending.iter().position(|b| *b == b'\n') {
            let event: StageEvent =
                serde_json::from_slice(&pending[..end]).map_err(|e| e.to_string())?;
            pending.drain(..=end);
            match event {
                StageEvent::Progress(current) if current > sequence => {
                    sequence = current;
                    checkpoint();
                }
                StageEvent::Progress(_) => {}
                StageEvent::Failed(error) => return Err(error),
                StageEvent::Complete(part) => {
                    if part.plan != expected {
                        return Err("worker returned a different staging plan".into());
                    }
                    checkpoint();
                    return Ok(part);
                }
            }
        }
    }
}

pub(crate) async fn run(state: &Arc<MasterState>, target: &str) -> Result<String> {
    let config = &state.config.append;
    config.validate()?;
    let mut endpoints = Vec::new();
    let mut shards = Vec::new();
    let mut seen = HashSet::new();
    // Capability probing is cheap and bounded; never send a new protocol RPC to
    // an older worker and silently fall back to its manifest-publishing merge.
    let probe_futures: Vec<_> = state
        .config
        .worker_endpoints
        .iter()
        .cloned()
        .map(|endpoint| {
            let client = state.http.clone();
            async move {
                let response = client
                    .get(format!(
                        "{}/api/v1/internal/merge-executor",
                        endpoint.trim_end_matches('/')
                    ))
                    .timeout(Duration::from_secs(5))
                    .send()
                    .await
                    .map_err(|e| e.to_string())?
                    .error_for_status()
                    .map_err(|e| e.to_string())?;
                let caps: Capabilities = response.json().await.map_err(|e| e.to_string())?;
                Ok::<_, String>((endpoint, caps))
            }
            .boxed()
        })
        .collect();
    let mut probes = stream::iter(probe_futures).buffer_unordered(8);
    while let Some(result) = probes.next().await {
        let (endpoint, caps) = match result {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(%error, "staging worker unavailable; healthy workers will read its WAL");
                continue;
            }
        };
        if caps.rollout_append_protocol != 1 {
            return Err(format!("worker {endpoint} lacks rollout append protocol 1"));
        }
        if !caps.owned_targets.iter().any(|t| t == "*" || t == target)
            || caps.drain_targets.iter().any(|t| t == "*" || t == target)
        {
            return Err(format!(
                "worker {endpoint} has not enabled owned merges for {target}"
            ));
        }
        if let Some(shard) = caps.shard_name {
            if seen.insert(shard.clone()) {
                shards.push(shard);
            }
        }
        endpoints.push(endpoint);
    }
    // Also visit historical shards whose worker no longer exists. Any staging
    // worker can read their immutable WAL; no writer epoch is acquired.
    for shard in &state.config.catchup.shards {
        if seen.insert(shard.clone()) {
            shards.push(shard.clone());
        }
    }
    if endpoints.is_empty() {
        return Err("no staging workers available".into());
    }
    let mut coordinator = AppendCoordinator::open(
        &state.rollout_uri(target),
        Some(lance_context_core::RolloutStore::build_session(
            32 * 1024 * 1024,
            32 * 1024 * 1024,
        )),
    )
    .await
    .map_err(|e| e.to_string())?;
    let started = tokio::time::Instant::now();
    let mut total = 0;
    for _ in 0..16 {
        if started.elapsed() >= Duration::from_secs(state.config.catchup.slice_secs) {
            break;
        }
        let reclaimed = run_pass(state, target, &endpoints, &shards, &mut coordinator).await?;
        total += reclaimed;
        if reclaimed == 0 {
            break;
        }
    }
    Ok(format!("staged append reclaimed {total} generations"))
}

async fn run_pass(
    state: &Arc<MasterState>,
    target: &str,
    endpoints: &[String],
    shards: &[String],
    coordinator: &mut AppendCoordinator,
) -> Result<usize> {
    let config = &state.config.append;
    let (plans, mut reclaimed) = coordinator
        .plan(
            shards,
            config.rollout_append_max_generations,
            config.rollout_append_max_bytes,
        )
        .await
        .map_err(|e| e.to_string())?;
    let stage_futures: Vec<_> = plans.into_iter().enumerate().map(|(i, plan)| {
        let client = state.http.clone();
        let endpoint = endpoints[i % endpoints.len()].clone();
        let fallback = endpoints[(i + 1) % endpoints.len()].clone();
        let target = target.to_owned();
        async move {
            match stage_remote(&client, &endpoint, &target, plan.clone()).await {
                Ok(part) => Ok(part),
                Err(error) => {
                    tracing::warn!(%target, %endpoint, %error, "retrying immutable staging on another worker");
                    stage_remote(&client, &fallback, &target, plan).await
                }
            }
        }.boxed()
    }).collect();
    let mut work = stream::iter(stage_futures).buffer_unordered(config.rollout_append_concurrency);
    let mut group = Vec::new();
    let mut errors = Vec::new();
    let mut flush = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        tokio::select! {
            next = work.next() => match next {
                Some(Ok(part)) => {
                    if group.is_empty() { flush = tokio::time::Instant::now() + Duration::from_secs(2); }
                    group.push(part);
                },
                Some(Err(error)) => errors.push(error),
                None => break,
            },
            _ = tokio::time::sleep_until(flush), if !group.is_empty() => {},
        }
        if group.len() >= config.rollout_append_concurrency
            || (!group.is_empty() && tokio::time::Instant::now() >= flush)
        {
            reclaimed += coordinator
                .commit(std::mem::take(&mut group))
                .await
                .map_err(|e| e.to_string())?;
        }
    }
    if !group.is_empty() {
        reclaimed += coordinator.commit(group).await.map_err(|e| e.to_string())?;
    }
    metrics::counter!("master_rollout_append_generations_reclaimed_total")
        .increment(reclaimed as u64);
    tracing::info!(%target, reclaimed, failed = errors.len(), version = coordinator.version(), "parallel rollout append completed");
    if !errors.is_empty() {
        return Err(format!(
            "staged append reclaimed {reclaimed}; {} shard(s) failed: {}",
            errors.len(),
            errors[0]
        ));
    }
    Ok(reclaimed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        extract::{Path, State},
        routing::{get, post},
        Json, Router,
    };
    use clap::Parser;
    use lance_context_core::{MergeMemoryBudget, RolloutStore, RolloutStoreOptions};
    use serde_json::json;

    #[derive(Clone)]
    struct Workers {
        uri: String,
        barrier: Arc<tokio::sync::Barrier>,
        budget: Arc<MergeMemoryBudget>,
    }

    async fn caps(Path(worker): Path<String>) -> Json<serde_json::Value> {
        Json(
            json!({"rollout_append_protocol": 1, "shard_name": worker, "owned_targets": ["hot"], "drain_targets": []}),
        )
    }
    async fn prepare(State(workers): State<Workers>, Json(plan): Json<AppendPlan>) -> String {
        // This test deadlocks if the master regresses to serial worker fan-out.
        workers.barrier.wait().await;
        let part =
            lance_context_core::rollout_append::stage(&workers.uri, plan, workers.budget, None)
                .await
                .unwrap();
        format!(
            "{}\n",
            serde_json::to_string(&StageEvent::Complete(part)).unwrap()
        )
    }

    #[tokio::test]
    async fn master_stages_two_workers_in_parallel_and_commits_one_version() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir
            .path()
            .join("hot.rollout.lance")
            .to_string_lossy()
            .to_string();
        for shard in ["a", "b"] {
            let store = RolloutStore::open_with_options(
                &uri,
                RolloutStoreOptions {
                    shard_id: Some(shard.into()),
                    merge_after_generations: Some(0),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let dto = serde_json::from_value(
                json!({"id": shard, "rollout_id": "r", "content": "payload"}),
            )
            .unwrap();
            store
                .add(&[lance_context_core::rollout_record_from_add_request(&dto)])
                .await
                .unwrap();
            store.flush().await.unwrap();
        }
        let before = lance::Dataset::open(&uri).await.unwrap().version().version;
        let workers = Workers {
            uri: uri.clone(),
            barrier: Arc::new(tokio::sync::Barrier::new(2)),
            budget: MergeMemoryBudget::new(8 * 1024 * 1024),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = Router::new()
            .route("/{worker}/api/v1/internal/merge-executor", get(caps))
            .route(
                "/{worker}/api/v1/internal/rollout-append/{target}",
                post(prepare),
            )
            .with_state(workers.clone());
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let mut config = crate::config::MasterConfig::parse_from([
            "test",
            "--data-dir",
            dir.path().to_str().unwrap(),
        ]);
        config.worker_endpoints =
            vec![format!("http://{address}/a"), format!("http://{address}/b")];
        config.append.rollout_append_targets = vec!["hot".into()];
        config.append.rollout_append_concurrency = 2;
        let state = MasterState::new(config).await.unwrap();
        let result = tokio::time::timeout(Duration::from_secs(30), run(&state, "hot"))
            .await
            .unwrap()
            .unwrap();
        assert!(result.contains("2 generations"), "{result}");
        let dataset = lance::Dataset::open(&uri).await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
        assert_eq!(
            dataset.version().version,
            before + 2,
            "one cutover metadata version and one combined append"
        );
        assert_eq!(workers.budget.reserved(), 0);
        server.abort();
    }

    #[tokio::test]
    async fn unchanged_remote_progress_does_not_reset_watchdog() {
        let dir = tempfile::tempdir().unwrap();
        let store = RolloutStore::open_with_options(
            dir.path().to_str().unwrap(),
            RolloutStoreOptions {
                shard_id: Some("a".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let dto = serde_json::from_value(json!({"id":"a", "rollout_id":"r"})).unwrap();
        store
            .add(&[lance_context_core::rollout_record_from_add_request(&dto)])
            .await
            .unwrap();
        store.flush().await.unwrap();
        let mut coordinator = AppendCoordinator::open(dir.path().to_str().unwrap(), None)
            .await
            .unwrap();
        let plan = coordinator
            .plan(&["a".into()], 1, 1024 * 1024)
            .await
            .unwrap()
            .0
            .remove(0);
        let part = StagedAppend {
            plan: plan.clone(),
            completed: 1,
            fragments: Vec::new(),
            rows: 0,
        };
        let body = format!(
            "{}\n{}\n{}\n{}\n",
            serde_json::to_string(&StageEvent::Progress(0)).unwrap(),
            serde_json::to_string(&StageEvent::Progress(1)).unwrap(),
            serde_json::to_string(&StageEvent::Progress(1)).unwrap(),
            serde_json::to_string(&StageEvent::Complete(part)).unwrap()
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/api/v1/internal/rollout-append/hot",
            post(move || {
                let body = body.clone();
                async move { body }
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let scope = lance_context_core::merge_write_scope::MergeWriteScope::default();
        scope
            .run(stage_remote(
                &reqwest::Client::new(),
                &format!("http://{address}"),
                "hot",
                plan,
            ))
            .await
            .unwrap();
        assert_eq!(
            scope.completed_steps(),
            2,
            "one actual progress update plus completion"
        );
        server.abort();
    }
}
