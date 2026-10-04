//! Master-owned, bounded Kubernetes catch-up jobs. No payload reads in admission.
mod executor;
mod kubernetes;
mod progress;
pub(crate) mod store;
#[cfg(test)]
mod tests;

use crate::{config::MasterConfig, error::MasterError, state::MasterState, stats_store::StatRow};
use axum::{
    extract::{Query, State},
    Json,
};
pub use executor::execute;
use kubernetes::Kubernetes;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use store::{Inventory, Record};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, clap::Args)]
pub struct CatchupConfig {
    /// Enable autonomous admission. Manual requests use the same policy.
    #[arg(long, env = "CATCHUP_ENABLED", default_value_t = false)]
    pub enabled: bool,
    /// Trusted PodSpec JSON. One container, pinned image, explicit resource bounds.
    #[arg(long, env = "CATCHUP_POD_TEMPLATE")]
    pub pod_template: Option<String>,
    #[arg(long, env = "CATCHUP_NAMESPACE", default_value = "default")]
    pub namespace: String,
    #[arg(long, env = "CATCHUP_MAX_JOBS", default_value_t = 4)]
    pub max_jobs: usize,
    #[arg(long, env = "CATCHUP_MIN_PENDING", default_value_t = 256)]
    pub min_pending: i64,
    #[arg(long, env = "CATCHUP_STATS_MAX_AGE_SECS", default_value_t = 900)]
    pub stats_max_age_secs: u64,
    #[arg(long, env = "CATCHUP_INTERVAL_SECS", default_value_t = 30)]
    pub interval_secs: u64,
    #[arg(long, env = "CATCHUP_SLICE_SECS", default_value_t = 1800)]
    pub slice_secs: u64,
    /// Maximum time without an execution starting, separate from merge progress.
    #[arg(long, env = "CATCHUP_STARTUP_TIMEOUT_SECS", default_value_t = 1800)]
    pub startup_timeout_secs: u64,
    /// Stable writer identities, not Pod IPs. Only sealed generations are merged.
    #[arg(long, env = "CATCHUP_SHARDS", value_delimiter = ',')]
    pub shards: Vec<String>,
    /// Generation count cap in addition to the byte cap; zero disables this cap.
    #[arg(long, env = "CATCHUP_MERGE_MAX_GENERATIONS", default_value_t = 64)]
    pub merge_max_generations: usize,
    #[arg(long, env = "CATCHUP_MERGE_MAX_BYTES", default_value_t = 67_108_864)]
    pub merge_max_bytes: usize,
    #[arg(
        long,
        env = "CATCHUP_MERGE_MEMORY_BYTES",
        default_value_t = 1_073_741_824
    )]
    pub merge_memory_bytes: usize,
    /// Prepare the next shard while the current shard commits. Commits stay ordered.
    #[arg(long, env = "CATCHUP_PIPELINE_ENABLED", default_value_t = true, action = clap::ArgAction::Set)]
    pub pipeline_enabled: bool,
    /// Native one-table executor mode; does not start the admin server or scanner.
    #[arg(long, env = "CATCHUP_TARGET", hide = true)]
    pub target: Option<String>,
    #[arg(long, env = "CATCHUP_JOB_NAME", hide = true)]
    pub job_name: Option<String>,
}
impl Default for CatchupConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            pod_template: None,
            namespace: "default".into(),
            max_jobs: 4,
            min_pending: 256,
            stats_max_age_secs: 900,
            interval_secs: 30,
            slice_secs: 1800,
            startup_timeout_secs: 1800,
            shards: Vec::new(),
            merge_max_generations: 64,
            merge_max_bytes: 67_108_864,
            merge_memory_bytes: 1_073_741_824,
            pipeline_enabled: true,
            target: None,
            job_name: None,
        }
    }
}
impl CatchupConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.enabled && self.target.is_none() && self.pod_template.is_none() {
            return Ok(());
        }
        if self.max_jobs == 0
            || self.max_jobs > 256
            || self.min_pending < 1
            || self.interval_secs == 0
            || self.stats_max_age_secs == 0
            || self.startup_timeout_secs < 120
            || self.startup_timeout_secs > 86_400
            || self.slice_secs < 120
            || self.slice_secs > 86_400
            || self.merge_max_bytes == 0
            || self.merge_memory_bytes < self.merge_max_bytes
            || self.shards.is_empty()
            || self.shards.len() > 256
            || self.shards.iter().any(|s| s.is_empty())
        {
            return Err("invalid catch-up limits or shard identities".into());
        }
        if self.enabled && self.pod_template.is_none() {
            return Err("catch-up requires CATCHUP_POD_TEMPLATE".into());
        }
        if self.target.is_some() && self.job_name.is_none() {
            return Err("executor requires CATCHUP_JOB_NAME".into());
        }
        if self.namespace.is_empty()
            || self.namespace.len() > 63
            || !self
                .namespace
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return Err("invalid catch-up namespace".into());
        }
        if self.enabled {
            kubernetes::read_template(self)?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trigger {
    pub target: String,
    pub reason: String,
    #[serde(default)]
    pub dry_run: bool,
}
#[derive(Debug, Serialize)]
pub struct Decision {
    pub target: String,
    pub decision: String,
    pub job: Option<String>,
}
impl Decision {
    fn new(target: &str, decision: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            decision: decision.into(),
            job: None,
        }
    }
}
#[derive(Deserialize)]
pub struct StatusQuery {
    pub target: String,
}

pub async fn status(
    State(state): State<Arc<MasterState>>,
    Query(query): Query<StatusQuery>,
) -> std::result::Result<Json<Option<Record>>, MasterError> {
    Inventory::new(&state)
        .get(&query.target)
        .await
        .map(Json)
        .map_err(MasterError::Internal)
}
pub async fn trigger(
    State(state): State<Arc<MasterState>>,
    Json(request): Json<Trigger>,
) -> std::result::Result<Json<Decision>, MasterError> {
    if request.target.is_empty()
        || request.target.len() > 512
        || request.reason.trim().is_empty()
        || request.reason.len() > 512
    {
        return Err(MasterError::InvalidRequest(
            "target and reason are required (maximum 512 bytes each)".into(),
        ));
    }
    // Followers may not have run the coordinated scanner. Refresh only this
    // small stats row, with a strict bound; never open the target dataset.
    let mut observed = state
        .stats_cache
        .read()
        .await
        .iter()
        .find(|r| r.name == request.target)
        .cloned();
    if let Ok(mut stats) = state.stats.try_lock() {
        if let Ok(Ok(row)) =
            tokio::time::timeout(Duration::from_secs(2), stats.get(&request.target)).await
        {
            observed = row;
        }
    }
    admit(&state, observed.as_ref(), &request)
        .await
        .map(Json)
        .map_err(MasterError::Internal)
}

fn eligibility(
    config: &MasterConfig,
    row: Option<&StatRow>,
    target: &str,
    now: i64,
) -> Option<&'static str> {
    if !config.catchup.enabled {
        return Some("disabled");
    }
    if !config.merge_rollout.owned(target) || config.merge_rollout.draining(target) {
        return Some("requires_owned_target");
    }
    let Some(row) = row else {
        return Some("no_stats");
    };
    if row.scanned_at > now
        || now.saturating_sub(row.scanned_at)
            > config.catchup.stats_max_age_secs.saturating_mul(1000) as i64
    {
        return Some("stale_stats");
    }
    if row.pending_wal_generations < config.catchup.min_pending {
        return Some("below_threshold");
    }
    None
}

async fn admit(
    state: &Arc<MasterState>,
    row: Option<&StatRow>,
    request: &Trigger,
) -> Result<Decision> {
    let now = chrono::Utc::now().timestamp_millis();
    let target = &request.target;
    if let Some(reason) = eligibility(&state.config, row, target, now) {
        return Ok(Decision::new(target, reason));
    }
    let inventory = Inventory::new(state);
    if let Some(record) = inventory.get(target).await? {
        if record.outcome.as_deref() == Some("succeeded")
            && row.is_some_and(|r| {
                record
                    .finished_at_ms
                    .is_some_and(|finished| r.scanned_at <= finished)
            })
        {
            return Ok(Decision::new(target, "awaiting_fresh_stats"));
        }
        if record.active || record.next_retry_ms > now {
            let mut decision = Decision::new(
                target,
                if record.active {
                    "already_active"
                } else {
                    "cooling_down"
                },
            );
            decision.job = Some(record.job);
            return Ok(decision);
        }
    }
    let coordinator = state.task_store.merge_coordinator();
    if coordinator.get(target).await?.is_some() {
        // The existing reconciler owns recovery. Do not provision idle waiters.
        return Ok(Decision::new(target, "awaiting_execution_recovery"));
    }
    if state
        .task_store
        .is_cooling_down(lance_context_api::TaskKind::MergeWal, target)
        .await
        .map_err(|e| e.to_string())?
    {
        return Ok(Decision::new(target, "cooling_down"));
    }
    if let Some(failure) = coordinator.failure(target, "master:catchup").await? {
        if failure.next_retry_ms > now as u64 {
            return Ok(Decision::new(target, "cooling_down"));
        }
    }
    if let Some(reason) = inventory.blocked(target).await? {
        return Ok(Decision::new(target, reason));
    }
    if request.dry_run {
        return Ok(Decision::new(target, "eligible_dry_run"));
    }
    inventory.ensure_policy(&state.config).await?;
    let result = inventory
        .reserve(
            target,
            &request.reason,
            row.unwrap().pending_wal_generations,
            now,
        )
        .await?;
    tracing::info!(%target, decision = %result.decision, job = ?result.job, reason = %request.reason, "catch-up admission");
    metrics::counter!("master_catchup_admissions_total", "decision" => result.decision.clone())
        .increment(1);
    Ok(result)
}

/// Independent from scheduler slots, scanners and master jobs. Only metadata.
pub fn spawn(state: &Arc<MasterState>) -> Option<tokio::task::JoinHandle<()>> {
    if !state.config.catchup.enabled && state.config.catchup.pod_template.is_none() {
        return None;
    }
    let weak = Arc::downgrade(state);
    Some(tokio::spawn(async move {
        let mut cursor = 0usize;
        loop {
            let Some(state) = weak.upgrade() else {
                return;
            };
            let interval = state.config.catchup.interval_secs;
            if let Err(error) = tick(&state, &mut cursor).await {
                tracing::error!(%error, "catch-up reconciliation failed; scheduler continues");
                metrics::counter!("master_catchup_reconcile_errors_total").increment(1);
            }
            drop(state);
            tokio::time::sleep(Duration::from_secs(interval)).await;
        }
    }))
}
async fn tick(state: &Arc<MasterState>, cursor: &mut usize) -> Result<()> {
    let inventory = Inventory::new(state);
    inventory.ensure_policy(&state.config).await?;
    let kube = Kubernetes::in_cluster(&state.config.catchup)?;
    // The lease reduces duplicate Kubernetes traffic; CAS inventory updates
    // remain correct even if a delayed replica outlives this coordination lease.
    if let Some(guard) = state
        .task_store
        .try_coordination_lock("catchup-jobs")
        .await
        .map_err(|e| e.to_string())?
    {
        use futures::{stream, StreamExt, TryStreamExt};
        let result = async {
            stream::iter(inventory.active().await?).map(|record| {
                let inventory = &inventory;
                let kube = &kube;
                async move {
                    let mut record = record;
                    let mut stop = record.termination_requested;
                    if !stop {
                    match progress::sample(state, &record).await {
                        Ok(sample) => {
                            let now = chrono::Utc::now().timestamp_millis();
                            let observed = progress::observe(record.progress.as_ref(), sample.clone(), now,
                                &state.config.catchup, state.config.maintenance.maintenance_idle_timeout_secs);
                            let mut updated = record.clone();
                            updated.progress = Some(observed);
                            if !inventory.update(&record, &updated).await? { return Ok(()); }
                            record = updated;
                            // Recheck immediately before acting: a stale sample is not
                            // permission to terminate a now-progressing execution.
                            if record.progress.as_ref().is_some_and(|p| p.stalled) {
                                stop = progress::revoke(state, &record, &sample).await?;
                                if stop {
                                    let mut updated = record.clone();
                                    updated.termination_requested = true;
                                    if !inventory.update(&record, &updated).await? { return Ok(()); }
                                    record = updated;
                                }
                            }
                        }
                        Err(error) => {
                            inventory.note_error(&record, &error).await?;
                            return Ok(());
                        }
                    }
                    }
                    match kube.reconcile(&state.config, &record, stop).await {
                        Ok((uid, terminal)) => {
                            if record.job_uid.is_none() && uid.is_some() {
                                let mut updated = record.clone();
                                updated.job_uid = uid;
                                if !inventory.update(&record, &updated).await? { return Ok(()); }
                                record = updated;
                            }
                            if let Some(success) = terminal {
                                inventory.complete(&record, success, chrono::Utc::now().timestamp_millis()).await?;
                            }
                        },
                        Err(error) => {
                            tracing::warn!(target = %record.target, job = %record.job, %error, "catch-up job unresolved; reservation retained");
                            metrics::counter!("master_catchup_job_errors_total").increment(1);
                            inventory.note_error(&record, &error).await?;
                        }
                    }
                    Ok::<_,String>(())
                }
            }).buffer_unordered(8).try_collect::<Vec<_>>().await?;
            Ok::<_,String>(())
        }.await;
        let released = state
            .task_store
            .release_coordination_lock(guard)
            .await
            .map_err(|e| e.to_string());
        result?;
        released?;
    }
    // Never wait behind a stats scan or perform fresh table/payload scans here.
    let snapshot = state.stats_cache.read().await.clone();
    let mut candidates: Vec<_> = snapshot
        .iter()
        .filter(|row| {
            eligibility(
                &state.config,
                Some(row),
                &row.name,
                chrono::Utc::now().timestamp_millis(),
            )
            .is_none()
        })
        .collect();
    candidates.sort_unstable_by_key(|r| std::cmp::Reverse(r.pending_wal_generations));
    let count = candidates.len();
    if count > 0 {
        candidates.rotate_left(*cursor % count);
    }
    *cursor = if count == 0 {
        0
    } else {
        (*cursor + 128) % count
    };
    for row in candidates.into_iter().take(128) {
        let decision = admit(
            state,
            Some(row),
            &Trigger {
                target: row.name.clone(),
                reason: "pending WAL exceeds threshold".into(),
                dry_run: false,
            },
        )
        .await?;
        if decision.decision == "capacity_exhausted" {
            break;
        }
    }
    metrics::gauge!("master_catchup_last_success_timestamp_seconds")
        .set(chrono::Utc::now().timestamp_millis() as f64 / 1000.0);
    Ok(())
}
