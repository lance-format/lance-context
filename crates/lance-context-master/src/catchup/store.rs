use super::{outcome::JobOutcome, Decision, Result};
use crate::state::MasterState;
use etcd_client::{Client, Compare, CompareOp, GetOptions, Txn, TxnOp};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub target: String,
    pub job: String,
    #[serde(default)]
    pub job_uid: Option<String>,
    pub slot: usize,
    pub attempt: u64,
    pub active: bool,
    #[serde(default)]
    pub termination_requested: bool,
    pub reason: String,
    pub pending_at_admission: i64,
    pub created_at_ms: i64,
    pub finished_at_ms: Option<i64>,
    pub consecutive_failures: u32,
    pub needs_attention: bool,
    pub next_retry_ms: i64,
    pub outcome: Option<String>,
    #[serde(default)]
    pub progress: Option<super::progress::Observation>,
}
pub(super) struct Inventory {
    pub client: Client,
    pub prefix: String,
    max_jobs: usize,
}
pub(crate) fn active_key(prefix: &str, target: &str) -> String {
    lance_context_merge::execution_key(prefix, target)
        .replace("/merge-executions/", "/catchup-active/")
}
impl Inventory {
    pub fn new(state: &MasterState) -> Self {
        Self {
            client: state.task_store.etcd_client().clone(),
            prefix: state.config.etcd.etcd_prefix.trim_end_matches('/').into(),
            max_jobs: state.config.catchup.max_jobs,
        }
    }
    fn record_key(&self, target: &str) -> String {
        lance_context_merge::execution_key(&self.prefix, target)
            .replace("/merge-executions/", "/catchup-records/")
    }
    fn slot_key(&self, slot: usize) -> String {
        format!("{}/catchup-slots/{slot:03}", self.prefix)
    }
    pub async fn get(&self, target: &str) -> Result<Option<Record>> {
        let response = self
            .client
            .clone()
            .get(self.record_key(target), None)
            .await
            .map_err(|e| e.to_string())?;
        response
            .kvs()
            .first()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()
    }
    pub async fn ensure_policy(&self, master: &crate::config::MasterConfig) -> Result<()> {
        let config = &master.catchup;
        let template = super::kubernetes::read_template(config)?;
        let mut policy = serde_json::json!({"namespace":config.namespace,"max_jobs":config.max_jobs,"template":template,
            "shards":config.shards,"merge_max_bytes":config.merge_max_bytes,"merge_memory_bytes":config.merge_memory_bytes,"slice":config.slice_secs,"startup_timeout":config.startup_timeout_secs,"idle_timeout":master.maintenance.maintenance_idle_timeout_secs});
        // Preserve the existing policy bytes when continuous service is off.
        if !config.continuous_targets.is_empty() {
            let mut targets = config.continuous_targets.clone();
            targets.sort();
            targets.dedup();
            policy["continuous_targets"] = serde_json::json!(targets);
        }
        // Default-off preserves existing policy bytes. Controllers opting in
        // must agree on the per-shard admission bound before creating Jobs.
        if config.grouped_prepare_max_bytes != 0 {
            policy["grouped_prepare_max_bytes"] =
                serde_json::json!(config.grouped_prepare_max_bytes);
        }
        if config.grouped_parallel_encoding {
            policy["grouped_parallel_encoding"] = serde_json::json!(true);
        }
        let policy = policy.to_string();
        let key = format!("{}/catchup-policy", self.prefix);
        let mut client = self.client.clone();
        client
            .txn(
                Txn::new()
                    .when([Compare::version(key.as_str(), CompareOp::Equal, 0)])
                    .and_then([TxnOp::put(key.as_str(), policy.clone(), None)]),
            )
            .await
            .map_err(|e| e.to_string())?;
        let response = client.get(key, None).await.map_err(|e| e.to_string())?;
        if response
            .kvs()
            .first()
            .is_none_or(|kv| kv.value() != policy.as_bytes())
        {
            return Err(
                "catch-up policy differs across masters; keep the existing policy until jobs drain"
                    .into(),
            );
        }
        Ok(())
    }
    pub async fn active(&self) -> Result<Vec<Record>> {
        let response = self
            .client
            .clone()
            .get(
                format!("{}/catchup-slots/", self.prefix),
                Some(GetOptions::new().with_prefix().with_limit(257)),
            )
            .await
            .map_err(|e| e.to_string())?;
        if response.kvs().len() > 256 || response.more() {
            return Err("catch-up inventory exceeds configured bounds".into());
        }
        response
            .kvs()
            .iter()
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .collect()
    }
    pub async fn blocked(&self, target: &str) -> Result<Option<&'static str>> {
        let key = lance_context_merge::target_lock_key(&self.prefix, target);
        if !self
            .client
            .clone()
            .get(key, None)
            .await
            .map_err(|e| e.to_string())?
            .kvs()
            .is_empty()
        {
            return Ok(Some("awaiting_current_task"));
        }
        if self.active().await?.len() >= self.max_jobs {
            return Ok(Some("capacity_exhausted"));
        }
        Ok(None)
    }
    pub async fn note_error(&self, record: &Record, error: &str) -> Result<()> {
        let mut updated = record.clone();
        updated.outcome = Some(error.chars().take(1024).collect());
        updated.needs_attention = true;
        updated.progress = None;
        self.update(record, &updated).await?;
        Ok(())
    }
    pub async fn update(&self, record: &Record, updated: &Record) -> Result<bool> {
        let old = serde_json::to_vec(record).map_err(|e| e.to_string())?;
        let new = serde_json::to_vec(&updated).map_err(|e| e.to_string())?;
        Ok(self
            .client
            .clone()
            .txn(
                Txn::new()
                    .when([
                        Compare::value(
                            self.record_key(&record.target),
                            CompareOp::Equal,
                            old.clone(),
                        ),
                        Compare::value(self.slot_key(record.slot), CompareOp::Equal, old),
                    ])
                    .and_then([
                        TxnOp::put(self.record_key(&record.target), new.clone(), None),
                        TxnOp::put(self.slot_key(record.slot), new, None),
                    ]),
            )
            .await
            .map_err(|e| e.to_string())?
            .succeeded())
    }
    pub async fn reserve(
        &self,
        target: &str,
        reason: &str,
        pending: i64,
        now: i64,
    ) -> Result<Decision> {
        let old = self.get(target).await?;
        if old
            .as_ref()
            .is_some_and(|r| r.active || r.next_retry_ms > now)
        {
            return Ok(Decision::new(target, "already_active_or_cooling_down"));
        }
        let key = self.record_key(target);
        let active = active_key(&self.prefix, target);
        let execution = lance_context_merge::execution_key(&self.prefix, target);
        let merge_claim = execution.replace("/merge-executions/", "/merge-claims/");
        let target_lock = lance_context_merge::target_lock_key(&self.prefix, target);
        let job = format!(
            "lc-catchup-{}",
            lance_context_core::generate_id().to_lowercase()
        );
        for slot in 0..self.max_jobs {
            let record = Record {
                target: target.into(),
                job: job.clone(),
                job_uid: None,
                slot,
                attempt: old.as_ref().map_or(1, |r| r.attempt.saturating_add(1)),
                active: true,
                termination_requested: false,
                reason: reason.into(),
                pending_at_admission: pending,
                created_at_ms: now,
                finished_at_ms: None,
                consecutive_failures: old.as_ref().map_or(0, |r| r.consecutive_failures),
                needs_attention: old.as_ref().is_some_and(|r| r.needs_attention),
                next_retry_ms: old.as_ref().map_or(0, |r| r.next_retry_ms),
                outcome: None,
                progress: None,
            };
            let value = serde_json::to_vec(&record).map_err(|e| e.to_string())?;
            let prior = match &old {
                Some(old) => Compare::value(
                    key.as_str(),
                    CompareOp::Equal,
                    serde_json::to_vec(old).map_err(|e| e.to_string())?,
                ),
                None => Compare::version(key.as_str(), CompareOp::Equal, 0),
            };
            if self
                .client
                .clone()
                .txn(
                    Txn::new()
                        .when([
                            prior,
                            Compare::version(self.slot_key(slot), CompareOp::Equal, 0),
                            Compare::version(active.as_str(), CompareOp::Equal, 0),
                            Compare::version(execution.as_str(), CompareOp::Equal, 0),
                            Compare::version(merge_claim.as_str(), CompareOp::Equal, 0),
                            Compare::version(target_lock.as_str(), CompareOp::Equal, 0),
                        ])
                        .and_then([
                            TxnOp::put(key.as_str(), value.clone(), None),
                            TxnOp::put(self.slot_key(slot), value, None),
                            TxnOp::put(active.as_str(), job.as_str(), None),
                        ]),
                )
                .await
                .map_err(|e| e.to_string())?
                .succeeded()
            {
                let mut decision = Decision::new(target, "reserved");
                decision.job = Some(job);
                return Ok(decision);
            }
        }
        // Distinguish busy tables from an actually exhausted fleet budget.
        let full = self.active().await?.len() >= self.max_jobs;
        Ok(Decision::new(
            target,
            if full {
                "capacity_exhausted"
            } else {
                "table_busy_or_admission_raced"
            },
        ))
    }
    pub(super) async fn complete(
        &self,
        record: &Record,
        outcome: JobOutcome,
        now: i64,
    ) -> Result<()> {
        let mut terminal = record.clone();
        terminal.active = false;
        terminal.finished_at_ms = Some(now);
        let delay_secs = match outcome {
            JobOutcome::Succeeded => {
                terminal.consecutive_failures = 0;
                terminal.needs_attention = false;
                terminal.outcome = Some("succeeded".into());
                60
            }
            JobOutcome::Deferred => {
                // Admission acquired no work. Keep earned failure debt and
                // attention, including deadlines surviving a controller restart.
                terminal.outcome = Some("deferred_before_claim".into());
                // Stable per-Job jitter; no synchronized etcd retry burst.
                3 + record.job.bytes().fold(0u64, |n, b| (n + u64::from(b)) % 3)
            }
            JobOutcome::Failed => {
                terminal.consecutive_failures = record.consecutive_failures.saturating_add(1);
                terminal.needs_attention = terminal.consecutive_failures >= 3;
                terminal.outcome = Some("failed; inspect Job status and executor logs".into());
                (60u64.saturating_mul(1u64 << terminal.consecutive_failures.min(6))).min(3600)
            }
        };
        terminal.next_retry_ms = now.saturating_add(delay_secs as i64 * 1000);
        if outcome == JobOutcome::Deferred {
            terminal.next_retry_ms = terminal.next_retry_ms.max(record.next_retry_ms);
        }
        let old = serde_json::to_vec(record).map_err(|e| e.to_string())?;
        let new = serde_json::to_vec(&terminal).map_err(|e| e.to_string())?;
        let active = active_key(&self.prefix, &record.target);
        let changed = self
            .client
            .clone()
            .txn(
                Txn::new()
                    .when([
                        Compare::value(
                            self.record_key(&record.target),
                            CompareOp::Equal,
                            old.clone(),
                        ),
                        Compare::value(self.slot_key(record.slot), CompareOp::Equal, old),
                        Compare::value(active.as_str(), CompareOp::Equal, record.job.as_bytes()),
                    ])
                    .and_then([
                        TxnOp::put(self.record_key(&record.target), new, None),
                        TxnOp::delete(self.slot_key(record.slot), None),
                        TxnOp::delete(active, None),
                    ]),
            )
            .await
            .map_err(|e| e.to_string())?
            .succeeded();
        if changed {
            tracing::info!(target = %record.target, job = %record.job, ?outcome, failures = terminal.consecutive_failures, "catch-up job terminal; capacity released");
            metrics::counter!("master_catchup_jobs_finished_total", "result" => match outcome { JobOutcome::Succeeded => "ok", JobOutcome::Deferred => "deferred", JobOutcome::Failed => "failed" }).increment(1);
        }
        Ok(())
    }
}
