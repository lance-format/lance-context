//! Native maintenance execution inside each master, sharing its local buffer budget.
use crate::{config::MasterConfig, state::MasterState};
use std::sync::Arc;

#[derive(Clone, Debug, Default, clap::Args)]
pub struct LocalMergeConfig {
    /// Route new rollout/generic demand to resident masters without a static
    /// table allowlist. Enable only after every legacy writer/helper has drained
    /// and all participants use persistent fenced admission. This flag is not
    /// evidence of that initial transition; retain its deployment evidence.
    #[arg(long, env = "MASTER_LOCAL_MERGE_AUTO", default_value_t = false)]
    pub master_local_merge_auto: bool,
    /// Execute sealed WAL merges in the master, including generic upsert tables.
    /// '*' selects explicitly owned targets unless automatic routing is enabled.
    /// Neither setting certifies that historical legacy writers have drained.
    #[arg(long, env = "MASTER_LOCAL_MERGE_TARGETS", value_delimiter = ',')]
    pub master_local_merge_targets: Vec<String>,
}

impl LocalMergeConfig {
    pub(crate) fn selected(&self, target: &str) -> bool {
        self.master_local_merge_targets
            .iter()
            .any(|t| t == "*" || t == target)
    }
}

pub(crate) fn enabled(config: &MasterConfig) -> bool {
    config.append.rollout_append_local || !config.local_merge.master_local_merge_targets.is_empty()
}

pub(crate) fn selected(config: &MasterConfig, target: &str) -> bool {
    let name = target.strip_prefix("generic:").unwrap_or(target);
    let dynamic = config.local_merge.master_local_merge_auto
        && config.merge_rollout.fence_legacy
        && !target.starts_with("datagen:")
        && lance_context_core::validate_store_name(name).is_ok();
    (config.merge_rollout.owned(target) || dynamic)
        && !config.merge_rollout.draining(target)
        && (config.append.local(target) || config.local_merge.selected(target))
}

/// Resolve the local selection before claiming. In particular an append wildcard
/// cannot accidentally reserve a slot for a generic table that still needs RPCs.
pub(crate) fn targets(config: &MasterConfig) -> Vec<String> {
    if config.local_merge.master_local_merge_auto {
        return vec!["*".into()];
    }
    config
        .merge_rollout
        .owned_targets
        .iter()
        .filter(|target| selected(config, target))
        .cloned()
        .collect()
}

pub(crate) fn validate(config: &MasterConfig) -> Result<(), String> {
    if config.local_merge.master_local_merge_auto
        && (!config.merge_rollout.fence_legacy
            || config.local_merge.master_local_merge_targets != ["*"]
            || !config.maintenance.fence_legacy_maintenance)
    {
        return Err("automatic local merges require persistent worker fencing, fenced master maintenance and MASTER_LOCAL_MERGE_TARGETS=*; initial legacy writer drain must be verified before enabling".into());
    }
    if config.local_merge.master_local_merge_targets.is_empty() {
        return Ok(());
    }
    let catchup = &config.catchup;
    if catchup.enabled || catchup.pod_template.is_some() || catchup.target.is_some() {
        return Err("master local merges cannot be combined with Kubernetes catch-up mode".into());
    }
    if catchup.shards.is_empty()
        || catchup.shards.len() > 256
        || catchup.shards.iter().any(|s| s.is_empty())
        || catchup.slice_secs == 0
        || catchup.merge_max_bytes == 0
        || config.append.rollout_append_local_memory_bytes < catchup.merge_max_bytes
        || config.merge_wal_concurrency == 0
    {
        return Err("master local merges require stable shards, bounded passes and a sufficient shared memory budget".into());
    }
    if config
        .merge_rollout
        .owned_targets
        .iter()
        .any(|target| selected(config, target) && config.append.enabled(target))
    {
        config.append.validate()?;
        if config.append.rollout_append_local_memory_bytes < config.append.rollout_append_max_bytes
        {
            return Err("local append batch exceeds shared memory budget".into());
        }
    }
    for target in &config.local_merge.master_local_merge_targets {
        if target != "*" && !selected(config, target) {
            return Err(format!(
                "master local merge requires a non-draining owned target: {target}"
            ));
        }
    }
    Ok(())
}

/// A prepared maintenance task already has a leased commit-turn marker which
/// prevents the next merge from stealing its turn. Observe it only after all
/// work in a shard pass completed. Mid-pass yields could repeatedly restart at
/// the first hot shard and starve the remaining shards.
pub(crate) async fn yield_to_maintenance(state: &MasterState, target: &str) -> bool {
    if state.config.catchup.target.is_some() || !selected(&state.config, target) {
        return false;
    }
    let reason = match state.task_store.maintenance_commit_requested(target).await {
        Ok(false) => return false,
        Ok(true) => "maintenance_waiting",
        Err(error) => {
            tracing::warn!(%target, %error, "yielding completed merge pass after commit-turn check failed");
            "check_unavailable"
        }
    };
    metrics::counter!("master_native_merge_pass_yields_total", "reason" => reason).increment(1);
    true
}

/// Closing executor admission also stops new resident batches. Already admitted
/// preparation/commits are joined normally; this is not a cancellation or fence.
pub(crate) fn drain_requested(state: &MasterState, target: &str) -> bool {
    !state.admission.is_accepting()
        && state.config.catchup.target.is_none()
        && selected(&state.config, target)
}

/// The caller owns the canonical task and maintenance fence. No private queue,
/// Pod, writer epoch, or independent per-table memory budget is created here.
pub(crate) async fn run(state: &Arc<MasterState>, target: &str) -> Result<String, String> {
    if !selected(&state.config, target) {
        return Err("target is not configured for master local merge".into());
    }
    let local = state
        .local_append
        .as_ref()
        .ok_or("missing shared local merge resources")?;
    crate::catchup::merge_local(state, target, local.session.clone(), local.budget.clone()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn local_selection_preserves_legacy_and_append_boundaries() {
        let mut config = MasterConfig::parse_from(["test"]);
        config.merge_rollout.owned_targets = vec!["rollout".into(), "generic:rows".into()];
        config.append.rollout_append_local = true;
        config.append.rollout_append_targets = vec!["*".into()];
        assert_eq!(targets(&config), ["rollout"]);
        config.local_merge.master_local_merge_targets = vec!["*".into()];
        assert_eq!(targets(&config), ["rollout", "generic:rows"]);
        assert!(!selected(&config, "legacy"));
        config.merge_rollout.drain_targets = vec!["generic:rows".into()];
        assert_eq!(targets(&config), ["rollout"]);
    }

    #[test]
    fn local_validation_requires_safe_resources_and_ownership() {
        let mut config = MasterConfig::parse_from(["test"]);
        config.local_merge.master_local_merge_targets = vec!["generic:rows".into()];
        config.catchup.shards = vec!["writer".into()];
        assert!(validate(&config).is_err());
        config.merge_rollout.owned_targets = vec!["generic:rows".into()];
        assert!(validate(&config).is_ok());
        config.append.rollout_append_local_memory_bytes = config.catchup.merge_max_bytes - 1;
        assert!(validate(&config).is_err());
        config.append.rollout_append_local_memory_bytes = config.catchup.merge_max_bytes;
        config.catchup.enabled = true;
        assert!(validate(&config).is_err());
        config.catchup.enabled = false;
        config.catchup.shards.clear();
        assert!(validate(&config).is_err());
        config.catchup.shards = vec!["writer".into()];
        config.local_merge.master_local_merge_targets = vec!["*".into()];
        config.merge_rollout.owned_targets.push("rollout".into());
        config.append.rollout_append_targets = vec!["rollout".into()];
        config.append.rollout_append_max_bytes =
            config.append.rollout_append_local_memory_bytes + 1;
        assert!(validate(&config).is_err());
    }
    #[test]
    fn automatic_selection_requires_fenced_protocol_and_bounded_resources() {
        let mut cfg = MasterConfig::parse_from(["test"]);
        cfg.local_merge.master_local_merge_auto = true;
        cfg.local_merge.master_local_merge_targets = vec!["*".into()];
        cfg.catchup.shards = vec!["writer".into()];
        assert!(validate(&cfg).is_err());
        assert!(!selected(&cfg, "new-hot"));
        cfg.merge_rollout.fence_legacy = true;
        assert!(validate(&cfg).is_err());
        cfg.maintenance.fence_legacy_maintenance = true;
        assert!(validate(&cfg).is_ok());
        assert!(selected(&cfg, "new-hot"));
        assert!(selected(&cfg, "generic:new-hot"));
        assert!(!selected(&cfg, "datagen:new-hot"));
        cfg.merge_rollout.drain_targets.push("new-hot".into());
        assert!(!selected(&cfg, "new-hot"));
        assert_eq!(targets(&cfg), ["*"]);
        cfg.catchup.merge_max_bytes = cfg.append.rollout_append_local_memory_bytes + 1;
        assert!(validate(&cfg).is_err());
    }
}

#[cfg(test)]
#[path = "local_merge_drain_tests.rs"]
mod drain_tests;
