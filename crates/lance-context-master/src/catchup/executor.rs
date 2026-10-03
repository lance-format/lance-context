use super::Result;
use crate::{config::MasterConfig, maintenance_execution, state::MasterState};
use lance_context_api::TaskKind;
use lance_context_core::{
    GenericStore, GenericStoreOptions, MergeMemoryBudget, RolloutStore, RolloutStoreOptions,
};
use lance_context_merge::MaintenanceKind;
use std::{sync::Arc, time::Duration};

/// One dedicated process, one table, serial shards, bounded passes. No server,
/// scanner, ingestion writer, or fleet task consumer starts in this mode.
pub async fn execute(mut config: MasterConfig, target: &str) -> Result<()> {
    config.catchup.enabled = false;
    config.catchup.validate()?;
    if !config.merge_rollout.owned(target) || config.merge_rollout.draining(target) {
        return Err("catch-up executor requires a non-draining owned target".into());
    }
    let job = config
        .catchup
        .job_name
        .clone()
        .ok_or("missing catch-up Job identity")?;
    let state = MasterState::new(config).await.map_err(|e| e.to_string())?;
    // Enqueue through the normal durable queue, then claim only this table.
    state
        .task_store
        .enqueue(TaskKind::MergeWal, target, Vec::new())
        .await
        .map_err(|e| e.to_string())?;
    let claim = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Some(claim) = state
                .task_store
                .claim_merge_target(target, &job)
                .await
                .map_err(|e| e.to_string())?
            {
                return Ok::<_, String>(claim);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    })
    .await
    .map_err(|_| "catch-up admission remained busy for 30 seconds")??;
    let result = async {
        // The claim CAS has excluded any live reconciler. A previous process
        // may still have a storage PUT in flight: fence before opening writers.
        let coordinator = state.task_store.merge_coordinator();
        if let Some(old) = coordinator.get(target).await? {
            crate::merge_execution::ensure_recovery_due(&coordinator, &old).await?;
            crate::merge_execution::recover_execution(
                &state,
                &coordinator,
                &state.task_store.merge_claim(&claim),
                old,
            )
            .await?;
        }
        maintenance_execution::run_as(
            &state,
            &claim,
            MaintenanceKind::Catchup,
            merge_passes(&state, target),
        )
        .await
    }
    .await;
    let error = result.as_ref().err().cloned();
    state
        .task_store
        .finish(claim, result)
        .await
        .map_err(|e| e.to_string())?;
    if let Some(error) = error {
        return Err(error);
    }
    Ok(())
}

async fn merge_passes(state: &Arc<MasterState>, target: &str) -> Result<String> {
    let config = &state.config.catchup;
    let session = RolloutStore::build_session(96 * 1024 * 1024, 32 * 1024 * 1024);
    let budget = MergeMemoryBudget::new(config.merge_memory_bytes);
    let mut total = 0usize;
    let started = tokio::time::Instant::now();
    // This is only an admission time slice. A progressing operation already
    // admitted may run beyond it; cancellation is based on lack of progress.
    let admit_for = Duration::from_secs(config.slice_secs);
    // Bound each slice even under continuous ingestion; later fresh stats may
    // request another Job. Visit every shard before repeating any hot shard.
    for _ in 0..16 {
        let before = total;
        for shard in &config.shards {
            if started.elapsed() >= admit_for {
                return Ok(format!(
                    "dedicated catch-up slice merged {total} generations"
                ));
            }
            let reclaimed = if let Some(name) = target.strip_prefix("generic:") {
                let options = GenericStoreOptions {
                    shard_id: Some(shard.clone()),
                    merge_after_generations: Some(0),
                    merge_max_generations: Some(8),
                    merge_max_bytes: Some(config.merge_max_bytes),
                    key_index_type: state.config.key_index_type,
                    merge_budget: Some(budget.clone()),
                    session: Some(session.clone()),
                    pending_generations_max: Some(0),
                    ..Default::default()
                };
                let mut store = GenericStore::open_existing(&state.generic_uri(name), options)
                    .await
                    .map_err(|e| e.to_string())?;
                // No resident writer was opened, so cleanup only sees sealed WAL.
                match store
                    .prepare_cleanup_merge()
                    .await
                    .map_err(|e| e.to_string())?
                {
                    Some((manifest_store, manifest, prepared)) => store
                        .commit_prepared_merge(&manifest_store, &manifest, prepared)
                        .await
                        .map_err(|e| e.to_string())?,
                    None => 0,
                }
            } else {
                let options = RolloutStoreOptions {
                    shard_id: Some(shard.clone()),
                    merge_after_generations: Some(0),
                    merge_max_generations: Some(8),
                    merge_max_bytes: Some(config.merge_max_bytes),
                    key_index_type: state.config.key_index_type,
                    merge_budget: Some(budget.clone()),
                    session: Some(session.clone()),
                    pending_generations_max: Some(0),
                    ..Default::default()
                };
                let mut store =
                    RolloutStore::open_existing_with_options(&state.rollout_uri(target), options)
                        .await
                        .map_err(|e| e.to_string())?;
                match store
                    .prepare_merge_if_ready(1)
                    .await
                    .map_err(|e| e.to_string())?
                {
                    Some((manifest_store, manifest, prepared)) => store
                        .commit_prepared_merge(&manifest_store, &manifest, prepared)
                        .await
                        .map_err(|e| e.to_string())?,
                    None => 0,
                }
            };
            total += reclaimed;
            metrics::counter!("master_catchup_generations_reclaimed_total")
                .increment(reclaimed as u64);
            tracing::info!(%target, %shard, reclaimed, total, "dedicated catch-up committed");
        }
        if total == before {
            break;
        }
    }
    Ok(format!("dedicated catch-up merged {total} generations"))
}
