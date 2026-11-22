//! Independent index discovery; execution uses the normal IndexId queue/fence.
use std::{sync::Arc, time::Duration};

use futures::{stream, StreamExt};
use lance_context_api::TaskKind;

use crate::{
    registered_store, scheduler, state::MasterState, stats_store::StatRow, task_store::TaskStore,
};

pub(crate) async fn sweep(
    state: &Arc<MasterState>,
    publisher: Option<&TaskStore>,
) -> lance::Result<usize> {
    let Some(guard) = state
        .task_store
        .try_coordination_lock("index-sweep")
        .await?
    else {
        return Ok(0);
    };
    let result = sweep_inner(state, publisher).await;
    let release = state.task_store.release_coordination_lock(guard).await;
    match (result, release) {
        (Ok(count), Ok(())) => Ok(count),
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
    }
}

async fn needs_index(state: &MasterState, row: &StatRow) -> lance::Result<bool> {
    if state
        .task_store
        .is_cooling_down(TaskKind::IndexId, &row.name)
        .await?
        || state
            .task_store
            .get_active_id(TaskKind::IndexId, &row.name)
            .await?
            .is_some()
    {
        return Ok(false);
    }
    let uri = scheduler::maintenance_uri(state, &row.name)
        .await
        .map_err(lance::Error::io)?;
    let store = registered_store::open(state, &uri).await?;
    // Stale stats can still list an empty base. WAL-only data is left to merge.
    if store.dataset().get_fragments().is_empty() {
        return Ok(false);
    }
    Ok(
        registered_store::index_work(&store, state.config.key_index_type)
            .await?
            .needed(),
    )
}

async fn sweep_inner(
    state: &Arc<MasterState>,
    publisher: Option<&TaskStore>,
) -> lance::Result<usize> {
    // Inventory comes from the existing scanner; coverage is checked against a
    // fresh manifest, not inferred from the stats row or a compaction threshold.
    let rows = state
        .stats
        .lock()
        .await
        .list_above_fragment_count(1, usize::MAX)
        .await?;
    let mut checks = stream::iter(
        rows.into_iter()
            .filter(|row| state.registered_target(&row.name).is_some() || !row.name.contains(':')),
    )
    .map(|row| async move {
        let result = tokio::time::timeout(Duration::from_secs(30), needs_index(state, &row)).await;
        (row, result)
    })
    .buffer_unordered(state.config.scan_concurrency.max(1));
    let mut queued = 0;
    while let Some((row, result)) = checks.next().await {
        let result =
            result.unwrap_or_else(|_| Err(lance::Error::io("index coverage check timed out")));
        let label = match &result {
            Ok(true) => "needed",
            Ok(false) => "skipped",
            Err(_) => "error",
        };
        metrics::counter!("master_index_sweep_checks_total", "result" => label).increment(1);
        match result {
            Ok(true) => {
                let before = state
                    .task_store
                    .get_active_id(TaskKind::IndexId, &row.name)
                    .await?;
                let task =
                    scheduler::enqueue_sweep_candidate(state, publisher, TaskKind::IndexId, &row)
                        .await?;
                if before.as_deref() != Some(task.id.as_str()) {
                    queued += 1;
                    if queued >= scheduler::MAX_SWEEP_ENQUEUE {
                        break;
                    }
                }
            }
            Ok(false) => {}
            Err(error) => tracing::warn!(target = %row.name, %error, "index coverage check failed"),
        }
    }
    Ok(queued)
}
