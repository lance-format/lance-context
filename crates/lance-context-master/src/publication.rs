//! Synchronous publication of worker-received data for a registered store.

use std::{sync::Arc, time::Duration};

use axum::{
    extract::{Path, State},
    http::StatusCode,
};
use futures::{stream, StreamExt};
use lance_context_core::validate_store_name;

use crate::{error::MasterError, state::MasterState};

/// Publish one managed store on every configured worker before recovery reads.
/// Writers must already be quiesced. This neither drains client queues nor merges
/// WAL generations, and it cannot recover an unavailable or retired writer.
pub(crate) async fn flush_table(
    State(state): State<Arc<MasterState>>,
    Path((kind, name)): Path<(String, String)>,
) -> Result<StatusCode, MasterError> {
    if state.config.managed_store_kind.as_deref() != Some(kind.as_str()) {
        return Err(MasterError::InvalidRequest(
            "store kind is not managed by this master".into(),
        ));
    }
    validate_store_name(&name).map_err(MasterError::InvalidRequest)?;
    let started = std::time::Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(state.config.table_flush_timeout_secs),
        async {
            let registry = state
                .managed_registry
                .as_ref()
                .expect("configured registry");
            if !registry
                .contains(&name)
                .await
                .map_err(MasterError::from_lance)?
            {
                return Err(MasterError::NotFound(format!(
                    "store '{name}' does not exist"
                )));
            }
            match &state.managed_workers {
                Some(workers) => flush_discovered(&state.http, workers, &kind, &name).await,
                None => {
                    flush_workers(&state.http, &state.config.worker_endpoints, &kind, &name).await
                }
            }
        },
    )
    .await
    .unwrap_or_else(|_| {
        Err(MasterError::Unavailable(
            "table publication timed out".into(),
        ))
    });
    metrics::histogram!("master_table_flush_duration_seconds")
        .record(started.elapsed().as_secs_f64());
    metrics::counter!("master_table_flush_total", "result" => if result.is_ok() { "ok" } else { "failed" })
        .increment(1);
    result.map(|()| StatusCode::NO_CONTENT)
}

pub(crate) async fn flush_discovered(
    http: &reqwest::Client,
    workers: &crate::worker_discovery::KubernetesWorkers,
    kind: &str,
    name: &str,
) -> Result<(), MasterError> {
    let before = workers.snapshot().await.map_err(MasterError::Unavailable)?;
    flush_workers(http, &before.endpoints(), kind, name).await?;
    let after = workers.snapshot().await.map_err(MasterError::Unavailable)?;
    if before != after {
        return Err(MasterError::Unavailable(
            "worker Service membership changed during publication; retry the barrier".into(),
        ));
    }
    Ok(())
}

async fn flush_workers(
    http: &reqwest::Client,
    endpoints: &[String],
    kind: &str,
    name: &str,
) -> Result<(), MasterError> {
    if endpoints.is_empty() {
        return Err(MasterError::Unavailable(
            "no worker endpoints configured".into(),
        ));
    }
    let urls = endpoints
        .iter()
        .map(|endpoint| {
            let mut url = reqwest::Url::parse(endpoint).map_err(|error| {
                MasterError::Unavailable(format!("invalid worker endpoint: {error}"))
            })?;
            if !matches!(url.scheme(), "http" | "https")
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return Err(MasterError::Unavailable(
                    "worker endpoint must be an HTTP(S) base URL".into(),
                ));
            }
            url.path_segments_mut()
                .map_err(|_| {
                    MasterError::Unavailable("worker endpoint cannot be a base URL".into())
                })?
                .pop_if_empty()
                .extend(["api", "v1", kind, name, "flush-local"]);
            Ok(url)
        })
        .collect::<Result<Vec<_>, MasterError>>()?;
    let results = stream::iter(urls)
        .map(|url| async move {
            let response = http.post(url.clone()).send().await.map_err(|error| {
                MasterError::Unavailable(format!("worker publication failed at {url}: {error}"))
            })?;
            if response.status() != StatusCode::NO_CONTENT {
                return Err(MasterError::Unavailable(format!(
                    "worker publication at {url} returned {}",
                    response.status()
                )));
            }
            Ok(())
        })
        .buffer_unordered(8)
        .collect::<Vec<_>>()
        .await;
    // A failed worker must not cancel the other publications or turn a partial
    // flush into success. The caller's deadline bounds the complete fan-out.
    for result in results {
        result?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
