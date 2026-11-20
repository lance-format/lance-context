//! Synchronous publication of worker-received data for a registered store.

use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};

use axum::{
    extract::{Path, State},
    http::StatusCode,
};
use futures::{stream, StreamExt};
use lance_context_core::validate_store_name;
use reqwest::header::{HeaderMap, RETRY_AFTER};

use crate::{error::MasterError, state::MasterState};

const WORKER_FLUSH_MAX_RETRIES: u32 = 3;
const WORKER_FLUSH_RETRY_BACKOFF: Duration = Duration::from_millis(500);

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
        .map(|url| flush_worker(http, url))
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

async fn flush_worker(http: &reqwest::Client, url: reqwest::Url) -> Result<(), MasterError> {
    for retries in 0..=WORKER_FLUSH_MAX_RETRIES {
        if retries > 0 {
            metrics::counter!("master_table_flush_worker_retries_total").increment(1);
        }
        let (error, retryable, retry_after) = match http.post(url.clone()).send().await {
            Ok(response) => {
                let status = response.status();
                if status == StatusCode::NO_CONTENT {
                    return Ok(());
                }
                (
                    format!("worker publication at {url} returned {status}"),
                    matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504),
                    publication_retry_after(response.headers()),
                )
            }
            Err(error) => (
                format!("worker publication failed at {url}: {error}"),
                error.is_connect() || error.is_timeout() || error.is_request(),
                None,
            ),
        };
        if !retryable || retries == WORKER_FLUSH_MAX_RETRIES {
            return Err(MasterError::Unavailable(error));
        }
        // Publication is idempotent even if the worker published but its response
        // was lost. Keep successful peers untouched and retain the original
        // membership snapshot for the final discovery check. The enclosing
        // table deadline covers requests, backoff and Retry-After together.
        let delay = (WORKER_FLUSH_RETRY_BACKOFF * (1 << retries))
            .mul_f64(0.5 + rand::random::<f64>() * 0.5)
            .max(retry_after.unwrap_or_default());
        tracing::warn!(
            worker = %url,
            retry = retries + 1,
            delay_ms = delay.as_millis(),
            %error,
            "retrying worker table publication"
        );
        tokio::time::sleep(delay).await;
    }
    unreachable!("the final attempt returns success or its publication error")
}

fn publication_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?;
    value
        .parse::<u64>()
        .map(Duration::from_secs)
        .ok()
        .or_else(|| {
            httpdate::parse_http_date(value)
                .ok()
                .map(|date| date.duration_since(SystemTime::now()).unwrap_or_default())
        })
}

#[cfg(test)]
mod tests;
