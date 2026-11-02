//! Cross-process Context HTTP qualification driver. Never accepts production URIs.
#![recursion_limit = "256"]

use lance_context_api::AddRecordRequest;
use lance_context_core::{
    record_from_add_request, ContextStore, ContextStoreOptions, MergeMemoryBudget,
};
use std::io::{self, Write};
use std::time::Duration;

const PREFIX: &str = "abfss://snowflake-rollouts@falconphxdataprocessing.dfs.core.windows.net/rocketkeep-staging/wal-lifetime-47748/";

fn emit(value: serde_json::Value) {
    println!("{value}");
    io::stdout().flush().unwrap();
}

fn command(expected: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    if line.trim() != expected {
        return Err(format!("expected staging command {expected}").into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let uri = std::env::args()
        .nth(1)
        .ok_or("missing isolated table URI")?;
    let name = uri.strip_prefix(PREFIX).ok_or("refusing non-staging URI")?;
    let nonce = name
        .strip_prefix("stage_context_")
        .and_then(|s| s.strip_suffix(".lance"))
        .ok_or("refusing non-fixture table")?;
    if nonce.len() != 32 || !nonce.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("fixture name must contain a fresh UUID nonce".into());
    }
    if !matches!(
        std::env::var("ROCKETKEEP_WAL_LIFECYCLE").as_deref(),
        Ok("retain" | "collect")
    ) {
        return Err("explicit staging WAL lifecycle mode required".into());
    }
    if std::env::var("ETCD_ENDPOINTS").is_ok_and(|s| !s.is_empty()) {
        return Err("staging driver must not inherit etcd endpoints".into());
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let mut store = rt.block_on(async {
        tokio::time::timeout(
            Duration::from_secs(60),
            ContextStore::open_with_options(
                &uri,
                ContextStoreOptions {
                    shard_id: Some(format!("stage-context-{nonce}")),
                    embedding_dim: Some(3),
                    merge_max_generations: Some(8),
                    merge_max_bytes: Some(64 * 1024 * 1024),
                    merge_budget: Some(MergeMemoryBudget::new(256 * 1024 * 1024)),
                    ..Default::default()
                },
            ),
        )
        .await
    })??;
    let records: Vec<_> = (0..8)
        .map(|i| {
            record_from_add_request(
                &AddRecordRequest {
                    role: "assistant".into(),
                    content_type: "application/octet-stream".into(),
                    text_payload: Some(format!("context-payload-{i}-{}", "c".repeat(8192))),
                    binary_payload: Some(vec![i as u8; 8192]),
                    external_id: Some(format!("external-{i}")),
                    embedding: Some(vec![i as f32, 1.0, 0.0]),
                    ..Default::default()
                },
                format!("record-{i}"),
                nonce.to_owned(),
            )
        })
        .collect();
    rt.block_on(async {
        tokio::time::timeout(Duration::from_secs(60), store.add(&records)).await
    })??;
    emit(serde_json::json!({"kind":"context_ready","uri":uri,"records":8}));
    command("merge")?;
    let merged = rt.block_on(async {
        tokio::time::timeout(Duration::from_secs(60), store.cleanup_wal()).await
    })??;
    if merged == 0 {
        return Err("fixture did not merge an actual WAL generation".into());
    }
    emit(serde_json::json!({"kind":"context_merged","generations":merged}));
    // Keep the runtime and collector alive while the independent HTTP readers
    // finish and the orchestrator verifies real backend retirement.
    command("close")?;
    rt.block_on(async { tokio::time::timeout(Duration::from_secs(30), store.close()).await })??;
    emit(serde_json::json!({"kind":"context_closed"}));
    Ok(())
}
