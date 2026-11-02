//! Cross-process Context HTTP qualification driver. Never accepts production URIs.
#![recursion_limit = "256"]

use lance_context_api::AddRecordRequest;
use lance_context_core::{
    discover_retired_wal_at_uri, discover_retired_wal_under_root, record_from_add_request,
    ContextStore, ContextStoreOptions, MergeMemoryBudget,
};
use std::io::{self, Write};
use std::time::Duration;

const PREFIX: &str = "abfss://snowflake-rollouts@falconphxdataprocessing.dfs.core.windows.net/rocketkeep-staging/wal-lifetime-47748/";

#[derive(Debug, PartialEq, Eq)]
enum FixtureScope<'a> {
    Table(&'a str),
    Root,
}

fn fixture_nonce<'a>(name: &'a str, prefix: &str, suffix: &str) -> Result<&'a str, &'static str> {
    let nonce = name
        .strip_prefix(prefix)
        .and_then(|s| s.strip_suffix(suffix))
        .ok_or("refusing non-fixture location")?;
    if nonce.len() != 32 || !nonce.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("fixture name must contain a fresh UUID nonce");
    }
    Ok(nonce)
}

fn fixture_scope<'a>(uri: &'a str, action: Option<&str>) -> Result<FixtureScope<'a>, &'static str> {
    let name = uri.strip_prefix(PREFIX).ok_or("refusing non-staging URI")?;
    match action {
        Some("recover-root") => {
            // Namespace collection is confined to a fresh fixture subdirectory,
            // never the shared staging parent or a caller-selected production root.
            fixture_nonce(name, "stage_root_", "")?;
            Ok(FixtureScope::Root)
        }
        None | Some("recover") => {
            let table = if let Some((root, table)) = name.split_once('/') {
                fixture_nonce(root, "stage_root_", "")?;
                table
            } else {
                name
            };
            Ok(FixtureScope::Table(fixture_nonce(
                table,
                "stage_context_",
                ".lance",
            )?))
        }
        _ => Err("unknown staging action"),
    }
}

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
    let action = std::env::args().nth(2);
    if std::env::args().nth(3).is_some() {
        return Err("unknown staging action".into());
    }
    let scope = fixture_scope(&uri, action.as_deref())?;
    if !matches!(
        std::env::var("ROCKETKEEP_WAL_LIFECYCLE").as_deref(),
        Ok("retain" | "collect")
    ) {
        return Err("explicit staging WAL lifecycle mode required".into());
    }
    if std::env::var("ETCD_ENDPOINTS").is_ok_and(|s| !s.is_empty()) {
        return Err("staging driver must not inherit etcd endpoints".into());
    }
    if action.is_some() && std::env::var("ROCKETKEEP_WAL_LIFECYCLE").as_deref() != Ok("collect") {
        return Err("cold recovery requires explicit collect mode".into());
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    if scope == FixtureScope::Root {
        rt.block_on(async {
            tokio::time::timeout(
                Duration::from_secs(120),
                discover_retired_wal_under_root(&uri),
            )
            .await
        })??;
        emit(serde_json::json!({"kind":"retirement_root_recovery_admitted","uri":uri}));
        command("close")?;
        emit(serde_json::json!({"kind":"retirement_root_recovery_closed"}));
        return Ok(());
    }
    if action.as_deref() == Some("recover") {
        // A fresh process receives only the canonical fixture URI. Do not open
        // a writer, read table payloads, or reuse the writer's remembered work.
        rt.block_on(async {
            tokio::time::timeout(Duration::from_secs(60), discover_retired_wal_at_uri(&uri)).await
        })??;
        emit(serde_json::json!({"kind":"retirement_recovery_admitted","uri":uri}));
        command("close")?;
        emit(serde_json::json!({"kind":"retirement_recovery_closed"}));
        return Ok(());
    }
    let FixtureScope::Table(nonce) = scope else {
        unreachable!("root recovery returned above");
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    const NONCE: &str = "0123456789abcdef0123456789abcdef";

    #[test]
    fn fixture_scope_allows_table_and_isolated_namespace() {
        for uri in [
            format!("{PREFIX}stage_context_{NONCE}.lance"),
            format!("{PREFIX}stage_root_{NONCE}/stage_context_{NONCE}.lance"),
        ] {
            for action in [None, Some("recover")] {
                assert_eq!(fixture_scope(&uri, action), Ok(FixtureScope::Table(NONCE)));
            }
        }
        assert_eq!(
            fixture_scope(&format!("{PREFIX}stage_root_{NONCE}"), Some("recover-root")),
            Ok(FixtureScope::Root)
        );
    }

    #[test]
    fn fixture_scope_rejects_escape_and_action_mismatch() {
        let root = format!("{PREFIX}stage_root_{NONCE}");
        let table = format!("{root}/stage_context_{NONCE}.lance");
        for (uri, action) in [
            (PREFIX.to_owned(), Some("recover-root")),
            (
                "abfss://production/table.lance".to_owned(),
                Some("recover-root"),
            ),
            (format!("{root}/.."), Some("recover-root")),
            (format!("{root}%2f.."), Some("recover-root")),
            (format!("{root}?suffix=other"), Some("recover-root")),
            (format!("{root}/"), Some("recover-root")),
            (format!("{PREFIX}stage_root_short"), Some("recover-root")),
            (root.clone(), None),
            (root, Some("recover")),
            (table.clone(), Some("recover-root")),
            (table.clone(), Some("unknown")),
            (format!("{table}/extra"), None),
            (format!("{PREFIX}../stage_context_{NONCE}.lance"), None),
        ] {
            assert!(fixture_scope(&uri, action).is_err(), "{uri} {action:?}");
        }
    }
}
