//! Process-local WAL evidence from the existing scanner. HTTP reads never open
//! stores or trigger scans. This is separate from atomic committed stats and is
//! not a lease, custody record, or proof that this process still owns the scan.
use std::collections::{BTreeMap, HashSet};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use axum::{body::Body, extract::State, http::StatusCode, response::Response};
use serde::Serialize;
use tokio::sync::Semaphore;

use crate::scanner::{ScanEntry, ScanKind};
use crate::state::MasterState;
use crate::stats_store::StatRow;

const MAX_TABLES: usize = 40_000;
const MAX_IDENTITY_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
const SCOPE: &str = "latest_shard_manifests_only";

#[derive(Clone, Serialize)]
struct Row {
    name: String,
    at: Option<f64>,
    pending: Option<i64>,
    scope: &'static str,
    error: Option<&'static str>,
    failed_at_ms: Option<i64>,
}

impl Row {
    fn unknown(name: String, error: &'static str, failed_at_ms: Option<i64>) -> Self {
        Self {
            name,
            at: None,
            pending: None,
            scope: SCOPE,
            error: Some(error),
            failed_at_ms,
        }
    }
}

struct Entry {
    identity: Option<(String, ScanKind)>,
    row: Arc<Row>,
    visited: bool,
}

#[derive(Clone, Serialize)]
struct Metadata {
    format: &'static str,
    scan_id: String,
    inventory_at_ms: i64,
    inventory_complete: bool,
    table_count: usize,
    attempted_tables: usize,
    scan_state: &'static str,
    finished_at_ms: Option<i64>,
}

struct View {
    metadata: Metadata,
    entries: BTreeMap<String, Entry>,
}

#[derive(Default)]
struct Inner {
    round: u64,
    revision: u64,
    view: Option<View>,
}

pub(crate) struct ScanObservations {
    inner: Mutex<Inner>,
    exports: Arc<Semaphore>,
}

impl Default for ScanObservations {
    fn default() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            exports: Arc::new(Semaphore::new(2)),
        }
    }
}

/// Dropping an interrupted/failed scan cannot leave an apparently active round.
/// Each update is fenced to this local generation, not an external ownership claim.
pub(crate) struct Round<'a> {
    cache: &'a ScanObservations,
    generation: u64,
}

impl ScanObservations {
    pub(crate) fn begin(
        &self,
        entries: &[ScanEntry],
        ambiguous: &HashSet<String>,
        at: i64,
    ) -> Round<'_> {
        let mut inner = self.inner.lock().unwrap();
        inner.round += 1;
        inner.revision += 1;
        let generation = inner.round;
        // Invalidate the previous inventory on overflow. Never truncate and
        // advertise the retained subset as a complete inventory.
        let valid_name =
            |name: &str| !name.is_empty() && name.len() <= 2048 && name.chars().count() <= 512;
        let bytes = entries
            .iter()
            .map(|e| e.name.len() + e.uri.len())
            .chain(ambiguous.iter().map(String::len))
            .sum::<usize>();
        if entries.len() + ambiguous.len() > MAX_TABLES
            || bytes > MAX_IDENTITY_BYTES
            || entries
                .iter()
                .any(|e| !valid_name(&e.name) || e.uri.len() > 8192)
            || ambiguous.iter().any(|name| !valid_name(name))
        {
            inner.view = None;
            return Round {
                cache: self,
                generation,
            };
        }
        let mut old = inner.view.take();
        let mut rows = BTreeMap::new();
        for entry in entries {
            let identity = Some((entry.uri.clone(), entry.kind));
            let prior = old.as_mut().and_then(|v| v.entries.remove(&entry.name));
            let row = match prior {
                Some(prior) if prior.identity == identity => prior.row,
                _ => Arc::new(Row::unknown(entry.name.clone(), "not_observed", None)),
            };
            if rows
                .insert(
                    entry.name.clone(),
                    Entry {
                        identity,
                        row,
                        visited: false,
                    },
                )
                .is_some()
            {
                inner.view = None;
                return Round {
                    cache: self,
                    generation,
                };
            }
        }
        for name in ambiguous {
            if rows
                .insert(
                    name.clone(),
                    Entry {
                        identity: None,
                        row: Arc::new(Row::unknown(name.clone(), "ambiguous_identity", Some(at))),
                        visited: false,
                    },
                )
                .is_some()
            {
                inner.view = None;
                return Round {
                    cache: self,
                    generation,
                };
            }
        }
        inner.view = Some(View {
            metadata: Metadata {
                format: "rocketkeep-scan-observations-v1",
                scan_id: lance_context_core::generate_id(),
                inventory_at_ms: at,
                inventory_complete: true,
                table_count: rows.len(),
                attempted_tables: 0,
                scan_state: "observing",
                finished_at_ms: None,
            },
            entries: rows,
        });
        Round {
            cache: self,
            generation,
        }
    }

    fn snapshot(&self) -> Result<(u64, Metadata, Vec<Arc<Row>>), StatusCode> {
        let inner = self.inner.lock().unwrap();
        let view = inner.view.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
        // Clone small immutable handles under the lock; serialize outside it.
        Ok((
            inner.revision,
            view.metadata.clone(),
            view.entries.values().map(|e| e.row.clone()).collect(),
        ))
    }

    async fn response(&self, executor_id: String) -> Result<Response, StatusCode> {
        let permit = self
            .exports
            .clone()
            .try_acquire_owned()
            .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;
        let (revision, metadata, rows) = self.snapshot()?;
        let bytes = tokio::task::spawn_blocking(move || {
            #[derive(Serialize)]
            struct Export<'a> {
                executor_id: String,
                revision: u64,
                #[serde(flatten)]
                metadata: Metadata,
                rows: Vec<&'a Row>,
            }
            let body = Export {
                executor_id,
                revision,
                metadata,
                rows: rows.iter().map(AsRef::as_ref).collect(),
            };
            let mut writer = BoundedWriter {
                bytes: Vec::new(),
                limit: MAX_RESPONSE_BYTES,
            };
            serde_json::to_writer(&mut writer, &body)
                .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
            // Keep the permit in the blocking task even if the request is cancelled.
            Ok::<_, StatusCode>((bytes::Bytes::from(writer.bytes), permit))
        })
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;
        // Retain admission through body delivery, not merely serialization.
        let body = Body::from_stream(futures::stream::unfold(
            bytes,
            |(mut bytes, permit)| async move {
                if bytes.is_empty() {
                    return None;
                }
                let chunk = bytes.split_to(bytes.len().min(64 * 1024));
                Some((Ok::<_, io::Error>(chunk), (bytes, permit)))
            },
        ));
        Response::builder()
            .header("content-type", "application/json")
            .header("cache-control", "no-store")
            .body(body)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }
}

impl Round<'_> {
    pub(crate) fn observe(&self, entry: &ScanEntry, row: Option<&StatRow>) {
        let mut inner = self.cache.inner.lock().unwrap();
        if inner.round != self.generation {
            return;
        }
        let Some(view) = inner.view.as_mut() else {
            return;
        };
        let Some(target) = view.entries.get_mut(&entry.name) else {
            return;
        };
        if target
            .identity
            .as_ref()
            .map(|(uri, kind)| (uri.as_str(), *kind))
            != Some((entry.uri.as_str(), entry.kind))
        {
            return;
        }
        if !target.visited {
            view.metadata.attempted_tables += 1;
            target.visited = true;
        }
        target.row = Arc::new(match row {
            Some(row)
                if row.name == entry.name
                    && row.uri == entry.uri
                    && row.pending_wal_generations >= 0
                    && row.scanned_at > 0 =>
            {
                Row {
                    name: entry.name.clone(),
                    at: Some(row.scanned_at as f64 / 1000.0),
                    pending: Some(row.pending_wal_generations),
                    scope: SCOPE,
                    error: None,
                    failed_at_ms: None,
                }
            }
            _ => Row::unknown(
                entry.name.clone(),
                "observation_failed",
                Some(chrono::Utc::now().timestamp_millis()),
            ),
        });
        inner.revision += 1;
    }

    pub(crate) fn finish(&self) {
        self.end("observed");
    }

    fn end(&self, status: &'static str) {
        let mut inner = self.cache.inner.lock().unwrap();
        if inner.round != self.generation {
            return;
        }
        let Some(view) = inner.view.as_mut() else {
            return;
        };
        if view.metadata.finished_at_ms.is_some() {
            return;
        }
        view.metadata.scan_state = status;
        view.metadata.finished_at_ms = Some(chrono::Utc::now().timestamp_millis());
        inner.revision += 1;
    }
}

impl Drop for Round<'_> {
    fn drop(&mut self) {
        self.end("interrupted");
    }
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if data.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("observation export exceeds byte limit"));
        }
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) async fn handler(State(state): State<Arc<MasterState>>) -> Result<Response, StatusCode> {
    state
        .scan_observations
        .response(state.admission.status().executor_id)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, uri: &str) -> ScanEntry {
        ScanEntry {
            name: name.into(),
            uri: uri.into(),
            kind: ScanKind::Rollout,
        }
    }
    fn row(entry: &ScanEntry, pending: i64, at: i64) -> StatRow {
        StatRow {
            name: entry.name.clone(),
            uri: entry.uri.clone(),
            row_count: 0,
            fragment_count: 0,
            last_updated: 0,
            pending_wal_generations: pending,
            last_compaction: -1,
            total_compactions: 0,
            scanned_at: at,
            version: 1,
        }
    }
    async fn read(cache: &ScanObservations) -> serde_json::Value {
        let response = cache.response("executor-1".into()).await.unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), MAX_RESPONSE_BYTES)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn preserves_original_time_and_immutable_response_across_rounds() {
        let cache = ScanObservations::default();
        let a = entry("a", "a.lance");
        let b = entry("b", "b.lance");
        let round = cache.begin(&[a.clone(), b.clone()], &HashSet::new(), 1000);
        round.observe(&a, Some(&row(&a, 7, 1234)));
        let captured = cache.response("executor-1".into()).await.unwrap();
        round.observe(&b, Some(&row(&b, 0, 2345)));
        round.finish();
        let first: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(captured.into_body(), MAX_RESPONSE_BYTES)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(first["rows"][0]["pending"], 7);
        assert_eq!(first["rows"][0]["at"], 1.234);
        assert_eq!(first["rows"][1]["error"], "not_observed");
        assert_eq!(first["attempted_tables"], 1);
        let next = cache.begin(&[a.clone(), b], &HashSet::new(), 900000);
        let current = read(&cache).await;
        assert_eq!(current["rows"][0]["at"], 1.234);
        assert_eq!(current["attempted_tables"], 0);
        assert_ne!(current["scan_id"], first["scan_id"]);
        // A superseded round cannot update or finish the new generation.
        round.observe(&a, Some(&row(&a, 99, 900001)));
        drop(round);
        assert_eq!(read(&cache).await["scan_state"], "observing");
        next.observe(&a, None);
        let failed = read(&cache).await;
        assert!(failed["rows"][0]["at"].is_null());
        assert!(failed["rows"][0]["pending"].is_null());
        assert!(failed["rows"][0]["failed_at_ms"].is_i64());
        drop(next);
        assert_eq!(read(&cache).await["scan_state"], "interrupted");
    }

    #[tokio::test]
    async fn replacement_removal_ambiguity_and_bad_counts_remain_unknown() {
        let cache = ScanObservations::default();
        let a = entry("a", "a.lance");
        let b = entry("b", "b.lance");
        let old = cache.begin(&[a.clone(), b.clone()], &HashSet::new(), 1);
        old.observe(&a, Some(&row(&a, 0, 100)));
        old.observe(&b, Some(&row(&b, 0, 100)));
        let replaced = entry("a", "different.lance");
        let new = cache.begin(&[replaced.clone()], &HashSet::from(["conflict".into()]), 2);
        let value = read(&cache).await;
        assert_eq!(value["table_count"], 2);
        assert_eq!(value["rows"][0]["error"], "not_observed");
        assert_eq!(value["rows"][1]["error"], "ambiguous_identity");
        new.observe(&a, Some(&row(&a, 0, 101))); // Wrong URI cannot revive old facts.
        assert_eq!(read(&cache).await["attempted_tables"], 0);
        new.observe(&replaced, Some(&row(&replaced, -1, 200)));
        assert!(read(&cache).await["rows"][0]["pending"].is_null());
        let mut changed_kind = replaced.clone();
        new.observe(&replaced, Some(&row(&replaced, 3, 300)));
        changed_kind.kind = ScanKind::Generic;
        let _third = cache.begin(&[changed_kind], &HashSet::new(), 3);
        assert_eq!(read(&cache).await["rows"][0]["error"], "not_observed");
    }

    #[tokio::test]
    async fn bounded_inventory_fails_closed_and_exports_hold_capacity_until_dropped() {
        let cache = ScanObservations::default();
        assert_eq!(
            cache.response("e".into()).await.unwrap_err(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        let a = entry("a", "a.lance");
        let _round = cache.begin(&[a.clone()], &HashSet::new(), 1);
        let first = cache.response("e".into()).await.unwrap();
        let second = cache.response("e".into()).await.unwrap();
        assert_eq!(
            cache.response("e".into()).await.unwrap_err(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(first);
        assert!(cache.response("e".into()).await.is_ok());
        drop(second);
        let _bad = cache.begin(&[a.clone(), a], &HashSet::new(), 2);
        assert!(cache.snapshot().is_err());
        let _long = cache.begin(&[entry(&"x".repeat(513), "a")], &HashSet::new(), 3);
        assert!(cache.snapshot().is_err());
        let many: Vec<_> = (0..=MAX_TABLES)
            .map(|i| entry(&i.to_string(), "a"))
            .collect();
        let _overflow = cache.begin(&many, &HashSet::new(), 4);
        assert!(cache.snapshot().is_err());
        let large: Vec<_> = (0..3000)
            .map(|i| entry(&i.to_string(), &"u".repeat(8192)))
            .collect();
        let _bytes = cache.begin(&large, &HashSet::new(), 5);
        assert!(cache.snapshot().is_err());
        let mut writer = BoundedWriter {
            bytes: Vec::new(),
            limit: 3,
        };
        assert!(writer.write_all(b"abc").is_ok());
        assert!(writer.write_all(b"d").is_err());
        assert_eq!(writer.bytes, b"abc");
    }

    #[tokio::test]
    async fn escaped_response_overflow_is_unavailable_not_truncated_success() {
        let cache = ScanObservations::default();
        // Small raw names can expand sixfold during JSON escaping. Exercise
        // the HTTP serialization limit, not only the raw inventory bound.
        let entries: Vec<_> = (0..11_000)
            .map(|i| entry(&format!("{:05}{}", i, "\u{0001}".repeat(507)), "a"))
            .collect();
        let _round = cache.begin(&entries, &HashSet::new(), 1);
        assert!(cache.snapshot().is_ok());
        assert_eq!(
            cache.response("e".into()).await.unwrap_err(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(cache.exports.available_permits(), 2);
    }

    #[tokio::test]
    #[ignore = "requires isolated loopback ETCD_TEST_ENDPOINTS and ETCD_TEST_EXCLUSIVE=1; serial publication tests"]
    async fn http_exports_first_table_while_real_scan_is_blocked_before_second_table() {
        use clap::Parser;
        use lance_context_api::{ColumnSpec, ColumnType, SchemaSpec, ID_COLUMN};
        use lance_context_core::{GenericStore, GenericStoreOptions};
        use std::time::Duration;
        use tower::ServiceExt;

        assert_eq!(std::env::var("ETCD_TEST_EXCLUSIVE").unwrap(), "1");
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let url = reqwest::Url::parse(&endpoint).unwrap();
        assert_eq!(url.host_str(), Some("127.0.0.1"));
        assert!(url.port().is_some());
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::config::MasterConfig::parse_from(["test"]);
        config.data_dir = dir.path().to_string_lossy().into_owned();
        config.etcd.etcd_endpoints = vec![endpoint];
        config.etcd.etcd_prefix = format!(
            "/incremental-scan-test/{}",
            lance_context_core::generate_id()
        );
        config.scan_concurrency = 1;
        config.merge_wal_interval_secs = 0;
        config.stats_scan_interval_secs = 0;
        config.stats_cold_retire_secs = 0;
        config.worker_endpoints.clear();
        let state = MasterState::new(config).await.unwrap();
        for name in ["first", "second"] {
            let uri = dir
                .path()
                .join(format!("{name}.generic.lance"))
                .to_string_lossy()
                .into_owned();
            let writer = GenericStore::open(
                &uri,
                SchemaSpec::new(vec![(
                    ID_COLUMN.into(),
                    ColumnSpec::required(ColumnType::String { large: false }),
                )]),
                GenericStoreOptions {
                    seal_on_add: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            writer
                .add(&[serde_json::json!({"id":"a"}).as_object().unwrap().clone()])
                .await
                .unwrap();
            state.generic_registry.upsert(name, &uri).await.unwrap();
        }
        let router = crate::routes::api_router().with_state(state.clone());
        async fn get(router: axum::Router) -> serde_json::Value {
            let response = router
                .oneshot(
                    axum::http::Request::builder()
                        .uri("/scan-observations")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), MAX_RESPONSE_BYTES)
                    .await
                    .unwrap(),
            )
            .unwrap()
        }
        let busy = crate::demand_publish::hold_scan_slots_for_test().await;
        let scan_state = state.clone();
        let scan = tokio::spawn(async move { crate::scanner::scan_once(&scan_state).await });
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if state
                    .scan_observations
                    .snapshot()
                    .is_ok_and(|(_, meta, _)| meta.attempted_tables == 1)
                {
                    break;
                }
                assert!(!scan.is_finished());
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let first = get(router.clone()).await;
        assert_eq!(first["executor_id"], state.admission.status().executor_id);
        assert_eq!(first["table_count"], 2);
        assert_eq!(first["rows"][0]["name"], "generic:first");
        assert_eq!(first["rows"][0]["pending"], 1);
        assert!(first["rows"][0]["at"].is_number());
        assert_eq!(first["rows"][1]["error"], "not_observed");
        assert!(state.stats_cache.read().await.is_empty());
        assert!(!scan.is_finished());
        drop(busy);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(30), scan)
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
            2
        );
        let done = get(router).await;
        assert_eq!(done["scan_id"], first["scan_id"]);
        assert_eq!(done["rows"][0]["at"], first["rows"][0]["at"]);
        assert_eq!(done["rows"][1]["pending"], 1);
        assert_eq!(done["scan_state"], "observed");
        assert_eq!(state.stats_cache.read().await.len(), 2);
        // Failed later observation must replace success with unknown, not zero.
        std::fs::rename(
            dir.path().join("first.generic.lance"),
            dir.path().join("removed.generic.lance"),
        )
        .unwrap();
        crate::scanner::scan_once(&state).await.unwrap();
        let after = get(crate::routes::api_router().with_state(state.clone())).await;
        assert_eq!(after["rows"][0]["error"], "observation_failed");
        assert!(after["rows"][0]["pending"].is_null());
        assert_eq!(first["rows"][0]["pending"], 1);
        // Atomic stats retain the previous success on a failed read as before.
        assert_eq!(
            state
                .stats_cache
                .read()
                .await
                .iter()
                .find(|r| r.name == "generic:first")
                .unwrap()
                .pending_wal_generations,
            1
        );
    }
}
