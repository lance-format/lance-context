//! Bounded request hints for recovery, independent of Kubernetes ingress logs.
//!
//! These are lossy observation hints, never merge ownership or write evidence.
//! Request handlers never wait for this cache; loss and incomplete snapshots are
//! explicit. No request body, query string, record ID, or storage payload is kept.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::Query;
use axum::http::{header, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{routing::get, Json, Router};
use serde_json::json;

const SHARDS: usize = 32;
const PER_SHARD: usize = 128;
const WINDOW: Duration = Duration::from_secs(300);
const MAX_REPLY: usize = 1024 * 1024;
const MAX_TARGET: usize = 1032;
// Observation threshold, not a deadline: never cancel or retry a request here.
const SLOW_READ: Duration = Duration::from_secs(5);

#[derive(Clone)]
struct Entry {
    touched: Instant,
    active_at: u64,
    read_error: Option<(Instant, u64, u16)>,
    read_slow: Option<(Instant, u64, u16, u64)>,
}

struct Signals {
    shards: [Mutex<HashMap<String, Entry>>; SHARDS],
    snapshot_gate: Mutex<()>,
    dropped: AtomicU64,
    last_drop_tick: AtomicU64,
    started: Instant,
    started_at: u64,
    enabled: bool,
}

fn wall_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

impl Signals {
    fn new(enabled: bool) -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(HashMap::new())),
            snapshot_gate: Mutex::new(()),
            dropped: AtomicU64::new(0),
            last_drop_tick: AtomicU64::new(0),
            started: Instant::now(),
            started_at: wall_millis(),
            enabled,
        }
    }

    fn shard(target: &str) -> usize {
        let mut hash = DefaultHasher::new();
        target.hash(&mut hash);
        hash.finish() as usize % SHARDS
    }

    fn tick(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.started)
            .as_millis()
            .min((u64::MAX - 1) as u128) as u64
            + 1
    }

    fn drop_event(&self, now: Instant) {
        self.last_drop_tick
            .fetch_max(self.tick(now), Ordering::Relaxed);
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }

    fn record(&self, target: String, read: bool, status: StatusCode, now: Instant, wall: u64) {
        self.record_with_latency(target, read, status, now, wall, None);
    }

    fn record_with_latency(
        &self,
        target: String,
        read: bool,
        status: StatusCode,
        now: Instant,
        wall: u64,
        elapsed: Option<Duration>,
    ) {
        let slow = elapsed.filter(|elapsed| read && *elapsed >= SLOW_READ);
        if !self.enabled
            || !(status.is_success()
                || status.is_redirection()
                || status.is_server_error()
                || slow.is_some())
        {
            return;
        }
        if target.len() > MAX_TARGET {
            self.drop_event(now);
            return;
        }
        let Ok(mut values) = self.shards[Self::shard(&target)].try_lock() else {
            self.drop_event(now);
            return;
        };
        if !values.contains_key(&target) && values.len() >= PER_SHARD {
            values.retain(|_, entry| now.saturating_duration_since(entry.touched) < WINDOW);
            if values.len() >= PER_SHARD {
                self.drop_event(now);
                return;
            }
        }
        let entry = values.entry(target).or_insert(Entry {
            touched: now,
            active_at: wall,
            read_error: None,
            read_slow: None,
        });
        entry.touched = now;
        entry.active_at = wall;
        if read && status.is_server_error() {
            entry.read_error = Some((now, wall, status.as_u16()));
        }
        if let Some(elapsed) = slow {
            entry.read_slow = Some((
                now,
                wall,
                status.as_u16(),
                elapsed.as_millis().min(u64::MAX as u128) as u64,
            ));
        }
    }

    fn snapshot(&self, nonce: &str, now: Instant) -> Response {
        let Ok(_guard) = self.snapshot_gate.try_lock() else {
            return unavailable("snapshot busy");
        };
        if !self.enabled {
            return unavailable("request signals disabled");
        }
        let dropped_before = self.dropped.load(Ordering::Relaxed);
        let mut entries = Vec::new();
        let mut complete = true;
        for shard in &self.shards {
            let Ok(values) = shard.try_lock() else {
                complete = false;
                continue;
            };
            for (name, entry) in values.iter() {
                if now.saturating_duration_since(entry.touched) < WINDOW {
                    entries.push((name.clone(), entry.clone()));
                }
            }
        }
        // Preserve recent failures, then slow reads, before ordinary activity.
        entries.sort_by_key(|(_, entry)| {
            let error = entry
                .read_error
                .filter(|(at, _, _)| now.saturating_duration_since(*at) < WINDOW);
            let slow = entry
                .read_slow
                .filter(|(at, _, _, _)| now.saturating_duration_since(*at) < WINDOW);
            (
                std::cmp::Reverse(error.map(|(_, at, _)| at)),
                std::cmp::Reverse(slow.map(|(_, at, _, _)| at)),
                std::cmp::Reverse(entry.active_at),
            )
        });
        let mut tables = Vec::new();
        let mut bytes = 1024; // Reserved envelope space, including a maximum nonce.
        for (name, entry) in entries {
            let error = entry
                .read_error
                .filter(|(at, _, _)| now.saturating_duration_since(*at) < WINDOW);
            let slow = entry
                .read_slow
                .filter(|(at, _, _, _)| now.saturating_duration_since(*at) < WINDOW);
            let row = json!({"name": name, "active_at_unix_millis": entry.active_at,
                "read_error_at_unix_millis": error.map(|(_, at, _)| at),
                "read_error_status": error.map(|(_, _, status)| status),
                "read_slow_at_unix_millis": slow.map(|(_, at, _, _)| at),
                "read_slow_status": slow.map(|(_, _, status, _)| status),
                "read_slow_header_millis": slow.map(|(_, _, _, millis)| millis)});
            let size = serde_json::to_vec(&row).expect("bounded scalar row").len() + 1;
            if bytes + size > MAX_REPLY {
                complete = false;
                continue;
            }
            bytes += size;
            tables.push(row);
        }
        let dropped = self.dropped.load(Ordering::Relaxed);
        // Incompleteness expires with the lost hints; a transient scrape/request
        // collision must not permanently disable recovery after traffic resumes.
        let last_drop = self.last_drop_tick.load(Ordering::Relaxed);
        let recent_loss =
            last_drop != 0 && self.tick(now).saturating_sub(last_drop) < WINDOW.as_millis() as u64;
        complete &= dropped == dropped_before && !recent_loss;
        let value = json!({"schema":1, "nonce":nonce,"sampled_at_unix_millis":wall_millis(),
            "coverage_started_at_unix_millis":self.started_at,"window_seconds":WINDOW.as_secs(),
            "slow_read_header_threshold_millis": SLOW_READ.as_millis() as u64,
            "complete":complete,"dropped_total":dropped,"tables":tables});
        let raw = serde_json::to_vec(&value).expect("bounded scalar envelope");
        if raw.len() > MAX_REPLY {
            return unavailable("snapshot exceeds budget");
        }
        (
            [
                (header::CACHE_CONTROL, "no-store"),
                (header::CONTENT_TYPE, "application/json"),
            ],
            raw,
        )
            .into_response()
    }
}

fn global() -> &'static Signals {
    static CACHE: OnceLock<Signals> = OnceLock::new();
    CACHE.get_or_init(|| {
        Signals::new(matches!(
            std::env::var("RECOVERY_SIGNALS_ENABLED").as_deref(),
            Ok("1" | "true")
        ))
    })
}

pub(super) fn target(path: &str, matched: &str, method: &Method) -> Option<(String, bool)> {
    if !global().enabled {
        return None;
    }
    let target = parse_target(path, matched, method);
    if let Some((name, _)) = &target {
        // Capture attempted activity before awaiting a handler that may stall.
        // This is a demand hint, never evidence that the operation progressed.
        global().record(
            name.clone(),
            false,
            StatusCode::OK,
            Instant::now(),
            wall_millis(),
        );
    }
    target
}

fn parse_target(path: &str, matched: &str, method: &Method) -> Option<(String, bool)> {
    if path.len() > 4096 {
        return None;
    }
    let mut parts = path.strip_prefix("/api/v1/")?.split('/');
    let family = parts.next()?;
    let suffix = matched.strip_prefix(&format!("/api/v1/{family}/{{name}}"))?;
    // Maintenance requests must not keep generating their own recovery demand.
    if !suffix.is_empty() && !suffix.starts_with('/') {
        return None;
    }
    let operation = suffix
        .strip_prefix('/')
        .unwrap_or(suffix)
        .split('/')
        .next()?;
    if !matches!(
        operation,
        "" | "records"
            | "rows"
            | "get-rows"
            | "search"
            | "retrieve"
            | "events"
            | "items"
            | "overview"
            | "root-status"
            | "roots"
            | "blobs"
    ) {
        return None;
    }
    let kind = match family {
        "rollouts" => "",
        "generic" => "generic:",
        "contexts" => "context:",
        "datagen" => "datagen:",
        _ => return None,
    };
    let segment = parts.next()?;
    // Axum decodes path parameters. Decode only this bounded segment, so an
    // escaped valid name cannot split one table into multiple recovery keys.
    let mut decoded = Vec::with_capacity(segment.len().min(1024));
    let mut chars = segment.bytes();
    while let Some(c) = chars.next() {
        let c = if c == b'%' {
            let hi = (chars.next()? as char).to_digit(16)?;
            let lo = (chars.next()? as char).to_digit(16)?;
            (hi * 16 + lo) as u8
        } else {
            c
        };
        if !c.is_ascii_alphanumeric() && !b"_.-".contains(&c) || decoded.len() >= 1024 {
            return None;
        }
        decoded.push(c);
    }
    let name = std::str::from_utf8(&decoded).ok()?;
    if name.is_empty()
        || name.len() > 1024
        || !name
            .as_bytes()
            .first()
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
        || matches!(name, "_registry" | "_stats")
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
    {
        return None;
    }
    let read = method == Method::GET
        || method == Method::HEAD
        || (method == Method::POST && matches!(suffix, "/get-rows" | "/search" | "/retrieve"));
    if !matches!(
        *method,
        Method::GET | Method::HEAD | Method::POST | Method::PUT | Method::PATCH
    ) {
        return None;
    }
    Some((format!("{kind}{name}"), read))
}

pub(super) fn record(target: Option<(String, bool)>, status: StatusCode, elapsed: Duration) {
    if let Some((name, true)) = target {
        if status.is_server_error() || elapsed >= SLOW_READ {
            global().record_with_latency(
                name,
                true,
                status,
                Instant::now(),
                wall_millis(),
                Some(elapsed),
            );
        }
    }
}

fn unavailable(message: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({"error":message})),
    )
        .into_response()
}

async fn observe(Query(query): Query<HashMap<String, String>>) -> Response {
    let nonce = query.get("nonce").map(String::as_str).unwrap_or("");
    if nonce.len() > 64
        || !nonce
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return (
            StatusCode::BAD_REQUEST,
            [(header::CACHE_CONTROL, "no-store")],
            Json(json!({"error":"invalid nonce"})),
        )
            .into_response();
    }
    global().snapshot(nonce, Instant::now())
}

pub(super) fn router() -> Router {
    Router::new().route("/api/v1/internal/recovery-signals", get(observe))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use serde_json::Value;

    async fn value(signals: &Signals, now: Instant) -> Value {
        let response = signals.snapshot("test-nonce", now);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = to_bytes(response.into_body(), MAX_REPLY).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn routes_keep_only_canonical_table_names_and_classify_reads() {
        for (path, matched, method, expected, read) in [
            (
                "/api/v1/rollouts/mai3_bigclimb_t1/records/secret-id/blob",
                "/api/v1/rollouts/{name}/records/{id}/blob",
                Method::GET,
                "mai3_bigclimb_t1",
                true,
            ),
            (
                "/api/v1/contexts/c/search",
                "/api/v1/contexts/{name}/search",
                Method::POST,
                "context:c",
                true,
            ),
            (
                "/api/v1/generic/g/get-rows",
                "/api/v1/generic/{name}/get-rows",
                Method::POST,
                "generic:g",
                true,
            ),
            (
                "/api/v1/datagen/d/events",
                "/api/v1/datagen/{name}/events",
                Method::POST,
                "datagen:d",
                false,
            ),
            (
                "/api/v1/rollouts/t/records",
                "/api/v1/rollouts/{name}/records",
                Method::POST,
                "t",
                false,
            ),
            (
                "/api/v1/contexts/c/records/batch",
                "/api/v1/contexts/{name}/records/batch",
                Method::PUT,
                "context:c",
                false,
            ),
        ] {
            assert_eq!(
                parse_target(path, matched, &method),
                Some((expected.to_owned(), read))
            );
        }
        assert_eq!(
            parse_target(
                "/api/v1/rollouts/%74/records",
                "/api/v1/rollouts/{name}/records",
                &Method::GET
            ),
            Some(("t".into(), true))
        );
        for (path, matched, method) in [
            ("/api/v1/rollouts/t/records", "unknown", Method::GET),
            (
                "/api/v1/rollouts/t/records",
                "/api/v1/generic/{name}/rows",
                Method::GET,
            ),
            (
                "/api/v1/internal/merge-wal/t",
                "/api/v1/internal/merge-wal/{name}",
                Method::POST,
            ),
            (
                "/api/v1/rollouts/t/compact/stats",
                "/api/v1/rollouts/{name}/compact/stats",
                Method::GET,
            ),
            (
                "/api/v1/generic/t/merge-wal",
                "/api/v1/generic/{name}/merge-wal",
                Method::POST,
            ),
            (
                "/api/v1/rollouts/t",
                "/api/v1/rollouts/{name}",
                Method::DELETE,
            ),
            (
                "/api/v1/rollouts/%2f/records",
                "/api/v1/rollouts/{name}/records",
                Method::GET,
            ),
            (
                "/api/v1/rollouts/../records",
                "/api/v1/rollouts/{name}/records",
                Method::GET,
            ),
            (
                "/api/v1/rollouts/_registry",
                "/api/v1/rollouts/{name}",
                Method::GET,
            ),
        ] {
            assert_eq!(parse_target(path, matched, &method), None, "{path}");
        }
    }

    #[tokio::test]
    async fn successful_activity_never_clears_unexpired_read_errors() {
        let signals = Signals::new(true);
        let now = Instant::now();
        signals.record("t".into(), true, StatusCode::SERVICE_UNAVAILABLE, now, 100);
        signals.record(
            "t".into(),
            false,
            StatusCode::OK,
            now + Duration::from_secs(200),
            200,
        );
        signals.record(
            "write-error".into(),
            false,
            StatusCode::SERVICE_UNAVAILABLE,
            now,
            100,
        );
        signals.record("not-found".into(), true, StatusCode::NOT_FOUND, now, 100);
        let v = value(&signals, now + Duration::from_secs(201)).await;
        assert_eq!(v["tables"][0]["name"], "t");
        assert_eq!(v["tables"][0]["read_error_status"], 503);
        assert_eq!(v["tables"][0]["active_at_unix_millis"], 200);
        assert_eq!(v["tables"][1]["read_error_status"], Value::Null);
        assert_eq!(v["tables"].as_array().unwrap().len(), 2);
        let v = value(&signals, now + WINDOW).await;
        assert_eq!(v["tables"].as_array().unwrap().len(), 1);
        assert_eq!(v["tables"][0]["read_error_status"], Value::Null);
        assert!(value(&signals, now + WINDOW * 2).await["tables"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn contention_never_waits_and_loss_remains_explicit() {
        let signals = Signals::new(true);
        let now = Instant::now();
        let response = {
            let _held = signals.shards[Signals::shard("t")].lock().unwrap();
            signals.record("t".into(), true, StatusCode::SERVICE_UNAVAILABLE, now, 1);
            signals.snapshot("n", now)
        };
        let v: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), MAX_REPLY).await.unwrap())
                .unwrap();
        assert_eq!(v["complete"], false);
        assert_eq!(v["dropped_total"], 1);
        signals.record("t".into(), true, StatusCode::OK, now, 2);
        let v = value(&signals, now).await;
        assert_eq!(v["complete"], false);
        assert_eq!(v["tables"].as_array().unwrap().len(), 1);
        assert_eq!(value(&signals, now + WINDOW * 2).await["complete"], true);
        let _gate = signals.snapshot_gate.lock().unwrap();
        assert_eq!(
            signals.snapshot("n", now).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            Signals::new(false).snapshot("n", now).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn slow_read_hints_include_404_preserve_status_and_expire_independently() {
        let signals = Signals::new(true);
        let now = Instant::now();
        for (name, read, status, elapsed) in [
            (
                "missing",
                true,
                StatusCode::NOT_FOUND,
                Duration::from_secs(241),
            ),
            ("successful", true, StatusCode::OK, SLOW_READ),
            ("failed", true, StatusCode::SERVICE_UNAVAILABLE, SLOW_READ),
            (
                "fast-miss",
                true,
                StatusCode::NOT_FOUND,
                SLOW_READ - Duration::from_millis(1),
            ),
            ("write", false, StatusCode::OK, Duration::from_secs(241)),
        ] {
            signals.record_with_latency(name.into(), read, status, now, 100, Some(elapsed));
        }
        // Continued activity and a successful read must not erase either hint.
        signals.record(
            "missing".into(),
            false,
            StatusCode::OK,
            now + Duration::from_secs(200),
            200,
        );
        signals.record_with_latency(
            "failed".into(),
            true,
            StatusCode::OK,
            now + Duration::from_secs(200),
            200,
            Some(Duration::from_millis(1)),
        );
        let v = value(&signals, now + Duration::from_secs(201)).await;
        let rows = v["tables"].as_array().unwrap();
        let missing = rows.iter().find(|r| r["name"] == "missing").unwrap();
        assert_eq!(missing["read_slow_header_millis"], 241_000);
        assert_eq!(missing["read_slow_status"], 404);
        assert_eq!(missing["read_error_status"], Value::Null);
        assert_eq!(missing["active_at_unix_millis"], 200);
        assert_eq!(missing["read_slow_at_unix_millis"], 100);
        assert!(!rows.iter().any(|r| r["name"] == "fast-miss"));
        assert_eq!(
            rows.iter().find(|r| r["name"] == "write").unwrap()["read_slow_status"],
            Value::Null
        );
        assert_eq!(
            rows.iter().find(|r| r["name"] == "successful").unwrap()["read_slow_header_millis"],
            5000
        );
        let failed = rows.iter().find(|r| r["name"] == "failed").unwrap();
        assert_eq!(failed["read_error_status"], 503);
        assert_eq!(failed["read_slow_status"], 503);
        let v = value(&signals, now + WINDOW).await;
        assert_eq!(v["tables"].as_array().unwrap().len(), 2);
        for row in v["tables"].as_array().unwrap() {
            assert_eq!(row["read_slow_at_unix_millis"], Value::Null);
            assert_eq!(row["read_slow_header_millis"], Value::Null);
            assert_eq!(row["read_error_status"], Value::Null);
        }
    }

    #[tokio::test]
    async fn slow_read_recording_keeps_the_nonblocking_loss_contract() {
        let signals = Signals::new(true);
        let now = Instant::now();
        {
            let _held = signals.shards[Signals::shard("slow")].lock().unwrap();
            signals.record_with_latency(
                "slow".into(),
                true,
                StatusCode::NOT_FOUND,
                now,
                1,
                Some(SLOW_READ),
            );
        }
        let v = value(&signals, now).await;
        assert_eq!(v["complete"], false);
        assert_eq!(v["dropped_total"], 1);
        assert!(v["tables"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn cardinality_bytes_and_ttl_are_bounded_and_errors_win_truncation() {
        let signals = Signals::new(true);
        let now = Instant::now();
        // Long valid names fill the reply before the cardinality limit.
        signals.record(
            "urgent".into(),
            true,
            StatusCode::SERVICE_UNAVAILABLE,
            now,
            1,
        );
        signals.record_with_latency(
            "slow".into(),
            true,
            StatusCode::NOT_FOUND,
            now,
            1,
            Some(SLOW_READ),
        );
        for n in 0..20_000 {
            signals.record(
                format!("{n}_{}", "x".repeat(1000)),
                false,
                StatusCode::OK,
                now,
                2,
            );
        }
        let count: usize = signals.shards.iter().map(|s| s.lock().unwrap().len()).sum();
        assert!(count <= SHARDS * PER_SHARD);
        assert!(signals
            .shards
            .iter()
            .all(|s| s.lock().unwrap().len() <= PER_SHARD));
        let v = value(&signals, now).await;
        assert_eq!(v["tables"][0]["name"], "urgent");
        assert_eq!(v["tables"][1]["name"], "slow");
        assert_eq!(v["complete"], false);
        assert!(v["dropped_total"].as_u64().unwrap() > 0);
        assert!(v["tables"].as_array().unwrap().len() < count);
        signals.record("fresh".into(), false, StatusCode::OK, now + WINDOW, 3);
        let v = value(&signals, now + WINDOW).await;
        assert_eq!(v["tables"].as_array().unwrap().len(), 1);
        assert_eq!(v["tables"][0]["name"], "fresh");
        signals.record(
            "x".repeat(MAX_TARGET + 1),
            false,
            StatusCode::OK,
            now + WINDOW,
            3,
        );
        assert_eq!(
            value(&signals, now + WINDOW).await["tables"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    #[ignore = "manual release-mode throughput measurement"]
    fn record_throughput() {
        let signals = Signals::new(true);
        let start = Instant::now();
        for n in 0..1_000_000 {
            let target = parse_target(
                "/api/v1/rollouts/mai3_bigclimb_t1/records",
                "/api/v1/rollouts/{name}/records",
                &Method::POST,
            )
            .unwrap()
            .0;
            signals.record(target, false, StatusCode::OK, Instant::now(), n);
        }
        eprintln!("1M parse+record operations: {:?}", start.elapsed());
        assert_eq!(signals.dropped.load(Ordering::Relaxed), 0);
    }
}
