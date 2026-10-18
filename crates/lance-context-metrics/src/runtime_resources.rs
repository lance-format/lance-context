//! Fresh cgroup-v2 observations for recovery admission; never a writer watchdog.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{routing::get, Json, Router};
use serde_json::json;

const MAX_FILE_BYTES: u64 = 8192;

#[derive(Debug, PartialEq)]
struct Snapshot {
    current: u64,
    limit: Option<u64>,
    oom: u64,
    oom_kill: u64,
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid cgroup memory observation",
    )
}

fn bounded(root: &Path, name: &str) -> io::Result<String> {
    let mut text = String::new();
    File::open(root.join(name))?
        .take(MAX_FILE_BYTES + 1)
        .read_to_string(&mut text)?;
    if text.len() as u64 > MAX_FILE_BYTES {
        return Err(invalid());
    }
    Ok(text)
}

fn snapshot(root: &Path) -> io::Result<Snapshot> {
    let current = bounded(root, "memory.current")?
        .trim()
        .parse::<u64>()
        .map_err(|_| invalid())?;
    let maximum = bounded(root, "memory.max")?;
    let limit = match maximum.trim() {
        "max" => None,
        value => {
            let limit = value.parse::<u64>().map_err(|_| invalid())?;
            if limit == 0 {
                return Err(invalid());
            }
            Some(limit)
        }
    };
    let events = bounded(root, "memory.events")?;
    let mut counters = BTreeMap::new();
    for line in events.lines() {
        let mut fields = line.split_whitespace();
        let name = fields.next().ok_or_else(invalid)?;
        let count = fields
            .next()
            .ok_or_else(invalid)?
            .parse::<u64>()
            .map_err(|_| invalid())?;
        if fields.next().is_some() || counters.insert(name, count).is_some() {
            return Err(invalid());
        }
    }
    Ok(Snapshot {
        current,
        limit,
        oom: *counters.get("oom").ok_or_else(invalid)?,
        oom_kill: *counters.get("oom_kill").ok_or_else(invalid)?,
    })
}

pub(super) fn router() -> Router {
    Router::new()
        .route("/api/v1/internal/runtime-resources", get(observe))
        .with_state(PathBuf::from("/sys/fs/cgroup"))
}

async fn observe(
    State(root): State<PathBuf>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let nonce = query.get("nonce").map(String::as_str).unwrap_or("");
    if nonce.len() > 64
        || !nonce
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return (
            StatusCode::BAD_REQUEST,
            [(header::CACHE_CONTROL, "no-store")],
            Json(json!({"error": "invalid observation nonce"})),
        )
            .into_response();
    }
    // These three small kernel pseudo-files do not access a table, acquire a
    // store lock, scrape peers, or run an external command. No prior sample is
    // cached: missing/unsupported cgroups cannot become a healthy zero sample.
    let value = match snapshot(&root) {
        Ok(value) => value,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                [(header::CACHE_CONTROL, "no-store")],
                Json(json!({"error": "cgroup v2 memory observation unavailable", "nonce": nonce})),
            )
                .into_response();
        }
    };
    let sampled_at = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(time) => time.as_millis(),
        Err(_) => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };
    (
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({
            "schema": 1,
            "nonce": nonce,
            "sampled_at_unix_millis": sampled_at,
            "cgroup_version": 2,
            "memory_current_bytes": value.current,
            "memory_max_bytes": value.limit,
            "oom": value.oom,
            "oom_kill": value.oom_kill,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "lance-runtime-resources-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&path).unwrap();
            fs::write(path.join("memory.current"), "100\n").unwrap();
            fs::write(path.join("memory.max"), "1000\n").unwrap();
            fs::write(
                path.join("memory.events"),
                "low 0\nhigh 0\nmax 0\noom 2\noom_kill 1\n",
            )
            .unwrap();
            Self(path)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn preserves_oom_evidence_and_unlimited_is_not_a_finite_budget() {
        let fixture = Fixture::new();
        assert_eq!(
            snapshot(&fixture.0).unwrap(),
            Snapshot {
                current: 100,
                limit: Some(1000),
                oom: 2,
                oom_kill: 1
            }
        );
        fs::write(fixture.0.join("memory.max"), "max\n").unwrap();
        assert_eq!(snapshot(&fixture.0).unwrap().limit, None);
    }

    #[test]
    fn partial_duplicate_or_oversized_observations_fail_closed() {
        let fixture = Fixture::new();
        for events in [
            "oom 0\n",
            "oom 0\noom 1\noom_kill 0\n",
            "oom invalid\noom_kill 0\n",
        ] {
            fs::write(fixture.0.join("memory.events"), events).unwrap();
            assert!(snapshot(&fixture.0).is_err());
        }
        fs::write(fixture.0.join("memory.current"), "1".repeat(8193)).unwrap();
        assert!(snapshot(&fixture.0).is_err());
    }

    #[tokio::test]
    async fn each_request_reads_fresh_values_and_failure_cannot_serve_old_success() {
        let fixture = Fixture::new();
        let query = || Query(HashMap::from([("nonce".to_owned(), "read-123".to_owned())]));
        for current in [100, 900] {
            fs::write(fixture.0.join("memory.current"), current.to_string()).unwrap();
            let response = observe(State(fixture.0.clone()), query()).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["nonce"], "read-123");
            assert_eq!(value["memory_current_bytes"], current);
            assert_eq!(value["oom_kill"], 1);
        }
        fs::remove_file(fixture.0.join("memory.events")).unwrap();
        assert_eq!(
            observe(State(fixture.0.clone()), query()).await.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
