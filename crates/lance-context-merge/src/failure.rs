//! Durable shard failures: task recreation must not reset a retry budget.
use crate::{ClaimProof, Coordinator, Execution, Result};
use etcd_client::{Compare, CompareOp, GetOptions, TxnOp};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Retryable,
    Deadline,
    DataOrConfiguration,
    OwnershipUnresolved,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardFailure {
    pub target: String,
    pub endpoint: String,
    pub consecutive_attempts: u32,
    pub class: FailureClass,
    pub last_error: String,
    pub last_failure_ms: u64,
    pub next_retry_ms: u64,
    pub needs_attention: bool,
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn classify(error: &str) -> FailureClass {
    let error = error.to_ascii_lowercase();
    if error.contains("ownership unresolved") {
        FailureClass::OwnershipUnresolved
    } else if is_preparation_conflict(&error) {
        FailureClass::Retryable
    } else if [
        "not found",
        "notfound",
        "corrupt",
        "schema",
        "invalid",
        "permission denied",
        "access denied",
        "accessdenied",
        "unauthorized",
        "unauthenticated",
        "forbidden",
        "protocol",
    ]
    .iter()
    .any(|s| error.contains(s))
        || has_http_status(&error, &["401", "403"])
    {
        FailureClass::DataOrConfiguration
    } else if has_http_status(&error, &["408", "429", "500", "502", "503", "504"]) {
        FailureClass::Retryable
    } else if error.contains("deadline") || error.contains("timeout") || error.contains("timed out")
    {
        FailureClass::Deadline
    } else {
        FailureClass::Retryable
    }
}

// These exact validation failures mean the prepared snapshot became stale.
// Lance wraps them in InvalidInput, but rebuilding against the latest snapshot
// can succeed. Do not exempt arbitrary errors merely containing "reprepare".
fn is_preparation_conflict(error: &str) -> bool {
    let message = error.strip_prefix("invalid user input: ").unwrap_or(error);
    let message = match message.split_once(", crates/lance-context-core/src/store_base.rs:") {
        Some((message, location)) => {
            let Some((line, column)) = location.split_once(':') else {
                return false;
            };
            if line.parse::<u32>().is_err() || column.parse::<u32>().is_err() {
                return false;
            }
            message
        }
        None => message,
    };
    matches!(
        message,
        "index dataset or schema changed; reprepare"
            | "index source fragment changed; reprepare"
            | "index metadata changed; reprepare"
            | "compaction dataset or schema changed; reprepare"
            | "compaction source fragment changed; reprepare"
    )
}

// Errors cross worker HTTP and durable task boundaries as strings. Recognize
// explicit HTTP status syntax, never digits in a request ID, path or timestamp.
fn has_http_status(error: &str, codes: &[&str]) -> bool {
    [
        "status code",
        "status_code",
        "statuscode",
        "status",
        "http/1.1",
        "http/2",
        "http",
    ]
    .iter()
    .any(|marker| {
        error.match_indices(marker).any(|(offset, matched)| {
            if error[..offset]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '/'))
            {
                return false;
            }
            let rest = &error[offset + matched.len()..];
            let value = rest.trim_start_matches(|c: char| {
                c.is_ascii_whitespace() || matches!(c, ':' | '=' | '(' | '\"' | '\'')
            });
            // Require a separator and a complete code, so HTTP401 and
            // status=401e-... cannot accidentally identify authentication.
            rest.len() != value.len()
                && codes.iter().any(|code| {
                    value.strip_prefix(code).is_some_and(|suffix| {
                        suffix.chars().next().is_none_or(|c| {
                            c.is_ascii_whitespace()
                                || matches!(c, ')' | '}' | ']' | ',' | ';' | ':' | '\"' | '\'')
                        })
                    })
                })
        })
    })
}

impl ShardFailure {
    fn retry_policy(class: &FailureClass, attempts: u32) -> (bool, u64) {
        let needs_attention = attempts >= 15
            || matches!(
                class,
                FailureClass::DataOrConfiguration | FailureClass::OwnershipUnresolved
            );
        let delay_secs = if *class == FailureClass::OwnershipUnresolved {
            (30u64.saturating_mul(1 << attempts.saturating_sub(1).min(4))).min(300)
        } else if needs_attention {
            3600
        } else if attempts < 3 {
            2
        } else {
            (30u64.saturating_mul(1 << (attempts - 3).min(5))).min(900)
        };
        (needs_attention, delay_secs)
    }

    // Read old records with corrected semantics without resetting attempts,
    // moving the failure timestamp, or mutating ownership/durable records.
    // The next fenced completion persists the normal retry result.
    fn corrected(mut self) -> Self {
        if self.class == FailureClass::DataOrConfiguration
            && is_preparation_conflict(&self.last_error.to_ascii_lowercase())
        {
            self.class = FailureClass::Retryable;
            let (attention, delay) = Self::retry_policy(&self.class, self.consecutive_attempts);
            self.needs_attention = attention;
            self.next_retry_ms = self.last_failure_ms.saturating_add(delay * 1000);
        }
        self
    }

    fn advance(
        target: &str,
        endpoint: &str,
        previous: Option<Self>,
        error: &str,
        now: u64,
    ) -> Self {
        let attempts = previous.map_or(1, |f| f.consecutive_attempts.saturating_add(1));
        let class = classify(error);
        // First task gets three attempts. Later tasks get one half-open probe,
        // never a fresh budget of three. Cap probing at once per hour.
        let (needs_attention, delay_secs) = Self::retry_policy(&class, attempts);
        Self {
            target: target.into(),
            endpoint: endpoint.into(),
            consecutive_attempts: attempts,
            class,
            last_error: error.chars().take(4096).collect(),
            last_failure_ms: now,
            next_retry_ms: now.saturating_add(delay_secs * 1000),
            needs_attention,
        }
    }
}

impl Coordinator {
    fn failures_prefix(&self) -> String {
        format!("{}/merge-failures/", self.prefix.trim_end_matches('/'))
    }
    fn failure_key(&self, target: &str, endpoint: &str) -> String {
        let hex = |s: &str| {
            s.as_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        format!(
            "{}{}/{}",
            self.failures_prefix(),
            hex(target),
            hex(endpoint)
        )
    }
    pub async fn failure(&self, target: &str, endpoint: &str) -> Result<Option<ShardFailure>> {
        let response = self
            .client
            .clone()
            .get(self.failure_key(target, endpoint), None)
            .await
            .map_err(|e| e.to_string())?;
        response
            .kvs()
            .first()
            .map(|kv| {
                serde_json::from_slice::<ShardFailure>(kv.value())
                    .map(ShardFailure::corrected)
                    .map_err(|e| e.to_string())
            })
            .transpose()
    }
    /// Commit the terminal outcome and its retry budget in the same etcd
    /// transaction that releases execution ownership. A master crash between
    /// release and bookkeeping cannot reset failed attempts.
    pub(crate) async fn completion_changes(
        &self,
        execution: &Execution,
    ) -> Result<(Vec<Compare>, Vec<TxnOp>)> {
        let key = self.failure_key(&execution.target, &execution.endpoint);
        let Some(error) = &execution.error else {
            return Ok((Vec::new(), vec![TxnOp::delete(key, None)]));
        };
        let response = self
            .client
            .clone()
            .get(key.clone(), None)
            .await
            .map_err(|e| e.to_string())?;
        let previous = response.kvs().first();
        let old = previous
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()?;
        let failure =
            ShardFailure::advance(&execution.target, &execution.endpoint, old, error, now_ms());
        Ok((
            vec![Compare::mod_revision(
                key.clone(),
                CompareOp::Equal,
                previous.map_or(0, |kv| kv.mod_revision()),
            )],
            vec![TxnOp::put(key, serde_json::to_vec(&failure).unwrap(), None)],
        ))
    }

    pub async fn record_failure(
        &self,
        proof: &ClaimProof,
        target: &str,
        endpoint: &str,
        error: &str,
    ) -> Result<ShardFailure> {
        let key = self.failure_key(target, endpoint);
        let response = self
            .client
            .clone()
            .get(key.clone(), None)
            .await
            .map_err(|e| e.to_string())?;
        let previous = response.kvs().first();
        let old = previous
            .map(|kv| serde_json::from_slice(kv.value()).map_err(|e| e.to_string()))
            .transpose()?;
        let failure = ShardFailure::advance(target, endpoint, old, error, now_ms());
        let revision = previous.map_or(0, |kv| kv.mod_revision());
        if !self
            .transact(
                vec![
                    Compare::value(proof.key.as_str(), CompareOp::Equal, proof.token.as_bytes()),
                    Compare::mod_revision(key.clone(), CompareOp::Equal, revision),
                ],
                vec![TxnOp::put(key, serde_json::to_vec(&failure).unwrap(), None)],
            )
            .await?
        {
            return Err("claim or failure record changed while recording merge failure".into());
        }
        Ok(failure)
    }
    pub async fn clear_failure(
        &self,
        proof: &ClaimProof,
        target: &str,
        endpoint: &str,
    ) -> Result<()> {
        if !self
            .transact(
                vec![Compare::value(
                    proof.key.as_str(),
                    CompareOp::Equal,
                    proof.token.as_bytes(),
                )],
                vec![TxnOp::delete(self.failure_key(target, endpoint), None)],
            )
            .await?
        {
            return Err("claim lost while recording merge recovery".into());
        }
        Ok(())
    }
    /// Bounded, cursor-based metadata scan; never opens table data or WAL.
    pub async fn failure_page(
        &self,
        after: Option<&str>,
        limit: i64,
    ) -> Result<(Vec<ShardFailure>, Option<String>)> {
        let prefix = self.failures_prefix();
        let (start, options) = match after {
            Some(key) if key.starts_with(&prefix) => {
                let mut end = prefix.as_bytes().to_vec();
                *end.last_mut().unwrap() += 1;
                (format!("{key}\0"), GetOptions::new().with_range(end))
            }
            Some(_) => return Err("invalid merge failure cursor".into()),
            None => (prefix, GetOptions::new().with_prefix()),
        };
        let response = self
            .client
            .clone()
            .get(start, Some(options.with_limit(limit.clamp(1, 256))))
            .await
            .map_err(|e| e.to_string())?;
        let rows = response
            .kvs()
            .iter()
            .map(|kv| {
                serde_json::from_slice::<ShardFailure>(kv.value())
                    .map(ShardFailure::corrected)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>>>()?;
        let next = if response.more() {
            response
                .kvs()
                .last()
                .map(|kv| String::from_utf8_lossy(kv.key()).into_owned())
        } else {
            None
        };
        Ok((rows, next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const STALE_INDEX: &str = "Invalid user input: index metadata changed; reprepare, crates/lance-context-core/src/store_base.rs:1529:24";

    #[test]
    fn optimistic_preparation_conflicts_keep_the_bounded_retry_budget() {
        for message in [
            "index metadata changed; reprepare",
            "index source fragment changed; reprepare",
            "index dataset or schema changed; reprepare",
            "compaction source fragment changed; reprepare",
            "compaction dataset or schema changed; reprepare",
        ] {
            let error = format!("Invalid user input: {message}, crates/lance-context-core/src/store_base.rs:1529:24");
            let mut previous = None;
            for attempt in 1..=16 {
                let next =
                    ShardFailure::advance("table", "master:index_id", previous, &error, 1000);
                assert_eq!(next.class, FailureClass::Retryable, "{error}");
                assert_eq!(next.consecutive_attempts, attempt);
                assert_eq!(next.needs_attention, attempt >= 15);
                if attempt < 3 {
                    assert_eq!(next.next_retry_ms, 3000);
                } else if attempt >= 15 {
                    assert_eq!(next.next_retry_ms, 3_601_000);
                }
                previous = Some(next);
            }
        }
        for error in [
            "Invalid user input: schema mismatch; reprepare",
            "Invalid user input: corrupt index metadata changed; reprepare",
            "Invalid user input: index metadata changed; reprepare: permission denied",
            "Invalid user input: index metadata changed; reprepare, crates/lance-context-core/src/store_base.rs:1529:24; HTTP 403",
            "Invalid user input: index metadata changed; reprepare, /unrelated/path.rs:1:2",
            "ownership unresolved: index metadata changed; reprepare",
        ] {
            assert_ne!(classify(error), FailureClass::Retryable, "{error}");
        }
    }

    #[test]
    fn old_conflict_records_recover_without_resetting_attempts_or_clock() {
        for attempts in [1, 4, 15, u32::MAX] {
            let old = ShardFailure {
                target: "table".into(),
                endpoint: "master:index_id".into(),
                consecutive_attempts: attempts,
                class: FailureClass::DataOrConfiguration,
                last_error: STALE_INDEX.into(),
                last_failure_ms: 1000,
                next_retry_ms: 3_601_000,
                needs_attention: true,
            };
            let corrected = old.clone().corrected();
            assert_eq!(corrected.class, FailureClass::Retryable);
            assert_eq!(corrected.consecutive_attempts, attempts);
            assert_eq!(corrected.last_failure_ms, 1000);
            assert_eq!(corrected.last_error, old.last_error);
            assert_eq!(corrected.needs_attention, attempts >= 15);
            assert_eq!(
                corrected.next_retry_ms,
                match attempts {
                    1 => 3000,
                    4 => 61_000,
                    _ => 3_601_000,
                }
            );
            let retry =
                ShardFailure::advance("table", "master:index_id", Some(old), STALE_INDEX, 5000);
            assert_eq!(retry.consecutive_attempts, attempts.saturating_add(1));
            assert_eq!(retry.class, FailureClass::Retryable);
        }
        let permanent = ShardFailure::advance("table", "worker", None, "schema mismatch", 1000);
        let before = serde_json::to_value(&permanent).unwrap();
        assert_eq!(serde_json::to_value(permanent.corrected()).unwrap(), before);
    }

    #[test]
    fn explicit_http_status_controls_backoff_not_request_id_digits() {
        let azure = include_str!("../tests/fixtures/azure-server-busy.txt");
        for error in [
            azure.to_string(),
            azure.replace(
                "bdb822a8-401e-00b9-57f1-545094000000",
                "b78c8e80-001e-005c-28f1-5401d6000000",
            ),
            azure.replace("401e", "403e"),
            "HTTP 429; request_id=403".into(),
            "HTTP 503; retry_timeout: 180s".into(),
            "temporary storage failure at /data/401/403.lance".into(),
            "temporary failure RequestId:401e-005c".into(),
            "temporary failure status=401e-005c".into(),
        ] {
            let failure = ShardFailure::advance("table", "worker", None, &error, 1000);
            assert_eq!(failure.class, FailureClass::Retryable, "{error}");
            assert!(!failure.needs_attention, "{error}");
            assert_eq!(failure.next_retry_ms, 3000, "{error}");
        }
        for error in [
            "HTTP 401",
            "HTTP/1.1 403",
            "HTTP/2 401",
            "Server returned non-2xx status code: 403",
            "status_code=401",
            "StatusCode(403)",
            "response status: 401",
            "{\"status\":403}",
        ] {
            let failure = ShardFailure::advance("table", "worker", None, error, 1000);
            assert_eq!(failure.class, FailureClass::DataOrConfiguration, "{error}");
            assert!(failure.needs_attention);
            assert_eq!(failure.next_retry_ms, 3_601_000);
        }
        assert_eq!(
            classify("storage operation timed out"),
            FailureClass::Deadline
        );
        assert_eq!(
            classify("merge ownership unresolved: HTTP 503"),
            FailureClass::OwnershipUnresolved
        );
    }

    #[test]
    fn persistent_budget_backoff_and_attention() {
        let mut old = None;
        for attempt in 1..=20 {
            let next =
                ShardFailure::advance("table", "worker", old, "temporary storage failure", 1000);
            assert_eq!(next.consecutive_attempts, attempt);
            assert_eq!(next.needs_attention, attempt >= 15);
            let expected_secs = match attempt {
                1 | 2 => 2,
                3 => 30,
                4 => 60,
                5 => 120,
                6 => 240,
                7 => 480,
                8..=14 => 900,
                _ => 3600,
            };
            assert_eq!(
                next.next_retry_ms - next.last_failure_ms,
                expected_secs * 1000
            );
            old = Some(next);
        }
        let broken =
            ShardFailure::advance("table", "worker", None, "Not found: /data/1.lance", 1000);
        assert!(broken.needs_attention);
        assert_eq!(broken.next_retry_ms, 3_601_000);
    }

    #[test]
    fn permanent_failures_and_unresolved_barriers_have_distinct_probe_budgets() {
        for error in [
            "NotFound: data/fragment.lance",
            "corrupt manifest",
            "schema mismatch",
            "Permission denied",
            "AccessDenied",
            "unauthenticated",
            "HTTP 403 Forbidden",
        ] {
            let failure = ShardFailure::advance("table", "worker", None, error, 1000);
            assert_eq!(failure.class, FailureClass::DataOrConfiguration);
            assert!(failure.needs_attention);
            assert_eq!(failure.next_retry_ms, 3_601_000);
        }
        let mut old = None;
        for expected_secs in [30, 60, 120, 240, 300, 300] {
            let next = ShardFailure::advance(
                "table",
                "worker",
                old,
                "merge ownership unresolved: recovery barrier failed: timeout",
                1000,
            );
            assert_eq!(next.class, FailureClass::OwnershipUnresolved);
            assert!(next.needs_attention);
            assert_eq!(
                next.next_retry_ms - next.last_failure_ms,
                expected_secs * 1000
            );
            old = Some(next);
        }
    }
}
