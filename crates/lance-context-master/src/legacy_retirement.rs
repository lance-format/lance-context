//! One-target legacy retirement: capture live identity, verify its termination,
//! persist a storage barrier plan, and publish a terminal execution atomically.
//! This module never kills a process, deletes a claim, or bypasses storage fencing.
use crate::{config::MasterConfig, error::MasterError, state::MasterState};
use axum::{extract::State, Json};
use lance_context_merge::{
    legacy_retirement::{ProcessIdentity, Retirement},
    Coordinator,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{sync::Arc, time::Duration};

type Result<T> = std::result::Result<T, String>;
const ACCOUNT: &str = "/var/run/secrets/kubernetes.io/serviceaccount";

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Prepare,
    Complete,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub action: Action,
    pub target: String,
    pub execution_id: String,
    pub namespace: String,
    pub pod: String,
    pub container: String,
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub retirement: Retirement,
    pub target_lock_released: bool,
}

/// A prepared journal is an explicit migration intent, not a timeout heuristic.
/// Canonical task retries finish that handoff themselves once the exact old
/// process is terminated. Unknown legacy writers still require preparation.
pub(crate) async fn reconcile_prepared(
    config: &MasterConfig,
    coordinator: &Coordinator,
    execution: lance_context_merge::Execution,
) -> Result<lance_context_merge::Execution> {
    if execution.protocol != 0
        || !matches!(
            execution.phase,
            lance_context_merge::Phase::Running | lance_context_merge::Phase::Uncertain
        )
    {
        return Ok(execution);
    }
    let Some(record) = coordinator.legacy_retirement(&execution.id).await? else {
        return Ok(execution);
    };
    let request = completion_request(&execution, &record)?;
    let result = execute(config, coordinator, request).await;
    metrics::counter!("master_legacy_retirement_attempts_total", "result" => if result.is_ok() { "completed" } else { "unresolved" }).increment(1);
    result?;
    let current = coordinator
        .get(&execution.target)
        .await?
        .ok_or("retired execution disappeared before canonical reconciliation")?;
    if current.id != execution.id || current.phase != lance_context_merge::Phase::Finished {
        return Err("retired execution changed before canonical reconciliation".into());
    }
    tracing::info!(target = %execution.target, execution_id = %execution.id,
        "verified legacy retirement completed by canonical merge task");
    Ok(current)
}

fn completion_request(
    execution: &lance_context_merge::Execution,
    record: &Retirement,
) -> Result<Request> {
    if record.completed || record.execution != *execution {
        return Err("prepared retirement does not match current execution".into());
    }
    Ok(Request {
        action: Action::Complete,
        target: execution.target.clone(),
        execution_id: execution.id.clone(),
        namespace: record.process.namespace.clone(),
        pod: record.process.pod.clone(),
        container: record.process.container.clone(),
    })
}

pub async fn route(
    State(state): State<Arc<MasterState>>,
    Json(request): Json<Request>,
) -> std::result::Result<Json<Response>, MasterError> {
    let _admission = state
        .admission
        .try_admit_named("legacy_retirement")
        .ok_or_else(|| MasterError::InvalidRequest("master is draining".into()))?;
    execute(
        &state.config,
        &state.task_store.merge_coordinator(),
        request,
    )
    .await
    .map(Json)
    .map_err(MasterError::Internal)
}

pub async fn execute(
    config: &MasterConfig,
    coordinator: &Coordinator,
    request: Request,
) -> Result<Response> {
    tokio::time::timeout(
        Duration::from_secs(120),
        execute_inner(config, coordinator, request),
    )
    .await
    .map_err(|_| "retirement incomplete; persisted journal and target lock retained".to_string())?
}

async fn execute_inner(
    config: &MasterConfig,
    coordinator: &Coordinator,
    request: Request,
) -> Result<Response> {
    for value in [&request.namespace, &request.pod, &request.container] {
        if value.is_empty()
            || value.len() > 253
            || !value
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-' || c == b'.')
        {
            return Err("invalid Kubernetes identity".into());
        }
    }
    if request.target.is_empty()
        || request.target.len() > 512
        || request.target.starts_with("datagen:")
    {
        return Err("retirement supports one explicit rollout/generic target".into());
    }
    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .add_root_certificate(
            reqwest::Certificate::from_pem(
                &std::fs::read(format!("{ACCOUNT}/ca.crt")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?,
        )
        .build()
        .map_err(|e| e.to_string())?;
    let prior = coordinator.legacy_retirement(&request.execution_id).await?;
    if let Some(record) = &prior {
        if record.execution.target != request.target
            || record.process.namespace != request.namespace
            || record.process.pod != request.pod
            || record.process.container != request.container
        {
            return Err("request does not match prepared retirement".into());
        }
        if record.completed {
            return Ok(Response {
                retirement: record.clone(),
                target_lock_released: false,
            });
        }
    }
    let execution = coordinator
        .get(&request.target)
        .await?
        .ok_or("execution not found")?;
    if execution.id != request.execution_id
        || execution.protocol != 0
        || !config
            .worker_endpoints
            .iter()
            .any(|e| e.trim_end_matches('/') == execution.endpoint.trim_end_matches('/'))
    {
        return Err("execution changed or endpoint is not a configured legacy worker".into());
    }
    let before = pod(&client, &request).await?;
    let address = worker_address(&before, &request, &execution.endpoint)?;
    let capability =
        bounded_json(client.get(format!("{address}/api/v1/internal/merge-executor"))).await?;
    let after = pod(&client, &request).await?;
    let identity = running_identity(&before, &request)?;
    if identity != running_identity(&after, &request)? {
        return Err("worker identity changed during capability probe".into());
    }
    match request.action {
        Action::Prepare => {
            if prior.is_some() {
                return Err(
                    "retirement already prepared; use complete after verified termination".into(),
                );
            }
            let instance = capability["instance"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or("worker instance absent")?;
            if capability["shard_name"].as_str() != Some(identity.2.as_str()) {
                return Err("worker shard identity does not match its Pod configuration".into());
            }
            let process = ProcessIdentity {
                namespace: request.namespace,
                pod: request.pod,
                container: request.container,
                pod_uid: identity.0,
                container_id: identity.1,
                writer_id: identity.2,
                instance: instance.into(),
            };
            let record = coordinator
                .prepare_legacy_retirement(&execution, process)
                .await?;
            Ok(Response {
                retirement: record,
                target_lock_released: false,
            })
        }
        Action::Complete => {
            let mut record =
                prior.ok_or("prepare must capture the live process before retirement")?;
            if execution != record.execution {
                return Err("execution changed after retirement preparation".into());
            }
            verify_terminated(&after, &request, &record.process, &capability)?;
            let replacement = bounded_json(
                client
                    .get(format!(
                        "{address}/api/v1/internal/merge-executor/retirement"
                    ))
                    .query(&[("target", &request.target)]),
            )
            .await?;
            verify_replacement(&replacement, &record, &capability, &config.data_dir)?;
            let uri = match request.target.strip_prefix("generic:") {
                Some(name) => format!(
                    "{}/{}.generic.lance",
                    config.data_dir.trim_end_matches('/'),
                    name
                ),
                None => format!(
                    "{}/{}.rollout.lance",
                    config.data_dir.trim_end_matches('/'),
                    request.target
                ),
            };
            if record.versions.is_none() {
                let versions = lance_context_core::legacy_retirement::snapshot_retired_writer(
                    &uri,
                    None,
                    &record.process.writer_id,
                )
                .await
                .map_err(|e| e.to_string())?;
                record = coordinator
                    .plan_legacy_retirement(&record, versions)
                    .await?;
            }
            lance_context_core::merge_write_scope::fence_manifest_versions(
                &uri,
                None,
                record.versions.as_ref().unwrap(),
                &execution.id,
            )
            .await
            .map_err(|e| e.to_string())?;
            // Recheck Kubernetes after storage work. A rolling/replaced process
            // never authorizes an incomplete handoff based on a stale ready probe.
            let latest = pod(&client, &request).await?;
            if running_identity(&latest, &request)? != identity {
                return Err("replacement changed during storage fencing; journal retained".into());
            }
            if !coordinator.finish_legacy_retirement(&record).await? {
                return Err(
                    "execution changed before terminal publication; journal retained".into(),
                );
            }
            record.completed = true;
            Ok(Response {
                retirement: record,
                target_lock_released: false,
            })
        }
    }
}

async fn bounded_json(request: reqwest::RequestBuilder) -> Result<Value> {
    let mut response = request
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        if bytes.len() + chunk.len() > 262_144 {
            return Err("retirement response exceeds metadata bound".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

async fn pod(client: &Client, request: &Request) -> Result<Value> {
    let token = std::fs::read_to_string(format!("{ACCOUNT}/token")).map_err(|e| e.to_string())?;
    bounded_json(
        client
            .get(format!(
                "https://kubernetes.default.svc/api/v1/namespaces/{}/pods/{}",
                request.namespace, request.pod
            ))
            .bearer_auth(token.trim()),
    )
    .await
}

fn status<'a>(pod: &'a Value, request: &Request) -> Result<&'a Value> {
    pod["status"]["containerStatuses"]
        .as_array()
        .ok_or("container statuses absent")?
        .iter()
        .find(|c| c["name"].as_str() == Some(&request.container))
        .ok_or("container status absent".into())
}

fn running_identity(pod: &Value, request: &Request) -> Result<(String, String, String)> {
    if pod["metadata"]["name"].as_str() != Some(&request.pod)
        || pod["metadata"]["namespace"].as_str() != Some(&request.namespace)
        || !pod["metadata"]["deletionTimestamp"].is_null()
    {
        return Err("Pod identity mismatch or deletion in progress".into());
    }
    let status = status(pod, request)?;
    if status["ready"] != true || status["state"]["running"].is_null() {
        return Err("worker is not running/ready".into());
    }
    let spec = pod["spec"]["containers"]
        .as_array()
        .ok_or("containers absent")?
        .iter()
        .find(|c| c["name"].as_str() == Some(&request.container))
        .ok_or("container spec absent")?;
    if spec["args"].as_array().is_some_and(|a| !a.is_empty())
        || spec["command"].as_array().is_some_and(|a| !a.is_empty())
    {
        return Err("custom worker command requires separate writer identity verification".into());
    }
    let mut writer = pod["spec"]["hostname"]
        .as_str()
        .unwrap_or(&request.pod)
        .to_owned();
    if let Some(env) = spec["env"].as_array() {
        for name in ["HOSTNAME", "INSTANCE_ID"] {
            if let Some(entry) = env.iter().find(|e| e["name"] == name) {
                writer = if let Some(value) = entry["value"].as_str() {
                    value.to_owned()
                } else if entry["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.name" {
                    request.pod.clone()
                } else {
                    return Err("unresolved worker writer identity".into());
                };
            }
        }
    }
    if writer.is_empty() {
        return Err("empty writer identity".into());
    }
    Ok((
        pod["metadata"]["uid"]
            .as_str()
            .ok_or("Pod UID absent")?
            .into(),
        status["containerID"]
            .as_str()
            .ok_or("container ID absent")?
            .into(),
        writer,
    ))
}

fn worker_address(pod: &Value, request: &Request, endpoint: &str) -> Result<String> {
    let url = reqwest::Url::parse(endpoint).map_err(|e| e.to_string())?;
    let ip = pod["status"]["podIP"].as_str().ok_or("Pod IP absent")?;
    let service = pod["spec"]["subdomain"].as_str().unwrap_or("");
    let dns = format!(
        "{}.{}.{}.svc.cluster.local",
        request.pod, service, request.namespace
    );
    let short = format!("{}.{}", request.pod, service);
    let namespaced = format!("{short}.{}", request.namespace);
    if url.scheme() != "http"
        || url.port() != Some(3000)
        || !["", "/"].contains(&url.path())
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || ![ip, dns.as_str(), short.as_str(), namespaced.as_str()]
            .contains(&url.host_str().unwrap_or(""))
    {
        return Err("endpoint is not this Pod's direct worker address".into());
    }
    Ok(format!("http://{ip}:3000"))
}

fn verify_terminated(
    pod: &Value,
    request: &Request,
    old: &ProcessIdentity,
    capability: &Value,
) -> Result<()> {
    let (uid, cid, writer) = running_identity(pod, request)?;
    let terminated = &status(pod, request)?["lastState"]["terminated"];
    if uid != old.pod_uid
        || cid == old.container_id
        || writer != old.writer_id
        || terminated["containerID"].as_str() != Some(&old.container_id)
        || terminated["exitCode"].as_i64().is_none()
        || terminated["finishedAt"].as_str().is_none()
        || capability["instance"]
            .as_str()
            .is_none_or(|s| s.is_empty() || s == old.instance)
        || capability["protocol"] != 2
        || capability["fenced_legacy_merges"] != true
    {
        return Err("exact old process termination and fenced replacement are required".into());
    }
    Ok(())
}

fn verify_replacement(
    value: &Value,
    record: &Retirement,
    capability: &Value,
    base_uri: &str,
) -> Result<()> {
    if value["instance"] != capability["instance"]
        || value["fenced_legacy_merges"] != true
        || value["base_uri"].as_str().map(|s| s.trim_end_matches('/'))
            != Some(base_uri.trim_end_matches('/'))
        || value["shard_name"].as_str() != Some(record.process.writer_id.as_str())
        || serde_json::from_value::<lance_context_merge::Execution>(value["execution"].clone())
            .ok()
            .as_ref()
            != Some(&record.execution)
    {
        return Err(
            "replacement storage namespace or durable execution does not match retirement".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> (Request, Value, ProcessIdentity, Value) {
        let request = Request {
            action: Action::Complete,
            target: "table".into(),
            execution_id: "unused".into(),
            namespace: "test".into(),
            pod: "worker-9".into(),
            container: "server".into(),
        };
        let pod = json!({
            "metadata":{"name":"worker-9","namespace":"test","uid":"same-pod"},
            "spec":{"hostname":"worker-9","subdomain":"workers","containers":[{"name":"server"}]},
            "status":{"podIP":"10.1.2.3","containerStatuses":[{"name":"server","ready":true,
                "containerID":"containerd://new","state":{"running":{}},
                "lastState":{"terminated":{"containerID":"containerd://old","exitCode":137,"finishedAt":"2026-10-22T00:00:00Z"}}}]}
        });
        let old = ProcessIdentity {
            namespace: "test".into(),
            pod: "worker-9".into(),
            pod_uid: "same-pod".into(),
            container: "server".into(),
            container_id: "containerd://old".into(),
            instance: "old-instance".into(),
            writer_id: "worker-9".into(),
        };
        let capability =
            json!({"instance":"new-instance","protocol":2,"fenced_legacy_merges":true});
        (request, pod, old, capability)
    }

    #[test]
    fn legacy_retirement_automatic_completion_cannot_adopt_another_execution() {
        let (_, _, process, _) = fixture();
        let mut execution =
            lance_context_merge::Execution::new("table", "worker", "legacy-http", 1);
        execution.protocol = 0;
        execution.phase = lance_context_merge::Phase::Running;
        let record = Retirement {
            execution: execution.clone(),
            process,
            versions: None,
            completed: false,
        };
        let request = completion_request(&execution, &record).unwrap();
        assert!(matches!(request.action, Action::Complete));
        assert_eq!(request.execution_id, execution.id);
        assert_eq!(request.pod, "worker-9");
        let mut successor = execution.clone();
        successor.id = lance_context_core::generate_id();
        assert!(completion_request(&successor, &record).is_err());
        let mut different_endpoint = execution.clone();
        different_endpoint.endpoint = "other-worker".into();
        assert!(completion_request(&different_endpoint, &record).is_err());
        let completed = Retirement {
            completed: true,
            ..record
        };
        assert!(completion_request(&execution, &completed).is_err());
    }

    #[test]
    fn legacy_retirement_requires_exact_termination_and_fenced_replacement() {
        let (request, pod, old, capability) = fixture();
        verify_terminated(&pod, &request, &old, &capability).unwrap();
        for (pointer, value) in [
            ("/metadata/uid", json!("replacement-pod")),
            ("/metadata/deletionTimestamp", json!("2026-10-22T00:00:01Z")),
            (
                "/status/containerStatuses/0/containerID",
                json!("containerd://old"),
            ),
            ("/status/containerStatuses/0/ready", json!(false)),
            ("/status/containerStatuses/0/lastState", json!({})),
            (
                "/status/containerStatuses/0/lastState/terminated/containerID",
                json!("containerd://unrelated"),
            ),
            (
                "/status/containerStatuses/0/lastState/terminated/finishedAt",
                Value::Null,
            ),
            ("/spec/hostname", json!("other-writer")),
        ] {
            let mut bad = pod.clone();
            if pointer == "/metadata/deletionTimestamp" {
                bad["metadata"]["deletionTimestamp"] = value;
            } else {
                *bad.pointer_mut(pointer).unwrap() = value;
            }
            assert!(
                verify_terminated(&bad, &request, &old, &capability).is_err(),
                "accepted {pointer}"
            );
        }
        for (field, value) in [
            ("instance", json!("old-instance")),
            ("protocol", json!(1)),
            ("fenced_legacy_merges", json!(false)),
        ] {
            let mut bad = capability.clone();
            bad[field] = value;
            assert!(
                verify_terminated(&pod, &request, &old, &bad).is_err(),
                "accepted {field}"
            );
        }
    }

    #[test]
    fn legacy_retirement_rejects_service_alias_and_cross_namespace_endpoints() {
        let (request, pod, _, _) = fixture();
        for good in [
            "http://10.1.2.3:3000",
            "http://worker-9.workers:3000",
            "http://worker-9.workers.test.svc.cluster.local:3000",
        ] {
            assert_eq!(
                worker_address(&pod, &request, good).unwrap(),
                "http://10.1.2.3:3000"
            );
        }
        for bad in [
            "http://workers:3000",
            "http://worker-9.workers.other.svc.cluster.local:3000",
            "http://10.1.2.4:3000",
            "http://10.1.2.3:3000?x=1",
            "http://user@10.1.2.3:3000",
        ] {
            assert!(worker_address(&pod, &request, bad).is_err());
        }
    }

    #[test]
    fn legacy_retirement_replacement_must_share_exact_execution_and_storage() {
        let (_, _, process, capability) = fixture();
        let record = Retirement {
            execution: lance_context_merge::Execution::new("table", "worker", "old", 1),
            process,
            versions: None,
            completed: false,
        };
        let value = json!({"instance":"new-instance","fenced_legacy_merges":true,"base_uri":"az://data/prefix","shard_name":"worker-9","execution":record.execution});
        verify_replacement(&value, &record, &capability, "az://data/prefix/").unwrap();
        for (field, replacement) in [
            ("base_uri", json!("az://wrong/prefix")),
            ("execution", Value::Null),
            ("shard_name", json!("worker-0")),
            ("instance", json!("another-instance")),
            ("fenced_legacy_merges", json!(false)),
        ] {
            let mut bad = value.clone();
            bad[field] = replacement;
            assert!(
                verify_replacement(&bad, &record, &capability, "az://data/prefix").is_err(),
                "accepted {field}"
            );
        }
    }
}
