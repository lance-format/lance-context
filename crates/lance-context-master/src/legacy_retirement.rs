//! One-target legacy retirement: capture live identity, verify its termination,
//! persist a storage barrier plan, and publish a terminal execution atomically.
//! This module never kills a process, deletes a claim, or bypasses storage fencing.
use crate::{config::MasterConfig, error::MasterError, state::MasterState};
use axum::{extract::State, Json};
use lance_context_merge::{
    legacy_retirement::{ProcessIdentity, Retirement, WorkerRetirementIntent, WorkerTermination},
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
    /// For a late-discovered target, reuse only the process identity from this
    /// completed retirement, after revalidating its exact death and successor.
    #[serde(default)]
    pub witness_execution_id: Option<String>,
    /// A process intent captured while the old worker was live, including when
    /// no target execution existed. Never accepted together with a task witness.
    #[serde(default)]
    pub worker_witness_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerAction {
    PrepareWorker,
    ConfirmWorkerTermination,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerRequest {
    pub action: WorkerAction,
    pub intent_id: String,
    pub endpoint: String,
    pub namespace: String,
    pub pod: String,
    pub container: String,
}

/// Preserve the target request wire format and the existing bounded CLI flag.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum Command {
    Worker(WorkerRequest),
    Target(Request),
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
    let record = match coordinator.legacy_retirement(&execution.id).await? {
        Some(record) => record,
        None => {
            let Some(intent) = coordinator
                .worker_retirement_for_endpoint(&execution.endpoint)
                .await?
            else {
                return Ok(execution);
            };
            // A process intent is not takeover permission. execute() rechecks
            // graceful death, the exact fenced successor, and target identity
            // before it creates the target's own retirement journal.
            execute(
                config,
                coordinator,
                worker_witness_request(&execution, &intent),
            )
            .await?
            .retirement
        }
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

fn worker_witness_request(
    execution: &lance_context_merge::Execution,
    intent: &WorkerRetirementIntent,
) -> Request {
    Request {
        action: Action::Prepare,
        target: execution.target.clone(),
        execution_id: execution.id.clone(),
        namespace: intent.process.namespace.clone(),
        pod: intent.process.pod.clone(),
        container: intent.process.container.clone(),
        witness_execution_id: None,
        worker_witness_id: Some(intent.id.clone()),
    }
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
        witness_execution_id: None,
        worker_witness_id: None,
    })
}

pub async fn route(
    State(state): State<Arc<MasterState>>,
    Json(request): Json<Command>,
) -> std::result::Result<Json<Value>, MasterError> {
    let _admission = state
        .admission
        .try_admit_named("legacy_retirement")
        .ok_or_else(|| MasterError::InvalidRequest("master is draining".into()))?;
    execute_command(
        &state.config,
        &state.task_store.merge_coordinator(),
        request,
    )
    .await
    .map(Json)
    .map_err(MasterError::Internal)
}

pub async fn execute_command(
    config: &MasterConfig,
    coordinator: &Coordinator,
    command: Command,
) -> Result<Value> {
    match command {
        Command::Target(request) => {
            serde_json::to_value(execute(config, coordinator, request).await?)
                .map_err(|e| e.to_string())
        }
        Command::Worker(request) => tokio::time::timeout(Duration::from_secs(120), async {
            match request.action {
                WorkerAction::PrepareWorker => {
                    let intent = prepare_worker(config, coordinator, request).await?;
                    Ok(serde_json::json!({"worker_intent":intent}))
                }
                WorkerAction::ConfirmWorkerTermination => {
                    let receipt = confirm_worker_termination(config, coordinator, request).await?;
                    Ok(serde_json::json!({"worker_termination":receipt}))
                }
            }
        })
        .await
        .map_err(|_| "worker preparation incomplete; no target ownership changed".to_string())?,
    }
}

async fn confirm_worker_termination(
    config: &MasterConfig,
    coordinator: &Coordinator,
    request: WorkerRequest,
) -> Result<WorkerTermination> {
    let endpoint = config
        .worker_endpoints
        .iter()
        .find(|e| e.trim_end_matches('/') == request.endpoint.trim_end_matches('/'))
        .ok_or("worker endpoint is not configured")?;
    let identity_request = Request {
        action: Action::Prepare,
        target: String::new(),
        execution_id: String::new(),
        namespace: request.namespace,
        pod: request.pod,
        container: request.container,
        witness_execution_id: None,
        worker_witness_id: None,
    };
    validate_kube_identity(&identity_request)?;
    let intent = coordinator
        .worker_retirement_intent(&request.intent_id)
        .await?
        .ok_or("worker retirement intent not found")?;
    validate_worker_intent(&identity_request, endpoint, &config.data_dir, &intent)?;
    let client = retirement_client()?;
    let before = pod(&client, &identity_request).await?;
    let address = worker_address(&before, &identity_request, endpoint)?;
    let capability =
        bounded_json(client.get(format!("{address}/api/v1/internal/merge-executor"))).await?;
    let after = pod(&client, &identity_request).await?;
    preserve_worker_termination(
        coordinator,
        &intent,
        &before,
        &after,
        &identity_request,
        &capability,
    )
    .await
}

async fn preserve_worker_termination(
    coordinator: &Coordinator,
    intent: &WorkerRetirementIntent,
    before: &Value,
    after: &Value,
    request: &Request,
    capability: &Value,
) -> Result<WorkerTermination> {
    let successor = capture_process(before, after, request, capability)?;
    if let Some(receipt) = coordinator.worker_termination(intent).await? {
        verify_receipt_successor(after, request, &receipt, capability)?;
        return Ok(receipt);
    }
    verify_graceful_worker_replacement(after, request, &intent.process, capability)?;
    coordinator
        .record_worker_termination(WorkerTermination {
            intent: intent.clone(),
            successor,
            finished_at: status(after, request)?["lastState"]["terminated"]["finishedAt"]
                .as_str()
                .ok_or("old process termination time absent")?
                .into(),
        })
        .await
}

/// A receipt survives later restarts; current readiness, Pod identity, shard,
/// and fenced admission must still be checked afresh. Target barriers are
/// deliberately outside this helper and remain mandatory at every handoff.
fn verify_receipt_successor(
    pod: &Value,
    request: &Request,
    receipt: &WorkerTermination,
    capability: &Value,
) -> Result<()> {
    receipt.validate()?;
    let old = &receipt.intent.process;
    let (uid, cid, writer) = running_identity(pod, request)?;
    if request.namespace != old.namespace
        || request.pod != old.pod
        || request.container != old.container
        || uid != old.pod_uid
        || cid == old.container_id
        || writer != old.writer_id
        || capability["shard_name"].as_str() != Some(old.writer_id.as_str())
        || capability["instance"]
            .as_str()
            .is_none_or(|s| s.is_empty() || s == old.instance)
        || capability["protocol"] != 2
        || capability["fenced_legacy_merges"] != true
    {
        return Err(
            "persisted worker death requires the same Pod and a current fenced successor".into(),
        );
    }
    Ok(())
}

async fn verify_retired_process(
    config: &MasterConfig,
    coordinator: &Coordinator,
    endpoint: &str,
    pod: &Value,
    request: &Request,
    old: &ProcessIdentity,
    capability: &Value,
) -> Result<()> {
    if let Some(intent) = coordinator.worker_retirement_for_endpoint(endpoint).await? {
        if intent.process == *old {
            validate_worker_intent(request, endpoint, &config.data_dir, &intent)?;
            if let Some(receipt) = coordinator.worker_termination(&intent).await? {
                return verify_receipt_successor(pod, request, &receipt, capability);
            }
        }
    }
    verify_terminated(pod, request, old, capability)
}

async fn prepare_worker(
    config: &MasterConfig,
    coordinator: &Coordinator,
    request: WorkerRequest,
) -> Result<WorkerRetirementIntent> {
    let endpoint = config
        .worker_endpoints
        .iter()
        .find(|e| e.trim_end_matches('/') == request.endpoint.trim_end_matches('/'))
        .ok_or("worker endpoint is not configured")?
        .clone();
    // These helpers inspect Kubernetes identity only. No execution or target
    // is fabricated, looked up, or reserved for worker preparation.
    let identity_request = Request {
        action: Action::Prepare,
        target: String::new(),
        execution_id: String::new(),
        namespace: request.namespace,
        pod: request.pod,
        container: request.container,
        witness_execution_id: None,
        worker_witness_id: None,
    };
    validate_kube_identity(&identity_request)?;
    if let Some(prior) = coordinator
        .worker_retirement_intent(&request.intent_id)
        .await?
    {
        validate_worker_intent(&identity_request, &endpoint, &config.data_dir, &prior)?;
        return Ok(prior);
    }
    let client = retirement_client()?;
    let before = pod(&client, &identity_request).await?;
    let address = worker_address(&before, &identity_request, &endpoint)?;
    let capability =
        bounded_json(client.get(format!("{address}/api/v1/internal/merge-executor"))).await?;
    let after = pod(&client, &identity_request).await?;
    let process = capture_process(&before, &after, &identity_request, &capability)?;
    coordinator
        .prepare_worker_retirement(WorkerRetirementIntent {
            id: request.intent_id,
            endpoint,
            base_uri: config.data_dir.trim_end_matches('/').to_owned(),
            process,
        })
        .await
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
    validate_kube_identity(&request)?;
    if request.worker_witness_id.is_some() && request.witness_execution_id.is_some() {
        return Err("choose one retirement witness".into());
    }
    if matches!(request.action, Action::Complete)
        && (request.worker_witness_id.is_some() || request.witness_execution_id.is_some())
    {
        return Err("retirement witnesses are only accepted during preparation".into());
    }
    if request.target.is_empty()
        || request.target.len() > 512
        || request.target.starts_with("datagen:")
    {
        return Err("retirement supports one explicit rollout/generic target".into());
    }
    let client = retirement_client()?;
    execute_target(config, coordinator, request, client).await
}

fn validate_kube_identity(request: &Request) -> Result<()> {
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
    Ok(())
}

fn retirement_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .add_root_certificate(
            reqwest::Certificate::from_pem(
                &std::fs::read(format!("{ACCOUNT}/ca.crt")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?,
        )
        .build()
        .map_err(|e| e.to_string())
}

async fn execute_target(
    config: &MasterConfig,
    coordinator: &Coordinator,
    request: Request,
    client: Client,
) -> Result<Response> {
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
            if let Some(intent_id) = &request.worker_witness_id {
                let intent = coordinator
                    .worker_retirement_intent(intent_id)
                    .await?
                    .ok_or("worker retirement intent not found")?;
                validate_worker_intent(&request, &execution.endpoint, &config.data_dir, &intent)?;
                // Capture death while lastState still names the old process;
                // later retries can use this evidence after another restart.
                preserve_worker_termination(
                    coordinator,
                    &intent,
                    &before,
                    &after,
                    &request,
                    &capability,
                )
                .await?;
                let prospective = Retirement {
                    execution: execution.clone(),
                    process: intent.process.clone(),
                    versions: None,
                    completed: false,
                };
                let replacement = bounded_json(
                    client
                        .get(format!(
                            "{address}/api/v1/internal/merge-executor/retirement"
                        ))
                        .query(&[("target", &request.target)]),
                )
                .await?;
                verify_replacement(&replacement, &prospective, &capability, &config.data_dir)?;
                let latest = pod(&client, &request).await?;
                if running_identity(&latest, &request)? != identity {
                    return Err("replacement changed during worker intent validation".into());
                }
                let record = coordinator
                    .prepare_legacy_retirement_from_worker(&execution, &intent)
                    .await?;
                return Ok(Response {
                    retirement: record,
                    target_lock_released: false,
                });
            }
            if let Some(witness_id) = &request.witness_execution_id {
                let witness = coordinator
                    .legacy_retirement(witness_id)
                    .await?
                    .ok_or("retirement witness not found")?;
                validate_witness(&request, &execution, &witness)?;
                verify_retired_process(
                    config,
                    coordinator,
                    &execution.endpoint,
                    &after,
                    &request,
                    &witness.process,
                    &capability,
                )
                .await?;
                let replacement = bounded_json(
                    client
                        .get(format!(
                            "{address}/api/v1/internal/merge-executor/retirement"
                        ))
                        .query(&[("target", &request.target)]),
                )
                .await?;
                let prospective = Retirement {
                    execution: execution.clone(),
                    process: witness.process.clone(),
                    versions: None,
                    completed: false,
                };
                verify_replacement(&replacement, &prospective, &capability, &config.data_dir)?;
                let latest = pod(&client, &request).await?;
                if running_identity(&latest, &request)? != identity {
                    return Err("replacement changed during witness validation".into());
                }
                let record = coordinator
                    .prepare_legacy_retirement_from_witness(&execution, &witness)
                    .await?;
                return Ok(Response {
                    retirement: record,
                    target_lock_released: false,
                });
            }
            let process = capture_process(&before, &after, &request, &capability)?;
            let record = coordinator
                .prepare_legacy_retirement(&execution, process)
                .await?;
            Ok(Response {
                retirement: record,
                target_lock_released: false,
            })
        }
        Action::Complete => {
            if request.witness_execution_id.is_some() {
                return Err("retirement witness is only accepted during preparation".into());
            }
            let mut record =
                prior.ok_or("prepare must capture the live process before retirement")?;
            if execution != record.execution {
                return Err("execution changed after retirement preparation".into());
            }
            verify_retired_process(
                config,
                coordinator,
                &execution.endpoint,
                &after,
                &request,
                &record.process,
                &capability,
            )
            .await?;
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
            lance_context_core::legacy_retirement::fence_retired_writer(
                &uri,
                None,
                &record.process.writer_id,
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

fn capture_process(
    before: &Value,
    after: &Value,
    request: &Request,
    capability: &Value,
) -> Result<ProcessIdentity> {
    let identity = running_identity(before, request)?;
    if identity != running_identity(after, request)? {
        return Err("worker identity changed during capability probe".into());
    }
    let instance = capability["instance"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("worker instance absent")?;
    if capability["shard_name"].as_str() != Some(identity.2.as_str()) {
        return Err("worker shard identity does not match its Pod configuration".into());
    }
    Ok(ProcessIdentity {
        namespace: request.namespace.clone(),
        pod: request.pod.clone(),
        container: request.container.clone(),
        pod_uid: identity.0,
        container_id: identity.1,
        writer_id: identity.2,
        instance: instance.into(),
    })
}

fn validate_worker_intent(
    request: &Request,
    endpoint: &str,
    base_uri: &str,
    intent: &WorkerRetirementIntent,
) -> Result<()> {
    if intent.endpoint != endpoint
        || intent.base_uri != base_uri.trim_end_matches('/')
        || intent.process.namespace != request.namespace
        || intent.process.pod != request.pod
        || intent.process.container != request.container
    {
        return Err("worker intent does not match endpoint, namespace, or Pod identity".into());
    }
    Ok(())
}

/// A zero-execution handoff has no per-target pre-shutdown journal. Require
/// graceful old-container completion, not an OOM or a forced termination, in
/// addition to the normal same-Pod fenced-successor proof.
fn verify_graceful_worker_replacement(
    pod: &Value,
    request: &Request,
    old: &ProcessIdentity,
    capability: &Value,
) -> Result<()> {
    verify_terminated(pod, request, old, capability)?;
    let terminated = &status(pod, request)?["lastState"]["terminated"];
    if terminated["exitCode"] != 0
        || terminated["reason"] != "Completed"
        || terminated["signal"].as_i64().is_some_and(|s| s != 0)
    {
        return Err("worker-only retirement requires graceful old-container completion".into());
    }
    Ok(())
}

fn validate_witness(
    request: &Request,
    execution: &lance_context_merge::Execution,
    witness: &Retirement,
) -> Result<()> {
    if !witness.completed
        || witness.versions.is_none()
        || witness.execution.id == execution.id
        || witness.execution.endpoint != execution.endpoint
        || witness.process.namespace != request.namespace
        || witness.process.pod != request.pod
        || witness.process.container != request.container
    {
        return Err("completed retirement witness must identify this exact worker".into());
    }
    Ok(())
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
            witness_execution_id: None,
            worker_witness_id: None,
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
    fn worker_intent_checks_lifecycle_and_storage_scope_without_a_target() {
        let (request, mut pod, process, capability) = fixture();
        let intent = WorkerRetirementIntent {
            id: "intent".into(),
            endpoint: "http://worker-9:3000".into(),
            base_uri: "az://data/root".into(),
            process: process.clone(),
        };
        validate_worker_intent(&request, &intent.endpoint, "az://data/root/", &intent).unwrap();
        for field in ["endpoint", "base_uri", "namespace", "pod", "container"] {
            let mut wrong = intent.clone();
            match field {
                "endpoint" => wrong.endpoint = "http://other:3000".into(),
                "base_uri" => wrong.base_uri = "az://other/root".into(),
                "namespace" => wrong.process.namespace = "other".into(),
                "pod" => wrong.process.pod = "other".into(),
                "container" => wrong.process.container = "other".into(),
                _ => unreachable!(),
            }
            assert!(
                validate_worker_intent(&request, &intent.endpoint, &intent.base_uri, &wrong)
                    .is_err(),
                "{field}"
            );
        }
        assert!(verify_graceful_worker_replacement(&pod, &request, &process, &capability).is_err());
        pod["status"]["containerStatuses"][0]["lastState"]["terminated"]["exitCode"] = json!(0);
        pod["status"]["containerStatuses"][0]["lastState"]["terminated"]["reason"] =
            json!("Completed");
        verify_graceful_worker_replacement(&pod, &request, &process, &capability).unwrap();
        for (pointer, value) in [
            (
                "/status/containerStatuses/0/lastState/terminated/reason",
                json!("OOMKilled"),
            ),
            (
                "/status/containerStatuses/0/lastState/terminated/signal",
                json!(9),
            ),
            ("/metadata/uid", json!("different-pod")),
            (
                "/status/containerStatuses/0/lastState/terminated/containerID",
                json!("other"),
            ),
        ] {
            let mut changed = pod.clone();
            // signal is optional in a Kubernetes termination record.
            if pointer.ends_with("/signal") {
                changed["status"]["containerStatuses"][0]["lastState"]["terminated"]["signal"] =
                    value;
            } else {
                *changed.pointer_mut(pointer).unwrap() = value;
            }
            assert!(
                verify_graceful_worker_replacement(&changed, &request, &process, &capability)
                    .is_err(),
                "{pointer}"
            );
        }
    }

    #[test]
    fn worker_death_receipt_survives_another_restart_but_not_identity_changes() {
        let (request, mut pod, process, mut capability) = fixture();
        capability["shard_name"] = json!(process.writer_id);
        pod["status"]["containerStatuses"][0]["lastState"]["terminated"]["exitCode"] = json!(0);
        pod["status"]["containerStatuses"][0]["lastState"]["terminated"]["reason"] =
            json!("Completed");
        verify_graceful_worker_replacement(&pod, &request, &process, &capability).unwrap();
        let receipt = WorkerTermination {
            intent: WorkerRetirementIntent {
                id: lance_context_core::generate_id(),
                endpoint: "http://worker:3000".into(),
                base_uri: "az://data/root".into(),
                process,
            },
            successor: capture_process(&pod, &pod, &request, &capability).unwrap(),
            finished_at: "2026-10-22T00:00:00Z".into(),
        };
        pod["status"]["containerStatuses"][0]["containerID"] = json!("containerd://third");
        pod["status"]["containerStatuses"][0]["lastState"]["terminated"]["containerID"] =
            json!("containerd://new");
        capability["instance"] = json!("third-instance");
        assert!(verify_terminated(&pod, &request, &receipt.intent.process, &capability).is_err());
        verify_receipt_successor(&pod, &request, &receipt, &capability).unwrap();
        for (pointer, value) in [
            ("/metadata/uid", json!("different-pod")),
            ("/metadata/deletionTimestamp", json!("2026-10-23T00:00:00Z")),
            ("/status/containerStatuses/0/ready", json!(false)),
            (
                "/status/containerStatuses/0/containerID",
                json!("containerd://old"),
            ),
            ("/spec/hostname", json!("different-writer")),
        ] {
            let mut changed = pod.clone();
            if pointer == "/metadata/deletionTimestamp" {
                changed["metadata"]["deletionTimestamp"] = value;
            } else {
                *changed.pointer_mut(pointer).unwrap() = value;
            }
            assert!(
                verify_receipt_successor(&changed, &request, &receipt, &capability).is_err(),
                "{pointer}"
            );
        }
        for (field, value) in [
            ("instance", json!("old-instance")),
            ("protocol", json!(1)),
            ("fenced_legacy_merges", json!(false)),
            ("shard_name", json!("other")),
        ] {
            let mut changed = capability.clone();
            changed[field] = value;
            assert!(
                verify_receipt_successor(&pod, &request, &receipt, &changed).is_err(),
                "{field}"
            );
        }
        let mut wrong = receipt;
        wrong.successor.pod_uid = "unrelated".into();
        assert!(verify_receipt_successor(&pod, &request, &wrong, &capability).is_err());
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn worker_death_receipt_allows_late_target_after_second_restart() {
        let endpoint = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let client = etcd_client::Client::connect([endpoint], None)
            .await
            .unwrap();
        let coordinator = Coordinator::new(
            client,
            format!("/master-worker-death/{}", lance_context_core::generate_id()),
        );
        let (request, mut pod, old, mut capability) = fixture();
        capability["shard_name"] = json!(old.writer_id);
        pod["status"]["containerStatuses"][0]["lastState"]["terminated"]["exitCode"] = json!(0);
        pod["status"]["containerStatuses"][0]["lastState"]["terminated"]["reason"] =
            json!("Completed");
        let intent = WorkerRetirementIntent {
            id: lance_context_core::generate_id(),
            endpoint: "http://worker:3000".into(),
            base_uri: "az://data/root".into(),
            process: old.clone(),
        };
        coordinator
            .prepare_worker_retirement(intent.clone())
            .await
            .unwrap();
        let receipt =
            preserve_worker_termination(&coordinator, &intent, &pod, &pod, &request, &capability)
                .await
                .unwrap();
        // Re-create the single-entry Kubernetes history after a second upgrade.
        pod["status"]["containerStatuses"][0]["containerID"] = json!("containerd://third");
        pod["status"]["containerStatuses"][0]["lastState"]["terminated"]["containerID"] =
            json!("containerd://new");
        capability["instance"] = json!("third-instance");
        assert_eq!(
            preserve_worker_termination(&coordinator, &intent, &pod, &pod, &request, &capability)
                .await
                .unwrap(),
            receipt
        );
        use clap::Parser;
        let config =
            MasterConfig::try_parse_from(["master", "--data-dir", "az://data/root"]).unwrap();
        verify_retired_process(
            &config,
            &coordinator,
            &intent.endpoint,
            &pod,
            &request,
            &old,
            &capability,
        )
        .await
        .unwrap();
        let mut wrong_scope = config.clone();
        wrong_scope.data_dir = "az://other/root".into();
        assert!(verify_retired_process(
            &wrong_scope,
            &coordinator,
            &intent.endpoint,
            &pod,
            &request,
            &old,
            &capability
        )
        .await
        .is_err());
        // No original lastState evidence may be manufactured for another intent.
        let mut unknown = intent.clone();
        unknown.id = lance_context_core::generate_id();
        assert!(preserve_worker_termination(
            &coordinator,
            &unknown,
            &pod,
            &pod,
            &request,
            &capability
        )
        .await
        .is_err());
        assert!(coordinator
            .worker_termination(&unknown)
            .await
            .unwrap()
            .is_none());
        assert!(coordinator.get(&request.target).await.unwrap().is_none());
    }

    #[test]
    fn worker_capture_rejects_changed_or_unready_process() {
        let (request, pod, _, mut capability) = fixture();
        capability["shard_name"] = json!("worker-9");
        let captured = capture_process(&pod, &pod, &request, &capability).unwrap();
        assert_eq!(captured.container_id, "containerd://new");
        let mut changed = pod.clone();
        changed["status"]["containerStatuses"][0]["containerID"] = json!("containerd://later");
        assert!(capture_process(&pod, &changed, &request, &capability).is_err());
        changed = pod.clone();
        changed["status"]["containerStatuses"][0]["ready"] = json!(false);
        assert!(capture_process(&pod, &changed, &request, &capability).is_err());
        capability["shard_name"] = json!("other");
        assert!(capture_process(&pod, &pod, &request, &capability).is_err());
    }

    #[test]
    fn worker_command_requires_explicit_action_and_does_not_accept_process_evidence() {
        let request = json!({"action":"prepare_worker","intent_id":"id","endpoint":"http://worker:3000","namespace":"test","pod":"worker","container":"server"});
        assert!(matches!(
            serde_json::from_value::<Command>(request.clone()).unwrap(),
            Command::Worker(_)
        ));
        let mut invented = request.clone();
        invented["process"] = json!({"container_id":"forged"});
        assert!(serde_json::from_value::<Command>(invented).is_err());
        let mut ambiguous = request;
        ambiguous["target"] = json!("table");
        assert!(serde_json::from_value::<Command>(ambiguous).is_err());
        let target = json!({"action":"prepare","target":"table","execution_id":"id","namespace":"test","pod":"worker","container":"server"});
        assert!(matches!(
            serde_json::from_value::<Command>(target).unwrap(),
            Command::Target(_)
        ));
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
    fn retirement_witness_must_be_complete_and_match_the_worker() {
        let (request, pod, process, capability) = fixture();
        let execution =
            lance_context_merge::Execution::new("late", "http://worker-9:3000", "legacy-http", 1);
        let mut previous = execution.clone();
        previous.id = "earlier".into();
        previous.target = "previous-table".into();
        let witness = Retirement {
            execution: previous,
            process,
            versions: Some(std::collections::BTreeMap::from([("base".into(), 12)])),
            completed: true,
        };
        validate_witness(&request, &execution, &witness).unwrap();
        verify_terminated(&pod, &request, &witness.process, &capability).unwrap();
        for field in [
            "completed",
            "versions",
            "namespace",
            "pod",
            "container",
            "endpoint",
            "id",
        ] {
            let mut wrong = witness.clone();
            match field {
                "completed" => wrong.completed = false,
                "versions" => wrong.versions = None,
                "namespace" => wrong.process.namespace = "other".into(),
                "pod" => wrong.process.pod = "other".into(),
                "container" => wrong.process.container = "other".into(),
                "endpoint" => wrong.execution.endpoint = "http://other:3000".into(),
                "id" => wrong.execution.id = execution.id.clone(),
                _ => unreachable!(),
            }
            assert!(
                validate_witness(&request, &execution, &wrong).is_err(),
                "{field}"
            );
        }
        let mut living = pod;
        living["status"]["containerStatuses"][0]["containerID"] = json!("containerd://old");
        assert!(verify_terminated(&living, &request, &witness.process, &capability).is_err());
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
