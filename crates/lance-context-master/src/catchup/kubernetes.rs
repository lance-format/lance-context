use super::{store::Record, CatchupConfig, Result};
use crate::config::MasterConfig;
use reqwest::{Client, Method, StatusCode};
use serde_json::{json, Value};
use std::{path::Path, time::Duration};

const SERVICE_ACCOUNT: &str = "/var/run/secrets/kubernetes.io/serviceaccount";

pub(super) fn read_template(config: &CatchupConfig) -> Result<Value> {
    let path = config
        .pod_template
        .as_ref()
        .ok_or("missing catch-up PodSpec template")?;
    let bytes = std::fs::read(path).map_err(|e| format!("read catch-up template: {e}"))?;
    if bytes.len() > 65_536 {
        return Err("catch-up template exceeds 64 KiB".into());
    }
    let spec: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate_template(&spec)?;
    Ok(spec)
}
fn validate_template(spec: &Value) -> Result<()> {
    let containers = spec["containers"]
        .as_array()
        .ok_or("template requires containers")?;
    if containers.len() != 1
        || spec.get("initContainers").is_some()
        || spec.get("ephemeralContainers").is_some()
        || spec.get("nodeName").is_some()
    {
        return Err("catch-up requires one container, no init containers or nodeName".into());
    }
    let c = &containers[0];
    let image = c["image"].as_str().ok_or("missing image")?;
    if !image.rsplit_once("@sha256:").is_some_and(|(_, digest)| {
        digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit())
    }) {
        return Err("catch-up image must be pinned by SHA-256 digest".into());
    }
    if !c["command"].as_array().is_some_and(|a| {
        a.len() == 1
            && a[0]
                .as_str()
                .is_some_and(|s| s.ends_with("lance-context-master"))
    }) {
        return Err("catch-up command must run the native lance-context-master binary".into());
    }
    for resource in ["cpu", "memory"] {
        let request = &c["resources"]["requests"][resource];
        let limit = &c["resources"]["limits"][resource];
        if request.as_str().is_none_or(|s| s.is_empty() || s == "0") || request != limit {
            return Err("catch-up requires equal, nonzero CPU/memory requests and limits".into());
        }
    }
    Ok(())
}

pub(super) struct Kubernetes {
    client: Client,
    base: String,
    token_file: String,
    pods_base: String,
}
impl Kubernetes {
    pub fn in_cluster(config: &CatchupConfig) -> Result<Self> {
        let ca = std::fs::read(Path::new(SERVICE_ACCOUNT).join("ca.crt"))
            .map_err(|e| format!("read Kubernetes CA: {e}"))?;
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .add_root_certificate(reqwest::Certificate::from_pem(&ca).map_err(|e| e.to_string())?)
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            client,
            base: format!(
                "https://kubernetes.default.svc/apis/batch/v1/namespaces/{}/jobs",
                config.namespace
            ),
            token_file: format!("{SERVICE_ACCOUNT}/token"),
            pods_base: format!(
                "https://kubernetes.default.svc/api/v1/namespaces/{}/pods",
                config.namespace
            ),
        })
    }
    async fn request(
        &self,
        method: Method,
        url: &str,
        body: Option<&Value>,
    ) -> Result<(StatusCode, Value)> {
        // Re-read projected tokens so long-lived masters survive token rotation.
        let token = std::fs::read_to_string(&self.token_file)
            .map_err(|e| format!("read Kubernetes token: {e}"))?;
        let patch = method == Method::PATCH;
        let mut request = self.client.request(method, url).bearer_auth(token.trim());
        if patch {
            request = request.header("Content-Type", "application/json-patch+json");
        }
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().await.map_err(|e| e.to_string())?;
        let status = response.status();
        let body = response.json().await.map_err(|e| e.to_string())?;
        Ok((status, body))
    }
    pub async fn reconcile(
        &self,
        config: &MasterConfig,
        record: &Record,
        stop: bool,
    ) -> Result<(Option<String>, Option<bool>)> {
        let (status, mut job) = self
            .request(Method::GET, &format!("{}/{}", self.base, record.job), None)
            .await?;
        if status == StatusCode::NOT_FOUND {
            if stop || record.job_uid.is_some() {
                return Err(
                    "catch-up Job missing after confirmation or startup stall; reservation retained"
                        .into(),
                );
            }
            let desired = render_job(config, record, read_template(&config.catchup)?);
            let (created, body) = self
                .request(Method::POST, &self.base, Some(&desired))
                .await?;
            // Ambiguous response and conflict both reconcile by the same name.
            if created == StatusCode::CONFLICT {
                return Ok((None, None));
            }
            if !created.is_success() {
                return Err(format!("Kubernetes Job creation returned {created}"));
            }
            job = body;
        } else if !status.is_success() {
            return Err(format!("Kubernetes Job lookup returned {status}"));
        }
        if job["metadata"]["annotations"]["lance-context/target"] != record.target
            || job["metadata"]["labels"]["lance-context/catchup"] != "native-v1"
        {
            return Err("Job identity mismatch; refusing adoption".into());
        }
        let uid = job["metadata"]["uid"]
            .as_str()
            .ok_or("Job has no UID")?
            .to_string();
        if record
            .job_uid
            .as_ref()
            .is_some_and(|expected| expected != &uid)
        {
            return Err("Job UID changed; refusing adoption or termination".into());
        }
        let terminal = job["status"]["conditions"]
            .as_array()
            .and_then(|conditions| {
                conditions.iter().find_map(|c| {
                    if c["status"] != "True" {
                        return None;
                    }
                    match c["type"].as_str() {
                        Some("Complete") => Some(true),
                        Some("Failed") => Some(false),
                        _ => None,
                    }
                })
            });
        let Some(success) = terminal else {
            if stop && record.job_uid.is_some() {
                let uid = job["metadata"]["uid"].as_str().ok_or("Job has no UID")?;
                let version = job["metadata"]["resourceVersion"]
                    .as_str()
                    .ok_or("Job has no resourceVersion")?;
                // Only a confirmed stall enables a deadline. CAS prevents a
                // delayed controller from touching a replaced/changed Job.
                let patch = json!([
                    {"op":"test","path":"/metadata/uid","value":uid},
                    {"op":"test","path":"/metadata/resourceVersion","value":version},
                    {"op":"add","path":"/spec/activeDeadlineSeconds","value":1}
                ]);
                let (status, _) = self
                    .request(
                        Method::PATCH,
                        &format!("{}/{}", self.base, record.job),
                        Some(&patch),
                    )
                    .await?;
                if !status.is_success() {
                    return Err(format!("stalled Job termination returned {status}"));
                }
                tracing::warn!(target = %record.target, job = %record.job,
                    "confirmed no-progress catch-up; requested Job termination, retaining ownership");
            }
            return Ok((Some(uid), None));
        };
        // Older Kubernetes versions publish Failed while Pods still terminate.
        // Check Job UID and every Pod phase before giving the resource slot back.
        if !uid.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err("invalid Job UID".into());
        }
        let url = format!(
            "{}?labelSelector=batch.kubernetes.io%2Fcontroller-uid%3D{uid}&limit=16",
            self.pods_base
        );
        let (status, pods) = self.request(Method::GET, &url, None).await?;
        if !status.is_success()
            || pods["metadata"]["continue"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
        {
            return Err("cannot confirm catch-up Pods terminated".into());
        }
        let items = pods["items"].as_array().ok_or("invalid Pod list")?;
        if items
            .iter()
            .any(|p| !matches!(p["status"]["phase"].as_str(), Some("Succeeded" | "Failed")))
        {
            return Ok((Some(uid), None));
        }
        Ok((Some(uid), Some(success)))
    }
}

pub(super) fn render_job(config: &MasterConfig, record: &Record, mut spec: Value) -> Value {
    let mut env = spec["containers"][0]["env"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut shards = config.catchup.shards.clone();
    if !shards.is_empty() {
        let start = record.attempt.saturating_sub(1) as usize % shards.len();
        shards.rotate_left(start);
    }
    let overrides = [
        ("DATA_DIR", config.data_dir.clone()),
        (
            "ROLLOUT_KEY_INDEX_TYPE",
            config.key_index_type.as_str().into(),
        ),
        ("ETCD_ENDPOINTS", config.etcd.etcd_endpoints.join(",")),
        ("ETCD_PREFIX", config.etcd.etcd_prefix.clone()),
        ("CATCHUP_ENABLED", "false".into()),
        (
            "CATCHUP_PIPELINE_ENABLED",
            config.catchup.pipeline_enabled.to_string(),
        ),
        ("CATCHUP_TARGET", record.target.clone()),
        ("CATCHUP_JOB_NAME", record.job.clone()),
        ("CATCHUP_SHARDS", shards.join(",")),
        (
            "CATCHUP_MERGE_MAX_GENERATIONS",
            config.catchup.merge_max_generations.to_string(),
        ),
        (
            "CATCHUP_MERGE_MAX_BYTES",
            config.catchup.merge_max_bytes.to_string(),
        ),
        (
            "CATCHUP_MERGE_MEMORY_BYTES",
            config.catchup.merge_memory_bytes.to_string(),
        ),
        ("MERGE_OWNED_TARGETS", record.target.clone()),
        ("MERGE_DRAIN_TARGETS", String::new()),
        ("CATCHUP_SLICE_SECS", config.catchup.slice_secs.to_string()),
        (
            "MAINTENANCE_IDLE_TIMEOUT_SECS",
            config.maintenance.maintenance_idle_timeout_secs.to_string(),
        ),
        ("ROLLOUT_CACHE_BYTES", "134217728".into()),
    ];
    for (name, value) in overrides {
        env.retain(|e| e["name"] != name);
        env.push(json!({"name": name, "value": value}));
    }
    spec["containers"][0]["env"] = json!(env);
    spec["containers"][0]["args"] = json!([]);
    spec["restartPolicy"] = json!("Never");
    spec["preemptionPolicy"] = json!("Never");
    spec["terminationGracePeriodSeconds"] = json!(30);
    spec.as_object_mut()
        .expect("validated PodSpec")
        .remove("activeDeadlineSeconds");
    // Dedicated executors need storage/etcd, never Kubernetes permissions.
    spec["automountServiceAccountToken"] = json!(false);
    json!({"apiVersion":"batch/v1","kind":"Job", "metadata":{"name":record.job,
        "namespace":config.catchup.namespace,"labels":{"lance-context/catchup":"native-v1"},
        "annotations":{"lance-context/target":record.target}},
        "spec":{"parallelism":1,"completions":1,"backoffLimit":0,
            "ttlSecondsAfterFinished":86400,
            "template":{"metadata":{"labels":{"lance-context/catchup":"native-v1"}},"spec":spec}}})
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::State, routing::get, Json, Router};
    use clap::Parser;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    struct Mock {
        creates: AtomicUsize,
        terminal: AtomicBool,
        patches: AtomicUsize,
        pods_stopped: AtomicBool,
        job: std::sync::Mutex<Option<Value>>,
    }
    async fn create(
        State(mock): State<Arc<Mock>>,
        Json(mut job): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        mock.creates.fetch_add(1, Ordering::SeqCst);
        job["metadata"]["uid"] = json!("test-uid");
        job["metadata"]["resourceVersion"] = json!("1");
        *mock.job.lock().unwrap() = Some(job);
        // The API committed the Job but the caller received an error.
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({})))
    }
    async fn lookup(State(mock): State<Arc<Mock>>) -> (StatusCode, Json<Value>) {
        let Some(mut job) = mock.job.lock().unwrap().clone() else {
            return (StatusCode::NOT_FOUND, Json(json!({})));
        };
        if mock.terminal.load(Ordering::SeqCst) {
            job["status"] = json!({"conditions":[{"type":"Failed","status":"True"}]});
        }
        (StatusCode::OK, Json(job))
    }
    async fn patch(
        State(mock): State<Arc<Mock>>,
        Json(patch): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        let mut guard = mock.job.lock().unwrap();
        let job = guard.as_mut().unwrap();
        assert_eq!(patch[0]["path"], "/metadata/uid");
        assert_eq!(patch[1]["path"], "/metadata/resourceVersion");
        if patch[0]["value"] != job["metadata"]["uid"]
            || patch[1]["value"] != job["metadata"]["resourceVersion"]
        {
            return (StatusCode::CONFLICT, Json(json!({})));
        }
        assert_eq!(patch[2]["value"], 1);
        mock.patches.fetch_add(1, Ordering::SeqCst);
        job["spec"]["activeDeadlineSeconds"] = json!(1);
        (StatusCode::OK, Json(job.clone()))
    }
    async fn pods(State(mock): State<Arc<Mock>>) -> Json<Value> {
        Json(
            json!({"items":[{"status":{"phase":if mock.pods_stopped.load(Ordering::SeqCst) {"Failed"} else {"Running"}}}]}),
        )
    }
    #[tokio::test]
    async fn ambiguous_create_is_adopted_and_terminating_pods_retain_capacity() {
        let mock = Arc::new(Mock {
            creates: AtomicUsize::new(0),
            terminal: AtomicBool::new(false),
            patches: AtomicUsize::new(0),
            pods_stopped: AtomicBool::new(false),
            job: std::sync::Mutex::new(None),
        });
        let app = Router::new()
            .route("/jobs", axum::routing::post(create))
            .route("/jobs/{name}", get(lookup).patch(patch))
            .route("/pods", get(pods))
            .with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "test-token").unwrap();
        let template = dir.path().join("pod.json");
        std::fs::write(&template, json!({"containers":[{"name":"executor","command":["lance-context-master"],"image":format!("native@sha256:{}","a".repeat(64)),"resources":{"requests":{"cpu":"1","memory":"8Gi"},"limits":{"cpu":"1","memory":"8Gi"}}}]}).to_string()).unwrap();
        let mut config = MasterConfig::parse_from(["master"]);
        config.catchup.pod_template = Some(template.to_string_lossy().into());
        let client = Kubernetes {
            client: Client::new(),
            base: format!("http://{addr}/jobs"),
            pods_base: format!("http://{addr}/pods"),
            token_file: token.to_string_lossy().into(),
        };
        let mut record = Record {
            target: "hot".into(),
            job: "lc-catchup-test".into(),
            job_uid: None,
            slot: 0,
            attempt: 1,
            active: true,
            termination_requested: false,
            reason: "test".into(),
            pending_at_admission: 1000,
            created_at_ms: 0,
            finished_at_ms: None,
            consecutive_failures: 0,
            needs_attention: false,
            next_retry_ms: 0,
            outcome: None,
            progress: None,
        };
        assert!(client.reconcile(&config, &record, false).await.is_err());
        assert_eq!(
            client.reconcile(&config, &record, false).await.unwrap().1,
            None
        );
        assert_eq!(mock.creates.load(Ordering::SeqCst), 1);
        record.job_uid = Some("test-uid".into());
        assert_eq!(mock.patches.load(Ordering::SeqCst), 0);
        client.reconcile(&config, &record, true).await.unwrap();
        assert_eq!(mock.patches.load(Ordering::SeqCst), 1);
        record.job_uid = Some("replaced-uid".into());
        assert!(client
            .reconcile(&config, &record, true)
            .await
            .unwrap_err()
            .contains("UID changed"));
        assert_eq!(mock.patches.load(Ordering::SeqCst), 1);
        record.job_uid = Some("test-uid".into());
        mock.terminal.store(true, Ordering::SeqCst);
        assert_eq!(
            client.reconcile(&config, &record, false).await.unwrap().1,
            None
        );
        mock.pods_stopped.store(true, Ordering::SeqCst);
        assert_eq!(
            client.reconcile(&config, &record, false).await.unwrap().1,
            Some(false)
        );
        server.abort();
    }
    #[test]
    fn invalid_or_unbounded_templates_are_rejected() {
        assert!(validate_template(&json!({"containers":[]})).is_err());
        assert!(validate_template(&json!({"containers":[{"image":"native:latest"}]})).is_err());
    }
}
