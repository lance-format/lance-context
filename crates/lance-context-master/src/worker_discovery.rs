//! Fresh Kubernetes Service membership for registered-store worker RPCs.
//!
//! Publication must not use a watch cache or the load-balanced Service address:
//! the former can omit a newly admitted writer and the latter reaches one worker.
use std::{collections::BTreeSet, net::IpAddr, path::Path, time::Duration};

use reqwest::Client;
use serde::Deserialize;

const SERVICE_ACCOUNT: &str = "/var/run/secrets/kubernetes.io/serviceaccount";

#[derive(Clone, Debug, clap::Args)]
pub struct WorkerDiscoveryConfig {
    /// Kubernetes Service whose EndpointSlices identify registered-store workers.
    /// Requires MANAGED_STORE_KIND; overrides WORKER_ENDPOINTS for that kind only.
    #[arg(long, env = "MANAGED_WORKER_SERVICE", requires = "managed_store_kind")]
    pub managed_worker_service: Option<String>,
    /// Worker Service namespace; defaults to the master's service-account namespace.
    #[arg(
        long,
        env = "MANAGED_WORKER_NAMESPACE",
        requires = "managed_worker_service"
    )]
    pub managed_worker_namespace: Option<String>,
    /// Named TCP endpoint port exposed by the worker Service.
    #[arg(long, env = "MANAGED_WORKER_PORT_NAME", default_value = "http")]
    pub managed_worker_port_name: String,
}

impl Default for WorkerDiscoveryConfig {
    fn default() -> Self {
        Self {
            managed_worker_service: None,
            managed_worker_namespace: None,
            managed_worker_port_name: "http".into(),
        }
    }
}

pub(crate) struct KubernetesWorkers {
    client: Client,
    url: String,
    token_file: std::path::PathBuf,
    service: String,
    port_name: String,
}

/// Include Pod identity as well as address so replacement during a barrier fails.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Worker {
    url: String,
    pod_uid: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Membership(BTreeSet<Worker>);

impl Membership {
    pub(crate) fn endpoints(&self) -> Vec<String> {
        self.0.iter().map(|worker| worker.url.clone()).collect()
    }
}

#[derive(Deserialize)]
struct SliceList {
    metadata: ListMetadata,
    items: Vec<Slice>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListMetadata {
    resource_version: String,
    #[serde(default, rename = "continue")]
    next: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Slice {
    address_type: String,
    endpoints: Vec<Endpoint>,
    ports: Vec<Port>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Endpoint {
    addresses: Vec<IpAddr>,
    target_ref: Option<Target>,
    // Deliberately do not filter readiness/serving/terminating: a withdrawn
    // endpoint may still hold acknowledged, unpublished writes.
}

#[derive(Deserialize)]
struct Target {
    uid: Option<String>,
}

#[derive(Deserialize)]
struct Port {
    name: Option<String>,
    protocol: Option<String>,
    port: Option<u16>,
}

impl KubernetesWorkers {
    pub(crate) fn in_cluster(config: &WorkerDiscoveryConfig) -> Result<Option<Self>, String> {
        let Some(service) = &config.managed_worker_service else {
            return Ok(None);
        };
        let directory = Path::new(SERVICE_ACCOUNT);
        let namespace = match &config.managed_worker_namespace {
            Some(namespace) => namespace.clone(),
            None => std::fs::read_to_string(directory.join("namespace"))
                .map_err(|error| format!("read worker discovery namespace: {error}"))?
                .trim()
                .to_string(),
        };
        for (label, value) in [("Service", service.as_str()), ("namespace", &namespace)] {
            if value.is_empty()
                || value.len() > 63
                || !value
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
                || value.starts_with('-')
                || value.ends_with('-')
            {
                return Err(format!("invalid worker discovery {label}"));
            }
        }
        if config.managed_worker_port_name.is_empty() {
            return Err("worker discovery requires a named endpoint port".into());
        }
        let ca = std::fs::read(directory.join("ca.crt"))
            .map_err(|error| format!("read Kubernetes CA: {error}"))?;
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .add_root_certificate(reqwest::Certificate::from_pem(&ca).map_err(|e| e.to_string())?)
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Some(Self {
            client,
            url: format!("https://kubernetes.default.svc/apis/discovery.k8s.io/v1/namespaces/{namespace}/endpointslices"),
            token_file: directory.join("token"),
            service: service.clone(),
            port_name: config.managed_worker_port_name.clone(),
        }))
    }

    pub(crate) async fn snapshot(&self) -> Result<Membership, String> {
        let mut workers = BTreeSet::new();
        let mut next = String::new();
        let mut revision = None;
        loop {
            // Projected service-account tokens rotate while a master is running.
            let token = std::fs::read_to_string(&self.token_file)
                .map_err(|error| format!("read Kubernetes token: {error}"))?;
            // Omit resourceVersion: Kubernetes' Most Recent read semantics are
            // required here. In particular, never request resourceVersion=0.
            let response = self
                .client
                .get(&self.url)
                .bearer_auth(token.trim())
                .query(&[
                    (
                        "labelSelector",
                        format!("kubernetes.io/service-name={}", self.service),
                    ),
                    ("limit", "500".to_string()),
                    ("continue", next.clone()),
                ])
                .send()
                .await
                .map_err(|error| format!("worker discovery request failed: {error}"))?;
            if response.status() != reqwest::StatusCode::OK {
                return Err(format!(
                    "worker discovery returned HTTP {}",
                    response.status()
                ));
            }
            let page: SliceList = response
                .json()
                .await
                .map_err(|error| format!("invalid worker EndpointSlice response: {error}"))?;
            if page.metadata.resource_version.is_empty()
                || revision
                    .as_ref()
                    .is_some_and(|r| r != &page.metadata.resource_version)
            {
                return Err("worker discovery changed revision during pagination".into());
            }
            revision = Some(page.metadata.resource_version);
            for slice in page.items {
                if slice.endpoints.is_empty() {
                    continue;
                }
                if !matches!(slice.address_type.as_str(), "IPv4" | "IPv6") {
                    return Err("worker discovery requires IP EndpointSlices".into());
                }
                let mut ports = slice.ports.iter().filter(|p| {
                    p.name.as_deref() == Some(&self.port_name)
                        && p.protocol.as_deref().unwrap_or("TCP") == "TCP"
                });
                let port = ports
                    .next()
                    .and_then(|p| p.port)
                    .filter(|p| *p != 0)
                    .ok_or("worker EndpointSlice is missing its named TCP port")?;
                if ports.next().is_some() {
                    return Err("worker EndpointSlice has duplicate named ports".into());
                }
                for endpoint in slice.endpoints {
                    if endpoint.addresses.is_empty() {
                        return Err(
                            "worker EndpointSlice contains an endpoint without addresses".into(),
                        );
                    }
                    for address in endpoint.addresses {
                        workers.insert(Worker {
                            url: format!("http://{}", std::net::SocketAddr::new(address, port)),
                            pod_uid: endpoint.target_ref.as_ref().and_then(|r| r.uid.clone()),
                        });
                    }
                }
            }
            if page.metadata.next.is_empty() {
                break;
            }
            if page.metadata.next == next {
                return Err("worker discovery repeated its pagination token".into());
            }
            next = page.metadata.next;
        }
        if workers.is_empty() {
            return Err("worker discovery returned no endpoints".into());
        }
        Ok(Membership(workers))
    }
}

#[cfg(test)]
mod tests;
