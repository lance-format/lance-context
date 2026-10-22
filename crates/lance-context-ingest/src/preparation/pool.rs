use super::*;
use bytes::BytesMut;

/// A bounded endpoint set. Authentication applies to the deployment's configured
/// table registry; put the HTTP service behind TLS outside a trusted local test.
pub struct Pool {
    client: reqwest::Client,
    endpoints: Vec<reqwest::Url>,
    token: String,
    timeout: Duration,
}
impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparationPool")
            .field("endpoints", &self.endpoints.len())
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}
impl Pool {
    pub fn new(endpoints: &[String], token: String, timeout: Duration) -> Result<Arc<Self>> {
        ensure!(
            (1..=8).contains(&endpoints.len())
                && (32..=1024).contains(&token.len())
                && token.bytes().all(|b| b.is_ascii_graphic())
                && (Duration::from_secs(1)..=Duration::from_secs(300)).contains(&timeout),
            "invalid remote preparation bounds"
        );
        let mut unique = HashSet::new();
        let endpoints = endpoints
            .iter()
            .map(|url| {
                let mut url = reqwest::Url::parse(url)?;
                ensure!(
                    matches!(url.scheme(), "http" | "https")
                        && url.host_str().is_some()
                        && url.username().is_empty()
                        && url.password().is_none()
                        && url.path() == "/"
                        && url.query().is_none()
                        && url.fragment().is_none()
                        && unique.insert(url.to_string()),
                    "invalid or duplicate preparer endpoint"
                );
                url.set_path(ROUTE);
                Ok(url)
            })
            .collect::<Result<Vec<_>>>()?;
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .tcp_nodelay(true)
            .pool_max_idle_per_host(8)
            .build()?;
        Ok(Arc::new(Self {
            client,
            endpoints,
            token,
            timeout,
        }))
    }
    pub(crate) async fn prepare(
        self: &Arc<Self>,
        ingress: &Ingress,
        pending: Arc<p::PendingPlan>,
        budget: Arc<MergeMemoryBudget>,
    ) -> Result<p::Descriptors> {
        let count: usize = pending.plan.sources.values().map(|s| s.records.len()).sum();
        let parts = self.endpoints.len().min(count);
        ensure!(parts > 0, "empty remote preparation");
        let maximum = ingress.client.max_descriptor_bytes();
        let workspace = (parts + 1)
            .checked_mul(4 * maximum as usize)
            .context("descriptor workspace overflow")?;
        ensure!(
            workspace <= budget.limit(),
            "preparation metadata exceeds publisher budget"
        );
        let reservation = budget.reserve(workspace).await;
        let tasks = (0..parts)
            .map(|part| Task::new(ingress, &pending, part, parts))
            .collect::<Vec<_>>();
        let pool = self.clone();
        // Cancellation only drops the observer. Join every admitted request;
        // remote jobs own separate source reservations and also survive disconnect.
        tokio::spawn(async move {
            let _reservation = reservation;
            let results =
                futures::future::join_all(tasks.iter().map(|task| pool.request(task, maximum)))
                    .await;
            let complete = results.into_iter().collect::<Result<Vec<_>>>()?;
            ensure!(complete.len() == parts, "incomplete preparation round");
            let mut paths = HashSet::new();
            let mut all = Vec::new();
            for result in complete {
                for fragment in fragments(&result, maximum)? {
                    for file in &fragment.files {
                        ensure!(
                            paths.insert(file.path.clone()),
                            "duplicate prepared file across partitions"
                        );
                    }
                    all.push(fragment);
                }
            }
            let result = p::Descriptors::lance_fragments(Bytes::from(serde_json::to_vec(&all)?))?;
            result.validate(maximum)?;
            Ok(result)
        })
        .await?
    }
    async fn request(&self, task: &Task, maximum: u64) -> Result<p::Descriptors> {
        let mut response = self
            .client
            .post(self.endpoints[task.part].clone())
            .bearer_auth(&self.token)
            .json(task)
            .send()
            .await?
            .error_for_status()?;
        ensure!(
            response
                .headers()
                .get(IDENTITY)
                .and_then(|v| v.to_str().ok())
                == Some(task.identity()?.as_str())
                && response
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    == Some(CONTENT_TYPE),
            "foreign preparation response"
        );
        ensure!(
            response.content_length().is_none_or(|n| n <= maximum),
            "oversized preparation response"
        );
        let mut bytes = BytesMut::new();
        while let Some(chunk) = response.chunk().await? {
            ensure!(
                chunk.len() <= maximum as usize - bytes.len(),
                "oversized preparation response"
            );
            bytes.extend_from_slice(&chunk);
        }
        p::Descriptors::lance_fragments(bytes.freeze())
    }
}
