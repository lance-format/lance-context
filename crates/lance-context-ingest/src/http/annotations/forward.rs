use super::*;
use bytes::BytesMut;
use futures::stream;
use http_body_util::BodyExt;
use tokio::time::{timeout_at, Instant};

#[derive(Clone)]
pub struct Forwarder {
    gate: Gateway,
    routes: Arc<BTreeMap<String, reqwest::Url>>,
    client: reqwest::Client,
    token: Arc<String>,
}
impl Forwarder {
    /// Resolve the owner addresses and table bindings from trusted deployment
    /// configuration. Forwarding ingresses have no Lance view or local applier.
    pub fn new(
        tables: Vec<(Ingress, String)>,
        keys: ReceiptKeys,
        options: Options,
        token: String,
    ) -> Result<Self> {
        ensure!(
            (32..=1024).contains(&token.len()) && token.bytes().all(|b| b.is_ascii_graphic()),
            "invalid annotation owner workload token"
        );
        let mut routes = BTreeMap::new();
        let mut ingresses = BTreeMap::new();
        for (ingress, address) in tables {
            ensure!(
                ingress.scope.contract.policy == WritePolicy::TurnAnnotations
                    && ingress.layout.buckets.len() == 1,
                "annotation route requires a single-shard annotation binding"
            );
            let reply = reply_limit(&ingress)?;
            let required = reply
                .checked_add(reply.max(ingress.scope.contract.max_record_bytes as usize))
                .context("annotation forwarding budget overflow")?;
            ensure!(
                required <= ingress.budget.limit(),
                "annotation forwarding exceeds ingress budget"
            );
            let url = reqwest::Url::parse(&address)?;
            ensure!(
                matches!(url.scheme(), "http" | "https")
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.path() == "/"
                    && url.query().is_none()
                    && url.fragment().is_none(),
                "invalid annotation owner address"
            );
            let name = ingress.scope.contract.qualified_table.clone();
            ensure!(
                routes.insert(name.clone(), url).is_none(),
                "duplicate annotation route"
            );
            ingresses.insert(name, ingress);
        }
        let client = reqwest::Client::builder()
            .timeout(options.body_total_timeout)
            .connect_timeout(options.admission_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .tcp_nodelay(true)
            .pool_max_idle_per_host(16)
            .build()?;
        Ok(Self {
            gate: Gateway::new(ingresses, keys, options)?,
            routes: Arc::new(routes),
            client,
            token: Arc::new(token),
        })
    }
    pub fn router(&self) -> Router {
        Router::new()
            .route("/api/v1/annotations/{table}/producer", post(open))
            .route("/api/v1/annotations/{table}/update", post(update))
            .route("/api/v1/annotations/{table}/read", post(read))
            .route("/api/v1/annotations/{table}/close", post(close))
            .with_state(self.clone())
    }
    pub fn observation(&self) -> Observation {
        self.gate.observation()
    }
    pub async fn shutdown(&self) {
        self.gate.shutdown().await;
    }

    async fn dispatch(self, table: String, request: Request, operation: Operation) -> Response {
        let version = request.version();
        let guard = match self.gate.enter(matches!(operation, Operation::Update)) {
            Ok(guard) => guard,
            Err(error) => return early(error, version),
        };
        match tokio::spawn(async move {
            let (response, reservation) = self.process(&table, request, operation).await?;
            Ok::<_, Failure>(hold_response(response, guard, Some(reservation)))
        })
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => early(error, version),
            Err(_) => Failure::unavailable("annotation_forward_failed", "unknown").into_response(),
        }
    }

    async fn process(
        &self,
        table: &str,
        request: Request,
        operation: Operation,
    ) -> HttpResult<(Response, MergeReservation)> {
        let ingress = self.gate.0.tables.get(table).ok_or(Failure {
            status: StatusCode::NOT_FOUND,
            code: "unknown_table",
            outcome: "not_submitted",
        })?;
        let maximum_reply =
            reply_limit(ingress).map_err(|_| Failure::bad("invalid_annotation_scope"))?;
        let (maximum, content_type) = match operation {
            Operation::Update => (ingress.scope.contract.max_record_bytes as usize, ARROW_FILE),
            Operation::Read => (maximum_reply, "application/json"),
            _ => (32 << 10, "application/json"),
        };
        let length = body::length(&request, maximum, content_type)?;
        let options = &self.gate.0.options;
        let reservation = tokio::time::timeout(
            options.admission_timeout,
            ingress.budget.reserve(length + maximum_reply),
        )
        .await
        .map_err(|_| Failure::unavailable("memory_capacity", "not_submitted"))?;
        let mut url = self.routes[table].clone();
        url.path_segments_mut()
            .map_err(|_| Failure::bad("invalid_owner_route"))?
            .extend(["api", "v1", "annotations", table, operation.suffix()]);
        let mut outgoing = self
            .client
            .post(url)
            .bearer_auth(self.token.as_str())
            .header(header::CONTENT_TYPE, content_type)
            .header(header::CONTENT_LENGTH, length);
        if matches!(operation, Operation::Update) {
            // Forward these credentials unchanged. The owner validates them.
            for name in [PRODUCER, SEQUENCE] {
                if request.headers().get_all(name).iter().count() != 1 {
                    return Err(Failure::bad("ambiguous_producer_headers"));
                }
                outgoing = outgoing.header(name, request.headers()[name].clone());
            }
        }
        let deadline = Instant::now() + options.body_total_timeout;
        let idle = options.body_idle_timeout;
        let frames = stream::try_unfold(
            (request.into_body(), length),
            move |(mut body, remaining)| async move {
                loop {
                    let frame = timeout_at(deadline.min(Instant::now() + idle), body.frame())
                        .await
                        .map_err(|_| std::io::Error::other("annotation body deadline"))?;
                    let Some(frame) = frame else {
                        if remaining != 0 {
                            return Err(std::io::Error::other("annotation body truncated"));
                        }
                        return Ok(None);
                    };
                    let bytes = frame
                        .map_err(std::io::Error::other)?
                        .into_data()
                        .map_err(|_| std::io::Error::other("annotation body trailers"))?;
                    if bytes.is_empty() {
                        continue;
                    }
                    if bytes.len() > remaining {
                        return Err(std::io::Error::other("annotation body too long"));
                    }
                    let next = remaining - bytes.len();
                    // Share the original frame. No Arrow decode, re-encode, or
                    // whole-request coalescing occurs in a forwarding ingress.
                    return Ok(Some((bytes, (body, next))));
                }
            },
        );
        // One send only. A transport failure cannot justify replaying a mutation
        // with a new sequence or silently resending an ambiguous request here.
        let mut upstream = outgoing
            .body(reqwest::Body::wrap_stream(frames))
            .send()
            .await
            .map_err(|_| Failure::unavailable("annotation_owner_unavailable", "unknown"))?;
        let status = upstream.status();
        let is_empty = status == StatusCode::NO_CONTENT;
        if !is_empty
            && upstream
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                != Some("application/json")
        {
            return Err(Failure::unavailable("invalid_owner_response", "unknown"));
        }
        if upstream
            .content_length()
            .is_some_and(|n| n > maximum_reply as u64)
        {
            return Err(Failure::unavailable("oversized_owner_response", "unknown"));
        }
        let mut bytes = BytesMut::new();
        while let Some(chunk) = upstream
            .chunk()
            .await
            .map_err(|_| Failure::unavailable("incomplete_owner_response", "unknown"))?
        {
            if chunk.len() > maximum_reply - bytes.len() {
                return Err(Failure::unavailable("oversized_owner_response", "unknown"));
            }
            bytes.extend_from_slice(&chunk);
        }
        let response = if is_empty {
            status.into_response()
        } else {
            (
                status,
                [(header::CONTENT_TYPE, "application/json")],
                bytes.freeze(),
            )
                .into_response()
        };
        Ok((response, reservation))
    }
}

async fn open(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Open).await
}
async fn update(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Update).await
}
async fn read(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Read).await
}
async fn close(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Close).await
}
