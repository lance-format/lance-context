use super::*;
use bytes::BytesMut;
use futures::stream;
use http_body_util::BodyExt;
use tokio::time::{timeout_at, Instant};

mod routes;
use routes::Routes;

#[derive(Clone)]
pub struct Forwarder {
    gate: Gateway,
    routes: Arc<BTreeMap<String, Routes>>,
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
            !tables.is_empty() && tables.len() <= 4096,
            "invalid annotation route count"
        );
        ensure!(
            (32..=1024).contains(&token.len()) && token.bytes().all(|b| b.is_ascii_graphic()),
            "invalid annotation owner workload token"
        );
        let mut routes = BTreeMap::<String, Routes>::new();
        let mut ingresses = BTreeMap::new();
        for (ingress, address) in tables {
            ensure!(
                matches!(
                    ingress.scope.contract.policy,
                    WritePolicy::TurnAnnotations | WritePolicy::TurnIngestion
                ),
                "annotation route requires an annotation binding"
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
            if let Some(existing) = routes.get_mut(&name) {
                existing.insert(ingress, url)?;
            } else {
                ingresses.insert(name.clone(), ingress.clone());
                routes.insert(name, Routes::new(ingress, url)?);
            }
        }
        for route in routes.values() {
            route.validate()?;
        }
        let client = reqwest::Client::builder()
            // An owner can reject before consuming a streamed upload. HTTP/2
            // cancels that stream without closing a connection while another
            // request body is being written. Owners accept HTTP/1 and HTTP/2;
            // forwarding ingresses require an HTTP/2-capable owner endpoint.
            .http2_prior_knowledge()
            .retry(reqwest::retry::never())
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
            .route("/api/v1/health", get(health))
            .route("/api/v1/annotations/{table}/producer", post(open))
            .route(
                "/api/v1/annotations/{table}/replace-producer",
                post(replace),
            )
            .route("/api/v1/annotations/{table}/update", post(update))
            .route("/api/v1/annotations/{table}/applied", post(applied))
            .route("/api/v1/annotations/{table}/read", post(read))
            .route("/api/v1/annotations/{table}/read-body", post(read_body))
            .route(
                "/api/v1/annotations/{table}/selected-body",
                post(selected_body),
            )
            .route(
                "/api/v1/annotations/{table}/filter-status",
                post(filter_status),
            )
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
        if matches!(operation, Operation::ReadBody) {
            return match self.process(&table, request, operation).await {
                Ok((response, reservation)) => hold_response(response, guard, Some(reservation)),
                Err(error) => early(error, version),
            };
        }
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
        let control_limit =
            reply_limit(ingress).map_err(|_| Failure::bad("invalid_annotation_scope"))?;
        let maximum_reply = if matches!(operation, Operation::ReadBody) {
            body_read::reply_limit(ingress).map_err(|_| Failure::bad("invalid_body_scope"))?
        } else {
            control_limit
        };
        let (maximum, content_type) = match operation {
            Operation::Update => (ingress.scope.contract.max_record_bytes as usize, ARROW_FILE),
            Operation::Read | Operation::ReadBody | Operation::FilterStatus => {
                (control_limit, "application/json")
            }
            _ => (32 << 10, "application/json"),
        };
        let length = body::length(&request, maximum, content_type)?;
        let options = &self.gate.0.options;
        if matches!(operation, Operation::ReadBody)
            && length
                .checked_add(maximum_reply)
                .is_none_or(|bytes| bytes > ingress.budget.limit())
        {
            return Err(Failure::unavailable("body_read_capacity", "not_submitted"));
        }
        let reservation = if matches!(operation, Operation::FilterStatus | Operation::ReadBody) {
            // Optional filtering must not queue a reservation in front of
            // ingestion at either the forwarding ingress or the owner.
            ingress
                .budget
                .try_reserve(length + maximum_reply)
                .ok_or_else(|| {
                    if matches!(operation, Operation::FilterStatus) {
                        self.gate.status_memory_refusal()
                    } else {
                        Failure::unavailable("body_read_capacity", "not_submitted")
                    }
                })?
        } else {
            tokio::time::timeout(
                options.admission_timeout,
                ingress.budget.reserve(length + maximum_reply),
            )
            .await
            .map_err(|_| {
                if matches!(operation, Operation::Update) {
                    self.gate.write_memory_refusal()
                } else {
                    Failure::unavailable("memory_capacity", "not_submitted")
                }
            })?
        };
        let deadline = Instant::now() + options.body_total_timeout;
        let (request, mut url) = self.routes[table]
            .select(request, operation, length, &self.gate.0.keys, options)
            .await?;
        url.path_segments_mut()
            .map_err(|_| Failure::bad("invalid_owner_route"))?
            .extend(["api", "v1", "annotations", table, operation.suffix()]);
        let mut outgoing = self
            .client
            .post(url)
            .bearer_auth(self.token.as_str())
            .header(header::CONTENT_TYPE, content_type)
            .header(header::CONTENT_LENGTH, length)
            .timeout(deadline.saturating_duration_since(Instant::now()));
        if matches!(operation, Operation::Update) {
            // Forward these credentials unchanged. The owner validates them.
            for name in [PRODUCER, SEQUENCE] {
                if request.headers().get_all(name).iter().count() != 1 {
                    return Err(Failure::bad("ambiguous_producer_headers"));
                }
                outgoing = outgoing.header(name, request.headers()[name].clone());
            }
            // Optional ack mode: forward at most one value unchanged.
            match request.headers().get_all(ACK_MODE).iter().count() {
                0 => {}
                1 => outgoing = outgoing.header(ACK_MODE, request.headers()[ACK_MODE].clone()),
                _ => return Err(Failure::bad("ambiguous_ack_mode")),
            }
        }
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
        let retry_after = upstream.headers().get(header::RETRY_AFTER).cloned();
        let is_empty = status == StatusCode::NO_CONTENT;
        let arrow_body = matches!(operation, Operation::ReadBody) && status == StatusCode::OK;
        let response_type = if arrow_body {
            ARROW_FILE
        } else {
            "application/json"
        };
        if !is_empty
            && upstream
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                != Some(response_type)
        {
            return Err(Failure::unavailable("invalid_owner_response", "unknown"));
        }
        if upstream
            .content_length()
            .is_some_and(|n| n > maximum_reply as u64)
        {
            return Err(Failure::unavailable("oversized_owner_response", "unknown"));
        }
        if arrow_body {
            let length = upstream
                .content_length()
                .filter(|length| *length > 0)
                .ok_or_else(|| Failure::unavailable("invalid_owner_response", "unknown"))?;
            // Share the owner's Arrow chunks, with no decode or whole-response
            // coalescing. A truncated/oversized stream remains an error.
            let frames = stream::try_unfold(
                (upstream, length),
                move |(mut upstream, remaining)| async move {
                    let chunk = timeout_at(deadline.min(Instant::now() + idle), upstream.chunk())
                        .await
                        .map_err(|_| std::io::Error::other("body response deadline"))?
                        .map_err(std::io::Error::other)?;
                    match chunk {
                        Some(chunk) if chunk.len() as u64 <= remaining => {
                            let remaining = remaining - chunk.len() as u64;
                            Ok(Some((chunk, (upstream, remaining))))
                        }
                        Some(_) => Err(std::io::Error::other("body response too long")),
                        None if remaining == 0 => Ok(None),
                        None => Err(std::io::Error::other("body response truncated")),
                    }
                },
            );
            let mut response = (
                [(header::CONTENT_TYPE, ARROW_FILE)],
                axum::body::Body::from_stream(frames),
            )
                .into_response();
            response
                .headers_mut()
                .insert(header::CONTENT_LENGTH, length.into());
            return Ok((response, reservation));
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
        let mut response = if is_empty {
            status.into_response()
        } else {
            (
                status,
                [(header::CONTENT_TYPE, "application/json")],
                bytes.freeze(),
            )
                .into_response()
        };
        if let Some(delay) = retry_after {
            response.headers_mut().insert(header::RETRY_AFTER, delay);
        }
        Ok((response, reservation))
    }
}

async fn health(State(g): State<Forwarder>) -> StatusCode {
    super::super::health(State(g.gate)).await
}

async fn open(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Open).await
}
async fn replace(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Replace).await
}
async fn update(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Update).await
}
async fn applied(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Applied).await
}
async fn read(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Read).await
}
async fn read_body(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::ReadBody).await
}
async fn close(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::Close).await
}
async fn filter_status(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::FilterStatus).await
}
async fn selected_body(State(g): State<Forwarder>, Path(t): Path<String>, r: Request) -> Response {
    g.dispatch(t, r, Operation::SelectedBody).await
}
