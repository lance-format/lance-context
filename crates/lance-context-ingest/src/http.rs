//! Bounded HTTP ingress over the public TLS WAL client. Install behind the
//! application's authenticated/authorized routing boundary. Credentials prove
//! producer/receipt identity, not permission for an end user to access a table.
use super::*;
use axum::{
    extract::{Path, Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use receipt::ReceiptKeys;
use std::{collections::BTreeMap, sync::Mutex};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

mod body;
#[cfg(feature = "http-client")]
pub mod journal;
#[cfg(feature = "materializer")]
mod read;

pub const ARROW_FILE: &str = "application/vnd.apache.arrow.file";
pub const PRODUCER: &str = "x-lance-producer";
pub const SEQUENCE: &str = "x-lance-sequence";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub data_requests: usize,
    pub control_requests: usize,
    pub admission_timeout: Duration,
    pub body_idle_timeout: Duration,
    pub body_total_timeout: Duration,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            data_requests: 64,
            control_requests: 16,
            admission_timeout: Duration::from_secs(5),
            body_idle_timeout: Duration::from_secs(10),
            body_total_timeout: Duration::from_secs(60),
        }
    }
}
#[derive(Default)]
struct Lifecycle {
    closed: bool,
    active: usize,
}
struct Inner {
    tables: BTreeMap<String, Ingress>,
    keys: ReceiptKeys,
    options: Options,
    data: Arc<Semaphore>,
    control: Arc<Semaphore>,
    lifecycle: Mutex<Lifecycle>,
    idle: Notify,
}
struct Guard {
    inner: Arc<Inner>,
    slot: Option<OwnedSemaphorePermit>,
}
impl Drop for Guard {
    fn drop(&mut self) {
        let mut lifecycle = self.inner.lifecycle.lock().unwrap();
        self.slot.take();
        lifecycle.active -= 1;
        drop(lifecycle);
        self.inner.idle.notify_waiters();
    }
}
#[derive(Clone)]
pub struct Gateway(Arc<Inner>);
#[derive(Debug, Serialize)]
pub struct Observation {
    pub data_inflight: usize,
    pub control_inflight: usize,
    pub draining: bool,
}
impl Gateway {
    /// Table bindings must be resolved and authorized before construction.
    /// Cloned ingresses should share their physical WAL client and memory budget.
    pub fn new(
        tables: BTreeMap<String, Ingress>,
        keys: ReceiptKeys,
        options: Options,
    ) -> Result<Self> {
        ensure!(
            !tables.is_empty() && tables.len() <= 4096,
            "invalid HTTP table count"
        );
        ensure!(
            tables.iter().all(|(name, table)| !name.is_empty()
                && name.len() <= 1024
                && !name.contains('/')
                && name == &table.scope.contract.qualified_table),
            "invalid HTTP table binding"
        );
        ensure!(
            (1..=256).contains(&options.data_requests)
                && (1..=64).contains(&options.control_requests)
                && (Duration::from_millis(1)..=Duration::from_secs(30))
                    .contains(&options.admission_timeout)
                && (Duration::from_millis(1)..=Duration::from_secs(60))
                    .contains(&options.body_idle_timeout)
                && (options.body_idle_timeout..=Duration::from_secs(300))
                    .contains(&options.body_total_timeout),
            "invalid HTTP admission/deadline limits"
        );
        Ok(Self(Arc::new(Inner {
            tables,
            keys,
            data: Arc::new(Semaphore::new(options.data_requests)),
            control: Arc::new(Semaphore::new(options.control_requests)),
            options,
            lifecycle: Mutex::new(Lifecycle::default()),
            idle: Notify::new(),
        })))
    }
    pub fn router(&self) -> Router {
        Router::new()
            .route("/api/v1/health", get(health))
            .route("/api/v1/ingest/{table}/producer", post(open))
            .route("/api/v1/ingest/{table}/append", post(append))
            .route("/api/v1/ingest/{table}/publication", post(publication))
            .route("/api/v1/ingest/{table}/close", post(close))
            .with_state(self.clone())
    }
    /// Add session reads using the application's resolved table storage.
    #[cfg(feature = "materializer")]
    pub fn router_with_reads(
        &self,
        storage: BTreeMap<String, crate::materializer::Storage>,
    ) -> Router {
        self.router().merge(read::router(self.clone(), storage))
    }
    pub fn observation(&self) -> Observation {
        Observation {
            data_inflight: self.0.options.data_requests - self.0.data.available_permits(),
            control_inflight: self.0.options.control_requests - self.0.control.available_permits(),
            draining: self.0.lifecycle.lock().unwrap().closed,
        }
    }
    /// Stop admission and join every admitted handler, including handlers whose
    /// HTTP client disconnected. Body reads have total/idle deadlines; WAL RPCs
    /// retain their configured exchange deadline. No submitted write is aborted.
    pub async fn shutdown(&self) {
        self.0.lifecycle.lock().unwrap().closed = true;
        loop {
            let idle = self.0.idle.notified();
            tokio::pin!(idle);
            idle.as_mut().enable();
            if self.0.lifecycle.lock().unwrap().active == 0 {
                return;
            }
            idle.await;
        }
    }
    fn enter(&self, data: bool) -> HttpResult<Guard> {
        let mut state = self.0.lifecycle.lock().unwrap();
        if state.closed {
            return Err(Failure::unavailable("draining", "not_submitted"));
        }
        let lane = if data { &self.0.data } else { &self.0.control };
        let slot = lane
            .clone()
            .try_acquire_owned()
            .map_err(|_| Failure::unavailable("request_capacity", "not_submitted"))?;
        state.active += 1;
        Ok(Guard {
            inner: self.0.clone(),
            slot: Some(slot),
        })
    }
}
#[derive(Debug)]
struct Failure {
    status: StatusCode,
    code: &'static str,
    outcome: &'static str,
}
type HttpResult<T> = std::result::Result<T, Failure>;
impl Failure {
    fn bad(code: &'static str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            outcome: "not_submitted",
        }
    }
    fn unavailable(code: &'static str, outcome: &'static str) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code,
            outcome,
        }
    }
}
impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        let mut response = (
            self.status,
            Json(serde_json::json!({
                "error": self.code, "operation_outcome": self.outcome,
                "retry_same_request": true,
            })),
        )
            .into_response();
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, "1".parse().unwrap());
        }
        response
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Opened {
    pub producer: String,
    pub generation: u64,
    pub scope: [u8; 32],
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Accepted {
    pub receipt: String,
    pub rows: u32,
    pub sequence: u64,
    pub group: u64,
    pub index: u64,
    pub digest: [u8; 32],
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptRequest {
    pub receipt: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloseRequest {
    pub producer: String,
    pub receipt: Option<String>,
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Visibility {
    pub minimum_read_version: Option<u64>,
}
#[derive(Clone, Copy)]
enum Action {
    Open,
    Append,
    Publication,
    Close,
}
async fn health(State(gateway): State<Gateway>) -> StatusCode {
    if gateway.0.lifecycle.lock().unwrap().closed {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}
async fn open(State(g): State<Gateway>, Path(table): Path<String>, request: Request) -> Response {
    dispatch(g, table, request, Action::Open).await
}
async fn append(State(g): State<Gateway>, Path(table): Path<String>, request: Request) -> Response {
    dispatch(g, table, request, Action::Append).await
}
async fn publication(
    State(g): State<Gateway>,
    Path(table): Path<String>,
    request: Request,
) -> Response {
    dispatch(g, table, request, Action::Publication).await
}
async fn close(State(g): State<Gateway>, Path(table): Path<String>, request: Request) -> Response {
    dispatch(g, table, request, Action::Close).await
}
async fn dispatch(gateway: Gateway, table: String, request: Request, action: Action) -> Response {
    let guard = match gateway.enter(matches!(action, Action::Append)) {
        Ok(guard) => guard,
        Err(error) => {
            let mut response = error.into_response();
            // Admission rejected this request without consuming its body.
            // Hyper may close HTTP/1 after dropping that unread body. Tell the
            // client before it pools the socket and races a following POST
            // against the close; never drain an unbounded rejected upload.
            if matches!(
                request.version(),
                axum::http::Version::HTTP_10 | axum::http::Version::HTTP_11
            ) {
                response
                    .headers_mut()
                    .insert(header::CONNECTION, "close".parse().unwrap());
            }
            return response;
        }
    };
    // The admitted task owns its guard and body independently of the socket.
    match tokio::spawn(async move {
        let _guard = guard;
        process(&gateway.0, &table, request, action).await
    })
    .await
    {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => error.into_response(),
        Err(_) => Failure::unavailable("handler_failed", "unknown").into_response(),
    }
}
async fn process(
    inner: &Inner,
    table: &str,
    request: Request,
    action: Action,
) -> HttpResult<Response> {
    let ingress = inner.tables.get(table).ok_or(Failure {
        status: StatusCode::NOT_FOUND,
        code: "unknown_table",
        outcome: "not_submitted",
    })?;
    if matches!(action, Action::Append) {
        return append_body(inner, ingress, request).await;
    }
    let data = body::read(request, 32 << 10, "application/json", &inner.options).await?;
    match action {
        Action::Open => {
            let open: ProducerOpen =
                serde_json::from_slice(&data).map_err(|_| Failure::bad("invalid_open"))?;
            let producer = ingress
                .open_producer(open)
                .await
                .map_err(|_| Failure::unavailable("open_outcome_unknown", "unknown"))?;
            let token = inner
                .keys
                .encode_producer(&producer)
                .map_err(|_| Failure::unavailable("credential_unavailable", "unknown"))?;
            let scope = super::receipt::scope_hash(&ingress.scope)
                .map_err(|_| Failure::unavailable("credential_unavailable", "unknown"))?;
            Ok(Json(Opened {
                producer: token,
                generation: producer.generation(),
                scope,
            })
            .into_response())
        }
        Action::Publication => {
            let query: ReceiptRequest =
                serde_json::from_slice(&data).map_err(|_| Failure::bad("invalid_receipt_query"))?;
            let accepted = ingress
                .restore_receipt(&inner.keys, &query.receipt)
                .map_err(|_| Failure::bad("invalid_receipt"))?;
            let status = ingress
                .publication_status(&accepted)
                .await
                .map_err(|_| Failure::unavailable("publication_unavailable", "unknown"))?;
            let version = match status {
                PublicationStatus::Pending => None,
                PublicationStatus::Published {
                    minimum_read_version,
                } => Some(minimum_read_version),
            };
            Ok(Json(Visibility {
                minimum_read_version: version,
            })
            .into_response())
        }
        Action::Close => {
            let close: CloseRequest =
                serde_json::from_slice(&data).map_err(|_| Failure::bad("invalid_close"))?;
            let producer = ingress
                .restore_producer(&inner.keys, &close.producer)
                .map_err(|_| Failure::bad("invalid_producer"))?;
            let receipt = close
                .receipt
                .as_ref()
                .map(|r| ingress.restore_receipt(&inner.keys, r))
                .transpose()
                .map_err(|_| Failure::bad("invalid_receipt"))?;
            ingress
                .close_producer(&producer, receipt.as_ref())
                .await
                .map_err(|_| Failure::unavailable("close_outcome_unknown", "unknown"))?;
            Ok(StatusCode::NO_CONTENT.into_response())
        }
        Action::Append => unreachable!(),
    }
}
async fn append_body(inner: &Inner, ingress: &Ingress, request: Request) -> HttpResult<Response> {
    if request.headers().get_all(PRODUCER).iter().count() != 1
        || request.headers().get_all(SEQUENCE).iter().count() != 1
    {
        return Err(Failure::bad("ambiguous_producer_headers"));
    }
    let producer = request
        .headers()
        .get(PRODUCER)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| Failure::bad("missing_producer"))?;
    let producer = ingress
        .restore_producer(&inner.keys, producer)
        .map_err(|_| Failure::bad("invalid_producer"))?;
    let sequence = request
        .headers()
        .get(SEQUENCE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0 && *v < u64::MAX)
        .ok_or_else(|| Failure::bad("invalid_sequence"))?;
    let length = body::length(
        &request,
        ingress.scope.contract.max_record_bytes as usize,
        ARROW_FILE,
    )?;
    let upload = tokio::time::timeout(
        inner.options.admission_timeout,
        ingress.reserve_upload(length),
    )
    .await
    .map_err(|_| Failure::unavailable("memory_capacity", "not_submitted"))?
    .map_err(|_| Failure::bad("invalid_upload_size"))?;
    let bytes = body::read_exact(request, length, &inner.options).await?;
    let prepared = ingress
        .prepare(upload, bytes, &producer, sequence)
        .await
        .map_err(|_| Failure::bad("invalid_rollout_ipc"))?;
    let accepted = ingress.append(&prepared).await.map_err(|error| {
        match error.downcast_ref::<AppendFailure>() {
            Some(AppendFailure::Sequence { .. } | AppendFailure::ProducerFenced) => Failure {
                status: StatusCode::CONFLICT,
                code: "producer_sequence_rejected",
                outcome: "rejected",
            },
            _ => Failure::unavailable("append_outcome_unknown", "unknown"),
        }
    })?;
    let receipt = inner
        .keys
        .encode(&accepted)
        .map_err(|_| Failure::unavailable("receipt_unavailable", "unknown"))?;
    Ok((
        StatusCode::ACCEPTED,
        Json(Accepted {
            receipt,
            rows: accepted.rows(),
            sequence,
            group: accepted.position().group,
            index: accepted.position().index,
            digest: accepted.meta().digest,
        }),
    )
        .into_response())
}
