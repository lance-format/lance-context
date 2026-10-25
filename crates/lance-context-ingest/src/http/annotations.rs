//! Narrow annotation RPCs and opaque forwarding to a deployment-authorized owner.
//! Install both routers behind the application's authentication boundary. The
//! fixed owner address can be a Kubernetes Service; no client supplies a URI.
use super::*;
use crate::annotations::{runtime::Runtime, Completed, MutationResult, Service};
use lance_context_wal::replication::publication as p;

#[cfg(feature = "service")]
mod forward;
mod owner;
#[cfg(feature = "service")]
pub use forward::Forwarder;
pub use owner::Owner;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationReply {
    pub receipt: String,
    pub completion: Completed,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointRequest {
    pub turn: String,
}

#[derive(Clone, Copy)]
enum Operation {
    Open,
    Update,
    Read,
    Close,
}
impl Operation {
    #[cfg(feature = "service")]
    fn suffix(self) -> &'static str {
        match self {
            Self::Open => "producer",
            Self::Update => "update",
            Self::Read => "read",
            Self::Close => "close",
        }
    }
    fn is_write(self) -> bool {
        !matches!(self, Self::Read)
    }
}

// Keep admission (and a forwarding reservation) with the actual outgoing Bytes,
// including bytes buffered by HTTP after the handler has returned.
fn hold_response(
    response: Response,
    guard: Guard,
    reservation: Option<MergeReservation>,
) -> Response {
    use futures::TryStreamExt;
    struct Held {
        bytes: Bytes,
        _guards: Arc<(Guard, Option<MergeReservation>)>,
    }
    impl AsRef<[u8]> for Held {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }
    let guards = Arc::new((guard, reservation));
    let (parts, body) = response.into_parts();
    let stream = body.into_data_stream().map_ok(move |bytes| {
        Bytes::from_owner(Held {
            bytes,
            _guards: guards.clone(),
        })
    });
    Response::from_parts(parts, axum::body::Body::from_stream(stream))
}

fn reply_limit(ingress: &Ingress) -> Result<usize> {
    ingress
        .schema
        .metadata()
        .get("lance-context.annotation-value-bytes")
        .context("annotation value bound missing")?
        .parse::<usize>()?
        .checked_mul(6)
        .and_then(|n| n.checked_add(64 << 10))
        .context("annotation response bound overflow")
}

fn early(error: Failure, version: axum::http::Version) -> Response {
    let mut response = error.into_response();
    if matches!(
        version,
        axum::http::Version::HTTP_10 | axum::http::Version::HTTP_11
    ) {
        response
            .headers_mut()
            .insert(header::CONNECTION, "close".parse().unwrap());
    }
    response
}

fn producer(
    ingress: &Ingress,
    keys: &ReceiptKeys,
    request: &Request,
) -> HttpResult<(Producer, u64)> {
    if request.headers().get_all(PRODUCER).iter().count() != 1
        || request.headers().get_all(SEQUENCE).iter().count() != 1
    {
        return Err(Failure::bad("ambiguous_producer_headers"));
    }
    let token = request
        .headers()
        .get(PRODUCER)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| Failure::bad("missing_producer"))?;
    let producer = ingress
        .restore_producer(keys, token)
        .map_err(|_| Failure::bad("invalid_producer"))?;
    let sequence = request
        .headers()
        .get(SEQUENCE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0 && *v < u64::MAX)
        .ok_or_else(|| Failure::bad("invalid_sequence"))?;
    Ok((producer, sequence))
}
