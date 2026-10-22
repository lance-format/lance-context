use super::*;
use bytes::BytesMut;
use http_body_util::BodyExt;
use tokio::time::{timeout_at, Instant};

pub(super) fn length(request: &Request, maximum: usize, content_type: &str) -> HttpResult<usize> {
    let headers = request.headers();
    if headers.contains_key(header::TRANSFER_ENCODING)
        || headers.get_all(header::CONTENT_LENGTH).iter().count() != 1
        || headers.get_all(header::CONTENT_TYPE).iter().count() != 1
        || headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            != Some(content_type)
    {
        return Err(Failure::bad("invalid_body_headers"));
    }
    headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0 && *n <= maximum)
        .ok_or_else(|| Failure::bad("invalid_content_length"))
}
pub(super) async fn read(
    request: Request,
    maximum: usize,
    content_type: &str,
    options: &Options,
) -> HttpResult<Bytes> {
    let length = length(&request, maximum, content_type)?;
    read_exact(request, length, options).await
}
pub(super) async fn read_exact(
    request: Request,
    length: usize,
    options: &Options,
) -> HttpResult<Bytes> {
    let deadline = Instant::now() + options.body_total_timeout;
    let mut body = request.into_body();
    let mut first: Option<Bytes> = None;
    let mut combined: Option<BytesMut> = None;
    let mut received = 0usize;
    loop {
        let frame = timeout_at(
            deadline.min(Instant::now() + options.body_idle_timeout),
            body.frame(),
        )
        .await
        .map_err(|_| Failure::bad("body_deadline"))?;
        let Some(frame) = frame else {
            break;
        };
        let frame = frame.map_err(|_| Failure::bad("body_disconnected"))?;
        let data = frame
            .into_data()
            .map_err(|_| Failure::bad("unexpected_body_trailers"))?;
        if data.is_empty() {
            continue;
        }
        if data.len() > length - received {
            return Err(Failure::bad("body_length_mismatch"));
        }
        received += data.len();
        if let Some(buffer) = combined.as_mut() {
            buffer.extend_from_slice(&data);
        } else if let Some(previous) = first.take() {
            let mut buffer = BytesMut::with_capacity(length);
            buffer.extend_from_slice(&previous);
            buffer.extend_from_slice(&data);
            combined = Some(buffer);
        } else {
            first = Some(data);
        }
    }
    if received != length {
        return Err(Failure::bad("body_length_mismatch"));
    }
    // A single HTTP frame retains its existing buffer. Fragmented bodies are
    // coalesced once; subsequent validation and WAL transport share these Bytes.
    Ok(combined.map(BytesMut::freeze).or(first).unwrap_or_default())
}
