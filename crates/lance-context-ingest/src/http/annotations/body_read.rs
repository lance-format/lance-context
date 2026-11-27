//! One original-schema row from the owner's compatible applied view.
use super::*;
use crate::annotations::single_body::Body as HeldBody;
use std::io::{self, Write};

pub(super) fn reply_limit(ingress: &Ingress) -> Result<usize> {
    usize::try_from(ingress.scope.contract.max_record_bytes)?
        .checked_add(64 << 10)
        .context("full body response limit overflow")
}

pub(super) async fn read(
    service: &Service,
    query: PointRequest,
    recovery_cut: u64,
    deadline: std::time::Duration,
) -> HttpResult<Response> {
    if !service.is_single_body() {
        return Err(Failure::bad("not_full_body_table"));
    }
    service
        .validate_turn(&query.turn)
        .map_err(|_| Failure::bad("invalid_body_key"))?;
    let ingress = service.ingress();
    let maximum = reply_limit(ingress).map_err(|_| Failure::bad("invalid_body_scope"))?;
    // Reserve scan/encoder workspace and final output in one nonwaiting step.
    // The returned body retains this charge through actual HTTP consumption.
    let workspace = (ingress.scope.contract.max_record_bytes as usize)
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(maximum))
        .ok_or_else(|| Failure::bad("invalid_body_scope"))?;
    // MergeMemoryBudget deliberately admits oversized merge work alone. A
    // point response instead fails closed at its configured resource limit.
    if workspace > ingress.budget.limit() {
        return Err(Failure::unavailable("body_read_capacity", "not_submitted"));
    }
    let held = ingress
        .budget
        .try_reserve(workspace)
        .ok_or_else(|| Failure::unavailable("body_read_capacity", "not_submitted"))?;
    let body = service
        .get_full_body_owned(&query.turn, recovery_cut, deadline, held)
        .await
        .map_err(|_| Failure::unavailable("body_read_unavailable", "not_submitted"))?;
    let Some(body) = body else {
        return Ok(StatusCode::NO_CONTENT.into_response());
    };
    let bytes = encode(body, maximum)
        .map_err(|_| Failure::unavailable("body_encoding_failed", "not_submitted"))?;
    let length = bytes.len();
    let mut response = ([(header::CONTENT_TYPE, ARROW_FILE)], bytes).into_response();
    // The guard wrapper turns the bytes into a stream, so set the bounded
    // length explicitly before its exact size hint is lost.
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, length.into());
    Ok(response)
}

fn encode(body: HeldBody, maximum: usize) -> Result<Bytes> {
    struct Output {
        bytes: Vec<u8>,
        maximum: usize,
    }
    impl Write for Output {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.maximum - self.bytes.len() {
                return Err(io::Error::other("encoded body exceeds admitted response"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    struct Held {
        bytes: Vec<u8>,
        _body: HeldBody,
    }
    impl AsRef<[u8]> for Held {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }
    let mut output = Output {
        bytes: Vec::new(),
        maximum,
    };
    // Allocate once after admission; no geometric growth or oversized output
    // can temporarily escape the response reservation.
    output.bytes.try_reserve_exact(maximum)?;
    {
        let mut writer =
            arrow_ipc::writer::FileWriter::try_new(&mut output, &body.batch().schema())?;
        writer.write(body.batch())?;
        writer.finish()?;
    }
    Ok(Bytes::from_owner(Held {
        bytes: output.bytes,
        _body: body,
    }))
}
