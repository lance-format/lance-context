//! Full original-schema point reads through the existing owner and ingress.
use super::*;
use std::io::Write;

pub(super) fn reply_limit(ingress: &Ingress) -> Result<usize> {
    (ingress.scope.contract.max_record_bytes as usize)
        .checked_mul(3)
        .context("full body response limit overflow")
}

/// Bound the encoded backing, including Vec growth, before it reaches HTTP.
/// Arrow encoder workspace remains separate from this output reservation.
struct Output {
    bytes: Vec<u8>,
    limit: usize,
    reservation: MergeReservation,
}

impl Write for Output {
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        let length = self
            .bytes
            .len()
            .checked_add(input.len())
            .filter(|&length| length <= self.limit)
            .ok_or_else(|| std::io::Error::other("full body response exceeds bound"))?;
        if length > self.bytes.capacity() {
            let capacity = length
                .checked_next_power_of_two()
                .unwrap_or(self.limit)
                .min(self.limit);
            let peak = capacity
                .checked_add(self.bytes.capacity())
                .ok_or_else(|| std::io::Error::other("body response growth overflow"))?;
            if peak > self.limit || !self.reservation.try_grow_to(peak) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "body response memory unavailable",
                ));
            }
            self.bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .map_err(std::io::Error::other)?;
            self.reservation.shrink_to(self.bytes.capacity());
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) async fn read(
    service: &Service,
    point: PointRequest,
    recovery_cut: u64,
    deadline: Duration,
) -> HttpResult<(Response, Option<MergeReservation>)> {
    if !service.is_single_body() {
        return Err(Failure::bad("not_full_body_table"));
    }
    let value = service
        .get_full_body_owned(&point.turn, recovery_cut, deadline)
        .await
        .map_err(|_| Failure::unavailable("full_body_read_unavailable", "not_submitted"))?;
    let Some(value) = value else {
        return Ok((
            Failure {
                status: StatusCode::NOT_FOUND,
                code: "turn_not_found",
                outcome: "not_submitted",
            }
            .into_response(),
            None,
        ));
    };
    let limit = reply_limit(service.ingress())
        .map_err(|_| Failure::bad("invalid_body_scope"))?
        .min(service.ingress().budget.limit());
    // Never wait for more memory while holding a staged frame or cold-read
    // reservation. Capacity pressure leaves writes and publication free to run.
    let reservation = service
        .ingress()
        .budget
        .try_reserve(0)
        .ok_or_else(|| Failure::unavailable("body_response_capacity", "not_submitted"))?;
    let (bytes, reservation) = tokio::task::spawn_blocking(move || {
        let output = Output {
            bytes: Vec::new(),
            limit,
            reservation,
        };
        let mut writer = arrow_ipc::writer::FileWriter::try_new(output, &value.batch().schema())?;
        writer.write(value.batch())?;
        writer.finish()?;
        let output = writer.into_inner()?;
        let bytes = Bytes::from(output.bytes);
        // The captured view and its Arrow backing live through encoding, even
        // if the HTTP caller disconnects while this blocking task is running.
        drop(value);
        Ok::<_, anyhow::Error>((bytes, output.reservation))
    })
    .await
    .map_err(|_| Failure::unavailable("body_encoding_failed", "not_submitted"))?
    .map_err(|error| {
        let pressured = error.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::WouldBlock)
        });
        Failure::unavailable(
            if pressured {
                "body_response_capacity"
            } else {
                "body_encoding_failed"
            },
            "not_submitted",
        )
    })?;
    Ok((
        ([(header::CONTENT_TYPE, ARROW_FILE)], bytes).into_response(),
        Some(reservation),
    ))
}
