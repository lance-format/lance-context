//! Caller admission for actual fetched payload backing. Local RPC limits are
//! independent; this hook neither reserves idle ceilings nor bills native Lance
//! allocations. Uploads and decoder allocations require their own admission.
use super::*;

/// A caller-owned charge. Dropping it returns admission to the caller's pool.
/// It travels with the backing allocation through every Bytes/Arrow clone.
pub trait Reservation: Send + Sync {
    fn bytes(&self) -> usize;
    /// Record allocated capacity, before exposing backing to another consumer.
    fn set_backed_bytes(&mut self, bytes: usize);
}

/// Nonblocking, all-or-nothing admission before a payload allocation. `None`
/// denotes temporary pressure; errors preserve permanent/configuration failures.
/// Do not wait or run a blocking cache reclaimer from this I/O callback.
pub trait Pool: Send + Sync {
    fn try_reserve(&self, bytes: usize) -> Result<Option<Box<dyn Reservation>>>;
}

/// Temporary capacity refusal, before receiving or allocating payload bytes.
/// Callers retaining an apply head can yield until their existing capacity gate
/// reopens. This does not retry or change the identity of a business mutation.
#[derive(Debug)]
pub struct Unavailable;

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WAL payload memory unavailable")
    }
}
impl std::error::Error for Unavailable {}

struct Charged {
    // Drop backing before returning its admission, including on cancellation.
    body: Bytes,
    _reservation: Box<dyn Reservation>,
}
impl AsRef<[u8]> for Charged {
    fn as_ref(&self) -> &[u8] {
        &self.body
    }
}

pub(super) async fn receive<R: AsyncRead + Unpin>(
    io: &mut R,
    bytes: usize,
    memory: Option<&dyn Pool>,
) -> Result<Bytes> {
    Ok(receive_with_capacity(io, bytes, memory).await?.0)
}

pub(super) async fn receive_with_capacity<R: AsyncRead + Unpin>(
    io: &mut R,
    bytes: usize,
    memory: Option<&dyn Pool>,
) -> Result<(Bytes, usize)> {
    if bytes == 0 {
        return Ok((Bytes::new(), 0));
    }
    let mut reservation = match memory {
        Some(memory) => Some(memory.try_reserve(bytes)?.ok_or(Unavailable)?),
        None => None,
    };
    if let Some(reservation) = &mut reservation {
        ensure!(reservation.bytes() >= bytes, "undersized payload admission");
        reservation.set_backed_bytes(0);
    }
    // Declared after the charge so an error drops backing before its charge.
    let mut body = BytesMut::with_capacity(bytes);
    if let Some(reservation) = &mut reservation {
        ensure!(
            body.capacity() <= reservation.bytes(),
            "payload allocation exceeds admission"
        );
        reservation.set_backed_bytes(body.capacity());
    }
    // The limit prevents spare capacity from consuming the next wire frame.
    let mut limited = io.take(bytes as u64);
    while limited.limit() > 0 {
        if limited.read_buf(&mut body).await? == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
    }
    let capacity = body.capacity();
    let body = body.freeze();
    let body = match reservation {
        Some(reservation) => Bytes::from_owner(Charged {
            body,
            _reservation: reservation,
        }),
        None => body,
    };
    Ok((body, capacity))
}
