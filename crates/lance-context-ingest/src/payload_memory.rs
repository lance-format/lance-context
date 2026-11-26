//! Admission follows retained upload backing, including local apply reuse.
#[cfg(feature = "materializer")]
use anyhow::ensure;
use anyhow::{bail, Result};
use bytes::Bytes;
use lance_context_wal::replication::runtime::client::payload_memory::Reservation;
#[cfg(feature = "materializer")]
use lance_context_wal::replication::runtime::client::payload_memory::{Pool, Unavailable};

#[cfg(feature = "materializer")]
pub(crate) fn reserve(
    memory: Option<&dyn Pool>,
    bytes: usize,
) -> Result<Option<Box<dyn Reservation>>> {
    let Some(memory) = memory else {
        return Ok(None);
    };
    let mut reservation = memory.try_reserve(bytes)?.ok_or(Unavailable)?;
    ensure!(reservation.bytes() >= bytes, "undersized upload admission");
    reservation.set_backed_bytes(0);
    Ok(Some(reservation))
}

struct Charged {
    // Drop backing before returning admission, including on cancellation.
    body: Bytes,
    _reservation: Box<dyn Reservation>,
}

impl AsRef<[u8]> for Charged {
    fn as_ref(&self) -> &[u8] {
        &self.body
    }
}

pub(crate) fn retain(
    body: Bytes,
    capacity: usize,
    mut reservation: Option<Box<dyn Reservation>>,
) -> Result<Bytes> {
    if let Some(reservation) = &mut reservation {
        if capacity < body.len() || capacity > reservation.bytes() {
            drop(body);
            bail!("upload backing exceeds admission");
        }
        reservation.set_backed_bytes(capacity);
    }
    Ok(match reservation {
        Some(reservation) => Bytes::from_owner(Charged {
            body,
            _reservation: reservation,
        }),
        None => body,
    })
}
