//! Caller-owned admission for native allocations. Reservations follow backing
//! allocations, independently of scheduler backpressure and cache eviction weight.

use std::{fmt::Debug, ptr::NonNull, sync::Arc};

use arrow_buffer::Buffer;
use bytes::Bytes;
use futures::future::BoxFuture;

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeMemoryKind {
    Read,
    Decode,
    Index,
}

/// Dropping the reservation releases its admission. `bytes` is allocation
/// capacity, including spare capacity retained by a truncated buffer.
pub trait NativeMemoryReservation: Send + Sync + Debug {
    fn bytes(&self) -> usize;

    /// Pending I/O has zero backing; set this once its bytes arrive.
    fn set_backed_bytes(&mut self, bytes: usize);
}

/// Admission is scoped to the supplied owner, never installed process-wide.
/// `reserve` must not wait indefinitely while the caller owns earlier buffers;
/// `try_reserve` must return immediately on refusal. Errors retain their causes
/// for the owning operation's recovery policy.
pub trait NativeMemoryPool: Send + Sync + Debug {
    fn reserve(
        &self,
        kind: NativeMemoryKind,
        bytes: usize,
    ) -> BoxFuture<'_, Result<Box<dyn NativeMemoryReservation>>>;

    fn try_reserve(
        &self,
        kind: NativeMemoryKind,
        bytes: usize,
    ) -> Result<Box<dyn NativeMemoryReservation>>;
}

pub type NativeMemory = Option<Arc<dyn NativeMemoryPool>>;
pub type NativeReservation = Option<Box<dyn NativeMemoryReservation>>;

/// Classify all allocations made by a component under one caller category.
pub fn with_kind(memory: NativeMemory, kind: NativeMemoryKind) -> NativeMemory {
    memory.map(|pool| Arc::new(KindMemoryPool { pool, kind }) as Arc<dyn NativeMemoryPool>)
}

#[derive(Debug)]
struct KindMemoryPool {
    pool: Arc<dyn NativeMemoryPool>,
    kind: NativeMemoryKind,
}

impl NativeMemoryPool for KindMemoryPool {
    fn reserve(
        &self,
        _kind: NativeMemoryKind,
        bytes: usize,
    ) -> BoxFuture<'_, Result<Box<dyn NativeMemoryReservation>>> {
        self.pool.reserve(self.kind, bytes)
    }

    fn try_reserve(
        &self,
        _kind: NativeMemoryKind,
        bytes: usize,
    ) -> Result<Box<dyn NativeMemoryReservation>> {
        self.pool.try_reserve(self.kind, bytes)
    }
}

pub async fn reserve_read(memory: &NativeMemory, bytes: usize) -> Result<NativeReservation> {
    match memory.as_ref().filter(|_| bytes != 0) {
        Some(memory) => {
            let mut reservation = memory.reserve(NativeMemoryKind::Read, bytes).await?;
            reservation.set_backed_bytes(0);
            Ok(Some(reservation))
        }
        None => Ok(None),
    }
}

pub fn try_reserve(
    memory: &NativeMemory,
    kind: NativeMemoryKind,
    bytes: usize,
) -> Result<NativeReservation> {
    memory
        .as_ref()
        .filter(|_| bytes != 0)
        .map(|memory| memory.try_reserve(kind, bytes))
        .transpose()
}

/// Admit a growing vector before reallocating. The old reservation is held
/// until reallocation has released its old backing. Call before extending;
/// attach the returned reservation to the vector's final shared allocation.
pub fn reserve_vec<T>(
    values: &mut Vec<T>,
    reservation: &mut NativeReservation,
    memory: &NativeMemory,
    kind: NativeMemoryKind,
    additional: usize,
) -> Result<()> {
    let required = values
        .len()
        .checked_add(additional)
        .ok_or_else(|| crate::Error::execution("native vector capacity overflow"))?;
    if required <= values.capacity() {
        return Ok(());
    }
    if memory.is_none() {
        values.reserve(additional);
        return Ok(());
    }
    let capacity = required.max(values.capacity().saturating_mul(2));
    let bytes = capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| crate::Error::execution("native vector byte capacity overflow"))?;
    let mut next = try_reserve(memory, kind, bytes)?;
    if let Some(next) = &mut next {
        next.set_backed_bytes(0);
    }
    values.reserve_exact(capacity - values.len());
    if let Some(next) = &mut next {
        next.set_backed_bytes(bytes);
    }
    *reservation = next;
    Ok(())
}

struct ReservedBytes {
    bytes: Bytes,
    _reservation: Box<dyn NativeMemoryReservation>,
}

impl AsRef<[u8]> for ReservedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

/// Preserve one reservation through every clone/slice of these bytes.
pub fn retain_bytes(bytes: Bytes, reservation: NativeReservation) -> Bytes {
    match reservation {
        Some(mut reservation) => {
            reservation.set_backed_bytes(reservation.bytes());
            Bytes::from_owner(ReservedBytes {
                bytes,
                _reservation: reservation,
            })
        }
        None => bytes,
    }
}

struct ReservedBuffer {
    buffer: Buffer,
    _reservation: Box<dyn NativeMemoryReservation>,
}

impl AsRef<[u8]> for ReservedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

/// Preserve alignment and the admission through Arrow clones and slices.
/// The original buffer remains the owner of the actual allocation.
pub fn retain_buffer(buffer: Buffer, reservation: NativeReservation) -> Buffer {
    let Some(mut reservation) = reservation else {
        return buffer;
    };
    reservation.set_backed_bytes(reservation.bytes());
    let bytes = Bytes::from_owner(ReservedBuffer {
        buffer,
        _reservation: reservation,
    });
    // SAFETY: the owner retains the original buffer without changing its
    // pointer or alignment, and every resulting slice retains that owner.
    unsafe {
        Buffer::from_custom_allocation(
            NonNull::new(bytes.as_ptr().cast_mut()).expect("buffer has a non-null pointer"),
            bytes.len(),
            Arc::new(bytes),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[derive(Debug)]
    struct Reservation(Arc<AtomicUsize>, usize);

    impl NativeMemoryReservation for Reservation {
        fn bytes(&self) -> usize {
            self.1
        }
        fn set_backed_bytes(&mut self, _: usize) {}
    }

    impl Drop for Reservation {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn reservation_follows_arrow_slice_after_original_is_dropped() {
        let released = Arc::new(AtomicUsize::new(0));
        let original = retain_buffer(
            Buffer::from_vec(vec![1_u64, 2, 3, 4]),
            Some(Box::new(Reservation(released.clone(), 32))),
        );
        let slice = original.slice_with_length(8, 8);
        assert_eq!(slice.as_ptr().align_offset(8), 0);
        drop(original);
        assert_eq!(released.load(Ordering::SeqCst), 0);
        assert_eq!(slice.typed_data::<u64>(), &[2]);
        drop(slice);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn reservation_follows_bytes_slice() {
        let released = Arc::new(AtomicUsize::new(0));
        let original = retain_bytes(
            Bytes::from(vec![1_u8, 2, 3]),
            Some(Box::new(Reservation(released.clone(), 3))),
        );
        let slice = original.slice(1..);
        drop(original);
        assert_eq!(released.load(Ordering::SeqCst), 0);
        assert_eq!(slice.as_ref(), &[2, 3]);
        drop(slice);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }
}
