//! Standalone experiment accounting, never substituted into an embedding host.
use futures::future::BoxFuture;
use lance_core::memory::{NativeMemoryKind, NativeMemoryPool, NativeMemoryReservation};
use serde::Serialize;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default, Serialize)]
pub struct Counters {
    pub reserved: usize,
    pub backed: usize,
    pub peak_reserved: usize,
    pub peak_backed: usize,
    pub requested_bytes: u64,
    pub refused: u64,
    /// Read, Decode, Index in that order. Allocation capacity, not RSS.
    pub by_kind: [usize; 3],
}

#[derive(Debug)]
pub struct Pool {
    limit: usize,
    counters: Arc<Mutex<Counters>>,
}

impl Pool {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            counters: Arc::default(),
        }
    }
    pub fn snapshot(&self) -> Counters {
        self.counters.lock().unwrap().clone()
    }
}

#[derive(Debug)]
struct Charge {
    counters: Arc<Mutex<Counters>>,
    bytes: usize,
    backed: usize,
    kind: usize,
}

impl NativeMemoryReservation for Charge {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn set_backed_bytes(&mut self, n: usize) {
        assert!(n <= self.bytes);
        let mut counters = self.counters.lock().unwrap();
        counters.backed = counters.backed - self.backed + n;
        self.backed = n;
        counters.peak_backed = counters.peak_backed.max(counters.backed);
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        let mut counters = self.counters.lock().unwrap();
        counters.reserved -= self.bytes;
        counters.backed -= self.backed;
        counters.by_kind[self.kind] -= self.bytes;
    }
}
impl NativeMemoryPool for Pool {
    fn reserve(
        &self,
        kind: NativeMemoryKind,
        bytes: usize,
    ) -> BoxFuture<'_, lance_core::Result<Box<dyn NativeMemoryReservation>>> {
        Box::pin(async move { self.try_reserve(kind, bytes) })
    }
    fn try_reserve(
        &self,
        kind: NativeMemoryKind,
        bytes: usize,
    ) -> lance_core::Result<Box<dyn NativeMemoryReservation>> {
        let kind = match kind {
            NativeMemoryKind::Read => 0,
            NativeMemoryKind::Decode => 1,
            NativeMemoryKind::Index => 2,
        };
        let mut counters = self.counters.lock().unwrap();
        counters.requested_bytes = counters.requested_bytes.saturating_add(bytes as u64);
        let Some(next) = counters
            .reserved
            .checked_add(bytes)
            .filter(|n| *n <= self.limit)
        else {
            counters.refused += 1;
            return Err(lance_core::Error::backpressure(
                "standalone probe native budget refused",
            ));
        };
        counters.reserved = next;
        counters.peak_reserved = counters.peak_reserved.max(next);
        counters.by_kind[kind] += bytes;
        Ok(Box::new(Charge {
            counters: self.counters.clone(),
            bytes,
            backed: 0,
            kind,
        }))
    }
}
