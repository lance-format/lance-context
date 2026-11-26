//! A single test budget for native backing allocations and DataFusion operators.
use futures::future::BoxFuture;
use lance::deps::datafusion::execution::memory_pool::{MemoryLimit, MemoryPool, MemoryReservation};
use lance_core::memory::{NativeMemoryKind, NativeMemoryPool, NativeMemoryReservation};
use std::{
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

#[derive(Debug, Default)]
struct Counts {
    held: AtomicUsize,
    peak: AtomicUsize,
    kinds: [AtomicUsize; 4],
    peaks: [AtomicUsize; 4],
    backed: AtomicUsize,
    backed_peak: AtomicUsize,
}

#[derive(Debug)]
pub(super) struct Memory {
    pub(super) limit: AtomicUsize,
    counts: Arc<Counts>,
}

impl Memory {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            limit: AtomicUsize::new(limit),
            counts: Arc::default(),
        }
    }

    fn charge(&self, kind: usize, bytes: usize) -> lance_core::Result<()> {
        let mut current = self.counts.held.load(Ordering::SeqCst);
        let previous = loop {
            let next = current
                .checked_add(bytes)
                .filter(|next| *next <= self.limit.load(Ordering::SeqCst))
                .ok_or_else(|| {
                    lance_core::Error::backpressure("test shared capacity exhausted".to_owned())
                })?;
            match self.counts.held.compare_exchange_weak(
                current,
                next,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(previous) => break previous,
                Err(observed) => current = observed,
            }
        };
        self.counts
            .peak
            .fetch_max(previous + bytes, Ordering::SeqCst);
        let current = self.counts.kinds[kind].fetch_add(bytes, Ordering::SeqCst) + bytes;
        self.counts.peaks[kind].fetch_max(current, Ordering::SeqCst);
        Ok(())
    }

    pub(super) async fn released(&self) -> anyhow::Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while self.counts.held.load(Ordering::SeqCst) != 0 {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await?;
        assert_eq!(self.counts.backed.load(Ordering::SeqCst), 0);
        assert!(self
            .counts
            .kinds
            .iter()
            .all(|n| n.load(Ordering::SeqCst) == 0));
        Ok(())
    }

    pub(super) fn report(&self, limit: usize) {
        let peaks = self
            .counts
            .peaks
            .each_ref()
            .map(|n| n.load(Ordering::SeqCst));
        let peak = self.counts.peak.load(Ordering::SeqCst);
        let backed = self.counts.backed_peak.load(Ordering::SeqCst);
        assert!(peak > 0 && peak <= limit && backed > 0 && backed <= peak);
        assert!(peaks[0] > 0 && peaks[1] > 0);
        eprintln!("bounded native readahead: rows=20480 fragments=20 peak_accounted_bytes={peak} native_backed_peak={backed} Read_Decode_Index_DF_peaks={peaks:?} limit={limit}; all released");
    }
}

#[derive(Debug)]
struct Charge {
    counts: Arc<Counts>,
    kind: usize,
    bytes: usize,
    backed: usize,
}

impl NativeMemoryReservation for Charge {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn set_backed_bytes(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        let current = if bytes >= self.backed {
            self.counts
                .backed
                .fetch_add(bytes - self.backed, Ordering::SeqCst)
                + bytes
                - self.backed
        } else {
            self.counts
                .backed
                .fetch_sub(self.backed - bytes, Ordering::SeqCst)
                - (self.backed - bytes)
        };
        self.counts.backed_peak.fetch_max(current, Ordering::SeqCst);
        self.backed = bytes;
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.counts.backed.fetch_sub(self.backed, Ordering::SeqCst);
        self.counts.kinds[self.kind].fetch_sub(self.bytes, Ordering::SeqCst);
        self.counts.held.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

impl NativeMemoryPool for Memory {
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
        self.charge(kind, bytes)?;
        Ok(Box::new(Charge {
            counts: self.counts.clone(),
            kind,
            bytes,
            backed: 0,
        }))
    }
}

impl fmt::Display for Memory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("shared test budget")
    }
}

impl MemoryPool for Memory {
    fn name(&self) -> &str {
        "shared test budget"
    }
    fn grow(&self, _reservation: &MemoryReservation, bytes: usize) {
        // DataFusion infallible growth is observable even if it exceeds the
        // capacity; the final peak assertion must catch that in this fixture.
        let held = self.counts.held.fetch_add(bytes, Ordering::SeqCst) + bytes;
        let kind = self.counts.kinds[3].fetch_add(bytes, Ordering::SeqCst) + bytes;
        self.counts.peak.fetch_max(held, Ordering::SeqCst);
        self.counts.peaks[3].fetch_max(kind, Ordering::SeqCst);
    }
    fn shrink(&self, _reservation: &MemoryReservation, bytes: usize) {
        self.counts.kinds[3].fetch_sub(bytes, Ordering::SeqCst);
        self.counts.held.fetch_sub(bytes, Ordering::SeqCst);
    }
    fn try_grow(
        &self,
        _reservation: &MemoryReservation,
        bytes: usize,
    ) -> lance::deps::datafusion::common::Result<()> {
        self.charge(3, bytes).map_err(|e| {
            lance::deps::datafusion::common::DataFusionError::ResourcesExhausted(e.to_string())
        })
    }
    fn reserved(&self) -> usize {
        self.counts.held.load(Ordering::SeqCst)
    }
    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(self.limit.load(Ordering::SeqCst))
    }
}
