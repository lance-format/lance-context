//! Fixed-size thread-local measurements of the ordinary engine write path.
//! Acquiring a lock does not establish a durable write or quorum receipt.

#[derive(Clone, Copy)]
pub(super) enum Gate {
    Maintenance,
    Append,
    Sync,
    Exclusive,
}

#[cfg(not(feature = "runtime"))]
pub(super) fn acquire<T>(_: Gate, acquire: impl FnOnce() -> T) -> T {
    acquire()
}

#[cfg(feature = "runtime")]
pub(super) fn acquire<T>(gate: Gate, acquire: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = acquire();
    let elapsed = start.elapsed();
    COUNTERS.with(|counters| {
        let mut values = counters.get();
        let value = &mut values[gate as usize];
        value.attempts = value.attempts.saturating_add(1);
        value.elapsed = value.elapsed.saturating_add(elapsed);
        value.max = value.max.max(elapsed);
        counters.set(values);
    });
    result
}

#[cfg(feature = "runtime")]
#[derive(Clone, Copy, Default)]
struct Counter {
    attempts: u64,
    elapsed: std::time::Duration,
    max: std::time::Duration,
}

#[cfg(feature = "runtime")]
thread_local! {
    static COUNTERS: std::cell::Cell<[Counter; 4]> = const {
        std::cell::Cell::new([Counter {
            attempts: 0,
            elapsed: std::time::Duration::ZERO,
            max: std::time::Duration::ZERO,
        }; 4])
    };
}

/// Completed lock acquisition attempts, including poisoned-lock errors.
/// Elapsed time includes waiting and lock acquisition, excluding lock hold time.
#[cfg(feature = "runtime")]
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct WaitInterval {
    pub attempts: u64,
    pub elapsed_micros: u64,
    pub max_elapsed_micros: u64,
}

/// Ordinary engine-write gate waits accumulated on one thread. Includes local
/// checkpoint writes on that thread; excludes other threads, group-state locks,
/// native engine I/O, and maintenance's own exclusive gate acquisitions.
/// Waits overlap poll elapsed time and are not client request latency.
#[cfg(feature = "runtime")]
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct StorageWaitObservation {
    pub maintenance: WaitInterval,
    pub append: WaitInterval,
    pub sync: WaitInterval,
    pub exclusive: WaitInterval,
}

#[cfg(feature = "runtime")]
pub(crate) fn snapshot() -> StorageWaitObservation {
    COUNTERS.with(|counters| {
        let values = counters.get();
        let interval = |gate: Gate| {
            let value = values[gate as usize];
            WaitInterval {
                attempts: value.attempts,
                elapsed_micros: value.elapsed.as_micros().min(u64::MAX as u128) as u64,
                max_elapsed_micros: value.max.as_micros().min(u64::MAX as u128) as u64,
            }
        };
        StorageWaitObservation {
            maintenance: interval(Gate::Maintenance),
            append: interval(Gate::Append),
            sync: interval(Gate::Sync),
            exclusive: interval(Gate::Exclusive),
        }
    })
}

#[cfg(all(test, feature = "runtime"))]
mod tests;
