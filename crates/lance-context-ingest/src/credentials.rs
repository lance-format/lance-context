//! Wait for temporary credential renewal outages inside the original operation.
use lance_io::object_store::StorageOptionsProvider;
use std::{collections::HashMap, io::ErrorKind, sync::Arc};

mod observation;
pub use observation::Observation;

/// Wrap the origin provider once, before constructing its storage-options
/// accessor. The provider keeps ownership of authentication, per-request
/// deadlines and its existing request retries. This wrapper waits between
/// exhausted *transient* renewal attempts; it does not replay a blob request,
/// application mutation or task, and does not change the provider's identity.
///
/// Transient failures keep the calling future pending until renewal succeeds
/// or the caller cancels it. There are no spawned tasks or detached retries.
/// Drop/timeout the operation to cancel its awaited provider future and backoff.
/// The origin provider owns any work it spawns and its own request timeout. It
/// must report known transient failures as
/// a direct `lance::Error::IO` source containing a `std::io::Error`.
#[derive(Clone)]
pub struct RenewingStorageOptionsProvider {
    inner: Arc<dyn StorageOptionsProvider>,
    counters: Arc<observation::Counters>,
}

impl std::fmt::Debug for RenewingStorageOptionsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenewingStorageOptionsProvider")
            .finish_non_exhaustive()
    }
}

impl RenewingStorageOptionsProvider {
    pub fn new(inner: Arc<dyn StorageOptionsProvider>) -> Self {
        Self {
            inner,
            counters: Arc::default(),
        }
    }

    /// Clones share observations; credentials and provider identity are omitted.
    pub fn observation(&self) -> Observation {
        self.counters.observation()
    }

    async fn fetch(&self, force: bool) -> lance::Result<Option<HashMap<String, String>>> {
        let mut call = self.counters.call();
        // Reuse the native storage client's backoff scale. Each renewal round
        // retains the origin provider's own bounded request policy; no second
        // attempt counter or total deadline terminates the owning operation.
        let backoff = object_store::BackoffConfig::default();
        let mut delay = backoff.init_backoff;
        loop {
            let result = {
                let _round = call.provider_round();
                if force {
                    self.inner.force_fetch_storage_options().await
                } else {
                    self.inner.fetch_storage_options().await
                }
            };
            match result {
                Err(error) if transient(&error) => {
                    call.retry();
                    {
                        let _backoff = self.counters.phase(true);
                        tokio::time::sleep(delay).await;
                    }
                    delay = delay.mul_f64(backoff.base).min(backoff.max_backoff);
                }
                result => {
                    call.finish(result.is_ok());
                    return result;
                }
            }
        }
    }
}

fn transient(error: &lance::Error) -> bool {
    let lance::Error::IO { source, .. } = error.find_root() else {
        return false;
    };
    // Do not cross an External/provider boundary or parse an error message.
    // In particular, a scoped accessor must not start another renewal loop,
    // and personal caller-denial errors retain their existing contract.
    source.downcast_ref::<std::io::Error>().is_some_and(|io| {
        matches!(
            io.kind(),
            ErrorKind::TimedOut
                | ErrorKind::WouldBlock
                | ErrorKind::ConnectionReset
                | ErrorKind::ConnectionAborted
                | ErrorKind::ConnectionRefused
                | ErrorKind::NotConnected
                | ErrorKind::BrokenPipe
                | ErrorKind::UnexpectedEof
        )
    })
}

#[async_trait::async_trait]
impl StorageOptionsProvider for RenewingStorageOptionsProvider {
    async fn fetch_storage_options(&self) -> lance::Result<Option<HashMap<String, String>>> {
        self.fetch(false).await
    }

    async fn force_fetch_storage_options(&self) -> lance::Result<Option<HashMap<String, String>>> {
        self.fetch(true).await
    }

    fn provider_id(&self) -> String {
        self.inner.provider_id()
    }
}

#[cfg(test)]
mod tests;
