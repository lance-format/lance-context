use super::*;
use object_store::{
    retry::{RetryContext, RetryFailure, RetryPolicy},
    UploadPart,
};

#[derive(Debug)]
struct Policy {
    stopped: watch::Receiver<()>,
    counters: Arc<Counters>,
    operation: Operation,
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Part,
    Completion,
}

impl Operation {
    fn waiting(self, counters: &Counters) -> &AtomicU64 {
        match self {
            Self::Part => &counters.waiting_part_retries,
            Self::Completion => &counters.waiting_completion_retries,
        }
    }

    fn decisions(self, counters: &Counters) -> &AtomicU64 {
        match self {
            Self::Part => &counters.part_retry_decisions,
            Self::Completion => &counters.completion_retry_decisions,
        }
    }
}

struct WaitingRetry(Arc<Counters>, Operation);

impl Drop for WaitingRetry {
    fn drop(&mut self) {
        self.1.waiting(&self.0).fetch_sub(1, Ordering::Relaxed);
    }
}

#[async_trait::async_trait]
impl RetryPolicy for Policy {
    async fn retry(&self, context: RetryContext) -> bool {
        if !matches!(context.failure, RetryFailure::Status(code) if matches!(code.as_u16(), 429 | 503))
        {
            return false;
        }
        // The SDK retains the same part ID or block list and shared payload, and refreshes
        // credentials when this policy permits another bounded request loop.
        let delay = Duration::from_secs((1u64 << context.attempt.saturating_sub(1).min(5)).min(30));
        let mut stopped = self.stopped.clone();
        self.operation
            .waiting(&self.counters)
            .fetch_add(1, Ordering::Relaxed);
        let _waiting = WaitingRetry(self.counters.clone(), self.operation);
        tokio::select! {
            biased;
            _ = stopped.changed() => false,
            _ = tokio::time::sleep(delay) => {
                self.operation.decisions(&self.counters).fetch_add(1, Ordering::Relaxed);
                true
            },
        }
    }
}

pub(super) async fn start(
    store: &UploadStore,
    path: &Path,
    opts: PutMultipartOptions,
) -> object_store::Result<Box<dyn MultipartUpload>> {
    let opts = opts
        .with_retry_policy(Arc::new(Policy {
            stopped: store.stopped.clone(),
            counters: store.counters.clone(),
            operation: Operation::Part,
        }))
        .with_azure_completion_retry_policy(Arc::new(Policy {
            stopped: store.stopped.clone(),
            counters: store.counters.clone(),
            operation: Operation::Completion,
        }));
    let mut stopped = store.stopped.clone();
    let inner = tokio::select! {
        biased;
        _ = stopped.changed() => return Err(cancelled()),
        result = store.inner.put_multipart_opts(path, opts) => result?,
    };
    Ok(Box::new(Upload { inner, stopped }))
}

#[derive(Debug)]
struct Upload {
    inner: Box<dyn MultipartUpload>,
    stopped: watch::Receiver<()>,
}

#[async_trait::async_trait]
impl MultipartUpload for Upload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        // Exactly one call: invoking put_part again would allocate a new
        // ordinal. Only the SDK may repeat the original logical part.
        let original = self.inner.put_part(data);
        let mut stopped = self.stopped.clone();
        Box::pin(async move {
            tokio::select! {
                biased;
                _ = stopped.changed() => Err(cancelled()),
                result = original => result,
            }
        })
    }

    async fn complete(&mut self) -> object_store::Result<PutResult> {
        tokio::select! {
            biased;
            _ = self.stopped.changed() => Err(cancelled()),
            result = self.inner.complete() => result,
        }
    }

    async fn abort(&mut self) -> object_store::Result<()> {
        self.inner.abort().await
    }
}

#[cfg(test)]
mod tests;
