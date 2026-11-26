use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    task::{Poll, Waker},
};

use futures::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use lance_core::{Error, Result};
use lance_encoding::decoder::{FilterExpression, ReadBatchTask};
use lance_io::ReadBatchParams;

use super::{DecodeEngine, PreparedProjection};

/// Called after failed decoder state is released. Returning `Ok(())` retries
/// only the failed physical selection; returning `Err` terminates that read.
/// Attempts start at one. Dropping the read future cancels the callback.
#[derive(Clone)]
pub struct ReadMemoryRetry(
    pub Arc<dyn Fn(Error, usize, &'static str) -> BoxFuture<'static, Result<()>> + Send + Sync>,
);

impl std::fmt::Debug for ReadMemoryRetry {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str("ReadMemoryRetry")
    }
}

/// Process-local read identity. Offsets count assigned physical rows, before
/// deletions or residual filters. `None` identifies the unassigned suffix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadRetryContext {
    pub stream_id: u64,
    pub row_offset: usize,
    pub assigned_rows: Option<u32>,
}

impl ReadRetryContext {
    /// Preserve the source error while attaching physical read progress.
    pub fn attach(self, error: Error) -> Error {
        Error::external(Box::new(ReadFailure {
            context: self,
            error,
        }))
    }
}

#[derive(Debug)]
struct ReadFailure {
    context: ReadRetryContext,
    error: Error,
}

impl std::fmt::Display for ReadFailure {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.error, out)
    }
}

impl std::error::Error for ReadFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Inspect physical progress without formatting provider errors or file paths.
pub fn read_retry_context(error: &Error) -> Option<ReadRetryContext> {
    let mut cause: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(read) = cause.downcast_ref::<ReadFailure>() {
            return Some(read.context);
        }
        if let Some(error) = cause.downcast_ref::<Error>() {
            if let Some(source) = error.external_source() {
                cause = source.as_ref();
                continue;
            }
        }
        cause = cause.source()?;
    }
}

// Only unassigned tasks belong to this stream. Emitted tasks own their exact
// selection and may finish concurrently, including after a suffix is reopened.
struct Pending {
    stream: Option<BoxStream<'static, ReadBatchTask>>,
    generation: usize,
    waker: Option<Waker>,
}

impl Pending {
    fn discard(shared: &Mutex<Self>, generation: usize) {
        let (stream, waker) = {
            let mut pending = shared.lock().unwrap();
            if pending.generation != generation {
                return;
            }
            pending.generation += 1;
            (pending.stream.take(), pending.waker.take())
        };
        // Release queued decode inputs even when the stream consumer is waiting
        // on this assigned task instead of polling the stream again.
        drop(stream);
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

struct Recovery {
    core: DecodeEngine,
    prepared: PreparedProjection,
    read_len: u64,
    params: ReadBatchParams,
    batch_size: u32,
    policy: ReadMemoryRetry,
    pending: Arc<Mutex<Pending>>,
    context: ReadRetryContext,
    selected_rows: usize,
    attempt: usize,
    terminal: bool,
}

impl Drop for Recovery {
    fn drop(&mut self) {
        let generation = self.pending.lock().unwrap().generation;
        Pending::discard(&self.pending, generation);
    }
}

impl Recovery {
    fn fail(&mut self, error: Error) -> ReadBatchTask {
        self.terminal = true;
        let generation = self.pending.lock().unwrap().generation;
        Pending::discard(&self.pending, generation);
        ReadBatchTask {
            num_rows: 0,
            task: futures::future::ready(Err(error)).boxed(),
        }
    }

    fn assigned(
        &self,
        task: ReadBatchTask,
        selection: ReadBatchParams,
        generation: usize,
    ) -> ReadBatchTask {
        let mut core = self.core.clone();
        // Adaptive decoding chose this boundary already. Replaying a smaller
        // byte-target batch would break the caller's physical row assignment.
        core.options.batch_size_bytes = None;
        let prepared = self.prepared.clone();
        let read_len = self.read_len;
        let policy = self.policy.clone();
        let pending = self.pending.clone();
        let num_rows = task.num_rows;
        let context = ReadRetryContext {
            assigned_rows: Some(num_rows),
            ..self.context
        };
        ReadBatchTask {
            num_rows,
            task: async move {
                let mut result = task.task.await;
                let mut attempt = 0;
                loop {
                    let error = match result {
                        Ok(batch) => return Ok(batch),
                        Err(error) => error,
                    };
                    Pending::discard(&pending, generation);
                    attempt += 1;
                    (policy.0)(context.attach(error), attempt, "file_batch").await?;
                    result = match core
                        .read_prepared_tasks_once(
                            selection.clone(),
                            num_rows,
                            prepared.clone(),
                            read_len,
                            FilterExpression::no_filter(),
                        )
                        .await
                    {
                        Err(error) => Err(error),
                        Ok(mut stream) => {
                            let result = match stream.next().await {
                                Some(next) => next.task.await.and_then(|batch| {
                                    if batch.num_rows() == num_rows as usize {
                                        Ok(batch)
                                    } else {
                                        Err(Error::internal(
                                            "read replay changed its assigned row count",
                                        ))
                                    }
                                }),
                                None => Err(Error::internal("read replay lost its assigned rows")),
                            };
                            drop(stream);
                            result
                        }
                    };
                }
            }
            .boxed(),
        }
    }

    async fn recover(&mut self, error: Error, phase: &'static str) -> Result<()> {
        self.attempt += 1;
        (self.policy.0)(self.context.attach(error), self.attempt, phase).await
    }

    async fn next(&mut self) -> Option<ReadBatchTask> {
        loop {
            if self.terminal {
                return None;
            }
            let reopen = {
                let pending = self.pending.lock().unwrap();
                pending.stream.is_none().then_some(pending.generation)
            };
            if let Some(generation) = reopen {
                if self.context.row_offset == self.selected_rows {
                    return None;
                }
                let suffix = match self.params.slice(
                    self.context.row_offset,
                    self.selected_rows - self.context.row_offset,
                ) {
                    Ok(suffix) => suffix,
                    Err(error) => return Some(self.fail(error)),
                };
                match self
                    .core
                    .read_prepared_tasks_once(
                        suffix,
                        self.batch_size,
                        self.prepared.clone(),
                        self.read_len,
                        FilterExpression::no_filter(),
                    )
                    .await
                {
                    Ok(stream) => {
                        let mut pending = self.pending.lock().unwrap();
                        if pending.generation == generation {
                            pending.stream = Some(stream);
                        }
                    }
                    Err(error) => {
                        if let Err(error) = self.recover(error, "file_stream_reopen").await {
                            return Some(self.fail(error));
                        }
                    }
                }
                continue;
            }
            let next = futures::future::poll_fn(|cx| {
                let mut pending = self.pending.lock().unwrap();
                pending.waker = Some(cx.waker().clone());
                let generation = pending.generation;
                match pending.stream.as_mut() {
                    Some(stream) => stream
                        .as_mut()
                        .poll_next(cx)
                        .map(|next| Some((generation, next))),
                    None => Poll::Ready(None),
                }
            })
            .await;
            let Some((generation, next)) = next else {
                continue;
            };
            let Some(task) = next else {
                if self.context.row_offset != self.selected_rows {
                    return Some(self.fail(Error::internal(
                        "read stream ended before all physical rows were assigned",
                    )));
                }
                self.terminal = true;
                Pending::discard(&self.pending, generation);
                return None;
            };
            if task.num_rows != 0 {
                let count = task.num_rows as usize;
                let selection = match self.params.slice(self.context.row_offset, count) {
                    Ok(selection) => selection,
                    Err(error) => return Some(self.fail(error)),
                };
                let task = self.assigned(task, selection, generation);
                self.context.row_offset += count;
                self.attempt = 0;
                return Some(task);
            }
            // Scheduler failure before assignment is a zero-row task. Its
            // output never supplies a cursor, so reopen the unassigned suffix.
            let error = match task.task.await {
                Ok(batch) => {
                    return Some(ReadBatchTask {
                        num_rows: 0,
                        task: futures::future::ready(Ok(batch)).boxed(),
                    });
                }
                Err(error) => error,
            };
            Pending::discard(&self.pending, generation);
            if let Err(error) = self.recover(error, "file_stream_next").await {
                return Some(self.fail(error));
            }
        }
    }
}

pub(super) async fn read(
    core: DecodeEngine,
    params: ReadBatchParams,
    batch_size: u32,
    prepared: PreparedProjection,
    read_len: u64,
    policy: ReadMemoryRetry,
) -> Result<BoxStream<'static, ReadBatchTask>> {
    static NEXT_READ: AtomicU64 = AtomicU64::new(1);
    let context = ReadRetryContext {
        stream_id: NEXT_READ.fetch_add(1, Ordering::Relaxed),
        row_offset: 0,
        assigned_rows: None,
    };
    let mut attempt = 0;
    let stream = loop {
        match core
            .read_prepared_tasks_once(
                params.clone(),
                batch_size,
                prepared.clone(),
                read_len,
                FilterExpression::no_filter(),
            )
            .await
        {
            Ok(stream) => break stream,
            Err(error) => {
                attempt += 1;
                (policy.0)(context.attach(error), attempt, "file_stream_open").await?;
            }
        }
    };
    let selected_rows = match &params {
        ReadBatchParams::Range(range) => range.len(),
        ReadBatchParams::Ranges(ranges) => ranges
            .iter()
            .map(|range| (range.end - range.start) as usize)
            .sum(),
        ReadBatchParams::Indices(indices) => indices.len(),
        ReadBatchParams::RangeFull => read_len as usize,
        ReadBatchParams::RangeTo(range) => range.end,
        ReadBatchParams::RangeFrom(range) => read_len as usize - range.start,
    };
    let recovery = Recovery {
        core,
        prepared,
        read_len,
        params,
        batch_size,
        policy,
        context,
        selected_rows,
        pending: Arc::new(Mutex::new(Pending {
            stream: Some(stream),
            generation: 0,
            waker: None,
        })),
        attempt: 0,
        terminal: false,
    };
    Ok(futures::stream::unfold(recovery, |mut recovery| async {
        recovery.next().await.map(|task| (task, recovery))
    })
    .boxed())
}
