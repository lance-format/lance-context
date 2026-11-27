//! One speculative whole window, owned and cancelled with the ordered driver.
use super::*;
use futures::future::BoxFuture;
use lance_context_wal::replication::Page;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    task::{Context, Poll},
};

pub(super) struct Query {
    service: Service,
    future: Option<BoxFuture<'static, Result<Option<single_body::Lookup>>>>,
    result: Option<Result<Option<single_body::Lookup>>>,
    native_running: Arc<AtomicBool>,
    requests: usize,
    bytes: u64,
    pub(super) speculative: bool,
    started: bool,
}

impl Query {
    pub(super) fn new(
        service: &Service,
        head: &Pending,
        tail: &VecDeque<Pending>,
        speculative: bool,
    ) -> Result<Option<Self>> {
        let mut keys = Vec::new();
        let mut bytes = 0u64;
        for record in std::iter::once(head).chain(tail) {
            let mixed::Decoded::Creates { rows, .. } = &record.decoded else {
                bail!("immutable body cannot contain patches")
            };
            keys.extend(rows.iter().map(|row| row.turn.as_str()));
            bytes = bytes.saturating_add(record.meta.body_bytes);
        }
        let Some(mut prepared) = service.prepare_body_lookup(&keys)? else {
            return Ok(None);
        };
        let native_running = Arc::new(AtomicBool::new(false));
        prepared.native_running = Some(native_running.clone());
        let owner = service.clone();
        let window = lookup_capture::Window {
            first_record: head.index,
            last_record: tail.back().unwrap_or(head).index,
            requests: tail.len() + 1,
            payload_bytes: bytes,
            speculative,
        };
        Ok(Some(Self {
            service: service.clone(),
            future: Some(Box::pin(async move {
                owner.lookup_prepared_bodies(prepared, window).await
            })),
            result: None,
            native_running,
            requests: tail.len() + 1,
            bytes,
            speculative,
            started: false,
        }))
    }

    pub(super) fn poll(&mut self, cx: &mut Context<'_>) {
        let Some(future) = self.future.as_mut() else {
            return;
        };
        if !self.started {
            self.started = true;
            if self.speculative {
                let mut observed = self.service.shared.body_lookups.lock().unwrap();
                observed.prefetched_lookups_started =
                    observed.prefetched_lookups_started.saturating_add(1);
            }
        }
        if let Poll::Ready(result) = future.as_mut().poll(cx) {
            if let Ok(Some(lookup)) = &result {
                let mut observed = self.service.shared.body_lookups.lock().unwrap();
                observed.record_window(lookup.keys.observation(), self.requests, self.bytes);
                if self.speculative {
                    observed.prefetched_lookups_completed =
                        observed.prefetched_lookups_completed.saturating_add(1);
                }
            }
            self.future = None;
            self.result = Some(result);
        }
    }

    pub(super) fn take_result(&mut self) -> Option<Result<Option<single_body::Lookup>>> {
        self.result.take()
    }
}

impl Drop for Query {
    fn drop(&mut self) {
        if self.speculative && self.future.is_some() {
            let mut observed = self.service.shared.body_lookups.lock().unwrap();
            observed.prefetched_lookups_cancelled =
                observed.prefetched_lookups_cancelled.saturating_add(1);
        }
    }
}

pub(super) struct Window {
    pub(super) head: Pending,
    pub(super) tail: VecDeque<Pending>,
    pub(super) page: Page,
    pub(super) query: Option<Query>,
    cut: Option<u64>,
}

pub(super) struct Ahead {
    pub(super) after: u64,
    service: Service,
    cut: Option<u64>,
    batch_wait: Duration,
    loading: Option<BoxFuture<'static, Result<Loaded>>>,
    result: Option<Result<Loaded>>,
    retry_at: Option<std::pin::Pin<Box<tokio::time::Sleep>>>,
    overlap_recorded: bool,
}

enum Loaded {
    /// No matching committed input yet. Retain the validated metadata cursor,
    /// not an application position or a claim that upstream input is exhausted.
    Gap(Page),
    Unavailable,
    Ready(Box<Window>),
}

impl Ahead {
    pub(super) fn new(
        service: &Service,
        after: u64,
        page: Page,
        cut: Option<u64>,
        batch_wait: Duration,
    ) -> Self {
        Self {
            after,
            service: service.clone(),
            cut,
            batch_wait,
            loading: Some(Self::load(service.clone(), after, page, cut, batch_wait)),
            result: None,
            retry_at: None,
            overlap_recorded: false,
        }
    }

    fn load(
        service: Service,
        after: u64,
        mut page: Page,
        cut: Option<u64>,
        batch_wait: Duration,
    ) -> BoxFuture<'static, Result<Loaded>> {
        Box::pin(async move {
            if page.entries.is_empty() {
                if cut.is_some_and(|cut| page.through >= cut) {
                    return Ok(Loaded::Unavailable);
                }
                let Some(next) = prefetch::source_page(&service, page.through, cut).await? else {
                    return Ok(Loaded::Gap(page));
                };
                page = next;
            }
            let Some((index, meta)) = page.entries.first().cloned() else {
                return Ok(Loaded::Gap(page));
            };
            ensure!(index > after, "unordered speculative source window");
            let Some(head) = prefetch::fetch(&service, index, meta).await? else {
                return Ok(Loaded::Unavailable);
            };
            if prefetch::exceeds_key_window(&service, &head)? {
                // Symmetric with the foreground: an oversize head may need the
                // whole lookup budget, so it is never speculated beside one.
                return Ok(Loaded::Unavailable);
            }
            page.entries.remove(0);
            let mut tail = VecDeque::new();
            let prepared = prefetch::prepare(&service, &head, &mut page, &mut tail, cut).await?;
            // A configured wait lets the successor collect later commits
            // and start its query while the foreground is still running.
            // Keep one absolute deadline inside this retained future, so
            // cancellation of Driver::step cannot restart the wait.
            let deferred = matches!(prepared.body_stop, Some(PrefetchStop::PageExhausted))
                && cut.is_none_or(|cut| page.through < cut);
            if deferred && !batch_wait.is_zero() {
                let deadline = tokio::time::Instant::now() + batch_wait;
                prefetch::coalesce(&service, &head, &mut page, &mut tail, cut, deadline).await?;
            }
            // Without an opt-in wait, a successor that merely exhausted the
            // committed page keeps promotion-time refill: records committed
            // before promotion join this window's single scan instead of
            // forcing a second one. A window sealed by a real bound (e.g.
            // `KeyBound`) cannot grow, so its query starts now and overlaps
            // the foreground. Exactness is unchanged: apply revalidates the
            // captured base.
            let query = if deferred && batch_wait.is_zero() {
                None
            } else {
                let Some(query) = Query::new(&service, &head, &tail, true)? else {
                    return Ok(Loaded::Unavailable);
                };
                Some(query)
            };
            Ok(Loaded::Ready(Box::new(Window {
                head,
                tail,
                page,
                query,
                cut,
            })))
        })
    }

    pub(super) fn poll(&mut self, cx: &mut Context<'_>, current: Option<&Query>) {
        loop {
            if let Some(loading) = self.loading.as_mut() {
                if let Poll::Ready(result) = loading.as_mut().poll(cx) {
                    self.loading = None;
                    self.result = Some(result);
                }
            }
            if !self.batch_wait.is_zero()
                && current.is_some_and(|query| query.future.is_some())
                && self.service.apply_backpressure().is_none()
                && matches!(self.result, Some(Ok(Loaded::Gap(_))))
            {
                let retry = self.retry_at.get_or_insert_with(|| {
                    // Reuse the configured batching cadence for an empty
                    // successor throughout a potentially long native scan.
                    Box::pin(tokio::time::sleep(
                        self.batch_wait.max(Duration::from_millis(5)),
                    ))
                });
                if retry.as_mut().poll(cx).is_ready() {
                    let Some(Ok(Loaded::Gap(page))) = self.result.take() else {
                        unreachable!("checked source gap")
                    };
                    self.retry_at = None;
                    self.loading = Some(Self::load(
                        self.service.clone(),
                        self.after,
                        page,
                        self.cut,
                        self.batch_wait,
                    ));
                    // Poll the new read now to register its waker. There is
                    // still only one successor and no retained payload at a gap.
                    continue;
                }
            }
            break;
        }
        if let Some(Ok(Loaded::Ready(window))) = self.result.as_mut() {
            let Some(query) = window.query.as_mut() else {
                return;
            };
            query.poll(cx);
            if !self.overlap_recorded
                && current.is_some_and(|query| query.native_running.load(Ordering::Relaxed))
                && query.native_running.load(Ordering::Relaxed)
            {
                self.overlap_recorded = true;
                let mut observed = query.service.shared.body_lookups.lock().unwrap();
                observed.history_query_overlaps = observed.history_query_overlaps.saturating_add(1);
            }
        }
    }

    pub(super) async fn ready(
        &mut self,
        service: &Service,
        batch_wait: Duration,
    ) -> Result<Option<Window>> {
        // Promote the decoded window without waiting for its query. It becomes
        // foreground and may overlap the following window's query in turn.
        futures::future::poll_fn(|cx| loop {
            self.poll(cx, None);
            let Some(result) = self.result.take() else {
                return Poll::Pending;
            };
            match result {
                Ok(Loaded::Ready(mut window)) if window.query.is_none() => {
                    let service = service.clone();
                    let deadline = tokio::time::Instant::now() + batch_wait;
                    // Retain the refill future in Ahead, not in this call's
                    // stack. Cancellation of step must preserve both admitted
                    // records and the next source cursor for the same driver.
                    self.loading = Some(Box::pin(async move {
                        prefetch::coalesce(
                            &service,
                            &window.head,
                            &mut window.page,
                            &mut window.tail,
                            window.cut,
                            deadline,
                        )
                        .await?;
                        let Some(query) = Query::new(&service, &window.head, &window.tail, false)?
                        else {
                            return Ok(Loaded::Unavailable);
                        };
                        window.query = Some(query);
                        Ok(Loaded::Ready(window))
                    }));
                }
                Ok(Loaded::Ready(window)) => return Poll::Ready(Ok(Some(*window))),
                Ok(Loaded::Gap(_) | Loaded::Unavailable) => return Poll::Ready(Ok(None)),
                Err(error) => return Poll::Ready(Err(error)),
            }
        })
        .await
    }
}

pub(super) async fn resolve(
    query: &mut Query,
    ahead: &mut Option<Ahead>,
) -> Result<Option<single_body::Lookup>> {
    futures::future::poll_fn(|cx| {
        // Foreground acquires native capacity before speculative work.
        query.poll(cx);
        if matches!(query.result, Some(Err(_)) | Some(Ok(None))) {
            return Poll::Ready(query.take_result().unwrap());
        }
        if let Some(ahead) = ahead {
            ahead.poll(cx, Some(query));
        }
        query.take_result().map_or(Poll::Pending, Poll::Ready)
    })
    .await
}

pub(super) async fn tick(ahead: &mut Option<Ahead>) {
    futures::future::poll_fn(|cx| {
        if let Some(ahead) = ahead {
            ahead.poll(cx, None);
        }
        Poll::Ready(())
    })
    .await
}

/// Drive `foreground` to completion, also polling the speculative successor
/// on each wakeup. The foreground is polled first so it claims native
/// capacity before speculative work; the successor never gates completion.
/// No task is spawned: dropping this future drops neither the successor nor
/// its retained records, which stay owned by the driver's `ahead` slot.
pub(super) async fn drive<F: std::future::Future>(
    foreground: F,
    ahead: &mut Option<Ahead>,
) -> F::Output {
    let mut foreground = std::pin::pin!(foreground);
    futures::future::poll_fn(|cx| {
        if let Poll::Ready(output) = foreground.as_mut().poll(cx) {
            return Poll::Ready(output);
        }
        if let Some(ahead) = ahead {
            ahead.poll(cx, None);
        }
        Poll::Pending
    })
    .await
}

#[cfg(test)]
mod drive_tests {
    use super::*;

    #[tokio::test]
    async fn drive_returns_foreground_without_successor() {
        let mut ahead = None;
        assert_eq!(drive(async { 7 }, &mut ahead).await, 7);
    }

    #[tokio::test]
    async fn drive_completes_pending_foreground() {
        let mut ahead = None;
        let out = drive(
            async {
                tokio::task::yield_now().await;
                3
            },
            &mut ahead,
        )
        .await;
        assert_eq!(out, 3);
    }
}
