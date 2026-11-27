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
    speculative: bool,
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
        Ok(Some(Self {
            service: service.clone(),
            future: Some(Box::pin(async move {
                owner.lookup_prepared_bodies(prepared).await
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
    pub(super) query: Query,
}

pub(super) struct Ahead {
    pub(super) after: u64,
    loading: Option<BoxFuture<'static, Result<Option<Window>>>>,
    result: Option<Result<Option<Window>>>,
    overlap_recorded: bool,
}

impl Ahead {
    pub(super) fn new(service: &Service, after: u64, mut page: Page, cut: Option<u64>) -> Self {
        let service = service.clone();
        Self {
            after,
            loading: Some(Box::pin(async move {
                if page.entries.is_empty() {
                    if cut.is_some_and(|cut| page.through >= cut) {
                        return Ok(None);
                    }
                    let Some(next) = prefetch::source_page(&service, page.through, cut).await?
                    else {
                        return Ok(None);
                    };
                    page = next;
                }
                let Some((index, meta)) = page.entries.first().cloned() else {
                    return Ok(None);
                };
                ensure!(index > after, "unordered speculative source window");
                let Some(head) = prefetch::fetch(&service, index, meta).await? else {
                    return Ok(None);
                };
                page.entries.remove(0);
                let mut tail = VecDeque::new();
                prefetch::prepare(&service, &head, &mut page, &mut tail, cut).await?;
                let Some(query) = Query::new(&service, &head, &tail, true)? else {
                    return Ok(None);
                };
                Ok(Some(Window {
                    head,
                    tail,
                    page,
                    query,
                }))
            })),
            result: None,
            overlap_recorded: false,
        }
    }

    pub(super) fn poll(&mut self, cx: &mut Context<'_>, current: Option<&Query>) {
        if let Some(loading) = self.loading.as_mut() {
            if let Poll::Ready(result) = loading.as_mut().poll(cx) {
                self.loading = None;
                self.result = Some(result);
            }
        }
        if let Some(Ok(Some(window))) = self.result.as_mut() {
            window.query.poll(cx);
            if !self.overlap_recorded
                && current.is_some_and(|query| query.native_running.load(Ordering::Relaxed))
                && window.query.native_running.load(Ordering::Relaxed)
            {
                self.overlap_recorded = true;
                let mut observed = window.query.service.shared.body_lookups.lock().unwrap();
                observed.history_query_overlaps = observed.history_query_overlaps.saturating_add(1);
            }
        }
    }

    pub(super) async fn ready(&mut self) -> Result<Option<Window>> {
        // Promote the decoded window without waiting for its query. It becomes
        // foreground and may overlap the following window's query in turn.
        futures::future::poll_fn(|cx| {
            self.poll(cx, None);
            self.result.take().map_or(Poll::Pending, Poll::Ready)
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
