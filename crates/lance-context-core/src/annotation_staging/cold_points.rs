//! Combine queued cold point reads against the exact same protected base.
use super::*;
use std::{collections::VecDeque, sync::Mutex as SyncMutex};
use tokio::sync::oneshot;

// No batching timer: idle reads start immediately, while reads arriving during
// native IO share the next scan. Bound both scan workspace and result keys.
const SCANS: usize = 2;
const KEYS: usize = 128;

#[derive(Clone)]
pub(super) struct ColdPoints {
    state: Arc<SyncMutex<Queue>>,
    slots: Arc<Semaphore>,
}

#[derive(Default)]
struct Queue {
    requests: VecDeque<Request>,
    running: usize,
}

struct Request {
    base: Arc<views::Base>,
    turn: Arc<str>,
    value_bytes: usize,
    reply: oneshot::Sender<Result<Option<TurnAnnotation>>>,
    // A cancelled caller must not release GC protection/admission while its
    // detached native scan still uses this snapshot. Queue admission also
    // bounds repeated resolves of the same captured PointRead.
    _reader: Arc<OwnedSemaphorePermit>,
    _slot: OwnedSemaphorePermit,
}

impl ColdPoints {
    pub fn new(readers: usize) -> Self {
        Self {
            state: Arc::new(SyncMutex::new(Queue::default())),
            slots: Arc::new(Semaphore::new(readers)),
        }
    }

    pub async fn read(
        &self,
        base: &Arc<views::Base>,
        turn: &Arc<str>,
        value_bytes: usize,
        reader: &Arc<OwnedSemaphorePermit>,
    ) -> Result<Option<TurnAnnotation>> {
        let slot = reserve(&self.slots, 1)?;
        let (reply, receive) = oneshot::channel();
        {
            let mut state = self.state.lock().unwrap();
            state.requests.push_back(Request {
                base: base.clone(),
                turn: turn.clone(),
                value_bytes,
                reply,
                _reader: reader.clone(),
                _slot: slot,
            });
            if state.running < SCANS {
                state.running += 1;
                // The worker owns only the queue, never PublishedValues or a
                // sender to itself. It exits as soon as the queue empties.
                tokio::spawn(run(self.state.clone()));
            }
        }
        receive
            .await
            .map_err(|_| Error::internal("annotation cold point worker stopped"))?
    }
}

async fn run(state: Arc<SyncMutex<Queue>>) {
    loop {
        let batch = {
            let mut state = state.lock().unwrap();
            state.requests.retain(|r| !r.reply.is_closed());
            let Some(first) = state.requests.pop_front() else {
                state.running -= 1;
                return;
            };
            let mut batch = vec![first];
            let mut remaining = VecDeque::new();
            for request in state.requests.drain(..) {
                if batch.len() < KEYS && Arc::ptr_eq(&request.base, &batch[0].base) {
                    batch.push(request);
                } else {
                    remaining.push_back(request);
                }
            }
            state.requests = remaining;
            batch
        };
        resolve(batch).await;
    }
}

async fn resolve(batch: Vec<Request>) {
    if batch.len() == 1 {
        let request = batch.into_iter().next().unwrap();
        let value = read_base(&request.base, &request.turn, request.value_bytes).await;
        let _ = request.reply.send(value);
        return;
    }
    let turns = batch.iter().map(|r| r.turn.as_ref()).collect();
    let rows = read_existing_with_scan_rows(
        &batch[0].base,
        &turns,
        &["labels", "status", "score", "id"],
        COLD_POINT_SCAN_ROWS,
    )
    .await;
    // Decode before releasing requests: rows borrow the request keys, and each
    // returned Arrow slice shares the scan's narrow buffers without copying.
    let results = match rows {
        Ok(rows) => batch
            .iter()
            .map(|r| {
                rows.get(r.turn.as_ref())
                    .map(|row| decode_base_row(row, 0, &r.turn, r.value_bytes))
                    .transpose()
            })
            .collect::<Vec<_>>(),
        Err(error) => {
            let shared = lance_core::error::CloneableError::new(error);
            batch
                .iter()
                .map(|_| Err(shared.clone().into_inner()))
                .collect()
        }
    };
    for (request, result) in batch.into_iter().zip(results) {
        let _ = request.reply.send(result);
    }
}
