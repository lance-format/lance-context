//! One task per lane: pull up to the record cap from the shared buffer,
//! encode, commit, record the pending Applied wait, pull again.
use std::{
    future::Future,
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use anyhow::{Context, Result};
use arrow_array::RecordBatch;
use lance_context_wal::replication::runtime::client::payload_memory::Reservation;

use super::{encode, LaneSlot, Pending, RowExceedsRecord, Shared, BACKOFF_MAX, BACKOFF_MIN};

pub(super) async fn run(
    shared: Arc<Shared>,
    slot: Arc<LaneSlot>,
    _slot_memory: Box<dyn Reservation>,
) {
    let bucket = slot.shard.bucket;
    loop {
        let idle = Instant::now();
        let pulled = match shared
            .buffer
            .pull(bucket, shared.max_rows, shared.max_bytes)
            .await
        {
            Ok(Some(pulled)) => pulled,
            Ok(None) => return,
            Err(_) => return, // buffer already failed; fence/send surface it
        };
        shared.metrics.lane_idle(idle.elapsed());
        let result = commit_rows(&shared, &slot, pulled.batches, pulled.rows).await;
        match result {
            Ok(()) => shared.buffer.done(),
            Err(error) => {
                shared.buffer.fail(&error);
                shared.buffer.done();
                return;
            }
        }
    }
}

async fn commit_rows(
    shared: &Shared,
    slot: &LaneSlot,
    batches: Vec<RecordBatch>,
    rows: usize,
) -> Result<()> {
    let schema = shared.spec.body_schema().clone();
    let body = if batches.len() == 1 {
        batches.into_iter().next().unwrap()
    } else {
        arrow_select::concat::concat_batches(&schema, &batches)?
    };
    debug_assert_eq!(body.num_rows(), rows);
    // Encoded size can exceed the in-memory estimate; halve until each part
    // fits one record (encoded-size oversize split).
    let mut parts = vec![body];
    while let Some(part) = parts.pop() {
        let encoded = encode(shared, slot.shard, &part)?;
        if encoded.len() > shared.max_bytes {
            if part.num_rows() == 1 {
                return Err(RowExceedsRecord {
                    shard: slot.shard.bucket,
                    encoded_bytes: encoded.len(),
                    max_record_bytes: shared.max_bytes,
                }
                .into());
            }
            drop(encoded);
            shared.metrics.oversize_slice();
            let half = part.num_rows() / 2;
            // Push the tail first so the head commits first.
            parts.push(part.slice(half, part.num_rows() - half));
            parts.push(part.slice(0, half));
            continue;
        }
        let keys = match &shared.key_cache {
            Some(_) => super::key_cache::batch_keys(&part, &shared.key_columns)?,
            None => Vec::new(),
        };
        commit(shared, slot, encoded, part.num_rows(), keys).await?;
    }
    Ok(())
}

async fn commit(
    shared: &Shared,
    slot: &LaneSlot,
    body: bytes::Bytes,
    rows: usize,
    keys: Vec<super::key_cache::Key>,
) -> Result<()> {
    let bytes = body.len();
    let started = Instant::now();
    let mut body = Some(body);
    let reply = retry(shared, || {
        let body = body.take();
        async move {
            match body {
                Some(body) => slot.client.commit(slot.key, body).await,
                // A failed attempt left the exact intent staged: resend it.
                None => slot
                    .client
                    .resume_commit(slot.key)
                    .await?
                    .context("staged intent disappeared"),
            }
        }
    })
    .await?;
    shared.metrics.committed(started.elapsed(), bytes, rows);
    if let Some(cache) = &shared.key_cache {
        // Durably committed: later resends of these turns can be dropped.
        cache.insert(&keys, super::key_cache::Reason::Commit)?;
    }
    let mut pending = slot.pending.lock().unwrap();
    match pending.as_mut() {
        Some(p) => {
            anyhow::ensure!(reply.index >= p.latest.index, "lane commit index regressed");
            p.latest = reply;
            p.records += 1;
            p.rows += rows;
            p.keys.extend(keys);
        }
        None => *pending = Some(Pending { latest: reply, records: 1, rows, keys }),
    }
    Ok(())
}

/// Retry transient network ambiguity with bounded exponential backoff.
pub(super) async fn retry<T, F, Fut>(shared: &Shared, mut attempt: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut delay = BACKOFF_MIN;
    loop {
        match attempt().await {
            Ok(value) => return Ok(value),
            Err(error) if (shared.transient)(&error) && !shared.stop.load(Ordering::Acquire) => {
                shared.metrics.retry();
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(BACKOFF_MAX);
            }
            Err(error) => return Err(error),
        }
    }
}
