use super::*;
use crate::replication::{
    tests::{batch, layout},
    Applied, Limits,
};
use std::path::Path;
use tokio::time::{sleep, timeout};

mod early;
#[cfg(feature = "peer-transport")]
mod fetch;
mod images;

// Actual dedicated workers and durable engines. This relay deliberately drops
// selected links; it is not the production peer transport or a throughput test.
struct Cluster {
    runtimes: Vec<Runtime>,
    handles: Arc<BTreeMap<u64, Handle>>,
    relays: Vec<tokio::task::JoinHandle<()>>,
    isolated: Arc<AtomicU64>,
    snapshot_transfers: Arc<AtomicU64>,
    hold_snapshots: Arc<AtomicBool>,
    held_snapshots: Arc<Mutex<Vec<Outbound>>>,
}
impl Cluster {
    fn open(root: &Path, options: Options) -> Result<Self> {
        let mut runtimes = Vec::new();
        let mut handles = BTreeMap::new();
        let mut outboxes = Vec::new();
        for id in 1..=3 {
            let node = Node::open(&root.join(id.to_string()), layout(), id, Limits::default())?;
            let (runtime, outbox) = Runtime::spawn(node, options.clone())?;
            handles.insert(id, runtime.handle());
            runtimes.push(runtime);
            outboxes.push(outbox);
        }
        let handles = Arc::new(handles);
        let isolated = Arc::new(AtomicU64::new(0));
        let snapshot_transfers = Arc::new(AtomicU64::new(0));
        let hold_snapshots = Arc::new(AtomicBool::new(false));
        let held_snapshots = Arc::new(Mutex::new(Vec::new()));
        let relays = outboxes
            .into_iter()
            .map(|mut outbox| {
                let handles = handles.clone();
                let isolated = isolated.clone();
                let snapshot_transfers = snapshot_transfers.clone();
                let hold_snapshots = hold_snapshots.clone();
                let held_snapshots = held_snapshots.clone();
                tokio::spawn(async move {
                    while let Some(outgoing) = outbox.recv().await {
                        let p = outgoing.packet();
                        let id = isolated.load(Ordering::Acquire);
                        if id == p.message.from || id == p.message.to {
                            continue;
                        }
                        if outgoing.snapshot_image().is_some() {
                            if hold_snapshots.load(Ordering::Acquire) {
                                held_snapshots.lock().unwrap().push(outgoing);
                                continue;
                            }
                            let delivered = images::transfer(&handles[&p.message.to], &outgoing)
                                .await
                                .is_ok();
                            if delivered {
                                snapshot_transfers.fetch_add(1, Ordering::Relaxed);
                            }
                            let _ = outgoing.finish_snapshot(delivered);
                            continue;
                        }
                        // Retain the outgoing credit until the destination takes its
                        // separately charged ownership. Entry payloads share Bytes.
                        let _ = handles[&p.message.to].step(p.clone());
                    }
                })
            })
            .collect();
        Ok(Self {
            runtimes,
            handles,
            relays,
            isolated,
            snapshot_transfers,
            hold_snapshots,
            held_snapshots,
        })
    }
    async fn elect(&self, id: u64, group: u64) -> Result<()> {
        self.handles[&id].campaign(group)?.await?;
        timeout(Duration::from_secs(10), async {
            loop {
                if self
                    .handles
                    .values()
                    .all(|h| h.observation(group).is_ok_and(|o| o.status.leader == id))
                {
                    return;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .context("worker election did not settle")?;
        Ok(())
    }
    async fn shutdown(mut self) -> Result<()> {
        for task in self.relays.drain(..) {
            task.abort();
            let _ = task.await;
        }
        self.held_snapshots.lock().unwrap().clear();
        for runtime in self.runtimes.drain(..) {
            runtime.shutdown().await?;
        }
        Ok(())
    }
}
impl Drop for Cluster {
    fn drop(&mut self) {
        for relay in &self.relays {
            relay.abort();
        }
    }
}
fn index(outcome: Outcome) -> u64 {
    match outcome {
        Outcome::Committed(Applied::Accepted { index }) => index,
        other => panic!("unexpected append outcome {other:?}"),
    }
}
async fn commit(handle: &Handle, group: u64, batch: Batch) -> Result<u64> {
    Ok(index(
        timeout(
            Duration::from_secs(10),
            handle.commit(Command::Append { group, batch })?,
        )
        .await??,
    ))
}
async fn wait_for(mut predicate: impl FnMut() -> bool) -> Result<()> {
    timeout(Duration::from_secs(5), async {
        while !predicate() {
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn durable_group_workers_preserve_payloads_and_exact_retries_after_reopen() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = Cluster::open(directory.path(), Options::default())?;
    c.elect(1, 10).await?;
    c.elect(2, 20).await?;
    let first = commit(&c.handles[&1], 10, batch(1)).await?;
    let mut other = batch(1);
    other.meta.bucket = 1;
    let second = commit(&c.handles[&2], 20, other.clone()).await?;
    assert_eq!(commit(&c.handles[&1], 10, batch(1)).await?, first);
    assert_eq!(commit(&c.handles[&2], 20, other.clone()).await?, second);
    let read = c.handles[&1]
        .query(10, query::Query::Accepted { after: 0, limit: 4 })?
        .await?;
    let query::Outcome::Observed {
        cut,
        value: query::Value::Accepted(page),
    } = read
    else {
        panic!("missing quorum page")
    };
    assert_eq!(page.entries, vec![(first, batch(1).meta)]);
    assert!(cut.applied >= first);
    wait_for(|| {
        c.handles.values().all(|h| {
            h.observation(10).unwrap().status.applied >= first
                && h.observation(20).unwrap().status.applied >= second
        })
    })
    .await?;
    c.shutdown().await?;
    // Reopen all three original engines and read the full bytes, not only a
    // counter or receipt. No in-memory worker survives these opens.
    for id in 1..=3 {
        let node = Node::open(
            &directory.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        assert_eq!(node.read_accepted(10, first)?, Some(batch(1)));
        assert_eq!(node.read_accepted(20, second)?, Some(other.clone()));
    }
    let c = Cluster::open(directory.path(), Options::default())?;
    c.elect(3, 10).await?;
    c.elect(1, 20).await?;
    assert_eq!(commit(&c.handles[&3], 10, batch(1)).await?, first);
    assert_eq!(commit(&c.handles[&1], 20, other).await?, second);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_keeps_shared_credits_until_quorum_resolves_and_control_stays_available(
) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let options = Options {
        data_requests: 1,
        tick_interval: Duration::from_secs(1),
        ..Options::default()
    };
    let c = Cluster::open(directory.path(), options)?;
    c.elect(1, 10).await?;
    commit(&c.handles[&1], 10, batch(1)).await?;
    c.isolated.store(1, Ordering::Release);
    let h = &c.handles[&1];
    let mut pending = h.commit(Command::Append {
        group: 10,
        batch: batch(2),
    })?;
    wait_for(|| h.observation(10).unwrap().status.pending == 1).await?;
    assert!(
        matches!(
            pending.0.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ),
        "isolated exact leader returned an ACK"
    );
    drop(pending);
    assert_eq!(h.shared.data.slots.available_permits(), 0);
    let mut other = batch(1);
    other.meta.bucket = 1;
    assert!(
        h.commit(Command::Append {
            group: 20,
            batch: other
        })
        .is_err(),
        "data credit was not shared across groups"
    );
    let read = h.query(10, query::Query::Accepted { after: 0, limit: 4 })?;
    drop(read);
    assert!(!h.observation(10)?.stopped);
    c.isolated.store(0, Ordering::Release);
    wait_for(|| h.shared.data.slots.available_permits() == 1).await?;
    let retry = commit(h, 10, batch(2)).await?;
    let read = h
        .query(10, query::Query::Accepted { after: 0, limit: 4 })?
        .await?;
    let query::Outcome::Observed {
        value: query::Value::Accepted(page),
        ..
    } = read
    else {
        panic!("missing accepted page")
    };
    assert_eq!(page.entries.len(), 2);
    assert_eq!(page.entries[1], (retry, batch(2).meta));
    c.isolated.store(1, Ordering::Release);
    let abandoned = h.commit(Command::Append {
        group: 10,
        batch: batch(3),
    })?;
    wait_for(|| h.observation(10).unwrap().status.pending == 1).await?;
    c.shutdown().await?;
    assert!(
        abandoned.await.is_err(),
        "shutdown fabricated a durable acknowledgment"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_peer_outbox_does_not_block_other_groups_or_cached_observation() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let node = Node::open(directory.path(), layout(), 1, Limits::default())?;
    let (runtime, mut outbox) = Runtime::spawn(
        node,
        Options {
            peer_requests: 1,
            peer_bytes: 4096,
            ..Options::default()
        },
    )?;
    let handle = runtime.handle();
    timeout(Duration::from_secs(5), handle.campaign(10)?).await??;
    wait_for(|| handle.peer_drops().0 > 0).await?;
    assert_eq!(outbox.len(), 1);
    timeout(Duration::from_secs(5), handle.campaign(20)?).await??;
    assert!(!handle.observation(20)?.stopped);
    let output = outbox
        .recv()
        .await
        .context("missing retained peer packet")?;
    assert!(output.packet().message.to == 2 || output.packet().message.to == 3);
    assert_eq!(handle.shared.outgoing.slots.available_permits(), 0);
    drop(output);
    runtime.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completed_unconsumed_replies_cannot_bypass_shared_admission() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = Cluster::open(
        directory.path(),
        Options {
            data_requests: 1,
            control_requests: 1,
            ..Options::default()
        },
    )?;
    c.elect(1, 10).await?;
    let h = &c.handles[&1];
    let reply = h.commit(Command::Append {
        group: 10,
        batch: batch(1),
    })?;
    wait_for(|| !reply.0.is_empty()).await?;
    assert_eq!(h.shared.data.slots.available_permits(), 0);
    assert!(h
        .commit(Command::Append {
            group: 10,
            batch: batch(2)
        })
        .is_err());
    let original = index(reply.await?);
    assert_eq!(commit(h, 10, batch(1)).await?, original);
    let page = h.query(
        10,
        query::Query::Accepted {
            after: 0,
            limit: 128,
        },
    )?;
    wait_for(|| !page.0.is_empty()).await?;
    assert!(h
        .query(10, query::Query::Accepted { after: 0, limit: 1 })
        .is_err());
    assert!(h.shared.control.bytes.available_permits() < Options::default().control_bytes - 32768);
    drop(page);
    assert_eq!(h.shared.control.slots.available_permits(), 1);
    h.query(10, query::Query::Accepted { after: 0, limit: 1 })?
        .await?;
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_publisher_reads_distinguish_byte_and_request_admission_limits() -> Result<()> {
    // Sixteen materializer tasks must fit both the response-byte and request
    // limits. Use real quorum reads, retaining each response credit until its
    // caller consumes the result, rather than measuring a semaphore in isolation.
    for (control_bytes, control_requests, accepted, byte_rejections, slot_rejections) in [
        (4 << 20, 64, 2, 14, 0),
        (32 << 20, 64, 16, 0, 0),
        (32 << 20, 2, 2, 0, 14),
    ] {
        let directory = tempfile::tempdir()?;
        let c = Cluster::open(
            directory.path(),
            Options {
                control_bytes,
                control_requests,
                ..Options::default()
            },
        )?;
        c.elect(1, 10).await?;
        let handle = &c.handles[&1];
        commit(handle, 10, batch(1)).await?;
        let before = handle.admission();
        let mut reads = Vec::new();
        for _ in 0..16 {
            match handle.query(
                10,
                query::Query::Publisher {
                    table: batch(1).meta.stream,
                },
            ) {
                Ok(read) => reads.push(read),
                Err(error) => assert!(error.to_string().contains("credits exhausted"), "{error:#}"),
            }
        }
        assert_eq!(reads.len(), accepted);
        let busy = handle.admission();
        assert_eq!(busy.control.capacity_bytes, control_bytes);
        assert_eq!(busy.control.capacity_requests, control_requests);
        assert_eq!(busy.control.used_requests, accepted);
        assert!(busy.control.used_bytes > 0 && busy.control.used_bytes <= control_bytes);
        assert_eq!(
            busy.control.rejected_bytes - before.control.rejected_bytes,
            byte_rejections
        );
        assert_eq!(
            busy.control.rejected_requests - before.control.rejected_requests,
            slot_rejections
        );
        assert_eq!(busy.data.rejected_requests, before.data.rejected_requests);
        assert_eq!(busy.data.rejected_bytes, before.data.rejected_bytes);
        for read in reads {
            assert!(matches!(
                read.await?,
                query::Outcome::Observed {
                    value: query::Value::Publisher(None),
                    ..
                }
            ));
        }
        let drained = handle.admission();
        assert_eq!(drained.control.used_requests, 0);
        assert_eq!(drained.control.used_bytes, 0);
        // Observing counters does not consume or restore capacity. Admission
        // recovers because the original pending owners released their credits.
        assert!(matches!(
            handle
                .query(
                    10,
                    query::Query::Publisher {
                        table: batch(1).meta.stream,
                    }
                )?
                .await?,
            query::Outcome::Observed { .. }
        ));
        c.shutdown().await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_durable_poll_stops_sibling_workers_and_releases_no_peer_response() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let mut nodes = (1..=3)
        .map(|id| {
            Node::open(
                &directory.path().join(id.to_string()),
                layout(),
                id,
                Limits::default(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    nodes[0].campaign(10)?;
    crate::replication::tests::pump(&mut nodes)?;
    assert_eq!(nodes[0].status(10)?.leader, 1);
    let node = nodes.remove(0);
    drop(nodes);
    node.groups[&10].raw.store().fail_next_sync_for_test();
    let (runtime, mut outbox) = Runtime::spawn(node, Options::default())?;
    let h = runtime.handle();
    assert!(timeout(
        Duration::from_secs(5),
        h.commit(Command::Append {
            group: 10,
            batch: batch(1)
        })?
    )
    .await?
    .is_err());
    wait_for(|| h.observation(10).unwrap().stopped && h.observation(20).unwrap().stopped).await?;
    assert!(h.campaign(20).is_err());
    assert!(
        outbox.try_recv().is_err(),
        "failed persistence released a peer response"
    );
    assert!(runtime.shutdown().await.is_err());
    // The uncertain files can be reopened only after all original workers join.
    let reopened = Node::open(&directory.path().join("1"), layout(), 1, Limits::default())?;
    assert_eq!(reopened.status(10)?.accepted_records, 0);
    assert_eq!(reopened.status(20)?.accepted_records, 0);
    Ok(())
}
