use super::*;
use crate::replication::{
    checkpoint::{Image, Part, CHUNK_BYTES},
    PreparedCheckpoint,
};
use uuid::Uuid;

// Test-only transfer relay. The actual network transport remains a separate
// qualification; this exercises worker ownership and durable installation.
pub(super) async fn transfer(destination: &Handle, outgoing: &Outbound) -> Result<()> {
    let image = outgoing
        .snapshot_image()
        .context("missing captured snapshot")?
        .clone();
    assert_eq!(image.snapshot()?, *outgoing.packet().message.get_snapshot());
    let prepared = receive_image(destination, outgoing.packet().clone(), image).await?;
    destination.step_snapshot(prepared)?.await
}

async fn receive_image(
    destination: &Handle,
    packet: Packet,
    image: Arc<Image>,
) -> Result<PreparedCheckpoint> {
    let mut receive = destination
        .receive_snapshot(packet, Uuid::new_v4())?
        .await?;
    let prepared = tokio::task::spawn_blocking(move || {
        for part in [Part::Metadata, Part::Payload] {
            let mut offset = 0;
            while offset < image.bytes(part) {
                let size = (image.bytes(part) - offset).min(CHUNK_BYTES as u64) as usize;
                let chunk = image.chunk(part, offset, size)?;
                receive.write(part, offset, &chunk)?;
                offset += chunk.len() as u64;
            }
        }
        receive.finish()
    })
    .await??;
    Ok(prepared)
}

async fn build(handle: &Handle, group: u64, id: Uuid) -> Result<PreparedCheckpoint> {
    let task = handle.prepare_checkpoint(group, id)?.await?;
    tokio::task::spawn_blocking(move || task.run()).await?
}

async fn checkpoint(handle: &Handle, group: u64) -> Result<Arc<Image>> {
    let prepared = build(handle, group, Uuid::new_v4()).await?;
    let image = prepared.image();
    handle.install_checkpoint(prepared)?.await?;
    Ok(image)
}

async fn collect(handle: &Handle) -> Result<u64> {
    let mut task = handle.prepare_image_collection()?.await?;
    tokio::task::spawn_blocking(move || -> Result<u64> {
        let mut removed = 0;
        loop {
            let progress = task.run_batch(1)?;
            assert!(progress.scanned_entries <= 1);
            removed += progress.removed_files;
            if progress.finished {
                return Ok(removed);
            }
        }
    })
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_snapshot_catches_up_compacted_follower_and_preserves_restart_retries() -> Result<()>
{
    snapshot_restart_retries(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipelined_replication_keeps_snapshot_ownership_and_restart_retries() -> Result<()> {
    snapshot_restart_retries(true).await
}

async fn snapshot_restart_retries(pipeline_replication: bool) -> Result<()> {
    let options = Options {
        pipeline_replication,
        ..Options::default()
    };
    let root = tempfile::tempdir()?;
    let c = Cluster::open(root.path(), options.clone())?;
    c.elect(1, 10).await?;
    c.elect(1, 20).await?;
    c.isolated.store(3, Ordering::Release);
    let mut expected = Vec::new();
    for group in [10, 20] {
        for sequence in 1..=4 {
            let mut b = batch(sequence);
            b.meta.bucket = u32::from(group == 20);
            let i = commit(&c.handles[&1], group, b.clone()).await?;
            expected.push((group, i, b));
        }
    }
    for group in [10, 20] {
        drop(checkpoint(&c.handles[&1], group).await?);
    }
    assert_eq!(c.handles[&3].observation(10)?.status.accepted_records, 0);
    c.isolated.store(0, Ordering::Release);
    wait_for(|| {
        c.snapshot_transfers.load(Ordering::Acquire) >= 2
            && [10, 20].into_iter().all(|group| {
                c.handles[&3]
                    .observation(group)
                    .unwrap()
                    .status
                    .accepted_records
                    == 4
            })
    })
    .await?;
    for group in [10, 20] {
        let mut next = batch(5);
        next.meta.bucket = u32::from(group == 20);
        let i = commit(&c.handles[&1], group, next.clone()).await?;
        expected.push((group, i, next));
    }
    wait_for(|| {
        c.handles.values().all(|h| {
            [10, 20]
                .into_iter()
                .all(|g| h.observation(g).unwrap().status.accepted_records == 5)
        })
    })
    .await?;
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        for (group, i, b) in &expected {
            assert_eq!(node.read_accepted(*group, *i)?, Some(b.clone()));
        }
    }
    let c = Cluster::open(root.path(), options)?;
    c.elect(3, 10).await?;
    c.elect(3, 20).await?;
    // Only the latest identity is exactly retryable; sequence5 must not have
    // created another copy when retried after every original engine reopened.
    for (group, i, b) in expected.iter().filter(|(_, _, b)| b.meta.sequence == 5) {
        assert_eq!(commit(&c.handles[&3], *group, b.clone()).await?, *i);
        assert_eq!(
            c.handles[&3].observation(*group)?.status.accepted_records,
            5
        );
    }
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_snapshot_retries_with_sender_control_credits_exhausted() -> Result<()> {
    let root = tempfile::tempdir()?;
    let options = Options {
        control_requests: 4,
        ..Options::default()
    };
    let c = Cluster::open(root.path(), options)?;
    c.elect(1, 10).await?;
    c.elect(1, 20).await?;
    c.isolated.store(3, Ordering::Release);
    let original = commit(&c.handles[&1], 10, batch(1)).await?;
    drop(checkpoint(&c.handles[&1], 10).await?);
    c.hold_snapshots.store(true, Ordering::Release);
    c.isolated.store(0, Ordering::Release);
    wait_for(|| !c.held_snapshots.lock().unwrap().is_empty()).await?;

    // Real unconsumed quorum-query replies retain every sender control slot.
    // Snapshot failure must still reach the owner and trigger a fresh transfer.
    let held: Vec<_> = (0..4)
        .map(|_| c.handles[&1].query(20, query::Query::Accepted { after: 0, limit: 1 }))
        .collect::<Result<_>>()?;
    assert!(c.handles[&1].report_snapshot(10, 3, false).is_err());
    assert_eq!(c.handles[&1].shared.control.slots.available_permits(), 0);
    c.hold_snapshots.store(false, Ordering::Release);
    // Simulate a cancelled transport task: no explicit success/failure callback.
    c.held_snapshots.lock().unwrap().clear();
    wait_for(|| {
        c.snapshot_transfers.load(Ordering::Acquire) > 0
            && c.handles[&3]
                .observation(10)
                .unwrap()
                .status
                .accepted_records
                == 1
    })
    .await?;
    assert_eq!(c.handles[&1].shared.control.slots.available_permits(), 0);
    assert_eq!(commit(&c.handles[&1], 10, batch(1)).await?, original);
    for reply in held {
        reply.await?;
    }
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        assert_eq!(node.read_accepted(10, original)?, Some(batch(1)));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_image_tasks_exclude_siblings_and_keep_engine_ownership_after_shutdown(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = Cluster::open(root.path(), Options::default())?;
    c.elect(1, 10).await?;
    c.elect(1, 20).await?;
    commit(&c.handles[&1], 10, batch(1)).await?;
    let mut b = batch(1);
    b.meta.bucket = 1;
    commit(&c.handles[&1], 20, b).await?;
    let handle = c.handles[&1].clone();
    let task = handle.prepare_checkpoint(10, Uuid::new_v4())?.await?;
    assert!(handle
        .prepare_checkpoint(20, Uuid::new_v4())?
        .await
        .is_err());
    assert!(handle.prepare_image_collection()?.await.is_err());
    // A captured image cut retains its old input while the owner keeps serving
    // ordinary appends. The later suffix must survive installation.
    let second = commit(&handle, 10, batch(2)).await?;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= second)
    })
    .await?;
    let prepared = tokio::task::spawn_blocking(move || task.run()).await??;
    let captured = prepared.image();
    let installation = handle.install_checkpoint(prepared)?;
    drop(installation); // Cancellation cannot revoke an already queued install.
    wait_for(|| {
        handle.shared.control.slots.available_permits() == Options::default().control_requests
    })
    .await?;
    drop(checkpoint(&handle, 10).await?);
    assert_eq!(
        collect(&handle).await?,
        0,
        "captured old image is still protected"
    );
    drop(captured);
    assert_eq!(collect(&handle).await?, 2);
    let maintenance = handle.prepare_log_maintenance()?.await?;
    tokio::task::spawn_blocking(move || maintenance.run()).await??;
    let task = handle.prepare_image_collection()?.await?;
    c.shutdown().await?;
    assert!(Node::open(&root.path().join("1"), layout(), 1, Limits::default()).is_err());
    drop(task);
    let node = Node::open(&root.path().join("1"), layout(), 1, Limits::default())?;
    assert_eq!(node.read_accepted(10, second)?, Some(batch(2)));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupted_snapshot_and_failed_install_never_acknowledge_or_release_peer_success(
) -> Result<()> {
    use raft_proto::eraftpb::{Message, MessageType};
    let root = tempfile::tempdir()?;
    let source_root = root.path().to_path_buf();
    let (image, term) = tokio::task::spawn_blocking(move || -> Result<_> {
        let mut nodes = (1..=3)
            .map(|id| {
                Node::open(
                    &source_root.join(id.to_string()),
                    layout(),
                    id,
                    Limits::default(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        nodes[0].campaign(10)?;
        crate::replication::tests::pump(&mut nodes)?;
        nodes[0].propose(10, 1, batch(1))?;
        crate::replication::tests::pump(&mut nodes)?;
        let prepared = nodes[0].prepare_checkpoint(10, Uuid::new_v4())?.run()?;
        let image = prepared.image();
        nodes[0].install_checkpoint(prepared)?;
        Ok((image, nodes[0].status(10)?.term))
    })
    .await??;
    // A fresh follower has none of this committed prefix. A stale snapshot
    // would be ignored and would not exercise the durable install failure.
    let node = Node::open(
        &root.path().join("receiver"),
        layout(),
        2,
        Limits::default(),
    )?;
    let store = node.groups[&10].raw.store().clone();
    let (runtime, mut outbox) = Runtime::spawn(node, Options::default())?;
    let handle = runtime.handle();
    let mut message = Message {
        from: 1,
        to: 2,
        term,
        ..Default::default()
    };
    message.set_msg_type(MessageType::MsgSnapshot);
    message.set_snapshot(image.snapshot()?);
    let packet = Packet {
        cluster: handle.cluster,
        layout: handle.layout,
        policy: handle.policy,
        group: 10,
        message,
    };
    let mut foreign = packet.clone();
    foreign.message.to = 3;
    assert!(handle.receive_snapshot(foreign, Uuid::new_v4()).is_err());
    let mut receiver = handle
        .receive_snapshot(packet.clone(), Uuid::new_v4())?
        .await?;
    assert!(handle.prepare_image_collection()?.await.is_err());
    let partial = image.chunk(Part::Metadata, 0, 8)?;
    let result = tokio::task::spawn_blocking(move || {
        receiver.write(Part::Metadata, 0, &partial)?;
        receiver.finish()
    })
    .await?;
    assert!(result.is_err());
    assert_eq!(handle.observation(10)?.status.applied, 0);
    assert_eq!(collect(&handle).await?, 2);
    let prepared = receive_image(&handle, packet, image.clone()).await?;
    store.fail_next_sync_for_test();
    assert!(
        timeout(Duration::from_secs(5), handle.step_snapshot(prepared)?)
            .await?
            .is_err()
    );
    wait_for(|| handle.observation(10).unwrap().stopped && handle.observation(20).unwrap().stopped)
        .await?;
    assert!(
        outbox.try_recv().is_err(),
        "failed snapshot persistence released a peer reply"
    );
    assert!(runtime.shutdown().await.is_err());
    drop(store);
    let node = Node::open(
        &root.path().join("receiver"),
        layout(),
        2,
        Limits::default(),
    )?;
    // An uncertain install may be recovered, but it was never acknowledged.
    let page = node.accepted(10, 0, 8)?;
    for (index, _) in page.entries {
        assert_eq!(node.read_accepted(10, index)?, Some(batch(1)));
    }
    Ok(())
}
