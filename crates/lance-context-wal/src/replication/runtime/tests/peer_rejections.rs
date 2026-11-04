use super::*;
use crate::{
    admission::DiskBudgetExhausted,
    raft_log::{DiskOptions, LogOptions},
};
use raft_proto::eraftpb::MessageType;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_reports_disk_refusal_separately_and_keeps_decision_after_reclaim() -> Result<()> {
    let root = tempfile::tempdir()?;
    let open = |id: u64| {
        Node::open_with_storage_options(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits {
                max_record_bytes: 1 << 20,
                max_uncommitted_bytes: 1 << 20,
                ..Limits::default()
            },
            LogOptions {
                target_file_bytes: 16 << 10,
                purge_threshold_bytes: 64 << 10,
            },
        )?
        .with_disk_budget(DiskOptions {
            max_bytes: 32 << 20,
            data_bytes: 8 << 20,
            headroom_bytes: 8 << 20,
        })
    };
    let mut nodes = (1..=3).map(open).collect::<Result<Vec<_>>>()?;
    nodes[0].campaign(10)?;
    crate::replication::tests::pump(&mut nodes)?;
    nodes[0].propose(10, 1, batch(1))?;
    let events = nodes[0].poll()?;
    let append = events
        .messages
        .iter()
        .find(|p| p.message.to == 2 && !p.message.entries.is_empty())
        .context("missing follower append")?
        .clone();
    let appended_index = append.message.entries.last().unwrap().index;

    // Real retained files exhaust the follower's data budget. The worker must
    // report that decision, not a later filesystem sample or a queue refusal.
    let occupied = root.path().join("2/retained-space-fixture");
    std::fs::write(&occupied, vec![7_u8; 9 << 20])?;
    let mut follower = nodes.remove(1);
    follower.prepare_disk_measurement()?.run()?;
    let expected = follower
        .step(append.clone())
        .unwrap_err()
        .downcast::<DiskBudgetExhausted>()?;
    let (runtime, mut outbox) = Runtime::spawn(
        follower,
        Options {
            tick_interval: Duration::from_secs(1),
            ..Options::default()
        },
    )?;
    let handle = runtime.handle();
    handle.step(append.clone())?;
    wait_for(|| handle.peer_drops().1 == 1).await?;
    let observed = handle.peer_rejections();
    assert_eq!(observed.disk_budget, 1);
    assert_eq!(observed.other, 0);
    let refusal = observed.last_disk_refusal.as_ref().unwrap();
    assert_eq!((refusal.group, refusal.from), (10, 1));
    assert_eq!(
        serde_json::to_value(&refusal.budget)?,
        serde_json::to_value(expected)?
    );
    assert_eq!(handle.admission().incoming_peers.rejected_requests, 0);
    assert_eq!(handle.admission().incoming_peers.rejected_bytes, 0);

    // A packet accepted into the runtime queue can still fail Node validation.
    // Its error is not disk pressure and must not overwrite the disk decision.
    let mut invalid = append.clone();
    invalid.message.set_msg_type(MessageType::MsgPropose);
    handle.step(invalid)?;
    wait_for(|| handle.peer_drops().1 == 2).await?;
    let historical = handle.peer_rejections();
    assert_eq!(historical.disk_budget, 1);
    assert_eq!(historical.other, 1);
    assert_eq!(
        serde_json::to_value(&historical.last_disk_refusal)?,
        serde_json::to_value(&observed.last_disk_refusal)?
    );

    // Reclaim space and deliver the same replication packet. This is Raft
    // retransmission, not a new business mutation. Observe its durable ACK.
    std::fs::remove_file(occupied)?;
    let current = handle.prepare_disk_measurement()?.await?.run()?;
    assert!(current.accounted_disk_bytes < refusal.budget.accounted_bytes);
    handle.step(append)?;
    timeout(Duration::from_secs(5), async {
        loop {
            let outgoing = outbox.recv().await.context("outbox closed")?;
            let message = &outgoing.packet().message;
            if message.get_msg_type() == MessageType::MsgAppendResponse
                && !message.reject
                && message.index == appended_index
            {
                return Ok::<_, anyhow::Error>(());
            }
        }
    })
    .await??;
    assert_eq!(handle.peer_drops().1, 2);
    assert_eq!(
        serde_json::to_value(handle.peer_rejections())?,
        serde_json::to_value(historical)?
    );
    runtime.shutdown().await?;
    drop(outbox);
    drop(handle);

    // Reopen the follower, deliver the original packets (including its stored
    // duplicate), then finish consensus and check applied bytes on every node.
    nodes.insert(1, open(2)?);
    for packet in events.messages {
        nodes[packet.message.to as usize - 1].step(packet)?;
    }
    crate::replication::tests::pump(&mut nodes)?;
    for node in &nodes {
        assert_eq!(node.read_accepted(10, appended_index)?, Some(batch(1)));
    }
    Ok(())
}
