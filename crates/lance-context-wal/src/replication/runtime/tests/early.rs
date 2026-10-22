use super::*;
use raft_proto::eraftpb::MessageType;

fn open_nodes(root: &Path) -> Result<Vec<Node>> {
    (1..=3)
        .map(|id| Node::open(&root.join(id.to_string()), layout(), id, Limits::default()))
        .collect()
}

// The test thread owns the disk barrier. Actual dedicated group workers and
// real log files perform the append; block_on only drives their async clients.
#[test]
fn pipelined_append_reaches_a_follower_during_blocked_leader_sync_without_ack() -> Result<()> {
    assert!(!Options::default().pipeline_replication);
    let executor = tokio::runtime::Runtime::new()?;
    for fail_sync in [false, true] {
        let directory = tempfile::tempdir()?;
        let mut nodes = open_nodes(directory.path())?;
        nodes[0].campaign(10)?;
        crate::replication::tests::pump(&mut nodes)?;
        let node = nodes.remove(0);
        let store = node.groups[&10].raw.store().clone();
        let (runtime, mut outbox) = Runtime::spawn(
            node,
            Options {
                pipeline_replication: true,
                peer_requests: 1,
                peer_bytes: 16 * 1024,
                tick_interval: Duration::from_secs(1),
                ..Options::default()
            },
        )?;
        let handle = runtime.handle();
        let blocked = store.block_sync_for_test();
        if fail_sync {
            store.fail_next_sync_for_test();
        }
        let body = batch(1);
        let pointer = body.body.as_ptr();
        let pending = handle.commit(Command::Append {
            group: 10,
            batch: body.clone(),
        })?;
        let outgoing = executor.block_on(async {
            timeout(Duration::from_secs(5), async {
                loop {
                    let value = outbox.recv().await.context("outbox closed")?;
                    if value.packet().message.get_msg_type() == MessageType::MsgAppend
                        && value
                            .packet()
                            .message
                            .entries
                            .iter()
                            .any(|entry| entry.data == body.body)
                    {
                        return Ok::<_, anyhow::Error>(value);
                    }
                }
            })
            .await?
        })?;
        assert!(outgoing.snapshot_image().is_none());
        let entry = outgoing
            .packet()
            .message
            .entries
            .iter()
            .find(|entry| entry.data == body.body)
            .unwrap();
        let original_index = entry.index;
        assert_eq!(
            entry.data.as_ptr(),
            pointer,
            "early transport copied payload"
        );
        executor.block_on(wait_for(|| {
            store.appended_index_for_test() == Some(original_index)
        }))?;
        // The second destination cannot bypass a held packet's node-wide slot.
        assert!(handle.peer_drops().0 > 0);
        assert_eq!(handle.shared.outgoing.slots.available_permits(), 0);
        let follower = outgoing.packet().message.to as usize - 2;
        nodes[follower].step(outgoing.packet().clone())?;
        let response = nodes[follower].poll()?;
        assert!(response.messages.iter().any(|packet| {
            packet.message.get_msg_type() == MessageType::MsgAppendResponse
                && !packet.message.reject
                && packet.message.index == original_index
        }));
        for packet in response.messages {
            handle.step(packet)?;
        }
        assert!(pending.0.is_empty(), "quorum response bypassed leader sync");
        drop(outgoing);
        assert_eq!(handle.shared.outgoing.slots.available_permits(), 1);
        drop(blocked);
        let outcome =
            executor.block_on(async { timeout(Duration::from_secs(5), pending).await })?;
        if fail_sync {
            assert!(outcome.is_err(), "failed leader sync released a client ACK");
            executor.block_on(wait_for(|| {
                handle.observation(10).unwrap().stopped && handle.observation(20).unwrap().stopped
            }))?;
            assert!(executor.block_on(runtime.shutdown()).is_err());
        } else {
            assert_eq!(index(outcome?), original_index);
            executor.block_on(runtime.shutdown())?;
        }
        drop(outbox);
        drop(handle);
        drop(store);
        drop(nodes);

        // The failed-sync case is uncertain, not rejected. Its follower has a
        // durable append; elect that follower after reopening every directory.
        let mut nodes = open_nodes(directory.path())?;
        nodes[follower + 1].campaign(10)?;
        crate::replication::tests::pump(&mut nodes)?;
        for node in &nodes {
            assert_eq!(node.read_accepted(10, original_index)?, Some(body.clone()));
        }
        nodes[follower + 1].propose(10, 99, body)?;
        let retried = crate::replication::tests::pump(&mut nodes)?;
        let retried = retried.into_iter().find(|done| done.request == 99).unwrap();
        assert_eq!(index(retried.outcome), original_index);
    }
    Ok(())
}

#[test]
fn pipelined_follower_sync_failure_releases_no_append_response() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let mut nodes = open_nodes(directory.path())?;
    nodes[0].campaign(10)?;
    crate::replication::tests::pump(&mut nodes)?;
    nodes[0].propose(10, 1, batch(1))?;
    let mut packets = Vec::new();
    let events = nodes[0].poll_with_early_messages(|packet| packets.push(packet))?;
    assert!(events.completions.is_empty());
    let append = packets
        .into_iter()
        .find(|packet| {
            packet.message.to == 2
                && packet.message.get_msg_type() == MessageType::MsgAppend
                && !packet.message.entries.is_empty()
        })
        .context("missing early append")?;
    nodes[1].step(append)?;
    nodes[1].groups[&10].raw.store().fail_next_sync_for_test();
    let mut responses = Vec::new();
    assert!(nodes[1]
        .poll_with_early_messages(|packet| responses.push(packet))
        .is_err());
    assert!(responses.is_empty(), "follower replied before durable sync");
    assert!(nodes[0].poll()?.completions.is_empty());
    assert!(nodes[1].status(20).is_err());
    Ok(())
}
