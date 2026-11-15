use super::*;
use raft_proto::eraftpb::{Message, MessageType};

#[test]
fn follower_heartbeat_response_waits_for_a_new_term_to_be_durable() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let mut nodes = (1..=3)
        .map(|id| {
            Node::open(
                &root.path().join(id.to_string()),
                layout(),
                id,
                Limits::default(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    nodes[0].campaign(10)?;
    crate::replication::tests::pump(&mut nodes)?;
    let follower = nodes.remove(1);
    let first_term = follower.groups[&10].raw.raft.term;
    let store = follower.groups[&10].raw.store().durable_log().clone();
    let mut message = Message::default();
    message.set_msg_type(MessageType::MsgHeartbeat);
    message.from = 1;
    message.to = 2;
    message.term = first_term + 1;
    message.context = b"new-term-read".to_vec().into();
    let packet = Packet {
        cluster: follower.layout.cluster,
        layout: follower.layout_hash,
        policy: follower.policy,
        group: 10,
        message,
    };
    let (runtime, mut outbox) = Runtime::spawn(follower, options())?;
    let handle = runtime.handle();
    let gate = store.block_sync_for_test();
    let checked = executor.block_on(async {
        handle.step(packet)?;
        wait_for(|| {
            handle
                .observation(10)
                .is_ok_and(|o| o.persistence.is_some_and(|p| p.pending_readies > 0))
        })
        .await?;
        ensure!(
            timeout(Duration::from_millis(200), outbox.recv())
                .await
                .is_err(),
            "new-term heartbeat reply escaped before native persistence"
        );
        Ok::<_, anyhow::Error>(())
    });
    drop(gate);
    let verified = executor.block_on(async {
        checked?;
        let reply = timeout(Duration::from_secs(5), outbox.recv())
            .await?
            .context("outbox closed")?;
        ensure!(reply.packet().message.get_msg_type() == MessageType::MsgHeartbeatResponse);
        ensure!(reply.packet().message.term == first_term + 1);
        ensure!(reply.packet().message.context.as_ref() == b"new-term-read");
        Ok::<_, anyhow::Error>(())
    });
    let stopped = executor.block_on(runtime.shutdown());
    drop(outbox);
    drop(handle);
    drop(store);
    stopped?;
    verified?;
    let reopened = Node::open(&root.path().join("2"), layout(), 2, Limits::default())?;
    ensure!(reopened.groups[&10].raw.raft.term == first_term + 1);
    Ok(())
}

#[test]
fn follower_heartbeat_does_not_advertise_an_unsynced_commit() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let mut nodes = (1..=3)
        .map(|id| {
            Node::open(
                &root.path().join(id.to_string()),
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
    let first = nodes[0].groups[&10].state.applied;
    nodes[0].propose(10, 2, batch(2))?;
    let initial = nodes[0].poll()?;
    let mut pending: std::collections::VecDeque<_> = initial.messages.into();
    // Let the real leader and third replica commit the second record while
    // withholding all replication to replica two.
    while let Some(packet) = pending.pop_front() {
        if packet.message.to == 2 {
            continue;
        }
        let node = &mut nodes[(packet.message.to - 1) as usize];
        node.step(packet)?;
        pending.extend(node.poll()?.messages);
    }
    let second = nodes[0].groups[&10].state.applied;
    ensure!(second > first && nodes[0].read_accepted(10, second)? == Some(batch(2)));
    let term = nodes[0].groups[&10].raw.raft.term;
    let mut response = Message::default();
    response.set_msg_type(MessageType::MsgHeartbeatResponse);
    response.from = 2;
    response.to = 1;
    response.term = term;
    let packet = Packet {
        cluster: nodes[0].layout.cluster,
        layout: nodes[0].layout_hash,
        policy: nodes[0].policy,
        group: 10,
        message: response,
    };
    nodes[0].step(packet)?;
    let mut pending: std::collections::VecDeque<_> = nodes[0].poll()?.messages.into();
    let mut append = None;
    while let Some(packet) = pending.pop_front() {
        if packet.message.to == 2 && !packet.message.entries.is_empty() {
            append = Some(packet);
            break;
        }
        // Its first empty append can probe beyond the follower's old suffix.
        // Deliver the real rejection so Raft resends the missing original data.
        let node = &mut nodes[(packet.message.to - 1) as usize];
        node.step(packet)?;
        pending.extend(node.poll()?.messages);
    }
    let append = append.context("missing real catch-up append")?;
    ensure!(append.message.commit == second);
    let follower = nodes.remove(1);
    let store = follower.groups[&10].raw.store().durable_log().clone();
    let mut heartbeat = append.clone();
    heartbeat.message.set_msg_type(MessageType::MsgHeartbeat);
    heartbeat.message.entries.clear();
    heartbeat.message.commit = first;
    heartbeat.message.context = b"durable-commit-read".to_vec().into();
    let (runtime, mut outbox) = Runtime::spawn(follower, options())?;
    let handle = runtime.handle();
    let gate = store.block_sync_for_test();
    let checked = executor.block_on(async {
        handle.step(append)?;
        wait_for(|| store.appended_index_for_test().is_some_and(|i| i >= second)).await?;
        handle.step(heartbeat)?;
        let reply = timeout(Duration::from_millis(500), outbox.recv())
            .await?
            .context("outbox closed")?;
        let message = &reply.packet().message;
        ensure!(
            message.get_msg_type() == MessageType::MsgHeartbeatResponse,
            "data acknowledgment escaped held native sync"
        );
        ensure!(message.context.as_ref() == b"durable-commit-read");
        ensure!(
            message.commit <= first,
            "heartbeat advertised unsynced commit {}",
            message.commit
        );
        ensure!(handle.observation(10)?.status.applied == first);
        Ok::<_, anyhow::Error>(())
    });
    drop(gate);
    let applied = executor.block_on(wait_for(|| {
        handle
            .observation(10)
            .is_ok_and(|o| o.status.applied >= second)
    }));
    let stopped = executor.block_on(runtime.shutdown());
    drop(outbox);
    drop(handle);
    drop(store);
    stopped?;
    checked?;
    applied?;
    let reopened = Node::open(&root.path().join("2"), layout(), 2, Limits::default())?;
    ensure!(reopened.read_accepted(10, first)? == Some(batch(1)));
    ensure!(reopened.read_accepted(10, second)? == Some(batch(2)));
    Ok(())
}

#[test]
fn follower_checkpoint_drain_keeps_heartbeats_live_without_early_ack() -> Result<()> {
    follower_checkpoint_stall(false)
}

#[test]
fn follower_checkpoint_metadata_keeps_heartbeats_live_without_early_ack() -> Result<()> {
    follower_checkpoint_stall(true)
}

fn follower_checkpoint_stall(metadata: bool) -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let term = c.handles[&1].observation(10)?.status.term;
    let mut prepared = Vec::new();
    let mut stores = Vec::new();
    for id in [2, 3] {
        prepared.push(executor.block_on(async {
            let task = c.handles[&id]
                .prepare_checkpoint(10, uuid::Uuid::new_v4())?
                .await?;
            tokio::task::spawn_blocking(move || task.run()).await?
        })?);
        stores.push(executor.block_on(c.handles[&id].local(10, |node| {
            Ok(node.groups[&10].raw.store().durable_log().clone())
        })?)?);
    }
    let before: Vec<_> = stores.iter().map(|s| s.appended_index_for_test()).collect();
    let gates: Vec<_> = stores.iter().map(|s| s.block_sync_for_test()).collect();
    let held = executor.block_on(async {
        let mut prepared = prepared.into_iter();
        let mut checkpoints = Vec::new();
        if metadata {
            for id in [2, 3] {
                checkpoints.push(c.handles[&id].install_checkpoint(prepared.next().unwrap())?);
            }
            wait_for(|| stores.iter().all(|s| s.appended_checkpoint_for_test().ok().flatten() == Some(first))).await?;
        }
        let append = c.handles[&1].commit(Command::Append { group: 10, batch: batch(2) })?;
        if metadata {
            wait_for(|| [2, 3].iter().all(|id| c.handles[id].observation(10).is_ok_and(|o| o.persistence.is_some_and(|p| p.pending_entries > 0)))).await?;
        } else {
            wait_for(|| stores.iter().zip(&before).all(|(s, before)| s.appended_index_for_test() > *before)).await?;
            for id in [2, 3] {
                checkpoints.push(c.handles[&id].install_checkpoint(prepared.next().unwrap())?);
            }
        }
        // A later control round confirms that the actual install request has
        // is still pending behind its original native write, not a fabricated flag.
        for id in [2, 3] {
            let next = c.handles[&id].prepare_checkpoint(10, uuid::Uuid::new_v4())?;
            ensure!(timeout(Duration::from_millis(500), next).await?.as_ref().err()
                .is_some_and(|e| e.to_string().contains("node image builder/receiver busy")));
        }
        let mut direct_refusals = Vec::new();
        for id in [2, 3] {
            let result = c.handles[&id].local(10, move |node| {
                let mut message = Message::default();
                message.set_msg_type(MessageType::MsgHeartbeat);
                message.from = 1;
                message.to = id;
                message.term = term;
                message.commit = first;
                message.context = b"checkpoint-read-barrier".to_vec().into();
                let packet = Packet {
                    cluster: node.layout.cluster, layout: node.layout_hash,
                    policy: node.policy, group: 10, message,
                };
                Ok(node.step(packet).err().map(|e| e.to_string()))
            })?.await?;
            if let Some(error) = result { direct_refusals.push((id, error)); }
        }
        let deadline = Instant::now() + Duration::from_millis(2500);
        let mut first_leader_change = None;
        while Instant::now() < deadline {
            ensure!(checkpoints.iter().all(|p| p.0.is_empty()), "checkpoint ACK preceded original sync");
            for (id, handle) in c.handles.iter() {
                let view = handle.observation(10)?;
                ensure!(!view.stopped, "worker failed: {:?}", view.failure);
                ensure!(view.observed_at.elapsed() < Duration::from_millis(500), "owner stalled");
                if (view.status.leader != 1 || view.status.term != term) && first_leader_change.is_none() {
                    first_leader_change = Some((*id, view.status.leader, view.status.term));
                }
            }
            for id in [2, 3] {
                ensure!(c.handles[&id].observation(10)?.status.applied == first,
                    "unsynced follower applied the pending append");
            }
            sleep(Duration::from_millis(10)).await;
        }
        let rejections: Vec<_> = c.handles.iter().map(|(id, h)| (*id, h.peer_rejections())).collect();
        ensure!(rejections.iter().all(|(_, r)| r.other == r.pending_persistence && r.pending_heartbeats == 0),
            "checkpoint backpressure included heartbeat or unclassified failures: {rejections:?}");
        eprintln!("follower_checkpoint_stall: metadata={metadata} direct_refusals={direct_refusals:?} first_leader_change={first_leader_change:?} rejections={rejections:?} append_ready={}", !append.0.is_empty());
        let early_reply = !append.0.is_empty();
        Ok::<_, anyhow::Error>((append, checkpoints, direct_refusals, first_leader_change, early_reply))
    });
    // Release and join the original native writers even when the observed
    // liveness invariant failed. No new application request is submitted.
    drop(gates);
    let checked = executor.block_on(async {
        let (append, checkpoints, refusals, changed, early_reply) = held?;
        for checkpoint in checkpoints {
            timeout(Duration::from_secs(10), checkpoint).await??;
        }
        let outcome = timeout(Duration::from_secs(10), append).await?;
        eprintln!("follower_checkpoint_original_outcome: {outcome:?}");
        let index = match outcome {
            Ok(Outcome::Committed(Applied::Accepted { index })) => Some(index),
            Ok(Outcome::Unknown) => None,
            other => anyhow::bail!("unexpected original append result: {other:?}"),
        };
        if let Some(index) = index {
            wait_for(|| {
                c.handles
                    .values()
                    .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= index))
            })
            .await?;
        }
        Ok::<_, anyhow::Error>((index, refusals, changed, early_reply))
    });
    let stopped = executor.block_on(c.shutdown());
    drop(stores);
    stopped?;
    let (index, refusals, changed, early_reply) = checked?;
    let mut expected = vec![(first, 1)];
    if let Some(index) = index {
        expected.push((index, 2));
    }
    reopen(root.path(), &expected)?;
    ensure!(
        refusals.is_empty(),
        "checkpoint drain rejected a current leader heartbeat: {refusals:?}"
    );
    ensure!(
        changed.is_none(),
        "checkpoint drain lost stable leadership: {changed:?}"
    );
    ensure!(
        !early_reply,
        "append completed while both follower syncs were held"
    );
    ensure!(
        index.is_some(),
        "checkpoint drain lost the original append completion"
    );
    Ok(())
}
