use super::*;
use raft::eraftpb::MessageType;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_peer_ack_completes_before_another_slow_data_batch() -> Result<()> {
    queued_peer_ack(8, true).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn peer_prefix_limit_still_visits_data_before_later_ack() -> Result<()> {
    queued_peer_ack(2, false).await
}

async fn queued_peer_ack(batch_requests: usize, expected_first_poll: bool) -> Result<()> {
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
    nodes[0].propose(10, 1, batch(1))?;
    let warmup = index(
        crate::replication::tests::pump(&mut nodes)?
            .into_iter()
            .find(|c| c.request == 1)
            .unwrap()
            .outcome,
    );
    let (runtime, mut outbox) = Runtime::spawn(
        nodes.remove(0),
        Options {
            batch_requests,
            batch_time: Duration::from_millis(100),
            tick_interval: Duration::from_secs(1),
            ..Options::default()
        },
    )?;
    let h = runtime.handle();
    let append = h.commit(Command::Append {
        group: 10,
        batch: batch(2),
    })?;
    let read = h.query(10, query::Query::Accepted { after: 0, limit: 8 })?;

    // Both responses come from real followers. Keep the actual append ACK
    // behind duplicate read-heartbeat responses, without fabricating a quorum.
    let mut held = std::collections::VecDeque::new();
    let (append_frame, heartbeat) = timeout(Duration::from_secs(5), async {
        let (mut append_frame, mut heartbeat) = (None, None);
        while append_frame.is_none() || heartbeat.is_none() {
            let frame = outbox.recv().await.context("worker outbox closed")?;
            let p = frame.packet();
            if p.group == 10
                && p.message.to == 2
                && p.message.entries.last().is_some_and(|e| e.index > warmup)
                && append_frame.is_none()
            {
                append_frame = Some(frame);
            } else if p.group == 10
                && p.message.to == 3
                && p.message.get_msg_type() == MessageType::MsgHeartbeat
                && !p.message.context.is_empty()
                && heartbeat.is_none()
            {
                heartbeat = Some(frame);
            } else {
                held.push_back(frame);
            }
        }
        Ok::<_, anyhow::Error>((append_frame.unwrap(), heartbeat.unwrap()))
    })
    .await??;
    let proposed = append_frame.packet().message.entries.last().unwrap().index;
    nodes[0].step(append_frame.packet().clone())?;
    drop(append_frame);
    let mut deferred = nodes[0].poll()?.messages;
    let position = deferred
        .iter()
        .position(|p| {
            p.group == 10
                && p.message.get_msg_type() == MessageType::MsgAppendResponse
                && !p.message.reject
                && p.message.index == proposed
        })
        .context("missing durable follower append ACK")?;
    let append_ack = deferred.remove(position);
    let context = heartbeat.packet().message.context.clone();
    nodes[1].step(heartbeat.packet().clone())?;
    drop(heartbeat);
    let mut responses = nodes[1].poll()?.messages;
    let position = responses
        .iter()
        .position(|p| {
            p.group == 10
                && p.message.get_msg_type() == MessageType::MsgHeartbeatResponse
                && p.message.context == context
        })
        .context("missing actual follower heartbeat response")?;
    let heartbeat_ack = responses.remove(position);
    deferred.extend(responses);

    let (entered, resume) = hold_worker(&h)?;
    entered.recv_timeout(Duration::from_secs(5))?;
    let before_quorum = append.0.is_empty();
    for _ in 0..2 {
        h.step(heartbeat_ack.clone())?;
    }
    h.step(append_ack)?;
    let (first_data, resume_first_data) = hold_lane(&h, true)?;
    let (second_data, resume_second_data) = hold_lane(&h, true)?;
    // Consume this batch's control budget independently of CPU/payload speed.
    sleep(Duration::from_millis(120)).await;
    drop(resume);
    first_data.recv_timeout(Duration::from_secs(5))?;
    let before_poll = append.0.is_empty();
    drop(resume_first_data);
    second_data.recv_timeout(Duration::from_secs(5))?;
    let completed_after_first_poll = !append.0.is_empty();
    drop(resume_second_data);

    // Finish the original operations and reopen all engines before evaluating
    // the scheduling expectation, including when the unchanged control fails.
    for packet in deferred {
        h.step(packet)?;
    }
    timeout(Duration::from_secs(10), async {
        loop {
            let outgoing = if let Some(frame) = held.pop_front() {
                Some(frame)
            } else {
                timeout(Duration::from_millis(50), outbox.recv())
                    .await
                    .ok()
                    .flatten()
            };
            if let Some(frame) = outgoing {
                let packet = frame.packet();
                nodes[packet.message.to as usize - 2].step(packet.clone())?;
            }
            for node in &mut nodes {
                for packet in node.poll()?.messages {
                    h.step(packet)?;
                }
            }
            if !append.0.is_empty()
                && !read.0.is_empty()
                && nodes
                    .iter()
                    .all(|n| n.status(10).unwrap().applied >= proposed)
            {
                return Ok::<_, anyhow::Error>(());
            }
        }
    })
    .await??;
    let committed = index(append.await?);
    assert_eq!(committed, proposed);
    let query::Outcome::Observed {
        cut,
        value: query::Value::Accepted(page),
    } = read.await?
    else {
        anyhow::bail!("missing quorum read");
    };
    assert!(cut.applied >= warmup && cut.applied >= cut.quorum_index);
    assert!(page.entries.contains(&(warmup, batch(1).meta)));
    runtime.shutdown().await?;
    drop(outbox);
    drop(nodes);
    for id in 1..=3 {
        let node = Node::open(
            &directory.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        assert_eq!(node.read_accepted(10, warmup)?, Some(batch(1)));
        assert_eq!(node.read_accepted(10, committed)?, Some(batch(2)));
    }
    println!(
        "PEER_PREFIX_EXPERIMENT {}",
        serde_json::json!({
            "batch_requests": batch_requests,
            "queued_peer_messages": 3,
            "before_quorum_unacknowledged": before_quorum,
            "before_poll_unacknowledged": before_poll,
            "completed_after_first_poll": completed_after_first_poll,
            "expected_first_poll": expected_first_poll,
            "reopened_replicas": 3,
            "scope": "controlled durable Rust scheduling test, not network or Lance throughput",
        })
    );
    assert!(before_quorum && before_poll);
    assert_eq!(completed_after_first_poll, expected_first_poll);
    Ok(())
}
