use super::*;

enum Pressure {
    NotReady,
    Bytes,
    Queue,
    Closed,
}

// Exercise the real dispatcher and both ownership budgets. A rejected packet
// must release runtime credits; a full/closed queue must also return its newly
// acquired destination credits. Accepted payloads retain the original Bytes.
async fn observe(pressure: Pressure) -> Result<()> {
    let body = Bytes::from(vec![0xa5; 1024]);
    let mut message = Message {
        from: 1,
        to: 2,
        term: 1,
        ..Default::default()
    };
    message.set_msg_type(MessageType::MsgAppend);
    message.set_entries(
        vec![Entry {
            index: 1,
            term: 1,
            data: body.clone(),
            ..Default::default()
        }]
        .into(),
    );
    let charge = message.compute_size() as usize + 256;
    let credits = super::super::super::Credits::new(3, charge * 3);
    let capacity = if matches!(pressure, Pressure::Bytes) {
        charge
    } else {
        charge * 3
    };
    let bytes = Arc::new(Semaphore::new(capacity));
    let (send, mut receive) = mpsc::channel(1);
    if matches!(pressure, Pressure::Closed) {
        receive.close();
    }
    let route = Route {
        send,
        ready: Arc::new(AtomicBool::new(!matches!(pressure, Pressure::NotReady))),
        bytes: bytes.clone(),
    };
    let (outbox, incoming) = mpsc::channel(3);
    for group in [10, 20, 10] {
        outbox
            .send(Outbound {
                packet: Packet {
                    cluster: uuid::Uuid::nil(),
                    layout: [0; 32],
                    policy: [0; 32],
                    group,
                    message: message.clone(),
                },
                // Runtime admission adds its own per-frame overhead.
                _credit: Some(credits.reserve(message.compute_size() as usize)?),
                image: None,
                snapshot_feedback: None,
            })
            .await
            .map_err(|_| anyhow::anyhow!("test outbox closed"))?;
    }
    drop(outbox);
    let (_stop, stopped) = watch::channel(false);
    let statistics = Arc::new(Statistics::default());
    let mut failures = Vec::new();
    dispatch(
        incoming,
        BTreeMap::from([(2, route)]),
        Arc::new(OnceLock::new()),
        stopped,
        statistics.clone(),
        |failed| failures.push(failed),
    )
    .await?;
    let counts = (
        statistics.enqueue_not_ready.load(Ordering::Relaxed),
        statistics.enqueue_byte_capacity.load(Ordering::Relaxed),
        statistics.enqueue_queue_full.load(Ordering::Relaxed),
        statistics.enqueue_queue_closed.load(Ordering::Relaxed),
    );
    let (expected, accepted) = match pressure {
        Pressure::NotReady => ((3, 0, 0, 0), false),
        Pressure::Bytes => ((0, 2, 0, 0), true),
        Pressure::Queue => ((0, 0, 2, 0), true),
        Pressure::Closed => ((0, 0, 0, 3), false),
    };
    assert_eq!(counts, expected);
    let rejected_groups = if accepted {
        vec![20, 10]
    } else {
        vec![10, 20, 10]
    };
    assert_eq!(
        failures,
        rejected_groups
            .into_iter()
            .map(|group| FailedAppend {
                group,
                to: 2,
                term: 1
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(
        statistics.dropped_outbound.load(Ordering::Relaxed),
        if accepted { 2 } else { 3 }
    );
    assert_eq!(credits.observation().used_requests, usize::from(accepted));
    assert_eq!(
        credits.observation().used_bytes,
        charge * usize::from(accepted)
    );
    assert_eq!(
        bytes.available_permits(),
        capacity - charge * usize::from(accepted)
    );
    if accepted {
        let queued = receive.try_recv()?;
        assert_eq!(
            queued.packet.packet().message.entries[0].data.as_ptr(),
            body.as_ptr()
        );
        drop(queued);
    }
    assert!(receive.try_recv().is_err());
    assert_eq!(credits.observation().used_requests, 0);
    assert_eq!(credits.observation().used_bytes, 0);
    assert_eq!(bytes.available_permits(), capacity);
    Ok(())
}

#[tokio::test]
async fn peer_enqueue_observes_not_ready_and_releases_runtime_credits() -> Result<()> {
    observe(Pressure::NotReady).await
}

#[tokio::test]
async fn peer_enqueue_observes_byte_pressure_separately_from_queue_count() -> Result<()> {
    observe(Pressure::Bytes).await
}

#[tokio::test]
async fn peer_enqueue_observes_full_queue_and_returns_destination_credits() -> Result<()> {
    observe(Pressure::Queue).await
}

#[tokio::test]
async fn peer_enqueue_observes_closed_queue_and_returns_destination_credits() -> Result<()> {
    observe(Pressure::Closed).await
}

#[tokio::test]
async fn rejected_append_returns_to_probe_and_reopens_the_acknowledged_record() -> Result<()> {
    use crate::replication::tests::pump;
    use raft::ProgressState;

    let directory = tempfile::tempdir()?;
    let open = || {
        (1..=3)
            .map(|id| {
                Node::open(
                    &directory.path().join(id.to_string()),
                    layout(),
                    id,
                    Limits::default(),
                )
            })
            .collect::<Result<Vec<_>>>()
    };
    let mut nodes = open()?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    let expected = batch(1);
    nodes[0].propose(10, 7, expected.clone())?;
    let mut events = nodes[0].poll()?;
    assert!(events.completions.is_empty());
    let selected = events
        .messages
        .iter()
        .position(|p| {
            p.group == 10 && p.message.to == 2 && p.message.get_msg_type() == MessageType::MsgAppend
        })
        .unwrap();
    let packet = events.messages.remove(selected);
    let failure = FailedAppend::capture(&packet).unwrap();
    assert_eq!(
        nodes[0].groups[&10].raw.raft.prs().get(2).unwrap().state,
        ProgressState::Replicate
    );

    // The real dispatcher rejects a native Ready packet at its byte limit.
    // Feed the resulting failure back before delivering the healthy voter ACK.
    let (send, incoming) = mpsc::channel(1);
    send.send(Outbound {
        packet,
        _credit: None,
        image: None,
        snapshot_feedback: None,
    })
    .await
    .map_err(|_| anyhow::anyhow!("test outbox closed"))?;
    drop(send);
    let (send, _receive) = mpsc::channel(1);
    let route = Route {
        send,
        ready: Arc::new(AtomicBool::new(true)),
        bytes: Arc::new(Semaphore::new(0)),
    };
    let (_stop, stopped) = watch::channel(false);
    let statistics = Arc::new(Statistics::default());
    let mut failures = Vec::new();
    dispatch(
        incoming,
        BTreeMap::from([(2, route)]),
        Arc::new(OnceLock::new()),
        stopped,
        statistics.clone(),
        |failed| failures.push(failed),
    )
    .await?;
    assert_eq!(statistics.enqueue_byte_capacity.load(Ordering::Relaxed), 1);
    assert_eq!(failures, vec![failure]);
    failure.apply(&mut nodes[0])?;
    assert_eq!(
        nodes[0].groups[&10].raw.raft.prs().get(2).unwrap().state,
        ProgressState::Probe
    );
    assert!(nodes[0].poll()?.completions.is_empty());
    for packet in events.messages {
        nodes[packet.message.to as usize - 1].step(packet)?;
    }
    let completed = pump(&mut nodes)?;
    let Outcome::Committed(Applied::Accepted { index }) = completed
        .into_iter()
        .find(|c| c.request == 7)
        .unwrap()
        .outcome
    else {
        anyhow::bail!("healthy quorum did not acknowledge the original record");
    };
    for _ in 0..10 {
        nodes[0].tick()?;
        pump(&mut nodes)?;
    }
    for node in &nodes {
        assert!(node.read_accepted(10, index)?.unwrap() == expected);
    }

    // Reopen as followers and elect a fresh term through the supported peer
    // protocol. Delayed old-term feedback must not reset the new leader.
    drop(nodes);
    nodes = open()?;
    assert_eq!(
        nodes[0].groups[&10].raw.raft.state,
        raft::StateRole::Follower
    );
    failure.apply(&mut nodes[0])?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    assert!(nodes[0].groups[&10].raw.raft.term > failure.term);
    assert_eq!(
        nodes[0].groups[&10].raw.raft.prs().get(2).unwrap().state,
        ProgressState::Replicate
    );
    failure.apply(&mut nodes[0])?;
    assert_eq!(
        nodes[0].groups[&10].raw.raft.prs().get(2).unwrap().state,
        ProgressState::Replicate
    );
    drop(nodes);
    for node in open()? {
        assert!(node.read_accepted(10, index)?.unwrap() == expected);
    }
    Ok(())
}

#[test]
fn append_feedback_excludes_quorum_responses_and_snapshot_delivery() {
    for kind in [
        MessageType::MsgAppendResponse,
        MessageType::MsgHeartbeat,
        MessageType::MsgHeartbeatResponse,
        MessageType::MsgSnapshot,
        MessageType::MsgReadIndexResp,
    ] {
        let mut message = Message::default();
        message.set_msg_type(kind);
        let packet = Packet {
            cluster: uuid::Uuid::nil(),
            layout: [0; 32],
            policy: [0; 32],
            group: 10,
            message,
        };
        assert_eq!(FailedAppend::capture(&packet), None);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejection_burst_leaves_control_capacity_for_quorum_reads() -> Result<()> {
    use crate::replication::{query, runtime::Work};

    let directory = tempfile::tempdir()?;
    let c = Cluster::open(directory.path()).await?;
    c.nodes[&1].handle.campaign(10)?.await?;
    let leader = c.leader(10).await?;
    let handle = c.nodes[&leader].handle.clone();
    let destination = (1..=3).find(|&id| id != leader).unwrap();
    let term = handle.observation(10)?.status.term;
    let (entered, arrived) = tokio::sync::oneshot::channel();
    let (release, resumed) = std::sync::mpsc::channel::<()>();
    handle.enqueue(
        handle.route(10)?,
        Work::Local(Box::new(move |_| {
            let _ = entered.send(());
            let _ = resumed.recv();
            Box::new(|| {})
        })),
        false,
    )?;
    timeout(Duration::from_secs(5), arrived).await??;

    let (outbox, incoming) = mpsc::channel(4096);
    for i in 0..4096 {
        let mut message = Message {
            from: leader,
            to: destination,
            term,
            ..Default::default()
        };
        message.set_msg_type(MessageType::MsgAppend);
        outbox
            .try_send(Outbound {
                packet: Packet {
                    cluster: uuid::Uuid::nil(),
                    layout: [0; 32],
                    policy: [0; 32],
                    group: if i % 2 == 0 { 10 } else { 20 },
                    message,
                },
                _credit: None,
                image: None,
                snapshot_feedback: None,
            })
            .map_err(|_| anyhow::anyhow!("test outbox full"))?;
    }
    drop(outbox);
    let (send, _receive) = mpsc::channel(1);
    let route = Route {
        send,
        ready: Arc::new(AtomicBool::new(false)),
        bytes: Arc::new(Semaphore::new(4096)),
    };
    let (_stop, stopped) = watch::channel(false);
    let statistics = Arc::new(Statistics::default());
    let mut feedback = AppendFeedback {
        handle: handle.clone(),
        pending: None,
    };
    dispatch(
        incoming,
        BTreeMap::from([(destination, route)]),
        Arc::new(OnceLock::new()),
        stopped,
        statistics.clone(),
        |failed| feedback.report(failed),
    )
    .await?;
    assert_eq!(statistics.enqueue_not_ready.load(Ordering::Relaxed), 4096);
    assert_eq!(handle.shared.control.observation().used_requests, 1);

    // Keep the worker paused while admitting real read-index requests. They
    // cannot free credits early and hide feedback occupying the control lane.
    let mut reads = Vec::new();
    for _ in 1..64 {
        reads.push(handle.query(10, query::Query::Accepted { after: 0, limit: 1 })?);
    }
    assert_eq!(handle.shared.control.observation().used_requests, 64);
    drop(release);
    for pending in reads {
        assert!(matches!(
            timeout(Duration::from_secs(10), pending).await??,
            query::Outcome::Observed { .. }
        ));
    }
    wait_for(|| !feedback.pending.as_ref().unwrap().0.is_empty()).await?;
    // Completed replies retain their admission credit until consumed too.
    assert_eq!(handle.shared.control.observation().used_requests, 1);

    // Completion permits a later hint, so limiting the burst does not disable
    // feedback permanently. This result is still delivered after durable poll.
    feedback.report(FailedAppend {
        group: 10,
        to: destination,
        term,
    });
    timeout(Duration::from_secs(10), feedback.pending.take().unwrap()).await??;
    assert_eq!(handle.shared.control.observation().used_requests, 0);
    c.shutdown().await?;
    Ok(())
}
