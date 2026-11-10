use super::*;

mod peers;

fn hold_worker(handle: &Handle) -> Result<(mpsc::Receiver<()>, mpsc::Sender<()>)> {
    hold_lane(handle, false)
}

fn hold_lane(handle: &Handle, data: bool) -> Result<(mpsc::Receiver<()>, mpsc::Sender<()>)> {
    let (entered, arrived) = mpsc::channel();
    let (resume, resumed) = mpsc::channel::<()>();
    handle.enqueue(
        handle.route(10)?,
        Work::Local(Box::new(move |_| {
            entered.send(()).unwrap();
            let _ = resumed.recv();
            Box::new(|| {})
        })),
        data,
    )?;
    Ok((arrived, resume))
}

async fn worker_polled(handle: &Handle) -> Result<()> {
    let (sent, received) = oneshot::channel();
    handle.enqueue(
        handle.route(10)?,
        Work::Local(Box::new(move |_| {
            Box::new(move || {
                let _ = sent.send(());
            })
        })),
        false,
    )?;
    timeout(Duration::from_secs(5), received).await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn append_intervals_separate_worker_queue_from_quorum_and_survive_caller_cancellation(
) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = Cluster::open(
        directory.path(),
        Options {
            tick_interval: Duration::from_secs(1),
            ..Options::default()
        },
    )?;
    c.elect(1, 10).await?;
    let h = &c.handles[&1];
    let warmup = index(
        h.commit(Command::Append {
            group: 10,
            batch: batch(1),
        })?
        .await?,
    );
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= warmup)
    })
    .await?;
    let before = h.observation(10)?.append.accepted;
    assert_eq!(before.completed, 1);

    c.isolated.store(1, Ordering::Release);
    let (entered, resume) = hold_worker(h)?;
    entered.recv_timeout(Duration::from_secs(5))?;
    let pending = h.commit(Command::Append {
        group: 10,
        batch: batch(2),
    })?;
    let queued_since = Instant::now();
    sleep(Duration::from_millis(20)).await;
    let queued_micros = queued_since.elapsed().as_micros() as u64;
    assert_eq!(h.observation(10)?.append.accepted.completed, 1);
    assert!(pending.0.is_empty());
    drop(resume);
    worker_polled(h).await?;
    // The local proposal and persistence have finished, but no follower can
    // acknowledge it. This interval belongs to completion, not the queue.
    let quorum_since = Instant::now();
    sleep(Duration::from_millis(20)).await;
    let quorum_micros = quorum_since.elapsed().as_micros() as u64;
    assert!(pending.0.is_empty(), "append bypassed quorum persistence");
    assert_eq!(h.observation(10)?.append.accepted.completed, 1);
    c.isolated.store(0, Ordering::Release);
    let second = index(timeout(Duration::from_secs(10), pending).await??);
    let measured = h.observation(10)?.append.accepted;
    assert_eq!(measured.completed, 2);
    assert!(measured.queue_micros - before.queue_micros >= queued_micros);
    assert!(measured.completion_micros - before.completion_micros >= quorum_micros);

    // Durable sequence rejection and a pre-proposal follower error must not
    // dilute accepted-request timings.
    assert!(matches!(
        h.commit(Command::Append {
            group: 10,
            batch: batch(4)
        })?
        .await?,
        Outcome::Committed(Applied::Rejected { .. })
    ));
    assert_eq!(h.observation(10)?.append.accepted.completed, 2);
    assert_eq!(h.observation(10)?.append.committed_other.completed, 1);
    assert!(c.handles[&2]
        .commit(Command::Append {
            group: 10,
            batch: batch(3)
        })?
        .await
        .is_err());
    assert_eq!(
        c.handles[&2]
            .observation(10)?
            .append
            .proposal_error
            .completed,
        1
    );

    // Once queued, a dropped caller does not cancel the underlying append.
    let (entered, resume) = hold_worker(h)?;
    entered.recv_timeout(Duration::from_secs(5))?;
    drop(h.commit(Command::Append {
        group: 10,
        batch: batch(3),
    })?);
    drop(resume);
    wait_for(|| {
        let observed = h.observation(10).unwrap();
        observed.append.accepted.completed == 3 && observed.status.accepted_records == 3
    })
    .await?;
    let third = h.observation(10)?.status.applied;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= third)
    })
    .await?;
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &directory.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        assert_eq!(node.read_accepted(10, warmup)?, Some(batch(1)));
        assert_eq!(node.read_accepted(10, second)?, Some(batch(2)));
        assert_eq!(node.read_accepted(10, third)?, Some(batch(3)));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_reads_share_one_quorum_before_slow_data_without_reusing_it_for_later_reads(
) -> Result<()> {
    use raft::eraftpb::MessageType;

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
            batch_requests: 32,
            batch_time: Duration::from_millis(10),
            tick_interval: Duration::from_secs(1),
            ..Options::default()
        },
    )?;
    let h = runtime.handle();
    let (entered, resume) = hold_worker(&h)?;
    entered.recv_timeout(Duration::from_secs(5))?;
    let mut reads = (0..16)
        .map(|_| h.query(10, query::Query::Accepted { after: 0, limit: 1 }))
        .collect::<Result<Vec<_>>>()?;
    // The initial control gate shares a round with the first data gate. The
    // second data gate consumes the *next* round's budget, independently of
    // payload size and CPU speed. All queries are already queued at this point.
    let (first_data, resume_first_data) = hold_lane(&h, true)?;
    let (second_data, resume_second_data) = hold_lane(&h, true)?;
    sleep(Duration::from_millis(20)).await;
    drop(resume);
    first_data.recv_timeout(Duration::from_secs(5))?;
    drop(resume_first_data);
    second_data.recv_timeout(Duration::from_secs(5))?;
    sleep(Duration::from_millis(20)).await;
    let before_quorum = reads.iter().all(|r| r.0.is_empty());
    drop(resume_second_data);

    // Deliver only the first nonempty read context to one real follower. Keep
    // all other frames, so no later quorum response can satisfy this sample.
    let mut held = std::collections::VecDeque::new();
    let first = timeout(Duration::from_secs(5), async {
        loop {
            let frame = outbox.recv().await.context("worker outbox closed")?;
            let p = frame.packet();
            if p.group == 10
                && p.message.to == 2
                && p.message.get_msg_type() == MessageType::MsgHeartbeat
                && !p.message.context.is_empty()
            {
                return Ok::<_, anyhow::Error>(frame);
            }
            held.push_back(frame);
        }
    })
    .await??;
    let context = first.packet().message.context.clone();
    nodes[0].step(first.packet().clone())?;
    drop(first);
    let mut responses = 0;
    for packet in nodes[0].poll()?.messages {
        ensure!(packet.message.to == 1);
        if packet.message.get_msg_type() == MessageType::MsgHeartbeatResponse
            && packet.message.context == context
        {
            responses += 1;
        }
        h.step(packet)?;
    }
    ensure!(responses == 1);
    wait_for(|| !reads[0].0.is_empty()).await?;
    worker_polled(&h).await?;
    let first_quorum_completed = reads.iter().filter(|r| !r.0.is_empty()).count();
    let later = h.query(10, query::Query::Accepted { after: 0, limit: 1 })?;
    worker_polled(&h).await?;
    let later_unacknowledged = later.0.is_empty();
    reads.push(later);
    let append = h.commit(Command::Append {
        group: 10,
        batch: batch(2),
    })?;
    let append_unacknowledged = append.0.is_empty();

    // Finish the original operations, shut down, and reopen every engine
    // before comparing schedulers. A failed performance expectation must not
    // leave blocked workers or unresolved writes behind.
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
                    ensure!(packet.message.to == 1);
                    h.step(packet)?;
                }
            }
            if reads.iter().all(|r| !r.0.is_empty())
                && !append.0.is_empty()
                && nodes.iter().all(|n| n.status(10).unwrap().applied > warmup)
            {
                return Ok::<_, anyhow::Error>(());
            }
        }
    })
    .await??;
    let committed = index(append.await?);
    for read in reads {
        let query::Outcome::Observed {
            cut,
            value: query::Value::Accepted(page),
        } = read.await?
        else {
            anyhow::bail!("missing quorum result");
        };
        assert!(cut.applied >= warmup && cut.applied >= cut.quorum_index);
        assert_eq!(page.entries, vec![(warmup, batch(1).meta)]);
    }
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
        "CONTROL_PREFIX_EXPERIMENT {}",
        serde_json::json!({
            "queued_reads": 16,
            "first_quorum_completed": first_quorum_completed,
            "before_quorum_unacknowledged": before_quorum,
            "later_read_unacknowledged": later_unacknowledged,
            "append_before_relay_unacknowledged": append_unacknowledged,
            "reopened_replicas": 3,
            "scope": "controlled native scheduling experiment; not network or Lance throughput"
        })
    );
    assert!(before_quorum && later_unacknowledged && append_unacknowledged);
    assert_eq!(
        first_quorum_completed, 16,
        "queued reads split by slow data"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_batch_polls_before_the_next_control_request_and_preserves_durable_ack() -> Result<()>
{
    let directory = tempfile::tempdir()?;
    let c = Cluster::open(
        directory.path(),
        Options {
            batch_requests: 32,
            batch_time: Duration::from_millis(1),
            tick_interval: Duration::from_secs(1),
            ..Options::default()
        },
    )?;
    c.elect(1, 10).await?;
    let h = &c.handles[&1];
    assert_eq!(c.nonempty_entry_packets.load(Ordering::Acquire), 0);

    // The first control operation occupies the current drain round. Queue an
    // append in its data lane, followed by another control operation. Holding
    // the first operation gives a deterministic slow round without assuming
    // CPU speed or adding a production delay.
    let (first_entered, first_resume) = hold_worker(h)?;
    first_entered.recv_timeout(Duration::from_secs(5))?;
    let pending = h.commit(Command::Append {
        group: 10,
        batch: batch(1),
    })?;
    let (second_entered, second_resume) = hold_worker(h)?;
    sleep(Duration::from_millis(20)).await;
    drop(first_resume);
    second_entered.recv_timeout(Duration::from_secs(5))?;

    // The first append must have crossed local persistence and entered actual
    // replica transport before the next slow request can occupy the worker.
    // Keep this result until after cleanup so the old behavior fails without
    // leaving a blocked thread or an unresolved mutation behind.
    let emitted = wait_for(|| c.nonempty_entry_packets.load(Ordering::Acquire) > 0).await;
    let unacknowledged = pending.0.is_empty();
    drop(second_resume);
    let committed = index(timeout(Duration::from_secs(10), pending).await??);
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= committed)
    })
    .await?;
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &directory.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        assert_eq!(node.read_accepted(10, committed)?, Some(batch(1)));
    }
    assert!(unacknowledged, "append bypassed quorum/apply completion");
    assert!(
        emitted.is_ok(),
        "slow drain round delayed replication behind the next control request"
    );
    Ok(())
}
