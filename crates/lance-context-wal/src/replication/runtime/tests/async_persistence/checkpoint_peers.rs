//! Replication can use existing bounded capacity while old Ready writes drain.
use super::*;

#[test]
fn checkpoint_drain_accepts_replication_without_advancing_durability() -> Result<()> {
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
        let original = c.handles[&1].commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        wait_for(|| {
            stores
                .iter()
                .zip(&before)
                .all(|(s, old)| s.appended_index_for_test() > *old)
        })
        .await?;
        let mut checkpoints = Vec::new();
        for (id, image) in [2, 3].into_iter().zip(prepared) {
            checkpoints.push(c.handles[&id].install_checkpoint(image)?);
            // Ordered control work proves the real install is draining.
            ensure!(
                !c.handles[&id]
                    .local(10, |node| node.groups[&10].accepts_async(&[]))?
                    .await?
            );
        }
        let pending_entries: Vec<_> = [2, 3]
            .into_iter()
            .map(|id| {
                Ok(c.handles[&id]
                    .observation(10)?
                    .persistence
                    .context("persistence missing")?
                    .pending_entries)
            })
            .collect::<Result<_>>()?;
        let next = c.handles[&1].commit(Command::Append {
            group: 10,
            batch: batch(3),
        })?;
        let deadline = Instant::now() + Duration::from_millis(2500);
        while Instant::now() < deadline {
            ensure!(
                original.0.is_empty() && next.0.is_empty(),
                "ACK preceded quorum sync"
            );
            ensure!(
                checkpoints.iter().all(|p| p.0.is_empty()),
                "checkpoint completed before sync"
            );
            for h in c.handles.values() {
                let view = h.observation(10)?;
                ensure!(!view.stopped, "owner failed: {:?}", view.failure);
                ensure!(
                    view.observed_at.elapsed() < Duration::from_millis(500),
                    "owner blocked"
                );
                ensure!(
                    view.status.leader == 1 && view.status.term == term,
                    "lost stable leader"
                );
            }
            for (offset, id) in [2, 3].into_iter().enumerate() {
                let view = c.handles[&id].observation(10)?;
                ensure!(
                    view.status.applied == first,
                    "follower applied unsynced input"
                );
                ensure!(
                    view.persistence
                        .context("persistence missing")?
                        .pending_entries
                        == pending_entries[offset],
                    "draining checkpoint collected another Ready"
                );
            }
            sleep(Duration::from_millis(10)).await;
        }
        let mut buffered = Vec::new();
        for id in [2, 3] {
            buffered.push(
                c.handles[&id]
                    .local(10, move |node| {
                        ensure!(node.read_accepted(10, first)? == Some(batch(1)));
                        let g = &node.groups[&10];
                        Ok(g.raw
                            .raft
                            .raft_log
                            .unstable
                            .entries
                            .iter()
                            .any(|e| e.index > first + 1))
                    })?
                    .await?,
            );
        }
        let refusals: Vec<_> = c
            .handles
            .iter()
            .map(|(id, h)| (*id, h.peer_rejections()))
            .collect();
        eprintln!("checkpoint peer admission: buffered={buffered:?} refusals={refusals:?}");
        Ok::<_, anyhow::Error>((original, next, checkpoints, buffered, refusals))
    });
    // Always release original sync operations; the baseline must also drain
    // and reopen every acknowledged value before reporting the regression.
    drop(gates);
    let mut expected = vec![(first, 1)];
    let checked = executor.block_on(async {
        let (original, next, checkpoints, buffered, refusals) = held?;
        for checkpoint in checkpoints {
            timeout(Duration::from_secs(10), checkpoint).await??;
        }
        for (request, sequence) in [(original, 2), (next, 3)] {
            let result = timeout(Duration::from_secs(10), request).await??;
            let Outcome::Committed(Applied::Accepted { index }) = result else {
                anyhow::bail!("original request did not commit");
            };
            expected.push((index, sequence));
        }
        let last = expected.last().unwrap().0;
        wait_for(|| {
            c.handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= last))
        })
        .await?;
        for id in [2, 3] {
            ensure!(c.handles[&id].observation(10)?.checkpoint_index == first);
        }
        Ok::<_, anyhow::Error>((buffered, refusals))
    });
    let stopped = executor.block_on(c.shutdown());
    drop(stores);
    stopped?;
    reopen(root.path(), &expected)?;
    let (buffered, refusals) = checked?;
    ensure!(buffered.iter().all(|present| *present) && refusals.iter().all(|(_, r)| r.other == 0 && r.disk_budget == 0),
        "checkpoint drain refused replication despite available capacity: buffered={buffered:?} refusals={refusals:?}");
    Ok(())
}

#[test]
fn checkpoint_drain_accepts_durable_responses_without_early_leader_ack() -> Result<()> {
    use raft_proto::eraftpb::MessageType;
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
    let first = nodes[0].status(10)?.applied;
    let leader = nodes.remove(0);
    let image = leader.prepare_checkpoint(10, uuid::Uuid::new_v4())?.run()?;
    let store = leader.groups[&10].raw.store().durable_log().clone();
    let before = store.appended_index_for_test();
    let (runtime, mut outbox) = Runtime::spawn(leader, options())?;
    let handle = runtime.handle();
    let gate = store.block_sync_for_test();
    let held = executor.block_on(async {
        let request = handle.commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        let mut responses = BTreeMap::new();
        // Hold the actual follower responses until the leader's checkpoint
        // drain starts. Each response came from a completed native write.
        timeout(Duration::from_secs(5), async {
            while responses.len() < 2 {
                let outgoing = outbox.recv().await.context("outbox closed")?;
                let packet = outgoing.packet().clone();
                let follower = &mut nodes[(packet.message.to - 2) as usize];
                follower.step(packet)?;
                for response in follower.poll()?.messages {
                    if response.message.get_msg_type() == MessageType::MsgAppendResponse
                        && !response.message.reject
                        && response.message.index > first
                    {
                        responses.insert(response.message.from, response);
                    } else {
                        handle.step(response)?;
                    }
                }
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        wait_for(|| store.appended_index_for_test() > before).await?;
        let checkpoint = handle.install_checkpoint(image)?;
        ensure!(
            !handle
                .local(10, |node| node.groups[&10].accepts_async(&[]))?
                .await?
        );
        let index = responses.values().next().unwrap().message.index;
        for response in responses.into_values() {
            handle.step(response)?;
        }
        // Peer-lane work may follow local control in this poll. A subsequent
        // control round observes both responses after they have been visited.
        sleep(Duration::from_millis(100)).await;
        let matched = handle
            .local(10, move |node| {
                let g = &node.groups[&10];
                Ok([2, 3]
                    .into_iter()
                    .all(|id| g.raw.raft.prs().get(id).is_some_and(|p| p.matched >= index)))
            })?
            .await?;
        ensure!(
            request.0.is_empty() && checkpoint.0.is_empty(),
            "leader ACK preceded its own sync"
        );
        ensure!(handle.observation(10)?.status.applied == first);
        let refusals = handle.peer_rejections();
        eprintln!("checkpoint durable responses: matched={matched} refusals={refusals:?}");
        Ok::<_, anyhow::Error>((request, checkpoint, index, matched, refusals))
    });
    drop(gate);
    let checked = executor.block_on(async {
        let (request, checkpoint, index, matched, refusals) = held?;
        timeout(Duration::from_secs(10), async {
            while handle.observation(10)?.status.applied < index
                || nodes
                    .iter()
                    .any(|n| n.status(10).is_ok_and(|s| s.applied < index))
            {
                let outgoing = outbox.recv().await.context("outbox closed")?;
                let packet = outgoing.packet().clone();
                let follower = &mut nodes[(packet.message.to - 2) as usize];
                follower.step(packet)?;
                for response in follower.poll()?.messages {
                    handle.step(response)?;
                }
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        let result = timeout(Duration::from_secs(5), request).await??;
        ensure!(
            matches!(result, Outcome::Committed(Applied::Accepted { index: got }) if got == index)
        );
        timeout(Duration::from_secs(5), checkpoint).await??;
        Ok::<_, anyhow::Error>((index, matched, refusals))
    });
    let stopped = executor.block_on(runtime.shutdown());
    drop(outbox);
    drop(handle);
    drop(store);
    drop(nodes);
    stopped?;
    let (index, matched, refusals) = checked?;
    reopen(root.path(), &[(first, 1), (index, 2)])?;
    ensure!(
        matched && refusals.pending_persistence == 0,
        "checkpoint discarded durable follower responses: matched={matched} refusals={refusals:?}"
    );
    Ok(())
}

#[test]
fn checkpoint_peer_input_stays_bounded_and_healthy_quorum_progresses() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let opts = Options {
        data_bytes: 32 << 20,
        peer_bytes: 28 << 20,
        ..options()
    };
    let window_bytes = opts.data_bytes + opts.peer_bytes + opts.control_bytes;
    let c = executor.block_on(async {
        let c = Cluster::open(root.path(), opts)?;
        c.elect(1, 10).await?;
        Ok::<_, anyhow::Error>(c)
    })?;
    let first = executor.block_on(commit(&c.handles[&1], 10, batch(1)))?;
    executor.block_on(wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= first))
    }))?;
    let prepared = executor.block_on(async {
        let task = c.handles[&3]
            .prepare_checkpoint(10, uuid::Uuid::new_v4())?
            .await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(
        c.handles[&3].local(10, |n| Ok(n.groups[&10].raw.store().durable_log().clone()))?,
    )?;
    let before = store.appended_index_for_test();
    let term = c.handles[&1].observation(10)?.status.term;
    let gate = store.block_sync_for_test();
    let mut expected = vec![(first, batch(1))];
    let held = executor.block_on(async {
        expected.push((commit(&c.handles[&1], 10, batch(2)).await?, batch(2)));
        wait_for(|| store.appended_index_for_test() > before).await?;
        let checkpoint = c.handles[&3].install_checkpoint(prepared)?;
        ensure!(!c.handles[&3].local(10, |n| n.groups[&10].accepts_async(&[]))?.await?);
        let mut full_at = None;
        let mut max_owned = 0;
        // With the real 16-message Raft inflight cap, four-MiB records reach
        // this 64-MiB byte window before the sender's packet limit stops it.
        for sequence in 3..=50 {
            let input = Batch::new(uuid::Uuid::from_u128(2), 0, uuid::Uuid::from_u128(3), sequence,
                bytes::Bytes::from(vec![sequence as u8; 4 << 20]));
            let index = commit(&c.handles[&1], 10, input.clone()).await?;
            expected.push((index, input));
            let owned = c.handles[&3].local(10, |n| {
                let g = &n.groups[&10];
                let unstable = g.raw.raft.raft_log.unstable.entries.iter()
                    .map(|e| e.compute_size() as usize + 2 * std::mem::size_of::<raft_proto::eraftpb::Entry>())
                    .sum::<usize>();
                let p = n.persistence_observation(10).context("persistence missing")?;
                Ok(unstable + p.pending_bytes + p.deferred_bytes)
            })?.await?;
            max_owned = max_owned.max(owned);
            ensure!(owned <= window_bytes, "peer input exceeded the original byte window");
            let view = c.handles[&3].observation(10)?;
            ensure!(!view.stopped && view.status.applied == first);
            ensure!(view.status.leader == 1 && view.status.term == term);
            ensure!(checkpoint.0.is_empty());
            if c.handles[&3].peer_rejections().pending_persistence > 0 {
                let start = *full_at.get_or_insert(sequence);
                if sequence >= start + 4 { break; }
            }
        }
        // The stalled replica must not have acknowledged its unsynced suffix;
        // the other two replicas are the real durable majority serving writes.
        let matched = c.handles[&1].local(10, |n| Ok(n.groups[&10].raw.raft.prs().get(3).unwrap().matched))?.await?;
        ensure!(matched == first, "stalled follower advertised unsynced replication");
        eprintln!("checkpoint peer boundedness: full_at={full_at:?} max_owned={max_owned} budget={window_bytes} healthy_commits={}", expected.len());
        Ok::<_, anyhow::Error>((checkpoint, full_at, max_owned))
    });
    drop(gate);
    let checked = executor.block_on(async {
        let (checkpoint, full_at, max_owned) = held?;
        timeout(Duration::from_secs(10), checkpoint).await??;
        let last = expected.last().unwrap().0;
        timeout(Duration::from_secs(30), async {
            while !c
                .handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= last))
            {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await?;
        Ok::<_, anyhow::Error>((full_at, max_owned))
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        for (index, input) in &expected {
            ensure!(node.read_accepted(10, *index)?.as_ref() == Some(input));
        }
    }
    let (full_at, max_owned) = checked?;
    ensure!(
        full_at.is_some() && max_owned > 32 << 20,
        "test did not exercise real byte pressure after filling checkpoint peer input"
    );
    Ok(())
}
