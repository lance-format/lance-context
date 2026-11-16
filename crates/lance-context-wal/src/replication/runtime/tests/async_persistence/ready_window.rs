//! A full native task queue need not exhaust bounded peer input memory.
use super::*;
use raft_proto::eraftpb::{Message, MessageType};

#[test]
fn full_ready_queue_retains_bounded_peer_input_without_early_ack() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let store = executor.block_on(
        c.handles[&3].local(10, |n| Ok(n.groups[&10].raw.store().durable_log().clone()))?,
    )?;
    let before = store.appended_index_for_test();
    let gate = store.block_sync_for_test();
    let mut expected = vec![(first, 1)];
    let held = executor.block_on(async {
        let second = commit(&c.handles[&1], 10, batch(2)).await?;
        expected.push((second, 2));
        wait_for(|| store.appended_index_for_test() > before).await?;
        // Real duplicate Append messages produce persisted replies while the
        // original native write is blocked. No queue limit is reduced and no
        // fake persistence receipt is injected. The local closure controls the
        // scheduling of this bounded burst, not its Raft or storage handling.
        let (readies, bytes, capacity, admitted) = c.handles[&3]
            .local(10, move |n| {
                let g = &n.groups[&10];
                let mut message = Message {
                    from: 1,
                    to: 3,
                    term: g.raw.raft.term,
                    index: second,
                    log_term: g.raw.raft.raft_log.term(second)?,
                    commit: first,
                    ..Default::default()
                };
                message.set_msg_type(MessageType::MsgAppend);
                let packet = Packet {
                    cluster: n.layout.cluster,
                    layout: n.layout_hash,
                    policy: n.policy,
                    group: 10,
                    message,
                };
                let capacity = crate::replication::async_persist::window(
                    &options(), &Limits::default(),
                )?.readies;
                for _ in 0..capacity {
                    if n.persistence_observation(10).unwrap().pending_readies == capacity {
                        break;
                    }
                    n.step(packet.clone())?;
                    let events = n.poll()?;
                    ensure!(events.completions.is_empty() && events.reads.is_empty());
                    ensure!(events.messages.iter().all(|p| {
                        p.message.get_msg_type() != MessageType::MsgAppendResponse
                    }), "unsynced Append reply escaped");
                }
                let p = n.persistence_observation(10).unwrap();
                ensure!(p.pending_readies == capacity, "native task queue was not filled");
                ensure!(!p.pending_checkpoint, "test must isolate Ready queue pressure");
                ensure!(p.pending_bytes + p.deferred_bytes < 1 << 20,
                    "test accidentally exhausted byte capacity");
                ensure!(!n.groups[&10].accepts_async(&[])?,
                    "full queue admitted a new persistence task");
                let admitted = n.step(packet).is_ok();
                Ok((p.pending_readies, p.pending_bytes + p.deferred_bytes, capacity, admitted))
            })?
            .await?;
        let third = commit(&c.handles[&1], 10, batch(3)).await?;
        expected.push((third, 3));
        // The healthy two replicas can acknowledge. The held follower must
        // retain the new entry without claiming it has persisted or applied it.
        let deadline = Instant::now() + Duration::from_millis(300);
        let mut buffered = false;
        while Instant::now() < deadline {
            buffered = c.handles[&3].local(10, move |n| {
                ensure!(n.status(10)?.applied == first);
                ensure!(n.persistence_observation(10).unwrap().pending_readies == capacity);
                Ok(n.groups[&10].raw.raft.raft_log.unstable.entries.iter()
                    .any(|e| e.index == third))
            })?.await?;
            if buffered { break; }
            sleep(Duration::from_millis(10)).await;
        }
        let matched = c.handles[&1].local(10, |n| {
            Ok(n.groups[&10].raw.raft.prs().get(3).unwrap().matched)
        })?.await?;
        ensure!(matched == first, "held follower advertised unsynced data");
        eprintln!("Ready queue peer input: readies={readies} capacity={capacity} bytes={bytes} admitted={admitted} buffered={buffered}");
        Ok::<_, anyhow::Error>((admitted, buffered))
    });
    // Release the original write and verify recovery even on the baseline.
    drop(gate);
    let drained = executor.block_on(async {
        let last = expected.last().unwrap().0;
        wait_for(|| {
            c.handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= last))
        })
        .await
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    drained?;
    reopen(root.path(), &expected)?;
    let (admitted, buffered) = held?;
    ensure!(
        admitted && buffered,
        "full Ready task queue refused peer input despite available bounded memory"
    );
    Ok(())
}
