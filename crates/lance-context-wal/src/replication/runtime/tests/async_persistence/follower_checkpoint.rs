use super::*;
use raft_proto::eraftpb::{Message, MessageType};

#[test]
fn follower_checkpoint_drain_keeps_heartbeats_live_without_early_ack() -> Result<()> {
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
        let append = c.handles[&1].commit(Command::Append { group: 10, batch: batch(2) })?;
        wait_for(|| stores.iter().zip(&before).all(|(s, before)| s.appended_index_for_test() > *before)).await?;
        let checkpoints = [2, 3].into_iter().zip(prepared)
            .map(|(id, image)| c.handles[&id].install_checkpoint(image))
            .collect::<Result<Vec<_>>>()?;
        // A later control round confirms that the actual install request has
        // queued behind the original native Ready write, not a fabricated flag.
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
        eprintln!("follower_checkpoint_drain: direct_refusals={direct_refusals:?} first_leader_change={first_leader_change:?} rejections={rejections:?} append_ready={}", !append.0.is_empty());
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
