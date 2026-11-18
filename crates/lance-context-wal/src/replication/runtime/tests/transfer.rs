use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn leader_transfer_preserves_committed_bytes_and_other_groups() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = Cluster::open(directory.path(), Options::default())?;
    c.elect(1, 10).await?;
    c.elect(1, 20).await?;
    let first = commit(&c.handles[&1], 10, batch(1)).await?;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= first))
    })
    .await?;
    assert!(!c.handles[&2].transfer_leader(10, 3)?.await?);
    assert!(!c.handles[&1].transfer_leader(10, 1)?.await?);
    assert!(c.handles[&1].transfer_leader(10, 4)?.await.is_err());
    c.handles[&2]
        .local(10, |node| {
            let before = node.status(10)?;
            for (sender, term) in [(3, before.term), (1, before.term - 1)] {
                let mut message = raft_proto::eraftpb::Message::default();
                message.set_msg_type(raft_proto::eraftpb::MessageType::MsgTimeoutNow);
                message.from = sender;
                message.to = 2;
                message.term = term;
                let packet = Packet {
                    cluster: node.layout.cluster,
                    layout: node.layout_hash,
                    policy: node.policy,
                    group: 10,
                    message,
                };
                assert!(node.step(packet).is_err());
            }
            let after = node.status(10)?;
            assert_eq!((after.leader, after.term), (before.leader, before.term));
            Ok(())
        })?
        .await?;
    let other_term = c.handles[&1].observation(20)?.status.term;

    // Issuing a transfer is not a leadership receipt. The intended target is
    // temporarily unreachable, while the other two voters still form a quorum.
    c.isolated.store(3, Ordering::Release);
    assert!(c.handles[&1].transfer_leader(10, 3)?.await?);
    sleep(Duration::from_millis(250)).await;
    assert_eq!(c.handles[&1].observation(10)?.status.leader, 1);
    c.isolated.store(0, Ordering::Release);
    // The first TimeoutNow can be lost. Once the ordinary transfer deadline
    // expires, a fresh request may be needed; neither request is a success ACK.
    sleep(Duration::from_secs(2)).await;
    c.handles[&1].transfer_leader(10, 3)?.await?;
    wait_for(|| {
        c.handles.values().all(|h| {
            h.observation(10)
                .is_ok_and(|o| o.status.leader == 3 && o.status.applied >= first)
        })
    })
    .await?;
    assert_eq!(commit(&c.handles[&3], 10, batch(1)).await?, first);
    let second = commit(&c.handles[&3], 10, batch(2)).await?;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= second))
    })
    .await?;
    for h in c.handles.values() {
        let other = h.observation(20)?.status;
        assert_eq!((other.leader, other.term), (1, other_term));
    }
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &directory.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        assert_eq!(node.read_accepted(10, first)?, Some(batch(1)));
        assert_eq!(node.read_accepted(10, second)?, Some(batch(2)));
    }
    Ok(())
}
