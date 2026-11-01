use super::*;
use std::collections::BTreeSet;

fn accepted() -> Query {
    Query::Accepted { after: 0, limit: 1 }
}

fn deliver(nodes: &mut [Node], packets: Vec<Packet>) -> Result<()> {
    for packet in packets {
        nodes[packet.message.to as usize - 1].step(packet)?;
    }
    Ok(())
}

#[test]
fn one_poll_shares_a_fresh_quorum_barrier_for_all_admitted_reads() -> Result<()> {
    let (_root, mut nodes) = nodes(Limits::default())?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    let batch = super::super::super::tests::batch(1);
    nodes[0].propose(10, 100, batch.clone())?;
    assert_eq!(pump(&mut nodes)?.completions.len(), 1);
    let before = nodes[0].groups[&10].raw.store().last_index()?;
    for request in 1..=32 {
        nodes[0].query(10, request, accepted())?;
    }
    assert_eq!(nodes[0].status(10)?.pending_reads, 32);
    let events = nodes[0].poll()?;
    assert!(events.reads.is_empty(), "batch bypassed the quorum");
    for peer in [2, 3] {
        assert_eq!(
            events
                .messages
                .iter()
                .filter(|p| p.message.to == peer
                    && p.message.get_msg_type() == MessageType::MsgHeartbeat
                    && !p.message.context.is_empty())
                .count(),
            1,
            "read batch sent more than one quorum heartbeat per peer"
        );
    }
    deliver(&mut nodes, events.messages)?;
    let results = pump(&mut nodes)?.reads;
    assert_eq!(results.len(), 32);
    let mut requests = BTreeSet::new();
    for result in results {
        assert!(requests.insert(result.request));
        let Outcome::Observed {
            cut,
            value: Value::Accepted(page),
        } = result.outcome
        else {
            anyhow::bail!("missing batched quorum read");
        };
        assert_eq!(cut.group, 10);
        assert!(cut.quorum_index > 0 && cut.applied >= cut.quorum_index);
        assert_eq!(page.through, cut.applied);
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.entries[0].1, batch.meta);
    }
    assert_eq!(nodes[0].status(10)?.pending_reads, 0);
    assert_eq!(nodes[0].groups[&10].raw.store().last_index()?, before);
    Ok(())
}

#[test]
fn later_arrivals_cannot_reuse_an_already_sent_barriers_quorum() -> Result<()> {
    let (_root, mut nodes) = nodes(Limits::default())?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    for request in 1..=2 {
        nodes[0].query(10, request, accepted())?;
    }
    let first = nodes[0].poll()?;
    assert!(first.reads.is_empty());
    // Both batches are pending when the first batch's responses arrive.
    for request in 3..=4 {
        nodes[0].query(10, request, accepted())?;
    }
    let second = nodes[0].poll()?;
    assert!(second.reads.is_empty());
    assert_ne!(
        first.messages[0].message.context,
        second.messages[0].message.context
    );
    deliver(&mut nodes, first.messages)?;
    let results = pump(&mut nodes)?.reads;
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.request <= 2));
    assert_eq!(nodes[0].status(10)?.pending_reads, 2);
    assert!(nodes[0].poll()?.reads.is_empty());
    deliver(&mut nodes, second.messages)?;
    let results = pump(&mut nodes)?.reads;
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.request >= 3));
    assert_eq!(nodes[0].status(10)?.pending_reads, 0);
    Ok(())
}

#[test]
fn batching_preserves_request_capacity_and_separate_group_quorums() -> Result<()> {
    let (_root, mut nodes) = nodes(Limits {
        max_pending: 3,
        ..Default::default()
    })?;
    nodes[0].campaign(10)?;
    nodes[0].campaign(20)?;
    pump(&mut nodes)?;
    for request in 1..=3 {
        nodes[0].query(10, request, accepted())?;
    }
    assert!(nodes[0].query(10, 4, accepted()).is_err());
    assert!(nodes[0]
        .propose(10, 4, super::super::super::tests::batch(1))
        .is_err());
    assert!(nodes[0].query(20, 1, accepted()).is_err());
    nodes[0].query(20, 4, accepted())?;
    let events = nodes[0].poll()?;
    let (first, second): (Vec<_>, Vec<_>) =
        events.messages.into_iter().partition(|p| p.group == 10);
    assert!(!first.is_empty() && !second.is_empty());
    deliver(&mut nodes, first)?;
    let results = pump(&mut nodes)?.reads;
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|r| r.group == 10));
    assert_eq!(nodes[0].status(20)?.pending_reads, 1);
    deliver(&mut nodes, second)?;
    let results = pump(&mut nodes)?.reads;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].group, 20);
    assert_eq!(results[0].request, 4);
    Ok(())
}

#[test]
fn leadership_loss_retires_both_queued_and_sent_read_batches() -> Result<()> {
    for send_first in [false, true] {
        let (_root, mut nodes) = nodes(Limits::default())?;
        nodes[0].campaign(10)?;
        pump(&mut nodes)?;
        for request in 1..=4 {
            nodes[0].query(10, request, accepted())?;
        }
        let held = if send_first {
            nodes[0].poll()?.messages
        } else {
            Vec::new()
        };
        let node = &nodes[0];
        let mut message = raft::eraftpb::Message::default();
        message.set_msg_type(MessageType::MsgHeartbeat);
        message.from = 2;
        message.to = 1;
        message.term = node.groups[&10].raw.raft.term + 1;
        let packet = Packet {
            cluster: node.layout.cluster,
            layout: node.layout_hash,
            policy: node.policy,
            group: 10,
            message,
        };
        nodes[0].step(packet)?;
        let events = nodes[0].poll()?;
        assert_eq!(events.reads.len(), 4);
        assert!(events.reads.iter().all(|r| r.outcome == Outcome::Unknown));
        assert_eq!(nodes[0].status(10)?.pending_reads, 0);
        deliver(&mut nodes, held)?;
        assert!(pump(&mut nodes)?.reads.is_empty());
    }
    Ok(())
}

#[test]
fn queued_batch_has_no_confirmation_and_does_not_cross_terms() -> Result<()> {
    let mut tracker = Tracker::default();
    tracker.enqueue(1, 2, accepted())?;
    tracker.enqueue(2, 2, accepted())?;
    let unissued = tracker.queued.unwrap();
    tracker.confirm(&unissued, 5)?;
    assert_eq!(tracker.pending[&unissued].confirmed, None);
    tracker.enqueue(3, 3, accepted())?;
    let issued = tracker.take_barrier(3).unwrap();
    assert_ne!(issued, unissued);
    let state = State {
        applied: 5,
        ..Default::default()
    };
    let cut = Cut {
        cluster: Uuid::from_u128(1),
        layout: [2; 32],
        group: 10,
        term: 3,
        quorum_index: 0,
        applied: 5,
    };
    let stale = tracker.finish(cut.clone(), &state, true, evaluate)?;
    assert_eq!(stale.len(), 2);
    assert!(stale.iter().all(|r| r.outcome == Outcome::Unknown));
    assert_eq!(tracker.len(), 1);
    tracker.confirm(&issued, 5)?;
    let done = tracker.finish(cut, &state, true, evaluate)?;
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].request, 3);
    assert_eq!(tracker.len(), 0);
    Ok(())
}
