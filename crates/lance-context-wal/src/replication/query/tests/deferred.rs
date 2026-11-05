use super::*;
use crate::replication::{tests::batch, Applied, Outcome as WriteOutcome};
use std::collections::BTreeSet;

fn contexts(messages: &[Packet]) -> BTreeSet<Vec<u8>> {
    messages
        .iter()
        .filter(|p| {
            p.group == 10
                && p.message.get_msg_type() == MessageType::MsgHeartbeat
                && !p.message.context.is_empty()
        })
        .map(|p| p.message.context.to_vec())
        .collect()
}

fn deliver(nodes: &mut [Node], messages: Vec<Packet>) -> Result<()> {
    for packet in messages {
        nodes[packet.message.to as usize - 1].step(packet)?;
    }
    Ok(())
}

#[test]
fn arrivals_during_a_quorum_round_coalesce_without_reusing_its_confirmation() -> Result<()> {
    let (root, mut nodes) = nodes(Limits::default())?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    nodes[0].propose(10, 100, batch(1))?;
    let warmup = pump(&mut nodes)?;
    let WriteOutcome::Committed(Applied::Accepted { index: first_index }) =
        warmup.completions[0].outcome
    else {
        anyhow::bail!("warmup not committed");
    };

    nodes[0].query(10, 1, Query::Accepted { after: 0, limit: 1 })?;
    let first = nodes[0].poll()?;
    assert!(first.reads.is_empty());
    let first_context = contexts(&first.messages);
    assert_eq!(first_context.len(), 1);
    let mut before_confirmation = first_context.clone();
    let mut held = Vec::new();
    for request in 2..=17 {
        // Each arrival has its own poll, while the first quorum is delayed.
        // This is distinct from batching requests already queued before a poll.
        nodes[0].query(10, request, Query::Accepted { after: 0, limit: 1 })?;
        let events = nodes[0].poll()?;
        assert!(events.reads.is_empty());
        before_confirmation.extend(contexts(&events.messages));
        held.extend(events.messages);
    }
    assert_eq!(nodes[0].status(10)?.pending_reads, 17);

    // Only the original round's replies reach the leader. Later arrivals must
    // remain pending even though this quorum completes after they arrived.
    deliver(&mut nodes, first.messages)?;
    let mut replies = Vec::new();
    for node in &mut nodes[1..] {
        replies.extend(node.poll()?.messages);
    }
    deliver(&mut nodes, replies)?;
    let first_completed = nodes[0].poll()?;
    let completed_before_fresh_quorum: Vec<_> =
        first_completed.reads.iter().map(|r| r.request).collect();
    let pending_after_first = nodes[0].status(10)?.pending_reads;
    let immediate_next = contexts(&first_completed.messages);
    let next_is_fresh = immediate_next.is_disjoint(&first_context);
    held.extend(first_completed.messages);

    // Resolve every original read and a real second append before comparing
    // schedulers. No ticks, sleeps, forced Raft state or second workload run.
    deliver(&mut nodes, held)?;
    let rest = pump(&mut nodes)?;
    let mut reads = first_completed.reads;
    reads.extend(rest.reads);
    assert_eq!(reads.len(), 17);
    assert_eq!(
        reads.iter().map(|r| r.request).collect::<BTreeSet<_>>(),
        (1..=17).collect()
    );
    for read in reads {
        let Outcome::Observed {
            cut,
            value: Value::Accepted(page),
        } = read.outcome
        else {
            anyhow::bail!("missing observed read");
        };
        assert!(cut.applied >= first_index && cut.applied >= cut.quorum_index);
        assert_eq!(page.entries, vec![(first_index, batch(1).meta)]);
    }
    assert_eq!(nodes[0].status(10)?.pending_reads, 0);
    nodes[0].propose(10, 101, batch(2))?;
    let committed = pump(&mut nodes)?;
    let WriteOutcome::Committed(Applied::Accepted {
        index: second_index,
    }) = committed.completions[0].outcome
    else {
        anyhow::bail!("second append not committed");
    };
    drop(nodes);
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            crate::replication::tests::layout(),
            id,
            Limits::default(),
        )?;
        assert_eq!(node.read_accepted(10, first_index)?, Some(batch(1)));
        assert_eq!(node.read_accepted(10, second_index)?, Some(batch(2)));
    }
    println!(
        "READ_ROUND_EXPERIMENT {}",
        serde_json::json!({
            "reads": 17,
            "contexts_before_confirmation": before_confirmation.len(),
            "first_quorum_completed": completed_before_fresh_quorum,
            "pending_after_first_quorum": pending_after_first,
            "immediate_next_contexts": immediate_next.len(),
            "next_context_is_fresh": next_is_fresh,
            "reopened_replicas": 3,
            "ticks": 0,
            "scope": "native delayed-quorum mechanism test; not cloud throughput"
        })
    );
    assert_eq!(completed_before_fresh_quorum, vec![1]);
    assert_eq!(pending_after_first, 16);
    assert!(next_is_fresh);
    assert_eq!(before_confirmation.len(), 1);
    assert_eq!(
        immediate_next.len(),
        1,
        "next batch waited for another poll"
    );
    Ok(())
}
