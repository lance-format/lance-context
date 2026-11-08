use super::*;
use crate::replication::tests::{batch, layout};

fn nodes() -> Result<(tempfile::TempDir, Vec<Node>)> {
    let root = tempfile::tempdir()?;
    let nodes = (1..=3)
        .map(|id| {
            Node::open(
                &root.path().join(id.to_string()),
                layout(),
                id,
                Limits::default(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((root, nodes))
}

fn pump(nodes: &mut [Node], connected: &[u64]) -> Result<Vec<Completion>> {
    let mut done = Vec::new();
    for _ in 0..100 {
        let mut packets = Vec::new();
        for node in nodes.iter_mut().filter(|n| connected.contains(&n.id)) {
            let events = node.poll()?;
            done.extend(events.completions);
            packets.extend(events.messages);
        }
        if packets.is_empty() {
            return Ok(done);
        }
        for packet in packets {
            if connected.contains(&packet.message.to) {
                nodes[packet.message.to as usize - 1].step(packet)?;
            }
        }
    }
    anyhow::bail!("committed source message pump did not settle")
}

fn request(after: u64) -> Request {
    Request {
        table: batch(1).meta.stream,
        bucket: 0,
        after,
        limit: 32,
    }
}

fn page(observed: &Observation) -> &Page {
    let Value::Page { page, .. } = &observed.value else {
        panic!("expected a committed source page");
    };
    page
}

#[test]
fn source_read_excludes_uncommitted_suffix_without_an_extra_quorum_round() -> Result<()> {
    let (_root, mut nodes) = nodes()?;
    nodes[0].campaign(10)?;
    pump(&mut nodes, &[1, 2, 3])?;
    nodes[0].propose(10, 1, batch(1))?;
    assert_eq!(pump(&mut nodes, &[1, 2, 3])?.len(), 1);
    let committed = nodes[0].read_source(10, &request(0))?;
    assert_eq!(page(&committed).entries.len(), 1);
    nodes[0].propose(10, 2, batch(2))?;
    // Persist the leader's new entry, withholding every follower message.
    let isolated = nodes[0].poll()?;
    assert!(isolated.completions.is_empty());
    let durable_end = nodes[0].groups[&10].raw.store().last_index()?;
    assert!(durable_end > committed.applied);
    let observed = nodes[0].read_source(10, &request(0))?;
    assert_eq!(observed, committed);
    assert_eq!(nodes[0].status(10)?.pending_reads, 0);
    let events = nodes[0].poll()?;
    assert!(events.reads.is_empty());
    assert!(events.messages.is_empty());
    assert_eq!(nodes[0].groups[&10].raw.store().last_index()?, durable_end);
    // The existing strong read still waits for its own fresh quorum.
    nodes[0].query(10, 3, query::Query::Accepted { after: 0, limit: 1 })?;
    let events = nodes[0].poll()?;
    assert!(events.reads.is_empty());
    assert!(!events.messages.is_empty());
    assert_eq!(nodes[0].status(10)?.pending_reads, 1);
    Ok(())
}

#[test]
fn isolated_old_leader_returns_only_its_committed_prefix_and_reports_cursor_ahead() -> Result<()> {
    let (_root, mut nodes) = nodes()?;
    nodes[0].campaign(10)?;
    pump(&mut nodes, &[1, 2, 3])?;
    nodes[0].propose(10, 1, batch(1))?;
    pump(&mut nodes, &[1, 2, 3])?;
    let old = nodes[0].read_source(10, &request(0))?;
    nodes[1].campaign(10)?;
    pump(&mut nodes, &[2, 3])?;
    // Recent leader contact suppresses votes until the majority's election
    // timeout. Advance only that partition; the isolated old leader has not
    // yet learned the new term or lost its own check-quorum interval.
    assert!(nodes[1].status(10)?.leader != 2);
    let mut successor = None;
    for _ in 0..100 {
        nodes[1].tick()?;
        nodes[2].tick()?;
        pump(&mut nodes, &[2, 3])?;
        successor = [1, 2]
            .into_iter()
            .find(|&i| nodes[i].status(10).is_ok_and(|s| s.leader == nodes[i].id));
        if successor.is_some() {
            break;
        }
    }
    let successor = successor.context("isolated majority did not elect a leader")?;
    nodes[successor].propose(10, 2, batch(2))?;
    assert_eq!(pump(&mut nodes, &[2, 3])?.len(), 1);
    let current = nodes[successor].read_source(10, &request(0))?;
    assert_eq!(page(&current).entries.len(), 2);
    assert!(current.applied > old.applied);
    assert_eq!(nodes[0].read_source(10, &request(0))?, old);
    let behind = nodes[0].read_source(10, &request(current.applied))?;
    assert_eq!(behind.applied, old.applied);
    assert_eq!(behind.value, Value::CursorAhead);
    // Deliver the new term. The old leader must now reject routed source reads.
    nodes[successor].query(10, 3, query::Query::Accepted { after: 0, limit: 1 })?;
    pump(&mut nodes, &[1, 2, 3])?;
    assert!(nodes[0].read_source(10, &request(0)).is_err());
    Ok(())
}

#[test]
fn cursor_ahead_is_distinct_from_an_empty_page() -> Result<()> {
    let (_root, mut nodes) = nodes()?;
    nodes[0].campaign(10)?;
    pump(&mut nodes, &[1, 2, 3])?;
    let applied = nodes[0].read_source(10, &request(0))?.applied;
    let observed = nodes[0].read_source(10, &request(applied + 1))?;
    assert_eq!(observed.applied, applied);
    assert_eq!(observed.value, Value::CursorAhead);
    Ok(())
}

#[test]
fn full_restart_and_election_keep_uncommitted_suffix_out_of_source_pages() -> Result<()> {
    let (root, mut nodes) = nodes()?;
    nodes[0].campaign(10)?;
    pump(&mut nodes, &[1, 2, 3])?;
    nodes[0].propose(10, 1, batch(1))?;
    pump(&mut nodes, &[1, 2, 3])?;
    let expected = page(&nodes[0].read_source(10, &request(0))?)
        .entries
        .clone();
    nodes[0].propose(10, 2, batch(2))?;
    assert!(nodes[0].poll()?.completions.is_empty());
    drop(nodes);
    let mut reopened = (1..=3)
        .map(|id| {
            Node::open(
                &root.path().join(id.to_string()),
                layout(),
                id,
                Limits::default(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    reopened[1].campaign(10)?;
    pump(&mut reopened, &[2, 3])?;
    let observed = reopened[1].read_source(10, &request(0))?;
    assert_eq!(page(&observed).entries, expected);
    assert_eq!(reopened[1].status(10)?.pending_reads, 0);
    Ok(())
}

#[test]
fn source_read_validates_scope_and_keeps_pages_bounded() -> Result<()> {
    let (_root, mut nodes) = nodes()?;
    nodes[0].campaign(10)?;
    pump(&mut nodes, &[1, 2, 3])?;
    for i in 1..=3 {
        nodes[0].propose(10, i, batch(i))?;
    }
    pump(&mut nodes, &[1, 2, 3])?;
    assert!(nodes[1].read_source(10, &request(0)).is_err());
    assert!(nodes[0].read_source(20, &request(0)).is_err());
    for bad in [
        Request {
            table: Uuid::nil(),
            ..request(0)
        },
        Request {
            limit: 0,
            ..request(0)
        },
        Request {
            limit: 129,
            ..request(0)
        },
        request(u64::MAX),
    ] {
        assert!(nodes[0].read_source(10, &bad).is_err());
    }
    let mut cursor = 0;
    let mut entries = Vec::new();
    loop {
        let observed = nodes[0].read_source(
            10,
            &Request {
                limit: 1,
                ..request(cursor)
            },
        )?;
        assert_eq!(observed.cluster, layout().cluster);
        assert_eq!(observed.layout, nodes[0].layout_hash);
        assert_eq!(observed.group, 10);
        let selected = page(&observed);
        assert!(selected.entries.len() <= 1);
        assert!(selected.through > cursor && selected.through <= observed.applied);
        entries.extend(selected.entries.iter().cloned());
        cursor = selected.through;
        if cursor == observed.applied {
            break;
        }
    }
    assert_eq!(
        entries.iter().map(|(_, m)| m.sequence).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    Ok(())
}
