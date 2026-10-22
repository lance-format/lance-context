use super::*;

fn nodes(limits: Limits) -> Result<(tempfile::TempDir, Vec<Node>)> {
    let root = tempfile::tempdir()?;
    let nodes = (1..=3)
        .map(|id| {
            Node::open(
                &root.path().join(id.to_string()),
                super::super::tests::layout(),
                id,
                limits.clone(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((root, nodes))
}
fn pump(nodes: &mut [Node]) -> Result<Events> {
    let mut result = Events::default();
    for _ in 0..100 {
        let mut messages = Vec::new();
        for node in &mut *nodes {
            let events = node.poll()?;
            result.completions.extend(events.completions);
            result.reads.extend(events.reads);
            messages.extend(events.messages);
        }
        if messages.is_empty() {
            return Ok(result);
        }
        for packet in messages {
            nodes[packet.message.to as usize - 1].step(packet)?;
        }
    }
    anyhow::bail!("query message pump did not settle")
}
fn evaluate(query: &Query, cut: &Cut, state: &State) -> Result<Value> {
    query.evaluate(
        state,
        cut,
        &super::super::tests::layout(),
        &Limits::default(),
        source::QuerySigners {
            catalog: None,
            source: None,
        },
    )
}

#[test]
fn compact_visibility_uses_catalog_quorum_and_fits_control_reply_budget() -> Result<()> {
    let (_root, mut nodes) = nodes(Limits::default())?;
    nodes[0].campaign(10)?;
    nodes[0].campaign(20)?;
    pump(&mut nodes)?;
    let acquire = publication::Acquire {
        table: Uuid::from_u128(10),
        destination: publication::Destination {
            namespace: Uuid::from_u128(11),
            dataset: "x".repeat(4096),
        },
        ingest: crate::ingest_contract::IngestContract {
            tenant: "x".repeat(1024),
            qualified_table: "x".repeat(1024),
            schema: [3; 32],
            policy: crate::ingest_contract::WritePolicy::ImmutableUniqueIds,
            max_record_bytes: 1024 * 1024,
            max_rows: 1024,
        },
        owner: Uuid::from_u128(12),
        request: Uuid::from_u128(13),
        expected_epoch: 0,
    };
    let query = Query::Visibility {
        table: acquire.table,
    };
    assert!(nodes[0].query(20, 1, query.clone()).is_err());
    nodes[0].query(10, 1, query.clone())?;
    assert!(matches!(
        &pump(&mut nodes)?.reads[0].outcome,
        Outcome::Observed {
            value: Value::Visibility(None),
            ..
        }
    ));
    nodes[0].acquire_publisher(2, acquire.clone())?;
    pump(&mut nodes)?;
    nodes[0].query(10, 3, query)?;
    let observed = pump(&mut nodes)?;
    let Outcome::Observed {
        cut,
        value: Value::Visibility(Some(view)),
    } = &observed.reads[0].outcome
    else {
        anyhow::bail!("missing quorum visibility");
    };
    assert!(cut.quorum_index > 0 && cut.applied >= cut.quorum_index);
    assert_eq!(view.destination, acquire.destination);
    assert_eq!(view.ingest, acquire.ingest);
    assert_eq!(view.incarnation, None);
    assert!(view.checkpoint.is_empty());
    // Maximum configured group count, even with maximal binding strings,
    // remains below the runtime's generic control reply allowance.
    let mut maximum = view.clone();
    maximum.incarnation = Some(Uuid::from_u128(100));
    maximum.checkpoint = (1..=64).map(|group| (group, u64::MAX - 1)).collect();
    assert!(bincode::serialized_size(&maximum)? < 16 * 1024);
    Ok(())
}

#[test]
fn new_leader_rejects_reads_until_its_first_current_term_entry_commits() -> Result<()> {
    let (_root, mut nodes) = nodes(Limits {
        max_pending: 1,
        ..Default::default()
    })?;
    nodes[0].campaign(10)?;
    let mut held_appends = Vec::new();
    for round in 0..100 {
        let mut messages = Vec::new();
        for node in &mut nodes {
            let events = node.poll()?;
            assert!(events.reads.is_empty() && events.completions.is_empty());
            messages.extend(events.messages);
        }
        if messages.is_empty() {
            break;
        }
        assert!(round < 99, "election did not settle");
        for packet in messages {
            if packet.message.get_msg_type() == MessageType::MsgAppend {
                held_appends.push(packet);
            } else {
                nodes[packet.message.to as usize - 1].step(packet)?;
            }
        }
    }
    let group = &nodes[0].groups[&10];
    assert_eq!(group.raw.raft.state, StateRole::Leader);
    assert!(!held_appends.is_empty());
    assert!(!group.raw.raft.commit_to_current_term());
    assert!(group.raw.store().last_index()? > group.state.applied);

    // raft-rs silently drops ReadIndex here. Do not consume the one available
    // request slot or token for a request it cannot yet process.
    let query = Query::Accepted { after: 0, limit: 1 };
    assert!(nodes[0].query(10, 7, query.clone()).is_err());
    assert_eq!(nodes[0].status(10)?.pending_reads, 0);
    assert!(nodes[0].poll()?.reads.is_empty());
    for packet in held_appends {
        nodes[packet.message.to as usize - 1].step(packet)?;
    }
    assert!(pump(&mut nodes)?.reads.is_empty());
    assert!(nodes[0].groups[&10].raw.raft.commit_to_current_term());

    // Restoring replication makes the same token admissible without a timeout,
    // capacity reset or leader change. The retry completes through a quorum.
    nodes[0].query(10, 7, query)?;
    let reads = pump(&mut nodes)?.reads;
    assert_eq!(reads.len(), 1);
    assert_eq!(reads[0].request, 7);
    assert!(matches!(reads[0].outcome, Outcome::Observed { .. }));
    assert_eq!(nodes[0].status(10)?.pending_reads, 0);
    Ok(())
}

#[test]
fn reads_share_capacity_and_tokens_with_appends_and_never_use_a_local_lease() -> Result<()> {
    let (_root, mut nodes) = nodes(Limits {
        max_pending: 1,
        ..Default::default()
    })?;
    nodes[0].campaign(10)?;
    nodes[0].campaign(20)?;
    pump(&mut nodes)?;
    let log_positions = nodes
        .iter()
        .map(|node| node.groups[&10].raw.store().last_index())
        .collect::<Result<Vec<_>, _>>()?;
    let query = Query::Accepted { after: 0, limit: 1 };
    nodes[0].query(10, 1, query.clone())?;
    assert_eq!(nodes[0].status(10)?.pending_reads, 1);
    assert!(nodes[0]
        .propose(10, 2, super::super::tests::batch(1))
        .is_err());
    assert!(nodes[0].query(20, 1, query.clone()).is_err());
    assert!(nodes[0].query(10, 2, query.clone()).is_err());
    assert!(nodes[1].query(10, 1, query.clone()).is_err());
    let events = nodes[0].poll()?;
    assert!(
        events.reads.is_empty(),
        "local leader released a read without quorum"
    );
    for packet in events.messages {
        nodes[packet.message.to as usize - 1].step(packet)?;
    }
    let events = pump(&mut nodes)?;
    assert_eq!(events.reads.len(), 1);
    assert!(
        matches!(&events.reads[0].outcome, Outcome::Observed { cut, value: Value::Accepted(page) } if cut.quorum_index > 0 && cut.applied >= cut.quorum_index && page.through == cut.applied)
    );
    for (node, expected) in nodes.iter().zip(log_positions) {
        assert_eq!(node.groups[&10].raw.store().last_index()?, expected);
        assert_eq!(node.groups[&10].raw.raft.raft_log.last_index(), expected);
    }
    nodes[0].propose(10, 2, super::super::tests::batch(1))?;
    assert!(nodes[0].query(10, 3, query.clone()).is_err());
    pump(&mut nodes)?;
    nodes[0].query(
        10,
        3,
        Query::Accepted {
            after: u64::MAX - 1,
            limit: 1,
        },
    )?;
    assert!(matches!(
        &pump(&mut nodes)?.reads[0].outcome,
        Outcome::Observed {
            value: Value::CursorAhead { .. },
            ..
        }
    ));
    assert_eq!(nodes[0].status(10)?.pending_reads, 0);
    assert!(nodes[0]
        .query(
            20,
            4,
            Query::Publisher {
                table: Uuid::from_u128(50)
            }
        )
        .is_err());
    assert!(nodes[0]
        .query(
            10,
            4,
            Query::Producer {
                table: Uuid::from_u128(50),
                bucket: 1,
                producer: Uuid::from_u128(51)
            }
        )
        .is_err());
    assert!(nodes[0]
        .query(
            10,
            4,
            Query::Accepted {
                after: 0,
                limit: 129
            }
        )
        .is_err());
    Ok(())
}

#[test]
fn read_contexts_do_not_cross_request_reuse_reopen_or_unapplied_cuts() -> Result<()> {
    let mut tracker = Tracker::default();
    let query = Query::Accepted { after: 0, limit: 1 };
    let context = tracker.enqueue(10, 2, query.clone())?;
    let mut state = State {
        applied: 4,
        ..Default::default()
    };
    let mut cut = Cut {
        cluster: Uuid::from_u128(1),
        layout: [2; 32],
        group: 10,
        term: 2,
        quorum_index: 0,
        applied: 4,
    };
    tracker.confirm(&context, 5)?;
    assert!(tracker
        .finish(cut.clone(), &state, true, evaluate)?
        .is_empty());
    state.applied = 5;
    cut.applied = 5;
    assert_eq!(
        tracker.finish(cut.clone(), &state, true, evaluate)?.len(),
        1
    );
    let next = tracker.enqueue(10, 2, query.clone())?;
    assert_ne!(next, context);
    tracker.confirm(&context, 5)?;
    assert!(tracker
        .finish(cut.clone(), &state, true, evaluate)?
        .is_empty());
    cut.term = 3;
    assert_eq!(
        tracker.finish(cut.clone(), &state, true, evaluate)?[0].outcome,
        Outcome::Unknown
    );
    let mut reopened = Tracker::default();
    assert_ne!(reopened.enqueue(10, 3, query)?, context);
    reopened.confirm(&context, 5)?;
    assert!(reopened
        .finish(cut.clone(), &state, true, evaluate)?
        .is_empty());
    assert_eq!(
        reopened.finish(cut, &state, false, evaluate)?[0].outcome,
        Outcome::Unknown
    );
    Ok(())
}

#[test]
fn shared_physical_sync_failure_fences_already_confirmed_reads() -> Result<()> {
    let (_root, mut nodes) = nodes(Limits::default())?;
    nodes[0].campaign(10)?;
    nodes[0].campaign(20)?;
    pump(&mut nodes)?;
    nodes[0].query(10, 1, Query::Accepted { after: 0, limit: 1 })?;
    let events = nodes[0].poll()?;
    assert!(events.reads.is_empty());
    for packet in events.messages {
        nodes[packet.message.to as usize - 1].step(packet)?;
    }
    let mut responses = Vec::new();
    for node in &mut nodes[1..] {
        responses.extend(node.poll()?.messages);
    }
    for packet in responses {
        nodes[packet.message.to as usize - 1].step(packet)?;
    }
    assert!(
        !nodes[0].groups[&10].raw.raft.read_states.is_empty(),
        "quorum read was not confirmed before sibling failure"
    );
    let mut batch = super::super::tests::batch(1);
    batch.meta.bucket = 1;
    nodes[0].propose(20, 2, batch)?;
    nodes[0].groups[&20].raw.store().fail_next_sync_for_test();
    assert!(
        nodes[0].poll().is_err(),
        "sibling persistence error released read events"
    );
    assert!(nodes[0].status(10).is_err());
    assert!(nodes[0]
        .query(10, 3, Query::Accepted { after: 0, limit: 1 })
        .is_err());
    Ok(())
}
