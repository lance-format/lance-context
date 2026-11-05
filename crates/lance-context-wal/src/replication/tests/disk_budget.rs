use super::*;
use crate::raft_log::{DiskOptions, LogOptions};

fn limits() -> Limits {
    Limits {
        max_record_bytes: 1 << 20,
        max_uncommitted_bytes: 1 << 20,
        ..Limits::default()
    }
}
fn open(root: &Path, id: u64) -> Result<Node> {
    Node::open_with_storage_options(
        &root.join(id.to_string()),
        layout(),
        id,
        limits(),
        LogOptions {
            target_file_bytes: 16 << 10,
            purge_threshold_bytes: 64 << 10,
        },
    )?
    .with_disk_budget(DiskOptions {
        max_bytes: 32 << 20,
        data_bytes: 8 << 20,
        headroom_bytes: 8 << 20,
    })
}
fn payload(group: u64, sequence: u64) -> Batch {
    Batch::new(
        Uuid::from_u128(2),
        u32::from(group == 20),
        Uuid::from_u128(group.into()),
        sequence,
        Bytes::from(vec![sequence as u8; 128 << 10]),
    )
}
fn packet(leader: &Node, to: u64, image: &checkpoint::Image) -> Result<Packet> {
    Ok(Packet {
        cluster: leader.layout.cluster,
        layout: leader.layout_hash,
        policy: leader.policy,
        group: 10,
        message: Message {
            from: 1,
            to,
            term: leader.status(10)?.term,
            msg_type: MessageType::MsgSnapshot,
            snapshot: Some(image.snapshot()?).into(),
            ..Default::default()
        },
    })
}

fn acquisition() -> publication::Acquire {
    publication::Acquire {
        table: Uuid::from_u128(100),
        owner: Uuid::from_u128(101),
        request: Uuid::from_u128(102),
        expected_epoch: 0,
        destination: publication::Destination {
            namespace: Uuid::from_u128(103),
            dataset: "test/table".into(),
        },
        ingest: crate::ingest_contract::IngestContract {
            tenant: "test".into(),
            qualified_table: "table".into(),
            schema: [1; 32],
            policy: crate::ingest_contract::WritePolicy::ImmutableUniqueIds,
            max_record_bytes: 1 << 20,
            max_rows: 1024,
        },
    }
}

#[test]
fn failed_peer_validation_releases_reservation_without_appending_a_prefix() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut nodes = (1..=3)
        .map(|id| open(root.path(), id))
        .collect::<Result<Vec<_>>>()?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    let previous_last = nodes[2].groups[&10].raw.raft.raft_log.last_index();
    nodes[0].propose(10, 100, payload(10, 1))?;
    nodes[0].propose(10, 101, payload(10, 2))?;
    let events = nodes[0].poll()?;
    let mut valid = events
        .messages
        .iter()
        .find(|p| p.message.to == 3 && !p.message.entries.is_empty())
        .context("leader did not send entries to the follower")?
        .clone();
    let entries: BTreeMap<_, _> = events
        .messages
        .iter()
        .filter(|p| p.message.to == 3)
        .flat_map(|p| p.message.entries.iter().map(|e| (e.index, e.clone())))
        .collect();
    assert_eq!(entries.len(), 2);
    valid.message.entries = entries.into_values().collect();
    let mut corrupt = valid.clone();
    let body = &mut corrupt.message.entries[1].data;
    let mut changed = body.to_vec();
    changed[0] ^= 1;
    *body = Bytes::from(changed);
    assert!(nodes[2].step(corrupt).is_err());
    assert_eq!(nodes[2].disk_budget().unwrap().reserved_proposal_bytes, 0);
    assert_eq!(
        nodes[2].groups[&10].raw.raft.raft_log.last_index(),
        previous_last
    );
    // Neither a valid prefix nor a higher term may reach Raft on failed validation.
    let previous_term = nodes[2].status(10)?.term;
    let mut malformed = valid.clone();
    malformed.message.term += 1;
    malformed.message.entries[1].context = Bytes::from_static(b"invalid metadata");
    assert!(nodes[2].step(malformed).is_err());
    assert_eq!(nodes[2].status(10)?.term, previous_term);
    assert_eq!(nodes[2].disk_budget().unwrap().reserved_proposal_bytes, 0);
    assert_eq!(
        nodes[2].groups[&10].raw.raft.raft_log.last_index(),
        previous_last
    );
    for message in events.messages {
        nodes[message.message.to as usize - 1].step(message)?;
    }
    assert!(nodes[2].disk_budget().unwrap().reserved_proposal_bytes > 0);
    pump(&mut nodes)?;
    assert_eq!(nodes[2].disk_budget().unwrap().reserved_proposal_bytes, 0);
    let indexes = valid
        .message
        .entries
        .iter()
        .map(|e| e.index)
        .collect::<Vec<_>>();
    drop(nodes);
    for id in 1..=3 {
        let node = open(root.path(), id)?;
        for (sequence, index) in (1..=2).zip(&indexes) {
            assert_eq!(node.read_accepted(10, *index)?, Some(payload(10, sequence)));
        }
    }
    Ok(())
}

/// Measures only the owner-thread rejection path, excluding transport and
/// payload construction. This is not a distributed throughput benchmark.
#[test]
#[ignore = "manual peer disk-pressure timing experiment"]
fn benchmark_peer_disk_pressure_rejection() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut nodes = (1..=3)
        .map(|id| {
            Node::open_with_storage_options(
                &root.path().join(id.to_string()),
                layout(),
                id,
                Limits {
                    max_record_bytes: 8 << 20,
                    max_uncommitted_bytes: 8 << 20,
                    ..Limits::default()
                },
                LogOptions {
                    target_file_bytes: 16 << 10,
                    purge_threshold_bytes: 64 << 10,
                },
            )?
            .with_disk_budget(DiskOptions {
                max_bytes: 64 << 20,
                data_bytes: 8 << 20,
                headroom_bytes: 24 << 20,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    let occupied = root.path().join("3").join("retained-space-fixture");
    std::fs::write(&occupied, vec![7_u8; 9 << 20])?;
    nodes[2].prepare_disk_measurement()?.run()?;
    let body_bytes = 4 << 20;
    nodes[0].propose(
        10,
        100,
        Batch::new(
            Uuid::from_u128(2),
            0,
            Uuid::from_u128(10),
            1,
            Bytes::from(vec![7; body_bytes]),
        ),
    )?;
    let events = nodes[0].poll()?;
    let deferred = events
        .messages
        .iter()
        .find(|p| p.message.to == 3 && !p.message.entries.is_empty())
        .context("missing peer append")?;
    let previous_last = nodes[2].groups[&10].raw.raft.raft_log.last_index();
    let attempts = 1024;
    for run in 0..5 {
        let started = std::time::Instant::now();
        for _ in 0..attempts {
            let error = nodes[2]
                .step(std::hint::black_box(deferred.clone()))
                .unwrap_err();
            assert!(error.is::<crate::admission::DiskBudgetExhausted>());
        }
        println!(
            "peer_pressure run={run} attempts={attempts} body_bytes={body_bytes} elapsed_us={}",
            started.elapsed().as_micros()
        );
    }
    assert_eq!(
        nodes[2].groups[&10].raw.raft.raft_log.last_index(),
        previous_last
    );
    assert_eq!(nodes[2].disk_budget().unwrap().reserved_proposal_bytes, 0);
    Ok(())
}

#[test]
fn pressured_follower_defers_new_data_and_catches_up_after_space_is_reclaimed() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut nodes = (1..=3)
        .map(|id| open(root.path(), id))
        .collect::<Result<Vec<_>>>()?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    // Consume real space in this replica's owned directory. Its two peers have
    // room, so a healthy quorum must remain able to accept the append.
    let occupied = root.path().join("3").join("retained-space-fixture");
    std::fs::write(&occupied, vec![7_u8; 9 << 20])?;
    let before = nodes[2].prepare_disk_measurement()?.run()?;
    assert!(before.accounted_disk_bytes > before.data_limit);
    nodes[0].propose(10, 100, payload(10, 1))?;
    let events = nodes[0].poll()?;
    let deferred = events
        .messages
        .iter()
        .find(|p| p.message.to == 3 && !p.message.entries.is_empty())
        .context("leader did not send the append to the pressured follower")?
        .clone();
    assert!(
        nodes[2].step(deferred.clone()).is_err(),
        "follower accepted a new payload despite exhausted data admission"
    );
    let mut corrupt = deferred.clone();
    let mut changed = corrupt.message.entries[0].data.to_vec();
    changed[0] ^= 1;
    corrupt.message.entries[0].data = Bytes::from(changed);
    // Pressure refuses the packet before its body is scanned. Once admission
    // reopens, the same corrupt packet must fail integrity validation instead.
    assert!(nodes[2]
        .step(corrupt.clone())
        .unwrap_err()
        .is::<crate::admission::DiskBudgetExhausted>());
    let previous_applied = nodes[2].status(10)?.applied;
    nodes[2].poll()?;
    assert_eq!(nodes[2].status(10)?.applied, previous_applied);
    for message in events.messages.into_iter().filter(|p| p.message.to != 3) {
        nodes[message.message.to as usize - 1].step(message)?;
    }
    let mut committed = None;
    for _ in 0..20 {
        let mut messages = Vec::new();
        for node in &mut nodes {
            let events = node.poll()?;
            for completion in events.completions {
                if completion.request == 100 {
                    let Outcome::Committed(Applied::Accepted { index }) = completion.outcome else {
                        anyhow::bail!("healthy quorum did not accept the append");
                    };
                    committed = Some(index);
                }
            }
            messages.extend(events.messages);
        }
        for message in messages {
            if message.message.to != 3 || message.message.entries.is_empty() {
                nodes[message.message.to as usize - 1].step(message)?;
            }
        }
        if committed.is_some() {
            break;
        }
    }
    let index = committed.context("healthy quorum failed to commit")?;
    assert_eq!(nodes[2].read_accepted(10, index)?, None);
    std::fs::remove_file(&occupied)?;
    nodes[2].prepare_disk_measurement()?.run()?;
    assert!(matches!(
        nodes[2]
            .step(corrupt)
            .unwrap_err()
            .downcast_ref::<crate::record::ValidationError>(),
        Some(crate::record::ValidationError::DigestMismatch)
    ));
    assert_eq!(nodes[2].disk_budget().unwrap().reserved_proposal_bytes, 0);
    nodes[2].step(deferred.clone())?;
    assert!(nodes[2].disk_budget().unwrap().reserved_proposal_bytes > 0);
    pump(&mut nodes)?;
    assert_eq!(nodes[2].disk_budget().unwrap().reserved_proposal_bytes, 0);
    for node in &nodes {
        assert_eq!(node.read_accepted(10, index)?, Some(payload(10, 1)));
    }
    // A lost response must not prevent an already persisted entry from being
    // acknowledged under pressure. Control-only replication also retains its
    // reserved window so publication/retirement need not wait for new data.
    std::fs::write(&occupied, vec![7_u8; 9 << 20])?;
    nodes[2].prepare_disk_measurement()?.run()?;
    nodes[2].step(deferred)?;
    assert_eq!(nodes[2].disk_budget().unwrap().reserved_proposal_bytes, 0);
    assert!(nodes[2].poll()?.messages.iter().any(|p| {
        p.message.get_msg_type() == MessageType::MsgAppendResponse
            && !p.message.reject
            && p.message.index == index
    }));
    nodes[0].acquire_publisher(101, acquisition())?;
    assert!(pump(&mut nodes)?.iter().any(|c| c.request == 101));
    assert!(nodes[2].publisher(Uuid::from_u128(100))?.is_some());
    std::fs::remove_file(occupied)?;
    drop(nodes);
    for id in 1..=3 {
        assert_eq!(
            open(root.path(), id)?.read_accepted(10, index)?,
            Some(payload(10, 1))
        );
    }
    Ok(())
}

#[test]
fn shared_budget_preserves_control_reopens_orphans_and_reclaims_only_deleted_images() -> Result<()>
{
    use checkpoint::Part;
    let root = tempfile::tempdir()?;
    let mut nodes = (1..=3)
        .map(|id| open(root.path(), id))
        .collect::<Result<Vec<_>>>()?;
    nodes[0].campaign(10)?;
    nodes[0].campaign(20)?;
    pump(&mut nodes)?;
    let mut accepted = Vec::new();
    for group in [10, 20] {
        for sequence in 1..=8 {
            let token = group * 100 + sequence;
            nodes[0].propose(group, token, payload(group, sequence))?;
            let events = pump(&mut nodes)?;
            let Outcome::Committed(Applied::Accepted { index }) = events
                .iter()
                .find(|e| e.request == token)
                .context("missing append completion")?
                .outcome
            else {
                anyhow::bail!("append rejected");
            };
            accepted.push((group, sequence, index));
        }
    }
    for node in &nodes {
        assert!(node.disk_budget().unwrap().accounted_disk_bytes >= 2 << 20);
    }
    let prepared = nodes[0]
        .prepare_checkpoint(10, Uuid::from_u128(900))?
        .run()?;
    let image = prepared.image();
    nodes[0].install_checkpoint(prepared)?;
    // A captured checkpoint must reserve exactly the same encoder lengths.
    let g = &nodes[0].groups[&10];
    let cut = checkpoint::Cut {
        group: 10,
        id: Uuid::from_u128(901),
        index: g.state.applied,
        term: g.raw.store().term(g.state.applied)?,
    };
    assert_eq!(
        checkpoint::build_sizes(&layout(), &limits(), &cut, &g.state)?,
        [image.bytes(Part::Metadata), image.bytes(Part::Payload)]
    );

    let before = nodes[2].disk_budget().unwrap().accounted_disk_bytes;
    let mut receiver =
        nodes[2].receive_snapshot(packet(&nodes[0], 3, &image)?, Uuid::from_u128(902))?;
    let expected_charge = nodes[2].disk_budget().unwrap().reserved_file_bytes;
    assert!(expected_charge >= image.bytes(Part::Payload));
    receiver.write(Part::Payload, 0, &image.chunk(Part::Payload, 0, 128 << 10)?)?;
    drop(receiver);
    assert_eq!(nodes[2].disk_budget().unwrap().reserved_file_bytes, 0);
    assert!(nodes[2].disk_budget().unwrap().accounted_disk_bytes >= before + expected_charge);

    // The measurement task owns the engine lock even after its Node is gone.
    let measure = nodes[2].prepare_disk_measurement()?;
    drop(nodes.pop());
    assert!(open(root.path(), 3).is_err());
    assert!(measure.run()?.accounted_disk_bytes >= 2 << 20);
    drop(measure);
    nodes.push(open(root.path(), 3)?);
    // Native recovery may truncate unused preallocated log tails. Validate the
    // current physical files, not monotonic disk usage across engine reopens.
    let mut occupied = 0;
    for entry in std::fs::read_dir(root.path().join("3"))? {
        let metadata = entry?.metadata()?;
        if !metadata.is_file() {
            continue;
        }
        let mut bytes = metadata.len();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            bytes = bytes.max(metadata.blocks() * 512);
        }
        occupied += bytes;
    }
    assert_eq!(
        nodes[2].disk_budget().unwrap().accounted_disk_bytes,
        occupied
    );
    let orphan_usage = nodes[2].disk_budget().unwrap().accounted_disk_bytes;
    let mut collection = nodes[2].prepare_image_collection()?;
    let mut removed = 0;
    loop {
        let p = collection.run_batch(2)?;
        removed += p.removed_files;
        if p.finished {
            break;
        }
    }
    drop(collection);
    assert_eq!(removed, 2);
    assert!(nodes[2].disk_budget().unwrap().accounted_disk_bytes < orphan_usage);

    // The same physical budget rejects a new image before either file exists.
    let budget = nodes[1].images.store().disk_budget().unwrap();
    let used = budget.snapshot().accounted_disk_bytes;
    budget.limit_files(usize::try_from(used + image.bytes(Part::Payload) / 2)?)?;
    let files = std::fs::read_dir(root.path().join("2"))?.count();
    assert!(nodes[1]
        .receive_snapshot(packet(&nodes[0], 2, &image)?, Uuid::from_u128(903))
        .is_err());
    assert_eq!(std::fs::read_dir(root.path().join("2"))?.count(), files);

    nodes[0].prepare_log_maintenance()?.run()?;
    let shared = nodes[0].images.store().disk_budget().unwrap().clone();
    let used = shared.snapshot().accounted_disk_bytes;
    shared.limit_data(usize::try_from(used + 4096)?)?;
    assert!(nodes[0].propose(20, 999, payload(20, 9)).is_err());
    nodes[0].acquire_publisher(1000, acquisition())?;
    assert!(pump(&mut nodes)?.iter().any(|e| e.request == 1000
        && matches!(
            e.outcome,
            Outcome::Committed(Applied::Publisher(publication::Reply::Acquired(_)))
        )));
    drop(image);
    drop(nodes);
    let mut nodes = (1..=3)
        .map(|id| open(root.path(), id))
        .collect::<Result<Vec<_>>>()?;
    for node in &nodes {
        for &(group, sequence, index) in &accepted {
            assert_eq!(
                node.read_accepted(group, index)?,
                Some(payload(group, sequence))
            );
        }
    }
    nodes[0].campaign(10)?;
    nodes[0].campaign(20)?;
    pump(&mut nodes)?;
    for group in [10, 20] {
        nodes[0].propose(group, 2000 + group, payload(group, 8))?;
        let expected = accepted
            .iter()
            .find(|(g, s, _)| *g == group && *s == 8)
            .unwrap()
            .2;
        assert!(pump(&mut nodes)?.iter().any(|e| e.request == 2000 + group
            && e.outcome == Outcome::Committed(Applied::Accepted { index: expected })));
    }
    Ok(())
}
