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
    let acquire = publication::Acquire {
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
    };
    nodes[0].acquire_publisher(1000, acquire)?;
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
