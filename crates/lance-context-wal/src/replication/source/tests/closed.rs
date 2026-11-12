use super::*;
use publication::{
    BootstrapManifest, ConfirmPublication, Descriptors, ManifestObject, PrepareRequest,
    ReserveManifest,
};
mod proof;

fn publisher(events: Events) -> Result<publication::Reply> {
    events
        .completions
        .into_iter()
        .find_map(|c| match c.outcome {
            Outcome::Committed(Applied::Publisher(reply)) => Some(reply),
            _ => None,
        })
        .context("publisher completion missing")
}

fn fixture() -> Result<(Cluster, Permit, Uuid, Uuid)> {
    fixture_with_background(100_000, 12)
}
fn fixture_with_background(
    max_records: u64,
    background: u64,
) -> Result<(Cluster, Permit, Uuid, Uuid)> {
    fixture_from(Cluster::with_policy(128, max_records)?, background)
}

fn fixture_from(mut c: Cluster, background: u64) -> Result<(Cluster, Permit, Uuid, Uuid)> {
    let authority = c.acquire(0)?;
    let incarnation = Uuid::from_u128(800);
    c.nodes[0].bootstrap_manifest(
        800,
        BootstrapManifest {
            authority,
            incarnation,
            request: Uuid::from_u128(801),
            version: 8,
            object: ManifestObject {
                path: "_versions/8.manifest".into(),
                bytes: 8192,
                v2_names: false,
            },
            transaction: Uuid::from_u128(802),
        },
    )?;
    let publication::Reply::ManifestReserved(base) = publisher(c.pump()?)? else {
        anyhow::bail!("bootstrap missing")
    };
    let plan = c.plan_with_background(|_| {}, background)?;
    let permit = c.permit(plan)?;
    assert!(matches!(c.pin(permit.clone())?, Reply::Pinned { .. }));
    Ok((c, permit, incarnation, base.identity))
}

fn closure_permit(c: &mut Cluster, plan: u64) -> Result<Option<ClosurePermit>> {
    c.nodes[0].query(
        10,
        803,
        Query::ClosePermit {
            table: c.table,
            transaction: c.transaction,
            plan_index: plan,
            source_group: 20,
        },
    )?;
    c.pump()?
        .reads
        .into_iter()
        .find_map(|read| match read.outcome {
            ReadOutcome::Observed {
                value: Value::ClosePermit(permit),
                ..
            } => Some(permit.map(|p| *p)),
            _ => None,
        })
        .context("closure query missing")
}

fn publish(c: &mut Cluster, incarnation: Uuid, predecessor: Uuid) -> Result<ClosurePermit> {
    let table = c.nodes[0].publisher(c.table)?.unwrap().clone();
    let pending = table.progress.pending.unwrap();
    let descriptors = Descriptors::lance_fragments(Bytes::from_static(b"unit-test-descriptors"))?;
    c.nodes[0].prepare_publication(
        804,
        PrepareRequest {
            authority: table.authority.clone(),
            transaction: c.transaction,
            plan_index: pending.index,
            descriptors: descriptors.clone(),
        },
    )?;
    let publication::Reply::Prepared { index } = publisher(c.pump()?)? else {
        anyhow::bail!("prepare missing")
    };
    c.nodes[0].reserve_manifest(
        805,
        ReserveManifest {
            authority: table.authority.clone(),
            incarnation,
            transaction: c.transaction,
            plan_index: pending.index,
            prepared_index: index,
            descriptor_digest: descriptors.digest,
            predecessor,
            target: 9,
            object: ManifestObject {
                path: format!("_versions/9.manifest-{}", Uuid::new_v4()),
                bytes: 8192,
                v2_names: false,
            },
        },
    )?;
    let publication::Reply::ManifestReserved(selected) = publisher(c.pump()?)? else {
        anyhow::bail!("reservation missing")
    };
    assert!(
        closure_permit(c, pending.index)?.is_none(),
        "selection alone is not a closure outcome"
    );
    c.nodes[0].confirm_publication(
        806,
        ConfirmPublication {
            authority: table.authority,
            incarnation,
            transaction: c.transaction,
            plan_index: pending.index,
            prepared_index: index,
            descriptor_digest: descriptors.digest,
            target: 9,
            manifest_identity: selected.identity,
        },
    )?;
    assert!(matches!(
        publisher(c.pump()?)?,
        publication::Reply::Published(_)
    ));
    closure_permit(c, pending.index)?.context("published closure permit missing")
}

fn close(c: &mut Cluster, permit: ClosurePermit) -> Result<Reply> {
    c.nodes[3].close_source(807, permit)?;
    c.pump()?
        .completions
        .into_iter()
        .find_map(|completion| match completion.outcome {
            Outcome::Committed(Applied::Source(reply)) => Some(reply),
            _ => None,
        })
        .context("source closure missing")
}

#[test]
fn committed_source_page_captures_published_closure_and_live_suffix_after_reopen() -> Result<()> {
    let (mut c, pin, incarnation, predecessor) = fixture()?;
    let permit = publish(&mut c, incarnation, predecessor)?;
    c.append(c.table, 3)?;
    assert!(matches!(close(&mut c, permit)?, Reply::Closed { .. }));
    for reopened in [false, true] {
        if reopened {
            for node in &mut c.nodes[3..] {
                let prepared = node.prepare_checkpoint(20, Uuid::new_v4())?.run()?;
                node.install_checkpoint(prepared)?;
            }
            c.reopen()?;
        }
        let observed = c.nodes[3].read_source(
            20,
            &crate::replication::source_read::Request {
                table: c.table,
                bucket: 1,
                after: 0,
                limit: 128,
            },
        )?;
        let crate::replication::source_read::Value::Page { page, closure } = &observed.value else {
            anyhow::bail!("expected a committed source page");
        };
        let closure = closure.as_ref().context("missing durable source closure")?;
        assert_eq!(
            closure.permit.closure().through,
            pin.selection().range.through
        );
        assert_eq!(closure.permit.closure().incarnation, incarnation);
        assert_eq!(page.through, observed.applied);
        assert_eq!(page.entries.len(), 1);
        assert!(page.entries[0].0 > pin.selection().range.through);
        assert_eq!(page.entries[0].1.sequence, 3);
        assert_eq!(page.entries[0].1.stream, c.table);
        assert_eq!(page.entries[0].1.bucket, 1);
        assert_eq!(
            bincode::deserialize::<crate::replication::source_read::Observation>(
                &bincode::serialize(&observed)?
            )?,
            observed
        );
    }
    Ok(())
}

#[test]
fn closure_retires_selected_payloads_preserves_suffix_and_rejects_delayed_pins() -> Result<()> {
    let (mut c, pin, incarnation, predecessor) = fixture()?;
    assert!(closure_permit(&mut c, pin.selection().plan_index)?.is_none());
    let permit = publish(&mut c, incarnation, predecessor)?;
    c.append(c.table, 3)?;
    let before = c.nodes[3].groups[&20].state.clone();
    let result = close(&mut c, permit.clone())?;
    assert!(matches!(result, Reply::Closed { .. }));
    assert_eq!(c.pin(pin.clone())?, Reply::Rejected(Rejection::Closed));
    assert_eq!(close(&mut c, permit.clone())?, result);
    let state = &c.nodes[3].groups[&20].state;
    assert!(state.pins.is_empty());
    assert_eq!(
        state.closed.len(),
        1,
        "raw streams must not allocate closure history"
    );
    assert_eq!(state.accepted.len() + 2, before.accepted.len());
    assert_eq!(state.retired.len(), 1);
    assert_eq!(state.producers, before.producers);
    assert_eq!(
        state.closed[&c.table].pin_index,
        before.pins[&c.table].index
    );
    assert_eq!(
        state.closed[&c.table].permit.closure().through,
        pin.selection().range.through
    );
    for (n, node) in c.nodes[..6].iter_mut().enumerate() {
        let group = if n < 3 { 10 } else { 20 };
        let image = node.prepare_checkpoint(group, Uuid::new_v4())?.run()?;
        node.install_checkpoint(image)?;
    }
    c.reopen()?;
    c.acquire(1)?;
    let refreshed = closure_permit(&mut c, pin.selection().plan_index)?.unwrap();
    assert_ne!(permit.catalog_cut(), refreshed.catalog_cut());
    assert_eq!(close(&mut c, refreshed)?, result);
    assert_eq!(c.pin(pin.clone())?, Reply::Rejected(Rejection::Closed));
    for node in &c.nodes[3..] {
        let state = &node.groups[&20].state;
        assert!(state.pins.is_empty());
        state.validate_checkpoint(&c.layout, 20, &c.limits, state.applied)?;
        for (index, meta) in &before.accepted {
            if meta.stream == c.table && *index <= pin.selection().range.through {
                assert!(node.read_accepted(20, *index)?.is_none());
                continue;
            }
            let record = node
                .read_accepted(20, *index)?
                .context("retained source disappeared")?;
            assert_eq!(&record.meta, meta);
            assert_eq!(record.body, Bytes::from(vec![meta.sequence as u8; 4096]));
        }
    }
    Ok(())
}

#[test]
fn empty_payload_images_preserve_latest_retry_and_captured_old_images() -> Result<()> {
    use checkpoint::Part;
    let (mut c, pin, incarnation, predecessor) = fixture_with_background(2, 0)?;
    let original_index = pin.selection().range.records.last().unwrap().0;
    let permit = publish(&mut c, incarnation, predecessor)?;
    let captured = c.nodes[3].prepare_checkpoint(20, Uuid::new_v4())?;
    assert!(matches!(close(&mut c, permit)?, Reply::Closed { .. }));
    let prepared = captured.run()?;
    let old_image = prepared.image();
    assert_eq!(old_image.bytes(Part::Payload), 8192);
    assert_eq!(
        old_image.chunk(Part::Payload, 0, 4096)?,
        Bytes::from(vec![1; 4096])
    );
    drop(prepared);
    for node in &mut c.nodes[3..] {
        let prepared = node.prepare_checkpoint(20, Uuid::new_v4())?.run()?;
        assert_eq!(prepared.image().bytes(Part::Payload), 0);
        node.install_checkpoint(prepared)?;
    }
    // A captured image remains readable after its node is dropped, and keeps
    // the directory locked until its last reader releases it.
    c.nodes.clear();
    assert_eq!(
        old_image.chunk(Part::Payload, 4096, 4096)?,
        Bytes::from(vec![2; 4096])
    );
    drop(old_image);
    c.reopen()?;
    let retry = Batch::new(
        c.table,
        1,
        Uuid::from_u128(94),
        2,
        Bytes::from(vec![2; 4096]),
    );
    c.nodes[3].propose(20, 810, retry.clone())?;
    assert!(c.pump()?.completions.iter().any(|c| c.request == 810
        && c.outcome
            == Outcome::Committed(Applied::Accepted {
                index: original_index
            })));
    assert_eq!(
        c.nodes[3].status(20)?.accepted_records,
        0,
        "retry resurrected payload"
    );
    let mut changed = retry;
    changed.body = Bytes::from(vec![9; 4096]);
    changed.meta.digest = Sha256::digest(&changed.body).into();
    c.nodes[3].propose(20, 811, changed)?;
    assert!(c
        .pump()?
        .completions
        .iter()
        .any(|c| c.request == 811
            && matches!(c.outcome, Outcome::Committed(Applied::Rejected { .. }))));
    // Both record slots were reclaimed, while the producer sequence continued.
    c.append(c.table, 3)?;
    c.append(c.table, 4)?;
    assert_eq!(c.nodes[3].status(20)?.accepted_records, 2);
    for node in &c.nodes[3..] {
        let state = &node.groups[&20].state;
        state.validate_checkpoint(&c.layout, 20, &c.limits, state.applied)?;
        assert!(node.read_accepted(20, original_index)?.is_none());
    }
    Ok(())
}

#[test]
fn retirement_boundaries_cannot_fabricate_retry_receipts_or_skip_retained_sequences() -> Result<()>
{
    let (mut c, _, incarnation, predecessor) = fixture()?;
    let permit = publish(&mut c, incarnation, predecessor)?;
    c.append(c.table, 3)?;
    close(&mut c, permit)?;
    let state = c.nodes[3].groups[&20].state.clone();
    let key = (c.table, 1, Uuid::from_u128(94));
    for mutation in 0..8 {
        let mut invalid = state.clone();
        match mutation {
            0 => {
                invalid.retired.clear();
            }
            1 => {
                invalid.producers.remove(&key);
            }
            2 => invalid.retired.get_mut(&key).unwrap().sequence += 1,
            3 => invalid.retired.get_mut(&key).unwrap().index = state.applied,
            4 => {
                invalid.closed.remove(&c.table);
            }
            5 => {
                invalid.accepted.retain(|_, meta| meta.stream != c.table);
            }
            6 => invalid.producers.get_mut(&key).unwrap().1 -= 1,
            7 => {
                invalid
                    .retired
                    .insert((Uuid::new_v4(), 1, Uuid::new_v4()), invalid.retired[&key]);
            }
            _ => unreachable!(),
        }
        assert!(
            invalid
                .validate_checkpoint(&c.layout, 20, &c.limits, invalid.applied)
                .is_err(),
            "mutation {mutation}"
        );
    }
    Ok(())
}

#[test]
fn closure_signature_scope_and_snapshot_fences_are_checked() -> Result<()> {
    let (mut c, pin, incarnation, predecessor) = fixture()?;
    let permit = publish(&mut c, incarnation, predecessor)?;
    let before = c.nodes[3].groups[&20].raw.raft.raft_log.last_index();
    for field in ["through", "plan_index", "group"] {
        let mut value = serde_json::to_value(&permit)?;
        let slot = &mut value["closure"][field];
        *slot = serde_json::json!(slot.as_u64().unwrap() + 1);
        assert!(c.nodes[3]
            .close_source(808, serde_json::from_value(value)?)
            .is_err());
    }
    let mut bytes = bincode::serialize(&permit)?;
    *bytes.last_mut().unwrap() ^= 1;
    assert!(c.nodes[3]
        .close_source(808, bincode::deserialize(&bytes)?)
        .is_err());
    assert_eq!(
        c.nodes[3].groups[&20].raw.raft.raft_log.last_index(),
        before
    );
    assert!(matches!(close(&mut c, permit)?, Reply::Closed { .. }));
    let state = c.nodes[3].groups[&20].state.clone();
    for mutation in 0..4 {
        let mut invalid = state.clone();
        let closed = Arc::make_mut(invalid.closed.get_mut(&c.table).unwrap());
        match mutation {
            0 => closed.index = state.applied + 1,
            1 => closed.pin_index = closed.index,
            2 => closed.pin_index = closed.permit.closure().through,
            3 => {
                invalid.pins.insert(
                    c.table,
                    Arc::new(Pin {
                        index: before,
                        permit: pin.clone(),
                    }),
                );
            }
            _ => unreachable!(),
        }
        assert!(invalid
            .validate_checkpoint(&c.layout, 20, &c.limits, invalid.applied)
            .is_err());
    }
    Ok(())
}
