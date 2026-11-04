use super::*;
use std::fs;

fn layout() -> Arc<Layout> {
    super::super::tests::layout()
}

fn fixture() -> Result<(State, BTreeMap<u64, Bytes>)> {
    let mut state = State::default();
    let mut bodies = BTreeMap::new();
    // Blank, exact retry and rejected entries leave gaps in accepted indexes.
    for (index, sequence) in [(1, 0), (2, 1), (3, 1), (4, 3), (5, 2), (6, 0)] {
        let mut entry = Entry {
            index,
            term: 2,
            ..Default::default()
        };
        if sequence > 0 {
            let batch = super::super::tests::batch(sequence);
            entry.context = state::encode_meta(&batch.meta)?.into();
            entry.data = batch.body.clone();
            bodies.insert(index, batch.body);
        }
        state.apply(&entry, &Limits::default())?;
    }
    Ok((state, bodies))
}

fn create(directory: &Path, id: u128) -> Result<Loaded> {
    let (state, bodies) = fixture()?;
    build(
        directory,
        &layout(),
        &Limits::default(),
        Cut {
            group: 10,
            index: state.applied,
            term: 2,
            id: Uuid::from_u128(id),
        },
        state,
        |index, meta| {
            let body = bodies[&index].clone();
            meta.validate_payload(&body, Limits::default().max_record_bytes as usize)?;
            Ok(body)
        },
    )
}

fn copy(source: &Image, receiver: &mut Receiver) -> Result<()> {
    for part in [Part::Metadata, Part::Payload] {
        let mut offset = 0;
        while offset < source.bytes(part) {
            let bytes = source.chunk(
                part,
                offset,
                (source.bytes(part) - offset).min(997) as usize,
            )?;
            receiver.write(part, offset, &bytes)?;
            offset += bytes.len() as u64;
        }
    }
    Ok(())
}

#[test]
fn exact_state_payloads_and_streamed_transfer_without_changing_source_image() -> Result<()> {
    let source = tempfile::tempdir()?;
    let target = tempfile::tempdir()?;
    let built = create(source.path(), 101)?;
    let snapshot = built.image.snapshot()?;
    let restored = load(source.path(), &snapshot, &layout(), &Limits::default(), 10)?;
    assert_eq!(built.state, restored.state);
    assert_eq!(
        restored.state.accepted.keys().copied().collect::<Vec<_>>(),
        [2, 5]
    );
    assert_eq!(restored.state.producers.values().next().unwrap().1, 5);
    let mut receiver = Receiver::new(
        target.path(),
        &snapshot,
        layout(),
        Limits::default(),
        10,
        Uuid::from_u128(102),
    )?;
    copy(&built.image, &mut receiver)?;
    let received = receiver.finish()?;
    assert_eq!(received.state, restored.state);
    assert_ne!(received.image.descriptor.id, built.image.descriptor.id);
    assert_eq!(built.image.snapshot()?, snapshot);
    for index in [2, 5] {
        assert_eq!(
            restored.image.read_for_validation(index)?,
            received.image.read_for_validation(index)?
        );
        assert_eq!(
            received.image.read_for_validation(index)?.unwrap(),
            fixture()?.1[&index]
        );
    }
    assert!(received.image.read_for_validation(3)?.is_none());
    assert!(built.image.chunk(Part::Payload, 0, 0).is_err());
    assert!(built
        .image
        .chunk(Part::Payload, 0, CHUNK_BYTES + 1)
        .is_err());
    assert!(built.image.chunk(Part::Payload, u64::MAX, 1).is_err());
    assert!(
        create(source.path(), 101).is_err(),
        "existing immutable files overwritten"
    );
    assert_eq!(
        load(source.path(), &snapshot, &layout(), &Limits::default(), 10)?.state,
        restored.state
    );
    Ok(())
}

#[test]
fn received_images_reject_partial_corrupt_and_out_of_order_bytes() -> Result<()> {
    let source = tempfile::tempdir()?;
    let target = tempfile::tempdir()?;
    let built = create(source.path(), 111)?;
    let snapshot = built.image.snapshot()?;
    let receive = |id| {
        Receiver::new(
            target.path(),
            &snapshot,
            layout(),
            Limits::default(),
            10,
            Uuid::from_u128(id),
        )
    };
    let mut partial = receive(112)?;
    let first = built.image.chunk(Part::Metadata, 0, 8)?;
    assert!(partial.write(Part::Metadata, 1, &first).is_err());
    partial.write(Part::Metadata, 0, &first)?;
    assert!(
        partial.finish().is_err(),
        "partial state became installable"
    );
    let mut bad = receive(113)?;
    for part in [Part::Metadata, Part::Payload] {
        let mut bytes = built
            .image
            .chunk(part, 0, built.image.bytes(part) as usize)?
            .to_vec();
        if part == Part::Payload {
            bytes[0] ^= 1;
        }
        bad.write(part, 0, &bytes)?;
    }
    assert!(
        bad.finish().is_err(),
        "wrong payload hash became installable"
    );
    let mut good = receive(114)?;
    copy(&built.image, &mut good)?;
    assert_eq!(good.finish()?.state, built.state);
    assert!(
        receive(114).is_err(),
        "interrupted or existing files overwritten"
    );
    Ok(())
}

#[test]
fn received_images_validate_file_and_record_digests_before_install() -> Result<()> {
    let source = tempfile::tempdir()?;
    let built = create(source.path(), 115)?;
    // Both complete-file checks remain mandatory, even though receiving no
    // longer hashes the same bytes once on write and again on load.
    for metadata in [true, false] {
        let target = tempfile::tempdir()?;
        let mut descriptor = built.image.descriptor.clone();
        if metadata {
            descriptor.metadata_digest[0] ^= 1;
        } else {
            descriptor.payload_digest[0] ^= 1;
        }
        let mut receiver = Receiver::new(
            target.path(),
            &descriptor.snapshot()?,
            layout(),
            Limits::default(),
            10,
            Uuid::new_v4(),
        )?;
        copy(&built.image, &mut receiver)?;
        let error = receiver.finish().err().context("invalid image accepted")?;
        assert!(error.to_string().contains("image file checksum mismatch"));
    }
    // A matching full-pack digest must not hide a changed record body.
    let target = tempfile::tempdir()?;
    let mut payload = built
        .image
        .chunk(Part::Payload, 0, built.image.bytes(Part::Payload) as usize)?
        .to_vec();
    payload[0] ^= 1;
    let mut descriptor = built.image.descriptor.clone();
    descriptor.payload_digest = Sha256::digest(&payload).into();
    let mut receiver = Receiver::new(
        target.path(),
        &descriptor.snapshot()?,
        layout(),
        Limits::default(),
        10,
        Uuid::new_v4(),
    )?;
    let metadata = built.image.chunk(
        Part::Metadata,
        0,
        built.image.bytes(Part::Metadata) as usize,
    )?;
    receiver.write(Part::Metadata, 0, &metadata)?;
    receiver.write(Part::Payload, 0, &payload)?;
    let error = receiver.finish().err().context("invalid record accepted")?;
    assert!(error.to_string().contains("image record checksum mismatch"));
    Ok(())
}

#[test]
fn installed_files_fail_closed_on_corruption_missing_parts_and_foreign_binding() -> Result<()> {
    let root = tempfile::tempdir()?;
    let built = create(root.path(), 121)?;
    let snapshot = built.image.snapshot()?;
    let policy = Limits {
        max_records: 999,
        ..Default::default()
    };
    assert!(load(root.path(), &snapshot, &layout(), &policy, 10).is_err());
    assert!(load(root.path(), &snapshot, &layout(), &Limits::default(), 20).is_err());
    let mut cut = snapshot.clone();
    cut.mut_metadata().index += 1;
    assert!(load(root.path(), &cut, &layout(), &Limits::default(), 10).is_err());
    let paths = built.image.descriptor.paths(root.path());
    let original = fs::read(&paths[1])?;
    let mut corrupt = original.clone();
    corrupt[0] ^= 1;
    fs::write(&paths[1], &corrupt)?;
    assert!(built.state.accepted[&2]
        .validate_payload(
            &built.image.read_for_validation(2)?.unwrap(),
            8 * 1024 * 1024
        )
        .is_err());
    assert!(load(root.path(), &snapshot, &layout(), &Limits::default(), 10).is_err());
    // Even a matching outer digest cannot hide a record/metadata mismatch.
    let mut descriptor = built.image.descriptor.clone();
    descriptor.payload_digest = Sha256::digest(&corrupt).into();
    assert!(load(
        root.path(),
        &descriptor.snapshot()?,
        &layout(),
        &Limits::default(),
        10
    )
    .is_err());
    fs::write(&paths[1], original)?;
    fs::remove_file(&paths[0])?;
    assert!(load(root.path(), &snapshot, &layout(), &Limits::default(), 10).is_err());
    Ok(())
}

#[test]
fn metadata_counts_frames_trailing_bytes_and_exact_producer_tails_are_validated() -> Result<()> {
    let root = tempfile::tempdir()?;
    let built = create(root.path(), 131)?;
    let path = built.image.descriptor.paths(root.path())[0].clone();
    let original = fs::read(&path)?;
    let try_bytes = |bytes: &[u8]| -> Result<()> {
        fs::write(&path, bytes)?;
        let mut descriptor = built.image.descriptor.clone();
        descriptor.metadata_bytes = bytes.len() as u64;
        descriptor.metadata_digest = Sha256::digest(bytes).into();
        assert!(load(
            root.path(),
            &descriptor.snapshot()?,
            &layout(),
            &Limits::default(),
            10
        )
        .is_err());
        Ok(())
    };
    let mut trailing = original.clone();
    trailing.push(0);
    try_bytes(&trailing)?;
    let mut frame = original.clone();
    frame[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    try_bytes(&frame)?;
    let binding_bytes = u32::from_le_bytes(original[8..12].try_into()?) as usize;
    let mut count = original.clone();
    count[12 + binding_bytes..20 + binding_bytes].copy_from_slice(&u64::MAX.to_le_bytes());
    try_bytes(&count)?;
    fs::write(&path, original)?;
    let (state, _) = fixture()?;
    for mutation in 0..7 {
        let mut invalid = state.clone();
        match mutation {
            0 => invalid.producers.clear(),
            1 => invalid.producers.values_mut().next().unwrap().1 = 2,
            2 => invalid.accepted.get_mut(&5).unwrap().sequence = 3,
            _ => {
                invalid.accepted.insert(7, invalid.accepted[&5].clone());
            }
        }
        assert!(invalid
            .validate_checkpoint(&layout(), 10, &Limits::default(), 6)
            .is_err());
    }
    Ok(())
}

#[test]
fn empty_application_cut_has_a_real_empty_pack_and_no_payload_fetch() -> Result<()> {
    let root = tempfile::tempdir()?;
    let state = State {
        applied: 1,
        ..Default::default()
    };
    let built = build(
        root.path(),
        &layout(),
        &Limits::default(),
        Cut {
            group: 10,
            index: 1,
            term: 1,
            id: Uuid::from_u128(141),
        },
        state.clone(),
        |_, _| anyhow::bail!("unexpected payload fetch"),
    )?;
    assert_eq!(built.image.bytes(Part::Payload), 0);
    assert_eq!(
        load(
            root.path(),
            &built.image.snapshot()?,
            &layout(),
            &Limits::default(),
            10
        )?
        .state,
        state
    );
    Ok(())
}

#[test]
fn mixed_catalog_image_preserves_payloads_and_rejects_corrupt_table_frames() -> Result<()> {
    use crate::ingest_contract::{IngestContract, WritePolicy};
    let root = tempfile::tempdir()?;
    let (mut state, bodies) = fixture()?;
    for id in [200, 201] {
        let batch = publication::Acquire {
            table: Uuid::from_u128(id),
            destination: publication::Destination {
                namespace: Uuid::from_u128(202),
                dataset: format!("tables/{id}"),
            },
            ingest: IngestContract {
                tenant: "test".into(),
                qualified_table: format!("table-{id}"),
                schema: [7; 32],
                policy: WritePolicy::ImmutableUniqueIds,
                max_record_bytes: 4096,
                max_rows: 32,
            },
            owner: Uuid::from_u128(203),
            request: Uuid::from_u128(id + 10),
            expected_epoch: 0,
        }
        .batch(&layout())?;
        let entry = Entry {
            index: state.applied + 1,
            term: 2,
            context: state::encode_meta(&batch.meta)?.into(),
            data: batch.body,
            ..Default::default()
        };
        state::validate_entry(&layout(), 10, &Limits::default(), &entry, true)?;
        state.apply(&entry, &Limits::default())?;
    }
    let table = &state.catalog.tables[&Uuid::from_u128(200)];
    let mut records = Vec::new();
    for index in 1..=publication::MAX_INPUTS as u64 {
        records.push((
            index,
            Batch::new(
                table.authority.table,
                1,
                Uuid::from_u128(230),
                index,
                Bytes::from_static(b"source"),
            )
            .meta,
        ));
    }
    let plan = publication::PlanRequest {
        authority: table.authority.clone(),
        plan: publication::Plan {
            transaction: Uuid::from_u128(231),
            read_version: 7,
            sources: BTreeMap::from([(
                20,
                publication::SourceRange {
                    after: 0,
                    through: publication::MAX_INPUTS as u64,
                    records,
                },
            )]),
            next_group: 10,
        },
    }
    .batch(&layout(), &Limits::default())?;
    assert!(plan.body.len() > publication::MAX_ACQUIRE_BYTES);
    let entry = Entry {
        index: state.applied + 1,
        term: 2,
        context: state::encode_meta(&plan.meta)?.into(),
        data: plan.body,
        ..Default::default()
    };
    state::validate_entry(&layout(), 10, &Limits::default(), &entry, true)?;
    state.apply(&entry, &Limits::default())?;
    let built = build(
        root.path(),
        &layout(),
        &Limits::default(),
        Cut {
            group: 10,
            index: state.applied,
            term: 2,
            id: Uuid::from_u128(220),
        },
        state.clone(),
        |index, meta| {
            let body = bodies[&index].clone();
            meta.validate_payload(&body, Limits::default().max_record_bytes as usize)?;
            Ok(body)
        },
    )?;
    let snapshot = built.image.snapshot()?;
    assert_eq!(
        load(root.path(), &snapshot, &layout(), &Limits::default(), 10)?.state,
        state
    );
    assert_eq!(built.image.bytes(Part::Payload), 8192);
    for index in [2, 5] {
        assert_eq!(
            built.image.read_for_validation(index)?,
            Some(bodies[&index].clone())
        );
    }
    let path = built.image.descriptor.paths(root.path())[0].clone();
    let original = fs::read(&path)?;
    // Locate frames from their encoded lengths, without assuming a fixed header.
    let binding_len = u32::from_le_bytes(original[8..12].try_into()?) as usize;
    let mut header = &original[12 + binding_len..];
    for count in [
        state.accepted.len(),
        state.producers.len(),
        state.catalog.tables.len(),
        state.pins.len(),
        state.closed.len(),
        state.retired.len(),
    ] {
        assert_eq!(read_u64(&mut header)?, count as u64);
    }
    let mut offset = original.len() - header.len();
    for _ in 0..state.accepted.len() + state.producers.len() {
        offset += 8; // accepted index
        let len = u32::from_le_bytes(original[offset..offset + 4].try_into()?) as usize;
        offset += 4 + len;
    }
    let first_len = u32::from_le_bytes(original[offset..offset + 4].try_into()?) as usize;
    let second = offset + 4 + first_len;
    let second_len = u32::from_le_bytes(original[second..second + 4].try_into()?) as usize;
    let first_table = publication::decode_table(&original[offset + 4..second])?;
    let second_table = publication::decode_table(&original[second + 4..second + 4 + second_len])?;
    for mutation in 0..7 {
        let mut bytes = original[..offset].to_vec();
        let mut tables = [first_table.clone(), second_table.clone()];
        match mutation {
            0 => tables.swap(0, 1),
            1 => tables[1] = tables[0].clone(),
            2 => tables[1].destination = tables[0].destination.clone(),
            3 => tables[1].authority.grant_index = state.applied + 1,
            4 => {
                Arc::make_mut(tables[0].progress.pending.as_mut().unwrap()).index =
                    state.applied + 1
            }
            5 => {
                Arc::make_mut(tables[0].progress.pending.as_mut().unwrap())
                    .plan
                    .sources
                    .get_mut(&20)
                    .unwrap()
                    .after = 1
            }
            _ => {
                Arc::make_mut(tables[0].progress.pending.as_mut().unwrap())
                    .plan
                    .sources
                    .get_mut(&20)
                    .unwrap()
                    .records[0]
                    .1
                    .bucket = 0
            }
        }
        for table in &tables {
            write_frame(&mut bytes, &publication::encode_table(table)?)?;
        }
        fs::write(&path, &bytes)?;
        let mut descriptor = built.image.descriptor.clone();
        descriptor.metadata_bytes = bytes.len() as u64;
        descriptor.metadata_digest = Sha256::digest(&bytes).into();
        assert!(load(
            root.path(),
            &descriptor.snapshot()?,
            &layout(),
            &Limits::default(),
            10
        )
        .is_err());
    }
    fs::write(&path, original)?;
    let target = tempfile::tempdir()?;
    let mut receiver = Receiver::new(
        target.path(),
        &snapshot,
        layout(),
        Limits::default(),
        10,
        Uuid::from_u128(221),
    )?;
    copy(&built.image, &mut receiver)?;
    assert_eq!(receiver.finish()?.state, state);
    Ok(())
}
