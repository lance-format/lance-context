use super::*;

fn layout() -> Arc<Layout> {
    crate::replication::tests::layout()
}
fn acquire() -> Acquire {
    super::super::tests::input()
}
fn catalog() -> Result<(Catalog, PlanRequest)> {
    let mut catalog = Catalog::default();
    let a = acquire();
    let batch = a.clone().batch(&layout())?;
    let Reply::Acquired(authority) = catalog.apply(decode(&batch.body)?, 4, &Limits::default())
    else {
        anyhow::bail!("missing authority");
    };
    let meta = Batch::new(
        a.table,
        1,
        Uuid::from_u128(90),
        1,
        Bytes::from_static(b"record"),
    )
    .meta;
    let request = PlanRequest {
        authority,
        plan: Plan {
            transaction: Uuid::from_u128(91),
            read_version: 7,
            sources: BTreeMap::from([(
                20,
                SourceRange {
                    after: 0,
                    through: 1000,
                    records: vec![(999, meta)],
                },
            )]),
            next_group: 10,
        },
    };
    Ok((catalog, request))
}
fn apply(
    catalog: &mut Catalog,
    request: PlanRequest,
    index: u64,
    limits: &Limits,
) -> Result<Reply> {
    let batch = request.batch(&layout(), limits)?;
    validate_record(&layout(), 10, limits, &batch.meta, &batch.body)?;
    Ok(catalog.apply(decode(&batch.body)?, index, limits))
}

#[test]
fn exact_plan_and_shared_view_survive_takeover_without_releasing_capacity() -> Result<()> {
    let (mut catalog, mut request) = catalog()?;
    let limits = Limits {
        max_active_plans: 1,
        ..Default::default()
    };
    assert_eq!(
        apply(&mut catalog, request.clone(), 5, &limits)?,
        Reply::Planned { index: 5 }
    );
    assert_eq!(
        apply(&mut catalog, request.clone(), 6, &limits)?,
        Reply::Planned { index: 5 }
    );
    let captured = catalog.tables[&request.authority.table].clone();
    let pending = captured.progress.pending.as_ref().unwrap();
    assert_eq!(pending.plan.target_version()?, 8);
    assert!(Arc::ptr_eq(
        pending,
        catalog.tables[&request.authority.table]
            .progress
            .pending
            .as_ref()
            .unwrap()
    ));
    let mut changed = request.clone();
    changed.plan.transaction = Uuid::from_u128(92);
    assert_eq!(
        apply(&mut catalog, changed, 7, &limits)?,
        Reply::Rejected(Rejection::PendingPlan)
    );

    let mut a = acquire();
    a.owner = Uuid::from_u128(93);
    a.request = Uuid::from_u128(94);
    a.expected_epoch = 1;
    let batch = a.batch(&layout())?;
    let Reply::Acquired(next) = catalog.apply(decode(&batch.body)?, 8, &limits) else {
        anyhow::bail!("takeover blocked by plan capacity");
    };
    assert_eq!(
        apply(&mut catalog, request.clone(), 9, &limits)?,
        Reply::Rejected(Rejection::Authority)
    );
    request.authority = next;
    assert_eq!(
        apply(&mut catalog, request.clone(), 10, &limits)?,
        Reply::Planned { index: 5 }
    );
    assert!(Arc::ptr_eq(
        pending,
        catalog.tables[&request.authority.table]
            .progress
            .pending
            .as_ref()
            .unwrap()
    ));
    assert_eq!(captured.authority.epoch, 1);
    catalog.validate(&layout(), 10, 10, &limits)?;

    let mut other = acquire();
    other.table = Uuid::from_u128(95);
    other.destination.dataset = "another/table".into();
    let batch = other.batch(&layout())?;
    let Reply::Acquired(other) = catalog.apply(decode(&batch.body)?, 11, &limits) else {
        anyhow::bail!("another table could not bind");
    };
    request.authority = other.clone();
    request.plan.sources.get_mut(&20).unwrap().records[0]
        .1
        .stream = other.table;
    assert_eq!(
        apply(&mut catalog, request, 12, &limits)?,
        Reply::Rejected(Rejection::Capacity)
    );
    assert_eq!(
        catalog.tables[&captured.authority.table]
            .progress
            .checkpoint
            .len(),
        0
    );
    assert_eq!(
        catalog.tables[&captured.authority.table]
            .progress
            .minimum_read_version,
        1
    );
    Ok(())
}

#[test]
fn source_positions_are_per_group_and_plans_cannot_skip_publication_checkpoints() -> Result<()> {
    let (mut catalog, mut request) = catalog()?;
    let limits = Limits::default();
    request.plan.sources.get_mut(&20).unwrap().after = 998;
    assert_eq!(
        apply(&mut catalog, request.clone(), 5, &limits)?,
        Reply::Rejected(Rejection::SourcePosition)
    );
    request.plan.sources.get_mut(&20).unwrap().after = 0;
    let mut local = request.clone();
    let mut range = local.plan.sources.remove(&20).unwrap();
    range.records[0].1.bucket = 0;
    local.plan.sources.insert(10, range);
    assert_eq!(
        apply(&mut catalog, local, 6, &limits)?,
        Reply::Rejected(Rejection::SourcePosition)
    );
    // The same numeric position is valid in an independent source group. A
    // catalog index must never be used as a global source high-water mark.
    assert_eq!(
        apply(&mut catalog, request, 7, &limits)?,
        Reply::Planned { index: 7 }
    );
    catalog.validate(&layout(), 10, 7, &limits)?;
    Ok(())
}

#[test]
fn plan_bounds_reject_bad_sources_before_admission_and_allow_large_metadata() -> Result<()> {
    let (mut catalog, request) = catalog()?;
    let limits = Limits::default();
    for mutate in [
        (|r: &mut PlanRequest| r.plan.transaction = Uuid::nil()) as fn(&mut PlanRequest),
        |r| r.plan.read_version = 0,
        |r| r.plan.read_version = u64::MAX,
        |r| r.plan.next_group = 99,
        |r| r.authority.layout[0] ^= 1,
        |r| r.plan.sources.get_mut(&20).unwrap().through = u64::MAX,
        |r| r.plan.sources.get_mut(&20).unwrap().records[0].0 = 0,
        |r| r.plan.sources.get_mut(&20).unwrap().records[0].1.bucket = 0,
        |r| r.plan.sources.get_mut(&20).unwrap().records[0].1.stream = Uuid::from_u128(100),
        |r| r.plan.sources.get_mut(&20).unwrap().records[0].1.sequence = 0,
        |r| r.plan.sources.get_mut(&20).unwrap().records[0].1.body_bytes = 0,
        |r| {
            let range = r.plan.sources.get_mut(&20).unwrap();
            range.records.push(range.records[0].clone());
        },
    ] {
        let mut invalid = request.clone();
        mutate(&mut invalid);
        assert!(invalid.clone().batch(&layout(), &limits).is_err());
        // A hand-crafted wire request must also fail validation at the replica.
        let batch = Command::Plan(Box::new(invalid)).batch(&layout())?;
        assert!(validate_record(&layout(), 10, &limits, &batch.meta, &batch.body).is_err());
    }
    let tiny = Limits {
        max_plan_source_bytes: 5,
        ..limits.clone()
    };
    assert!(request.clone().batch(&layout(), &tiny).is_err());
    let mut aggregate = request.clone();
    let range = aggregate.plan.sources.get_mut(&20).unwrap();
    range.records[0].1.body_bytes = 4;
    range.records.push((
        1000,
        BatchMeta {
            sequence: 2,
            ..range.records[0].1.clone()
        },
    ));
    assert!(aggregate
        .batch(
            &layout(),
            &Limits {
                max_plan_source_bytes: 6,
                ..limits.clone()
            }
        )
        .is_err());
    let mut large = request;
    let range = large.plan.sources.get_mut(&20).unwrap();
    range.through = MAX_INPUTS as u64 + 1;
    let meta = range.records[0].1.clone();
    range.records = (1..=MAX_INPUTS as u64)
        .map(|index| {
            (
                index,
                BatchMeta {
                    sequence: index,
                    ..meta.clone()
                },
            )
        })
        .collect();
    let batch = large.clone().batch(&layout(), &limits)?;
    assert!(batch.body.len() > MAX_ACQUIRE_BYTES);
    assert_eq!(
        apply(&mut catalog, large.clone(), 5, &limits)?,
        Reply::Planned { index: 5 }
    );
    let table = &catalog.tables[&large.authority.table];
    assert_eq!(decode_table(&encode_table(table)?)?, *table);
    // Each group fits separately, but the limit is shared by the whole plan.
    let mut aggregate = large.clone();
    let range = aggregate.plan.sources.get_mut(&20).unwrap();
    let mut second = range.clone();
    range.records.truncate(MAX_INPUTS / 2 + 1);
    second.records.truncate(MAX_INPUTS / 2 + 1);
    for (_, meta) in &mut second.records {
        meta.bucket = 0;
    }
    aggregate.plan.sources.insert(10, second);
    assert!(aggregate.clone().batch(&layout(), &limits).is_err());
    let invalid = Command::Plan(Box::new(aggregate)).batch(&layout())?;
    assert!(validate_record(&layout(), 10, &limits, &invalid.meta, &invalid.body).is_err());
    large
        .plan
        .sources
        .get_mut(&20)
        .unwrap()
        .records
        .push((MAX_INPUTS as u64 + 1, meta));
    assert!(large.batch(&layout(), &limits).is_err());
    Ok(())
}

#[test]
fn pending_plan_checkpoint_validation_checks_inner_semantics() -> Result<()> {
    let (mut catalog, request) = catalog()?;
    let limits = Limits::default();
    apply(&mut catalog, request.clone(), 5, &limits)?;
    let saved = catalog.tables[&request.authority.table].clone();
    for mutate in [
        (|p: &mut PendingPlan| p.index = 0) as fn(&mut PendingPlan),
        |p| p.index = 4,
        |p| p.index = 6,
        |p| p.plan.read_version = u64::MAX,
        |p| p.plan.sources.get_mut(&20).unwrap().after = 1,
        |p| p.plan.sources.get_mut(&20).unwrap().records[0].1.body_bytes = 2 * 1024 * 1024,
    ] {
        let mut table = saved.clone();
        mutate(Arc::make_mut(table.progress.pending.as_mut().unwrap()));
        catalog.tables.insert(request.authority.table, table);
        assert!(catalog.validate(&layout(), 10, 5, &limits).is_err());
    }
    assert_eq!(saved.progress.pending.as_ref().unwrap().index, 5);
    catalog
        .tables
        .insert(request.authority.table, saved.clone());
    let mut other = saved;
    other.authority.table = Uuid::from_u128(101);
    other.destination.dataset = "other/table".into();
    Arc::make_mut(other.progress.pending.as_mut().unwrap())
        .plan
        .sources
        .get_mut(&20)
        .unwrap()
        .records[0]
        .1
        .stream = other.authority.table;
    catalog.tables.insert(other.authority.table, other);
    catalog.validate(&layout(), 10, 5, &limits)?;
    assert!(catalog
        .validate(
            &layout(),
            10,
            5,
            &Limits {
                max_active_plans: 1,
                ..limits
            }
        )
        .is_err());
    Ok(())
}

#[test]
fn wire_and_checkpoint_maps_cannot_hide_duplicate_source_groups() -> Result<()> {
    #[derive(Serialize)]
    struct WirePlan<'a> {
        transaction: Uuid,
        read_version: u64,
        sources: Vec<(u64, &'a SourceRange)>,
        next_group: u64,
    }
    fn replace_once(body: &[u8], old: &[u8], new: &[u8]) -> Vec<u8> {
        let positions: Vec<_> = body
            .windows(old.len())
            .enumerate()
            .filter_map(|(i, bytes)| (bytes == old).then_some(i))
            .collect();
        assert_eq!(positions.len(), 1);
        let start = positions[0];
        [
            body[..start].to_vec(),
            new.to_vec(),
            body[start + old.len()..].to_vec(),
        ]
        .concat()
    }
    let (mut catalog, request) = catalog()?;
    let limits = Limits::default();
    let original = codec(MAX_COMMAND_BYTES).serialize(&request.plan)?;
    let range = &request.plan.sources[&20];
    let duplicate = codec(MAX_COMMAND_BYTES).serialize(&WirePlan {
        transaction: request.plan.transaction,
        read_version: request.plan.read_version,
        sources: vec![(20, range), (20, range)],
        next_group: request.plan.next_group,
    })?;
    let batch = request.clone().batch(&layout(), &limits)?;
    let bad = replace_once(&batch.body, &original, &duplicate);
    let decoded: Envelope = codec(MAX_COMMAND_BYTES).deserialize(&bad)?;
    let Command::Plan(normalized) = decoded.command else {
        anyhow::bail!("wrong command")
    };
    assert_eq!(
        *normalized, request,
        "bincode did not normalize the malformed map"
    );
    assert!(decode(&bad).is_err());
    apply(&mut catalog, request.clone(), 5, &limits)?;
    let table = &catalog.tables[&request.authority.table];
    // Public query tables include the advancement field; checkpoint frames
    // preserve the legacy prefix and carry advancement in a tagged suffix.
    let query = codec(MAX_TABLE_BYTES).serialize(table)?;
    let bad_query = replace_once(&query, &original, &duplicate);
    assert_eq!(
        codec(MAX_TABLE_BYTES).deserialize::<Table>(&bad_query)?,
        *table
    );
    let checkpoint = encode_table(table)?;
    assert_eq!(decode_table(&checkpoint)?, *table);
    let bad_checkpoint = replace_once(&checkpoint, &original, &duplicate);
    assert!(decode_table(&bad_checkpoint).is_err());
    Ok(())
}
