use super::*;

fn fixture() -> Result<(Catalog, PlanRequest)> {
    fixture_with_policy(WritePolicy::ImmutableUniqueIds)
}

fn fixture_with_policy(policy: WritePolicy) -> Result<(Catalog, PlanRequest)> {
    let layout = crate::replication::tests::layout();
    let limits = Limits::default();
    let mut catalog = Catalog::default();
    let mut a = super::super::tests::input();
    a.ingest.policy = policy;
    let batch = a.clone().batch(&layout)?;
    let Reply::Acquired(authority) = catalog.apply(decode(&batch.body)?, 4, &limits) else {
        anyhow::bail!("missing authority");
    };
    let request = PlanRequest {
        authority,
        plan: Plan {
            transaction: Uuid::from_u128(91),
            read_version: 7,
            next_group: 10,
            sources: BTreeMap::from([(
                20,
                SourceRange {
                    after: 0,
                    through: 1000,
                    records: vec![(
                        999,
                        Batch::new(
                            a.table,
                            1,
                            Uuid::from_u128(90),
                            1,
                            Bytes::from_static(b"data"),
                        )
                        .meta,
                    )],
                },
            )]),
        },
    };
    Ok((catalog, request))
}

fn apply(
    catalog: &mut Catalog,
    request: PrepareRequest,
    index: u64,
    limits: &Limits,
) -> Result<Reply> {
    let layout = crate::replication::tests::layout();
    let batch = request.batch(&layout, limits)?;
    validate_record(&layout, 10, limits, &batch.meta, &batch.body)?;
    Ok(catalog.apply(decode(&batch.body)?, index, limits))
}

#[test]
fn descriptors_keep_original_index_and_shared_bytes_through_takeover() -> Result<()> {
    for policy in [
        WritePolicy::ImmutableUniqueIds,
        WritePolicy::TurnAnnotations,
    ] {
        takeover(policy)?;
    }
    Ok(())
}

fn takeover(policy: WritePolicy) -> Result<()> {
    let (mut catalog, plan) = fixture_with_policy(policy.clone())?;
    let descriptors = |bytes| match policy {
        WritePolicy::ImmutableUniqueIds => Descriptors::lance_fragments(bytes),
        WritePolicy::TurnAnnotations => Descriptors::lance_annotations(bytes),
    };
    let layout = crate::replication::tests::layout();
    let limits = Limits {
        max_active_plans: 1,
        ..Default::default()
    };
    let mut request = PrepareRequest {
        authority: plan.authority.clone(),
        transaction: plan.plan.transaction,
        plan_index: 5,
        descriptors: descriptors(Bytes::from_static(b"fragment-descriptors"))?,
    };
    assert_eq!(
        apply(&mut catalog, request.clone(), 5, &limits)?,
        Reply::Rejected(Rejection::PlanIdentity)
    );
    let batch = plan.clone().batch(&layout, &limits)?;
    assert_eq!(
        catalog.apply(decode(&batch.body)?, 5, &limits),
        Reply::Planned { index: 5 }
    );
    let before = catalog.tables[&plan.authority.table].clone();
    for mutation in 0..2 {
        let mut wrong = request.clone();
        if mutation == 0 {
            wrong.plan_index += 1;
        } else {
            wrong.transaction = Uuid::new_v4();
        }
        assert_eq!(
            apply(&mut catalog, wrong, 6, &limits)?,
            Reply::Rejected(Rejection::PlanIdentity)
        );
    }
    assert_eq!(
        apply(&mut catalog, request.clone(), 6, &limits)?,
        Reply::Prepared { index: 6 }
    );
    let captured = catalog.tables[&plan.authority.table].clone();
    let saved = captured.progress.prepared.as_ref().unwrap();
    assert!(before.progress.prepared.is_none());
    assert!(Arc::ptr_eq(
        before.progress.pending.as_ref().unwrap(),
        captured.progress.pending.as_ref().unwrap()
    ));
    assert!(Arc::ptr_eq(
        saved,
        catalog.tables[&plan.authority.table]
            .progress
            .prepared
            .as_ref()
            .unwrap()
    ));
    assert_eq!(
        apply(&mut catalog, request.clone(), 7, &limits)?,
        Reply::Prepared { index: 6 }
    );
    let mut changed = request.clone();
    changed.descriptors = descriptors(Bytes::from_static(b"different-files"))?;
    assert_eq!(
        apply(&mut catalog, changed, 8, &limits)?,
        Reply::Rejected(Rejection::PreparedConflict)
    );
    let mut acquire = super::super::tests::input();
    acquire.ingest.policy = policy.clone();
    acquire.owner = Uuid::from_u128(93);
    acquire.request = Uuid::from_u128(94);
    acquire.expected_epoch = 1;
    let batch = acquire.batch(&layout)?;
    let Reply::Acquired(authority) = catalog.apply(decode(&batch.body)?, 9, &limits) else {
        anyhow::bail!("takeover blocked by active-plan capacity");
    };
    assert_eq!(
        apply(&mut catalog, request.clone(), 10, &limits)?,
        Reply::Rejected(Rejection::Authority)
    );
    request.authority = authority;
    assert_eq!(
        apply(&mut catalog, request, 11, &limits)?,
        Reply::Prepared { index: 6 }
    );
    let recovered = decode_table(&encode_table(&catalog.tables[&plan.authority.table])?)?;
    assert_eq!(recovered.progress.prepared.as_deref(), Some(saved.as_ref()));
    catalog.validate(&layout, 10, 11, &limits)?;
    let mut wrong_policy = catalog.clone();
    wrong_policy
        .tables
        .get_mut(&plan.authority.table)
        .unwrap()
        .ingest
        .policy = match policy {
        WritePolicy::ImmutableUniqueIds => WritePolicy::TurnAnnotations,
        WritePolicy::TurnAnnotations => WritePolicy::ImmutableUniqueIds,
    };
    assert!(wrong_policy.validate(&layout, 10, 11, &limits).is_err());
    for mutation in 0..5 {
        let mut invalid = catalog.clone();
        let progress = &mut invalid
            .tables
            .get_mut(&plan.authority.table)
            .unwrap()
            .progress;
        let prepared = Arc::make_mut(progress.prepared.as_mut().unwrap());
        match mutation {
            0 => prepared.index = prepared.plan_index,
            1 => prepared.plan_index += 1,
            2 => prepared.transaction = Uuid::new_v4(),
            3 => prepared.descriptors.digest[0] ^= 1,
            4 => progress.pending = None,
            _ => unreachable!(),
        }
        assert!(invalid.validate(&layout, 10, 11, &limits).is_err());
    }
    assert_eq!(captured.authority.epoch, 1);
    assert!(catalog.tables[&plan.authority.table]
        .progress
        .checkpoint
        .is_empty());
    Ok(())
}

#[test]
fn annotation_descriptors_require_an_annotation_binding_and_a_source_prefix() -> Result<()> {
    let layout = crate::replication::tests::layout();
    let limits = Limits::default();
    for policy in [
        WritePolicy::ImmutableUniqueIds,
        WritePolicy::TurnAnnotations,
    ] {
        for maintenance in [false, true] {
            for format in 1..=4 {
                let (mut catalog, mut plan) = fixture_with_policy(policy.clone())?;
                if maintenance {
                    plan.plan.sources.clear();
                }
                let batch = plan.clone().batch(&layout, &limits)?;
                assert_eq!(
                    catalog.apply(decode(&batch.body)?, 5, &limits),
                    Reply::Planned { index: 5 }
                );
                let accepted = if maintenance {
                    matches!(format, 2 | 3)
                } else {
                    match policy {
                        WritePolicy::ImmutableUniqueIds => format == 1,
                        WritePolicy::TurnAnnotations => format == 4,
                    }
                };
                let before = catalog.clone();
                let request = PrepareRequest {
                    authority: plan.authority.clone(),
                    transaction: plan.plan.transaction,
                    plan_index: 5,
                    descriptors: Descriptors::new(format, Bytes::from_static(b"metadata"))?,
                };
                assert_eq!(
                    apply(&mut catalog, request, 6, &limits)?,
                    if accepted {
                        Reply::Prepared { index: 6 }
                    } else {
                        Reply::Rejected(Rejection::PreparedConflict)
                    },
                    "policy={policy:?}, maintenance={maintenance}, format={format}"
                );
                if !accepted {
                    assert_eq!(catalog, before);
                }
                catalog.validate(&layout, 10, 6, &limits)?;
                assert!(catalog.tables[&plan.authority.table]
                    .progress
                    .checkpoint
                    .is_empty());
            }
        }
    }
    Ok(())
}

#[test]
fn descriptor_budget_digest_and_wire_type_limits_apply_before_admission() -> Result<()> {
    let (_, plan) = fixture()?;
    let layout = crate::replication::tests::layout();
    let limits = Limits {
        max_descriptor_bytes: 512,
        ..Default::default()
    };
    let request = PrepareRequest {
        authority: plan.authority,
        transaction: plan.plan.transaction,
        plan_index: 5,
        descriptors: Descriptors::lance_fragments(Bytes::from(vec![7; 512]))?,
    };
    request.clone().batch(&layout, &limits)?;
    for mutation in 0..6 {
        let mut invalid = request.clone();
        match mutation {
            0 => invalid.descriptors = Descriptors::lance_fragments(Bytes::from(vec![7; 513]))?,
            1 => invalid.descriptors.bytes = Bytes::new(),
            2 => invalid.descriptors.digest[0] ^= 1,
            3 => invalid.descriptors.format = 5,
            4 => invalid.authority.layout[0] ^= 1,
            5 => invalid.plan_index = 0,
            _ => unreachable!(),
        }
        assert!(invalid.clone().batch(&layout, &limits).is_err());
        let batch = Command::Prepare(Box::new(invalid)).batch(&layout)?;
        assert!(validate_record(&layout, 10, &limits, &batch.meta, &batch.body).is_err());
    }
    assert!(Descriptors::lance_fragments(Bytes::from(vec![0; MAX_DESCRIPTOR_BYTES + 1])).is_err());
    let mut large = request;
    large.descriptors = Descriptors::lance_fragments(Bytes::from(vec![7; MAX_DESCRIPTOR_BYTES]))?;
    large.batch(
        &layout,
        &Limits {
            max_descriptor_bytes: MAX_DESCRIPTOR_BYTES as u64,
            ..Default::default()
        },
    )?;
    Ok(())
}
