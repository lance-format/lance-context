use super::*;
#[tokio::test]
async fn mixed_replays_an_unprepared_cut_then_drains_bodies_and_annotations() -> Result<()> {
    resume(0).await
}

#[tokio::test]
async fn mixed_recovers_exact_prepared_pair_without_reimporting_turns() -> Result<()> {
    resume(1).await
}

#[tokio::test]
async fn mixed_resolves_unknown_selection_and_replays_only_its_suffix() -> Result<()> {
    resume(2).await
}

async fn resume(stage: u8) -> Result<()> {
    let root = artifact_root()?;
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed(&c, &narrow, &body).await?;
    let (a, driver) = connect(&c, &narrow, &body).await?;
    let (stop, task) = run(driver);
    let p = producer(&a, 1).await?;
    let initial = request(
        &a,
        &p,
        1,
        encoded(create(&["A", "A", "B"], &["s", "wrong", "s"], &[1, 9, 2])?)?,
    )
    .await?;
    a.submit(&initial, Duration::from_secs(10)).await?;
    check(&a, "A", "s", 1, "pending", None).await?;
    let (other, driver) = connect(&c, &narrow, &body).await?;
    let (stop_other, other_task) = run(driver);
    check(&other, "A", "s", 1, "pending", None).await?;
    stop_other.send(()).unwrap();
    other_task.await??;
    assert_eq!(
        other
            .phase_observation()
            .iter()
            .find(|p| p.phase == "annotation_fetch")
            .unwrap()
            .completed,
        1,
        "another service must fetch the original committed create"
    );
    drop(other);
    let update = request(
        &a,
        &p,
        2,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), None)?,
            &limits(),
        )?)?,
    )
    .await?;
    let outcome = a.submit(&update, Duration::from_secs(10)).await?;
    let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
    let selection = annotation_publication::select_range(
        &c,
        frozen.sources().records.clone(),
        frozen.through(),
        Uuid::parse_str(frozen.identity())?,
    )
    .await?;
    let original = if stage > 0 {
        let prepared = frozen
            .prepare_mixed(&MergeMemoryBudget::new(8 << 20), 8192)
            .await?;
        let transaction = prepared.annotations().merge().transaction.clone();
        let repeated = frozen
            .prepare_mixed(&MergeMemoryBudget::new(8 << 20), 8192)
            .await?;
        assert_eq!(repeated.annotations().merge().transaction, transaction);
        // Reusing a frozen preparation must not write or commit its body again.
        // These observations come from real native Lance operations above.
        for (name, count) in [
            ("annotation_body_prepare", 1),
            ("annotation_body_write", 1),
            ("annotation_body_commit", 1),
            ("annotation_narrow_prepare", 2),
        ] {
            let observations = a.phase_observation();
            let phase = observations.iter().find(|p| p.phase == name).unwrap();
            assert_eq!(phase.started, count, "{name}");
            assert_eq!(phase.completed, count, "{name}");
            assert_eq!(phase.failed, 0, "{name}");
            assert_eq!(phase.cancelled, 0, "{name}");
        }
        let descriptors = crate::annotation_publication::mixed::persist_descriptors(
            prepared.annotations(),
            &selection,
            &narrow,
            &body,
            c.client.max_descriptor_bytes(),
        )
        .await?;
        let commit = annotation_publication::save(&c, &selection, descriptors.clone()).await?;
        if stage == 2 {
            let result = crate::annotation_publication::mixed::publish(
                Arc::new(annotation_publication::LostReply(Arc::new(
                    c.client.manifest_catalog(),
                ))),
                &annotation_publication::one_shard(),
                &narrow,
                &body,
                commit,
                &selection,
                &descriptors,
                prepared.annotations().base().clone(),
            )
            .await;
            assert!(result.is_err(), "selected-manifest reply must be lost");
        }
        Some(transaction)
    } else {
        None
    };
    let q = producer(&a, 2).await?;
    let suffix = request(
        &a,
        &q,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A"], None, Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    let suffix_outcome = a.submit(&suffix, Duration::from_secs(10)).await?;
    let reimport = request(
        &a,
        &q,
        2,
        encoded(create(
            &["A", "C", "C"],
            &["wrong", "t", "wrong"],
            &[9, 3, 8],
        )?)?,
    )
    .await?;
    let reimport_outcome = a.submit(&reimport, Duration::from_secs(10)).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    stop.send(()).unwrap();
    task.await??;
    // Two creates (including re-import), two patches and point reads above.
    // Observe can poll idle/control progress. The submitting service shares
    // all four uploads, still decodes/applies each exactly once, and point
    // reads do not repeat that work. The separate service above fetches WAL.
    let phases = a.phase_observation();
    for (name, count) in [
        ("annotation_fetch", 0),
        ("annotation_decode", 4),
        ("annotation_create", 2),
        ("annotation_patch", 2),
    ] {
        let phase = phases
            .iter()
            .find(|p| p.phase == name)
            .expect("missing apply phase");
        assert_eq!(phase.started, count, "{name}");
        assert_eq!(phase.completed, count, "{name}");
        assert_eq!(phase.failed, 0, "{name}");
        assert_eq!(phase.cancelled, 0, "{name}");
    }
    let observe = phases
        .iter()
        .find(|p| p.phase == "annotation_observe")
        .unwrap();
    assert!(observe.completed >= 4);
    assert_eq!(observe.failed, 0);
    assert_eq!(observe.started, observe.completed + observe.cancelled);
    drop((frozen, a));
    c.shutdown().await?;
    c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    assert_eq!(a.submit(&update, Duration::from_secs(30)).await?, outcome);
    assert_eq!(
        a.submit(&reimport, Duration::from_secs(30)).await?,
        reimport_outcome
    );
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "C", "t", 3, "pending", None).await?;
    annotation_runtime::drained(&c, &a, reimport_outcome.index).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert!(reimport_outcome.index > suffix_outcome.index);
    if let Some(original) = original {
        let first = annotation_publication::read(
            &c,
            &narrow,
            Some(selection.pending.plan.target_version()?),
        )
        .await?;
        assert_eq!(first.read_transaction().await?.as_ref(), Some(&original));
    }
    let selected = annotation_publication::read(&c, &narrow, None).await?;
    assert_eq!(selected.count_rows(None).await?, 3);
    let reference = BodySnapshot::read(&selected).await?.unwrap();
    let selected_body =
        mixed::load_body(&body, &reference, &binding().incarnation.to_string()).await?;
    assert_eq!(selected_body.count_rows(None).await?, 3);
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    assert_eq!(
        a.submit(&reimport, Duration::from_secs(10)).await?,
        reimport_outcome
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn mixed_single_low_rate_import_drains_without_a_following_record() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed(&c, &narrow, &body).await?;
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let p = producer(&a, 1).await?;
    let first = request(&a, &p, 1, encoded(create(&["A"], &["s"], &[1])?)?).await?;
    let result = a.submit(&first, Duration::from_secs(10)).await?;
    annotation_runtime::drained(&c, &a, result.index).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    check(&a, "A", "s", 1, "pending", None).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    Ok(())
}
