//! Patch lookahead shares reads, never application or request outcomes.
use super::*;

fn update(ids: &[&str], status: Option<&str>, score: Option<f32>) -> Result<RecordBatch> {
    mixed::wire::patches(&patch(ids, status, score)?, &limits())
}

async fn populated(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    budget: Arc<MergeMemoryBudget>,
    ids: &[&str],
) -> Result<()> {
    seed(c, narrow, body).await?;
    let (a, driver) = with_budget(c, narrow, body, budget.clone()).await?;
    let (stop, task) = run(driver);
    // Bootstrap owns slot 0; the lookahead backlog uses slots 1 through 10.
    let producer = producer(&a, 15).await?;
    for (offset, ids) in ids.chunks(binding().max_rows as usize).enumerate() {
        let input = request(
            &a,
            &producer,
            offset as u64 + 1,
            encoded(create(ids, &vec!["s"; ids.len()], &vec![7; ids.len()])?)?,
        )
        .await?;
        a.submit(&input, Duration::from_secs(5)).await?;
    }
    pressure::publish(c, &a, narrow, body).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(a);
    assert_eq!(budget.reserved(), 0);
    // Reconnect before patching so the published-value cache cannot satisfy
    // these lookups; all initial values must come from the selected Lance base.
    Ok(())
}

#[tokio::test]
async fn patch_lookup_preserves_ack_freeze_and_interleaved_create_order() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    let budget = MergeMemoryBudget::new(64 << 20);
    populated(&c, &narrow, &body, budget.clone(), &["A", "B", "C"]).await?;
    let (a, mut driver) = with_budget(&c, &narrow, &body, budget.clone()).await?;
    let inputs = Arc::new(
        backlog(
            &c,
            &a,
            vec![
                update(&["A"], Some("reviewed"), None)?,
                update(&["A"], None, Some(0.9))?,
                update(&["missing"], Some("bad"), None)?,
                create(&["D"], &["s"], &[4])?,
                update(&["D"], Some("created-then-updated"), Some(0.7))?,
                update(&["B"], Some("later-base"), Some(0.5))?,
                create(&["E"], &["s"], &[8])?,
            ],
        )
        .await?,
    );
    let ack = tokio::spawn({
        let a = a.clone();
        let inputs = inputs.clone();
        async move { a.submit(&inputs[1].0, Duration::from_secs(30)).await }
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while !a
            .phase_observation()
            .iter()
            .any(|p| p.phase == "annotation_apply_wait" && p.started > 0)
        {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    driver.step_for_test().await?;
    assert_eq!(completed(&a, "annotation_decode"), 7);
    assert_eq!(completed(&a, "annotation_patch_lookup"), 1);
    assert_eq!(completed(&a, "annotation_registration_lookup"), 1);
    assert_eq!(a.observation().await.0.applied, inputs[0].1);
    assert_eq!(a.source_observation().records, 1);
    assert!(
        !ack.is_finished(),
        "a prefetched patch must not be acknowledged"
    );
    let frozen = a.freeze(Uuid::new_v4()).await?.unwrap();
    assert_eq!(frozen.through(), inputs[0].1);
    assert_eq!(frozen.sources().records.len(), 1);
    driver.step_for_test().await?;
    assert_eq!(ack.await??.index, inputs[1].1);
    pressure::publish_frozen(&c, &a, &narrow, &body, frozen).await?;
    assert_eq!(a.observation().await.0.materialized, inputs[0].1);
    // A changed base invalidates the lookup. The later A update remains active,
    // D is registered before its patch, and B must use the compatible new base.
    for (_, index) in inputs.iter().skip(2) {
        driver.step_for_test().await?;
        assert_eq!(a.observation().await.0.applied, *index);
    }
    let (stop, task) = run(driver);
    check(&a, "A", "s", 7, "reviewed", Some(0.9)).await?;
    check(&a, "B", "s", 7, "later-base", Some(0.5)).await?;
    check(&a, "C", "s", 7, "pending", None).await?;
    check(&a, "D", "s", 4, "created-then-updated", Some(0.7)).await?;
    check(&a, "E", "s", 8, "pending", None).await?;
    for (offset, (input, index)) in inputs.iter().enumerate() {
        let result = a.submit(input, Duration::from_secs(5)).await?;
        assert_eq!(result.index, *index);
        if offset == 2 {
            assert!(
                matches!(result.result, service::MutationResult::Rejected(ref r)
                if r.reason == lance_context_core::annotation_staging::RejectionReason::MissingTurn)
            );
        } else {
            assert_eq!(result.result, service::MutationResult::Applied);
        }
    }
    pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert_eq!(budget.reserved(), 0);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[tokio::test]
async fn patch_lookup_suffix_replays_after_owner_and_wal_restart() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    let budget = MergeMemoryBudget::new(64 << 20);
    let ids = (0..10).map(|i| format!("turn-{i}")).collect::<Vec<_>>();
    let turns = ids.iter().map(String::as_str).collect::<Vec<_>>();
    populated(&c, &narrow, &body, budget.clone(), &turns).await?;
    let (a, mut driver) = with_budget(&c, &narrow, &body, budget.clone()).await?;
    let batches = ids
        .iter()
        .enumerate()
        .map(|(i, id)| update(&[id], Some("updated"), Some(i as f32)))
        .collect::<Result<Vec<_>>>()?;
    let inputs = backlog(&c, &a, batches).await?;
    driver.step_for_test().await?;
    assert_eq!(completed(&a, "annotation_decode"), 8);
    assert_eq!(completed(&a, "annotation_patch_lookup"), 1);
    assert_eq!(a.observation().await.0.applied, inputs[0].1);
    pressure::publish(&c, &a, &narrow, &body).await?;
    assert!(budget.reserved() > 0);
    drop(driver);
    drop(a);
    assert_eq!(budget.reserved(), 0);
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (a, driver) = with_budget(&c, &narrow, &body, budget.clone()).await?;
    assert_eq!(a.observation().await.0.materialized, inputs[0].1);
    let (stop, task) = run(driver);
    for (i, id) in ids.iter().enumerate() {
        check(&a, id, "s", 7, "updated", Some(i as f32)).await?;
        let result = a.submit(&inputs[i].0, Duration::from_secs(5)).await?;
        assert_eq!(result.index, inputs[i].1);
        assert_eq!(result.result, service::MutationResult::Applied);
    }
    pressure::publish(&c, &a, &narrow, &body).await?;
    assert_eq!(budget.reserved(), 0);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

#[tokio::test]
async fn patch_lookup_does_not_wait_for_lookahead_or_lookup_workspace() -> Result<()> {
    for available_mib in [1, 2] {
        let root = artifact_root()?;
        let c =
            network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
        let narrow = storage(root.path(), "annotations.lance").await?;
        let body = storage(root.path(), "body.lance").await?;
        let budget = MergeMemoryBudget::new(4 << 20);
        populated(&c, &narrow, &body, budget.clone(), &["A", "B"]).await?;
        let held = budget.reserve((4 - available_mib) << 20).await;
        let (a, mut driver) = with_budget(&c, &narrow, &body, budget.clone()).await?;
        let inputs = backlog(
            &c,
            &a,
            vec![
                update(&["A"], Some("first"), Some(0.1))?,
                update(&["B"], Some("second"), Some(0.2))?,
            ],
        )
        .await?;
        // One permit fits only the head; two fit both decoded requests but
        // leave no workspace for the optional lookup. Both must still apply.
        for (_, index) in &inputs {
            tokio::time::timeout(Duration::from_secs(5), driver.step_for_test()).await??;
            assert_eq!(a.observation().await.0.applied, *index);
            assert!(budget.reserved() <= budget.limit());
        }
        let (stop, task) = run(driver);
        check(&a, "A", "s", 7, "first", Some(0.1)).await?;
        check(&a, "B", "s", 7, "second", Some(0.2)).await?;
        pressure::publish(&c, &a, &narrow, &body).await?;
        assert_eq!(budget.reserved(), held.bytes());
        stop.send(()).unwrap();
        task.await??;
        drop(held);
        assert_eq!(budget.reserved(), 0);
        c.shutdown().await?;
    }
    Ok(())
}
