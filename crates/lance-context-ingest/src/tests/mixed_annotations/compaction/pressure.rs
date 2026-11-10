//! Bounded, queryable staging while native compaction is stalled.
use super::*;
use crate::annotations::ApplyBackpressure;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bounded_mixed_updates_during_native_maintenance() -> Result<()> {
    for body_pause in [true, false] {
        for restart in [false, true] {
            tokio::time::timeout(Duration::from_secs(90), exercise(body_pause, restart)).await??;
        }
    }
    Ok(())
}

async fn exercise(body_pause: bool, restart: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(fault_store::Faults::default());
    let narrow = if body_pause {
        storage(root.path(), "annotations.lance").await?
    } else {
        fault_store::storage(&root.path().join("annotations.lance"), faults.clone()).await?
    };
    let body = if body_pause {
        fault_store::storage(&root.path().join("body.lance"), faults.clone()).await?
    } else {
        storage(root.path(), "body.lance").await?
    };
    fragmented(&c, &narrow, &body).await?;
    let before_body_ids = ids(&selected_body(&c, &narrow, &body).await?).await?;
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    let (_, _, recovery_cut) = runtime.http_binding();
    let p = producer(&a, 2).await?;
    let (entered, resume) = faults.pause_data_write();
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let paused_path = tokio::time::timeout(Duration::from_secs(20), entered).await??;
    assert!(std::path::Path::new("/")
        .join(paused_path.as_ref())
        .exists());
    let before = annotation_publication::state(&c).await?;
    let pending = before.progress.pending.as_ref().unwrap();
    assert!(pending.plan.is_maintenance());
    assert!(before.progress.prepared.is_none());
    let cut = before.progress.checkpoint[&10];

    // Four actual mixed WAL frames fit the original fixture's source bound.
    // Each successful return must be visible while maintenance still owns its
    // earlier empty-source plan, including a new body and independent fields.
    let operations = [
        mixed::wire::patches(&patch(&["A"], Some("reviewed"), None)?, &limits())?,
        create(&["D"], &["s"], &[4])?,
        mixed::wire::patches(&patch(&["A"], None, Some(0.9))?, &limits())?,
        mixed::wire::patches(&patch(&["D"], None, Some(0.8))?, &limits())?,
    ];
    let mut acknowledged = Vec::new();
    for (i, batch) in operations.into_iter().enumerate() {
        let prepared = request(&a, &p, i as u64 + 1, encoded(batch)?).await?;
        let result = a.submit(&prepared, Duration::from_secs(10)).await?;
        acknowledged.push((prepared, result));
    }
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "D", "s", 4, "pending", Some(0.8)).await?;
    assert_eq!(a.source_observation().records, 4);
    assert_eq!(a.observation().await.0.materialized, cut);

    // A fifth frame is durable but cannot apply until a source generation
    // retires. Submit once and keep its original future; no business retry.
    let fifth = Arc::new(
        request(
            &a,
            &p,
            5,
            encoded(mixed::wire::patches(
                &patch(&["A"], Some("accepted"), None)?,
                &limits(),
            )?)?,
        )
        .await?,
    );
    let waiting_service = a.clone();
    let waiting_request = fifth.clone();
    let waiting = tokio::spawn(async move {
        waiting_service
            .submit(&waiting_request, Duration::from_secs(60))
            .await
    });
    tokio::time::timeout(Duration::from_secs(10), async {
        while a.apply_backpressure() != Some(ApplyBackpressure::SourceWindow) {
            ensure!(!waiting.is_finished(), "unapplied fifth update returned");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    let observed = a.observation().await.0;
    // Applied progress may also cover non-mutating consensus entries. The
    // source bound, outstanding fifth ACK and current-value checks below
    // establish that the fifth mutation has not applied.
    assert!(observed.applied >= acknowledged.last().unwrap().1.index);
    assert!(observed.retained_bytes <= limits().retained_bytes);
    assert!(a.retained_body_frames().1 <= 8 << 20);
    for _ in 0..16 {
        let current = a
            .get_owned("A", recovery_cut, Duration::from_secs(2))
            .await?
            .unwrap();
        assert_eq!(current.status.as_deref(), Some("reviewed"));
        assert_eq!(current.score, Some(0.9));
        let new = a
            .get_owned("D", recovery_cut, Duration::from_secs(2))
            .await?
            .unwrap();
        assert_eq!(new.status.as_deref(), Some("pending"));
        assert_eq!(new.score, Some(0.8));
    }
    assert!(!waiting.is_finished());
    let held = annotation_publication::state(&c).await?;
    assert_eq!(held.progress.pending.as_ref().unwrap().index, pending.index);
    assert_eq!(held.progress.checkpoint[&10], cut);
    assert_eq!(held.manifests, before.manifests);

    let (a, stop, task, completed) = if restart {
        // Stop the owner before it has prepared or selected any maintenance
        // metadata. Its successor must resolve the retained original plan.
        stop.send(()).unwrap();
        task.await??;
        // Stopping the applier wakes its unapplied waiter with an error. It
        // may already have returned before a task cancellation is requested;
        // require the actual failed result rather than a scheduling race.
        let stopped = tokio::time::timeout(Duration::from_secs(5), waiting).await??;
        assert_eq!(
            stopped.unwrap_err().to_string(),
            "annotation applier stopped"
        );
        drop(a);
        drop(resume);
        let (replacement, runtime) = compact_owner(&c, &narrow, &body).await?;
        let (stop, task) = annotation_runtime::run_owner(runtime);
        // Resolve the original uncertain request using its exact identity.
        let completed = replacement.submit(&fifth, Duration::from_secs(20)).await?;
        (replacement, stop, task, completed)
    } else {
        resume.send(()).unwrap();
        let completed = waiting.await??;
        (a, stop, task, completed)
    };
    annotation_runtime::drained(&c, &a, completed.index).await?;
    check(&a, "A", "s", 1, "accepted", Some(0.9)).await?;
    check(&a, "B", "s", 2, "pending", None).await?;
    check(&a, "C", "s", 3, "pending", None).await?;
    check(&a, "D", "s", 4, "pending", Some(0.8)).await?;
    // Only the latest result survives subsequent requests and recovery.
    // Rejecting older identities must leave the current value unchanged.
    for (request, _) in acknowledged {
        let error = a
            .submit(&request, Duration::from_secs(10))
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<crate::AppendFailure>(),
            Some(&crate::AppendFailure::Sequence { expected: 6 })
        );
    }
    assert_eq!(a.submit(&fifth, Duration::from_secs(10)).await?, completed);
    check(&a, "A", "s", 1, "accepted", Some(0.9)).await?;
    check(&a, "D", "s", 4, "pending", Some(0.8)).await?;
    let selected = annotation_publication::read(&c, &narrow, None).await?;
    assert_eq!(selected.count_rows(None).await?, 4);
    let selected_body = selected_body(&c, &narrow, &body).await?;
    assert_eq!(selected_body.count_rows(None).await?, 4);
    let after_body_ids = ids(&selected_body).await?;
    for (key, original) in before_body_ids {
        assert_eq!(after_body_ids[&key], original);
    }
    assert_eq!(a.retained_body_frames(), (0, 0));
    assert_eq!(a.source_observation().records, 0);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    Ok(())
}
