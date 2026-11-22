use super::*;
use crate::tests::annotation_runtime::automatic_reclamation::deletion_faults::fault_store;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn background_compaction_keeps_draining_and_rebases_over_new_turns() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), exercise(false, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn background_compaction_discards_changed_fragments_and_collects_unused_output() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(90), exercise(true, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn background_compaction_owner_replacement_keeps_acknowledged_publications() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), exercise(false, true)).await?
}

async fn submit(
    runtime: &mut service::runtime::Runtime,
    service: &Service,
    input: PreparedAppend,
) -> Result<service::Completed> {
    let submitting = service.clone();
    let task =
        tokio::spawn(async move { submitting.submit(&input, Duration::from_secs(30)).await });
    while !task.is_finished() {
        runtime.apply_once_for_test().await?;
        tokio::task::yield_now().await;
    }
    task.await?
}

async fn exercise(change_source: bool, replace_owner: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let faults = Arc::new(fault_store::Faults::default());
    let narrow =
        fault_store::storage(&root.path().join("annotations.lance"), faults.clone()).await?;
    let body = storage(root.path(), "body.lance").await?;
    fragmented(&c, &narrow, &body).await?;
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    let mut runtime =
        runtime.with_exclusive_reclamation(service::runtime::reclamation::Options {
            max_objects: 2,
            max_bytes: 16 << 10,
            delete_parallelism: 2,
            interval: Duration::from_millis(5),
        })?;
    let original_ids = ids(&selected_body(&c, &narrow, &body).await?).await?;
    let held = a.capture_turn_for_test("A").await?;
    let (entered, resume) = faults.pause_data_write();
    while runtime.compaction_preparation_finished_for_test().is_none() {
        // Body maintenance may run first. Only narrow writes use this fault.
        runtime.publish_once_for_test().await?;
    }
    let path = tokio::time::timeout(Duration::from_secs(20), entered).await??;
    let output = std::path::Path::new("/").join(path.as_ref());
    assert!(output.exists(), "the original native output must exist");
    assert_eq!(
        runtime.compaction_preparation_finished_for_test(),
        Some(false)
    );
    assert!(annotation_publication::state(&c)
        .await?
        .progress
        .pending
        .is_none());

    let p = producer(&a, 2).await?;
    let mut last = None;
    // The source window holds only four records. Eight separately drained
    // publications prove that the native stall cannot merely be hidden there.
    for sequence in 1..=8 {
        let id = format!("new-{sequence}");
        let batch = if change_source && sequence == 1 {
            mixed::wire::patches(&patch(&["A"], Some("reviewed"), Some(0.9))?, &limits())?
        } else {
            create(&[&id], &["s"], &[sequence as u8 + 10])?
        };
        let input = request(&a, &p, sequence, encoded(batch)?).await?;
        let result = submit(&mut runtime, &a, input).await?;
        while a.observation().await.0.materialized < result.index {
            runtime.publish_once_for_test().await?;
            assert!(runtime.collect_once_for_test().await?);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(a.source_observation().records, 0);
        assert!(
            output.exists(),
            "GC deleted an unselected compaction output"
        );
        assert_eq!(
            runtime.compaction_preparation_finished_for_test(),
            Some(false)
        );
        last = Some(result);
    }
    let before = annotation_publication::read(&c, &narrow, None).await?;
    let checkpoint = before
        .read_transaction()
        .await?
        .unwrap()
        .transaction_properties;
    let body_ids = ids(&selected_body(&c, &narrow, &body).await?).await?;
    if replace_owner {
        // Drop the owning publisher while its native writer is still paused.
        // No durable maintenance plan exists; the successor recovers the
        // independently acknowledged data publications, never the old job.
        drop((held, runtime));
        let (next, runtime) = compact_owner(&c, &narrow, &body).await?;
        assert_eq!(next.observation().await.0.materialized, last.unwrap().index);
        resume.send(()).unwrap();
        while a.phase_observation().iter().any(|phase| {
            phase.phase == "native_compaction"
                && phase.started > phase.completed + phase.failed + phase.cancelled
        }) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            annotation_publication::read(&c, &narrow, None)
                .await?
                .version_id(),
            before.version_id(),
            "an abandoned native job published after owner replacement"
        );
        let runtime =
            runtime.with_exclusive_reclamation(service::runtime::reclamation::Options {
                max_objects: 2,
                max_bytes: 16 << 10,
                delete_parallelism: 2,
                interval: Duration::from_millis(5),
            })?;
        let (stop, task) = annotation_runtime::run_owner(runtime);
        while output.exists() {
            ensure!(
                !task.is_finished(),
                "successor stopped before collecting the abandoned output"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        for sequence in 1..=8 {
            check(
                &next,
                &format!("new-{sequence}"),
                "s",
                sequence + 10,
                "pending",
                None,
            )
            .await?;
        }
        assert_eq!(
            ids(&selected_body(&c, &narrow, &body).await?).await?,
            body_ids
        );
        stop.send(()).unwrap();
        task.await??;
        return c.shutdown().await;
    }
    resume.send(()).unwrap();
    while runtime.compaction_preparation_finished_for_test() == Some(false) {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    runtime.publish_once_for_test().await?;
    assert_eq!(runtime.compaction_preparation_finished_for_test(), None);
    let selected = annotation_publication::read(&c, &narrow, None).await?;
    if change_source {
        assert_eq!(
            selected.version_id(),
            before.version_id(),
            "stale rewrite was published"
        );
        // Once the invalid job is gone its fresh, unreferenced data file is
        // collectable. Captured old reads still protect their own source files.
        while output.exists() {
            assert!(runtime.collect_once_for_test().await?);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
        // An invalidated job must not leave maintenance permanently deferred.
        while annotation_publication::read(&c, &narrow, None)
            .await?
            .fragments()
            .len()
            > 1
        {
            runtime.publish_once_for_test().await?;
            assert_eq!(runtime.compaction_preparation_finished_for_test(), None);
        }
    } else {
        assert_eq!(selected.version_id(), before.version_id() + 1);
        let transaction = selected.read_transaction().await?.unwrap();
        assert_eq!(transaction.transaction_properties, checkpoint);
        let Operation::Rewrite { groups, .. } = transaction.operation else {
            anyhow::bail!("background compaction did not publish a rewrite");
        };
        assert!(groups.iter().flat_map(|g| &g.new_fragments).any(|f| {
            f.files
                .iter()
                .any(|file| output.file_name().unwrap() == file.path.as_str())
        }));
        for _ in 0..12 {
            assert!(runtime.collect_once_for_test().await?);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(output.exists(), "GC deleted the selected rewrite");
    }
    assert_eq!(a.observation().await.0.materialized, last.unwrap().index);
    assert_eq!(
        ids(&selected_body(&c, &narrow, &body).await?).await?,
        body_ids
    );
    for (id, stable) in &original_ids {
        assert_eq!(body_ids[id], *stable);
    }
    assert_eq!(
        held.annotations()
            .resolve()
            .await?
            .unwrap()
            .status
            .as_deref(),
        Some("pending")
    );
    let lance_context_core::annotation_staging::TurnBodySource::Materialized(old) =
        held.body_source()
    else {
        anyhow::bail!("captured body has no materialized source");
    };
    let reference = BodySnapshot::read(old).await?.unwrap();
    let cold_body = storage(root.path(), "body.lance").await?;
    let captured =
        mixed::load_body(&cold_body, &reference, &binding().incarnation.to_string()).await?;
    assert_eq!(ids(&captured).await?, original_ids);
    drop((a, runtime));
    c.shutdown().await
}
