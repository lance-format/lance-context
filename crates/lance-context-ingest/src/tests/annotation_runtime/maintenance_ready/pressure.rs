use super::*;
use crate::annotations::{ApplyBackpressure, Completed};

async fn blocked_update(
    runtime: &mut Runtime,
    service: &Service,
    producer: &Producer,
    sequence: u64,
    body: Bytes,
) -> Result<tokio::task::JoinHandle<Result<Completed>>> {
    let prepared = request(service, producer, sequence, body).await?;
    let submitting = service.clone();
    let task =
        tokio::spawn(async move { submitting.submit(&prepared, Duration::from_secs(30)).await });
    tokio::time::timeout(Duration::from_secs(30), async {
        while service.apply_backpressure() != Some(ApplyBackpressure::SourceWindow) {
            ensure!(
                !task.is_finished(),
                "update did not wait for source capacity"
            );
            runtime.apply_once_for_test().await?;
            tokio::task::yield_now().await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    assert!(
        !task.is_finished(),
        "unapplied update must not return success"
    );
    Ok(task)
}

async fn finish_update(
    runtime: &mut Runtime,
    task: tokio::task::JoinHandle<Result<Completed>>,
) -> Result<Completed> {
    tokio::time::timeout(Duration::from_secs(30), async {
        while !task.is_finished() {
            runtime.apply_once_for_test().await?;
            tokio::task::yield_now().await;
        }
        task.await?
    })
    .await?
}

#[tokio::test]
async fn pressured_publication_defers_one_maintenance_turn_without_starving_indexes() -> Result<()>
{
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let state = annotation_publication::state(&c).await?;
    let (service, mut runtime) = Runtime::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        Configuration {
            limits: limits.clone(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(512 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1,
                max_records: 1,
                max_batch_wait: Duration::from_secs(5),
                status_index: Some(crate::materializer::StatusIndex {
                    min_rows: 1,
                    ..Default::default()
                }),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
    )
    .await?;
    let producer = producer(&service, 1).await?;
    let first = apply(
        &mut runtime,
        &service,
        &producer,
        1,
        wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?,
    )
    .await?;
    runtime.publish_once_for_test().await?;
    assert_eq!(service.observation().await.0.materialized, first.index);
    let mut current = apply(
        &mut runtime,
        &service,
        &producer,
        2,
        wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let mut materialized = first.index;

    // Keep an applied batch and its next committed update waiting on a bounded
    // source window. Use actual native index work, not an artificial timer.
    for round in 0..4 {
        let sequence = round + 3;
        let status = format!("update-{sequence}");
        let pending = blocked_update(
            &mut runtime,
            &service,
            &producer,
            sequence,
            wire(&limits, &["T"], 1, Some(&status), 0, None)?,
        )
        .await?;
        assert_eq!(service.source_observation().records, 1);
        let before = annotation_publication::read(&c, &storage, None).await?;
        runtime.publish_once_for_test().await?;
        if round % 2 == 1 {
            // Only one maintenance turn may be deferred. Continued pressure
            // must still publish a real index before another data batch.
            assert_eq!(service.observation().await.0.materialized, materialized);
            let indexed = annotation_publication::read(&c, &storage, None).await?;
            assert!(indexed.version_id() > before.version_id());
            assert!(!indexed.load_indices().await?.is_empty());
            runtime.publish_once_for_test().await?;
        }
        assert_eq!(
            service.observation().await.0.materialized,
            current.index,
            "source pressure must get one publication before new maintenance"
        );
        materialized = current.index;
        current = finish_update(&mut runtime, pending).await?;
        assert_eq!(current.result, MutationResult::Applied);
    }

    let (stop, task) = run_owner(runtime);
    drained(&c, &service, current.index).await?;
    let value = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(value.status.as_deref(), Some("update-6"));
    assert_eq!(value.score, Some(0.9));
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let base = annotation_publication::read(&c, &storage, None).await?;
            let covered: BTreeSet<_> = base
                .load_indices()
                .await?
                .iter()
                .filter(|i| i.name == "lance_context_annotation_status")
                .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
                .collect();
            if base
                .fragments()
                .iter()
                .all(|f| covered.contains(&(f.id as u32)))
                && annotation_publication::state(&c)
                    .await?
                    .progress
                    .pending
                    .is_none()
            {
                break Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await??;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
