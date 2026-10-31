use super::*;
use lance::index::DatasetIndexExt;
use std::collections::BTreeSet;

async fn apply(
    runtime: &mut Runtime,
    service: &Service,
    producer: &Producer,
    sequence: u64,
    body: Bytes,
) -> Result<crate::annotations::Completed> {
    let prepared = request(service, producer, sequence, body).await?;
    let service = service.clone();
    let task =
        tokio::spawn(async move { service.submit(&prepared, Duration::from_secs(30)).await });
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
async fn ready_annotations_publish_after_one_maintenance_then_idle_indexes_catch_up() -> Result<()>
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
                turn_id_index: Some(crate::materializer::TurnIdIndex { min_rows: 1 }),
                status_index: Some(crate::materializer::StatusIndex::default()),
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
    let second = apply(
        &mut runtime,
        &service,
        &producer,
        2,
        wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;

    // Both indexes have real work. One maintenance turn is guaranteed even
    // with an already-ready source batch, so sustained writes cannot starve it.
    runtime.publish_once_for_test().await?;
    let maintained = annotation_publication::read(&c, &storage, None).await?;
    assert_eq!(maintained.load_indices().await?.len(), 1);
    assert_eq!(service.observation().await.0.materialized, first.index);

    // The remaining index must not take another turn ahead of this ready batch.
    runtime.publish_once_for_test().await?;
    assert_eq!(
        service.observation().await.0.materialized,
        second.index,
        "ready annotation batch delayed behind another maintenance turn"
    );
    assert_eq!(service.source_observation().records, 0);

    let (stop, task) = run_owner(runtime);
    let value = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(value.status.as_deref(), Some("reviewed"));
    assert_eq!(value.score, Some(0.9));
    // No further writes: both indexes must still catch up to the selected base.
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let base = annotation_publication::read(&c, &storage, None).await?;
            let indexes = base.load_indices().await?;
            let complete = ["lance_context_turn_id", "lance_context_annotation_status"]
                .into_iter()
                .all(|name| {
                    let covered = indexes
                        .iter()
                        .filter(|i| i.name == name)
                        .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
                        .collect::<BTreeSet<_>>();
                    base.fragments()
                        .iter()
                        .all(|f| covered.contains(&(f.id as u32)))
                });
            if complete
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
