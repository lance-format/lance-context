use super::*;

#[tokio::test]
async fn prepared_id_index_yields_once_to_source_pressure_then_publishes() -> Result<()> {
    exercise(false).await
}

#[tokio::test]
async fn prepared_status_index_yields_once_without_restoring_old_annotations() -> Result<()> {
    exercise(true).await
}

async fn exercise(status_index: bool) -> Result<()> {
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
            publication_budget: MergeMemoryBudget::new(1024 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1,
                max_records: 1,
                max_batch_wait: Duration::from_secs(5),
                turn_id_index: (!status_index)
                    .then_some(crate::materializer::TurnIdIndex { min_rows: 1 }),
                status_index: status_index.then_some(crate::materializer::StatusIndex {
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

    // Finish real native preparation while the source window is empty. No
    // catalog plan is held, so the subsequent pressure can still drain first.
    runtime.publish_once_for_test().await?;
    tokio::time::timeout(Duration::from_secs(30), async {
        while runtime.index_preparation_finished_for_test() == Some(false) {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    assert_eq!(runtime.index_preparation_finished_for_test(), Some(true));
    assert!(annotation_publication::state(&c)
        .await?
        .progress
        .pending
        .is_none());

    apply(
        &mut runtime,
        &service,
        &producer,
        2,
        wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let third = blocked_update(
        &mut runtime,
        &service,
        &producer,
        3,
        wire(&limits, &["T"], 1, Some("accepted"), 0, None)?,
    )
    .await?;
    let cut = service.observation().await.0.applied;
    runtime.publish_once_for_test().await?;
    assert_eq!(
        service.observation().await.0.materialized,
        cut,
        "finished index preparation must yield once to a full source window"
    );
    assert_eq!(service.source_observation().records, 0);
    assert_eq!(runtime.index_preparation_finished_for_test(), Some(true));
    assert!(
        !third.is_finished(),
        "publication cannot ACK an unapplied update"
    );
    assert_eq!(
        finish_update(&mut runtime, third).await?.result,
        MutationResult::Applied
    );

    // Keep pressure present: the exact retained preparation now gets its turn,
    // rebased over the intervening data publication. It cannot starve forever.
    let fourth = blocked_update(
        &mut runtime,
        &service,
        &producer,
        4,
        wire(&limits, &["T"], 0, None, 1, Some(0.8))?,
    )
    .await?;
    runtime.publish_once_for_test().await?;
    assert_eq!(service.observation().await.0.materialized, cut);
    assert_eq!(service.source_observation().records, 1);
    assert_eq!(runtime.index_preparation_finished_for_test(), None);
    let indexed = annotation_publication::read(&c, &storage, None).await?;
    let name = if status_index {
        "lance_context_annotation_status"
    } else {
        "lance_context_turn_id"
    };
    assert!(indexed.load_indices().await?.iter().any(|i| i.name == name));
    assert!(!fourth.is_finished());
    runtime.publish_once_for_test().await?;
    assert_eq!(service.source_observation().records, 0);
    let last = finish_update(&mut runtime, fourth).await?;
    assert_eq!(last.result, MutationResult::Applied);

    let (stop, task) = run_owner(runtime);
    drained(&c, &service, last.index).await?;
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("accepted"));
    assert_eq!(current.score, Some(0.8));
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
