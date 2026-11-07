use super::*;

#[tokio::test]
async fn ready_updates_defer_small_status_build_then_idle_tail_catches_up() -> Result<()> {
    scenario(3, false, false).await
}

#[tokio::test]
async fn status_tail_at_threshold_advances_despite_ready_updates() -> Result<()> {
    scenario(2, true, false).await
}

#[tokio::test]
async fn ready_id_work_does_not_admit_a_small_preferred_status_build() -> Result<()> {
    scenario(3, false, true).await
}

async fn scenario(min_rows: usize, expect_maintenance: bool, competing_id: bool) -> Result<()> {
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
    let configuration = |indexing: bool| Configuration {
        limits: limits.clone(),
        upload_budget: MergeMemoryBudget::new(8 << 20),
        apply_budget: MergeMemoryBudget::new(8 << 20),
        publication_budget: MergeMemoryBudget::new(512 << 20),
        publication: Options {
            max_source_bytes: 1 << 20,
            min_source_bytes: 1,
            max_records: 1,
            max_batch_wait: Duration::from_secs(5),
            turn_id_index: (indexing && competing_id)
                .then_some(crate::materializer::TurnIdIndex { min_rows: 1 }),
            status_index: indexing.then_some(crate::materializer::StatusIndex {
                min_rows,
                ..Default::default()
            }),
            ..Default::default()
        },
    };
    if competing_id {
        // Start at a durable version that prefers status work, with neither
        // index installed. Each warm-up is a real RF3 update and publication.
        let (service, mut runtime) = connect_owner(&c, &storage, configuration(false)).await?;
        let producer = producer(&service, 2).await?;
        for sequence in 1..=3 {
            let completed = apply(
                &mut runtime,
                &service,
                &producer,
                sequence,
                wire(&limits, &["T"], 0, None, 1, Some(sequence as f32))?,
            )
            .await?;
            runtime.publish_once_for_test().await?;
            assert_eq!(service.observation().await.0.materialized, completed.index);
        }
        let base = annotation_publication::read(&c, &storage, None).await?;
        assert_eq!(base.version_id(), 4);
        assert!(base.load_indices().await?.is_empty());
    }
    let (service, mut runtime) = connect_owner(&c, &storage, configuration(true)).await?;
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
    runtime.publish_once_for_test().await?;
    let base = annotation_publication::read(&c, &storage, None).await?;
    assert_eq!(
        base.load_indices().await?.len(),
        usize::from(expect_maintenance)
    );
    assert_eq!(
        service.observation().await.0.materialized,
        if expect_maintenance {
            first.index
        } else {
            second.index
        }
    );
    if expect_maintenance {
        runtime.publish_once_for_test().await?;
    }
    assert_eq!(service.observation().await.0.materialized, second.index);
    assert_eq!(service.source_observation().records, 0);
    // Point reads also need the applier to advance their quorum read barrier.
    let (stop, task) = run_owner(runtime);
    let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));

    // The earlier deferral must not suppress maintenance on an idle version.
    // A real native index covers both original turns, including the updated T.
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let base = annotation_publication::read(&c, &storage, None).await?;
            let indices = base.load_indices().await?;
            let complete = ["lance_context_annotation_status", "lance_context_turn_id"]
                .into_iter()
                .filter(|name| competing_id || *name == "lance_context_annotation_status")
                .all(|name| {
                    let covered: BTreeSet<_> = indices
                        .iter()
                        .filter(|i| i.name == name)
                        .flat_map(|i| i.fragment_bitmap.as_ref().unwrap().iter())
                        .collect();
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

async fn connect_owner(
    c: &network::Cluster,
    storage: &Storage,
    configuration: Configuration,
) -> Result<(Service, Runtime)> {
    let state = annotation_publication::state(c).await?;
    Runtime::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        configuration,
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
    )
    .await
}
