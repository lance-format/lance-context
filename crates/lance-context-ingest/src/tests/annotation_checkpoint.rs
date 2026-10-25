use super::annotations::{connect, producer, request, run, wire};
use super::*;
use crate::{
    annotation_publication as publication,
    annotations::{FrozenCheckpoint, MutationResult, PreparedCheckpoint, Service},
    materializer::Storage,
};
use lance_context_core::annotation_staging::Limits;

pub(super) async fn prepare(
    c: &network::Cluster,
    frozen: &FrozenCheckpoint,
    after: u64,
) -> Result<(
    PreparedCheckpoint,
    publication::Selection,
    p::Descriptors,
    p::lance::ManifestCommit,
)> {
    let mut records = Vec::new();
    let mut cursor = after;
    while cursor < frozen.through() {
        let response = c
            .client
            .request_routed(Request::Query {
                group: 10,
                query: query::Query::AcceptedBucket {
                    table: binding().table,
                    bucket: 0,
                    after: cursor,
                    limit: 128,
                },
            })
            .await?;
        let Reply::Query(query::Outcome::Observed {
            value: query::Value::Accepted(page),
            ..
        }) = response.reply
        else {
            bail!("checkpoint source unavailable");
        };
        assert!(page.through > cursor);
        records.extend(
            page.entries
                .into_iter()
                .filter(|(i, _)| *i <= frozen.through()),
        );
        cursor = page.through;
    }
    assert_eq!(records, frozen.sources().records);
    assert_eq!(after, frozen.sources().after);
    let selection = annotation_publication::select_range(
        c,
        records,
        frozen.through(),
        Uuid::parse_str(frozen.identity())?,
    )
    .await?;
    let prepared = frozen
        .prepare(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let descriptors = publication::descriptors(prepared.annotations(), &selection)?;
    let commit = annotation_publication::save(c, &selection, descriptors.clone()).await?;
    Ok((prepared, selection, descriptors, commit))
}

#[tokio::test]
async fn annotation_source_record_limit_drains_repeated_updates_of_one_turn() -> Result<()> {
    bounded_prefix(false).await
}

#[tokio::test]
async fn annotation_source_byte_limit_drains_repeated_updates_of_one_turn() -> Result<()> {
    bounded_prefix(true).await
}

async fn bounded_prefix(byte_limit: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits::default();
    let body = wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?;
    let mut binding = binding();
    binding.max_record_bytes = body.len() as u32;
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding, &limits)?)
        .await?;
    let (service, driver) = Service::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding,
        &storage,
        limits,
        MergeMemoryBudget::new(8 << 20),
        MergeMemoryBudget::new(8 << 20),
    )
    .await?;
    let driver = driver.with_publication_limits(
        if byte_limit { p::MAX_INPUTS } else { 2 },
        if byte_limit {
            (body.len() * 2) as u64
        } else {
            c.client.max_plan_source_bytes()
        },
    )?;
    let (stop, task) = run(driver);
    let producer = producer(&service, 1).await?;
    let mut after = 0;
    for round in 0..3 {
        for offset in 1..=2 {
            let request = request(&service, &producer, round * 3 + offset, body.clone()).await?;
            assert_eq!(
                service
                    .submit(&request, Duration::from_secs(10))
                    .await?
                    .result,
                MutationResult::Applied
            );
        }
        assert_eq!(service.source_observation().records, 2);
        assert_eq!(service.source_observation().bytes, (body.len() * 2) as u64);
        assert_eq!(service.observation().await.0.active_entries, 1);
        let third = Arc::new(request(&service, &producer, round * 3 + 3, body.clone()).await?);
        let submitted = tokio::spawn({
            let service = service.clone();
            let third = third.clone();
            async move { service.submit(&third, Duration::from_secs(30)).await }
        });
        tokio::time::timeout(Duration::from_secs(10), async {
            while !service.observation().await.1 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?;
        assert!(
            !submitted.is_finished(),
            "source pressure must not acknowledge or reject the pending mutation"
        );
        assert_eq!(service.source_observation().records, 2);
        after = publish(&c, &service, &storage, after).await?;
        let completed = submitted.await??;
        assert_eq!(completed.result, MutationResult::Applied);
        assert_eq!(
            service.submit(&third, Duration::from_secs(10)).await?,
            completed
        );
        assert_eq!(
            service
                .get("T", Duration::from_secs(10))
                .await?
                .unwrap()
                .status
                .as_deref(),
            Some("reviewed")
        );
        after = publish(&c, &service, &storage, after).await?;
        assert_eq!(service.source_observation().records, 0);
        assert_eq!(service.source_observation().bytes, 0);
        let (observed, pressured) = service.observation().await;
        assert!(!pressured);
        assert_eq!(
            (
                observed.retained_entries,
                observed.retained_bytes,
                observed.retained_metadata_bytes
            ),
            (0, 0, 0)
        );
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

async fn publish(
    c: &network::Cluster,
    service: &Service,
    storage: &Storage,
    after: u64,
) -> Result<u64> {
    let frozen = service.freeze(Uuid::new_v4()).await?.unwrap();
    let (prepared, selection, descriptors, commit) = prepare(c, &frozen, after).await?;
    let selected = publication::publish(
        Arc::new(c.client.manifest_catalog()),
        &annotation_publication::one_shard(),
        storage,
        commit,
        &selection,
        &descriptors,
        prepared.annotations().base().clone(),
    )
    .await?;
    service
        .install_published(&prepared, Arc::new(selected))
        .await?;
    annotation_publication::finish(c).await?;
    Ok(frozen.through())
}

#[tokio::test]
async fn annotation_checkpoint_preserves_concurrent_suffix_and_results_after_source_gc(
) -> Result<()> {
    roundtrip(false).await
}

#[tokio::test]
async fn annotation_checkpoint_resolves_selected_publication_with_lost_reply() -> Result<()> {
    roundtrip(true).await
}

async fn roundtrip(lost_reply: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits::default();
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let (mut service, driver) = connect(&c, &storage, &limits).await?;
    let (mut stop, mut task) = run(driver);
    let a = producer(&service, 1).await?;
    let b = producer(&service, 2).await?;
    let suffix_producer = producer(&service, 3).await?;
    let rejected = request(
        &service,
        &a,
        1,
        wire(&limits, &["absent"], 2, None, 0, None)?,
    )
    .await?;
    let rejected_result = service.submit(&rejected, Duration::from_secs(5)).await?;
    assert!(matches!(
        rejected_result.result,
        MutationResult::Rejected(_)
    ));
    let accepted = request(
        &service,
        &b,
        1,
        wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let accepted_result = service.submit(&accepted, Duration::from_secs(5)).await?;
    let frozen = service.freeze(Uuid::new_v4()).await?.unwrap();
    let through = frozen.through();
    let (prepared, selection, descriptors, commit) = prepare(&c, &frozen, 0).await?;
    // This suffix is acknowledged while the captured files/results remain frozen.
    let suffix = request(
        &service,
        &suffix_producer,
        1,
        wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?,
    )
    .await?;
    let suffix_result = service.submit(&suffix, Duration::from_secs(5)).await?;
    let catalog: Arc<dyn p::lance::ManifestCatalog> = if lost_reply {
        Arc::new(annotation_publication::LostReply(Arc::new(
            c.client.manifest_catalog(),
        )))
    } else {
        Arc::new(c.client.manifest_catalog())
    };
    let selected = publication::publish(
        catalog,
        &annotation_publication::one_shard(),
        &storage,
        commit,
        &selection,
        &descriptors,
        prepared.annotations().base().clone(),
    )
    .await;
    if lost_reply {
        assert!(selected.is_err());
        stop.send(()).unwrap();
        task.await??;
        c.shutdown().await?;
        c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
        let (recovered, driver) = connect(&c, &storage, &limits).await?;
        service = recovered;
        (stop, task) = run(driver);
        // Selection before confirmation is resolved from the scoped catalog and
        // exact persisted descriptors, not a version number or blob listing.
        assert_eq!(
            service.submit(&rejected, Duration::from_secs(5)).await?,
            rejected_result
        );
    } else {
        service
            .install_published(&prepared, Arc::new(selected?))
            .await?;
    }
    drop(prepared);
    drop(frozen);
    annotation_publication::finish(&c).await?;
    let current = service.get("T", Duration::from_secs(5)).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));
    assert_eq!(
        service.submit(&rejected, Duration::from_secs(5)).await?,
        rejected_result
    );
    assert_eq!(
        service.submit(&accepted, Duration::from_secs(5)).await?,
        accepted_result
    );
    let through = publish(&c, &service, &storage, through).await?;
    assert_eq!(
        service.submit(&suffix, Duration::from_secs(5)).await?,
        suffix_result
    );
    let (observed, _) = service.observation().await;
    assert_eq!(
        (
            observed.active_entries,
            observed.frozen_entries,
            observed.retained_bytes,
            observed.retained_metadata_bytes
        ),
        (0, 0, 0, 0)
    );

    // An entirely rejected batch still publishes its result/cut, with no new
    // column files, so invalid requests cannot accumulate forever in the WAL.
    let rejected_only = request(
        &service,
        &a,
        2,
        wire(&limits, &["absent"], 2, None, 0, None)?,
    )
    .await?;
    let only_result = service
        .submit(&rejected_only, Duration::from_secs(5))
        .await?;
    assert!(matches!(only_result.result, MutationResult::Rejected(_)));
    let frozen = service.freeze(Uuid::new_v4()).await?.unwrap();
    let (prepared, selection, descriptors, commit) = prepare(&c, &frozen, through).await?;
    assert_eq!(prepared.annotations().merge().stats.bytes_written, 0);
    let before = prepared.annotations().base().manifest().fragments.clone();
    let selected = publication::publish(
        Arc::new(c.client.manifest_catalog()),
        &annotation_publication::one_shard(),
        &storage,
        commit,
        &selection,
        &descriptors,
        prepared.annotations().base().clone(),
    )
    .await?;
    assert_eq!(selected.manifest().fragments, before);
    service
        .install_published(&prepared, Arc::new(selected))
        .await?;
    annotation_publication::finish(&c).await?;
    let through = frozen.through();
    drop(prepared);
    drop(frozen);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (recovered, driver) = connect(&c, &storage, &limits).await?;
    service = recovered;
    let (stop, task) = run(driver);
    assert_eq!(
        service
            .submit(&rejected_only, Duration::from_secs(5))
            .await?,
        only_result
    );
    assert_eq!(
        service.submit(&accepted, Duration::from_secs(5)).await?,
        accepted_result
    );
    assert_eq!(
        service.submit(&suffix, Duration::from_secs(5)).await?,
        suffix_result
    );
    assert!(service
        .submit(&rejected, Duration::from_secs(5))
        .await
        .is_err());
    assert_eq!(
        service.get("T", Duration::from_secs(5)).await?.unwrap(),
        current
    );
    let fetched = c
        .client
        .request_routed(Request::Fetch {
            group: 10,
            index: accepted_result.index,
        })
        .await?;
    assert!(
        matches!(fetched.reply, Reply::Record(None)),
        "retired payload must not be needed to resolve an exact request"
    );

    // Reuse a closed and fully retired producer slot. Its old snapshot entry
    // must not turn a new generation's successful request into an old rejection.
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Producer(
            producer::Command {
                cluster: annotation_publication::one_shard().cluster,
                layout: annotation_publication::one_shard().fingerprint()?,
                group: 10,
                table: binding().table,
                slot: 1,
                action: producer::Action::Close {
                    generation: 1,
                    expected_last: Some(producer::Last {
                        sequence: 2,
                        index: only_result.index,
                    }),
                },
            },
        ))))
        .await?;
    assert!(matches!(
        response.reply,
        Reply::Commit(Outcome::Committed(Applied::Producer(
            producer::Reply::Closed { .. }
        )))
    ));
    let reused = service
        .open_producer(ProducerOpen {
            bucket: 0,
            slot: 1,
            expected_generation: 1,
            operation: Uuid::new_v4(),
        })
        .await?;
    let new_request = request(
        &service,
        &reused,
        1,
        wire(&limits, &["T"], 1, Some("new-generation"), 0, None)?,
    )
    .await?;
    let new_result = service.submit(&new_request, Duration::from_secs(5)).await?;
    assert_eq!(new_result.result, MutationResult::Applied);
    publish(&c, &service, &storage, through).await?;
    assert!(service
        .submit(&rejected_only, Duration::from_secs(5))
        .await
        .is_err());
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let (service, driver) = connect(&c, &storage, &limits).await?;
    let (stop, task) = run(driver);
    assert_eq!(
        service.submit(&new_request, Duration::from_secs(5)).await?,
        new_result
    );
    assert_eq!(
        service
            .get("T", Duration::from_secs(5))
            .await?
            .unwrap()
            .status
            .as_deref(),
        Some("new-generation")
    );
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
