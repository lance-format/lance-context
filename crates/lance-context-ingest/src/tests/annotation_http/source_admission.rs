//! A full apply window must stop new HTTP proposals while owned reads continue.
use super::*;
use crate::annotations::ApplyBackpressure;

#[tokio::test]
async fn source_pressure_refuses_before_wal_and_preserves_owned_reads() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
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
    annotation_publication::seed_contract(
        &c,
        &storage,
        annotations::Service::contract(&binding(), &limits)?,
    )
    .await?;
    let (service, mut runtime) = annotation_runtime::owner(&c, &storage, &limits).await?;
    let owner = Owner::new(&runtime, keys()?, options())?;
    let server = Server::start(owner.router(), None).await?;
    let (ga, a) = frontend(&c, &limits, &server.uri()).await?;
    let (gb, b) = frontend(&c, &limits, &server.uri()).await?;
    let client = reqwest::Client::builder()
        .http2_prior_knowledge()
        .timeout(Duration::from_secs(10))
        .build()?;
    let first = open(&client, &a, 1).await?;
    let producer = service
        .ingress()
        .restore_producer(&keys()?, &first.producer)?;

    // Apply two records without polling the publisher. The fixture has a real
    // two-record source limit, independent of how often the same turn changes.
    for (sequence, body) in [
        super::super::annotations::wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?,
        super::super::annotations::wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
    ]
    .into_iter()
    .enumerate()
    {
        let prepared =
            super::super::annotations::request(&service, &producer, sequence as u64 + 1, body)
                .await?;
        let accepted = service.ingress().append(&prepared).await?;
        while service.observation().await.0.applied < accepted.position.index {
            runtime.apply_once_for_test().await?;
            tokio::task::yield_now().await;
        }
        service.submit(&prepared, Duration::from_secs(10)).await?;
    }
    let third = super::super::annotations::request(
        &service,
        &producer,
        3,
        super::super::annotations::wire(&limits, &["T"], 1, Some("queued"), 0, None)?,
    )
    .await?;
    let queued = service.ingress().append(&third).await?;
    while service.apply_backpressure() != Some(ApplyBackpressure::SourceWindow) {
        runtime.apply_once_for_test().await?;
        tokio::task::yield_now().await;
    }
    let second = open(&client, &b, 2).await?;
    let second_producer = service
        .ingress()
        .restore_producer(&keys()?, &second.producer)?;
    let body = super::super::annotations::wire(&limits, &["T"], 0, None, 1, Some(0.5))?;
    let prepared =
        super::super::annotations::request(&service, &second_producer, 1, body.clone()).await?;
    let refusal = tokio::time::timeout(Duration::from_secs(1), async {
        let response = update(&client, &b, &second, 1, body.len())
            .body(body.clone())
            .send()
            .await?;
        let status = response.status();
        let value = response.json::<serde_json::Value>().await?;
        Ok::<_, anyhow::Error>((status, value))
    })
    .await;
    let (_, durable) = crate::reader::observe(
        service.ingress(),
        10,
        lance_context_wal::replication::query::Query::Producer {
            table: prepared.meta().stream,
            bucket: prepared.meta().bucket,
            producer: prepared.meta().producer,
        },
    )
    .await?;
    let visible = tokio::time::timeout(Duration::from_secs(1), async {
        tokio::try_join!(point(&client, &a), point(&client, &b))
    })
    .await;

    // Resume the same owner and submit the original refused identity once.
    // Cleanup precedes assertions so the old behavior does not strand tasks.
    let (stop, task) = annotation_runtime::run_owner(runtime);
    annotation_runtime::drained(&c, &service, queued.position.index).await?;
    let accepted: MutationReply = update(&client, &a, &second, 1, body.len())
        .body(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    annotation_runtime::drained(&c, &service, accepted.completion.index).await?;
    let current = point(&client, &b).await?.unwrap();
    ga.shutdown().await;
    gb.shutdown().await;
    a.shutdown().await?;
    b.shutdown().await?;
    owner.shutdown().await;
    server.shutdown().await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await?;

    let (status, rejection) = refusal??;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(rejection["error"], "annotation_apply_capacity");
    assert_eq!(rejection["operation_outcome"], "not_submitted");
    assert!(matches!(
        durable,
        lance_context_wal::replication::query::Value::Producer(None)
    ));
    let (left, right) = visible??;
    assert_eq!(left, right);
    let value = left.unwrap();
    assert_eq!(value.status.as_deref(), Some("reviewed"));
    assert_eq!(value.score, Some(0.9));
    assert_eq!(
        accepted.completion.result,
        annotations::MutationResult::Applied
    );
    assert_eq!(current.status.as_deref(), Some("queued"));
    assert_eq!(current.score, Some(0.5));
    Ok(())
}
