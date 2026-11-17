use super::*;
use lance_context_core::annotation_staging::StatusMatches;

#[tokio::test]
async fn status_filter_waits_for_reader_and_captures_after_concurrent_publication() -> Result<()> {
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
        readers: 1,
    };
    annotation_publication::seed_contract(
        &c,
        &storage,
        annotations::Service::contract(&binding(), &limits)?,
    )
    .await?;
    let (service, runtime) = annotation_runtime::owner(&c, &storage, &limits).await?;
    let (_, _, recovery_cut) = runtime.http_binding();
    let owner = Owner::new(&runtime, keys()?, options())?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let server = Server::start(owner.router(), None).await?;
    let (ga, a) = frontend(&c, &limits, &server.uri()).await?;
    let (gb, b) = frontend(&c, &limits, &server.uri()).await?;
    let client = reqwest::Client::builder()
        .retry(reqwest::retry::never())
        .timeout(Duration::from_secs(30))
        .build()?;
    let producer = open(&client, &a, 1).await?;
    let initial = point(&client, &a).await?.unwrap();
    let held = service.capture_read_for_test("T").await?;
    let query = |deadline| {
        service.filter_status_owned(
            Some("reviewed".into()),
            16,
            64 << 10,
            recovery_cut,
            deadline,
        )
    };

    // Poll the serving-layer query itself, proving it reached reader admission.
    // Cancellation and timeout must remove waiters without acquiring workspace.
    let cancelled_waited = {
        let cancelled = query(Duration::from_secs(5));
        tokio::pin!(cancelled);
        futures::poll!(&mut cancelled).is_pending()
    };
    let expired = query(Duration::from_millis(20)).await;
    let readers_after_cancel = service.observation().await.0.readers;
    let workspace_after_cancel = service.ingress().budget.reserved();
    let pending = query(Duration::from_secs(30));
    tokio::pin!(pending);
    let (waited, ready) = match futures::poll!(&mut pending) {
        std::task::Poll::Pending => (true, None),
        std::task::Poll::Ready(result) => (false, Some(result)),
    };

    // Publication must proceed while the query waits. Its eventual snapshot
    // must contain this update rather than the base from before admission.
    let body = super::super::annotations::wire(&limits, &["T"], 1, Some("reviewed"), 1, Some(0.9))?;
    let applied: MutationReply = update(&client, &b, &producer, 1, body.len())
        .body(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    annotation_runtime::drained(&c, &service, applied.completion.index).await?;
    let held_value = held.resolve().await?.unwrap();
    drop(held);
    let result = match ready {
        Some(result) => result,
        None => pending.await,
    };
    let readers_after = service.observation().await.0.readers;

    let mut forwarded = Vec::new();
    for ingress in [&a, &b] {
        forwarded.push(
            post(&client, ingress, "filter-status")
                .json(&http::annotations::StatusRequest {
                    bucket: 0,
                    status: Some("reviewed".into()),
                    limit: 16,
                })
                .send()
                .await?
                .error_for_status()?
                .json::<StatusMatches>()
                .await?,
        );
    }
    stop.send(()).unwrap();
    task.await??;
    ga.shutdown().await;
    gb.shutdown().await;
    a.shutdown().await?;
    b.shutdown().await?;
    owner.shutdown().await;
    server.shutdown().await?;
    c.shutdown().await?;

    assert!(
        cancelled_waited && waited,
        "status filter did not wait for a reader"
    );
    let expired = expired.err().context("status query did not expire")?;
    assert!(format!("{expired:#}").contains("annotation status query deadline expired"));
    assert_eq!(readers_after_cancel, 1);
    assert_eq!(workspace_after_cancel, 0);
    assert_eq!(held_value, initial);
    let (result, reservation) = result?.context("status query refused free workspace")?;
    drop(reservation);
    for value in forwarded.into_iter().chain([result]) {
        assert_eq!(value.ids, ["T"]);
        assert!(value.materialized_through >= applied.completion.index);
    }
    assert_eq!(readers_after, 0);
    assert_eq!(service.ingress().budget.reserved(), 0);
    Ok(())
}
