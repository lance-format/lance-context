use super::*;

#[tokio::test]
async fn owned_point_reads_wait_for_bounded_admission_and_capture_after_publication() -> Result<()>
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
    let owner = Owner::new(
        &runtime,
        keys()?,
        http::Options {
            control_requests: 2,
            ..options()
        },
    )?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let server = Server::start(owner.router(), None).await?;
    let (ga, a) = frontend(&c, &limits, &server.uri()).await?;
    let (gb, b) = frontend(&c, &limits, &server.uri()).await?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let producer = open(&client, &a, 1).await?;
    let initial = point(&client, &a).await?.unwrap();
    let held = service.capture_read_for_test("T").await?;

    // Poll explicitly: merely spawning an unscheduled read cannot prove that
    // it waits. Cancellation and the existing deadline must release waiters.
    let waited = {
        let pending = service.get_owned("T", recovery_cut, Duration::from_secs(5));
        tokio::pin!(pending);
        futures::poll!(&mut pending).is_pending()
    };
    let expired = service
        .get_owned("T", recovery_cut, Duration::from_millis(20))
        .await
        .unwrap_err();
    let readers_after_cancel = service.observation().await.0.readers;

    let phase = || {
        service
            .phase_observation()
            .into_iter()
            .find(|p| p.phase == "annotation_point_read")
            .unwrap()
    };
    let started = phase().started;
    let left = tokio::spawn(
        post(&client, &a, "read")
            .json(&PointRequest { turn: "T".into() })
            .send(),
    );
    let right = tokio::spawn(
        post(&client, &b, "read")
            .json(&PointRequest { turn: "T".into() })
            .send(),
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while phase().started < started + 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let observed = phase();
    let queued = observed.started - observed.completed - observed.cancelled;
    // The existing HTTP gate bounds waiters; there is no extra unbounded queue.
    let overflow = post(&client, &server, "read")
        .json(&PointRequest { turn: "T".into() })
        .send()
        .await?;
    let overflow_status = overflow.status();
    let overflow: serde_json::Value = overflow.json().await?;

    // Waiting reads must not hold the state lock or a stale base. Publish an
    // update while they wait, then require both ingress reads to see it.
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
    let left = left.await??;
    let right = right.await??;
    let left_status = left.status();
    let right_status = right.status();
    let left: serde_json::Value = left.json().await?;
    let right: serde_json::Value = right.json().await?;
    let readers_after = service.observation().await.0.readers;

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
        waited,
        "admitted point read failed instead of waiting for a reader"
    );
    assert!(format!("{expired:#}").contains("annotation owned read deadline expired"));
    assert_eq!(readers_after_cancel, 1);
    assert_eq!(queued, 2);
    assert_eq!(overflow_status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(overflow["error"], "request_capacity");
    assert_eq!(held_value, initial);
    assert_eq!(left_status, StatusCode::OK);
    assert_eq!(right_status, StatusCode::OK);
    for value in [left, right] {
        let value: TurnAnnotation = serde_json::from_value(value)?;
        assert_eq!(value.status.as_deref(), Some("reviewed"));
        assert_eq!(value.score, Some(0.9));
    }
    assert_eq!(readers_after, 0);
    Ok(())
}
