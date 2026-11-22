use super::*;

#[tokio::test]
async fn owned_point_reads_wait_for_bounded_admission_and_capture_after_publication() -> Result<()>
{
    queued_point_reads(2).await
}

#[tokio::test]
async fn large_http_control_budget_keeps_waiters_bounded_and_reads_current() -> Result<()> {
    queued_point_reads(128).await
}

async fn queued_point_reads(control_requests: usize) -> Result<()> {
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
            control_requests,
            ..options()
        },
    )?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let server = Server::start(owner.router(), None).await?;
    let (ga, a) = frontend_with_options(
        &c,
        &limits,
        &server.uri(),
        http::Options {
            control_requests,
            ..options()
        },
    )
    .await?;
    let (gb, b) = frontend_with_options(
        &c,
        &limits,
        &server.uri(),
        http::Options {
            control_requests,
            ..options()
        },
    )
    .await?;
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
    let reads: Vec<_> = (0..control_requests)
        .map(|i| {
            tokio::spawn(
                post(&client, if i % 2 == 0 { &a } else { &b }, "read")
                    .json(&PointRequest { turn: "T".into() })
                    .send(),
            )
        })
        .collect();
    tokio::time::timeout(Duration::from_secs(5), async {
        while phase().started < started + control_requests as u64 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .with_context(|| {
        format!(
            "read admission: phase={:?}, owner={:?}, ingress_a={:?}, ingress_b={:?}, finished={}",
            phase(),
            serde_json::to_value(owner.observation()).unwrap(),
            serde_json::to_value(ga.observation()).unwrap(),
            serde_json::to_value(gb.observation()).unwrap(),
            reads.iter().filter(|read| read.is_finished()).count(),
        )
    })?;
    let observed = phase();
    let queued = observed.started - observed.completed - observed.cancelled;
    assert_eq!(owner.observation().control_inflight, control_requests);
    assert_eq!(service.observation().await.0.readers, 1);
    // The existing HTTP gate bounds waiters; there is no extra unbounded queue.
    let overflow = post(&client, &a, "read")
        .json(&PointRequest { turn: "T".into() })
        .send()
        .await?;
    let overflow_status = overflow.status();
    let overflow: serde_json::Value = overflow.json().await?;
    assert_eq!(owner.observation().local_control_request_refusals, 1);
    assert_eq!(owner.observation().local_data_request_refusals, 0);
    assert_eq!(ga.observation().local_control_request_refusals, 0);
    assert_eq!(gb.observation().local_control_request_refusals, 0);

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
    let mut values = Vec::new();
    for read in reads {
        let response = read.await??.error_for_status()?;
        values.push(response.json::<TurnAnnotation>().await?);
    }
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
    assert_eq!(queued, control_requests as u64);
    assert_eq!(overflow_status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(overflow["error"], "request_capacity");
    assert_eq!(held_value, initial);
    assert_eq!(values.len(), control_requests);
    for value in values {
        assert_eq!(value.status.as_deref(), Some("reviewed"));
        assert_eq!(value.score, Some(0.9));
    }
    assert_eq!(readers_after, 0);
    assert_eq!(owner.observation().control_inflight, 0);
    Ok(())
}
