use super::*;
use crate::tests::{annotation_shards::Physical, mixed_annotations as fixture};
mod application;
mod clusters;
mod status_filter;

async fn routes(
    c: &network::Cluster,
    layout: &Arc<Layout>,
    shards: &[(&Physical, &Server)],
) -> Result<Vec<(Ingress, String)>> {
    let budget = MergeMemoryBudget::new(8 << 20);
    let mut routes = Vec::new();
    for (shard, server) in shards {
        routes.push((
            annotations::mixed::connect_shard_ingress(
                c.client.clone(),
                layout.clone(),
                shard.binding.clone(),
                &fixture::limits(),
                budget.clone(),
                shard.shard,
            )
            .await?,
            server.uri(),
        ));
    }
    Ok(routes)
}

async fn open_bucket(
    client: &reqwest::Client,
    server: &Server,
    bucket: u32,
) -> Result<http::Opened> {
    Ok(post(client, server, "producer")
        .json(&ProducerOpen {
            bucket,
            slot: 1,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

async fn mutate(
    client: &reqwest::Client,
    server: &Server,
    producer: &http::Opened,
    sequence: u64,
    body: Bytes,
) -> Result<MutationReply> {
    Ok(update(client, server, producer, sequence, body.len())
        .body(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

async fn read(client: &reqwest::Client, server: &Server, turn: &str) -> Result<TurnAnnotation> {
    post(client, server, "read")
        .json(&PointRequest { turn: turn.into() })
        .send()
        .await?
        .error_for_status()?
        .json::<Option<TurnAnnotation>>()
        .await?
        .context("sharded point missing")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sharded_forwarders_route_original_batches_and_points_across_owner_replacement(
) -> Result<()> {
    exercise(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn configured_shard_ingresses_and_bootstrap_preserve_routing_across_owner_replacement(
) -> Result<()> {
    exercise(true).await
}

async fn exercise(application_roles: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut definition = (*layout()).clone();
    definition.buckets = vec![10, 20];
    let layout = Arc::new(definition);
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let (a, b) = if application_roles {
        (
            application::provision(&c, root.path(), 0).await?,
            application::provision(&c, root.path(), 1).await?,
        )
    } else {
        (
            Physical::provision(&c, &layout, root.path(), 0).await?,
            Physical::provision(&c, &layout, root.path(), 1).await?,
        )
    };
    let ta = a.turn(&layout);
    let tb = b.turn(&layout);
    let (sa, ra) = a.owner(&c, &layout, Duration::from_secs(60)).await?;
    let (sb, rb) = b.owner(&c, &layout, Duration::from_millis(20)).await?;
    let oa = Owner::new(&ra, keys()?, options())?;
    let ob = Owner::new(&rb, keys()?, options())?;
    let owner_a = Server::start(oa.router(), None).await?;
    let owner_b = Server::start(ob.router(), None).await?;
    let (_stop_a, task_a) = annotation_runtime::run_owner(ra);
    let (stop_b, task_b) = annotation_runtime::run_owner(rb);
    let configured = routes(&c, &layout, &[(&a, &owner_a), (&b, &owner_b)]).await?;
    assert!(
        Forwarder::new(
            vec![configured[0].clone()],
            keys()?,
            options(),
            TOKEN.into()
        )
        .is_err(),
        "incomplete logical routes must fail at startup"
    );
    assert!(Forwarder::new(
        vec![configured[0].clone(), configured[0].clone()],
        keys()?,
        options(),
        TOKEN.into()
    )
    .is_err());
    let ga = Forwarder::new(configured, keys()?, options(), TOKEN.into())?;
    let gb = Forwarder::new(
        routes(&c, &layout, &[(&b, &owner_b), (&a, &owner_a)]).await?,
        keys()?,
        options(),
        TOKEN.into(),
    )?;
    let ingress_a = if application_roles {
        application::frontend(&c, root.path(), &[(&a, &owner_a), (&b, &owner_b)]).await?
    } else {
        Server::start(ga.router(), None).await?
    };
    let ingress_b = if application_roles {
        application::frontend(&c, root.path(), &[(&b, &owner_b), (&a, &owner_a)]).await?
    } else {
        Server::start(gb.router(), None).await?
    };
    let client = reqwest::Client::builder()
        .retry(reqwest::retry::never())
        .timeout(Duration::from_secs(30))
        .build()?;
    let pa = open_bucket(&client, &ingress_a, 0).await?;
    let pb = open_bucket(&client, &ingress_b, 1).await?;
    let bytes_a = a.encode(fixture::create(&[&ta], &["same-session"], &[7])?)?;
    let bytes_b = b.encode(fixture::create(&[&tb], &["same-session"], &[9])?)?;
    let (ack_a, ack_b) = tokio::try_join!(
        mutate(&client, &ingress_b, &pa, 1, bytes_a.clone()),
        mutate(&client, &ingress_a, &pb, 1, bytes_b.clone())
    )?;
    for (physical, bytes, ack) in [(&a, &bytes_a, &ack_a), (&b, &bytes_b, &ack_b)] {
        assert_eq!(ack.completion.meta.stream, physical.binding.table);
        assert_eq!(ack.completion.meta.bucket, physical.shard.bucket);
        let fetched = c
            .client
            .request_routed(Request::Fetch {
                group: layout.group_for_bucket(physical.shard.bucket)?,
                index: ack.completion.index,
            })
            .await?;
        assert_eq!(
            &fetched.body, bytes,
            "ingress must preserve the original Arrow IPC bytes"
        );
    }
    assert_eq!(
        read(&client, &ingress_a, &ta).await?.status.as_deref(),
        Some("pending")
    );
    assert_eq!(
        read(&client, &ingress_b, &tb).await?.status.as_deref(),
        Some("pending")
    );
    mutate(
        &client,
        &ingress_a,
        &pa,
        2,
        a.patch(&ta, Some("reviewed"), None)?,
    )
    .await?;
    mutate(
        &client,
        &ingress_b,
        &pb,
        2,
        b.patch(&tb, Some("reviewed"), None)?,
    )
    .await?;

    // Both producer credentials are valid. A's credential routes to A, where
    // the whole mixed-shard batch is rejected before consuming sequence three.
    let wrong = a.encode(annotations::mixed::wire::patches(
        &fixture::patch(&[&ta, &tb], Some("wrong"), None)?,
        &fixture::limits(),
    )?)?;
    let refused = update(&client, &ingress_b, &pa, 3, wrong.len())
        .body(wrong)
        .send()
        .await?;
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        refused.json::<serde_json::Value>().await?["operation_outcome"],
        "not_submitted"
    );
    let score_a = a.patch(&ta, None, Some(0.7))?;
    let final_a = mutate(&client, &ingress_b, &pa, 3, score_a.clone()).await?;
    let current_a = read(&client, &ingress_a, &ta).await?;
    assert_eq!(current_a.status.as_deref(), Some("reviewed"));
    assert_eq!(current_a.score, Some(0.7));
    let mut invalid = pa.producer.clone().into_bytes();
    invalid[10] = if invalid[10] == b'A' { b'B' } else { b'A' };
    let invalid = http::Opened {
        producer: String::from_utf8(invalid)?,
        ..pa.clone()
    };
    let refused = update(&client, &ingress_a, &invalid, 4, score_a.len())
        .body(score_a.clone())
        .send()
        .await?;
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);

    assert!(sa.source_observation().records > 0);
    task_a.abort();
    assert!(task_a.await.unwrap_err().is_cancelled());
    let address_a = owner_a.address;
    owner_a.shutdown().await?;
    oa.shutdown().await;
    drop((oa, sa));
    // A is unavailable, but B's original runtime and both ingresses continue.
    let missing = post(&client, &ingress_b, "read")
        .json(&PointRequest { turn: ta.clone() })
        .send()
        .await?;
    assert_eq!(missing.status(), StatusCode::SERVICE_UNAVAILABLE);
    let final_b = mutate(&client, &ingress_a, &pb, 3, b.patch(&tb, None, Some(0.9))?).await?;
    b.drained(&c, &layout, &sb, final_b.completion.index)
        .await?;
    let current_b = read(&client, &ingress_b, &tb).await?;
    assert_eq!(current_b.status.as_deref(), Some("reviewed"));
    assert_eq!(current_b.score, Some(0.9));

    let (sa, ra) = a.owner(&c, &layout, Duration::from_millis(20)).await?;
    let oa = Owner::new(&ra, keys()?, options())?;
    let owner_a = Server::start(oa.router(), Some(address_a)).await?;
    let (stop_a, task_a) = annotation_runtime::run_owner(ra);
    assert_eq!(read(&client, &ingress_b, &ta).await?, current_a);
    assert_eq!(mutate(&client, &ingress_a, &pa, 3, score_a).await?, final_a);
    a.drained(&c, &layout, &sa, final_a.completion.index)
        .await?;
    fixture::check(&sa, &ta, "same-session", 7, "reviewed", Some(0.7)).await?;
    fixture::check(&sb, &tb, "same-session", 9, "reviewed", Some(0.9)).await?;

    // Close also selects by the authenticated producer scope, through either
    // ingress, without changing the existing control protocol.
    for (producer, reply) in [(&pa, &final_a), (&pb, &final_b)] {
        let closed = post(&client, &ingress_b, "close")
            .json(&http::CloseRequest {
                producer: producer.producer.clone(),
                receipt: Some(reply.receipt.clone()),
            })
            .send()
            .await?;
        assert_eq!(closed.status(), StatusCode::NO_CONTENT);
    }
    ga.shutdown().await;
    gb.shutdown().await;
    ingress_a.shutdown().await?;
    ingress_b.shutdown().await?;
    oa.shutdown().await;
    ob.shutdown().await;
    owner_a.shutdown().await?;
    owner_b.shutdown().await?;
    stop_a.send(()).unwrap();
    stop_b.send(()).unwrap();
    task_a.await??;
    task_b.await??;
    c.shutdown().await?;
    Ok(())
}
