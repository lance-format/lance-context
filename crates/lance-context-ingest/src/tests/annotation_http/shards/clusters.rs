use super::*;

fn definition(group: u64) -> Arc<Layout> {
    let mut definition = (*layout()).clone();
    definition.cluster = Uuid::new_v4();
    definition.groups = BTreeMap::from([(group, vec![1, 2, 3])]);
    definition.buckets = vec![group; 2];
    Arc::new(definition)
}

async fn configured(
    a: (&network::Cluster, &Arc<Layout>, &Physical, &Server),
    b: (&network::Cluster, &Arc<Layout>, &Physical, &Server),
) -> Result<Vec<(Ingress, String)>> {
    let budget = MergeMemoryBudget::new(8 << 20);
    let mut routes = Vec::new();
    for (cluster, layout, physical, server) in [a, b] {
        routes.push((
            annotations::mixed::connect_shard_ingress(
                cluster.client.clone(),
                layout.clone(),
                physical.binding.clone(),
                &fixture::limits(),
                budget.clone(),
                physical.shard,
            )
            .await?,
            server.uri(),
        ));
    }
    Ok(routes)
}

fn validate_routes(routes: &[(Ingress, String)]) -> Result<()> {
    let rejects = |routes| {
        assert!(Forwarder::new(routes, keys().unwrap(), options(), TOKEN.into()).is_err());
    };
    rejects(vec![routes[0].clone()]);
    rejects(vec![routes[0].clone(), routes[0].clone()]);
    let mut different_count = routes.to_vec();
    Arc::make_mut(&mut different_count[1].0.layout).buckets = vec![20; 4];
    rejects(different_count);
    let mut conflict = routes.to_vec();
    Arc::make_mut(&mut conflict[1].0.scope).cluster = routes[0].0.scope.cluster;
    rejects(conflict);
    let mut budget = routes.to_vec();
    budget[1].0.budget = MergeMemoryBudget::new(8 << 20);
    rejects(budget);
    let mut tenant = routes.to_vec();
    Arc::make_mut(&mut tenant[1].0.scope).contract.tenant = "another-tenant".into();
    rejects(tenant);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn independent_wal_clusters_preserve_routes_and_drain_during_peer_cluster_outage(
) -> Result<()> {
    let root_a = tempfile::tempdir()?;
    let root_b = tempfile::tempdir()?;
    let la = definition(10);
    let lb = definition(20);
    let ca = network::Cluster::open_signed(root_a.path(), la.clone()).await?;
    let cb = network::Cluster::open_signed(root_b.path(), lb.clone()).await?;
    assert_ne!(ca.client.catalog_group(), cb.client.catalog_group());
    let a = Physical::provision(&ca, &la, root_a.path(), 0).await?;
    let b = Physical::provision(&cb, &lb, root_b.path(), 1).await?;
    let ta = a.turn(&la);
    let tb = b.turn(&lb);
    assert_eq!(a.turn(&lb), ta);
    assert_eq!(b.turn(&la), tb);
    let (sa, ra) = a.owner(&ca, &la, Duration::from_secs(60)).await?;
    let (sb, rb) = b.owner(&cb, &lb, Duration::from_millis(20)).await?;
    let oa = Owner::new(&ra, keys()?, options())?;
    let ob = Owner::new(&rb, keys()?, options())?;
    let owner_a = Server::start(oa.router(), None).await?;
    let owner_b = Server::start(ob.router(), None).await?;
    let (_stop_a, task_a) = annotation_runtime::run_owner(ra);
    let (stop_b, task_b) = annotation_runtime::run_owner(rb);
    let routes = configured((&ca, &la, &a, &owner_a), (&cb, &lb, &b, &owner_b)).await?;
    validate_routes(&routes)?;
    let ga = Forwarder::new(routes.clone(), keys()?, options(), TOKEN.into())?;
    let gb = Forwarder::new(
        routes.iter().cloned().rev().collect(),
        keys()?,
        options(),
        TOKEN.into(),
    )?;
    let ingress_a = Server::start(ga.router(), None).await?;
    let ingress_b = Server::start(gb.router(), None).await?;
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
    for (c, l, physical, bytes, ack) in [
        (&ca, &la, &a, &bytes_a, &ack_a),
        (&cb, &lb, &b, &bytes_b, &ack_b),
    ] {
        let fetched = c
            .client
            .request_routed(Request::Fetch {
                group: l.group_for_bucket(physical.shard.bucket)?,
                index: ack.completion.index,
            })
            .await?;
        assert_eq!(
            &fetched.body, bytes,
            "forwarded Arrow bytes must be unchanged"
        );
    }

    // A valid signature for the same table and bucket in a foreign cluster
    // must not become an authorized route or consume the producer sequence.
    let original = routes[0].0.restore_producer(&keys()?, &pa.producer)?;
    for change_layout in [false, true] {
        let mut foreign = original.clone();
        let scope = Arc::make_mut(&mut foreign.scope);
        if change_layout {
            scope.layout = lb.fingerprint()?;
        } else {
            scope.cluster = lb.cluster;
        }
        let opened = http::Opened {
            producer: keys()?.encode_producer(&foreign)?,
            ..pa.clone()
        };
        let refused = update(&client, &ingress_b, &opened, 2, bytes_a.len())
            .body(bytes_a.clone())
            .send()
            .await?;
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
        let failure = refused.json::<serde_json::Value>().await?;
        assert_eq!(failure["error"], "foreign_producer_route");
        assert_eq!(failure["operation_outcome"], "not_submitted");
    }
    mutate(
        &client,
        &ingress_b,
        &pa,
        2,
        a.patch(&ta, Some("reviewed"), None)?,
    )
    .await?;
    let score = a.patch(&ta, None, Some(0.7))?;
    let final_a = mutate(&client, &ingress_a, &pa, 3, score.clone()).await?;
    let current_a = read(&client, &ingress_b, &ta).await?;
    assert_eq!(current_a.status.as_deref(), Some("reviewed"));
    assert_eq!(current_a.score, Some(0.7));

    assert!(sa.source_observation().records > 0);
    task_a.abort();
    assert!(task_a.await.unwrap_err().is_cancelled());
    let address_a = owner_a.address;
    owner_a.shutdown().await?;
    oa.shutdown().await;
    drop((oa, sa));
    ca.shutdown().await?;
    let missing = post(&client, &ingress_b, "read")
        .json(&PointRequest { turn: ta.clone() })
        .send()
        .await?;
    assert_eq!(missing.status(), StatusCode::SERVICE_UNAVAILABLE);
    // All three WAL nodes and the owner for A are down. B still publishes and
    // drops its staging/source state through its own catalog and data group.
    mutate(
        &client,
        &ingress_a,
        &pb,
        2,
        b.patch(&tb, Some("reviewed"), None)?,
    )
    .await?;
    let final_b = mutate(&client, &ingress_b, &pb, 3, b.patch(&tb, None, Some(0.9))?).await?;
    b.drained(&cb, &lb, &sb, final_b.completion.index).await?;
    let current_b = read(&client, &ingress_a, &tb).await?;
    assert_eq!(current_b.status.as_deref(), Some("reviewed"));
    assert_eq!(current_b.score, Some(0.9));

    let ca = network::Cluster::open_signed(root_a.path(), la.clone()).await?;
    let (sa, ra) = a.owner(&ca, &la, Duration::from_millis(20)).await?;
    let oa = Owner::new(&ra, keys()?, options())?;
    let owner_a = Server::start(oa.router(), Some(address_a)).await?;
    let (stop_a, task_a) = annotation_runtime::run_owner(ra);
    assert_eq!(read(&client, &ingress_b, &ta).await?, current_a);
    assert_eq!(mutate(&client, &ingress_a, &pa, 3, score).await?, final_a);
    a.drained(&ca, &la, &sa, final_a.completion.index).await?;
    fixture::check(&sa, &ta, "same-session", 7, "reviewed", Some(0.7)).await?;
    fixture::check(&sb, &tb, "same-session", 9, "reviewed", Some(0.9)).await?;
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
    ca.shutdown().await?;
    cb.shutdown().await?;
    Ok(())
}
