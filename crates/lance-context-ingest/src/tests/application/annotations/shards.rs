//! Real application roles: one killed shard must not stop another from draining.
use super::*;
use crate::tests::mixed_annotations as fixture;
use crate::{annotations::Shard, tests::annotation_shards::Physical};
use lance::dataset::builder::DatasetBuilder;
use lance_context_core::annotation_staging::BodySnapshot;

async fn state(c: &network::Cluster, physical: &Physical) -> Result<Box<p::Table>> {
    let response = c
        .client
        .request_routed(Request::Query {
            group: c.client.catalog_group(),
            query: query::Query::Publisher {
                table: physical.binding.table,
            },
        })
        .await?;
    let Reply::Query(query::Outcome::Observed {
        value: query::Value::Publisher(Some(view)),
        ..
    }) = response.reply
    else {
        bail!("missing physical shard");
    };
    Ok(view)
}

async fn drained(c: &network::Cluster, physical: &Physical, through: u64) -> Result<()> {
    let group = c
        .application
        .layout
        .group_for_bucket(physical.shard.bucket)?;
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let view = state(c, physical).await?;
            if view.progress.pending.is_none()
                && view.progress.checkpoint.get(&group).copied().unwrap_or(0) >= through
                && matches!(
                    c.client
                        .request_routed(Request::Fetch {
                            group,
                            index: through
                        })
                        .await?
                        .reply,
                    Reply::Record(None)
                )
            {
                assert_eq!(view.progress.checkpoint.len(), 1);
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?
}

async fn point(client: &reqwest::Client, base: &str, turn: &str) -> Result<TurnAnnotation> {
    Ok(post(client, base, "read")
        .json(&PointRequest { turn: turn.into() })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

async fn persisted(
    c: &network::Cluster,
    physical: &Physical,
    turn: &str,
    byte: u8,
    score: f32,
) -> Result<()> {
    let handler = p::lance::ManagedManifest::reader(
        Arc::new(c.client.manifest_catalog()),
        physical.narrow.base.clone(),
        &c.application.layout,
        physical.binding.table,
        physical.binding.incarnation,
    )?;
    // Fresh readers resolve each physical shard's selected publication.
    let narrow = DatasetBuilder::from_uri(&physical.narrow.uri)
        .with_commit_handler(Arc::new(handler))
        .load()
        .await?;
    let rows = narrow
        .scan()
        .project(&["id", "status", "score"])?
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let row = rows.iter().find(|row| row.num_rows() == 1).unwrap();
    assert_eq!(
        row.column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        turn
    );
    assert_eq!(
        row.column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "reviewed"
    );
    let scores = row
        .column(2)
        .as_any()
        .downcast_ref::<arrow_array::Float32Array>()
        .unwrap();
    assert!(!scores.is_null(0));
    assert_eq!(scores.value(0), score);
    let reference = BodySnapshot::read(&narrow)
        .await?
        .context("body snapshot missing")?;
    let body = crate::annotations::mixed::load_body(
        &physical.body,
        &reference,
        &physical.binding.incarnation.to_string(),
    )
    .await?;
    let rows = body
        .scan()
        .project(&["id", "binary_payload"])?
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let row = rows.iter().find(|row| row.num_rows() == 1).unwrap();
    assert_eq!(
        row.column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        turn
    );
    assert_eq!(
        row.column(1)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap()
            .value(0),
        vec![byte; 16 << 10]
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires INGEST_TEST_BIN and ETCD_TEST_ENDPOINTS; real sharded owner/ingress processes"]
async fn sharded_executable_recovers_acked_suffix_while_other_owner_drains() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), exercise()).await?
}

async fn exercise() -> Result<()> {
    let binary = std::env::var("INGEST_TEST_BIN")?;
    let endpoints = std::env::var("ETCD_TEST_ENDPOINTS")?;
    let root = tempfile::Builder::new()
        .prefix("sharded-application-")
        .tempdir_in(std::env::var("INGEST_TEST_ARTIFACTS")?)?
        .keep();
    let mut definition = (*layout()).clone();
    definition.buckets = vec![10, 20];
    let c = network::Cluster::open_signed(&root, Arc::new(definition)).await?;
    let reserves = (0..5)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0"))
        .collect::<std::io::Result<Vec<_>>>()?;
    let addresses = reserves
        .iter()
        .map(|l| l.local_addr())
        .collect::<std::io::Result<Vec<_>>>()?;
    let bases: Vec<_> = addresses.iter().map(|a| format!("http://{a}")).collect();
    let token = root.join("token");
    let receipts = root.join("receipts.json");
    std::fs::write(&token, TOKEN)?;
    std::fs::write(
        &receipts,
        serde_json::to_vec(&serde_json::json!({"active":1,"keys":{"1":vec![17u8;32]}}))?,
    )?;
    let prefix = format!("/lance-context-test/sharded-application/{}", Uuid::new_v4());
    let mut physicals = Vec::new();
    let mut configs = Vec::new();
    for bucket in 0..2 {
        let mut binding = binding();
        binding.table = Uuid::new_v4();
        binding.incarnation = Uuid::new_v4();
        binding.destination.dataset = format!("annotations-{bucket}.lance");
        let physical = Physical {
            narrow: fixture::storage(&root, &binding.destination.dataset).await?,
            body: fixture::storage(&root, &format!("body-{bucket}.lance")).await?,
            binding,
            shard: Shard { bucket },
        };
        let mut config = c.application.clone();
        config.ingress_memory_bytes = 8 << 20;
        config.materializer_memory_bytes = 8 << 20;
        config.tables = vec![config::Table {
            binding: physical.binding.clone(),
            uri: physical.narrow.uri.clone(),
            storage_options: Default::default(),
            bootstrap: bootstrap::identity(),
        }];
        config.etcd = Some(config::Etcd {
            endpoints: endpoints.split(',').map(str::to_owned).collect(),
            prefix: prefix.clone(),
            ttl_seconds: 10,
            timeout: Duration::from_secs(3),
            tls: None,
        });
        config.http = Some(config::Http {
            listen: addresses[bucket as usize],
            bearer_token_file: token.clone(),
            receipt_key_file: receipts.clone(),
            options: crate::http::Options {
                body_total_timeout: Duration::from_secs(10),
                ..Default::default()
            },
        });
        config.annotations = Some(config::Annotations {
            owner_url: bases[bucket as usize].clone(),
            shard: Some(physical.shard),
            routes: vec![],
            wal_clusters: BTreeMap::new(),
            apply_memory_bytes: 8 << 20,
            staging: fixture::limits(),
            source: None,
            projection: None,
            body: Some(config::Body {
                uri: physical.body.uri.clone(),
                storage_options: Default::default(),
            }),
            exclusive_reclamation: None,
        });
        config.worker.publication.max_source_bytes = 1 << 20;
        config.worker.publication.min_source_bytes = 1 << 20;
        config.worker.publication.max_records = 64;
        config.worker.publication.max_batch_wait = if bucket == 0 {
            Duration::from_secs(60)
        } else {
            Duration::from_millis(20)
        };
        physicals.push(physical);
        configs.push(config);
    }
    let files: Vec<_> = (0..5)
        .map(|i| root.join(format!("config-{i}.json")))
        .collect();
    for (i, config) in configs.iter().enumerate() {
        std::fs::write(&files[i], serde_json::to_vec(config)?)?;
        for n in 0..2 {
            Process::spawn(
                &binary,
                &files[i],
                "annotation-bootstrap",
                &root.join(format!("bootstrap-{i}-{n}.log")),
            )?
            .wait_success()
            .await?;
        }
    }
    let routes: Vec<_> = physicals
        .iter()
        .enumerate()
        .map(|(i, physical)| config::AnnotationRoute {
            binding: physical.binding.clone(),
            shard: physical.shard,
            owner_url: bases[i].clone(),
            wal_cluster: None,
        })
        .collect();
    for i in 2..4 {
        let mut config = configs[0].clone();
        config.annotations.as_mut().unwrap().routes = routes.clone();
        config.http.as_mut().unwrap().listen = addresses[i];
        config.tables[0].uri = "unsupported://ingress-must-not-open-narrow".into();
        config
            .annotations
            .as_mut()
            .unwrap()
            .body
            .as_mut()
            .unwrap()
            .uri = "unsupported://ingress-must-not-open-body".into();
        std::fs::write(&files[i], serde_json::to_vec(&config)?)?;
    }
    let mut competing = configs[0].clone();
    competing.http.as_mut().unwrap().listen = addresses[4];
    std::fs::write(&files[4], serde_json::to_vec(&competing)?)?;
    drop(reserves);
    let client = reqwest::Client::builder()
        .retry(reqwest::retry::never())
        .timeout(Duration::from_secs(15))
        .build()?;
    let mut a = Process::spawn(
        &binary,
        &files[0],
        "annotation-owner",
        &root.join("owner-a.log"),
    )?;
    let mut b = Process::spawn(
        &binary,
        &files[1],
        "annotation-owner",
        &root.join("owner-b.log"),
    )?;
    ready(&client, &bases[0], TOKEN, &mut a).await?;
    ready(&client, &bases[1], TOKEN, &mut b).await?;
    let before_a = state(&c, &physicals[0]).await?.authority;
    let before_b = state(&c, &physicals[1]).await?.authority;
    let mut competitor = Process::spawn(
        &binary,
        &files[4],
        "annotation-owner",
        &root.join("competitor.log"),
    )?;
    failed(&mut competitor).await?;
    assert!(std::fs::read_to_string(root.join("competitor.log"))?
        .contains("annotation shard already claimed"));
    assert_eq!(state(&c, &physicals[0]).await?.authority, before_a);
    let mut ingress_a = Process::spawn(
        &binary,
        &files[2],
        "annotation-ingress",
        &root.join("ingress-a.log"),
    )?;
    let mut ingress_b = Process::spawn(
        &binary,
        &files[3],
        "annotation-ingress",
        &root.join("ingress-b.log"),
    )?;
    ready(&client, &bases[2], TOKEN, &mut ingress_a).await?;
    ready(&client, &bases[3], TOKEN, &mut ingress_b).await?;
    let turns: Vec<_> = physicals
        .iter()
        .map(|p| p.turn(&c.application.layout))
        .collect();
    let mut producers = Vec::new();
    for i in 0..2 {
        let producer: Opened = post(&client, &bases[2], "producer")
            .json(&ProducerOpen {
                bucket: i as u32,
                slot: 1,
                expected_generation: 0,
                operation: Uuid::new_v4(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        update(
            &client,
            &bases[2],
            &producer,
            1,
            physicals[i].encode(fixture::create(
                &[&turns[i]],
                &["same-session"],
                &[7 + i as u8],
            )?)?,
        )
        .await?;
        assert_eq!(
            point(&client, &bases[3], &turns[i])
                .await?
                .status
                .as_deref(),
            Some("pending")
        );
        update(
            &client,
            &bases[3],
            &producer,
            2,
            physicals[i].patch(&turns[i], Some("reviewed"), None)?,
        )
        .await?;
        producers.push(producer);
    }
    let score_a = physicals[0].patch(&turns[0], None, Some(0.7))?;
    let ack_a = update(&client, &bases[2], &producers[0], 3, score_a.clone()).await?;
    let visible_a = point(&client, &bases[3], &turns[0]).await?;
    assert_eq!(visible_a.status.as_deref(), Some("reviewed"));
    assert_eq!(visible_a.score, Some(0.7));
    assert!(
        state(&c, &physicals[0])
            .await?
            .progress
            .checkpoint
            .get(&10)
            .copied()
            .unwrap_or(0)
            < ack_a.completion.index
    );
    assert!(matches!(
        c.client
            .request_routed(Request::Fetch {
                group: 10,
                index: ack_a.completion.index
            })
            .await?
            .reply,
        Reply::Record(Some(_))
    ));
    a.0.kill()?;
    ensure!(!a.0.wait()?.success(), "owner SIGKILL failed");
    let unavailable = post(&client, &bases[3], "read")
        .json(&PointRequest {
            turn: turns[0].clone(),
        })
        .send()
        .await?;
    assert_eq!(
        unavailable.status(),
        reqwest::StatusCode::SERVICE_UNAVAILABLE
    );
    let ack_b = update(
        &client,
        &bases[2],
        &producers[1],
        3,
        physicals[1].patch(&turns[1], None, Some(0.9))?,
    )
    .await?;
    drained(&c, &physicals[1], ack_b.completion.index).await?;
    let visible_b = point(&client, &bases[3], &turns[1]).await?;
    assert_eq!(visible_b.status.as_deref(), Some("reviewed"));
    assert_eq!(visible_b.score, Some(0.9));
    assert_eq!(state(&c, &physicals[1]).await?.authority, before_b);
    let mut etcd =
        etcd_client::Client::connect(endpoints.split(',').collect::<Vec<_>>(), None).await?;
    let key = format!(
        "{}/{}/{}",
        prefix, c.application.layout.cluster, physicals[0].binding.table
    );
    tokio::time::timeout(Duration::from_secs(30), async {
        while !etcd.get(key.clone(), None).await?.kvs().is_empty() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    configs[0].worker.publication.max_batch_wait = Duration::from_millis(20);
    let replacement_file = root.join("replacement.json");
    std::fs::write(&replacement_file, serde_json::to_vec(&configs[0])?)?;
    let mut replacement = Process::spawn(
        &binary,
        &replacement_file,
        "annotation-owner",
        &root.join("replacement.log"),
    )?;
    ready(&client, &bases[0], TOKEN, &mut replacement).await?;
    assert_eq!(point(&client, &bases[2], &turns[0]).await?, visible_a);
    // Explicit idempotency probe of one known ACK; no automatic mutation retries.
    assert_eq!(
        update(&client, &bases[3], &producers[0], 3, score_a).await?,
        ack_a
    );
    drained(&c, &physicals[0], ack_a.completion.index).await?;
    assert!(state(&c, &physicals[0]).await?.authority.epoch > before_a.epoch);
    assert_eq!(state(&c, &physicals[1]).await?.authority, before_b);
    assert_eq!(point(&client, &bases[3], &turns[0]).await?, visible_a);
    assert_eq!(point(&client, &bases[2], &turns[1]).await?, visible_b);
    ingress_a.stop().await?;
    ingress_b.stop().await?;
    replacement.stop().await?;
    b.stop().await?;
    persisted(&c, &physicals[0], &turns[0], 7, 0.7).await?;
    persisted(&c, &physicals[1], &turns[1], 8, 0.9).await?;
    c.shutdown().await
}
