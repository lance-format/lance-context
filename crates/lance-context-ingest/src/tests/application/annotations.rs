use super::*;
use crate::{
    http::annotations::{MutationReply, PointRequest},
    materializer::Storage,
    tests::{annotation_publication, annotations::wire},
};
use lance_context_api::annotation::TurnAnnotation;
use lance_context_core::annotation_staging::Limits;

const TOKEN: &str = "annotation-application-test-token-0001";

fn post(client: &reqwest::Client, base: &str, route: &str) -> reqwest::RequestBuilder {
    client
        .post(format!("{base}/api/v1/annotations/rollouts/{route}"))
        .bearer_auth(TOKEN)
}
async fn update(
    client: &reqwest::Client,
    base: &str,
    producer: &Opened,
    sequence: u64,
    body: Bytes,
) -> Result<MutationReply> {
    Ok(post(client, base, "update")
        .header(crate::http::PRODUCER, &producer.producer)
        .header(crate::http::SEQUENCE, sequence)
        .header(reqwest::header::CONTENT_TYPE, crate::http::ARROW_FILE)
        .body(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}
async fn read(client: &reqwest::Client, base: &str) -> Result<TurnAnnotation> {
    Ok(post(client, base, "read")
        .json(&PointRequest { turn: "T".into() })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}
async fn failed(process: &mut Process) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Some(status) = process.0.try_wait()? {
                ensure!(!status.success(), "failed role exited successfully");
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?
}
async fn reclaimed(c: &network::Cluster, index: u64) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let response = c
                .client
                .request_routed(Request::Fetch { group: 10, index })
                .await?;
            if matches!(response.reply, Reply::Record(None)) {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires INGEST_TEST_BIN and ETCD_TEST_ENDPOINTS; executed by application CI"]
async fn annotation_executable_two_ingresses_restart_owner_and_fence_lost_lease() -> Result<()> {
    let binary = std::env::var("INGEST_TEST_BIN")?;
    let endpoints = std::env::var("ETCD_TEST_ENDPOINTS")?;
    let root = tempfile::Builder::new()
        .prefix("annotation-application-")
        .tempdir_in(std::env::var("INGEST_TEST_ARTIFACTS")?)?
        .keep();
    let c = network::Cluster::open_signed(&root, annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.join("turns.lance").to_str().unwrap().into(),
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
    let original = bootstrap::annotations::original(&storage).await?;
    let source = crate::bootstrap::annotations::Snapshot {
        version: original.version_id(),
        transaction: Uuid::parse_str(&original.read_transaction().await?.unwrap().uuid)?,
    };
    let mut config = c.application.clone();
    config.tables.push(config::Table {
        binding: binding(),
        uri: storage.uri.clone(),
        storage_options: Default::default(),
        bootstrap: bootstrap::identity(),
    });
    let prefix = format!(
        "/lance-context-test/annotation-application/{}",
        Uuid::new_v4()
    );
    config.etcd = Some(config::Etcd {
        endpoints: endpoints.split(',').map(str::to_owned).collect(),
        prefix: prefix.clone(),
        ttl_seconds: 10,
        timeout: Duration::from_secs(3),
        tls: None,
    });
    let reserves: Vec<_> = (0..4)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0"))
        .collect::<std::io::Result<_>>()?;
    let addresses: Vec<_> = reserves
        .iter()
        .map(|l| l.local_addr())
        .collect::<std::io::Result<_>>()?;
    let bases: Vec<_> = addresses.iter().map(|a| format!("http://{a}")).collect();
    let bearer = root.join("token");
    std::fs::write(&bearer, TOKEN)?;
    let receipts = root.join("receipts.json");
    std::fs::write(
        &receipts,
        serde_json::to_vec(&serde_json::json!({"active":1,"keys":{"1":vec![17u8;32]}}))?,
    )?;
    config.http = Some(config::Http {
        listen: addresses[0],
        bearer_token_file: bearer,
        receipt_key_file: receipts,
        options: crate::http::Options {
            body_total_timeout: Duration::from_secs(10),
            ..Default::default()
        },
    });
    config.annotations = Some(config::Annotations {
        owner_url: bases[0].clone(),
        apply_memory_bytes: 8 << 20,
        staging: limits.clone(),
        source: Some(source),
    });
    config.worker.publication.max_records = 2;
    config.worker.publication.min_source_bytes = 1 << 20;
    config.worker.publication.max_batch_wait = Duration::from_secs(60);
    let files: Vec<_> = (0..4)
        .map(|i| root.join(format!("config-{i}.json")))
        .collect();
    for (i, file) in files.iter().enumerate() {
        config.http.as_mut().unwrap().listen = addresses[i];
        // Forwarding roles must not touch Lance even with unusable storage options.
        if i == 1 || i == 2 {
            config.tables[0].uri = "unsupported://forwarder-must-not-open-lance".into();
        } else {
            config.tables[0].uri = storage.uri.clone();
        }
        std::fs::write(file, serde_json::to_vec(&config)?)?;
    }
    drop(reserves);
    let mut bootstrap = Process::spawn(
        &binary,
        &files[0],
        "annotation-bootstrap",
        &root.join("bootstrap.log"),
    )?;
    bootstrap.wait_success().await?;
    assert_eq!(
        Dataset::open(&storage.uri).await?.manifest().fragments,
        original.manifest().fragments
    );
    let mut owner = Process::spawn(
        &binary,
        &files[0],
        "annotation-owner",
        &root.join("owner.log"),
    )?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;
    ready(&client, &bases[0], TOKEN, &mut owner).await?;
    let authority = annotation_publication::state(&c).await?.authority;
    let mut competitor = Process::spawn(
        &binary,
        &files[3],
        "annotation-owner",
        &root.join("competitor.log"),
    )?;
    failed(&mut competitor).await?;
    assert!(std::fs::read_to_string(root.join("competitor.log"))?
        .contains("annotation shard already claimed"));
    assert_eq!(
        annotation_publication::state(&c).await?.authority,
        authority
    );
    let mut a = Process::spawn(
        &binary,
        &files[1],
        "annotation-ingress",
        &root.join("a.log"),
    )?;
    let mut b = Process::spawn(
        &binary,
        &files[2],
        "annotation-ingress",
        &root.join("b.log"),
    )?;
    ready(&client, &bases[1], TOKEN, &mut a).await?;
    ready(&client, &bases[2], TOKEN, &mut b).await?;
    let initial = read(&client, &bases[2]).await?;
    assert_eq!(initial.score, Some(0.5));
    assert!(initial.labels.is_empty());
    assert_eq!(initial.status, None);
    for base in &bases[..3] {
        assert_eq!(
            client
                .get(format!("{base}/api/v1/health"))
                .send()
                .await?
                .status(),
            401
        );
    }
    let producer: Opened = post(&client, &bases[1], "producer")
        .json(&ProducerOpen {
            bucket: 0,
            slot: 1,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let body = wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?;
    update(&client, &bases[1], &producer, 1, body).await?;
    assert_eq!(
        read(&client, &bases[2]).await?.status.as_deref(),
        Some("reviewed")
    );
    let body = wire(&limits, &["T"], 0, None, 1, Some(0.9))?;
    let published = update(&client, &bases[2], &producer, 2, body.clone()).await?;
    reclaimed(&c, published.completion.index).await?;
    assert_eq!(
        update(&client, &bases[1], &producer, 2, body).await?,
        published
    );
    let body = wire(&limits, &["T"], 1, Some("acknowledged-suffix"), 0, None)?;
    let suffix = update(&client, &bases[1], &producer, 3, body.clone()).await?;
    assert_eq!(read(&client, &bases[2]).await?.score, Some(0.9));
    assert!(matches!(
        c.client
            .request_routed(Request::Fetch {
                group: 10,
                index: suffix.completion.index
            })
            .await?
            .reply,
        Reply::Record(Some(_))
    ));
    owner.0.kill()?;
    ensure!(!owner.0.wait()?.success(), "owner SIGKILL failed");
    let mut etcd =
        etcd_client::Client::connect(endpoints.split(',').collect::<Vec<_>>(), None).await?;
    let claim_key = format!("{}/{}/{}", prefix, config.layout.cluster, binding().table);
    // A crash leaves the original claim until its real etcd lease expires.
    let expired_at = tokio::time::Instant::now();
    tokio::time::timeout(Duration::from_secs(60), async {
        while !etcd.get(claim_key.clone(), None).await?.kvs().is_empty() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await
    .context("old annotation owner lease did not expire")??;
    eprintln!(
        "annotation owner lease observed expired after {:?}",
        expired_at.elapsed()
    );
    let mut replacement = Process::spawn(
        &binary,
        &files[0],
        "annotation-owner",
        &root.join("replacement.log"),
    )?;
    ready(&client, &bases[0], TOKEN, &mut replacement).await?;
    let value = read(&client, &bases[2]).await?;
    assert_eq!(value.status.as_deref(), Some("acknowledged-suffix"));
    assert_eq!(value.score, Some(0.9));
    assert_eq!(
        update(&client, &bases[2], &producer, 3, body).await?,
        suffix
    );
    let clear = wire(&limits, &["T"], 2, None, 2, None)?;
    let cleared = update(&client, &bases[2], &producer, 4, clear).await?;
    reclaimed(&c, cleared.completion.index).await?;
    let value = read(&client, &bases[1]).await?;
    assert_eq!((value.status, value.score), (None, None));
    assert!(value.labels.is_empty());
    let lease = etcd.get(claim_key, None).await?.kvs()[0].lease();
    etcd.lease_revoke(lease).await?;
    failed(&mut replacement).await?;
    assert!(std::fs::read_to_string(root.join("replacement.log"))?
        .contains("annotation owner lease lost"));
    a.stop().await?;
    b.stop().await?;
    etcd.delete(
        prefix,
        Some(etcd_client::DeleteOptions::new().with_prefix()),
    )
    .await?;
    c.shutdown().await
}
