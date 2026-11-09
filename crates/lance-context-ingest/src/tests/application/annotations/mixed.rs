use super::*;
use crate::{annotations::mixed, tests::mixed_annotations as fixture};
use lance_context_core::annotation_staging::BodySnapshot;

#[cfg(unix)]
mod publication_failure;

enum Failure {
    Sigkill,
    SigkillStaging,
    #[cfg(unix)]
    Publication,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires INGEST_TEST_BIN and ETCD_TEST_ENDPOINTS; real mixed owner/ingress processes"]
async fn mixed_executable_recovers_acked_import_and_annotations_after_sigkill() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Failure::Sigkill)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires INGEST_TEST_BIN and ETCD_TEST_ENDPOINTS; real mixed owner/ingress processes"]
async fn mixed_executable_reads_recovered_annotations_before_publication() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Failure::SigkillStaging)).await?
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires INGEST_TEST_BIN and ETCD_TEST_ENDPOINTS; real owner publication failure"]
async fn mixed_executable_recovers_after_native_write_failure_without_resubmitting_requests(
) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(120), exercise(Failure::Publication)).await?
}

async fn exercise(failure: Failure) -> Result<()> {
    let binary = std::env::var("INGEST_TEST_BIN")?;
    let endpoints = std::env::var("ETCD_TEST_ENDPOINTS")?;
    let root = tempfile::Builder::new()
        .prefix("mixed-application-")
        .tempdir_in(std::env::var("INGEST_TEST_ARTIFACTS")?)?
        .keep();
    let c = network::Cluster::open_signed(&root, annotation_publication::one_shard()).await?;
    let narrow = fixture::storage(&root, "annotations.lance").await?;
    let body = fixture::storage(&root, "body.lance").await?;
    let mut config = c.application.clone();
    config.ingress_memory_bytes = 8 << 20;
    config.materializer_memory_bytes = 8 << 20;
    config.tables.push(config::Table {
        binding: binding(),
        uri: narrow.uri.clone(),
        storage_options: Default::default(),
        bootstrap: bootstrap::identity(),
    });
    let prefix = format!("/lance-context-test/mixed-application/{}", Uuid::new_v4());
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
    let token = root.join("token");
    let receipts = root.join("receipts.json");
    std::fs::write(&token, TOKEN)?;
    std::fs::write(
        &receipts,
        serde_json::to_vec(&serde_json::json!({"active":1,"keys":{"1":vec![17u8;32]}}))?,
    )?;
    config.http = Some(config::Http {
        listen: addresses[0],
        bearer_token_file: token,
        receipt_key_file: receipts,
        options: crate::http::Options {
            body_total_timeout: Duration::from_secs(10),
            ..Default::default()
        },
    });
    config.annotations = Some(config::Annotations {
        shard: None,
        routes: vec![],
        wal_clusters: BTreeMap::new(),
        owner_url: bases[0].clone(),
        apply_memory_bytes: 8 << 20,
        staging: fixture::limits(),
        source: None,
        projection: None,
        body: Some(config::Body {
            uri: body.uri.clone(),
            storage_options: Default::default(),
        }),
        exclusive_reclamation: Some(crate::annotations::runtime::reclamation::Options {
            max_objects: 2,
            max_bytes: 16 << 10,
            delete_parallelism: 2,
            interval: Duration::from_millis(10),
        }),
    });
    config.worker.publication.max_source_bytes = 1 << 20;
    config.worker.publication.min_source_bytes = 1 << 20;
    config.worker.publication.max_records = 2;
    // Same deterministic two-record publication window as the existing process
    // recovery test. The single-record suffix must remain unmaterialized.
    config.worker.publication.max_batch_wait = Duration::from_secs(60);
    config.worker.publication.retained_manifest_versions = 2;
    let files: Vec<_> = (0..4)
        .map(|i| root.join(format!("config-{i}.json")))
        .collect();
    for (i, file) in files.iter().enumerate() {
        let mut role = config.clone();
        role.http.as_mut().unwrap().listen = addresses[i];
        if i == 1 || i == 2 {
            role.tables[0].uri = "unsupported://ingress-must-not-open-narrow".into();
            role.annotations
                .as_mut()
                .unwrap()
                .body
                .as_mut()
                .unwrap()
                .uri = "unsupported://ingress-must-not-open-body".into();
        }
        std::fs::write(file, serde_json::to_vec(&role)?)?;
    }
    // Use the real provisioning entry point, including repeated invocation.
    // No test fixture creates the paired snapshots for these processes.
    for i in 0..2 {
        Process::spawn(
            &binary,
            &files[0],
            "annotation-bootstrap",
            &root.join(format!("bootstrap-{i}.log")),
        )?
        .wait_success()
        .await?;
    }
    drop(reserves);
    let mut owner = Process::spawn(
        &binary,
        &files[0],
        "annotation-owner",
        &root.join("owner.log"),
    )?;
    let client = reqwest::Client::builder()
        .retry(reqwest::retry::never())
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
    let first = fixture::encoded(fixture::create(&["T"], &["s"], &[1])?)?;
    update(&client, &bases[1], &producer, 1, first).await?;
    assert_eq!(
        read(&client, &bases[2]).await?.status.as_deref(),
        Some("pending")
    );
    let patch = fixture::encoded(mixed::wire::patches(
        &fixture::patch(&["T"], Some("reviewed"), Some(0.9))?,
        &fixture::limits(),
    )?)?;
    let patched = update(&client, &bases[2], &producer, 2, patch).await?;
    reclaimed(&c, patched.completion.index).await?;
    let materialized_version = annotation_publication::read(&c, &narrow, None)
        .await?
        .version_id();
    let materialized_cut = annotation_publication::state(&c).await?.progress.checkpoint;
    let (suffix, current_status, current_score) = if matches!(failure, Failure::Sigkill) {
        (fixture::create(&["U"], &["s"], &[2])?, "reviewed", 0.9)
    } else {
        (
            mixed::wire::patches(
                &fixture::patch(&["T"], Some("approved"), Some(0.7))?,
                &fixture::limits(),
            )?,
            "approved",
            0.7,
        )
    };
    let suffix = fixture::encoded(suffix)?;
    let imported = update(&client, &bases[1], &producer, 3, suffix.clone()).await?;
    let visible: TurnAnnotation = post(&client, &bases[2], "read")
        .json(&PointRequest {
            turn: if matches!(failure, Failure::Sigkill) {
                "U"
            } else {
                "T"
            }
            .into(),
        })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(
        visible.status.as_deref(),
        Some(if matches!(failure, Failure::Sigkill) {
            "pending"
        } else {
            current_status
        })
    );
    if !matches!(failure, Failure::Sigkill) {
        assert_eq!(visible.score, Some(current_score));
    }
    assert_eq!(
        annotation_publication::read(&c, &narrow, None)
            .await?
            .count_rows(None)
            .await?,
        1
    );
    assert!(matches!(
        c.client
            .request_routed(Request::Fetch {
                group: 10,
                index: imported.completion.index
            })
            .await?
            .reply,
        Reply::Record(Some(_))
    ));
    let (through, next_sequence) = match failure {
        Failure::Sigkill | Failure::SigkillStaging => {
            owner.0.kill()?;
            ensure!(!owner.0.wait()?.success(), "owner SIGKILL failed");
            (imported.completion.index, 4)
        }
        #[cfg(unix)]
        Failure::Publication => {
            let through = publication_failure::fail(
                &c,
                &root,
                &client,
                &bases[1],
                &producer,
                &mut owner,
                imported.completion.index,
            )
            .await?;
            // A failed body upload leaves the owner and its ACKed narrow
            // view available through both ingresses until a fenced restart.
            for base in &bases[1..=2] {
                let current = read(&client, base).await?;
                assert_eq!(current.status.as_deref(), Some(current_status));
                assert_eq!(current.score, Some(current_score));
            }
            assert!(owner.0.try_wait()?.is_none());
            owner.stop().await?;
            // An exited owner must yield an explicit unavailable response,
            // not a successful stale base read or an invented missing turn.
            for base in &bases[1..=2] {
                let response = post(&client, base, "read")
                    .json(&PointRequest { turn: "T".into() })
                    .send()
                    .await?;
                assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
            }
            (through, 5)
        }
    };
    let mut etcd =
        etcd_client::Client::connect(endpoints.split(',').collect::<Vec<_>>(), None).await?;
    let key = format!("{}/{}/{}", prefix, config.layout.cluster, binding().table);
    tokio::time::timeout(Duration::from_secs(30), async {
        while !etcd.get(key.clone(), None).await?.kvs().is_empty() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    // Keep the existing two-record window for the staging-read case. A single
    // retained update must be readable while the native version is unchanged.
    // The other cases exercise automatic drain with no new business requests.
    if !matches!(failure, Failure::SigkillStaging) {
        config.worker.publication.max_batch_wait = Duration::from_millis(20);
    }
    std::fs::write(&files[0], serde_json::to_vec(&config)?)?;
    let mut replacement = Process::spawn(
        &binary,
        &files[0],
        "annotation-owner",
        &root.join("replacement.log"),
    )?;
    ready(&client, &bases[0], TOKEN, &mut replacement).await?;
    for base in &bases[1..=2] {
        if !matches!(failure, Failure::SigkillStaging) {
            let visible: TurnAnnotation = post(&client, base, "read")
                .json(&PointRequest { turn: "U".into() })
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            assert_eq!(visible.status.as_deref(), Some("pending"));
        }
        let current = read(&client, base).await?;
        assert_eq!(current.status.as_deref(), Some(current_status));
        assert_eq!(current.score, Some(current_score));
    }
    if matches!(failure, Failure::SigkillStaging) {
        // Check after both reads: a recovery that published first must not
        // accidentally satisfy a claim about visibility from recovered staging.
        assert_eq!(
            annotation_publication::read(&c, &narrow, None)
                .await?
                .version_id(),
            materialized_version
        );
        assert_eq!(
            annotation_publication::state(&c).await?.progress.checkpoint,
            materialized_cut
        );
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: imported.completion.index
                })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
        a.stop().await?;
        b.stop().await?;
        replacement.stop().await?;
        return c.shutdown().await;
    }
    reclaimed(&c, through).await?;
    assert!(annotation_publication::state(&c).await?.authority.epoch > authority.epoch);
    assert_eq!(
        annotation_publication::read(&c, &narrow, None)
            .await?
            .count_rows(None)
            .await?,
        2
    );
    if matches!(failure, Failure::Sigkill) {
        // Check an exact known result only after independent recovery and reads.
        assert_eq!(
            update(&client, &bases[2], &producer, 3, suffix).await?,
            imported
        );
    }
    let reimport = fixture::encoded(fixture::create(
        &["T", "U", "V"],
        &["wrong", "wrong", "s"],
        &[9, 9, 3],
    )?)?;
    let latest = update(
        &client,
        &bases[2],
        &producer,
        next_sequence,
        reimport.clone(),
    )
    .await?;
    let current = read(&client, &bases[1]).await?;
    assert_eq!(current.status.as_deref(), Some(current_status));
    assert_eq!(current.score, Some(current_score));
    reclaimed(&c, latest.completion.index).await?;
    if matches!(failure, Failure::Sigkill) {
        assert_eq!(
            update(&client, &bases[1], &producer, next_sequence, reimport).await?,
            latest
        );
    }
    let selected = annotation_publication::read(&c, &narrow, None).await?;
    assert_eq!(selected.count_rows(None).await?, 3);
    let reference = BodySnapshot::read(&selected).await?.unwrap();
    let selected_body =
        mixed::load_body(&body, &reference, &binding().incarnation.to_string()).await?;
    let rows = selected_body
        .scan()
        .project(&["id", "binary_payload"])?
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let mut seen = std::collections::BTreeSet::new();
    for row in rows {
        let ids = row
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let payloads = row
            .column(1)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        for i in 0..row.num_rows() {
            let byte = match ids.value(i) {
                "T" => 1,
                "U" => 2,
                "V" => 3,
                id => panic!("unexpected ID {id}"),
            };
            assert!(seen.insert(ids.value(i).to_owned()));
            assert_eq!(payloads.value(i), vec![byte; 16 << 10]);
        }
    }
    assert_eq!(seen.len(), 3);
    a.stop().await?;
    b.stop().await?;
    replacement.stop().await?;
    let before = annotation_publication::state(&c).await?;
    Process::spawn(
        &binary,
        &files[0],
        "annotation-bootstrap",
        &root.join("bootstrap-after-publication.log"),
    )?
    .wait_success()
    .await?;
    assert_eq!(annotation_publication::state(&c).await?, before);
    c.shutdown().await
}
