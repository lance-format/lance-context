use super::*;
use crate::tests::{annotation_runtime::lost_prepare::proxy::Proxy, mixed_annotations as fixture};

async fn frontend(
    c: &network::Cluster,
    uri: &str,
    narrow: &Storage,
    body: &Storage,
    root: &std::path::Path,
) -> Result<Server> {
    frontend_with_binding(c, uri, narrow, body, root, binding()).await
}

async fn frontend_with_binding(
    c: &network::Cluster,
    uri: &str,
    narrow: &Storage,
    body: &Storage,
    root: &std::path::Path,
    binding: TableBinding,
) -> Result<Server> {
    frontend_with_settings(
        c,
        uri,
        narrow,
        body,
        root,
        FrontendSettings {
            binding,
            limits: fixture::limits(),
            http: options(),
            ingress_bytes: 8 << 20,
        },
    )
    .await
}

struct FrontendSettings {
    binding: TableBinding,
    limits: Limits,
    http: http::Options,
    ingress_bytes: usize,
}

async fn frontend_with_settings(
    c: &network::Cluster,
    uri: &str,
    narrow: &Storage,
    body: &Storage,
    root: &std::path::Path,
    settings: FrontendSettings,
) -> Result<Server> {
    use crate::application::{self, config};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let token = root.join(format!("token-{}", address.port()));
    let receipts = root.join(format!("receipts-{}.json", address.port()));
    std::fs::write(&token, TOKEN)?;
    std::fs::write(
        &receipts,
        serde_json::to_vec(&serde_json::json!({"active":1,"keys":{"1":vec![17u8;32]}}))?,
    )?;
    let mut config = c.application.clone();
    config.ingress_memory_bytes = settings.ingress_bytes;
    config.materializer_memory_bytes = 8 << 20;
    config.tables.push(config::Table {
        binding: settings.binding,
        uri: narrow.uri.clone(),
        storage_options: narrow.options.clone(),
        bootstrap: crate::bootstrap::Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
    });
    config.http = Some(config::Http {
        listen: address,
        bearer_token_file: token,
        receipt_key_file: receipts,
        options: settings.http,
    });
    config.annotations = Some(config::Annotations {
        shard: None,
        routes: vec![],
        wal_clusters: BTreeMap::new(),
        owner_url: uri.into(),
        apply_memory_bytes: 8 << 20,
        staging: settings.limits,
        source: None,
        projection: None,
        body: Some(config::Body {
            uri: body.uri.clone(),
            storage_options: body.options.clone(),
        }),
        exclusive_reclamation: None,
    });
    let config: application::Config = serde_json::from_slice(&serde_json::to_vec(&config)?)?;
    config.validate()?;
    let mut invalid = config.clone();
    invalid
        .annotations
        .as_mut()
        .unwrap()
        .body
        .as_mut()
        .unwrap()
        .uri = narrow.uri.clone();
    assert!(invalid.validate().is_err());
    // The existing bootstrap role must not create a patch-only binding when
    // configured for an already provisioned pair.
    assert!(application::annotations::provision(&config).await.is_err());
    let (stop, stopped) = oneshot::channel();
    let task = tokio::spawn(async move {
        application::annotations::serve_ingress(&config, listener, async {
            let _ = stopped.await;
        })
        .await
        .map_err(std::io::Error::other)
    });
    Ok(Server {
        address,
        stop,
        task,
    })
}

async fn submit(
    client: &reqwest::Client,
    server: &Server,
    p: &http::Opened,
    sequence: u64,
    body: &Bytes,
) -> Result<MutationReply> {
    Ok(update(client, server, p, sequence, body.len())
        .body(body.clone())
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_http_two_ingresses_read_acked_staging_and_recover_unknown_publication() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        tempfile::Builder::new()
            .prefix("mixed-http-")
            .disable_cleanup(true)
            .tempdir_in(parent)?
    } else {
        tempfile::tempdir()?
    };
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = fixture::storage(root.path(), "annotations.lance").await?;
    let body = fixture::storage(root.path(), "body.lance").await?;
    fixture::seed(&c, &narrow, &body).await?;
    let initial_head = annotation_publication::state(&c)
        .await?
        .manifests
        .as_ref()
        .unwrap()
        .head;
    let proxy = Proxy::open(&c, root.path()).await?;
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let (service, runtime) = fixture::owner(&c, &narrow, &body).await?;
    let owner = Owner::new(&runtime, keys()?, options())?;
    let (_stop, task) = annotation_runtime::run_owner(runtime);
    let server = Server::start(owner.router(), None).await?;
    let address = server.address;
    let a = frontend(&c, &server.uri(), &narrow, &body, root.path()).await?;
    let b = frontend(&c, &server.uri(), &narrow, &body, root.path()).await?;
    let client = reqwest::Client::builder()
        .retry(reqwest::retry::never())
        .timeout(Duration::from_secs(30))
        .build()?;
    let p = open(&client, &a, 1).await?;
    let q = open(&client, &b, 2).await?;
    let first = fixture::encoded(fixture::create(&["T"], &["s"], &[1])?)?;
    let created = submit(&client, &a, &p, 1, &first).await?;
    assert_eq!(
        created.completion.meta,
        Batch::new(
            binding().table,
            0,
            producer::producer_id(1, p.generation),
            1,
            first.clone()
        )
        .meta
    );
    tokio::time::timeout(Duration::from_secs(30), proxy.committed.notified()).await?;
    assert_eq!(
        annotation_publication::state(&c)
            .await?
            .manifests
            .as_ref()
            .unwrap()
            .head,
        initial_head
    );
    assert_eq!(
        annotation_publication::read(&c, &narrow, None)
            .await?
            .count_rows(None)
            .await?,
        0
    );
    let visible = point(&client, &b).await?.unwrap();
    assert_eq!(visible.status.as_deref(), Some("pending"));
    assert_eq!(visible.score, None);
    let patch = fixture::encoded(annotations::mixed::wire::patches(
        &fixture::patch(&["T"], Some("reviewed"), Some(0.9))?,
        &fixture::limits(),
    )?)?;
    let patched = submit(&client, &b, &q, 1, &patch).await?;
    let visible = point(&client, &a).await?.unwrap();
    assert_eq!(visible.status.as_deref(), Some("reviewed"));
    assert_eq!(visible.score, Some(0.9));
    let reimport = fixture::encoded(fixture::create(&["T", "U"], &["wrong", "s"], &[9, 2])?)?;
    let imported = submit(&client, &a, &p, 2, &reimport).await?;
    assert_eq!(point(&client, &b).await?, Some(visible.clone()));
    fixture::check(&service, "T", "s", 1, "reviewed", Some(0.9)).await?;
    fixture::check(&service, "U", "s", 2, "pending", None).await?;
    for reply in [&created, &patched, &imported] {
        assert!(matches!(
            direct
                .request_routed(Request::Fetch {
                    group: 10,
                    index: reply.completion.index
                })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
    }
    // Drop exactly the already committed Prepare response, after both ingress
    // paths have returned later successes. Reuse the durable native pair.
    proxy.release.notify_one();
    let error = tokio::time::timeout(Duration::from_secs(30), task)
        .await??
        .unwrap_err();
    assert!(format!("{error:#}").contains("outcome unknown"));
    c.client = direct;
    let (new_service, runtime) = fixture::owner(&c, &narrow, &body).await?;
    let new_owner = Owner::new(&runtime, keys()?, options())?;
    let (new_stop, new_task) = annotation_runtime::run_owner(runtime);
    for ingress in [&a, &b] {
        let response = post(&client, ingress, "read")
            .json(&PointRequest { turn: "T".into() })
            .send()
            .await?;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let _ = response.bytes().await?;
    }
    owner.shutdown().await;
    server.shutdown().await?;
    drop((service, proxy));
    let server = Server::start(new_owner.router(), Some(address)).await?;
    assert_eq!(point(&client, &a).await?, Some(visible.clone()));
    assert_eq!(point(&client, &b).await?, Some(visible));
    assert_eq!(submit(&client, &b, &p, 2, &reimport).await?, imported);
    assert_eq!(submit(&client, &a, &q, 1, &patch).await?, patched);
    annotation_runtime::drained(&c, &new_service, imported.completion.index).await?;
    assert_eq!(new_service.retained_body_frames(), (0, 0));
    fixture::check(&new_service, "T", "s", 1, "reviewed", Some(0.9)).await?;
    fixture::check(&new_service, "U", "s", 2, "pending", None).await?;
    assert_eq!(
        annotation_publication::read(&c, &narrow, None)
            .await?
            .count_rows(None)
            .await?,
        2
    );
    for reply in [&created, &patched, &imported] {
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: reply.completion.index
                })
                .await?
                .reply,
            Reply::Record(None)
        ));
    }
    a.shutdown().await?;
    b.shutdown().await?;
    new_owner.shutdown().await;
    server.shutdown().await?;
    new_stop.send(()).unwrap();
    new_task.await??;
    c.shutdown().await
}
