use super::*;
use crate::application::{self, config, Config};
use crate::bootstrap::Identity;

fn configuration(c: &network::Cluster, physical: &Physical, owner_url: String) -> Config {
    let mut config = c.application.clone();
    config.ingress_memory_bytes = 8 << 20;
    config.materializer_memory_bytes = 8 << 20;
    config.tables = vec![config::Table {
        binding: physical.binding.clone(),
        uri: physical.narrow.uri.clone(),
        storage_options: physical.narrow.options.clone(),
        bootstrap: Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
    }];
    config.annotations = Some(config::Annotations {
        owner_url,
        shard: Some(physical.shard),
        routes: vec![],
        wal_clusters: BTreeMap::new(),
        apply_memory_bytes: 8 << 20,
        staging: fixture::limits(),
        source: None,
        projection: None,
        body: Some(config::Body {
            uri: physical.body.uri.clone(),
            storage_options: physical.body.options.clone(),
        }),
        exclusive_reclamation: None,
    });
    config
}

pub(super) async fn provision(
    c: &network::Cluster,
    root: &std::path::Path,
    bucket: u32,
) -> Result<Physical> {
    let mut binding = binding();
    binding.table = Uuid::new_v4();
    binding.incarnation = Uuid::new_v4();
    binding.destination.dataset = format!("configured-{bucket}.lance");
    let narrow = fixture::storage(root, &binding.destination.dataset).await?;
    let body = fixture::storage(root, &format!("configured-body-{bucket}.lance")).await?;
    let physical = Physical {
        binding,
        narrow,
        body,
        shard: annotations::Shard { bucket },
    };
    let config = configuration(c, &physical, "http://unused-owner:8080".into());
    let config: Config = serde_json::from_slice(&serde_json::to_vec(&config)?)?;
    config.validate()?;
    application::annotations::provision(&config).await?;
    // Explicit repeated bootstrap reuses the same persisted table identities.
    application::annotations::provision(&config).await?;
    Ok(physical)
}

pub(super) async fn frontend(
    c: &network::Cluster,
    root: &std::path::Path,
    routes: &[(&Physical, &Server)],
) -> Result<Server> {
    let mut config = configuration(c, routes[0].0, routes[0].1.uri());
    config.annotations.as_mut().unwrap().routes = routes
        .iter()
        .map(|(physical, owner)| config::AnnotationRoute {
            binding: physical.binding.clone(),
            shard: physical.shard,
            owner_url: owner.uri(),
            wal_cluster: None,
        })
        .collect();
    serve(config, root).await
}

pub(super) async fn clustered_frontend(
    root: &std::path::Path,
    routes: [(&network::Cluster, &Physical, &Server); 2],
) -> Result<Server> {
    let (local, physical, owner) = routes[0];
    let mut config = configuration(local, physical, owner.uri());
    let remote = &routes[1].0.application;
    let connection = config::WalConnection {
        layout: remote.layout.clone(),
        limits: remote.limits.clone(),
        endpoints: remote.endpoints.clone(),
        tls: remote.tls.clone(),
        rpc: remote.rpc.clone(),
    };
    let annotations = config.annotations.as_mut().unwrap();
    annotations.wal_clusters.insert("remote".into(), connection);
    annotations.routes = routes
        .iter()
        .enumerate()
        .map(|(i, (_, physical, owner))| config::AnnotationRoute {
            binding: physical.binding.clone(),
            shard: physical.shard,
            owner_url: owner.uri(),
            wal_cluster: (i == 1).then(|| "remote".into()),
        })
        .collect();
    config.validate()?;
    let mut unknown = config.clone();
    unknown.annotations.as_mut().unwrap().routes[1].wal_cluster = Some("missing".into());
    assert!(unknown.validate().is_err());
    let mut mismatch = config.clone();
    mismatch
        .annotations
        .as_mut()
        .unwrap()
        .wal_clusters
        .get_mut("remote")
        .unwrap()
        .layout
        .buckets
        .extend_from_within(..);
    assert!(mismatch.validate().is_err());
    let mut duplicate = config.clone();
    duplicate
        .annotations
        .as_mut()
        .unwrap()
        .wal_clusters
        .get_mut("remote")
        .unwrap()
        .layout
        .cluster = config.layout.cluster;
    assert!(duplicate.validate().is_err());
    let mut local = config.clone();
    local.annotations.as_mut().unwrap().routes[0].wal_cluster = Some("remote".into());
    assert!(local.validate().is_err());
    let mut unused = config.clone();
    unused.annotations.as_mut().unwrap().routes[1].wal_cluster = None;
    assert!(unused.validate().is_err());
    serve(config, root).await
}

async fn serve(mut config: Config, root: &std::path::Path) -> Result<Server> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let token = root.join(format!("shard-token-{}", address.port()));
    let receipts = root.join(format!("shard-receipts-{}.json", address.port()));
    std::fs::write(&token, TOKEN)?;
    std::fs::write(
        &receipts,
        serde_json::to_vec(&serde_json::json!({"active":1,"keys":{"1":vec![17u8;32]}}))?,
    )?;
    if let Some(wal) = config
        .annotations
        .as_mut()
        .unwrap()
        .wal_clusters
        .get_mut("remote")
    {
        // Exercise relative TLS paths by actually connecting the remote cluster
        // through a serialized application config, not just inspecting strings.
        for (i, path) in [
            &mut wal.tls.certificate,
            &mut wal.tls.private_key,
            &mut wal.tls.roots,
        ]
        .into_iter()
        .enumerate()
        {
            let name = format!("remote-tls-{}-{i}.pem", address.port());
            std::fs::copy(&*path, root.join(&name))?;
            *path = name.into();
        }
    }
    config.http = Some(config::Http {
        listen: address,
        bearer_token_file: token,
        receipt_key_file: receipts,
        options: options(),
    });
    let file = root.join(format!("shard-ingress-{}.json", address.port()));
    std::fs::write(&file, serde_json::to_vec(&config)?)?;
    let config = Config::read(&file)?;
    let mut missing = config.clone();
    missing.annotations.as_mut().unwrap().routes.pop();
    assert!(
        missing.validate().is_err(),
        "incomplete map must fail before connecting"
    );
    let mut foreign = config.clone();
    foreign.annotations.as_mut().unwrap().routes[0]
        .binding
        .table = Uuid::new_v4();
    assert!(
        foreign.validate().is_err(),
        "local binding cannot silently change physical tables"
    );
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
