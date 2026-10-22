use super::*;
use crate::{
    application::{self, config, Registry},
    materializer::Storage,
    preparation::Pool,
};
use std::net::SocketAddr;
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};

struct Server {
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<Result<()>>>,
}
impl Drop for Server {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }
}
impl Server {
    async fn start(config: application::Config, address: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(address).await?;
        let registry = Registry::connect(&config).await?;
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            application::serve_preparer(&registry, &config, listener, async {
                let _ = stopped.await;
            })
            .await
        });
        Ok(Self {
            stop: Some(stop),
            task: Some(task),
        })
    }
    async fn finish(mut self) -> Result<()> {
        let _ = self.stop.take().unwrap().send(());
        tokio::time::timeout(Duration::from_secs(20), self.task.take().unwrap()).await???;
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn independent_preparers_require_every_partition_and_recover_the_durable_aggregate(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("distributed.lance");
    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let ingress = connect(&cluster, binding()).await?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let mut expected = BTreeMap::new();
    for bucket in 0..4 {
        let producer = ingress
            .open_producer(ProducerOpen {
                bucket,
                slot: bucket,
                expected_generation: 0,
                operation: Uuid::new_v4(),
            })
            .await?;
        let ids = [format!("{bucket}-first"), format!("{bucket}-second")];
        let session = session(bucket);
        let body = ipc_file(&[batch(&[&ids[0], &ids[1]], &[&session, &session])?])?;
        let prepared = ingress
            .prepare(
                ingress.reserve_upload(body.len()).await?,
                body.clone(),
                &producer,
                1,
            )
            .await?;
        ingress.append(&prepared).await?;
        materializer::remember(&mut expected, body)?;
    }
    let token = "independent-preparer-authorization-test-0001";
    let token_file = root.path().join("preparer.token");
    std::fs::write(&token_file, token)?;
    let mut config = cluster.application.clone();
    config.tables.push(config::Table {
        binding: binding(),
        uri: path.to_str().unwrap().into(),
        storage_options: Default::default(),
        bootstrap: bootstrap::identity(),
    });
    config.worker.publication.pipeline_preparation = true;
    config.preparation = Some(config::Preparation {
        bearer_token_file: token_file,
        active: 2,
        retained_results: 4,
        ..Default::default()
    });
    let first_port = std::net::TcpListener::bind("127.0.0.1:0")?;
    let second_port = std::net::TcpListener::bind("127.0.0.1:0")?;
    let first_address = first_port.local_addr()?;
    let second_address = second_port.local_addr()?;
    drop(first_port);
    let first_server = Server::start(config.clone(), first_address).await?;
    let endpoints = [
        format!("http://{first_address}"),
        format!("http://{second_address}"),
    ];
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let health = format!("{}/api/v1/health", endpoints[0]);
    assert_eq!(client.get(&health).send().await?.status(), 401);
    assert_eq!(
        client
            .post(format!("{}{}", endpoints[0], crate::preparation::ROUTE))
            .bearer_auth(token)
            .body(vec![0u8; 4097])
            .send()
            .await?
            .status(),
        400
    );
    drop(second_port); // The second real endpoint is unavailable for the first round.
    let pool = Pool::new(&endpoints, token.to_owned(), Duration::from_secs(10))?;
    let budget = MergeMemoryBudget::new(32 << 20);
    let mut publisher = materializer::publisher(&ingress, &storage, &budget, 1).await?;
    publisher.use_preparers(pool);
    assert!(publisher.prepare_next().await.is_err());
    let pending = publisher
        .state()
        .await?
        .progress
        .pending
        .context("missing durable plan")?;
    assert!(
        publisher.state().await?.progress.prepared.is_none(),
        "partial round was prepared"
    );
    assert_eq!(
        publisher.read_dataset(None).await?.count_rows(None).await?,
        0
    );
    assert_eq!(budget.reserved(), 0);
    let first: serde_json::Value = client
        .get(&health)
        .bearer_auth(token)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(
        first["completed"], 1,
        "healthy sibling did not finish its files"
    );
    assert_eq!(first["active"], 0);
    let second_server = Server::start(config.clone(), second_address).await?;
    let saved = publisher
        .prepare_next()
        .await?
        .context("complete round not prepared")?;
    assert_eq!(saved.plan_index, pending.index);
    assert_eq!(saved.transaction, pending.plan.transaction);
    for endpoint in &endpoints {
        let health: serde_json::Value = client
            .get(format!("{endpoint}/api/v1/health"))
            .bearer_auth(token)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        assert_eq!(
            health["completed"], 1,
            "retry rewrote the healthy partition"
        );
        assert_eq!(health["reserved_bytes"], 0);
        let fetch = health["phases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["phase"] == "fetch")
            .unwrap();
        assert_eq!(
            fetch["completed"], 2,
            "preparer did not fetch exactly its assigned half"
        );
    }
    assert_eq!(publisher.read_dataset(None).await?.version().version, 1);
    assert_eq!(
        publisher.read_dataset(None).await?.count_rows(None).await?,
        0
    );
    first_server.finish().await?;
    second_server.finish().await?;
    drop((publisher, ingress));
    cluster.shutdown().await?;

    // Lose both volatile preparer caches and reopen the actual WAL directories.
    // The complete aggregate is durable; recovery must not need remote writers.
    let cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&cluster, binding()).await?;
    let mut publisher = materializer::publisher(&ingress, &storage, &budget, 2).await?;
    assert_eq!(publisher.prepare_next().await?.unwrap(), saved);
    publisher
        .commit_pending()
        .await?
        .context("publication missing")?;
    publisher
        .finish_pending()
        .await?
        .context("source advancement missing")?;
    materializer::verify(&publisher, &expected, 2).await?;
    assert!(publisher.run_once().await?.is_none());
    assert_eq!(budget.reserved(), 0);
    cluster.shutdown().await?;
    Ok(())
}
