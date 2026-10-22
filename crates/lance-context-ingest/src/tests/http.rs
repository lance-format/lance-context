use super::*;
use crate::http::{
    self,
    journal::{Config, DurableClient},
    Accepted, Gateway, Opened,
};
use axum::{
    body::to_bytes,
    extract::{Request, State},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use lance_context_wal::client_journal::Settings;
use std::{
    collections::HashMap,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
};

#[derive(Default)]
struct LostAck {
    lost: AtomicBool,
    accepted: Mutex<Option<Accepted>>,
    producer: Mutex<Option<String>>,
}
async fn lose_one_ack(State(state): State<Arc<LostAck>>, request: Request, next: Next) -> Response {
    let append = request.uri().path().ends_with("/append");
    let token = request
        .headers()
        .get(http::PRODUCER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let response = next.run(request).await;
    if append
        && response.status() == axum::http::StatusCode::ACCEPTED
        && !state.lost.swap(true, Ordering::AcqRel)
    {
        let bytes = to_bytes(response.into_body(), 16 << 10).await.unwrap();
        *state.accepted.lock().unwrap() = Some(serde_json::from_slice(&bytes).unwrap());
        *state.producer.lock().unwrap() = token;
        // The real gateway has durably accepted the append, but the client sees
        // an intermediary failure and must retain its journaled original body.
        return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    response
}
struct HttpServer {
    gateway: Gateway,
    address: std::net::SocketAddr,
    stop: oneshot::Sender<()>,
    task: JoinHandle<std::io::Result<()>>,
}
impl HttpServer {
    async fn start(
        ingress: Ingress,
        lost: Option<Arc<LostAck>>,
        options: http::Options,
    ) -> Result<Self> {
        let gateway = Gateway::new(
            BTreeMap::from([("rollouts".into(), ingress)]),
            ReceiptKeys::new(1, [(1, [0x42; 32])])?,
            options,
        )?;
        let mut app = gateway.router();
        if let Some(lost) = lost {
            app = app.layer(middleware::from_fn_with_state(lost, lose_one_ack));
        }
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let (stop, shutdown) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown.await;
                })
                .await
        });
        Ok(Self {
            gateway,
            address,
            stop,
            task,
        })
    }
    fn url(&self) -> reqwest::Url {
        format!("http://{}/", self.address).parse().unwrap()
    }
    fn endpoint(&self, action: &str) -> String {
        format!("{}api/v1/ingest/rollouts/{action}", self.url())
    }
    async fn shutdown(self) -> Result<()> {
        self.gateway.shutdown().await;
        let _ = self.stop.send(());
        self.task.await??;
        Ok(())
    }
}
fn transport() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(15))
        .build()?)
}
fn opens() -> Vec<ProducerOpen> {
    (0..2)
        .map(|bucket| ProducerOpen {
            bucket,
            slot: bucket,
            expected_generation: 0,
            operation: Uuid::from_u128(100 + bucket as u128),
        })
        .collect()
}
async fn writer(
    root: &Path,
    server: &HttpServer,
    budget: Arc<MergeMemoryBudget>,
) -> Result<DurableClient> {
    DurableClient::open(
        Config {
            directory: root.join("journal"),
            opens: opens(),
            base: server.url(),
            request_timeout: Duration::from_secs(15),
            journal: Settings {
                state_bytes: 16 << 20,
                queue_bytes: 8 << 20,
                batch_bytes: 2 << 20,
                file_bytes: 64 << 10,
                log_bytes: 96 << 20,
                ..Settings::default()
            },
        },
        &layout(),
        binding(),
        transport()?,
        budget,
    )
    .await
}
fn data(bucket: u32, round: usize) -> Result<Bytes> {
    let session = session(bucket);
    ipc_file(&[batch(
        &[
            &format!("{bucket}-{round}-a"),
            &format!("{bucket}-{round}-b"),
        ],
        &[&session, &session],
    )?])
}
async fn storage(path: &Path) -> Result<crate::materializer::Storage> {
    let base = Dataset::open(path.to_str().unwrap()).await?;
    let object_base = base
        .manifest_location()
        .path
        .as_ref()
        .rsplit_once("/_versions/")
        .context("manifest base")?
        .0
        .to_owned();
    Ok(crate::materializer::Storage {
        uri: path.to_str().unwrap().into(),
        base: object_store::path::Path::parse(object_base)?,
        options: HashMap::new(),
        session: Arc::new(lance::session::Session::default()),
    })
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lost_http_ack_recovers_shared_journal_across_gateway_and_wal_restart_then_publishes_lance(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("rollouts.lance");
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&c, &path, &binding()).await?;
    let storage = storage(&path).await?;
    let ingress = connect(&c, binding()).await?;
    let lost = Arc::new(LostAck::default());
    let server = HttpServer::start(
        ingress.clone(),
        Some(lost.clone()),
        http::Options::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(64 << 20);
    let client = writer(root.path(), &server, budget.clone()).await?;
    let mut expected = BTreeMap::new();
    let first = data(0, 1)?;
    materializer::remember(&mut expected, first.clone())?;
    assert!(client.append(0, first.clone()).await.is_err());
    let accepted = lost
        .accepted
        .lock()
        .unwrap()
        .clone()
        .context("failure was not injected after durable acceptance")?;
    assert!(client.last_accepted(0).await?.is_none());
    assert_eq!(client.stage(0, first.clone()).await?, 1);
    assert_eq!(client.stage(0, Bytes::copy_from_slice(&first)).await?, 1);
    assert!(
        client.append(0, data(0, 99)?).await.is_err(),
        "unresolved intent was replaced"
    );
    let second = data(1, 1)?;
    materializer::remember(&mut expected, second.clone())?;
    let known = client.append(1, second).await?;
    let producer = lost.producer.lock().unwrap().clone().unwrap();
    let keys = ReceiptKeys::new(1, [(1, [0x42; 32])])?;
    assert!(ingress.restore_receipt(&keys, &producer).is_err());
    assert!(ingress.restore_producer(&keys, &accepted.receipt).is_err());
    let mut altered = producer.clone().into_bytes();
    altered[20] = if altered[20] == b'A' { b'B' } else { b'A' };
    assert!(ingress
        .restore_producer(&keys, std::str::from_utf8(&altered)?)
        .is_err());
    client.shutdown().await?;
    assert_eq!(budget.reserved(), 0);
    server.shutdown().await?;
    drop(ingress);
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&c, binding()).await?;
    let server = HttpServer::start(ingress.clone(), None, http::Options::default()).await?;
    let client = writer(root.path(), &server, budget.clone()).await?;
    assert_eq!(client.last_accepted(1).await?, Some(known.clone()));
    let retried = client
        .resume(0)
        .await?
        .context("lost ACK intent was not recovered")?;
    assert_eq!(
        retried, accepted,
        "gateway failover changed the original ACK"
    );
    assert!(client.resume(0).await?.is_none());
    let materializer_budget = MergeMemoryBudget::new(32 << 20);
    let mut publisher =
        materializer::publisher(&ingress, &storage, &materializer_budget, 1).await?;
    let prepared = publisher
        .prepare_next()
        .await?
        .context("missing staged files")?;
    // Restart the application endpoints and all original WAL directories with
    // a real prepared Lance result but before selecting its manifest.
    drop(publisher);
    client.shutdown().await?;
    server.shutdown().await?;
    drop(ingress);
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&c, binding()).await?;
    let server = HttpServer::start(ingress.clone(), None, http::Options::default()).await?;
    let client = writer(root.path(), &server, budget.clone()).await?;
    assert_eq!(client.last_accepted(0).await?, Some(accepted.clone()));
    let mut publisher =
        materializer::publisher(&ingress, &storage, &materializer_budget, 2).await?;
    assert_eq!(publisher.prepare_next().await?.unwrap(), prepared);
    assert!(publisher.run_once().await?.is_some());
    materializer::verify(&publisher, &expected, 2).await?;
    for ack in [&accepted, &known] {
        let response: http::Visibility = transport()?
            .post(server.endpoint("publication"))
            .json(&http::ReceiptRequest {
                receipt: ack.receipt.clone(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        assert_eq!(response.minimum_read_version, Some(2));
    }
    let a = data(0, 2)?;
    let b = data(1, 2)?;
    materializer::remember(&mut expected, a.clone())?;
    materializer::remember(&mut expected, b.clone())?;
    let (a, b) = tokio::try_join!(client.append(0, a), client.append(1, b))?;
    assert_eq!((a.sequence, b.sequence), (2, 2));
    assert!(client.statistics().requests >= 4);
    assert!(publisher.run_once().await?.is_some());
    materializer::verify(&publisher, &expected, 3).await?;
    client.shutdown().await?;
    let client = writer(root.path(), &server, budget.clone()).await?;
    assert_eq!(client.last_accepted(0).await?, Some(a));
    assert_eq!(client.last_accepted(1).await?, Some(b));
    assert!(client.resume(0).await?.is_none());
    assert!(publisher.run_once().await?.is_none());
    materializer::verify(&publisher, &expected, 3).await?;
    client.shutdown().await?;
    server.shutdown().await?;
    drop((publisher, ingress));
    c.shutdown().await?;
    assert_eq!(budget.reserved(), 0);
    Ok(())
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn saturated_http_data_lane_keeps_control_available_and_drain_joins_disconnected_body(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&c, &root.path().join("rollouts.lance"), &binding()).await?;
    let ingress = connect(&c, binding()).await?;
    let server = HttpServer::start(
        ingress.clone(),
        None,
        http::Options {
            data_requests: 1,
            body_idle_timeout: Duration::from_secs(60),
            body_total_timeout: Duration::from_secs(60),
            ..http::Options::default()
        },
    )
    .await?;
    let client = transport()?;
    let opened: Opened = client
        .post(server.endpoint("producer"))
        .json(&opens()[0])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let mut slow = TcpStream::connect(server.address).await?;
    slow.write_all(format!("POST /api/v1/ingest/rollouts/append HTTP/1.1\r\nHost: localhost\r\nContent-Type: {}\r\nContent-Length: 4096\r\nx-lance-producer: {}\r\nx-lance-sequence: 1\r\n\r\nx", http::ARROW_FILE, opened.producer).as_bytes()).await?;
    tokio::time::timeout(Duration::from_secs(5), async {
        while server.gateway.observation().data_inflight != 1 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let overloaded = client
        .post(server.endpoint("append"))
        .header("content-type", http::ARROW_FILE)
        .header(http::PRODUCER, &opened.producer)
        .header(http::SEQUENCE, "1")
        .body(data(0, 1)?)
        .send()
        .await?;
    assert_eq!(
        overloaded.status(),
        reqwest::StatusCode::SERVICE_UNAVAILABLE
    );
    let opened_again: Opened = client
        .post(server.endpoint("producer"))
        .json(&opens()[0])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(opened_again.producer, opened.producer);
    let gateway = server.gateway.clone();
    let drain = tokio::spawn(async move { gateway.shutdown().await });
    tokio::time::timeout(Duration::from_secs(5), async {
        while !server.gateway.observation().draining {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    assert!(
        !drain.is_finished(),
        "shutdown released a still-admitted body reader"
    );
    slow.shutdown().await?;
    let mut discarded = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), slow.read_to_end(&mut discarded)).await;
    tokio::time::timeout(Duration::from_secs(5), drain).await??;
    assert_eq!(server.gateway.observation().data_inflight, 0);
    server.shutdown().await?;
    let server = HttpServer::start(ingress.clone(), None, http::Options::default()).await?;
    let accepted: Accepted = client
        .post(server.endpoint("append"))
        .header("content-type", http::ARROW_FILE)
        .header(http::PRODUCER, &opened.producer)
        .header(http::SEQUENCE, "1")
        .body(data(0, 1)?)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(
        accepted.sequence, 1,
        "partial body advanced the producer sequence"
    );
    server.shutdown().await?;
    drop(ingress);
    c.shutdown().await
}
