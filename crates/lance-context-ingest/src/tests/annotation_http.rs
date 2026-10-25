use super::*;
use crate::{
    annotations,
    http::{
        self,
        annotations::{Forwarder, MutationReply, Owner, PointRequest},
    },
    materializer::Storage,
};
use lance_context_api::annotation::TurnAnnotation;
use lance_context_core::annotation_staging::Limits;
use reqwest::StatusCode;
use tokio::sync::oneshot;

const TOKEN: &str = "annotation-test-workload-token-00000000";

mod batches;
mod pressure;

struct Server {
    address: std::net::SocketAddr,
    stop: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}
impl Server {
    async fn start(router: axum::Router, address: Option<std::net::SocketAddr>) -> Result<Self> {
        let listener = tokio::net::TcpListener::bind(
            address.unwrap_or_else(|| "127.0.0.1:0".parse().unwrap()),
        )
        .await?;
        let address = listener.local_addr()?;
        let (stop, stopped) = oneshot::channel();
        let app = router.layer(axum::middleware::from_fn(
            |request: axum::extract::Request, next: axum::middleware::Next| async move {
                use axum::response::IntoResponse;
                if request
                    .headers()
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    != Some(concat!(
                        "Bearer ",
                        "annotation-test-workload-token-00000000"
                    ))
                {
                    return StatusCode::UNAUTHORIZED.into_response();
                }
                next.run(request).await
            },
        ));
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        });
        Ok(Self {
            address,
            stop,
            task,
        })
    }
    fn uri(&self) -> String {
        format!("http://{}", self.address)
    }
    async fn shutdown(self) -> Result<()> {
        let _ = self.stop.send(());
        tokio::time::timeout(Duration::from_secs(30), self.task).await???;
        Ok(())
    }
}

fn keys() -> Result<ReceiptKeys> {
    ReceiptKeys::new(1, [(1, [17; 32])])
}
fn options() -> http::Options {
    http::Options {
        data_requests: 2,
        control_requests: 4,
        body_total_timeout: Duration::from_secs(30),
        ..Default::default()
    }
}
async fn frontend(c: &network::Cluster, limits: &Limits, uri: &str) -> Result<(Forwarder, Server)> {
    let ingress = annotations::connect_ingress(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        limits,
        MergeMemoryBudget::new(8 << 20),
    )
    .await?;
    let gateway = Forwarder::new(
        vec![(ingress, uri.into())],
        keys()?,
        options(),
        TOKEN.into(),
    )?;
    let server = Server::start(gateway.router(), None).await?;
    Ok((gateway, server))
}
fn post(client: &reqwest::Client, server: &Server, operation: &str) -> reqwest::RequestBuilder {
    client
        .post(format!(
            "{}/api/v1/annotations/rollouts/{operation}",
            server.uri()
        ))
        .bearer_auth(TOKEN)
}
async fn open(client: &reqwest::Client, server: &Server, slot: u32) -> Result<http::Opened> {
    Ok(post(client, server, "producer")
        .json(&ProducerOpen {
            bucket: 0,
            slot,
            expected_generation: 0,
            operation: Uuid::from_u128(100 + slot as u128),
        })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}
fn update(
    client: &reqwest::Client,
    server: &Server,
    producer: &http::Opened,
    sequence: u64,
    length: usize,
) -> reqwest::RequestBuilder {
    post(client, server, "update")
        .header(http::PRODUCER, &producer.producer)
        .header(http::SEQUENCE, sequence)
        .header(reqwest::header::CONTENT_TYPE, http::ARROW_FILE)
        .header(reqwest::header::CONTENT_LENGTH, length)
}
async fn point(client: &reqwest::Client, server: &Server) -> Result<Option<TurnAnnotation>> {
    Ok(post(client, server, "read")
        .json(&PointRequest { turn: "T".into() })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

#[tokio::test]
async fn annotation_http_two_ingresses_keep_values_and_exact_retries_through_owner_replacement(
) -> Result<()> {
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
        readers: 16,
    };
    annotation_publication::seed_contract(
        &c,
        &storage,
        annotations::Service::contract(&binding(), &limits)?,
    )
    .await?;
    let (service, runtime) = annotation_runtime::owner(&c, &storage, &limits).await?;
    let owner = Owner::new(
        &runtime,
        keys()?,
        http::Options {
            data_requests: 1,
            ..options()
        },
    )?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let server = Server::start(owner.router(), None).await?;
    let address = server.address;
    let (ga, a) = frontend(&c, &limits, &server.uri()).await?;
    let (gb, b) = frontend(&c, &limits, &server.uri()).await?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let producer = open(&client, &a, 1).await?;
    let producer_b = open(&client, &b, 2).await?;
    let first = super::annotations::wire(&limits, &["T"], 1, Some("reviewed"), 1, Some(0.9))?;
    let reply: MutationReply = update(&client, &a, &producer, 1, first.len())
        .body(first.clone())
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(
        reply.completion.meta,
        Batch::new(binding().table, 0, producer::producer_id(1, 1), 1, first).meta
    );
    let value = point(&client, &b).await?.unwrap();
    assert_eq!(value.status.as_deref(), Some("reviewed"));
    assert_eq!(value.score, Some(0.9));

    // One partial upload occupies the owner's data lane. The other ingress
    // must still read, and an excess upload gets a definite pre-submit refusal.
    let body = super::annotations::wire(&limits, &["T"], 1, Some("after-upload"), 0, None)?;
    let (chunks, receive) = tokio::sync::mpsc::channel::<Bytes>(2);
    chunks.send(body.slice(..body.len() / 2)).await?;
    let stream = futures::stream::unfold(receive, |mut receive| async {
        receive
            .recv()
            .await
            .map(|chunk| (Ok::<_, std::io::Error>(chunk), receive))
    });
    let pending = tokio::spawn(
        update(&client, &a, &producer, 2, body.len())
            .body(reqwest::Body::wrap_stream(stream))
            .send(),
    );
    tokio::time::timeout(Duration::from_secs(10), async {
        while owner.observation().data_inflight != 1 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert_eq!(point(&client, &b).await?.unwrap(), value);
    let refused = update(&client, &b, &producer_b, 1, body.len())
        .body(body.clone())
        .send()
        .await?;
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        refused.json::<serde_json::Value>().await?["operation_outcome"],
        "not_submitted"
    );
    chunks.send(body.slice(body.len() / 2..)).await?;
    drop(chunks);
    let updated: MutationReply = pending.await??.error_for_status()?.json().await?;
    assert_eq!(
        updated.completion.meta,
        Batch::new(
            binding().table,
            0,
            producer::producer_id(1, 1),
            2,
            body.clone()
        )
        .meta
    );
    annotation_runtime::drained(&service, updated.completion.index).await?;
    let current = point(&client, &b).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("after-upload"));
    assert_eq!(current.score, Some(0.9));
    let retry: MutationReply = update(&client, &b, &producer, 2, body.len())
        .body(body.clone())
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(retry, updated);
    let fetched = c
        .client
        .request_routed(Request::Fetch {
            group: 10,
            index: updated.completion.index,
        })
        .await?;
    assert!(matches!(fetched.reply, Reply::Record(None)));

    // Acquire the successor while the old HTTP listener still accepts traffic.
    // Its epoch checks must refuse both routes before replacing the stable URL.
    let (new_service, runtime) = annotation_runtime::owner(&c, &storage, &limits).await?;
    let new_owner = Owner::new(&runtime, keys()?, options())?;
    let (new_stop, new_task) = annotation_runtime::run_owner(runtime);
    for frontend in [&a, &b] {
        let old = post(&client, frontend, "read")
            .json(&PointRequest { turn: "T".into() })
            .send()
            .await?;
        assert_eq!(old.status(), StatusCode::SERVICE_UNAVAILABLE);
        let _ = old.bytes().await?;
    }
    let _ = stop.send(());
    // Fencing may already have stopped the old publisher before shutdown.
    let _ = task.await?;
    owner.shutdown().await;
    server.shutdown().await?;
    let server = Server::start(new_owner.router(), Some(address)).await?;
    assert_eq!(point(&client, &a).await?.unwrap(), current);
    assert_eq!(point(&client, &b).await?.unwrap(), current);
    let retry: MutationReply = update(&client, &a, &producer, 2, body.len())
        .body(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(retry, updated);
    let body = super::annotations::wire(&limits, &["T"], 2, None, 2, None)?;
    let latest: MutationReply = update(&client, &b, &producer, 3, body.len())
        .body(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let cleared = point(&client, &a).await?.unwrap();
    assert_eq!((cleared.status, cleared.score), (None, None));
    assert_eq!(cleared.labels, current.labels);
    annotation_runtime::drained(&new_service, latest.completion.index).await?;
    let closed = post(&client, &a, "close")
        .json(&http::CloseRequest {
            producer: producer.producer,
            receipt: Some(latest.receipt),
        })
        .send()
        .await?;
    assert_eq!(closed.status(), StatusCode::NO_CONTENT);
    let reopened: http::Opened = post(&client, &b, "producer")
        .json(&ProducerOpen {
            bucket: 0,
            slot: 1,
            expected_generation: 1,
            operation: Uuid::new_v4(),
        })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(reopened.generation, 2);
    ga.shutdown().await;
    gb.shutdown().await;
    a.shutdown().await?;
    b.shutdown().await?;
    new_owner.shutdown().await;
    server.shutdown().await?;
    new_stop.send(()).unwrap();
    new_task.await??;
    c.shutdown().await
}
