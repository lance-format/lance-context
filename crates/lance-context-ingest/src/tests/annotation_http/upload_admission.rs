//! Memory admission must not block other streams on the owner connection.
use super::*;
use crate::annotations::runtime::{Configuration, Runtime};
use crate::materializer::{Acquisition, Options};
use std::collections::BTreeSet;
use std::sync::Mutex;

fn started(service: &annotations::Service, phase: &str) -> u64 {
    service
        .phase_observation()
        .iter()
        .find(|p| p.phase == phase)
        .unwrap()
        .started
}

async fn wait_started(service: &annotations::Service, phase: &str, count: u64) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(10), async {
        while started(service, phase) < count {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await?;
    Ok(())
}

fn stream(receive: tokio::sync::mpsc::Receiver<Bytes>) -> reqwest::Body {
    reqwest::Body::wrap_stream(futures::stream::unfold(receive, |mut receive| async {
        receive
            .recv()
            .await
            .map(|b| (Ok::<_, std::io::Error>(b), receive))
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn memory_refusal_does_not_wait_for_an_admitted_upload_on_the_same_connection() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits {
        retained_bytes: 8 << 20,
        retained_entries: 128,
        value_bytes: 512 << 10,
        readers: 16,
    };
    let status = "s".repeat(500 << 10);
    let body = super::super::annotations::wire(&limits, &["T"; 8], 1, Some(&status), 1, Some(0.9))?;
    ensure!(body.len() > 3 << 20 && body.len() < 5 << 20);
    let mut binding = binding();
    binding.max_record_bytes = 5 << 20;
    annotation_publication::seed_contract(
        &c,
        &storage,
        annotations::Service::contract(&binding, &limits)?,
    )
    .await?;
    let state = annotation_publication::state(&c).await?;
    let upload_budget = MergeMemoryBudget::new(20 << 20);
    let (service, runtime) = Runtime::connect(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding.clone(),
        storage.clone(),
        Configuration {
            limits: limits.clone(),
            upload_budget: upload_budget.clone(),
            apply_budget: MergeMemoryBudget::new(32 << 20),
            publication_budget: MergeMemoryBudget::new(64 << 20),
            publication: Options {
                max_source_bytes: 8 << 20,
                min_source_bytes: 4 << 20,
                max_records: 2,
                max_batch_wait: Duration::from_millis(20),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
    )
    .await?;
    let owner_options = http::Options {
        admission_timeout: Duration::from_secs(30),
        ..options()
    };
    let owner = Owner::new(&runtime, keys()?, owner_options)?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let connections = Arc::new(Mutex::new(BTreeSet::new()));
    let seen = connections.clone();
    let router = owner.router().layer(axum::middleware::from_fn(
        move |request: axum::extract::Request, next: axum::middleware::Next| {
            let seen = seen.clone();
            async move {
                assert_eq!(request.version(), axum::http::Version::HTTP_2);
                if request.uri().path().ends_with("/update") {
                    let address = request
                        .extensions()
                        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
                        .unwrap()
                        .0;
                    seen.lock().unwrap().insert(address);
                }
                next.run(request).await
            }
        },
    ));
    let owner_server = Server::start(router, None).await?;
    let ingress_budget = MergeMemoryBudget::new(64 << 20);
    let ingress = annotations::connect_ingress(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding,
        &limits,
        ingress_budget.clone(),
    )
    .await?;
    let gateway = Forwarder::new(
        vec![(ingress, owner_server.uri())],
        keys()?,
        http::Options {
            admission_timeout: Duration::from_millis(100),
            ..options()
        },
        TOKEN.into(),
    )?;
    let front = Server::start(gateway.router(), None).await?;
    let client = reqwest::Client::builder()
        // Require stream cancellation on both hops. With HTTP/1, rejecting an
        // unfinished upload can close the connection before its response body
        // reaches the caller; that is an unknown transport outcome, not proof
        // that this HTTP/2 admission check blocked.
        .http2_prior_knowledge()
        .retry(reqwest::retry::never())
        .timeout(Duration::from_secs(30))
        .build()?;
    let a = open(&client, &front, 1).await?;
    let b = open(&client, &front, 2).await?;
    let (a_tx, a_rx) = tokio::sync::mpsc::channel(2);
    a_tx.send(body.slice(..16384)).await?;
    let a_task = tokio::spawn(
        update(&client, &front, &a, 1, body.len())
            .body(stream(a_rx))
            .send(),
    );
    wait_started(&service, "annotation_upload_read", 1).await?;
    ensure!(
        upload_budget.reserved() > 10 << 20,
        "first upload must occupy over half the budget"
    );
    let (b_tx, b_rx) = tokio::sync::mpsc::channel(2);
    b_tx.send(body.clone()).await?;
    drop(b_tx);
    let mut b_task = tokio::spawn(
        update(&client, &front, &b, 1, body.len())
            .body(stream(b_rx))
            .send(),
    );
    wait_started(&service, "annotation_upload_admission", 2).await?;
    assert_eq!(started(&service, "annotation_upload_read"), 1);
    // The refusal must arrive while A still holds its reservation and has no
    // remaining body available. A 30-second admission timer cannot satisfy this.
    let b_response = tokio::time::timeout(Duration::from_secs(5), &mut b_task).await???;
    assert_eq!(b_response.version(), reqwest::Version::HTTP_2);
    assert_eq!(b_response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let refusal: serde_json::Value = b_response.json().await?;
    assert_eq!(refusal["error"], "memory_capacity");
    assert_eq!(refusal["operation_outcome"], "not_submitted");
    assert_eq!(owner.observation().local_write_memory_refusals, 1);
    assert_eq!(gateway.observation().local_write_memory_refusals, 0);
    a_tx.send(body.slice(16384..)).await?;
    drop(a_tx);
    let a_reply: MutationReply = a_task.await??.error_for_status()?.json().await?;
    assert_eq!(
        a_reply.completion.result,
        annotations::MutationResult::Applied
    );
    // Only the explicit not-submitted refusal permits this exact request to
    // be sent again. It must first commit after A, not resolve an earlier write.
    let b_reply: MutationReply = update(&client, &front, &b, 1, body.len())
        .body(body.clone())
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(
        b_reply.completion.result,
        annotations::MutationResult::Applied
    );
    assert!(b_reply.completion.index > a_reply.completion.index);
    let current = point(&client, &front)
        .await?
        .context("acknowledged turn missing")?;
    assert_eq!(current.status.as_deref(), Some(status.as_str()));
    assert_eq!(current.score, Some(0.9));
    let connection_count = connections.lock().unwrap().len();
    assert_eq!(
        connection_count, 1,
        "both updates must share one owner HTTP/2 connection"
    );
    // A later refusal at the ingress must not be attributed to the owner.
    let occupied = ingress_budget.reserve(ingress_budget.limit()).await;
    let refused = update(&client, &front, &b, 2, body.len())
        .body(body.clone())
        .send()
        .await?;
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    let refusal: serde_json::Value = refused.json().await?;
    assert_eq!(refusal["error"], "memory_capacity");
    assert_eq!(refusal["operation_outcome"], "not_submitted");
    assert_eq!(gateway.observation().local_write_memory_refusals, 1);
    assert_eq!(owner.observation().local_write_memory_refusals, 1);
    drop(occupied);
    let admitted: MutationReply = update(&client, &front, &b, 2, body.len())
        .body(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(
        admitted.completion.result,
        annotations::MutationResult::Applied
    );
    assert!(admitted.completion.index > b_reply.completion.index);
    assert_eq!(gateway.observation().local_write_memory_refusals, 1);
    assert_eq!(owner.observation().local_write_memory_refusals, 1);
    assert_eq!(gateway.observation().local_status_memory_refusals, 0);
    assert_eq!(owner.observation().local_status_memory_refusals, 0);
    gateway.shutdown().await;
    front.shutdown().await?;
    owner.shutdown().await;
    owner_server.shutdown().await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
