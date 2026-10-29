//! Real owner transport: early refusal and a lost committed response.
use super::*;
use http_body_util::BodyExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forwarding_preserves_large_refusals_and_does_not_replay_lost_replies() -> Result<()> {
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
        retained_bytes: 2 << 20,
        retained_entries: 128,
        value_bytes: 256 << 10,
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
    let requests = Arc::new(AtomicUsize::new(0));
    let lose_reply = Arc::new(AtomicBool::new(false));
    let lost = Arc::new(std::sync::Mutex::new(None::<MutationReply>));
    let (observed, fault, captured) = (requests.clone(), lose_reply.clone(), lost.clone());
    let router = owner.router().layer(axum::middleware::from_fn(
        move |request: axum::extract::Request, next: axum::middleware::Next| {
            let (observed, fault, captured) = (observed.clone(), fault.clone(), captured.clone());
            async move {
                assert_eq!(request.version(), axum::http::Version::HTTP_2);
                observed.fetch_add(1, Ordering::SeqCst);
                let update = request.uri().path().ends_with("/update");
                let mut response = next.run(request).await;
                if response.status() == StatusCode::SERVICE_UNAVAILABLE {
                    // Use a distinct owner delay to detect a lost or replaced
                    // header at the forwarding ingress.
                    response
                        .headers_mut()
                        .insert(reqwest::header::RETRY_AFTER, "3".parse().unwrap());
                }
                if update
                    && response.status() == StatusCode::OK
                    && fault.swap(false, Ordering::SeqCst)
                {
                    let (mut parts, body) = response.into_parts();
                    let bytes = body.collect().await.unwrap().to_bytes();
                    *captured.lock().unwrap() = Some(serde_json::from_slice(&bytes).unwrap());
                    parts.headers.remove(axum::http::header::CONTENT_LENGTH);
                    let failed = futures::stream::once(async {
                        Err::<Bytes, _>(std::io::Error::other("test lost committed reply"))
                    });
                    return axum::response::Response::from_parts(
                        parts,
                        axum::body::Body::from_stream(failed),
                    );
                }
                response
            }
        },
    ));
    let server = Server::start(router, None).await?;
    let (ga, a) = frontend(&c, &limits, &server.uri()).await?;
    let (gb, b) = frontend(&c, &limits, &server.uri()).await?;
    let client = reqwest::Client::builder()
        .http1_only()
        .timeout(Duration::from_secs(30))
        .build()?;
    let p = open(&client, &a, 1).await?;
    let q = open(&client, &b, 2).await?;
    let body = super::super::annotations::wire(&limits, &["T"], 1, Some("held"), 0, None)?;
    let (chunks, receive) = tokio::sync::mpsc::channel::<Bytes>(2);
    chunks.send(body.slice(..body.len() / 2)).await?;
    let frames = futures::stream::unfold(receive, |mut receive| async {
        receive
            .recv()
            .await
            .map(|b| (Ok::<_, std::io::Error>(b), receive))
    });
    let pending = tokio::spawn(
        update(&client, &a, &p, 1, body.len())
            .body(reqwest::Body::wrap_stream(frames))
            .send(),
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while owner.observation().data_inflight != 1 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await?;
    let status = "s".repeat(128 << 10);
    let large = super::super::annotations::wire(&limits, &["T"], 1, Some(&status), 1, Some(0.9))?;
    assert!(large.len() > 128 << 10);
    for _ in 0..64 {
        let response = update(&client, &b, &q, 1, large.len())
            .body(large.clone())
            .send()
            .await?;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[reqwest::header::RETRY_AFTER], "3");
        let reply: serde_json::Value = response.json().await?;
        assert_eq!(reply["operation_outcome"], "not_submitted");
        assert_eq!(reply["error"], "request_capacity");
    }
    chunks.send(body.slice(body.len() / 2..)).await?;
    drop(chunks);
    let held: MutationReply = pending.await??.error_for_status()?.json().await?;
    annotation_runtime::drained(&c, &service, held.completion.index).await?;
    assert_eq!(
        point(&client, &b).await?.unwrap().status.as_deref(),
        Some("held")
    );

    // The previously refused identity now commits, but loses its response.
    // The ingress must report unknown and send exactly one owner request.
    let before = requests.load(Ordering::SeqCst);
    lose_reply.store(true, Ordering::SeqCst);
    let response = update(&client, &b, &q, 1, large.len())
        .body(large.clone())
        .send()
        .await?;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response.json::<serde_json::Value>().await?["operation_outcome"],
        "unknown"
    );
    assert_eq!(requests.load(Ordering::SeqCst), before + 1);
    let current = point(&client, &a).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some(status.as_str()));
    assert_eq!(current.score, Some(0.9));
    let expected = lost
        .lock()
        .unwrap()
        .clone()
        .context("missing committed response")?;
    assert_eq!(
        expected.completion.result,
        annotations::MutationResult::Applied
    );
    let retry: MutationReply = update(&client, &a, &q, 1, large.len())
        .body(large)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(retry, expected);
    ga.shutdown().await;
    gb.shutdown().await;
    a.shutdown().await?;
    b.shutdown().await?;
    owner.shutdown().await;
    server.shutdown().await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
