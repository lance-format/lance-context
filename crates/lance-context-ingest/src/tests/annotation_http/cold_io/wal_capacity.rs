//! A quorum capacity rejection must remain distinguishable from response loss.
use super::*;

#[tokio::test]
async fn wal_capacity_rejects_new_updates_but_preserves_exact_results_and_reads() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed_with_record_limit(
        root.path(),
        annotation_publication::one_shard(),
        1,
    )
    .await?;
    let gates = Arc::new(store::Gates::default());
    let storage = store::storage(&root.path().join("turns.lance"), gates.clone()).await?;
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
    let owner = Owner::new(&runtime, keys()?, options())?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let server = Server::start(owner.router(), None).await?;
    let (ga, a) = frontend(&c, &limits, &server.uri()).await?;
    let (gb, b) = frontend(&c, &limits, &server.uri()).await?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let producer = open(&client, &a, 1).await?;
    let mut publication = gates.pause_put();
    let first =
        super::super::super::annotations::wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?;
    let accepted: MutationReply = update(&client, &a, &producer, 1, first.len())
        .body(first.clone())
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    publication.entered().await?;

    // Keep the accepted record live by blocking its native publication. The
    // next sequence reaches a real replicated capacity decision, not an HTTP
    // concurrency or upload-memory refusal.
    let second = super::super::super::annotations::wire(&limits, &["T"], 0, None, 1, Some(0.9))?;
    let response = update(&client, &b, &producer, 2, second.len())
        .body(second.clone())
        .send()
        .await?;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let rejection: serde_json::Value = response.json().await?;
    assert_eq!(rejection["error"], "wal_capacity");
    assert_eq!(rejection["operation_outcome"], "rejected");
    let retry: MutationReply = update(&client, &b, &producer, 1, first.len())
        .body(first)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(retry.completion, accepted.completion);
    let current = point(&client, &a).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, None);

    publication.release();
    annotation_runtime::drained(&c, &service, accepted.completion.index).await?;
    // Wait for the existing source-closure protocol, not just local staging
    // retirement, before deliberately resubmitting the rejected original bytes.
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let reply = c
                .client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: accepted.completion.index,
                })
                .await?;
            if matches!(reply.reply, Reply::Record(None)) {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await??;
    let next: MutationReply = update(&client, &a, &producer, 2, second.len())
        .body(second)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(next.completion.result, annotations::MutationResult::Applied);
    assert!(next.completion.index > accepted.completion.index);
    let current = point(&client, &b).await?.unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));
    annotation_runtime::drained(&c, &service, next.completion.index).await?;

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
