//! Drop an actual successful RF3 Prepare reply for a paired body/annotation cut.
use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use prost::Message;
use std::sync::atomic::Ordering;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_unknown_prepare_preserves_acked_suffix_and_original_pair() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("mixed-lost-prepare-");
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        builder.disable_cleanup(true).tempdir_in(parent)?
    } else {
        builder.tempdir()?
    };
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    seed(&c, &narrow, &body).await?;
    let base = annotation_publication::read(&c, &narrow, None).await?;
    let proxy = Proxy::open(&c, root.path()).await?;
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let (a, runtime) = owner(&c, &narrow, &body).await?;
    let (_stop, task) = annotation_runtime::run_owner(runtime);
    let p = producer(&a, 1).await?;
    let q = producer(&a, 2).await?;
    let r = producer(&a, 3).await?;
    let s = producer(&a, 4).await?;
    let first = request(&a, &p, 1, encoded(create(&["A"], &["s"], &[1])?)?).await?;
    let first_result = a.submit(&first, Duration::from_secs(30)).await?;
    assert_eq!(first_result.result, service::MutationResult::Applied);
    tokio::time::timeout(Duration::from_secs(30), proxy.committed.notified()).await?;

    // The real quorum has committed the original pair. While the publisher
    // waits for its reply, apply must preserve both later fields and new turns.
    let score = request(
        &a,
        &q,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A"], None, Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    let status = request(
        &a,
        &r,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), None)?,
            &limits(),
        )?)?,
    )
    .await?;
    let (score_result, status_result) = tokio::join!(
        a.submit(&score, Duration::from_secs(30)),
        a.submit(&status, Duration::from_secs(30)),
    );
    let score_result = score_result?;
    let status_result = status_result?;
    let reimport = request(
        &a,
        &s,
        1,
        encoded(create(&["A", "B"], &["wrong", "t"], &[9, 2])?)?,
    )
    .await?;
    let reimport_result = a.submit(&reimport, Duration::from_secs(30)).await?;
    for result in [&score_result, &status_result, &reimport_result] {
        assert_eq!(result.result, service::MutationResult::Applied);
    }
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "B", "t", 2, "pending", None).await?;
    let current = a.get("A", Duration::from_secs(30)).await?.unwrap();
    let retained = annotation_publication::state(&c).await?;
    let pending = retained.progress.pending.clone().unwrap();
    let prepared = retained.progress.prepared.clone().unwrap();
    assert_eq!(prepared.descriptors.format, 4);
    // Decode the already committed descriptors, never prepare another pair.
    let original = Transaction::try_from(lance_table::format::pb::Transaction::decode(
        prepared.descriptors.bytes.clone(),
    )?)?;
    let reference: BodySnapshot = serde_json::from_str(
        original.transaction_properties.as_ref().unwrap()[BODY_SNAPSHOT_PROPERTY].as_str(),
    )?;
    let original_body =
        mixed::load_body(&body, &reference, &binding().incarnation.to_string()).await?;
    let body_transaction = original_body.read_transaction().await?.unwrap();
    assert_eq!(body_transaction.uuid, original.uuid);
    assert_eq!(original_body.count_rows(None).await?, 1);
    assert_eq!(original.uuid, pending.plan.transaction.to_string());
    assert_eq!(retained.manifests.as_ref().unwrap().head, base.version_id());
    assert_eq!(
        pending.plan.sources[&10].records,
        vec![(first_result.index, first_result.meta.clone())]
    );
    assert!(pending.plan.sources[&10].through < score_result.index.min(status_result.index));

    proxy.release.notify_one();
    // The proxy blocks all voter observations for the unchanged 30-second RPC deadline.
    let error = tokio::time::timeout(Duration::from_secs(45), task)
        .await??
        .unwrap_err();
    proxy.restore_observations();
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    let context = format!("{error:#}");
    assert!(
        context.contains("WAL RPC prepare") && context.contains("outcome unknown"),
        "{context}"
    );
    assert!(a.get_turn("A", Duration::from_secs(1)).await.is_err());
    let stopped = annotation_publication::state(&c).await?;
    assert_eq!(stopped.progress.prepared.as_ref(), Some(&prepared));
    assert_eq!(stopped.manifests.as_ref().unwrap().head, base.version_id());
    let requests = [
        (&first, &first_result),
        (&score, &score_result),
        (&status, &status_result),
        (&reimport, &reimport_result),
    ];
    for (_, result) in requests {
        assert!(matches!(
            direct
                .request_routed(Request::Fetch {
                    group: 10,
                    index: result.index,
                })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
    }
    drop(a);
    c.client = direct;
    drop(proxy);
    c.shutdown().await?;

    // Reopen the original RF3 disks and acquire a new owner. Its first visible
    // value includes the suffix; recovery must reuse the exact original pair.
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard())
        .await
        .context("reopen original mixed WAL replicas")?;
    assert_eq!(
        annotation_publication::state(&c).await?.progress.prepared,
        Some(prepared)
    );
    let (a, runtime) = owner(&c, &narrow, &body)
        .await
        .context("resolve the unknown mixed preparation")?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    assert_eq!(a.get("A", Duration::from_secs(30)).await?.unwrap(), current);
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "B", "t", 2, "pending", None).await?;
    annotation_runtime::drained(&c, &a, reimport_result.index).await?;
    assert_eq!(a.retained_body_frames(), (0, 0));
    let selected =
        annotation_publication::read(&c, &narrow, Some(pending.plan.target_version()?)).await?;
    assert_eq!(selected.read_transaction().await?, Some(original));
    assert_eq!(
        BodySnapshot::read(&selected).await?,
        Some(reference.clone())
    );
    let selected_body =
        mixed::load_body(&body, &reference, &binding().incarnation.to_string()).await?;
    assert_eq!(
        selected_body.read_transaction().await?,
        Some(body_transaction)
    );
    for (request, result) in requests {
        assert_eq!(&a.submit(request, Duration::from_secs(30)).await?, result);
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: result.index,
                })
                .await?
                .reply,
            Reply::Record(None)
        ));
    }
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "B", "t", 2, "pending", None).await?;
    let latest = annotation_publication::read(&c, &narrow, None).await?;
    assert_eq!(latest.count_rows(None).await?, 2);
    let latest_reference = BodySnapshot::read(&latest).await?.unwrap();
    let latest_body =
        mixed::load_body(&body, &latest_reference, &binding().incarnation.to_string()).await?;
    assert_eq!(latest_body.count_rows(None).await?, 2);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
