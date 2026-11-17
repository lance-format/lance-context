use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use prost::Message;
use std::sync::atomic::Ordering;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_body_compaction_lost_prepare_retains_concurrent_create_and_patch() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = artifact_root()?;
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    let (last, last_result) = fragmented(&c, &narrow, &body).await?;
    let original_ids = ids(&selected_body(&c, &narrow, &body).await?).await?;
    let base = annotation_publication::read(&c, &narrow, None).await?;
    let proxy = Proxy::open(&c, root.path()).await?;
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    let (_stop, task) = annotation_runtime::run_owner(runtime);
    tokio::time::timeout(Duration::from_secs(30), proxy.committed.notified()).await?;
    let held = a.get_turn("A", Duration::from_secs(10)).await?.unwrap();
    let p = producer(&a, 2).await?;
    let q = producer(&a, 3).await?;
    let update = request(
        &a,
        &p,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    let create = request(&a, &q, 1, encoded(create(&["D"], &["s"], &[4])?)?).await?;
    let update_result = a.submit(&update, Duration::from_secs(10)).await?;
    let create_result = a.submit(&create, Duration::from_secs(10)).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "D", "s", 4, "pending", None).await?;
    let retained = annotation_publication::state(&c).await?;
    let pending = retained.progress.pending.clone().unwrap();
    let prepared = retained.progress.prepared.clone().unwrap();
    assert!(pending.plan.is_maintenance());
    assert_eq!(prepared.descriptors.format, 4);
    assert_eq!(retained.manifests.as_ref().unwrap().head, base.version_id());
    let expected = Transaction::try_from(lance_table::format::pb::Transaction::decode(
        prepared.descriptors.bytes.clone(),
    )?)?;
    let reference: BodySnapshot = serde_json::from_str(
        &expected.transaction_properties.as_ref().unwrap()[BODY_SNAPSHOT_PROPERTY],
    )?;
    let compacted = mixed::load_body(&body, &reference, &binding().incarnation.to_string()).await?;
    assert_eq!(compacted.fragments().len(), 1);
    let original_body_transaction = compacted.read_transaction().await?.unwrap();
    assert_eq!(original_body_transaction.uuid, expected.uuid);
    assert_eq!(ids(&compacted).await?, original_ids);
    let through = retained.progress.checkpoint[&10];
    assert!(update_result.index > through && create_result.index > through);

    proxy.release.notify_one();
    // The proxy blocks all voter observations for the unchanged 30-second RPC deadline.
    let error = tokio::time::timeout(Duration::from_secs(45), task)
        .await??
        .unwrap_err();
    proxy.restore_observations();
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    let message = format!("{error:#}");
    assert!(message.contains("outcome unknown"), "{message}");
    let stopped = annotation_publication::state(&c).await?;
    assert_eq!(stopped.progress.prepared.as_ref(), Some(&prepared));
    assert_eq!(stopped.progress.checkpoint[&10], through);
    for result in [&update_result, &create_result] {
        assert!(matches!(
            direct
                .request_routed(Request::Fetch {
                    group: 10,
                    index: result.index
                })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
    }
    assert_eq!(held.annotation.status.as_deref(), Some("pending"));
    assert_eq!(
        held.body()
            .column_by_name("binary_payload")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap()
            .value(0),
        vec![1; 16 << 10]
    );
    drop((held, a));
    c.client = direct;
    drop(proxy);
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    assert_eq!(
        annotation_publication::state(&c).await?.progress.prepared,
        Some(prepared)
    );
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "D", "s", 4, "pending", None).await?;
    annotation_runtime::drained(&c, &a, create_result.index).await?;
    let original =
        annotation_publication::read(&c, &narrow, Some(pending.plan.target_version()?)).await?;
    assert_eq!(original.read_transaction().await?, Some(expected));
    assert_eq!(
        BodySnapshot::read(&original).await?,
        Some(reference.clone())
    );
    let original_body =
        mixed::load_body(&body, &reference, &binding().incarnation.to_string()).await?;
    assert_eq!(
        original_body.read_transaction().await?,
        Some(original_body_transaction)
    );
    assert_eq!(a.submit(&last, Duration::from_secs(10)).await?, last_result);
    for (request, result) in [(&update, &update_result), (&create, &create_result)] {
        assert_eq!(&a.submit(request, Duration::from_secs(10)).await?, result);
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: result.index
                })
                .await?
                .reply,
            Reply::Record(None)
        ));
    }
    let current_body = selected_body(&c, &narrow, &body).await?;
    let current_ids = ids(&current_body).await?;
    assert_eq!(current_ids.len(), 4);
    for (id, stable) in original_ids {
        assert_eq!(current_ids[&id], stable);
    }
    assert_eq!(a.retained_body_frames(), (0, 0));
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "B", "s", 2, "pending", None).await?;
    check(&a, "C", "s", 3, "pending", None).await?;
    check(&a, "D", "s", 4, "pending", None).await?;
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
