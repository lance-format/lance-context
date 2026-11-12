use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use std::sync::atomic::Ordering;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_status_index_unknown_prepare_resolves_with_maintenance_disabled() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), lost_prepare(false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_status_index_empty_retirement_recovers_with_maintenance_disabled() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), lost_prepare(true)).await?
}

async fn leave_empty_definition(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
) -> Result<()> {
    let (_a, runtime) = owner_with_indexes(c, narrow, body, 64, false, true, false).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let base = annotation_publication::read(c, narrow, None).await?;
            if base.load_indices().await?.iter().any(|i| i.name == NAME)
                && annotation_publication::state(c)
                    .await?
                    .progress
                    .pending
                    .is_none()
            {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await??;
    stop.send(()).unwrap();
    task.await??;

    let (a, runtime) = owner_with_indexes(c, narrow, body, 64, false, false, false).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let patcher = producer(&a, 3).await?;
    let input = request(
        &a,
        &patcher,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A", "B", "C"], Some("reviewed"), Some(0.1))?,
            &limits(),
        )?)?,
    )
    .await?;
    let result = a.submit(&input, Duration::from_secs(10)).await?;
    assert_eq!(result.result, service::MutationResult::Applied);
    annotation_runtime::drained(c, &a, result.index).await?;
    stop.send(()).unwrap();
    task.await??;
    let base = annotation_publication::read(c, narrow, None).await?;
    let indices = base.load_indices().await?;
    let status = indices
        .iter()
        .filter(|i| i.name == NAME)
        .collect::<Vec<_>>();
    assert_eq!(status.len(), 1);
    assert!(status[0].fragment_bitmap.as_ref().unwrap().is_empty());
    Ok(())
}

async fn lost_prepare(retire_empty: bool) -> Result<()> {
    let root = artifact_root()?;
    let mut c =
        network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    let (previous, previous_result) =
        compaction::fragmented_with_slot(&c, &narrow, &body, 1).await?;
    if retire_empty {
        leave_empty_definition(&c, &narrow, &body).await?;
    }
    let base = annotation_publication::read(&c, &narrow, None).await?;
    let proxy = Proxy::open(&c, root.path()).await?;
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let (a, runtime) = owner_with_indexes(&c, &narrow, &body, 64, false, true, false).await?;
    let (_stop, task) = annotation_runtime::run_owner(runtime);
    tokio::time::timeout(Duration::from_secs(30), proxy.committed.notified()).await?;
    let patcher = producer(&a, 2).await?;
    let update = request(
        &a,
        &patcher,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    let result = a.submit(&update, Duration::from_secs(10)).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    let state = annotation_publication::state(&c).await?;
    let pending = state.progress.pending.clone().unwrap();
    let prepared = state.progress.prepared.clone().unwrap();
    assert!(pending.plan.is_maintenance());
    assert_eq!(prepared.descriptors.format, 3);
    let expected = crate::materializer::stage::annotation_compaction_transaction(
        &base,
        &pending,
        &prepared.descriptors,
    )
    .await?;
    assert!(
        matches!(&expected.operation, Operation::CreateIndex {new_indices, ..} if new_indices.len() == 1 && new_indices[0].name == NAME)
    );
    if retire_empty {
        let Operation::CreateIndex {
            removed_indices, ..
        } = &expected.operation
        else {
            unreachable!();
        };
        assert_eq!(removed_indices.len(), 1);
        assert!(removed_indices[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .is_empty());
        // A removal must not let a stale build pass the normal snapshot guard.
        use prost::Message;
        let mut change: serde_json::Value = serde_json::from_slice(&prepared.descriptors.bytes)?;
        let bytes: Vec<u8> = serde_json::from_value(change["index"].clone())?;
        let mut index = lance_table::format::pb::IndexMetadata::decode(bytes.as_slice())?;
        index.dataset_version -= 1;
        change["index"] = serde_json::to_value(index.encode_to_vec())?;
        let invalid = p::Descriptors::lance_index(Bytes::from(serde_json::to_vec(&change)?))?;
        let error = crate::materializer::stage::annotation_compaction_transaction(
            &base, &pending, &invalid,
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("invalid empty status index replacement"));
    }
    assert!(result.index > state.progress.checkpoint[&10]);
    proxy.release.notify_one();
    // The proxy blocks all voter observations for the unchanged 30-second RPC deadline.
    let error = tokio::time::timeout(Duration::from_secs(45), task)
        .await??
        .unwrap_err();
    proxy.restore_observations();
    assert!(format!("{error:#}").contains("outcome unknown"));
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    drop(a);
    c.client = direct;
    drop(proxy);
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    assert_eq!(
        annotation_publication::state(&c).await?.progress.prepared,
        Some(prepared)
    );
    // Recovery must interpret the retained descriptor, even when no new index
    // work is allowed on the replacement owner.
    let (a, runtime) = owner_with_indexes(&c, &narrow, &body, 64, false, false, false).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    annotation_runtime::drained(&c, &a, result.index).await?;
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    assert_eq!(a.submit(&update, Duration::from_secs(10)).await?, result);
    assert_eq!(
        a.submit(&previous, Duration::from_secs(10)).await?,
        previous_result
    );
    let original =
        annotation_publication::read(&c, &narrow, Some(pending.plan.target_version()?)).await?;
    assert_eq!(original.read_transaction().await?, Some(expected));
    assert_eq!(original.manifest().fragments, base.manifest().fragments);
    let current = annotation_publication::read(&c, &narrow, None).await?;
    for predicate in [
        "status = 'reviewed'",
        "status = 'pending'",
        "status IS NULL",
    ] {
        assert_eq!(
            ids(&current, predicate, true).await?,
            ids(&current, predicate, false).await?
        );
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
