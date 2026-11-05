use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
enum Case {
    Committed,
    Timeout,
    Absent,
    DifferentClosure,
    Fenced,
}

async fn exercise(case: Case) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("closures.lance");
    let mut cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let execute = matches!(case, Case::Committed | Case::Timeout | Case::Fenced);
    let proxy = Proxy::close_fault(
        &cluster,
        root.path(),
        execute,
        if matches!(case, Case::Timeout) {
            Duration::from_secs(2)
        } else {
            Duration::from_secs(30)
        },
    )
    .await?;
    let direct = std::mem::replace(&mut cluster.client, proxy.client.clone());
    let ingress = connect(&cluster, binding()).await?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(32 << 20);
    let mut publisher = publisher(&ingress, &storage, &budget, 1).await?;
    let open = ProducerOpen {
        bucket: 0,
        slot: 0,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    append(&ingress, &open, 1, &["turn"]).await?;
    publisher.prepare_next().await?.unwrap();
    let pending = publisher.state().await?.progress.pending.unwrap();
    publisher.commit_pending().await?.unwrap();
    let group = *pending.plan.sources.keys().next().unwrap();
    let query::Value::ClosePermit(Some(permit)) = crate::materializer::query(
        &ingress,
        publisher.authority().group,
        query::Query::ClosePermit {
            table: publisher.authority().table,
            transaction: pending.plan.transaction,
            plan_index: pending.index,
            source_group: group,
        },
    )
    .await?
    else {
        bail!("missing test closure permit");
    };
    let mut permit = *permit;
    if matches!(case, Case::DifferentClosure) {
        // Persist a real closure, then intercept a different request before it
        // executes. Its unknown outcome cannot be proven by the earlier receipt.
        assert!(matches!(
            crate::materializer::commit(&direct, Command::Close(permit.clone())).await?,
            Applied::Source(_)
        ));
        let mut value = serde_json::to_value(&permit)?;
        value["closure"]["published_index"] = (permit.closure().published_index + 1).into();
        permit = serde_json::from_value(value)?;
    }
    let expected = permit.closure().clone();
    let authority = publisher.authority().clone();
    let operation = tokio::spawn(async move { publisher.close_source(permit).await });
    tokio::time::timeout(Duration::from_secs(10), proxy.committed.notified()).await?;
    if matches!(case, Case::Fenced) {
        cluster.client = direct.clone();
        let ingress = connect(&cluster, binding()).await?;
        let successor = super::publisher(&ingress, &storage, &budget, authority.epoch).await?;
        assert_ne!(successor.authority(), &authority);
    }
    if !matches!(case, Case::Timeout) {
        proxy.release.notify_one();
    }
    let result = tokio::time::timeout(Duration::from_secs(10), operation).await??;
    assert_eq!(
        result.is_ok(),
        matches!(case, Case::Committed | Case::Timeout),
        "{result:?}"
    );
    if let Ok(proof) = result {
        assert_eq!(proof.closed().permit.closure(), &expected);
    }
    assert_eq!(proxy.closes.load(Ordering::SeqCst), 1, "Close resubmitted");
    // Let the held proxy task exit even when the real client deadline expired.
    proxy.release.notify_one();
    cluster.client = direct;
    drop(proxy);
    cluster.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_close_with_lost_reply_is_observed_without_resubmission() -> Result<()> {
    exercise(Case::Committed).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_close_after_client_timeout_is_observed_without_resubmission() -> Result<()> {
    exercise(Case::Timeout).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn absent_unknown_close_remains_unknown() -> Result<()> {
    exercise(Case::Absent).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_close_requires_the_exact_closure() -> Result<()> {
    exercise(Case::DifferentClosure).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_close_cannot_bypass_owner_fencing() -> Result<()> {
    exercise(Case::Fenced).await
}
