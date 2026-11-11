use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
enum Case {
    Committed,
    Timeout,
    Absent,
    PreviousClosure,
    PersistentUnknown,
    DifferentClosure,
    Fenced,
    AbsentFenced,
}

async fn exercise(case: Case) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("closures.lance");
    let mut cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let budget = MergeMemoryBudget::new(32 << 20);
    let open = ProducerOpen {
        bucket: 0,
        slot: 0,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    let previous = matches!(case, Case::PreviousClosure);
    let epoch = if previous {
        let ingress = connect(&cluster, binding()).await?;
        let mut first = publisher(&ingress, &storage, &budget, 1).await?;
        append(&ingress, &open, 1, &["prefix"]).await?;
        first
            .run_once()
            .await?
            .context("first publication missing")?;
        assert!(first.state().await?.progress.pending.is_none());
        first.authority().epoch
    } else {
        1
    };
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
    let mut publisher = publisher(&ingress, &storage, &budget, epoch).await?;
    append(&ingress, &open, if previous { 2 } else { 1 }, &["turn"]).await?;
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
            Applied::Source(lance_context_wal::replication::source::Reply::Closed { .. })
        ));
        let mut value = serde_json::to_value(&permit)?;
        value["closure"]["published_index"] = (permit.closure().published_index + 1).into();
        permit = serde_json::from_value(value)?;
    }
    let expected = permit.closure().clone();
    let authority = publisher.authority().clone();
    let operation = tokio::spawn(async move { publisher.close_source(permit).await });
    tokio::time::timeout(Duration::from_secs(10), proxy.committed.notified()).await?;
    if matches!(case, Case::Fenced | Case::AbsentFenced) {
        cluster.client = direct.clone();
        let ingress = connect(&cluster, binding()).await?;
        let successor = super::publisher(&ingress, &storage, &budget, authority.epoch).await?;
        assert_ne!(successor.authority(), &authority);
    }
    if matches!(case, Case::PersistentUnknown) {
        proxy.unknown_closes.store(true, Ordering::SeqCst);
    }
    if !matches!(case, Case::Timeout) {
        proxy.release.notify_one();
    }
    let result = tokio::time::timeout(Duration::from_secs(10), operation).await??;
    assert_eq!(
        result.is_ok(),
        matches!(
            case,
            Case::Committed | Case::Timeout | Case::Absent | Case::PreviousClosure
        ),
        "{result:?}"
    );
    match result {
        Ok(proof) => assert_eq!(proof.closed().permit.closure(), &expected),
        Err(error) => {
            let reason = match case {
                Case::PersistentUnknown => "source closure proof unavailable",
                Case::DifferentClosure => "foreign source closure proof",
                Case::Fenced | Case::AbsentFenced => {
                    "publisher fenced or table incarnation changed"
                }
                _ => unreachable!(),
            };
            assert!(format!("{error:#}").contains(reason), "{error:#}");
        }
    }
    let resumed = matches!(
        case,
        Case::Absent | Case::PreviousClosure | Case::PersistentUnknown
    );
    assert_eq!(
        proxy.closes.load(Ordering::SeqCst),
        if resumed { 2 } else { 1 }
    );
    let permits = proxy.close_permits.lock().unwrap().clone();
    assert_eq!(permits.len(), if resumed { 2 } else { 1 });
    assert!(
        permits.iter().all(|p| p == &permits[0]),
        "original permit changed"
    );
    assert_eq!(permits[0].closure(), &expected);
    assert_eq!(
        proxy.dropped.load(Ordering::SeqCst),
        usize::from(!matches!(case, Case::Timeout))
            + usize::from(matches!(case, Case::PersistentUnknown)),
        "timeout case must finish while the original reply is still held"
    );
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
async fn absent_unknown_close_resumes_the_exact_authorized_command() -> Result<()> {
    exercise(Case::Absent).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn previous_closure_does_not_stop_the_next_authorized_publication() -> Result<()> {
    exercise(Case::PreviousClosure).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeatedly_unknown_close_stops_after_one_exact_resubmission() -> Result<()> {
    exercise(Case::PersistentUnknown).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_close_requires_the_exact_closure() -> Result<()> {
    exercise(Case::DifferentClosure).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_close_cannot_bypass_owner_fencing() -> Result<()> {
    exercise(Case::Fenced).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn absent_unknown_close_is_not_resubmitted_after_owner_fencing() -> Result<()> {
    exercise(Case::AbsentFenced).await
}
