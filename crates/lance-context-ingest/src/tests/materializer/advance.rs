use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use lance_context_wal::replication::publication as p;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
enum Case {
    Committed,
    Absent,
    DifferentPublication,
    DifferentClosures,
    Fenced,
    AbsentFenced,
    PersistentUnknown,
    Unobservable,
    PendingPublicationChanged,
    PendingClosuresChanged,
}

async fn exercise(case: Case) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("advances.lance");
    let mut cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let proxy = Proxy::open_advance(
        &cluster,
        root.path(),
        matches!(case, Case::Committed | Case::Fenced | Case::Unobservable),
    )
    .await?;
    if matches!(case, Case::AbsentFenced | Case::PersistentUnknown) {
        proxy
            .pause_uncommitted_advance
            .store(true, Ordering::SeqCst);
    }
    if matches!(case, Case::Unobservable) {
        proxy
            .block_advance_observations
            .store(true, Ordering::SeqCst);
    }
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
    let published = publisher.commit_pending().await?.unwrap();
    let mut proofs = BTreeMap::new();
    for group in pending.plan.sources.keys() {
        let query::Value::ClosePermit(Some(permit)) = crate::materializer::query(
            &ingress,
            publisher.authority().group,
            query::Query::ClosePermit {
                table: publisher.authority().table,
                transaction: pending.plan.transaction,
                plan_index: pending.index,
                source_group: *group,
            },
        )
        .await?
        else {
            bail!("missing closure permit")
        };
        assert!(matches!(
            crate::materializer::commit(&direct, Command::Close(*permit)).await?,
            Applied::Source(lance_context_wal::replication::source::Reply::Closed { .. })
        ));
        let query::Value::ClosedProof(Some(proof)) = crate::materializer::query(
            &ingress,
            *group,
            query::Query::ClosedProof {
                table: publisher.authority().table,
            },
        )
        .await?
        else {
            bail!("missing closed proof")
        };
        proofs.insert(*group, *proof);
    }
    let mut request = p::AdvancePublication {
        authority: publisher.authority().clone(),
        published_index: published.index,
        proofs,
    };
    if matches!(case, Case::DifferentPublication | Case::DifferentClosures) {
        assert!(matches!(
            crate::materializer::commit(&direct, Command::Advance(request.clone())).await?,
            Applied::Publisher(p::Reply::Advanced(_))
        ));
        // An earlier completion cannot prove a different request completed.
        request.proofs.clear();
        if matches!(case, Case::DifferentPublication) {
            request.published_index += 1;
        }
    }
    match case {
        Case::PendingPublicationChanged => request.published_index += 1,
        Case::PendingClosuresChanged => request.proofs.clear(),
        _ => {}
    }
    let original = request.clone();
    let authority = publisher.authority().clone();
    let operation = tokio::spawn(async move { publisher.record_advance(request).await });
    if matches!(
        case,
        Case::Committed
            | Case::Fenced
            | Case::Unobservable
            | Case::AbsentFenced
            | Case::PersistentUnknown
    ) {
        tokio::time::timeout(Duration::from_secs(10), proxy.committed.notified()).await?;
        if matches!(case, Case::Fenced | Case::AbsentFenced) {
            cluster.client = direct.clone();
            let ingress = connect(&cluster, binding()).await?;
            let successor = super::publisher(&ingress, &storage, &budget, authority.epoch).await?;
            assert_ne!(successor.authority(), &authority);
        }
        if matches!(case, Case::PersistentUnknown) {
            proxy.unknown_advances.store(true, Ordering::SeqCst);
        }
        proxy.release.notify_one();
    }
    let result = tokio::time::timeout(Duration::from_secs(45), operation).await??;
    if matches!(case, Case::Unobservable) {
        proxy.restore_observations();
    }
    assert_eq!(
        result.is_ok(),
        matches!(case, Case::Committed | Case::Absent),
        "{result:?}"
    );
    if let Ok(receipt) = result {
        assert_eq!(receipt.publication, published);
        assert_eq!(receipt.sources[&10], pending.plan.sources[&10].through);
    }
    assert_eq!(
        proxy.dropped.load(Ordering::SeqCst),
        if matches!(case, Case::PersistentUnknown) {
            2
        } else {
            1
        }
    );
    assert_eq!(
        proxy.advances.load(Ordering::SeqCst),
        if matches!(case, Case::Absent | Case::PersistentUnknown) {
            2
        } else {
            1
        },
        "unexpected Advance submissions"
    );
    let requests = proxy.advance_requests.lock().unwrap().clone();
    assert_eq!(requests.len(), proxy.advances.load(Ordering::SeqCst));
    assert!(
        requests.iter().all(|r| r == &original),
        "original advancement changed"
    );
    if !matches!(case, Case::Unobservable) {
        assert_eq!(proxy.blocked_reads.load(Ordering::SeqCst), 0);
    }
    let response = direct
        .request_routed(Request::Query {
            group: authority.group,
            query: query::Query::Publisher {
                table: authority.table,
            },
        })
        .await?;
    let Reply::Query(query::Outcome::Observed {
        value: query::Value::Publisher(Some(view)),
        ..
    }) = response.reply
    else {
        bail!("missing test state")
    };
    assert_eq!(
        view.progress.advanced.is_none(),
        matches!(
            case,
            Case::AbsentFenced
                | Case::PersistentUnknown
                | Case::PendingPublicationChanged
                | Case::PendingClosuresChanged
        )
    );
    assert_eq!(
        view.progress.pending.is_some(),
        matches!(
            case,
            Case::AbsentFenced
                | Case::PersistentUnknown
                | Case::PendingPublicationChanged
                | Case::PendingClosuresChanged
        )
    );
    cluster.client = direct;
    drop(proxy);
    cluster.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_unknown_advance_is_observed_without_resubmission() -> Result<()> {
    exercise(Case::Committed).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn absent_unknown_advance_resumes_original_publication() -> Result<()> {
    exercise(Case::Absent).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_advance_requires_same_publication() -> Result<()> {
    exercise(Case::DifferentPublication).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_advance_requires_same_closures() -> Result<()> {
    exercise(Case::DifferentClosures).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_advance_cannot_bypass_owner_fencing() -> Result<()> {
    exercise(Case::Fenced).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unrecorded_advance_is_not_resubmitted_after_owner_fencing() -> Result<()> {
    exercise(Case::AbsentFenced).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeatedly_unknown_advance_stops_after_one_exact_resubmission() -> Result<()> {
    exercise(Case::PersistentUnknown).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unavailable_advance_observation_preserves_failure() -> Result<()> {
    exercise(Case::Unobservable).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_advance_does_not_resubmit_a_different_pending_publication() -> Result<()> {
    exercise(Case::PendingPublicationChanged).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_advance_does_not_resubmit_a_different_pending_closure_set() -> Result<()> {
    exercise(Case::PendingClosuresChanged).await
}
