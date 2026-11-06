use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use lance_context_wal::replication::publication as p;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
enum Case {
    Committed,
    Absent,
    DifferentContent,
    DifferentPlan,
    DifferentTransaction,
    Fenced,
    Unobservable,
}

async fn exercise(case: Case) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("preparations.lance");
    let mut cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let execute = matches!(case, Case::Committed | Case::Fenced | Case::Unobservable);
    let proxy = Proxy::open_prepare(
        &cluster,
        root.path(),
        execute,
        !matches!(case, Case::Unobservable),
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
    let publisher = publisher(&ingress, &storage, &budget, 1).await?;
    let before = publisher.state().await?;
    let plan = p::Plan {
        transaction: Uuid::new_v4(),
        read_version: before.manifests.as_ref().unwrap().head,
        sources: BTreeMap::new(),
        next_group: 10,
    };
    publisher.record_plan(plan.clone()).await?;
    let pending = publisher.state().await?.progress.pending.clone().unwrap();
    // These cases exercise catalog identity matching, not native file publication.
    // The annotation runtime regression separately publishes actual Lance files.
    let descriptors = p::Descriptors::lance_compaction(Bytes::from_static(b"[]"))?;
    let mut request = p::PrepareRequest {
        authority: publisher.authority().clone(),
        transaction: plan.transaction,
        plan_index: pending.index,
        descriptors: descriptors.clone(),
    };
    if matches!(
        case,
        Case::DifferentContent | Case::DifferentPlan | Case::DifferentTransaction
    ) {
        let response = direct
            .request_routed(Request::Commit(Box::new(Command::Prepare(request.clone()))))
            .await?;
        assert!(matches!(
            response.reply,
            Reply::Commit(Outcome::Committed(Applied::Publisher(
                p::Reply::Prepared { .. }
            )))
        ));
        match case {
            Case::DifferentContent => {
                request.descriptors = p::Descriptors::lance_compaction(Bytes::from_static(b"[ ]"))?
            }
            Case::DifferentPlan => request.plan_index += 1,
            Case::DifferentTransaction => request.transaction = Uuid::new_v4(),
            _ => unreachable!(),
        }
    }
    let authority = publisher.authority().clone();
    let operation = tokio::spawn(async move { publisher.record_preparation(request).await });
    if execute {
        tokio::time::timeout(Duration::from_secs(10), proxy.committed.notified()).await?;
        if matches!(case, Case::Fenced) {
            cluster.client = direct.clone();
            let ingress = connect(&cluster, binding()).await?;
            let successor = super::publisher(&ingress, &storage, &budget, authority.epoch).await?;
            assert_ne!(successor.authority(), &authority);
        }
        proxy.release.notify_one();
    }
    let result = tokio::time::timeout(Duration::from_secs(10), operation).await??;
    assert_eq!(
        result.is_ok(),
        matches!(case, Case::Committed),
        "{result:?}"
    );
    assert_eq!(
        proxy.prepares.load(Ordering::SeqCst),
        1,
        "Prepare was resubmitted"
    );
    assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(
        proxy.blocked_reads.load(Ordering::SeqCst),
        usize::from(matches!(case, Case::Unobservable))
    );
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
        bail!("missing test state");
    };
    assert_eq!(view.manifests.as_ref().unwrap().head, plan.read_version);
    if matches!(case, Case::Absent) {
        assert!(view.progress.prepared.is_none());
    } else {
        let prepared = view.progress.prepared.as_ref().unwrap();
        assert_eq!(prepared.transaction, plan.transaction);
        assert_eq!(prepared.plan_index, pending.index);
        assert_eq!(prepared.descriptors, descriptors);
    }
    cluster.client = direct;
    drop(proxy);
    cluster.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_unknown_preparation_is_observed_without_resubmission() -> Result<()> {
    exercise(Case::Committed).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn absent_unknown_preparation_remains_unknown() -> Result<()> {
    exercise(Case::Absent).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_preparation_requires_exact_descriptors() -> Result<()> {
    exercise(Case::DifferentContent).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_preparation_requires_original_plan_index() -> Result<()> {
    exercise(Case::DifferentPlan).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_preparation_requires_original_transaction() -> Result<()> {
    exercise(Case::DifferentTransaction).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_preparation_cannot_bypass_owner_fencing() -> Result<()> {
    exercise(Case::Fenced).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unavailable_preparation_observation_preserves_failure() -> Result<()> {
    exercise(Case::Unobservable).await
}
