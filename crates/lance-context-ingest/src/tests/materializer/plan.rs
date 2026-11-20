use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use lance_context_wal::replication::publication as p;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
enum Case {
    Committed,
    Absent,
    DelayedCommit,
    DifferentContent,
    Fenced,
    FencedRetry,
    ChangedPredecessor,
    Cancelled,
}

async fn exercise(case: Case, rpc_unknown: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("plans.lance");
    let mut cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let proxy = Proxy::open_plan(
        &cluster,
        root.path(),
        matches!(case, Case::Committed | Case::Fenced),
    )
    .await?;
    proxy.rpc_unknown_plans.store(rpc_unknown, Ordering::SeqCst);
    proxy
        .unknown_plans
        .store(matches!(case, Case::Cancelled), Ordering::SeqCst);
    proxy.pause_plan_retry.store(
        matches!(case, Case::DelayedCommit | Case::FencedRetry),
        Ordering::SeqCst,
    );
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
    let view = publisher.state().await?;
    let mut plan = p::Plan {
        transaction: Uuid::new_v4(),
        read_version: view.manifests.as_ref().unwrap().head,
        sources: BTreeMap::new(),
        next_group: 10,
    };
    if matches!(case, Case::DifferentContent) {
        submit_plan(&direct, publisher.authority(), &plan).await?;
        // Same UUID, different version: UUID equality alone must not confirm it.
        plan.read_version += 1;
    }
    if matches!(case, Case::ChangedPredecessor) {
        plan.read_version += 1;
    }
    let attempted = plan.clone();
    let original_authority = publisher.authority().clone();
    let operation = tokio::spawn(async move { publisher.record_plan(attempted).await });
    let mut delayed_index = None;
    if matches!(
        case,
        Case::Committed | Case::Fenced | Case::DelayedCommit | Case::FencedRetry
    ) {
        tokio::time::timeout(Duration::from_secs(10), proxy.committed.notified()).await?;
        if matches!(case, Case::Fenced | Case::FencedRetry) {
            cluster.client = direct.clone();
            let ingress = connect(&cluster, binding()).await?;
            let successor =
                super::publisher(&ingress, &storage, &budget, original_authority.epoch).await?;
            assert_ne!(successor.authority(), &original_authority);
        }
        if matches!(case, Case::DelayedCommit) {
            // Commit the original request after the quorum read saw no plan,
            // while its exact retry is waiting to reach the actual WAL.
            delayed_index = Some(submit_plan(&direct, &original_authority, &plan).await?);
        }
        proxy.release.notify_one();
    }
    if matches!(case, Case::Cancelled) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while proxy.dropped.load(Ordering::SeqCst) < 3 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await?;
        assert!(!operation.is_finished());
        operation.abort();
        assert!(operation.await.unwrap_err().is_cancelled());
        let count = proxy.plans.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert_eq!(proxy.plans.load(Ordering::SeqCst), count);
    } else {
        let result = tokio::time::timeout(Duration::from_secs(10), operation).await??;
        assert_eq!(
            result.is_ok(),
            matches!(case, Case::Committed | Case::Absent | Case::DelayedCommit),
            "{result:?}"
        );
        assert_eq!(proxy.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(
            proxy.plans.load(Ordering::SeqCst),
            if matches!(case, Case::Absent | Case::DelayedCommit | Case::FencedRetry) {
                2
            } else {
                1
            }
        );
    }
    let expected_request = p::PlanRequest {
        authority: original_authority.clone(),
        plan: plan.clone(),
    };
    assert!(proxy
        .plan_requests
        .lock()
        .unwrap()
        .iter()
        .all(|request| request == &expected_request));
    cluster.client = direct;
    // Query through the proxy-independent client, including when fenced.
    let response = cluster
        .client
        .request_routed(Request::Query {
            group: original_authority.group,
            query: query::Query::Publisher {
                table: original_authority.table,
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
    if matches!(
        case,
        Case::Cancelled | Case::ChangedPredecessor | Case::FencedRetry
    ) {
        assert!(view.progress.pending.is_none());
    } else {
        let pending = view.progress.pending.as_ref().unwrap();
        if let Some(index) = delayed_index {
            assert_eq!(pending.index, index, "retry replaced the original plan");
        }
        assert_eq!(pending.plan.transaction, plan.transaction);
        assert_eq!(
            pending.plan.read_version,
            if matches!(case, Case::DifferentContent) {
                plan.read_version - 1
            } else {
                plan.read_version
            }
        );
    }
    drop(proxy);
    cluster.shutdown().await
}

async fn submit_plan(client: &Client, authority: &p::Authority, plan: &p::Plan) -> Result<u64> {
    let response = client
        .request_routed(Request::Commit(Box::new(Command::Plan(p::PlanRequest {
            authority: authority.clone(),
            plan: plan.clone(),
        }))))
        .await?;
    let Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Planned { index }))) =
        response.reply
    else {
        bail!("test plan rejected: {:?}", response.reply)
    };
    Ok(index)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_unknown_plan_is_observed_without_resubmission() -> Result<()> {
    exercise(Case::Committed, false).await?;
    exercise(Case::Committed, true).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn absent_unknown_plan_reissues_identical_selection() -> Result<()> {
    exercise(Case::Absent, false).await?;
    exercise(Case::Absent, true).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_plan_requires_full_content_match() -> Result<()> {
    exercise(Case::DifferentContent, false).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_plan_cannot_bypass_owner_fencing() -> Result<()> {
    exercise(Case::Fenced, false).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delayed_original_plan_commit_retains_original_index() -> Result<()> {
    exercise(Case::DelayedCommit, false).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plan_retry_cannot_bypass_new_owner_after_observation() -> Result<()> {
    exercise(Case::FencedRetry, true).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_plan_with_changed_predecessor_is_not_reissued() -> Result<()> {
    exercise(Case::ChangedPredecessor, true).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn continuously_unknown_plan_can_be_cancelled_without_background_retries() -> Result<()> {
    exercise(Case::Cancelled, false).await?;
    exercise(Case::Cancelled, true).await
}
