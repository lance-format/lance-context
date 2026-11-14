use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use lance_context_wal::replication::source;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
enum Case {
    Committed,
    Absent,
    CommittedFenced,
    AbsentFenced,
    PersistentUnknown,
    Unobservable,
}

async fn exercise(case: Case) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("pins.lance");
    let mut cluster = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&cluster, &path, &binding()).await?;
    let proxy = Proxy::open_pin(
        &cluster,
        root.path(),
        matches!(
            case,
            Case::Committed | Case::CommittedFenced | Case::Unobservable
        ),
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
    let mut publisher = publisher(&ingress, &storage, &budget, 1).await?;
    let authority = publisher.authority().clone();
    let (body, _) = append(
        &ingress,
        &ProducerOpen {
            bucket: 0,
            slot: 0,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        },
        1,
        &["a", "b"],
    )
    .await?;
    let mut expected = BTreeMap::new();
    remember(&mut expected, body)?;
    if matches!(case, Case::Unobservable) {
        for node in 1..=3 {
            proxy
                .client
                .request(
                    node,
                    Request::Query {
                        group: authority.group,
                        query: query::Query::Publisher {
                            table: authority.table,
                        },
                    },
                )
                .await?;
        }
    }
    let operation = tokio::spawn(async move {
        let result = publisher.prepare_next().await;
        (publisher, result)
    });
    tokio::time::timeout(Duration::from_secs(15), proxy.committed.notified()).await?;
    if matches!(case, Case::CommittedFenced | Case::AbsentFenced) {
        cluster.client = direct.clone();
        let ingress = connect(&cluster, binding()).await?;
        let successor = super::publisher(&ingress, &storage, &budget, authority.epoch).await?;
        assert_ne!(successor.authority(), &authority);
    }
    if matches!(case, Case::PersistentUnknown) {
        proxy.unknown_pins.store(true, Ordering::SeqCst);
    }
    proxy.release.notify_one();
    let (publisher, result) = tokio::time::timeout(Duration::from_secs(45), operation).await??;
    if matches!(case, Case::Unobservable) {
        proxy.restore_observations();
    }
    assert_eq!(
        result.is_ok(),
        matches!(case, Case::Committed | Case::Absent),
        "{result:?}"
    );
    let requests = proxy.pin_requests.lock().unwrap().clone();
    assert_eq!(
        requests.len(),
        if matches!(case, Case::Absent | Case::PersistentUnknown) {
            2
        } else {
            1
        }
    );
    assert!(
        requests.iter().all(|request| request == &requests[0]),
        "original pin command changed"
    );
    let selection = requests[0].selection();
    let response = direct
        .request_routed(Request::Query {
            group: selection.group,
            query: query::Query::SourcePin {
                table: authority.table,
            },
        })
        .await?;
    let Reply::Query(query::Outcome::Observed {
        value: query::Value::SourcePin(pin),
        ..
    }) = response.reply
    else {
        bail!("missing source observation");
    };
    if matches!(case, Case::AbsentFenced | Case::PersistentUnknown) {
        assert!(pin.is_none());
    } else {
        let pin = pin.context("original pin missing")?;
        assert_eq!(pin.permit.selection(), selection);
        let response = direct
            .request_routed(Request::Commit(Box::new(Command::Pin(requests[0].clone()))))
            .await?;
        assert!(
            matches!(response.reply,
                Reply::Commit(Outcome::Committed(Applied::Source(source::Reply::Pinned { index }))) if index == pin.index
            ),
            "duplicate pin did not preserve the first durable result"
        );
    }
    if matches!(case, Case::Committed | Case::Absent) {
        assert!(result?.is_some());
        assert!(publisher.commit_pending().await?.is_some());
        assert!(publisher.finish_pending().await?.is_some());
        verify(&publisher, &expected, 2).await?;
    }
    drop((publisher, ingress, proxy));
    cluster.client = direct;
    cluster.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_unknown_pin_is_observed_without_resubmission() -> Result<()> {
    exercise(Case::Committed).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn absent_unknown_pin_resumes_the_original_selection_and_publishes() -> Result<()> {
    exercise(Case::Absent).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn committed_pin_does_not_bypass_owner_fencing() -> Result<()> {
    exercise(Case::CommittedFenced).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn absent_pin_is_not_resubmitted_after_owner_fencing() -> Result<()> {
    exercise(Case::AbsentFenced).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeatedly_unknown_pin_stops_after_one_exact_resubmission() -> Result<()> {
    exercise(Case::PersistentUnknown).await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unavailable_pin_observation_preserves_failure() -> Result<()> {
    exercise(Case::Unobservable).await
}
