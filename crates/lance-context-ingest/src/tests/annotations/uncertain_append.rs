//! Quorum acceptance resolves an unknown append, but never substitutes for apply.
use super::*;
use crate::tests::annotation_runtime::lost_prepare::proxy::Proxy;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, PartialEq)]
enum Case {
    Accepted,
    Absent,
    DifferentBytes,
    RejectedMutation,
    PreviousSequence,
    LateCommit,
    PersistentUnknown,
    ProducerFenced,
    LostRetryReply,
    RetryRejectedMutation,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_observes_exact_acceptance_and_waits_for_apply() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Case::Accepted)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_without_acceptance_resubmits_exact_request() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Case::Absent)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_does_not_accept_same_sequence_with_different_bytes() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Case::DifferentBytes)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_preserves_the_applied_mutation_rejection() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Case::RejectedMutation)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_after_previous_sequence_resubmits_exact_request() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Case::PreviousSequence)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_that_commits_after_observation_is_not_applied_twice() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Case::LateCommit)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_resubmission_is_bounded_and_preserves_unknown() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Case::PersistentUnknown)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_resubmission_cannot_bypass_producer_fence() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Case::ProducerFenced)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_observes_lost_exact_resubmission_reply() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), exercise(Case::LostRetryReply)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_append_resubmission_preserves_applied_rejection() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(60),
        exercise(Case::RetryRejectedMutation),
    )
    .await?
}

async fn exercise(case: Case) -> Result<()> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("annotation-uncertain-append-");
    let root = match std::env::var("INGEST_TEST_ARTIFACTS") {
        Ok(parent) => builder.tempdir_in(parent)?,
        Err(_) => builder.tempdir()?,
    }
    .keep();
    let mut c = network::Cluster::open_signed(&root, annotation_publication::one_shard()).await?;
    let storage = Storage::resolve(
        root.join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let limits = Limits::default();
    annotation_publication::seed_contract(&c, &storage, Service::contract(&binding(), &limits)?)
        .await?;
    let execute = matches!(case, Case::Accepted | Case::RejectedMutation);
    let proxy = Proxy::open_append(&c, &root, execute).await?;
    proxy
        .unknown_appends
        .store(case == Case::PersistentUnknown, Ordering::SeqCst);
    proxy
        .pause_append_retry
        .store(case == Case::LateCommit, Ordering::SeqCst);
    proxy
        .lose_append_retry_reply
        .store(case == Case::LostRetryReply, Ordering::SeqCst);
    let direct = std::mem::replace(&mut c.client, proxy.client.clone());
    let (service, driver) = connect(&c, &storage, &limits).await?;
    let producer = producer(&service, 1).await?;
    let turn = if matches!(case, Case::RejectedMutation | Case::RetryRejectedMutation) {
        "missing"
    } else {
        "T"
    };
    if case == Case::PreviousSequence {
        let previous = request(
            &service,
            &producer,
            1,
            wire(&limits, &["T"], 0, None, 1, Some(0.9))?,
        )
        .await?;
        assert!(matches!(
            direct
                .request_routed(Request::Append {
                    group: 10,
                    batch: previous.batch.clone()
                })
                .await?
                .reply,
            Reply::Commit(Outcome::Committed(Applied::Accepted { .. }))
        ));
    }
    let input = Arc::new(
        request(
            &service,
            &producer,
            if case == Case::PreviousSequence { 2 } else { 1 },
            wire(&limits, &[turn], 1, Some("reviewed"), 0, None)?,
        )
        .await?,
    );
    let submit = tokio::spawn({
        let service = service.clone();
        let input = input.clone();
        async move { service.submit(&input, Duration::from_secs(30)).await }
    });
    tokio::time::timeout(Duration::from_secs(20), proxy.committed.notified()).await?;
    if case == Case::DifferentBytes {
        let other = request(
            &service,
            &producer,
            1,
            wire(&limits, &[turn], 1, Some("other"), 0, None)?,
        )
        .await?;
        assert!(matches!(
            direct
                .request_routed(Request::Append {
                    group: 10,
                    batch: other.batch.clone(),
                })
                .await?
                .reply,
            Reply::Commit(Outcome::Committed(Applied::Accepted { .. }))
        ));
    }
    if case == Case::ProducerFenced {
        let layout = annotation_publication::one_shard();
        let command = producer::Command {
            cluster: layout.cluster,
            layout: layout.fingerprint()?,
            group: 10,
            table: input.meta().stream,
            slot: 1,
            action: producer::Action::Revoke {
                generation: producer.generation(),
            },
        };
        assert!(matches!(
            direct
                .request_routed(Request::Commit(Box::new(Command::Producer(command))))
                .await?
                .reply,
            Reply::Commit(Outcome::Committed(Applied::Producer(
                producer::Reply::Closed { last: None }
            )))
        ));
    }
    proxy.release.notify_one();
    let late_index = if case == Case::LateCommit {
        tokio::time::timeout(Duration::from_secs(20), proxy.committed.notified()).await?;
        // The first observation found no acceptance. Commit the original exact
        // bytes before forwarding the paused resubmission to the real RF3 WAL.
        let reply = direct
            .request_routed(Request::Append {
                group: 10,
                batch: input.batch.clone(),
            })
            .await?
            .reply;
        let Reply::Commit(Outcome::Committed(Applied::Accepted { index })) = reply else {
            bail!("original late commit did not succeed: {reply:?}");
        };
        proxy.release.notify_one();
        Some(index)
    } else {
        None
    };
    let successful = !matches!(
        case,
        Case::DifferentBytes | Case::PersistentUnknown | Case::ProducerFenced
    );
    if successful {
        tokio::time::timeout(Duration::from_secs(20), async {
            while proxy.appends.load(Ordering::SeqCst) == 0 {
                assert!(
                    !submit.is_finished(),
                    "append ended before exact resubmission was accepted"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?;
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !submit.is_finished(),
            "quorum acceptance must not acknowledge an unapplied mutation"
        );
    }
    let (stop, task) = run(driver);
    let result = submit.await?;
    if successful {
        let completed = result?;
        assert_eq!(completed.meta, input.batch.meta);
        if let Some(index) = late_index {
            assert_eq!(
                completed.index, index,
                "retry returned a second application position"
            );
            let reply = direct
                .request_routed(Request::Query {
                    group: 10,
                    query: query::Query::Accepted {
                        after: index - 1,
                        limit: 128,
                    },
                })
                .await?
                .reply;
            let Reply::Query(query::Outcome::Observed {
                value: query::Value::Accepted(page),
                ..
            }) = reply
            else {
                bail!("accepted records unavailable: {reply:?}");
            };
            assert_eq!(
                page.entries,
                vec![(index, input.meta().clone())],
                "resubmission created a second accepted source record"
            );
        }
        if matches!(case, Case::RejectedMutation | Case::RetryRejectedMutation) {
            assert!(matches!(
                completed.result,
                MutationResult::Rejected(ref r) if r.row == 0 && r.reason == RejectionReason::MissingTurn
            ));
            assert!(service.get(turn, Duration::from_secs(10)).await?.is_none());
        } else {
            assert_eq!(completed.result, MutationResult::Applied);
            assert_eq!(
                service
                    .get(turn, Duration::from_secs(10))
                    .await?
                    .unwrap()
                    .status
                    .as_deref(),
                Some("reviewed")
            );
            if case == Case::PreviousSequence {
                assert_eq!(
                    service
                        .get(turn, Duration::from_secs(10))
                        .await?
                        .unwrap()
                        .score,
                    Some(0.9)
                );
            }
        }
    } else {
        let error = result.unwrap_err();
        assert!(
            matches!(
                error.downcast_ref::<AppendFailure>(),
                Some(AppendFailure::Unknown)
            ),
            "{error:#}"
        );
        let current = service.get(turn, Duration::from_secs(10)).await?.unwrap();
        assert_eq!(
            current.status.as_deref(),
            if case == Case::DifferentBytes {
                Some("other")
            } else {
                Some("pending")
            }
        );
    }
    assert_eq!(
        proxy.appends.load(Ordering::SeqCst),
        usize::from(successful)
    );
    assert_eq!(
        proxy.dropped.load(Ordering::SeqCst),
        if matches!(case, Case::PersistentUnknown | Case::LostRetryReply) {
            2
        } else {
            1
        }
    );
    // Redirects may add wire attempts, but every attempt must retain the same
    // full metadata and IPC payload. Persistent Unknown must stop after two.
    {
        let attempts = proxy.append_requests.lock().unwrap();
        for (meta, body) in attempts.iter() {
            assert_eq!(meta, input.meta());
            assert_eq!(body.as_slice(), input.body().as_ref());
        }
        if case == Case::PersistentUnknown {
            assert_eq!(attempts.len(), 2);
        }
    }
    stop.send(()).unwrap();
    task.await??;
    drop(service);
    c.client = direct;
    drop(proxy);
    c.shutdown().await
}
