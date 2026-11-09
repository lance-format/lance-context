use super::*;
use crate::replication::ProposalRejection;

#[test]
fn cancelled_validation_keeps_credit_until_the_blocking_task_releases_its_payload() -> Result<()> {
    let executor = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(1)
        .enable_all()
        .build()?;
    let root = tempfile::tempdir()?;
    let node = Node::open(root.path(), layout(), 1, Limits::default())?;
    let (runtime, _outbox) = Runtime::spawn(
        node,
        Options {
            data_requests: 1,
            ..Options::default()
        },
    )?;
    let handle = runtime.handle();
    let (entered, arrived) = mpsc::channel();
    let (release, resumed) = mpsc::channel::<()>();
    let blocked = executor.spawn_blocking(move || {
        entered.send(()).unwrap();
        let _ = resumed.recv_timeout(Duration::from_secs(10));
    });
    arrived.recv_timeout(Duration::from_secs(5))?;
    let submission = executor.spawn({
        let handle = handle.clone();
        async move { handle.append(10, batch(1)).await }
    });
    executor.block_on(wait_for(|| handle.admission().data.used_requests == 1))?;
    submission.abort();
    assert!(executor
        .block_on(submission)
        .err()
        .context("validation caller was not cancelled")?
        .is_cancelled());
    // The validation closure is queued behind the real blocked pool thread.
    // Dropping its JoinHandle must not release its still-owned body allowance.
    assert!(handle.admission().data.used_bytes >= 4096);
    let error = executor
        .block_on(handle.append(10, batch(2)))
        .err()
        .context("cancelled validation released admission too early")?;
    assert!(matches!(
        error.downcast_ref::<SubmissionFailure>(),
        Some(SubmissionFailure::RequestCredits)
    ));
    // Data reads share the occupied request allowance. Control metadata reads
    // still reach the group worker while validation waits on the blocking pool.
    assert!(matches!(
        handle
            .read_accepted(10, 1)
            .err()
            .context("data read bypassed admission")?
            .downcast_ref::<SubmissionFailure>(),
        Some(SubmissionFailure::RequestCredits)
    ));
    let response = handle.read_source(
        10,
        source_read::Request {
            table: batch(1).meta.stream,
            bucket: 0,
            after: 0,
            limit: 1,
        },
    )?;
    let error = executor
        .block_on(async { timeout(Duration::from_secs(5), response).await })?
        .err()
        .context("unelected node served source metadata")?;
    assert!(matches!(
        error.downcast_ref::<ProposalRejection>(),
        Some(ProposalRejection::NotLeader)
    ));
    drop(release);
    executor.block_on(blocked)?;
    executor.block_on(wait_for(|| handle.admission().data.used_requests == 0))?;
    assert_eq!(handle.admission().data.used_bytes, 0);
    assert_eq!(handle.observation(10)?.status.accepted_records, 0);
    assert!(executor
        .block_on(async { handle.read_accepted(10, 1)?.await })?
        .is_none());
    executor.block_on(runtime.shutdown())?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn large_prevalidated_burst_keeps_budgets_and_recovers_exact_payloads() -> Result<()> {
    for prevalidated in [false, true] {
        let root = tempfile::tempdir()?;
        let c = Cluster::open(
            root.path(),
            Options {
                pipeline_replication: true,
                ..Options::default()
            },
        )?;
        c.elect(1, 10).await?;
        let handle = c.handles[&1].clone();
        let (entered, arrived) = mpsc::channel();
        let (release, resumed) = mpsc::channel::<()>();
        handle.enqueue(
            handle.route(10)?,
            Work::Local(Box::new(move |_| {
                entered.send(()).unwrap();
                let _ = resumed.recv_timeout(Duration::from_secs(10));
                Box::new(|| {})
            })),
            false,
        )?;
        arrived.recv_timeout(Duration::from_secs(5))?;
        let bodies: Vec<_> = (0..8)
            .map(|n| {
                Batch::new(
                    batch(1).meta.stream,
                    0,
                    uuid::Uuid::from_u128(100 + n),
                    1,
                    bytes::Bytes::from(vec![n as u8; 4 << 20]),
                )
            })
            .collect();
        let before_drops = handle.peer_drops().0;
        let started = Instant::now();
        let mut pending = Vec::new();
        for body in &bodies {
            pending.push(if prevalidated {
                handle.append(10, body.clone()).await?
            } else {
                handle.commit(Command::Append {
                    group: 10,
                    batch: body.clone(),
                })?
            });
        }
        let admission = handle.admission();
        assert_eq!(admission.data.used_requests, 8);
        assert!(admission.data.used_bytes <= admission.data.capacity_bytes);
        assert!(pending.iter().all(|p| p.0.is_empty()));
        drop(release);
        let mut accepted = Vec::new();
        for reply in pending {
            accepted.push(index(timeout(Duration::from_secs(10), reply).await??));
        }
        let elapsed = started.elapsed();
        wait_for(|| {
            c.handles.values().all(|h| {
                h.observation(10)
                    .is_ok_and(|o| o.status.accepted_records == 8)
            })
        })
        .await?;
        assert_eq!(handle.admission().data.used_requests, 0);
        assert_eq!(handle.admission().data.used_bytes, 0);
        let drops = handle.peer_drops().0 - before_drops;
        println!(
            "prevalidated={prevalidated} eight_4mib_burst_elapsed_us={} outgoing_drops={drops}",
            elapsed.as_micros()
        );
        drop(handle);
        c.shutdown().await?;
        for id in 1..=3 {
            let node = Node::open(
                &root.path().join(id.to_string()),
                layout(),
                id,
                Limits::default(),
            )?;
            for (index, body) in accepted.iter().zip(&bodies) {
                assert_eq!(node.read_accepted(10, *index)?.as_ref(), Some(body));
            }
        }
    }
    Ok(())
}
