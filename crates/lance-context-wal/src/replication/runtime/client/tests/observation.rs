use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn append_diagnostics_distinguish_disk_refusals_from_unknown_outcomes() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let handle = &c.net.nodes[&leader].handle;
    let request = Request::Append {
        group: 10,
        batch: batch(1),
    };

    let slots = handle
        .shared
        .data
        .slots
        .clone()
        .try_acquire_many_owned(handle.shared.data.capacity_slots as u32)?;
    assert!(matches!(
        c.client.request(leader, request.clone()).await?.reply,
        Reply::Unknown
    ));
    assert_eq!(handle.append_observation()["request_credits"], 1);
    drop(slots);

    let bytes = handle
        .shared
        .data
        .bytes
        .clone()
        .try_acquire_many_owned(handle.shared.data.capacity_bytes as u32)?;
    assert!(matches!(
        c.client.request(leader, request.clone()).await?.reply,
        Reply::Unknown
    ));
    assert_eq!(handle.append_observation()["byte_credits"], 1);
    drop(bytes);

    let first = c.client.request(leader, request.clone()).await?;
    let index = accepted(&first);
    drop(first);
    let retry = c.client.request(leader, request.clone()).await?;
    assert_eq!(accepted(&retry), index);
    drop(retry);

    let follower = *c.net.nodes.keys().find(|&&id| id != leader).unwrap();
    assert!(matches!(
        c.client.request(follower, request.clone()).await?.reply,
        Reply::Redirect { .. }
    ));
    assert_eq!(
        c.net.nodes[&follower].handle.append_observation()["redirect"],
        1
    );

    handle.disk.as_ref().unwrap().limit_data(1)?;
    assert!(matches!(
        c.client.request(leader, request).await?.reply,
        Reply::AppendNotSubmittedDiskBudget
    ));
    let observed = handle.append_observation();
    assert_eq!(observed["disk_budget"], 1);
    assert_eq!(observed["accepted"], 2);
    assert_eq!(observed["started"], 5);
    assert_eq!(observed["execution_other"], 0);
    assert_eq!(observed["cancelled"], 0);
    assert_eq!(observed.values().sum::<u64>(), 2 * observed["started"]);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disk_refused_append_keeps_identity_and_does_not_resolve_a_prior_commit() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let handle = &c.net.nodes[&leader].handle;
    let budget = handle.disk.as_ref().unwrap();
    let request = Request::Append {
        group: 10,
        batch: batch(1),
    };
    let reserve = || {
        let view = budget.snapshot();
        // Use the separate control headroom to tolerate concurrent heartbeat
        // accounting while occupying the complete producer admission window.
        budget.reserve(
            (view.data_limit - view.accounted_disk_bytes) as usize,
            false,
        )
    };
    let held = reserve()?;
    assert!(matches!(
        c.client.request_routed(request.clone()).await?.reply,
        Reply::AppendNotSubmittedDiskBudget
    ));
    assert_eq!(handle.observation(10)?.status.accepted_records, 0);
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, None);
    drop(held);

    // Explicitly submit the identical request after the reservation releases.
    let first = c.client.request_routed(request.clone()).await?;
    let index = accepted(&first);
    drop(first);
    let held = reserve()?;
    assert!(matches!(
        c.client.request_routed(request.clone()).await?.reply,
        Reply::AppendNotSubmittedDiskBudget
    ));
    // This later refusal cannot undo or deny the first durable acceptance.
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, Some(index));
    drop(held);
    let retried = c.client.request_routed(request).await?;
    assert_eq!(accepted(&retried), index);
    drop(retried);
    assert_eq!(handle.observation(10)?.status.accepted_records, 1);
    assert_eq!(handle.append_observation()["disk_budget"], 2);
    assert_eq!(handle.append_observation()["started"], 4);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_append_execution_is_not_reported_as_rejected_or_accepted() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open_with_options(
        dir.path(),
        Options {
            timeout: Duration::from_millis(300),
            ..Options::default()
        },
        Options::default(),
    )
    .await?;
    let leader = c.leader().await?;
    let handle = &c.net.nodes[&leader].handle;
    let (entered, arrived) = tokio::sync::oneshot::channel();
    let (release, resumed) = std::sync::mpsc::channel::<()>();
    // Keep the actual group owner from processing the queued append until the
    // RPC deadline expires. Dropping the RPC future must not discard its intent.
    handle.enqueue(
        handle.route(10)?,
        Work::Local(Box::new(move |_| {
            let _ = entered.send(());
            let _ = resumed.recv();
            Box::new(|| {})
        })),
        false,
    )?;
    arrived.await?;
    let request = Request::Append {
        group: 10,
        batch: batch(1),
    };
    let expired = c.client.request(leader, request.clone()).await;
    drop(release);
    assert!(expired.is_err());
    assert_eq!(handle.append_observation()["cancelled"], 1);
    assert_eq!(handle.append_observation()["accepted"], 0);
    // The timed-out operation can still commit. Retry the original bytes and
    // verify one accepted record and the original durable position.
    let retried = c.client.request_routed(request).await?;
    let index = accepted(&retried);
    drop(retried);
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, Some(index));
    assert_eq!(handle.observation(10)?.status.accepted_records, 1);
    assert_eq!(handle.append_observation()["accepted"], 1);
    c.shutdown().await
}
