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
    assert!(handle.last_append_disk_refusal().is_none());
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
    let decision = handle.last_append_disk_refusal().unwrap();
    assert_eq!(decision.limit_bytes, 1);
    assert!(decision.requested_bytes > decision.limit_bytes);
    assert_eq!(observed["disk_budget"], 1);
    assert_eq!(observed["accepted"], 2);
    assert_eq!(observed["started"], 5);
    assert_eq!(observed["execution_other"], 0);
    assert_eq!(observed["cancelled"], 0);
    assert_eq!(observed.values().sum::<u64>(), 2 * observed["started"]);
    let execution = handle.append_execution_timing();
    assert_eq!(execution.started, 5);
    assert_eq!(execution.finished, 5);
    assert!(execution.elapsed_micros >= execution.max_elapsed_micros);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn append_disk_refusal_retains_decision_after_reservations_release() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(dir.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let handle = &c.net.nodes[&leader].handle;
    let budget = handle.disk.as_ref().unwrap();
    let request = Request::Append {
        group: 10,
        batch: batch(1),
    };
    // Populate both kinds of reservation. Control headroom permits holding the
    // producer window without racing heartbeat accounting at its exact limit.
    let file = budget.reserve_files(&[4097])?;
    let view = budget.snapshot();
    let proposal_bytes = view.data_limit - view.accounted_disk_bytes - view.reserved_file_bytes;
    let proposal = budget.reserve(proposal_bytes as usize, false)?;
    assert!(matches!(
        c.client.request(leader, request.clone()).await?.reply,
        Reply::AppendNotSubmittedDiskBudget
    ));
    let decision = handle.last_append_disk_refusal().unwrap();
    assert_eq!(decision.limit_bytes, view.data_limit);
    assert_eq!(decision.reserved_file_bytes, 8192);
    assert!(decision.reserved_proposal_bytes >= proposal_bytes);
    assert!(decision.requested_bytes > 0);
    assert!(
        decision.accounted_bytes
            + decision.reserved_proposal_bytes
            + decision.reserved_file_bytes
            + decision.requested_bytes
            > decision.limit_bytes
    );
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, None);

    drop(proposal);
    drop(file);
    budget.refresh()?;
    assert_eq!(budget.snapshot().reserved_file_bytes, 0);
    let retry = c.client.request(leader, request.clone()).await?;
    let index = accepted(&retry);
    drop(retry);
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, Some(index));
    // Status retains the failure's exact values, even after pressure is gone
    // and the same request succeeds. It must not synthesize a fresh measurement.
    assert_eq!(
        serde_json::to_value(handle.last_append_disk_refusal().unwrap())?,
        serde_json::to_value(decision)?
    );
    assert_eq!(handle.append_observation()["disk_budget"], 1);

    budget.limit_data(1)?;
    assert!(matches!(
        c.client.request(leader, request).await?.reply,
        Reply::AppendNotSubmittedDiskBudget
    ));
    assert_eq!(handle.last_append_disk_refusal().unwrap().limit_bytes, 1);
    assert_eq!(handle.append_observation()["disk_budget"], 2);
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
    let execution = handle.append_execution_timing();
    assert_eq!(execution.started, 1);
    assert_eq!(execution.finished, 1);
    assert!(execution.elapsed_micros >= 200_000);
    let response = &c.client.append_timings()["response"];
    assert_eq!(response.started, 1);
    assert_eq!(response.finished, 1);
    assert!(response.elapsed_micros >= 200_000);
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn append_timings_separate_byte_and_connection_waits() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let c = RpcCluster::open(
        dir.path(),
        Options {
            connections: 1,
            ..Options::default()
        },
    )
    .await?;
    let leader = c.leader().await?;
    let handle = &c.net.nodes[&leader].handle;
    let byte_permits = c.client.budgets.data.available_permits() as u32;
    let bytes = c
        .client
        .budgets
        .data
        .clone()
        .acquire_many_owned(byte_permits)
        .await?;
    let connection = c.client.nodes[&leader]
        .data
        .slots
        .clone()
        .acquire_owned()
        .await?;
    let client = c.client.clone();
    let request = client.request(
        leader,
        Request::Append {
            group: 10,
            batch: batch(1),
        },
    );
    tokio::pin!(request);
    assert!(timeout(Duration::from_millis(30), &mut request)
        .await
        .is_err());
    let blocked = c.client.append_timings();
    assert_eq!(blocked["admission"].started, 1);
    assert_eq!(blocked["admission"].finished, 0);
    assert_eq!(blocked["connection"].started, 0);
    assert_eq!(handle.append_execution_timing().started, 0);
    drop(bytes);

    assert!(timeout(Duration::from_millis(30), &mut request)
        .await
        .is_err());
    let blocked = c.client.append_timings();
    assert_eq!(blocked["admission"].finished, 1);
    assert!(blocked["admission"].elapsed_micros >= 20_000);
    assert_eq!(blocked["connection"].started, 1);
    assert_eq!(blocked["connection"].finished, 0);
    assert_eq!(blocked["send"].started, 0);
    assert_eq!(handle.append_execution_timing().started, 0);
    drop(connection);

    let completed = request.await?;
    let index = accepted(&completed);
    drop(completed);
    // Control traffic must not enter the append phase counters.
    assert_eq!(producer(&c.client, leader, &batch(1)).await?, Some(index));
    let observed = c.client.append_timings();
    for phase in observed.values() {
        assert_eq!(phase.started, 1);
        assert_eq!(phase.finished, 1);
        assert_eq!(phase.elapsed_micros, phase.max_elapsed_micros);
    }
    assert!(observed["connection"].elapsed_micros >= 20_000);
    assert_eq!(handle.append_execution_timing().finished, 1);
    c.shutdown().await
}
