//! Full-body HTTP reads across WAL retirement, owner takeover, and restart.
use super::*;
use crate::http::annotations::PointRequest;
use lance_context_core::MergeReservation;

pub(super) async fn idle_budget(budget: &Arc<MergeMemoryBudget>) -> Result<MergeReservation> {
    tokio::time::timeout(DEADLINE, budget.reserve(budget.limit()))
        .await
        .context("HTTP read budget did not release")
}

fn request(query: &PointRequest) -> Result<axum::extract::Request> {
    let body = serde_json::to_vec(query)?;
    Ok(axum::http::Request::builder()
        .method("POST")
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .header(axum::http::header::CONTENT_LENGTH, body.len())
        .body(axum::body::Body::from(body))?)
}

pub(super) async fn admission_and_response_lifetime(
    owner: &Owner,
    service: &Service,
    budget: &Arc<MergeMemoryBudget>,
    missing: &PointRequest,
    query: &PointRequest,
) -> Result<()> {
    let invalid = owner
        .read_body_for_test(request(&PointRequest {
            turn: String::new(),
        })?)
        .await;
    assert_eq!(invalid.status(), axum::http::StatusCode::BAD_REQUEST);
    drop(invalid);
    let held = idle_budget(budget).await?;
    let refused = owner.read_body_for_test(request(missing)?).await;
    assert_eq!(
        refused.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    let error = axum::body::to_bytes(refused.into_body(), 4096).await?;
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&error)?["error"],
        "body_read_capacity"
    );
    drop(held);

    // The existing frozen body stays protected even after the response handler
    // returns. Cancelling an unconsumed response must release both resources.
    let before = service.observation().await.0.readers;
    let response = owner.read_body_for_test(request(query)?).await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert!(budget.reserved() > 0);
    assert_eq!(service.observation().await.0.readers, before + 1);
    drop(response);
    assert_eq!(budget.reserved(), 0);
    assert_eq!(service.observation().await.0.readers, before);

    // Consuming the body transfers ownership to Bytes. Response completion
    // cannot release its charge while the transport still retains those bytes.
    let response = owner.read_body_for_test(request(query)?).await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await?;
    assert!(budget.reserved() > 0);
    assert_eq!(service.observation().await.0.readers, before + 1);
    drop(bytes);
    assert_eq!(budget.reserved(), 0);
    assert_eq!(service.observation().await.0.readers, before);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_http_fences_takeover_during_native_read() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), fenced_read()).await?
}

async fn fenced_read() -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let input = body("session", "one", 3)?;
    let spec = TurnIngestionSpec::new(
        input.schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let shard = Shard::for_session(&layout, &binding(), "session")?;
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let directory = root.path().join("bodies.lance");
    let storage = fault_store::storage(&directory, Arc::default()).await?;
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        layout.clone(),
        binding(),
        storage.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &fixture::limits(),
        shard,
        &spec,
    )
    .await?;
    let mut physical = Physical {
        binding: binding(),
        shard,
        storage,
    };
    let budget = MergeMemoryBudget::new(32 << 20);
    let (service, runtime) = physical.owner(&c, &layout, &spec, budget.clone()).await?;
    let (stop, task) = run_owner(runtime);
    let lane = service
        .open_producer(ProducerOpen {
            bucket: shard.bucket,
            slot: 1,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .await?;
    let append = super::super::request(&service, &lane, 1, physical.wire(&spec, &input)?).await?;
    let ack = service.submit(&append, DEADLINE).await?;
    drop(append);
    physical.drained(&c, &service, ack.index).await?;
    stop.send(()).unwrap();
    task.await??;
    drop(service);

    // Fresh native caches guarantee the next body lookup reaches real IO.
    let faults = Arc::new(fault_store::Faults::default());
    physical.storage = fault_store::storage(&directory, faults.clone()).await?;
    let (_, old_runtime) = physical.owner(&c, &layout, &spec, budget.clone()).await?;
    let old_owner = Owner::new(&old_runtime, keys()?, Default::default())?;
    let old_server = Server::start(old_owner.router()).await?;
    let (stop_old, stopped_old) = tokio::sync::oneshot::channel();
    let old_work = old_runtime.run(async {
        let _ = stopped_old.await;
    });
    tokio::pin!(old_work);
    let query = PointRequest {
        turn: spec.row_key(&input, 0, 4096)?,
    };
    let endpoint = old_server.base.join(&format!(
        "/api/v1/annotations/{}/read-body",
        binding().qualified_table
    ))?;
    let (entered, resume) = faults.pause_data_read();
    let pending = tokio::spawn(transport()?.post(endpoint.clone()).json(&query).send());
    tokio::select! {
        result = &mut old_work => bail!("old runtime stopped before native read: {result:?}"),
        result = tokio::time::timeout(DEADLINE, entered) => { result??; }
    }
    assert!(!pending.is_finished());
    // Keep the old runtime alive but unpolled during takeover. Its background
    // shutdown cannot substitute for the HTTP handler's own epoch fence.

    physical.storage = fault_store::storage(&directory, Arc::default()).await?;
    let (current, successor) = physical
        .owner(&c, &layout, &spec, MergeMemoryBudget::new(32 << 20))
        .await?;
    let (stop_new, new_task) = run_owner(successor);
    resume.send(()).unwrap();
    let inflight = pending.await??;
    let stale = transport()?.post(endpoint).json(&query).send().await?;
    for response in [inflight, stale] {
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.json::<serde_json::Value>().await?["error"],
            "annotation_owner_changed"
        );
    }
    check(&current, &spec, &input).await?;
    drop(idle_budget(&budget).await?);
    let _ = stop_old.send(());
    let _ = old_work.await;
    old_owner.shutdown().await;
    old_server.shutdown().await?;
    stop_new.send(()).unwrap();
    new_task.await??;
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_http_crosses_wal_retirement_with_one_reader_permit() -> Result<()> {
    tokio::time::timeout(DEADLINE, read_across_retirement(false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_http_uses_installed_cut_when_apply_memory_is_full() -> Result<()> {
    tokio::time::timeout(DEADLINE, read_across_retirement(true)).await?
}

async fn read_across_retirement(apply_pressure: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let input = body("session", "retired-before-fetch", 7)?;
    let spec = TurnIngestionSpec::new(
        input.schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let shard = Shard::for_session(&layout, &binding(), "session")?;
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let storage = fixture::storage(root.path(), "bodies.lance").await?;
    let limits = lance_context_core::annotation_staging::Limits {
        readers: 1,
        ..fixture::limits()
    };
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        layout.clone(),
        binding(),
        storage.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &limits,
        shard,
        &spec,
    )
    .await?;
    let physical = Physical {
        binding: binding(),
        shard,
        storage,
    };
    let (service, runtime) = physical
        .owner_with_limits(&c, &layout, &spec, MergeMemoryBudget::new(32 << 20), limits)
        .await?;
    let mut runtime = runtime.with_owner_batch(Default::default())?;
    let lane = service
        .open_producer(ProducerOpen {
            bucket: shard.bucket,
            slot: 1,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .await?;
    let append = super::super::request(&service, &lane, 1, physical.wire(&spec, &input)?).await?;
    let ack = {
        let submitted = service.submit(&append, DEADLINE);
        tokio::pin!(submitted);
        loop {
            runtime.apply_once_for_test().await?;
            tokio::select! {
                result = &mut submitted => break result?,
                _ = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        }
    };
    drop(append);
    assert_eq!(ack.result, MutationResult::Applied);
    assert!(service.selected_full_body(DEADLINE).await?.through < ack.index);

    let owner = Owner::new(&runtime, keys()?, Default::default())?;
    let server = Server::start(owner.router()).await?;
    let (mut captured, resume) = service.pause_body_fetch_for_test();
    let read = {
        let base = server.base.clone();
        let spec = spec.clone();
        let input = input.clone();
        tokio::spawn(async move { http_check(&base, &spec, &input).await })
    };
    loop {
        tokio::select! {
            biased;
            result = &mut captured => { result?; break; }
            _ = tokio::time::sleep(Duration::from_millis(5)) => {
                runtime.apply_once_for_test().await?;
            }
        }
    }
    assert_eq!(service.observation().await.0.readers, 1);
    assert!(!read.is_finished());

    publish_and_retire(&c, &physical, &service, &mut runtime, ack.index, None).await?;
    assert_eq!(service.observation().await.0.readers, 1);
    let apply_hold = if apply_pressure {
        let budget = service.apply_budget_for_test();
        let hold = idle_budget(budget).await?;
        assert_eq!(budget.reserved(), budget.limit());
        Some(hold)
    } else {
        None
    };
    resume
        .send(())
        .expect("body read still waiting for WAL fetch");
    read.await??;
    drop(apply_hold);
    owner.shutdown().await;
    server.shutdown().await?;
    drop((runtime, service));
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn body_commit_ack_survives_owner_and_source_leader_shutdown() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(45), Box::pin(ack_across_restart(false))).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn body_commit_ack_survives_reopen_of_all_replica_disks() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(45), Box::pin(ack_across_restart(true))).await?
}

async fn ack_across_restart(restart_all: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let layout = layout();
    let input = body("session", "committed-before-owner-loss", 81)?;
    let spec = TurnIngestionSpec::new(
        input.schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let shard = Shard::for_session(&layout, &binding(), "session")?;
    let mut c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let storage = fixture::storage(root.path(), "bodies.lance").await?;
    crate::bootstrap::provision_body_shard(
        c.client.clone(),
        layout.clone(),
        binding(),
        storage.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        &fixture::limits(),
        shard,
        &spec,
    )
    .await?;
    let physical = Physical {
        binding: binding(),
        shard,
        storage,
    };
    let uploads = MergeMemoryBudget::new(32 << 20);
    let (old_service, old_runtime) = physical.owner(&c, &layout, &spec, uploads.clone()).await?;
    let old_owner = Owner::new(&old_runtime, keys()?, Default::default())?;
    let old_server = Server::start(old_owner.router()).await?;
    let journal = root.path().join("producer-journal");
    let client = physical
        .client(&journal, old_server.base.clone(), &layout, &spec, 1)
        .await?;
    let ack = client.commit(0, physical.wire(&spec, &input)?).await?;
    // The old applier never runs. This is purely a durable WAL commit ACK.
    assert!(old_service.observation().await.0.applied < ack.index);
    assert!(client.selected_full_body().await?.through < ack.index);
    client.shutdown().await?;
    old_owner.shutdown().await;
    old_server.shutdown().await?;
    drop((old_runtime, old_service));
    let group = layout.group_for_bucket(shard.bucket)?;
    let stopped = if restart_all {
        c.shutdown().await?;
        c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
        None
    } else {
        Some(c.stop_source_leader(group).await?)
    };

    let (service, mut runtime) = physical.owner(&c, &layout, &spec, uploads).await?;
    let owner = Owner::new(&runtime, keys()?, Default::default())?;
    let server = Server::start(owner.router()).await?;
    let client = physical
        .client(&journal, server.base.clone(), &layout, &spec, 1)
        .await?;
    let recovered = client
        .last_committed(0)
        .await?
        .context("durable journal lost the ACK")?;
    assert_eq!(recovered.index, ack.index);
    assert_eq!(recovered.receipt, ack.receipt);
    assert_eq!(recovered.meta, ack.meta);
    let recovery_cut = runtime.http_binding().2.max(ack.index);
    while service.observation().await.0.applied < recovery_cut {
        runtime.apply_once_for_test().await?;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let applied = client
        .wait_applied(0, &recovered)
        .await
        .context("recover original receipt")?;
    assert_eq!(applied.completion.index, ack.index);
    assert_eq!(applied.completion.result, MutationResult::Applied);
    assert!(client.selected_full_body().await?.through < ack.index);
    http_check(&server.base, &spec, &input)
        .await
        .context("read replayed staging before publication")?;

    publish_and_retire(&c, &physical, &service, &mut runtime, ack.index, stopped).await?;
    let selected = client.selected_full_body().await?;
    assert!(selected.through >= ack.index);
    assert_eq!(selected.rows, Some(1));
    http_check(&server.base, &spec, &input).await?;
    client.shutdown().await?;
    owner.shutdown().await;
    server.shutdown().await?;
    drop((runtime, service));
    c.shutdown().await
}

// A selected manifest alone does not prove that the bytes were retired. Query
// each live voter directly before allowing the pending read to continue.
async fn publish_and_retire(
    c: &network::Cluster,
    physical: &Physical,
    service: &Service,
    runtime: &mut Runtime,
    cut: u64,
    stopped: Option<u64>,
) -> Result<()> {
    while service.observation().await.0.materialized < cut {
        runtime.publish_once_for_test().await?;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    physical.drained(c, service, cut).await?;
    let group = service
        .ingress()
        .layout
        .group_for_bucket(physical.shard.bucket)?;
    for node in (1..=3).filter(|&node| Some(node) != stopped) {
        loop {
            let response = c
                .client
                .request(node, Request::Fetch { group, index: cut })
                .await?;
            if matches!(response.reply, Reply::Record(None)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    Ok(())
}
