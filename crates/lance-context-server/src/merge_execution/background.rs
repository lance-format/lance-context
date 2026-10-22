//! Count/timer and direct HTTP merges retain their executor independently of
//! callers. Admission uses the same durable owner and commit guard as fanout.
use super::*;
use tokio::sync::{oneshot, OwnedSemaphorePermit};

type Reply = oneshot::Receiver<Result<usize, String>>;

struct TargetGuard {
    state: Arc<AppState>,
    target: String,
    request: bool,
}
impl Drop for TargetGuard {
    fn drop(&mut self) {
        let targets = if self.request {
            &self.state.merge_executions.request_targets
        } else {
            &self.state.merge_executions.background_targets
        };
        targets.lock().unwrap().remove(&self.target);
    }
}

/// Demand must reach the master even when every worker execution slot is
/// occupied. Metadata publication has its own small bound and per-table
/// coalescing; no payload slot or flush-pass wait is needed.
pub(crate) fn publish_background_request(state: Arc<AppState>, target: String) {
    let Ok(permit) = state
        .merge_executions
        .request_slots
        .clone()
        .try_acquire_owned()
    else {
        return;
    };
    if !state
        .merge_executions
        .request_targets
        .lock()
        .unwrap()
        .insert(target.clone())
    {
        return;
    }
    let guard = TargetGuard {
        state: state.clone(),
        target: target.clone(),
        request: true,
    };
    tokio::spawn(async move {
        let _permit = permit;
        let _guard = guard;
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            state.merge_executions.request_merge(&target),
        )
        .await;
        let label = match result {
            Ok(Ok(())) => "ok",
            outcome => {
                tracing::warn!(target, ?outcome, "master merge demand publication deferred");
                "failed"
            }
        };
        metrics::counter!("rollout_wal_background_requests_total", "result" => label).increment(1);
    });
}

/// Never wait for a slot or coordinator IO on the flush sweep. Coalescing and
/// the pre-acquired process slot bound both admission requests and executors.
/// Dropping the returned receiver does not drop an admitted storage write.
pub(crate) async fn spawn_background(
    state: Arc<AppState>,
    target: String,
) -> Result<Reply, AppError> {
    state.merge_executions.legacy_allowed(&target)?;
    if !state.merge_executions.rollout.fence_legacy
        && state.merge_executions.fencing_mode_known() != Some(true)
    {
        return Err(AppError::InvalidRequest(
            "background fencing is disabled".into(),
        ));
    }
    let endpoint = state.merge_executions.endpoint.clone().ok_or_else(|| {
        AppError::InvalidRequest("background merge executor endpoint is missing".into())
    })?;
    let slot = state
        .merge_slots
        .as_ref()
        .ok_or_else(|| AppError::InvalidRequest("background merges require bounded slots".into()))?
        .clone()
        .try_acquire_owned()
        .map_err(|_| AppError::Overloaded("merge slots are occupied".into()))?;
    if !state
        .merge_executions
        .background_targets
        .lock()
        .unwrap()
        .insert(target.clone())
    {
        return Err(AppError::Overloaded(
            "table already has a local merge admission".into(),
        ));
    }
    let guard = TargetGuard {
        state: state.clone(),
        target: target.clone(),
        request: false,
    };
    let mut execution = Execution::new(
        &target,
        &endpoint,
        state.merge_executions.instance(),
        state.merge_executions.timeout_secs,
    );
    execution.background = true;
    execution.queue_timeout_secs = state.merge_executions.queue_timeout_secs;
    execution.idle_timeout_secs = state.merge_executions.idle_timeout_secs;
    let (cancel, cancelled) = watch::channel(false);
    {
        let mut running = state.merge_executions.running.lock().await;
        if state
            .merge_executions
            .closing
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(AppError::Overloaded(
                "merge executor is shutting down".into(),
            ));
        }
        running.insert(execution.id.clone(), cancel);
    }
    let (reply, result) = oneshot::channel();
    // Retain reconciliation BEFORE the first fallible coordinator operation.
    tokio::spawn(async move {
        let _guard = guard;
        let outcome = std::panic::AssertUnwindSafe(execute_background(
            state.clone(),
            &execution,
            cancelled,
            slot,
        ))
        .catch_unwind()
        .await
        .unwrap_or_else(|_| Err("background executor panicked; durable fence retained".into()));
        if let Err(error) = &outcome {
            tracing::warn!(target = %target, id = %execution.id, %error, "background merge did not complete");
        }
        state
            .merge_executions
            .running
            .lock()
            .await
            .remove(&execution.id);
        let _ = reply.send(outcome);
    });
    Ok(result)
}

async fn execute_background(
    state: Arc<AppState>,
    execution: &Execution,
    cancelled: watch::Receiver<bool>,
    slot: OwnedSemaphorePermit,
) -> Result<usize, String> {
    if !state
        .merge_executions
        .fenced_mode()
        .await
        .map_err(|e| format!("{e:?}"))?
    {
        return Err("fenced merge protocol is not active".into());
    }
    let coordinator = state
        .merge_executions
        .coordinator()
        .await
        .map_err(|e| format!("{e:?}"))?;
    if *cancelled.borrow() {
        return Err("background admission cancelled before reservation".into());
    }
    loop {
        match coordinator.reserve_background(execution).await {
            Ok(true) => break,
            outcome => {
                match coordinator.get(&execution.target).await {
                    Ok(Some(current)) if current.id == execution.id => break,
                    Ok(_) if matches!(outcome, Ok(false)) => {
                        return Err("background merge deferred by ownership, admission gate or retry budget".into());
                    }
                    _ => {
                        // An uncertain transaction may have admitted this exact
                        // execution. Never start an untracked fallback or forget it.
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        }
    }
    run(
        state,
        coordinator.clone(),
        execution.clone(),
        cancelled,
        Some(slot),
    )
    .await?;
    loop {
        match coordinator.get(&execution.target).await {
            Ok(Some(current)) if current.id == execution.id => {
                if !matches!(current.phase, Phase::Finished | Phase::Recovered) {
                    return Err(
                        "background merge needs master storage recovery; fence retained".into(),
                    );
                }
                match coordinator.release_background(&current).await {
                    // False preserves a terminal record for an existing master
                    // claim. Only that master may transfer the target lock.
                    Ok(_) => return current.error.map_or(Ok(current.reclaimed), Err),
                    Err(error) => {
                        tracing::warn!(id = %execution.id, %error, "retrying background terminal release")
                    }
                }
            }
            Ok(_) => return Err("background execution reconciled by another owner".into()),
            Err(error) => {
                tracing::warn!(id = %execution.id, %error, "reconciling background terminal outcome")
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweeper;
    use lance_context_core::{rollout_record_from_add_request, RolloutStore};
    use tokio::sync::RwLock;

    async fn fixture(slots: usize, idle: u64) -> (Arc<AppState>, Coordinator, tempfile::TempDir) {
        let client =
            etcd_client::Client::connect([std::env::var("ETCD_TEST_ENDPOINTS").unwrap()], None)
                .await
                .unwrap();
        let coordinator = Coordinator::new(
            client,
            format!("/background-runtime/{}", uuid::Uuid::new_v4()),
        );
        let dir = tempfile::tempdir().unwrap();
        let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
        state.merge_slots = Some(Arc::new(tokio::sync::Semaphore::new(slots)));
        state.rollout_merge_after_generations = 1;
        state.merge_executions = Executions::new(Some(coordinator.clone()), 3600);
        state.merge_executions.rollout.fence_legacy = true;
        state.merge_executions.endpoint = Some("http://127.0.0.1:1".into());
        state.merge_executions.idle_timeout_secs = idle;
        (Arc::new(state), coordinator, dir)
    }

    async fn pending(state: &Arc<AppState>, name: &str) -> Arc<RwLock<RolloutStore>> {
        let _ = rollouts::create_rollout_store(
            State(state.clone()),
            Json(lance_context_api::CreateRolloutStoreRequest {
                name: name.into(),
                storage_options: None,
            }),
        )
        .await
        .unwrap();
        let store = state.get_or_open_rollout_store(name).await.unwrap();
        let row = rollout_record_from_add_request(&lance_context_api::AddRolloutRequest {
            id: "row".into(),
            rollout_id: "trajectory".into(),
            ..Default::default()
        });
        store.read().await.add(&[row]).await.unwrap();
        store.read().await.flush().await.unwrap();
        store
    }

    async fn running(coordinator: &Coordinator, target: &str) -> Execution {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Some(e) = coordinator.get(target).await.unwrap() {
                    if e.phase == Phase::Running {
                        return e;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn disconnected_http_keeps_admitted_executor_and_reclaims_real_wal() {
        let (state, coordinator, _dir) = fixture(1, 60).await;
        let store = pending(&state, "hot").await;
        let held = store.write().await;
        let caller = tokio::spawn(rollouts::merge_wal(
            State(state.clone()),
            axum::extract::Path("hot".into()),
        ));
        let e = running(&coordinator, "hot").await;
        assert!(e.background);
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        assert_eq!(state.merge_slots.as_ref().unwrap().available_permits(), 0);
        assert_eq!(coordinator.get("hot").await.unwrap().unwrap().id, e.id);
        drop(held);
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                if !state
                    .merge_executions
                    .background_targets
                    .lock()
                    .unwrap()
                    .contains("hot")
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        assert!(coordinator.get("hot").await.unwrap().is_none());
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            0
        );
        assert!(store.read().await.get_by_id("row").await.unwrap().is_some());
        assert_eq!(state.merge_slots.as_ref().unwrap().available_permits(), 1);
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn shutdown_handoff_cancels_waiting_http_merge_and_successor_reclaims_wal() {
        let (state, coordinator, dir) = fixture(2, 600).await;
        let store = pending(&state, "hot").await;
        let held = store.write().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = crate::routes::router().with_state(state.clone());
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let shutdown_state = state.clone();
        let server = tokio::spawn(crate::drain::serve(
            listener,
            app,
            Duration::from_millis(100),
            async move {
                stopped.await.unwrap();
                shutdown_state.merge_executions.begin_shutdown().await;
            },
        ));
        let request = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
            stream.write_all(b"POST /api/v1/internal/merge-wal/hot HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.unwrap();
            String::from_utf8(response).unwrap()
        });
        let old = running(&coordinator, "hot").await;
        stop.send(()).unwrap();
        // The locked store stays locked: shutdown must cancel this executor,
        // not wait for its 600-second idle deadline or an unrelated unlock.
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let response = request.await.unwrap();
        assert!(response.starts_with("HTTP/1.1 500"), "{response}");
        tokio::time::timeout(Duration::from_secs(5), state.merge_executions.shutdown())
            .await
            .unwrap();
        assert!(coordinator.get("hot").await.unwrap().is_none());
        assert!(state.merge_executions.running.lock().await.is_empty());
        // A racing timer or already-accepted HTTP handler cannot admit more
        // detached work after the initial cancellation snapshot.
        assert!(spawn_background(state.clone(), "late".into())
            .await
            .is_err());
        let late = Execution::new(
            "late",
            "http://127.0.0.1:1",
            state.merge_executions.instance(),
            3600,
        );
        assert!(
            matches!(super::super::start(State(state.clone()), Json(late)).await, Err(AppError::Overloaded(message)) if message == "merge executor is shutting down")
        );
        assert!(coordinator.get("late").await.unwrap().is_none());
        drop(held);
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            1
        );
        state.shutdown().await;

        let mut successor = AppState::new_for_test(dir.path().to_path_buf()).await;
        successor.merge_slots = Some(Arc::new(tokio::sync::Semaphore::new(1)));
        successor.merge_executions = Executions::new(Some(coordinator.clone()), 3600);
        successor.merge_executions.rollout.fence_legacy = true;
        successor.merge_executions.endpoint = Some("http://127.0.0.1:1".into());
        let successor = Arc::new(successor);
        assert_ne!(successor.merge_executions.instance(), old.instance);
        // Cancellation retains the ordinary retry budget. Wait for its exact
        // first retry deadline rather than erasing failures during handoff.
        let failure = coordinator
            .failure("hot", &old.endpoint)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(failure.consecutive_attempts, 1);
        let wait_ms = failure
            .next_retry_ms
            .saturating_sub(lance_context_merge::failure::now_ms());
        assert!(
            wait_ms <= 2_000,
            "unexpected cancellation retry policy: {failure:?}"
        );
        tokio::time::sleep(Duration::from_millis(wait_ms + 10)).await;
        let Json(_) =
            rollouts::merge_wal(State(successor.clone()), axum::extract::Path("hot".into()))
                .await
                .unwrap();
        let recovered = successor.get_or_open_rollout_store("hot").await.unwrap();
        assert_eq!(
            recovered
                .read()
                .await
                .pending_wal_generations()
                .await
                .unwrap(),
            0
        );
        assert!(recovered
            .read()
            .await
            .get_by_id("row")
            .await
            .unwrap()
            .is_some());
        successor.shutdown().await;
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn count_and_timer_triggers_execute_without_holding_flush_sweep() {
        let (state, coordinator, _dir) = fixture(1, 60).await;
        let store = pending(&state, "hot").await;
        // A reader permits the count predicate and preparation, but blocks
        // commit. The sweeper must return while the owned executor waits.
        let held = store.read().await;
        tokio::time::timeout(
            Duration::from_secs(2),
            sweeper::flush_pass_coordinated(
                vec![("hot".into(), store.clone())],
                Duration::from_secs(5),
                state.merge_slots.clone(),
                Some(state.clone()),
            ),
        )
        .await
        .unwrap();
        let e = running(&coordinator, "hot").await;
        tokio::time::timeout(
            Duration::from_secs(2),
            sweeper::merge_pass_coordinated(
                vec![("hot".into(), store.clone())],
                Duration::from_secs(5),
                Some(state.clone()),
            ),
        )
        .await
        .unwrap();
        assert_eq!(coordinator.get("hot").await.unwrap().unwrap().id, e.id);
        drop(held);
        tokio::time::timeout(Duration::from_secs(20), async {
            while store.read().await.pending_wal_generations().await.unwrap() != 0 {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        state.merge_executions.shutdown().await;
        assert!(coordinator.get("hot").await.unwrap().is_none());
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn busy_slots_do_not_create_unbounded_admissions_or_hide_other_flushes() {
        let (state, coordinator, _dir) = fixture(1, 60).await;
        let store = pending(&state, "hot").await;
        let held = state.acquire_merge_slot().await;
        for _ in 0..32 {
            assert!(matches!(
                spawn_background(state.clone(), "hot".into()).await,
                Err(AppError::Overloaded(_))
            ));
        }
        tokio::time::timeout(
            Duration::from_secs(2),
            sweeper::flush_pass_coordinated(
                vec![("hot".into(), store.clone())],
                Duration::from_secs(5),
                state.merge_slots.clone(),
                Some(state.clone()),
            ),
        )
        .await
        .unwrap();
        assert!(state.merge_executions.running.lock().await.is_empty());
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                if coordinator
                    .request_page(None)
                    .await
                    .unwrap()
                    .0
                    .iter()
                    .any(|r| r.target == "hot")
                {
                    break;
                }
                // Publication has a bounded metadata timeout. Model the next
                // periodic flush after a deferred publish while retaining the
                // occupied execution slot; retry must not need merge capacity.
                tokio::time::timeout(
                    Duration::from_secs(2),
                    sweeper::flush_pass_coordinated(
                        vec![("hot".into(), store.clone())],
                        Duration::from_secs(5),
                        state.merge_slots.clone(),
                        Some(state.clone()),
                    ),
                )
                .await
                .expect("metadata retry must not hold the flush sweep");
                assert!(state.merge_executions.running.lock().await.is_empty());
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .expect("saturated worker must still request master catchup");
        assert!(coordinator.get("hot").await.unwrap().is_none());
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            1
        );
        drop(held);
        assert_eq!(
            spawn_background(state.clone(), "hot".into())
                .await
                .unwrap()
                .await
                .unwrap()
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn another_tables_success_does_not_renew_a_stalled_executor() {
        let (state, coordinator, _dir) = fixture(2, 10).await;
        let slow = pending(&state, "slow").await;
        let fast = pending(&state, "fast").await;
        let held = slow.write().await;
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let commits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let healthy = tokio::spawn({
            let state = state.clone();
            let stop = stop.clone();
            let commits = commits.clone();
            async move {
                while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                    let reclaimed = spawn_background(state.clone(), "fast".into())
                        .await
                        .unwrap()
                        .await
                        .unwrap()
                        .unwrap();
                    assert!(reclaimed > 0);
                    let id = commits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let row =
                        rollout_record_from_add_request(&lance_context_api::AddRolloutRequest {
                            id: format!("row-{id}"),
                            rollout_id: "trajectory".into(),
                            ..Default::default()
                        });
                    fast.read().await.add(&[row]).await.unwrap();
                    fast.read().await.flush().await.unwrap();
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        });
        // Establish actual commit traffic before the idle window starts. Cold
        // index/storage setup can otherwise consume the entire short window on
        // a busy test host without ever exercising concurrent progress.
        tokio::time::timeout(Duration::from_secs(30), async {
            while commits.load(std::sync::atomic::Ordering::SeqCst) < 2 {
                assert!(!healthy.is_finished());
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        let stalled = spawn_background(state.clone(), "slow".into())
            .await
            .unwrap();
        running(&coordinator, "slow").await;
        let before = commits.load(std::sync::atomic::Ordering::SeqCst);
        let error = tokio::time::timeout(Duration::from_secs(30), stalled)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        // Real commits continue throughout the stalled table's idle window.
        assert!(!healthy.is_finished());
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        healthy.await.unwrap();
        assert!(commits.load(std::sync::atomic::Ordering::SeqCst) - before >= 2);
        assert!(error.contains("no-progress"), "{error}");
        assert!(coordinator.get("slow").await.unwrap().is_none());
        drop(held);
        assert_eq!(
            slow.read().await.pending_wal_generations().await.unwrap(),
            1
        );
        assert_eq!(state.merge_slots.as_ref().unwrap().available_permits(), 2);
    }
    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn restarted_worker_keeps_fencing_after_flag_rollback() {
        let (state, coordinator, dir) = fixture(1, 60).await;
        let store = pending(&state, "hot").await;
        assert!(state.merge_executions.fenced_mode().await.unwrap());
        let mut restarted = AppState::new_for_test(dir.path().to_path_buf()).await;
        restarted.merge_slots = Some(Arc::new(tokio::sync::Semaphore::new(1)));
        restarted.merge_executions = Executions::new(Some(coordinator.clone()), 3600);
        restarted.merge_executions.endpoint = Some("http://127.0.0.1:1".into());
        assert!(!restarted.merge_executions.rollout.fence_legacy);
        assert!(restarted.merge_executions.fenced_mode().await.unwrap());
        let restarted = Arc::new(restarted);
        let held = store.write().await;
        // The reopened worker has its own handle. Hold its lock to inspect the
        // actual HTTP admission before allowing storage work to run.
        let reopened = restarted.get_or_open_rollout_store("hot").await.unwrap();
        let blocked = reopened.write().await;
        let task = tokio::spawn(rollouts::merge_wal(
            State(restarted.clone()),
            axum::extract::Path("hot".into()),
        ));
        assert!(running(&coordinator, "hot").await.background);
        drop(blocked);
        drop(held);
        let Json(result) = task.await.unwrap().unwrap();
        assert_eq!(result.reclaimed, 1);
        assert!(coordinator.get("hot").await.unwrap().is_none());
        let missing_endpoint = Executions::new(Some(coordinator), 3600);
        assert!(
            missing_endpoint.fenced_mode().await.is_err(),
            "rollback must not silently restore legacy writes"
        );
    }
    #[tokio::test]
    #[ignore = "requires isolated local ETCD_TEST_ENDPOINTS"]
    async fn fenced_mode_blocks_untracked_worker_compaction_before_store_lock() {
        let (state, _coordinator, _dir) = fixture(1, 60).await;
        let store = pending(&state, "hot").await;
        let held = store.read().await;
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            rollouts::compact_rollout(
                State(state.clone()),
                axum::extract::Path("hot".into()),
                Json(serde_json::from_value(serde_json::json!({})).unwrap()),
            ),
        )
        .await
        .unwrap();
        assert!(matches!(result, Err(AppError::Overloaded(_))));
        drop(held);
        assert_eq!(
            store.read().await.pending_wal_generations().await.unwrap(),
            1
        );
    }
}
