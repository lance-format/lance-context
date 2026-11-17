use super::*;

#[cfg(unix)]
mod checkpoint_crash;

mod checkpoint_peers;
mod checkpoint_queue;
mod follower_checkpoint;
mod producer_checkpoint;
mod ready_window;

fn executor() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?)
}

fn options() -> Options {
    Options {
        async_persistence: true,
        ..Options::default()
    }
}

async fn seeded(root: &Path) -> Result<(Cluster, u64)> {
    let c = Cluster::open(root, options())?;
    c.elect(1, 10).await?;
    let first = commit(&c.handles[&1], 10, batch(1)).await?;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= first))
    })
    .await?;
    Ok((c, first))
}

fn reopen(root: &Path, expected: &[(u64, u64)]) -> Result<()> {
    for id in 1..=3 {
        let node = Node::open(&root.join(id.to_string()), layout(), id, Limits::default())?;
        for &(index, sequence) in expected {
            ensure!(node.read_accepted(10, index)? == Some(batch(sequence)));
        }
    }
    Ok(())
}

#[test]
fn async_persistence_runtime_keeps_leader_live_without_early_ack() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let store = executor.block_on(c.handles[&1].local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let term = c.handles[&1].observation(10)?.status.term;
    let before = store.appended_index_for_test();
    let gate = store.block_sync_for_test();
    let held = executor.block_on(async {
        let pending = c.handles[&1].commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        wait_for(|| store.appended_index_for_test() > before).await?;
        let deadline = Instant::now() + Duration::from_millis(2500);
        while Instant::now() < deadline {
            ensure!(pending.0.is_empty(), "ACK preceded original native sync");
            for h in c.handles.values() {
                let observation = h.observation(10)?;
                ensure!(
                    !observation.stopped,
                    "worker stopped: {:?}",
                    observation.failure
                );
                ensure!(
                    observation.status.leader == 1 && observation.status.term == term,
                    "sync stall caused an election"
                );
            }
            let observation = c.handles[&1].observation(10)?;
            ensure!(
                observation.status.applied == first && observation.status.committed == first,
                "visible storage escaped the durable application/status boundary"
            );
            ensure!(
                observation.observed_at.elapsed() < Duration::from_millis(500),
                "owner event loop blocked"
            );
            sleep(Duration::from_millis(10)).await;
        }
        let old = timeout(
            Duration::from_millis(500),
            c.handles[&1].local(10, move |node| node.read_accepted(10, first))?,
        )
        .await??;
        ensure!(
            old == Some(batch(1)),
            "stable-prefix read failed while sync was held"
        );
        Ok::<_, anyhow::Error>(pending)
    });
    drop(gate);
    let verified = executor.block_on(async {
        let pending = held?;
        let result = timeout(Duration::from_secs(10), pending).await??;
        let Outcome::Committed(Applied::Accepted { index }) = result else {
            anyhow::bail!("unexpected append result {result:?}")
        };
        ensure!(commit(&c.handles[&1], 10, batch(2)).await? == index);
        wait_for(|| {
            c.handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= index))
        })
        .await?;
        let io = c.handles[&1]
            .observation(10)?
            .persistence
            .context("missing native-worker observation")?
            .io;
        ensure!(
            io.max_elapsed_micros >= 2_000_000,
            "moving I/O off poll hid the deliberate sync wait from observations"
        );
        ensure!(io.storage_wait.sync.max_elapsed_micros >= 2_000_000);
        Ok::<_, anyhow::Error>(index)
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    reopen(root.path(), &[(first, 1), (verified?, 2)])
}

#[test]
fn async_persistence_runtime_requires_synced_follower_quorum() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let a = executor.block_on(
        c.handles[&2].local(10, |n| Ok(n.groups[&10].raw.store().durable_log().clone()))?,
    )?;
    let b = executor.block_on(
        c.handles[&3].local(10, |n| Ok(n.groups[&10].raw.store().durable_log().clone()))?,
    )?;
    let before_a = a.appended_index_for_test();
    let before_b = b.appended_index_for_test();
    let gate_a = a.block_sync_for_test();
    let gate_b = b.block_sync_for_test();
    let pending = executor.block_on(async {
        let pending = c.handles[&1].commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        wait_for(|| {
            a.appended_index_for_test() > before_a && b.appended_index_for_test() > before_b
        })
        .await?;
        sleep(Duration::from_millis(200)).await;
        ensure!(
            pending.0.is_empty(),
            "cached follower data counted as durable quorum"
        );
        ensure!(c
            .handles
            .values()
            .all(|h| h.observation(10).is_ok_and(|o| o.status.applied == first)));
        Ok::<_, anyhow::Error>(pending)
    });
    drop(gate_a);
    let verified = executor.block_on(async {
        let result = timeout(Duration::from_secs(10), pending?).await??;
        let Outcome::Committed(Applied::Accepted { index }) = result else {
            anyhow::bail!("unexpected append result {result:?}")
        };
        ensure!(
            c.handles[&3].observation(10)?.status.applied == first,
            "blocked follower applied early"
        );
        Ok::<_, anyhow::Error>(index)
    });
    drop(gate_b);
    let verified = executor.block_on(async {
        let index = verified?;
        wait_for(|| {
            c.handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= index))
        })
        .await?;
        Ok::<_, anyhow::Error>(index)
    });
    let stopped = executor.block_on(c.shutdown());
    drop(a);
    drop(b);
    stopped?;
    reopen(root.path(), &[(first, 1), (verified?, 2)])
}

#[test]
fn async_persistence_checkpoint_waits_for_original_write_and_keeps_its_reply() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let prepared = executor.block_on(async {
        let task = c.handles[&1]
            .prepare_checkpoint(10, uuid::Uuid::new_v4())?
            .await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(
        c.handles[&1].local(10, |n| Ok(n.groups[&10].raw.store().durable_log().clone()))?,
    )?;
    let before = store.appended_index_for_test();
    let gate = store.block_sync_for_test();
    let waiting = executor.block_on(async {
        let pending = c.handles[&1].commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        wait_for(|| store.appended_index_for_test() > before).await?;
        let checkpoint = c.handles[&1].install_checkpoint(prepared)?;
        sleep(Duration::from_millis(100)).await;
        ensure!(
            pending.0.is_empty(),
            "append escaped checkpoint barrier before sync"
        );
        ensure!(
            checkpoint.0.is_empty(),
            "checkpoint completed ahead of original write"
        );
        Ok::<_, anyhow::Error>((pending, checkpoint))
    });
    drop(gate);
    let verified = executor.block_on(async {
        let (pending, checkpoint) = waiting?;
        timeout(Duration::from_secs(10), checkpoint).await??;
        let result = timeout(Duration::from_secs(10), pending).await??;
        let Outcome::Committed(Applied::Accepted { index }) = result else {
            anyhow::bail!("lost original completion {result:?}")
        };
        ensure!(c.handles[&1].observation(10)?.checkpoint_index == first);
        wait_for(|| {
            c.handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= index))
        })
        .await?;
        Ok::<_, anyhow::Error>(index)
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    reopen(root.path(), &[(first, 1), (verified?, 2)])
}

#[test]
fn async_persistence_native_sync_failure_never_returns_success() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let store = executor.block_on(
        c.handles[&1].local(10, |n| Ok(n.groups[&10].raw.store().durable_log().clone()))?,
    )?;
    store.fail_next_sync_for_test();
    let checked = executor.block_on(async {
        let pending = c.handles[&1].commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        let result = timeout(Duration::from_secs(10), pending).await?;
        ensure!(
            result.is_err() || matches!(result, Ok(Outcome::Unknown)),
            "failed sync produced a completion: {result:?}"
        );
        wait_for(|| c.handles[&1].observation(10).is_ok_and(|o| o.stopped)).await?;
        let observation = c.handles[&1].observation(10)?;
        ensure!(
            observation
                .failure
                .as_deref()
                .is_some_and(|e| e.contains("engine write panicked")),
            "test did not observe the original injected sync failure: {:?}",
            observation.failure
        );
        ensure!(observation.status.applied == first);
        ensure!(
            observation
                .persistence
                .context("missing failed writer observation")?
                .io
                .failed
                == 1
        );
        Ok::<_, anyhow::Error>(())
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    ensure!(stopped.is_err(), "shutdown hid the native write failure");
    checked?;
    reopen(root.path(), &[(first, 1)])
}

#[test]
fn async_persistence_window_backpressures_then_drains_without_lost_acks() -> Result<()> {
    full_window(Duration::ZERO)
}

#[test]
fn async_persistence_full_window_keeps_quorum_live_during_slow_disk() -> Result<()> {
    full_window(Duration::from_millis(2500))
}

fn full_window(sustain_for: Duration) -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let store = executor.block_on(
        c.handles[&1].local(10, |n| Ok(n.groups[&10].raw.store().durable_log().clone()))?,
    )?;
    let gate = store.block_sync_for_test();
    let filled = executor.block_on(async {
        let handle = &c.handles[&1];
        let mut requests = Vec::new();
        let mut full = false;
        for offset in 0..128 {
            let mut input = batch(offset + 2);
            input.meta.producer = uuid::Uuid::from_u128(1000 + u128::from(offset));
            input.meta.sequence = 1;
            let request = handle.commit(Command::Append {
                group: 10,
                batch: input.clone(),
            })?;
            requests.push((input, request));
            wait_for(|| {
                handle
                    .observation(10)
                    .is_ok_and(|o| o.status.pending >= requests.len())
                    || requests.last().is_some_and(|(_, r)| !r.0.is_empty())
            })
            .await?;
            if requests
                .last()
                .is_some_and(|(_, request)| !request.0.is_empty())
            {
                let (_, request) = requests.pop().unwrap();
                ensure!(
                    request.await.is_err(),
                    "blocked native sync produced a successful reply"
                );
            }
            let usage = timeout(
                Duration::from_millis(500),
                handle.local(10, |node| {
                    let super::super::super::LogView::Buffered(log) = node.groups[&10].raw.store()
                    else {
                        anyhow::bail!("missing buffered storage")
                    };
                    Ok(log.usage())
                })?,
            )
            .await??;
            ensure!(usage.readies <= 128 && usage.bytes <= 132 * 1024 * 1024);
            ensure!(handle.observation(10)?.status.applied == first);
            if usage.readies == 128 {
                eprintln!(
                    "async window full: readies={}, entries={}, bytes={}, admitted={}",
                    usage.readies,
                    usage.entries,
                    usage.bytes,
                    requests.len()
                );
                full = true;
                break;
            }
        }
        ensure!(full, "test did not reach the actual pending Ready bound");
        for _ in 0..8 {
            let request = handle.commit(Command::Append {
                group: 10,
                batch: batch(2),
            })?;
            let refused = timeout(Duration::from_millis(500), request).await?;
            ensure!(
                refused.is_err(),
                "full window did not refuse before mutation: {refused:?}"
            );
            ensure!(
                !handle.observation(10)?.stopped,
                "window pressure fenced the runtime"
            );
        }
        let original = handle.observation(10)?.status;
        let held_until = Instant::now() + sustain_for;
        while Instant::now() < held_until {
            for (id, peer) in c.handles.iter() {
                let view = peer.observation(10)?;
                ensure!(
                    view.status.leader == 1 && view.status.term == original.term,
                    "full persistence window changed leadership: node={id}, initial={original:?}, observed={:?}, persistence={:?}, rejections={:?}",
                    view.status, view.persistence, peer.peer_rejections()
                );
            }
            ensure!(
                requests.iter().all(|(_, request)| request.0.is_empty()),
                "full-window test observed an ACK before the original native sync"
            );
            sleep(Duration::from_millis(10)).await;
        }
        Ok::<_, anyhow::Error>(requests)
    });
    drop(gate);
    let checked = executor.block_on(async {
        let mut expected = vec![(first, batch(1))];
        for (input, request) in filled? {
            let result = timeout(Duration::from_secs(10), request).await??;
            let Outcome::Committed(Applied::Accepted { index }) = result else {
                anyhow::bail!("admitted original did not complete: {result:?}")
            };
            expected.push((index, input));
        }
        let last = expected.iter().map(|(index, _)| *index).max().unwrap();
        wait_for(|| {
            c.handles.values().all(|h| {
                h.observation(10)
                    .is_ok_and(|o| o.status.applied >= last && o.status.pending == 0)
            })
        })
        .await?;
        let usage = c.handles[&1]
            .local(10, |node| {
                let super::super::super::LogView::Buffered(log) = node.groups[&10].raw.store()
                else {
                    anyhow::bail!("missing buffered storage")
                };
                Ok(log.usage())
            })?
            .await?;
        // An empty heartbeat Ready may be in flight; no payload may remain.
        ensure!(usage.entries == 0, "payload window did not drain");
        Ok::<_, anyhow::Error>(expected)
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        for (index, input) in checked
            .as_ref()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
        {
            ensure!(node.read_accepted(10, *index)? == Some(input.clone()));
        }
    }
    Ok(())
}

#[test]
fn async_persistence_full_window_still_steps_down_without_quorum() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let store = executor.block_on(c.handles[&1].local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let gate = store.block_sync_for_test();
    let checked = executor.block_on(async {
        let handle = &c.handles[&1];
        let mut requests = Vec::new();
        for offset in 0..128 {
            let mut input = batch(offset + 2);
            input.meta.producer = uuid::Uuid::from_u128(1000 + u128::from(offset));
            input.meta.sequence = 1;
            requests.push(handle.commit(Command::Append {
                group: 10,
                batch: input,
            })?);
            wait_for(|| {
                handle
                    .observation(10)
                    .is_ok_and(|o| o.status.pending >= requests.len())
                    || requests.last().is_some_and(|r| !r.0.is_empty())
            })
            .await?;
            if let Some(request) = requests.pop_if(|r| !r.0.is_empty()) {
                ensure!(request.await.is_err());
            }
            if handle.observation(10)?.persistence.unwrap().pending_readies == 128 {
                break;
            }
        }
        ensure!(handle.observation(10)?.persistence.unwrap().pending_readies == 128);
        c.isolated.store(1, Ordering::Release);
        timeout(
            Duration::from_secs(3),
            wait_for(|| handle.observation(10).is_ok_and(|o| o.status.leader != 1)),
        )
        .await??;
        let view = handle.observation(10)?;
        ensure!(!view.stopped, "worker stopped instead of losing quorum");
        ensure!(
            view.persistence.unwrap().pending_readies == 128,
            "test no longer held the full native persistence window"
        );
        for mut request in requests {
            if let Ok(done) = request.0.try_recv() {
                ensure!(
                    !matches!(done.result, Ok(Outcome::Committed(_))),
                    "commit ACK escaped blocked native persistence"
                );
            }
        }
        Ok::<_, anyhow::Error>(())
    });
    drop(gate);
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    checked?;
    reopen(root.path(), &[(first, 1)])
}

#[test]
fn async_persistence_checkpoint_sync_stall_keeps_leader_live() -> Result<()> {
    checkpoint_stall(false)
}

#[test]
fn async_persistence_checkpoint_bounds_admission_during_continuous_requests() -> Result<()> {
    checkpoint_stall(true)
}

fn checkpoint_stall(continuous_requests: bool) -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let prepared = executor.block_on(async {
        let task = c.handles[&1]
            .prepare_checkpoint(10, uuid::Uuid::new_v4())?
            .await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(
        c.handles[&1].local(10, |n| Ok(n.groups[&10].raw.store().durable_log().clone()))?,
    )?;
    let term = c.handles[&1].observation(10)?.status.term;
    let before = store.appended_index_for_test();
    let gate = store.block_sync_for_test();
    let waiting = executor.block_on(async {
        let pending = c.handles[&1].commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        wait_for(|| store.appended_index_for_test() > before).await?;
        let checkpoint = c.handles[&1].install_checkpoint(prepared)?;
        let deadline = Instant::now() + Duration::from_millis(2500);
        let mut maximum_staleness = Duration::ZERO;
        let mut first_leader_change = None;
        let mut refused = 0;
        let mut queued = Vec::new();
        while Instant::now() < deadline {
            ensure!(pending.0.is_empty(), "ACK preceded held native sync");
            ensure!(checkpoint.0.is_empty(), "checkpoint preceded held native sync");
            for (id, handle) in c.handles.iter() {
                let observation = handle.observation(10)?;
                maximum_staleness = maximum_staleness.max(observation.observed_at.elapsed());
                if (observation.status.leader != 1 || observation.status.term != term)
                    && first_leader_change.is_none()
                {
                    first_leader_change = Some((*id, observation.status.leader, observation.status.term));
                }
            }
            if continuous_requests {
                // Control admission precedes this data request. All subsequent
                // writes retain existing data credits without adding Ready
                // work. Once full, the original request limit rejects callers.
                match c.handles[&1].commit(Command::Append {
                    group: 10,
                    batch: batch(queued.len() as u64 + 3),
                }) {
                    Ok(next) => queued.push(next),
                    Err(error) => {
                        ensure!(matches!(error.downcast_ref(), Some(SubmissionFailure::RequestCredits)),
                            "unexpected checkpoint admission error: {error}");
                        refused += 1;
                    }
                }
                ensure!(queued.iter().all(|request| request.0.is_empty()),
                    "checkpoint drain completed an admitted request prematurely");
            }
            sleep(Duration::from_millis(10)).await;
        }
        eprintln!(
            "checkpoint_stall_observation: hold_ms=2500 max_owner_age_ms={} first_leader_change={:?}",
            maximum_staleness.as_millis(), first_leader_change
        );
        ensure!(
            maximum_staleness < Duration::from_millis(500),
            "checkpoint sync barrier blocked the owner event loop: max_age_ms={} first_leader_change={:?}",
            maximum_staleness.as_millis(), first_leader_change
        );
        ensure!(first_leader_change.is_none(), "checkpoint sync barrier caused leader change");
        ensure!(
            pending.0.is_empty(),
            "append escaped checkpoint barrier before sync"
        );
        ensure!(
            checkpoint.0.is_empty(),
            "checkpoint completed ahead of original write"
        );
        ensure!(!continuous_requests || refused >= 10);
        ensure!(!continuous_requests || queued.len() + 1 == options().data_requests);
        Ok::<_, anyhow::Error>((pending, checkpoint, queued))
    });
    drop(gate);
    let mut accepted = vec![(first, 1)];
    let verified = executor.block_on(async {
        let (pending, checkpoint, queued) = waiting?;
        timeout(Duration::from_secs(10), checkpoint).await??;
        let result = timeout(Duration::from_secs(10), pending).await??;
        let Outcome::Committed(Applied::Accepted { index }) = result else {
            anyhow::bail!("lost original completion {result:?}")
        };
        accepted.push((index, 2));
        for (offset, request) in queued.into_iter().enumerate() {
            let result = timeout(Duration::from_secs(10), request).await??;
            let Outcome::Committed(Applied::Accepted { index }) = result else {
                anyhow::bail!("lost queued completion {result:?}")
            };
            accepted.push((index, offset as u64 + 3));
        }
        ensure!(c.handles[&1].observation(10)?.checkpoint_index == first);
        let last = accepted.last().unwrap().0;
        wait_for(|| {
            c.handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= last))
        })
        .await?;
        Ok::<_, anyhow::Error>(())
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    reopen(root.path(), &accepted)?;
    verified
}

#[test]
fn async_persistence_checkpoint_shutdown_retains_original_owners_until_sync() -> Result<()> {
    checkpoint_interrupted(false)
}

#[test]
fn async_persistence_checkpoint_native_failure_cannot_complete_successfully() -> Result<()> {
    checkpoint_interrupted(true)
}

fn checkpoint_interrupted(fail_sync: bool) -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let handle = c.handles[&1].clone();
    let prepared = executor.block_on(async {
        let task = handle.prepare_checkpoint(10, uuid::Uuid::new_v4())?.await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(handle.local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let before = store.appended_index_for_test();
    let gate = store.block_sync_for_test();
    let waiting = executor.block_on(async {
        let append = handle.commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        wait_for(|| store.appended_index_for_test() > before).await?;
        let checkpoint = handle.install_checkpoint(prepared)?;
        // A following control round proves that the queued checkpoint has
        // yielded the owner, without releasing its original image admission.
        let second = handle.prepare_checkpoint(10, uuid::Uuid::new_v4())?;
        ensure!(timeout(Duration::from_millis(500), second)
            .await?
            .as_ref()
            .err()
            .is_some_and(|error| error
                .to_string()
                .contains("node image builder/receiver busy")));
        ensure!(checkpoint.0.is_empty() && append.0.is_empty());
        ensure!(handle.admission().control.used_requests == 1);
        let queued = handle.commit(Command::Append {
            group: 10,
            batch: batch(3),
        })?;
        if fail_sync {
            store.fail_next_sync_for_test();
        } else {
            for runtime in &c.runtimes {
                runtime.stop();
            }
        }
        Ok::<_, anyhow::Error>((append, checkpoint, queued))
    });
    // The fault/stop is applied before releasing the original native write.
    drop(gate);
    let observed_failure = if fail_sync && waiting.is_ok() {
        executor.block_on(wait_for(|| handle.observation(10).is_ok_and(|o| o.stopped)))
    } else {
        Ok(())
    };
    let stopped = executor.block_on(c.shutdown());
    let checked = executor.block_on(async {
        let (append, checkpoint, queued) = waiting?;
        ensure!(timeout(Duration::from_secs(5), checkpoint).await?.is_err());
        let result = timeout(Duration::from_secs(5), append).await?;
        ensure!(result.is_err() || matches!(result, Ok(Outcome::Unknown)));
        let result = timeout(Duration::from_secs(5), queued).await?;
        ensure!(result.is_err() || matches!(result, Ok(Outcome::Unknown)));
        ensure!(handle.admission().control.used_requests == 0);
        ensure!(handle.admission().data.used_requests == 0);
        ensure!(handle.observation(10)?.stopped);
        Ok::<_, anyhow::Error>(())
    });
    drop(store);
    checked?;
    observed_failure?;
    if fail_sync {
        ensure!(stopped.is_err(), "shutdown hid the original native failure");
        ensure!(handle
            .observation(10)?
            .failure
            .as_deref()
            .is_some_and(|error| error.contains("engine write panicked")));
    } else {
        stopped?;
    }
    // Reopen proves that shutdown joined the original writer and released the
    // engine/image owners; the already acknowledged prefix is still readable.
    reopen(root.path(), &[(first, 1)])
}

#[test]
fn async_persistence_checkpoint_metadata_sync_keeps_leader_and_reads_live() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let prepared = executor.block_on(async {
        let task = c.handles[&1]
            .prepare_checkpoint(10, uuid::Uuid::new_v4())?
            .await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(c.handles[&1].local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let term = c.handles[&1].observation(10)?.status.term;
    let gate = store.block_sync_for_test();
    let waiting = executor.block_on(async {
        let checkpoint = c.handles[&1].install_checkpoint(prepared)?;
        // Prove that it is the checkpoint metadata's own sync being held,
        // rather than merely a pending Ready preceding its installation.
        wait_for(|| store.appended_checkpoint_for_test().ok().flatten() == Some(first)).await?;
        let read = c.handles[&1].local(10, move |node| node.read_accepted(10, first))?;
        let append = c.handles[&1].commit(Command::Append { group: 10, batch: batch(2) })?;
        wait_for(|| c.handles[&1].observation(10).is_ok_and(|o| o.status.pending == 1)).await?;
        let deadline = Instant::now() + Duration::from_millis(2500);
        let mut maximum_staleness = Duration::ZERO;
        let mut first_leader_change = None;
        while Instant::now() < deadline {
            ensure!(checkpoint.0.is_empty(), "checkpoint ACK preceded its metadata sync");
            ensure!(append.0.is_empty(), "suffix ACK preceded checkpoint metadata sync");
            for (id, handle) in c.handles.iter() {
                let view = handle.observation(10)?;
                maximum_staleness = maximum_staleness.max(view.observed_at.elapsed());
                if (view.status.leader != 1 || view.status.term != term) && first_leader_change.is_none() {
                    first_leader_change = Some((*id, view.status.leader, view.status.term));
                }
            }
            ensure!(c.handles[&1].observation(10)?.checkpoint_index < first,
                "durable checkpoint observation advanced before native sync");
            sleep(Duration::from_millis(10)).await;
        }
        eprintln!("checkpoint_metadata_stall: hold_ms=2500 max_owner_age_ms={} first_leader_change={:?} read_completed={}",
            maximum_staleness.as_millis(), first_leader_change, !read.0.is_empty());
        ensure!(maximum_staleness < Duration::from_millis(500), "checkpoint metadata sync blocked owner");
        ensure!(first_leader_change.is_none(), "checkpoint metadata sync changed leader visibility");
        ensure!(!read.0.is_empty(), "stable-prefix read waited for checkpoint metadata sync");
        ensure!(read.await? == Some(batch(1)));
        Ok::<_, anyhow::Error>((checkpoint, append))
    });
    drop(gate);
    let verified = executor.block_on(async {
        let (checkpoint, append) = waiting?;
        timeout(Duration::from_secs(10), checkpoint).await??;
        let index = index(timeout(Duration::from_secs(10), append).await??);
        wait_for(|| {
            c.handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= index))
        })
        .await?;
        ensure!(c.handles[&1].observation(10)?.checkpoint_index == first);
        Ok::<_, anyhow::Error>(index)
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    reopen(root.path(), &[(first, 1), (verified?, 2)])
}

#[test]
fn async_persistence_checkpoint_full_queue_shutdown_joins_original_metadata_write() -> Result<()> {
    checkpoint_metadata_interrupted(false)
}

#[test]
fn async_persistence_checkpoint_metadata_failure_fences_queued_suffix() -> Result<()> {
    checkpoint_metadata_interrupted(true)
}

fn checkpoint_metadata_interrupted(fail_sync: bool) -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    let handle = c.handles[&1].clone();
    let prepared = executor.block_on(async {
        let task = handle.prepare_checkpoint(10, uuid::Uuid::new_v4())?.await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(handle.local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let gate = store.block_sync_for_test();
    let held = executor.block_on(async {
        let checkpoint = handle.install_checkpoint(prepared)?;
        wait_for(|| store.appended_checkpoint_for_test().ok().flatten() == Some(first)).await?;
        let mut requests = Vec::new();
        for offset in 0..128 {
            let mut input = batch(offset + 2);
            input.meta.producer = uuid::Uuid::from_u128(1000 + u128::from(offset));
            input.meta.sequence = 1;
            requests.push(handle.commit(Command::Append {
                group: 10,
                batch: input,
            })?);
            wait_for(|| {
                handle
                    .observation(10)
                    .is_ok_and(|o| o.status.pending >= requests.len())
                    || requests.last().is_some_and(|request| !request.0.is_empty())
            })
            .await?;
            if let Some(request) = requests.pop_if(|request| !request.0.is_empty()) {
                ensure!(
                    request.await.is_err(),
                    "checkpoint-blocked write completed successfully"
                );
            }
            let io = handle
                .observation(10)?
                .persistence
                .context("missing persistence")?;
            ensure!(io.pending_checkpoint && io.pending_readies < 128);
            if io.pending_readies == 127 {
                break;
            }
        }
        let io = handle.observation(10)?.persistence.unwrap();
        ensure!(
            io.pending_checkpoint && io.pending_readies == 127,
            "did not fill original native queue"
        );
        ensure!(checkpoint.0.is_empty() && requests.iter().all(|request| request.0.is_empty()));
        let read = timeout(
            Duration::from_millis(500),
            handle.local(10, move |node| node.read_accepted(10, first))?,
        )
        .await??;
        ensure!(
            read == Some(batch(1)),
            "full native queue lost readable checkpoint prefix"
        );
        ensure!(handle.admission().control.used_requests == 1);
        if fail_sync {
            store.fail_next_sync_for_test();
        } else {
            for runtime in &c.runtimes {
                runtime.stop();
            }
        }
        Ok::<_, anyhow::Error>((checkpoint, requests))
    });
    drop(gate);
    let observed_failure = if fail_sync && held.is_ok() {
        executor.block_on(wait_for(|| handle.observation(10).is_ok_and(|o| o.stopped)))
    } else {
        Ok(())
    };
    // Joining would deadlock if the checkpoint were allowed to overfill the
    // original completion queue alongside 128 Ready tasks during shutdown.
    let stopped = executor.block_on(async { timeout(Duration::from_secs(10), c.shutdown()).await });
    let verified = executor.block_on(async {
        let (checkpoint, requests) = held?;
        ensure!(timeout(Duration::from_secs(5), checkpoint).await?.is_err());
        for request in requests {
            let result = timeout(Duration::from_secs(5), request).await?;
            ensure!(result.is_err() || matches!(result, Ok(Outcome::Unknown)));
        }
        ensure!(
            handle.admission().data.used_requests == 0
                && handle.admission().control.used_requests == 0
        );
        Ok::<_, anyhow::Error>(())
    });
    drop(store);
    observed_failure?;
    verified?;
    if fail_sync {
        ensure!(stopped?.is_err(), "shutdown hid checkpoint sync failure");
    } else {
        stopped??;
    }
    reopen(root.path(), &[(first, 1)])
}

#[test]
fn async_persistence_checkpoint_transfers_readable_image_before_metadata_receipt() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let (c, first) = executor.block_on(seeded(root.path()))?;
    c.isolated.store(3, Ordering::Release);
    let second = executor.block_on(commit(&c.handles[&1], 10, batch(2)))?;
    let prepared = executor.block_on(async {
        let task = c.handles[&1]
            .prepare_checkpoint(10, uuid::Uuid::new_v4())?
            .await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(c.handles[&1].local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let gate = store.block_sync_for_test();
    let held = executor.block_on(async {
        let checkpoint = c.handles[&1].install_checkpoint(prepared)?;
        wait_for(|| store.appended_checkpoint_for_test().ok().flatten() == Some(second)).await?;
        ensure!(c.handles[&3].observation(10)?.status.applied < second);
        c.isolated.store(0, Ordering::Release);
        wait_for(|| {
            c.snapshot_transfers.load(Ordering::Acquire) > 0
                && c.handles[&3]
                    .observation(10)
                    .is_ok_and(|o| o.status.applied >= second)
        })
        .await?;
        // The sender's descriptor has not synced, but its image contains only
        // previously acknowledged state. The receiver syncs its own descriptor.
        ensure!(checkpoint.0.is_empty());
        ensure!(c.handles[&1].observation(10)?.checkpoint_index < second);
        for (index, sequence) in [(first, 1), (second, 2)] {
            let value = timeout(
                Duration::from_millis(500),
                c.handles[&3].local(10, move |node| node.read_accepted(10, index))?,
            )
            .await??;
            ensure!(
                value == Some(batch(sequence)),
                "transferred image lost payload"
            );
        }
        Ok::<_, anyhow::Error>(checkpoint)
    });
    drop(gate);
    let verified = executor.block_on(async {
        timeout(Duration::from_secs(10), held?).await??;
        ensure!(commit(&c.handles[&1], 10, batch(2)).await? == second);
        Ok::<_, anyhow::Error>(())
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    verified?;
    reopen(root.path(), &[(first, 1), (second, 2)])
}
