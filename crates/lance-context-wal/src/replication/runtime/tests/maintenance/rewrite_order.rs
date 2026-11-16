use super::*;

#[test]
fn checkpoint_uses_native_file_headroom_before_a_slow_rewrite() -> Result<()> {
    check_rewrite_order(false)
}

#[test]
fn pressure_checkpoints_other_groups_before_a_due_slow_rewrite() -> Result<()> {
    check_rewrite_order(true)
}

fn check_rewrite_order(hold_first_checkpoint: bool) -> Result<()> {
    let executor = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let root = tempfile::tempdir()?;
    let c = executor.block_on(async {
        Cluster::open_with_storage(
            root.path(),
            Options::default(),
            Some(disk()),
            crate::raft_log::LogOptions {
                target_file_bytes: 8 << 20,
                purge_threshold_bytes: 8 << 20,
            },
        )
    })?;
    let records = executor.block_on(async {
        let mut records = Vec::new();
        for (group, bucket) in [(10, 0), (20, 1)] {
            c.elect(1, group).await?;
            let value = Batch::new(
                Uuid::from_u128(2),
                bucket,
                Uuid::from_u128(3),
                1,
                bytes::Bytes::from(vec![group as u8; 1 << 20]),
            );
            let index = commit(&c.handles[&1], group, value.clone()).await?;
            // Exact retries make the retained native log substantially larger
            // than the one accepted body that each checkpoint must preserve.
            for _ in 0..24 {
                ensure!(commit(&c.handles[&1], group, value.clone()).await? == index);
            }
            let through = c.handles[&1].observation(group)?.status.applied;
            wait_for(|| {
                c.handles
                    .values()
                    .all(|h| h.observation(group).unwrap().status.applied >= through)
            })
            .await?;
            records.push((group, index, through, value));
        }
        Ok::<_, anyhow::Error>(records)
    })?;
    let handle = &c.handles[&3];
    let store = executor.block_on(handle.local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let measured = handle.prepare_disk_measurement();
    let measured = executor.block_on(measured?)?.run()?;
    let budget = handle.disk.as_ref().unwrap();
    let room = if hold_first_checkpoint {
        512 << 10
    } else {
        12 << 20
    };
    // The first case still admits a maximum proposal, but lies inside the
    // native engine's two-file pressure window. The second makes log work due
    // during the first checkpoint, before the other group has been captured.
    budget.limit_data(usize::try_from(measured.accounted_disk_bytes)? + room)?;
    if !hold_first_checkpoint {
        drop(budget.reserve(8 << 20, true)?);
    }
    let blocked_sync = hold_first_checkpoint.then(|| store.block_sync_for_test());
    let paused_rewrite = store.pause_rewrite_sync_for_test();
    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let task = executor.spawn(maintenance::run(
        handle.clone(),
        vec![10, 20],
        maintenance::Options {
            checkpoint_interval_ms: 60_000,
            checkpoint_trailing_bytes: 128 << 10,
            log_interval_ms: 25,
            ..options()
        },
        progress.clone(),
        stopped,
    ));
    let built = if hold_first_checkpoint {
        executor.block_on(wait_for(|| {
            std::fs::read_dir(root.path().join("3"))
                .unwrap()
                .any(|entry| {
                    let entry = entry.unwrap();
                    entry.path().extension().is_some_and(|ext| ext == "pack")
                        && entry.metadata().unwrap().len() == 1 << 20
                })
        }))
    } else {
        Ok(())
    };
    if hold_first_checkpoint {
        std::thread::sleep(Duration::from_millis(75));
    }
    drop(blocked_sync);
    let reached_rewrite = paused_rewrite.wait();
    let cuts = records
        .iter()
        .map(|(group, _, _, _)| handle.observation(*group).map(|o| o.checkpoint_index))
        .collect::<Result<Vec<_>>>();
    // Always release the real native sync and join the scheduler on failure.
    stop.send(true)?;
    drop(paused_rewrite);
    executor.block_on(async {
        task.await??;
        c.shutdown().await
    })?;
    drop(store);
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        for (group, index, _, value) in &records {
            ensure!(node.read_accepted(*group, *index)? == Some(value.clone()));
            ensure!(node.status(*group)?.accepted_records == 1);
        }
    }
    built?;
    reached_rewrite.context("fixture did not reach native rewrite sync")?;
    let cuts = cuts?;
    eprintln!("checkpoint cuts before native rewrite sync: {cuts:?}");
    for ((group, _, through, _), cut) in records.iter().zip(cuts) {
        ensure!(
            cut >= *through,
            "group {group} checkpoint was delayed behind pressure rewriting: {cut} < {through}"
        );
    }
    ensure!(progress.lock().unwrap().attempts_failed == 0);
    Ok(())
}
