use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_pressure_checkpoints_before_age_without_rewrite_admission_failure() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = Cluster::open_with_storage(
        root.path(),
        Options::default(),
        Some(disk()),
        crate::raft_log::LogOptions {
            target_file_bytes: 64 << 10,
            purge_threshold_bytes: 64 << 10,
        },
    )?;
    c.elect(1, 10).await?;
    let value = Batch::new(
        Uuid::from_u128(2),
        0,
        Uuid::from_u128(3),
        1,
        bytes::Bytes::from(vec![7; 1 << 20]),
    );
    let index = commit(&c.handles[&1], 10, value.clone()).await?;
    for _ in 0..32 {
        ensure!(commit(&c.handles[&1], 10, value.clone()).await? == index);
    }
    let through = c.handles[&1].observation(10)?.status.applied;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= through)
    })
    .await?;
    let handle = &c.handles[&3];
    let measured = handle.prepare_disk_measurement()?.await?.run()?;
    ensure!(measured.accounted_disk_bytes > 16 << 20);
    let budget = handle.disk.as_ref().unwrap();
    // Reproduce the two different admission windows with real native log bytes:
    // new data lacks room while native rewriting still has its reserved headroom.
    // Lower only this test's follower data ceiling after building the log; no
    // outstanding reservation is released or converted into fictitious free space.
    budget.limit_data(16 << 20)?;
    ensure!(budget.reserve(1 << 20, true).is_err());
    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(maintenance::run(
        handle.clone(),
        vec![10],
        maintenance::Options {
            checkpoint_interval_ms: 60_000,
            checkpoint_trailing_bytes: 128 << 10,
            log_interval_ms: 25,
            ..options()
        },
        progress.clone(),
        stopped,
    ));
    let recovered = timeout(Duration::from_secs(3), async {
        loop {
            if handle.observation(10)?.checkpoint_index >= through
                && budget.reserve(1 << 20, true).is_ok()
            {
                return Ok::<_, anyhow::Error>(());
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    stop.send(true)?;
    task.await??;
    let final_progress = progress.lock().unwrap().clone();
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        ensure!(node.read_accepted(10, index)? == Some(value.clone()));
        ensure!(node.status(10)?.accepted_records == 1);
    }
    ensure!(
        final_progress.attempts_failed == 0,
        "native maintenance failed instead of reclaiming admitted rewrite work"
    );
    recovered.context("data admission remained blocked until the ordinary checkpoint age despite available rewrite headroom")??;
    ensure!(
        final_progress.checkpoints == 1,
        "pressure must not rebuild an already checkpointed cut"
    );
    Ok(())
}
