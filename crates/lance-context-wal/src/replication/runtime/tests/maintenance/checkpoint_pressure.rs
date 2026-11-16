use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rewrite_pressure_checkpoints_before_age_without_rebuilding_the_same_cut() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = Cluster::open_with_disk(root.path(), Options::default(), Some(disk()))?;
    c.elect(1, 10).await?;
    let value = Batch::new(
        Uuid::from_u128(2),
        0,
        Uuid::from_u128(3),
        1,
        bytes::Bytes::from(vec![7; 1 << 20]),
    );
    let index = commit(&c.handles[&1], 10, value.clone()).await?;
    // Exact retries enter Raft but leave one accepted record. This creates real
    // native rewrite work while the next application image stays small.
    for _ in 0..352 {
        ensure!(commit(&c.handles[&1], 10, value.clone()).await? == index);
    }
    let through = c.handles[&1].observation(10)?.status.applied;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= through)
    })
    .await?;
    // Seed the independent rewrite queue with real native passes. A later pass
    // must reserve that live queue as well as its bounded append prefix. None
    // of these exact retries has been covered by an application checkpoint.
    let budget = c.handles[&3].disk.as_ref().unwrap();
    let pressure = budget.reserve(128 << 20, false)?;
    for _ in 0..2 {
        let rewritten = c.handles[&3].prepare_log_maintenance()?.await?;
        tokio::task::spawn_blocking(move || rewritten.run()).await??;
    }
    drop(pressure);
    let rewrite_bytes = std::fs::read_dir(root.path().join("3"))?
        .map(|entry| {
            let entry = entry?;
            if entry.path().extension().is_some_and(|ext| ext == "rewrite") {
                Ok(entry.metadata()?.len())
            } else {
                Ok(0)
            }
        })
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .sum::<u64>();
    ensure!(
        rewrite_bytes > 128 << 20,
        "fixture did not seed live rewrite data"
    );
    let measured = c.handles[&3].prepare_disk_measurement()?.await?.run()?;
    // Leave the required 256 MiB control headroom plus 2 MiB for the small
    // application image. The independent rewrite queue plus bounded append
    // prefix and segment allowance must not fit. Keep this reservation until
    // the scheduler checkpoints; reducing it below the control headroom would
    // also block the image and could not exercise checkpoint-driven recovery.
    let held = budget.reserve(
        usize::try_from(measured.max_log_bytes - measured.accounted_disk_bytes - (258 << 20))?,
        false,
    )?;
    // Isolate a failed rewrite reservation: retained bytes must still lie
    // outside the proactive two-file checkpoint window. A temporary reservation
    // alone must not bypass the normal checkpoint cadence.
    ensure!(!c.handles[&3].data_checkpoint_pressure());
    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(maintenance::run(
        c.handles[&3].clone(),
        vec![10],
        maintenance::Options {
            checkpoint_interval_ms: 60_000,
            checkpoint_trailing_bytes: 128 << 10,
            log_interval_ms: 100,
            ..options()
        },
        progress.clone(),
        stopped,
    ));
    let checked: Result<()> = async {
        failure_has_occurred(&progress).await?;
        ensure!(progress.lock().unwrap().log_passes == 0);
        timeout(Duration::from_secs(2), async {
            loop {
                if c.handles[&3].observation(10)?.checkpoint_index >= through {
                    return Ok::<_, anyhow::Error>(());
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .context("rewrite pressure did not checkpoint before the 60 second age")??;
        let count = progress.lock().unwrap().checkpoints;
        ensure!(count == 1);
        sleep(Duration::from_millis(250)).await;
        ensure!(
            progress.lock().unwrap().checkpoints == count,
            "pressure rebuilt an already covered cut"
        );
        ensure!(!c.handles[&3].observation(10)?.stopped);
        Ok(())
    }
    .await;
    drop(held);
    let reclaimed = wait_for(|| progress.lock().unwrap().log_passes > 0).await;
    stop.send(true)?;
    task.await??;
    c.shutdown().await?;
    checked?;
    reclaimed?;
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
    Ok(())
}
