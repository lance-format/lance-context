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
    for _ in 0..384 {
        ensure!(commit(&c.handles[&1], 10, value.clone()).await? == index);
    }
    let through = c.handles[&1].observation(10)?.status.applied;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= through)
    })
    .await?;
    let measured = c.handles[&3].prepare_disk_measurement()?.await?.run()?;
    let budget = c.handles[&3].disk.as_ref().unwrap();
    // Real admission leaves room for the small captured application image, but
    // not the retained native entries plus segment allowance. Do not enlarge the
    // budget or release this reservation until the scheduler checkpoints.
    let held = budget.reserve(
        usize::try_from(measured.max_log_bytes - measured.accounted_disk_bytes - (258 << 20))?,
        false,
    )?;
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
