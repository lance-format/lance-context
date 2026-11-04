use super::*;
use bytes::Bytes;
use std::time::Instant;

async fn scheduled_checkpoint(
    handle: &Handle,
    through: u64,
    entries: u64,
    interval_ms: u64,
) -> Result<()> {
    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(maintenance::run(
        handle.clone(),
        vec![10],
        maintenance::Options {
            interval_ms: 25,
            checkpoint_entries: entries,
            checkpoint_interval_ms: interval_ms,
            checkpoint_trailing_bytes: 2 << 20,
            log_interval_ms: 25,
            scan_entries: 8,
        },
        progress.clone(),
        stopped,
    ));
    let covered = timeout(Duration::from_secs(3), async {
        while handle.observation(10)?.checkpoint_index < through {
            sleep(Duration::from_millis(5)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await;
    stop.send(true)?;
    task.await??;
    assert_eq!(progress.lock().unwrap().attempts_failed, 0);
    covered??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retained_tail_hints_do_not_bypass_checkpoint_cadence() -> Result<()> {
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
    let handle = &c.handles[&1];
    let mut records = Vec::new();
    for sequence in 1..=64 {
        let value = Batch::new(
            Uuid::from_u128(2),
            0,
            Uuid::from_u128(3),
            sequence,
            Bytes::from(vec![sequence as u8; 16 << 10]),
        );
        records.push((commit(handle, 10, value.clone()).await?, value));
    }
    let captured = handle.prepare_checkpoint(10, Uuid::new_v4())?.await?;
    let built = tokio::task::spawn_blocking(move || captured.run()).await??;
    handle.install_checkpoint_retaining(built, 2 << 20)?.await?;
    let mut hinted = false;
    for _ in 0..4 {
        let work = handle.prepare_log_maintenance()?.await?;
        hinted |= tokio::task::spawn_blocking(move || work.run())
            .await??
            .contains(&10);
    }
    assert!(
        hinted,
        "native maintenance did not produce the retained-tail hint"
    );

    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let started = Instant::now();
    let task = tokio::spawn(maintenance::run(
        handle.clone(),
        vec![10],
        maintenance::Options {
            interval_ms: 25,
            checkpoint_entries: 512,
            checkpoint_interval_ms: 30_000,
            checkpoint_trailing_bytes: 2 << 20,
            log_interval_ms: 25,
            scan_entries: 8,
        },
        progress.clone(),
        stopped,
    ));
    for round in 0..4 {
        let before = progress.lock().unwrap().log_passes;
        for sequence in (65 + round * 8)..=(72 + round * 8) {
            let value = Batch::new(
                Uuid::from_u128(2),
                0,
                Uuid::from_u128(3),
                sequence,
                Bytes::from(vec![sequence as u8; 16 << 10]),
            );
            records.push((commit(handle, 10, value.clone()).await?, value));
        }
        wait_for(|| progress.lock().unwrap().log_passes >= before + 3).await?;
    }
    stop.send(true)?;
    task.await??;
    let observed = progress.lock().unwrap().clone();
    assert!(started.elapsed() < Duration::from_secs(15));
    assert_eq!(observed.attempts_failed, 0);
    // Both configured triggers must still make progress with a retained tail:
    // first use the entry threshold, then only the elapsed-time threshold.
    for (sequence, entries, interval_ms) in [(97, 1, 30_000), (98, u64::MAX, 100)] {
        let value = Batch::new(
            Uuid::from_u128(2),
            0,
            Uuid::from_u128(3),
            sequence,
            Bytes::from(vec![sequence as u8; 16 << 10]),
        );
        let index = commit(handle, 10, value.clone()).await?;
        records.push((index, value));
        scheduled_checkpoint(handle, index, entries, interval_ms).await?;
    }
    let through = records.last().unwrap().0;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= through)
    })
    .await?;
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        for (index, expected) in &records {
            assert_eq!(node.read_accepted(10, *index)?, Some(expected.clone()));
        }
    }
    assert_eq!(
        observed.checkpoints, 0,
        "retained-tail hints bypassed both checkpoint entry and time thresholds"
    );
    Ok(())
}
