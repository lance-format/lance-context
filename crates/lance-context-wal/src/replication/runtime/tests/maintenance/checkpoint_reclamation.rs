use super::*;
use bytes::Bytes;

fn native_bytes(directory: &Path) -> Result<u64> {
    let mut bytes = 0;
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if entry
            .path()
            .extension()
            .is_some_and(|ext| ext == "raftlog" || ext == "rewrite")
        {
            bytes += entry.metadata()?.len();
        }
    }
    Ok(bytes)
}

// Real RF3 workers, checkpoint files and native reclamation. Hold the first
// checkpoint's sync long enough to make log maintenance due, then hold the
// second group's owner. Reclamation must not wait for that unrelated group.
#[test]
fn due_log_reclamation_precedes_another_groups_checkpoint() -> Result<()> {
    let executor = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let root = tempfile::tempdir()?;
    let c = executor.block_on(async {
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
        c.elect(1, 20).await?;
        Ok::<_, anyhow::Error>(c)
    })?;
    let mut records = Vec::new();
    let mut other_value = batch(1);
    other_value.meta.bucket = 1;
    let other = executor.block_on(async {
        for sequence in 1..=32 {
            let value = Batch::new(
                Uuid::from_u128(2),
                0,
                Uuid::from_u128(3),
                sequence,
                Bytes::from(vec![sequence as u8; 256 << 10]),
            );
            records.push((commit(&c.handles[&1], 10, value.clone()).await?, value));
        }
        let other = commit(&c.handles[&1], 20, other_value.clone()).await?;
        let through = records.last().unwrap().0;
        wait_for(|| {
            c.handles.values().all(|h| {
                h.observation(10).unwrap().status.applied >= through
                    && h.observation(20).unwrap().status.applied >= other
            })
        })
        .await?;
        Ok::<_, anyhow::Error>(other)
    })?;
    let handle = &c.handles[&3];
    let store =
        executor.block_on(handle.local(10, |node| Ok(node.groups[&10].raw.store().clone()))?)?;
    let (release_other, held_other) = std::sync::mpsc::channel::<()>();
    let (entered, entered_other) = tokio::sync::oneshot::channel();
    let other_work = handle.local(20, move |_| {
        let _ = entered.send(());
        let _ = held_other.recv();
        Ok(())
    })?;
    executor.block_on(entered_other)?;

    let blocked_sync = store.block_sync_for_test();
    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let task = executor.spawn(maintenance::run(
        handle.clone(),
        vec![10, 20],
        maintenance::Options {
            interval_ms: 100,
            checkpoint_entries: 1,
            checkpoint_interval_ms: 60_000,
            log_interval_ms: 100,
            ..options()
        },
        progress.clone(),
        stopped,
    ));
    let built = executor.block_on(wait_for(|| {
        std::fs::read_dir(root.path().join("3"))
            .unwrap()
            .any(|entry| {
                let entry = entry.unwrap();
                entry.path().extension().is_some_and(|ext| ext == "pack")
                    && entry.metadata().unwrap().len() == 8 << 20
            })
    }));
    // The sync guard prevents installation while the elapsed-time trigger
    // becomes due. This does not depend on how quickly the filesystem writes.
    std::thread::sleep(Duration::from_millis(150));
    let before = native_bytes(&root.path().join("3"))?;
    drop(blocked_sync);
    let reclaimed = executor.block_on(async {
        built?;
        wait_for(|| progress.lock().unwrap().checkpoints >= 1).await?;
        timeout(Duration::from_secs(2), async {
            while progress.lock().unwrap().log_passes == 0 {
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .context("due native reclamation waited for the next group's checkpoint")?;
        native_bytes(&root.path().join("3"))
    });
    // Always release the owner and join submitted work, including on the
    // negative control, before checking the result or reopening the engine.
    stop.send(true)?;
    drop(release_other);
    executor.block_on(async {
        other_work.await?;
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
        for (index, expected) in &records {
            assert_eq!(node.read_accepted(10, *index)?, Some(expected.clone()));
        }
        assert_eq!(node.read_accepted(20, other)?, Some(other_value.clone()));
    }
    let after = reclaimed?;
    eprintln!("native bytes before={before}, after={after}; all RF3 records reopened");
    assert!(
        after + (4 << 20) < before,
        "obsolete native bytes were not reclaimed"
    );
    assert_eq!(progress.lock().unwrap().attempts_failed, 0);
    Ok(())
}
