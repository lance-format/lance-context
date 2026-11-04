use super::*;
use crate::replication::checkpoint::{Image, Part};
use bytes::Bytes;

async fn checkpoint(handle: &Handle, id: Uuid) -> Result<Arc<Image>> {
    let captured = handle.prepare_checkpoint(10, id)?.await?;
    let built = tokio::task::spawn_blocking(move || captured.run()).await??;
    let image = built.image();
    handle.install_checkpoint(built)?.await?;
    Ok(image)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retired_checkpoint_releases_admission_before_next_tick_without_losing_live_reader(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = Cluster::open_with_disk(root.path(), Options::default(), Some(disk()))?;
    c.elect(1, 10).await?;
    let handle = &c.handles[&1];
    let first = Batch::new(
        Uuid::from_u128(2),
        0,
        Uuid::from_u128(3),
        1,
        Bytes::from(vec![7; 3 << 20]),
    );
    let first_index = commit(handle, 10, first.clone()).await?;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= first_index)
    })
    .await?;
    let pinned_id = Uuid::new_v4();
    let pinned = checkpoint(handle, pinned_id).await?;
    let second_index = commit(handle, 10, batch(2)).await?;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= second_index)
    })
    .await?;
    let retired_id = Uuid::new_v4();
    let retired = checkpoint(handle, retired_id).await?;
    let obsolete_bytes = retired.bytes(Part::Payload);
    drop(retired);
    let maintenance = handle.prepare_log_maintenance()?.await?;
    tokio::task::spawn_blocking(move || maintenance.run()).await??;
    let used = handle.prepare_disk_measurement()?.await?.run()?;
    let budget = handle.disk.as_ref().unwrap();
    budget.limit_data(usize::try_from(
        used.accounted_disk_bytes + obsolete_bytes / 2,
    )?)?;
    let third_index = commit(handle, 10, batch(3)).await?;
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= third_index)
    })
    .await?;

    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(maintenance::run(
        handle.clone(),
        vec![10],
        maintenance::Options {
            // The first tick is immediate. Expire the assertion well before
            // the next tick so ordinary periodic GC cannot hide the gap.
            interval_ms: 30_000,
            checkpoint_entries: 1,
            checkpoint_interval_ms: 60_000,
            log_interval_ms: 60_000,
            scan_entries: 2,
            ..options()
        },
        progress.clone(),
        stopped,
    ));
    wait_for(|| progress.lock().unwrap().checkpoints >= 1).await?;
    let reclaimed = timeout(Duration::from_secs(3), async {
        while progress.lock().unwrap().collected_bytes < obsolete_bytes {
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    stop.send(true)?;
    task.await??;

    // On the old ordering, prove this is an actual disk-admission failure:
    // the superseded files occupy the window after installation has finished.
    if reclaimed.is_err() {
        let before = handle.prepare_disk_measurement()?.await?.run()?;
        let refused = handle
            .commit(Command::Append {
                group: 10,
                batch: batch(4),
            })?
            .await
            .unwrap_err();
        assert!(refused.is::<crate::admission::DiskBudgetExhausted>());
        assert!(before.accounted_disk_bytes > before.data_limit);
        eprintln!("retired checkpoint still occupies producer budget: {refused}");
    }

    // A real collector releases only obsolete images; the older captured
    // reader and the installed image remain protected in either ordering.
    let mut collection = handle.prepare_image_collection()?.await?;
    while !collection.run_batch(2)?.finished {}
    drop(collection);
    let image_path = |id| {
        root.path()
            .join("1")
            .join(format!("context-application-10-{id}.pack"))
    };
    assert!(!image_path(retired_id).exists());
    assert!(image_path(pinned_id).exists());
    assert_eq!(
        pinned.chunk(Part::Payload, 0, 4096)?,
        first.body.slice(..4096)
    );
    let fourth_index = commit(handle, 10, batch(4)).await?;
    drop(pinned);
    let mut collection = handle.prepare_image_collection()?.await?;
    while !collection.run_batch(2)?.finished {}
    drop(collection);
    assert!(!image_path(pinned_id).exists());
    wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).unwrap().status.applied >= fourth_index)
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
        for (index, expected) in [
            (first_index, first.clone()),
            (second_index, batch(2)),
            (third_index, batch(3)),
            (fourth_index, batch(4)),
        ] {
            assert_eq!(node.read_accepted(10, index)?, Some(expected));
        }
    }
    reclaimed.context(
        "installed checkpoint left obsolete files charged until the next maintenance tick",
    )?;
    Ok(())
}
