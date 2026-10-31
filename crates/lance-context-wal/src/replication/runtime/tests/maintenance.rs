//! Exercise the service scheduler against real Raft workers and physical logs.
use super::*;
use crate::{raft_log::DiskOptions, service::maintenance};
use raft_proto::eraftpb::{Message, MessageType};
use uuid::Uuid;

fn disk() -> DiskOptions {
    DiskOptions {
        max_bytes: 1 << 30,
        data_bytes: 512 << 20,
        headroom_bytes: 256 << 20,
    }
}

// A long normal interval makes an erroneous full-interval postponement visible.
// Failed work should become eligible at the next existing scheduler tick.
fn options() -> maintenance::Options {
    maintenance::Options {
        interval_ms: 25,
        checkpoint_entries: u64::MAX,
        checkpoint_interval_ms: 4000,
        log_interval_ms: 60_000,
        scan_entries: 8,
    }
}

async fn failure_has_occurred(progress: &Arc<Mutex<maintenance::Progress>>) -> Result<()> {
    timeout(Duration::from_secs(8), async {
        loop {
            if progress.lock().unwrap().attempts_failed > 0 {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .context("expected maintenance admission failure did not occur")
}

fn verify_reopened(root: &Path, index: u64) -> Result<()> {
    for id in 1..=3 {
        let node = Node::open(&root.join(id.to_string()), layout(), id, Limits::default())?;
        assert_eq!(node.read_accepted(10, index)?, Some(batch(1)));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn busy_image_wait_preserves_log_progress_and_checkpoint_eligibility() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = Cluster::open_with_disk(root.path(), Options::default(), Some(disk()))?;
    c.elect(1, 10).await?;
    let index = commit(&c.handles[&1], 10, batch(1)).await?;
    wait_for(|| c.handles[&3].observation(10).unwrap().status.applied >= index).await?;
    // Hold the same physical-node permit that a snapshot receiver/build uses.
    let busy = c.handles[&3]
        .prepare_checkpoint(10, Uuid::new_v4())?
        .await?;
    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(maintenance::run(
        c.handles[&3].clone(),
        vec![10],
        maintenance::Options {
            log_interval_ms: 1000,
            ..options()
        },
        progress.clone(),
        stopped,
    ));
    // Collection also uses this permit. Keep it until the first checkpoint age
    // expires, and require independent log maintenance to advance while waiting.
    sleep(Duration::from_millis(4500)).await;
    let while_busy = progress.lock().unwrap().clone();
    assert!(
        while_busy.log_passes >= 2,
        "image wait must not block log maintenance"
    );
    assert_eq!(
        while_busy.attempts_failed, 0,
        "image contention must wait for admission"
    );
    assert_eq!(progress.lock().unwrap().checkpoints, 0);
    drop(busy);
    let result = timeout(Duration::from_secs(2), async {
        loop {
            if c.handles[&3].observation(10)?.checkpoint_index >= index {
                return Ok::<_, anyhow::Error>(());
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .context("released checkpoint permit was postponed for a full checkpoint interval");
    stop.send(true)?;
    task.await??;
    c.shutdown().await?;
    result??;
    verify_reopened(root.path(), index)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stalled_snapshot_allows_log_maintenance_and_cancellable_image_wait() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = Cluster::open_with_disk(root.path(), Options::default(), Some(disk()))?;
    c.elect(1, 10).await?;
    let index = commit(&c.handles[&1], 10, batch(1)).await?;
    wait_for(|| c.handles[&3].observation(10).unwrap().status.applied >= index).await?;
    let capture = c.handles[&1]
        .prepare_checkpoint(10, Uuid::new_v4())?
        .await?;
    let built = tokio::task::spawn_blocking(move || capture.run()).await??;
    let image = built.image();
    c.handles[&1].install_checkpoint(built)?.await?;
    let packet = Packet {
        cluster: c.handles[&1].cluster,
        layout: c.handles[&1].layout,
        policy: c.handles[&1].policy,
        group: 10,
        message: Message {
            from: 1,
            to: 3,
            term: c.handles[&1].observation(10)?.status.term,
            msg_type: MessageType::MsgSnapshot,
            snapshot: Some(image.snapshot()?).into(),
            ..Default::default()
        },
    };
    // Real reception owns the image slot and partially created files. Do not
    // feed it bytes: neither log maintenance nor shutdown can depend on EOF.
    let receiver = c.handles[&3]
        .receive_snapshot(packet, Uuid::new_v4())?
        .await?;
    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(maintenance::run(
        c.handles[&3].clone(),
        vec![10],
        maintenance::Options {
            log_interval_ms: 100,
            ..options()
        },
        progress.clone(),
        stopped,
    ));
    wait_for(|| progress.lock().unwrap().log_passes >= 2).await?;
    assert_eq!(progress.lock().unwrap().attempts_failed, 0);
    let next = commit(&c.handles[&1], 10, batch(2)).await?;
    wait_for(|| c.handles[&3].observation(10).unwrap().status.applied >= next).await?;
    stop.send(true)?;
    timeout(Duration::from_secs(2), task)
        .await
        .context("maintenance shutdown waited for a stalled snapshot")???;
    assert_eq!(progress.lock().unwrap().checkpoints, 0);
    drop(receiver);
    drop(image);
    c.shutdown().await?;
    verify_reopened(root.path(), index)?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        assert_eq!(node.read_accepted(10, next)?, Some(batch(2)));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_checkpoint_captures_fresh_state_and_skips_an_already_covered_cut() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = Cluster::open(root.path(), Options::default())?;
    c.elect(1, 10).await?;
    let first = commit(&c.handles[&1], 10, batch(1)).await?;
    let handle = c.handles[&1].clone();
    let held = handle.prepare_checkpoint(10, Uuid::new_v4())?.await?;
    let waiting = handle.prepare_checkpoint_wait(10, Uuid::new_v4());
    tokio::pin!(waiting);
    assert!(timeout(Duration::from_millis(50), &mut waiting)
        .await
        .is_err());
    let next = commit(&handle, 10, batch(2)).await?;
    assert!(next > first);
    drop(held);
    let capture = timeout(Duration::from_secs(2), waiting)
        .await??
        .await?
        .unwrap();
    let built = tokio::task::spawn_blocking(move || capture.run()).await??;
    assert_eq!(built.image().snapshot()?.get_metadata().index, next);
    let covered = handle.prepare_checkpoint_wait(10, Uuid::new_v4());
    tokio::pin!(covered);
    assert!(timeout(Duration::from_millis(50), &mut covered)
        .await
        .is_err());
    handle.install_checkpoint(built)?.await?;
    assert!(timeout(Duration::from_secs(2), covered)
        .await??
        .await?
        .is_none());
    // Neither cancelled/completed admission nor an empty capture retains the
    // physical slot. An immediate image scan must still be admissible.
    drop(handle.prepare_image_collection()?.await?);
    wait_for(|| c.handles[&3].observation(10).unwrap().status.applied >= next).await?;
    c.shutdown().await?;
    verify_reopened(root.path(), first)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn released_rewrite_headroom_does_not_wait_another_log_interval() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = Cluster::open_with_disk(root.path(), Options::default(), Some(disk()))?;
    c.elect(1, 10).await?;
    let index = commit(&c.handles[&1], 10, batch(1)).await?;
    wait_for(|| c.handles[&3].observation(10).unwrap().status.applied >= index).await?;
    let budget = c.handles[&3].disk.as_ref().unwrap();
    let measured = c.handles[&3].prepare_disk_measurement()?.await?.run()?;
    // Reserve headroom without creating extra data or changing disk ceilings.
    // A native rewrite must reject until this concurrent reservation is freed.
    let held = budget.reserve(
        usize::try_from(measured.max_log_bytes - measured.accounted_disk_bytes - (1 << 20))?,
        false,
    )?;
    let progress = Arc::new(Mutex::new(maintenance::Progress::default()));
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(maintenance::run(
        c.handles[&3].clone(),
        vec![10],
        maintenance::Options {
            checkpoint_interval_ms: 60_000,
            log_interval_ms: 4000,
            ..options()
        },
        progress.clone(),
        stopped,
    ));
    failure_has_occurred(&progress).await?;
    assert_eq!(progress.lock().unwrap().log_passes, 0);
    drop(held);
    let result = timeout(Duration::from_secs(2), async {
        loop {
            if progress.lock().unwrap().log_passes > 0 {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .context("released rewrite headroom was postponed for a full log interval");
    stop.send(true)?;
    task.await??;
    c.shutdown().await?;
    result?;
    verify_reopened(root.path(), index)
}
