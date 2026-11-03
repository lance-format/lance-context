//! Compare suffix catch-up with a checkpoint that overtakes an in-flight image.
use super::*;

async fn successful_slow_snapshot(
    checkpoint_during_transfer: bool,
    retained_bytes: u64,
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut c = Images::open(root.path()).await?;
    c.net.nodes[&1].handle.campaign(10)?.await?;
    let leader_id = c.net.leader(10).await?;
    let follower = (1..=3).rfind(|id| *id != leader_id).unwrap();
    c.stop(follower).await?;
    let leader = c.net.nodes[&leader_id].handle.clone();
    let mut expected = Vec::new();
    for sequence in 1..=4 {
        expected.push(append(&leader, 10, sequence).await?);
    }
    wait_for(|| {
        c.net
            .nodes
            .values()
            .all(|n| n.handle.observation(10).unwrap().status.accepted_records == 4)
    })
    .await?;
    let original = checkpoint(&leader, 10).await?;
    let original_index = original.snapshot()?.get_metadata().index;
    drop(original);
    c.net.start(
        root.path(),
        follower,
        TcpListener::bind(c.net.addresses[&follower]).await?,
    )?;
    let marker = root.path().join("received-first-chunk");
    c.attach_options(
        follower,
        c.listener(follower)?,
        c.quotas[&follower].clone(),
        snapshot::Options {
            transfer_timeout: Duration::from_secs(60),
            pause_after_payload_chunk: Some(marker.clone()),
            ..Default::default()
        },
    )?;
    let statistics = c.services[&leader_id].statistics().clone();
    let receiver = c.services[&follower].statistics().clone();
    wait_for(|| marker.exists()).await?;
    assert_eq!(statistics.sent.load(Ordering::Acquire), 0);
    assert_eq!(receiver.installed.load(Ordering::Acquire), 0);

    // The slow follower is receiving real TLS payload bytes. Its existing
    // transfer remains connected while the healthy quorum commits a suffix.
    timeout(Duration::from_secs(10), async {
        for sequence in 5..=12 {
            expected.push(append(&leader, 10, sequence).await?);
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    // The healthy voter already has the suffix before another checkpoint.
    // It must not contribute unrelated image deliveries to these counters.
    wait_for(|| {
        c.net.nodes.iter().all(|(id, n)| {
            *id == follower || n.handle.observation(10).unwrap().status.accepted_records == 12
        })
    })
    .await?;
    let latest_index = if checkpoint_during_transfer {
        let task = leader.prepare_checkpoint(10, uuid::Uuid::new_v4())?.await?;
        let newer = tokio::task::spawn_blocking(move || task.run()).await??;
        let index = newer.image().snapshot()?.get_metadata().index;
        leader
            .install_checkpoint_retaining(newer, retained_bytes)?
            .await?;
        assert!(index > original_index);
        index
    } else {
        original_index
    };
    assert_eq!(statistics.sent.load(Ordering::Acquire), 0);
    assert_eq!(receiver.installed.load(Ordering::Acquire), 0);
    assert_eq!(leader.observation(10)?.checkpoint_index, latest_index);
    std::fs::write(marker.with_extension("resume"), b"resume")?;

    wait_for(|| {
        c.net.nodes[&follower]
            .handle
            .observation(10)
            .unwrap()
            .status
            .accepted_records
            == 12
    })
    .await?;
    let extra_image = checkpoint_during_transfer && retained_bytes == 0;
    // Without a retained tail, a checkpoint overtaking this successful
    // transfer requires another image. With a sufficient tail, the same
    // follower replays the suffix despite the newer installed checkpoint.
    let expected_images = if extra_image { 2 } else { 1 };
    wait_for(|| {
        statistics.sent.load(Ordering::Acquire) == expected_images
            && receiver.installed.load(Ordering::Acquire) == expected_images
    })
    .await?;
    assert_eq!(receiver.installed.load(Ordering::Acquire), expected_images);
    assert_eq!(receiver.failed.load(Ordering::Acquire), 0);
    assert_eq!(statistics.failed.load(Ordering::Acquire), 0);
    c.shutdown().await?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        for (index, batch) in &expected {
            assert_eq!(node.read_accepted(10, *index)?, Some(batch.clone()));
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn successful_slow_snapshot_catches_up_from_retained_suffix() -> Result<()> {
    successful_slow_snapshot(false, 0).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn successful_slow_snapshot_needs_another_image_when_checkpoint_overtakes_it() -> Result<()> {
    successful_slow_snapshot(true, 0).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn successful_slow_snapshot_uses_tail_despite_newer_checkpoint() -> Result<()> {
    successful_slow_snapshot(true, 16 << 20).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reopening_retained_index_one_still_loads_the_application_checkpoint() -> Result<()> {
    let root = tempfile::tempdir()?;
    let c = Images::open(root.path()).await?;
    c.net.nodes[&1].handle.campaign(10)?.await?;
    let leader_id = c.net.leader(10).await?;
    let leader = &c.net.nodes[&leader_id].handle;
    let mut expected = Vec::new();
    for sequence in 1..=4 {
        expected.push(append(leader, 10, sequence).await?);
    }
    let task = leader.prepare_checkpoint(10, uuid::Uuid::new_v4())?.await?;
    let image = tokio::task::spawn_blocking(move || task.run()).await??;
    let index = image.image().snapshot()?.get_metadata().index;
    leader
        .install_checkpoint_retaining(image, 16 << 20)?
        .await?;
    assert_eq!(leader.observation(10)?.checkpoint_index, index);
    c.shutdown().await?;
    let node = Node::open(
        &root.path().join(leader_id.to_string()),
        layout(),
        leader_id,
        Limits::default(),
    )?;
    assert_eq!(raft::Storage::first_index(node.groups[&10].raw.store())?, 1);
    assert_eq!(node.checkpoint_index(10)?, index);
    assert!(node.groups[&10].image.is_some());
    assert!(node.prepare_checkpoint(10, uuid::Uuid::new_v4()).is_err());
    for (index, batch) in expected {
        assert_eq!(node.read_accepted(10, index)?, Some(batch));
    }
    Ok(())
}
