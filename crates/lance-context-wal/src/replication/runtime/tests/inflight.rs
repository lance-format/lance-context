use super::*;
use crate::replication::tests::pump;

// A data-lane fence runs after preceding proposals and releases after their poll.
async fn proposed(handle: &Handle, group: u64) -> Result<()> {
    let (sent, received) = oneshot::channel();
    handle.enqueue(
        handle.route(group)?,
        Work::Local(Box::new(move |_| {
            Box::new(move || {
                let _ = sent.send(());
            })
        })),
        true,
    )?;
    timeout(Duration::from_secs(5), received).await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn configured_window_limits_each_follower_then_resumes_after_durable_replies() -> Result<()> {
    for window in [1, 3] {
        let directory = tempfile::tempdir()?;
        let mut nodes = (1..=3)
            .map(|id| {
                Node::open(
                    &directory.path().join(id.to_string()),
                    layout(),
                    id,
                    Limits::default(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        for group in [10, 20] {
            nodes[0].campaign(group)?;
        }
        pump(&mut nodes)?;
        let leader = nodes.remove(0);
        let (runtime, mut outgoing) = Runtime::spawn(
            leader,
            Options {
                raft_max_inflight_msgs: window,
                tick_interval: Duration::from_secs(1),
                ..Options::default()
            },
        )?;
        let handle = runtime.handle();
        let mut pending = Vec::new();
        for sequence in 1..=4 {
            for group in [10, 20] {
                let mut input = batch(sequence);
                input.meta.bucket = u32::from(group == 20);
                pending.push(handle.commit(Command::Append {
                    group,
                    batch: input,
                })?);
                proposed(&handle, group).await?;
            }
        }

        let mut held = std::collections::VecDeque::new();
        let mut sent = BTreeMap::<(u64, u64), usize>::new();
        while let Ok(packet) = outgoing.try_recv() {
            let p = packet.packet();
            if !p.message.entries.is_empty() {
                *sent.entry((p.group, p.message.to)).or_default() += 1;
            }
            held.push_back(packet);
        }
        assert_eq!(
            sent,
            BTreeMap::from([
                ((10, 2), window),
                ((10, 3), window),
                ((20, 2), window),
                ((20, 3), window),
            ])
        );
        assert!(pending.iter().all(|p| p.0.is_empty()));

        // Release real follower persistence replies. A full window must resume
        // and commit every queued request without changing the quorum boundary.
        let relay_handle = handle.clone();
        let relay = tokio::spawn(async move {
            loop {
                let packet = match held.pop_front() {
                    Some(packet) => packet,
                    None => match outgoing.recv().await {
                        Some(packet) => packet,
                        None => return Ok::<_, anyhow::Error>(()),
                    },
                };
                let p = packet.packet();
                let follower = &mut nodes[p.message.to as usize - 2];
                follower.step(p.clone())?;
                for response in follower.poll()?.messages {
                    relay_handle.step(response)?;
                }
            }
        });
        for request in pending {
            assert!(matches!(
                timeout(Duration::from_secs(5), request).await??,
                Outcome::Committed(Applied::Accepted { .. })
            ));
        }
        relay.abort();
        match relay.await {
            Err(error) if error.is_cancelled() => {}
            result => result??,
        }
        runtime.shutdown().await?;

        let reopened = Node::open(&directory.path().join("1"), layout(), 1, Limits::default())?;
        for group in [10, 20] {
            assert_eq!(reopened.status(group)?.accepted_records, 4);
        }
        let (restarted, _) = Runtime::spawn(
            reopened,
            Options {
                raft_max_inflight_msgs: 64,
                ..Options::default()
            },
        )?;
        restarted.shutdown().await?;
    }
    Ok(())
}

#[test]
fn invalid_window_rejects_startup_without_retaining_the_directory() -> Result<()> {
    let directory = tempfile::tempdir()?;
    for window in [0, 257] {
        let node = Node::open(directory.path(), layout(), 1, Limits::default())?;
        assert!(Runtime::spawn(
            node,
            Options {
                raft_max_inflight_msgs: window,
                ..Options::default()
            }
        )
        .is_err());
    }
    drop(Node::open(
        directory.path(),
        layout(),
        1,
        Limits::default(),
    )?);
    Ok(())
}
