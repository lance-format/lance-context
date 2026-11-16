//! Requests admitted to the bounded data lane survive a checkpoint drain.
use super::*;

#[test]
fn async_persistence_checkpoint_drain_retains_bounded_data_queue() -> Result<()> {
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let c = executor.block_on(async {
        let c = Cluster::open(
            root.path(),
            Options {
                data_requests: 8,
                ..options()
            },
        )?;
        c.elect(1, 10).await?;
        Ok::<_, anyhow::Error>(c)
    })?;
    let handle = &c.handles[&1];
    let first = executor.block_on(commit(handle, 10, batch(1)))?;
    let prepared = executor.block_on(async {
        let task = handle.prepare_checkpoint(10, uuid::Uuid::new_v4())?.await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(handle.local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let before = store.appended_index_for_test();
    let term = handle.observation(10)?.status.term;
    let gate = store.block_sync_for_test();
    let held = executor.block_on(async {
        let original = handle.commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        wait_for(|| store.appended_index_for_test() > before).await?;
        let checkpoint = handle.install_checkpoint(prepared)?;
        // FIFO control work establishes that the checkpoint drain is active;
        // this does not depend on a timed race with the data lane.
        let draining = handle.local(10, |node| node.groups[&10].accepts_async(&[]))?;
        ensure!(!timeout(Duration::from_millis(500), draining).await??);
        let queued = (3..=9)
            .map(|sequence| {
                handle.commit(Command::Append {
                    group: 10,
                    batch: batch(sequence),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let overflow = handle.commit(Command::Append {
            group: 10,
            batch: batch(10),
        });
        let bounded = overflow.as_ref().err().is_some_and(|error| {
            matches!(
                error.downcast_ref(),
                Some(SubmissionFailure::RequestCredits)
            )
        });
        sleep(Duration::from_millis(200)).await;
        let retained = queued.iter().all(|request| request.0.is_empty());
        ensure!(original.0.is_empty() && checkpoint.0.is_empty());
        ensure!(handle.admission().data.used_requests == 8);
        let old = timeout(
            Duration::from_millis(500),
            handle.local(10, move |node| node.read_accepted(10, first))?,
        )
        .await??;
        ensure!(
            old == Some(batch(1)),
            "drain blocked the durable prefix read"
        );
        ensure!(c.handles.values().all(|h| h
            .observation(10)
            .is_ok_and(|o| { !o.stopped && o.status.leader == 1 && o.status.term == term })));
        Ok::<_, anyhow::Error>((original, checkpoint, queued, bounded, retained))
    });
    drop(gate);
    let mut accepted = vec![(first, 1)];
    let checked = executor.block_on(async {
        let (original, checkpoint, queued, bounded, retained) = held?;
        let second = index(timeout(Duration::from_secs(10), original).await??);
        accepted.push((second, 2));
        timeout(Duration::from_secs(10), checkpoint).await??;
        let mut refused = 0;
        for (offset, request) in queued.into_iter().enumerate() {
            match timeout(Duration::from_secs(10), request).await? {
                Ok(Outcome::Committed(Applied::Accepted { index })) => {
                    accepted.push((index, offset as u64 + 3));
                }
                _ => refused += 1,
            }
        }
        let last = accepted.last().unwrap().0;
        wait_for(|| {
            c.handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= last))
        })
        .await?;
        ensure!(handle.observation(10)?.checkpoint_index == first);
        ensure!(handle.admission().data.used_requests == 0);
        ensure!(handle.admission().control.used_requests == 0);
        Ok::<_, anyhow::Error>((bounded, retained, refused))
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    // Even the unfixed regression reopens every value it did acknowledge.
    reopen(root.path(), &accepted)?;
    let (bounded, retained, refused) = checked?;
    ensure!(bounded, "checkpoint wait escaped the existing data credits");
    ensure!(
        retained && refused == 0 && accepted.len() == 9,
        "checkpoint drain prematurely completed queued requests: retained={retained} refused={refused}"
    );
    Ok(())
}
