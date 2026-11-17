//! Reproduce producer admission during a real native checkpoint drain.
use super::*;
use crate::replication::{producer, query, ProposalRejection};
use uuid::Uuid;

fn open(slot: u32) -> Result<producer::Command> {
    Ok(producer::Command {
        cluster: layout().cluster,
        layout: layout().fingerprint()?,
        group: 10,
        table: Uuid::from_u128(2),
        slot,
        action: producer::Action::Open {
            expected_generation: 0,
            operation: Uuid::from_u128(100 + u128::from(slot)),
            bucket: 0,
        },
    })
}

fn data(sequence: u64) -> Batch {
    let mut value = batch(sequence);
    value.meta.producer = producer::producer_id(0, 1);
    value
}

async fn lifecycle(handle: &Handle, command: producer::Command) -> Result<producer::Reply> {
    let outcome = timeout(
        Duration::from_secs(10),
        handle.commit(Command::Producer(command))?,
    )
    .await??;
    let Outcome::Committed(Applied::Producer(reply)) = outcome else {
        anyhow::bail!("producer command did not commit: {outcome:?}");
    };
    Ok(reply)
}

async fn slot(handle: &Handle, slot: u32) -> Result<producer::View> {
    let outcome = timeout(
        Duration::from_secs(10),
        handle.query(10, query::Query::ProducerSlot { slot })?,
    )
    .await??;
    let query::Outcome::Observed {
        value: query::Value::ProducerSlot(Some(view)),
        ..
    } = outcome
    else {
        anyhow::bail!("producer quorum view unavailable: {outcome:?}");
    };
    Ok(view)
}

#[test]
fn producer_open_during_checkpoint_is_rejected_before_proposal_and_exact_cas_is_safe() -> Result<()>
{
    let executor = executor()?;
    let root = tempfile::tempdir()?;
    let limits = Limits {
        producer_slots: 2,
        ..Limits::default()
    };
    let c = executor.block_on(async {
        let c = Cluster::open_with_limits(
            root.path(),
            options(),
            None,
            crate::raft_log::LogOptions::default(),
            limits.clone(),
        )?;
        c.elect(1, 10).await?;
        ensure!(matches!(
            lifecycle(&c.handles[&1], open(0)?).await?,
            producer::Reply::Opened {
                generation: 1,
                closed: false,
                ..
            }
        ));
        Ok::<_, anyhow::Error>(c)
    })?;
    let handle = &c.handles[&1];
    let first = executor.block_on(commit(handle, 10, data(1)))?;
    let image = executor.block_on(async {
        let task = handle.prepare_checkpoint(10, Uuid::new_v4())?.await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(handle.local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let appended = store.appended_index_for_test();
    let gate = store.block_sync_for_test();
    let held = executor.block_on(async {
        let original = handle.commit(Command::Append {
            group: 10,
            batch: data(2),
        })?;
        wait_for(|| store.appended_index_for_test() > appended).await?;
        let checkpoint = handle.install_checkpoint(image)?;
        // FIFO control work proves the real drain is active, without a sleep race.
        let before = handle
            .local(10, |node| {
                ensure!(!node.groups[&10].accepts_async(&[])?);
                Ok(node.groups[&10].raw.raft.raft_log.last_index())
            })?
            .await?;
        let rejected = timeout(
            Duration::from_secs(1),
            handle.commit(Command::Producer(open(1)?))?,
        )
        .await?;
        ensure!(
            rejected.as_ref().err().is_some_and(|error| matches!(
                error.downcast_ref::<ProposalRejection>(),
                Some(ProposalRejection::PendingCapacity)
            )),
            "expected preproposal checkpoint refusal: {rejected:?}"
        );
        let unchanged = handle
            .local(10, move |node| {
                let group = &node.groups[&10];
                let view = group.state.producer_view(1).context("missing slot")?;
                Ok(group.raw.raft.raft_log.last_index() == before
                    && view.generation == 0
                    && view.lease.is_none())
            })?
            .await?;
        ensure!(
            unchanged,
            "rejected Open changed log or producer allocation"
        );
        ensure!(original.0.is_empty() && checkpoint.0.is_empty());
        Ok::<_, anyhow::Error>((original, checkpoint))
    });
    // Release the original sync even if a diagnostic assertion fails.
    drop(gate);
    let checked = executor.block_on(async {
        let (original, checkpoint) = held?;
        let second = index(timeout(Duration::from_secs(10), original).await??);
        timeout(Duration::from_secs(10), checkpoint).await??;
        let absent = slot(handle, 1).await?;
        ensure!(absent.generation == 0 && absent.lease.is_none());
        let command = open(1)?;
        let expected = producer::producer_id(1, 1);
        // An identical logical Open can be resolved again after a lost reply;
        // it must not allocate another generation or accept a different UUID.
        for _ in 0..2 {
            ensure!(matches!(
                lifecycle(handle, command.clone()).await?,
                producer::Reply::Opened { producer, generation: 1, closed: false }
                    if producer == expected
            ));
        }
        let mut foreign = command.clone();
        if let producer::Action::Open { operation, .. } = &mut foreign.action {
            *operation = Uuid::from_u128(999);
        }
        ensure!(matches!(
            lifecycle(handle, foreign).await?,
            producer::Reply::Fenced
        ));
        let mut close = command.clone();
        close.action = producer::Action::Revoke { generation: 1 };
        ensure!(matches!(
            lifecycle(handle, close).await?,
            producer::Reply::Closed { last: None }
        ));
        ensure!(matches!(
            lifecycle(handle, command).await?,
            producer::Reply::Opened { producer, generation: 1, closed: true }
                if producer == expected
        ));
        let final_view = slot(handle, 1).await?;
        ensure!(final_view.generation == 1 && final_view.lease.unwrap().closed);
        let through = handle.observation(10)?.status.applied;
        wait_for(|| {
            c.handles
                .values()
                .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= through))
        })
        .await?;
        Ok::<_, anyhow::Error>(second)
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    let second = checked?;
    for id in 1..=3 {
        let node = Node::open(
            &root.path().join(id.to_string()),
            layout(),
            id,
            limits.clone(),
        )?;
        ensure!(node.read_accepted(10, first)? == Some(data(1)));
        ensure!(node.read_accepted(10, second)? == Some(data(2)));
        let view = node.groups[&10]
            .state
            .producer_view(1)
            .context("missing slot")?;
        ensure!(view.generation == 1 && view.lease.unwrap().closed);
    }
    Ok(())
}
