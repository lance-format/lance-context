use super::*;

/// A local committed read must not wait for the *next* unrelated sync. It is
/// neither a fresh quorum observation nor an acknowledgement of a new append.
#[test]
fn committed_fetch_finishes_before_a_later_failed_sync_without_acknowledging_new_data() -> Result<()>
{
    let executor = tokio::runtime::Runtime::new()?;
    for pipeline_replication in [false, true] {
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
        nodes[0].campaign(10)?;
        crate::replication::tests::pump(&mut nodes)?;
        let original = batch(1);
        nodes[0].propose(10, 1, original.clone())?;
        let committed = crate::replication::tests::pump(&mut nodes)?;
        let original_index = index(
            committed
                .into_iter()
                .find(|c| c.request == 1)
                .unwrap()
                .outcome,
        );
        let proposed_index = original_index + 1;
        let node = nodes.remove(0);
        let store = node.groups[&10].raw.store().clone();
        let (runtime, outbox) = Runtime::spawn(
            node,
            Options {
                pipeline_replication,
                tick_interval: Duration::from_secs(1),
                batch_requests: 16,
                ..Options::default()
            },
        )?;
        let handle = runtime.handle();

        // Freeze the actual worker while all three data-lane requests queue.
        // Dropping the sender also releases this gate on any error path.
        let (entered, arrived) = mpsc::channel();
        let (resume, resumed) = mpsc::channel::<()>();
        handle.enqueue(
            handle.route(10)?,
            Work::Local(Box::new(move |_| {
                entered.send(()).unwrap();
                let _ = resumed.recv();
                Box::new(|| {})
            })),
            false,
        )?;
        arrived.recv_timeout(Duration::from_secs(5))?;
        let blocked = store.block_sync_for_test();
        store.fail_next_sync_for_test();
        let pending = handle.commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        let existing = handle.read_accepted(10, original_index)?;
        let uncommitted = handle.read_accepted(10, proposed_index)?;
        drop(resume);
        executor.block_on(wait_for(|| {
            store.appended_index_for_test() == Some(proposed_index)
        }))?;

        let existing_ready = !existing.0.is_empty();
        let uncommitted_ready = !uncommitted.0.is_empty();
        let new_append_unacknowledged = pending.0.is_empty();
        drop(blocked);
        let existing_result = executor.block_on(existing);
        let uncommitted_result = executor.block_on(uncommitted);
        assert!(executor
            .block_on(async { timeout(Duration::from_secs(5), pending).await })?
            .is_err());
        assert!(executor.block_on(runtime.shutdown()).is_err());
        drop(outbox);
        drop(handle);
        drop(store);
        drop(nodes);

        // Real files reopen without a producer replay. A failed new sync cannot
        // invalidate the old committed response or make the new row accepted.
        let reopened = Node::open(&directory.path().join("1"), layout(), 1, Limits::default())?;
        assert_eq!(
            reopened.read_accepted(10, original_index)?,
            Some(original.clone())
        );
        assert!(reopened.read_accepted(10, proposed_index)?.is_none());
        assert!(new_append_unacknowledged, "new append bypassed persistence");
        assert!(
            existing_ready,
            "committed fetch waited for an unrelated sync"
        );
        assert!(
            uncommitted_ready,
            "absent local row waited for an unrelated sync"
        );
        assert_eq!(existing_result?, Some(original));
        assert!(uncommitted_result?.is_none());
    }
    Ok(())
}
