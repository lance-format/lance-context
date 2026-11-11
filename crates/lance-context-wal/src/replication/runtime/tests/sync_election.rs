use super::*;
use anyhow::bail;

// Three real durable engines and dedicated group workers, with the ordinary
// 100 ms ticker and an in-process peer relay. This is a controlled mechanism
// test, not a TCP benchmark or proof of a particular cloud incident's cause.
fn sync_stall_election(pipeline_replication: bool) -> Result<()> {
    let executor = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let directory = tempfile::tempdir()?;
    let options = Options {
        pipeline_replication,
        tick_interval: Duration::from_millis(100),
        ..Options::default()
    };
    let c = executor.block_on(async {
        let c = Cluster::open(directory.path(), options.clone())?;
        c.elect(1, 10).await?;
        Ok::<_, anyhow::Error>(c)
    })?;
    let first = executor.block_on(commit(&c.handles[&1], 10, batch(1)))?;
    executor.block_on(wait_for(|| {
        c.handles
            .values()
            .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= first))
    }))?;
    let old_term = c.handles[&1].observation(10)?.status.term;
    let store = executor
        .block_on(c.handles[&1].local(10, |node| Ok(node.groups[&10].raw.store().clone()))?)?;
    let before_append = store.appended_index_for_test();
    let sync_guard = store.block_sync_for_test();
    let started = std::time::Instant::now();
    // Keep every fallible observation inside this result. Always release the
    // gate before propagating errors or joining workers, including test failure.
    let stalled = executor.block_on(async {
        let pending = c.handles[&1].commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        wait_for(|| store.appended_index_for_test() > before_append).await?;
        let proposed = store
            .appended_index_for_test()
            .context("no native append while sync is held")?;
        let (leader, term) = timeout(Duration::from_secs(10), async {
            loop {
                let a = c.handles[&2].observation(10)?.status;
                let b = c.handles[&3].observation(10)?.status;
                if matches!(a.leader, 2 | 3)
                    && a.leader == b.leader
                    && a.term == b.term
                    && a.term > old_term
                {
                    return Ok::<_, anyhow::Error>((a.leader, a.term));
                }
                ensure!(pending.0.is_empty(), "blocked leader released a completion");
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .context("unblocked replicas did not automatically elect a leader")??;
        ensure!(pending.0.is_empty(), "quorum bypassed blocked leader sync");
        Ok::<_, anyhow::Error>((pending, leader, term, proposed))
    });
    let held_ms = started.elapsed().as_millis();
    drop(sync_guard);
    let checked = executor.block_on(async {
        let (pending, leader, term, proposed) = stalled?;
        let original = timeout(Duration::from_secs(10), pending).await??;
        wait_for(|| {
            c.handles.values().all(|h| {
                h.observation(10)
                    .is_ok_and(|o| o.status.leader == leader && o.status.term >= term)
            })
        })
        .await?;
        let resolved = commit(&c.handles[&leader], 10, batch(2)).await?;
        ensure!(
            commit(&c.handles[&leader], 10, batch(2)).await? == resolved,
            "exact retry created another accepted position"
        );
        match original {
            Outcome::Unknown => {}
            Outcome::Committed(Applied::Accepted { index }) => {
                ensure!(index == resolved, "successful original and retry disagree");
            }
            ref other => bail!("unexpected original outcome {other:?}"),
        }
        let read = c.handles[&leader]
            .query(10, query::Query::Accepted { after: 0, limit: 8 })?
            .await?;
        let query::Outcome::Observed {
            value: query::Value::Accepted(page),
            ..
        } = read
        else {
            bail!("missing quorum accepted page");
        };
        ensure!(
            page.entries == vec![(first, batch(1).meta), (resolved, batch(2).meta)],
            "old ACKed prefix was lost or exact retry duplicated the request"
        );
        wait_for(|| {
            c.handles.values().all(|h| {
                h.observation(10)
                    .is_ok_and(|o| o.status.applied >= resolved)
            })
        })
        .await?;
        eprintln!(
            "sync-election pipeline={pipeline_replication} held_ms={held_ms} old_term={old_term} new_term={term} leader={leader} proposed={proposed} resolved={resolved} original={original:?}"
        );
        Ok::<_, anyhow::Error>(resolved)
    });
    let stopped = executor.block_on(c.shutdown());
    drop(store);
    stopped?;
    let resolved = checked?;
    // No original worker or engine handle survives. Verify exact full bytes
    // independently on all replicas, then verify dedup from reopened state.
    for id in 1..=3 {
        let node = Node::open(
            &directory.path().join(id.to_string()),
            layout(),
            id,
            Limits::default(),
        )?;
        ensure!(node.read_accepted(10, first)? == Some(batch(1)));
        ensure!(node.read_accepted(10, resolved)? == Some(batch(2)));
    }
    executor.block_on(async {
        let c = Cluster::open(directory.path(), options)?;
        let verified = async {
            c.elect(3, 10).await?;
            ensure!(commit(&c.handles[&3], 10, batch(2)).await? == resolved);
            Ok::<_, anyhow::Error>(())
        }
        .await;
        c.shutdown().await?;
        verified
    })
}

#[test]
fn blocked_sync_preserves_exact_retry_through_automatic_election() -> Result<()> {
    sync_stall_election(false)
}

#[test]
fn pipelined_blocked_sync_preserves_exact_retry_through_automatic_election() -> Result<()> {
    sync_stall_election(true)
}
