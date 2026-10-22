use super::*;
use crate::replication::runtime::Work;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn idle_time_does_not_consume_the_next_requests_durability_deadline() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let server_timeout = Duration::from_secs(1);
    let c = RpcCluster::open_with_options(
        directory.path(),
        Options {
            timeout: server_timeout,
            ..Options::default()
        },
        Options {
            connections: 1,
            timeout: Duration::from_secs(3),
            ..Options::default()
        },
    )
    .await?;
    let leader = c.leader().await?;
    let first = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: batch(1),
            },
        )
        .await?;
    let original_index = accepted(&first);
    drop(first);
    let idle_since = tokio::time::Instant::now();

    // Capture the real storage handle on its owner, without another data RPC.
    let handle = &c.net.nodes[&leader].handle;
    let (send_store, receive_store) = tokio::sync::oneshot::channel();
    handle.enqueue(
        handle.route(10)?,
        Work::Local(Box::new(move |node| {
            let store = node.groups[&10].raw.store().clone();
            Box::new(move || {
                let _ = send_store.send(store);
            })
        })),
        false,
    )?;
    let store = receive_store.await?;
    let blocking_store = store.clone();
    let (entered, arrived) = tokio::sync::oneshot::channel();
    let (release, resumed) = std::sync::mpsc::channel::<()>();
    let blocked = tokio::task::spawn_blocking(move || {
        let _gate = blocking_store.block_sync_for_test();
        let _ = entered.send(());
        let _ = resumed.recv();
    });
    arrived.await?;
    tokio::time::sleep_until(idle_since + Duration::from_millis(600)).await;
    let client = c.client.clone();
    let append = tokio::spawn(async move {
        client
            .request(
                leader,
                Request::Append {
                    group: 10,
                    batch: batch(2),
                },
            )
            .await
    });
    let entered_sync = timeout(Duration::from_millis(250), async {
        while store.appended_index_for_test() != Some(original_index + 1) {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await;

    // The bytes have reached the actual journal, but quorum ACK must wait for
    // its sync. This crosses the previous idle deadline while staying within
    // the new request's full processing budget.
    tokio::time::sleep_until(idle_since + server_timeout + Duration::from_millis(150)).await;
    let completed_before_sync = append.is_finished();
    drop(release);
    blocked.await?;
    let result = append.await?;
    let fetched = c
        .client
        .request(
            leader,
            Request::Fetch {
                group: 10,
                index: original_index + 1,
            },
        )
        .await;
    drop(store);
    c.shutdown().await?;

    entered_sync.context("request did not reach real sync within the test window")?;
    ensure!(
        !completed_before_sync,
        "connection expired during an admitted durability wait"
    );
    let reply = result.context("append lost its processing budget to prior idle time")?;
    assert_eq!(accepted(&reply), original_index + 1);
    let fetched = fetched?;
    assert!(matches!(&fetched.reply, Reply::Record(Some(meta)) if meta == &batch(2).meta));
    assert_eq!(fetched.body, batch(2).body);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn partial_request_frames_still_expire_and_release_the_connection() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = RpcCluster::open_with_options(
        directory.path(),
        Options {
            timeout: Duration::from_millis(250),
            server_connections: 1,
            ..Options::default()
        },
        Options::default(),
    )
    .await?;
    let leader = c.leader().await?;
    for partial_header in [true, false] {
        let mut stream = c.raw(leader, true).await?;
        let began = tokio::time::Instant::now();
        if partial_header {
            stream.write_all(&MAGIC[..1]).await?;
        } else {
            let original = batch(1);
            let metadata = codec().serialize(&(
                scope(&c.net.nodes[&leader].handle),
                WireRequest::Append {
                    group: 10,
                    meta: original.meta,
                },
            ))?;
            let mut fixed = [0; PREFIX];
            fixed[..8].copy_from_slice(MAGIC);
            fixed[8..16].copy_from_slice(&1u64.to_le_bytes());
            fixed[16..20].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
            fixed[20..24].copy_from_slice(&(original.body.len() as u32).to_le_bytes());
            stream.write_all(&fixed).await?;
            stream.write_all(&metadata).await?;
            // Valid frame metadata, but the declared body never arrives.
        }
        stream.flush().await?;
        let mut byte = [0];
        let closed = timeout(Duration::from_secs(2), stream.read(&mut byte)).await?;
        assert!(
            matches!(closed, Ok(0) | Err(_)),
            "partial frame received an unexpected reply"
        );
        assert!(
            began.elapsed() >= Duration::from_millis(200),
            "valid partial frame rejected before its timeout"
        );
        drop(stream);
    }
    let response = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: batch(1),
            },
        )
        .await?;
    accepted(&response);
    drop(response);
    c.shutdown().await
}
