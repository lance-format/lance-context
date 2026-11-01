use super::*;
use crate::replication::runtime::Credits;

fn queued() -> Result<(Queued, Credits, Arc<Semaphore>, Bytes)> {
    let body = Bytes::from(vec![0xa5; 1024]);
    let mut message = Message {
        from: 1,
        to: 2,
        ..Default::default()
    };
    message.set_msg_type(MessageType::MsgAppend);
    message.set_entries(
        vec![Entry {
            index: 1,
            term: 1,
            data: body.clone(),
            ..Default::default()
        }]
        .into(),
    );
    let size = message.compute_size() as usize;
    let credits = Credits::new(1, size + 256);
    let bytes = Arc::new(Semaphore::new(size + 256));
    let (send, mut receive) = mpsc::channel(1);
    let route = Route {
        send,
        ready: Arc::new(AtomicBool::new(true)),
        bytes: bytes.clone(),
    };
    assert!(route.enqueue(
        Outbound {
            packet: Packet {
                cluster: uuid::Uuid::nil(),
                layout: [0; 32],
                policy: [0; 32],
                group: 10,
                message,
            },
            _credit: Some(credits.reserve(size)?),
            image: None,
            snapshot_feedback: None,
        },
        &Statistics::default(),
    ));
    Ok((receive.try_recv()?, credits, bytes, body))
}

#[tokio::test]
async fn blocked_send_separates_queue_wait_and_keeps_both_credits() -> Result<()> {
    let (queued, credits, bytes, body) = queued()?;
    let statistics = DestinationStatistics::default();
    assert_eq!(
        queued.packet.packet().message.entries[0].data.as_ptr(),
        body.as_ptr()
    );
    // Neither interval claims a frame that has not reached the send loop.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(statistics.queue_wait.snapshot().finished, 0);
    let (mut write, mut read) = tokio::io::duplex(64);
    let mut send = Box::pin(send_queued(&mut write, &queued, &statistics));
    // Borrowing the future preserves the blocked send after the observer timeout.
    assert!(timeout(Duration::from_millis(10), send.as_mut())
        .await
        .is_err());
    assert_eq!(statistics.queue_wait.snapshot().finished, 1);
    assert!(statistics.queue_wait.snapshot().elapsed_micros >= 20_000);
    assert_eq!(statistics.send.snapshot().started, 1);
    assert_eq!(statistics.send.snapshot().finished, 0);
    assert_eq!(statistics.sent_frames.load(Ordering::Relaxed), 0);
    assert_eq!(statistics.sent_payload_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(credits.observation().used_requests, 1);
    assert_eq!(bytes.available_permits(), 0);
    let (sent, received) = tokio::join!(send, async {
        let length = read.read_u32().await? as usize;
        ensure!(length <= MAX_HEADER, "unexpected test frame header");
        let mut header = vec![0; length];
        read.read_exact(&mut header).await?;
        let mut payload = vec![0; body.len()];
        read.read_exact(&mut payload).await?;
        assert_eq!(payload.as_slice(), body.as_ref());
        Ok::<_, anyhow::Error>(())
    });
    sent?;
    received?;
    assert_eq!(statistics.send.snapshot().finished, 1);
    assert_eq!(statistics.sent_frames.load(Ordering::Relaxed), 1);
    assert_eq!(
        statistics.sent_payload_bytes.load(Ordering::Relaxed),
        body.len() as u64
    );
    assert_eq!(credits.observation().used_requests, 1);
    drop(queued);
    assert_eq!(credits.observation().used_requests, 0);
    assert_eq!(credits.observation().used_bytes, 0);
    assert!(bytes.available_permits() > body.len());
    Ok(())
}

#[tokio::test]
async fn cancelled_partial_send_records_time_without_success_or_credit_leak() -> Result<()> {
    let (queued, credits, bytes, _) = queued()?;
    let statistics = DestinationStatistics::default();
    let (mut write, _blocked_read) = tokio::io::duplex(64);
    assert!(timeout(
        Duration::from_millis(10),
        send_queued(&mut write, &queued, &statistics),
    )
    .await
    .is_err());
    assert_eq!(statistics.queue_wait.snapshot().finished, 1);
    assert_eq!(statistics.send.snapshot().started, 1);
    assert_eq!(statistics.send.snapshot().finished, 1);
    assert!(statistics.send.snapshot().elapsed_micros > 0);
    assert_eq!(statistics.sent_frames.load(Ordering::Relaxed), 0);
    assert_eq!(statistics.sent_payload_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(credits.observation().used_requests, 1);
    assert_eq!(bytes.available_permits(), 0);
    // The send loop drops the uncertain connection and the retained packet.
    drop(write);
    drop(queued);
    assert_eq!(credits.observation().used_requests, 0);
    assert_eq!(credits.observation().used_bytes, 0);
    assert!(bytes.available_permits() > 0);
    Ok(())
}
