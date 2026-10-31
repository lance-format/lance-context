use super::*;

enum Pressure {
    NotReady,
    Bytes,
    Queue,
    Closed,
}

// Exercise the real dispatcher and both ownership budgets. A rejected packet
// must release runtime credits; a full/closed queue must also return its newly
// acquired destination credits. Accepted payloads retain the original Bytes.
async fn observe(pressure: Pressure) -> Result<()> {
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
    let charge = message.compute_size() as usize + 256;
    let credits = super::super::super::Credits::new(3, charge * 3);
    let capacity = if matches!(pressure, Pressure::Bytes) {
        charge
    } else {
        charge * 3
    };
    let bytes = Arc::new(Semaphore::new(capacity));
    let (send, mut receive) = mpsc::channel(1);
    if matches!(pressure, Pressure::Closed) {
        receive.close();
    }
    let route = Route {
        send,
        ready: Arc::new(AtomicBool::new(!matches!(pressure, Pressure::NotReady))),
        bytes: bytes.clone(),
    };
    let (outbox, incoming) = mpsc::channel(3);
    for group in [10, 20, 10] {
        outbox
            .send(Outbound {
                packet: Packet {
                    cluster: uuid::Uuid::nil(),
                    layout: [0; 32],
                    policy: [0; 32],
                    group,
                    message: message.clone(),
                },
                _credit: Some(credits.reserve(charge)?),
                image: None,
                snapshot_feedback: None,
            })
            .await
            .map_err(|_| anyhow::anyhow!("test outbox closed"))?;
    }
    drop(outbox);
    let (_stop, stopped) = watch::channel(false);
    let statistics = Arc::new(Statistics::default());
    dispatch(
        incoming,
        BTreeMap::from([(2, route)]),
        Arc::new(OnceLock::new()),
        stopped,
        statistics.clone(),
    )
    .await?;
    let counts = (
        statistics.enqueue_not_ready.load(Ordering::Relaxed),
        statistics.enqueue_byte_capacity.load(Ordering::Relaxed),
        statistics.enqueue_queue_full.load(Ordering::Relaxed),
        statistics.enqueue_queue_closed.load(Ordering::Relaxed),
    );
    let (expected, accepted) = match pressure {
        Pressure::NotReady => ((3, 0, 0, 0), false),
        Pressure::Bytes => ((0, 2, 0, 0), true),
        Pressure::Queue => ((0, 0, 2, 0), true),
        Pressure::Closed => ((0, 0, 0, 3), false),
    };
    assert_eq!(counts, expected);
    assert_eq!(
        statistics.dropped_outbound.load(Ordering::Relaxed),
        if accepted { 2 } else { 3 }
    );
    assert_eq!(credits.observation().used_requests, usize::from(accepted));
    assert_eq!(
        credits.observation().used_bytes,
        charge * usize::from(accepted)
    );
    assert_eq!(
        bytes.available_permits(),
        capacity - charge * usize::from(accepted)
    );
    if accepted {
        let queued = receive.try_recv()?;
        assert_eq!(
            queued.packet.packet().message.entries[0].data.as_ptr(),
            body.as_ptr()
        );
        drop(queued);
    }
    assert!(receive.try_recv().is_err());
    assert_eq!(credits.observation().used_requests, 0);
    assert_eq!(credits.observation().used_bytes, 0);
    assert_eq!(bytes.available_permits(), capacity);
    Ok(())
}

#[tokio::test]
async fn peer_enqueue_observes_not_ready_and_releases_runtime_credits() -> Result<()> {
    observe(Pressure::NotReady).await
}

#[tokio::test]
async fn peer_enqueue_observes_byte_pressure_separately_from_queue_count() -> Result<()> {
    observe(Pressure::Bytes).await
}

#[tokio::test]
async fn peer_enqueue_observes_full_queue_and_returns_destination_credits() -> Result<()> {
    observe(Pressure::Queue).await
}

#[tokio::test]
async fn peer_enqueue_observes_closed_queue_and_returns_destination_credits() -> Result<()> {
    observe(Pressure::Closed).await
}
