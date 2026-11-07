use super::*;

fn outbound(kind: MessageType, index: u64, body: Bytes) -> Outbound {
    let mut message = Message {
        from: 1,
        to: 2,
        index,
        ..Default::default()
    };
    message.set_msg_type(kind);
    if !body.is_empty() {
        message.set_entries(
            vec![Entry {
                index,
                term: 1,
                data: body,
                ..Default::default()
            }]
            .into(),
        );
    }
    Outbound {
        packet: Packet {
            cluster: uuid::Uuid::nil(),
            layout: [0; 32],
            policy: [0; 32],
            group: 10,
            message,
        },
        _credit: None,
        image: None,
        snapshot_feedback: None,
    }
}

// The production destination queue has no draining sender while the previous
// frame is blocked. Data pressure must leave a bounded path for a durable ACK.
#[tokio::test]
async fn full_data_queue_still_admits_a_raft_ack() -> Result<()> {
    let (route, mut receive) = Route::new(32, 32 << 20, 4 << 20, Arc::new(AtomicBool::new(true)));
    let statistics = Statistics::default();
    let body = Bytes::from(vec![0xa5; 64 << 10]);
    let mut accepted = 0;
    for index in 1..=33 {
        if !route.enqueue(
            outbound(MessageType::MsgAppend, index, body.clone()),
            &statistics,
        ) {
            break;
        }
        accepted += 1;
    }
    assert!(accepted > 0 && accepted < 33);
    assert!(
        route.enqueue(
            outbound(MessageType::MsgAppendResponse, 99, Bytes::new()),
            &statistics,
        ),
        "a full data queue dropped a small Raft ACK"
    );
    let first = receive.recv().await.context("missing queued ACK")?;
    assert_eq!(
        first.packet.packet().message.get_msg_type(),
        MessageType::MsgAppendResponse
    );
    for index in 1..=accepted {
        let next = receive.recv().await.context("missing queued data")?;
        assert_eq!(next.packet.packet().message.index, index);
        assert_eq!(
            next.packet.packet().message.entries[0].data.as_ptr(),
            body.as_ptr()
        );
    }
    Ok(())
}

fn route(slots: usize, capacity: usize, largest: usize) -> (Route, Receiver) {
    Route::new(slots, capacity, largest, Arc::new(AtomicBool::new(true)))
}

#[tokio::test]
async fn control_flood_is_bounded_and_cannot_starve_ordered_appends() -> Result<()> {
    let (route, mut receive) = route(32, 32 << 20, 4 << 20);
    let statistics = Statistics::default();
    let body = Bytes::from_static(b"original payload");
    for index in 1..=28 {
        // Include empty Appends between nonempty ones. They must not overtake.
        let payload = if index % 2 == 0 {
            Bytes::new()
        } else {
            body.clone()
        };
        assert!(route.enqueue(
            outbound(MessageType::MsgAppend, index, payload),
            &statistics
        ));
    }
    for index in 1..=4 {
        assert!(route.enqueue(
            outbound(MessageType::MsgHeartbeat, index, Bytes::new()),
            &statistics
        ));
    }
    assert!(!route.enqueue(
        outbound(MessageType::MsgHeartbeat, 5, Bytes::new()),
        &statistics
    ));
    assert!(!route.enqueue(
        outbound(MessageType::MsgAppend, 29, body.clone()),
        &statistics
    ));
    for index in 1..=28 {
        // Keep refilling the control lane, including while data is ready.
        for _ in 0..4 {
            let queued = receive.recv().await.context("missing control")?;
            assert_eq!(
                queued.packet.packet().message.get_msg_type(),
                MessageType::MsgHeartbeat
            );
            drop(queued);
            assert!(route.enqueue(
                outbound(MessageType::MsgHeartbeat, 99, Bytes::new()),
                &statistics
            ));
        }
        let queued = receive.recv().await.context("data was starved")?;
        assert_eq!(
            queued.packet.packet().message.get_msg_type(),
            MessageType::MsgAppend
        );
        assert_eq!(queued.packet.packet().message.index, index);
        if index % 2 != 0 {
            assert_eq!(
                queued.packet.packet().message.entries[0].data.as_ptr(),
                body.as_ptr()
            );
        }
    }
    drop(route);
    for _ in 0..4 {
        assert!(receive.recv().await.is_some());
    }
    assert!(receive.recv().await.is_none());
    Ok(())
}

#[test]
fn data_byte_pressure_preserves_ack_capacity_without_expanding_total_budget() -> Result<()> {
    let (route, mut receive) = route(32, 16 << 10, 4 << 10);
    let statistics = Statistics::default();
    let body = Bytes::from(vec![0x42; 4 << 10]);
    let credits = crate::replication::runtime::Credits::new(32, 32 << 10);
    let mut accepted = 0;
    for index in 1..=33 {
        let mut packet = outbound(MessageType::MsgAppend, index, body.clone());
        packet._credit = Some(credits.reserve(packet.packet().message.compute_size() as usize)?);
        if !route.enqueue(packet, &statistics) {
            break;
        }
        accepted += 1;
    }
    assert_eq!(statistics.enqueue_byte_capacity.load(Ordering::Relaxed), 1);
    assert_eq!(credits.observation().used_requests, accepted);
    for kind in [
        MessageType::MsgAppendResponse,
        MessageType::MsgHeartbeatResponse,
        MessageType::MsgRequestVoteResponse,
        MessageType::MsgRequestPreVoteResponse,
    ] {
        let mut packet = outbound(kind, 99, Bytes::new());
        packet._credit = Some(credits.reserve(packet.packet().message.compute_size() as usize)?);
        assert!(route.enqueue(packet, &statistics));
    }
    assert_eq!(credits.observation().used_requests, accepted + 4);
    // Dequeue is not send completion: destination and runtime bytes stay held.
    let before = route.bytes.available_permits();
    let in_flight = receive.try_recv()?;
    assert_eq!(route.bytes.available_permits(), before);
    assert!(route.bytes.available_permits() >= 3 << 10);
    drop(in_flight);
    while receive.try_recv().is_ok() {}
    assert_eq!(credits.observation().used_requests, 0);
    assert_eq!(credits.observation().used_bytes, 0);
    assert_eq!(route.bytes.available_permits(), 16 << 10);
    // The data-only byte permits were released too.
    assert!(route.enqueue(outbound(MessageType::MsgAppend, 100, body), &statistics));
    drop(receive);
    assert_eq!(route.bytes.available_permits(), 16 << 10);
    Ok(())
}

#[test]
fn priority_never_bypasses_frame_size_or_payload_limits() -> Result<()> {
    let (route, mut receive) = route(2, 32 << 10, 4 << 10);
    let statistics = Statistics::default();
    assert!(route.enqueue(
        outbound(MessageType::MsgAppend, 1, Bytes::new()),
        &statistics
    ));
    for kind in [
        MessageType::MsgHeartbeat,
        MessageType::MsgHeartbeatResponse,
        MessageType::MsgAppendResponse,
        MessageType::MsgRequestVote,
        MessageType::MsgRequestVoteResponse,
        MessageType::MsgRequestPreVote,
        MessageType::MsgRequestPreVoteResponse,
    ] {
        let mut oversized = outbound(kind, 2, Bytes::new());
        oversized.packet.message.context = Bytes::from(vec![0; 1024]);
        assert!(!route.enqueue(oversized, &statistics));
        assert!(!route.enqueue(
            outbound(kind, 2, Bytes::from_static(b"payload")),
            &statistics
        ));
        assert!(route.enqueue(outbound(kind, 3, Bytes::new()), &statistics));
        // Consume each small control so capacity is available for the next kind.
        // The data lane remains blocked for this admission-only check.
        let mut held_data = None;
        loop {
            let queued = receive.try_recv()?;
            if queued.packet.packet().message.get_msg_type() == kind {
                break;
            }
            held_data = Some(queued);
        }
        if let Some(queued) = held_data {
            assert!(route.enqueue(queued.packet, &statistics));
        }
    }
    drop(receive);
    assert!(!route.enqueue(
        outbound(MessageType::MsgHeartbeat, 4, Bytes::new()),
        &statistics
    ));
    assert_eq!(route.bytes.available_permits(), 32 << 10);
    Ok(())
}

#[test]
fn smallest_queue_and_maximum_frame_remain_admissible() -> Result<()> {
    let statistics = Statistics::default();
    let packet = outbound(MessageType::MsgAppend, 1, Bytes::from(vec![1; 4096]));
    let capacity = packet.packet().message.compute_size() as usize + 256;
    for slots in [1, 2, 32, 256] {
        let (route, mut receive) = route(slots, capacity, capacity);
        let packet = outbound(MessageType::MsgAppend, 1, Bytes::from(vec![1; 4096]));
        assert!(route.enqueue(packet, &statistics));
        assert_eq!(route.bytes.available_permits(), 0);
        assert!(!route.enqueue(
            outbound(MessageType::MsgHeartbeat, 2, Bytes::new()),
            &statistics
        ));
        drop(receive.try_recv()?);
        assert_eq!(route.bytes.available_permits(), capacity);
        assert!(route.enqueue(
            outbound(MessageType::MsgHeartbeat, 2, Bytes::new()),
            &statistics
        ));
        receive.close();
        assert!(!route.enqueue(
            outbound(MessageType::MsgAppend, 3, Bytes::new()),
            &statistics
        ));
        assert!(!route.enqueue(
            outbound(MessageType::MsgHeartbeat, 3, Bytes::new()),
            &statistics
        ));
        drop(receive);
        assert_eq!(route.bytes.available_permits(), capacity);
    }
    Ok(())
}
