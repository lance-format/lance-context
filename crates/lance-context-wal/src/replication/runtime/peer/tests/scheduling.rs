use super::*;

// Exercise dispatch and both credit owners with a ready destination consumer.
// This isolates scheduling from network delays: no persistence or application
// acknowledgements are simulated by the consumer.
#[tokio::test(flavor = "current_thread")]
async fn ready_peer_receiver_progresses_during_burst_dispatch() -> Result<()> {
    for offered in [32, 128, 512] {
        let queue_capacity = 32;
        let mut message = Message {
            from: 1,
            to: 2,
            term: 1,
            ..Default::default()
        };
        message.set_msg_type(MessageType::MsgHeartbeat);
        let charge = message.compute_size() as usize + 256;
        let credits = super::super::super::Credits::new(offered, charge * offered);
        let bytes = Arc::new(Semaphore::new(charge * offered));
        let (send, mut receive) = mpsc::channel::<Queued>(queue_capacity);
        let route = Route {
            send,
            ready: Arc::new(AtomicBool::new(true)),
            bytes: bytes.clone(),
        };
        let (ready, waiting) = tokio::sync::oneshot::channel();
        let consumer = tokio::spawn(async move {
            let _ = ready.send(());
            let mut delivered = 0;
            while let Some(queued) = receive.recv().await {
                assert_eq!(
                    queued.packet.packet().message.get_msg_type(),
                    MessageType::MsgHeartbeat
                );
                delivered += 1;
            }
            delivered
        });
        waiting.await?;
        let (outbox, incoming) = mpsc::channel(offered);
        for _ in 0..offered {
            outbox
                .try_send(Outbound {
                    packet: Packet {
                        cluster: uuid::Uuid::nil(),
                        layout: [0; 32],
                        policy: [0; 32],
                        group: 10,
                        message: message.clone(),
                    },
                    _credit: Some(credits.reserve(message.compute_size() as usize)?),
                    image: None,
                    snapshot_feedback: None,
                })
                .map_err(|_| anyhow::anyhow!("diagnostic outbox admission failed"))?;
        }
        drop(outbox);
        let (_stop, stopped) = watch::channel(false);
        let statistics = Arc::new(Statistics::default());
        tokio::spawn(dispatch(
            incoming,
            BTreeMap::from([(2, route)]),
            Arc::new(OnceLock::new()),
            stopped,
            statistics.clone(),
        ))
        .await??;
        let delivered = consumer.await?;
        let dropped = statistics.dropped_outbound.load(Ordering::Relaxed);
        let queue_full = statistics.enqueue_queue_full.load(Ordering::Relaxed);
        assert_eq!(delivered + dropped as usize, offered);
        assert_eq!(delivered, offered);
        assert_eq!(dropped, 0);
        assert_eq!(queue_full, dropped);
        assert_eq!(statistics.enqueue_byte_capacity.load(Ordering::Relaxed), 0);
        assert_eq!(credits.observation().used_requests, 0);
        assert_eq!(credits.observation().used_bytes, 0);
        assert_eq!(bytes.available_permits(), charge * offered);
    }
    Ok(())
}
