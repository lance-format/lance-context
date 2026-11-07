//! Reserve a small part of the existing destination budget for Raft progress.
use super::*;
use raft_proto::eraftpb::MessageType;

const CONTROL_BYTES: usize = 1024;
const CONTROL_BURST: usize = 4;

pub(super) struct Route {
    send: mpsc::Sender<Queued>,
    control: Option<mpsc::Sender<Queued>>,
    ready: Arc<AtomicBool>,
    pub(super) bytes: Arc<Semaphore>,
    data_bytes: Option<Arc<Semaphore>>,
}

pub(super) struct Receiver {
    data: mpsc::Receiver<Queued>,
    control: Option<mpsc::Receiver<Queued>>,
    controls: usize,
}

impl Route {
    pub(super) fn new(
        slots: usize,
        capacity: usize,
        largest: usize,
        ready: Arc<AtomicBool>,
    ) -> (Self, Receiver) {
        // A one-slot configuration must still admit data. The sum of both
        // channel capacities is exactly the original queue limit.
        let reserved = if slots == 1 {
            0
        } else {
            (slots / 4).clamp(1, 4)
        };
        let (send, data) = mpsc::channel(slots - reserved);
        let (control, receive) = if reserved == 0 {
            (None, None)
        } else {
            let (send, receive) = mpsc::channel(reserved);
            (Some(send), Some(receive))
        };
        // Never shrink below the configured maximum frame. With no spare byte
        // budget the lanes still share the original hard cap, without a reserve.
        let reserve_bytes = (reserved * CONTROL_BYTES).min(capacity.saturating_sub(largest));
        (
            Self {
                send,
                control,
                ready,
                bytes: Arc::new(Semaphore::new(capacity)),
                data_bytes: (reserve_bytes > 0)
                    .then(|| Arc::new(Semaphore::new(capacity - reserve_bytes))),
            },
            Receiver {
                data,
                control: receive,
                controls: 0,
            },
        )
    }

    pub(super) fn enqueue(&self, packet: Outbound, statistics: &Statistics) -> bool {
        if !self.ready.load(Ordering::Acquire) {
            statistics.enqueue_not_ready.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let message = &packet.packet().message;
        let size = message.compute_size().saturating_add(256);
        // Empty Appends stay with data too: sending a later Append first can
        // make a follower reject its predecessor. Oversized contexts and any
        // unexpected payload also use the ordinary lane and byte allowance.
        let control = self.control.as_ref().filter(|_| {
            size as usize <= CONTROL_BYTES
                && message.entries.is_empty()
                && !message.has_snapshot()
                && matches!(
                    message.get_msg_type(),
                    MessageType::MsgHeartbeat
                        | MessageType::MsgHeartbeatResponse
                        | MessageType::MsgAppendResponse
                        | MessageType::MsgRequestVote
                        | MessageType::MsgRequestVoteResponse
                        | MessageType::MsgRequestPreVote
                        | MessageType::MsgRequestPreVoteResponse
                )
        });
        let data_bytes = if control.is_none() {
            match self
                .data_bytes
                .as_ref()
                .map(|b| b.clone().try_acquire_many_owned(size))
            {
                Some(Ok(permit)) => Some(permit),
                Some(Err(_)) => {
                    statistics
                        .enqueue_byte_capacity
                        .fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                None => None,
            }
        } else {
            None
        };
        let Ok(bytes) = self.bytes.clone().try_acquire_many_owned(size) else {
            statistics
                .enqueue_byte_capacity
                .fetch_add(1, Ordering::Relaxed);
            return false;
        };
        match control.unwrap_or(&self.send).try_send(Queued {
            packet,
            _bytes: bytes,
            _data_bytes: data_bytes,
            enqueued: Instant::now(),
        }) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                statistics
                    .enqueue_queue_full
                    .fetch_add(1, Ordering::Relaxed);
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                statistics
                    .enqueue_queue_closed
                    .fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }
}

impl Receiver {
    pub(super) fn try_recv(&mut self) -> Result<Queued, mpsc::error::TryRecvError> {
        if self.controls >= CONTROL_BURST {
            if let Ok(packet) = self.data.try_recv() {
                self.controls = 0;
                return Ok(packet);
            }
        }
        if let Some(control) = &mut self.control {
            if let Ok(packet) = control.try_recv() {
                self.controls = self.controls.saturating_add(1);
                return Ok(packet);
            }
        }
        let packet = self.data.try_recv()?;
        self.controls = 0;
        Ok(packet)
    }

    pub(super) async fn recv(&mut self) -> Option<Queued> {
        if let Ok(packet) = self.try_recv() {
            return Some(packet);
        }
        let Some(control) = &mut self.control else {
            return self.data.recv().await;
        };
        let next = if self.controls >= CONTROL_BURST {
            tokio::select! {
                biased;
                Some(packet) = self.data.recv() => Some((packet, false)),
                Some(packet) = control.recv() => Some((packet, true)),
                else => None,
            }
        } else {
            tokio::select! {
                biased;
                Some(packet) = control.recv() => Some((packet, true)),
                Some(packet) = self.data.recv() => Some((packet, false)),
                else => None,
            }
        };
        next.map(|(packet, control)| {
            self.controls = if control {
                self.controls.saturating_add(1)
            } else {
                0
            };
            packet
        })
    }

    #[cfg(test)]
    pub(super) fn close(&mut self) {
        self.data.close();
        if let Some(control) = &mut self.control {
            control.close();
        }
    }
}
