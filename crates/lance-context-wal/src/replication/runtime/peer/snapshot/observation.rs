//! Latest actual outgoing transfer per configured destination. The destination
//! permit serializes its groups; this is bounded observation, not transfer history.
use super::*;
use std::time::Instant;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "service", derive(serde::Serialize))]
pub struct TransferObservation {
    pub group: u64,
    pub to: u64,
    pub term: u64,
    pub index: u64,
    /// Local to this runtime and group. It is not a request or publication ID.
    pub attempt: u64,
    pub metadata_bytes: u64,
    pub payload_bytes: u64,
    pub started_ago_ms: u64,
    /// Running elapsed time, or the fixed duration of a completed transfer.
    pub elapsed_ms: u64,
    /// None while running; true only after the remote durable installation ACK.
    /// Raft can still ignore the subsequent feedback if its term has changed.
    pub delivered: Option<bool>,
}

pub(super) struct Transfer {
    observation: TransferObservation,
    started: Instant,
    finished: Option<Instant>,
}

fn millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

impl Statistics {
    pub fn outbound_observations(&self) -> Vec<TransferObservation> {
        self.outgoing
            .values()
            .filter_map(|slot| {
                let guard = slot.lock().ok()?;
                let transfer = guard.as_ref()?;
                let now = Instant::now();
                let mut view = transfer.observation.clone();
                view.started_ago_ms = millis(now.duration_since(transfer.started));
                view.elapsed_ms = millis(
                    transfer
                        .finished
                        .unwrap_or(now)
                        .duration_since(transfer.started),
                );
                Some(view)
            })
            .collect()
    }

    pub(super) fn started(&self, outgoing: &Outbound) {
        let packet = outgoing.packet();
        let Some(slot) = self.outgoing.get(&packet.message.to) else {
            return;
        };
        let Some(attempt) = outgoing.snapshot_attempt() else {
            return;
        };
        let Some(image) = outgoing.snapshot_image() else {
            return;
        };
        let Ok(mut guard) = slot.lock() else {
            return;
        };
        *guard = Some(Transfer {
            observation: TransferObservation {
                group: packet.group,
                to: packet.message.to,
                term: packet.message.term,
                index: packet.message.get_snapshot().get_metadata().index,
                attempt,
                metadata_bytes: image.bytes(Part::Metadata),
                payload_bytes: image.bytes(Part::Payload),
                started_ago_ms: 0,
                elapsed_ms: 0,
                delivered: None,
            },
            started: Instant::now(),
            finished: None,
        });
    }

    pub(super) fn finished(&self, outgoing: &Outbound, delivered: bool) {
        let packet = outgoing.packet();
        let Some(slot) = self.outgoing.get(&packet.message.to) else {
            return;
        };
        let Ok(mut guard) = slot.lock() else {
            return;
        };
        let Some(transfer) = guard.as_mut() else {
            return;
        };
        if transfer.observation.group == packet.group
            && Some(transfer.observation.attempt) == outgoing.snapshot_attempt()
        {
            transfer.finished = Some(Instant::now());
            transfer.observation.delivered = Some(delivered);
        }
    }
}
