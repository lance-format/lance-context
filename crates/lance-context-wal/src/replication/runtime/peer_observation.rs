use crate::admission::DiskBudgetExhausted;
use raft_proto::eraftpb::MessageType;
use serde::Serialize;

/// Decision-time accounting for the most recent peer disk admission refusal.
/// This remains historical after space is reclaimed; it is not current usage.
#[derive(Clone, Debug, Serialize)]
pub struct PeerDiskRefusal {
    pub group: u64,
    pub from: u64,
    pub budget: DiskBudgetExhausted,
}

/// Bounded, process-lifetime observation of Node::step failures. No payloads,
/// error strings or per-request history are retained. Successful steps do not
/// lock or update this observation. Counters saturate instead of wrapping.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PeerRejections {
    pub disk_budget: u64,
    pub other: u64,
    /// Subset of `other`: bounded native persistence admission refused a step.
    pub pending_persistence: u64,
    /// Subset of `pending_persistence`: heartbeat or heartbeat-response packets.
    pub pending_heartbeats: u64,
    pub last_disk_refusal: Option<PeerDiskRefusal>,
}
impl PeerRejections {
    pub(super) fn record(
        &mut self,
        group: u64,
        from: u64,
        kind: MessageType,
        error: &anyhow::Error,
    ) {
        if let Some(budget) = error.downcast_ref::<DiskBudgetExhausted>() {
            self.disk_budget = self.disk_budget.saturating_add(1);
            self.last_disk_refusal = Some(PeerDiskRefusal {
                group,
                from,
                budget: budget.clone(),
            });
        } else {
            self.other = self.other.saturating_add(1);
            if error.is::<super::super::async_persist::PeerPersistenceCapacity>() {
                self.pending_persistence = self.pending_persistence.saturating_add(1);
                if matches!(
                    kind,
                    MessageType::MsgHeartbeat | MessageType::MsgHeartbeatResponse
                ) {
                    self.pending_heartbeats = self.pending_heartbeats.saturating_add(1);
                }
            }
        }
    }
}
