use crate::admission::DiskBudgetExhausted;
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
    pub last_disk_refusal: Option<PeerDiskRefusal>,
}
impl PeerRejections {
    pub(super) fn record(&mut self, group: u64, from: u64, error: &anyhow::Error) {
        if let Some(budget) = error.downcast_ref::<DiskBudgetExhausted>() {
            self.disk_budget = self.disk_budget.saturating_add(1);
            self.last_disk_refusal = Some(PeerDiskRefusal {
                group,
                from,
                budget: budget.clone(),
            });
        } else {
            self.other = self.other.saturating_add(1);
        }
    }
}
