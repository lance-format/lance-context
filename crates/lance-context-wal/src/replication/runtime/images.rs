//! Owner-thread transitions for separately scheduled image and maintenance work.
//! Control credits charge the scheduling envelope. Image tasks own a separate
//! physical-node permit and their captured state is bounded by Node policy, not
//! by the control-lane byte allowance. Callers must budget and join that work.
use super::*;
use crate::{
    raft_log::LogMaintenance,
    replication::{CheckpointTask, ImageCollection, PreparedCheckpoint, SnapshotReceiver},
};

impl Handle {
    // Only these typed library operations can access the owner. Their result is
    // held until the next successful durable poll; an error/panic drops the
    // completion as uncertain. Heavy image work stays in the returned task.
    fn local<T: Send + 'static>(
        &self,
        group: u64,
        operation: impl FnOnce(&mut Node) -> Result<T> + Send + 'static,
    ) -> Result<Pending<T>> {
        let route = self.route(group)?;
        let credit = self.shared.control.reserve(4096)?;
        let (reply, receive) = oneshot::channel();
        let completion = Completion {
            reply,
            _credit: credit,
        };
        self.enqueue(
            route,
            Work::Local(Box::new(move |node| {
                let result = operation(node);
                Box::new(move || completion.finish(result))
            })),
            false,
        )?;
        Ok(Pending(receive))
    }

    /// Capture an applied cut. Run the returned task on an admitted blocking
    /// worker, then install its result. Its physical-node permit spans all
    /// sibling groups and remains owned after this future is consumed.
    /// Do not enable compaction before the transport serves snapshot messages.
    pub fn prepare_checkpoint(
        &self,
        group: u64,
        id: uuid::Uuid,
    ) -> Result<Pending<CheckpointTask>> {
        self.local(group, move |node| node.prepare_checkpoint(group, id))
    }

    /// Install a completed local image and logically compact its prefix. The
    /// completion follows persistence. Later suffix entries stay in the log.
    pub fn install_checkpoint(&self, image: PreparedCheckpoint) -> Result<Pending<()>> {
        self.local(image.group(), move |node| node.install_checkpoint(image))
    }

    /// Create an authenticated snapshot receiver after the caller has reserved
    /// transfer and disk capacity. Feed sequential chunks and finish it on an
    /// admitted blocking worker. This admission is not a peer acknowledgment.
    pub fn receive_snapshot(
        &self,
        packet: Packet,
        local_id: uuid::Uuid,
    ) -> Result<Pending<SnapshotReceiver>> {
        self.route(packet.group)?;
        ensure!(
            packet.cluster == self.cluster
                && packet.layout == self.layout
                && packet.policy == self.policy
                && packet.message.to == self.id
                && packet.message.from != self.id
                && self.voters[&packet.group].contains(&packet.message.from)
                && packet.message.get_msg_type() == raft_proto::eraftpb::MessageType::MsgSnapshot
                && packet.message.entries.is_empty()
                && packet.message.has_snapshot()
                && packet.message.compute_size() <= 4096,
            "invalid runtime snapshot route or descriptor"
        );
        self.local(packet.group, move |node| {
            node.receive_snapshot(packet, local_id)
        })
    }

    /// Hand a fully validated received image back to Raft. The completion waits
    /// for durable poll; Raft may ignore an obsolete cut. It is not a write ACK.
    pub fn step_snapshot(&self, image: PreparedCheckpoint) -> Result<Pending<()>> {
        self.local(image.group(), move |node| node.step_snapshot(image))
    }

    /// Report the original snapshot transfer outcome to its sender's Raft group.
    /// A successful socket write alone does not prove follower persistence.
    pub fn report_snapshot(&self, group: u64, to: u64, delivered: bool) -> Result<Pending<()>> {
        self.local(group, move |node| {
            node.report_snapshot(group, to, delivered)
        })
    }

    fn physical_group(&self) -> Result<u64> {
        self.routes
            .first_key_value()
            .map(|(&group, _)| group)
            .context("runtime has no groups")
    }

    /// Capture one resumable physical-node image scan, including sibling groups.
    /// Run bounded batches outside the owner; keep the cursor until finished.
    pub fn prepare_image_collection(&self) -> Result<Pending<ImageCollection>> {
        self.local(self.physical_group()?, |node| {
            node.prepare_image_collection()
        })
    }

    /// Capture one physical-engine maintenance handle, not one per Raft group.
    /// Its native pass can delay all persistence and requires rewrite headroom.
    /// The embedding service must serialize and join this blocking work before
    /// shutdown/reopen. Returned handles keep the original engine lock alive.
    pub fn prepare_log_maintenance(&self) -> Result<Pending<LogMaintenance>> {
        self.local(self.physical_group()?, |node| {
            node.prepare_log_maintenance()
        })
    }
}
