//! Owner-thread transitions for separately scheduled image and maintenance work.
//! Control credits charge the scheduling envelope. Image tasks own a separate
//! physical-node permit and their captured state is bounded by Node policy, not
//! by the control-lane byte allowance. Callers must budget and join that work.
use super::*;
use crate::{
    raft_log::LogMaintenance,
    replication::{CheckpointTask, ImageCollection, PreparedCheckpoint, SnapshotReceiver},
};

pub(super) struct CheckpointWork {
    image: PreparedCheckpoint,
    retained_bytes: u64,
    reply: Completion<()>,
    started: bool,
    previous: Option<Arc<crate::replication::checkpoint::Image>>,
}

impl CheckpointWork {
    fn finish(self: Box<Self>, result: Result<()>) {
        let Self {
            image,
            previous,
            reply,
            ..
        } = *self;
        // A sibling group can receive the caller's next checkpoint as soon as
        // the reply wakes it. Release physical-node admission before that wake.
        drop(image);
        drop(previous);
        reply.finish(result);
    }
}

#[cfg(all(test, feature = "peer-transport"))]
#[path = "images/completion_test.rs"]
mod completion_test;

// Every queued result retains its original outgoing credit. The lane has as
// many slots as the entire physical node's outgoing allowance, so it cannot
// fill while another valid owner is trying to return its slot. Control-lane
// saturation must never suppress snapshot failure feedback.
pub(super) struct SnapshotResult {
    to: u64,
    term: u64,
    index: u64,
    attempt: u64,
    delivered: bool,
    pub(super) credit: Credit,
}

pub(super) struct SnapshotFeedback {
    to: u64,
    term: u64,
    index: u64,
    attempt: u64,
    send: mpsc::SyncSender<SnapshotResult>,
    wake: Thread,
    shared: Arc<Shared>,
}
impl SnapshotFeedback {
    fn send(self, credit: Credit, delivered: bool) -> Result<()> {
        let result = self.send.try_send(SnapshotResult {
            to: self.to,
            term: self.term,
            index: self.index,
            attempt: self.attempt,
            delivered,
            credit,
        });
        if result.is_err() {
            // Disconnection during shutdown is expected. Otherwise fence the
            // node instead of silently abandoning Raft's pending snapshot.
            self.shared.stopped.store(true, Ordering::Release);
        }
        self.wake.unpark();
        result.map_err(|_| anyhow::anyhow!("snapshot feedback unavailable; runtime stopped"))
    }
}

impl Outbound {
    /// Local attempt identity for transport observation, outside the wire protocol.
    pub(super) fn snapshot_attempt(&self) -> Option<u64> {
        self.snapshot_feedback
            .as_ref()
            .map(|feedback| feedback.attempt)
    }

    /// Consume a snapshot owner and return its transfer result on a reserved
    /// lane. Report success only after the remote durable installation response.
    /// This queues local Raft feedback; it is not a client durability receipt.
    /// Dropping an unfinished owner automatically reports failure, including
    /// queue rejection, network cancellation and disconnection.
    pub fn finish_snapshot(mut self, delivered: bool) -> Result<()> {
        let feedback = self
            .snapshot_feedback
            .take()
            .context("outbound packet is not a snapshot")?;
        feedback.send(self._credit.take().expect("outbound credit"), delivered)
    }
}
impl Drop for Outbound {
    fn drop(&mut self) {
        if let Some(feedback) = self.snapshot_feedback.take() {
            let _ = feedback.send(self._credit.take().expect("outbound credit"), false);
        }
    }
}

impl Worker {
    pub(super) fn checkpoint_draining(&self) -> bool {
        self.checkpoint.as_ref().is_some_and(|work| !work.started)
    }

    pub(super) fn queue_checkpoint(&mut self, work: Box<CheckpointWork>) {
        if self.checkpoint.is_some() {
            work.finish(Err(anyhow::anyhow!("checkpoint already queued")));
            return;
        }
        self.node
            .groups
            .get_mut(&self.group)
            .unwrap()
            .drain_for_checkpoint(true);
        // Retain the original completion, control credit and physical-node
        // image permit across every poll. No task or waiter is spawned here.
        self.checkpoint = Some(work);
    }

    pub(super) fn install_drained_checkpoint(
        &mut self,
        after_poll: &mut Vec<AfterPoll>,
    ) -> Result<bool> {
        let Some(work) = self.checkpoint.as_mut() else {
            return Ok(false);
        };
        if work.started {
            if self.node.checkpoint_index(self.group)? < work.image.index() {
                return Ok(false);
            }
            let work = self.checkpoint.take().unwrap();
            let result = self.node.finish_async_checkpoint(&work.image);
            after_poll.push(Box::new(move || work.finish(result)));
            return Ok(true);
        }
        if !self.node.groups[&self.group].persistence_idle() {
            return Ok(false);
        }
        if self.options.async_persistence {
            match self
                .node
                .start_async_checkpoint(&work.image, work.retained_bytes)
            {
                Ok(previous) => {
                    work.previous = previous;
                    work.started = true;
                }
                Err(error) => {
                    let work = self.checkpoint.take().unwrap();
                    after_poll.push(Box::new(move || work.finish(Err(error))));
                }
            }
            self.node
                .groups
                .get_mut(&self.group)
                .unwrap()
                .drain_for_checkpoint(false);
            return Ok(true);
        }
        let CheckpointWork {
            image,
            retained_bytes,
            reply,
            ..
        } = *self.checkpoint.take().unwrap();
        self.node
            .groups
            .get_mut(&self.group)
            .unwrap()
            .drain_for_checkpoint(false);
        // The synchronous runtime retains its original installation protocol.
        let result = self
            .node
            .install_checkpoint_retaining(image, retained_bytes);
        after_poll.push(Box::new(move || reply.finish(result)));
        Ok(true)
    }

    pub(super) fn snapshot_feedback(&mut self, packet: &Packet) -> Result<SnapshotFeedback> {
        self.snapshot_attempt = self
            .snapshot_attempt
            .checked_add(1)
            .context("snapshot attempt identities exhausted")?;
        self.snapshot_attempts
            .insert(packet.message.to, self.snapshot_attempt);
        Ok(SnapshotFeedback {
            to: packet.message.to,
            term: packet.message.term,
            index: packet.message.get_snapshot().get_metadata().index,
            attempt: self.snapshot_attempt,
            send: self.snapshot_feedback.clone(),
            wake: thread::current(),
            shared: self.shared.clone(),
        })
    }

    pub(super) fn snapshot_result(&mut self, result: &SnapshotResult) -> Result<()> {
        if self.snapshot_attempts.get(&result.to) != Some(&result.attempt) {
            return Ok(());
        }
        self.snapshot_attempts.remove(&result.to);
        let raft = &self.node.groups[&self.group].raw.raft;
        if raft.term == result.term
            && raft
                .prs()
                .get(result.to)
                .is_some_and(|progress| progress.pending_snapshot == result.index)
        {
            self.node
                .report_snapshot(self.group, result.to, result.delivered)?;
        }
        Ok(())
    }
}

impl Handle {
    pub fn disk_budget(&self) -> Option<crate::admission::Snapshot> {
        self.disk.as_ref().map(|budget| budget.snapshot())
    }

    #[cfg(feature = "service")]
    pub(crate) fn data_checkpoint_pressure(&self) -> bool {
        self.disk_budget().is_some_and(|disk| {
            let proposal = self
                .max_record
                .saturating_add(crate::replication::PROPOSAL_DISK_OVERHEAD);
            // Native pressure rewriting starts with two target files of room
            // left. Checkpoint before copying a prefix that the image can
            // retire, even if another maximum-size proposal still fits.
            // Reservations alone cannot be reclaimed by a checkpoint.
            disk.data_limit.saturating_sub(disk.accounted_disk_bytes)
                < self.log_file_headroom.max(proposal as u64)
        })
    }

    /// Capture a blocking directory measurement retaining the physical engine.
    /// Run this on the maintenance worker, never on an async executor thread.
    pub fn prepare_disk_measurement(&self) -> Result<Pending<crate::raft_log::DiskMeasurement>> {
        let group = *self.routes.keys().next().context("no hosted group")?;
        self.local(group, |node| node.prepare_disk_measurement())
    }
    /// Validate exact incoming image lengths without creating files. The
    /// authenticated transport uses this before reserving physical disk space.
    pub fn snapshot_sizes(&self, packet: Packet) -> Result<Pending<[u64; 2]>> {
        self.local(packet.group, move |node| node.snapshot_sizes(&packet))
    }

    // Only these typed library operations can access the owner. Their result is
    // held until the next successful durable poll; an error/panic drops the
    // completion as uncertain. Heavy image work stays in the returned task.
    pub(super) fn local<T: Send + 'static>(
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

    /// Queue the service's single maintenance task behind other image owners.
    /// No worker, control credit or captured cut is retained during admission.
    #[cfg(feature = "service")]
    pub(crate) async fn prepare_checkpoint_wait(
        &self,
        group: u64,
        id: uuid::Uuid,
    ) -> Result<Pending<Option<CheckpointTask>>> {
        self.route(group)?;
        let permit = self
            .images
            .upgrade()
            .context("node image owner stopped")?
            .acquire_queued()
            .await?;
        self.local(group, move |node| {
            node.prepare_checkpoint_admitted(group, id, permit)
        })
    }

    /// Install a completed local image and logically compact its prefix. The
    /// completion follows persistence. Later suffix entries stay in the log.
    pub fn install_checkpoint(&self, image: PreparedCheckpoint) -> Result<Pending<()>> {
        self.install_checkpoint_retaining(image, 0)
    }

    /// Install an application checkpoint with a byte-bounded replication tail.
    pub fn install_checkpoint_retaining(
        &self,
        image: PreparedCheckpoint,
        retained_bytes: u64,
    ) -> Result<Pending<()>> {
        let route = self.route(image.group())?;
        let credit = self.shared.control.reserve(4096)?;
        let (reply, receive) = oneshot::channel();
        self.enqueue(
            route,
            Work::Checkpoint(Box::new(CheckpointWork {
                image,
                retained_bytes,
                started: false,
                previous: None,
                reply: Completion {
                    reply,
                    _credit: credit,
                },
            })),
            false,
        )?;
        Ok(Pending(receive))
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

    /// Manual control-lane reporting for callers managing Node-level traffic.
    /// Runtime transports must consume `Outbound::finish_snapshot` instead: it
    /// retains reserved feedback capacity and rejects stale transfer outcomes.
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

    #[cfg(feature = "service")]
    pub(crate) async fn prepare_image_collection_wait(&self) -> Result<Pending<ImageCollection>> {
        let group = self.physical_group()?;
        let permit = self
            .images
            .upgrade()
            .context("node image owner stopped")?
            .acquire_queued()
            .await?;
        self.local(group, move |node| {
            node.prepare_image_collection_admitted(permit)
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
