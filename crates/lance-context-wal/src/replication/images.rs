use super::*;
use checkpoint::{Cut, Image, Loaded, Part};
use raft_proto::eraftpb::Snapshot;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, Weak,
    },
};

mod collection;
pub use collection::{ImageCollection, ImageCollectionProgress};

pub(super) struct Owner {
    directory: PathBuf,
    busy: AtomicBool,
    groups: BTreeSet<u64>,
    registry: Mutex<Registry>,
    // Images keep this owner alive. Its engine lock excludes a new incarnation
    // until readers and transfers from the old incarnation have released them.
    store: GroupLog,
}

#[derive(Default)]
struct Registry {
    live: BTreeMap<(u64, Uuid), Weak<Image>>,
    // A split group owner may be dropped while its siblings keep running.
    // Its durable recovery image remains required even without a live Arc.
    installed: BTreeMap<u64, (u64, Uuid)>,
}
impl Owner {
    pub(super) fn configure_disk(
        self: &Arc<Self>,
        options: crate::raft_log::DiskOptions,
    ) -> Result<()> {
        let _permit = self.acquire()?;
        self.store.configure_disk(options)
    }
    pub(super) fn store(&self) -> &GroupLog {
        &self.store
    }
    pub fn new(directory: &Path, groups: BTreeSet<u64>, store: GroupLog) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            directory: directory.canonicalize()?,
            busy: AtomicBool::new(false),
            groups,
            registry: Mutex::new(Registry::default()),
            store,
        }))
    }

    pub fn track(self: &Arc<Self>, image: &Arc<Image>) -> Result<()> {
        let key = image.identity();
        ensure!(self.groups.contains(&key.0), "unhosted application image");
        image.keep_owner(self)?;
        let mut registry = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("image registry poisoned"))?;
        let live = &mut registry.live;
        live.retain(|_, image| image.strong_count() > 0);
        if let Some(previous) = live.get(&key).and_then(Weak::upgrade) {
            ensure!(
                Arc::ptr_eq(&previous, image),
                "duplicate live image identity"
            );
        } else {
            live.insert(key, Arc::downgrade(image));
        }
        Ok(())
    }

    pub fn installed(self: &Arc<Self>, image: &Arc<Image>) -> Result<()> {
        self.track(image)?;
        let mut registry = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("image registry poisoned"))?;
        let key = image.identity();
        registry.installed.insert(key.0, key);
        Ok(())
    }
    fn acquire(self: &Arc<Self>) -> Result<Permit> {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| anyhow::anyhow!("node image builder/receiver busy"))?;
        Ok(Permit(self.clone()))
    }
}
struct Permit(Arc<Owner>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.busy.store(false, Ordering::Release);
    }
}

/// Captures a committed application cut without copying payload bodies. Run it
/// on a bounded blocking worker; the node may keep polling and appending. One
/// build/receive permit spans every group sharing this physical node. It also
/// prevents another checkpoint from compacting this task's captured log input.
pub struct CheckpointTask {
    permit: Permit,
    store: GroupLog,
    previous: Option<Arc<Image>>,
    layout: Arc<Layout>,
    limits: Limits,
    cut: Cut,
    state: State,
}
impl CheckpointTask {
    pub fn run(self) -> Result<PreparedCheckpoint> {
        let Self {
            permit,
            store,
            previous,
            layout,
            limits,
            cut,
            state,
        } = self;
        let group = cut.group;
        let _disk = permit
            .0
            .store
            .disk_budget()
            .map(|budget| {
                budget.reserve_files(&checkpoint::build_sizes(&layout, &limits, &cut, &state)?)
            })
            .transpose()?;
        let loaded = checkpoint::build(
            &permit.0.directory,
            &layout,
            &limits,
            cut,
            state,
            |index, meta| read_body(&store, previous.as_deref(), index, meta),
        )?;
        permit.0.track(&loaded.image)?;
        Ok(PreparedCheckpoint {
            permit,
            group,
            loaded,
            received: None,
        })
    }
}

/// Complete, validated and synced files. This value cannot be constructed from
/// unverified descriptors. Holding it reserves the node's image worker slot;
/// dropping it abandons the candidate and retains its files as orphans.
pub struct PreparedCheckpoint {
    permit: Permit,
    group: u64,
    loaded: Loaded,
    received: Option<Packet>,
}
impl PreparedCheckpoint {
    /// The hosted group bound to these fully validated image files.
    pub fn group(&self) -> u64 {
        self.group
    }
    pub fn image(&self) -> Arc<Image> {
        self.loaded.image.clone()
    }
}

/// Bounded transfer state. Socket I/O stays outside the Raft owner; move this
/// object to the admitted worker and feed exact, sequential file chunks.
pub struct SnapshotReceiver {
    packet: Packet,
    receiver: checkpoint::Receiver,
    // Declared after receiver: buffered file writes must finish before the
    // pending charge turns into counted retained growth, including on Drop.
    _disk: Option<crate::admission::FileReservation>,
    // Release the image/engine owner only after buffered writers and accounting.
    permit: Permit,
}
impl SnapshotReceiver {
    pub fn write(&mut self, part: Part, offset: u64, bytes: &[u8]) -> Result<()> {
        self.receiver.write(part, offset, bytes)
    }
    pub fn finish(self) -> Result<PreparedCheckpoint> {
        let loaded = self.receiver.finish()?;
        self.permit.0.track(&loaded.image)?;
        Ok(PreparedCheckpoint {
            permit: self.permit,
            group: self.packet.group,
            loaded,
            received: Some(self.packet),
        })
    }
}

pub(super) fn read_body(
    store: &GroupLog,
    image: Option<&Image>,
    index: u64,
    meta: &BatchMeta,
) -> Result<bytes::Bytes> {
    if let Some(image) = image {
        if let Some(body) = image.read_for_validation(index)? {
            return Ok(body);
        }
    }
    let mut entries = store.entries(
        index,
        index + 1,
        None,
        raft::GetEntriesContext::empty(false),
    )?;
    let entry = entries.pop().context("missing application image input")?;
    ensure!(entry.index == index, "image input index mismatch");
    ensure!(
        &state::decode_meta(&entry.context)? == meta,
        "accepted metadata mismatch"
    );
    Ok(entry.data)
}

impl Node {
    pub fn prepare_checkpoint(&self, group: u64, id: Uuid) -> Result<CheckpointTask> {
        self.healthy()?;
        ensure!(!id.is_nil(), "nil application image id");
        let g = self.groups.get(&group).context("unhosted group")?;
        let cut = Cut {
            group,
            id,
            index: g.state.applied,
            term: g.raw.store().term(g.state.applied)?,
        };
        ensure!(
            cut.index >= g.raw.store().first_index()?,
            "no new committed state to checkpoint"
        );
        Ok(CheckpointTask {
            permit: self.images.acquire()?,
            store: g.raw.store().clone(),
            previous: g.image.clone(),
            layout: self.layout.clone(),
            limits: self.limits.clone(),
            cut,
            state: g.state.clone(),
        })
    }

    /// Install a completed local build. Inputs after the captured cut remain in
    /// the log. The full image is now available for a lagging follower's transfer;
    /// callers must service MsgSnapshot before enabling periodic compaction.
    /// This logically compacts only; it does not bound retained disk space.
    pub fn install_checkpoint(&mut self, prepared: PreparedCheckpoint) -> Result<()> {
        self.healthy()?;
        ensure!(
            Arc::ptr_eq(&prepared.permit.0, &self.images),
            "foreign node image worker"
        );
        ensure!(
            prepared.received.is_none(),
            "received image needs step_snapshot"
        );
        let g = self
            .groups
            .get_mut(&prepared.group)
            .context("unhosted group")?;
        let snapshot = prepared.loaded.image.snapshot()?;
        ensure!(
            prepared.loaded.state.applied <= g.state.applied,
            "checkpoint exceeds applied state"
        );
        if let Err(error) = g
            .raw
            .store()
            .checkpoint(&snapshot)
            .and_then(|_| self.images.installed(&prepared.loaded.image))
        {
            self.failed = true;
            return Err(error);
        }
        g.image = Some(prepared.loaded.image);
        Ok(())
    }

    /// Capture immediately after poll, before installing another checkpoint.
    /// Keep the returned Arc until the outgoing transfer finishes. A transport
    /// can discard an obsolete message and report failure so Raft retries.
    pub fn snapshot_image(&self, group: u64, snapshot: &Snapshot) -> Result<Arc<Image>> {
        self.healthy()?;
        let image = self
            .groups
            .get(&group)
            .context("unhosted group")?
            .image
            .as_ref()
            .context("no installed application image")?;
        ensure!(
            image.snapshot()? == *snapshot,
            "outgoing snapshot is obsolete"
        );
        Ok(image.clone())
    }

    /// Validate an incoming descriptor and return exact file lengths without
    /// creating files. Disk admission must precede `receive_snapshot`.
    pub fn snapshot_sizes(&self, packet: &Packet) -> Result<[u64; 2]> {
        self.healthy()?;
        self.validate_packet(packet)?;
        ensure!(
            packet.message.get_msg_type() == MessageType::MsgSnapshot
                && packet.message.entries.is_empty()
                && packet.message.has_snapshot(),
            "expected snapshot-only message"
        );
        checkpoint::snapshot_sizes(
            packet.message.get_snapshot(),
            &self.layout,
            &self.limits,
            packet.group,
        )
    }

    pub fn receive_snapshot(&self, packet: Packet, local_id: Uuid) -> Result<SnapshotReceiver> {
        let sizes = self.snapshot_sizes(&packet)?;
        let permit = self.images.acquire()?;
        let disk = self
            .images
            .store
            .disk_budget()
            .map(|budget| budget.reserve_files(&sizes))
            .transpose()?;
        let receiver = checkpoint::Receiver::new(
            &self.images.directory,
            packet.message.get_snapshot(),
            self.layout.clone(),
            self.limits.clone(),
            packet.group,
            local_id,
        )?;
        Ok(SnapshotReceiver {
            permit,
            packet,
            receiver,
            _disk: disk,
        })
    }

    /// Only a complete image can reach RawNode. It may now be stale and ignored;
    /// otherwise poll persists the received snapshot, HardState and new suffix
    /// atomically before releasing the follower response. Keep polling during
    /// download; no Ready is retained across a network or file wait.
    pub fn step_snapshot(&mut self, mut prepared: PreparedCheckpoint) -> Result<()> {
        self.healthy()?;
        ensure!(
            Arc::ptr_eq(&prepared.permit.0, &self.images),
            "foreign node image worker"
        );
        let mut packet = prepared
            .received
            .take()
            .context("local image needs install_checkpoint")?;
        self.validate_packet(&packet)?;
        let snapshot = prepared.loaded.image.snapshot()?;
        packet.message.set_snapshot(snapshot.clone());
        let g = self
            .groups
            .get_mut(&prepared.group)
            .context("unhosted group")?;
        ensure!(
            g.incoming.is_none(),
            "poll the previous received snapshot first"
        );
        g.raw.step(packet.message)?;
        if g.raw.raft.raft_log.unstable.snapshot.as_ref() == Some(&snapshot) {
            g.incoming = Some(prepared);
        }
        Ok(())
    }

    pub fn report_snapshot(&mut self, group: u64, to: u64, delivered: bool) -> Result<()> {
        self.healthy()?;
        ensure!(
            to != self.id
                && self
                    .layout
                    .groups
                    .get(&group)
                    .context("unknown group")?
                    .contains(&to),
            "invalid snapshot destination"
        );
        self.groups
            .get_mut(&group)
            .context("unhosted group")?
            .raw
            .report_snapshot(
                to,
                if delivered {
                    raft::SnapshotStatus::Finish
                } else {
                    raft::SnapshotStatus::Failure
                },
            );
        Ok(())
    }
}

pub(super) fn install_received(
    g: &mut Group,
    ready: &raft::Ready,
    id: u64,
    events: &mut Events,
) -> Result<()> {
    let prepared = g
        .incoming
        .take()
        .context("received Raft snapshot has no prepared application image")?;
    ensure!(
        prepared.loaded.image.snapshot()? == *ready.snapshot(),
        "prepared image differs from Ready"
    );
    g.raw.store().install_snapshot(
        ready.snapshot(),
        ready.entries(),
        ready.hs().context("received snapshot has no HardState")?,
    )?;
    prepared.permit.0.installed(&prepared.loaded.image)?;
    g.state = prepared.loaded.state;
    g.image = Some(prepared.loaded.image);
    for (_, (request, _, _reservation)) in std::mem::take(&mut g.pending) {
        events.completions.push(Completion {
            group: id,
            request,
            outcome: Outcome::Unknown,
        });
    }
    Ok(())
}
