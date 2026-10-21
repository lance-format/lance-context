//! Synchronous, transport-independent Raft ingestion for static RF3 groups.
//!
//! `propose` queues input; only `poll` can return a committed result. All outgoing
//! messages and completions follow durable Ready/LightReady persistence. Exact
//! retries enter consensus again and return the original accepted index after
//! commit. An isolated former leader cannot answer retries from local state.
//!
//! The owner schedules ticks, bounded network I/O and calls on one thread. This
//! component supports explicit application checkpoints and prepared snapshot
//! transfer and explicit collection of unreferenced application images. Raft
//! segment reclamation and production scheduling remain separate responsibilities.
//! Optional shared node disk admission covers proposals and image files while
//! preserving a window for accepted peer traffic; it is not a filesystem quota.
//! Message scope checks are not authentication. Do not expose it to untrusted
//! transport or enable ingestion without the remaining admission/lifecycle work.

pub mod checkpoint;
mod images;
pub use images::{ImageCollection, ImageCollectionProgress};
pub mod producer;
pub mod publication;
pub mod query;
#[cfg(feature = "runtime")]
pub mod runtime;
pub mod source;
mod state;
#[cfg(test)]
mod tests;

use std::{collections::BTreeMap, path::Path, sync::Arc};

use anyhow::{ensure, Context, Result};
use protobuf::Message as _;
use raft::{RawNode, StateRole, Storage};
use raft_proto::eraftpb::{Entry, Message, MessageType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    placement::Layout,
    raft_log::{GroupLog, LogMaintenance, LogOptions, PhysicalLog},
    record::{Batch, BatchMeta},
};
pub use images::{CheckpointTask, PreparedCheckpoint, SnapshotReceiver};
use state::State;

/// Immutable, replicated-state bounds. All peers and reopens must use the same
/// policy. These bounds do not account for queues in the caller's transport.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    /// Public Ed25519 key for catalog dependency attestations. Key changes are
    /// application-policy migrations; private key bytes are never persisted here.
    pub catalog_key: Option<[u8; 32]>,
    /// Optional per-group keys for quorum-confirmed source closure proofs.
    /// A nonempty policy must cover every group with distinct keys, separate
    /// from the catalog key. Installing it is an explicit policy migration.
    #[serde(default)]
    pub source_keys: BTreeMap<u64, [u8; 32]>,
    pub max_tables: u64,
    pub max_active_plans: u64,
    pub max_plan_source_bytes: u64,
    pub max_descriptor_bytes: u64,
    pub max_manifest_records: u64,
    pub max_record_bytes: u64,
    pub max_records: u64,
    pub max_producers: u64,
    pub max_pending: u64,
    pub max_uncommitted_bytes: u64,
    /// Fixed slots per group; zero retains UUID-only semantics and disk identity.
    /// Set only on fresh groups. Changing this policy requires migration.
    #[serde(default)]
    pub producer_slots: u32,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            catalog_key: None,
            source_keys: BTreeMap::new(),
            max_tables: 1024,
            max_active_plans: 128,
            max_plan_source_bytes: 64 * 1024 * 1024,
            max_descriptor_bytes: 256 * 1024,
            max_manifest_records: 4096,
            max_record_bytes: 8 * 1024 * 1024,
            max_records: 100_000,
            max_producers: 16_384,
            max_pending: 128,
            max_uncommitted_bytes: 64 * 1024 * 1024,
            producer_slots: 0,
        }
    }
}
impl Limits {
    fn identity(&self) -> Result<Vec<u8>> {
        ensure!(
            (1..=16_384).contains(&self.max_tables)
                && self.catalog_key.is_none_or(|key| key != [0; 32])
                && self.source_keys.len() <= 64
                && (self.source_keys.is_empty() || self.catalog_key.is_some())
                && self.source_keys.iter().all(|(group, key)| {
                    *group > 0 && *key != [0; 32] && Some(*key) != self.catalog_key
                })
                && self
                    .source_keys
                    .values()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == self.source_keys.len()
                && (1..=256).contains(&self.max_active_plans)
                && (1..=256 * 1024 * 1024).contains(&self.max_plan_source_bytes)
                && (1..=publication::MAX_DESCRIPTOR_BYTES as u64)
                    .contains(&self.max_descriptor_bytes)
                && (1..=16_384).contains(&self.max_manifest_records)
                && (1..=8 * 1024 * 1024).contains(&self.max_record_bytes)
                && (1..=500_000).contains(&self.max_records)
                && (1..=100_000).contains(&self.max_producers)
                && self.producer_slots <= producer::MAX_SLOTS
                && u64::from(self.producer_slots) <= self.max_producers
                && (1..=4096).contains(&self.max_pending)
                && self.max_uncommitted_bytes >= self.max_record_bytes.max(1024 * 1024)
                && self.max_uncommitted_bytes <= 256 * 1024 * 1024,
            "invalid replication limits"
        );
        if self.producer_slots > 0 {
            let mut legacy = self.clone();
            legacy.producer_slots = 0;
            return Ok(bincode::serialize(&(
                "lance-context/replicated-append/v11",
                legacy.identity()?,
                self.producer_slots,
            ))?);
        }
        if self.source_keys.is_empty() {
            // Bincode structs and tuples encode their fields in order. Keep the
            // exact old policy bytes when source attestations are not enabled.
            return Ok(bincode::serialize(&(
                "lance-context/replicated-append/v9",
                self.catalog_key,
                self.max_tables,
                self.max_active_plans,
                self.max_plan_source_bytes,
                self.max_descriptor_bytes,
                self.max_manifest_records,
                self.max_record_bytes,
                self.max_records,
                self.max_producers,
                self.max_pending,
                self.max_uncommitted_bytes,
            ))?);
        }
        Ok(bincode::serialize(&(
            "lance-context/replicated-append/v10",
            (
                self.catalog_key,
                &self.source_keys,
                self.max_tables,
                self.max_active_plans,
                self.max_plan_source_bytes,
                self.max_descriptor_bytes,
                self.max_manifest_records,
                self.max_record_bytes,
                self.max_records,
                self.max_producers,
                self.max_pending,
                self.max_uncommitted_bytes,
            ),
        ))?)
    }
}

/// Transport supplies a bounded protobuf decoder. Scope checks prevent mixing
/// groups/layouts/policies; authenticated peer identity is still the caller's job.
#[derive(Clone, Debug)]
pub struct Packet {
    pub cluster: Uuid,
    pub layout: [u8; 32],
    pub policy: [u8; 32],
    pub group: u64,
    pub message: Message,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Applied {
    Publisher(publication::Reply),
    /// A new append or exact latest retry. Index identifies the original append.
    Accepted {
        index: u64,
    },
    /// Gap, stale sequence or changed content at an already-used sequence.
    Rejected {
        expected_sequence: u64,
    },
    Capacity,
    Source(source::Reply),
    Producer(producer::Reply),
    /// Stale generation, wrong lane binding or a closed lane attempting new data.
    ProducerFenced,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Committed(Applied),
    /// Leadership changed before this owner observed the result. The append may
    /// still commit; retry its exact identity and body, not a new sequence.
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Completion {
    pub group: u64,
    /// Caller correlation within this process; never persisted as a producer ID.
    pub request: u64,
    pub outcome: Outcome,
}

#[derive(Default)]
pub struct Events {
    pub messages: Vec<Packet>,
    pub completions: Vec<Completion>,
    pub reads: Vec<query::Completion>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Status {
    pub leader: u64,
    pub term: u64,
    pub committed: u64,
    pub applied: u64,
    pub accepted_records: usize,
    pub producers: usize,
    pub pending: usize,
    pub pending_reads: usize,
}

/// A bounded local view captured together with its next cursor. Preserve the
/// group/layout scope; this is neither a quorum read nor proof of Lance visibility.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    pub entries: Vec<(u64, BatchMeta)>,
    /// All accepted records after the requested cursor through this position
    /// are included. May advance over rejected, duplicate or blank entries.
    pub through: u64,
}

struct Group {
    raw: RawNode<GroupLog>,
    state: State,
    // Index AND term prevents suffix replacement from completing another input.
    pending: BTreeMap<(u64, u64), (u64, BatchMeta, Option<crate::admission::Reservation>)>,
    reads: query::Tracker,
    image: Option<Arc<checkpoint::Image>>,
    incoming: Option<PreparedCheckpoint>,
}

/// Owns all Raft groups in one shared physical engine. Do not call concurrently
/// or retain raw group storage handles outside it. An internal/disk error fences
/// this node; drop it and any maintenance handles, reconcile outstanding requests
/// and reopen the original directories.
pub struct Node {
    signer: Option<Arc<source::CatalogSigner>>,
    source_signers: BTreeMap<u64, Arc<source::SourceSigner>>,
    layout: Arc<Layout>,
    layout_hash: [u8; 32],
    policy: [u8; 32],
    limits: Limits,
    id: u64,
    groups: BTreeMap<u64, Group>,
    images: Arc<images::Owner>,
    failed: bool,
}

impl Node {
    /// Configure one shared physical-directory budget before splitting group
    /// owners or starting transports. Reopens reconstruct installed/orphan/log
    /// usage from the same directory. This does not alter replicated identity.
    pub fn with_disk_budget(self, options: crate::raft_log::DiskOptions) -> Result<Self> {
        options.validate()?;
        ensure!(
            self.groups.len()
                == self
                    .layout
                    .groups
                    .values()
                    .filter(|v| v.contains(&self.id))
                    .count(),
            "configure disk admission before splitting groups"
        );
        let allocation = self.images.store().allocation_headroom()?;
        let reserve = self
            .limits
            .max_uncommitted_bytes
            .checked_mul(self.groups.len() as u64)
            .and_then(|n| n.checked_add(allocation))
            .context("replication headroom overflow")?;
        ensure!(
            options.headroom_bytes as u64 >= reserve,
            "headroom must cover hosted uncommitted windows and two log files"
        );
        self.images.configure_disk(options)?;
        Ok(self)
    }

    pub fn disk_budget(&self) -> Option<crate::admission::Snapshot> {
        self.images.store().disk_budget().map(|b| b.snapshot())
    }

    pub fn prepare_disk_measurement(&self) -> Result<crate::raft_log::DiskMeasurement> {
        self.healthy()?;
        self.images.store().disk_measurement()
    }
    /// Give each group to an independent worker while retaining the shared
    /// physical engine and its sync/failure fence. Consumes the aggregate owner;
    /// it never clones a RawNode or permits two owners of one group.
    pub fn into_groups(self) -> Vec<Self> {
        self.groups
            .into_iter()
            .map(|(id, group)| Self {
                signer: if id == publication::control_group(&self.layout) {
                    self.signer.clone()
                } else {
                    None
                },
                source_signers: self
                    .source_signers
                    .get(&id)
                    .map_or_else(BTreeMap::new, |s| BTreeMap::from([(id, s.clone())])),
                layout: self.layout.clone(),
                layout_hash: self.layout_hash,
                policy: self.policy,
                limits: self.limits.clone(),
                id: self.id,
                groups: BTreeMap::from([(id, group)]),
                images: self.images.clone(),
                failed: self.failed,
            })
            .collect()
    }

    pub fn open(directory: &Path, layout: Arc<Layout>, id: u64, limits: Limits) -> Result<Self> {
        Self::open_with_storage_options(directory, layout, id, limits, LogOptions::default())
    }

    /// Open with local log rotation and maintenance thresholds. These settings
    /// do not change the replicated layout, application policy or disk quota.
    pub fn open_with_storage_options(
        directory: &Path,
        layout: Arc<Layout>,
        id: u64,
        limits: Limits,
        storage: LogOptions,
    ) -> Result<Self> {
        let layout_hash = layout.fingerprint()?;
        let identity = limits.identity()?;
        ensure!(
            limits.source_keys.is_empty() || limits.source_keys.keys().eq(layout.groups.keys()),
            "source signing policy must cover exactly the configured groups"
        );
        let policy = Sha256::digest(&identity).into();
        let log = PhysicalLog::open_with_options(directory, &layout, id, storage)?;
        let hosted: std::collections::BTreeSet<_> = layout
            .groups
            .iter()
            .filter(|(_, voters)| voters.contains(&id))
            .map(|(group, _)| *group)
            .collect();
        let image_store = log.group(*hosted.iter().next().context("node has no hosted group")?)?;
        let images = images::Owner::new(directory, hosted, image_store)?;
        let mut groups = BTreeMap::new();
        let logger = slog::Logger::root(slog::Discard, slog::o!());
        for (group, voters) in &layout.groups {
            if !voters.contains(&id) {
                continue;
            }
            let store = log.group(*group)?;
            store.bind_application(&identity)?;
            let committed = store.initial_state()?.hard_state.commit;
            let (mut state, image) = if store.first_index()? > 1 {
                let snapshot = store.snapshot(store.first_index()? - 1, id)?;
                let loaded = checkpoint::load(directory, &snapshot, &layout, &limits, *group)?;
                images.installed(&loaded.image)?;
                (loaded.state, Some(loaded.image))
            } else {
                (State::with_producer_slots(limits.producer_slots), None)
            };
            while state.applied < committed {
                let entries = store.entries(
                    state.applied + 1,
                    (state.applied + 257).min(committed + 1),
                    Some(1024 * 1024),
                    raft::GetEntriesContext::empty(false),
                )?;
                ensure!(!entries.is_empty(), "missing committed recovery entries");
                for entry in entries {
                    state::validate_entry(&layout, *group, &limits, &entry, true)?;
                    state.apply(&entry, &limits)?;
                }
            }
            let config = raft::Config {
                id,
                applied: committed,
                election_tick: 10,
                heartbeat_tick: 2,
                check_quorum: true,
                pre_vote: true,
                read_only_option: raft::ReadOnlyOption::Safe,
                max_size_per_msg: 1024 * 1024,
                max_inflight_msgs: 16,
                max_uncommitted_size: limits.max_uncommitted_bytes,
                max_committed_size_per_ready: 1024 * 1024,
                ..Default::default()
            };
            let raw = RawNode::new(&config, store, &logger)?;
            groups.insert(
                *group,
                Group {
                    raw,
                    state,
                    pending: BTreeMap::new(),
                    reads: query::Tracker::default(),
                    image,
                    incoming: None,
                },
            );
        }
        Ok(Self {
            signer: None,
            source_signers: BTreeMap::new(),
            layout,
            layout_hash,
            policy,
            limits,
            id,
            groups,
            images,
            failed: false,
        })
    }

    fn healthy(&self) -> Result<()> {
        ensure!(!self.failed, "replication node fenced; drop and reopen it");
        // Another group shares this engine's uncertain-write fence.
        for group in self.groups.values() {
            group.raw.store().initial_state()?;
        }
        Ok(())
    }

    /// Prepare one physical-engine maintenance handle, shared by all groups on
    /// this node. Run it on a bounded blocking worker. A native pass excludes
    /// payload reads and persistence and can delay polling until it completes.
    pub fn prepare_log_maintenance(&self) -> Result<LogMaintenance> {
        self.healthy()?;
        Ok(self
            .groups
            .first_key_value()
            .context("node has no hosted groups")?
            .1
            .raw
            .store()
            .maintenance())
    }

    /// Install the catalog's private dependency-attestation key. The persisted
    /// policy pins its public key. Source-only replicas need only the public key.
    pub fn set_catalog_signer(&mut self, signer: Arc<source::CatalogSigner>) -> Result<()> {
        self.healthy()?;
        ensure!(
            self.groups
                .contains_key(&publication::control_group(&self.layout))
                && self.limits.catalog_key == Some(signer.public_key()),
            "catalog signer does not match hosted scope and policy"
        );
        self.signer = Some(signer);
        Ok(())
    }

    /// Provision this source group's key without exposing the catalog's private
    /// key to source-only replicas. The persisted policy pins the public key.
    pub fn set_source_signer(
        &mut self,
        group: u64,
        signer: Arc<source::SourceSigner>,
    ) -> Result<()> {
        self.healthy()?;
        ensure!(
            self.groups.contains_key(&group)
                && self.limits.source_keys.get(&group) == Some(&signer.public_key()),
            "source signer does not match hosted scope and policy"
        );
        self.source_signers.insert(group, signer);
        Ok(())
    }

    pub fn tick(&mut self) -> Result<()> {
        self.healthy()?;
        for group in self.groups.values_mut() {
            group.raw.tick();
        }
        Ok(())
    }

    pub fn campaign(&mut self, group: u64) -> Result<()> {
        self.healthy()?;
        self.groups
            .get_mut(&group)
            .context("unhosted group")?
            .raw
            .campaign()?;
        Ok(())
    }

    pub fn status(&self, group: u64) -> Result<Status> {
        self.healthy()?;
        let g = self.groups.get(&group).context("unhosted group")?;
        Ok(Status {
            leader: g.raw.raft.leader_id,
            term: g.raw.raft.term,
            committed: g.raw.store().initial_state()?.hard_state.commit,
            applied: g.state.applied,
            accepted_records: g.state.accepted.len(),
            producers: g.state.producers.len(),
            pending: g.pending.len(),
            pending_reads: g.reads.len(),
        })
    }

    fn request_pending(&self, request: u64) -> bool {
        self.groups
            .values()
            .any(|g| g.pending.values().any(|p| p.0 == request) || g.reads.contains(request))
    }

    /// Queue a bounded safe read-index request at this group's leader. Only a
    /// successful poll can return the captured application view. This neither
    /// forwards follower reads nor treats the leader's local state as a lease.
    /// Until an entry in the leader's current term commits, admission fails
    /// without reserving capacity or the token; retry after driving replication.
    pub fn query(&mut self, group: u64, request: u64, query: query::Query) -> Result<()> {
        self.healthy()?;
        ensure!(
            !self.request_pending(request),
            "request token already pending"
        );
        query.validate(&self.layout, group)?;
        if matches!(
            query,
            query::Query::Pin { .. } | query::Query::ClosePermit { .. }
        ) {
            ensure!(
                self.signer.is_some(),
                "catalog dependency signer unavailable"
            );
        }
        if matches!(query, query::Query::ClosedProof { .. }) {
            ensure!(
                self.source_signers.contains_key(&group),
                "source signer unavailable"
            );
        }
        let g = self.groups.get_mut(&group).context("unhosted group")?;
        ensure!(g.raw.raft.state == StateRole::Leader, "not leader");
        // raft-rs 0.7 silently drops ReadIndex before this condition holds.
        // Reject before tracking so a new leader cannot strand accepted reads.
        ensure!(
            g.raw.raft.commit_to_current_term(),
            "leader has not committed a current-term entry; retry after replication"
        );
        ensure!(
            g.pending.len() + g.reads.len() < self.limits.max_pending as usize,
            "pending request capacity"
        );
        let context = g.reads.enqueue(request, g.raw.raft.term, query)?;
        g.raw.read_index(context);
        Ok(())
    }

    /// Queue a binding/epoch CAS in the one catalog group. Only a committed
    /// Publisher completion from poll grants authority; no local cache shortcut.
    pub fn acquire_publisher(&mut self, request: u64, acquire: publication::Acquire) -> Result<()> {
        let batch = acquire.batch(&self.layout)?;
        self.propose(publication::control_group(&self.layout), request, batch)
    }

    /// Persist an immutable publication plan through the catalog quorum. The
    /// plan is not a source pin or permission to commit an external manifest.
    pub fn plan_publication(&mut self, request: u64, plan: publication::PlanRequest) -> Result<()> {
        let batch = plan.batch(&self.layout, &self.limits)?;
        self.propose(publication::control_group(&self.layout), request, batch)
    }

    /// Persist one immutable descriptor set for an exact pending plan. The
    /// caller must separately validate source pins and coordinate Lance commits.
    pub fn prepare_publication(
        &mut self,
        request: u64,
        prepared: publication::PrepareRequest,
    ) -> Result<()> {
        let batch = prepared.batch(&self.layout, &self.limits)?;
        self.propose(publication::control_group(&self.layout), request, batch)
    }

    /// Adopt an immutable starting manifest after the caller has verified it
    /// and excluded unmanaged writers/cleaners for this dataset incarnation.
    pub fn bootstrap_manifest(
        &mut self,
        request: u64,
        bootstrap: publication::BootstrapManifest,
    ) -> Result<()> {
        let batch = bootstrap.batch(&self.layout)?;
        self.propose(publication::control_group(&self.layout), request, batch)
    }

    /// Select the unique manifest path for the exact immutable prepared plan.
    /// An enqueue is not publication; wait for durable quorum completion.
    pub fn reserve_manifest(
        &mut self,
        request: u64,
        reservation: publication::ReserveManifest,
    ) -> Result<()> {
        let batch = reservation.batch(&self.layout)?;
        self.propose(publication::control_group(&self.layout), request, batch)
    }

    /// Retire old catalog directory records while preserving the bootstrap,
    /// current head and active publication dependencies. Wait for a committed
    /// result. This is not permission to delete immutable Lance objects.
    pub fn retire_manifest_history(
        &mut self,
        request: u64,
        retirement: publication::RetireManifestHistory,
    ) -> Result<()> {
        let batch = retirement.batch(&self.layout)?;
        self.propose(publication::control_group(&self.layout), request, batch)
    }

    /// Durably resolve the selected manifest against its exact pending plan.
    /// This retains all source dependencies; it does not authorize reclamation.
    pub fn confirm_publication(
        &mut self,
        request: u64,
        confirmation: publication::ConfirmPublication,
    ) -> Result<()> {
        let batch = confirmation.batch(&self.layout)?;
        self.propose(publication::control_group(&self.layout), request, batch)
    }

    /// Advance independent source cursors only after every selected source
    /// proves the exact durable closure. Completion permits the next plan;
    /// physical file reclamation still requires a durable application image.
    pub fn advance_publication(
        &mut self,
        request: u64,
        advancement: publication::AdvancePublication,
    ) -> Result<()> {
        let batch = advancement.batch(&self.layout, &self.limits)?;
        self.propose(publication::control_group(&self.layout), request, batch)
    }

    /// Verify a catalog quorum-read attestation and persist the exact complete
    /// table prefix in its source group. Completion, never enqueue, pins it.
    pub fn pin_source(&mut self, request: u64, permit: source::Permit) -> Result<()> {
        let group = permit.selection().group;
        let batch = permit.batch(&self.layout, &self.limits)?;
        self.propose(group, request, batch)
    }

    /// Persist a catalog-authorized source closure and reject delayed pin
    /// requests for the closed plan. Retire selected payload metadata while
    /// preserving latest producer receipts; physical collection is separate.
    pub fn close_source(&mut self, request: u64, permit: source::ClosurePermit) -> Result<()> {
        let group = permit.closure().group;
        let batch = permit.batch(&self.layout, &self.limits)?;
        self.propose(group, request, batch)
    }

    /// Local applied view, not a quorum read or proof that an owner is current.
    pub fn publisher(&self, table: Uuid) -> Result<Option<&publication::Table>> {
        self.healthy()?;
        Ok(self
            .groups
            .get(&publication::control_group(&self.layout))
            .context("catalog group not hosted")?
            .state
            .catalog
            .tables
            .get(&table))
    }

    /// Replicate a trusted producer lifecycle command. Completion is released
    /// only after durable quorum commit; an ambiguous open must retry its exact CAS.
    pub fn producer_command(&mut self, request: u64, command: producer::Command) -> Result<()> {
        let group = command.group;
        self.propose(group, request, command.batch(&self.layout, &self.limits)?)
    }

    /// Ok means queued locally, NOT durable or committed. A token may have only
    /// one unresolved request on this node. No retry result bypasses consensus.
    pub fn propose(&mut self, group: u64, request: u64, batch: Batch) -> Result<()> {
        self.healthy()?;
        ensure!(
            !self.request_pending(request),
            "request token already pending"
        );
        state::validate_meta(
            &self.layout,
            group,
            &self.limits,
            &batch.meta,
            &batch.body,
            true,
        )?;
        let g = self.groups.get_mut(&group).context("unhosted group")?;
        ensure!(g.raw.raft.state == StateRole::Leader, "not leader");
        ensure!(
            g.pending.len() + g.reads.len() < self.limits.max_pending as usize,
            "pending proposal capacity"
        );
        let index = g.raw.raft.raft_log.last_index() + 1;
        let term = g.raw.raft.term;
        let reservation = g
            .raw
            .store()
            .disk_budget()
            .map(|budget| {
                let bytes = batch
                    .body
                    .len()
                    .checked_add(4096)
                    .context("proposal disk reservation overflow")?;
                budget.reserve(bytes, batch.meta.kind == crate::record::RecordKind::Data)
            })
            .transpose()?;
        // RawNode::propose accepts Vec and would copy the opaque Bytes. This is
        // the equivalent raft-rs proposal path with shared entry.data ownership.
        let entry = Entry {
            context: state::encode_meta(&batch.meta)?.into(),
            data: batch.body,
            ..Default::default()
        };
        let mut message = Message::default();
        message.set_msg_type(MessageType::MsgPropose);
        message.from = self.id;
        message.set_entries(vec![entry].into());
        g.raw.raft.step(message)?;
        ensure!(
            g.raw.raft.raft_log.last_index() == index,
            "proposal was not queued"
        );
        g.pending
            .insert((index, term), (request, batch.meta, reservation));
        Ok(())
    }

    fn validate_packet(&self, packet: &Packet) -> Result<()> {
        ensure!(
            packet.cluster == self.layout.cluster
                && packet.layout == self.layout_hash
                && packet.policy == self.policy,
            "foreign replication scope or policy"
        );
        let voters = self
            .layout
            .groups
            .get(&packet.group)
            .context("unknown group")?;
        let message = &packet.message;
        ensure!(
            message.to == self.id && message.from != self.id && voters.contains(&message.from),
            "invalid Raft peer route"
        );
        ensure!(self.groups.contains_key(&packet.group), "unhosted group");
        ensure!(
            message.compute_size() <= 16 * 1024 * 1024,
            "oversized message"
        );
        Ok(())
    }

    pub fn step(&mut self, packet: Packet) -> Result<()> {
        self.healthy()?;
        self.validate_packet(&packet)?;
        let message = packet.message;
        ensure!(
            matches!(
                message.get_msg_type(),
                MessageType::MsgAppend
                    | MessageType::MsgAppendResponse
                    | MessageType::MsgHeartbeat
                    | MessageType::MsgHeartbeatResponse
                    | MessageType::MsgRequestVote
                    | MessageType::MsgRequestVoteResponse
                    | MessageType::MsgRequestPreVote
                    | MessageType::MsgRequestPreVoteResponse
            ),
            "unsupported peer message"
        );
        ensure!(
            message.snapshot.is_none(),
            "snapshot requires receive_snapshot and step_snapshot"
        );
        for entry in &message.entries {
            state::validate_entry(&self.layout, packet.group, &self.limits, entry, true)?;
        }
        self.groups
            .get_mut(&packet.group)
            .context("unhosted group")?
            .raw
            .step(message)?;
        Ok(())
    }

    /// Performs blocking storage I/O. The transport may drop output messages;
    /// Raft will retransmit. On error NONE of this call's events are released.
    pub fn poll(&mut self) -> Result<Events> {
        self.healthy()?;
        let result = self.drive();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn drive(&mut self) -> Result<Events> {
        let mut events = Events::default();
        for (id, group) in &mut self.groups {
            let mut outbound = Vec::new();
            while group.raw.has_ready() {
                let mut ready = group.raw.ready();
                if ready.snapshot().is_empty() {
                    group.raw.store().persist(ready.entries(), ready.hs())?;
                } else {
                    images::install_received(group, &ready, *id, &mut events)?;
                }
                outbound.extend(ready.take_messages());
                outbound.extend(ready.take_persisted_messages());
                for read in ready.take_read_states() {
                    group.reads.confirm(&read.request_ctx, read.index)?;
                }
                apply(
                    group,
                    *id,
                    ready.take_committed_entries(),
                    &self.limits,
                    &mut events,
                )?;
                let mut light = group.raw.advance(ready);
                if light.commit_index().is_some() {
                    group
                        .raw
                        .store()
                        .persist(&[], Some(&group.raw.raft.hard_state()))?;
                }
                outbound.extend(light.take_messages());
                apply(
                    group,
                    *id,
                    light.take_committed_entries(),
                    &self.limits,
                    &mut events,
                )?;
                group.raw.advance_apply();
            }
            // Old-term pending entries may still commit after leader loss. They
            // must not be reported as a definite rejection or leak forever.
            let uncertain: Vec<_> = group
                .pending
                .keys()
                .copied()
                .filter(|(_, term)| {
                    group.raw.raft.state != StateRole::Leader || *term != group.raw.raft.term
                })
                .collect();
            for key in uncertain {
                let (request, _, _reservation) = group.pending.remove(&key).unwrap();
                events.completions.push(Completion {
                    group: *id,
                    request,
                    outcome: Outcome::Unknown,
                });
            }
            events.reads.extend(group.reads.finish(
                query::Cut {
                    cluster: self.layout.cluster,
                    layout: self.layout_hash,
                    group: *id,
                    term: group.raw.raft.term,
                    quorum_index: 0,
                    applied: group.state.applied,
                },
                &group.state,
                group.raw.raft.state == StateRole::Leader,
                |query, cut, state| {
                    query.evaluate(
                        state,
                        cut,
                        &self.layout,
                        &self.limits,
                        source::QuerySigners {
                            catalog: self.signer.as_deref(),
                            source: self.source_signers.get(id).map(Arc::as_ref),
                        },
                    )
                },
            )?);
            events
                .messages
                .extend(outbound.into_iter().map(|message| Packet {
                    cluster: self.layout.cluster,
                    layout: self.layout_hash,
                    policy: self.policy,
                    group: *id,
                    message,
                }));
        }
        self.healthy()?;
        Ok(events)
    }

    /// Local accepted rows at this durable applied cut. This is not a quorum
    /// read or a producer ACK and cannot be used to shortcut retry completion.
    pub fn accepted(&self, group: u64, after: u64, limit: usize) -> Result<Page> {
        self.healthy()?;
        let state = &self.groups.get(&group).context("unhosted group")?.state;
        state.page(after, limit)
    }

    pub fn read_accepted(&self, group: u64, index: u64) -> Result<Option<Batch>> {
        self.healthy()?;
        let group = self.groups.get(&group).context("unhosted group")?;
        let Some(meta) = group.state.accepted.get(&index) else {
            return Ok(None);
        };
        let body = images::read_body(group.raw.store(), group.image.as_deref(), index, meta)?;
        meta.validate_payload(&body, self.limits.max_record_bytes as usize)?;
        Ok(Some(Batch {
            meta: meta.clone(),
            body,
        }))
    }
}

fn apply(
    group: &mut Group,
    id: u64,
    entries: Vec<Entry>,
    limits: &Limits,
    events: &mut Events,
) -> Result<()> {
    let durable = group.raw.store().initial_state()?.hard_state.commit;
    for entry in entries {
        ensure!(
            entry.index <= durable,
            "application precedes durable commit"
        );
        let result = group.state.apply(&entry, limits)?;
        if let Some((request, meta, _reservation)) =
            group.pending.remove(&(entry.index, entry.term))
        {
            ensure!(
                state::decode_meta(&entry.context)? == meta,
                "pending proposal identity changed"
            );
            events.completions.push(Completion {
                group: id,
                request,
                outcome: Outcome::Committed(result.context("pending proposal applied as blank")?),
            });
        }
    }
    Ok(())
}
