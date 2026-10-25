//! Recoverable publication of session-sharded rollout records into real Lance.
//! Scheduling ownership is supplied by the application. Durable catalog epochs,
//! fixed transaction targets and signed source closure enforce correctness.
use super::*;
use futures::{stream, StreamExt, TryStreamExt};
use lance::{dataset::builder::DatasetBuilder, Dataset};
use lance_context_wal::replication::{publication as p, source};
use object_store::path::Path as ObjectPath;
use std::collections::{BTreeMap, HashMap};

mod compaction;
pub mod observation;
mod pipeline;
mod read_policy;
mod session_index;
pub(crate) mod stage;
pub use compaction::SmallFileCompaction;
use observation::{Counters as PhaseCounters, Phase};
pub use read_policy::SourceReadPolicy;
pub(crate) use session_index::scan_fragments as session_scan_fragments;
pub use session_index::SessionIndex;

/// Resolve this from the authorized namespace, never from an append request.
#[derive(Clone)]
pub struct Storage {
    pub uri: String,
    pub base: ObjectPath,
    pub options: HashMap<String, String>,
    pub session: Arc<lance::session::Session>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub max_source_bytes: usize,
    /// Accumulate this many source bytes before publication. Zero flushes on
    /// every poll, preserving the existing behavior.
    pub min_source_bytes: usize,
    /// Flush a smaller tail after this delay, on the next scheduled poll.
    pub max_batch_wait: Duration,
    pub max_records: usize,
    pub fetch_parallelism: usize,
    pub write_parallelism: usize,
    /// Initial voter for immutable source reads; all other voters remain fallbacks.
    /// Experimental spreading changes read placement, not durable source identity.
    pub source_read_policy: SourceReadPolicy,
    /// Feed bounded decoded streams to file writers before all sources arrive.
    /// Experimental; disabled until a same-workload comparison supports it.
    pub pipeline_preparation: bool,
    /// Each poll bounds scanning of other tables sharing the same source group.
    pub scan_pages_per_group: usize,
    /// Catalog metadata only; retiring it never deletes immutable blob objects.
    pub retained_manifest_versions: usize,
    /// Compact small published fragments between appends or after the WAL scan catches up.
    /// All table publishers must support compaction before enabling this.
    pub compaction: Option<SmallFileCompaction>,
    /// Build bounded native rollout_id index segments between append publications.
    /// All catalog voters and table publishers must support index maintenance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_index: Option<SessionIndex>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_source_bytes: 64 << 20,
            min_source_bytes: 0,
            max_batch_wait: Duration::from_secs(1),
            max_records: 128,
            fetch_parallelism: 4,
            write_parallelism: 2,
            source_read_policy: SourceReadPolicy::FirstVoter,
            pipeline_preparation: false,
            scan_pages_per_group: 4,
            retained_manifest_versions: 64,
            compaction: None,
            session_index: None,
        }
    }
}
impl Options {
    pub(crate) fn workspace(
        &self,
        contract: &IngestContract,
        bytes: usize,
        records: usize,
    ) -> Result<usize> {
        bytes
            .checked_mul(3)
            .and_then(|v| {
                records
                    .checked_mul(contract.max_rows as usize)
                    .and_then(|n| n.checked_mul(64))
                    .and_then(|n| v.checked_add(n))
            })
            .context("materializer workspace overflow")
    }
    pub(super) fn validate(&self, ingress: &Ingress, budget: &MergeMemoryBudget) -> Result<()> {
        if let Some(index) = &self.session_index {
            index.validate(budget.limit())?;
        }
        if let Some(compaction) = &self.compaction {
            compaction.validate(
                ingress.scope.contract.max_record_bytes as usize,
                budget.limit(),
            )?;
        }
        ensure!(
            self.max_source_bytes >= ingress.scope.contract.max_record_bytes as usize
                && self.max_source_bytes as u64 <= ingress.client.max_plan_source_bytes(),
            "materializer source cap differs from WAL/ingress policy"
        );
        ensure!(
            self.min_source_bytes <= self.max_source_bytes
                && self.max_batch_wait <= Duration::from_secs(60),
            "invalid materializer batch thresholds"
        );
        ensure!(
            (1..=p::MAX_INPUTS).contains(&self.max_records)
                && (1..=8).contains(&self.fetch_parallelism)
                && (1..=8).contains(&self.write_parallelism)
                && (1..=32).contains(&self.scan_pages_per_group)
                && (2..=p::MAX_MANIFEST_VERSIONS - 2).contains(&self.retained_manifest_versions),
            "invalid materializer concurrency or history bounds"
        );
        ensure!(
            self.workspace(
                &ingress.scope.contract,
                self.max_source_bytes,
                self.max_records
            )? <= budget.limit(),
            "materializer maximum workspace exceeds shared budget"
        );
        Ok(())
    }
}

/// Persist this acquisition identity before its first RPC, or derive it from
/// an existing durable scheduling claim. A timeout does not justify a takeover.
#[derive(Clone, Debug)]
pub struct Acquisition {
    pub owner: Uuid,
    pub request: Uuid,
    pub expected_epoch: u64,
}

/// Volatile hints from quorum-observed pages containing no rows for this table.
/// Keep them across scheduling claims, without advancing durable checkpoints.
#[cfg(feature = "worker")]
pub(crate) struct ScanProgress {
    scope: Arc<Scope>,
    empty_cursors: BTreeMap<u64, u64>,
    batch_started: Option<tokio::time::Instant>,
}

pub struct Publisher {
    ingress: Ingress,
    storage: Storage,
    options: Options,
    budget: Arc<MergeMemoryBudget>,
    authority: p::Authority,
    // Only skip pages proven to contain no records for this table. This hint
    // never advances a durable source checkpoint or skips a selected record.
    empty_cursors: BTreeMap<u64, u64>,
    batch_started: Option<tokio::time::Instant>,
    plan_attempt: Option<p::Plan>,
    source_scan_idle: bool,
    prepared_attempt: Option<p::PrepareRequest>,
    phases: Arc<PhaseCounters>,
    #[cfg(feature = "service")]
    preparers: Option<Arc<crate::preparation::Pool>>,
}

impl Publisher {
    pub async fn acquire(
        ingress: Ingress,
        storage: Storage,
        options: Options,
        budget: Arc<MergeMemoryBudget>,
        acquisition: Acquisition,
    ) -> Result<Self> {
        options.validate(&ingress, &budget)?;
        ensure!(
            !storage.uri.is_empty() && !acquisition.owner.is_nil() && !acquisition.request.is_nil(),
            "invalid publisher configuration"
        );
        let applied = commit(
            &ingress.client,
            Command::Acquire(p::Acquire {
                table: ingress.scope.table,
                destination: ingress.scope.destination.clone(),
                ingest: ingress.scope.contract.clone(),
                owner: acquisition.owner,
                request: acquisition.request,
                expected_epoch: acquisition.expected_epoch,
            }),
        )
        .await?;
        let Applied::Publisher(p::Reply::Acquired(authority)) = applied else {
            bail!("publisher acquisition rejected: {applied:?}");
        };
        ensure!(
            authority.cluster == ingress.scope.cluster
                && authority.layout == ingress.scope.layout
                && authority.group == ingress.client.catalog_group()
                && authority.table == ingress.scope.table
                && authority.owner == acquisition.owner
                && authority.epoch
                    == acquisition
                        .expected_epoch
                        .checked_add(1)
                        .context("publisher epoch exhausted")?,
            "foreign publisher authority"
        );
        let value = Self {
            ingress,
            storage,
            options,
            budget,
            authority,
            empty_cursors: BTreeMap::new(),
            batch_started: None,
            plan_attempt: None,
            source_scan_idle: false,
            prepared_attempt: None,
            phases: Arc::default(),
            #[cfg(feature = "service")]
            preparers: None,
        };
        value.state().await?;
        Ok(value)
    }
    pub fn phase_observation(&self) -> [observation::Observation; 10] {
        self.phases.observation()
    }
    #[cfg(feature = "worker")]
    pub(crate) fn waiting_for_batch(&self) -> bool {
        self.batch_started.is_some()
    }
    #[cfg(feature = "worker")]
    pub(crate) fn observe_with(&mut self, phases: Arc<PhaseCounters>) {
        self.phases = phases;
    }
    #[cfg(feature = "worker")]
    pub(crate) fn resume_scanning(&mut self, progress: ScanProgress) -> Result<()> {
        ensure!(
            progress.scope == self.ingress.scope,
            "scan progress belongs to another table binding"
        );
        self.empty_cursors = progress.empty_cursors;
        self.batch_started = progress.batch_started;
        Ok(())
    }
    #[cfg(feature = "worker")]
    pub(crate) fn take_scan_progress(&mut self) -> ScanProgress {
        ScanProgress {
            scope: self.ingress.scope.clone(),
            empty_cursors: std::mem::take(&mut self.empty_cursors),
            batch_started: self.batch_started.take(),
        }
    }
    pub fn authority(&self) -> &p::Authority {
        &self.authority
    }
    /// File preparation may run on separate processes. This publisher retains
    /// sole authority to record the complete result and commit the manifest.
    #[cfg(feature = "service")]
    pub fn use_preparers(&mut self, pool: Arc<crate::preparation::Pool>) {
        self.preparers = Some(pool);
    }
    pub async fn state(&self) -> Result<Box<p::Table>> {
        self.phases
            .clone()
            .measure(Phase::State, self.state_inner())
            .await
    }
    async fn state_inner(&self) -> Result<Box<p::Table>> {
        let query::Value::Publisher(Some(view)) = query(
            &self.ingress,
            self.authority.group,
            query::Query::Publisher {
                table: self.authority.table,
            },
        )
        .await?
        else {
            bail!("publisher binding unavailable");
        };
        ensure!(
            view.authority == self.authority
                && view.destination == self.ingress.scope.destination
                && view.ingest == self.ingress.scope.contract
                && view
                    .manifests
                    .as_ref()
                    .is_some_and(|m| m.incarnation == self.ingress.scope.incarnation),
            "publisher fenced or table incarnation changed"
        );
        Ok(view)
    }
    /// Readers of this managed table must use its catalog-selected manifest.
    pub async fn read_dataset(&self, version: Option<u64>) -> Result<Dataset> {
        read_dataset(&self.ingress, &self.storage, version).await
    }

    /// Select a complete bounded prefix, pin every source, write immutable Lance
    /// files and record descriptors. Existing prepared files are reused.
    pub async fn prepare_next(&mut self) -> Result<Option<Arc<p::Prepared>>> {
        self.phases
            .clone()
            .measure(Phase::Prepare, self.prepare_next_inner())
            .await
    }
    async fn prepare_next_inner(&mut self) -> Result<Option<Arc<p::Prepared>>> {
        let mut view = self.state().await?;
        if view.progress.pending.is_none() {
            self.retire_history(&view).await?;
            if self.plan_attempt.is_none() {
                // The durable completion receipt survives claim rotation and
                // takeover. Allow one bounded rewrite after an append, then
                // give WAL publication another turn even if small files remain.
                if view
                    .progress
                    .advanced
                    .as_ref()
                    .is_some_and(|completed| !completed.sources.is_empty())
                {
                    self.plan_attempt = self.maintenance_plan(&view).await?;
                }
                if self.plan_attempt.is_none() {
                    self.plan_attempt = self.select(&view).await?;
                }
                if self.plan_attempt.is_none() && self.source_scan_idle {
                    self.plan_attempt = self.maintenance_plan(&view).await?;
                }
            }
            let Some(plan) = self.plan_attempt.clone() else {
                return Ok(None);
            };
            let reply = commit(
                &self.ingress.client,
                Command::Plan(p::PlanRequest {
                    authority: self.authority.clone(),
                    plan,
                }),
            )
            .await?;
            ensure!(
                matches!(reply, Applied::Publisher(p::Reply::Planned { .. })),
                "publication selection rejected: {reply:?}"
            );
            view = self.state().await?;
        }
        self.plan_attempt = None;
        if let Some(prepared) = view.progress.prepared {
            self.prepared_attempt = None;
            return Ok(Some(prepared));
        }
        let pending = view.progress.pending.context("publication plan missing")?;
        ensure!(
            pending.plan.source_bytes()? <= self.options.max_source_bytes as u64
                && pending
                    .plan
                    .sources
                    .values()
                    .map(|s| s.records.len())
                    .sum::<usize>()
                    <= self.options.max_records,
            "recovered plan exceeds this worker's configured limits"
        );
        if self
            .prepared_attempt
            .as_ref()
            .is_some_and(|p| p.plan_index != pending.index)
        {
            self.prepared_attempt = None;
        }
        if self.prepared_attempt.is_none() {
            self.pin(&pending).await?;
            let descriptors = self.materialize(&pending).await?;
            descriptors.validate(self.ingress.client.max_descriptor_bytes())?;
            self.prepared_attempt = Some(p::PrepareRequest {
                authority: self.authority.clone(),
                transaction: pending.plan.transaction,
                plan_index: pending.index,
                descriptors,
            });
        }
        let reply = commit(
            &self.ingress.client,
            Command::Prepare(self.prepared_attempt.as_ref().unwrap().clone()),
        )
        .await?;
        ensure!(
            matches!(reply, Applied::Publisher(p::Reply::Prepared { .. })),
            "materialization result rejected: {reply:?}"
        );
        Ok(self.state().await?.progress.prepared)
    }
    async fn maintenance_plan(&self, view: &p::Table) -> Result<Option<p::Plan>> {
        if Self::compaction_first(view.progress.minimum_read_version) {
            if let Some(plan) = compaction::plan(self, view).await? {
                return Ok(Some(plan));
            }
        }
        if let Some(plan) = session_index::plan(self, view).await? {
            return Ok(Some(plan));
        }
        compaction::plan(self, view).await
    }

    fn compaction_first(read_version: u64) -> bool {
        // During backlog, each append/maintenance pair advances two versions.
        // Rotate from durable state so takeover cannot reset the preference.
        (read_version / 2) % 2 == 1
    }

    async fn materialize(&self, pending: &Arc<p::PendingPlan>) -> Result<p::Descriptors> {
        if pending.plan.is_maintenance() {
            if Self::compaction_first(pending.plan.read_version)
                && compaction::has_work(self, pending.plan.read_version).await?
            {
                return compaction::prepare(self, pending).await;
            }
            if let Some(index) = session_index::prepare(self, pending).await? {
                return Ok(index);
            }
            return compaction::prepare(self, pending).await;
        }
        #[cfg(feature = "service")]
        if let Some(pool) = &self.preparers {
            return pool
                .prepare(&self.ingress, pending.clone(), self.budget.clone())
                .await;
        }
        let dataset = self.read_dataset(Some(pending.plan.read_version)).await?;
        let bytes = usize::try_from(pending.plan.source_bytes()?)?;
        let count = pending.plan.sources.values().map(|s| s.records.len()).sum();
        let reservation = Arc::new(
            self.phases
                .measure(Phase::Admission, async {
                    Ok::<_, anyhow::Error>(
                        self.budget
                            .reserve(self.options.workspace(
                                &self.ingress.scope.contract,
                                bytes,
                                count,
                            )?)
                            .await,
                    )
                })
                .await?,
        );
        stage::materialize(
            self.ingress.clone(),
            self.options.clone(),
            pending.clone(),
            dataset,
            reservation,
            self.phases.clone(),
        )
        .await
    }

    /// Publish the fixed target and confirm its exact identity. Errors preserve
    /// both the prepared files and source pins for explicit reconciliation.
    pub async fn commit_pending(&self) -> Result<Option<Arc<p::Published>>> {
        self.phases
            .clone()
            .measure(Phase::Commit, self.commit_pending_inner())
            .await
    }
    async fn commit_pending_inner(&self) -> Result<Option<Arc<p::Published>>> {
        let view = self.state().await?;
        if let Some(published) = view.progress.published {
            return Ok(Some(published));
        }
        let Some(pending) = view.progress.pending else {
            return Ok(None);
        };
        let prepared = view
            .progress
            .prepared
            .context("pending plan has no prepared files")?;
        let manifests = view.manifests.context("managed manifest index missing")?;
        let target = pending.plan.target_version()?;
        if !manifests.versions.contains_key(&target) {
            let predecessor = manifests
                .versions
                .get(&pending.plan.read_version)
                .context("publication predecessor unavailable")?;
            let base = self.read_dataset(Some(pending.plan.read_version)).await?;
            stage::publish(self, &pending, &prepared, predecessor, base).await?;
        }
        // A lost commit reply is reconciled by this quorum-selected record;
        // absence on a local replica is never treated as a failed publication.
        let selected = self
            .state()
            .await?
            .manifests
            .context("manifest index missing")?
            .versions
            .get(&target)
            .cloned()
            .context("fixed target not selected")?;
        ensure!(
            selected.transaction == pending.plan.transaction
                && selected.plan_index == pending.index
                && selected.prepared_index == prepared.index
                && selected.descriptor_digest == prepared.descriptors.digest,
            "fixed target belongs to another publication"
        );
        let reply = commit(
            &self.ingress.client,
            Command::Confirm(p::ConfirmPublication {
                authority: self.authority.clone(),
                incarnation: self.ingress.scope.incarnation,
                transaction: pending.plan.transaction,
                plan_index: pending.index,
                prepared_index: prepared.index,
                descriptor_digest: prepared.descriptors.digest,
                target,
                manifest_identity: selected.identity,
            }),
        )
        .await?;
        let Applied::Publisher(p::Reply::Published(published)) = reply else {
            bail!("publication confirmation rejected: {reply:?}");
        };
        Ok(Some(published))
    }

    /// Retire payloads only with catalog-issued closure authorization, then
    /// advance independent source cursors using their quorum closure proofs.
    pub async fn finish_pending(&self) -> Result<Option<Arc<p::Advanced>>> {
        self.phases
            .clone()
            .measure(Phase::Retire, self.finish_pending_inner())
            .await
    }
    async fn finish_pending_inner(&self) -> Result<Option<Arc<p::Advanced>>> {
        let view = self.state().await?;
        let Some(pending) = view.progress.pending else {
            return Ok(None);
        };
        let published = view
            .progress
            .published
            .context("publication has not been confirmed")?;
        let published = &published;
        let pending = &pending;
        let proofs = stream::iter(
            pending
                .plan
                .sources
                .keys()
                .copied()
                .map(|group| async move {
                    let query::Value::ClosePermit(Some(permit)) = query(
                        &self.ingress,
                        self.authority.group,
                        query::Query::ClosePermit {
                            table: self.authority.table,
                            transaction: pending.plan.transaction,
                            plan_index: pending.index,
                            source_group: group,
                        },
                    )
                    .await?
                    else {
                        bail!("source closure authorization unavailable");
                    };
                    let expected = permit.closure().clone();
                    let closed = commit(&self.ingress.client, Command::Close(*permit)).await?;
                    ensure!(
                        matches!(closed, Applied::Source(source::Reply::Closed { .. })),
                        "source closure rejected: {closed:?}"
                    );
                    let query::Value::ClosedProof(Some(proof)) = query(
                        &self.ingress,
                        group,
                        query::Query::ClosedProof {
                            table: self.authority.table,
                        },
                    )
                    .await?
                    else {
                        bail!("source closure proof unavailable");
                    };
                    ensure!(
                        proof.closed().permit.closure() == &expected
                            && expected.published_index == published.index
                            && proof.source_cut().group == group,
                        "foreign source closure proof"
                    );
                    Ok::<_, anyhow::Error>((group, *proof))
                }),
        )
        .buffer_unordered(self.options.fetch_parallelism)
        .try_collect::<BTreeMap<_, _>>()
        .await?;
        let reply = commit(
            &self.ingress.client,
            Command::Advance(p::AdvancePublication {
                authority: self.authority.clone(),
                published_index: published.index,
                proofs,
            }),
        )
        .await?;
        let Applied::Publisher(p::Reply::Advanced(advanced)) = reply else {
            bail!("publication advancement rejected: {reply:?}");
        };
        Ok(Some(advanced))
    }
    pub async fn run_once(&mut self) -> Result<Option<Arc<p::Advanced>>> {
        if self.prepare_next().await?.is_none() {
            return Ok(None);
        }
        self.commit_pending().await?;
        self.finish_pending().await
    }

    pub(crate) async fn pin(&self, pending: &p::PendingPlan) -> Result<()> {
        stream::iter(
            pending
                .plan
                .sources
                .keys()
                .copied()
                .map(|group| async move {
                    let query::Value::Pin(Some(permit)) = query(
                        &self.ingress,
                        self.authority.group,
                        query::Query::Pin {
                            table: self.authority.table,
                            transaction: pending.plan.transaction,
                            plan_index: pending.index,
                            source_group: group,
                        },
                    )
                    .await?
                    else {
                        bail!("source pin authorization unavailable");
                    };
                    let selected = permit.selection();
                    ensure!(
                        selected.table == self.authority.table
                            && selected.transaction == pending.plan.transaction
                            && selected.plan_index == pending.index
                            && selected.read_version == pending.plan.read_version
                            && selected.group == group
                            && selected.ingest == self.ingress.scope.contract
                            && selected.range == pending.plan.sources[&group],
                        "pin differs from selected prefix"
                    );
                    let reply = commit(&self.ingress.client, Command::Pin(*permit)).await?;
                    ensure!(
                        matches!(reply, Applied::Source(source::Reply::Pinned { .. })),
                        "source pin rejected: {reply:?}"
                    );
                    Ok::<_, anyhow::Error>(())
                }),
        )
        .buffer_unordered(self.options.fetch_parallelism)
        .try_collect::<Vec<_>>()
        .await?;
        Ok(())
    }
    pub(crate) async fn retire_history(&self, view: &p::Table) -> Result<()> {
        let manifests = view
            .manifests
            .as_ref()
            .context("manifest history missing")?;
        if manifests.versions.len() <= self.options.retained_manifest_versions {
            return Ok(());
        }
        let retain_from = manifests
            .head
            .saturating_sub(self.options.retained_manifest_versions as u64 - 2)
            .max(manifests.initial_version);
        let reply = commit(
            &self.ingress.client,
            Command::RetireHistory(p::RetireManifestHistory {
                authority: self.authority.clone(),
                incarnation: self.ingress.scope.incarnation,
                retain_from,
            }),
        )
        .await?;
        ensure!(
            matches!(reply, Applied::Publisher(p::Reply::ManifestHistory(_))),
            "manifest history retirement rejected: {reply:?}"
        );
        Ok(())
    }
    async fn select(&mut self, view: &p::Table) -> Result<Option<p::Plan>> {
        self.source_scan_idle = false;
        let mut caught_up = true;
        let mut groups = self
            .ingress
            .layout
            .groups
            .keys()
            .copied()
            .collect::<Vec<_>>();
        if self.options.compaction.is_some() || self.options.session_index.is_some() {
            // Maintenance replaces the last append receipt. Rotate from the
            // durable version instead: at most one maintenance operation
            // between appends advances version/2 by at most one per selection.
            // Every group gets the first turn without adding catalog state.
            let offset = (view.progress.minimum_read_version / 2) % groups.len() as u64;
            groups.rotate_left(offset as usize);
        } else if let Some(last) = view
            .progress
            .advanced
            .as_ref()
            .and_then(|a| a.sources.keys().next_back())
        {
            let offset = groups.iter().position(|g| g > last).unwrap_or(0);
            groups.rotate_left(offset);
        }
        let mut sources = BTreeMap::new();
        let mut bytes = 0usize;
        let mut count = 0usize;
        let mut full = false;
        let mut next_group = groups[0];
        for (ordinal, &group) in groups.iter().enumerate() {
            let mut at_end = false;
            let after = view.progress.checkpoint.get(&group).copied().unwrap_or(0);
            let mut cursor = self
                .empty_cursors
                .get(&group)
                .copied()
                .unwrap_or(after)
                .max(after);
            let mut records = Vec::new();
            let mut through = cursor;
            for _ in 0..self.options.scan_pages_per_group {
                let query::Value::Accepted(page) = query(
                    &self.ingress,
                    group,
                    query::Query::Accepted {
                        after: cursor,
                        limit: 128,
                    },
                )
                .await?
                else {
                    bail!("source page unavailable at quorum");
                };
                let mut stopped = false;
                for (index, meta) in page.entries {
                    if meta.stream != self.authority.table {
                        continue;
                    }
                    let size = usize::try_from(meta.body_bytes)?;
                    ensure!(
                        meta.kind == RecordKind::Data
                            && size <= self.ingress.scope.contract.max_record_bytes as usize
                            && self.ingress.layout.group_for_bucket(meta.bucket)? == group,
                        "foreign source metadata"
                    );
                    if size > self.options.max_source_bytes - bytes
                        || count == self.options.max_records
                    {
                        stopped = true;
                        full = true;
                        break;
                    }
                    bytes += size;
                    count += 1;
                    through = index;
                    records.push((index, meta));
                }
                if stopped {
                    break;
                }
                through = page.through;
                if records.is_empty() {
                    self.empty_cursors.insert(group, through);
                }
                if page.through == cursor {
                    at_end = true;
                    break;
                }
                if !records.is_empty() {
                    break;
                }
                cursor = page.through;
            }
            caught_up &= at_end;
            if !records.is_empty() {
                sources.insert(
                    group,
                    p::SourceRange {
                        after,
                        through,
                        records,
                    },
                );
                next_group = groups[(ordinal + 1) % groups.len()];
            }
            if bytes == self.options.max_source_bytes || count == self.options.max_records {
                break;
            }
        }
        if sources.is_empty() {
            self.batch_started = None;
            self.source_scan_idle = caught_up;
            return Ok(None);
        }
        let started = self
            .batch_started
            .get_or_insert_with(tokio::time::Instant::now);
        if !full
            && bytes < self.options.min_source_bytes
            && count < self.options.max_records
            && started.elapsed() < self.options.max_batch_wait
        {
            return Ok(None);
        }
        self.batch_started = None;
        Ok(Some(p::Plan {
            transaction: Uuid::new_v4(),
            read_version: view.progress.minimum_read_version,
            sources,
            next_group,
        }))
    }
}

/// Open a catalog-selected dataset without acquiring publisher ownership.
/// The application resolves and authorizes both the binding and storage first.
pub async fn read_dataset(
    ingress: &Ingress,
    storage: &Storage,
    version: Option<u64>,
) -> Result<Dataset> {
    let reader = p::lance::ManagedManifest::reader(
        Arc::new(ingress.client.manifest_catalog()),
        storage.base.clone(),
        &ingress.layout,
        ingress.scope.table,
        ingress.scope.incarnation,
    )?;
    let mut builder = DatasetBuilder::from_uri(&storage.uri)
        .with_storage_options(storage.options.clone())
        .with_session(storage.session.clone())
        .with_commit_handler(Arc::new(reader));
    if let Some(version) = version {
        builder = builder.with_version(version);
    }
    let dataset = builder.load().await?;
    let schema: arrow_schema::Schema = dataset.schema().into();
    ensure!(
        schema == rollout_schema(),
        "managed dataset is not the bound rollout schema"
    );
    Ok(dataset)
}

pub(super) async fn query(
    ingress: &Ingress,
    group: u64,
    value: query::Query,
) -> Result<query::Value> {
    let response = ingress
        .client
        .request_routed(Request::Query {
            group,
            query: value,
        })
        .await?;
    let Reply::Query(query::Outcome::Observed { cut, value }) = response.reply else {
        bail!("quorum observation unavailable");
    };
    ensure!(
        cut.cluster == ingress.scope.cluster
            && cut.layout == ingress.scope.layout
            && cut.group == group
            && cut.term > 0
            && cut.quorum_index > 0
            && cut.applied >= cut.quorum_index,
        "foreign quorum observation"
    );
    Ok(value)
}
pub(crate) async fn commit(client: &Client, command: Command) -> Result<Applied> {
    let response = client
        .request_routed(Request::Commit(Box::new(command)))
        .await?;
    let Reply::Commit(Outcome::Committed(applied)) = response.reply else {
        bail!("durable command outcome unknown; reconcile the original operation");
    };
    Ok(applied)
}
