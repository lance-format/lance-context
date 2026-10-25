//! Immutable selection of source records and a fixed Lance transaction target.
//! Source groups must subsequently verify their complete prefixes before pinning.
use super::*;

#[cfg(test)]
mod tests;

pub const MAX_INPUTS: usize = 512;
pub(super) const MAX_PLAN_BYTES: usize = 128 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRange {
    pub after: u64,
    pub through: u64,
    pub records: Vec<(u64, BatchMeta)>,
}
impl SourceRange {
    pub(in crate::replication) fn validate(
        &self,
        layout: &Layout,
        table: Uuid,
        group: u64,
        limits: &Limits,
    ) -> Result<()> {
        ensure!(
            layout.groups.contains_key(&group)
                && self.after < self.through
                && self.through < u64::MAX
                && !self.records.is_empty()
                && self.records.len() <= MAX_INPUTS
                && self.records.windows(2).all(|w| w[0].0 < w[1].0),
            "invalid publication source range"
        );
        let mut bytes = 0u64;
        for (index, meta) in &self.records {
            ensure!(
                *index > self.after
                    && *index <= self.through
                    && meta.stream == table
                    && meta.kind == RecordKind::Data
                    && !meta.producer.is_nil()
                    && meta.sequence > 0
                    && meta.sequence < u64::MAX
                    && meta.body_bytes > 0
                    && meta.body_bytes <= limits.max_record_bytes
                    && layout.group_for_bucket(meta.bucket)? == group,
                "invalid publication source metadata"
            );
            bytes = bytes
                .checked_add(meta.body_bytes)
                .context("source byte sum overflow")?;
        }
        ensure!(
            bytes <= limits.max_plan_source_bytes,
            "publication source byte limit"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub transaction: Uuid,
    pub read_version: u64,
    /// An empty selection denotes maintenance of the fixed read version. Its
    /// immutable prepared descriptors select the operation; no source cursor advances.
    pub sources: BTreeMap<u64, SourceRange>,
    /// Resume fair source selection from this group after publication.
    pub next_group: u64,
}
impl Plan {
    pub fn is_maintenance(&self) -> bool {
        self.sources.is_empty()
    }

    /// Legacy name for a source-free maintenance plan.
    pub fn is_compaction(&self) -> bool {
        self.is_maintenance()
    }

    pub fn target_version(&self) -> Result<u64> {
        ensure!(
            self.read_version > 0,
            "publication requires an existing dataset"
        );
        self.read_version
            .checked_add(1)
            .context("Lance version exhausted")
    }

    pub fn source_bytes(&self) -> Result<u64> {
        self.sources
            .values()
            .flat_map(|range| &range.records)
            .try_fold(0u64, |sum, (_, meta)| {
                sum.checked_add(meta.body_bytes)
                    .context("source byte sum overflow")
            })
    }

    pub(super) fn validate(&self, layout: &Layout, table: Uuid, limits: &Limits) -> Result<()> {
        self.target_version()?;
        ensure!(
            !self.transaction.is_nil()
                && !table.is_nil()
                && self.sources.len() <= layout.groups.len()
                && layout.groups.contains_key(&self.next_group),
            "invalid publication plan identity or group selection"
        );
        let mut count = 0usize;
        for (group, range) in &self.sources {
            count = count
                .checked_add(range.records.len())
                .context("plan input count overflow")?;
            ensure!(count <= MAX_INPUTS, "publication input count limit");
            range.validate(layout, table, *group, limits)?;
        }
        ensure!(
            self.source_bytes()? <= limits.max_plan_source_bytes,
            "publication source byte limit"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanRequest {
    pub authority: Authority,
    pub plan: Plan,
}
impl PlanRequest {
    pub(in crate::replication) fn batch(self, layout: &Layout, limits: &Limits) -> Result<Batch> {
        self.validate(layout, limits)?;
        Command::Plan(Box::new(self)).batch(layout)
    }
    pub(super) fn validate(&self, layout: &Layout, limits: &Limits) -> Result<()> {
        self.authority.validate(layout)?;
        self.plan.validate(layout, self.authority.table, limits)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingPlan {
    pub index: u64,
    pub plan: Plan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    /// Independent source-group positions; never collapse these to one maximum.
    pub checkpoint: BTreeMap<u64, u64>,
    pub minimum_read_version: u64,
    /// Cloning a table view shares the immutable metadata; it never clones bodies.
    pub pending: Option<Arc<PendingPlan>>,
    /// Materialization result shares ownership independently of the selection.
    pub prepared: Option<Arc<Prepared>>,
    /// Durable selected-manifest resolution; sources remain pinned for closure.
    pub published: Option<Arc<Published>>,
    /// Latest completed advancement, retained for ambiguous exact retries.
    pub advanced: Option<Arc<Advanced>>,
}
impl Default for Progress {
    fn default() -> Self {
        Self {
            checkpoint: BTreeMap::new(),
            minimum_read_version: 1,
            pending: None,
            prepared: None,
            published: None,
            advanced: None,
        }
    }
}
impl Progress {
    pub(super) fn matches(&self, plan: &Plan, ingest: &IngestContract) -> bool {
        plan.read_version >= self.minimum_read_version
            && plan.sources.iter().all(|(group, range)| {
                range.after == self.checkpoint.get(group).copied().unwrap_or(0)
                    && range
                        .records
                        .iter()
                        .all(|(_, meta)| meta.body_bytes <= u64::from(ingest.max_record_bytes))
            })
    }

    pub(super) fn validate(
        &self,
        layout: &Layout,
        table: &Table,
        cut: u64,
        limits: &Limits,
    ) -> Result<()> {
        if let Some(advanced) = &self.advanced {
            advanced.validate(table, cut)?;
        }
        if let Some(published) = &self.published {
            published.validate(table, cut)?;
        }
        ensure!(
            self.minimum_read_version > 0
                && self.checkpoint.len() <= layout.groups.len()
                && self.checkpoint.iter().all(|(group, position)| {
                    layout.groups.contains_key(group) && *position > 0 && *position < u64::MAX
                }),
            "invalid publication progress"
        );
        if let Some(prepared) = &self.prepared {
            let pending = self
                .pending
                .as_ref()
                .context("prepared result without plan")?;
            prepared.descriptors.validate(limits.max_descriptor_bytes)?;
            ensure!(
                prepared
                    .descriptors
                    .matches_plan(&pending.plan, &table.ingest.policy)
                    && prepared.plan_index == pending.index
                    && prepared.transaction == pending.plan.transaction
                    && prepared.index > pending.index
                    && prepared.index <= cut
                    && prepared.index != table.authority.grant_index,
                "invalid prepared checkpoint identity or index"
            );
        }
        if let Some(pending) = &self.pending {
            pending
                .plan
                .validate(layout, table.authority.table, limits)?;
            ensure!(
                pending.index > 0
                    && pending.index <= cut
                    && pending.index != table.authority.grant_index
                    && pending
                        .plan
                        .sources
                        .get(&table.authority.group)
                        .is_none_or(|range| range.through < pending.index)
                    && self.matches(&pending.plan, &table.ingest),
                "invalid pending publication plan"
            );
        }
        Ok(())
    }
}

impl Catalog {
    pub(super) fn plan(&mut self, request: PlanRequest, index: u64, limits: &Limits) -> Reply {
        let Some(table) = self.tables.get(&request.authority.table) else {
            return Reply::Rejected(Rejection::Authority);
        };
        if table.authority != request.authority {
            return Reply::Rejected(Rejection::Authority);
        }
        if let Some(pending) = &table.progress.pending {
            return if pending.plan == request.plan {
                Reply::Planned {
                    index: pending.index,
                }
            } else {
                Reply::Rejected(Rejection::PendingPlan)
            };
        }
        if !table.progress.matches(&request.plan, &table.ingest)
            || table
                .manifests
                .as_ref()
                .is_some_and(|m| m.head != request.plan.read_version)
            || request
                .plan
                .sources
                .get(&table.authority.group)
                .is_some_and(|range| range.through >= index)
        {
            return Reply::Rejected(Rejection::SourcePosition);
        }
        if self
            .tables
            .values()
            .filter(|t| t.progress.pending.is_some())
            .count() as u64
            >= limits.max_active_plans
        {
            return Reply::Rejected(Rejection::Capacity);
        }
        self.tables
            .get_mut(&request.authority.table)
            .expect("known table")
            .progress
            .pending = Some(Arc::new(PendingPlan {
            index,
            plan: request.plan,
        }));
        Reply::Planned { index }
    }
}
