//! Bounded physical maintenance through the existing empty-source WAL plan.
use super::*;

impl Publisher {
    async fn prefer_annotation_index(&mut self, version: u64) -> Result<bool> {
        Ok(!materializer::Publisher::compaction_first(version)
            && self.catalog.has_annotation_index_work(version).await?)
    }

    pub(super) async fn compact_next(&mut self, view: &p::Table) -> Result<bool> {
        if let Some(pending) = &view.progress.pending {
            if !pending.plan.is_maintenance() {
                return Ok(false);
            }
        } else {
            let source = self.service.source_observation();
            // The first annotation publication installs the durable cut/result
            // checkpoint. Spend at most the existing batch-wait window starting
            // further rewrites between publications. This reduces the fragment
            // fanout of subsequent narrow updates without an unbounded burst.
            // A retained plan always resolves above; idle maintenance may drain.
            // After one maintenance turn, a ready or pressured source batch
            // takes the next turn. The first maintenance after a data publication
            // remains eligible, so continuous writes still advance maintenance.
            if (self.options.compaction.is_none()
                && self.options.turn_id_index.is_none()
                && self.options.status_index.is_none())
                || view
                    .progress
                    .checkpoint
                    .get(&self.service.group()?)
                    .copied()
                    .unwrap_or(0)
                    == 0
                || (source.records != 0
                    && !view
                        .progress
                        .advanced
                        .as_ref()
                        .is_some_and(|a| !a.sources.is_empty())
                    && (source.bytes >= self.options.min_source_bytes as u64
                        || source.records >= self.options.max_records
                        || self.service.apply_backpressure().is_some()
                        || self.compaction_started.is_none_or(|started| {
                            started.elapsed()
                                >= self.options.max_batch_wait.min(Duration::from_secs(1))
                        })))
            {
                return Ok(false);
            }
            // Both datasets and their index coverage are fixed by this selected
            // narrow version. Recheck after publication, not on every idle poll.
            // A retained plan above always resolves, regardless of this hint.
            if self.compaction_checked_version == Some(view.progress.minimum_read_version) {
                return Ok(false);
            }
            let plan = if !self
                .prefer_annotation_index(view.progress.minimum_read_version)
                .await?
                && self
                    .body_compaction_candidate(view.progress.minimum_read_version)
                    .await?
                    .is_some()
            {
                Some(p::Plan {
                    transaction: Uuid::new_v4(),
                    read_version: view.progress.minimum_read_version,
                    sources: BTreeMap::new(),
                    next_group: self.service.group()?,
                })
            } else {
                self.catalog.maintenance_plan(view).await?
            };
            let Some(plan) = plan else {
                // Remember only a completed no-candidate check, never a budget
                // skip, a failed read, or a plan that still needs to finish.
                self.compaction_checked_version = Some(view.progress.minimum_read_version);
                return Ok(false);
            };
            // One slow rewrite can exceed the window. Its durable plan must
            // finish, but after that pending annotations get the next turn.
            self.compaction_started
                .get_or_insert_with(tokio::time::Instant::now);
            self.catalog.retire_history(view).await?;
            self.catalog.record_plan(plan).await?;
        }
        // A maintenance plan already exists, so the ordinary publisher only
        // prepares that exact plan. It cannot select an ingestion append here.
        let current = self.catalog.state().await?;
        let pending = current
            .progress
            .pending
            .as_ref()
            .context("maintenance plan missing")?;
        if current.progress.prepared.is_none()
            && (self
                .prefer_annotation_index(pending.plan.read_version)
                .await?
                || !self.prepare_body_compaction(pending).await?)
        {
            self.catalog
                .prepare_next()
                .await?
                .context("annotation compaction preparation missing")?;
        }
        let ready = self.catalog.state().await?;
        let selected = self.resolve_prepared(&ready).await?;
        let transaction = selected
            .read_transaction()
            .await?
            .context("selected compaction has no transaction")?;
        {
            let _ordered = self.service.shared.ordered.lock().await;
            if matches!(
                transaction.operation,
                lance_table::transaction::Operation::Rewrite { .. }
            ) {
                self.service
                    .shared
                    .staging
                    .install_compacted(&transaction, selected)
                    .await?;
            } else if matches!(
                transaction.operation,
                lance_table::transaction::Operation::CreateIndex { .. }
            ) {
                self.service
                    .shared
                    .staging
                    .install_indexed(&transaction, selected)
                    .await?;
            } else {
                self.service
                    .shared
                    .staging
                    .install_body_compacted(&transaction, selected)
                    .await?;
            }
        }
        // An empty-source maintenance completion changes neither the source
        // checkpoint nor the in-memory result snapshot or active source window.
        self.catalog.finish_pending().await?;
        Ok(true)
    }
}
