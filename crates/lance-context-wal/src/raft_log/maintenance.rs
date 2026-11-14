//! Physical segment maintenance after durable logical checkpointing.
use super::*;

/// A handle to one physical engine's maintenance boundary, shared by every
/// hosted group. Run it on a bounded blocking worker, outside the Raft loop.
/// It retains the native directory lock until dropped, including after all
/// group owners have been dropped.
pub struct LogMaintenance {
    node: Arc<NodeEngine>,
}

impl LogMaintenance {
    /// Rewrite retained entries and metadata, then delete obsolete native log
    /// segments. Returned group IDs are checkpoint suggestions, not permission
    /// to truncate those groups. This never advances a logical checkpoint or
    /// deletes application snapshots or Lance objects.
    ///
    /// Pending appends and checkpoint metadata first cross a physical sync
    /// barrier. Sizing and that barrier briefly exclude foreground work; the
    /// native rewrite then overlaps appends and reads. Its eligible append prefix
    /// and tombstones are captured with the rewrite reservation. Native payload
    /// readers are excluded only for physical file retirement. An error or panic
    /// fences every group on the node. This API does not bound filesystem latency.
    pub fn run(&self) -> Result<Vec<u64>> {
        let _job = self
            .node
            .maintenance_job
            .lock()
            .map_err(|_| anyhow::anyhow!("maintenance job lock poisoned"))?;
        // A refresh cannot measure uncounted native rewrite growth while its
        // reservation is in flight. Ordinary appends retain their growth counter.
        let _files = self
            .node
            .filesystem
            .write()
            .map_err(|_| anyhow::anyhow!("filesystem lifecycle lock poisoned"))?;
        self.node.check()?;
        let groups = match self.run_pass(false) {
            Ok(groups) => groups,
            Err(error)
                if error.is::<crate::admission::DiskBudgetExhausted>()
                    && self
                        .node
                        .engine
                        .purge_manager()
                        .has_obsolete_rewrite_entries() =>
            {
                // Normal reservation failed before touching storage. Retained
                // append data can make that allowance too large even when the
                // smaller rewrite-queue pass can reclaim obsolete bytes. Keep
                // its independent reservation and all sync/failure fencing.
                // No-garbage admission failures and native I/O errors stay errors.
                return self.run_pass(true).with_context(|| {
                    format!("normal rewrite admission failed before reclamation: {error:#}")
                });
            }
            Err(error) => return Err(error),
        };
        if self.node.budget.get().is_some_and(|budget| {
            let observed = budget.snapshot();
            observed
                .accounted_disk_bytes
                .saturating_add(observed.reserved_proposal_bytes)
                .saturating_add(observed.reserved_file_bytes)
                .saturating_add(self.node.target_file_bytes.saturating_mul(2))
                > observed.data_limit
        }) {
            // Native rewrite eligibility counts dead entries, not their bytes.
            // A few compacted large records can keep admission closed while
            // many tiny live metadata values suppress the normal heuristic.
            // Pending image files share producer admission's disk charge even
            // before their writes become measured retained bytes.
            // Reserve fresh headroom against the first pass's measured result;
            // never assume that two rewrites fit one reservation.
            self.run_pass(true)?;
        }
        Ok(groups)
    }

    fn run_pass(&self, force_rewrite: bool) -> Result<Vec<u64>> {
        // Admission failure occurs before touching native storage and must not
        // fence a healthy node. Existing accepted traffic still owns its window.
        // Exclude writes while sizing live entries. Dead segments need deletion,
        // not another directory-sized reservation that can prevent their cleanup.
        let (_reservation, boundary) = {
            let _maintenance = self
                .node
                .maintenance_gate
                .write()
                .map_err(|_| anyhow::anyhow!("maintenance gate poisoned"))?;
            let _sync = self
                .node
                .sync_gate
                .lock()
                .map_err(|_| anyhow::anyhow!("sync gate poisoned"))?;
            let _exclusive = self
                .node
                .append_gate
                .write()
                .map_err(|_| anyhow::anyhow!("append gate poisoned"))?;
            self.node.check()?;
            // Freeze eligibility with the allowance. Later appends can move
            // the native ratio-based watermark, but not this pass's copy cap.
            let boundary = self.node.engine.purge_manager().capture_append_boundary();
            let reservation = self
                .node
                .budget
                .get()
                .map(|budget| {
                    // Pressure reclamation touches only the rewrite queue.
                    // Charging the unrelated append tail again can itself close
                    // producer admission while this background pass runs.
                    let live = if force_rewrite {
                        self.node.engine.estimated_rewrite_queue_bytes()
                    } else {
                        self.node.engine.estimated_rewrite_bytes_for(&boundary)
                    };
                    let bytes = live
                        .and_then(|live| {
                            usize::try_from(self.node.target_file_bytes)
                                .ok()?
                                .checked_mul(2)?
                                .checked_add(live)
                        })
                        .context("native rewrite reservation overflow")?;
                    budget.reserve_rewrite(bytes)
                })
                .transpose()?;
            if let Err(error) = self.node.sync_locked() {
                self.node.failed.store(true, Ordering::Release);
                return Err(error);
            }
            // Later appends and their tombstones stay outside this pass. Rewrite
            // queue values cannot grow while our node-wide job guard is held.
            (reservation, boundary)
        };
        let result = self.run_native(force_rewrite, boundary);
        if result.is_err() {
            self.node.failed.store(true, Ordering::Release);
        }
        result
    }

    fn run_native(
        &self,
        force_rewrite: bool,
        boundary: raft_engine::internals::PurgeBoundary,
    ) -> Result<Vec<u64>> {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if force_rewrite {
                self.node.engine.purge_manager().reclaim_rewrite_queue()?;
                Ok(Vec::new())
            } else {
                self.node
                    .engine
                    .purge_manager()
                    .purge_expired_files_before(boundary)
            }
        }));
        let mut groups = match outcome {
            Ok(Ok(groups)) => groups,
            Ok(Err(error)) => {
                self.node.failed.store(true, Ordering::Release);
                return Err(error).context("uncertain log maintenance; reopen the node log");
            }
            Err(_) => {
                self.node.failed.store(true, Ordering::Release);
                anyhow::bail!("log maintenance panicked; reopen the node log");
            }
        };
        groups.retain(|group| *group != 0);
        groups.sort_unstable();
        groups.dedup();
        if let Some(budget) = self.node.budget.get() {
            // Native rewrites have completed. The filesystem lifecycle excludes
            // other unlink/rename work; concurrent appends use the growth counter.
            budget.refresh_exclusive()?;
        }
        self.node.check()?;
        Ok(groups)
    }
}

impl GroupLog {
    /// Return a node-wide physical-maintenance handle. One pass serves all
    /// groups on this engine; scheduling a pass per group repeats the same work.
    pub fn maintenance(&self) -> LogMaintenance {
        LogMaintenance {
            node: self.node.clone(),
        }
    }
}

#[cfg(test)]
mod tests;
