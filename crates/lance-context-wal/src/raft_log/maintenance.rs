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
    /// barrier. Payload reads and writes are excluded for this maintenance pass;
    /// an error or panic fences every group on the node. The native pass has no
    /// per-call byte/time bound, so callers must measure its latency and reserve
    /// rewrite headroom. This API alone does not impose a disk quota.
    pub fn run(&self) -> Result<Vec<u64>> {
        // Admission failure occurs before touching native storage and must not
        // fence a healthy node. Existing accepted traffic still owns its window.
        let _reservation = self
            .node
            .budget
            .get()
            .map(|budget| {
                budget.reserve(
                    usize::try_from(budget.snapshot().accounted_disk_bytes)?,
                    false,
                )
            })
            .transpose()?;
        let result = self.run_locked();
        if result.is_err() {
            self.node.failed.store(true, Ordering::Release);
        }
        result
    }

    fn run_locked(&self) -> Result<Vec<u64>> {
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
        // The maintenance gate drained pending appends and checkpoint writes.
        // Establish an explicit durable marker before the native rewrite too.
        self.node.sync_locked()?;
        let _files = self
            .node
            .filesystem
            .write()
            .map_err(|_| anyhow::anyhow!("filesystem lifecycle lock poisoned"))?;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.node.engine.purge_expired_files()
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
            // Native rewrites are outside Engine::write. While every appender
            // and image collector is excluded, replace the measured baseline.
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
