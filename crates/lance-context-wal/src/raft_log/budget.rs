//! One physical directory's admission and measurement lifecycle.
use super::*;
#[cfg(any(test, feature = "replication"))]
use crate::admission::Budget;
use crate::admission::Snapshot;

/// Local admission policy, independent of replicated layout/state. Producer
/// proposals stop at `data_bytes`; new images preserve `headroom_bytes` for
/// accepted Raft/control traffic and physical rewrites. Peer writes must still
/// persist and can consume this window. This is not a hard filesystem quota.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "service", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "service", serde(deny_unknown_fields))]
pub struct DiskOptions {
    pub max_bytes: usize,
    pub data_bytes: usize,
    pub headroom_bytes: usize,
}
#[cfg(any(test, feature = "replication"))]
impl DiskOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.max_bytes >= 4 << 20
                && self.headroom_bytes > 0
                && self.data_bytes > 0
                && self.data_bytes < self.max_bytes
                && self.headroom_bytes <= self.max_bytes - self.data_bytes,
            "invalid node disk admission policy"
        );
        Ok(())
    }
}

/// A blocking measurement task retaining the engine's exclusive directory lock.
/// No payload reads/decoding. Appends overlap safely via their growth counter;
/// native rewrite and image deletion are excluded by the lifecycle guard.
pub struct DiskMeasurement {
    node: Arc<NodeEngine>,
}
impl DiskMeasurement {
    pub fn run(&self) -> Result<Snapshot> {
        self.node.check()?;
        let budget = self
            .node
            .budget
            .get()
            .context("node disk budget not configured")?;
        budget.refresh()?;
        self.node.check()?;
        Ok(budget.snapshot())
    }
}

impl GroupLog {
    #[cfg(feature = "replication")]
    pub(crate) fn allocation_headroom(&self) -> Result<u64> {
        self.node
            .target_file_bytes
            .checked_mul(2)
            .context("log file headroom overflow")
    }
    #[cfg(any(test, feature = "replication"))]
    pub(crate) fn configure_disk(&self, options: DiskOptions) -> Result<()> {
        options.validate()?;
        let _guard = self
            .node
            .maintenance_gate
            .write()
            .map_err(|_| anyhow::anyhow!("maintenance gate poisoned"))?;
        self.node.check()?;
        ensure!(
            self.node.budget.get().is_none(),
            "node disk budget already configured"
        );
        let _files = self
            .node
            .filesystem
            .write()
            .map_err(|_| anyhow::anyhow!("filesystem lifecycle lock poisoned"))?;
        let budget = Budget::new(
            &self.node.directory,
            self.node.written.clone(),
            options.max_bytes,
            self.node.filesystem.clone(),
        )?;
        budget.limit_data(options.data_bytes)?;
        budget.limit_files(options.max_bytes - options.headroom_bytes)?;
        self.node
            .budget
            .set(budget)
            .map_err(|_| anyhow::anyhow!("node disk budget already configured"))?;
        Ok(())
    }
    #[cfg(any(test, feature = "replication"))]
    pub(crate) fn disk_budget(&self) -> Option<&Arc<Budget>> {
        self.node.budget.get()
    }
    pub fn disk_measurement(&self) -> Result<DiskMeasurement> {
        ensure!(
            self.node.budget.get().is_some(),
            "node disk budget not configured"
        );
        Ok(DiskMeasurement {
            node: self.node.clone(),
        })
    }
    #[cfg(any(test, feature = "replication"))]
    pub(crate) fn image_collection_guard(&self) -> Result<std::sync::RwLockWriteGuard<'_, ()>> {
        self.node
            .filesystem
            .write()
            .map_err(|_| anyhow::anyhow!("maintenance gate poisoned"))
    }
    #[cfg(feature = "replication")]
    pub(crate) fn refresh_disk_exclusive(&self) -> Result<()> {
        if let Some(budget) = self.node.budget.get() {
            budget.refresh_exclusive()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_collection_lifecycle_does_not_block_a_real_log_append() -> Result<()> {
        let layout = Layout {
            cluster: uuid::Uuid::from_u128(1),
            nodes: (1..=3)
                .map(|id| (id, format!("wal-{id}.example.invalid:7000")))
                .collect(),
            groups: BTreeMap::from([(10, vec![1, 2, 3]), (20, vec![1, 2, 3])]),
            buckets: vec![10, 20],
        };
        let directory = tempfile::tempdir()?;
        let log = PhysicalLog::open(directory.path(), &layout, 1)?;
        let image_owner = log.group(10)?;
        image_owner.configure_disk(DiskOptions {
            max_bytes: 512 << 20,
            data_bytes: 128 << 20,
            headroom_bytes: 256 << 20,
        })?;
        let lifecycle = image_owner.image_collection_guard()?;
        let writer = log.group(20)?;
        let (sent, received) = std::sync::mpsc::sync_channel(1);
        let thread = std::thread::spawn(move || {
            let result = writer.persist(
                &[Entry {
                    index: 1,
                    term: 1,
                    data: vec![7; 4096].into(),
                    ..Default::default()
                }],
                Some(&HardState {
                    term: 1,
                    vote: 1,
                    commit: 1,
                    ..Default::default()
                }),
            );
            let _ = sent.send(result);
        });
        let result = received.recv_timeout(std::time::Duration::from_secs(10));
        drop(lifecycle);
        thread.join().unwrap();
        result.context("image deletion lifecycle blocked an independent log append")??;
        assert_eq!(log.group(20)?.initial_state()?.hard_state.commit, 1);
        assert!(
            image_owner
                .disk_budget()
                .unwrap()
                .snapshot()
                .accounted_disk_bytes
                > 4096
        );
        Ok(())
    }
}
