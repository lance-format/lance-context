//! Node-wide proposal admission, shared by every physical group on one engine.
//! This reserves a CONTROL WINDOW, not filesystem extents or a bound on future
//! catch-up traffic. Replication must still persist already accepted entries;
//! snapshot/GC and replica-pressure coordination are required for sustained use.
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex, RwLock, TryLockError,
};

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub measured_disk_bytes: u64,
    pub accounted_disk_bytes: u64,
    pub reserved_proposal_bytes: u64,
    pub reserved_file_bytes: u64,
    pub data_limit: u64,
    pub file_limit: u64,
    pub max_log_bytes: u64,
}
#[derive(Debug)]
struct Accounting {
    measured: u64,
    counter_at_measurement: u64,
    pending: u64,
    pending_files: u64,
}
#[derive(Debug)]
pub struct Budget {
    directory: PathBuf,
    written: Arc<AtomicU64>,
    data_limit: AtomicU64,
    file_limit: AtomicU64,
    failed: AtomicBool,
    max: u64,
    accounting: Mutex<Accounting>,
    filesystem: Arc<RwLock<()>>,
}
impl Budget {
    /// `written` must count every encoded engine append and rewrite, across all
    /// groups and peer/control traffic. It must never reset during this budget's
    /// lifetime. Files are direct children of `directory`; no recursive scan.
    /// Hold `filesystem` for writing during file deletion/purge/rename. Appends
    /// may overlap refresh, provided the counter includes their encoded bytes.
    pub fn new(
        directory: &Path,
        written: Arc<AtomicU64>,
        max: usize,
        filesystem: Arc<RwLock<()>>,
    ) -> Result<Arc<Self>> {
        ensure!(max >= 4 * 1024 * 1024, "log budget must be at least 4 MiB");
        let counter = written.load(Ordering::Acquire);
        let measured = retained_disk_bytes(directory)?;
        // Reserve 25%, up to 256 MiB, from new producer proposals. It remains
        // available for publisher/control records and previously accepted Raft
        // traffic. This is explicitly not a guarantee against a full filesystem.
        let reserve = (max / 4).min(256 * 1024 * 1024);
        Ok(Arc::new(Self {
            directory: directory.to_owned(),
            written,
            filesystem,
            max: max as u64,
            data_limit: AtomicU64::new((max - reserve) as u64),
            file_limit: AtomicU64::new(max as u64),
            failed: AtomicBool::new(false),
            accounting: Mutex::new(Accounting {
                measured,
                counter_at_measurement: counter,
                pending: 0,
                pending_files: 0,
            }),
        }))
    }
    /// Tighten new-proposal admission to leave additional space for checkpoint
    /// copies and engine rewrites. All groups observe the same ceiling. Existing
    /// reservations remain charged; lowering the limit does not revoke them.
    /// The total budget and already accepted replication traffic are unchanged.
    pub fn limit_data(&self, bytes: usize) -> Result<()> {
        let bytes = u64::try_from(bytes)?;
        let _state = self.accounting.lock().unwrap();
        ensure!(
            bytes > 0 && bytes <= self.data_limit.load(Ordering::Acquire),
            "data limit must be positive and cannot raise the producer ceiling"
        );
        self.data_limit.store(bytes, Ordering::Release);
        Ok(())
    }
    fn accounted(&self, state: &Accounting) -> u64 {
        state.measured.saturating_add(
            self.written
                .load(Ordering::Acquire)
                .saturating_sub(state.counter_at_measurement),
        )
    }
    /// No directory walk on successful admission. The engine's exact encoded
    /// write counts include peer traffic and control/HardState writes across all
    /// groups; pending proposals are charged atomically on top of that baseline.
    pub fn reserve(self: &Arc<Self>, bytes: usize, data: bool) -> Result<Reservation> {
        self.reserve_below(bytes, false, || {
            if data {
                self.data_limit.load(Ordering::Acquire)
            } else {
                self.max
            }
        })
    }
    fn reserve_below(
        self: &Arc<Self>,
        bytes: usize,
        file: bool,
        limit: impl FnOnce() -> u64,
    ) -> Result<Reservation> {
        let bytes = u64::try_from(bytes)?;
        let mut state = self.accounting.lock().unwrap();
        ensure!(
            !self.failed.load(Ordering::Acquire),
            "disk accounting failed"
        );
        ensure!(
            self.accounted(&state)
                .checked_add(state.pending)
                .and_then(|n| n.checked_add(bytes))
                .is_some_and(|n| n <= limit()),
            "node proposal disk budget exhausted; data admission preserves control window"
        );
        state.pending += bytes;
        if file {
            state.pending_files += bytes;
        }
        Ok(Reservation {
            budget: self.clone(),
            bytes,
            file,
        })
    }

    /// Preserve a separate log/control/rewrite window from new image files.
    /// Apply before exposing the node. Already held reservations remain charged.
    pub fn limit_files(&self, bytes: usize) -> Result<()> {
        let bytes = u64::try_from(bytes)?;
        let _state = self.accounting.lock().unwrap();
        ensure!(
            bytes > 0 && bytes <= self.file_limit.load(Ordering::Acquire),
            "invalid file ceiling"
        );
        self.file_limit.store(bytes, Ordering::Release);
        Ok(())
    }

    /// Reserve complete file lengths before creation, rounded independently to
    /// 4 KiB. Keep the owner until all writes (including buffered drops) finish.
    /// Drop converts the whole reservation to counted growth: interrupted and
    /// installed files remain charged until a later directory measurement.
    /// Filesystem allocation can exceed this rounding; this is admission with
    /// reserved headroom, not an OS quota or a physical free-space guarantee.
    pub fn reserve_files(self: &Arc<Self>, lengths: &[u64]) -> Result<FileReservation> {
        let bytes = lengths.iter().try_fold(0u64, |sum, length| {
            let rounded = length
                .checked_add(4095)
                .map(|n| n / 4096 * 4096)
                .ok_or_else(|| anyhow::anyhow!("file reservation overflow"))?;
            sum.checked_add(rounded)
                .ok_or_else(|| anyhow::anyhow!("file reservation overflow"))
        })?;
        Ok(FileReservation(self.reserve_below(
            usize::try_from(bytes)?,
            true,
            || self.file_limit.load(Ordering::Acquire),
        )?))
    }
    /// Refresh after persistence and on explicit status reads. Sampling the
    /// counter BEFORE reading directory metadata makes overlapping appends count
    /// twice at worst; it cannot subtract a write the measurement did not see.
    /// Engine append writes must not acquire the accounting mutex: independent
    /// group writes can still coalesce in the engine.
    pub fn refresh(&self) -> Result<()> {
        ensure!(
            !self.failed.load(Ordering::Acquire),
            "disk accounting failed"
        );
        let _files = match self.filesystem.try_read() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(()),
            Err(TryLockError::Poisoned(_)) => anyhow::bail!("maintenance filesystem lock poisoned"),
        };
        self.refresh_exclusive()
    }
    /// Caller owns a directory lifecycle read/write guard. No unlink/purge can
    /// overlap this measurement or its accounting update. Engine appends can.
    pub(crate) fn refresh_exclusive(&self) -> Result<()> {
        let counter = self.written.load(Ordering::Acquire);
        let measured = retained_disk_bytes(&self.directory)?;
        let mut state = self.accounting.lock().unwrap();
        if counter >= state.counter_at_measurement {
            state.measured = measured;
            state.counter_at_measurement = counter;
        }
        Ok(())
    }
    pub fn snapshot(&self) -> Snapshot {
        let state = self.accounting.lock().unwrap();
        Snapshot {
            measured_disk_bytes: state.measured,
            accounted_disk_bytes: self.accounted(&state),
            reserved_proposal_bytes: state.pending - state.pending_files,
            reserved_file_bytes: state.pending_files,
            data_limit: self.data_limit.load(Ordering::Acquire),
            file_limit: self.file_limit.load(Ordering::Acquire),
            max_log_bytes: self.max,
        }
    }
}

/// A physical-file operation's charge outlives network cancellation. Unlike a
/// proposal reservation, dropping this owner does not return retained disk space.
pub struct FileReservation(Reservation);
impl Drop for FileReservation {
    fn drop(&mut self) {
        let budget = &self.0.budget;
        // Publish growth before Reservation's drop releases its pending charge.
        // A racing refresh sees the existing files or retains this growth delta.
        if budget
            .written
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(self.0.bytes)
            })
            .is_err()
        {
            budget.failed.store(true, Ordering::Release);
        }
    }
}
#[derive(Debug)]
pub struct Reservation {
    budget: Arc<Budget>,
    bytes: u64,
    file: bool,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut state = self.budget.accounting.lock().unwrap();
        state.pending -= self.bytes;
        if self.file {
            state.pending_files -= self.bytes;
        }
    }
}

fn retained_disk_bytes(directory: &Path) -> Result<u64> {
    let mut bytes = 0u64;
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        ensure!(
            !entry.file_type()?.is_symlink(),
            "WAL accounting directory contains a symlink"
        );
        let metadata = entry.metadata()?;
        if !metadata.is_file() {
            continue;
        }
        let mut occupied = metadata.len();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            occupied = occupied.max(metadata.blocks().saturating_mul(512));
        }
        bytes = bytes
            .checked_add(occupied)
            .ok_or_else(|| anyhow::anyhow!("disk usage overflow"))?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interrupted_files_stay_charged_until_measurement_and_actual_collection() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let lifecycle = Arc::new(RwLock::new(()));
        let budget = Budget::new(
            directory.path(),
            Arc::new(AtomicU64::new(0)),
            8 << 20,
            lifecycle.clone(),
        )?;
        budget.limit_data(4 << 20)?;
        budget.limit_files(5 << 20)?;
        let receiving = budget.reserve_files(&[3 << 20, 1])?;
        assert!(budget.reserve(1 << 20, true).is_err());
        let path = directory.path().join("partial.pack");
        std::fs::write(&path, vec![7; 1 << 20])?;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)?
            .sync_all()?;
        let metadata = std::fs::metadata(&path)?;
        let occupied = metadata.len();
        #[cfg(unix)]
        let occupied = {
            use std::os::unix::fs::MetadataExt;
            occupied.max(metadata.blocks() * 512)
        };
        // Allocation may exceed logical length, but must still distinguish the
        // partial file from the larger unfinished-file reservation in this case.
        assert!((1 << 20..3 << 20).contains(&occupied));
        // A measurement while the writer still exists cannot return its credit.
        budget.refresh()?;
        drop(receiving);
        assert_eq!(budget.snapshot().reserved_proposal_bytes, 0);
        assert!(budget.snapshot().accounted_disk_bytes >= 3 << 20);
        assert!(budget.reserve(2 << 20, true).is_err());
        budget.refresh()?;
        assert_eq!(budget.snapshot().accounted_disk_bytes, occupied);
        let reopened = Budget::new(
            directory.path(),
            Arc::new(AtomicU64::new(0)),
            8 << 20,
            lifecycle.clone(),
        )?;
        assert_eq!(reopened.snapshot().accounted_disk_bytes, occupied);
        {
            let _files = lifecycle.write().unwrap();
            std::fs::remove_file(path)?;
            budget.refresh_exclusive()?;
        }
        assert_eq!(budget.snapshot().accounted_disk_bytes, 0);
        assert!(budget.reserve_files(&[u64::MAX]).is_err());
        assert!(budget.reserve_files(&[5 << 20, 1]).is_err());
        assert!(
            budget.reserve(6 << 20, false).is_ok(),
            "control window remains independent"
        );
        Ok(())
    }
    #[test]
    fn lower_ceiling_keeps_pending_proposals_and_maintenance_charged() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let budget = Budget::new(
            directory.path(),
            Arc::new(AtomicU64::new(0)),
            8 * 1024 * 1024,
            Arc::new(RwLock::new(())),
        )?;
        let data = budget.reserve(4 * 1024 * 1024, true)?;
        budget.limit_data(2 * 1024 * 1024)?;
        assert!(budget.limit_data(0).is_err());
        assert!(budget.limit_data(3 * 1024 * 1024).is_err());
        assert_eq!(budget.snapshot().data_limit, 2 * 1024 * 1024);
        assert_eq!(budget.snapshot().reserved_proposal_bytes, 4 * 1024 * 1024);
        assert!(budget.reserve(1, true).is_err());
        let maintenance = budget.reserve(4 * 1024 * 1024, false)?;
        assert!(budget.reserve(1, false).is_err());
        drop(data);
        // Maintenance still occupies the shared producer ceiling.
        assert!(budget.reserve(1, true).is_err());
        drop(maintenance);
        assert!(budget.reserve(2 * 1024 * 1024, true).is_ok());
        Ok(())
    }
    #[test]
    fn racing_groups_share_admission_and_preserve_control_window() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let written = Arc::new(AtomicU64::new(0));
        let budget = Budget::new(
            directory.path(),
            written.clone(),
            8 * 1024 * 1024,
            Arc::new(RwLock::new(())),
        )?;
        budget.refresh()?;
        let gate = std::sync::Barrier::new(8);
        let reservations = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    let (budget, gate) = (budget.clone(), &gate);
                    scope.spawn(move || {
                        gate.wait();
                        budget.reserve(1024 * 1024, true).ok()
                    })
                })
                .collect();
            threads
                .into_iter()
                .filter_map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(reservations.len(), 6);
        assert!(budget.reserve(1, true).is_err());
        let control = budget.reserve(1024 * 1024, false)?;
        assert_eq!(budget.snapshot().reserved_proposal_bytes, 7 * 1024 * 1024);
        drop(reservations);
        drop(control);
        assert_eq!(budget.snapshot().reserved_proposal_bytes, 0);
        // Peer/HardState traffic must be included even though it never passed
        // through producer admission and may be written by another local group.
        written.fetch_add(6 * 1024 * 1024, Ordering::Release);
        assert!(budget.reserve(1, true).is_err());
        assert!(budget.reserve(1024, false).is_ok());
        Ok(())
    }

    #[test]
    fn collection_refresh_releases_measured_bytes_but_not_pending_proposals() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("segment");
        std::fs::write(&path, vec![7; 6 * 1024 * 1024])?;
        let written = Arc::new(AtomicU64::new(6 * 1024 * 1024));
        let lifecycle = Arc::new(RwLock::new(()));
        let budget = Budget::new(
            directory.path(),
            written.clone(),
            8 * 1024 * 1024,
            lifecycle.clone(),
        )?;
        assert!(budget.reserve(1, true).is_err());
        let pending = budget.reserve(1024, false)?;
        {
            let _guard = lifecycle.write().unwrap();
            std::fs::remove_file(&path)?;
            budget.refresh()?; // Maintenance owns the lifecycle lock: no scan.
            assert!(budget.reserve(1, true).is_err());
        }
        budget.refresh()?;
        assert_eq!(budget.snapshot().accounted_disk_bytes, 0);
        assert_eq!(budget.snapshot().reserved_proposal_bytes, 1024);
        assert!(budget.reserve(6 * 1024 * 1024, true).is_err());
        drop(pending);
        let all = budget.reserve(6 * 1024 * 1024, true)?;
        assert!(budget.reserve(1, true).is_err());
        drop(all);
        // Peer appends bypassing producer admission still count without a scan.
        std::fs::write(&path, vec![9; 6 * 1024 * 1024])?;
        written.fetch_add(6 * 1024 * 1024, Ordering::Release);
        assert!(budget.reserve(1, true).is_err());
        budget.refresh()?;
        assert!(budget.reserve(1, true).is_err());
        assert!(budget.reserve(usize::MAX, false).is_err());
        Ok(())
    }
}
