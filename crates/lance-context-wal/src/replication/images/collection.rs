//! Collection of application-image files in the exclusively owned node directory.
use super::*;
use std::{
    ffi::OsStr,
    fs::{self, File, ReadDir},
};

const MAX_SCAN_ENTRIES: usize = 4096;

/// One bounded directory-scan batch. Removed bytes are file lengths, not a
/// measurement of storage-device free space or physical bytes reclaimed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageCollectionProgress {
    pub scanned_entries: u64,
    pub retained_files: u64,
    pub removed_files: u64,
    pub removed_bytes: u64,
    pub finished: bool,
}

/// A resumable scan on a bounded blocking worker. Its node-wide image permit
/// excludes builders, receivers and checkpoint installation, while ordinary
/// append/poll work may continue. Drain or drop it before creating another image.
/// A storage handle keeps the exclusive engine lock alive even if all Node
/// owners are dropped; another incarnation cannot race this collector.
pub struct ImageCollection {
    permit: Permit,
    entries: ReadDir,
    protected: BTreeSet<(u64, Uuid)>,
    finished: bool,
}

impl ImageCollection {
    /// Inspect at most `max_entries` directory entries, including unrelated or
    /// protected files. Only canonical, unreferenced image files of hosted groups
    /// are removed. A successful deleting batch syncs the containing directory.
    /// Retain this cursor for the next batch; restarting each scan can starve
    /// later entries. No file payload is decoded, read or rewritten.
    pub fn run_batch(&mut self, max_entries: usize) -> Result<ImageCollectionProgress> {
        ensure!(
            (1..=MAX_SCAN_ENTRIES).contains(&max_entries),
            "invalid image collection batch size"
        );
        self.permit.0.store.initial_state()?;
        let _files = self.permit.0.store.image_collection_guard()?;
        let mut progress = ImageCollectionProgress::default();
        for _ in 0..max_entries {
            if self.finished {
                break;
            }
            let Some(entry) = self.entries.next() else {
                self.finished = true;
                break;
            };
            let entry = entry?;
            progress.scanned_entries += 1;
            let Some(key) = image_key(&entry.file_name()) else {
                continue;
            };
            if !self.permit.0.groups.contains(&key.0) {
                continue;
            }
            if self.protected.contains(&key) {
                progress.retained_files += 1;
                continue;
            }
            // The node owns this directory. Keep unexpected symlinks, hard
            // links, subdirectories and names outside its image namespace.
            let metadata = fs::symlink_metadata(entry.path())?;
            if !metadata.file_type().is_file() {
                continue;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if metadata.nlink() != 1 {
                    continue;
                }
            }
            let bytes = progress
                .removed_bytes
                .checked_add(metadata.len())
                .context("image collection byte counter overflow")?;
            fs::remove_file(entry.path())?;
            progress.removed_files += 1;
            progress.removed_bytes = bytes;
        }
        if progress.removed_files > 0 {
            File::open(&self.permit.0.directory)?.sync_all()?;
        }
        progress.finished = self.finished;
        if self.finished {
            self.permit.0.store.refresh_disk_exclusive()?;
        }
        Ok(progress)
    }
}

fn image_key(name: &OsStr) -> Option<(u64, Uuid)> {
    let name = name.to_str()?.strip_prefix("context-application-")?;
    let stem = name
        .strip_suffix(".state")
        .or_else(|| name.strip_suffix(".pack"))?;
    let (group, id) = stem.split_once('-')?;
    let number: u64 = group.parse().ok()?;
    let uuid = Uuid::parse_str(id).ok()?;
    (number > 0 && number.to_string() == group && !uuid.is_nil() && uuid.to_string() == id)
        .then_some((number, uuid))
}

impl Node {
    /// Collect obsolete and abandoned application-image files. Current installed
    /// images and captured readers/transfers remain protected, including images
    /// of sibling group owners returned by `into_groups`. This does not collect
    /// Raft-engine segments or Lance objects and does not enforce a disk quota.
    pub fn prepare_image_collection(&self) -> Result<ImageCollection> {
        self.healthy()?;
        let permit = self.images.acquire()?;
        let protected = {
            let mut registry = self
                .images
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("image registry poisoned"))?;
            registry.live.retain(|_, image| image.strong_count() > 0);
            registry
                .live
                .keys()
                .chain(registry.installed.values())
                .copied()
                .collect()
        };
        let entries = fs::read_dir(&self.images.directory)?;
        Ok(ImageCollection {
            permit,
            entries,
            protected,
            finished: false,
        })
    }
}
