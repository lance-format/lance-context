//! Bounded-size diagnostics from an already loaded manifest, without I/O.
use lance_table::format::Manifest;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct FragmentSummary {
    pub fragments: usize,
    pub known_size_fragments: usize,
    /// Subtotal for fragments with all file sizes known. None means overflow;
    /// incomplete coverage is reported by known_size_fragments, not as zero.
    pub known_size_bytes: Option<u64>,
    pub known_physical_rows_fragments: usize,
    /// Physical rows, including deleted rows. None means subtotal overflow.
    pub known_physical_rows: Option<u64>,
    pub fragments_with_deletions: usize,
    /// Cumulative size candidates, not a compaction plan. Does not check native
    /// index coverage, min_fragments, task row/byte limits or memory admission.
    pub small_fragments: [SmallFragmentSummary; 4],
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SmallFragmentSummary {
    pub target_file_bytes: u64,
    /// Known nonzero size strictly below target/2, nonzero trustworthy physical
    /// rows, and no deletion file. A fragment may appear at several targets.
    pub fragments: usize,
    pub bytes: Option<u64>,
    pub physical_rows: Option<u64>,
}

impl From<&Manifest> for FragmentSummary {
    fn from(manifest: &Manifest) -> Self {
        let mut result = Self {
            fragments: manifest.fragments.len(),
            known_size_fragments: 0,
            known_size_bytes: Some(0),
            known_physical_rows_fragments: 0,
            known_physical_rows: Some(0),
            fragments_with_deletions: 0,
            small_fragments: [16, 32, 64, 128].map(|mib| SmallFragmentSummary {
                target_file_bytes: mib << 20,
                fragments: 0,
                bytes: Some(0),
                physical_rows: Some(0),
            }),
        };
        for fragment in manifest.fragments.iter() {
            let size = (!fragment.files.is_empty()).then_some(()).and_then(|()| {
                fragment.files.iter().try_fold(0u64, |bytes, file| {
                    bytes.checked_add(file.file_size_bytes.get()?.get())
                })
            });
            let rows = manifest.writer_version.as_ref().and_then(|_| {
                (!fragment.files.is_empty())
                    .then_some(fragment.physical_rows)
                    .flatten()
                    .and_then(|rows| u64::try_from(rows).ok())
            });
            if let Some(size) = size {
                result.known_size_fragments += 1;
                result.known_size_bytes = result.known_size_bytes.and_then(|n| n.checked_add(size));
            }
            if let Some(rows) = rows {
                result.known_physical_rows_fragments += 1;
                result.known_physical_rows =
                    result.known_physical_rows.and_then(|n| n.checked_add(rows));
            }
            if fragment.deletion_file.is_some() {
                result.fragments_with_deletions += 1;
                continue;
            }
            let (Some(size), Some(rows)) = (size, rows) else {
                continue;
            };
            for small in &mut result.small_fragments {
                if size > 0 && rows > 0 && size < small.target_file_bytes / 2 {
                    small.fragments += 1;
                    small.bytes = small.bytes.and_then(|n| n.checked_add(size));
                    small.physical_rows = small.physical_rows.and_then(|n| n.checked_add(rows));
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use lance_io::utils::CachedFileSize;
    use std::sync::Arc;

    #[tokio::test]
    async fn loaded_manifest_summary_handles_gaps_boundaries_and_deletions_without_files(
    ) -> lance::Result<()> {
        let root = tempfile::tempdir()?;
        let batch = RecordBatch::try_from_iter([(
            "id",
            Arc::new(Int32Array::from(vec![1, 2, 3])) as arrow_array::ArrayRef,
        )])?;
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let mut dataset =
            lance::Dataset::write(reader, root.path().to_str().unwrap(), None).await?;
        dataset.delete("id = 2").await?;
        let mut manifest = dataset.manifest().clone();
        root.close()?;

        // Metadata diagnostics continue after all physical files are gone.
        let summary = FragmentSummary::from(&manifest);
        assert_eq!(summary.fragments_with_deletions, 1);
        assert_eq!(summary.known_physical_rows, Some(3));
        assert!(summary.small_fragments.iter().all(|s| s.fragments == 0));
        let mut template = manifest.fragments[0].clone();
        template.deletion_file = None;
        template.files[0].file_size_bytes = CachedFileSize::new(8 << 20);
        let mut small = template.clone();
        small.files[0].file_size_bytes = CachedFileSize::new((8 << 20) - 1);
        let mut missing_size = template.clone();
        missing_size.files[0].file_size_bytes = CachedFileSize::unknown();
        let mut missing_rows = template.clone();
        missing_rows.physical_rows = None;
        let mut empty = template.clone();
        empty.files.clear();
        manifest.fragments = Arc::new(vec![template, small, missing_size, missing_rows, empty]);
        let summary = FragmentSummary::from(&manifest);
        assert_eq!(summary.fragments, 5);
        assert_eq!(summary.known_size_fragments, 3);
        assert_eq!(summary.known_size_bytes, Some((24 << 20) - 1));
        assert_eq!(summary.known_physical_rows_fragments, 3);
        assert_eq!(summary.known_physical_rows, Some(9));
        assert_eq!(summary.small_fragments[0].fragments, 1);
        assert_eq!(summary.small_fragments[0].bytes, Some((8 << 20) - 1));
        assert_eq!(summary.small_fragments[1].fragments, 2);
        assert_eq!(summary.small_fragments[1].bytes, Some((16 << 20) - 1));
        assert_eq!(summary.small_fragments[1].physical_rows, Some(6));
        let encoded = serde_json::to_value(&summary).unwrap();
        assert_eq!(encoded["small_fragments"][0]["target_file_bytes"], 16 << 20);

        manifest.writer_version = None;
        let legacy = FragmentSummary::from(&manifest);
        assert_eq!(legacy.known_physical_rows_fragments, 0);
        assert!(legacy.small_fragments.iter().all(|s| s.fragments == 0));
        manifest.fragments = Arc::new(Vec::new());
        let empty = FragmentSummary::from(&manifest);
        assert_eq!(empty.fragments, 0);
        assert_eq!(empty.known_size_bytes, Some(0));
        Ok(())
    }
}
