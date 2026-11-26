use lance_table::format::Manifest;

/// Count live rows using only an already-loaded manifest. Unlike native
/// count_rows(None), this never opens data/deletion files to fill metadata gaps.
/// Missing, legacy or inconsistent counts are unavailable, never assumed zero.
pub fn manifest_rows(manifest: &Manifest) -> Option<u64> {
    if manifest.fragments.is_empty() {
        return Some(0);
    }
    manifest.writer_version.as_ref()?;
    manifest.fragments.iter().try_fold(0u64, |total, fragment| {
        if fragment.files.is_empty() {
            return None;
        }
        let deleted = match &fragment.deletion_file {
            Some(file) => file.num_deleted_rows?,
            None => 0,
        };
        let rows = fragment.physical_rows?.checked_sub(deleted)?;
        total.checked_add(u64::try_from(rows).ok()?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance::Result;
    use std::sync::Arc;

    #[tokio::test]
    async fn live_manifest_counts_or_unavailable_without_opening_files() -> Result<()> {
        let root = tempfile::tempdir()?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset =
            lance::Dataset::write(reader, root.path().to_str().unwrap(), None).await?;
        assert_eq!(manifest_rows(dataset.manifest()), Some(3));
        dataset.delete("id = 2").await?;
        assert_eq!(manifest_rows(dataset.manifest()), Some(2));

        // All counts below must work without any files or fallback I/O.
        let original = dataset.manifest().clone();
        root.close()?;
        assert_eq!(manifest_rows(&original), Some(2));
        let mut missing = original.clone();
        Arc::make_mut(&mut missing.fragments)[0].physical_rows = None;
        assert_eq!(manifest_rows(&missing), None);
        let mut missing = original.clone();
        Arc::make_mut(&mut missing.fragments)[0]
            .deletion_file
            .as_mut()
            .unwrap()
            .num_deleted_rows = None;
        assert_eq!(manifest_rows(&missing), None);
        let mut invalid = original.clone();
        Arc::make_mut(&mut invalid.fragments)[0]
            .deletion_file
            .as_mut()
            .unwrap()
            .num_deleted_rows = Some(4);
        assert_eq!(manifest_rows(&invalid), None);
        let mut invalid = original.clone();
        Arc::make_mut(&mut invalid.fragments)[0].files.clear();
        assert_eq!(manifest_rows(&invalid), None);
        let mut legacy = original;
        legacy.writer_version = None;
        assert_eq!(manifest_rows(&legacy), None);
        legacy.fragments = Arc::new(Vec::new());
        assert_eq!(manifest_rows(&legacy), Some(0));
        Ok(())
    }
}
