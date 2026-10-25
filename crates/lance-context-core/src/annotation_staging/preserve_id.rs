use super::*;
use lance_table::transaction::{Operation, UpdateMode};
use std::collections::BTreeSet;

/// The ID is a merge key, never an annotation mutation. Lance 13's narrow
/// merge writes it again and marks its index stale. Keep its original physical
/// column instead. Do not restore index metadata over a replaced column.
pub(super) fn preserve_id(base: &Dataset, operation: &mut Operation) -> Result<()> {
    let Operation::Update {
        removed_fragment_ids,
        updated_fragments,
        new_fragments,
        fields_modified,
        compacted_sstables,
        fields_for_preserving_frag_bitmap,
        update_mode: Some(UpdateMode::RewriteColumns),
        inserted_rows_filter: None,
        ..
    } = operation
    else {
        return Err(invalid("unexpected annotation merge operation"));
    };
    if !removed_fragment_ids.is_empty()
        || !new_fragments.is_empty()
        || !compacted_sstables.is_empty()
        || !fields_for_preserving_frag_bitmap.is_empty()
        || updated_fragments.is_empty()
    {
        return Err(invalid("annotation merge changed fragment membership"));
    }
    let id = base.schema().field_id("id")?;
    let expected_fields: BTreeSet<_> = base
        .schema()
        .project(&["id", "labels", "status", "score"])?
        .field_ids()
        .into_iter()
        .map(|f| f as u32)
        .collect();
    if fields_modified.iter().copied().collect::<BTreeSet<_>>() != expected_fields {
        return Err(invalid("annotation merge changed unexpected fields"));
    }
    for updated in updated_fragments {
        let original = base
            .get_fragment(updated.id as usize)
            .ok_or_else(|| invalid("annotation merge changed an unknown fragment"))?;
        let original = original.metadata();
        // RewriteColumns preserves physical row positions, including deleted
        // positions. Only its file mappings and last-update lineage may differ.
        let mut unchanged = updated.clone();
        unchanged.files.clone_from(&original.files);
        unchanged.last_updated_at_version_meta = original.last_updated_at_version_meta.clone();
        if &unchanged != original
            || !original.overlays.is_empty()
            || updated.files.len() != original.files.len() + 1
            || original
                .files
                .iter()
                .flat_map(|f| f.fields.iter())
                .filter(|&&f| f == id)
                .count()
                != 1
        {
            return Err(invalid("annotation merge changed physical row layout"));
        }
        let (old_files, new_files) = updated.files.split_at_mut(original.files.len());
        let new_file = &mut new_files[0];
        if new_file.fields.iter().filter(|&&f| f == id).count() != 1
            || new_file
                .fields
                .iter()
                .any(|&f| f < 0 || !expected_fields.contains(&(f as u32)))
            || original
                .files
                .iter()
                .any(|f| f.path == new_file.path && f.base_id == new_file.base_id)
        {
            return Err(invalid("unexpected annotation column file"));
        }
        for (old, original) in old_files.iter_mut().zip(&original.files) {
            let mut expected = original.clone();
            expected.fields = original
                .fields
                .iter()
                .map(|f| if new_file.fields.contains(f) { -2 } else { *f })
                .collect::<Vec<_>>()
                .into();
            if *old != expected {
                return Err(invalid("annotation merge changed an original column file"));
            }
            // Restore only the original ID slot; all annotation tombstones stay.
            old.fields = old
                .fields
                .iter()
                .zip(original.fields.iter())
                .map(|(field, original)| if *original == id { id } else { *field })
                .collect::<Vec<_>>()
                .into();
        }
        new_file.fields = new_file
            .fields
            .iter()
            .map(|&field| if field == id { -2 } else { field })
            .collect::<Vec<_>>()
            .into();
    }
    fields_modified.retain(|&field| field != id as u32);
    Ok(())
}
