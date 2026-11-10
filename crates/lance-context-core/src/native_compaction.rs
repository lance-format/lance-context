//! Prepare native index remaps before a managed compaction is published.
use lance::dataset::{
    index::DatasetIndexRemapperOptions,
    optimize::{IndexRemapperOptions, RewriteResult},
};
use lance::index::DatasetIndexExt;
use lance::Dataset;
use lance_core::utils::row_addr_remap::{GroupInput, RowAddrRemap};
use lance_core::{Error, Result};
use lance_table::transaction::{Operation, RewriteGroup, RewrittenIndex};
use roaring::RoaringTreemap;
use std::collections::HashSet;
use std::io::Cursor;

/// Prepare one bounded, append-only rewrite against its exact source snapshot.
/// The caller assigns final fragment IDs under its publication fence, retains
/// its workspace reservation, and persists the returned operation before commit.
/// This writes index files but never commits or reserves a manifest version.
/// Never call it again to resolve an uncertain publication: reuse that persisted
/// operation, including the original remapped index files.
pub async fn prepare_rewrite(
    base: &Dataset,
    result: RewriteResult,
    maximum_mapping_bytes: usize,
) -> Result<Operation> {
    let invalid = |message: &str| Error::invalid_input(message.to_owned());
    if result.read_version != base.version_id()
        || !base.manifest().uses_stable_row_ids()
        || !(2..=64).contains(&result.original_fragments.len())
        || result.new_fragments.is_empty()
        || result.new_fragments.len() >= result.original_fragments.len()
    {
        return Err(invalid("unsupported native compaction source"));
    }
    let mut expected = RoaringTreemap::new();
    let mut old_ids = Vec::new();
    for fragment in &result.original_fragments {
        let id = u32::try_from(fragment.id)
            .map_err(|_| invalid("compaction source ID exceeds the address domain"))?;
        let rows = fragment
            .physical_rows
            .and_then(|rows| u32::try_from(rows).ok())
            .filter(|&rows| rows > 0)
            .ok_or_else(|| invalid("compaction source has unknown or oversized row count"))?;
        if fragment.deletion_file.is_some()
            || !fragment.overlays.is_empty()
            || old_ids.last().is_some_and(|&last| last >= id)
            || !base.manifest().fragments.contains(fragment)
        {
            return Err(invalid(
                "compaction source differs from its ordered snapshot",
            ));
        }
        old_ids.push(id);
        let first = u64::from(id) << 32;
        expected.insert_range(first..first + u64::from(rows));
    }
    let high_water = base
        .manifest()
        .max_fragment_id
        .ok_or_else(|| invalid("compaction source has no fragment high-water mark"))?;
    let mut new_frags = Vec::new();
    for (ordinal, fragment) in result.new_fragments.iter().enumerate() {
        let id = high_water
            .checked_add(ordinal as u32 + 1)
            .ok_or_else(|| invalid("compaction fragment IDs exhausted"))?;
        let rows = fragment
            .physical_rows
            .and_then(|rows| u32::try_from(rows).ok())
            .filter(|&rows| rows > 0)
            .ok_or_else(|| invalid("compaction output has unknown or oversized row count"))?;
        if fragment.id != u64::from(id)
            || fragment.deletion_file.is_some()
            || !fragment.overlays.is_empty()
            || fragment.files.is_empty()
        {
            return Err(invalid(
                "compaction output differs from its assigned layout",
            ));
        }
        new_frags.push((id, rows));
    }
    if expected.len()
        != new_frags
            .iter()
            .map(|(_, rows)| u64::from(*rows))
            .sum::<u64>()
    {
        return Err(invalid("compaction changed the row count"));
    }
    let indices = base.load_indices().await?;
    let mut required_remaps = HashSet::new();
    for index in indices.iter() {
        // Deferred reuse histories need their own validated publication path.
        if lance_index::is_system_index(index) {
            return Err(invalid("native compaction cannot bypass a system index"));
        }
        let coverage = index
            .fragment_bitmap
            .as_ref()
            .ok_or_else(|| invalid("compaction index coverage is unknown"))?;
        let covered = coverage.contains(old_ids[0]);
        if old_ids.iter().any(|id| coverage.contains(*id) != covered) {
            return Err(invalid("compaction mixes index coverage"));
        }
        if covered && index.results_are_row_addrs() {
            if !index.supports_eager_remap() {
                return Err(invalid("compaction cannot preserve this address index"));
            }
            required_remaps.insert(index.uuid);
        }
    }
    let mut rewritten_indices = Vec::new();
    if !required_remaps.is_empty() {
        let bytes = result
            .row_addrs
            .as_ref()
            .filter(|bytes| !bytes.is_empty() && bytes.len() <= maximum_mapping_bytes)
            .ok_or_else(|| invalid("compaction address mapping is missing or oversized"))?;
        let mut reader = Cursor::new(bytes);
        let recorded = RoaringTreemap::deserialize_from(&mut reader)
            .map_err(|error| invalid(&error.to_string()))?;
        if reader.position() != bytes.len() as u64 || recorded != expected {
            return Err(invalid(
                "compaction address mapping differs from its source rows",
            ));
        }
        let remapper = DatasetIndexRemapperOptions::default()
            .create_remapper(base)
            .await?
            .ok_or_else(|| invalid("compaction index remapper is unavailable"))?;
        let mapping = RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: recorded,
            old_frag_ids: old_ids.clone(),
            new_frags,
        }])?;
        rewritten_indices = remapper
            .remap_indices(
                mapping,
                &old_ids.into_iter().map(u64::from).collect::<Vec<_>>(),
            )
            .await?
            .into_iter()
            .map(|index| RewrittenIndex {
                old_id: index.old_id,
                new_id: index.new_id,
                new_index_details: index.index_details,
                new_index_version: index.index_version,
                new_index_files: index.files,
            })
            .collect();
        if rewritten_indices.len() != required_remaps.len()
            || rewritten_indices
                .iter()
                .map(|index| index.old_id)
                .collect::<HashSet<_>>()
                != required_remaps
        {
            return Err(invalid(
                "native compaction did not remap every covered address index",
            ));
        }
    }
    Ok(Operation::Rewrite {
        groups: vec![RewriteGroup {
            old_fragments: result.original_fragments,
            new_fragments: result.new_fragments,
        }],
        rewritten_indices,
        frag_reuse_index: None,
    })
}

/// Validate persisted index remaps without writing files or rerunning a remap.
/// The caller separately validates the rewrite's source fragments, assigned IDs,
/// row counts, and publication identity against this same snapshot.
pub async fn validate_index_remaps(base: &Dataset, operation: &Operation) -> Result<()> {
    let invalid = |message: &str| Error::invalid_input(message.to_owned());
    let Operation::Rewrite {
        groups,
        rewritten_indices,
        frag_reuse_index: None,
    } = operation
    else {
        return Err(invalid("unsupported prepared native compaction"));
    };
    if groups.len() != 1 || groups[0].old_fragments.is_empty() {
        return Err(invalid(
            "prepared native compaction has no single source group",
        ));
    }
    let indices = base.load_indices().await?;
    let mut required = HashSet::new();
    let old_ids = groups[0]
        .old_fragments
        .iter()
        .map(|fragment| {
            u32::try_from(fragment.id)
                .map_err(|_| invalid("prepared compaction source exceeds the address domain"))
        })
        .collect::<Result<Vec<_>>>()?;
    for index in indices.iter() {
        if lance_index::is_system_index(index) {
            return Err(invalid("prepared compaction cannot bypass a system index"));
        }
        let coverage = index
            .fragment_bitmap
            .as_ref()
            .ok_or_else(|| invalid("prepared compaction index coverage is unknown"))?;
        let covered = coverage.contains(old_ids[0]);
        if old_ids.iter().any(|id| coverage.contains(*id) != covered) {
            return Err(invalid("prepared compaction mixes index coverage"));
        }
        if covered && index.results_are_row_addrs() {
            if !index.supports_eager_remap() {
                return Err(invalid(
                    "prepared compaction cannot preserve this address index",
                ));
            }
            required.insert(index.uuid);
        }
    }
    let mut new_ids = HashSet::new();
    for remap in rewritten_indices {
        let original = indices
            .iter()
            .find(|index| index.uuid == remap.old_id)
            .ok_or_else(|| invalid("prepared compaction remaps an unknown index"))?;
        if !required.remove(&remap.old_id)
            || remap.new_id.is_nil()
            || indices.iter().any(|index| index.uuid == remap.new_id)
            || !new_ids.insert(remap.new_id)
            || remap.new_index_version != original.index_version
            || !original
                .index_details
                .as_ref()
                .is_some_and(|details| details.type_url == remap.new_index_details.type_url)
            || remap
                .new_index_files
                .as_ref()
                .is_none_or(|files| files.is_empty())
        {
            return Err(invalid(
                "prepared compaction changed index identity or omitted its files",
            ));
        }
    }
    if !required.is_empty() {
        return Err(invalid(
            "prepared compaction omitted a required index remap",
        ));
    }
    Ok(())
}
